import pandas as pd
import numpy as np
import scanpy
import dendropy
from typing import Union, List, Tuple, Callable, Dict, Optional
import re
import sys
from preprocess import get_top_down_cluster_ids


def dist_matrix_to_numpy(pdm, names):
  n = len(names)
  dists = np.zeros((n, n))
  for i in range(n - 1):
    for j in range(i, len(names)):
      dists[i][j] = _cell_dist(pdm, names[i], names[j])
      dists[j][i] = dists[i][j]
  return dists

def _cell_dist(pdm, c1, c2):
  return pdm(pdm.taxon_namespace.get_taxon(c1), pdm.taxon_namespace.get_taxon(c2))


def _leaf_label(nd) -> Optional[str]:
    if getattr(nd, "taxon", None) is not None and nd.taxon is not None and nd.taxon.label is not None:
        return str(nd.taxon.label).strip()
    if getattr(nd, "label", None) is not None and nd.label is not None:
        return str(nd.label).strip()
    return None

def _preprocess_tree_for_lca(tree: dendropy.Tree):
    nodes = list(tree.preorder_node_iter())
    node_to_id = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)

    depth = np.zeros(n_nodes, dtype=np.float64)
    for nd in tree.preorder_node_iter():
        i = node_to_id[nd]
        if nd.parent_node is None:
            depth[i] = 0.0
        else:
            p = node_to_id[nd.parent_node]
            elen = nd.edge.length or 0.0
            depth[i] = depth[p] + float(elen)

    euler, level = [], []
    first = np.full(n_nodes, -1, dtype=np.int64)

    def dfs(u, h):
        uid = node_to_id[u]
        if first[uid] == -1:
            first[uid] = len(euler)
        euler.append(uid)
        level.append(h)
        for v in u.child_node_iter():
            dfs(v, h + 1)
            euler.append(uid)
            level.append(h)

    dfs(tree.seed_node, 0)
    euler = np.asarray(euler, dtype=np.int32)
    level = np.asarray(level, dtype=np.int32)

    m = len(level)
    K = m.bit_length()
    st = np.empty((K, m), dtype=np.int32)
    st[0] = np.arange(m, dtype=np.int32)
    for k in range(1, K):
        span = 1 << k
        half = span >> 1
        upto = m - span + 1
        left = st[k - 1, :upto]
        right = st[k - 1, half:half + upto]
        take_left = (level[left] <= level[right])
        st[k, :upto] = np.where(take_left, left, right)

    label_to_id: Dict[str, int] = {}
    leaf_names_all: List[str] = []
    for nd in tree.leaf_node_iter():
        lab = _leaf_label(nd)
        if lab is None:
            continue
        label_to_id[lab] = node_to_id[nd]
        leaf_names_all.append(lab)

    return label_to_id, depth, euler, level, first, st

def _lca_id(u_id: int, v_id: int, first, st, euler, level) -> int:
    iu, iv = int(first[u_id]), int(first[v_id])
    if iu > iv:
        iu, iv = iv, iu
    width = iv - iu + 1
    k = width.bit_length() - 1
    i1 = st[k, iu]
    i2 = st[k, iv - (1 << k) + 1]
    return int(euler[i1] if level[i1] <= level[i2] else euler[i2])

def normalize_identity(s: str) -> str:
    return s

def normalize_strip_dash_number(s: str) -> str:
    return re.sub(r"-\d+$", "", s)

def normalize_barcode_only(s: str) -> str:
    # 'L8.AAAC...-1' -> 'AAAC...'
    m = re.search(r'(?:^|\.)([^.\-]+)(?:-\d+)?$', s)
    return m.group(1) if m else s


def compute_distance_matrix_for_names(
    tree: dendropy.Tree,
    names: Union[np.ndarray, List[str]],
    dtype=np.float32,
    memmap_path: Optional[str] = None,
    block_rows: int = 1024,
    return_condensed: bool = False,
    name_normalizer: Callable[[str], str] = normalize_identity,
    on_missing: str = "error",             # "error" | "drop" | "map"
    name_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    names = np.asarray(names)
    label_to_id, depth, euler, level, first, st = _preprocess_tree_for_lca(tree)

    ids_list: List[int] = []
    missing: List[str] = []

    norm_index: Dict[str, int] = {name_normalizer(k): v for k, v in label_to_id.items()}

    for nm in names:
        key = name_normalizer(str(nm).strip())
        nid = norm_index.get(key, None)
        if nid is None:
            missing.append(str(nm))
            ids_list.append(-1)
        else:
            ids_list.append(int(nid))

    if missing:
        if on_missing == "error":
            raise ValueError(
                f"{len(missing)} names not found after normalization, e.g. {missing[:5]}"
            )
        elif on_missing == "map":
            if not name_map:
                raise ValueError("on_missing='map' but name_map is None.")
            fixed = 0
            for i, nm in enumerate(names):
                if ids_list[i] != -1:
                    continue
                tgt = name_map.get(str(nm), None)
                if tgt is None:
                    continue
                key2 = name_normalizer(str(tgt))
                nid2 = norm_index.get(key2, None)
                if nid2 is not None:
                    ids_list[i] = int(nid2)
                    fixed += 1
            still = [str(nm) for i, nm in enumerate(names) if ids_list[i] < 0]
            if still:
                sys.stderr.write(f"[warn] {len(still)} names still missing, e.g. {still[:5]}\n")
            else:
                sys.stderr.write(f"[info] all missing resolved by mapping ({fixed}).\n")
        elif on_missing == "drop":
            keep = np.array([i != -1 for i in ids_list], dtype=bool)
            dropped = (~keep).sum()
            sys.stderr.write(f"[warn] dropping {dropped} names, e.g. {np.array(missing[:5])}\n")
            names = names[keep]
            ids_list = [i for i in ids_list if i != -1]
        else:
            raise ValueError("on_missing must be one of: 'error', 'drop', 'map'")

    ids = np.array(ids_list, dtype=np.int32)
    if (ids < 0).any():
        bad = [str(nm) for i, nm in enumerate(names) if ids[i] < 0][:5]
        raise ValueError(f"Unresolved missing names remain, e.g. {bad}")

    n = len(ids)

    # condensed
    if return_condensed:
        m = n * (n - 1) // 2
        out = np.empty(m, dtype=dtype)
        k = 0
        for i in range(n - 1):
            ui = int(ids[i]); di = depth[ui]
            for j in range(i + 1, n):
                vj = int(ids[j])
                a = _lca_id(ui, vj, first, st, euler, level)
                out[k] = di + depth[vj] - 2.0 * depth[a]
                k += 1
        return out, names

    # full matrix
    if memmap_path is None:
        D = np.empty((n, n), dtype=dtype)
    else:
        D = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=(n, n))
    D.fill(0)

    for i0 in range(0, n, block_rows):
        i1 = min(i0 + block_rows, n)
        for i in range(i0, i1):
            ui = int(ids[i]); di = depth[ui]
            row = D[i]
            for j in range(i + 1, n):
                vj = int(ids[j])
                a = _lca_id(ui, vj, first, st, euler, level)
                row[j] = di + depth[vj] - 2.0 * depth[a]

    D += D.T
    np.fill_diagonal(D, 0)
    return D, names



def preprocess_adata_Fam(adata, name, prefix, data_path, meta_df):

  adata = adata.copy()

  tree = dendropy.Tree.get(path=f'{data_path}/KPTracer-Data/trees/{name}_tree.nwk', schema='newick')
  for edge in tree.edges():
    edge.length = 1
  print('height:', tree.max_distance_from_root())

  tumor_meta = meta_df[meta_df['Tumor'].str.startswith(prefix) & meta_df['TS_Present'] & meta_df['RNA_Present']]
  tumor_adata = adata[adata.obs['Tumor'].str.startswith(prefix)]
  (tumor_meta['Unnamed: 0'].value_counts() != 1).sum()  # check that cells are unique

  # find intersection of cells in both modalities
  tree_cells = set([c.taxon.label for c in tree.leaf_nodes()])
  meta_cells = set(tumor_meta['Unnamed: 0'])
  adata_cells = set(tumor_adata.obs.index)
  print(len(tree_cells), len(meta_cells), len(adata_cells))
  print('meta but not tree (vice versa): ', len(meta_cells - tree_cells), '(', len(tree_cells - meta_cells), ')')
  print('tree but not adata (vice versa): ', len(tree_cells - adata_cells), '(', len(adata_cells - tree_cells), ')')
  print('meta but not adata (vice versa): ', len(meta_cells - adata_cells), '(', len(adata_cells - meta_cells), ')')

  intersect_cells = meta_cells
  intersect_cells -= intersect_cells - tree_cells
  intersect_cells -= intersect_cells - adata_cells
  print('intersection: ', len(intersect_cells))

  # prune tree
  tree.retain_taxa_with_labels(intersect_cells, suppress_unifurcations=True)
  for edge in tree.edges():
    # set edge lengths to 1 again as we deleted some degree 1 nodes
    edge.length = 1
  print('height:', tree.max_distance_from_root())
  print('leaves:', len(tree.leaf_nodes()))
  tree.write(path=f'{data_path}/preprocessed/{name}_pruned.nwk', schema='newick')

  # get degrees of nodes:
  np.unique(list(map(lambda x: x.num_child_nodes(), tree.nodes())), return_counts=True)

  pruned_adata = tumor_adata[tumor_adata.obs.index.isin(intersect_cells)]
  print(pruned_adata.X.shape)
  pruned_adata.write(f'{data_path}/preprocessed/{name}_pruned_adata.h5ad')

  # index is cell names
  pd.DataFrame(pruned_adata.obs.index).to_csv(f'{data_path}/preprocessed/{name}_cells.txt', index=False, columns=[0], header=False)

  # tumor association
  tumor_meta.set_index('Unnamed: 0').loc[pruned_adata.obs.index].to_csv(
      f'{data_path}/preprocessed/{name}_tumors.txt', index=False, columns=['Tumor'], header=False)

  with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
    labels = np.array(f.read().splitlines())
  cell_types = pd.DataFrame(labels, columns=['Cell names']).set_index('Cell names')
  cell_types = cell_types.join(adata.obs['Cluster-Name'])
  assert (labels == cell_types.index).all()  # check order
  print(cell_types)

  print(cell_types['Cluster-Name'].value_counts())

  cell_types.to_csv(f'{data_path}/preprocessed/{name}_cell_types.txt', header=False, index=False)

  tree = dendropy.Tree.get(path=f'{data_path}/preprocessed/{name}_pruned.nwk', schema='newick')
  with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
    cell_names = np.array(f.read().splitlines())
  # computing the phylogenetic_distance_matrix with dendropy can take a few minutes
  dist_matrix = dist_matrix_to_numpy(tree.phylogenetic_distance_matrix(), cell_names)
  np.save(f'{data_path}/preprocessed/{name}_dist_matrix.npy', dist_matrix)


def preprocess_adata(adata, name, data_path, meta_df):

  adata = adata.copy()

  tree = dendropy.Tree.get(path=f'{data_path}/KPTracer-Data/trees/{name}_tree.nwk', schema='newick')
  for edge in tree.edges():
    edge.length = 1
  print('height:', tree.max_distance_from_root())

  tumor_meta = meta_df[(meta_df['Tumor'] == name) & meta_df['TS_Present'] & meta_df['RNA_Present']]
  tumor_adata = adata[adata.obs['Tumor'] == name]
  (tumor_meta['Unnamed: 0'].value_counts() != 1).sum()  # check that cells are unique

  # find intersection of cells in both modalities
  tree_cells = set([c.taxon.label for c in tree.leaf_nodes()])
  meta_cells = set(tumor_meta['Unnamed: 0'])
  adata_cells = set(tumor_adata.obs.index)
  print(len(tree_cells), len(meta_cells), len(adata_cells))
  print('meta but not tree (vice versa): ', len(meta_cells - tree_cells), '(', len(tree_cells - meta_cells), ')')
  print('tree but not adata (vice versa): ', len(tree_cells - adata_cells), '(', len(adata_cells - tree_cells), ')')
  print('meta but not adata (vice versa): ', len(meta_cells - adata_cells), '(', len(adata_cells - meta_cells), ')')

  intersect_cells = meta_cells
  intersect_cells -= intersect_cells - tree_cells
  intersect_cells -= intersect_cells - adata_cells
  print('intersection: ', len(intersect_cells))

  # prune tree
  tree.retain_taxa_with_labels(intersect_cells, suppress_unifurcations=True)
  for edge in tree.edges():
    # set edge lengths to 1 again as we deleted some degree 1 nodes
    edge.length = 1
  print('height:', tree.max_distance_from_root())
  print('leaves:', len(tree.leaf_nodes()))
  tree.write(path=f'{data_path}/preprocessed/{name}_pruned.nwk', schema='newick')

  # get degrees of nodes:
  np.unique(list(map(lambda x: x.num_child_nodes(), tree.nodes())), return_counts=True)

  pruned_adata = tumor_adata[tumor_adata.obs.index.isin(intersect_cells)]
  print(pruned_adata.X.shape)
  pruned_adata.write(f'{data_path}/preprocessed/{name}_pruned_adata.h5ad')


  # remove rare genes
  keep = np.array((pruned_adata.X > 0).sum(axis=0) >= 10).flatten()
  print('genes expressed in 10 or more cells:', np.sum(keep))
  gene_names = adata.var_names[keep]
  gene_counts = pruned_adata.X[:, keep].toarray()
  print(gene_counts.shape)

  # L1 normalization on cells to [0, 10000]
  cell_sums = gene_counts.sum(axis=1)
  gene_counts[cell_sums > 0] = gene_counts[cell_sums > 0] / cell_sums[cell_sums > 0].reshape(-1, 1)
  gene_counts *= 10000
  # log2(1 + x) tranform
  gene_counts = np.log2(1 + gene_counts)

  # index is cell names
  pd.DataFrame(pruned_adata.obs.index).to_csv(f'{data_path}/preprocessed/{name}_cells.txt', index=False, columns=[0], header=False)

  # tumor association
  tumor_meta.set_index('Unnamed: 0').loc[pruned_adata.obs.index].to_csv(
      f'{data_path}/preprocessed/{name}_tumors.txt', index=False, columns=['Tumor'], header=False)

  df = pd.DataFrame(gene_counts)
  df.columns = gene_names
  df.to_csv(f'{data_path}/preprocessed/{name}_normalized_log_counts.txt', index=False)

  with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
    labels = np.array(f.read().splitlines())
  cell_types = pd.DataFrame(labels, columns=['Cell names']).set_index('Cell names')
  cell_types = cell_types.join(adata.obs['Cluster-Name'])
  assert (labels == cell_types.index).all()  # check order
  print(cell_types)

  print(cell_types['Cluster-Name'].value_counts())

  cell_types.to_csv(f'{data_path}/preprocessed/{name}_cell_types.txt', header=False, index=False)

  tree = dendropy.Tree.get(path=f'{data_path}/preprocessed/{name}_pruned.nwk', schema='newick')
  with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
    cell_names = np.array(f.read().splitlines())
  # computing the phylogenetic_distance_matrix with dendropy can take a few minutes
  dist_matrix = dist_matrix_to_numpy(tree.phylogenetic_distance_matrix(), cell_names)
  np.save(f'{data_path}/preprocessed/{name}_dist_matrix.npy', dist_matrix)


def preprocess_adata_big_num(adata, name, data_path, meta_df):
  adata = adata.copy()

  tree = dendropy.Tree.get(path=f'{data_path}/KPTracer-Data/trees/{name}_tree.nwk', schema='newick')
  for edge in tree.edges():
    edge.length = 1
  print('height:', tree.max_distance_from_root())

  tumor_meta = meta_df[(meta_df['Tumor'] == name) & meta_df['TS_Present'] & meta_df['RNA_Present']]
  tumor_adata = adata[adata.obs['Tumor'] == name]
  (tumor_meta['Unnamed: 0'].value_counts() != 1).sum()  # check that cells are unique

  # find intersection of cells in both modalities
  tree_cells = set([c.taxon.label for c in tree.leaf_nodes()])
  meta_cells = set(tumor_meta['Unnamed: 0'])
  adata_cells = set(tumor_adata.obs.index)
  print(len(tree_cells), len(meta_cells), len(adata_cells))
  print('meta but not tree (vice versa): ', len(meta_cells - tree_cells), '(', len(tree_cells - meta_cells), ')')
  print('tree but not adata (vice versa): ', len(tree_cells - adata_cells), '(', len(adata_cells - tree_cells), ')')
  print('meta but not adata (vice versa): ', len(meta_cells - adata_cells), '(', len(adata_cells - meta_cells), ')')

  intersect_cells = meta_cells
  intersect_cells -= intersect_cells - tree_cells
  intersect_cells -= intersect_cells - adata_cells
  print('intersection: ', len(intersect_cells))

  # prune tree
  tree.retain_taxa_with_labels(intersect_cells, suppress_unifurcations=True)
  for edge in tree.edges():
    # set edge lengths to 1 again as we deleted some degree 1 nodes
    edge.length = 1
  print('height:', tree.max_distance_from_root())
  print('leaves:', len(tree.leaf_nodes()))
  tree.write(path=f'{data_path}/preprocessed/{name}_pruned.nwk', schema='newick')

  # get degrees of nodes:
  np.unique(list(map(lambda x: x.num_child_nodes(), tree.nodes())), return_counts=True)

  pruned_adata = tumor_adata[tumor_adata.obs.index.isin(intersect_cells)]
  print(pruned_adata.X.shape)
  pruned_adata.write(f'{data_path}/preprocessed/{name}_pruned_adata.h5ad')

  # Get top-down clusters and filter cells based on cluster size
  tree_path = f'{data_path}/preprocessed/{name}_pruned.nwk'
  with open(f'{data_path}/preprocessed/{name}_cells.txt', 'w') as f:
      f.write('\n'.join(pruned_adata.obs.index))
  with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
      labels_in_order = np.array(f.read().splitlines())

  # Calculate top-down clusters
  _, top_down_clusters = get_top_down_cluster_ids(tree_path, labels_in_order, depth=4)
  # Add subtree information to adata
  pruned_adata.obs['subtree'] = top_down_clusters
  print(pruned_adata.obs['subtree'].value_counts())

  # Filter cells based on subtree cluster size (example: keep clusters with at least 10 cells)
  cluster_counts = pruned_adata.obs['subtree'].value_counts()
  clusters_to_keep = cluster_counts[cluster_counts >= 30].index
  filtered_adata = pruned_adata[pruned_adata.obs['subtree'].isin(clusters_to_keep)]
  filtered_cells = filtered_adata.obs.index
  print(f'Filtered to {len(filtered_cells)} cells after subtree size filtering')

  # Prune tree to filtered cells
  filtered_tree = dendropy.Tree.get(path=tree_path, schema='newick')
  filtered_tree.retain_taxa_with_labels(filtered_cells, suppress_unifurcations=True)
  for edge in filtered_tree.edges():
      edge.length = 1
  print('Filtered tree height:', filtered_tree.max_distance_from_root())
  print('Filtered leaves:', len(filtered_tree.leaf_nodes()))
  filtered_tree.write(path=f'{data_path}/preprocessed/{name}_filtered_pruned.nwk', schema='newick')

  # Save filtered cells
  pd.DataFrame(filtered_adata.obs.index).to_csv(f'{data_path}/preprocessed/{name}_filtered_cells.txt',
                                                index=False, columns=[0], header=False)

  # Save tumor association for filtered cells
  tumor_meta.set_index('Unnamed: 0').loc[filtered_adata.obs.index].to_csv(
      f'{data_path}/preprocessed/{name}_filtered_tumors.txt', index=False, columns=['Tumor'], header=False)

  # Save cell types for filtered cells
  with open(f'{data_path}/preprocessed/{name}_filtered_cells.txt') as f:
      labels = np.array(f.read().splitlines())
  cell_types = pd.DataFrame(labels, columns=['Cell names']).set_index('Cell names')
  cell_types = cell_types.join(adata.obs['Cluster-Name'])
  assert (labels == cell_types.index).all()
  print(cell_types['Cluster-Name'].value_counts())
  cell_types.to_csv(f'{data_path}/preprocessed/{name}_filtered_cell_types.txt', header=False, index=False)

  # Compute and save distance matrix for filtered cells
  with open(f"{data_path}/preprocessed/{name}_filtered_cells.txt") as f:
      cell_names = np.array(f.read().splitlines())
  dist_matrix, names_out = compute_distance_matrix_for_names(
      filtered_tree,
      cell_names,
      dtype=np.float32,
      memmap_path=None,
      block_rows=1024,
      return_condensed=False,
      name_normalizer=normalize_identity,
      on_missing="error",
  )
  assert np.array_equal(names_out, cell_names)
  np.save(f'{data_path}/preprocessed/{name}_filtered_dist_matrix.npy', dist_matrix)

  # Save the filtered AnnData object
  filtered_adata.write(f'{data_path}/preprocessed/{name}_filtered_pruned_adata.h5ad')



if __name__ == "__main__":
    data_path = '/data/DeepTracing/experiments/Tumor/data'
    meta_df = pd.read_csv(f'{data_path}/KPTracer-Data/KPTracer_meta.csv')
    print(meta_df)
    adata = scanpy.read_h5ad(f'{data_path}/KPTracer-Data/expression/adata_processed.combined.h5ad')
    print(adata)

    preprocess_adata_Fam(adata = adata, name = '3515_Lkb1_T1_Fam', prefix= '3515_Lkb1',
                     data_path = data_path, meta_df = meta_df)

    preprocess_adata_big_num(adata = adata, name = '3724_NT_T1',
                     data_path = data_path, meta_df = meta_df)



