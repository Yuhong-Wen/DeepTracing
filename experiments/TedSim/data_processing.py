import pathlib
import newick
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import lineageot.inference as lot_inf
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union
import networkx as nx
from scipy.sparse import issparse
from Bio import Phylo
import io
from copy import deepcopy
import anndata
import pickle



def create_anndata(root: pathlib.Path, *, p_a: float) -> AnnData:
    """Create an AnnData object from TedSim output files."""
    # Read count matrix (transposed to cells x genes)
    counts = pd.read_csv(root / f"counts_tedsim_{p_a}.csv").T.values
    cell_meta = pd.read_csv(root / f"cell_meta_tedsim_{p_a}.csv", index_col=0)

    # Read barcode matrix and handle missing values
    barcodes = pd.read_csv(
        root / f"character_matrix_{p_a}.txt",
        sep=" ",
    ).values
    barcodes[barcodes == "-"] = -1
    barcodes = barcodes.astype(int)

    # Create AnnData object
    adata = AnnData(
        X=counts.astype(float),
        obs=cell_meta,
        obsm={"barcodes": barcodes},
        dtype=np.float32
    )

    # Process metadata
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    adata.obs["depth"] = adata.obs["depth"].astype(int)
    adata.obs["parent"] = adata.obs["parent"].astype("category")

    # Read phylogenetic tree
    with open(root / f"tree_gt_bin_tedsim_{p_a}.newick", "r") as f:
        tree = newick.load(f)[0].newick

    # Add global metadata
    adata.uns["tree"] = tree
    adata.uns["sim_params"] = {
        "p_a": p_a,
    }

    assert (adata.obsm["barcodes"] == -1).sum() == 0, "Invalid barcodes present"
    return adata

def prepare_data(fpath: Path,
        *,
        ttp: float = 100.0,

):
    import scanpy as sc
    adata = sc.read(fpath)
    tree = adata.uns["tree"]
    rna = adata.X.A.copy() if issparse(adata.X) else adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    # if ssr is not None:
    #     barcodes = stochastic_silencing(barcodes, stochastic_silencing_rate=ssr)
    true_tree = build_true_trees(
        rna,
        barcodes,
        meta=adata.obs,
        tree=tree,
        ttp=ttp
    )

    return true_tree

def build_true_trees(
        rna: np.ndarray,
        barcodes: np.ndarray,
        meta: pd.DataFrame,
        *,
        tree: str,
        ttp: float = 100.0):
    print(f"rna.shape[0]: {rna.shape[0]}")
    print(f"Type of rna.shape[0]: {type(rna.shape[0])}")
    # cell_arr_adata = [rna[nid] for nid in range(rna.shape[0])]
    seed = np.random.randint(2**31)
    cell_arr_adata = [
    lot_inf.sim.Cell(rna[nid], barcodes[nid], seed) for nid in range(rna.shape[0])
]

    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = newick2digraph(tree)
    G = annotate(G, cell_arr_adata, metadata, ttp=ttp)
    for s, t in G.edges:
        sn, tn = G.nodes[s], G.nodes[t]
        assert is_valid_edge(sn, tn), (s, t)
    tree = G
    return tree


def process_data(
        rna_arrays: Mapping[Literal["early", "late"], np.ndarray],
        *,
        n_pcs: int = 30,
        lognorm: bool = True,
        pca: bool = True,
) -> np.ndarray:
    adata_early = AnnData(rna_arrays["early"], dtype=float)
    adata_late = AnnData(rna_arrays["late"], dtype=float)
    import scanpy as sc
    adata = sc.concat([adata_early, adata_late], label="time", keys=["0", "1"])
    if lognorm:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if pca:
        sc.pp.highly_variable_genes(adata)
        sc.tl.pca(adata, use_highly_variable=False)

        return adata.obsm["X_pca"][:, :n_pcs].astype(float, copy=True)
    else:
        return adata.X.astype(float, copy=True)



def cut_at_depth(G: nx.DiGraph, *, max_depth: Optional[int] = None) -> nx.DiGraph:
    if max_depth is None:
        return deepcopy(G)
    selected_nodes = [n for n in G.nodes if G.nodes[n]["node_depth"] <= max_depth]
    G = deepcopy(G.subgraph(selected_nodes).copy())

    # relabel because of LOT
    leaves = sorted(n for n in G.nodes if not list(G.successors(n)))
    for new_name, n in enumerate(leaves):
        G = nx.relabel_nodes(G, {n: new_name}, copy=False)
    return G

def is_leaf(G: nx.DiGraph, n: Any) -> bool:
    return not list(nx.descendants(G, n))



def is_valid_edge(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
    r"""Assumes the following state tree:

       /-4
      7
     / \-3
    5
     \ /-1
      6
       \-2
    """
    state_tree = nx.from_edgelist([(5, 6), (5, 7), (6, 1), (6, 2), (7, 3), (7, 4)])
    try:
        # parent, cluster, depth
        p1, c1, d1 = n1["parent"], n1["cluster"], n1["depth"]
        p2, c2, d2 = n2["parent"], n2["cluster"], n2["depth"]
    except KeyError:
        # no metadata, assume true
        return True

    # root, anything is permitted
    if (p1, c1, d1) == (5, 6, 0):
        return True

    # sanity checks
    assert p1 in [5, 6, 7], p1
    assert p2 in [5, 6, 7], p2
    assert c1 in [1, 2, 3, 4, 6, 7], c1
    assert c2 in [1, 2, 3, 4, 6, 7], c2

    if p1 == p2:
        if c1 == c2:
            # check if depth of a parent is <=
            return d1 <= d2
        # sanity check that clusters are valid siblings
        return (c1, c2) in state_tree.edges

    # parent-cluster relationship
    assert c1 == p2, (c1, p2)
    # valid transition
    assert (c1, c2) in state_tree.edges, (c1, c2)
    return True


def newick2digraph(tree: str) -> nx.DiGraph:
    def trav(clade, prev: Any, depth: int) -> None:
        nonlocal cnt
        if depth == 0:
            name = "root"
        else:
            name = clade.name
            if name is None:
                name = cnt
                cnt -= 1
            else:
                name = int(name[1:]) - 1

        G.add_node(name, node_depth=depth)
        if prev is not None:
            G.add_edge(prev, name)

        for c in clade.clades:
            trav(c, name, depth + 1)

    G = nx.DiGraph()
    cnt = -1
    tree = Phylo.read(io.StringIO(tree), "newick")
    trav(tree.clade, None, 0)

    start = max([n for n in G.nodes if n != "root"]) + 1
    for n in list(nx.dfs_preorder_nodes(G)):
        if n == "root":
            pass
        if is_leaf(G, n):
            continue

        assert start not in G.nodes
        G = nx.relabel_nodes(G, {n: start}, copy=False)
        start += 1

    return G


def annotate(
        G: nx.DiGraph,
        cell_arr_data: List[lot_inf.sim.Cell],
        meta: List[Dict[str, Any]],
        ttp: int = 100,
) -> nx.DiGraph:
    G = G.copy()
    n_leaves = len([n for n in G.nodes if not len(list(G.successors(n)))])
    assert (n_leaves & (n_leaves - 1)) == 0, f"{n_leaves} is not power of 2"
    max_depth = int(np.log2(n_leaves))

    n_expected_nodes = 2 ** (max_depth + 1) - 1
    assert len(G) == n_expected_nodes, "graph is not a full binary tree"

    if len(cell_arr_data) != n_expected_nodes:  # missing root, add after observed nodes
        seed = np.random.randint(2**31)
        dummy_cell = lot_inf.sim.Cell([], [], seed)
        cell_arr_data += [dummy_cell]
        meta += [{}]

    for nid in G.nodes:
        depth = G.nodes[nid]["node_depth"]
        metadata = {
            **meta[nid],  # contains `depth`, which is different from `node_depth`
            "cell": cell_arr_data[nid],
            "nid": nid,
            "time": depth * ttp,
            "time_to_parent": ttp,
        }
        G.nodes[nid].update(metadata)

    for eid in G.edges:
        G.edges[eid].update({"time": ttp})
    return nx.relabel_nodes(G, {n_leaves: "root"}, copy=False)


def truncate_tree(G: nx.DiGraph, depth: int):
    """
    Processing the tree 'G' returned by the 'annotate' function:
    Truncate the tree at 'depth' and extract all nodes at or above that depth (retaining the original node ID).
    Return all information of each node, including gene expression, barcode, time, etc.

    : paramG: annotated nx. DiGraph cell evolutionary tree
    : param depth: Truncated depth, extract this depth and all nodes before it
    Return: (Induction point information, overall dataset information)
    """
    all_nodes = {n: G.nodes[n] for n in G.nodes if G.nodes[n]["node_depth"] <= depth}

    all_nodes_filtered = {n: data for n, data in all_nodes.items() if
                          not isinstance(data["cell"].x, list) or len(data["cell"].x) > 0}

    return all_nodes_filtered

def create_adata_from_tree(all_data):
    """
    """
    def construct_adata(data):
        """ Convert dictionary data to AnnData object (retaining original cell ID, including cluster information) """
        cell_ids = list(data.keys())

        X = np.array([data[n]["cell"].x for n in cell_ids])

        obs = pd.DataFrame({
            "time": [data[n]["time"] for n in cell_ids],
            "node_depth": [data[n]["node_depth"] for n in cell_ids],
            "cluster": [data[n].get("cluster", -1) for n in cell_ids]
        }, index=cell_ids)

        obsm = {
            "barcodes": np.array([data[n]["cell"].barcode for n in cell_ids])
        }
        adata = anndata.AnnData(X=X, obs=obs, obsm=obsm)

        return adata

    adata_all = construct_adata(all_data)
    return adata_all


def subtree_clusters(adata, tree, depth, obs_feature):
    adata = adata.copy()
    split_depth = depth
    tree = tree.copy()
    subtree_clusters = []
    for node_id in adata.obs.index:
        current_node = node_id
        while True:
            if current_node == 'root':
                subtree_clusters.append("unassigned")
                break
            try:
                current_depth = tree.nodes[int(current_node)]["node_depth"]
            except (KeyError, ValueError):
                subtree_clusters.append("unassigned")
                break
            if current_depth == split_depth:
                subtree_clusters.append(current_node)
                break
            predecessors = list(tree.predecessors(int(current_node)))
            if not predecessors:  # Handle root node case
                subtree_clusters.append("unassigned")
                break
            current_node = predecessors[0]
    # **Step 2: Ensure subtree is categorical data**
    adata.obs[obs_feature] = pd.Categorical(subtree_clusters).astype(str)
    return adata



if __name__ == "__main__":
    p_a_list = [0.2, 0.4, 0.6, 0.8]
    BASE_DIR = Path("./data")
    OUT_DIR = pathlib.Path("output_data")
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    for p_a in p_a_list:
        ROOT_DIR = BASE_DIR / f"{p_a}"
        try:
            adata = create_anndata(ROOT_DIR, p_a=p_a)
            sc.write(OUT_DIR / f"raw_adata_{p_a}.h5ad", adata)
        except FileNotFoundError:
            print(f"File not found: p_a={p_a}")

        data_file = f"./output_data/raw_adata_{p_a}.h5ad"
        adata_raw = sc.read(data_file)
        print(adata_raw.X.shape)

        TTP = 100
        max_depth = 12
        tree = prepare_data(data_file, ttp=TTP)
        tree_file = OUT_DIR / f"tree_{p_a}.pkl"

        with open(tree_file, "wb") as f:
            pickle.dump(tree, f)

        with open(tree_file, "rb") as f:
            loaded_tree = pickle.load(f)

        tree = loaded_tree

        all_data = truncate_tree(tree, depth=max_depth)
        adata_all = create_adata_from_tree(all_data)

        mask = adata_all.obs['node_depth'].isin([max_depth])
        adata_need = adata_all[mask, :].copy()
        adata_need.X = np.array(adata_need.X) if isinstance(adata_need.X, pd.Series) else adata_need.X

        adata_need = subtree_clusters(adata_need, tree, 4, 'subtree')
        print("all dataset:", adata_need)

        adata_need.obs.index = adata_need.obs.index.astype(str)
        for col in adata_need.obs.columns:
            if adata_need.obs[col].dtype == 'object':
                adata_need.obs[col] = adata_need.obs[col].astype(str)

        adata_need.write(OUT_DIR / f'preprocessed_adata_{p_a}.h5ad')