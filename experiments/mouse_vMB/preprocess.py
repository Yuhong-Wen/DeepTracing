import numpy as np
import torch
from sklearn_extra.cluster import KMedoids
import pandas as pd
import scanpy as sc

def encode_mutation_patterns_to_obsm(
    adata,
    mode: str = "both",
    v1_col: str = "patternv1",
    v2_col: str = "patternv2",
    obsm_key: str = "X_clone_barcode_mt",
    none_tokens=("NONE","NA","",None,"NaN","nan","NAN"),
    start_index: int = 1,
    sort_tokens: bool = True
):
    """
    Encode the mutation string into an integer matrix and write it to adata. obsm [obsm-key].
    -Mode="v1": encodes only patterned v1
    -Mode="v2": encodes only patternv2
    -Mode="both": First concatenate V2 | V1, then split and encode them together (aligned with your R-side merge logic)
    Simultaneously write the mapping dictionary and column names into adata.uns for reproduction.
    """
    if mode not in {"v1","v2","both"}:
        raise ValueError("mode must be one of {'v1','v2','both'}")

    if mode == "v1":
        series = adata.obs.get(v1_col)
        if series is None:
            raise KeyError(f"Column '{v1_col}' not found in adata.obs")
        raw_patterns = series.astype(str).tolist()
    elif mode == "v2":
        series = adata.obs.get(v2_col)
        if series is None:
            raise KeyError(f"Column '{v2_col}' not found in adata.obs")
        raw_patterns = series.astype(str).tolist()
    else:  # both
        s1 = adata.obs.get(v2_col)
        s2 = adata.obs.get(v1_col)
        if s1 is None or s2 is None:
            raise KeyError(f"Columns '{v2_col}' and/or '{v1_col}' not found in adata.obs")
        raw_patterns = (s1.astype(str) + "|" + s2.astype(str)).tolist()

    token_rows = []
    for s in raw_patterns:
        if s in none_tokens:
            token_rows.append([])
            continue
        if mode == "both":
            parts = []
            for seg in s.split("|"):
                if seg in none_tokens:
                    continue
                parts.extend([t for t in seg.split("_") if t not in none_tokens])
            token_rows.append(parts)
        else:
            parts = [t for t in s.split("_") if t not in none_tokens]
            token_rows.append(parts)

    max_len = max((len(r) for r in token_rows), default=0)
    if max_len == 0:
        encoded = np.zeros((adata.n_obs, 1), dtype=int)
        adata.obsm[obsm_key] = encoded
        adata.uns[f"{obsm_key}_map"] = {}
        adata.uns[f"{obsm_key}_columns"] = ["mut_1"]
        return

    padded_rows = [r + ["NONE"]*(max_len - len(r)) for r in token_rows]

    flat = np.concatenate([np.array(r, dtype=object) for r in padded_rows])
    uniq = pd.unique(flat)
    valid_tokens = [u for u in uniq if u not in none_tokens]
    if sort_tokens:
        valid_tokens = sorted(valid_tokens)

    token2id = {tok: (i + start_index) for i, tok in enumerate(valid_tokens)}
    for nt in none_tokens:
        token2id[nt] = 0

    encoded = np.empty((len(padded_rows), max_len), dtype=int)
    for i, row in enumerate(padded_rows):
        encoded[i, :] = [token2id.get(tok, 0) for tok in row]

    adata.obsm[obsm_key] = encoded

    adata.uns[f"{obsm_key}_map"] = token2id
    adata.uns[f"{obsm_key}_columns"] = [f"mut_{j+1}" for j in range(max_len)]



def barcode_distances_between_arrays(barcode_array1, barcode_array2):
    """
    计算两个 barcode 数组之间的所有成对距离，返回距离矩阵
    """
    num_cells1 = barcode_array1.shape[0]
    num_cells2 = barcode_array2.shape[0]

    # 初始化距离矩阵，大小为 (num_cells1, num_cells2)
    distance_matrix = np.zeros((num_cells1, num_cells2))

    for i in range(num_cells1):
        barcode1 = barcode_array1[i, :]
        for j in range(num_cells2):
            barcode2 = barcode_array2[j, :]

            # 计算 scaled Hamming distance
            distance_matrix[i, j] = scaled_Hamming_distance(barcode1, barcode2)

    return distance_matrix


def scaled_Hamming_distance(barcode1, barcode2):
    """
    Computes the distance between two barcodes, adjusted for

    (1) the number of sites where both cells were measured and

    (2) distance between two scars is twice the distance from

        scarred to unscarred
    """
    shared_indices = (barcode1 >= 0) & (barcode2 >= 0)
    b1 = barcode1[shared_indices]

    # There may not be any sites where both were measured
    if len(b1) == 0:
        return np.nan
    b2 = barcode2[shared_indices]

    differences = b1 != b2
    double_scars = differences & (b1 != 0) & (b2 != 0)

    if isinstance(differences, torch.Tensor):
        differences = differences.cpu().numpy()
    if isinstance(double_scars, torch.Tensor):
        double_scars = double_scars.cpu().numpy()

    return (np.sum(differences) + np.sum(double_scars)) / len(b1)


def normalize(adata,size_factors=True, logtrans_input=True, normalize_input=True):
    adata.raw = adata.copy()
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def assign_subtree_by_depth(tree, max_depth=3):
    barcode_to_subtree = {}
    subtree_id = 1

    def traverse(node, depth):
        nonlocal subtree_id
        if depth == max_depth or node.is_tip():
            for tip in node.tips():
                barcode_to_subtree[tip.name] = subtree_id
            subtree_id += 1
        else:
            for child in node.children:
                traverse(child, depth + 1)

    traverse(tree.root(), depth=0)
    return barcode_to_subtree


def state_classification(adata, early_cell_types, late_cell_types, early_state_name, late_state_name):
    adata.obs['states'] = None
    adata.obs.loc[adata.obs['Cell_type'].isin(early_cell_types), 'states'] = early_state_name
    adata.obs.loc[adata.obs['Cell_type'].isin(late_cell_types), 'states'] = late_state_name

    adata.obs['states'] = adata.obs['states'].fillna('unknown')
    return adata

def inducing_points_select(adata, distance_matrix, select_feature, all_indices,
                                 MIN_POINTS_PER_CLUSTER=1,
                                 CELLS_PER_ADDITIONAL_POINT=50):
    inducing_points = []

    for t in np.unique(select_feature):
        position_in_cluster = np.where(select_feature == t)[0]
        cluster_size = len(position_in_cluster)
        if cluster_size == 0:
            continue

        n_points = max(MIN_POINTS_PER_CLUSTER,
                       min(cluster_size,
                           MIN_POINTS_PER_CLUSTER + cluster_size // CELLS_PER_ADDITIONAL_POINT))

        cluster_dist_matrix = distance_matrix[np.ix_(position_in_cluster, position_in_cluster)]

        if n_points == 1:
            dist_sums = cluster_dist_matrix.sum(axis=1) - np.diag(cluster_dist_matrix)
            min_idx_in_cluster = np.argmin(dist_sums)
            selected_index = all_indices[position_in_cluster[min_idx_in_cluster]]
            inducing_points.append(selected_index)
        else:
            kmedoids = KMedoids(n_clusters=n_points,
                                metric="precomputed",
                                init='k-medoids++',
                                random_state=42)
            kmedoids.fit(cluster_dist_matrix)
            medoid_indices_in_cluster = kmedoids.medoid_indices_
            selected_indices = all_indices[position_in_cluster[medoid_indices_in_cluster]].tolist()
            inducing_points.extend(selected_indices)
    return inducing_points