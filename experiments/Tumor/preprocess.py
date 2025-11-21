from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scanpy as sc
from sklearn_extra.cluster import KMedoids
import dendropy



def get_top_down_cluster_ids(tree_path, labels, depth=2):
  tree = dendropy.Tree.get(path=tree_path, schema='newick')
  cluster_roots = [tree.seed_node]
  for i in range(depth):
    cluster_children = []
    for root in cluster_roots:
      if root.is_leaf():
        cluster_children.append(root)
      else:
        cluster_children.extend(list(root.child_node_iter()))
    cluster_roots = cluster_children
  labels_to_ids = {}
  cluster_id = 1
  for root in cluster_roots:
    for node in root.leaf_nodes():
      labels_to_ids[node.taxon.label] = cluster_id
    cluster_id += 1
  return np.array([node.label for node in cluster_roots]), np.array([labels_to_ids[x] for x in labels])


def normalize(adata,size_factors=True, logtrans_input=True, normalize_input=True):
    """
    Keep raw. X and perform normalization without removing genes or cells.

    Parameters:
    -Adata: AnnData object, single-cell data to be processed
    - logtrans_input: bool, Whether to perform logarithmic transformation on the data
    - normalize_input: bool, Whether to standardize the data

    Returns:
    -Adata: Processed AnnData object
    """

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


def inducing_points_select_tumor(adata, distance_matrix, select_feature, all_indices,
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
