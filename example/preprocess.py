from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scanpy as sc
from TedSim_data_prepare import scaled_Hamming_distance
from sklearn_extra.cluster import KMedoids


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


def inducing_points_select(adata, barcodes, select_feature, all_indices,
                           MIN_POINTS_PER_CLUSTER=2,
                           CELLS_PER_ADDITIONAL_POINT=50):
    inducing_points = []

    for t in np.unique(select_feature):
        position_in_cluster = np.where(select_feature == t)[0]
        if len(position_in_cluster) == 0:
            continue

        cluster_size = len(position_in_cluster)
        n_points = max(MIN_POINTS_PER_CLUSTER,
                       min(cluster_size,
                           MIN_POINTS_PER_CLUSTER + cluster_size // CELLS_PER_ADDITIONAL_POINT))

        cluster_barcodes = barcodes[position_in_cluster]

        if n_points == 1:
            dist_sum = np.zeros(cluster_size)
            for i in range(cluster_size):
                for j in range(cluster_size):
                    if i != j:
                        dist_sum[i] += scaled_Hamming_distance(
                            cluster_barcodes[i], cluster_barcodes[j])

            # Select the point with the smallest sum of distances as the center
            selected_position = position_in_cluster[np.argmin(dist_sum)]
            selected_indices = [all_indices[selected_position]]
        else:
            def custom_distance(b1, b2):
                return scaled_Hamming_distance(b1, b2)

            dist_matrix = np.zeros((cluster_size, cluster_size))
            for i in range(cluster_size):
                for j in range(i + 1, cluster_size):
                    d = custom_distance(cluster_barcodes[i], cluster_barcodes[j])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

            kmedoids = KMedoids(n_clusters=n_points, metric="precomputed",
                                init='k-medoids++', random_state=42)
            kmedoids.fit(dist_matrix)

            medoid_indices = kmedoids.medoid_indices_
            selected_positions = position_in_cluster[medoid_indices]
            selected_indices = all_indices[selected_positions].tolist()

        inducing_points.extend(selected_indices)

    return inducing_points

