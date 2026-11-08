from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scanpy as sc
from sklearn_extra.cluster import KMedoids
import torch


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



def barcode_distances_between_arrays(barcode_array1, barcode_array2):
    """
    Calculate all pairwise distances between two barcode arrays and return the distance matrix
    """
    num_cells1 = barcode_array1.shape[0]
    num_cells2 = barcode_array2.shape[0]

    # (num_cells1, num_cells2)
    distance_matrix = np.zeros((num_cells1, num_cells2))

    for i in range(num_cells1):
        barcode1 = barcode_array1[i, :]
        for j in range(num_cells2):
            barcode2 = barcode_array2[j, :]

            # scaled Hamming distance
            distance_matrix[i, j] = scaled_Hamming_distance(barcode1, barcode2)

    return distance_matrix

def barcode_distances(barcode_array):
    """
    Computes all pairwise lineage distances between barcodes
    """
    num_cells = barcode_array.shape[0]
    lineage_distances = np.zeros([num_cells, num_cells])

    for i in range(num_cells):
        barcode1 = barcode_array[i,:]
        for j in range(i):
            barcode2 = barcode_array[j]

            lineage_distances[i,j] = scaled_Hamming_distance(barcode1, barcode2)
            lineage_distances[j,i] = lineage_distances[i,j]
    return lineage_distances


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

