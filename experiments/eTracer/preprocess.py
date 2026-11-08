from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scanpy as sc
import numpy as np
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

def inducing_points_select_from_dist(
    select_feature,
    all_indices,
    full_dist_matrix,
    MIN_POINTS_PER_CLUSTER=2,
    CELLS_PER_ADDITIONAL_POINT=200,
):
    select_feature = np.asarray(select_feature)
    all_indices = np.asarray(all_indices, dtype=int)

    assert len(select_feature) == len(all_indices)
    assert all_indices.min() >= 0
    assert all_indices.max() < full_dist_matrix.shape[0]
    assert full_dist_matrix.shape[0] == full_dist_matrix.shape[1]

    inducing_points = []

    for t in np.unique(select_feature):

        positions = np.where(select_feature == t)[0]

        if len(positions) == 0:
            continue

        cluster_size = len(positions)

        cluster_indices = all_indices[positions]

        candidate_indices = np.unique(cluster_indices)

        n_candidates = len(candidate_indices)

        if n_candidates == 0:
            continue

        n_points = max(
            MIN_POINTS_PER_CLUSTER,
            int(np.ceil(cluster_size / CELLS_PER_ADDITIONAL_POINT)),
        )

        n_points = min(
            n_points,
            n_candidates,
        )

        cluster_dist_matrix = full_dist_matrix[
            np.ix_(candidate_indices, candidate_indices)
        ]

        if n_points == 1:
            medoid_local_idx = np.argmin(
                cluster_dist_matrix.sum(axis=1)
            )

            selected_indices = candidate_indices[
                [medoid_local_idx]
            ]

        else:
            kmedoids = KMedoids(
                n_clusters=n_points,
                metric="precomputed",
                init="k-medoids++",
                random_state=42,
            )

            kmedoids.fit(cluster_dist_matrix)

            selected_indices = candidate_indices[
                kmedoids.medoid_indices_
            ]

        inducing_points.extend(
            selected_indices.tolist()
        )

    return inducing_points
