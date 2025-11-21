from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import scanpy as sc
import pylab as plt
import seaborn as sns
import pandas as pd
from scipy import sparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import umap
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

