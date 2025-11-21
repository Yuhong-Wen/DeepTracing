import torch
import scanpy as sc
from preprocess import barcode_distances_between_arrays, assign_subtree_by_depth, encode_mutation_patterns_to_obsm
import anndata as ad
from tree_utils import calculate_all_tree_distances
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import os
from skbio import DistanceMatrix
from skbio.tree import nj

torch.manual_seed(0)

def gunzip_file(gz_path, output_path=None):
    if output_path is None:
        output_path = gz_path.with_suffix('')
    if not output_path.exists():
        with gzip.open(gz_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    return output_path


def load_10x_sample(sample_dir: Path):
    barcodes_gz = next(sample_dir.glob("*barcodes.tsv.gz"))
    features_gz = next(sample_dir.glob("*features.tsv.gz"))
    matrix_gz = next(sample_dir.glob("*matrix.mtx.gz"))

    barcodes = gunzip_file(barcodes_gz)
    features = gunzip_file(features_gz)
    matrix = gunzip_file(matrix_gz)

    adata = sc.read_mtx(str(matrix)).T
    adata.var_names = pd.read_csv(features, header=None, sep="\t")[1].values
    adata.obs_names = pd.read_csv(barcodes, header=None)[0].values

    adata.obs['sample'] = sample_dir.name
    return adata


def process_E_rep_data(
        base_path,
        E_rep_id,
        string_to_remove="_1"
):
    """
    Process single-cell mutation data for different E_rep samples
    Parameters:
    base_path: Base directory path where all data files are located
    E_rep_id: Identifier for the sample (e.g., "E15_rep1")
    string_to_remove: String to remove from cell names, default is "_1"

    Returns:
    adata_subset: Processed AnnData object
    """
    base_path = Path(base_path)

    # Read data files
    print(f"Reading data for {E_rep_id}...")
    mutation_matrix = pd.read_csv(base_path / f"{E_rep_id}_seq_data_mt.csv")
    cell_barcodes = pd.read_csv(base_path / f"{E_rep_id}_CellBC.csv")
    cell_type = pd.read_csv(base_path / f"{E_rep_id}_Celltype.csv")
    cell_patternv2 = pd.read_csv(base_path / f"{E_rep_id}_patternv2.csv")
    cell_patternv1 = pd.read_csv(base_path / f"{E_rep_id}_patternv1.csv")

    # Merge data
    mutation_matrix['Cell.BC'] = cell_barcodes['x']
    mutation_matrix['Cell.type'] = cell_type['x']
    mutation_matrix['patternv2'] = cell_patternv2['x']
    mutation_matrix['patternv1'] = cell_patternv1['x']
    mutation_matrix = mutation_matrix.set_index('Cell.BC')

    # Load 10x data
    adata = load_10x_sample(base_path / f"GSM6422662_snapCREST_{E_rep_id}")

    print(f"Processing data: {E_rep_id}")
    print(f"Original AnnData: {adata}")

    # Standardize cell names
    mutation_matrix.index = mutation_matrix.index.str.replace(string_to_remove, "", regex=False)
    adata.obs_names = adata.obs_names.str.replace(string_to_remove, "", regex=False)

    # Find shared cells
    shared_cells = mutation_matrix.index.intersection(adata.obs_names)
    print(f"Found {len(shared_cells)} shared cells")

    # Subset data
    mutation_subset = mutation_matrix.loc[shared_cells]
    adata_subset = adata[shared_cells].copy()

    # Ensure consistent ordering
    mutation_subset = mutation_subset.loc[adata_subset.obs_names]

    # Add mutation data and annotations to AnnData object
    numeric_cols = mutation_subset.drop(columns=['Cell.type', 'patternv2', 'patternv1']).columns
    adata_subset.obsm['mutation_barcode'] = mutation_subset[numeric_cols].values
    adata_subset.obs['Cell_type'] = mutation_subset['Cell.type'].values
    adata_subset.obs['patternv2'] = mutation_subset['patternv2'].values
    adata_subset.obs['patternv1'] = mutation_subset['patternv1'].values

    # Save results
    output_filename = base_path / f"adata_{E_rep_id}.h5ad"
    adata_subset.write(output_filename)
    print(f"Processed data saved to: {output_filename}")
    return adata_subset



def run_phylogenetic_analysis(
        adata,
        mode="both",
        obsm_key="X_clone_barcode_mt",
        max_depth=5,
        output_dir="./results",
        adata_output_path="./adata_combined_E11_E15.h5ad"
):
    encode_mutation_patterns_to_obsm(adata, mode=mode, obsm_key=obsm_key)

    barcodes_all = np.asarray(adata.obsm[obsm_key])
    unique_barcodes, first_idx, inv_barcode = np.unique(
        barcodes_all, axis=0, return_index=True, return_inverse=True
    )
    indices_all_barcode = inv_barcode

    distance_output_dir = Path(output_dir)
    distance_output_dir.mkdir(parents=True, exist_ok=True)

    distance_matrix_file = distance_output_dir / "unique_distance_matrix_barcode_both.npy"
    if not os.path.isfile(distance_matrix_file):
        unique_distance_matrix_barcode = barcode_distances_between_arrays(unique_barcodes, unique_barcodes)
        np.save(distance_matrix_file, unique_distance_matrix_barcode)
    unique_distance_matrix_barcode = np.load(distance_matrix_file)

    cell_ids = [f"barcode_{i}" for i in range(len(unique_barcodes))]
    dm = DistanceMatrix(unique_distance_matrix_barcode, ids=cell_ids)
    tree = nj(dm)
    tree = tree.root_at_midpoint()

    barcode_to_subtree = assign_subtree_by_depth(tree, max_depth=max_depth)
    barcode_labels = [barcode_to_subtree.get(f"barcode_{i}", 0) for i in range(len(unique_barcodes))]
    cell_subtree_labels = [barcode_labels[idx] for idx in indices_all_barcode]
    adata.obs["subtree"] = cell_subtree_labels

    all_cell_ids = [f"barcode_{idx}" for idx in range(len(unique_barcodes))]
    tree_distance_matrix = calculate_all_tree_distances(tree, cell_ids)
    print(f"Tree distance matrix shape: {tree_distance_matrix.shape}")

    tree_distance_matrix_file = distance_output_dir / "tree_distance_matrix.npy"
    np.save(tree_distance_matrix_file, tree_distance_matrix)

    adata.write(adata_output_path)

    print(f"Tree distance matrix saved to: {tree_distance_matrix_file}")
    print(f"AnnData object saved to: {adata_output_path}")

    return tree_distance_matrix


if __name__ == "__main__":
    adata_E11 = process_E_rep_data(base_path="/data/DeepTracing/experiments/mouse_vMB/data", E_rep_id="E11_rep1")
    adata_E11.obs["timepoint"] = 11.5
    adata_E15 = process_E_rep_data(base_path="/data/DeepTracing/experiments/mouse_vMB/data", E_rep_id="E15_rep1")
    adata_E15.obs["timepoint"] = 15.5

    adata_E11.var_names_make_unique()
    adata_E15.var_names_make_unique()

    adata = ad.concat([adata_E11, adata_E15], join="outer", label="batch", keys=["E11.5", "E15.5"])

    if hasattr(adata.X, "toarray"):
        dense_array = adata.X.toarray()
    else:
        dense_array = np.asarray(adata.X)
    adata.X = dense_array

    adata.obs['clone_barcode'] = adata.obs['patternv2'].astype(str) + '|' + adata.obs['patternv1'].astype(str)
    clone_counts = adata.obs['clone_barcode'].value_counts()
    cell_num = 6
    valid_clones = clone_counts[clone_counts >= cell_num].index
    adata_clones = adata[adata.obs['clone_barcode'].isin(valid_clones)].copy()
    adata = adata_clones

    tree_distance_matrix = run_phylogenetic_analysis(adata)

