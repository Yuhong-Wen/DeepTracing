# -*- coding: utf-8 -*-
"""eTRACER preparation: raw scRNA + lineage mapping + Rep3-only distance indexes.

Runs both original preprocessing stages in memory. Final results_all are written
straight to BASE_DIR / "processed_data"; no Rep3 subdirectory or intermediate
whole-cohort h5ad is generated.
"""
from pathlib import Path
import re
import json
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
from scipy import sparse
BASE_DIR = Path("./data/Tumor_Immune_Evasion_eTRACER")
MTX_FILE = BASE_DIR / "scRNA/sc_UMI_count.mtx"
GENE_FILE = BASE_DIR / "scRNA/genes.tsv"
CELL_FILE = BASE_DIR / "scRNA/cells.tsv"
META_FILE = BASE_DIR / "scRNA/scRNA.metadata.csv"
LINEAGE_DIR = BASE_DIR / "LineageTracing/single_cell"
LINEAGE_FILE = LINEAGE_DIR / "sc_cell_lineage.csv"
OUTPUT_DIR = BASE_DIR / "processed_data"

# Only Rep3 (sample a3030)
REP_NAME = "Rep3"
REP_CONFIG = {"sample_key": "a3030", "allele_file": LINEAGE_DIR / "Replicate3_allele_table.csv"}
TIME_WEEK = {"T1": 0, "T2": 4, "T3": 6}
STAGE_NAME = {"T1": "BL", "T2": "EE", "T3": "LE"}
def prepare_adata():
    """Construct the raw-count AnnData and merge lineage labels (original stage 1)."""
    print("\n" + "1. Reading metadata")
    meta = pd.read_csv(META_FILE, index_col=0, low_memory=False, )
    meta.index = meta.index.astype(str)
    print("metadata shape:", meta.shape)
    print("metadata columns:", meta.columns.tolist())
    if "cellName" not in meta.columns:
        raise ValueError("cellName is missing from metadata.")
    meta["cellName"] = (meta["cellName"] .astype(str))
    print("\nmetadata head:")
    print(meta[["time", "cellName", "lineageGrp", "tumor_state", ]].head())
    # 2. Read lineage mapping

    print("\n" + "2. Reading lineage mapping")
    lineage = pd.read_csv(LINEAGE_FILE)
    lineage["cellBC"] = (lineage["cellBC"] .astype(str))
    lineage["lineageGrp"] = (lineage["lineageGrp"] .astype(str))
    print("lineage shape:", lineage.shape)
    print("n lineage cells:", lineage["cellBC"].nunique())
    print("n lineage groups:", lineage["lineageGrp"].nunique())
    # 3. Read expression matrix

    print("\n" + "3. Reading expression matrix")
    X = mmread(MTX_FILE)
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()
    genes = pd.read_csv(GENE_FILE, header=None, )[0].astype(str).values
    cells = pd.read_csv(CELL_FILE, header=None, )[0].astype(str).values
    print("raw matrix shape:", X.shape)
    print("n genes:", len(genes))
    print("n cells:", len(cells))
    # 4. Detect orientation

    print("\n" + "4. Detecting matrix orientation")
    if X.shape == (len(genes), len(cells), ):
        print("Detected gene x cell matrix.")
        X = X.T.tocsr()
    elif X.shape == (len(cells), len(genes), ):
        print("Detected cell x gene matrix.")
    else:
        raise ValueError(f"Unexpected matrix shape: {X.shape}; " f"genes={len(genes)}, cells={len(cells)}")
    print("final cell x gene shape:", X.shape)
    # 5. Match expression cells to metadata

    print("\n" + "5. Matching expression cells")
    cells = pd.Index(cells.astype(str))
    n_match_meta_index = (cells.isin(meta.index).sum())
    n_match_cellname = (cells.isin(meta["cellName"]).sum())
    print("match metadata index:", n_match_meta_index)
    print("match metadata cellName:", n_match_cellname)
    # 6. Align metadata

    if n_match_meta_index >= n_match_cellname:
        print("Using metadata index " "for expression alignment.")
        meta_aligned = (meta.reindex(cells))
    else:
        print("Using metadata cellName " "for expression alignment.")
        meta_by_cellname = (meta .drop_duplicates("cellName") .set_index("cellName", drop=False, ))
        meta_aligned = (meta_by_cellname .reindex(cells))
    unmatched = (meta_aligned .isna() .all(axis=1) .sum())
    print("unmatched cells:", unmatched)
    if unmatched > 0:
        print("[WARNING] Some cells " "could not be matched.")
    # 7. Create AnnData

    print("\n" + "6. Constructing AnnData")
    adata = ad.AnnData(X=X, obs=meta_aligned.copy(), )
    adata.obs_names = cells
    adata.var_names = pd.Index(genes.astype(str))
    adata.var_names_make_unique()
    print(adata)
    # 8. Store raw counts

    adata.layers["counts"] = adata.X.copy()
    # 9. Merge lineage mapping

    print("\n" + "7. Merging lineage information")
    lineage_map = (lineage .drop_duplicates("cellBC") .set_index("cellBC")["lineageGrp"])
    adata.obs["lineageGrp_from_table"] = (adata.obs["cellName"] .astype(str) .map(lineage_map))
    # 10. Compare lineage labels

    if (
        "lineageGrp"
        in
        adata.obs.columns
    ):
        valid = (adata.obs["lineageGrp"].notna() & adata.obs["lineageGrp_from_table"].notna())
        if valid.sum() > 0:

            agreement = (
                adata.obs.loc[
                    valid,
                    "lineageGrp",
                ].astype(str)
                ==
                adata.obs.loc[
                    valid,
                    "lineageGrp_from_table",
                ].astype(str)
            ).mean()
            print("lineage label agreement:", agreement)
        else:
            agreement = np.nan
            print("No overlapping lineage " "labels found.")
    else:
        agreement = np.nan
    # 11. Build final lineage label

    adata.obs[
        "lineageGrp_final"
    ] = (
        adata.obs[
            "lineageGrp_from_table"
        ]
    )
    if ("lineageGrp" in adata.obs.columns):
        missing = (adata.obs["lineageGrp_final"].isna())
        adata.obs.loc[missing, "lineageGrp_final", ] = (adata.obs.loc[missing, "lineageGrp"])
    # 12. Clean lineage labels

    invalid_labels = {
        "None",
        "nan",
        "NaN",
        "NA",
        "",
    }
    lineage_str = (adata.obs["lineageGrp_final"] .astype(str))
    adata.obs["has_lineage"] = (adata.obs["lineageGrp_final"].notna() & ~lineage_str.isin(invalid_labels))
    # 13. Statistics

    print("\n" + "8. Statistics")
    print("\nLineage availability:")
    print(adata.obs["has_lineage"].value_counts())
    if ("time" in adata.obs.columns):
        print("\nCells per time:")
        print(adata.obs["time"].value_counts())
        print("\nLineage availability " "per time:")
        print(pd.crosstab(adata.obs["time"], adata.obs["has_lineage"], ))
    n_clones = (adata.obs.loc[adata.obs["has_lineage"], "lineageGrp_final", ] .nunique())
    print("\nNumber of clones:", n_clones)
    # 14. Cross-time clone statistics

    if (
        "time"
        in
        adata.obs.columns
    ):
        tmp = (adata.obs.loc[adata.obs["has_lineage"], ["lineageGrp_final", "time", ], ])
        clone_time = (pd.crosstab(tmp["lineageGrp_final"], tmp["time"], ))
        n_times = (clone_time .gt(0) .sum(axis=1))
        print("\nNumber of stages " "per clone:")
        print(n_times .value_counts() .sort_index())
        print("\nClones appearing " "in >1 stage:", (n_times > 1).sum())
        print("\nClones appearing " "in all stages:", (n_times == clone_time.shape[1]).sum())
    # 15. Final summary

    print("\n" + "9. Final AnnData summary")
    print(adata)

    columns_to_show = [
        c
        for c in [
            "time",
            "cellName",
            "lineageGrp",
            "lineageGrp_from_table",
            "lineageGrp_final",
            "tumor_state",
            "has_lineage",
        ]
        if c in adata.obs.columns
    ]
    print(adata.obs[columns_to_show].head(10))
    return adata


def section(title):
    print(f"\n{title}")


def infer_replicate(cell_name):
    return "Rep3" if "a3030" in str(cell_name) else None
def infer_stage(cell_name):
    s = str(cell_name)
    if s.startswith("T1_"):
        return "T1"
    elif s.startswith("T2_"):
        return "T2"
    elif s.startswith("T3_"):
        return "T3"
    return None
def parse_indel_length(allele):
    """
    78:40D -> 40
    93:1I  -> 1
    85:15I92:1I -> 15
    """
    matches = re.findall(r"\d+:(\d+)[DI]", str(allele), )
    if len(matches) == 0:
        return None
    return max(int(x) for x in matches)
def build_binary_edit_matrix(allele_df, cell_order, ):
    """
    Reconstruct binary cell x edit matrix.

    This definition has already been validated against
    author's C1-4560.csv with 100% agreement.
    """
    section("Building binary edit matrix")
    cells = pd.Index(np.asarray(cell_order, dtype=str))
    cell_set = set(cells)
    df = allele_df[allele_df["cellBC"].astype(str).isin(cell_set)].copy()

    # exclude Acta1
    df = df[df["intBC"].astype(str)!= "Acta1"].copy()
    # exclude WT / None
    allele_str = df["allele"].astype(str)
    valid = ~allele_str.isin(["None", "nan", "NaN", "NA", ""])
    df = df[valid].copy()
    # >50 bp filter
    df["indel_length"] = (df["allele"].map(parse_indel_length))
    too_long = (df["indel_length"].notna()&(df["indel_length"] > 50))
    df = df[~too_long].copy()

    # exact feature definition validated against author data
    df["edit_event"] = (df["intBC"].astype(str) + "_" + df["allele"].astype(str))
    df = (df[["cellBC", "edit_event"]].drop_duplicates())
    events = pd.Index(sorted(df["edit_event"].unique()))
    cell_to_idx = pd.Series(np.arange(len(cells)), index=cells)
    event_to_idx = pd.Series(np.arange(len(events)), index=events)
    rows = (cell_to_idx.loc[df["cellBC"].astype(str)].values)
    cols = (event_to_idx.loc[df["edit_event"]].values)
    values = np.ones(len(df), dtype=np.uint8)
    X = sparse.csr_matrix((values, (rows, cols), ), shape=(len(cells), len(events), ), dtype=np.uint8, )
    X.data[:] = 1
    n_edit = np.asarray(X.sum(axis=1)).ravel()
    print("matrix shape:", X.shape)
    print("number of edit features:", len(events))
    print("zero-edit cells:", int((n_edit == 0).sum()))
    return X, events, n_edit
def get_unique_edit_profiles(X_edit):
    """
    Factorize cell edit vectors.

    Analogous to:

        pd.factorize(adata.obs["lineage"])

    but here lineage identity is the complete evolving
    barcode binary vector, NOT clone ID.

    Returns
    -------
    X_unique:
        unique binary edit profiles

    inverse_indices:
        for every cell, which unique profile it belongs to
    """
    section("Factorizing unique lineage edit profiles")

    # dense uint8
    #
    # Typical matrix is roughly 10k-20k cells x ~2000 events,
    # which is manageable.
    X_dense = (
        X_edit
        .toarray()
        .astype(np.uint8)
    )

    # Pack bits first:
    # greatly reduces memory and makes np.unique faster.
    X_packed = np.packbits(
        X_dense,
        axis=1,
    )

    unique_packed, first_indices, inverse_indices = (
        np.unique(
            X_packed,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
    )

    # recover original binary profiles
    X_unique = X_dense[first_indices]
    print("number of cells:", X_dense.shape[0])
    print("number of unique edit profiles:", X_unique.shape[0])
    print("compression ratio:", X_unique.shape[0] / X_dense.shape[0])
    del X_dense
    del X_packed
    del unique_packed
    gc.collect()
    return (X_unique, inverse_indices.astype(np.int32), first_indices, )
def compute_unique_hamming_block(X_unique, output_file, block_size=1000, ):
    section("Computing unique lineage distance")
    X = sparse.csr_matrix(X_unique.astype(np.uint16))
    n_unique = X.shape[0]
    n_features = X.shape[1]
    print("unique profiles:", n_unique)
    print("features:", n_features)
    size_gb = (n_unique * n_unique * 2 / 1024**3)
    print(f"distance matrix size: {size_gb:.3f} GB")

    # 直接创建标准 .npy 文件
    D = np.lib.format.open_memmap(
        output_file,
        mode="w+",
        dtype=np.uint16,
        shape=(n_unique, n_unique),
    )
    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.int32)
    for start in range(0, n_unique, block_size):
        end = min(start + block_size, n_unique)
        print(f"{start:,} -> {end:,} / {n_unique:,}")
        shared = (X[start:end] @ X.T).toarray().astype(np.int32)
        dist = (row_sum[start:end, None] + row_sum[None, :] - 2 * shared)
        D[start:end, :] = (dist.astype(np.uint16))
        D.flush()
    idx = np.arange(n_unique)
    assert np.all(D[idx, idx] == 0)
    rng = np.random.default_rng(0)
    ii = rng.integers(0, n_unique, size=10000)
    jj = rng.integers(0, n_unique, size=10000)
    assert np.all(D[ii, jj] == D[jj, ii])
    print("distance range:", D.min(), D.max())
    return D
def compute_unique_hamming(X_unique):
    """
    Compute distance only between unique lineage profiles.

    D[a,b] = number of binary edit features that differ.
    """
    X = sparse.csr_matrix(X_unique.astype(np.uint16))
    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.int32)
    shared = (X @ X.T).toarray().astype(np.int32)
    D = (row_sum[:, None] + row_sum[None, :] - 2 * shared).astype(np.uint16)
    n_unique = X.shape[0]
    idx = np.arange(n_unique)
    assert np.all(D[idx, idx] == 0)
    rng = np.random.default_rng(0)
    ii = rng.integers(0, n_unique, size=10000)
    jj = rng.integers(0, n_unique, size=10000)
    assert np.all(D[ii, jj] == D[jj, ii])
    print("distance range:", D.min(), D.max())
    print("unique lineage profiles:", X_unique.shape[0])
    print("edit features:", X_unique.shape[1])
    print("distance shape:", D.shape)
    print("distance range:", D.min(), D.max())
    return D
def build_time_index(adata):
    """
    Equivalent to factorizing lineage in your C. elegans example.

    Here unique times are fixed:
        BL = 0 week
        EE = 4 week
        LE = 6 week
    """
    unique_times = np.array([0, 4, 6], dtype=np.uint8)
    time_to_index = {0: 0, 4: 1, 6: 2}
    indices = np.asarray([time_to_index[int(x)] for x in adata.obs["time_week"]], dtype=np.int8, )

    # unique-level time distance
    D_time = np.abs(
        unique_times[:, None].astype(np.int16)
        -
        unique_times[None, :].astype(np.int16)
    ).astype(np.uint8)
    return (unique_times, indices, D_time)
def process_rep(adata, rep_name, config):
    section("PROCESSING Rep3")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Replicate subset
    adata_rep_all = (adata[adata.obs["replicate"]== rep_name].copy())
    print("all cells:", adata_rep_all.n_obs)

    # allele table
    allele_df = pd.read_csv(config["allele_file"], low_memory=False)
    allele_df["cellBC"] = (allele_df["cellBC"].astype(str))
    allele_cells = set(allele_df["cellBC"])

    # initial filtering
    cell_name = (adata_rep_all.obs["cellName"].astype(str))
    has_clone = (adata_rep_all.obs["has_lineage"].astype(bool))
    has_allele = (cell_name.isin(allele_cells))
    usable = (has_clone & has_allele)
    print("has clone:", int(has_clone.sum()))
    print("in allele table:", int(has_allele.sum()))
    print("initial usable:", int(usable.sum()))
    adata_rep = (adata_rep_all[usable].copy())

    # edit matrix
    cell_order = (adata_rep.obs["cellName"].astype(str).tolist())
    X_edit, events, n_edit = (build_binary_edit_matrix(allele_df, cell_order))

    # Remove zero-edit cells
    informative = (n_edit > 0)
    print("zero edit removed:", int((~informative).sum()))
    adata_rep = (adata_rep[informative].copy())
    X_edit = (X_edit[informative].tocsr())
    n_edit = (n_edit[informative])


    # Filter clones by size
    clone_col = "lineageGrp_final"
    clone_counts = (adata_rep.obs[clone_col].value_counts())
    valid_clones = clone_counts[clone_counts >= 30].index
    clone_mask = (adata_rep.obs[clone_col].isin(valid_clones).to_numpy())
    print("clones before size filtering:", adata_rep.obs[clone_col].nunique())
    print("cells before size filtering:", adata_rep.n_obs)
    print("clones retained (>=30 cells):", len(valid_clones))
    print("cells retained:", int(clone_mask.sum()))

    # subset AnnData
    adata_rep = (adata_rep[clone_mask].copy())

    # IMPORTANT:
    # subset edit matrix using exactly the same mask
    X_edit = (X_edit[clone_mask].tocsr())
    n_edit = (n_edit[clone_mask])

    # re-attach edit count
    adata_rep.obs["n_edit_events"] = n_edit
    adata_rep.obs["lineage_usable"] = True


    # UNIQUE lineage profiles
    (
        X_unique,
        lineage_indices,
        first_indices,
    ) = get_unique_edit_profiles(
        X_edit
    )

    # This is the key mapping used during training
    adata_rep.obs["lineage_index"] = lineage_indices

    # UNIQUE lineage distance
    lineage_distance = compute_unique_hamming_block(
        X_unique,
        OUTPUT_DIR / "lineage_distance.npy",
        block_size=1000,
    )

    # Time index + unique time distance
    (
        unique_times,
        time_indices,
        time_distance,
    ) = build_time_index(
        adata_rep
    )
    adata_rep.obs["time_index"] = time_indices

    # Strong alignment checks
    assert (len(lineage_indices) == adata_rep.n_obs)
    assert (len(time_indices) == adata_rep.n_obs)

    # reconstruct some random cell-pair distances
    rng = np.random.default_rng(0)
    n_check = min(10000, adata_rep.n_obs * 2)
    ii = rng.integers(0, adata_rep.n_obs, n_check)
    jj = rng.integers(0, adata_rep.n_obs, n_check)

    # lineage lookup
    test_lineage = (
        lineage_distance[
            lineage_indices[ii],
            lineage_indices[jj],
        ]
    )

    # direct edit-vector Hamming
    Xi = X_edit[ii].toarray()
    Xj = X_edit[jj].toarray()
    direct_lineage = (Xi != Xj).sum(axis=1)
    assert np.array_equal(test_lineage, direct_lineage)

    # time lookup
    test_time = (
        time_distance[
            time_indices[ii],
            time_indices[jj],
        ]
    )
    direct_time = np.abs(adata_rep.obs["time_week"].values[ii] - adata_rep.obs["time_week"].values[jj])
    assert np.array_equal(test_time, direct_time, )
    print("cell-distance index validation: PASSED")


    # Save
    section("Saving processed data")

    # AnnData:
    #
    # obs includes:
    # lineage_index
    # time_index
    # replicate
    # stage
    # time_week
    # clone
    adata_rep.write_h5ad(OUTPUT_DIR / "eTracer_lineage_usable.h5ad", compression="gzip")

    # unique time distance: only 3 x 3
    np.save(OUTPUT_DIR / "time_distance.npy", time_distance)

    # time values
    np.save(OUTPUT_DIR / "unique_times.npy", unique_times)

    # unique edit profiles
    sparse.save_npz(OUTPUT_DIR / "unique_edit_profiles.npz", sparse.csr_matrix(X_unique))

    # raw cell edit matrix, useful for auditing
    sparse.save_npz(OUTPUT_DIR / "cell_edit_matrix.npz", X_edit)

    # edit feature names
    pd.DataFrame({"edit_event": events}).to_csv(OUTPUT_DIR / "edit_events.tsv", sep="\t", index=False)

    # metadata
    obs_cols = [
        "cellName",
        "replicate",
        "stage",
        "stage_name",
        "time_week",
        "time_index",
        "lineageGrp_final",
        "lineage_index",
        "n_edit_events",
        "tumor_state",
    ]
    adata_rep.obs[obs_cols].to_csv(OUTPUT_DIR / "cells.tsv", sep="\t")

    report = {
        "replicate": rep_name,
        "n_cells": int(adata_rep.n_obs),
        "n_genes": int(adata_rep.n_vars),
        "n_edit_features": int(X_edit.shape[1]),
        "n_unique_lineage_profiles": int(X_unique.shape[0]),
        "lineage_distance_shape": list(lineage_distance.shape),
        "time_distance_shape": list(time_distance.shape),
        "lineage_index_min": int(lineage_indices.min()),
        "lineage_index_max": int(lineage_indices.max()),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(report, f, indent=2, )
    print(report)
    return report
def main():
    # Original stage 1, now returned in memory rather than saved/reloaded.
    adata = prepare_adata()

    # Original stage 2 annotations, then process only Rep3.
    section("Annotating experimental hierarchy")
    cell_name = adata.obs["cellName"].astype(str)
    adata.obs["replicate"] = cell_name.map(infer_replicate)
    adata.obs["stage"] = cell_name.map(infer_stage)
    adata.obs["stage_name"] = adata.obs["stage"].map(STAGE_NAME)
    adata.obs["time_week"] = adata.obs["stage"].map(TIME_WEEK)

    rep_mask = adata.obs["replicate"] == "Rep3"
    if not rep_mask.any():
        raise RuntimeError("No Rep3 cells found.")

    print(pd.crosstab(adata.obs["replicate"], adata.obs["stage_name"]))
    if not (adata.obs["replicate"] == "Rep3").any():
        raise RuntimeError("No Rep3 cells found.")
    if adata.obs["stage"].isna().any():
        raise RuntimeError("Some cells have unknown stage.")
    report = process_rep(adata, REP_NAME, REP_CONFIG)
    pd.DataFrame([report]).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    gc.collect()
    section("DONE")
if __name__ == "__main__":
    main()
