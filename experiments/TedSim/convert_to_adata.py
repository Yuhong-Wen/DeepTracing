import pathlib
import newick
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from pathlib import Path

def create_anndata(root: pathlib.Path, *, p_a: float) -> AnnData:
    """Create AnnData object from TedSim output file"""
    # Read the count matrix (transposed to cells × genes)
    counts = pd.read_csv(root / f"counts_tedsim_{p_a}.csv").T.values
    cell_meta = pd.read_csv(root / f"cell_meta_tedsim_{p_a}.csv", index_col=0)

    # Read barcode matrix and process missing values
    barcodes = pd.read_csv(
        root / f"character_matrix_{p_a}.txt",
        sep=" ",
    ).values
    barcodes[barcodes == "-"] = -1
    barcodes = barcodes.astype(int)


    adata = AnnData(
        X=counts.astype(float),
        obs=cell_meta,
        obsm={"barcodes": barcodes},
        dtype=np.float32
    )

    # Processing metadata
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    adata.obs["depth"] = adata.obs["depth"].astype(int)
    adata.obs["parent"] = adata.obs["parent"].astype("category")

    # Read the phylogenetic tree
    with open(root / f"tree_gt_bin_tedsim_{p_a}.newick", "r") as f:
        tree = newick.load(f)[0].newick

    # Add global metadata
    adata.uns["tree"] = tree
    adata.uns["sim_params"] = {
        "p_a": p_a,
    }

    assert (adata.obsm["barcodes"] == -1).sum() == 0, "Invalid barcode exists"
    return adata

if __name__ == "__main__":
    p_a_list = [0.2, 0.4, 0.6, 0.8]
    BASE_DIR = Path("/data/DeepTracing/experiments/TedSim/data")
    OUT_DIR = pathlib.Path("output_adatas")
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    for p_a in p_a_list:
        ROOT_DIR = BASE_DIR / f"{p_a}"
        try:
            adata = create_anndata(ROOT_DIR, p_a=p_a)
            sc.write(OUT_DIR / f"adata_{p_a}.h5ad", adata)
        except FileNotFoundError:
            print(f"File does not exist: p_a={p_a}")


    
