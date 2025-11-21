import torch
import os
import pandas as pd
import scanpy as sc
import numpy as np
from pathlib import Path
from sklearn.cross_decomposition import CCA
from preprocess import normalize

torch.manual_seed(0)

def CCA_corr_analyse(adata, latent):
    adata = adata.copy()
    cca_genes_latent = []

    for gene_idx in range(adata.X.shape[1]):  # adata.X.shape[1]
        gene_expression = adata.X[:, gene_idx].flatten()

        cca = CCA(n_components=1)
        cca.fit(gene_expression.reshape(-1, 1), latent)

        # Canonical Correlation
        canonical_corr = cca.score(gene_expression.reshape(-1, 1), latent)
        cca_genes_latent.append(canonical_corr)
    gene_names = adata.var_names
    cca_corr_df = pd.DataFrame(cca_genes_latent, columns=['cca_corr'], index=gene_names)
    cca_corr_df_sorted = cca_corr_df.sort_values(by='cca_corr', ascending=False)
    return cca_corr_df_sorted

def plot_top_genes(adata, rep_key, corr_df_sorted, top_genes_num):
    adata = adata.copy()

    sc.pp.neighbors(adata,use_rep=rep_key)
    sc.tl.umap(adata)
    top_genes = corr_df_sorted.head(top_genes_num).index
    gene_expression = adata[:, top_genes].X

    for i, gene in enumerate(top_genes):
        adata.obs[gene + '_expression'] = gene_expression[:, i].toarray().flatten()
    sc.settings.figdir = "./figures"
    os.makedirs(sc.settings.figdir, exist_ok=True)
    sc.settings.file_format_figs = "pdf"
    for gene in top_genes:
        sc.pl.embedding(adata,basis="umap", color=[gene + '_expression'], size=10,
                        save=f"_{gene}_{rep_key}_umap",
        cmap='viridis')




if __name__ == "__main__":
    data_path = 'path/to/your/data'
    name = '3515_Lkb1_T1_Fam'

    output_dir = Path(f"./results/tumor_{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pruned_adata = sc.read_h5ad(f'{data_path}/preprocessed/{name}_pruned_adata.h5ad')
    print(pruned_adata)

    keep = np.array((pruned_adata.X > 0).sum(axis=0) >= 10).flatten()
    print('genes expressed in 10 or more cells:', np.sum(keep))

    pruned_adata = pruned_adata[:, keep].copy()
    if hasattr(pruned_adata.X, "toarray"):
        dense_array = pruned_adata.X.toarray()
    else:
        dense_array = np.asarray(pruned_adata.X)
    pruned_adata.X = dense_array

    sc.pp.highly_variable_genes(pruned_adata, flavor="seurat_v3", n_top_genes=200, subset=True)
    adata = normalize(pruned_adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    gp_latent = np.loadtxt('.../3515_gp_latent.txt', delimiter=",")
    gp_cca_corr_df_sorted = CCA_corr_analyse(adata, gp_latent)
    print(gp_cca_corr_df_sorted.head())

    plot_top_genes(adata, "X_latent", gp_cca_corr_df_sorted, 2)