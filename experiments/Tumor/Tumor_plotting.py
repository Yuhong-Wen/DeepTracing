import scanpy as sc
from preprocess import normalize, get_top_down_cluster_ids
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.cross_decomposition import CCA
import pandas as pd
import os
torch.manual_seed(0)


def plt_show_adata(adata, color_feature, size, name, results_path,
                      n_pca_comps=50, n_neighbors=15, metric='euclidean',
                      random_state=42, recompute=False):

    if recompute or 'X_pca' not in adata.obsm:
        sc.tl.pca(
            adata,
            n_comps=n_pca_comps,
            svd_solver='arpack',
            random_state=random_state
        )

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pca_comps,
        metric=metric,
        use_rep='X_pca',
        random_state=random_state
    )
    sc.tl.umap(adata, random_state=random_state)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    sc.pl.umap(
        adata,
        color=[color_feature],
        size=size,
        legend_loc="right margin",
        frameon=False,
        legend_fontsize=8,
        legend_fontweight="normal",
        ax=ax,
        show=False,
        title='UMAP (based on PCA)'
    )

    plt.subplots_adjust(left=0.05, right=0.85)

    if ax.get_legend():
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.legend(
            handles=handles,
            labels=labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            fontsize=8,
            frameon=False
        )

    plt.savefig(
        f'{results_path}/{name}_adata_{color_feature}_UMAP.pdf',
        bbox_inches='tight'
    )
    plt.show()


def plot_umap_embeddings(adata, rep_keys, color_keys, size, name, results_path):
    for rep_key in rep_keys:

        sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=15)
        sc.tl.umap(adata)

        for color_key in color_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            sc.pl.umap(adata, color=[color_key], size=size, legend_loc="right margin",
                       frameon=False,
                       legend_fontsize=8,
                       legend_fontweight="normal",
                       ax=ax,
                       show=False)
            ax.set_title(f"{rep_key} : {color_key}", fontsize=14)
            plt.subplots_adjust(left=0.1, right=0.8)
            if ax.get_legend():

                handles, labels = ax.get_legend_handles_labels()

                ax.get_legend().remove()

                legend = ax.legend(handles=handles, labels=labels, loc="center left",
                                    bbox_to_anchor=(1.02, 0.5),
                                    ncol=1,
                                    fontsize=8, frameon=False)
            plt.savefig(f'{results_path}/{name}_adata_embeddings_{rep_key}_{color_key}.pdf', dpi=300, bbox_inches="tight")
            plt.show()


def CCA_corr_analyse(adata, latent):
    adata = adata.copy()
    cca_genes_latent = []
    for gene_idx in range(adata.X.shape[1]):
        gene_expression = adata.X[:, gene_idx].flatten()
        cca = CCA(n_components=1)
        cca.fit(gene_expression.reshape(-1, 1), latent)
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
    sc.settings.figdir = f"./results_3515/Figures"
    os.makedirs(sc.settings.figdir, exist_ok=True)
    sc.settings.file_format_figs = "pdf"
    for gene in top_genes:
        sc.pl.embedding(adata,basis="umap", color=[gene + '_expression'], size=10,
                        save=f"_{gene}_{rep_key}_umap",
        cmap='viridis')

#
# if __name__ == "__main__":
#     data_path = '/data/DeepTracing/experiments/Tumor/data'
#     name = '3515_Lkb1_T1_Fam'
#     pruned_adata = sc.read_h5ad(f'{data_path}/preprocessed/{name}_pruned_adata.h5ad')
#     print(pruned_adata)
#
#     keep = np.array((pruned_adata.X > 0).sum(axis=0) >= 10).flatten()
#     print('genes expressed in 10 or more cells:', np.sum(keep))
#
#     pruned_adata = pruned_adata[:, keep].copy()
#     if hasattr(pruned_adata.X, "toarray"):
#         dense_array = pruned_adata.X.toarray()
#     else:
#         dense_array = np.asarray(pruned_adata.X)
#     pruned_adata.X = dense_array
#
#     sc.pp.highly_variable_genes(pruned_adata, flavor="seurat_v3", n_top_genes=200, subset=True)
#
#     pruned_adata = normalize(pruned_adata,
#                              size_factors=True,
#                              normalize_input=True,
#                              logtrans_input=True)
#     print(pruned_adata)
#
#     adata = pruned_adata
#
#     tree_path = f'{data_path}/preprocessed/{name}_pruned.nwk'
#     with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
#         labels_in_order = np.array(f.read().splitlines())
#     _, top_down_clusters = get_top_down_cluster_ids(tree_path, labels_in_order, depth=3)
#     pruned_adata.obs['subtree'] = top_down_clusters
#
#     adata.obs['subtree'] = adata.obs['subtree'].astype('category')
#     adata.obs['Tumor'] = adata.obs['Tumor'].astype('category')
#     adata.obs['Cluster-Name'] = adata.obs['Cluster-Name'].astype('category')
#
#     size = 10
#     results_path = '.results_3515/Figures'
#     os.makedirs(results_path, exist_ok=True)
#
#     plt_show_adata(adata, 'Tumor', size, name, results_path)
#     plt_show_adata(adata, 'subtree', size, name, results_path)
#     plt_show_adata(adata, 'Cluster-Name', size, name, results_path)
#
#     final_latent = np.loadtxt('3515_final_latent.txt', delimiter=",")
#     gp_latent = np.loadtxt('3515_gp_latent.txt', delimiter=",")
#     gaussian_latent = np.loadtxt('3515_gaussian_latent.txt', delimiter=",")
#
#     PORCELAN_results_path = f"./Tumor_PORCELAN/results_all"
#     PORCELAN_latent = np.load(f'{PORCELAN_results_path}/{name}_latent_genes_PORCELAN_1000.npy')
#
#     adata.obsm["DeepTracing_UE"] = final_latent
#     adata.obsm["DeepTracing_LE"] = gp_latent
#     adata.obsm["DeepTracing_IE"] = gaussian_latent
#     adata.obsm["PORCELAN"] = PORCELAN_latent
#
#     color_keys = ["Tumor", "subtree"]
#     plot_umap_embeddings(adata, rep_keys=["DeepTracing_UE", "DeepTracing_LE", "DeepTracing_IE"],
#                          color_keys=color_keys, size=10, name=name, results_path=results_path)
#
#     X_cca_corr_df_sorted = CCA_corr_analyse(adata, final_latent)
#     print(X_cca_corr_df_sorted.head())
#     plot_top_genes(adata, "DeepTracing_UE", X_cca_corr_df_sorted, 2)
