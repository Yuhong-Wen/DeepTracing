import os
import scib
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np
from preprocess import normalize


custom_palette = [
    "#4E79A7",  # muted blue
    "#F28E2B",  # warm orange
    "#59A14F",  # soft green
    "#EDC948",  # muted yellow
    "#76B7B2",  # teal/cyan
    "#B07AA1",  # purple
    "#FF9DA7",  # soft pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray

    "#6B8FD6",  # soft cornflower blue
    "#F6A75D",  # peach orange
    "#7FB77E",  # sage green
    "#F2D35B",  # sunflower yellow
    "#88C6C1",  # aqua teal
    "#C08CB9",  # lilac
    "#FFB7BC",  # light rose pink
    "#A88F79",  # taupe brown
    "#C8C3C0",  # light warm gray

    "#3F5F8C",  # deep muted navy
    "#D77C35",  # burnt orange
    "#4F8A54",  # forest green
    "#D1B741",  # mustard yellow
    "#5EA1A1",  # dusty teal
    "#946C91",  # mauve purple
    "#E88A94",  # coral pink
    "#826A58",  # cocoa brown
    "#A5A09D",  # cool gray

    "#7AA0CC",  # sky slate blue
    "#FFBE78",  # apricot
    "#A3D39C",  # mint green
]

def plot_show_adata(adata, color_feature, size=10, color_map=None):

    categories = adata.obs[color_feature].astype(str).unique().tolist()

    if color_map is not None and isinstance(color_map, dict):
        palette = [color_map.get(cat, "#CCCCCC") for cat in categories]
    else:
        num_clusters = len(categories)
        if num_clusters <= 12:
            palette = sns.color_palette("Set2", n_colors=num_clusters)
        elif num_clusters <= 20:
            palette = sns.color_palette("tab20", n_colors=num_clusters)
        else:
            palette = sns.color_palette("husl", n_colors=num_clusters)

        color_map = dict(zip(categories, palette))

    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)

    fig, ax = plt.subplots(figsize=(12, 8))
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
        palette=palette
    )

    plt.subplots_adjust(left=0.1, right=0.8)
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

    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'./figures/adata_{color_feature}.pdf', dpi=300, bbox_inches="tight")
    plt.show()


def plot_umap_embeddings(adata, rep_keys, color_keys, size=10):
    for rep_key in rep_keys:

        sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=15)
        sc.tl.umap(adata)

        for color_key in color_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            sc.pl.embedding(
                adata,
                basis="umap",
                color=[color_key],
                size=size,
                legend_loc="right margin",
                legend_fontsize=8,
                frameon=False,
                ax=ax,
                show=False
            )

            plt.subplots_adjust(left=0.1, right=0.8)
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

            plt.title(f"{rep_key} - {color_key}")

            os.makedirs('figures', exist_ok=True)
            plt.savefig(f'./figures/adata_embeddings_{rep_key}_{color_key}.pdf', dpi=300, bbox_inches="tight")
            plt.show()


def calculate_metrics(adata, rep_dict):
    results = []
    for rep_name, latent_matrix in rep_dict.items():
        adata_copy = adata.copy()

        if not pd.api.types.is_categorical_dtype(adata_copy.obs['Cell_type']):
            adata_copy.obs['Cell_type'] = adata_copy.obs['Cell_type'].astype('category')

        sc.pp.neighbors(adata_copy, use_rep=rep_name, n_neighbors=15)
        score = scib.me.graph_connectivity(adata_copy, label_key="Cell_type")

        result = {
            "rep": rep_name,
            "graph_connectivity": score
        }

        for label_key, n_clusters in [("Cell_type", len(adata.obs['Cell_type'].unique().tolist())),
                                      ("subtree", len(adata.obs['subtree'].unique().tolist()))]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            pred_labels = kmeans.fit_predict(latent_matrix)
            result[f"{label_key}_ari"] = adjusted_rand_score(adata.obs[label_key], pred_labels)
            result[f"{label_key}_nmi"] = normalized_mutual_info_score(adata.obs[label_key], pred_labels)

            if label_key == "Cell_type":
                result["silhouette"] = silhouette_score(latent_matrix, pred_labels)
                result["ch_score"] = calinski_harabasz_score(latent_matrix, pred_labels)
                result["db_score"] = davies_bouldin_score(latent_matrix, pred_labels)

        results.append(result)
    return pd.DataFrame(results)

if __name__ == "__main__":
    adata = sc.read('.../adata_combined_E11_E15.h5ad')
    if hasattr(adata.X, "toarray"):
        dense_array = adata.X.toarray()
    else:
        dense_array = np.asarray(adata.X)
    adata.X = dense_array

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000, subset=True)
    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    adata.obs['states'] = adata.obs['states'].astype('category')
    plot_show_adata(adata, color_feature='states', size=10)

    adata.obs['Cell_type'] = adata.obs['Cell_type'].astype('category')
    plot_show_adata(adata, color_feature="Cell_type", size=10, color_map=custom_palette)

    adata.obs['timepoint'] = adata.obs['timepoint'].astype('category')
    plot_show_adata(adata, color_feature="timepoint", size=10)

    final_latent = np.loadtxt('.../final_latent.txt', delimiter=",")
    gp_latent = np.loadtxt('.../gp_latent.txt', delimiter=",")
    gp_T_latent = np.loadtxt('.../gp_T_latent.txt', delimiter=",")
    gaussian_latent = np.loadtxt('.../gaussian_latent_with_tree.txt', delimiter=",")

    adata.obsm["X_latent"] = final_latent
    adata.obsm["gp_latent"] = gp_latent
    adata.obsm["gp_T_latent"] = gp_T_latent
    adata.obsm["gaussian_latent"] = gaussian_latent

    rep_keys = ["X_latent", "gp_latent", "gp_T_latent", "gaussian_latent"]
    for rep_key in rep_keys:
        if rep_key in adata.obsm and len(adata.obsm[rep_key].shape) == 1:
            adata.obsm[rep_key] = adata.obsm[rep_key].reshape(-1, 1)

    color_keys = ["states", "Cell_type", 'timepoint']
    plot_umap_embeddings(adata, rep_keys=rep_keys, color_keys=color_keys, size=10)