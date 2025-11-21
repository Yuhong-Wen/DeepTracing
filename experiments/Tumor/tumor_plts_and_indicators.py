import scanpy as sc
from preprocess import normalize, get_top_down_cluster_ids
import matplotlib.pyplot as plt
import torch
import pandas as pd
import scib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import numpy as np

torch.manual_seed(0)


def calculate_metrics(adata, rep_dict):
    results = []

    n_clusters_cluster = len(adata.obs['Cluster-Name'].unique())
    n_clusters_subtree = len(adata.obs['subtree'].unique())
    n_clusters_tumor = len(adata.obs['Tumor'].unique())

    for rep_name, latent_matrix in rep_dict.items():
        adata_copy = adata.copy()

        if not pd.api.types.is_categorical_dtype(adata_copy.obs['Cluster-Name']):
            print("Converting 'cluster' column to categorical dtype")
            adata_copy.obs['Cluster-Name'] = adata_copy.obs['Cluster-Name'].astype('category')

        sc.pp.neighbors(adata_copy, use_rep=rep_name, n_neighbors=15)
        score = scib.me.graph_connectivity(adata_copy, label_key="Cluster-Name")

        result = {
            "rep": rep_name,
            "graph_connectivity": score
        }

        for label_key, n_clusters in [("Cluster-Name", n_clusters_cluster), ("subtree", n_clusters_subtree), ("Tumor", n_clusters_tumor)]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            pred_labels = kmeans.fit_predict(latent_matrix)
            result[f"{label_key}_ARI"] = adjusted_rand_score(adata.obs[label_key], pred_labels)
            result[f"{label_key}_NMI"] = normalized_mutual_info_score(adata.obs[label_key], pred_labels)

            if label_key == "Cluster-Name":
                result["silhouette"] = silhouette_score(latent_matrix, pred_labels)
                result["ch_score"] = calinski_harabasz_score(latent_matrix, pred_labels)
                result["db_score"] = davies_bouldin_score(latent_matrix, pred_labels)

        results.append(result)

    return pd.DataFrame(results)

def plt_show_adata(adata, color_feature, size, name, results_path):
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.umap(adata, color=[color_feature], size=size, legend_loc="right margin",
               frameon=False,
               legend_fontsize=8,
               legend_fontweight="normal",
               ax=ax,
               show=False)
    plt.subplots_adjust(left=0.1, right=0.8)
    if ax.get_legend():
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        legend = ax.legend(handles=handles, labels=labels, loc="center left",
                            bbox_to_anchor=(1.02, 0.5),
                            ncol=1,
                            fontsize=8, frameon=False)
    plt.savefig(f'{results_path}/{name}_adata_{color_feature}.pdf', dpi=300, bbox_inches="tight")
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


if __name__ == "__main__":
    data_path = '/data/DeepTracing/experiments/Tumor/data'
    name = '3515_Lkb1_T1_Fam'
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

    pruned_adata = normalize(pruned_adata,
                             size_factors=True,
                             normalize_input=True,
                             logtrans_input=True)
    print(pruned_adata)

    adata = pruned_adata

    tree_path = f'{data_path}/preprocessed/{name}_pruned.nwk'
    with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
        labels_in_order = np.array(f.read().splitlines())
    _, top_down_clusters = get_top_down_cluster_ids(tree_path, labels_in_order, depth=3)
    pruned_adata.obs['subtree'] = top_down_clusters

    adata.obs['subtree'] = adata.obs['subtree'].astype('category')
    adata.obs['Tumor'] = adata.obs['Tumor'].astype('category')
    adata.obs['Cluster-Name'] = adata.obs['Cluster-Name'].astype('category')

    size = 10
    results_path = '/data/DeepTracing/experiments/Tumor/figures_3515'

    plt_show_adata(adata, 'Tumor', size, name, results_path)
    plt_show_adata(adata, 'subtree', size, name, results_path)
    plt_show_adata(adata, 'Cluster-Name', size, name, results_path)

    final_latent = np.loadtxt(f'./results/tumor_{name}/tumor_{name}_final_latent.txt', delimiter=",")
    gp_latent = np.loadtxt(f'./results/tumor_{name}/tumor_{name}_gp_latent.txt', delimiter=",")
    gaussian_latent = np.loadtxt(f'./results/tumor_{name}/tumor_{name}_gaussian_latent.txt', delimiter=",")

    PORCELAN_results_path = f"/data/DeepTracing/experiments/Tumor/Tumor_PORCELAN/results"
    PORCELAN_latent = np.load(f'{PORCELAN_results_path}/{name}_latent_genes_PORCELAN_1000.npy')

    adata.obsm["DeepTracing_UE"] = final_latent
    adata.obsm["DeepTracing_LE"] = gp_latent
    adata.obsm["DeepTracing_IE"] = gaussian_latent
    adata.obsm["PORCELAN"] = PORCELAN_latent

    color_keys = ["Tumor", "subtree", "Cluster-Name"]
    plot_umap_embeddings(adata, rep_keys=["DeepTracing_UE", "DeepTracing_LE", "DeepTracing_IE"],
                         color_keys=color_keys, size=10, name=name, results_path=results_path)

    rep_dict = {
        "X": adata.X,
        "DeepTracing_UE": final_latent,
        "DeepTracing_LE": gp_latent,
        "DeepTracing_IE": gaussian_latent,
        "X_scVI": adata.obsm["X_scVI"],
        "PORCELAN": PORCELAN_latent,
    }

    df = calculate_metrics(adata, rep_dict)
    df.to_csv(f"./results_3515/all_indicators_results.csv", index=False)
    print(df)
