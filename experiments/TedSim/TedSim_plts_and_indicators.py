import scanpy as sc
from preprocess import normalize
from TedSim_data_prepare import *
import pickle
import scib
import seaborn as sns
import matplotlib.pyplot as plt
from tree_utils import tree_draw_subtrees, tree_draw_clustres, custom_palette, subtree_clusters
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

def calculate_metrics(adata, rep_dict):
    results = []
    n_clusters_cluster = len(adata.obs['cluster'].unique())
    n_clusters_subtree = len(adata.obs['subtree'].unique())
    for rep_name, latent_matrix in rep_dict.items():
        adata_copy = adata.copy()

        if not pd.api.types.is_categorical_dtype(adata_copy.obs['cluster']):
            print("Converting 'cluster' column to categorical dtype")
            adata_copy.obs['cluster'] = adata_copy.obs['cluster'].astype('category')

        sc.pp.neighbors(adata_copy, use_rep=rep_name, n_neighbors=15)
        score = scib.me.graph_connectivity(adata_copy, label_key="cluster")

        result = {
            "rep": rep_name,
            "graph_connectivity": score
        }

        for label_key, n_clusters in [("cluster", n_clusters_cluster), ("subtree", n_clusters_subtree)]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            pred_labels = kmeans.fit_predict(latent_matrix)
            result[f"{label_key}_ari"] = adjusted_rand_score(adata.obs[label_key], pred_labels)
            result[f"{label_key}_nmi"] = normalized_mutual_info_score(adata.obs[label_key], pred_labels)

            if label_key == "cluster":
                result["silhouette"] = silhouette_score(latent_matrix, pred_labels)
                result["ch_score"] = calinski_harabasz_score(latent_matrix, pred_labels)
                result["db_score"] = davies_bouldin_score(latent_matrix, pred_labels)
        results.append(result)
    return pd.DataFrame(results)

def plot_show_adata(adata, color_feature, size=10, color_map=None, output_dir="./"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if color_map is not None:
        categories = adata.obs[color_feature].astype(str).unique()
        palette = [color_map.get(str(cat), "#CCCCCC") for cat in categories]
    else:
        unique_vals = adata.obs[color_feature].unique()
        num_clusters = len(unique_vals)
        if num_clusters <= 12:
            palette = custom_palette[:num_clusters]
        elif num_clusters <= 20:
            palette = sns.color_palette("tab20", n_colors=num_clusters)
        else:
            palette = sns.color_palette("husl", n_colors=num_clusters)

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


    save_path = output_dir / f'adata_{color_feature}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Image saved to: {save_path.resolve()}")
    plt.show()

def plot_umap_embeddings(adata, rep_keys, color_keys, output_dir="./"):
    for rep_key in rep_keys:

        sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=15)
        sc.tl.umap(adata)

        for color_key in color_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            sc.pl.embedding(
                adata,
                basis="umap",
                color=[color_key],
                size=10,
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
            save_path = output_dir / f'adata_embeddings_{rep_key}_{color_key}.pdf'
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Image saved to: {save_path.resolve()}")
            plt.show()


if __name__ == "__main__":
    p_a = 0.2
    running_seed = 0
    BASE_DIR = Path("/data/DeepTracing/experiments/TedSim")
    data_file = BASE_DIR / f"output_adatas/adata_{p_a}.h5ad"
    adata_raw = sc.read(data_file)
    print(adata_raw.X.shape)

    output_dir = Path(f"./results/running_seed_{running_seed}/{p_a}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_file = output_dir / f"tree_{p_a}.pkl"
    with open(tree_file, "rb") as f:
        loaded_tree = pickle.load(f)
    tree = loaded_tree

    all_data = truncate_tree(tree, depth=12)
    adata_all = create_adata_from_tree(all_data)

    mask = adata_all.obs['node_depth'].isin([12])
    adata_need = adata_all[mask, :].copy()
    adata_need.X = np.array(adata_need.X) if isinstance(adata_need.X, pd.Series) else adata_need.X

    adata_need = subtree_clusters(adata_need, tree, 4, 'subtree')

    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000, subset=True)
    adata = normalize(adata_need,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    adata.obs['cluster'] = adata.obs['cluster'].astype('category')
    adata.obs['subtree'] = adata.obs['subtree'].astype('category')

    plot_show_adata(adata, color_feature="cluster", size=10, output_dir=output_dir)
    plot_show_adata(adata, color_feature="subtree", size=10, output_dir=output_dir)

    tree_draw_clustres(adata, tree=tree, depth=6, path=str(output_dir), color_by='cluster')
    tree_draw_subtrees(adata, tree=tree, depth=6, path=str(output_dir), color_by='subtree')


    final_latent_file_path = str(output_dir / f"final_latent_{p_a}.txt")
    gp_latent_file_path = str(output_dir / f"gp_latent_{p_a}.txt")
    gaussian_latent_file_path = str(output_dir / f"gaussian_latent_{p_a}.txt")

    scVI_output_dir = Path(f".../results_scvi/running_seed_{running_seed}")
    scVI_latent_file_path = str(scVI_output_dir / f"{p_a}/scvi_latent_{p_a}_{running_seed}.txt")

    final_latent = np.loadtxt(final_latent_file_path, delimiter=",")
    gp_latent = np.loadtxt(gp_latent_file_path, delimiter=",")
    gaussian_latent = np.loadtxt(gaussian_latent_file_path, delimiter=",")
    scVI_latent = np.loadtxt(scVI_latent_file_path, delimiter=",")

    adata.obsm["DeepTracing_UE"] = final_latent
    adata.obsm["DeepTracing_LE"] = gp_latent
    adata.obsm["DeepTracing_IE"] = gaussian_latent
    adata.obsm["scVI"] = scVI_latent

    color_keys = ["cluster", "subtree"]
    plot_umap_embeddings(adata, rep_keys=["DeepTracing_UE", "DeepTracing_LE", "DeepTracing_IE",
                                          ], color_keys=color_keys, output_dir=output_dir)

    rep_dict = {
        "X": adata.X,
        "DeepTracing_UE": final_latent,
        "DeepTracing_LE": gp_latent,
        "DeepTracing_IE": gaussian_latent,
        "scVI": scVI_latent
    }

    df = calculate_metrics(adata, rep_dict)
    df["p_a"] = p_a
    df.to_csv(f"./results/running_seed_{running_seed}/indicators_results_{p_a}.csv", index=False)
    print(df)