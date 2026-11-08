"""TedSim UMAP plotting and scIB evaluation utilities.

Data loading, normalization, embedding selection, and experiment configuration
are controlled by the calling notebook.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import scanpy as sc
import seaborn as sns
from tree_utils import custom_palette
from scib_metrics.benchmark import Benchmarker, BioConservation

node_palette = ['#4E79A7', '#F28E2B', '#EDC948', '#59A14F', '#76B7B2', '#B07AA1', '#BAB0AC', '#9C755F', '#5CA8D6']

RAW_METRIC_NAMES = {
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "silhouette_label": "Silhouette label",
}
METRIC_COLS = list(RAW_METRIC_NAMES.values())

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


def plot_umap_embeddings(adata, rep_keys, color_keys, output_dir="./", color_map=None):

    adata_copy = adata.copy()

    for rep_key in rep_keys:
        if adata_copy.obsm[rep_key].ndim == 1:
            adata_copy.obsm[rep_key] = adata_copy.obsm[rep_key].reshape(-1, 1)

        try:
            sc.pp.neighbors(adata_copy, use_rep=rep_key, n_neighbors=15)
            sc.tl.umap(adata_copy)
        except Exception as e:
            continue

        for color_key in color_keys:
            if color_key not in adata_copy.obs.columns:
                continue

            if color_map is not None:
                categories = adata_copy.obs[color_key].astype(str).unique()
                palette = [color_map.get(str(cat), "#CCCCCC") for cat in categories]
            else:
                unique_vals = adata_copy.obs[color_key].unique()
                num_clusters = len(unique_vals)
                if num_clusters <= 12:
                    palette = custom_palette[:num_clusters]
                elif num_clusters <= 20:
                    palette = sns.color_palette("tab20", n_colors=num_clusters)
                else:
                    palette = sns.color_palette("husl", n_colors=num_clusters)

            fig, ax = plt.subplots(figsize=(12, 8))
            sc.pl.embedding(
                adata_copy,
                basis="umap",
                color=[color_key],
                size=10,
                legend_loc="right margin",
                legend_fontsize=8,
                legend_fontweight="normal",
                frameon=False,
                ax=ax,
                show=False,
                palette=palette,
                title=f'{rep_key} - UMAP'
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

            save_path = Path(output_dir) / f'adata_embeddings_{rep_key}_{color_key}.pdf'
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Image saved to: {save_path.resolve()}")
            plt.show()


def add_embeddings(adata, embedding_files):
    """Each file must have shape (n_cells, n_latent) and match obs_names order."""
    for key, filename in embedding_files.items():
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"Embedding file not found: {filename}")
        if filename.suffix.lower() == ".npy":
            array = np.load(filename)
        elif filename.suffix.lower() == ".txt":
            array = np.loadtxt(filename, delimiter=",")
        else:
            raise ValueError(f"Unsupported file format for {key}: {filename}")

        array = np.asarray(array)
        if array.ndim != 2 or array.shape[0] != adata.n_obs:
            raise ValueError(
                f"{key}: expected ({adata.n_obs}, n_latent), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or infinite values")
        adata.obsm[key] = array
        print(f"Loaded {key}: {array.shape}")
    return adata


def draw_summary_svg(summary, filename):
    score_cols = [
        "Cell Type Conservation",
        "Clone Conservation",
    ]
    n = len(summary)
    fig, ax = plt.subplots(figsize=(9, max(2.6, 1.25 + 0.58 * n)))
    ax.set_xlim(0, 8.1)
    ax.set_ylim(-0.8, n + 0.8)
    ax.axis("off")

    xs = [3.0, 5.55]
    bar_width = 1.5
    ax.text(0.15, n + 0.35, "Embedding", weight="bold", fontsize=11, va="center")
    for x, col in zip(xs, score_cols):
        ax.text(x + 0.83, n + 0.35, col, weight="bold", fontsize=10,
                ha="center", va="center")

    for i, (method, row) in enumerate(summary.iterrows()):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.39, y + 0.39, color="#F6F7F9", zorder=0)
        ax.text(0.15, y, str(method), va="center", fontsize=10)
        for x, col in zip(xs, score_cols):
            value = float(row[col])
            ax.add_patch(Rectangle((x, y - 0.18), bar_width, 0.36,
                                   facecolor="#E8EEF3", edgecolor="none"))
            ax.add_patch(Rectangle((x, y - 0.18),
                                   bar_width * float(np.clip(value, 0, 1)), 0.36,
                                   facecolor="#447DB4", edgecolor="none"))
            ax.text(x + bar_width + 0.10, y, f"{value:.3f}",
                    fontsize=10, va="center", ha="left")

    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def extract_official_bio_metrics(bm, embedding_keys):
    """Read the three original scIB metrics, without cross-method min-max scaling.

    scib-metrics 0.5.1 get_results() unconditionally references the
    absent Batch correction column. The numerical values are stored in
    bm._results immediately after benchmark(), so no recalculation or
    substitute metric implementation is required.
    """
    if not getattr(bm, "_benchmarked", False):
        raise RuntimeError("Call bm.benchmark() before extracting results_all")

    result = bm._results
    missing = [k for k in RAW_METRIC_NAMES if k not in result.index]
    if missing:
        raise RuntimeError(
            f"Missing computed metrics: {missing}. "
            f"Available: {list(result.index)}"
        )

    # Raw result matrix: rows=metric internal names; columns=embeddings.
    df = result.loc[list(RAW_METRIC_NAMES), embedding_keys].T.copy()
    df = df.rename(columns=RAW_METRIC_NAMES).astype(float)
    df.index.name = "Embedding"

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("The benchmark returned NaN/Inf for some metrics: "
                         + str(df[~np.isfinite(df)].stack(dropna=False)))

    # scIB's official Bio conservation aggregation: simple mean of metrics.
    # In the original Benchmarker.get_results():
    # df.groupby('Metric Type').mean().transpose()['Bio conservation'].
    df["Bio conservation"] = df[METRIC_COLS].mean(axis=1)
    return df


def draw_bio_metrics_svg(df, label_name, filename):
    """Plot an editable SVG metric table in scIB's circle/bar visual style.

    This is needed on scib-metrics 0.5.1 since its official plotting API
    calls get_results(), which crashes in bio-only mode.
    """
    from matplotlib.patches import Circle

    score_label = label_name + " Conservation"
    ordered = df.sort_values("Bio conservation", ascending=False)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(13, max(3, 1.8 + 0.59 * n)))
    ax.set_xlim(0, 12.1)
    ax.set_ylim(-0.75, n + 1.05)
    ax.axis("off")

    x_method = 0.20
    x_metrics = [4.1, 5.5, 6.9]
    x_score = 8.3
    score_width = 1.9

    ax.text(x_method, n + 0.34, "Embedding", fontsize=11, weight="bold", va="center")
    for x, name in zip(x_metrics, METRIC_COLS):
        title = name.replace(" ", "\n", 1)
        ax.text(x, n + 0.34, title, ha="center", va="center", fontsize=9.6, weight="bold")
    ax.text(x_score + score_width / 2, n + 0.34,
            score_label.replace(" Conservation", "\nConservation"),
            ha="center", va="center", fontsize=10, weight="bold")

    cmap = plt.get_cmap("PRGn")
    for i, (method, row) in enumerate(ordered.iterrows()):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.44, y + 0.44, color="#F6F7F9", zorder=0)
        ax.text(x_method, y, str(method), fontsize=10.2, va="center")

        for x, col in zip(x_metrics, METRIC_COLS):
            val = float(row[col])
            rgba = cmap(float(np.clip(val, 0, 1)))
            ax.add_patch(Circle((x, y), radius=0.33,
                                facecolor=rgba, edgecolor="#999999", linewidth=0.45))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if luminance > 0.55 else "white")

        score = float(row["Bio conservation"])
        ax.barh(y, score_width, height=0.37, left=x_score,
                color="#E9EFF3", edgecolor="none")
        ax.barh(y, score_width * float(np.clip(score, 0, 1)),
                height=0.37, left=x_score, color="#447DB4", edgecolor="none")
        ax.text(x_score + score_width + 0.12, y, f"{score:.3f}",
                fontsize=9.8, va="center", ha="left")

    ax.plot([0.12, 11.75], [n - 0.52, n - 0.52], color="#B8C0C9", lw=0.8)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(filename, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_scib(
    adata,
    embedding_keys,
    cell_type_key="cell_type",
    clone_key="clone",
    output_dir="evaluation_metrics_results",
    n_jobs=8,
):
    """Evaluate three scIB metrics per label and save three SVG tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_keys = list(embedding_keys)
    if not embedding_keys:
        raise ValueError("embedding_keys must not be empty")

    for label_key in (cell_type_key, clone_key):
        if label_key not in adata.obs:
            raise KeyError(f"Missing adata.obs[{label_key!r}]")
        if adata.obs[label_key].isna().any():
            raise ValueError(f"Missing labels in {label_key!r}")
        if adata.obs[label_key].nunique() < 2:
            raise ValueError(f"Label {label_key!r} needs >= 2 groups for ASW")

    for key in embedding_keys:
        if key not in adata.obsm:
            raise KeyError(f"Missing adata.obsm[{key!r}]")
        x = np.asarray(adata.obsm[key])
        if x.ndim != 2 or x.shape[0] != adata.n_obs:
            raise ValueError(f"Invalid embedding shape for {key}: {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite embedding values in {key}")

    adata = adata.copy()
    adata.obs["_evaluation_metrics_dummy_batch"] = "all"
    bio_metrics = BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=True,
        silhouette_label=True,
        clisi_knn=False,
    )
    tables = []
    benchmarkers = {}

    for label_name, label_key in (
        ("Cell Type", cell_type_key),
        ("Clone", clone_key),
    ):
        print(f"\nEvaluating {label_name} Conservation...")
        bm = Benchmarker(
            adata,
            batch_key="_evaluation_metrics_dummy_batch",
            label_key=label_key,
            embedding_obsm_keys=embedding_keys,
            pre_integrated_embedding_obsm_key=embedding_keys[0],
            bio_conservation_metrics=bio_metrics,
            batch_correction_metrics=None,
            n_jobs=n_jobs,
        )

        # v0.5.1 maps None to a default BatchCorrection() instance. Force
        # bio-only before benchmark(), and verify the active metric group.
        bm._batch_correction_metrics = None
        bm._metric_collection_dict = {"Bio conservation": bio_metrics}
        print("Enabled metric groups:", list(bm._metric_collection_dict))

        bm.benchmark()
        benchmarkers[label_name] = bm

        # No bm.get_results() or bm.plot_results_table() here: both crash
        # with only a Bio conservation group in scib-metrics 0.5.1.
        raw = extract_official_bio_metrics(bm, embedding_keys)
        raw.to_csv(output_dir / (label_name.replace(" ", "_") + "_metrics.csv"),
                   float_format="%.6f")

        selected = raw.rename(columns={
            **{col: f"{label_name} | {col}" for col in METRIC_COLS},
            "Bio conservation": f"{label_name} Conservation",
        })
        tables.append(selected)

        subdir = output_dir / label_name.replace(" ", "_")
        subdir.mkdir(parents=True, exist_ok=True)
        draw_bio_metrics_svg(raw, label_name, subdir / "evaluation_metrics_results.svg")

    full = pd.concat(tables, axis=1)

    full.index.name = "Embedding"
    full.to_csv(output_dir / "all_metrics.csv", float_format="%.6f")

    summary = full[[
        "Cell Type Conservation", "Clone Conservation"
    ]].copy()
    summary.to_csv(output_dir / "summary_scores.csv", float_format="%.6f")
    draw_summary_svg(summary, output_dir / "summary_scores.svg")
    return full, summary, benchmarkers


METHOD_ORDER = [
    "X_pca", "scVI", "DeepTracing_IV",
    "DeepTracing_UV", "DeepTracing_LV", "PORCELAN",
]
DISPLAY_NAMES = {
    "X_pca": "PCA",
    "scVI": "scVI",
    "DeepTracing_IV": "DeepTracing-IV",
    "DeepTracing_UV": "DeepTracing-UV",
    "DeepTracing_LV": "DeepTracing-LV",
    "PORCELAN": "PORCELAN",
}
METHOD_COLORS = {
    "X_pca": "#4C72B0",
    "scVI": "#DD8452",
    "DeepTracing_IV": "#C44E52",
    "DeepTracing_UV": "#55A868",
    "DeepTracing_LV": "#8172B2",
    "PORCELAN": "#937860",
}

X_COL = "Clone Conservation"
Y_COL = "Cell Type Conservation"

# A legend is clearer than direct labels when several points overlap.
SHOW_POINT_LABELS = False
POINT_SIZE = 145

SCATTER_RCPARAMS = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.linewidth": 1.1,
    "svg.fonttype": "none",  # Keep text editable in SVG
}



def load_data(base_dir, pa_list):
    """Load all requested pa datasets; fail loudly on missing methods/values."""
    tables = {}
    for pa in pa_list:
        path = Path(base_dir) / f"p_a_{pa}" / "summary_scores.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing result file: {path}")

        df = pd.read_csv(path, index_col=0)
        for col in (X_COL, Y_COL):
            if col not in df.columns:
                raise KeyError(f"{path} is missing column {col!r}")

        missing = [method for method in METHOD_ORDER if method not in df.index]
        if missing:
            raise ValueError(f"{path}: missing embeddings: {missing}")

        df = df.loc[METHOD_ORDER, [X_COL, Y_COL]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError(f"{path} contains non-finite conservation scores")
        tables[pa] = df
    return tables


def padded_axis_range(values, step=0.1):
    """Round data limits outward to neat tick values and add margins."""
    values = np.asarray(values, dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    pad = max(0.035, (vmax - vmin) * 0.07)
    low = math.floor((vmin - pad) / step) * step
    high = math.ceil((vmax + pad) / step) * step
    # Conservation is expected to be in [0, 1].
    low, high = max(0.0, low), min(1.0, high)
    if high <= low:
        low, high = max(0.0, vmin - 0.1), min(1.0, vmax + 0.1)
    return round(low, 5), round(high, 5)


def plot_one_pa(pa, df, xlim, ylim, output_dir):
    fig, ax = plt.subplots(figsize=(7.0, 5.8))

    # Median lines from six methods at the CURRENT p_a (not pooled over p_a).
    clone_median = float(df.loc[METHOD_ORDER, X_COL].median())
    celltype_median = float(df.loc[METHOD_ORDER, Y_COL].median())
    median_style = dict(color="#707070", linestyle="--", linewidth=1.3,
                        alpha=0.85, zorder=2)
    ax.axvline(x=clone_median, **median_style)
    ax.axhline(y=celltype_median, **median_style)

    # Draw square markers. NO diagonal y=x line.
    for method in METHOD_ORDER:
        x = float(df.loc[method, X_COL])
        y = float(df.loc[method, Y_COL])
        ax.scatter(
            x, y,
            marker="s", s=POINT_SIZE,
            c=METHOD_COLORS[method],
            edgecolors="black", linewidths=0.65,
            zorder=3,
        )
        if SHOW_POINT_LABELS:
            # Optional labels; legend remains the default to avoid collisions.
            dx = -6 if x > xlim[0] + 0.75 * (xlim[1] - xlim[0]) else 6
            ax.annotate(
                DISPLAY_NAMES[method], (x, y), xytext=(dx, 7),
                textcoords="offset points", fontsize=9,
                ha="right" if dx < 0 else "left", va="bottom",
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.round(np.arange(xlim[0], xlim[1] + 1e-8, 0.1), 2))
    ax.set_yticks(np.round(np.arange(ylim[0], ylim[1] + 1e-8, 0.1), 2))

    ax.set_xlabel(X_COL, fontsize=13, labelpad=9)
    ax.set_ylabel(Y_COL, fontsize=13, labelpad=9)
    ax.set_title(rf"$p_a={pa}$", fontsize=17, pad=14)
    ax.grid(True, color="#B0B0B0", alpha=0.24, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, direction="out", length=4.5)
    # Do not force equal aspect; keep a consistent 7x5.8 layout.

    legend_handles = [
        Patch(facecolor=METHOD_COLORS[m], edgecolor="black", linewidth=0.5,
              label=DISPLAY_NAMES[m])
        for m in METHOD_ORDER
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
        ncol=3, frameon=False, fontsize=10, columnspacing=1.5, handlelength=1.4,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"conservation_scatter_pa_{pa}.svg"
    fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight", facecolor="white")

    print(f"p_a={pa}: Clone median={clone_median:.4f}, Cell Type median={celltype_median:.4f}")
    print(f"Saved: {output_path}")
    return fig


def plot_all_pa_scatter(base_dir, pa_list=(0.2, 0.4, 0.6, 0.8), output_dir=None, show=True):
    """Plot four scatter charts with shared axis limits and per-p_a medians."""
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_dir / "scatter_by_pa"
    tables = load_data(base_dir, pa_list)
    all_x = np.concatenate([t[X_COL].to_numpy() for t in tables.values()])
    all_y = np.concatenate([t[Y_COL].to_numpy() for t in tables.values()])
    xlim = padded_axis_range(all_x)
    ylim = padded_axis_range(all_y)
    print(f"Axis limits: x={xlim}, y={ylim}")
    with plt.rc_context(SCATTER_RCPARAMS):
        for pa, df in tables.items():
            fig = plot_one_pa(pa, df, xlim, ylim, output_dir)
            if show:
                plt.show()
            plt.close(fig)
    return tables
