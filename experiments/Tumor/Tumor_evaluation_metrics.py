"""Bio-only scIB evaluation and visualization for tumor type and clone labels."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scib_metrics.benchmark import Benchmarker, BioConservation


RAW_METRIC_NAMES = {
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "silhouette_label": "Silhouette label",
}
METRIC_COLS = list(RAW_METRIC_NAMES.values())


def extract_official_bio_metrics(bm, embedding_keys):
    """Extract the three original scIB metrics without rescaling."""
    if not getattr(bm, "_benchmarked", False):
        raise RuntimeError("Call bm.benchmark() before extracting results_all")
    result = bm._results
    missing = [key for key in RAW_METRIC_NAMES if key not in result.index]
    if missing:
        raise RuntimeError(f"Missing computed scIB metrics: {missing}. Available: {list(result.index)}")
    df = result.loc[list(RAW_METRIC_NAMES), embedding_keys].T.copy()
    df = df.rename(columns=RAW_METRIC_NAMES).astype(float)
    df.index.name = "Embedding"
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("The benchmark returned NaN/Inf values")
    df["Bio conservation"] = df[METRIC_COLS].mean(axis=1)
    return df


def draw_summary_svg(summary, filename, show=True):
    score_cols = list(summary.columns)
    n = len(summary)
    fig, ax = plt.subplots(figsize=(9.5, max(2.6, 1.25 + 0.58 * n)))
    ax.set_xlim(0, 8.05)
    ax.set_ylim(-0.8, n + 0.8)
    ax.axis("off")
    xs, bar_width = [3.0, 5.55], 1.5
    ax.text(0.15, n + 0.35, "Embedding", weight="bold", fontsize=11, va="center")
    for x, col in zip(xs, score_cols):
        ax.text(x + 0.83, n + 0.35, col, weight="bold", fontsize=10, ha="center", va="center")
    for i, (method, row) in enumerate(summary.iterrows()):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.39, y + 0.39, color="#F6F7F9", zorder=0)
        ax.text(0.15, y, str(method), va="center", fontsize=10)
        for x, col in zip(xs, score_cols):
            value = float(row[col])
            ax.add_patch(Rectangle((x, y - 0.18), bar_width, 0.36, facecolor="#E8EEF3", edgecolor="none"))
            ax.add_patch(Rectangle((x, y - 0.18), bar_width * float(np.clip(value, 0, 1)), 0.36, facecolor="#447DB4", edgecolor="none"))
            ax.text(x + bar_width + 0.10, y, f"{value:.3f}", fontsize=10, va="center", ha="left")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(filename, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def draw_bio_metrics_svg(df, label_name, filename, show=True):
    ordered = df.sort_values("Bio conservation", ascending=False)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(13, max(3, 1.8 + 0.59 * n)))
    ax.set_xlim(0, 12.1)
    ax.set_ylim(-0.75, n + 1.05)
    ax.axis("off")
    x_method, x_metrics, x_score, score_width = 0.20, [4.1, 5.5, 6.9], 8.3, 1.9
    ax.text(x_method, n + 0.34, "Embedding", fontsize=11, weight="bold", va="center")
    for x, metric_name in zip(x_metrics, METRIC_COLS):
        ax.text(x, n + 0.34, metric_name.replace(" ", "\n", 1), ha="center", va="center", fontsize=9.6, weight="bold")
    ax.text(x_score + score_width / 2, n + 0.34, (label_name + " Conservation").replace(" Conservation", "\nConservation"), ha="center", va="center", fontsize=10, weight="bold")
    cmap = plt.get_cmap("PRGn")
    for i, (method, row) in enumerate(ordered.iterrows()):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.44, y + 0.44, color="#F6F7F9", zorder=0)
        ax.text(x_method, y, str(method), fontsize=10.2, va="center")
        for x, col in zip(x_metrics, METRIC_COLS):
            val = float(row[col])
            rgba = cmap(float(np.clip(val, 0, 1)))
            ax.add_patch(Circle((x, y), radius=0.33, facecolor=rgba, edgecolor="#999999", linewidth=0.45))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black" if luminance > 0.55 else "white")
        score = float(row["Bio conservation"])
        ax.barh(y, score_width, height=0.37, left=x_score, color="#E9EFF3", edgecolor="none")
        ax.barh(y, score_width * float(np.clip(score, 0, 1)), height=0.37, left=x_score, color="#447DB4", edgecolor="none")
        ax.text(x_score + score_width + 0.12, y, f"{score:.3f}", fontsize=9.8, va="center", ha="left")
    ax.plot([0.12, 11.75], [n - 0.52, n - 0.52], color="#B8C0C9", lw=0.8)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(filename, format="svg", bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def draw_conservation_scatter(summary, filename, x_col="Lineage Conservation",
                              y_col="Tumor Type Conservation", method_order=None, show=True):
    """Preserve the original square markers, method colors and scatter layout."""
    import math
    from matplotlib.patches import Patch

    if method_order is None:
        method_order = list(summary.index)
    data = summary.loc[method_order, [x_col, y_col]]
    colors = {
        "PCA": "#4C72B0", "scVI": "#DD8452", "DeepTracing_IV": "#C44E52",
        "DeepTracing_UV": "#55A868", "DeepTracing_LV": "#8172B2", "PORCELAN": "#937860",
    }
    display_names = {
        "PCA": "PCA", "scVI": "scVI", "DeepTracing_IV": "DeepTracing IV",
        "DeepTracing_UV": "DeepTracing UV", "DeepTracing_LV": "DeepTracing LV",
        "PORCELAN": "PORCELAN",
    }
    step = 0.05

    def axis_range(values):
        values = np.asarray(values, dtype=float)
        vmin, vmax = float(values.min()), float(values.max())
        pad = max(0.035, (vmax - vmin) * 0.07)
        low = max(0.0, math.floor((vmin - pad) / step) * step)
        high = min(1.0, math.ceil((vmax + pad) / step) * step)
        if high <= low:
            low, high = max(0.0, vmin - step), min(1.0, vmax + step)
        return round(low, 5), round(high, 5)

    with plt.rc_context({
        "font.family": "DejaVu Serif", "mathtext.fontset": "stix",
        "font.size": 11, "axes.linewidth": 1.1,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        xlim, ylim = axis_range(data[x_col]), axis_range(data[y_col])
        x_median, y_median = float(data[x_col].median()), float(data[y_col].median())
        median_style = dict(color="#707070", linestyle="--", linewidth=1.3, alpha=0.85, zorder=2)
        ax.axvline(x=x_median, **median_style)
        ax.axhline(y=y_median, **median_style)
        for method, row in data.iterrows():
            ax.scatter(float(row[x_col]), float(row[y_col]), marker="s", s=145,
                       c=colors[method], edgecolors="black", linewidths=0.65, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        x_ticks = np.round(np.arange(xlim[0], xlim[1] + 1e-8, step), 2)
        y_ticks = np.round(np.arange(ylim[0], ylim[1] + 1e-8, step), 2)
        ax.set_xticks(x_ticks[x_ticks <= 1.0 + 1e-8])
        ax.set_yticks(y_ticks[y_ticks <= 1.0 + 1e-8])
        ax.set_xlabel(x_col, fontsize=13, labelpad=9)
        ax.set_ylabel(y_col, fontsize=13, labelpad=9)
        ax.grid(True, color="#B0B0B0", alpha=0.24, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11, direction="out", length=4.5)
        handles = [Patch(facecolor=colors[m], edgecolor="black", linewidth=0.5,
                         label=display_names[m]) for m in method_order]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
                  ncol=3, frameon=False, fontsize=10, columnspacing=1.5, handlelength=1.4)
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(filename.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)

def evaluate_scib(
    adata,
    embedding_keys,
    label_configs,
    output_dir="evaluation_metrics_results",
    n_jobs=8,
    show=True,
    scatter_pairs=None,
):
    """Evaluate preloaded embeddings in an existing AnnData object.

    This module does not load data, construct labels, calculate PCA, or load
    embeddings. Prepare all of those in the calling notebook and pass the
    ready-to-evaluate AnnData, embedding keys, and label configurations here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_keys = list(embedding_keys)
    if not embedding_keys:
        raise ValueError("embedding_keys must not be empty")
    if not label_configs:
        raise ValueError("label_configs must not be empty")

    for label_name, label_key in label_configs:
        if label_key not in adata.obs:
            raise KeyError(f"Missing adata.obs[{label_key!r}] for {label_name}")
        if adata.obs[label_key].isna().any() or adata.obs[label_key].nunique() < 2:
            raise ValueError(f"Invalid or insufficient labels in {label_key!r}")
    for key in embedding_keys:
        if key not in adata.obsm:
            raise KeyError(f"Missing adata.obsm[{key!r}]")
        values = np.asarray(adata.obsm[key])
        if values.ndim != 2 or values.shape[0] != adata.n_obs or not np.isfinite(values).all():
            raise ValueError(f"Invalid embedding values or shape for {key}: {values.shape}")

    eval_adata = adata.copy()
    eval_adata.obs["_evaluation_metrics_dummy_batch"] = "all"
    bio_metrics = BioConservation(
        isolated_labels=False, nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=True, silhouette_label=True, clisi_knn=False,
    )
    tables, benchmarkers = [], {}
    for label_name, label_key in label_configs:
        print(f"Evaluating {label_name} Conservation...")
        bm = Benchmarker(
            eval_adata, batch_key="_evaluation_metrics_dummy_batch", label_key=label_key,
            embedding_obsm_keys=embedding_keys, pre_integrated_embedding_obsm_key=embedding_keys[0],
            bio_conservation_metrics=bio_metrics, batch_correction_metrics=None, n_jobs=n_jobs,
        )
        bm._batch_correction_metrics = None
        bm._metric_collection_dict = {"Bio conservation": bio_metrics}
        bm.benchmark()
        benchmarkers[label_name] = bm
        raw = extract_official_bio_metrics(bm, embedding_keys)
        raw.to_csv(output_dir / f"{label_name.replace(' ', '_')}_metrics.csv", float_format="%.6f")
        tables.append(raw.rename(columns={
            **{col: f"{label_name} | {col}" for col in METRIC_COLS},
            "Bio conservation": f"{label_name} Conservation",
        }))

    full = pd.concat(tables, axis=1)
    full.index.name = "Embedding"
    full.to_csv(output_dir / "all_metrics.csv", float_format="%.6f")
    conservation_cols = [f"{label_name} Conservation" for label_name, _ in label_configs]
    summary = full[conservation_cols].copy()
    summary.to_csv(output_dir / "summary_scores.csv", float_format="%.6f")

    if scatter_pairs is None:
        scatter_pairs = [(conservation_cols[-1], conservation_cols[0])] if len(conservation_cols) >= 2 else []
    scatter_files = []
    for x_col, y_col in scatter_pairs:
        filename = output_dir / f"{x_col.replace(' ', '_')}_vs_{y_col.replace(' ', '_')}.svg"
        draw_conservation_scatter(summary, filename, x_col=x_col, y_col=y_col,
                                  method_order=embedding_keys, show=show)
        scatter_files.append(filename)
    return full, summary, benchmarkers, scatter_files
