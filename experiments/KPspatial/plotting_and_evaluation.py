"""KP-spatial scIB conservation evaluation and notebook scatter plotting.

Uses three bio-conservation metrics for cell type, lineage and tumor type.
The former overall harmonic-mean score is deliberately omitted.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle
import matplotlib.colors as mcolors
import seaborn as sns

RAW_METRIC_NAMES = {
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "silhouette_label": "Silhouette label",
}
METRIC_COLS = list(RAW_METRIC_NAMES.values())
LABEL_CONFIGS = [("Cell Type", "cell_type"), ("Lineage", "root_branch"), ("Tumor Type", "tumor_id")]
EMBEDDING_KEYS = ["PCA", "scVI", "DeepTracing_UV", "DeepTracing_LV", "DeepTracing_SV", "DeepTracing_IV", "PORCELAN"]
METHOD_ORDER = ["PCA", "scVI", "DeepTracing_IV", "DeepTracing_UV", "DeepTracing_LV", "DeepTracing_SV", "PORCELAN"]
DISPLAY_NAMES = {"PCA": "PCA", "scVI": "scVI", "DeepTracing_IV": "DeepTracing IV", "DeepTracing_UV": "DeepTracing UV", "DeepTracing_LV": "DeepTracing LV", "DeepTracing_SV": "DeepTracing SV", "PORCELAN": "PORCELAN"}
METHOD_COLORS = {"PCA": "#4C72B0", "scVI": "#DD8452", "DeepTracing_IV": "#C44E52", "DeepTracing_UV": "#55A868", "DeepTracing_LV": "#8172B2", "DeepTracing_SV": "#CCB974", "PORCELAN": "#937860"}
SCATTER_PAIRS = [("Lineage Conservation", "Tumor Type Conservation"), ("Lineage Conservation", "Cell Type Conservation")]
SHOW_POINT_LABELS = False
POINT_SIZE = 145
TICK_STEP = 0.05
FIG_SIZE = (7.0, 5.8)



def audit_kp_scib_inputs(adata, output_dir="./evalustion_metrics_results", n_neighbors=15):
    """Audit embedding integrity and label-neighborhood consistency before scIB.

    Note: shape and neighborhood tests cannot establish that externally saved
    latent rows match adata.obs_names. Verify cell order at export/loading.
    """
    from sklearn.neighbors import NearestNeighbors

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not adata.obs_names.is_unique:
        raise ValueError("adata.obs_names must be unique")
    if adata.n_obs <= n_neighbors:
        raise ValueError("Not enough cells for the requested neighbor count")

    print(f"Cells: {adata.n_obs}; genes: {adata.n_vars}")
    print("Embedding row identity: UNVERIFIED (latent files have no cell identifiers).")
    for name, key in LABEL_CONFIGS:
        if key not in adata.obs:
            raise KeyError(f"Missing label column: {key}")
        counts = adata.obs[key].value_counts(dropna=False)
        print(f"{name}: {len(counts)} categories; smallest group={counts.min()}, largest group={counts.max()}")

    rows = []
    for key in EMBEDDING_KEYS:
        if key not in adata.obsm:
            raise KeyError(f"Missing embedding: {key}")
        x = np.asarray(adata.obsm[key], dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != adata.n_obs or x.shape[1] < 1:
            raise ValueError(f"{key}: incorrect embedding shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{key}: nonfinite values")
        variance = x.var(axis=0)
        if np.all(variance < 1e-12):
            raise ValueError(f"{key}: all latent dimensions are constant")

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
        nn.fit(x)
        neighbor_idx = nn.kneighbors(x, return_distance=False)
        # Remove each cell's own index even when duplicate vectors are present.
        neighbor_idx = np.stack([row[row != i][:n_neighbors] for i, row in enumerate(neighbor_idx)])
        for label_name, label_key in LABEL_CONFIGS:
            labels = adata.obs[label_key].astype(str).to_numpy()
            purity = float((labels[:, None] == labels[neighbor_idx]).mean())
            counts = pd.Series(labels).value_counts().to_numpy(dtype=np.float64)
            baseline = float(np.sum(counts * (counts - 1)) / (adata.n_obs * (adata.n_obs - 1)))
            rows.append({"Embedding": key, "Label": label_name,
                         "kNN purity": purity, "Random expectation": baseline,
                         "Purity lift": purity / baseline if baseline > 0 else np.nan,
                         "Latent dimensions": x.shape[1],
                         "Constant dimensions": int((variance < 1e-12).sum())})

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "embedding_alignment_diagnostics.csv", index=False, float_format="%.6f")
    print("\nNearest-neighbor label checks (diagnostic only; not scIB metrics):")
    print(report.pivot(index="Embedding", columns="Label", values="Purity lift").round(3).to_string())
    print("IMPORTANT: Good purity does not prove row alignment, and low purity does not prove a mismatch.")
    print("For exact validation, store/export obs_names alongside every latent matrix and reindex by cell ID.")
    return report


def draw_summary_svg(summary, filename):
    score_cols = list(summary.columns)
    n = len(summary)
    n_cols = len(score_cols)

    fig, ax = plt.subplots(figsize=(3.0 + 2.3 * n_cols, max(2.6, 1.25 + 0.58 * n)))
    ax.set_xlim(0, 3.0 + 2.3 * n_cols - 0.2)
    ax.set_ylim(-0.8, n + 0.8)
    ax.axis("off")

    bar_width = 1.4
    xs = [2.4 + 2.25 * i for i in range(n_cols)]

    ax.text(0.15, n + 0.35, "Embedding", weight="bold", fontsize=11, va="center")
    for x, col in zip(xs, score_cols):
        ax.text(x + bar_width / 2, n + 0.35, col, weight="bold",
                fontsize=10, ha="center", va="center")

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
    if not getattr(bm, "_benchmarked", False):
        raise RuntimeError("Call bm.benchmark() before extracting results_all")

    result = bm._results
    missing = [k for k in RAW_METRIC_NAMES if k not in result.index]
    if missing:
        raise RuntimeError(f"Missing computed scIB metrics: {missing}. "
                           f"Available: {list(result.index)}")

    df = result.loc[list(RAW_METRIC_NAMES), embedding_keys].T.copy()
    df = df.rename(columns=RAW_METRIC_NAMES).astype(float)
    df.index.name = "Embedding"

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("The benchmark returned NaN/Inf for some metrics: "
                         + str(df[~np.isfinite(df)].stack(dropna=False)))

    df["Bio conservation"] = df[METRIC_COLS].mean(axis=1)
    return df


def draw_bio_metrics_svg(df, label_name, filename):
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
        ax.text(x, n + 0.34, title, ha="center", va="center",
                fontsize=9.6, weight="bold")
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
    label_configs,
    output_dir="scib_results",
    n_jobs=8,
):
    """Evaluate three scIB metrics per label and save three SVG tables.

    Parameters
    ----------
    label_configs : list[tuple[str, str]]
        [(display_name, adata.obs_key), ...]
        [("Cell Type", "cell_type"), ("Lineage", "root_branch"), ("Tumor Type", "tumor_id")]
    """
    from scib_metrics.benchmark import Benchmarker, BioConservation
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_keys = list(embedding_keys)
    if not embedding_keys:
        raise ValueError("embedding_keys must not be empty")
    if not label_configs:
        raise ValueError("label_configs must not be empty")


    for label_name, label_key in label_configs:
        if label_key not in adata.obs:
            raise KeyError(f"Missing adata.obs[{label_key!r}] (for {label_name})")
        if adata.obs[label_key].isna().any():
            raise ValueError(f"Missing labels in {label_key!r} (for {label_name})")
        if adata.obs[label_key].nunique() < 2:
            raise ValueError(f"Label {label_key!r} (for {label_name}) needs >= 2 groups")


    for key in embedding_keys:
        if key not in adata.obsm:
            raise KeyError(f"Missing adata.obsm[{key!r}]")
        x = np.asarray(adata.obsm[key])
        if x.ndim != 2 or x.shape[0] != adata.n_obs:
            raise ValueError(f"Invalid embedding shape for {key}: {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite embedding values in {key}")

    adata = adata.copy()
    adata.obs["_scib_dummy_batch"] = "all"

    bio_metrics = BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=True,
        silhouette_label=True,
        clisi_knn=False,
    )

    tables = []
    benchmarkers = {}

    for label_name, label_key in label_configs:
        print(f"\nEvaluating {label_name} Conservation (obs['{label_key}'])...")
        bm = Benchmarker(
            adata,
            batch_key="_scib_dummy_batch",
            label_key=label_key,
            embedding_obsm_keys=embedding_keys,
            pre_integrated_embedding_obsm_key=embedding_keys[0],
            bio_conservation_metrics=bio_metrics,
            batch_correction_metrics=None,
            n_jobs=n_jobs,
        )


        bm._batch_correction_metrics = None
        bm._metric_collection_dict = {"Bio conservation": bio_metrics}
        print("Enabled metric groups:", list(bm._metric_collection_dict))

        bm.benchmark()
        benchmarkers[label_name] = bm

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
        draw_bio_metrics_svg(raw, label_name, subdir / "scib_results.svg")

    # Combine conservation tables without an overall score.
    full = pd.concat(tables, axis=1)
    conservation_cols = [f"{label_name} Conservation" for label_name, _ in label_configs]
    full.index.name = "Embedding"
    full.to_csv(output_dir / "all_metrics.csv", float_format="%.6f")
    summary = full[conservation_cols].copy()
    summary.to_csv(output_dir / "summary_scores.csv", float_format="%.6f")
    draw_summary_svg(summary, output_dir / "summary_scores.svg")
    return full, summary, benchmarkers


def padded_axis_range(values, step=TICK_STEP):

    values = np.asarray(values, dtype=float)

    vmin = float(values.min())
    vmax = float(values.max())

    # Match the padding strategy of the second script
    pad = max(0.035, (vmax - vmin) * 0.07)

    low = math.floor((vmin - pad) / step) * step
    high = math.ceil((vmax + pad) / step) * step

    # Conservation scores are in [0, 1]
    low = max(0.0, low)
    high = min(1.0, high)

    if high <= low:
        low = max(0.0, vmin - step)
        high = min(1.0, vmax + step)

    return round(low, 5), round(high, 5)


def plot_scatter(df, methods, x_col, y_col, output_dir="./scib_results", show=True):
    """Plot one Conservation pair with the original plot styling."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    xlim = padded_axis_range(df[x_col].values)
    ylim = padded_axis_range(df[y_col].values)
    x_median, y_median = float(df[x_col].median()), float(df[y_col].median())
    median_style = dict(color="#707070", linestyle="--", linewidth=1.3, alpha=0.85, zorder=2)
    ax.axvline(x=x_median, **median_style)
    ax.axhline(y=y_median, **median_style)

    for method in methods:
        x, y = float(df.loc[method, x_col]), float(df.loc[method, y_col])
        ax.scatter(x, y, marker="s", s=POINT_SIZE, c=METHOD_COLORS.get(method, "#999999"),
                   edgecolors="black", linewidths=0.65, zorder=3)
        if SHOW_POINT_LABELS:
            dx = -6 if x > xlim[0] + 0.75 * (xlim[1] - xlim[0]) else 6
            ax.annotate(DISPLAY_NAMES.get(method, method), (x, y), xytext=(dx, 7),
                        textcoords="offset points", fontsize=9,
                        ha="right" if dx < 0 else "left", va="bottom")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for axis, bounds in ((ax.xaxis, xlim), (ax.yaxis, ylim)):
        ticks = np.round(np.arange(bounds[0], bounds[1] + 1e-8, TICK_STEP), 2)
        axis.set_ticks(ticks[ticks <= 1.0 + 1e-8])
    ax.set_xlabel(x_col, fontsize=13, labelpad=9)
    ax.set_ylabel(y_col, fontsize=13, labelpad=9)
    ax.grid(True, color="#B0B0B0", alpha=0.24, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, direction="out", length=4.5)
    handles = [Patch(facecolor=METHOD_COLORS.get(m, "#999999"), edgecolor="black", linewidth=0.5,
                     label=DISPLAY_NAMES.get(m, m)) for m in methods]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3,
              frameon=False, fontsize=10, columnspacing=1.5, handlelength=1.4)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{x_col.replace(' ', '_')}_vs_{y_col.replace(' ', '_')}_scatter"
    svg_path, pdf_path = output_dir / f"{stem}.svg", output_dir / f"{stem}.pdf"
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"{x_col} median={x_median:.4f}; {y_col} median={y_median:.4f}")
    print(f"Saved: {svg_path} and {pdf_path}")
    return svg_path, pdf_path


def run_kp_scib(adata, output_dir="./scib_results", n_jobs=8, show=True):
    """Calculate scIB scores and display two independent Conservation plots."""
    full, summary, benchmarkers = evaluate_scib(
        adata=adata, embedding_keys=EMBEDDING_KEYS, label_configs=LABEL_CONFIGS,
        output_dir=output_dir, n_jobs=n_jobs,
    )
    ordered = [name for name in METHOD_ORDER if name in summary.index]
    if not ordered:
        raise ValueError("No configured embeddings were found in scIB results_all")
    scatter_paths = {}
    with plt.rc_context({"font.family": "DejaVu Serif", "mathtext.fontset": "stix",
                         "font.size": 11, "axes.linewidth": 1.1,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        for x_col, y_col in SCATTER_PAIRS:
            scatter_input = summary.loc[ordered, [x_col, y_col]].apply(pd.to_numeric, errors="raise")
            scatter_paths[y_col] = plot_scatter(scatter_input, ordered, x_col, y_col,
                                                output_dir=output_dir, show=show)
    return full, summary, benchmarkers, scatter_paths


def plot_show_spatial(
    adata,
    color_feature,
    size=8,
    alpha=0.9,
    color_map=None,
    output_dir="./",
    use_gradient_for_categories=False,
    gradient_cmap="viridis",
    spatial_key="spatial",
    invert_y=True,
    show_axis=False,
    figsize=(8, 8),
    dpi=300,
    save_format="pdf",
    legend_fontsize=8,
    legend_ncol=1,
    title=None,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not found in adata.obsm.")

    spatial = np.asarray(adata.obsm[spatial_key])
    if spatial.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must have at least two columns.")

    x = spatial[:, 0]
    y = spatial[:, 1]

    # Determine whether the feature is categorical or continuous
    is_categorical = False

    if color_feature in adata.obs.columns:
        values = adata.obs[color_feature]
        if isinstance(values.dtype, pd.CategoricalDtype):
            is_categorical = True
        elif values.dtype == object or values.dtype.name == "category":
            is_categorical = True
        elif values.dtype.name in ["bool", "boolean"]:
            is_categorical = True
    elif color_feature in adata.var_names:
        values = adata[:, color_feature].X
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values).ravel()
        is_categorical = False
    else:
        raise KeyError(
            f"color_feature='{color_feature}' was not found in adata.obs or adata.var_names."
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---------------- Continuous variable ----------------
    if not is_categorical:
        vals = np.asarray(values, dtype=float)

        scat = ax.scatter(x, y, c=vals, s=size, alpha=alpha, cmap=gradient_cmap, linewidths=0,)

        cbar = plt.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_feature, fontsize=legend_fontsize + 1)
        cbar.ax.tick_params(labelsize=legend_fontsize)

    # ---------------- Categorical variable ----------------
    else:
        if not isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype("category")
        else:
            values = values.cat.remove_unused_categories()

        categories = list(values.cat.categories)
        codes = values.cat.codes.to_numpy()
        n = len(categories)

        # Custom colors
        if color_map is not None:
            palette = [color_map.get(str(cat), "#CCCCCC") for cat in categories]

        # Gradient colors for categories
        elif use_gradient_for_categories:
            cmap = plt.get_cmap(gradient_cmap)
            positions = np.linspace(0.35, 0.95, n)
            palette = [mcolors.to_hex(cmap(pos)) for pos in positions]

        # Automatic palette
        else:
            if n <= 10:
                palette = sns.color_palette("tab10", n_colors=n)
            elif n <= 20:
                palette = sns.color_palette("tab20", n_colors=n)
            else:
                palette = sns.color_palette("husl", n_colors=n)

        colors = np.array(
            [palette[code] if code >= 0 else (0, 0, 0, 0) for code in codes]
        )

        ax.scatter(x, y, c=colors, s=size, alpha=alpha, linewidths=0)

        # Manual legend
        handles = [
          plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=palette[i],
                markersize=6,
                label=str(categories[i]),
            )
            for i in range(n)
        ]

        ax.legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=False,
        )

    # Style
    ax.set_aspect("equal")

    if invert_y:
        ax.invert_yaxis()

    if title is not None:
        ax.set_title(title, fontsize=12)

    if show_axis:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    save_path = output_dir / f"spatial_{color_feature}.{save_format}"
    plt.savefig(
        save_path,
        facecolor="white",
        edgecolor="none",
        transparent=False,
        bbox_inches="tight",
    )
    print(f"Image saved to: {save_path.resolve()}")
    plt.show()
