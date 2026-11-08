import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D

import pandas as pd

def plot_show_adata(
    adata,
    color_feature,
    size=10,
    color_map=None,
    output_dir="./",
    use_gradient_for_categories=False,
    gradient_cmap="viridis"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if color_map is not None:
        categories = list(adata.obs[color_feature].cat.categories)
        palette = [color_map.get(str(cat), "#CCCCCC") for cat in categories]

    elif use_gradient_for_categories:
        categories = list(adata.obs[color_feature].cat.categories)
        n = len(categories)

        cmap = cm.get_cmap(gradient_cmap)

        positions = np.linspace(0.35, 0.95, n)
        palette = [mcolors.to_hex(cmap(x)) for x in positions]

    else:
        unique_vals = adata.obs[color_feature].astype(str).unique()
        num_clusters = len(unique_vals)

        if num_clusters <= 10:
            palette = sns.color_palette("tab10", n_colors=num_clusters)
        elif num_clusters <= 20:
            palette = sns.color_palette("husl", n_colors=num_clusters)
        else:
            palette = sns.color_palette("husl", n_colors=num_clusters)

    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    else:
        print("already")

    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, random_state=0)
    sc.tl.umap(adata, random_state=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
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
    plt.savefig(save_path, facecolor='white', edgecolor='none', transparent=False)
    print(f"Image saved to: {save_path.resolve()}")
    plt.show()

def plot_umap_embeddings(
    adata,
    rep_keys,
    color_keys,
    size=10,
    color_map=None,
    output_dir="./",
    use_gradient_for_categories=False,
    gradient_cmap="viridis"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rep_key in rep_keys:

        # Compute neighbors and UMAP.
        sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=15, random_state=0)
        sc.tl.umap(adata, random_state=0)

        for color_key in color_keys:

            # Match the color assignment used by plot_show_adata.
            if color_map is not None:
                categories = list(adata.obs[color_key].cat.categories)
                palette = [color_map.get(str(cat), "#CCCCCC") for cat in categories]

            elif use_gradient_for_categories:
                categories = list(adata.obs[color_key].cat.categories)
                n = len(categories)

                cmap = plt.get_cmap(gradient_cmap)

                positions = np.linspace(0.35, 0.95, n)
                palette = [mcolors.to_hex(cmap(x)) for x in positions]

            else:
                unique_vals = adata.obs[color_key].astype(str).unique()
                num_clusters = len(unique_vals)

                if num_clusters <= 10:
                    palette = sns.color_palette("tab10", n_colors=num_clusters)
                elif num_clusters <= 20:
                    palette = sns.color_palette("tab20", n_colors=num_clusters)
                else:
                    palette = sns.color_palette("husl", n_colors=num_clusters)

            # Plot.
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            sc.pl.umap(
                adata,
                color=[color_key],
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

            plt.title(f"{rep_key} - {color_key}")

            save_path = output_dir / f'adata_embeddings_{rep_key}_{color_key}.pdf'
            plt.savefig(save_path, facecolor='white', edgecolor='none', transparent=False)
            print(f"Image saved to: {save_path.resolve()}")
            plt.show()

def _draw_segmented_arrow(ax, x0, y0, x1, y1,
                          seg_len=0.8,
                          gap_frac=0.30,
                          lw=0.95, color="black", head_scale=7,
                          alpha=0.78, zorder=6, min_segments=1,
                          shrink_start=5,  #
                          shrink_end=10):  #

    dx, dy = float(x1 - x0), float(y1 - y0)
    total = float(np.hypot(dx, dy))
    if total < 1e-6 or seg_len <= 0:
        return

    n_segments = max(min_segments, int(round(total / seg_len)))
    pitch = total / n_segments
    ux, uy = dx / total, dy / total

    for s in range(n_segments):
        s0 = s * pitch
        s1 = (s + 1) * pitch
        s1_draw = s1 if s == n_segments - 1 else s1 - pitch * gap_frac

        sA = shrink_start if s == 0 else 0
        sB = shrink_end if s == n_segments - 1 else 0

        ax.annotate(
            "",
            xy=(x0 + ux * s1_draw, y0 + uy * s1_draw),
            xytext=(x0 + ux * s0, y0 + uy * s0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                            mutation_scale=head_scale,
                            shrinkA=sA, shrinkB=sB, alpha=alpha),
            zorder=zorder,
        )

def plot_clone_time_state_umap(
    adata,
    rep_key="DeepTracing_UV",
    clone_key="top_clone",
    state_key="tumor_state",
    time_key="time",
    time_order=(0, 4, 6),
    clones_to_draw=None,
    color_map=None,
    output_dir="./",
    min_cells=3,
    size=10,
    center_size=90,
    arrow_lw=0.95,
    arrow_seg_len=0.8,
    arrow_gap_frac=0.30,
    arrow_head_scale=7,
    arrow_min_segments=1,
    arrow_shrink_start=5,
    arrow_shrink_end=10,
    alpha_bg=0.75,
    recalculate_umap=True,
    show=True,
):
    """Plot all cells; overlay (clone,time,state) centroids and adjacent-time arrows.

    Returns centers, edges; centers also include full-dimensional UE centroid for
    subsequent decoder analyses. Lines represent observed state-matched changes,
    not experimentally proven ancestor-descendant transitions.
    """
    for k in (clone_key, state_key, time_key):
        if k not in adata.obs:
            raise KeyError(f"adata.obs missing {k!r}")
    if rep_key not in adata.obsm:
        raise KeyError(f"adata.obsm missing {rep_key!r}")

    # Crucial: recompute UMAP for ALL cells, exactly like original function.
    if recalculate_umap or "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=15, random_state=0)
        sc.tl.umap(adata, random_state=0)

    times_numeric = pd.to_numeric(adata.obs[time_key].astype(str), errors="coerce")
    if times_numeric.isna().all():
        raise ValueError(f"{time_key!r} is not numeric; use the column containing 0, 4, 6")
    observed_times = set(times_numeric.dropna().unique())
    if not set(time_order).intersection(observed_times):
        raise ValueError(f"No requested times {time_order} found in {time_key}: {sorted(observed_times)}")
    print(f"Observed times in {time_key}: {sorted(observed_times)}")

    # Clone colors are assigned on the full AnnData categorical ordering.
    clone_values = adata.obs[clone_key].astype("category")
    categories = list(clone_values.cat.categories)
    if color_map is not None:
        palette = [color_map.get(str(k), "#CCCCCC") for k in categories]
    else:
        import seaborn as sns
        if len(categories) <= 10:
            palette = sns.color_palette("tab10", n_colors=len(categories))
        elif len(categories) <= 20:
            palette = sns.color_palette("tab20", n_colors=len(categories))
        else:
            palette = sns.color_palette("husl", n_colors=len(categories))
    color_lookup = dict(zip(map(str, categories), palette))

    # Aggregate only target clones, but UMAP background always retains ALL cells.
    clones = ([x for x in categories if str(x).lower() not in ("other", "others", "nan")]
              if clones_to_draw is None else list(clones_to_draw))
    chosen = adata.obs[clone_key].astype(str).isin([str(x) for x in clones])
    chosen &= times_numeric.isin(time_order)

    coordinates = np.asarray(adata.obsm["X_umap"])
    z = np.asarray(adata.obsm[rep_key])
    df = pd.DataFrame({
        "clone": adata.obs[clone_key].astype(str).to_numpy(),
        "state": adata.obs[state_key].astype(str).to_numpy(),
        "time": times_numeric.to_numpy(),
        "umap_1": coordinates[:, 0],
        "umap_2": coordinates[:, 1],
    })
    latent_cols = [f"z_{i}" for i in range(z.shape[1])]
    for j, col in enumerate(latent_cols):
        df[col] = z[:, j]
    grouped = df.loc[chosen.to_numpy()].groupby(["clone", "time", "state"], sort=False)
    centers = grouped[["umap_1", "umap_2"] + latent_cols].mean().reset_index()
    n_df = grouped.size().reset_index(name="n_cells")
    centers = centers.merge(n_df, on=["clone", "time", "state"], validate="one_to_one")
    centers = centers.loc[centers.n_cells >= min_cells].reset_index(drop=True)

    center_lookup = {(r.clone, r.time, r.state): r for r in centers.itertuples(index=False)}
    edge_rows = []
    for clone in clones:
        for state in centers.loc[centers.clone == str(clone), "state"].unique():
            for t0, t1 in zip(time_order[:-1], time_order[1:]):
                p = center_lookup.get((str(clone), t0, state))
                q = center_lookup.get((str(clone), t1, state))
                if p is not None and q is not None:
                    edge_rows.append(dict(clone=str(clone), state=state,
                                          time_from=t0, time_to=t1,
                                          x_from=p.umap_1, y_from=p.umap_2,
                                          x_to=q.umap_1, y_to=q.umap_2))
    edges = pd.DataFrame(edge_rows, columns=["clone", "state", "time_from", "time_to",
                                             "x_from", "y_from", "x_to", "y_to"])

    # Original Scanpy plotting path, with same full dataset + category colors.
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    sc.pl.umap(adata, color=[clone_key], size=size,
               legend_loc="right margin", frameon=False, legend_fontsize=8,
               legend_fontweight="normal", ax=ax, show=False, palette=palette)
    # Scanpy can reorder categories; palette belongs to the full category list.
    if ax.collections:
        for collection in ax.collections:
            collection.set_alpha(alpha_bg)

    for edge in edge_rows:
        _draw_segmented_arrow(
            ax,
            edge["x_from"], edge["y_from"],
            edge["x_to"], edge["y_to"],
            seg_len=arrow_seg_len,
            gap_frac=arrow_gap_frac,
            lw=arrow_lw,
            head_scale=arrow_head_scale,
            min_segments=arrow_min_segments,
            zorder=6,
            shrink_start=arrow_shrink_start,
            shrink_end=arrow_shrink_end,
        )

    for clone, chunk in centers.groupby("clone", sort=False):
        ax.scatter(chunk.umap_1, chunk.umap_2, s=center_size,
                   color=color_lookup.get(str(clone), "#CCCCCC"),
                   edgecolors="black", linewidths=0.5, zorder=5)

    # Keep Scanpy clone legend. No marker legend necessary.
    ax.set_title(f"{rep_key} - {clone_key}")
    plt.subplots_adjust(left=0.1, right=0.8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"umap_{rep_key}_{clone_key}_time_state_arrow"
    figure_path = output_dir / f"{prefix}.pdf"
    fig.savefig(figure_path, facecolor="white", edgecolor="none", transparent=False,
                bbox_inches="tight")
    centers.to_csv(output_dir / f"{prefix}_centers.csv", index=False)
    edges.to_csv(output_dir / f"{prefix}_edges.csv", index=False)
    print(f"Saved: {figure_path.resolve()} | centers={len(centers)} edges={len(edges)}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return centers, edges

# Evaluation and conservation plotting (adapted from the standalone scripts).
# Uses embeddings already stored in adata.obsm by Figure4_S3.ipynb.
from matplotlib.patches import Rectangle, Patch
import math

RAW_METRIC_NAMES = {
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "silhouette_label": "Silhouette label",
}
METRIC_COLS = list(RAW_METRIC_NAMES.values())

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

        # In v0.5.1, None activates default BatchCorrection; enforce bio-only.
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

    # Combine results_all across the selected labels.
    full = pd.concat(tables, axis=1)
    conservation_cols = [f"{label_name} Conservation" for label_name, _ in label_configs]
    full.index.name = "Embedding"
    full.to_csv(output_dir / "all_metrics.csv", float_format="%.6f")

    summary = full[conservation_cols].copy()
    summary.to_csv(output_dir / "summary_scores.csv", float_format="%.6f")
    draw_summary_svg(summary, output_dir / "summary_scores.svg")
    return full, summary, benchmarkers


# Scatter appearance and method order correspond to notebook representations.
METHOD_ORDER = ["X_pca", "scVI", "DeepTracing_UV", "DeepTracing_LV", "DeepTracing_TV", "DeepTracing_IV"]
DISPLAY_NAMES = {
    "X_pca": "PCA", "scVI": "scVI", "DeepTracing_UV": "DeepTracing UV",
    "DeepTracing_LV": "DeepTracing LV", "DeepTracing_TV": "DeepTracing TV",
    "DeepTracing_IV": "DeepTracing IV",
}
METHOD_COLORS = {
    "X_pca": "#4C72B0", "scVI": "#DD8452", "DeepTracing_UV": "#55A868",
    "DeepTracing_LV": "#8172B2", "DeepTracing_TV": "#CCB974", "DeepTracing_IV": "#C44E52",
}
PAIRS = [("Lineage Conservation", "Stage Conservation")]
SHOW_POINT_LABELS = False
POINT_SIZE = 145
TICK_STEP_X = 0.05
TICK_STEP_Y = 0.1

def padded_axis_range(values, step, hard_min=None):
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    pad = max(0.025, (vmax - vmin) * 0.10)
    low = math.floor((vmin - pad) / step) * step

    # If padding would exceed the theoretical maximum of 1.0,
    # leave space to 1.02 instead of rounding up to 1.1.
    # This avoids clipping points at 1.0 without showing ticks above 1.0.
    if vmax + pad > 1.0:
        high = 1.02
    else:
        high = math.ceil((vmax + pad) / step) * step

    if hard_min is not None:
        low = max(hard_min, low)
    else:
        low = max(0.0, low)

    if high <= low:
        low = hard_min if hard_min is not None else max(0.0, vmin - step)
        high = vmax + step

    return round(low, 5), round(high, 5)

def plot_pair(df, x_col, y_col, out_dir):
    # The Lineage axis has no fixed lower bound.
    xlim = padded_axis_range(df[x_col].values, step=TICK_STEP_X)

    # The Stage axis has a minimum lower bound of 0.1.
    ylim = padded_axis_range(df[y_col].values, step=TICK_STEP_Y, hard_min=0.1)

    # Use a square 7.5-by-7.5-inch canvas.
    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)

    # Preserve the original plotting settings.

    x_median = float(df[x_col].median())
    y_median = float(df[y_col].median())

    median_style = dict(
        color="#707070", linestyle="--", linewidth=1.3,
        alpha=0.85, zorder=2,
    )
    ax.axvline(x=x_median, **median_style)
    ax.axhline(y=y_median, **median_style)

    for method in METHOD_ORDER:
        x = float(df.loc[method, x_col])
        y = float(df.loc[method, y_col])

        ax.scatter(
            x, y,
            marker="s",
            s=POINT_SIZE,
            c=METHOD_COLORS[method],
            edgecolors="black",
            linewidths=0.65,
            zorder=3,
        )

        if SHOW_POINT_LABELS:
            ax.annotate(
                DISPLAY_NAMES[method], (x, y),
                xytext=(6, 7), textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Do not force equal data-axis scaling; fill the square canvas.
    # Axis tick spacing follows each metric range independently.

    # Set ticks separately for both axes.
    x_ticks = np.round(np.arange(xlim[0], xlim[1] + 1e-8, TICK_STEP_X), 2)
    y_ticks = np.round(np.arange(ylim[0], ylim[1] + 1e-8, TICK_STEP_Y), 2)

    # Suppress tick labels above 1.0.
    x_ticks = x_ticks[x_ticks <= 1.0 + 1e-8]
    y_ticks = y_ticks[y_ticks <= 1.0 + 1e-8]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xlabel(x_col, fontsize=13, labelpad=9)
    ax.set_ylabel(y_col, fontsize=13, labelpad=9)

    ax.grid(True, color="#B0B0B0", alpha=0.24, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, direction="out", length=4.5)

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor="black",
            linewidth=0.5,
            label=DISPLAY_NAMES[m],
        )
        for m in METHOD_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # Position the legend just below the x-axis.
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=1.5,
        handlelength=1.4,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{x_col.replace(' ', '_')}_vs_{y_col.replace(' ', '_')}_scatter"
    svg_path = out_dir / f"{stem}.svg"
    pdf_path = out_dir / f"{stem}.pdf"

    # Save without bbox_inches="tight" when using constrained_layout.
    fig.savefig(svg_path, format="svg", dpi=300, facecolor="white")
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    plt.show()  # Display the generated scatter in Jupyter.
    plt.close(fig)

    print(f"[{x_col} vs {y_col}]")
    print(f"  {x_col} median : {x_median:.4f}")
    print(f"  {y_col} median : {y_median:.4f}")
    print(f"  axis limits   : x={xlim}, y={ylim}")
    print(f"  saved         : {svg_path}")
    print(f"  saved         : {pdf_path}")


def plot_conservation_scatter(summary, output_dir, pairs=PAIRS):
    """Draw original pairwise conservation scatter from in-memory summary scores.

    SVG and PDF are saved in output_dir; the figures are also displayed inline.
    """
    output_dir = Path(output_dir)
    required = sorted({col for pair in pairs for col in pair})
    missing_cols = [col for col in required if col not in summary.columns]
    missing_methods = [method for method in METHOD_ORDER if method not in summary.index]
    if missing_cols or missing_methods:
        raise ValueError(f"Scatter inputs missing: columns={missing_cols}, embeddings={missing_methods}")
    df = summary.loc[METHOD_ORDER, required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Non-finite conservation scores detected")
    print("\nConservation scores:\n", df.round(4))
    style = {"font.family": "DejaVu Serif", "mathtext.fontset": "stix", "font.size": 11,
             "axes.linewidth": 1.1, "svg.fonttype": "none", "pdf.fonttype": 42}
    with plt.rc_context(style):
        for x_col, y_col in pairs:
            plot_pair(df, x_col, y_col, output_dir)

