"""Rep3 temporal gene analysis: raw counts, decoder means, Top20 heatmaps and bars.

All output files are written under ./results_gene_analyse by default.
Raw counts must come from the original, unnormalized h5ad, not the normalized
AnnData used for plotting embeddings. Decoder means must match original cell
order and exact training HVG order (4000 genes).
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
GENE_ANALYSIS_ROOT = Path('./results_gene_analyse')
GENE_SHOW_FIGURES = True
raw_DATA_FILE = Path('./data/Tumor_Immune_Evasion_eTRACER/processed_data/eTracer_lineage_usable.h5ad')
raw_OUT = GENE_ANALYSIS_ROOT / 'raw_temporal_DE'
raw_CLONE_KEY = 'lineageGrp_final'
raw_STATE_KEY = 'tumor_state'
raw_TIME_KEY = 'time_week'
raw_TIME_PAIRS = [(0, 4), (4, 6)]
raw_MIN_CELLS_PER_TIME = 10
raw_MIN_EXPRESSED_CELLS = 10
raw_MIN_DETECTION_FRACTION = 0.05
raw_LOG2FC_THRESHOLD = 0.5
raw_FDR_THRESHOLD = 0.05
raw_TARGET_SUM = 10000.0
raw_PSEUDOCOUNT = 1.0
raw_MIN_TOTAL_COUNTS = 1

def raw_clean_filename(s):
    return re.sub('[^a-zA-Z0-9_.-]+', '_', str(s))[:90]

def raw_matrix_sums(X):
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=0)).ravel().astype(float)
    return np.asarray(X.sum(axis=0), dtype=float).ravel()

def raw_matrix_counts(X):
    if sparse.issparse(X):
        return np.asarray(X.getnnz(axis=0)).ravel().astype(float)
    return np.count_nonzero(X, axis=0).astype(float)

def raw_ensure_counts(adata):
    """Source is original file .X (before the user's HVG/normalization)."""
    X = adata.X
    if X is None:
        raise ValueError('adata.X is empty. Need original raw counts.')
    arr = X.data if sparse.issparse(X) else np.asarray(X)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError('Expression matrix must contain finite nonnegative raw counts.')
    if not np.allclose(arr[:min(arr.size, 100000)], np.rint(arr[:min(arr.size, 100000)]), atol=1e-05):
        raise ValueError('adata.X does not appear to be raw integer counts. Point DATA_FILE to the original unnormalized h5ad.')
    return X.tocsr() if sparse.issparse(X) else np.asarray(X)

def raw_make_group_stats(X, idx, totals):
    """Produce per-gene means and Welch t-test sufficient stats without densifying NxG."""
    part = X[idx, :]
    norm = raw_TARGET_SUM / totals[idx]
    if sparse.issparse(part):
        normx = sparse.diags(norm) @ part
        normx = normx.tocsr()
        logs = normx.copy().astype(np.float64)
        logs.data = np.log1p(logs.data)
        mu = raw_matrix_sums(normx) / len(idx)
        log_mean = raw_matrix_sums(logs) / len(idx)
        log_squares = raw_matrix_sums(logs.multiply(logs)) / len(idx)
    else:
        normx = np.asarray(part, dtype=np.float64) * norm[:, None]
        logs = np.log1p(normx)
        mu = normx.mean(axis=0)
        log_mean = logs.mean(axis=0)
        log_squares = np.square(logs).mean(axis=0)
    n = len(idx)
    log_var = np.maximum((log_squares - log_mean ** 2) * n / max(n - 1, 1), 0)
    detected = raw_matrix_counts(part)
    return {'n': n, 'mu': mu, 'mean_log': log_mean, 'var_log': log_var, 'detected': detected, 'frac': detected / n}

def raw_compare(st0, st1, genes, group_type, group_name, t0, t1):
    n0, n1 = (st0['n'], st1['n'])
    log2fc = np.log2((st1['mu'] + raw_PSEUDOCOUNT) / (st0['mu'] + raw_PSEUDOCOUNT))
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat, pval = stats.ttest_ind_from_stats(mean1=st1['mean_log'], std1=np.sqrt(st1['var_log']), nobs1=n1, mean2=st0['mean_log'], std2=np.sqrt(st0['var_log']), nobs2=n0, equal_var=False)
    testable = (st0['detected'] + st1['detected'] >= raw_MIN_EXPRESSED_CELLS) & (np.maximum(st0['frac'], st1['frac']) >= raw_MIN_DETECTION_FRACTION)
    pval = np.where(np.isfinite(pval), pval, 1.0)
    pval = np.clip(pval, 0, 1)
    padj = np.full(len(genes), np.nan)
    if testable.any():
        padj[testable] = multipletests(pval[testable], method='fdr_bh')[1]
    result = pd.DataFrame({'gene': genes, 'group_type': group_type, 'group': group_name, 'time_from': t0, 'time_to': t1, 'n_from': n0, 'n_to': n1, 'mean_normalized_from': st0['mu'], 'mean_normalized_to': st1['mu'], 'log2FC': log2fc, 'abs_log2FC': np.abs(log2fc), 'mean_log1p_from': st0['mean_log'], 'mean_log1p_to': st1['mean_log'], 'pct_from': st0['frac'], 'pct_to': st1['frac'], 'welch_t': tstat, 'p_value': pval, 'fdr': padj, 'tested': testable, 'significant': testable & (padj < raw_FDR_THRESHOLD) & (np.abs(log2fc) >= raw_LOG2FC_THRESHOLD)})
    valid = result.loc[result.tested]
    summary = {'group_type': group_type, 'group': group_name, 'time_from': t0, 'time_to': t1, 'comparison': f'{t0}->{t1}', 'n_from': n0, 'n_to': n1, 'tested_genes': len(valid), 'median_abs_log2FC': valid.abs_log2FC.median() if len(valid) else np.nan, 'fraction_large_fc': (valid.abs_log2FC >= raw_LOG2FC_THRESHOLD).mean() if len(valid) else np.nan, 'n_sig': int(valid.significant.sum()), 'fraction_sig': valid.significant.mean() if len(valid) else np.nan}
    return (result, summary)

def raw_plots(summary):
    metric_list = [('median_abs_log2FC', 'Median |log2FC|'), ('fraction_large_fc', f'Fraction genes |log2FC| >= {raw_LOG2FC_THRESHOLD}'), ('fraction_sig', 'Fraction significant genes (exploratory)')]
    types = ['Lineage']
    periods = [f'{a}->{b}' for a, b in raw_TIME_PAIRS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    rng = np.random.default_rng(42)
    for ax, (key, label) in zip(axes, metric_list):
        positions = np.arange(len(types))
        for j, t in enumerate(types):
            for k, period in enumerate(periods):
                sub = summary.loc[(summary.group_type == t) & (summary.comparison == period), key].dropna().to_numpy()
                offset = (k - 1) * 0.23
                xx = np.full(len(sub), j + offset) + rng.uniform(-0.05, 0.05, len(sub))
                ax.scatter(xx, sub, s=14, alpha=0.5, color=['#325d88', '#c27c3d', '#538d75'][k], label=period if j == 0 else None)
                if len(sub):
                    ax.plot([j + offset - 0.07, j + offset + 0.07], [np.median(sub)] * 2, lw=2.5, color='black')
        ax.set_xticks(positions)
        ax.set_xticklabels(types)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.15)
    axes[0].legend(title='Time', frameon=False, fontsize=8)
    fig.savefig(raw_OUT / 'comparison_overview.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    agg = summary.groupby(['group_type', 'comparison'], observed=True)[[c for c, _ in metric_list]].median().reset_index()
    agg.to_csv(raw_OUT / 'summary_by_category_and_time.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (key, label) in zip(axes, metric_list):
        data = np.full((len(types), len(periods)), np.nan)
        for j, t in enumerate(types):
            for k, period in enumerate(periods):
                s = agg.loc[(agg.group_type == t) & (agg.comparison == period), key]
                if len(s):
                    data[j, k] = s.iloc[0]
        image = ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods)
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types)
        ax.set_title(label, fontsize=10)
        for j in range(len(types)):
            for k in range(len(periods)):
                if np.isfinite(data[j, k]):
                    ax.text(k, j, f'{data[j, k]:.3f}', ha='center', va='center', fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.73)
    fig.savefig(raw_OUT / 'category_median_heatmaps.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

def raw_main():
    raw_OUT.mkdir(parents=True, exist_ok=True)
    gene_dir = raw_OUT / 'gene_tables'
    gene_dir.mkdir(exist_ok=True)
    if not raw_DATA_FILE.exists():
        raise FileNotFoundError(f'Missing {raw_DATA_FILE.resolve()}')
    a = ad.read_h5ad(raw_DATA_FILE)
    sc.pp.highly_variable_genes(a, flavor='seurat_v3', n_top_genes=4000, subset=True)
    for col in [raw_CLONE_KEY, raw_TIME_KEY]:
        if col not in a.obs:
            raise KeyError(f'obs missing {col}. Available: {list(a.obs.columns)}')
    X = raw_ensure_counts(a)
    genes = a.var_names.astype(str).to_numpy()
    totals = np.asarray(X.sum(axis=1)).ravel().astype(float)
    keep = np.flatnonzero(totals >= raw_MIN_TOTAL_COUNTS)
    if len(keep) < len(totals):
        print(f'Excluding {len(totals) - len(keep)} zero-count cells')
        a = a[keep].copy()
        X = X[keep]
        totals = totals[keep]
    obs = a.obs.copy()
    obs[raw_CLONE_KEY] = obs[raw_CLONE_KEY].astype('string').fillna('Missing')
    obs['_time_num'] = pd.to_numeric(obs[raw_TIME_KEY].astype(str), errors='coerce')
    if obs['_time_num'].isna().any():
        raise ValueError('time must be numeric 0,4,6; check adata.obs[time]')
    print(f'Loaded {X.shape[0]} cells x {X.shape[1]} genes (original unnormalized counts).')
    print('Time counts:', obs['_time_num'].value_counts().sort_index().to_dict())
    print('Lineage count:', obs[raw_CLONE_KEY].nunique())
    masks = {'Lineage': obs[raw_CLONE_KEY]}
    summaries = []
    skipped = []
    for kind, identifiers in masks.items():
        group_names = sorted(identifiers.dropna().unique())
        for name in group_names:
            if 'Missing' in str(name):
                continue
            imask = (identifiers == name).to_numpy(dtype=bool)
            time_stats = {}
            for t in sorted({x for pair in raw_TIME_PAIRS for x in pair}):
                ix = np.flatnonzero(imask & (obs['_time_num'].to_numpy() == t))
                if len(ix) >= raw_MIN_CELLS_PER_TIME:
                    time_stats[t] = raw_make_group_stats(X, ix, totals)
            for t0, t1 in raw_TIME_PAIRS:
                if t0 not in time_stats or t1 not in time_stats:
                    skipped.append({'group_type': kind, 'group': name, 'time_from': t0, 'time_to': t1, 'n_from': int(np.sum(imask & (obs['_time_num'].to_numpy() == t0))), 'n_to': int(np.sum(imask & (obs['_time_num'].to_numpy() == t1)))})
                    continue
                res, s = raw_compare(time_stats[t0], time_stats[t1], genes, kind, name, t0, t1)
                summaries.append(s)
                fname = f'{kind}__{raw_clean_filename(name)}__{t0}_to_{t1}.csv.gz'
                res.to_csv(gene_dir / fname, index=False, compression='gzip')
                print(f"{kind:11s} {str(name):22s} {t0}->{t1} n={s['n_from']}/{s['n_to']} tested={s['tested_genes']} significant={s['n_sig']}")
    if not summaries:
        raise RuntimeError('No comparisons passed MIN_CELLS_PER_TIME; reduce threshold or inspect labels.')
    summary = pd.DataFrame(summaries)
    summary.to_csv(raw_OUT / 'all_comparison_summary.csv', index=False)
    pd.DataFrame(skipped).to_csv(raw_OUT / 'skipped_comparisons.csv', index=False)
    raw_plots(summary)
    print('\nCategory medians (descriptive, not a statistical test):')
    print(summary.groupby(['group_type', 'comparison'])[['median_abs_log2FC', 'fraction_large_fc', 'fraction_sig']].median().round(4).to_string())
    print('\nSaved to:', raw_OUT.resolve())
    print('NOTE: cell-level Welch p-values are exploratory, not biological-replicate inference.')
den_DATA_FILE = Path('./data/Tumor_Immune_Evasion_eTRACER/processed_data/eTracer_lineage_usable.h5ad')
den_DENOISED_FILE = Path('./denoised_counts.txt')
den_OUT = GENE_ANALYSIS_ROOT / 'denoised_temporal_DE'
den_TRAIN_HVG_FILE = None
den_CLONE_KEY = 'lineageGrp_final'
den_STATE_KEY = 'tumor_state'
den_TIME_KEY = 'time_week'
den_TIME_PAIRS = [(0, 4), (4, 6)]
den_N_HVG = 4000
den_MIN_CELLS_PER_TIME = 10
den_MIN_EXPRESSED_CELLS = 10
den_MIN_DETECTION_FRACTION = 0.05
den_MIN_TOTAL_COUNTS = 1
den_TARGET_SUM = 10000.0
den_PSEUDOCOUNT = 1.0
den_LOG2FC_THRESHOLD = 0.5
den_FDR_THRESHOLD = 0.05
den_TOP_N = 20
den_MIN_SIGNIFICANT_GROUPS = {'Lineage': 2}

def den_safe_name(s):
    return re.sub('[^a-zA-Z0-9_.-]+', '_', str(s))[:90]

def den_natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)', str(s))]

def den_load_gene_names(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        col = 'gene' if 'gene' in df.columns else df.columns[0]
        names = df[col].astype(str).tolist()
    else:
        names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(names) != den_N_HVG or len(set(names)) != den_N_HVG:
        raise ValueError('Saved training HVG list must contain exactly 4000 unique names.')
    return names

def den_load_inputs():
    if not den_DATA_FILE.exists():
        raise FileNotFoundError(f'Raw AnnData not found: {den_DATA_FILE}')
    if not den_DENOISED_FILE.exists():
        raise FileNotFoundError(f'Denoised matrix not found: {den_DENOISED_FILE}; update DENOISED_FILE.')
    a = ad.read_h5ad(den_DATA_FILE)
    for key in [den_CLONE_KEY, den_TIME_KEY]:
        if key not in a.obs:
            raise KeyError(f'adata.obs missing {key}')
    if not a.var_names.is_unique:
        raise ValueError('adata.var_names must be unique to align decoder genes.')
    raw_counts = a.X.tocsr() if sparse.issparse(a.X) else np.asarray(a.X)
    finite_data = raw_counts.data if sparse.issparse(raw_counts) else raw_counts
    if not np.all(np.isfinite(finite_data)) or np.any(finite_data < 0):
        raise ValueError('Original adata.X must contain nonnegative finite raw counts')
    if finite_data.size and (not np.allclose(finite_data[:100000], np.rint(finite_data[:100000]), atol=1e-05)):
        raise ValueError('Original adata.X does not look like integer raw counts.')
    if den_TRAIN_HVG_FILE is not None:
        names = den_load_gene_names(den_TRAIN_HVG_FILE)
    else:
        sc.pp.highly_variable_genes(a, flavor='seurat_v3', n_top_genes=den_N_HVG, subset=True)
        names = a.var_names.astype(str).tolist()
        print('[HVG] Recomputed training-like seurat_v3 4000 HVGs. Verify exact gene/order alignment.')
    if len(names) != den_N_HVG:
        raise ValueError(f'Expected 4000 HVGs, got {len(names)}')
    if den_TRAIN_HVG_FILE is not None:
        if not set(names).issubset(set(a.var_names.astype(str))):
            raise ValueError('Training gene list contains genes absent from raw AnnData.')
        a = a[:, names].copy()
    X_raw = a.X.tocsr() if sparse.issparse(a.X) else np.asarray(a.X)
    den = np.loadtxt(den_DENOISED_FILE, delimiter=',', dtype=np.float64, ndmin=2)
    if den.shape != (a.n_obs, a.n_vars):
        raise ValueError(f'Denoised matrix shape {den.shape}, expected {(a.n_obs, a.n_vars)}; verify file and order.')
    if not np.all(np.isfinite(den)) or np.min(den) < 0:
        raise ValueError('Denoised decoder mean contains negative/nonfinite values.')
    totals = np.asarray(X_raw.sum(axis=1)).ravel().astype(np.float64)
    if np.any(totals < den_MIN_TOTAL_COUNTS):
        raise ValueError('Found HVG zero-count cell: need to match the training cell-filtering exactly. Cannot remove rows without verifying denoised-matrix row alignment.')
    median_lib = float(np.median(totals))
    if median_lib <= 0:
        raise ValueError('Median library size must be positive')
    den_10k = den * (den_TARGET_SUM / median_lib)
    print(f'[DATA] cells={a.n_obs} genes={a.n_vars} median_HVG_counts={median_lib:.3f} decoder_to_10k={den_TARGET_SUM / median_lib:.6g}')
    print('[DATA] Denoised predictions are compared after fixed scale conversion, NOT independent per-cell normalization.')
    (den_OUT / 'training_gene_order.txt').write_text('\n'.join(names) + '\n')
    return (a.obs.copy(), X_raw, den_10k, totals, np.array(names, dtype=object))

def den_raw_detection(part):
    if sparse.issparse(part):
        detected = np.asarray(part.getnnz(axis=0)).ravel()
    else:
        detected = np.count_nonzero(part, axis=0)
    return detected.astype(np.int64)

def den_group_stats(den_10k, X_raw, idx):
    arr = den_10k[idx, :]
    logs = np.log1p(arr)
    n = len(idx)
    det = den_raw_detection(X_raw[idx, :])
    return {'n': n, 'mu': arr.mean(axis=0), 'mean_log': logs.mean(axis=0), 'var_log': logs.var(axis=0, ddof=1), 'detected': det, 'frac': det / n}

def den_compare(s0, s1, genes, kind, group, t0, t1):
    n0, n1 = (s0['n'], s1['n'])
    fc = np.log2((s1['mu'] + den_PSEUDOCOUNT) / (s0['mu'] + den_PSEUDOCOUNT))
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat, pval = stats.ttest_ind_from_stats(mean1=s1['mean_log'], std1=np.sqrt(s1['var_log']), nobs1=n1, mean2=s0['mean_log'], std2=np.sqrt(s0['var_log']), nobs2=n0, equal_var=False)
    tested = (s0['detected'] + s1['detected'] >= den_MIN_EXPRESSED_CELLS) & (np.maximum(s0['frac'], s1['frac']) >= den_MIN_DETECTION_FRACTION)
    pval = np.where(np.isfinite(pval), pval, 1.0)
    pval = np.clip(pval, 0, 1)
    fdr = np.full(len(genes), np.nan, dtype=float)
    if tested.any():
        fdr[tested] = multipletests(pval[tested], method='fdr_bh')[1]
    sig = tested & (fdr < den_FDR_THRESHOLD) & (np.abs(fc) >= den_LOG2FC_THRESHOLD)
    result = pd.DataFrame({'gene': genes, 'group_type': kind, 'group': group, 'time_from': t0, 'time_to': t1, 'n_from': n0, 'n_to': n1, 'mean_normalized_from': s0['mu'], 'mean_normalized_to': s1['mu'], 'log2FC': fc, 'abs_log2FC': np.abs(fc), 'mean_log1p_from': s0['mean_log'], 'mean_log1p_to': s1['mean_log'], 'pct_from': s0['frac'], 'pct_to': s1['frac'], 'welch_t': tstat, 'p_value': pval, 'fdr': fdr, 'tested': tested, 'significant': sig})
    ss = result.loc[tested]
    summary = {'group_type': kind, 'group': group, 'time_from': t0, 'time_to': t1, 'comparison': f'{t0}->{t1}', 'n_from': n0, 'n_to': n1, 'tested_genes': len(ss), 'median_abs_log2FC': ss.abs_log2FC.median() if len(ss) else np.nan, 'fraction_large_fc': (ss.abs_log2FC >= den_LOG2FC_THRESHOLD).mean() if len(ss) else np.nan, 'n_sig': int(ss.significant.sum()), 'fraction_sig': ss.significant.mean() if len(ss) else np.nan}
    return (result, summary)

def den_heatmap_for(df, groups, genes, kind, t0, t1):
    mat = df.loc[df.gene.isin(genes)].pivot(index='gene', columns='group', values='log2FC').reindex(index=genes, columns=groups)
    tested = df.loc[df.gene.isin(genes)].pivot(index='gene', columns='group', values='tested').reindex(index=genes, columns=groups).fillna(False)
    mat = mat.where(tested)
    mat.to_csv(den_OUT / f'{kind}__{t0}_to_{t1}__top{den_TOP_N}_score_matrix.csv')
    arr = mat.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 0.1)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#d8d8d8')
    fig_w = max(9, min(29, 3.0 + len(groups) * 0.42))
    fig_h = max(6.5, 2.0 + len(genes) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(arr), aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=90, fontsize=8)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9)
    ax.set_xlabel('Lineage')
    ax.set_ylabel('Selected genes')
    ax.set_title(f'DeepTracing denoised | {kind} | {t0}->{t1} | Top {len(genes)} recurrent genes')
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.025)
    cbar.set_label('Denoised predicted log2FC (later / earlier)')
    fig.savefig(den_OUT / f'{kind}__{t0}_to_{t1}__top{den_TOP_N}_heatmap.png', dpi=260)
    fig.savefig(den_OUT / f'{kind}__{t0}_to_{t1}__top{den_TOP_N}_heatmap.pdf')
    plt.close(fig)

def den_make_top_heatmaps(summaries):
    for kind in ['Lineage']:
        for t0, t1 in den_TIME_PAIRS:
            info = summaries[(summaries.group_type == kind) & (summaries.time_from == t0) & (summaries.time_to == t1)]
            if info.empty:
                continue
            groupnames = sorted(info['group'].astype(str).tolist(), key=den_natural_key)
            frames = []
            for name in groupnames:
                file = den_OUT / 'gene_tables' / f'{kind}__{den_safe_name(name)}__{t0}_to_{t1}.csv.gz'
                tab = pd.read_csv(file, usecols=['gene', 'log2FC', 'fdr', 'tested', 'significant'])
                tab['group'] = name
                frames.append(tab)
            df = pd.concat(frames, ignore_index=True)
            tested_df = df.loc[df.tested].copy()
            if tested_df.empty:
                continue
            stats_df = tested_df.groupby('gene').agg(support=('significant', 'sum'), median_abs_log2FC=('log2FC', lambda s: float(np.median(np.abs(s)))), n_groups_tested=('group', 'nunique')).reset_index()
            stats_df = stats_df.sort_values(['support', 'median_abs_log2FC', 'gene'], ascending=[False, False, True])
            stats_df.to_csv(den_OUT / f'{kind}__{t0}_to_{t1}__gene_ranking.csv', index=False)
            min_support = den_MIN_SIGNIFICANT_GROUPS[kind]
            eligible = stats_df[stats_df.support >= min_support]
            chosen = (eligible if len(eligible) >= den_TOP_N else stats_df).head(den_TOP_N)
            names = chosen.gene.tolist()
            pd.DataFrame({'gene': names}).to_csv(den_OUT / f'{kind}__{t0}_to_{t1}__top{den_TOP_N}_genes.csv', index=False)
            den_heatmap_for(df, groupnames, names, kind, t0, t1)
            print(f'[HEATMAP] {kind} {t0}->{t1} groups={len(groupnames)} selected={len(names)}')

def den_main():
    den_OUT.mkdir(parents=True, exist_ok=True)
    gene_dir = den_OUT / 'gene_tables'
    gene_dir.mkdir(parents=True, exist_ok=True)
    obs, Xraw, den_10k, totals, genes = den_load_inputs()
    for key in [den_CLONE_KEY]:
        obs[key] = obs[key].astype('string').fillna('Missing')
    time = pd.to_numeric(obs[den_TIME_KEY].astype(str), errors='coerce').to_numpy()
    if not np.isfinite(time).all():
        raise ValueError(f'{den_TIME_KEY} must be numeric.')
    kind_series = {'Lineage': obs[den_CLONE_KEY]}
    summaries = []
    skipped = []
    for kind, labels in kind_series.items():
        label_vals = labels.astype(str).to_numpy()
        for name in sorted(pd.unique(label_vals), key=den_natural_key):
            if 'Missing' in name:
                continue
            imask = label_vals == name
            stats_by_time = {}
            counts_by_time = {}
            for t in [0, 4, 6]:
                idx = np.flatnonzero(imask & (time == t))
                counts_by_time[t] = len(idx)
                if len(idx) >= den_MIN_CELLS_PER_TIME:
                    stats_by_time[t] = den_group_stats(den_10k, Xraw, idx)
            for t0, t1 in den_TIME_PAIRS:
                if t0 not in stats_by_time or t1 not in stats_by_time:
                    skipped.append({'group_type': kind, 'group': name, 'time_from': t0, 'time_to': t1, 'n_from': counts_by_time[t0], 'n_to': counts_by_time[t1]})
                    continue
                result, summary = den_compare(stats_by_time[t0], stats_by_time[t1], genes, kind, name, t0, t1)
                summaries.append(summary)
                result.to_csv(gene_dir / f'{kind}__{den_safe_name(name)}__{t0}_to_{t1}.csv.gz', index=False, compression='gzip')
                print(f"[DE] {kind:11s} {name:19s} {t0}->{t1} n={summary['n_from']}/{summary['n_to']} tested={summary['tested_genes']} candidates={summary['n_sig']}")
    if not summaries:
        raise RuntimeError('No group comparison qualified. Check group/time labels and MIN_CELLS_PER_TIME.')
    summary = pd.DataFrame(summaries)
    summary.to_csv(den_OUT / 'all_comparison_summary.csv', index=False)
    pd.DataFrame(skipped).to_csv(den_OUT / 'skipped_comparisons.csv', index=False)
    ag = summary.groupby(['group_type', 'comparison'])[['median_abs_log2FC', 'fraction_large_fc', 'fraction_sig']].median()
    ag.to_csv(den_OUT / 'summary_by_category_and_time.csv')
    den_make_top_heatmaps(summary)
    print('\n[SUMMARY] category medians:')
    print(ag.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'\n[DONE] {den_OUT.resolve()}')
    print('CAUTION: Welch/FDR on decoder-denoised values is exploratory, not an independently calibrated DE test.')
heat_RESULTS_ROOT = GENE_ANALYSIS_ROOT
heat_RAW_DIR = heat_RESULTS_ROOT / 'raw_temporal_DE'
heat_DENOISED_DIR = heat_RESULTS_ROOT / 'denoised_temporal_DE'
heat_OUTPUT_DIR = heat_RESULTS_ROOT / 'raw_vs_denoised_top20_median_abs_heatmap'
heat_GROUP_TYPE = 'Lineage'
heat_TIME_PAIRS = [(0, 4), (4, 6)]
heat_TOP_N = 20
heat_LOG2FC_CUTOFF = 0.5
heat_FDR_CUTOFF = 0.05
heat_FIXED_VMAX = None

def heat_natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)', str(s))]

def heat_safe_name(s):
    return re.sub('[^a-zA-Z0-9_.-]+', '_', str(s))[:90]

def heat_load_summary(directory):
    path = directory / 'all_comparison_summary.csv'
    if not path.exists():
        raise FileNotFoundError('Missing summary: {}\nCheck RESULTS_ROOT in CONFIG.'.format(path.resolve()))
    df = pd.read_csv(path)
    required = {'group_type', 'group', 'time_from', 'time_to'}
    if not required.issubset(df.columns):
        raise ValueError('Summary {} missing columns {}'.format(path, required - set(df.columns)))
    df['group'] = df['group'].astype(str)
    return df

def heat_common_groups(raw_summary, den_summary, t0, t1):

    def groups(frame):
        x = frame.loc[(frame.group_type == heat_GROUP_TYPE) & (frame.time_from == t0) & (frame.time_to == t1), 'group']
        return set(x.astype(str))
    return sorted(groups(raw_summary) & groups(den_summary), key=heat_natural_key)

def heat_load_tables(directory, groups, t0, t1, allowed_genes=None):
    frames = []
    for name in groups:
        file = directory / 'gene_tables' / '{}__{}__{}_to_{}.csv.gz'.format(heat_GROUP_TYPE, heat_safe_name(name), t0, t1)
        if not file.exists():
            raise FileNotFoundError('Missing DE gene table: {}'.format(file.resolve()))
        tab = pd.read_csv(file, usecols=['gene', 'log2FC', 'fdr', 'tested'])
        tab['gene'] = tab['gene'].astype(str)
        tab['tested'] = tab['tested'].fillna(False).astype(bool)
        if allowed_genes is not None:
            tab = tab[tab.gene.isin(allowed_genes)].copy()
        if tab.gene.duplicated().any():
            raise ValueError('Duplicated gene in {}'.format(file))
        tab['group'] = str(name)
        frames.append(tab)
    return pd.concat(frames, ignore_index=True)

def heat_rank_genes_by_median_abs(df):
    """Latest requested selection logic: Top20 ONLY by median |log2FC|."""
    good = df[df.tested & np.isfinite(df.log2FC)].copy()
    if good.empty:
        raise ValueError('No tested genes after restricting to comparable groups and genes.')
    ranked = good.groupby('gene').agg(median_abs_log2FC=('log2FC', lambda x: float(np.median(np.abs(x)))), median_signed_log2FC=('log2FC', lambda x: float(np.median(x))), n_groups_tested=('group', 'nunique'), support=('log2FC', lambda x: int(np.sum(np.abs(np.asarray(x, dtype=float)) >= heat_LOG2FC_CUTOFF)))).reset_index()
    ranked = ranked.sort_values(['median_abs_log2FC', 'gene'], ascending=[False, True], kind='stable').reset_index(drop=True)
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    top = ranked.head(heat_TOP_N).copy()
    return (top, ranked)

def heat_matrix_for(df, genes, groups):
    sub = df[df.gene.isin(genes)].copy()
    sub.loc[~sub.tested, 'log2FC'] = np.nan
    mat = sub.pivot(index='gene', columns='group', values='log2FC')
    return mat.reindex(index=genes, columns=groups)

def heat_make_figure(matrices, groups, t0, t1, raw_top_df, den_top_df):
    raw_top = raw_top_df['gene'].tolist()
    den_top = den_top_df['gene'].tolist()
    shared = set(raw_top) & set(den_top)
    arrays = [m.to_numpy(dtype=float) for m in matrices]
    finite_parts = [a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()]
    finite = np.concatenate(finite_parts) if finite_parts else np.array([0.0])
    vmax = float(heat_FIXED_VMAX) if heat_FIXED_VMAX is not None else max(0.5, float(np.quantile(np.abs(finite), 0.98)) if finite.size else 0.5)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#dddddd')
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    figw = max(13.5, min(30, 7.4 + len(groups) * 0.68))
    figh = max(8.8, 3.0 + heat_TOP_N * 0.4)
    fig, axs = plt.subplots(1, 2, figsize=(figw, figh), facecolor='white')
    titles = ['(A) Raw Top{} in Raw'.format(heat_TOP_N), '(B) Denoised Top{} in Denoised'.format(heat_TOP_N)]
    im = None
    for k, (ax, mat, title) in enumerate(zip(axs, matrices, titles)):
        arr = mat.to_numpy(dtype=float)
        im = ax.imshow(np.ma.masked_invalid(arr), aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index.tolist(), fontsize=10)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=90, fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.4)
        ax.tick_params(which='minor', left=False, bottom=False)
        ax.tick_params(axis='both', which='major', length=0)
        ax.set_xlabel(heat_GROUP_TYPE.replace('_', ' x '), fontsize=11)
        ax.set_ylabel('Gene', fontsize=11)
        for label in ax.get_yticklabels():
            gene = label.get_text()
            if gene in shared:
                label.set_color('#222222')
            elif k == 1:
                label.set_color('#D97706')
                label.set_fontweight('bold')
            else:
                label.set_color('#888888')
    fig.suptitle('Raw vs DeepTracing denoised | {} | {} -> {} | Lineages={} | Shared genes: {}/{}'.format(heat_GROUP_TYPE.replace('_', ' x '), t0, t1, len(groups), len(shared), heat_TOP_N), fontsize=14, fontweight='bold', y=0.985)
    fig.subplots_adjust(left=0.085, right=0.9, bottom=0.18, top=0.89, wspace=0.36)
    cax = fig.add_axes([0.925, 0.24, 0.015, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('log2FC (later / earlier)', fontsize=10)
    fig.text(0.5, 0.06, 'Top{} genes are selected ONLY by median |log2FC| across all eligible/common {} comparisons.  Heatmap values are signed log2FC.  Gene labels: black = shared; orange = Denoised-only; gray = Raw-only.  Shared color scale: +/-{:.2f}.'.format(heat_TOP_N, heat_GROUP_TYPE, vmax), ha='center', fontsize=9)
    heat_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = '{}__{}_to_{}__raw_vs_denoised_top{}__median_abs_heatmap'.format(heat_GROUP_TYPE, t0, t1, heat_TOP_N)
    fig.savefig(heat_OUTPUT_DIR / (stem + '.png'), dpi=260, facecolor='white', bbox_inches='tight')
    fig.savefig(heat_OUTPUT_DIR / (stem + '.pdf'), facecolor='white', bbox_inches='tight')
    fig.savefig(heat_OUTPUT_DIR / (stem + '.svg'), facecolor='white', bbox_inches='tight')
    if GENE_SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return (stem, vmax)

def heat_process(t0, t1, raw_summary, den_summary, allowed):
    groups = heat_common_groups(raw_summary, den_summary, t0, t1)
    if not groups:
        print('[SKIP] {}->{} no common comparisons'.format(t0, t1))
        return
    raw = heat_load_tables(heat_RAW_DIR, groups, t0, t1, allowed)
    den = heat_load_tables(heat_DENOISED_DIR, groups, t0, t1, allowed)
    common_genes = set(raw.gene) & set(den.gene)
    if not common_genes:
        raise RuntimeError('No genes in common between Raw and Denoised tables.')
    raw = raw[raw.gene.isin(common_genes)].copy()
    den = den[den.gene.isin(common_genes)].copy()
    raw_top_df, raw_rank = heat_rank_genes_by_median_abs(raw)
    den_top_df, den_rank = heat_rank_genes_by_median_abs(den)
    raw_top = raw_top_df['gene'].tolist()
    den_top = den_top_df['gene'].tolist()
    overlap_genes = sorted(set(raw_top) & set(den_top), key=heat_natural_key)
    den_only = sorted(set(den_top) - set(raw_top), key=heat_natural_key)
    raw_only = sorted(set(raw_top) - set(den_top), key=heat_natural_key)
    matrices = [heat_matrix_for(raw, raw_top, groups), heat_matrix_for(den, den_top, groups)]
    stem, vmax = heat_make_figure(matrices, groups, t0, t1, raw_top_df, den_top_df)
    matrices[0].to_csv(heat_OUTPUT_DIR / (stem + '__raw_top_in_raw.csv'))
    matrices[1].to_csv(heat_OUTPUT_DIR / (stem + '__denoised_top_in_denoised.csv'))
    raw_top_df.to_csv(heat_OUTPUT_DIR / (stem + '__raw_top20_ranked.csv'), index=False)
    den_top_df.to_csv(heat_OUTPUT_DIR / (stem + '__denoised_top20_ranked.csv'), index=False)
    raw_rank.to_csv(heat_OUTPUT_DIR / (stem + '__raw_ranking.csv'), index=False)
    den_rank.to_csv(heat_OUTPUT_DIR / (stem + '__denoised_ranking.csv'), index=False)
    pd.DataFrame({'gene': overlap_genes}).to_csv(heat_OUTPUT_DIR / (stem + '__overlap.csv'), index=False)
    pd.DataFrame({'gene': den_only}).to_csv(heat_OUTPUT_DIR / (stem + '__denoised_only.csv'), index=False)
    pd.DataFrame({'gene': raw_only}).to_csv(heat_OUTPUT_DIR / (stem + '__raw_only.csv'), index=False)
    print('[SAVED] {} | clones={} shared genes={} overlap={}/{} | scale +/-{:.3f}'.format(stem, len(groups), len(common_genes), len(overlap_genes), heat_TOP_N, vmax))
    print('  Shared genes:', ', '.join(overlap_genes) if overlap_genes else 'None')
    print('  Denoised-only:', ', '.join(den_only) if den_only else 'None')
    print('  Raw-only:', ', '.join(raw_only) if raw_only else 'None')

def heat_main():
    raw_summary = heat_load_summary(heat_RAW_DIR)
    den_summary = heat_load_summary(heat_DENOISED_DIR)
    order_file = heat_DENOISED_DIR / 'training_gene_order.txt'
    if not order_file.exists():
        raise FileNotFoundError('Missing {}. Use output of analyze_denoised_temporal_DE.py'.format(order_file.resolve()))
    allowed = {x.strip() for x in order_file.read_text().splitlines() if x.strip()}
    if len(allowed) != 4000:
        raise ValueError('Training gene order has {} unique genes; expected 4000.'.format(len(allowed)))
    heat_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Restricting Raw and Denoised to the same 4000 training genes and all common eligible groups.')
    for t0, t1 in heat_TIME_PAIRS:
        heat_process(t0, t1, raw_summary, den_summary, allowed)
    print('Done:', heat_OUTPUT_DIR.resolve())
bar_RESULTS_ROOT = GENE_ANALYSIS_ROOT
bar_RAW_DIR = bar_RESULTS_ROOT / 'raw_temporal_DE'
bar_DENOISED_DIR = bar_RESULTS_ROOT / 'denoised_temporal_DE'
bar_OUTPUT_DIR = bar_RESULTS_ROOT / 'raw_vs_denoised_top20_bars'
bar_GROUP_TYPE = 'Lineage'
bar_TIME_PAIRS = [(0, 4), (4, 6)]
bar_TOP_N = 20
bar_LOG2FC_CUTOFF = 0.5
bar_FDR_CUTOFF = 0.05
bar_MIN_SIGNIFICANT_GROUPS = {'Lineage': 2}

def bar_natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)', str(s))]

def bar_safe_name(s):
    return re.sub('[^a-zA-Z0-9_.-]+', '_', str(s))[:90]

def bar_load_summary(directory):
    path = directory / 'all_comparison_summary.csv'
    if not path.exists():
        raise FileNotFoundError('Missing summary: {}\nCheck RESULTS_ROOT in CONFIG.'.format(path.resolve()))
    df = pd.read_csv(path)
    required = {'group_type', 'group', 'time_from', 'time_to'}
    if not required.issubset(df.columns):
        raise ValueError('Summary {} missing columns {}'.format(path, required - set(df.columns)))
    df['group'] = df['group'].astype(str)
    return df

def bar_common_groups(raw_summary, den_summary, t0, t1):

    def groups(frame):
        x = frame.loc[(frame.group_type == bar_GROUP_TYPE) & (frame.time_from == t0) & (frame.time_to == t1), 'group']
        return set(x.astype(str))
    return sorted(groups(raw_summary) & groups(den_summary), key=bar_natural_key)

def bar_load_tables(directory, groups, t0, t1, allowed_genes=None):
    frames = []
    for name in groups:
        file = directory / 'gene_tables' / '{}__{}__{}_to_{}.csv.gz'.format(bar_GROUP_TYPE, bar_safe_name(name), t0, t1)
        if not file.exists():
            raise FileNotFoundError('Missing DE gene table: {}'.format(file.resolve()))
        tab = pd.read_csv(file, usecols=['gene', 'log2FC', 'fdr', 'tested'])
        tab['gene'] = tab['gene'].astype(str)
        tab['tested'] = tab['tested'].fillna(False).astype(bool)
        if allowed_genes is not None:
            tab = tab[tab.gene.isin(allowed_genes)].copy()
        if tab.gene.duplicated().any():
            raise ValueError('Duplicated gene in {}'.format(file))
        tab['group'] = str(name)
        frames.append(tab)
    return pd.concat(frames, ignore_index=True)

def bar_rank_genes(df):
    """Latest Top20 logic: rank only by median |log2FC|."""
    good = df[df.tested & np.isfinite(df.log2FC)].copy()
    if good.empty:
        raise ValueError('No tested genes after restricting to comparable groups and genes.')
    good['significant'] = good.fdr.fillna(1).lt(bar_FDR_CUTOFF) & good.log2FC.abs().ge(bar_LOG2FC_CUTOFF)
    ranked = good.groupby('gene').agg(support=('significant', 'sum'), median_abs_log2FC=('log2FC', lambda x: float(np.median(np.abs(x)))), median_signed_log2FC=('log2FC', 'median'), n_groups_tested=('group', 'nunique'), fraction_up=('log2FC', lambda x: float(np.mean(x > 0))), fraction_down=('log2FC', lambda x: float(np.mean(x < 0)))).reset_index()
    ranked['direction'] = np.sign(ranked['median_signed_log2FC'])
    ranked['direction_consistency'] = np.where(ranked['direction'] > 0, ranked['fraction_up'], np.where(ranked['direction'] < 0, ranked['fraction_down'], np.nan))
    ranked['plot_value'] = ranked['median_abs_log2FC']
    ranked = ranked.sort_values(['median_abs_log2FC', 'gene'], ascending=[False, True], kind='stable').reset_index(drop=True)
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    return (ranked.head(bar_TOP_N).copy(), ranked)

def bar__label_color(gene, shared, panel):
    if gene in shared:
        return '#222222'
    if panel == 'raw':
        return '#888888'
    return '#D97706'

def bar_make_figure(raw_top_df, den_top_df, t0, t1, groups):
    """Same plot parameters as original signed version, with absolute values."""
    shared = set(raw_top_df['gene']) & set(den_top_df['gene'])
    vals_all = np.r_[raw_top_df['plot_value'].to_numpy(dtype=float), den_top_df['plot_value'].to_numpy(dtype=float)]
    bound = max(0.5, float(np.max(np.abs(vals_all))) * 1.16)
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.2), facecolor='white', sharey=True)
    panels = [(axs[0], raw_top_df, '(A) Raw Top{} in Raw'.format(bar_TOP_N), 'raw'), (axs[1], den_top_df, '(B) Denoised Top{} in Denoised'.format(bar_TOP_N), 'den')]
    for ax, dfp, title, panel in panels:
        genes = dfp['gene'].tolist()
        values = dfp['plot_value'].to_numpy(dtype=float)
        xpos = np.arange(len(dfp))
        colors = '#BF4A48'
        ax.bar(xpos, values, width=0.76, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='#343434', linewidth=1.1)
        ax.set_xticks(xpos)
        ax.set_xticklabels(genes, rotation=65, ha='right', rotation_mode='anchor', fontsize=9)
        ax.set_xlim(-0.65, len(dfp) - 0.35)
        ax.set_ylim(0, bound)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.grid(axis='y', color='#E5E7EB', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
        for label in ax.get_xticklabels():
            gene = label.get_text()
            label.set_color(bar__label_color(gene, shared, panel))
            if panel == 'den' and gene not in shared:
                label.set_fontweight('bold')
        ax.set_xlabel('Gene (ranked by median |log2FC|)', fontsize=10)
    axs[0].set_ylabel('Median |log2FC| across eligible Lineages', fontsize=11)
    fig.suptitle('Raw vs DeepTracing denoised | {} | {} -> {} | Lineages={} | Shared genes: {}/{}'.format(bar_GROUP_TYPE.replace('_', ' x '), t0, t1, len(groups), len(shared), bar_TOP_N), fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.29, top=0.86, wspace=0.12)
    fig.text(0.5, 0.065, 'Gene order: median |log2FC| across tested Lineages (descending).  Bars: median |log2FC| (positive magnitudes only).', ha='center', fontsize=9)
    fig.text(0.5, 0.038, 'Gene labels: black = shared; orange = Denoised-only; gray = Raw-only.  Both panels share the same y-axis range.', ha='center', fontsize=9)
    bar_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = '{}__{}_to_{}__raw_vs_denoised_top{}__absolute_vertical_bars'.format(bar_GROUP_TYPE, t0, t1, bar_TOP_N)
    for ext, opts in [('png', {'dpi': 260}), ('pdf', {}), ('svg', {})]:
        fig.savefig(bar_OUTPUT_DIR / (stem + '.' + ext), facecolor='white', bbox_inches='tight', **opts)
    if GENE_SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return stem

def bar_process(t0, t1, raw_summary, den_summary, allowed):
    groups = bar_common_groups(raw_summary, den_summary, t0, t1)
    if not groups:
        print('[SKIP] {}->{} no common comparisons'.format(t0, t1))
        return
    raw = bar_load_tables(bar_RAW_DIR, groups, t0, t1, allowed)
    den = bar_load_tables(bar_DENOISED_DIR, groups, t0, t1, allowed)
    common_genes = set(raw.gene) & set(den.gene)
    if not common_genes:
        raise RuntimeError('No genes in common between Raw and Denoised tables.')
    raw = raw[raw.gene.isin(common_genes)].copy()
    den = den[den.gene.isin(common_genes)].copy()
    raw_top_df, raw_rank = bar_rank_genes(raw)
    den_top_df, den_rank = bar_rank_genes(den)
    raw_top = raw_top_df['gene'].tolist()
    den_top = den_top_df['gene'].tolist()
    overlap_genes = sorted(set(raw_top) & set(den_top), key=bar_natural_key)
    den_only = sorted(set(den_top) - set(raw_top), key=bar_natural_key)
    raw_only = sorted(set(raw_top) - set(den_top), key=bar_natural_key)
    stem = bar_make_figure(raw_top_df, den_top_df, t0, t1, groups)
    raw_top_df.to_csv(bar_OUTPUT_DIR / (stem + '__raw_top20_ranked.csv'), index=False)
    den_top_df.to_csv(bar_OUTPUT_DIR / (stem + '__denoised_top20_ranked.csv'), index=False)
    raw_rank.to_csv(bar_OUTPUT_DIR / (stem + '__raw_ranking.csv'), index=False)
    den_rank.to_csv(bar_OUTPUT_DIR / (stem + '__denoised_ranking.csv'), index=False)
    pd.DataFrame({'gene': overlap_genes}).to_csv(bar_OUTPUT_DIR / (stem + '__overlap.csv'), index=False)
    pd.DataFrame({'gene': den_only}).to_csv(bar_OUTPUT_DIR / (stem + '__denoised_only.csv'), index=False)
    pd.DataFrame({'gene': raw_only}).to_csv(bar_OUTPUT_DIR / (stem + '__raw_only.csv'), index=False)
    print('[SAVED] {} | groups={} shared genes={} overlap={}/{}'.format(stem, len(groups), len(common_genes), len(overlap_genes), bar_TOP_N))
    print('  groups:', ', '.join(groups))
    print('  shared:', ', '.join(overlap_genes) if overlap_genes else 'None')
    print('  denoised-only:', ', '.join(den_only) if den_only else 'None')
    print('  raw-only:', ', '.join(raw_only) if raw_only else 'None')

def bar_main():
    raw_summary = bar_load_summary(bar_RAW_DIR)
    den_summary = bar_load_summary(bar_DENOISED_DIR)
    order_file = bar_DENOISED_DIR / 'training_gene_order.txt'
    if not order_file.exists():
        raise FileNotFoundError('Missing {}. Use output of analyze_denoised_temporal_DE.py'.format(order_file.resolve()))
    allowed = {x.strip() for x in order_file.read_text().splitlines() if x.strip()}
    if len(allowed) != 4000:
        raise ValueError('Training gene order has {} unique genes; expected 4000.'.format(len(allowed)))
    bar_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Restricting Raw and Denoised to the same 4000 training genes and selected common groups.')
    for t0, t1 in bar_TIME_PAIRS:
        bar_process(t0, t1, raw_summary, den_summary, allowed)
    print('Done:', bar_OUTPUT_DIR.resolve())

def run_temporal_gene_analysis(raw_adata_file='./data/Tumor_Immune_Evasion_eTRACER/processed_data/eTracer_lineage_usable.h5ad', denoised_file='./results_Rep3/train/denoised_counts.txt', output_dir='./results_gene_analyse', training_hvg_file=None, show=True, calculate=True):
    """Compute temporal DE, then plot matched Top20 heatmaps and bars.

    The raw .h5ad must contain unnormalized integer counts in .X.
    Denoised matrix must be cell-by-training-gene aligned to the raw dataset.
    If calculate=False, only regenerate comparison figures from saved CSVs.
    """
    global GENE_ANALYSIS_ROOT, GENE_SHOW_FIGURES
    global raw_DATA_FILE, raw_OUT, raw_TIME_PAIRS
    global den_DATA_FILE, den_DENOISED_FILE, den_OUT, den_TRAIN_HVG_FILE
    global heat_RESULTS_ROOT, heat_RAW_DIR, heat_DENOISED_DIR, heat_OUTPUT_DIR
    global bar_RESULTS_ROOT, bar_RAW_DIR, bar_DENOISED_DIR, bar_OUTPUT_DIR
    GENE_ANALYSIS_ROOT = Path(output_dir)
    GENE_SHOW_FIGURES = bool(show)
    raw_DATA_FILE = Path(raw_adata_file)
    den_DATA_FILE = Path(raw_adata_file)
    den_DENOISED_FILE = Path(denoised_file)
    den_TRAIN_HVG_FILE = Path(training_hvg_file) if training_hvg_file is not None else None
    raw_OUT = GENE_ANALYSIS_ROOT / 'raw_temporal_DE'
    den_OUT = GENE_ANALYSIS_ROOT / 'denoised_temporal_DE'
    heat_RESULTS_ROOT = GENE_ANALYSIS_ROOT
    heat_RAW_DIR = raw_OUT
    heat_DENOISED_DIR = den_OUT
    heat_OUTPUT_DIR = GENE_ANALYSIS_ROOT / 'raw_vs_denoised_top20_median_abs_heatmap'
    bar_RESULTS_ROOT = GENE_ANALYSIS_ROOT
    bar_RAW_DIR = raw_OUT
    bar_DENOISED_DIR = den_OUT
    bar_OUTPUT_DIR = GENE_ANALYSIS_ROOT / 'raw_vs_denoised_top20_bars'
    raw_TIME_PAIRS = [(0, 4), (4, 6)]
    GENE_ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    if calculate:
        print('Computing raw temporal differential expression')
        raw_main()
        print('Computing decoder-based temporal differential expression')
        den_main()
    print('Drawing matched Top20 heatmaps')
    heat_main()
    print('Drawing matched Top20 bar charts')
    bar_main()
    print('Gene analysis completed:', GENE_ANALYSIS_ROOT.resolve())
    return {'raw': raw_OUT, 'denoised': den_OUT, 'heatmaps': heat_OUTPUT_DIR, 'bars': bar_OUTPUT_DIR}
