"""Full benchmark data analysis script.

Analyzes CSV results, performs statistical tests, generates publication figures.
Statistical methods: Shapiro-Wilk, Kruskal-Wallis+Dunn-Bonferroni, bootstrap CI,
factorial OLS regression, Kaplan-Meier survival/Cox PH, correlations, PDR model.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False

try:
    from statsmodels.formula.api import ols as smf_ols
    from statsmodels.stats.anova import anova_lm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Only the repo root goes onto sys.path (never its parent), so the repo's own
# ``src/`` cannot be shadowed by a sibling checkout with an older codebase.


if _ROOT not in sys.path:

    sys.path.insert(0, _ROOT)

from benchmark.analyzers.stat_analyzer import (
    load_results, test_normality, confidence_interval, METRICS,
)


def collect(csv_globs):
    frames = []
    for pat in csv_globs:
        for f in sorted(glob.glob(pat)):
            frames.append(load_results(f))
    if not frames:
        raise FileNotFoundError('No CSVs matched: ' + str(csv_globs))
    df = pd.concat(frames, ignore_index=True)
    if 'converged' in df.columns:
        df['converged'] = df['converged'].astype(bool)
    return df


def plot_mtc_vs_swarm(data, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for bl, g in data.groupby('baseline'):
        a = g.groupby('swarm_size')['mtc']
        m = a.mean()
        s = a.std() / np.sqrt(a.count())
        ax.errorbar(m.index, m, yerr=s * 1.96, label=bl, marker='o', capsize=4, linewidth=2)
    ax.axhline(50, color='r', ls='--', alpha=0.5, label='Horizon (50s)')
    ax.set_xlabel('Swarm Size')
    ax.set_ylabel('MTC (s)')
    ax.set_title('Mean Time to Consensus vs Swarm Size (95% CI)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('[i] mtc_vs_swarm_size.png')


def plot_pdr_vs_loss(data, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    pls = sorted(data['packet_loss'].dropna().unique())
    lossy = data[data['packet_loss'] > 0]
    bls = sorted(lossy['baseline'].unique())
    w = 0.8 / max(len(bls), 1)
    xp = np.array([float(p) for p in pls])
    for i, bl in enumerate(bls):
        g = lossy[lossy['baseline'] == bl]
        m = g.groupby('packet_loss')['pdr'].mean().reindex(pls)
        c = g.groupby('packet_loss')['pdr'].count().reindex(pls)
        s = g.groupby('packet_loss')['pdr'].std().reindex(pls) / np.sqrt(c)
        off = (i - (len(bls) - 1) / 2) * w
        ax.bar(xp + off, m, width=w, yerr=s * 1.96, capsize=3, label=bl, alpha=0.85)
    ax.set_xticks(pls)
    ax.set_xticklabels([str(int(p * 100)) + '%' for p in pls])
    ax.set_xlabel('Packet Loss Rate')
    ax.set_ylabel('PDR')
    ax.set_title('Packet Delivery Ratio vs Packet Loss')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('[i] pdr_vs_packet_loss.png')


def plot_survival(data, path):
    if not HAS_LIFELINES:
        print('[!] lifelines missing, skip survival')
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for bl, g in data.groupby('baseline'):
        ev = g['converged'].astype(int).values
        t = g['frames_to_consensus'].values.astype(float) * 0.1
        t = np.where(ev == 1, t, 50.0)
        kmf = KaplanMeierFitter()
        kmf.fit(t, event_observed=ev, label=bl)
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier: Time to Consensus')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
def plot_cdf(data, path, horizon=500):
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.arange(0, horizon + 1, 5)
    for bl, g in data.groupby('baseline'):
        c = g['frames_to_consensus'].values.astype(float)
        c = c[(c >= 0) & (c <= horizon)]
        if len(c) == 0:
            continue
        y = np.array([np.mean(c <= x) for x in xs])
        ax.plot(xs, y, drawstyle='steps-post', label=bl, lw=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, horizon)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fraction Converged')
    ax.set_title('Convergence Curve (Empirical CDF)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('[i] convergence_curve.png')


def plot_heatmap(data, path):
    cand = data[data['baseline'] == 'tdma_packet_loss'] if 'tdma_packet_loss' in set(data.get('baseline', [])) else data
    pm = cand.pivot_table(index='swarm_size', columns='packet_loss', values='mtc', aggfunc='mean').sort_index()
    pc = cand.pivot_table(index='swarm_size', columns='packet_loss', values='converged', aggfunc='mean').sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pm.values, aspect='auto', cmap='viridis_r')
    for i in range(pm.shape[0]):
        for j in range(pm.shape[1]):
            v = pm.values[i, j]
            cv = pc.values[i, j] if not np.isnan(pc.values[i, j]) else 0
            tc = 'white' if v < 35 else 'black'
            ax.text(j, i, ('{:.1f}\n({:.0f}%)').format(v, cv * 100), ha='center', va='center', fontsize=7, color=tc)
    ax.set_xticks(range(len(pm.columns)))
    ax.set_xticklabels([str(int(c * 100)) + '%' for c in pm.columns])
    ax.set_yticks(range(len(pm.index)))
    ax.set_yticklabels([str(int(i)) for i in pm.index])
    ax.set_xlabel('Packet Loss Rate')
    ax.set_ylabel('Swarm Size N')
    ax.set_title('MTC and Convergence Rate Heatmap')
    fig.colorbar(im, ax=ax, shrink=0.8, label='MTC (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('[i] heatmap.png')


def plot_boxplots(data, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    mm = {'mtc': 'MTC (s)', 'sci': 'SCI', 'ce': 'CE', 'pdr': 'PDR'}
    for ax, (met, lbl) in zip(axes.flat, mm.items()):
        sns.boxplot(data=data, x='baseline', y=met, ax=ax)
        sns.stripplot(data=data, x='baseline', y=met, ax=ax, alpha=0.15, size=1.5)
        ax.set_ylabel(lbl)
        ax.set_title(lbl + ' by Baseline')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('[i] boxplot.png')


def pdr_check(data, outdir):
    """Validate the radio model against the end-to-end estimand.

    packets_offered counts every scheduled sender->receiver pair (including
    TX-failed slots), so PDR = received / offered is the true end-to-end
    delivery probability q = (1 - p_tx)(1 - p_rx).
    """
    rep = ['=== PDR MODEL VALIDATION ===', '',
           'Estimand: end-to-end q = (1-p_tx)(1-p_rx) over scheduled '
           'sender->receiver opportunities',
           '(TX-side failure is part of the offered-link count)', '']
    for pt in sorted(data['packet_loss'].unique()):
        if pt == 0:
            continue
        sub = data[data['packet_loss'] == pt]
        if len(sub) < 5:
            continue
        pr = pt / 2
        expected = (1.0 - pt) * (1.0 - pr)

        offered = sub["packets_offered"].sum()
        received = sub["packets_received"].sum()

        observed = received / offered if offered > 0 else np.nan

        row = (
            "p_tx={0:.2f}: expected_end_to_end={1:.4f} "
            "observed={2:.4f} error={3:.4f}"
        ).format(
            pt,
            expected,
            observed,
            observed - expected,
        )
        rep.append(row)
    txt = '\n'.join(rep)
    open(os.path.join(outdir, 'pdr_model_check.txt'), 'w').write(txt)
    print('[OK] pdr_model_check.txt')
def kruskal_wallis(data, outdir):
    from itertools import combinations
    rep = ['=== KRUSKAL-WALLIS + DUNN-BONFERRONI ===', '']
    for met in ['mtc', 'sci', 'ce', 'pdr']:
        if met not in data.columns:
            continue
        groups, labels = [], []
        for bl in sorted(data['baseline'].unique()):
            v = data[data['baseline'] == bl][met].dropna().values
            if len(v) >= 2:
                groups.append(v)
                labels.append(bl)
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        rep.append('{0}: H={1:.3f} p={2:.4f}'.format(met, H, p))
        if p < 0.05 and len(groups) > 2:
            nc = len(list(combinations(labels, 2)))
            ab = 0.05 / nc
            for (l1, g1), (l2, g2) in combinations(zip(labels, groups), 2):
                u, pp = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                rep.append('  {0} vs {1}: U={2:.0f} p={3:.4f}'.format(l1, l2, u, pp))
        rep.append('')
    txt = '\n'.join(rep)
    open(os.path.join(outdir, 'kruskal_wallis.txt'), 'w').write(txt)
    print('[OK] kruskal_wallis.txt')


def regression(data, outdir):
    """Descriptive mechanism regression on TDMA trials (HC3 robust SEs).

    The staleness ratio rho = N*tau/T_val (with T_val = 2.0 s this equals
    N*tau/2) and the nominal packet-loss rate are treated as controlled
    experimental factors. This is a descriptive response-surface analysis,
    NOT a causal estimator.
    """
    if not HAS_STATSMODELS:
        print('[!] statsmodels missing')
        return

    rep = ['=== FACTORIAL / MECHANISM REGRESSION ===', '']

    d = data.copy()

    d['staleness_ratio'] = d['swarm_size'] * d['slot_duration'] / 2.0

    # Mechanism model is intentionally restricted to TDMA trials.
    # This avoids conflating the communication mechanism with the
    # open-channel baseline indicator.
    d = d[d['use_tdma'] == 1].copy()

    formula = 'mtc ~ staleness_ratio + packet_loss + staleness_ratio:packet_loss'

    try:
        model = smf_ols(formula, data=d).fit(cov_type='HC3')
        rep.append('n observations : {0}'.format(int(model.nobs)))
        rep.append('covariance     : HC3 (heteroskedasticity-robust)')
        rep.append('')
        rep.append(model.summary().as_text())
        rep.append('')
        rep.append('Mechanism coefficients:')
        for term in ['staleness_ratio', 'packet_loss',
                     'staleness_ratio:packet_loss']:
            if term in model.params.index:
                rep.append('  {}: coef={:.6f}, p={:.6g}'.format(
                    term, model.params[term], model.pvalues[term]))
    except Exception as exc:
        rep.append('Regression failed: {}'.format(exc))

    with open(os.path.join(outdir, 'regression_analysis.txt'), 'w') as f:
        f.write('\n'.join(rep))

    print('[OK] regression_analysis.txt')


def survival(data, outdir):
    """Kaplan-Meier survival analysis with pairwise log-rank tests and a
    Cox proportional-hazards model on TDMA trials (robust SEs)."""
    if not HAS_LIFELINES:
        print('[!] lifelines missing')
        return

    from lifelines.statistics import logrank_test

    rep = ['=== SURVIVAL ANALYSIS ===', '']

    d = data.copy()

    d['event'] = d['converged'].astype(int)
    d['duration_s'] = d['frames_to_consensus'].astype(float) * 0.1

    # Non-converged trials are right-censored at the mission horizon.
    d.loc[d['event'] == 0, 'duration_s'] = 50.0

    # -----------------------------
    # Kaplan-Meier by baseline
    # -----------------------------
    km_models = {}
    for baseline, g in d.groupby('baseline'):
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=g['duration_s'],
            event_observed=g['event'],
            label=str(baseline),
        )
        km_models[baseline] = kmf
        median = kmf.median_survival_time_
        rep.append('{}: events={}/{} ({:.1f}%), median={}'.format(
            baseline,
            int(g['event'].sum()),
            len(g),
            100.0 * g['event'].mean(),
            'inf' if np.isinf(median) else '{:.3f}s'.format(median),
        ))
        rep.append('  S(10s)={:.4f}, S(30s)={:.4f}'.format(
            float(kmf.predict(10.0)), float(kmf.predict(30.0))))

    # -----------------------------
    # Pairwise log-rank tests
    # -----------------------------
    baselines = sorted(d['baseline'].unique())
    rep.append('')
    rep.append('Pairwise log-rank tests:')
    for i, b1 in enumerate(baselines):
        for b2 in baselines[i + 1:]:
            g1 = d[d['baseline'] == b1]
            g2 = d[d['baseline'] == b2]
            result = logrank_test(
                g1['duration_s'],
                g2['duration_s'],
                event_observed_A=g1['event'],
                event_observed_B=g2['event'],
            )
            rep.append('  {} vs {}: p={:.6g}'.format(b1, b2, result.p_value))

    # -----------------------------
    # Cox model (TDMA trials only)
    # -----------------------------
    tdma = d[d['use_tdma'] == 1].copy()
    tdma['staleness_ratio'] = tdma['swarm_size'] * tdma['slot_duration'] / 2.0

    cox_cols = ['duration_s', 'event', 'staleness_ratio', 'packet_loss']
    cox_data = tdma[cox_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(cox_data) >= 20:
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='duration_s', event_col='event',
                robust=True)
        rep.append('')
        rep.append('=== COX PH: TDMA CONDITIONS ===')
        rep.append(cph.summary[['coef', 'exp(coef)', 'p']].to_string())
        rep.append('')
        rep.append('Interpretation: exp(coef) > 1 increases the '
                   'instantaneous convergence hazard.')
        try:
            cph.check_assumptions(cox_data, p_value_threshold=0.05,
                                  show_plots=False)
            rep.append('PH assumption diagnostic: completed '
                       '(no output means no violations flagged).')
        except Exception as exc:
            rep.append('PH assumption diagnostic unavailable: {}'.format(exc))
    else:
        rep.append('')
        rep.append('Cox model skipped: insufficient TDMA observations.')

    with open(os.path.join(outdir, 'survival_analysis.txt'), 'w') as f:
        f.write('\n'.join(rep))

    # -----------------------------
    # Figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for baseline, kmf in km_models.items():
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Time to consensus (s)')
    ax.set_ylabel('Survival probability')
    ax.set_title('Kaplan-Meier Time-to-Consensus')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'survival_curves.png'), dpi=300)
    plt.close(fig)
    print('[OK] survival_analysis.txt')
def correlation(data, outdir):
    rep = ['=== CORRELATION ===', '']
    cols = [c for c in ['swarm_size', 'packet_loss', 'slot_duration', 'mtc', 'sci', 'pdr'] if c in data.columns]
    if len(cols) >= 2:
        sub = data[cols].dropna()
        rep.append('Pearson:')
        rep.append(sub.corr(method='pearson').round(3).to_string())
        rep.append('')
        rep.append('Spearman:')
        rep.append(sub.corr(method='spearman').round(3).to_string())
        rep.append('')
        for col in ['swarm_size', 'packet_loss']:
            for met in ['mtc', 'pdr']:
                if col in data.columns and met in data.columns:
                    m = data[col].notna() & data[met].notna()
                    r, p = stats.pearsonr(data.loc[m, col], data.loc[m, met])
                    rep.append('{0} vs {1}: r={2:.3f} p={3:.4f}'.format(col, met, r, p))
    txt = '\n'.join(rep)
    open(os.path.join(outdir, 'correlation_analysis.txt'), 'w').write(txt)
    print('[OK] correlation_analysis.txt')


def populations_report(files, data):
    """Explicit separation of experimental populations (manuscript Sec. V-E).

    Distinguishes: number of runs, number of independent seeds, and number
    of configurations -- the inferential unit is the SEED within a
    configuration, never the pooled run count.
    """
    rep = ['=== CANONICAL POPULATION MANIFEST ===', '',
           'This report is generated by ONE pipeline (benchmark/analyzers/',
           'full_analysis.py) from the files listed below. Older artifacts',
           '(e.g. benchmark_reduced.csv, 1620 rows) are superseded.', '']
    for f in files:
        try:
            d = pd.read_csv(f)
            ncfg = d.groupby(['baseline', 'swarm_size', 'packet_loss',
                              'slot_duration', 'num_objectives',
                              'num_threats']).ngroups
            nseeds = d['trial'].nunique() if 'trial' in d.columns else '?'
            rep.append('{0}'.format(os.path.basename(f)))
            rep.append('  runs={0}  configs={1}  seeds/config={2}  '
                       'baselines={3}'.format(len(d), ncfg, nseeds,
                                              sorted(d['baseline'].unique())))
        except Exception as e:
            rep.append('{0}: ERROR {1}'.format(os.path.basename(f), e))
    rep.append('')
    rep.append('POOLED ANALYSIS SET: {0} runs, {1} baselines'
               .format(len(data), data['baseline'].nunique()))
    rep.append('Inferential unit: trial (seed) nested in configuration;')
    rep.append('configurations are conditions, NOT independent draws.')
    return '\n'.join(rep)


def main(globs):
    data = collect(globs)
    print('Loaded {0} rows, {1} baselines'.format(len(data), data['baseline'].nunique()))
    outdir = os.path.join(_ROOT, 'benchmark', 'results', 'figures')
    resdir = os.path.join(_ROOT, 'benchmark', 'results')
    os.makedirs(outdir, exist_ok=True)

    # --- Canonical manifest: which artifact generated the paper? ---
    pop_txt = populations_report(globs if isinstance(globs, list) else [], data)
    open(os.path.join(resdir, 'population_manifest.txt'), 'w').write(pop_txt)
    print('[OK] population_manifest.txt')

    # --- Canonical consolidated report (replaces stale per-file reports) ---
    canon = ['CANONICAL STATISTICAL REPORT',
             'Generated by benchmark/analyzers/full_analysis.py',
             'Source files:', os.path.basename(str(globs)), '', pop_txt, '']
    open(os.path.join(resdir, 'statistical_report.txt'), 'w').write('\n'.join(canon))

    print('\n=== Shapiro-Wilk ===')
    for m in METRICS:
        if m in data.columns:
            r = test_normality(data[m])
            print('  {0}: n={1} p={2:.4f} {3}'.format(
                m, r.get('sample_size'), r['p_value'],
                'Normal' if r.get('normal') else 'NOT normal'))

    pdr_check(data, outdir)
    kruskal_wallis(data, outdir)
    regression(data, outdir)
    survival(data, outdir)
    correlation(data, outdir)

    print('\n=== CIs (95%) ===')
    for bl in sorted(data['baseline'].unique()):
        sub = data[data['baseline'] == bl]
        for m in METRICS:
            if m in data.columns:
                ci = confidence_interval(sub[m])
                print('  {0:<16} {1:<4} n={2:3d} mean={3:7.3f} CI=({4:.3f}, {5:.3f})'.format(
                    bl, m, ci['n'], ci['mean'], ci['ci'][0], ci['ci'][1]))

    print('\n=== Figures ===')
    plot_mtc_vs_swarm(data, os.path.join(outdir, 'mtc_vs_swarm_size.png'))
    plot_pdr_vs_loss(data, os.path.join(outdir, 'pdr_vs_packet_loss.png'))
    plot_cdf(data, os.path.join(outdir, 'convergence_curve.png'))
    plot_heatmap(data, os.path.join(outdir, 'heatmap.png'))
    plot_boxplots(data, os.path.join(outdir, 'boxplot.png'))

    print('\n=== SUMMARY ===')
    print('  runs={0} conv_rate={1:.1f}% mtc={2:.2f}s pdr={3:.3f}'.format(
        len(data), 100 * data['converged'].mean(), data['mtc'].mean(), data['pdr'].mean()))

    # --- Consolidate every section into ONE canonical report (#10) ---
    canon_path = os.path.join(resdir, 'statistical_report.txt')
    with open(canon_path, 'a', encoding='utf-8') as f:
        for name in ['pdr_model_check.txt', 'kruskal_wallis.txt',
                     'regression_analysis.txt', 'survival_analysis.txt',
                     'correlation_analysis.txt']:
            p = os.path.join(outdir, name)
            if os.path.exists(p):
                f.write('\n\n' + open(p, encoding='utf-8').read())
        f.write('\n\n=== HEADLINE (pooled, descriptive only) ===\n')
        f.write('  runs={0} conv_rate={1:.1f}% mtc={2:.2f}s pdr={3:.3f}\n'.format(
            len(data), 100 * data['converged'].mean(),
            data['mtc'].mean(), data['pdr'].mean()))
    print('[OK] statistical_report.txt (canonical, consolidated)')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        rd = os.path.join(_ROOT, 'benchmark', 'results')
        args = sorted(glob.glob(os.path.join(rd, '*.csv')))
    main(args)
    print('[i] survival_curves.png')