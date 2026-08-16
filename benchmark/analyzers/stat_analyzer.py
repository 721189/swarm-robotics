"""Statistical analysis for benchmark results.

Wraps scipy/pandas so the analyses stay one-liners inside the stat runner.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJECT_ROOT, os.path.dirname(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["mtc", "sci", "ce", "pdr"]


def load_results(filename: str) -> pd.DataFrame:
    """Load a benchmark CSV (with dtypes normalised)."""
    df = pd.read_csv(filename)
    # Normalise column names just in case.
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "use_tdma" in df.columns:
        df["use_tdma"] = df["use_tdma"].astype(int)
    return df


def _clean(series: pd.Series) -> pd.Series:
    """Drop sentinel error rows and NaN."""
    s = pd.to_numeric(series, errors="coerce")
    s = s[(s >= 0) | s.isna()]  # drop -1 sentinels
    return s.dropna()


def test_normality(data: pd.Series) -> dict:
    """Shapiro-Wilk normality test; dict with statistic/p-value/verdict."""
    clean = _clean(data)
    if len(clean) < 3 or len(clean) > 5000:
        return {"statistic": np.nan, "p_value": np.nan,
                "sample_size": int(len(clean)),
                "normal": False, "reason": "sample size outside Shapiro range"}
    stat, p_value = stats.shapiro(clean)
    return {"statistic": stat, "p_value": p_value,
            "sample_size": int(len(clean)), "normal": bool(p_value > 0.05)}


def run_anova(data: pd.DataFrame, metric: str, group: str) -> dict:
    """One-way ANOVA of ``metric`` across ``group`` categories."""
    clean = data.copy()
    if "use_tdma" in clean.columns and group == "use_tdma":
        clean = clean[clean["use_tdma"].isin([0, 1])]
        clean = clean.rename(columns={"use_tdma": "__use_tdma__"})
        group = "__use_tdma__"
    groups = [g[metric].dropna() for _, g in clean.groupby(group, dropna=True)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"f_stat": np.nan, "p_value": np.nan,
                "significant": False, "k": 0, "n": 0,
                "reason": "fewer than 2 non-degenerate groups"}
    f_stat, p_value = stats.f_oneway(*groups)
    n_total = int(sum(len(g) for g in groups))
    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "k": len(groups),
        "n": n_total,
    }


def confidence_interval(data: pd.Series, confidence: float = 0.95) -> dict:
    """Sample mean, SEM, and t-distribution confidence interval."""
    clean = _clean(data)
    n = len(clean)
    if n == 0:
        return {"mean": np.nan, "sem": np.nan, "ci": (np.nan, np.nan), "n": 0}
    mean = float(np.mean(clean))
    sem = float(stats.sem(clean)) if n > 1 else 0.0
    if n > 1:
        lo, hi = stats.t.interval(confidence, n - 1, mean, sem)
    else:
        lo, hi = mean, mean
    return {"mean": mean, "sem": sem, "ci": (float(lo), float(hi)), "n": n}


def aggregate_by(data: pd.DataFrame, group: str, metric: str
                 ) -> pd.DataFrame:
    """Compact mean/std/n/CI table for a metric grouped by a column."""
    out = []
    for key, g in data.groupby(group, dropna=False):
        ci = confidence_interval(g[metric])
        out.append({
            group: key,
            "n": ci["n"],
            "mean": ci["mean"],
            "std": float(np.std(g[metric].dropna())) if len(g) > 1 else 0.0,
            "sem": ci["sem"],
            "ci_lo": ci["ci"][0],
            "ci_hi": ci["ci"][1],
        })
    return pd.DataFrame(out).sort_values(by="mean")