"""Case-level (cluster) bootstrap with 95% CIs.

Cases are resampled with replacement; each resampled copy of a case is relabelled
(case_id -> "case#k") so that metrics which group by case_id weight a case drawn
twice twice, instead of silently merging the copies.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def resample_cases(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    cases = df["case_id"].unique()
    groups = {c: g for c, g in df.groupby("case_id")}
    draw = rng.choice(cases, size=len(cases), replace=True)
    parts = []
    for k, c in enumerate(draw):
        g = groups[c].copy()
        g["case_id"] = f"{c}#{k}"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def bootstrap_ci(
    df: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame], float],
    n: int = 1000,
    level: float = 0.95,
    seed: int = 20260515,
) -> tuple[float, float, float]:
    """Return (bootstrap mean, lower_ci, upper_ci)."""
    rng = np.random.default_rng(seed)
    boots = [statistic_fn(resample_cases(df, rng)) for _ in range(n)]
    arr = np.asarray(boots, dtype=float)
    alpha = (1 - level) / 2
    return (float(np.nanmean(arr)), float(np.nanquantile(arr, alpha)),
            float(np.nanquantile(arr, 1 - alpha)))


def bootstrap_many(
    df: pd.DataFrame,
    fns: dict[str, Callable[[pd.DataFrame], float | dict]],
    n: int = 1000,
    level: float = 0.95,
    seed: int = 20260515,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap several statistics on the SAME resamples. A fn may return a dict of
    named floats; each key becomes its own statistic (prefixed 'name.' when name != '')."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n):
        sub = resample_cases(df, rng)
        for name, fn in fns.items():
            val = fn(sub)
            if isinstance(val, dict):
                for k, v in val.items():
                    key = f"{name}.{k}" if name else k
                    samples.setdefault(key, []).append(float(v))
            else:
                samples.setdefault(name, []).append(float(val))
    alpha = (1 - level) / 2
    out = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = (float(np.nanmean(arr)), float(np.nanquantile(arr, alpha)),
                  float(np.nanquantile(arr, 1 - alpha)))
    return out
