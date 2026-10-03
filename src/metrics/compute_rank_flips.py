"""Rank Flip Rate: fraction of (model-pair, simulator-pair) comparisons whose order reverses.

`margin` makes the statistic practically meaningful: a reversal only counts when
BOTH score gaps exceed `margin` in absolute value (e.g. the within-policy seed noise
floor). margin=0 reproduces the nominal definition.
"""
from itertools import combinations

import pandas as pd


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    dynamic = df[df["simulator"] != "full_info_static"]
    if dynamic.empty:
        return pd.DataFrame()
    return dynamic.groupby(["simulator", "agent_model"])["normalized"].mean().unstack("simulator")


def rank_flip_details(df: pd.DataFrame, margin: float = 0.0) -> pd.DataFrame:
    pivot = _pivot(df)
    if pivot.empty:
        return pd.DataFrame()
    rows = []
    for s, sp in combinations(pivot.columns.tolist(), 2):
        for i, j in combinations(pivot.index.tolist(), 2):
            d1 = float(pivot.loc[i, s] - pivot.loc[j, s])
            d2 = float(pivot.loc[i, sp] - pivot.loc[j, sp])
            nominal = d1 * d2 < 0
            rows.append({
                "sim_a": s, "sim_b": sp, "model_i": i, "model_j": j,
                "gap_a": round(d1, 4), "gap_b": round(d2, 4),
                "nominal_flip": int(nominal),
                "flip": int(nominal and abs(d1) > margin and abs(d2) > margin),
                "margin": margin,
            })
    return pd.DataFrame(rows)


def rank_flip_rate(df: pd.DataFrame, margin: float = 0.0) -> float:
    det = rank_flip_details(df, margin)
    if det.empty:
        return 0.0
    return float(det["flip"].mean())
