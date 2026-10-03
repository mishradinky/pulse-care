"""Ranking Instability: average normalized Kendall-tau distance across simulator pairs."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def normalized_kendall_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    if len(r1) < 2:
        return 0.0
    tau, _ = kendalltau(r1, r2)
    if np.isnan(tau):
        return 0.0
    return float((1.0 - tau) / 2.0)


def ranking_instability(df: pd.DataFrame) -> float:
    dynamic = df[df["simulator"] != "full_info_static"]
    if dynamic.empty:
        return 0.0
    pivot = (
        dynamic.groupby(["simulator", "agent_model"])["normalized"]
        .mean()
        .unstack("simulator")
    )
    sims = pivot.columns.tolist()
    if len(sims) < 2:
        return 0.0
    dists = []
    for a, b in combinations(sims, 2):
        ra = pivot[a].rank(ascending=False).values
        rb = pivot[b].rank(ascending=False).values
        dists.append(normalized_kendall_distance(ra, rb))
    return float(np.mean(dists)) if dists else 0.0
