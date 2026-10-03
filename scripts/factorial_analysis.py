"""Factorial simulator analysis: which simulator property drives the effect?

Fits, on the factorial-cell rows (simulator names fx_v?_d?_l?_x?):

  normalized  ~ verbosity + withholding + low_literacy + distractors
                + case FE + model FE (+ arch FE)
  under_esc   ~ same (linear probability)
  disclosure_failure ~ same

with case-cluster-robust standard errors. With the 8-cell resolution-IV fraction the
four main effects are unconfounded with two-way interactions; with the full 16 cells the
two-way interactions are added.

Outputs: outputs/tables/T_factorial_effects.csv, T_factorial_cells.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import subsets as S
from src.metrics.compute_escalation_metrics import ORD

FACTORS = ["verbosity", "withholding", "low_literacy", "distractors"]


def _factors_from_name(name: str) -> dict | None:
    # fx_v0_d1_l0_x1
    try:
        parts = name.split("_")
        return {"verbosity": int(parts[1][1]), "withholding": int(parts[2][1]),
                "low_literacy": int(parts[3][1]), "distractors": int(parts[4][1])}
    except Exception:  # noqa: BLE001
        return None


def ols_cluster(y: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    G = len(np.unique(clusters))
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(clusters):
        idx = clusters == g
        u = X[idx].T @ e[idx]
        meat += np.outer(u, u)
    n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def fit(d: pd.DataFrame, y_col: str, interactions: bool) -> list[dict]:
    y = d[y_col].to_numpy(float)
    cols = list(FACTORS)
    X = d[FACTORS].to_numpy(float)
    if interactions:
        for a, b in combinations(FACTORS, 2):
            X = np.column_stack([X, d[a].to_numpy(float) * d[b].to_numpy(float)])
            cols.append(f"{a}:{b}")
    # fixed effects
    for fe in ["case_id", "agent_model", "agent_arch"]:
        dummies = pd.get_dummies(d[fe], prefix=fe, drop_first=True).to_numpy(float)
        if dummies.shape[1]:
            X = np.column_stack([X, dummies])
            cols += [f"{fe}_fe"] * dummies.shape[1]
    X = np.column_stack([np.ones(len(d)), X]); cols = ["intercept"] + cols
    beta, se = ols_cluster(y, X, d["case_id"].to_numpy())
    G = d["case_id"].nunique()
    rows = []
    for name, b, s in zip(cols, beta, se):
        if name.endswith("_fe") or name == "intercept":
            continue
        t = b / s if s > 0 else np.nan
        p = 2 * (1 - stats.t.cdf(abs(t), df=max(G - 1, 1))) if np.isfinite(t) else np.nan
        rows.append({"outcome": y_col, "term": name, "estimate": b, "se_cluster_case": s,
                     "t": t, "p": p, "ci_lo": b - 1.96 * s, "ci_hi": b + 1.96 * s, "n": len(d),
                     "n_cases": G})
    return rows


def run(scored_csv: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(scored_csv)
    d = S.factorial(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    if d.empty:
        print("[factorial] no factorial rows yet. Run: python -m src.batch_driver --subset factorial")
        pd.DataFrame(columns=["outcome", "term", "estimate"]).to_csv(out_dir / "T_factorial_effects.csv", index=False)
        return pd.DataFrame()
    fac = d["simulator"].apply(_factors_from_name)
    d = d[fac.notna()].copy()
    for f in FACTORS:
        d[f] = fac[fac.notna()].apply(lambda z: z[f]).to_numpy()
    d["under_esc"] = (d["escalation_level"].map(ORD) < d["correct_escalation"].map(ORD)).astype(int)
    d["emergency_miss"] = ((d["correct_escalation"] == "emergency_911")
                           & (d["escalation_level"] != "emergency_911")).astype(int)
    n_cells = d["simulator"].nunique()
    interactions = n_cells >= 16

    cells = d.groupby(["simulator"] + FACTORS).agg(
        n=("normalized", "size"), mean_score=("normalized", "mean"), sd_score=("normalized", "std"),
        under_esc=("under_esc", "mean"), disclosure_failure=("disclosure_failure", "mean"),
        severe_leakage=("leakage_level", lambda s: (s == 2).mean())).reset_index()
    cells.round(4).to_csv(out_dir / "T_factorial_cells.csv", index=False)

    rows = []
    for y in ["normalized", "under_esc", "disclosure_failure"]:
        rows += fit(d, y, interactions)
    eff = pd.DataFrame(rows)
    eff.round(4).to_csv(out_dir / "T_factorial_effects.csv", index=False)
    (out_dir / "factorial_summary.json").write_text(json.dumps({
        "n_rows": int(len(d)), "n_cells": int(n_cells), "design": "full 2^4" if interactions else "2^(4-1) fraction",
        "models": sorted(d["agent_model"].unique().tolist()), "n_cases": int(d["case_id"].nunique())}, indent=2))
    print(f"[factorial] {len(d)} rows, {n_cells} cells, design={'full' if interactions else 'fraction'}")
    print(eff.round(4).to_string(index=False))
    return eff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--out", default="outputs/tables")
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.out))
