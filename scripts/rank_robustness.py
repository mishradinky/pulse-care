"""Margin-aware and inferential ranking-flip analysis.

Nominal rank flips can come from tiny mean-score differences. This script reports:

  1. T_rank_flips_margin.csv   flip rate when a reversal only counts if BOTH gaps
                               exceed a margin (0, 0.02, 0.05, 0.10 and the seed
                               noise floor from noise_floor.json when available)
  2. T_rank_flip_detail.csv    every (model pair, simulator pair) with both gaps
  3. T_rank_pairwise_tests.csv per simulator and model pair: paired difference across
                               (case, arch) units, t-based 95% CI, Wilcoxon p
  4. T_rank_flip_bootstrap.csv case-level bootstrap probability that each pairwise
                               order reverses between two simulators, and P(i beats j)
  5. rank_robustness.json      summary incl. 'meaningful_flip_rate' = reversals where
                               both gaps exceed the noise floor AND both paired tests
                               are significant at 0.05
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
from src.metrics.bootstrap_ci import resample_cases
from src.metrics.compute_rank_flips import rank_flip_details

MARGINS = [0.0, 0.02, 0.05, 0.10]


def pairwise_tests(prim: pd.DataFrame) -> pd.DataFrame:
    unit = ["case_id", "agent_arch"]
    piv = prim.pivot_table(index=unit + ["simulator"], columns="agent_model",
                           values="normalized", aggfunc="mean").reset_index()
    models = [c for c in piv.columns if c not in unit + ["simulator"]]
    rows = []
    for sim, g in piv.groupby("simulator"):
        for i, j in combinations(models, 2):
            d = (g[i] - g[j]).dropna().to_numpy()
            if len(d) < 3:
                continue
            mean = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
            tcrit = stats.t.ppf(0.975, len(d) - 1)
            try:
                wp = float(stats.wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                wp = 1.0
            tp = float(stats.ttest_1samp(d, 0.0).pvalue)
            rows.append({"simulator": sim, "model_i": i, "model_j": j, "n_units": len(d),
                         "mean_diff": mean, "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se,
                         "t_p": tp, "wilcoxon_p": wp, "significant_05": int(wp < 0.05),
                         "winner": i if mean > 0 else j})
    return pd.DataFrame(rows)


def flip_bootstrap(prim: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    det0 = rank_flip_details(prim)
    keys = ["sim_a", "sim_b", "model_i", "model_j"]
    counts = {tuple(r[k] for k in keys): {"flip": 0, "i_beats_j_a": 0, "i_beats_j_b": 0}
              for _, r in det0.iterrows()}
    for _ in range(n):
        sub = resample_cases(prim, rng)
        det = rank_flip_details(sub)
        for _, r in det.iterrows():
            c = counts.get(tuple(r[k] for k in keys))
            if c is None:
                continue
            c["flip"] += int(r["nominal_flip"])
            c["i_beats_j_a"] += int(r["gap_a"] > 0)
            c["i_beats_j_b"] += int(r["gap_b"] > 0)
    rows = []
    for k, c in counts.items():
        rows.append({**dict(zip(keys, k)), "p_flip": c["flip"] / n,
                     "p_i_beats_j_sim_a": c["i_beats_j_a"] / n,
                     "p_i_beats_j_sim_b": c["i_beats_j_b"] / n, "n_boot": n})
    return pd.DataFrame(rows)


def run(scored_csv: Path, out_dir: Path, n_boot: int = 1000, seed: int = 20260515) -> dict:
    df = pd.read_csv(scored_csv)
    prim = S.primary(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    nf_path = out_dir / "noise_floor.json"
    noise = json.loads(nf_path.read_text())["within_policy_range_mean"] if nf_path.exists() else None
    margins = MARGINS + ([noise] if noise is not None else [])

    det_all = pd.concat([rank_flip_details(prim, m) for m in margins], ignore_index=True)
    det_all.to_csv(out_dir / "T_rank_flip_detail.csv", index=False)
    summ = (det_all.groupby("margin").agg(flips=("flip", "sum"), total=("flip", "size"))
            .reset_index())
    summ["flip_rate"] = (summ["flips"] / summ["total"]).round(4)
    summ["is_noise_floor"] = (summ["margin"] == noise).astype(int) if noise is not None else 0
    summ.to_csv(out_dir / "T_rank_flips_margin.csv", index=False)

    tests = pairwise_tests(prim)
    tests.round(4).to_csv(out_dir / "T_rank_pairwise_tests.csv", index=False)

    boot = flip_bootstrap(prim, n_boot, seed)
    boot.round(4).to_csv(out_dir / "T_rank_flip_bootstrap.csv", index=False)

    # meaningful flips: nominal flip, both gaps > noise floor, both pairwise tests significant
    base = rank_flip_details(prim, 0.0)
    sig = {(r.simulator, r.model_i, r.model_j): bool(r.significant_05) for r in tests.itertuples()}
    thr = noise if noise is not None else 0.05
    meaningful = 0
    for r in base.itertuples():
        if r.nominal_flip and abs(r.gap_a) > thr and abs(r.gap_b) > thr \
           and sig.get((r.sim_a, r.model_i, r.model_j), False) \
           and sig.get((r.sim_b, r.model_i, r.model_j), False):
            meaningful += 1
    out = {
        "nominal_flip_rate": float(base["nominal_flip"].mean()),
        "n_comparisons": int(len(base)),
        "noise_floor_used": thr,
        "noise_floor_source": "T_seed_variance.csv" if noise is not None else "default 0.05",
        "meaningful_flip_rate": meaningful / max(len(base), 1),
        "flip_rate_by_margin": {str(r.margin): float(r.flip_rate) for r in summ.itertuples()},
        "min_abs_gap_among_nominal_flips": float(
            base.loc[base["nominal_flip"] == 1, ["gap_a", "gap_b"]].abs().min().min())
        if base["nominal_flip"].any() else None,
        "mean_p_flip_bootstrap": float(boot["p_flip"].mean()) if len(boot) else None,
        "max_p_flip_bootstrap": float(boot["p_flip"].max()) if len(boot) else None,
    }
    (out_dir / "rank_robustness.json").write_text(json.dumps(out, indent=2))
    print("[rank_robustness]")
    print(summ.to_string(index=False))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.out), n_boot=a.n_boot)
