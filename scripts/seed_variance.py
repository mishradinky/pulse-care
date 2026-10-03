"""Multi-seed variance decomposition: between-simulator effect vs within-policy sampling noise.

Uses every condition (case, simulator, model, arch, backbone) that has >= 2 seed
replicates (the `repeatability*` subsets). For each such condition it treats seeds as
replicates and simulators as the treatment factor and reports:

  * within-policy SD / range of the normalized score (the noise floor)
  * between-simulator range on seed-averaged scores (the effect of interest)
  * additive two-way ANOVA per (case, model, arch): SS_sim, SS_seed_resid, F, p, ICC
  * escalation: within-policy instability (same condition, different seeds) vs
    between-policy instability (same seed, different simulators)
  * noise-floor-adjusted score sensitivity

Outputs (outputs/tables):
  T_seed_variance.csv           one-row summary
  T_seed_variance_by_case.csv   per (case, model, arch) ANOVA
  T_seed_conditions.csv         per condition within-seed statistics
  noise_floor.json              consumed by rank_robustness.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import subsets as S
from src.metrics.compute_escalation_metrics import ORD

# agent_model_id is part of the condition: seeds of different API model versions are never
# treated as replicates (Mistral Small 3.2 -> Mistral Small 4 change, 2026-09-27)
COND = ["case_id", "simulator", "agent_model", "agent_model_id", "agent_arch", "sim_backbone"]
POLICY = ["case_id", "agent_model", "agent_model_id", "agent_arch", "sim_backbone"]


def _anova_two_way(g: pd.DataFrame) -> dict:
    """Additive model score ~ simulator + seed within one (case, model, arch, backbone)."""
    y = g["normalized"].to_numpy(float)
    n = len(y)
    grand = y.mean()
    ss_tot = float(((y - grand) ** 2).sum())
    sim_means = g.groupby("simulator")["normalized"].transform("mean").to_numpy()
    seed_means = g.groupby("seed_offset")["normalized"].transform("mean").to_numpy()
    ss_sim = float(((sim_means - grand) ** 2).sum())
    ss_seed = float(((seed_means - grand) ** 2).sum())
    ss_res = max(ss_tot - ss_sim - ss_seed, 0.0)
    a = g["simulator"].nunique()
    b = g["seed_offset"].nunique()
    df_sim, df_seed, df_res = a - 1, b - 1, max(n - a - b + 1, 1)
    ms_sim, ms_res = ss_sim / max(df_sim, 1), ss_res / df_res
    f = ms_sim / ms_res if ms_res > 0 else np.inf
    p = float(1 - stats.f.cdf(f, df_sim, df_res)) if np.isfinite(f) else 0.0
    var_res = ms_res
    var_sim = max((ms_sim - ms_res) / b, 0.0)
    icc = var_sim / (var_sim + var_res) if (var_sim + var_res) > 0 else np.nan
    return {"n": n, "n_sims": a, "n_seeds": b, "ss_sim": ss_sim, "ss_seed": ss_seed,
            "ss_resid": ss_res, "F_sim": float(f), "p_sim": p,
            "var_between_sim": var_sim, "var_within_seed": var_res, "icc_sim": icc}


def run(scored_csv: Path, out_dir: Path) -> dict:
    df = pd.read_csv(scored_csv)
    rep = S.repeat(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rep.empty:
        print("[seed_variance] No conditions with >= 2 seeds. Run --subset repeatability first.")
        return {}
    rep["pred_ord"] = rep["escalation_level"].map(ORD)

    # per-condition within-seed statistics
    cond = rep.groupby(COND).agg(
        n_seeds=("seed_offset", "nunique"),
        mean=("normalized", "mean"),
        sd=("normalized", lambda s: s.std(ddof=1)),
        rng=("normalized", lambda s: s.max() - s.min()),
        esc_nunique=("pred_ord", "nunique"),
    ).reset_index()
    cond["esc_unstable_within"] = (cond["esc_nunique"] > 1).astype(int)
    cond.round(4).to_csv(out_dir / "T_seed_conditions.csv", index=False)

    pooled_within_sd = float(np.sqrt((cond["sd"] ** 2).mean()))
    mean_within_range = float(cond["rng"].mean())
    esc_within = float(cond["esc_unstable_within"].mean())

    # between-simulator on seed-averaged scores per policy; a policy needs replicates under
    # >= 2 simulators (fairness_null adds cooperative-only replicates, which must not count)
    pol_all = cond.groupby(POLICY).agg(rng=("mean", lambda s: s.max() - s.min()),
                                      n_sims=("simulator", "nunique"))
    pol = pol_all.loc[pol_all["n_sims"] >= 2, "rng"]
    between_range_seedavg = float(pol.mean()) if len(pol) else np.nan
    n_policies_multi_sim = int(len(pol))

    # per-seed score sensitivity (each seed is a full replicate of the design)
    per_seed = []
    esc_between_per_seed = []
    for seed, g in rep.groupby("seed_offset"):
        gg = g.groupby(POLICY)
        multi = gg["simulator"].nunique()
        ok = multi[multi >= 2].index
        if len(ok) == 0:
            continue
        rng_ = (gg["normalized"].max() - gg["normalized"].min()).loc[ok]
        per_seed.append({"seed_offset": int(seed), "score_sensitivity": float(rng_.mean()),
                         "n_groups": int(len(ok))})
        esc_between_per_seed.append(float((gg["pred_ord"].nunique().loc[ok] > 1).mean()))
    per_seed_df = pd.DataFrame(per_seed)
    ss_by_seed_mean = float(per_seed_df["score_sensitivity"].mean()) if len(per_seed_df) else np.nan
    ss_by_seed_sd = float(per_seed_df["score_sensitivity"].std(ddof=1)) if len(per_seed_df) > 1 else np.nan
    esc_between = float(np.mean(esc_between_per_seed)) if esc_between_per_seed else np.nan

    # ANOVA per policy
    an_rows = []
    for key, g in rep.groupby(POLICY):
        if g["simulator"].nunique() < 2 or g["seed_offset"].nunique() < 2:
            continue
        r = _anova_two_way(g)
        an_rows.append({**dict(zip(POLICY, key)), **r})
    an = pd.DataFrame(an_rows)
    an.round(4).to_csv(out_dir / "T_seed_variance_by_case.csv", index=False)

    # pooled ANOVA with case fixed effects
    pooled = {}
    if len(an):
        ss_sim = an["ss_sim"].sum(); ss_res = an["ss_resid"].sum(); ss_seed = an["ss_seed"].sum()
        df_sim = int((an["n_sims"] - 1).sum()); df_res = int((an["n"] - an["n_sims"] - an["n_seeds"] + 1).sum())
        f = (ss_sim / df_sim) / (ss_res / df_res) if ss_res > 0 else np.inf
        pooled = {"pooled_F_sim": float(f), "pooled_p_sim": float(1 - stats.f.cdf(f, df_sim, df_res)),
                  "pooled_eta2_sim": float(ss_sim / (ss_sim + ss_seed + ss_res)),
                  "pooled_eta2_seed_resid": float(ss_res / (ss_sim + ss_seed + ss_res)),
                  "frac_policies_sim_significant_05": float((an["p_sim"] < 0.05).mean()),
                  "median_icc_sim": float(an["icc_sim"].median())}

    # noise-floor-adjusted score sensitivity on the primary set
    prim = S.primary(df)
    prim_ss = float((prim.groupby(["case_id", "agent_model", "agent_arch"])["normalized"]
                     .agg(lambda s: s.max() - s.min())).mean()) if len(prim) else np.nan

    summary = {
        "n_conditions_with_replicates": int(len(cond)),
        "n_policies": int(len(an)),
        "n_policies_with_ge2_simulators": n_policies_multi_sim,
        "models_covered": ",".join(sorted(rep["agent_model"].unique())),
        "model_ids_covered": ",".join(sorted(rep["agent_model_id"].unique())),
        "models_excluded_version_change": ",".join(sorted(
            m for m, v in S.model_versions(df).items() if len(v) > 1 and m not in set(rep["agent_model"]))),
        "archs_covered": ",".join(sorted(rep["agent_arch"].unique())),
        "seeds_per_condition": float(cond["n_seeds"].mean()),
        "within_policy_sd_pooled": pooled_within_sd,
        "within_policy_range_mean": mean_within_range,
        "between_sim_range_seedavg": between_range_seedavg,
        "score_sensitivity_per_seed_mean": ss_by_seed_mean,
        "score_sensitivity_per_seed_sd": ss_by_seed_sd,
        "score_sensitivity_primary_seed0_all_cases": prim_ss,
        "score_sensitivity_noise_adjusted": (between_range_seedavg - mean_within_range),
        "ratio_between_to_within_range": (between_range_seedavg / mean_within_range
                                          if mean_within_range > 0 else np.nan),
        "escalation_instability_within_policy": esc_within,
        "escalation_instability_between_sim": esc_between,
        **pooled,
    }
    pd.DataFrame([summary]).round(4).to_csv(out_dir / "T_seed_variance.csv", index=False)
    (out_dir / "noise_floor.json").write_text(json.dumps({
        "within_policy_sd": pooled_within_sd,
        "within_policy_range_mean": mean_within_range,
        "source": "T_seed_variance.csv",
    }, indent=2))
    print("[seed_variance]")
    for k, v in summary.items():
        print(f"  {k:45s} {v}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--out", default="outputs/tables")
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.out))
