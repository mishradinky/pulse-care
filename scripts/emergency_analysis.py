"""Emergency-case evidence: exact counts and intervals instead of a bare rate.

Only a handful of cases carry ground truth emergency_911, so EMR is reported as
k-of-n with Clopper-Pearson exact CIs, a case-level cluster bootstrap CI, and a
per-case / per-model / per-simulator breakdown. Every emergency miss is exported
for clinician verification.

Outputs (outputs/tables):
  T_emergency_cases.csv      per emergency case: n trajectories, misses, exact CI
  T_emergency_by_model.csv   per (model, simulator): k/n, exact CI, near-miss split
  T_emergency_summary.csv    pooled EMR: exact CI + case-cluster bootstrap CI
docs/manual_review_emergency.csv   every miss + matched hits for clinician review
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import subsets as S
from src.metrics.compute_escalation_metrics import ORD


def exact_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    lo = stats.beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def run(scored_csv: Path, out_dir: Path, review_out: Path, n_boot: int = 2000,
        seed: int = 20260515) -> dict:
    df = pd.read_csv(scored_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    prim = S.primary(df)
    em = prim[prim["correct_escalation"] == "emergency_911"].copy()
    em["pred_ord"] = em["escalation_level"].map(ORD)
    em["miss"] = (em["escalation_level"] != "emergency_911").astype(int)
    em["near_miss_ed"] = ((em["miss"] == 1) & (em["pred_ord"] == 2)).astype(int)
    em["severe_miss"] = ((em["miss"] == 1) & (em["pred_ord"] <= 1)).astype(int)

    # per case
    rows = []
    for cid, g in em.groupby("case_id"):
        k, n = int(g["miss"].sum()), len(g)
        lo, hi = exact_ci(k, n)
        rows.append({"case_id": cid, "category": g["category"].iloc[0], "n_trajectories": n,
                     "misses": k, "emr": k / n, "ci_lo": lo, "ci_hi": hi,
                     "near_miss_ed_referral": int(g["near_miss_ed"].sum()),
                     "severe_miss_le_urgent": int(g["severe_miss"].sum()),
                     "models_missing": ",".join(sorted(g.loc[g["miss"] == 1, "agent_model"].unique()))})
    by_case = pd.DataFrame(rows)
    by_case.round(4).to_csv(out_dir / "T_emergency_cases.csv", index=False)

    # per model x simulator
    rows = []
    for (m, s), g in em.groupby(["agent_model", "simulator"]):
        k, n = int(g["miss"].sum()), len(g)
        lo, hi = exact_ci(k, n)
        rows.append({"agent_model": m, "simulator": s, "n": n, "misses": k, "emr": k / n,
                     "ci_lo": lo, "ci_hi": hi, "near_miss_ed_referral": int(g["near_miss_ed"].sum()),
                     "severe_miss_le_urgent": int(g["severe_miss"].sum())})
    for m, g in em.groupby("agent_model"):
        k, n = int(g["miss"].sum()), len(g)
        lo, hi = exact_ci(k, n)
        rows.append({"agent_model": m, "simulator": "ALL", "n": n, "misses": k, "emr": k / n,
                     "ci_lo": lo, "ci_hi": hi, "near_miss_ed_referral": int(g["near_miss_ed"].sum()),
                     "severe_miss_le_urgent": int(g["severe_miss"].sum())})
    pd.DataFrame(rows).round(4).to_csv(out_dir / "T_emergency_by_model.csv", index=False)

    # pooled
    k, n = int(em["miss"].sum()), len(em)
    lo, hi = exact_ci(k, n)
    rng = np.random.default_rng(seed)
    cases = em["case_id"].unique()
    per_case = {c: g["miss"].to_numpy() for c, g in em.groupby("case_id")}
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(cases, size=len(cases), replace=True)
        arr = np.concatenate([per_case[c] for c in draw])
        boots.append(arr.mean())
    blo, bhi = np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan)
    summary = {
        "n_emergency_cases": int(len(cases)), "n_cases_total": int(prim["case_id"].nunique()),
        "n_trajectories": n, "misses": k, "emr": k / n if n else np.nan,
        "exact_ci_lo": lo, "exact_ci_hi": hi,
        "case_bootstrap_ci_lo": float(blo), "case_bootstrap_ci_hi": float(bhi),
        "near_miss_ed_referral": int(em["near_miss_ed"].sum()),
        "severe_miss_le_urgent": int(em["severe_miss"].sum()),
        "cases_with_zero_misses": int((by_case["misses"] == 0).sum()) if len(by_case) else 0,
        "subset": "primary",
    }
    pd.DataFrame([summary]).round(4).to_csv(out_dir / "T_emergency_summary.csv", index=False)

    # clinician verification sheet: every miss (all subsets) + matched hits
    allem = df[df["correct_escalation"] == "emergency_911"].copy()
    allem["miss"] = (allem["escalation_level"] != "emergency_911").astype(int)
    misses = allem[allem["miss"] == 1]
    hits = allem[allem["miss"] == 0].sample(n=min(len(misses), (allem["miss"] == 0).sum()),
                                            random_state=seed)
    sheet = pd.concat([misses, hits], ignore_index=True)
    sheet["auto_verdict"] = np.where(sheet["miss"] == 1, "miss", "hit")
    for col in ["human_escalation", "human_miss_confirmed", "human_pass_fail", "reviewer_notes"]:
        sheet[col] = ""
    sheet["file_path"] = sheet["trajectory_id"].apply(lambda t: f"outputs/trajectories/{t}.json")
    review_out.parent.mkdir(parents=True, exist_ok=True)
    sheet.drop(columns=["miss"]).to_csv(review_out, index=False)

    print("[emergency_analysis]")
    print(pd.DataFrame([summary]).T.to_string(header=False))
    print(by_case.to_string(index=False))
    print(f"  clinician verification sheet: {review_out} ({len(misses)} misses + {len(hits)} hits)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--review-out", default="docs/manual_review_emergency.csv")
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.out), Path(a.review_out))
