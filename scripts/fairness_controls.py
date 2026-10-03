"""Demographic-disparity controls: separate demographic effects from simulator noise
and judge instability.

Reports, with case-level bootstrap CIs:
  cad_escalation        CAD (deterministic escalation mismatch on swap pairs)
  cad_null_escalation   same statistic on (seed 0, seed k) baseline pairs = noise floor
  cad_excess            cad_escalation - cad_null_escalation (with CI; excludes 0?)
  cad_judged_*          LLM-judge CAD per judge column (disparity_severity, disparity_severity__<judge>)
  judge agreement       Cohen's kappa between judge columns where >1 judge exists
  by variant / model    breakdown of the deterministic CAD
  permutation test      is cad_escalation larger than cad_null beyond chance?

Outputs (outputs/tables): T_fairness_controls.csv, T_fairness_pairs.csv,
                          T_fairness_judges.csv, fairness_controls.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.metrics import subsets as S
from src.metrics.bootstrap_ci import resample_cases
from src.metrics.compute_fairness import cad_metrics, fairness_pairs, null_pairs


def _kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = sum(m[i].sum() * m[:, i].sum() for i in range(len(cats))) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def run(scored_csv: Path, out_dir: Path, n_boot: int = 1000, seed: int = 20260515) -> dict:
    df = pd.read_csv(scored_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    fair = S.fairness(df)
    rep = S.repeat(df)
    pairs = fairness_pairs(fair)
    npairs = null_pairs(rep)
    if pairs.empty:
        print("[fairness_controls] no counterfactual pairs found")
        return {}
    pairs.to_csv(out_dir / "T_fairness_pairs.csv", index=False)

    point = cad_metrics(fair, null_df=rep)

    # bootstrap over cases (fairness cases and repeat cases resampled independently)
    rng = np.random.default_rng(seed)
    boots = {"cad_escalation": [], "cad_null_escalation": [], "cad_excess": [],
             "cad_judged_major": []}
    for _ in range(n_boot):
        f_sub = resample_cases(fair, rng)
        r_sub = resample_cases(rep, rng) if len(rep) else rep
        m = cad_metrics(f_sub, null_df=r_sub)
        for k in boots:
            boots[k].append(m[k])
    ci = {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))
          for k, v in boots.items() if len(v)}

    # permutation test: pool swap pairs and null pairs, shuffle labels
    perm_p = np.nan
    if not npairs.empty:
        obs = point["cad_excess"]
        x = np.concatenate([pairs["escalation_differs"].to_numpy(), npairs["escalation_differs"].to_numpy()])
        n1 = len(pairs)
        cnt = 0
        for _ in range(5000):
            rng.shuffle(x)
            if x[:n1].mean() - x[n1:].mean() >= obs:
                cnt += 1
        perm_p = cnt / 5000

    rows = []
    def add(name, val, lo=np.nan, hi=np.nan, n="", note=""):
        rows.append({"statistic": name, "value": val, "ci_lo": lo, "ci_hi": hi, "n": n, "note": note})
    add("cad_escalation", point["cad_escalation"], *ci.get("cad_escalation", (np.nan, np.nan)),
        point["n_pairs"], "CAD: escalation differs between baseline and swap")
    add("cad_null_escalation", point["cad_null_escalation"], *ci.get("cad_null_escalation", (np.nan, np.nan)),
        point["n_null_pairs"], "noise floor: escalation differs between two seeds of the same baseline")
    add("cad_excess", point["cad_excess"], *ci.get("cad_excess", (np.nan, np.nan)), "",
        f"cad_escalation - cad_null; permutation p={perm_p:.4f}" if not np.isnan(perm_p) else "no null pairs")
    add("cad_judged_major", point["cad_judged_major"], *ci.get("cad_judged_major", (np.nan, np.nan)),
        point["n_pairs"], "LLM judge (disparity_severity == major)")
    add("cad_judged_any", point["cad_judged_any"], n=point["n_pairs"], note="LLM judge major or minor")
    for v, val in point["by_variant"].items():
        add(f"cad_escalation[{v}]", val, n=int((pairs["demographic_variant"] == v).sum()))
    for mdl, val in point["by_model"].items():
        add(f"cad_escalation[{mdl}]", val, n=int((pairs["agent_model"] == mdl).sum()))
    if "score_delta" in pairs:
        add("mean_score_delta_swap_minus_base", float(pairs["score_delta"].mean()),
            n=len(pairs), note="normalized score, swap minus baseline")
    if not npairs.empty and "score_delta" in npairs:
        add("mean_abs_score_delta_null", float(npairs["score_delta"].abs().mean()), n=len(npairs))
        add("mean_abs_score_delta_swap", float(pairs["score_delta"].abs().mean()), n=len(pairs))

    # judge columns
    # judge verdicts on swap pairs live in disparity_severity[__judge]; *__null columns hold
    # verdicts on reseed pairs and are handled separately below
    judge_cols = [c for c in pairs.columns if c.startswith("disparity_severity") and not c.endswith("__null")]
    jrows = []
    for c in judge_cols:
        sev = pairs[c].astype(str)
        jrows.append({"judge_column": c, "major_rate": float((sev == "major").mean()),
                      "any_rate": float(sev.isin(["major", "minor"]).mean()),
                      "n_scored": int(sev.isin(["none", "minor", "major"]).sum())})
    # judge false-positive rate: verdicts on same-patient reseed pairs
    for c in [c for c in npairs.columns if c.startswith("disparity_severity") and c.endswith("__null")]:
        sev = npairs[c].astype(str)
        ok = sev.isin(["none", "minor", "major"])
        if ok.any():
            jrows.append({"judge_column": c + " (null pairs = judge false-positive rate)",
                          "major_rate": float((sev[ok] == "major").mean()),
                          "any_rate": float(sev[ok].isin(["major", "minor"]).mean()),
                          "n_scored": int(ok.sum())})
    for i in range(len(judge_cols)):
        for j in range(i + 1, len(judge_cols)):
            a, b = pairs[judge_cols[i]].astype(str), pairs[judge_cols[j]].astype(str)
            ok = a.isin(["none", "minor", "major"]) & b.isin(["none", "minor", "major"])
            if ok.sum():
                jrows.append({"judge_column": f"kappa({judge_cols[i]},{judge_cols[j]})",
                              "major_rate": _kappa(a[ok], b[ok]),
                              "any_rate": _kappa((a[ok] == "major").astype(int), (b[ok] == "major").astype(int)),
                              "n_scored": int(ok.sum())})
    pd.DataFrame(jrows).round(4).to_csv(out_dir / "T_fairness_judges.csv", index=False)

    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(out_dir / "T_fairness_controls.csv", index=False)
    out = {"point": {k: v for k, v in point.items()}, "ci": ci, "permutation_p": perm_p,
           "judges": jrows, "n_boot": n_boot}
    (out_dir / "fairness_controls.json").write_text(json.dumps(out, indent=2, default=float))
    print("[fairness_controls]")
    print(tab.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.out), n_boot=a.n_boot)
