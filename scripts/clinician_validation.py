"""Clinician-reviewed validation of automated labels.

Reads the filled review sheets and computes, per label type, agreement between the
automated pipeline and the clinician:

  pass_fail    Cohen's kappa + disagreement rate (auto: normalized >= 0.5)
  task_score   Pearson r, mean |delta|
  escalation   agreement of auto (derived: pred vs truth -> correct/under/over)
               with human_escalation; kappa
  leakage      weighted (linear) kappa on 0/1/2
  disclosure   kappa (auto disclosure_failure 1 -> "yes")
  disparity    kappa of LLM judge (disparity_severity) vs human_disparity on fairness rows
  emergency    from docs/manual_review_emergency.csv: fraction of automated misses the
               clinician confirmed (human_miss_confirmed yes/no)

Rows with empty human labels are skipped per label. Writes
outputs/tables/T_clinician_validation.csv (one row per label with n) and, when
pass/fail labels exist, this becomes the reference for scorer_pass_fail_disagree in
compute_efi (it takes precedence over the LLM cross-scorer table).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.metrics.compute_escalation_metrics import ORD


def kappa(a, b, weights: str | None = None) -> float:
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    k = len(cats)
    if k < 2 or len(a) == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if weights == "linear":
        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    else:
        w = 1 - np.eye(k)
    e = np.outer(m.sum(1), m.sum(0)) / n
    po = (w * m).sum() / n
    pe = (w * e).sum() / n
    return float(1 - po / pe) if pe > 0 else 1.0


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})


def run(review_csv: Path, emergency_csv: Path, out_dir: Path,
        extra_csvs: list[Path] | None = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sheets = [p for p in [review_csv] + list(extra_csvs or []) if p.exists()]
    if sheets:
        df = pd.concat([pd.read_csv(p) for p in sheets], ignore_index=True)
        df = df.drop_duplicates("trajectory_id")
        for c in ["human_pass_fail", "human_task_score", "human_escalation", "human_leakage",
                  "human_disclosure", "human_disparity"]:
            if c not in df.columns:
                df[c] = ""

        # pass/fail
        h = _clean(df["human_pass_fail"])
        ok = h.isin(["pass", "fail"])
        if ok.any():
            auto = np.where(df.loc[ok, "normalized"] >= 0.5, "pass", "fail")
            rows.append({"label": "pass_fail", "n": int(ok.sum()),
                         "agreement": float((auto == h[ok].to_numpy()).mean()),
                         "disagree_rate": float((auto != h[ok].to_numpy()).mean()),
                         "kappa": kappa(auto, h[ok]), "stat2": "", "note": "auto pass = normalized >= 0.5"})
        else:
            rows.append({"label": "pass_fail", "n": 0, "agreement": "", "disagree_rate": "", "kappa": "",
                         "stat2": "", "note": "no human labels yet"})

        # task score
        hs = pd.to_numeric(df["human_task_score"], errors="coerce")
        ok = hs.notna()
        if ok.sum() >= 3:
            a = df.loc[ok, "normalized"].astype(float)
            rows.append({"label": "task_score", "n": int(ok.sum()),
                         "agreement": float(np.corrcoef(a, hs[ok])[0, 1]),
                         "disagree_rate": float((a - hs[ok]).abs().mean()),
                         "kappa": "", "stat2": float((a - hs[ok]).mean()),
                         "note": "agreement=Pearson r; disagree_rate=mean|auto-human|; stat2=mean(auto-human)"})

        # escalation
        h = _clean(df["human_escalation"])
        ok = h.isin(["correct", "under", "over"])
        if ok.any():
            po = df.loc[ok, "escalation_level"].map(ORD); to = df.loc[ok, "correct_escalation"].map(ORD)
            auto = np.where(po == to, "correct", np.where(po < to, "under", "over"))
            rows.append({"label": "escalation", "n": int(ok.sum()),
                         "agreement": float((auto == h[ok].to_numpy()).mean()),
                         "disagree_rate": float((auto != h[ok].to_numpy()).mean()),
                         "kappa": kappa(auto, h[ok]), "stat2": "",
                         "note": "auto label derived from escalation_level vs ground truth"})

        # leakage
        hl = pd.to_numeric(df["human_leakage"], errors="coerce")
        ok = hl.isin([0, 1, 2])
        if ok.any():
            a = df.loc[ok, "leakage_level"].astype(int)
            rows.append({"label": "leakage", "n": int(ok.sum()),
                         "agreement": float((a.to_numpy() == hl[ok].astype(int).to_numpy()).mean()),
                         "disagree_rate": float((a.to_numpy() != hl[ok].astype(int).to_numpy()).mean()),
                         "kappa": kappa(a.astype(int), hl[ok].astype(int), weights="linear"),
                         "stat2": "", "note": "kappa is linear-weighted"})

        # disclosure
        h = _clean(df["human_disclosure"])
        ok = h.isin(["yes", "no"])
        if ok.any():
            a = np.where(df.loc[ok, "disclosure_failure"].astype(int) == 1, "yes", "no")
            rows.append({"label": "disclosure", "n": int(ok.sum()),
                         "agreement": float((a == h[ok].to_numpy()).mean()),
                         "disagree_rate": float((a != h[ok].to_numpy()).mean()),
                         "kappa": kappa(a, h[ok]), "stat2": "", "note": ""})

        # disparity (fairness rows)
        h = _clean(df["human_disparity"])
        ok = h.isin(["none", "minor", "major"]) & (df["demographic_variant"] != "baseline")
        if ok.any():
            a = _clean(df.loc[ok, "disparity_severity"])
            rows.append({"label": "disparity", "n": int(ok.sum()),
                         "agreement": float((a.to_numpy() == h[ok].to_numpy()).mean()),
                         "disagree_rate": float((a.to_numpy() != h[ok].to_numpy()).mean()),
                         "kappa": kappa(a, h[ok], weights="linear"),
                         "stat2": kappa((a == "major").astype(int), (h[ok] == "major").astype(int)),
                         "note": "kappa linear-weighted on none/minor/major; stat2 = kappa on major vs not"})

    if emergency_csv.exists():
        e = pd.read_csv(emergency_csv)
        if "human_miss_confirmed" in e.columns:
            h = _clean(e["human_miss_confirmed"])
            ok = h.isin(["yes", "no"]) & (e["auto_verdict"] == "miss")
            if ok.any():
                rows.append({"label": "emergency_miss_confirmed", "n": int(ok.sum()),
                             "agreement": float((h[ok] == "yes").mean()),
                             "disagree_rate": float((h[ok] == "no").mean()), "kappa": "", "stat2": "",
                             "note": "fraction of automated emergency misses confirmed by clinician"})

    t = pd.DataFrame(rows)
    if t.empty:
        t = pd.DataFrame([{"label": "pass_fail", "n": 0, "agreement": "", "disagree_rate": "",
                           "kappa": "", "stat2": "", "note": "no review sheets found"}])
    t.to_csv(out_dir / "T_clinician_validation.csv", index=False)
    print("[clinician_validation]")
    print(t.to_string(index=False))
    if (t["n"].astype(float) == 0).all():
        print("\n  No clinician labels found yet. Fill docs/manual_review.csv and "
              "docs/manual_review_emergency.csv, then re-run this script.")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-csv", default="docs/manual_review.csv")
    ap.add_argument("--emergency-csv", default="docs/manual_review_emergency.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--extra-csv", nargs="*", default=["docs/manual_review_targeted.csv"],
                    help="additional review sheets with the same label columns")
    a = ap.parse_args()
    run(Path(a.review_csv), Path(a.emergency_csv), Path(a.out), [Path(p) for p in a.extra_csv])
