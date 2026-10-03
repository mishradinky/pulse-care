"""Cross-scorer agreement with independent (non-evaluated) scorers and consensus.

The primary task scorer (claude_haiku_4_5) and the original alternative (gpt_4o) are both
evaluated agent models. This script re-scores a fixed stratified sample with any number of
scorers - by default the independent ones defined under `scorers:` in model_config.yaml -
and reports:

  T_scorer_agreement.csv         one row per alternative scorer vs primary
                                 (kappa pass/fail, r, mean |delta|, disagree rate,
                                 is_evaluated_model) plus a `consensus` row where the
                                 consensus score is the median across all scorers and
                                 pass/fail is the majority vote
  T_scorer_pairwise.csv          pass/fail kappa and score r for every scorer pair
  T_scorer_agreement_detail.csv  per-trajectory scores from every scorer (leakage too)

Scorers already present in the detail CSV are not re-run unless --force. The sample is
the trajectory set already in the detail CSV (so every scorer sees the same 60), or a
fresh stratified sample when none exists.

Usage:
  python scripts/cross_scorer_agreement.py --scorers claude_sonnet_4_6 gpt_4_1 --parallel 2
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

import numpy as np
import pandas as pd
import yaml
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from src.metrics.compute_task_scores import score_leakage, score_trajectory

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = Exception

SEED = 20260515
TARGET_N = 60
STRATIFY_BY = ["simulator", "agent_model"]
PRIMARY = "claude_haiku_4_5"
EVALUATED = {"claude_haiku_4_5", "gpt_4o", "mistral_small_3_2"}


def cohen_kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0 or len(cats) < 2:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = sum((m[i].sum() / n) * (m[:, i].sum() / n) for i in range(len(cats)))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


@retry(retry=retry_if_exception_type(OpenAIRateLimitError),
       wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
def _score(traj, gt, cfg, simulate):
    task = score_trajectory(traj, gt, cfg, simulate=simulate)
    leak = score_leakage(traj, gt, cfg, simulate=simulate)
    return task, leak


def _migrate_legacy(detail: pd.DataFrame) -> pd.DataFrame:
    if "alt_normalized" in detail.columns and "gpt_4o_normalized" not in detail.columns:
        detail = detail.rename(columns={"alt_normalized": "gpt_4o_normalized", "alt_raw": "gpt_4o_raw"})
    return detail


def _scorer_cfg(model_cfg: dict, name: str) -> dict:
    if name in model_cfg.get("scorers", {}):
        return model_cfg["scorers"][name]
    if name in model_cfg.get("patient_simulator_backbones", {}):
        return model_cfg["patient_simulator_backbones"][name]
    if name in model_cfg.get("agents", {}):
        return model_cfg["agents"][name]
    raise KeyError(f"scorer '{name}' not found in model_config.yaml")


def run(scored_csv: Path, trajectories_dir: Path, cases_dir: Path, model_cfg_path: Path,
        out_dir: Path, scorers: list[str], n: int = TARGET_N, seed: int = SEED,
        simulate: bool = False, parallel: int = 1, force: bool = False) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    df = pd.read_csv(scored_csv)
    dyn = df[df["simulator"] != "full_info_static"].copy()
    model_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))

    detail_path = out_dir / "T_scorer_agreement_detail.csv"
    if detail_path.exists():
        detail = _migrate_legacy(pd.read_csv(detail_path))
        sample_df = dyn[dyn["trajectory_id"].isin(detail["trajectory_id"])].copy()
        print(f"[cross_scorer] reusing existing sample of {len(sample_df)} trajectories")
    else:
        strat = dyn.groupby(STRATIFY_BY)
        per_cell = max(1, n // len(strat))
        parts = [g.sample(n=min(per_cell, len(g)), random_state=rng.randint(0, 2**31)) for _, g in strat]
        sample_df = pd.concat(parts, ignore_index=True).head(n)
        detail = sample_df[["trajectory_id", "simulator", "agent_model"]].copy()
        detail["primary_normalized"] = sample_df["normalized"].values
        detail["primary_raw"] = sample_df["raw_total"].values
    # keep primary columns in sync with scored csv
    detail = detail.drop(columns=[c for c in ["primary_normalized", "primary_raw", "primary_leakage"]
                                  if c in detail.columns])
    detail = detail.merge(sample_df[["trajectory_id", "normalized", "raw_total", "leakage_level"]]
                          .rename(columns={"normalized": "primary_normalized", "raw_total": "primary_raw",
                                           "leakage_level": "primary_leakage"}),
                          on="trajectory_id", how="left")

    for scorer in scorers:
        col = f"{scorer}_normalized"
        if col in detail.columns and not force and detail[col].notna().all():
            print(f"[cross_scorer] {scorer}: already scored, skipping (use --force to redo)")
            continue
        cfg = _scorer_cfg(model_cfg, scorer)
        print(f"[cross_scorer] scoring {len(sample_df)} trajectories with {scorer} ({cfg.get('model')}) ...")

        def _one(row: dict) -> dict | None:
            tf = trajectories_dir / f"{row['trajectory_id']}.json"
            cf = cases_dir / f"{row['case_id']}.json"
            if not tf.exists() or not cf.exists():
                return None
            traj = json.loads(tf.read_text(encoding="utf-8"))
            case = json.loads(cf.read_text(encoding="utf-8"))
            gt = traj.get("ground_truth", case["ground_truth"])
            try:
                task, leak = _score(traj, gt, cfg, simulate)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR {scorer} on {row['trajectory_id']}: {e}")
                return None
            return {"trajectory_id": row["trajectory_id"], col: task.get("normalized", np.nan),
                    f"{scorer}_raw": task.get("raw_total", np.nan),
                    f"{scorer}_leakage": leak.get("leakage_level", np.nan)}

        res = []
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(_one, r) for r in sample_df.to_dict("records")]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                if r:
                    res.append(r)
                if i % 10 == 0 or i == len(futs):
                    print(f"  [{i}/{len(futs)}] ok: {len(res)}")
        new = pd.DataFrame(res)
        if new.empty:
            print(f"  no results for {scorer}")
            continue
        detail = detail.drop(columns=[c for c in new.columns if c != "trajectory_id" and c in detail.columns])
        detail = detail.merge(new, on="trajectory_id", how="left")

    detail.to_csv(detail_path, index=False)

    # summaries
    score_cols = {"primary": "primary_normalized"}
    for c in detail.columns:
        if c.endswith("_normalized") and c != "primary_normalized":
            score_cols[c[:-len("_normalized")]] = c
    pf = {k: (detail[c] >= 0.5).astype(int) for k, c in score_cols.items()}
    rows = []
    for k, c in score_cols.items():
        if k == "primary":
            continue
        ok = detail[c].notna() & detail["primary_normalized"].notna()
        a, b = detail.loc[ok, "primary_normalized"], detail.loc[ok, c]
        rows.append({"n_trajectories": int(ok.sum()), "primary_scorer": PRIMARY, "alt_scorer": k,
                     "kappa_pass_fail": round(cohen_kappa(pf["primary"][ok].tolist(), pf[k][ok].tolist()), 4),
                     "score_correlation": round(float(a.corr(b)), 4),
                     "mean_abs_delta": round(float((a - b).abs().mean()), 4),
                     "mean_signed_delta_alt_minus_primary": round(float((b - a).mean()), 4),
                     "scorer_disagree_rate": round(float((pf["primary"][ok] != pf[k][ok]).mean()), 4),
                     "is_evaluated_model": int(k in EVALUATED)})
    if len(score_cols) >= 3:
        mat = detail[list(score_cols.values())]
        cons = mat.median(axis=1)
        maj = (mat >= 0.5).sum(axis=1) > (mat.notna().sum(axis=1) / 2)
        ok = mat.notna().all(axis=1)
        a = detail.loc[ok, "primary_normalized"]
        rows.append({"n_trajectories": int(ok.sum()), "primary_scorer": PRIMARY,
                     "alt_scorer": f"consensus[{'+'.join(score_cols)}]",
                     "kappa_pass_fail": round(cohen_kappa(pf["primary"][ok].tolist(), maj[ok].astype(int).tolist()), 4),
                     "score_correlation": round(float(a.corr(cons[ok])), 4),
                     "mean_abs_delta": round(float((a - cons[ok]).abs().mean()), 4),
                     "mean_signed_delta_alt_minus_primary": round(float((cons[ok] - a).mean()), 4),
                     "scorer_disagree_rate": round(float((pf["primary"][ok] != maj[ok].astype(int)).mean()), 4),
                     "is_evaluated_model": 0})
        detail["consensus_normalized"] = cons
        detail["consensus_pass"] = maj.astype(int)
        detail.to_csv(detail_path, index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "T_scorer_agreement.csv", index=False)

    prs = []
    for (ka, ca), (kb, cb) in combinations(score_cols.items(), 2):
        ok = detail[ca].notna() & detail[cb].notna()
        if ok.sum() < 3:
            continue
        prs.append({"scorer_a": ka, "scorer_b": kb, "n": int(ok.sum()),
                    "kappa_pass_fail": round(cohen_kappa(pf[ka][ok].tolist(), pf[kb][ok].tolist()), 4),
                    "score_correlation": round(float(detail.loc[ok, ca].corr(detail.loc[ok, cb])), 4),
                    "mean_abs_delta": round(float((detail.loc[ok, ca] - detail.loc[ok, cb]).abs().mean()), 4)})
    pd.DataFrame(prs).to_csv(out_dir / "T_scorer_pairwise.csv", index=False)

    print("[cross_scorer]")
    print(summary.to_string(index=False))
    if prs:
        print(pd.DataFrame(prs).to_string(index=False))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--trajectories", default="outputs/trajectories")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--scorers", nargs="+", default=["gpt_4o", "claude_sonnet_4_6", "gpt_4_1"],
                    help="scorer keys from model_config.yaml `scorers:` (or backbones/agents)")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-score scorers already present")
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.trajectories), Path(a.cases), Path(a.model_cfg), Path(a.out),
        scorers=a.scorers, simulate=a.simulate, parallel=a.parallel, force=a.force)
