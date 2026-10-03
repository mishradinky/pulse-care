"""Score fairness: LLM-judged action disparity for counterfactual pairs.

Pairs each non-baseline trajectory with its matching baseline trajectory (same case,
simulator, model, arch, simulator backbone, seed), calls the action-disparity judge and
writes the severity into a column of all_scored.csv.

Controls for judge instability and simulator noise:
  --scorer-backbone X --out-col disparity_severity__X   second judge, separate column
  --pair-mode null                                      judge (seed 0, seed k) BASELINE pairs
                                                        so the judge's false-positive rate can
                                                        be measured (writes to the seed-k row)

The deterministic CAD (escalation mismatch) does not need this script; see
scripts/fairness_controls.py.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

import pandas as pd
import yaml

from src.model_clients import make_client
from src.output_parser import parse_with_retry

KEYS = ["case_id", "simulator", "agent_model", "agent_arch", "sim_backbone", "seed_offset"]


def _score_disparity_pair(baseline_traj: dict, variant_traj: dict, scorer_cfg: dict,
                          simulate: bool = False) -> str:
    client = make_client(scorer_cfg["provider"], scorer_cfg, simulate=simulate)
    prompt_tpl = Path("prompts/scorer_action_disparity.txt").read_text(encoding="utf-8")
    prompt = (prompt_tpl.replace("{T_A_JSON}", json.dumps(baseline_traj))
              .replace("{T_B_JSON}", json.dumps(variant_traj)))
    obj, _ = parse_with_retry(client, [{"role": "user", "content": prompt}], system=None,
                              schema=None, temperature=0.0, max_tokens=512, retries=2)
    if not isinstance(obj, dict):
        return "none"
    return obj.get("severity", "none")


def run(trajectories_dir: Path, scored_csv: Path, scorer_cfg: dict, simulate: bool = False,
        parallel: int = 4, out_col: str = "disparity_severity", pair_mode: str = "variant",
        only_missing: bool = True) -> pd.DataFrame:
    df = pd.read_csv(scored_csv)
    if out_col not in df.columns:
        df[out_col] = "none"
    # pairs must be the same API model version (Mistral Small 3.2 -> 4 change); legacy rows
    # without agent_model_id are filled from the May model ids
    from src.metrics.subsets import with_model_ids
    df = with_model_ids(df)
    keys = [k for k in KEYS if k in df.columns] + ["agent_model_id"]
    base_all = df[df["demographic_variant"] == "baseline"]

    if pair_mode == "variant":
        targets = df[df["demographic_variant"] != "baseline"].copy()
        base_map = {tuple(r[k] for k in keys): r["trajectory_id"] for _, r in base_all.iterrows()}
        def base_id(row):
            return base_map.get(tuple(row[k] for k in keys))
    else:  # null: seed k>0 baseline rows vs seed 0 baseline rows
        targets = base_all[(base_all["seed_offset"].astype(int) > 0)
                           & (base_all["simulator"] != "full_info_static")].copy()
        k0 = [k for k in keys if k != "seed_offset"]
        base_map = {tuple(r[k] for k in k0): r["trajectory_id"]
                    for _, r in base_all[base_all["seed_offset"].astype(int) == 0].iterrows()}
        def base_id(row):
            return base_map.get(tuple(row[k] for k in k0))

    if only_missing:
        targets = targets[~targets[out_col].astype(str).isin(["major", "minor"]) |
                          (targets[out_col].astype(str) == "none") | targets[out_col].isna()]
        # only_missing skips rows that already have a judged label other than the default
        judged = df.loc[targets.index, out_col].astype(str).isin(["major", "minor"])
        targets = targets[~judged]
    if targets.empty:
        print(f"[fairness_scorer] nothing to score (pair_mode={pair_mode}, col={out_col})")
        return df
    print(f"[fairness_scorer] scoring {len(targets)} pairs (pair_mode={pair_mode}) -> {out_col}")

    def _load(tid: str) -> dict | None:
        p = trajectories_dir / f"{tid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _one(row: dict) -> tuple[str, str]:
        bid = base_id(row)
        if not bid:
            return row["trajectory_id"], "none"
        b, v = _load(bid), _load(row["trajectory_id"])
        if not b or not v:
            return row["trajectory_id"], "none"
        try:
            return row["trajectory_id"], _score_disparity_pair(b, v, scorer_cfg, simulate)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {row['trajectory_id']}: {e}")
            return row["trajectory_id"], "none"

    updates = {}
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = [ex.submit(_one, r) for r in targets.to_dict("records")]
        for i, f in enumerate(as_completed(futs), 1):
            tid, sev = f.result()
            updates[tid] = sev
            if i % 20 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}]")
    df[out_col] = df.apply(lambda r: updates.get(r["trajectory_id"], r[out_col]), axis=1)
    df.to_csv(scored_csv, index=False)
    print(f"[fairness_scorer] updated {len(updates)} rows in {scored_csv}")
    print(df.loc[df["trajectory_id"].isin(updates), out_col].value_counts().to_string())
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="outputs/trajectories")
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    ap.add_argument("--scorer-backbone", default="claude_haiku_4_5")
    ap.add_argument("--out-col", default=None,
                    help="column to write (default: disparity_severity for the primary judge, "
                         "disparity_severity__<scorer> otherwise)")
    ap.add_argument("--pair-mode", choices=["variant", "null"], default="variant")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="re-judge rows that already have a label")
    a = ap.parse_args()
    mc = yaml.safe_load(Path(a.model_cfg).read_text(encoding="utf-8"))
    cfg = (mc.get("scorers", {}).get(a.scorer_backbone)
           or mc["patient_simulator_backbones"].get(a.scorer_backbone)
           or mc["agents"][a.scorer_backbone])
    out_col = a.out_col or ("disparity_severity" if a.scorer_backbone == "claude_haiku_4_5"
                            else f"disparity_severity__{a.scorer_backbone}")
    if a.pair_mode == "null" and a.out_col is None:
        out_col = f"{out_col}__null"
    run(Path(a.trajectories), Path(a.scored_csv), cfg, simulate=a.simulate, parallel=a.parallel,
        out_col=out_col, pair_mode=a.pair_mode, only_missing=not a.redo)
