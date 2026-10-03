"""Score only trajectory files not yet in all_scored.csv, then append."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

import pandas as pd
import yaml

from src.metrics.compute_task_scores import _score_one, score_all
from concurrent.futures import ThreadPoolExecutor, as_completed


def score_incremental(
    trajectories_dir: Path,
    cases_dir: Path,
    scored_csv: Path,
    scorer_cfg: dict,
    simulate: bool = False,
    parallel: int = 8,
) -> pd.DataFrame:
    existing_ids: set[str] = set()
    if scored_csv.exists():
        existing_df = pd.read_csv(scored_csv)
        existing_ids = set(existing_df["trajectory_id"].tolist())
        print(f"[incremental] Existing scored: {len(existing_ids)}")

    all_files = sorted(trajectories_dir.glob("*.json"))
    new_files = [f for f in all_files if f.stem not in existing_ids]
    print(f"[incremental] New trajectories to score: {len(new_files)}")

    if not new_files:
        print("[incremental] Nothing to score.")
        return pd.read_csv(scored_csv) if scored_csv.exists() else pd.DataFrame()

    rows = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_score_one, tf, cases_dir, scorer_cfg, simulate): tf for tf in new_files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result:
                rows.append(result)
            if done % 10 == 0 or done == len(new_files):
                print(f"  [{done}/{len(new_files)}] scored: {len(rows)} ok")

    new_df = pd.DataFrame(rows)
    if scored_csv.exists():
        existing_df = pd.read_csv(scored_csv)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    scored_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(scored_csv, index=False)
    print(f"[incremental] Saved {len(combined)} total rows -> {scored_csv}")
    return combined


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="outputs/trajectories")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    model_cfg = yaml.safe_load(Path(args.model_cfg).read_text(encoding="utf-8"))
    scorer_cfg = model_cfg["patient_simulator_backbones"]["claude_haiku_4_5"]

    score_incremental(
        trajectories_dir=Path(args.trajectories),
        cases_dir=Path(args.cases),
        scored_csv=Path(args.scored_csv),
        scorer_cfg=scorer_cfg,
        simulate=args.simulate,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    _main()
