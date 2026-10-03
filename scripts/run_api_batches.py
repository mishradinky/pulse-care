"""One-command orchestrator for the extended API runs.

For each stage, in order:
  1. dry-run the subset and print how many trajectories are still pending
  2. run the subset with the given parallelism (skips completed files, resumable)
  3. score the new trajectory files with the primary scorer (Claude Haiku 4.5)
then the scorer / judge stages, then the full analysis pipeline.

Stages (default order; pick a subset with --stages):
  repeatability_full   200 new   seed noise floor for every model x architecture
  fairness_null         69 new   baseline-vs-baseline reseed pairs (CAD noise floor)
  factorial            192 new   which simulator property drives the effect
  extended_models_lite  96 new   Sonnet 4.6 / GPT-4.1 / Mistral Large on 8 cases
  emergency_ext        180 new   6 new emergency cases (+ static)  [clinician review first]
  extended_models      288+72    full extended-model coverage (optional, large)
  scorers                        independent task scorers on the fixed 60-trajectory sample
  judges                         disparity judge on swap pairs (2nd judge) and on null pairs
  analysis                       scripts/run_full_analysis.py

Examples:
  python scripts/run_api_batches.py --dry-run
  python scripts/run_api_batches.py --stages repeatability_full fairness_null --parallel 2
  python scripts/run_api_batches.py --stages scorers judges analysis
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = sys.executable

TRAJECTORY_STAGES: dict[str, list[str]] = {
    "repeatability_full": ["repeatability_full"],
    "fairness_null": ["fairness_null"],
    "factorial": ["factorial"],
    "extended_models_lite": ["extended_models_lite"],
    "emergency_ext": ["emergency_ext", "emergency_ext_static"],
    "extended_models": ["extended_models", "extended_models_static"],
}
# default = original models only (Haiku / GPT-4o / Mistral Small agents, Haiku simulator+scorer,
# GPT-4o judge). `scorers` and `extended_models_lite` add other LLMs and are opt-in.
DEFAULT_ORDER = ["repeatability_full", "fairness_null", "factorial", "emergency_ext", "judges", "analysis"]


def run(cmd: list[str], check: bool = True) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed with exit code {r.returncode}: {' '.join(cmd)}")
    return r.returncode


def score_new() -> None:
    run([PY, "-c",
         "from pathlib import Path; import yaml; from scripts.score_incremental import score_incremental; "
         "mc=yaml.safe_load(Path('configs/model_config.yaml').read_text(encoding='utf-8')); "
         "score_incremental(trajectories_dir=Path('outputs/trajectories'), cases_dir=Path('cases'), "
         "scored_csv=Path('outputs/scored/all_scored.csv'), "
         "scorer_cfg=mc['scorers']['claude_haiku_4_5'], simulate=False, parallel=4)"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=DEFAULT_ORDER)
    ap.add_argument("--parallel", type=int, default=2, help="<=2 when GPT-4o is involved (30K TPM)")
    ap.add_argument("--dry-run", action="store_true", help="only print pending counts")
    ap.add_argument("--scorers", nargs="*", default=["claude_sonnet_4_6", "gpt_4_1"])
    ap.add_argument("--second-judge", default="none",
                    help="extra disparity judge (e.g. claude_sonnet_4_6); 'none' keeps only the original GPT-4o judge")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict trajectory stages to these agent models (e.g. claude_haiku_4_5 gpt_4o while Mistral is blocked)")
    ap.add_argument("--primary-judge", default="gpt_4o",
                    help="judge that produced the existing disparity_severity column")
    a = ap.parse_args()

    t0 = time.time()
    for stage in a.stages:
        print(f"\n{'=' * 70}\nSTAGE {stage}\n{'=' * 70}")
        if stage in TRAJECTORY_STAGES:
            for subset in TRAJECTORY_STAGES[stage]:
                run([PY, "-m", "src.batch_driver", "--subset", subset, "--dry-run"])
                if a.dry_run:
                    continue
                mflag = ["--models", *a.models] if a.models else []
                run([PY, "-m", "src.batch_driver", "--subset", subset, "--parallel", str(a.parallel), *mflag])
                # second pass with parallel=1 picks up anything that failed on rate limits
                run([PY, "-m", "src.batch_driver", "--subset", subset, "--parallel", "1", *mflag])
            if not a.dry_run:
                score_new()
        elif stage == "scorers":
            if a.dry_run:
                print("would re-score the 60-trajectory sample with:", a.scorers); continue
            run([PY, "scripts/cross_scorer_agreement.py", "--scorers", *a.scorers,
                 "--parallel", str(a.parallel)])
        elif stage == "judges":
            if a.dry_run:
                print(f"would judge swap pairs with {a.second_judge} and null pairs with "
                      f"{a.primary_judge} and {a.second_judge}"); continue
            run([PY, "scripts/score_fairness.py", "--scorer-backbone", a.primary_judge, "--pair-mode", "null",
                 "--out-col", "disparity_severity__null", "--parallel", str(a.parallel)])
            if a.second_judge != "none":
                run([PY, "scripts/score_fairness.py", "--scorer-backbone", a.second_judge,
                     "--out-col", f"disparity_severity__{a.second_judge}", "--parallel", str(a.parallel)])
                run([PY, "scripts/score_fairness.py", "--scorer-backbone", a.second_judge, "--pair-mode", "null",
                     "--out-col", f"disparity_severity__{a.second_judge}__null", "--parallel", str(a.parallel)])
        elif stage == "analysis":
            if a.dry_run:
                print("would run scripts/run_full_analysis.py"); continue
            run([PY, "scripts/run_full_analysis.py"])
        else:
            raise SystemExit(f"unknown stage {stage}; choose from {DEFAULT_ORDER + ['extended_models']}")
    print(f"\n[run_api_batches] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
