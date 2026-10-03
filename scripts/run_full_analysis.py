"""Run the complete post-batch analysis pipeline.

Executes in order:
  1. Incremental scoring of any new trajectory files (primary scorer)
  2. Seed-variance decomposition           -> T_seed_variance*, noise_floor.json
  3. Margin-aware ranking robustness       -> T_rank_flips_margin, T_rank_pairwise_tests, ...
  4. Emergency-case evidence               -> T_emergency_*, docs/manual_review_emergency.csv
  5. Fairness controls (CAD vs noise floor)-> T_fairness_controls, T_fairness_pairs, ...
  6. Clinician validation (if labels exist)-> T_clinician_validation
  7. Factorial simulator effects (if run)  -> T_factorial_effects
  8. EFI + metric accounting (bootstrap)   -> T11_efi, T_metric_accounting, efi_summary.json
  9. Figures (F1-F13)
 10. Tables (T1-T10)
 11. Audit report

Usage:
  python scripts/run_full_analysis.py [--simulate] [--n-boot 1000]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

import yaml

from scripts import (
    clinician_validation,
    emergency_analysis,
    factorial_analysis,
    fairness_controls,
    rank_robustness,
    seed_variance,
)
from scripts.score_incremental import score_incremental
from src.metrics.compute_efi import run as compute_efi

SCORED = Path("outputs/scored/all_scored.csv")
TABLES = Path("outputs/tables")


def main(simulate: bool = False, n_boot: int = 1000) -> None:
    t0 = time.time()
    print("=" * 60)
    print("PULSE-Care Full Analysis Pipeline")
    print("=" * 60)

    model_cfg = yaml.safe_load(Path("configs/model_config.yaml").read_text(encoding="utf-8"))
    scorer_cfg = model_cfg.get("scorers", model_cfg["patient_simulator_backbones"])["claude_haiku_4_5"]

    print("\n[1/11] Scoring new trajectories...")
    score_incremental(trajectories_dir=Path("outputs/trajectories"), cases_dir=Path("cases"),
                      scored_csv=SCORED, scorer_cfg=scorer_cfg, simulate=simulate, parallel=8)

    print("\n[2/11] Seed variance...")
    seed_variance.run(SCORED, TABLES)
    print("\n[3/11] Ranking robustness...")
    rank_robustness.run(SCORED, TABLES, n_boot=n_boot)
    print("\n[4/11] Emergency evidence...")
    emergency_analysis.run(SCORED, TABLES, Path("docs/manual_review_emergency.csv"))
    print("\n[5/11] Fairness controls...")
    fairness_controls.run(SCORED, TABLES, n_boot=n_boot)
    print("\n[6/11] Clinician validation...")
    clinician_validation.run(Path("docs/manual_review.csv"), Path("docs/manual_review_emergency.csv"), TABLES)
    print("\n[7/11] Factorial effects...")
    factorial_analysis.run(SCORED, TABLES)
    print("\n[8/11] EFI + metric accounting...")
    compute_efi(scored_csv=SCORED, out_dir=TABLES, n_boot=n_boot)

    print("\n[9/11] Figures...")
    from analysis.make_figures import make_all as make_figures
    make_figures(Path("outputs/scored"), Path("outputs/figures"))
    print("\n[10/11] Tables...")
    from analysis.make_tables import make_all as make_tables
    make_tables(Path("outputs/scored"), TABLES)
    print("\n[11/11] Audit report...")
    try:
        from analysis.render_audit import render
        render(Path("analysis/audit_report_template.md"), TABLES, Path("outputs/audit_report.md"))
    except Exception as e:  # noqa: BLE001
        print(f"  audit report skipped: {e}")

    print(f"\n[done] Full analysis complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    main(simulate=args.simulate, n_boot=args.n_boot)
