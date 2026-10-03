"""
End-to-end PULSE-Care simulation runner.

Runs a mini experiment (static + 5-case main subset) using the MockClient,
scores all trajectories, computes EFI metrics, generates figures and tables,
and renders the audit report.

Usage:
    python run_simulation.py [--cases N] [--parallel P]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from src.batch_driver import run_subset
from src.metrics.compute_task_scores import score_all
from src.metrics.compute_efi import run as run_efi
from analysis.make_tables import make_all as make_tables
from analysis.make_figures import make_all as make_figures
from analysis.render_audit import render as render_audit


def _patch_config_for_sim(exp_cfg: dict, n_cases: int) -> dict:
    """Shrink the config for a quick simulation run."""
    import copy
    cfg = copy.deepcopy(exp_cfg)
    # Use just claude_haiku_4_5 for speed
    cfg["subsets"]["static"]["agent_models"] = ["claude_haiku_4_5"]
    cfg["subsets"]["main"]["agent_models"] = ["claude_haiku_4_5"]
    cfg["subsets"]["main"]["agents"] = ["single"]
    # Limit cases
    all_cases = sorted(Path(cfg["cases_dir"]).glob("*.json"))
    selected = [p.stem for p in all_cases[:n_cases]]
    cfg["subsets"]["static"]["cases"] = selected
    cfg["subsets"]["main"]["cases"] = selected
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="PULSE-Care end-to-end simulation")
    ap.add_argument("--cases", type=int, default=5, help="Number of cases to use (default 5)")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--exp-cfg", default="configs/experiment_config.yaml")
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    args = ap.parse_args()

    exp_cfg = yaml.safe_load(Path(args.exp_cfg).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(args.model_cfg).read_text(encoding="utf-8"))
    exp_cfg = _patch_config_for_sim(exp_cfg, args.cases)

    print("=" * 60)
    print("PULSE-Care End-to-End Simulation")
    print(f"  Cases: {args.cases}   Parallel: {args.parallel}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Run static baseline
    print("\n[Step 1/6] Running static baseline trajectories...")
    run_subset("static", exp_cfg, model_cfg, parallel=args.parallel, simulate=True)

    # Step 2: Run main experiment
    print("\n[Step 2/6] Running main experiment (4 simulators × cases)...")
    run_subset("main", exp_cfg, model_cfg, parallel=args.parallel, simulate=True)

    # Step 3: Score all trajectories
    print("\n[Step 3/6] Scoring trajectories...")
    scorer_cfg = model_cfg["patient_simulator_backbones"]["claude_haiku_4_5"]
    traj_dir = Path(exp_cfg["output_dir"]) / "trajectories"
    scored_dir = Path(exp_cfg["output_dir"]) / "scored"
    df = score_all(traj_dir, Path(exp_cfg["cases_dir"]), scored_dir, scorer_cfg, simulate=True)
    print(f"  Scored {len(df)} trajectories.")

    # Step 4: Compute EFI metrics
    print("\n[Step 4/6] Computing EFI metrics...")
    tables_dir = Path(exp_cfg["output_dir"]) / "tables"
    scored_csv = scored_dir / "all_scored.csv"
    efi = run_efi(scored_csv, tables_dir)
    print(f"  EFI-Core={efi['EFI_Core']:.4f}  EFI-Full={efi['EFI_Full']:.4f}")

    # Step 5: Generate tables and figures
    print("\n[Step 5/6] Generating tables and figures...")
    make_tables(scored_dir, tables_dir)
    figures_dir = Path(exp_cfg["output_dir"]) / "figures"
    make_figures(scored_dir, figures_dir)

    # Step 6: Render audit report
    print("\n[Step 6/6] Rendering audit report...")
    render_audit(
        Path("analysis/audit_report_template.md"),
        tables_dir,
        Path(exp_cfg["output_dir"]) / "audit_report.md",
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Simulation complete in {elapsed:.1f}s")
    print(f"  Trajectories : {len(list(traj_dir.glob('*.json')))}")
    print(f"  EFI-Core     : {efi['EFI_Core']:.4f}")
    print(f"  EFI-Full     : {efi['EFI_Full']:.4f}")
    print(f"  Figures      : {len(list(figures_dir.glob('*.png')))}/13")
    print(f"  Tables       : {len(list(tables_dir.glob('*.csv')))}/11")
    print(f"  Audit report : {exp_cfg['output_dir']}/audit_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
