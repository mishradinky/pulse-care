"""Parallel sweep over the experiment matrix."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from src.runner import run_trajectory


def _load_cases(cfg: dict, exp_cfg: dict) -> list[dict]:
    cases_dir = Path(exp_cfg["cases_dir"])
    all_case_files = sorted(cases_dir.glob("*.json"))

    subset_cases = cfg.get("cases")
    if subset_cases == "all":
        return [json.loads(f.read_text(encoding="utf-8")) for f in all_case_files]

    if isinstance(subset_cases, list):
        result = []
        for cid in subset_cases:
            p = cases_dir / f"{cid}.json"
            if p.exists():
                result.append(json.loads(p.read_text(encoding="utf-8")))
        return result

    # cases_subset: N (select first N)
    n = cfg.get("cases_subset", len(all_case_files))
    return [json.loads(f.read_text(encoding="utf-8")) for f in all_case_files[:n]]


def _enumerate_runs(
    subset_name: str, subset_cfg: dict, exp_cfg: dict
) -> list[dict[str, Any]]:
    cases = _load_cases(subset_cfg, exp_cfg)
    simulators = subset_cfg["simulators"]
    agents = subset_cfg["agents"]
    models = subset_cfg["agent_models"]
    backbone = subset_cfg["sim_backbone"]

    demographic_variants = subset_cfg.get("demographic_variants", ["baseline"])
    reseeds = subset_cfg.get("reseeds", 1)

    runs = []
    for case in cases:
        for sim in simulators:
            for agent in agents:
                for model in models:
                    for demo in demographic_variants:
                        for seed_off in range(reseeds):
                            runs.append({
                                "case": case,
                                "simulator": sim,
                                "agent": agent,
                                "model": model,
                                "sim_backbone": backbone,
                                "demographic_variant": demo,
                                "seed_offset": seed_off,
                            })
    return runs


# Trajectory files from the first experiment run carry the legacy key `gpt_5` for
# gpt-4o-2024-08-06 (see README "Data notes"). Treat them as completed gpt_4o runs so
# that re-running a subset never re-bills them.
LEGACY_MODEL_ALIASES = {"gpt_4o": ["gpt_4o", "gpt_5"]}


def _output_exists(run: dict[str, Any], output_dir: Path) -> bool:
    """Skip run if any matching trajectory file already exists (legacy aliases included)."""
    for model in LEGACY_MODEL_ALIASES.get(run["model"], [run["model"]]):
        for backbone in LEGACY_MODEL_ALIASES.get(run["sim_backbone"], [run["sim_backbone"]]):
            pattern = "__".join([
                run["case"]["case_id"], run["simulator"], run["agent"], model, backbone,
                run["demographic_variant"], str(run["seed_offset"]),
            ])
            if any(output_dir.glob(f"{pattern}_*.json")):
                return True
    return False


def _execute_run(
    run: dict[str, Any],
    model_cfg: dict,
    exp_cfg: dict,
    output_dir: Path,
    simulate: bool,
) -> dict[str, Any]:
    try:
        out = run_trajectory(
            case=run["case"],
            simulator_variant=run["simulator"],
            agent_architecture=run["agent"],
            model_key=run["model"],
            sim_backbone_key=run["sim_backbone"],
            model_cfg=model_cfg,
            exp_cfg=exp_cfg,
            output_dir=output_dir,
            demographic_variant=run["demographic_variant"],
            seed_offset=run["seed_offset"],
            simulate=simulate,
        )
        return {"status": "ok", "out": str(out)}
    except Exception as e:
        return {"status": "error", "error": str(e), "run": run["case"]["case_id"]}


def run_subset(
    subset_name: str,
    exp_cfg: dict,
    model_cfg: dict,
    parallel: int = 1,
    simulate: bool = False,
    skip_existing: bool = True,
) -> list[dict]:
    subset_cfg = exp_cfg["subsets"][subset_name]
    runs = _enumerate_runs(subset_name, subset_cfg, exp_cfg)
    output_dir = Path(exp_cfg["output_dir"]) / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing:
        pending = [r for r in runs if not _output_exists(r, output_dir)]
        skipped = len(runs) - len(pending)
        if skipped:
            print(f"[batch_driver] Skipping {skipped} already-completed trajectories.")
        runs = pending

    print(f"[batch_driver] Running {len(runs)} trajectories (subset={subset_name}, parallel={parallel})")

    results = []

    if parallel <= 1:
        iterator = tqdm(runs, desc=subset_name) if _HAS_TQDM else runs
        for run in iterator:
            r = _execute_run(run, model_cfg, exp_cfg, output_dir, simulate)
            results.append(r)
            if r["status"] == "error":
                print(f"  ERROR: {r['error']} (case={r.get('run')})")
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {
                ex.submit(_execute_run, run, model_cfg, exp_cfg, output_dir, simulate): run
                for run in runs
            }
            bar = tqdm(total=len(futures), desc=subset_name) if _HAS_TQDM else None
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                if r["status"] == "error":
                    print(f"  ERROR: {r['error']}")
                if bar:
                    bar.update(1)
            if bar:
                bar.close()

    ok = sum(1 for r in results if r["status"] == "ok")
    err = len(results) - ok
    print(f"[batch_driver] Done: {ok} ok, {err} errors.")
    return results


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                    help="subset name from experiment_config.yaml (see `subsets:`)")
    ap.add_argument("--config", default="configs/experiment_config.yaml")
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--simulate", action="store_true", help="Use mock LLM client")
    ap.add_argument("--no-skip", action="store_true", help="Re-run even if output exists")
    ap.add_argument("--dry-run", action="store_true", help="Only count pending trajectories")
    ap.add_argument("--models", nargs="*", default=None,
                    help="only run these agent models of the subset (e.g. when one provider is down)")
    args = ap.parse_args()

    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(args.model_cfg).read_text(encoding="utf-8"))
    if args.subset not in exp_cfg["subsets"]:
        raise SystemExit(f"Unknown subset '{args.subset}'. Available: {', '.join(exp_cfg['subsets'])}")
    if args.models:
        exp_cfg["subsets"][args.subset]["agent_models"] = [
            m for m in exp_cfg["subsets"][args.subset]["agent_models"] if m in args.models]
        print(f"[batch_driver] restricted to agent models: {exp_cfg['subsets'][args.subset]['agent_models']}")
    for m in exp_cfg["subsets"][args.subset]["agent_models"]:
        if m not in model_cfg["agents"]:
            raise SystemExit(f"agent model '{m}' is not defined in {args.model_cfg}")
    if args.dry_run:
        subset_cfg = exp_cfg["subsets"][args.subset]
        runs = _enumerate_runs(args.subset, subset_cfg, exp_cfg)
        out = Path(exp_cfg["output_dir"]) / "trajectories"
        pending = [r for r in runs if not _output_exists(r, out)]
        print(f"[batch_driver] subset={args.subset}: {len(runs)} total, "
              f"{len(runs) - len(pending)} done, {len(pending)} pending")
        return

    run_subset(
        subset_name=args.subset,
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        parallel=args.parallel,
        simulate=args.simulate,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    _main()
