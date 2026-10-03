"""Score trajectories with the 7-item rubric via the scorer LLM (or mock)."""
from __future__ import annotations

import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

from src.model_clients import make_client
from src.output_parser import parse_with_retry


def score_trajectory(
    traj: dict,
    ground_truth: dict,
    scorer_cfg: dict,
    simulate: bool = False,
) -> dict:
    client = make_client(scorer_cfg["provider"], scorer_cfg, simulate=simulate)
    schema = json.loads(Path("schemas/scorer_output.json").read_text(encoding="utf-8"))["task_score"]
    prompt = (
        Path("prompts/scorer_task_score.txt").read_text(encoding="utf-8")
        .replace("{GROUND_TRUTH_JSON}", json.dumps(ground_truth))
        .replace("{TRAJECTORY_JSON}", json.dumps(traj))
    )
    obj, _ = parse_with_retry(
        client,
        [{"role": "user", "content": prompt}],
        system=None,
        schema=schema,
        temperature=0.0,
        max_tokens=1024,
        retries=2,
    )
    return obj


def score_leakage(
    traj: dict,
    ground_truth: dict,
    scorer_cfg: dict,
    simulate: bool = False,
) -> dict:
    client = make_client(scorer_cfg["provider"], scorer_cfg, simulate=simulate)
    diagnosis_terms = ground_truth.get("correct_differential", [])
    prompt = (
        Path("prompts/scorer_leakage.txt").read_text(encoding="utf-8")
        .replace("{DIAGNOSIS_TERMS_LIST}", json.dumps(diagnosis_terms))
        .replace("{TRAJECTORY_JSON}", json.dumps(traj))
    )
    obj, _ = parse_with_retry(
        client,
        [{"role": "user", "content": prompt}],
        system=None,
        schema=None,
        temperature=0.0,
        max_tokens=512,
        retries=2,
    )
    if not isinstance(obj, dict):
        obj = {"leakage_level": 0, "offending_utterances": [], "justification": "parse error"}
    obj.setdefault("leakage_level", 0)
    return obj


_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "is", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "or", "and", "but",
    "any", "all", "no", "not", "with", "by", "as", "if", "it", "its",
}


def compute_disclosure_failure(traj: dict, critical_facts: list[str]) -> int:
    all_clinician_text = " ".join(
        t["text"].lower()
        for v in traj["transcript"]
        for t in v.get("dialogue", [])
        if t.get("speaker") == "clinician"
    )
    for cf in critical_facts:
        tokens = [
            tok for tok in cf.lower().split()
            if len(tok) >= 4 and tok not in _STOPWORDS
        ]
        if tokens and any(tok in all_clinician_text for tok in tokens):
            return 0
    return 1


def _score_one(tf: Path, cases_dir: Path, scorer_cfg: dict, simulate: bool) -> dict | None:
    try:
        traj = json.loads(tf.read_text(encoding="utf-8"))
        case_path = cases_dir / f"{traj['case_id']}.json"
        if not case_path.exists():
            return None
        case = json.loads(case_path.read_text(encoding="utf-8"))
        gt = traj.get("ground_truth", case["ground_truth"])

        task = score_trajectory(traj, gt, scorer_cfg, simulate=simulate)
        leak = score_leakage(traj, gt, scorer_cfg, simulate=simulate)
        disc = compute_disclosure_failure(traj, case["critical_facts"])

        last_assessment = {}
        for v in reversed(traj["transcript"]):
            if "assessment" in v:
                last_assessment = v["assessment"]
                break

        return {
            "trajectory_id": tf.stem,
            "case_id": traj["case_id"],
            "category": traj.get("category", case["category"]),
            "simulator": traj["simulator"],
            "agent_arch": traj["agent_arch"],
            "agent_model": traj["agent_model"],
            "sim_backbone": traj["sim_backbone"],
            "demographic_variant": traj.get("demographic_variant", "baseline"),
            "seed_offset": traj.get("seed_offset", 0),
            "agent_model_id": traj.get("agent_model_id", ""),
            "normalized": task.get("normalized", 0.0),
            "raw_total": task.get("raw_total", 0),
            "escalation_level": last_assessment.get("escalation_level", "routine"),
            "correct_escalation": gt.get("correct_escalation", "routine"),
            "leakage_level": leak.get("leakage_level", 0),
            "disclosure_failure": disc,
            "disparity_severity": "none",
        }
    except Exception as e:
        print(f"  ERROR scoring {tf.name}: {e}")
        return None


def score_all(
    trajectories_dir: Path,
    cases_dir: Path,
    output_dir: Path,
    scorer_cfg: dict,
    simulate: bool = False,
    parallel: int = 4,
) -> pd.DataFrame:
    traj_files = sorted(trajectories_dir.glob("*.json"))
    print(f"[scorer] Scoring {len(traj_files)} trajectories (parallel={parallel})...")

    rows = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_score_one, tf, cases_dir, scorer_cfg, simulate): tf for tf in traj_files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result:
                rows.append(result)
            if done % 50 == 0 or done == len(traj_files):
                print(f"  [{done}/{len(traj_files)}] scored so far: {len(rows)} ok")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "all_scored.csv"
    df.to_csv(out_path, index=False)
    print(f"[scorer] Saved {len(df)} rows -> {out_path}")
    return df


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="outputs/trajectories")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--out", default="outputs/scored")
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    model_cfg = yaml.safe_load(Path(args.model_cfg).read_text(encoding="utf-8"))
    scorer_cfg = model_cfg["patient_simulator_backbones"]["claude_haiku_4_5"]

    score_all(
        trajectories_dir=Path(args.trajectories),
        cases_dir=Path(args.cases),
        output_dir=Path(args.out),
        scorer_cfg=scorer_cfg,
        simulate=args.simulate,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    _main()
