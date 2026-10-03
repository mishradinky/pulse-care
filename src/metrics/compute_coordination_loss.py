"""Coordination Trace Loss: fraction of team trajectories missing role contributions."""
import json
from pathlib import Path


def coordination_trace_loss(team_records: list[dict]) -> float:
    if not team_records:
        return 0.0
    required_keys = [
        "clinician_contribution",
        "safety_checker_contribution",
        "pharm_nurse_contribution",
    ]
    MIN_CHARS = 50  # below this threshold a contribution is considered degenerate
    bad = 0
    for rec in team_records:
        for v in rec.get("transcript", []):
            tr = v.get("assessment", {}).get("trace", {})
            if not tr or any(len(tr.get(k, "")) < MIN_CHARS for k in required_keys):
                bad += 1
                break
    return bad / len(team_records)


def coordination_trace_loss_from_df(scored_dir: Path) -> float:
    """Load team trajectories from scored dir and compute CTL."""
    import glob

    team_files = list((scored_dir.parent / "trajectories").glob("*__team__*.json"))
    if not team_files:
        return 0.0
    records = [json.loads(f.read_text(encoding="utf-8")) for f in team_files]
    return coordination_trace_loss(records)
