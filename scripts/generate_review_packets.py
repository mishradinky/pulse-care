"""Generate per-trajectory review packets for manual annotation.

Reads docs/manual_review.csv and outputs one Markdown file per trajectory
to docs/review_packets/, containing:
  - Case metadata
  - All 3 visits' patient-agent dialogue
  - Agent's final assessment at each visit
  - Empty annotation form

Also writes docs/review_packets/CALIBRATION/ for 3 anchor examples.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

ANCHOR_COUNT = 3


def _esc_label(code: str) -> str:
    return {
        "routine": "Routine (no urgent referral)",
        "urgent_outpatient": "Urgent outpatient",
        "ed_referral": "ED referral",
        "emergency_911": "Emergency 911",
    }.get(code, code)


def render_packet(row: dict, traj: dict, cases_dir: Path, is_calibration: bool = False) -> str:
    """Render one trajectory as a Markdown review packet."""
    lines = []
    lines.append(f"# Review Packet: `{row['trajectory_id']}`")
    lines.append("")
    if is_calibration:
        lines.append("> **CALIBRATION EXAMPLE** — Pre-labeled by research team.")
        lines.append("")

    # Metadata
    lines.append("## Case Metadata")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    for k in ["case_id", "category", "simulator", "agent_arch", "agent_model",
               "sim_backbone", "demographic_variant"]:
        lines.append(f"| {k} | `{row.get(k, '')}` |")
    lines.append(f"| Auto normalized score | `{row.get('normalized', '')}` |")
    lines.append(f"| Auto escalation | `{row.get('escalation_level', '')}` |")
    lines.append(f"| Auto leakage | `{row.get('leakage_level', '')}` |")
    lines.append("")

    # Ground truth (from case file)
    gt = traj.get("ground_truth", {})
    if gt:
        lines.append("## Ground Truth (do not use until after labeling)")
        lines.append("<details><summary>Click to reveal</summary>")
        lines.append("")
        lines.append(f"- Correct differential: {', '.join(gt.get('correct_differential', []))}")
        lines.append(f"- Required workup: {', '.join(gt.get('required_workup', []))}")
        lines.append(f"- Correct management: {', '.join(gt.get('correct_management', []))}")
        lines.append(f"- Correct escalation: `{gt.get('correct_escalation', '')}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Visits
    for v in traj.get("transcript", []):
        visit_num = v.get("visit", "?")
        lines.append(f"---")
        lines.append(f"## Visit {visit_num}")
        lines.append("")

        dialogue = v.get("dialogue", [])
        if dialogue:
            lines.append("### Dialogue")
            lines.append("")
            for turn in dialogue:
                speaker = turn.get("speaker", "?").capitalize()
                text = turn.get("text", "")
                lines.append(f"**{speaker}:** {text}")
                lines.append("")

        assessment = v.get("assessment", {})
        if assessment:
            lines.append("### Agent Assessment")
            lines.append("")
            lines.append(f"**Escalation decision:** `{_esc_label(assessment.get('escalation_level', ''))}`")
            lines.append("")

            ddx = assessment.get("differential_diagnosis", [])
            if ddx:
                lines.append("**Differential diagnosis:**")
                for d in ddx[:5]:
                    lines.append(f"- {d}")
                lines.append("")

            workup = assessment.get("workup_ordered", [])
            if workup:
                lines.append("**Workup ordered:**")
                for w in workup[:6]:
                    lines.append(f"- {w}")
                lines.append("")

            mgmt = assessment.get("management_plan", [])
            if mgmt:
                lines.append("**Management plan:**")
                for m in mgmt[:5]:
                    lines.append(f"- {m}")
                lines.append("")

            safety = assessment.get("safety_concerns", [])
            if safety:
                lines.append("**Safety concerns:**")
                for s in safety[:4]:
                    lines.append(f"- {s}")
                lines.append("")

            note = assessment.get("communication_note", "")
            if note:
                lines.append(f"**Communication note:** {note}")
                lines.append("")

            trace = assessment.get("trace", {})
            if trace:
                lines.append("**Team coordination trace:**")
                for role_key, role_label in [
                    ("clinician_contribution", "Clinician"),
                    ("safety_checker_contribution", "Safety checker"),
                    ("pharm_nurse_contribution", "Pharm/nurse"),
                ]:
                    contrib = trace.get(role_key, "")
                    if contrib:
                        lines.append(f"- *{role_label}:* {contrib}")
                lines.append("")

    # Annotation form
    lines.append("---")
    lines.append("## Annotation Form")
    lines.append("")
    lines.append("Fill in the values below and copy them to `docs/manual_review.csv`.")
    lines.append("")
    if is_calibration:
        lines.append("**Pre-filled (calibration):**")
        lines.append("")
        lines.append(f"- `human_pass_fail`: {row.get('human_pass_fail', '_to be filled_')}")
        lines.append(f"- `human_task_score`: {row.get('human_task_score', '_to be filled_')}")
        lines.append(f"- `human_escalation`: {row.get('human_escalation', '_to be filled_')}")
        lines.append(f"- `human_leakage`: {row.get('human_leakage', '_to be filled_')}")
        lines.append(f"- `human_disclosure`: {row.get('human_disclosure', '_to be filled_')}")
        lines.append(f"- `reviewer_notes`: {row.get('reviewer_notes', '')}")
    else:
        lines.append("| Label | Your value |")
        lines.append("|-------|-----------|")
        lines.append("| `human_pass_fail` (pass/fail) | |")
        lines.append("| `human_task_score` (0.0–1.0) | |")
        lines.append("| `human_escalation` (correct/under/over) | |")
        lines.append("| `human_leakage` (0/1/2) | |")
        lines.append("| `human_disclosure` (yes/no) | |")
        lines.append("| `reviewer_notes` | |")
    lines.append("")

    return "\n".join(lines)


def run(
    review_csv: Path,
    trajectories_dir: Path,
    cases_dir: Path,
    out_dir: Path,
) -> None:
    df = pd.read_csv(review_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = out_dir / "CALIBRATION"
    cal_dir.mkdir(exist_ok=True)

    # Pick 3 anchor trajectories (highest/lowest/middle normalized scores from main subset)
    main = df[(df["demographic_variant"] == "baseline") & (df["sim_backbone"] == "claude_haiku_4_5")]
    anchors = set()
    if not main.empty:
        sorted_main = main.sort_values("normalized")
        for idx in [0, len(sorted_main) // 2, len(sorted_main) - 1]:
            anchors.add(sorted_main.iloc[idx]["trajectory_id"])

    ok = 0
    missing = 0
    for _, row in df.iterrows():
        tid = row["trajectory_id"]
        tf = trajectories_dir / f"{tid}.json"
        if not tf.exists():
            missing += 1
            continue

        traj = json.loads(tf.read_text(encoding="utf-8"))
        is_cal = tid in anchors
        packet = render_packet(row.to_dict(), traj, cases_dir, is_calibration=is_cal)

        if is_cal:
            target = cal_dir / f"{tid}.md"
        else:
            target = out_dir / f"{tid}.md"

        target.write_text(packet, encoding="utf-8")
        ok += 1

    print(f"[review_packets] Written {ok} packets ({missing} missing trajectory files) -> {out_dir}")
    print(f"  Calibration packets: {len(anchors)} in {cal_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-csv", default="docs/manual_review.csv")
    ap.add_argument("--trajectories", default="outputs/trajectories")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--out", default="docs/review_packets")
    args = ap.parse_args()

    run(
        review_csv=Path(args.review_csv),
        trajectories_dir=Path(args.trajectories),
        cases_dir=Path(args.cases),
        out_dir=Path(args.out),
    )
