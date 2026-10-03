"""Patch only the disclosure_failure column in all_scored.csv using Fix 4 logic.

Reads trajectory JSONs and recomputes disclosure_failure with the stopword
filter — no LLM calls. All other columns (including disparity_severity) are
left untouched.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics.compute_task_scores import compute_disclosure_failure

SCORED_CSV = Path("outputs/scored/all_scored.csv")
TRAJ_DIR = Path("outputs/trajectories")
CASES_DIR = Path("cases")


def main():
    df = pd.read_csv(SCORED_CSV)
    print(f"Loaded {len(df)} rows from {SCORED_CSV}")
    print(f"disparity_severity BEFORE patch:")
    print(df["disparity_severity"].value_counts().to_string())
    print(f"disclosure_failure BEFORE patch (mean): {df['disclosure_failure'].mean():.4f}")

    updated = 0
    missing_traj = 0
    missing_case = 0

    for i, row in df.iterrows():
        tid = row["trajectory_id"]
        tf = TRAJ_DIR / f"{tid}.json"
        if not tf.exists():
            missing_traj += 1
            continue
        case_path = CASES_DIR / f"{row['case_id']}.json"
        if not case_path.exists():
            missing_case += 1
            continue
        traj = json.loads(tf.read_text(encoding="utf-8"))
        case = json.loads(case_path.read_text(encoding="utf-8"))
        df.at[i, "disclosure_failure"] = compute_disclosure_failure(
            traj, case["critical_facts"]
        )
        updated += 1

    df.to_csv(SCORED_CSV, index=False)
    print(f"\nPatched {updated} rows  (missing traj: {missing_traj}, missing case: {missing_case})")
    print(f"disclosure_failure AFTER patch (mean): {df['disclosure_failure'].mean():.4f}")
    print(f"disparity_severity AFTER patch:")
    print(df["disparity_severity"].value_counts().to_string())


if __name__ == "__main__":
    main()
