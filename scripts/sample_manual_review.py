"""Stratified sample 110 trajectories for manual review.

Stratification variables: category, simulator, agent_model (per experiment_config.yaml).
Uses proportional allocation with a minimum of 1 per cell.
Outputs: docs/manual_review.csv with trajectory_id, file_path, and scoring columns.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

TARGET_N = 110
SEED = 20260515
STRATIFY_BY = ["category", "simulator", "agent_model"]
MIN_FAIRNESS = 20   # at least 20 fairness-subset (non-baseline demographic variant) rows
MIN_BACKBONE = 10   # at least 10 backbone-subset (non-haiku sim_backbone) rows


def sample_manual_review(
    scored_csv: Path,
    trajectories_dir: Path,
    out_path: Path,
    target_n: int = TARGET_N,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = random.Random(seed)
    df = pd.read_csv(scored_csv)

    # ── Fairness sub-sample (non-baseline demographic variant, cooperative sim) ──
    fairness_pool = df[
        (df["simulator"] != "full_info_static") &
        (df["demographic_variant"] != "baseline")
    ].copy()
    n_fairness = min(MIN_FAIRNESS, len(fairness_pool))
    fairness_sample = fairness_pool.sample(n=n_fairness, random_state=rng.randint(0, 2**31)) if n_fairness else pd.DataFrame()

    # ── Backbone sub-sample (sim_backbone != claude_haiku_4_5) ──
    backbone_pool = df[
        (df["sim_backbone"] != "claude_haiku_4_5") &
        (df["demographic_variant"] == "baseline") &
        (df["simulator"] != "full_info_static")
    ].copy()
    n_backbone = min(MIN_BACKBONE, len(backbone_pool))
    backbone_sample = backbone_pool.sample(n=n_backbone, random_state=rng.randint(0, 2**31)) if n_backbone else pd.DataFrame()

    # ── Main proportional sample (dynamic, baseline, haiku backbone) ──
    already_sampled = set(fairness_sample["trajectory_id"].tolist() if not fairness_sample.empty else []) | \
                      set(backbone_sample["trajectory_id"].tolist() if not backbone_sample.empty else [])
    n_main = target_n - n_fairness - n_backbone

    dyn = df[
        (df["simulator"] != "full_info_static") &
        (df["demographic_variant"] == "baseline") &
        (df["sim_backbone"] == "claude_haiku_4_5") &
        (~df["trajectory_id"].isin(already_sampled))
    ].copy()

    strat_counts = dyn.groupby(STRATIFY_BY).size().reset_index(name="n_total")
    total = strat_counts["n_total"].sum()
    strat_counts["alloc"] = ((strat_counts["n_total"] / total) * n_main).round().astype(int)
    strat_counts["alloc"] = strat_counts["alloc"].clip(lower=1)

    diff = strat_counts["alloc"].sum() - n_main
    if diff > 0:
        idx = strat_counts["alloc"].nlargest(diff).index
        strat_counts.loc[idx, "alloc"] -= 1
    elif diff < 0:
        idx = strat_counts["alloc"].nlargest(-diff).index
        strat_counts.loc[idx, "alloc"] += 1

    sampled_rows = []
    for _, row in strat_counts.iterrows():
        mask = (dyn[STRATIFY_BY[0]] == row[STRATIFY_BY[0]])
        for col in STRATIFY_BY[1:]:
            mask = mask & (dyn[col] == row[col])
        pool = dyn[mask]
        n = min(row["alloc"], len(pool))
        if n > 0:
            chosen = pool.sample(n=n, random_state=rng.randint(0, 2**31))
            sampled_rows.append(chosen)

    # Combine all three sub-samples
    parts = [p for p in [fairness_sample, backbone_sample] + sampled_rows if not (isinstance(p, pd.DataFrame) and p.empty)]

    sample = pd.concat(parts, ignore_index=True)

    # Add trajectory file path
    def find_traj_path(traj_id: str) -> str:
        p = trajectories_dir / f"{traj_id}.json"
        return str(p) if p.exists() else ""

    sample["file_path"] = sample["trajectory_id"].apply(find_traj_path)

    # Add manual review columns (empty for reviewers to fill)
    for col in ["human_pass_fail", "human_task_score", "human_escalation", "human_leakage",
                "human_disclosure", "reviewer_notes"]:
        sample[col] = ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"[manual_review] Sampled {len(sample)} trajectories -> {out_path}")
    print(f"  Distribution by simulator:\n{sample['simulator'].value_counts().to_string()}")
    print(f"  Distribution by category:\n{sample['category'].value_counts().to_string()}")
    print(f"  Distribution by model:\n{sample['agent_model'].value_counts().to_string()}")
    return sample


if __name__ == "__main__":
    sample_manual_review(
        scored_csv=Path("outputs/scored/all_scored.csv"),
        trajectories_dir=Path("outputs/trajectories"),
        out_path=Path("docs/manual_review.csv"),
    )
