"""Targeted (adversarial) manual-review sample, complementing the stratified 110.

Selects trajectories where automated labels matter most and are most likely to be wrong:
  A. high score variance      : (case, model, arch) groups with the largest max-min across
                                 simulators -> all 4 trajectories of the top groups
  B. rank-flip cells           : trajectories of the model pairs / simulators that flip
  C. escalation disagreement   : groups whose escalation differs across simulators
  D. leakage-suspected         : leakage_level >= 1
  E. CAD-positive pairs        : baseline + swap trajectories whose escalation differs
  F. scorer disagreement       : pass/fail differs between primary and alt scorer
  G. random baseline           : uniform from the primary set

Each stratum contributes up to --per-stratum rows (default 10); rows already in
docs/manual_review.csv are excluded. Output docs/manual_review_targeted.csv with the same
label columns as the main sheet plus `stratum` and `human_unsafe_reassurance` (yes/no).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.metrics import subsets as S
from src.metrics.compute_escalation_metrics import ORD
from src.metrics.compute_fairness import fairness_pairs
from src.metrics.compute_rank_flips import rank_flip_details

LABELS = ["human_pass_fail", "human_task_score", "human_escalation", "human_leakage",
          "human_disclosure", "human_disparity", "human_unsafe_reassurance", "reviewer_notes"]


def run(scored_csv: Path, review_csv: Path, tables_dir: Path, out_path: Path,
        per_stratum: int = 10, seed: int = 20260515) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(scored_csv)
    prim = S.primary(df)
    already = set(pd.read_csv(review_csv)["trajectory_id"]) if review_csv.exists() else set()
    picks: list[tuple[str, str]] = []

    def take(ids, stratum):
        ids = [i for i in ids if i not in already and i not in {p[0] for p in picks}]
        for i in ids[:per_stratum]:
            picks.append((i, stratum))

    grp = ["case_id", "agent_model", "agent_arch"]
    # A. high score variance
    rng_ = prim.groupby(grp)["normalized"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    ids = []
    for key in rng_.index[: max(1, per_stratum // 4)]:
        m = (prim[grp] == pd.Series(key, index=grp)).all(axis=1)
        ids += prim.loc[m, "trajectory_id"].tolist()
    take(ids, "A_high_score_variance")

    # B. rank-flip cells
    det = rank_flip_details(prim)
    flips = det[det["nominal_flip"] == 1]
    ids = []
    for r in flips.itertuples():
        m = prim["simulator"].isin([r.sim_a, r.sim_b]) & prim["agent_model"].isin([r.model_i, r.model_j])
        ids += prim.loc[m, "trajectory_id"].tolist()
    rng.shuffle(ids)
    take(ids, "B_rank_flip_cells")

    # C. escalation disagreement across simulators
    p = prim.copy(); p["pred_ord"] = p["escalation_level"].map(ORD)
    nun = p.groupby(grp)["pred_ord"].nunique()
    ids = []
    for key in nun[nun > 1].sample(frac=1, random_state=seed).index[: max(1, per_stratum // 4)]:
        m = (p[grp] == pd.Series(key, index=grp)).all(axis=1)
        ids += p.loc[m, "trajectory_id"].tolist()
    take(ids, "C_escalation_disagreement")

    # D. leakage-suspected
    ids = prim[prim["leakage_level"] >= 1].sample(frac=1, random_state=seed)["trajectory_id"].tolist()
    take(ids, "D_leakage_suspected")

    # E. CAD-positive pairs (both members)
    pairs = fairness_pairs(S.fairness(df))
    ids = []
    if not pairs.empty:
        pos = pairs[pairs["escalation_differs"] == 1].sample(frac=1, random_state=seed)
        for r in pos.itertuples():
            ids += [r.trajectory_id_base, r.trajectory_id_var]
    take(ids, "E_cad_positive_pair")

    # F. scorer disagreement
    det_p = tables_dir / "T_scorer_agreement_detail.csv"
    ids = []
    if det_p.exists():
        sd = pd.read_csv(det_p)
        alt_cols = [c for c in sd.columns if c.endswith("_normalized") and c not in ("primary_normalized", "consensus_normalized")]
        if "alt_normalized" in sd.columns:
            alt_cols.append("alt_normalized")
        dis = pd.Series(False, index=sd.index)
        for c in alt_cols:
            dis |= (sd["primary_normalized"] >= 0.5) != (sd[c] >= 0.5)
        ids = sd.loc[dis, "trajectory_id"].tolist()
        rng.shuffle(ids)
    take(ids, "F_scorer_disagreement")

    # G. random
    ids = prim.sample(frac=1, random_state=seed + 1)["trajectory_id"].tolist()
    take(ids, "G_random")

    sel = pd.DataFrame(picks, columns=["trajectory_id", "stratum"])
    out = sel.merge(df, on="trajectory_id", how="left")
    out["file_path"] = out["trajectory_id"].apply(lambda t: f"outputs/trajectories/{t}.json")
    for c in LABELS:
        out[c] = ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[targeted_review] {len(out)} rows -> {out_path}")
    print(out["stratum"].value_counts().to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--review-csv", default="docs/manual_review.csv")
    ap.add_argument("--tables", default="outputs/tables")
    ap.add_argument("--out", default="docs/manual_review_targeted.csv")
    ap.add_argument("--per-stratum", type=int, default=10)
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.review_csv), Path(a.tables), Path(a.out), per_stratum=a.per_stratum)
