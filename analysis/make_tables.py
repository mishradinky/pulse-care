"""Generate T1-T11 summary tables from scored data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.metrics.compute_escalation_metrics import ORD


def _load(inputs: Path) -> pd.DataFrame:
    p = inputs / "all_scored.csv"
    if not p.exists():
        raise FileNotFoundError(f"Scored CSV not found at {p}")
    return pd.read_csv(p)


def _dyn(df: pd.DataFrame) -> pd.DataFrame:
    """Primary analysis set, dynamic simulators only (T2-T5, T7, T8).

    Restricting to the primary set (24 original cases, 3 models, Haiku simulator backbone,
    baseline demographics, seed 0, the 4 named variants) keeps these tables identical to the
    metrics in T11 and excludes reseeds, swaps, other backbones and factorial cells.
    """
    from src.metrics import subsets as S
    return S.primary(df)


def t1_case_summary(df: pd.DataFrame, out: Path) -> None:
    """T1: case distribution by category."""
    t = (
        df.drop_duplicates("case_id")
        .groupby("category")["case_id"]
        .count()
        .reset_index(name="n_cases")
    )
    t.to_csv(out / "T1_case_summary.csv", index=False)


def t2_score_by_simulator(df: pd.DataFrame, out: Path) -> None:
    """T2: mean normalized score by (agent_model, simulator)."""
    dyn = _dyn(df)
    t = (
        dyn.groupby(["agent_model", "simulator"])["normalized"]
        .agg(["mean", "std", "count"])
        .round(4)
        .reset_index()
    )
    t.to_csv(out / "T2_score_by_simulator.csv", index=False)


def t3_ranking_by_simulator(df: pd.DataFrame, out: Path) -> None:
    """T3: agent model rank per simulator."""
    dyn = _dyn(df)
    pivot = dyn.groupby(["simulator", "agent_model"])["normalized"].mean().unstack("simulator")
    ranked = pivot.rank(ascending=False).astype(int)
    ranked.to_csv(out / "T3_ranking_by_simulator.csv")


def t4_kendall_matrix(df: pd.DataFrame, out: Path) -> None:
    """T4: pairwise Kendall-tau distance matrix."""
    from itertools import combinations
    from scipy.stats import kendalltau
    import numpy as np

    dyn = _dyn(df)
    pivot = dyn.groupby(["simulator", "agent_model"])["normalized"].mean().unstack("simulator")
    sims = pivot.columns.tolist()
    mat = pd.DataFrame(index=sims, columns=sims, dtype=float)
    for i in sims:
        for j in sims:
            if i == j:
                mat.loc[i, j] = 0.0
            else:
                ri = pivot[i].rank(ascending=False).values
                rj = pivot[j].rank(ascending=False).values
                tau, _ = kendalltau(ri, rj)
                mat.loc[i, j] = round((1.0 - tau) / 2.0, 4)
    mat.to_csv(out / "T4_kendall_matrix.csv")


def t5_rank_flips(df: pd.DataFrame, out: Path) -> None:
    """T5: rank flip counts per simulator pair."""
    from itertools import combinations

    dyn = _dyn(df)
    pivot = dyn.groupby(["simulator", "agent_model"])["normalized"].mean().unstack("simulator")
    sims = pivot.columns.tolist()
    models = pivot.index.tolist()
    rows = []
    for s, sp in combinations(sims, 2):
        flips = 0
        for i, j in combinations(models, 2):
            d1 = pivot.loc[i, s] - pivot.loc[j, s]
            d2 = pivot.loc[i, sp] - pivot.loc[j, sp]
            if d1 * d2 < 0:
                flips += 1
        total = len(list(combinations(models, 2)))
        rows.append({"sim_a": s, "sim_b": sp, "flips": flips, "total_pairs": total,
                     "flip_rate": round(flips / total, 4) if total else 0.0})
    pd.DataFrame(rows).to_csv(out / "T5_rank_flips.csv", index=False)


def t6_escalation_gaps(df: pd.DataFrame, out: Path) -> None:
    """T6: escalation accuracy by simulator and model (primary set incl. static).

    EMR denominator = emergency-case trajectories in the cell (counts reported).
    Rates are written as percentages to avoid decimal/percent ambiguity in prose.
    """
    from src.metrics import subsets as S

    d = S.primary(df, include_static=True)
    d["true_ord"] = d["correct_escalation"].map(ORD).fillna(0).astype(int)
    d["pred_ord"] = d["escalation_level"].map(ORD).fillna(0).astype(int)
    d["gap"] = d["pred_ord"] - d["true_ord"]
    rows = []
    for (m, s), g in d.groupby(["agent_model", "simulator"]):
        em = g[g["true_ord"] == 3]
        k = int((em["pred_ord"] < 3).sum())
        rows.append({
            "agent_model": m, "simulator": s, "n": len(g),
            "under_pct": round(100 * (g["gap"] <= -1).mean(), 1),
            "exact_pct": round(100 * (g["gap"] == 0).mean(), 1),
            "over_pct": round(100 * (g["gap"] >= 1).mean(), 1),
            "emergency_misses": f"{k}/{len(em)}",
            "emr_pct": round(100 * k / len(em), 1) if len(em) else "",
        })
    pd.DataFrame(rows).to_csv(out / "T6_escalation_gaps.csv", index=False)

    # by simulator, models pooled
    rows = []
    for s, g in d.groupby("simulator"):
        em = g[g["true_ord"] == 3]
        k = int((em["pred_ord"] < 3).sum())
        rows.append({
            "simulator": s, "n": len(g),
            "under_pct": round(100 * (g["gap"] <= -1).mean(), 1),
            "exact_pct": round(100 * (g["gap"] == 0).mean(), 1),
            "over_pct": round(100 * (g["gap"] >= 1).mean(), 1),
            "emergency_misses": f"{k}/{len(em)}",
            "emr_pct": round(100 * k / len(em), 1) if len(em) else "",
        })
    pd.DataFrame(rows).to_csv(out / "T_escalation_by_variant.csv", index=False)


def t7_leakage(df: pd.DataFrame, out: Path) -> None:
    """T7: leakage rates by simulator."""
    from src.metrics import subsets as S
    t = (
        S.primary(df, include_static=True).groupby("simulator")["leakage_level"]
        .agg(
            none_rate=lambda x: (x == 0).mean(),
            minor_rate=lambda x: (x == 1).mean(),
            severe_rate=lambda x: (x == 2).mean(),
        )
        .round(4)
        .reset_index()
    )
    t.to_csv(out / "T7_leakage.csv", index=False)


def t8_disclosure_failure(df: pd.DataFrame, out: Path) -> None:
    """T8: disclosure failure rate by (model, simulator)."""
    from src.metrics import subsets as S
    t = (
        S.primary(df, include_static=True).groupby(["agent_model", "simulator"])["disclosure_failure"]
        .mean()
        .round(4)
        .reset_index(name="disclosure_failure_rate")
    )
    t.to_csv(out / "T8_disclosure_failure.csv", index=False)


def t9_fairness(df: pd.DataFrame, out: Path) -> None:
    """T9: CAD per demographic variant and model, on matched (baseline, swap) pairs.

    cad_escalation = escalation differs (deterministic), cad_judged_major = LLM judge.
    Denominator is the number of matched pairs, not the number of cases.
    """
    from src.metrics import subsets as S
    from src.metrics.compute_fairness import fairness_pairs

    pairs = fairness_pairs(S.fairness(df))
    if pairs.empty:
        pd.DataFrame(columns=["group", "level", "n_pairs", "cad_escalation", "cad_judged_major"]).to_csv(
            out / "T9_fairness.csv", index=False)
        return
    rows = []
    for group in ["demographic_variant", "agent_model", "agent_arch"]:
        for level, g in pairs.groupby(group):
            rows.append({"group": group, "level": level, "n_pairs": len(g),
                         "cad_escalation": g["escalation_differs"].mean(),
                         "cad_judged_major": (g["disparity_severity"].astype(str) == "major").mean()
                         if "disparity_severity" in g else float("nan"),
                         "mean_score_delta": g["score_delta"].mean() if "score_delta" in g else float("nan")})
    rows.append({"group": "ALL", "level": "ALL", "n_pairs": len(pairs),
                 "cad_escalation": pairs["escalation_differs"].mean(),
                 "cad_judged_major": (pairs["disparity_severity"].astype(str) == "major").mean()
                 if "disparity_severity" in pairs else float("nan"),
                 "mean_score_delta": pairs["score_delta"].mean() if "score_delta" in pairs else float("nan")})
    pd.DataFrame(rows).round(4).to_csv(out / "T9_fairness.csv", index=False)


def t10_backbone_sensitivity(df: pd.DataFrame, out: Path) -> None:
    """T10: per-cell |delta| between the primary simulator backbone and EACH alternative."""
    from src.metrics import subsets as S

    bb = S.backbone(df)
    if bb.empty or bb["sim_backbone"].nunique() < 2:
        pd.DataFrame(columns=["case_id", "simulator", "agent_model", "agent_arch", "alt_backbone", "delta"]).to_csv(
            out / "T10_backbone_sensitivity.csv", index=False)
        return
    pivot = bb.pivot_table(index=["case_id", "simulator", "agent_model", "agent_arch"],
                           columns="sim_backbone", values="normalized", aggfunc="mean")
    rows = []
    for alt in [b for b in pivot.columns if b != S.PRIMARY_BACKBONE]:
        d = (pivot[S.PRIMARY_BACKBONE] - pivot[alt]).dropna()
        for idx, val in d.items():
            rows.append({"case_id": idx[0], "simulator": idx[1], "agent_model": idx[2],
                         "agent_arch": idx[3], "alt_backbone": alt, "delta": abs(val),
                         "signed_delta_primary_minus_alt": val})
    pd.DataFrame(rows).round(4).to_csv(out / "T10_backbone_sensitivity.csv", index=False)


def t11_efi(out: Path) -> None:
    """T11: EFI summary (read from compute_efi output)."""
    p = out / "T11_efi.csv"
    if not p.exists():
        efi_src = out / "T11_efi.csv"
    # Already produced by compute_efi.py; just confirm it exists
    if p.exists():
        print(f"[tables] T11 already exists at {p}")


def make_all(inputs: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = _load(inputs)
    t1_case_summary(df, out)
    t2_score_by_simulator(df, out)
    t3_ranking_by_simulator(df, out)
    t4_kendall_matrix(df, out)
    t5_rank_flips(df, out)
    t6_escalation_gaps(df, out)
    t7_leakage(df, out)
    t8_disclosure_failure(df, out)
    t9_fairness(df, out)
    t10_backbone_sensitivity(df, out)
    t11_efi(out)
    print(f"[tables] All tables written to {out}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="outputs/scored")
    ap.add_argument("--out", default="outputs/tables")
    args = ap.parse_args()
    make_all(Path(args.inputs), Path(args.out))


if __name__ == "__main__":
    _main()
