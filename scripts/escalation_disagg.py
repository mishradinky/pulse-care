"""Step 7: Disaggregate escalation by direction (under vs over) per simulator variant.

Produces T_escalation_disagg.csv and F_escalation_disagg.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.9)

ORD = {"routine": 0, "urgent_outpatient": 1, "ed_referral": 2, "emergency_911": 3}
SIM_ORDER = ["cooperative", "sparse", "verbose", "low_health_literacy", "full_info_static"]


def run(scored_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(scored_csv)
    df["true_ord"] = df["correct_escalation"].map(ORD).fillna(0).astype(int)
    df["pred_ord"] = df["escalation_level"].map(ORD).fillna(0).astype(int)
    df["gap"] = df["pred_ord"] - df["true_ord"]

    # Per simulator × model: under / exact / over rates + EMR
    rows = []
    # Primary analysis set only (baseline demographics, seed 0, primary backbone,
    # primary models) so rates match T11 / T_metric_accounting. EMR denominator is
    # the number of emergency-case trajectories in the cell, reported with counts.
    from src.metrics import subsets as S
    df = S.primary(df, include_static=True)
    for (sim, model), g in df.groupby(["simulator", "agent_model"]):
        n = len(g)
        under = (g["gap"] < 0).mean()
        exact = (g["gap"] == 0).mean()
        over = (g["gap"] > 0).mean()
        em = g[g["true_ord"] == 3]
        n_em = len(em)
        k_miss = int((em["pred_ord"] < 3).sum())
        rows.append({
            "simulator": sim,
            "agent_model": model,
            "n": n,
            "under_rate": round(under, 4),
            "exact_rate": round(exact, 4),
            "over_rate": round(over, 4),
            "n_emergency": n_em,
            "emergency_misses": k_miss,
            "emr": round(k_miss / n_em, 4) if n_em else float("nan"),
        })
    result = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "T_escalation_disagg.csv", index=False)
    print(result.to_string(index=False))

    # Figure: stacked bar — under / exact / over per (simulator, model)
    pivot_under = result.pivot(index="simulator", columns="agent_model", values="under_rate")
    pivot_exact = result.pivot(index="simulator", columns="agent_model", values="exact_rate")
    pivot_over  = result.pivot(index="simulator", columns="agent_model", values="over_rate")

    sims = [s for s in SIM_ORDER if s in result["simulator"].unique()]
    models = result["agent_model"].unique().tolist()

    x = np.arange(len(sims))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    colors_under = ["#d73027", "#f46d43", "#fdae61"]
    colors_exact = ["#1a9850", "#66bd63", "#a6d96a"]
    colors_over  = ["#4575b4", "#74add1", "#abd9e9"]

    for i, model in enumerate(models):
        offset = (i - 1) * width
        u = [pivot_under.loc[s, model] if s in pivot_under.index else 0 for s in sims]
        e = [pivot_exact.loc[s, model] if s in pivot_exact.index else 0 for s in sims]
        o = [pivot_over.loc[s, model] if s in pivot_over.index else 0 for s in sims]
        b1 = ax.bar(x + offset, u, width, label=f"{model} under", color=colors_under[i])
        b2 = ax.bar(x + offset, e, width, bottom=u, label=f"{model} exact", color=colors_exact[i])
        b3 = ax.bar(x + offset, o, width, bottom=[ui+ei for ui,ei in zip(u,e)],
                    label=f"{model} over", color=colors_over[i])

    ax.set_xticks(x)
    ax.set_xticklabels(sims, rotation=15)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("F: Escalation direction (under / exact / over) by simulator and model")
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=7)

    plt.tight_layout()
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "F_escalation_disagg.png", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / "F_escalation_disagg.pdf", bbox_inches="tight")
    plt.close()
    print(f"[escalation_disagg] Table -> {out_dir}/T_escalation_disagg.csv")
    print(f"[escalation_disagg] Figure -> {fig_dir}/F_escalation_disagg.png")


if __name__ == "__main__":
    run(
        scored_csv=Path("outputs/scored/all_scored.csv"),
        out_dir=Path("outputs/tables"),
    )
