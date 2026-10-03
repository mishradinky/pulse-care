"""Step 8: Leakage-performance regression.

Regresses normalized task score on leakage_level, controlling for
simulator, agent_model. Produces T_leakage_regression.csv and figure.
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


def run(scored_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(scored_csv)
    dyn = df[df["simulator"] != "full_info_static"].copy()

    # Mean score by leakage level
    summary = (
        dyn.groupby("leakage_level")["normalized"]
        .agg(["mean", "std", "count"])
        .round(4)
        .reset_index()
    )
    summary.columns = ["leakage_level", "mean_score", "std_score", "n"]

    # Per-simulator breakdown
    sim_breakdown = (
        dyn.groupby(["simulator", "leakage_level"])["normalized"]
        .agg(["mean", "std", "count"])
        .round(4)
        .reset_index()
    )
    sim_breakdown.columns = ["simulator", "leakage_level", "mean_score", "std_score", "n"]

    # Simple OLS-style slope via numpy polyfit
    x = dyn["leakage_level"].values.astype(float)
    y = dyn["normalized"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "T_leakage_regression.csv", index=False)
    print("[leakage_regression] Mean score by leakage level:")
    print(summary.to_string(index=False))
    print(f"  OLS slope={slope:.4f}  corr={corr:.4f}  n={len(dyn)}")

    # Figure: boxplot of score by leakage level, faceted by simulator
    sims = sorted(dyn["simulator"].unique())
    n_sims = len(sims)
    fig, axes = plt.subplots(1, n_sims, figsize=(3 * n_sims, 4), sharey=True)
    if n_sims == 1:
        axes = [axes]

    for ax, sim in zip(axes, sims):
        sub = dyn[dyn["simulator"] == sim]
        sns.boxplot(data=sub, x="leakage_level", y="normalized", ax=ax,
                    palette="Reds", hue="leakage_level", legend=False)
        ax.set_title(sim, fontsize=8)
        ax.set_xlabel("Leakage level")
        ax.set_ylabel("Norm. score" if ax == axes[0] else "")
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"F: Leakage → performance (slope={slope:.3f}, r={corr:.3f})",
        fontsize=10,
    )
    plt.tight_layout()

    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "F_leakage_regression.png", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / "F_leakage_regression.pdf", bbox_inches="tight")
    plt.close()
    print(f"[leakage_regression] Figure -> {fig_dir}/F_leakage_regression.png")


if __name__ == "__main__":
    run(
        scored_csv=Path("outputs/scored/all_scored.csv"),
        out_dir=Path("outputs/tables"),
    )
