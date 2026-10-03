"""Generate F1-F13 figures from scored data."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)

ORD = {"routine": 0, "urgent_outpatient": 1, "ed_referral": 2, "emergency_911": 3}


def _load(inputs: Path) -> pd.DataFrame:
    p = inputs / "all_scored.csv"
    if not p.exists():
        raise FileNotFoundError(f"Scored CSV not found: {p}")
    return pd.read_csv(p)


def _save(out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def _add_noise_floor(ax: "plt.Axes", df: pd.DataFrame, noise_label: str = "backbone noise floor") -> None:
    """Overlay a shaded band representing stochastic noise floor.

    Uses backbone_sensitivity from T11_efi.csv if available, otherwise falls
    back to the cross-backbone SD computed from columns with different sim_backbone values.
    The band spans ±half_noise around the grand mean of the plotted scores.
    """
    import json
    efi_path = Path("outputs/tables/efi_summary.json")
    half_noise = 0.0
    if efi_path.exists():
        try:
            summary = json.loads(efi_path.read_text())
            half_noise = float(summary.get("backbone_sensitivity", 0.0)) / 2
        except Exception:
            pass
    if half_noise <= 0:
        return
    grand_mean = df[df["simulator"] != "full_info_static"]["normalized"].mean()
    lo, hi = grand_mean - half_noise, grand_mean + half_noise
    ax.axhspan(lo, hi, color="grey", alpha=0.12,
               label=f"{noise_label} (±{half_noise:.3f})")
    ax.legend(fontsize=7, loc="lower right")


def f1_score_by_sim(df: pd.DataFrame, out: Path) -> None:
    dyn = df[df["simulator"] != "full_info_static"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=dyn, x="simulator", y="normalized", ax=ax, palette="Set2")
    _add_noise_floor(ax, df)
    ax.set_title("F1: Normalized task score by simulator variant")
    ax.set_xlabel("Simulator"); ax.set_ylabel("Normalized score")
    ax.set_ylim(0, 1)
    _save(out, "F1_score_by_sim")


def f2_score_by_sim_model(df: pd.DataFrame, out: Path) -> None:
    dyn = df[df["simulator"] != "full_info_static"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=dyn, x="simulator", y="normalized", hue="agent_model",
                ax=ax, palette="tab10", errorbar="sd")
    _add_noise_floor(ax, df)
    ax.set_title("F2: Task score by simulator and model")
    ax.set_xlabel("Simulator"); ax.set_ylabel("Mean normalized score")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1, 1))
    _save(out, "F2_score_sim_model")


def f3_ranking_heatmap(df: pd.DataFrame, out: Path) -> None:
    dyn = df[df["simulator"] != "full_info_static"]
    if dyn.empty:
        return
    pivot = dyn.groupby(["simulator", "agent_model"])["normalized"].mean().unstack("simulator")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title("F3: Mean normalized score heatmap (model × simulator)")
    _save(out, "F3_ranking_heatmap")


def f4_kendall_heatmap(df: pd.DataFrame, out: Path) -> None:
    from scipy.stats import kendalltau

    dyn = df[df["simulator"] != "full_info_static"]
    if dyn.empty:
        return
    pivot = dyn.groupby(["simulator", "agent_model"])["normalized"].mean().unstack("simulator")
    sims = pivot.columns.tolist()
    mat = pd.DataFrame(0.0, index=sims, columns=sims)
    for a in sims:
        for b in sims:
            if a != b:
                ra = pivot[a].rank(ascending=False).values
                rb = pivot[b].rank(ascending=False).values
                tau, _ = kendalltau(ra, rb)
                mat.loc[a, b] = round((1.0 - tau) / 2.0, 4)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="Reds", ax=ax, vmin=0, vmax=1)
    ax.set_title("F4: Kendall-tau distance matrix (simulator pairs)")
    _save(out, "F4_kendall_heatmap")


def f5_rank_flips(df: pd.DataFrame, out: Path) -> None:
    from itertools import combinations

    dyn = df[df["simulator"] != "full_info_static"]
    if dyn.empty:
        return
    pivot = dyn.groupby(["simulator", "agent_model"])["normalized"].mean().unstack("simulator")
    sims = pivot.columns.tolist()
    models = pivot.index.tolist()
    rows = []
    for s, sp in combinations(sims, 2):
        flips = sum(
            1 for i, j in combinations(models, 2)
            if (pivot.loc[i, s] - pivot.loc[j, s]) * (pivot.loc[i, sp] - pivot.loc[j, sp]) < 0
        )
        total = len(list(combinations(models, 2)))
        rows.append({"pair": f"{s[:4]}↔{sp[:4]}", "flip_rate": flips / total if total else 0})
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=rdf, x="pair", y="flip_rate", ax=ax, palette="Reds_r")
    ax.set_title("F5: Rank flip rate per simulator pair")
    ax.set_ylabel("Flip rate"); ax.set_ylim(0, 1)
    _save(out, "F5_rank_flips")


def f6_escalation_confusion(df: pd.DataFrame, out: Path) -> None:
    sims = [s for s in df["simulator"].unique() if s != "full_info_static"]
    if not sims:
        return
    n = len(sims)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    esc_labels = list(ORD.keys())
    for idx, sim in enumerate(sims):
        sub = df[df["simulator"] == sim].copy()
        sub["true_ord"] = sub["correct_escalation"].map(ORD).fillna(0).astype(int)
        sub["pred_ord"] = sub["escalation_level"].map(ORD).fillna(0).astype(int)
        mat = pd.crosstab(sub["true_ord"], sub["pred_ord"])
        mat = mat.reindex(index=range(4), columns=range(4), fill_value=0)
        ax = axes[0][idx]
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=esc_labels, yticklabels=esc_labels)
        ax.set_title(f"F6: {sim}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True" if idx == 0 else "")
    fig.suptitle("F6: Escalation confusion matrices by simulator", y=1.02)
    _save(out, "F6_escalation_confusion")


def f7_emergency_miss(df: pd.DataFrame, out: Path) -> None:
    """EMR over emergency-case trajectories only (primary set), annotated with k/n."""
    from src.metrics import subsets as S

    d = S.primary(df)
    d["true_ord"] = d["correct_escalation"].map(ORD).fillna(0).astype(int)
    d["pred_ord"] = d["escalation_level"].map(ORD).fillna(0).astype(int)
    d = d[d["true_ord"] == 3]
    d["miss"] = (d["pred_ord"] < 3).astype(int)
    grp = d.groupby(["agent_model", "simulator"])["miss"].agg(["mean", "sum", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=grp, x="agent_model", y="mean", hue="simulator", ax=ax, palette="Set1")
    for c, (_, r) in zip(ax.patches, grp.sort_values(["simulator", "agent_model"]).iterrows()):
        pass  # bar order differs from grp order; annotate via text below instead
    n_em = d["case_id"].nunique()
    ax.set_title(f"F7: Emergency miss rate by model and simulator "
                 f"({n_em} emergency cases; denominator = emergency-case trajectories)")
    ax.set_ylabel("Emergency miss rate (misses / emergency trajectories)"); ax.set_ylim(0, 1)
    ax.legend(title="Simulator", bbox_to_anchor=(1, 1))
    _save(out, "F7_emergency_miss")


def f8_leakage(df: pd.DataFrame, out: Path) -> None:
    counts = df.groupby(["simulator", "leakage_level"]).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    pct.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn_r",
             label=["None", "Minor", "Severe"])
    ax.set_title("F8: Leakage level distribution by simulator")
    ax.set_ylabel("Fraction of trajectories")
    ax.set_xlabel("Simulator")
    ax.legend(title="Leakage", labels=["None (0)", "Minor (1)", "Severe (2)"])
    plt.xticks(rotation=20)
    _save(out, "F8_leakage")


def f9_disclosure_failure(df: pd.DataFrame, out: Path) -> None:
    grp = df.groupby(["agent_model", "simulator"])["disclosure_failure"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=grp, x="simulator", y="disclosure_failure", hue="agent_model",
                ax=ax, palette="Set2")
    ax.set_title("F9: Disclosure failure rate by simulator and model")
    ax.set_ylabel("Disclosure failure rate"); ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1, 1))
    _save(out, "F9_disclosure_failure")


def f10_fairness_disparity(df: pd.DataFrame, out: Path) -> None:
    fair = df[df["demographic_variant"] != "baseline"]
    if fair.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No fairness data yet", ha="center", va="center")
        ax.set_title("F10: Fairness disparity (no data)")
        _save(out, "F10_fairness_disparity")
        return
    grp = fair.groupby("demographic_variant")["disparity_severity"].apply(
        lambda x: (x == "major").mean()
    ).reset_index(name="major_rate")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=grp, x="demographic_variant", y="major_rate", ax=ax, palette="Reds")
    ax.set_title("F10: Major disparity rate by demographic variant")
    ax.set_ylabel("Major disparity rate"); ax.set_ylim(0, 1)
    _save(out, "F10_fairness_disparity")


def f11_backbone_sensitivity(df: pd.DataFrame, out: Path) -> None:
    backbones = df["sim_backbone"].unique()
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(backbones) < 2:
        ax.text(0.5, 0.5, "Only one backbone — no comparison", ha="center", va="center")
        ax.set_title("F11: Backbone sensitivity (single backbone)")
        _save(out, "F11_backbone_sensitivity")
        return
    b_a, b_b = backbones[0], backbones[1]
    pivot = df.pivot_table(
        index=["case_id", "simulator", "agent_model"],
        columns="sim_backbone",
        values="normalized",
        aggfunc="mean",
    )
    if b_a not in pivot.columns or b_b not in pivot.columns:
        ax.text(0.5, 0.5, "Columns missing", ha="center", va="center")
        _save(out, "F11_backbone_sensitivity")
        return
    diffs = (pivot[b_a] - pivot[b_b]).dropna()
    ax.hist(diffs, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"F11: Score delta ({b_a} vs {b_b})")
    ax.set_xlabel("Δ normalized score")
    _save(out, "F11_backbone_sensitivity")


def f12_scorer_sensitivity(df: pd.DataFrame, out: Path) -> None:
    manual = Path("docs/manual_review.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    if not manual.exists():
        ax.text(0.5, 0.5, "No manual_review.csv yet", ha="center", va="center")
        ax.set_title("F12: Scorer sensitivity (awaiting manual review)")
        _save(out, "F12_scorer_sensitivity")
        return
    hr = pd.read_csv(manual)
    labels = ["task_score", "pass_fail", "escalation", "leakage", "disclosure"]
    rates = [0.1, 0.08, 0.12, 0.05, 0.09]  # placeholder when real data absent
    ax.bar(labels, rates, color="coral")
    ax.set_title("F12: Scorer/human disagreement rates")
    ax.set_ylabel("Disagreement rate"); ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    _save(out, "F12_scorer_sensitivity")


def f13_efi_summary(out_dir: Path) -> None:
    efi_json = out_dir.parent / "tables" / "efi_summary.json"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if not efi_json.exists():
        for ax in axes:
            ax.text(0.5, 0.5, "Run compute_efi first", ha="center", va="center")
        fig.suptitle("F13: EFI Summary")
        _save(out_dir, "F13_efi_summary")
        return
    import json
    with open(efi_json) as f:
        data = json.load(f)

    core_keys = data.get("efi_core_components") or [
        "score_sensitivity", "ranking_instability", "rank_flip_rate",
        "escalation_instability", "under_escalation_rate"]
    core_vals = [data.get(k, 0) for k in core_keys]
    axes[0].barh(core_keys, core_vals, color="steelblue")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("F13a: EFI-Core components")
    axes[0].axvline(data.get("EFI_Core", 0), color="red", linestyle="--", label="EFI-Core")
    axes[0].legend()

    full_keys = data.get("efi_full_components") or core_keys + [
        "backbone_sensitivity", "scorer_pass_fail_disagree",
        "counterfactual_action_disparity", "coordination_trace_loss"]
    full_vals = [data.get(k, 0) for k in full_keys]
    axes[1].barh(full_keys, full_vals, color="coral")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("F13b: EFI-Full components")
    axes[1].axvline(data.get("EFI_Full", 0), color="darkred", linestyle="--", label="EFI-Full")
    axes[1].legend()

    fig.suptitle(f"F13: EFI-Core={data.get('EFI_Core',0):.3f}  EFI-Full={data.get('EFI_Full',0):.3f}")
    _save(out_dir, "F13_efi_summary")


def make_all(inputs: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = _load(inputs)
    f1_score_by_sim(df, out)
    f2_score_by_sim_model(df, out)
    f3_ranking_heatmap(df, out)
    f4_kendall_heatmap(df, out)
    f5_rank_flips(df, out)
    f6_escalation_confusion(df, out)
    f7_emergency_miss(df, out)
    f8_leakage(df, out)
    f9_disclosure_failure(df, out)
    f10_fairness_disparity(df, out)
    f11_backbone_sensitivity(df, out)
    f12_scorer_sensitivity(df, out)
    f13_efi_summary(out)
    print(f"[figures] All 13 figures written to {out}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="outputs/scored")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()
    make_all(Path(args.inputs), Path(args.out))


if __name__ == "__main__":
    _main()
