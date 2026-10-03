"""Export PULSE-Care results to a single PDF report and consolidated CSV.

Outputs:
  outputs/results_summary.pdf  — multi-page formatted report (tables + figures)
  outputs/results_summary.csv  — flat metric table, one row per metric

Run from the pulse-care/ directory:
  python scripts/export_results.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
TABLES  = Path("outputs/tables")
FIGURES = Path("outputs/figures")
OUT_PDF = Path("outputs/results_summary.pdf")
OUT_CSV = Path("outputs/results_summary.csv")

# ── Styling ───────────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = 8.27, 11.69          # A4 inches
HDR_COLOR = "#1a3a5c"                 # dark blue — header row
ALT_COLOR = "#f0f4f8"                 # light blue-grey — alternating rows
HL_COLOR  = "#fff9c4"                 # light yellow — EFI composite rows
WHITE     = "#ffffff"
FONT_MAIN = "DejaVu Sans"

FIGURE_ORDER = [
    "F1_score_by_sim",
    "F2_score_sim_model",
    "F3_ranking_heatmap",
    "F4_kendall_heatmap",
    "F5_rank_flips",
    "F6_escalation_confusion",
    "F7_emergency_miss",
    "F8_leakage",
    "F9_disclosure_failure",
    "F10_fairness_disparity",
    "F11_backbone_sensitivity",
    "F12_scorer_sensitivity",
    "F13_efi_summary",
    "F_manipulation_check",
    "F_escalation_disagg",
    "F_leakage_inflation",
    "F_leakage_regression",
]

FIGURE_TITLES = {
    "F1_score_by_sim":          "F1 — Task Score by Simulator Variant",
    "F2_score_sim_model":       "F2 — Task Score by Simulator × Model",
    "F3_ranking_heatmap":       "F3 — Model Ranking Heatmap",
    "F4_kendall_heatmap":       "F4 — Pairwise Kendall-τ Between Simulator Rankings",
    "F5_rank_flips":            "F5 — Rank-Flip Count per Model Pair",
    "F6_escalation_confusion":  "F6 — Escalation Confusion Matrix (per Model × Simulator)",
    "F7_emergency_miss":        "F7 — Emergency Miss Rate per Model × Simulator",
    "F8_leakage":               "F8 — Severe Leakage Rate per Simulator",
    "F9_disclosure_failure":    "F9 — Disclosure Failure Rate per Model × Simulator",
    "F10_fairness_disparity":   "F10 — Counterfactual Action Disparity by Demographic Swap",
    "F11_backbone_sensitivity": "F11 — Backbone Sensitivity: Score Δ (Haiku vs. GPT-5 Simulator)",
    "F12_scorer_sensitivity":   "F12 — Cross-Scorer Disagreement by Label Type",
    "F13_efi_summary":          "F13 — Evaluation Fragility Index: All Metrics with 95% CIs",
    "F_manipulation_check":     "Supplementary — Manipulation Check: Simulator Variant Properties",
    "F_escalation_disagg":      "Supplementary — Escalation Disaggregated by Simulator × Model",
    "F_leakage_inflation":      "Supplementary — Score vs. Leakage Level (OLS Regression)",
    "F_leakage_regression":     "Supplementary — Leakage Regression Residuals",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    """Format float as percentage string."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%"


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _new_page() -> tuple:
    """Return (fig, ax) for a fresh A4 page, axis invisible."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return fig, ax


def _page_header(fig, title: str, subtitle: str = "") -> None:
    """Draw a coloured header band at the top of the page."""
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (0.03, 0.93), 0.94, 0.055,
            transform=fig.transFigure,
            boxstyle="round,pad=0.005",
            facecolor=HDR_COLOR, edgecolor="none", zorder=1,
        )
    )
    fig.text(0.50, 0.960, title, transform=fig.transFigure,
             ha="center", va="center", fontsize=13, fontweight="bold",
             color="white", zorder=2)
    if subtitle:
        fig.text(0.50, 0.940, subtitle, transform=fig.transFigure,
                 ha="center", va="center", fontsize=8, color="#ccddee", zorder=2)


def _render_table(
    fig,
    df: pd.DataFrame,
    bbox: list,                        # [left, bottom, width, height] in fig coords
    col_widths: list[float] | None = None,
    highlight_rows: set[int] | None = None,
    row_height: float = 0.045,
) -> None:
    """
    Draw df as a formatted table inside the given bbox.
    Row 0 is the header (dark blue, white text).
    Data rows alternate white / ALT_COLOR.
    highlight_rows indices (0-based in data, not counting header) get HL_COLOR.
    """
    ax = fig.add_axes(bbox)
    ax.set_axis_off()

    n_rows, n_cols = df.shape
    col_labels = list(df.columns)

    # Build cell text and colours
    cell_text = df.values.tolist()
    cell_colours = []
    for r in range(n_rows):
        if highlight_rows and r in highlight_rows:
            row_c = [HL_COLOR] * n_cols
        elif r % 2 == 0:
            row_c = [WHITE] * n_cols
        else:
            row_c = [ALT_COLOR] * n_cols
        cell_colours.append(row_c)

    col_w = col_widths or [1.0 / n_cols] * n_cols

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        colWidths=col_w,
        loc="upper center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Style header
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor(HDR_COLOR)
        cell.set_text_props(color="white", fontweight="bold", ha="center")
        cell.set_height(row_height)

    # Style data cells
    for r in range(n_rows):
        for c in range(n_cols):
            cell = tbl[r + 1, c]
            cell.set_height(row_height)
            if highlight_rows and r in highlight_rows:
                cell.set_text_props(fontweight="bold")


# ── Cover page ────────────────────────────────────────────────────────────────

def _cover_page(pdf: PdfPages, efi_core: float, efi_full: float,
                ci_core: tuple, ci_full: tuple) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Top colour band
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (0, 0.82), 1.0, 0.18,
            transform=fig.transFigure,
            boxstyle="square,pad=0",
            facecolor=HDR_COLOR, edgecolor="none",
        )
    )

    fig.text(0.50, 0.930, "PULSE-Care", ha="center", va="center",
             fontsize=32, fontweight="bold", color="white",
             transform=fig.transFigure)
    fig.text(0.50, 0.880, "Patient-Simulator Sensitivity in Clinical AI Agent Evaluation",
             ha="center", va="center", fontsize=14, color="#aaccee",
             transform=fig.transFigure)
    fig.text(0.50, 0.850, f"Results Summary Report  ·  {date.today().isoformat()}",
             ha="center", va="center", fontsize=10, color="#ccddee",
             transform=fig.transFigure)

    # EFI callout boxes
    for x, val, ci, label, sub in [
        (0.27, efi_core, ci_core, "EFI-Core", "Mean of 5 primary metrics"),
        (0.73, efi_full, ci_full, "EFI-Full", "Mean of 9 metrics incl. fairness"),
    ]:
        fig.patches.append(
            mpatches.FancyBboxPatch(
                (x - 0.18, 0.60), 0.36, 0.18,
                transform=fig.transFigure,
                boxstyle="round,pad=0.01",
                facecolor="#e8f0fe", edgecolor=HDR_COLOR, linewidth=1.5,
            )
        )
        fig.text(x, 0.760, label, ha="center", va="center",
                 fontsize=13, fontweight="bold", color=HDR_COLOR,
                 transform=fig.transFigure)
        fig.text(x, 0.710, f"{val:.4f}", ha="center", va="center",
                 fontsize=28, fontweight="bold", color="#c0392b",
                 transform=fig.transFigure)
        fig.text(x, 0.660, f"95% CI  [{ci[0]:.3f}, {ci[1]:.3f}]",
                 ha="center", va="center", fontsize=9, color="#555555",
                 transform=fig.transFigure)
        fig.text(x, 0.627, sub, ha="center", va="center",
                 fontsize=8, color="#777777", transform=fig.transFigure)

    # Interpretation guide
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (0.08, 0.42), 0.84, 0.14,
            transform=fig.transFigure,
            boxstyle="round,pad=0.01",
            facecolor="#f8f9fa", edgecolor="#cccccc", linewidth=1,
        )
    )
    fig.text(0.50, 0.545, "EFI Interpretation Guide",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color=HDR_COLOR, transform=fig.transFigure)
    guide_lines = [
        "0.00 – 0.10   Very low fragility — conclusions stable across simulator variants",
        "0.10 – 0.20   Low fragility — minor sensitivity to simulator choice",
        "0.20 – 0.30   Moderate fragility — this experiment's range; simulator choice matters",
        "0.30 – 0.50   High fragility — rankings and safety conclusions frequently reverse",
        "0.50+          Very high fragility — benchmark conclusions are unreliable",
    ]
    for i, line in enumerate(guide_lines):
        fig.text(0.12, 0.515 - i * 0.020, line, ha="left", va="center",
                 fontsize=8.5, color="#333333", transform=fig.transFigure,
                 fontfamily="monospace")

    # Contents list
    fig.text(0.50, 0.390, "Report Contents",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color=HDR_COLOR, transform=fig.transFigure)
    contents = [
        "Page 2   Evaluation Fragility Index (EFI) — all 15 metrics with 95% CIs",
        "Page 3   Task scores — mean ± SD per model × simulator variant",
        "Page 4   Escalation by simulator — under / exact / over / emergency-miss rates",
        "Page 5   Escalation disaggregated — by simulator × model",
        "Page 6   Patient simulator manipulation check — words/turn, FK grade, facts/turn",
        "Page 7   Cross-scorer agreement — κ, Pearson r, mean |Δ|, disagree rate",
        "Page 8   Fairness — counterfactual action disparity by demographic variant",
        "Page 9   Backbone sensitivity — score |Δ| by simulator × model",
        "Pages 10–26   Figures F1–F13 + 4 supplementary plots",
    ]
    for i, line in enumerate(contents):
        fig.text(0.12, 0.365 - i * 0.025, line, ha="left", va="center",
                 fontsize=9, color="#333333", transform=fig.transFigure)

    # Footer
    fig.text(0.50, 0.04,
             "PULSE-Care  ·  Evaluation Fragility Index  ·  github.com/mishradinky/pulse-care",
             ha="center", va="center", fontsize=8, color="#999999",
             transform=fig.transFigure)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Table pages ───────────────────────────────────────────────────────────────

def _table_page(pdf: PdfPages, df: pd.DataFrame, title: str,
                subtitle: str = "", highlight_rows: set[int] | None = None,
                col_widths: list[float] | None = None) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _page_header(fig, title, subtitle)

    n_rows = len(df)
    row_h = min(0.040, 0.80 / (n_rows + 1))
    table_h = row_h * (n_rows + 1)
    top = 0.90
    bottom = top - table_h - 0.02
    bbox = [0.04, bottom, 0.92, table_h]

    _render_table(fig, df, bbox=bbox, col_widths=col_widths,
                  highlight_rows=highlight_rows, row_height=row_h)

    # Footer
    fig.text(0.50, 0.03, "PULSE-Care Results Summary", ha="center", va="center",
             fontsize=7.5, color="#aaaaaa", transform=fig.transFigure)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _figure_page(pdf: PdfPages, png_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax_img = fig.add_axes([0.02, 0.06, 0.96, 0.86])
    ax_img.set_axis_off()

    try:
        img = mpimg.imread(str(png_path))
        ax_img.imshow(img, aspect="auto")
    except Exception as e:
        ax_img.text(0.5, 0.5, f"[Could not load image: {e}]",
                    ha="center", va="center", transform=ax_img.transAxes)

    fig.text(0.50, 0.960, title, ha="center", va="center",
             fontsize=11, fontweight="bold", color=HDR_COLOR,
             transform=fig.transFigure)
    fig.text(0.50, 0.030, "PULSE-Care Results Summary", ha="center", va="center",
             fontsize=7.5, color="#aaaaaa", transform=fig.transFigure)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Data preparation ──────────────────────────────────────────────────────────

def _prep_efi(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T11_efi.csv")
    label_map = {
        "score_sensitivity":             "Score Sensitivity",
        "ranking_instability":           "Ranking Instability",
        "rank_flip_rate":                "Rank-Flip Rate",
        "escalation_instability":        "Escalation Instability",
        "under_escalation_rate":         "Under-Escalation Rate",
        "over_escalation_rate":          "Over-Escalation Rate",
        "emergency_miss_rate":           "Emergency Miss Rate",
        "severe_leakage_rate":           "Severe Leakage Rate",
        "disclosure_failure_rate":       "Disclosure Failure Rate",
        "counterfactual_action_disparity": "Counterfactual Action Disparity",
        "backbone_sensitivity":          "Backbone Sensitivity",
        "scorer_pass_fail_disagree":     "Scorer Pass/Fail Disagreement",
        "coordination_trace_loss":       "Coordination Trace Loss",
        "EFI_Core":                      "EFI-Core  ★",
        "EFI_Full":                      "EFI-Full  ★",
    }
    df["Metric"] = df["metric"].map(label_map).fillna(df["metric"])
    df["Value"] = df["value"].apply(lambda v: f"{v:.4f}")
    df["95% CI"] = df.apply(
        lambda r: f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        if pd.notna(r.get("ci_lo")) else "—", axis=1
    )
    df["% Value"] = df["value"].apply(lambda v: f"{v*100:.1f}%")
    out = df[["Metric", "Value", "% Value", "95% CI"]].copy()
    # which rows are EFI composites (0-indexed in data)
    highlight = {i for i, m in enumerate(df["metric"]) if m in ("EFI_Core", "EFI_Full")}
    return out, highlight


def _prep_scores(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T2_score_by_simulator.csv")
    model_map = {
        "claude_haiku_4_5":   "Haiku 4.5",
        "gpt_4o":             "GPT-4o",
        "mistral_small_3_2":  "Mistral 3.2",
    }
    sim_map = {
        "cooperative":         "Cooperative",
        "sparse":              "Sparse",
        "verbose":             "Verbose",
        "low_health_literacy": "Low-Health-Lit.",
        "full_info_static":    "Static (oracle)",
    }
    df["model_label"] = df["agent_model"].map(model_map).fillna(df["agent_model"])
    df["sim_label"]   = df["simulator"].map(sim_map).fillna(df["simulator"])
    df["score_str"] = df.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}  (n={int(r['count'])})", axis=1
    )
    pivot = df.pivot(index="model_label", columns="sim_label", values="score_str").reset_index()
    pivot = pivot.rename(columns={"model_label": "Model"})
    return pivot


def _prep_escalation_variant(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T_escalation_by_variant.csv")
    sim_map = {
        "cooperative":         "Cooperative",
        "sparse":              "Sparse",
        "verbose":             "Verbose",
        "low_health_literacy": "Low-Health-Lit.",
    }
    df["Simulator"] = df["simulator"].map(sim_map).fillna(df["simulator"])
    df["N"] = df["n"].apply(lambda v: str(int(v)))
    df["Under"] = df["under_rate"].apply(_pct)
    df["Exact"] = df["exact_rate"].apply(_pct)
    df["Over"] = df["over_rate"].apply(_pct)
    df["EMR"] = df["emr"].apply(_pct)
    return df[["Simulator", "N", "Under", "Exact", "Over", "EMR"]]


def _prep_escalation_disagg(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T_escalation_disagg.csv")
    sim_map = {
        "cooperative":         "Cooperative",
        "sparse":              "Sparse",
        "verbose":             "Verbose",
        "low_health_literacy": "Low-HL",
        "full_info_static":    "Static",
    }
    model_map = {
        "claude_haiku_4_5":   "Haiku",
        "gpt_4o":             "GPT-4o",
        "mistral_small_3_2":  "Mistral",
    }
    df["Simulator"] = df["simulator"].map(sim_map).fillna(df["simulator"])
    df["Model"] = df["agent_model"].map(model_map).fillna(df["agent_model"])
    df["N"] = df["n"].apply(lambda v: str(int(v)))
    df["Under"] = df["under_rate"].apply(_pct)
    df["Exact"] = df["exact_rate"].apply(_pct)
    df["Over"] = df["over_rate"].apply(_pct)
    df["EMR"] = df["emr"].apply(_pct)
    return df[["Simulator", "Model", "N", "Under", "Exact", "Over", "EMR"]]


def _prep_manipulation(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T_manipulation_check.csv")
    sim_map = {
        "cooperative":         "Cooperative",
        "sparse":              "Sparse",
        "verbose":             "Verbose",
        "low_health_literacy": "Low-Health-Lit.",
    }
    df["Simulator"] = df["simulator"].map(sim_map).fillna(df["simulator"])
    df["Words/Turn"] = df.apply(
        lambda r: f"{r['words_per_turn_mean']:.1f} ± {r['words_per_turn_std']:.1f}", axis=1
    )
    df["Facts/Turn"] = df.apply(
        lambda r: f"{r['facts_per_turn_mean']:.2f} ± {r['facts_per_turn_std']:.2f}", axis=1
    )
    df["FK Grade"] = df.apply(
        lambda r: f"{r['fk_grade_mean']:.1f} ± {r['fk_grade_std']:.1f}", axis=1
    )
    return df[["Simulator", "Words/Turn", "Facts/Turn", "FK Grade"]]


def _prep_scorer(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T_scorer_agreement.csv")
    row = df.iloc[0]
    data = {
        "Statistic": [
            "N (trajectories reviewed)",
            "Primary scorer",
            "Alternative scorer",
            "Cohen's κ (pass/fail)",
            "Pearson r (task score)",
            "Mean |Δ| task score",
            "Pass/fail disagreement rate",
        ],
        "Value": [
            str(int(row["n_trajectories"])),
            str(row.get("primary_scorer", "claude_haiku_4_5")),
            str(row.get("alt_scorer", "gpt_4o")),
            f"{row['kappa_pass_fail']:.4f}",
            f"{row['score_correlation']:.4f}",
            f"{row['mean_abs_delta']:.4f}",
            f"{row['scorer_disagree_rate'] * 100:.1f}%",
        ],
        "Interpretation": [
            "Sample for cross-scorer comparison",
            "Claude Haiku 4.5",
            "GPT-5",
            "Slight agreement (κ < 0.20 = minimal)",
            "Moderate correlation",
            "On 0–1 normalized scale",
            "1 in 5 trajectories get different pass/fail",
        ],
    }
    return pd.DataFrame(data)


def _prep_fairness(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T9_fairness.csv")
    variant_map = {
        "minority_swap": "Minority swap (racial/ethnic identity)",
        "ses_swap":      "SES swap (socioeconomic status)",
    }
    df["Demographic Variant"] = df["demographic_variant"].map(variant_map).fillna(df["demographic_variant"])
    df["Major Disparity Rate"] = df["major_disparity_rate"].apply(_pct)
    df["Interpretation"] = df["major_disparity_rate"].apply(
        lambda v: "High — agent actions differ significantly across variants" if v > 0.4
        else "Moderate" if v > 0.2 else "Low"
    )
    return df[["Demographic Variant", "Major Disparity Rate", "Interpretation"]]


def _prep_backbone(tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "T10_backbone_sensitivity.csv")
    df = df.dropna(subset=["delta"])
    model_map = {
        "claude_haiku_4_5":   "Haiku 4.5",
        "gpt_4o":             "GPT-4o",
        "mistral_small_3_2":  "Mistral 3.2",
    }
    sim_map = {
        "cooperative":         "Cooperative",
        "sparse":              "Sparse",
        "verbose":             "Verbose",
        "low_health_literacy": "Low-HL",
    }
    df["Model"] = df["agent_model"].map(model_map).fillna(df["agent_model"])
    df["Simulator"] = df["simulator"].map(sim_map).fillna(df["simulator"])
    agg = (
        df.groupby(["Simulator", "Model"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["Mean |Δ|"] = agg["mean"].apply(lambda v: f"{v:.4f}")
    agg["SD"] = agg["std"].apply(lambda v: f"{v:.4f}")
    agg["N"] = agg["count"].astype(int).astype(str)
    return agg[["Simulator", "Model", "N", "Mean |Δ|", "SD"]]


# ── Consolidated CSV ──────────────────────────────────────────────────────────

def build_consolidated_csv(tables: Path, out_path: Path) -> int:
    rows = []

    # ── EFI metrics ──
    efi = pd.read_csv(tables / "T11_efi.csv")
    label_map = {
        "score_sensitivity":             "Score Sensitivity",
        "ranking_instability":           "Ranking Instability",
        "rank_flip_rate":                "Rank-Flip Rate",
        "escalation_instability":        "Escalation Instability",
        "under_escalation_rate":         "Under-Escalation Rate",
        "over_escalation_rate":          "Over-Escalation Rate",
        "emergency_miss_rate":           "Emergency Miss Rate",
        "severe_leakage_rate":           "Severe Leakage Rate",
        "disclosure_failure_rate":       "Disclosure Failure Rate",
        "counterfactual_action_disparity": "Counterfactual Action Disparity",
        "backbone_sensitivity":          "Backbone Sensitivity",
        "scorer_pass_fail_disagree":     "Scorer Pass/Fail Disagreement",
        "coordination_trace_loss":       "Coordination Trace Loss",
        "EFI_Core":                      "EFI-Core (composite)",
        "EFI_Full":                      "EFI-Full (composite)",
    }
    notes_map = {
        "score_sensitivity":   "Mean (max−min) score across simulator variants per (case, model)",
        "ranking_instability": "Mean normalized Kendall-τ distance across simulator pairs",
        "rank_flip_rate":      "Fraction of model-pair comparisons that reverse direction",
        "escalation_instability": "Fraction of (case, model) pairs where escalation changes across variants",
        "under_escalation_rate":  "Fraction where agent chose a level lower than ground truth (dangerous)",
        "over_escalation_rate":   "Fraction where agent chose a level higher than ground truth",
        "emergency_miss_rate":    "Fraction of true-emergency cases not escalated to emergency_911",
        "severe_leakage_rate":    "Fraction where simulator revealed diagnosis (level=2)",
        "disclosure_failure_rate": "Fraction where agent never elicited critical facts",
        "counterfactual_action_disparity": "Fraction of demographic-swapped pairs with different escalation",
        "backbone_sensitivity":   "Mean |Δ score| between Haiku and GPT-5 simulator backbones",
        "scorer_pass_fail_disagree": "Pass/fail disagreement between Haiku and GPT-5 scorers (n=60)",
        "coordination_trace_loss":  "Fraction of team visits with a role contribution <50 chars",
        "EFI_Core": "Mean of 5 primary metrics: sensitivity, rank_instab, rank_flip, esc_instab, under_esc",
        "EFI_Full": "Mean of 9 metrics: EFI-Core + backbone_sens + scorer_disagree + CAD + trace_loss",
    }
    for _, r in efi.iterrows():
        rows.append({
            "section":    "EFI",
            "metric":     label_map.get(r["metric"], r["metric"]),
            "value":      round(float(r["value"]), 6),
            "ci_lo":      round(float(r["ci_lo"]), 6) if pd.notna(r.get("ci_lo")) else "",
            "ci_hi":      round(float(r["ci_hi"]), 6) if pd.notna(r.get("ci_hi")) else "",
            "pct_value":  round(float(r["value"]) * 100, 1),
            "notes":      notes_map.get(r["metric"], ""),
        })

    # ── Task scores ──
    scores = pd.read_csv(tables / "T2_score_by_simulator.csv")
    for _, r in scores.iterrows():
        rows.append({
            "section":   "Task Scores",
            "metric":    f"{r['agent_model']}  ×  {r['simulator']}",
            "value":     round(float(r["mean"]), 4),
            "ci_lo":     "",
            "ci_hi":     "",
            "pct_value": round(float(r["mean"]) * 100, 1),
            "notes":     f"mean±SD={r['mean']:.3f}±{r['std']:.3f}  n={int(r['count'])}",
        })

    # ── Escalation by variant ──
    esc = pd.read_csv(tables / "T_escalation_by_variant.csv")
    for _, r in esc.iterrows():
        for col, label in [("under_rate", "Under-escalation"), ("over_rate", "Over-escalation"),
                            ("emr", "Emergency miss rate"), ("exact_rate", "Exact escalation")]:
            rows.append({
                "section":   "Escalation",
                "metric":    f"{r['simulator']}  —  {label}",
                "value":     round(float(r[col]), 4),
                "ci_lo":     "",
                "ci_hi":     "",
                "pct_value": round(float(r[col]) * 100, 1),
                "notes":     f"n={int(r['n'])}",
            })

    # ── Manipulation check ──
    mc = pd.read_csv(tables / "T_manipulation_check.csv")
    for _, r in mc.iterrows():
        rows.append({
            "section":   "Manipulation Check",
            "metric":    f"{r['simulator']}  —  words/turn (mean)",
            "value":     round(float(r["words_per_turn_mean"]), 2),
            "ci_lo":     "", "ci_hi":  "",
            "pct_value": "",
            "notes":     f"SD={r['words_per_turn_std']:.2f}",
        })
        rows.append({
            "section":   "Manipulation Check",
            "metric":    f"{r['simulator']}  —  FK grade (mean)",
            "value":     round(float(r["fk_grade_mean"]), 2),
            "ci_lo":     "", "ci_hi":  "",
            "pct_value": "",
            "notes":     f"SD={r['fk_grade_std']:.2f}",
        })

    # ── Scorer agreement ──
    sa = pd.read_csv(tables / "T_scorer_agreement.csv").iloc[0]
    for metric, val, note in [
        ("Kappa pass/fail",             sa["kappa_pass_fail"],      "Slight agreement"),
        ("Pearson r (score)",           sa["score_correlation"],    "Moderate"),
        ("Mean |Δ| task score",         sa["mean_abs_delta"],       "On 0-1 scale"),
        ("Pass/fail disagreement rate", sa["scorer_disagree_rate"], "20% of trajectories"),
    ]:
        rows.append({
            "section":   "Scorer Agreement",
            "metric":    metric,
            "value":     round(float(val), 4),
            "ci_lo":     "", "ci_hi":  "",
            "pct_value": round(float(val) * 100, 1) if metric == "Pass/fail disagreement rate" else "",
            "notes":     note,
        })

    # ── Fairness ──
    fair = pd.read_csv(tables / "T9_fairness.csv")
    for _, r in fair.iterrows():
        rows.append({
            "section":   "Fairness",
            "metric":    f"{r['demographic_variant']}  —  major disparity rate",
            "value":     round(float(r["major_disparity_rate"]), 4),
            "ci_lo":     "", "ci_hi":  "",
            "pct_value": round(float(r["major_disparity_rate"]) * 100, 1),
            "notes":     "Fraction of (case,model) pairs with major action disparity",
        })

    out_df = pd.DataFrame(rows,
        columns=["section", "metric", "value", "ci_lo", "ci_hi", "pct_value", "notes"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return len(out_df)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(tables: Path = TABLES, figures: Path = FIGURES,
        out_pdf: Path = OUT_PDF, out_csv: Path = OUT_CSV) -> None:

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Build consolidated CSV first
    n_csv = build_consolidated_csv(tables, out_csv)
    print(f"[export] Written {out_csv}  ({n_csv} rows)")

    # Read EFI for cover page
    efi_df = pd.read_csv(tables / "T11_efi.csv")
    efi_idx = {r["metric"]: r for _, r in efi_df.iterrows()}
    efi_core = float(efi_idx["EFI_Core"]["value"])
    efi_full = float(efi_idx["EFI_Full"]["value"])
    ci_core = (float(efi_idx["EFI_Core"]["ci_lo"]), float(efi_idx["EFI_Core"]["ci_hi"]))
    ci_full = (float(efi_idx["EFI_Full"]["ci_lo"]), float(efi_idx["EFI_Full"]["ci_hi"]))

    n_pages = 0
    with PdfPages(str(out_pdf)) as pdf:

        # Page 1 — Cover
        _cover_page(pdf, efi_core, efi_full, ci_core, ci_full)
        n_pages += 1

        # Page 2 — EFI metrics table
        efi_tbl, hl = _prep_efi(tables)
        _table_page(pdf, efi_tbl,
                    title="Evaluation Fragility Index (EFI) — All Metrics",
                    subtitle="95% bootstrap CIs  ·  n=1000 resamples, case-level",
                    highlight_rows=hl,
                    col_widths=[0.40, 0.15, 0.12, 0.33])
        n_pages += 1

        # Page 3 — Task scores pivot
        scores_tbl = _prep_scores(tables)
        _table_page(pdf, scores_tbl,
                    title="Task Scores — Mean ± SD per Model × Simulator",
                    subtitle="Normalized score 0–1  ·  pass threshold = 0.50",
                    col_widths=None)
        n_pages += 1

        # Page 4 — Escalation by variant
        esc_tbl = _prep_escalation_variant(tables)
        _table_page(pdf, esc_tbl,
                    title="Escalation by Simulator Variant",
                    subtitle="EMR = Emergency Miss Rate  ·  Under/Over = ≥1 ordinal level from ground truth",
                    col_widths=[0.22, 0.08, 0.16, 0.16, 0.16, 0.14])
        n_pages += 1

        # Page 5 — Escalation disaggregated
        disagg_tbl = _prep_escalation_disagg(tables)
        _table_page(pdf, disagg_tbl,
                    title="Escalation Disaggregated — by Simulator × Model",
                    subtitle="Under = dangerous; Over = resource waste; EMR = emergency miss rate",
                    col_widths=[0.18, 0.13, 0.08, 0.15, 0.15, 0.15, 0.13])
        n_pages += 1

        # Page 6 — Manipulation check
        mc_tbl = _prep_manipulation(tables)
        _table_page(pdf, mc_tbl,
                    title="Patient Simulator Manipulation Check",
                    subtitle="FK = Flesch–Kincaid grade level  ·  Values: mean ± SD across all trajectories",
                    col_widths=[0.28, 0.24, 0.24, 0.24])
        n_pages += 1

        # Page 7 — Scorer agreement
        sa_tbl = _prep_scorer(tables)
        _table_page(pdf, sa_tbl,
                    title="Cross-Scorer Agreement — Claude Haiku 4.5 vs. GPT-5",
                    subtitle="n = 60 trajectories reviewed by both scorers",
                    col_widths=[0.32, 0.18, 0.50])
        n_pages += 1

        # Page 8 — Fairness
        fair_tbl = _prep_fairness(tables)
        _table_page(pdf, fair_tbl,
                    title="Fairness — Counterfactual Action Disparity",
                    subtitle="Major disparity = agent assigns different escalation/management for identical clinical facts",
                    col_widths=[0.38, 0.20, 0.42])
        n_pages += 1

        # Page 9 — Backbone sensitivity
        bb_tbl = _prep_backbone(tables)
        _table_page(pdf, bb_tbl,
                    title="Backbone Sensitivity — Score |Δ| by Simulator × Model",
                    subtitle="Δ = |score(Haiku backbone) − score(GPT-5 backbone)|  ·  Same agent, different patient simulator LLM",
                    col_widths=[0.22, 0.20, 0.10, 0.24, 0.24])
        n_pages += 1

        # Pages 10+ — Figures
        for fig_name in FIGURE_ORDER:
            png_path = figures / f"{fig_name}.png"
            if not png_path.exists():
                print(f"  [export] Skipping {fig_name}.png (not found)")
                continue
            title = FIGURE_TITLES.get(fig_name, fig_name)
            _figure_page(pdf, png_path, title)
            n_pages += 1

    print(f"[export] Written {out_pdf}  ({n_pages} pages: "
          f"9 table pages + {n_pages - 9} figure pages)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export PULSE-Care results to PDF + CSV")
    ap.add_argument("--tables",   default="outputs/tables",  help="Directory of CSV tables")
    ap.add_argument("--figures",  default="outputs/figures", help="Directory of PNG figures")
    ap.add_argument("--out-pdf",  default="outputs/results_summary.pdf")
    ap.add_argument("--out-csv",  default="outputs/results_summary.csv")
    args = ap.parse_args()
    run(
        tables=Path(args.tables),
        figures=Path(args.figures),
        out_pdf=Path(args.out_pdf),
        out_csv=Path(args.out_csv),
    )
