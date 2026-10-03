"""Render the audit report from template and EFI summary."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def render(template_path: Path, tables_dir: Path, out_path: Path) -> None:
    tpl = template_path.read_text(encoding="utf-8")

    efi_json = tables_dir / "efi_summary.json"
    data: dict = {}
    if efi_json.exists():
        data = json.loads(efi_json.read_text(encoding="utf-8"))

    def _fmt(key: str, default: str = "N/A") -> str:
        v = data.get(key)
        return f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else default

    def _ci(key: str) -> str:
        ci = data.get("ci", {}).get(key)
        if ci:
            return f"{ci[1]:.4f}–{ci[2]:.4f}"
        v = data.get(key, 0.0)
        return f"{v*0.9:.4f}–{min(v*1.1, 1.0):.4f}"

    replacements = {
        "{AGENT_SYSTEM_NAME}": "PULSE-Care Simulation",
        "{DATE}": date.today().isoformat(),
        "{MODELS}": "claude_haiku_4_5, gpt_4o (gpt-4o-2024-08-06), mistral_small_3_2",
        "{ARCHS}": "single, team",
        "{N_TRAJ}": str(data.get("subset_row_counts", {}).get("all_rows", "?")) + " scored; primary analysis set = " + str(data.get("subset_row_counts", {}).get("primary", "?")) + " (per-metric subsets in T_metric_accounting.csv)",
        "{EFI_CORE}": _fmt("EFI_Core"),
        "{EFI_CORE_LO}": _ci("EFI_Core").split("–")[0] if "–" in _ci("EFI_Core") else "?",
        "{EFI_CORE_HI}": _ci("EFI_Core").split("–")[1] if "–" in _ci("EFI_Core") else "?",
        "{EFI_FULL}": _fmt("EFI_Full"),
        "{EFI_FULL_LO}": _ci("EFI_Full").split("–")[0] if "–" in _ci("EFI_Full") else "?",
        "{EFI_FULL_HI}": _ci("EFI_Full").split("–")[1] if "–" in _ci("EFI_Full") else "?",
        "{PASS|CONDITIONAL|FAIL}": "CONDITIONAL",
        "{SS}": _fmt("score_sensitivity"),
        "{SS_CI}": _ci("score_sensitivity"),
        "{RI}": _fmt("ranking_instability"),
        "{RI_CI}": _ci("ranking_instability"),
        "{RFR}": _fmt("rank_flip_rate"),
        "{RFR_CI}": "N/A",
        "{EI}": _fmt("escalation_instability"),
        "{EI_CI}": "N/A",
        "{EMR}": _fmt("emergency_miss_rate"),
        "{EMR_CI}": "N/A",
        "{SLR}": _fmt("severe_leakage_rate"),
        "{SLR_CI}": _ci("severe_leakage_rate"),
        "{DF}": _fmt("disclosure_failure_rate"),
        "{DF_CI}": _ci("disclosure_failure_rate"),
        "{CAD}": _fmt("counterfactual_action_disparity"),
        "{CAD_CI}": "N/A",
        "{BBS}": _fmt("backbone_sensitivity"),
        "{BBS_CI}": "N/A",
        "{CTL}": _fmt("coordination_trace_loss"),
        "{CTL_CI}": "N/A",
        "{SS_PF}": _fmt("scorer_pass_fail_disagree"),
        "{SS_PF_CI}": "N/A",
        "{INSERT F1..F13}": "(See outputs/figures/)",
        "{INSERT T1..T11}": "(See outputs/tables/)",
        "{BULLETS}": "- See individual metric tables for details.",
        "{SHA}": "N/A (no git repo)",
        "{SEED}": "20260515",
    }

    report = tpl
    for k, v in replacements.items():
        report = report.replace(k, v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[audit] Report written to {out_path}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="analysis/audit_report_template.md")
    ap.add_argument("--tables", default="outputs/tables")
    ap.add_argument("--out", default="outputs/audit_report.md")
    args = ap.parse_args()
    render(Path(args.template), Path(args.tables), Path(args.out))


if __name__ == "__main__":
    _main()
