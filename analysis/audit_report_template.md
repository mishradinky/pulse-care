# PULSE-Care Audit Report — {AGENT_SYSTEM_NAME}, {DATE}

## 1. Executive summary
- Agent models audited: {MODELS}
- Agent architectures: {ARCHS}
- Trajectories: {N_TRAJ}
- **EFI-Core**: {EFI_CORE} (95% CI {EFI_CORE_LO}–{EFI_CORE_HI})
- **EFI-Full**: {EFI_FULL} (95% CI {EFI_FULL_LO}–{EFI_FULL_HI})
- **Verdict**: {PASS|CONDITIONAL|FAIL}

## 2. Headline numbers

| Metric | Value | 95% CI |
|---|---|---|
| Score Sensitivity | {SS} | {SS_CI} |
| Ranking Instability (Kendall) | {RI} | {RI_CI} |
| Rank Flip Rate | {RFR} | {RFR_CI} |
| Escalation Instability | {EI} | {EI_CI} |
| Emergency Miss Rate | {EMR} | {EMR_CI} |
| Severe Leakage Rate | {SLR} | {SLR_CI} |
| Disclosure Failure | {DF} | {DF_CI} |
| Counterfactual Action Disparity | {CAD} | {CAD_CI} |
| Backbone Sensitivity | {BBS} | {BBS_CI} |
| Coordination Trace Loss | {CTL} | {CTL_CI} |
| Scorer Pass/Fail Disagreement | {SS_PF} | {SS_PF_CI} |

## 3. Figures

{INSERT F1..F13}

## 4. Tables

{INSERT T1..T11}

## 5. Risks identified

{BULLETS}

## 6. Recommendations

{BULLETS}

## 7. Reproducibility
- Commit SHA: {SHA}
- Seed: {SEED}
- Model snapshots: claude-haiku-4-5-20251001 / gpt-4o-2024-08-06 / mistral-small-2506 (May 2026 runs) and mistral-small-2603 = Mistral Small 4 (September 2026 revision runs; see efi_summary.json agent_model_versions)
- Bootstrap n=1000, case-level
