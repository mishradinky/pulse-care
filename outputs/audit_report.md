# PULSE-Care Audit Report — PULSE-Care Simulation, 2026-09-27

## 1. Executive summary
- Agent models audited: claude_haiku_4_5, gpt_4o (gpt-4o-2024-08-06), mistral_small_3_2
- Agent architectures: single, team
- Trajectories: 1542 scored; primary analysis set = 576 (per-metric subsets in T_metric_accounting.csv)
- **EFI-Core**: 0.2435 (95% CI 0.2024–0.2815)
- **EFI-Full**: 0.2000 (95% CI 0.1736–0.2266)
- **Verdict**: CONDITIONAL

## 2. Headline numbers

| Metric | Value | 95% CI |
|---|---|---|
| Score Sensitivity | 0.2471 | 0.2067–0.2897 |
| Ranking Instability (Kendall) | 0.1667 | 0.0000–0.2222 |
| Rank Flip Rate | 0.1667 | N/A |
| Escalation Instability | 0.6042 | N/A |
| Emergency Miss Rate | 0.1458 | N/A |
| Severe Leakage Rate | 0.0538 | 0.0312–0.0799 |
| Disclosure Failure | 0.3785 | 0.2534–0.5035 |
| Counterfactual Action Disparity | 0.2569 | N/A |
| Backbone Sensitivity | 0.1183 | N/A |
| Coordination Trace Loss | 0.0069 | N/A |
| Scorer Pass/Fail Disagreement | 0.2000 | N/A |

## 3. Figures

(See outputs/figures/)

## 4. Tables

(See outputs/tables/)

## 5. Risks identified

- See individual metric tables for details.

## 6. Recommendations

- See individual metric tables for details.

## 7. Reproducibility
- Commit SHA: N/A (no git repo)
- Seed: 20260515
- Model snapshots: claude-haiku-4-5-20251001 / gpt-4o-2024-08-06 / mistral-small-2506 (May 2026 runs) and mistral-small-2603 = Mistral Small 4 (September 2026 revision runs; see efi_summary.json agent_model_versions)
- Bootstrap n=1000, case-level
