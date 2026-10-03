"""Escalation metrics: under/over-escalation rates, instability, emergency miss rate.

Definitions:
  under_escalation_rate  = #rows with pred < truth / #rows
  over_escalation_rate   = #rows with pred > truth / #rows
  emergency_miss_rate    = #rows (truth == emergency_911 and pred != emergency_911)
                           / #rows with truth == emergency_911
  escalation_instability = fraction of policy groups
                           (case, model, arch, sim_backbone, seed, demographic)
                           observed under >= 2 dynamic simulators whose visit-3
                           escalation level is NOT identical across simulators
                           (an indicator).
  escalation_dispersion  = mean within-group std (ddof=0) of the ordinal level
                           (the pre-revision definition, kept for traceability).

The caller decides which rows to pass (see src/metrics/subsets.py).
"""
import pandas as pd

ORD = {"routine": 0, "urgent_outpatient": 1, "ed_referral": 2, "emergency_911": 3}

GROUP_COLS = ["case_id", "agent_model", "agent_arch", "sim_backbone",
              "seed_offset", "demographic_variant"]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["true_ord"] = d["correct_escalation"].map(ORD).fillna(0).astype(int)
    d["pred_ord"] = d["escalation_level"].map(ORD).fillna(0).astype(int)
    d["gap"] = d["pred_ord"] - d["true_ord"]
    return d


def _group_cols(d: pd.DataFrame) -> list[str]:
    return [c for c in GROUP_COLS if c in d.columns]


def escalation_metrics(df: pd.DataFrame) -> dict:
    d = _prep(df)
    n = len(d)
    under = float((d["gap"] <= -1).mean()) if n else 0.0
    over = float((d["gap"] >= 1).mean()) if n else 0.0

    em = d[d["true_ord"] == 3]
    n_em = len(em)
    n_miss = int((em["pred_ord"] < 3).sum())
    emr = n_miss / n_em if n_em else 0.0

    dynamic = d[d["simulator"] != "full_info_static"]
    inst, disp, n_groups = 0.0, 0.0, 0
    if not dynamic.empty:
        g = dynamic.groupby(_group_cols(dynamic))
        multi = g["simulator"].nunique()
        eligible = multi[multi >= 2].index
        if len(eligible):
            nun = g["pred_ord"].nunique().loc[eligible]
            inst = float((nun > 1).mean())
            disp = float(g["pred_ord"].std(ddof=0).loc[eligible].mean())
            n_groups = int(len(eligible))

    return {
        "under_escalation_rate": under,
        "over_escalation_rate": over,
        "emergency_miss_rate": float(emr),
        "escalation_instability": inst,
        "escalation_dispersion": disp,
        "n_rows": int(n),
        "n_emergency_rows": int(n_em),
        "n_emergency_misses": n_miss,
        "n_instability_groups": n_groups,
    }
