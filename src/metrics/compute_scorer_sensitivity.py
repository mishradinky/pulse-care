"""Scorer sensitivity: disagreement rate between auto scorer and human reviewer."""
import pandas as pd


def scorer_sensitivity(
    auto: pd.DataFrame,
    human: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    m = auto.merge(human, on=["trajectory_id"], suffixes=("_auto", "_human"))
    if m.empty:
        return {k: 0.0 for k in
                ["task_score_disagree", "pass_fail_disagree", "escalation_disagree",
                 "leakage_disagree", "disclosure_disagree"]}
    out = {}
    if "normalized_auto" in m and "normalized_human" in m:
        out["task_score_disagree"] = float(
            (m["normalized_auto"].round(2) != m["normalized_human"].round(2)).mean()
        )
        out["pass_fail_disagree"] = float(
            ((m["normalized_auto"] >= threshold) != (m["normalized_human"] >= threshold)).mean()
        )
    else:
        out["task_score_disagree"] = 0.0
        out["pass_fail_disagree"] = 0.0

    for col in ["escalation", "leakage", "disclosure"]:
        a_col = f"{col}_auto" if f"{col}_auto" in m else f"{col}_level_auto"
        h_col = f"{col}_human" if f"{col}_human" in m else f"{col}_level_human"
        if a_col in m.columns and h_col in m.columns:
            out[f"{col}_disagree"] = float((m[a_col] != m[h_col]).mean())
        else:
            out[f"{col}_disagree"] = 0.0

    return out
