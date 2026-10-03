"""Score Sensitivity: avg (max - min) normalized score across simulator variants."""
import pandas as pd


def score_sensitivity(df: pd.DataFrame) -> float:
    """Mean of (max - min) normalized score per (case, model, arch) across simulators."""
    sims = df["simulator"].unique()
    dynamic_sims = [s for s in sims if s != "full_info_static"]
    df_dyn = df[df["simulator"].isin(dynamic_sims)]
    if df_dyn.empty:
        return 0.0
    _group_cols = ["case_id", "agent_model", "agent_arch"]
    for _col in ["sim_backbone", "seed_offset"]:
        if _col in df_dyn.columns:
            _group_cols.append(_col)
    grp = df_dyn.groupby(_group_cols)["normalized"]
    return float((grp.max() - grp.min()).mean())
