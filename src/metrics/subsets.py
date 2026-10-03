"""Canonical analysis subsets.

Every headline metric must declare which rows it is computed on. This module is the
single place that defines those filters so the README, T11 and the metric
accounting table all agree.

Primary analysis set ("primary"):
    dynamic simulators (cooperative, sparse, verbose, low_health_literacy)
    x primary agent models x both architectures
    x primary simulator backbone (claude_haiku_4_5)
    x baseline demographics x seed_offset == 0
"""
from __future__ import annotations

import pandas as pd

PRIMARY_MODELS = ["claude_haiku_4_5", "gpt_4o", "mistral_small_3_2"]
PRIMARY_BACKBONE = "claude_haiku_4_5"
DYNAMIC_SIMS = ["cooperative", "sparse", "verbose", "low_health_literacy"]
STATIC_SIM = "full_info_static"
ARCHS = ["single", "team"]

# Exact API model ids behind each key in the May 2026 runs. Trajectory files written before
# 2026-09-27 carry no `agent_model_id`; they are filled from this map. New files record the id
# themselves, which is how the Mistral Small 3.2 (2506) -> Mistral Small 4 (2603) change is
# tracked per trajectory.
LEGACY_MODEL_IDS = {
    "claude_haiku_4_5": "claude-haiku-4-5-20251001",
    "gpt_4o": "gpt-4o-2024-08-06",
    "mistral_small_3_2": "mistral-small-2506",
}

# The 24 original cases define the primary set; cases added later (emergency extension) are
# analysed as an extension so that headline metrics stay comparable across revisions.
ORIGINAL_24 = [
    "cardio_001", "cardio_002", "cardio_003", "cardio_004", "cardio_005",
    "infectious_001", "infectious_002", "infectious_003", "infectious_004", "infectious_005",
    "medsafety_001", "medsafety_002", "medsafety_003", "medsafety_004", "medsafety_005",
    "mental_001", "mental_002", "mental_003", "mental_004",
    "primary_001", "primary_002", "primary_003", "primary_004", "primary_005",
]


def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def with_model_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `agent_model_id` filled for legacy rows."""
    d = df.copy()
    if "agent_model_id" not in d.columns:
        d["agent_model_id"] = ""
    d["agent_model_id"] = d["agent_model_id"].fillna("").astype(str)
    missing = d["agent_model_id"] == ""
    d.loc[missing, "agent_model_id"] = d.loc[missing, "agent_model"].map(LEGACY_MODEL_IDS).fillna("")
    return d


def model_versions(df: pd.DataFrame) -> dict[str, list[str]]:
    """Which API model ids stand behind each model key in the data (for disclosure)."""
    d = with_model_ids(df)
    return {m: sorted(g["agent_model_id"].unique().tolist()) for m, g in d.groupby("agent_model")}


def primary(
    df: pd.DataFrame,
    models: list[str] | None = None,
    include_static: bool = False,
    archs: list[str] | None = None,
    cases: list[str] | None = None,
) -> pd.DataFrame:
    """Rows of the primary analysis set (see module docstring). `cases` overrides the
    default restriction to the 24 original cases (pass e.g. ORIGINAL_24 + extension ids)."""
    models = models or PRIMARY_MODELS
    sims = DYNAMIC_SIMS + ([STATIC_SIM] if include_static else [])
    m = df["simulator"].isin(sims) & df["agent_model"].isin(models)
    if cases is not None:
        m &= df["case_id"].isin(cases)
    elif df["case_id"].isin(ORIGINAL_24).any():
        m &= df["case_id"].isin(ORIGINAL_24)   # exclude extension cases from the primary set
    if _has(df, "sim_backbone"):
        m &= df["sim_backbone"] == PRIMARY_BACKBONE
    if _has(df, "demographic_variant"):
        m &= df["demographic_variant"] == "baseline"
    if _has(df, "seed_offset"):
        m &= df["seed_offset"].fillna(0).astype(int) == 0
    if archs:
        m &= df["agent_arch"].isin(archs)
    return df[m].copy()


def static(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    models = models or PRIMARY_MODELS
    m = (df["simulator"] == STATIC_SIM) & df["agent_model"].isin(models)
    if _has(df, "demographic_variant"):
        m &= df["demographic_variant"] == "baseline"
    if _has(df, "seed_offset"):
        m &= df["seed_offset"].fillna(0).astype(int) == 0
    return df[m].copy()


def repeat(df: pd.DataFrame) -> pd.DataFrame:
    """All dynamic baseline rows (any seed) for (case, sim, model, arch, backbone)
    conditions that have >= 2 seeds. Used for within-policy variance."""
    d = with_model_ids(df[df["simulator"].isin(DYNAMIC_SIMS)])
    if _has(d, "demographic_variant"):
        d = d[d["demographic_variant"] == "baseline"]
    # replicates must come from the SAME API model id (a version change is not a reseed)
    keys = ["case_id", "simulator", "agent_model", "agent_model_id", "agent_arch", "sim_backbone"]
    n_seeds = d.groupby(keys)["seed_offset"].transform("nunique")
    return d[n_seeds >= 2].copy()


def fairness(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """All rows (baseline + swaps) for cases/conditions that have at least one
    demographic swap trajectory."""
    models = models or PRIMARY_MODELS
    d = df[df["agent_model"].isin(models) & df["simulator"].isin(DYNAMIC_SIMS)]
    if not _has(d, "demographic_variant"):
        return d.iloc[0:0].copy()
    swap_keys = d[d["demographic_variant"] != "baseline"][
        ["case_id", "simulator", "agent_model", "agent_arch", "sim_backbone"]
    ].drop_duplicates()
    return d.merge(swap_keys, how="inner").copy()


def backbone(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Baseline, seed-0 dynamic rows for conditions run under >= 2 simulator backbones."""
    models = models or PRIMARY_MODELS
    d = df[df["agent_model"].isin(models) & df["simulator"].isin(DYNAMIC_SIMS)]
    if _has(d, "demographic_variant"):
        d = d[d["demographic_variant"] == "baseline"]
    if _has(d, "seed_offset"):
        d = d[d["seed_offset"].fillna(0).astype(int) == 0]
    keys = ["case_id", "simulator", "agent_model", "agent_arch"]
    n_bb = d.groupby(keys)["sim_backbone"].transform("nunique")
    return d[n_bb >= 2].copy()


def factorial(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from factorial simulator cells (simulator names starting with 'fx_')."""
    return df[df["simulator"].astype(str).str.startswith("fx_")].copy()


def describe(df: pd.DataFrame) -> dict:
    """Row counts of each canonical subset (for the accounting table)."""
    return {
        "all_rows": int(len(df)),
        "primary": int(len(primary(df))),
        "static": int(len(static(df))),
        "repeat": int(len(repeat(df))),
        "fairness": int(len(fairness(df))),
        "backbone": int(len(backbone(df))),
        "factorial": int(len(factorial(df))),
    }
