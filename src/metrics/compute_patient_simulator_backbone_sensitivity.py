"""Backbone sensitivity: mean |delta normalized score| between the primary patient-simulator
backbone and each alternative backbone, over matched (case, simulator, model, arch) cells.

With more than one alternative backbone the headline value is the mean over alternatives;
per-backbone values are returned by `backbone_sensitivity_detail`.
"""
import pandas as pd

PRIMARY_BACKBONE = "claude_haiku_4_5"


def backbone_sensitivity_detail(df: pd.DataFrame, primary: str = PRIMARY_BACKBONE) -> dict:
    if "sim_backbone" not in df.columns:
        return {}
    backbones = [b for b in df["sim_backbone"].unique() if b != primary]
    if not backbones or primary not in set(df["sim_backbone"]):
        return {}
    pivot = df.pivot_table(
        index=["case_id", "simulator", "agent_model", "agent_arch"],
        columns="sim_backbone", values="normalized", aggfunc="mean",
    )
    out = {}
    for b in backbones:
        if b in pivot.columns:
            d = (pivot[primary] - pivot[b]).dropna()
            if len(d):
                out[b] = {"mean_abs_delta": float(d.abs().mean()),
                          "mean_signed_delta": float(d.mean()), "n_cells": int(len(d))}
    return out


def backbone_sensitivity(df: pd.DataFrame, primary: str = PRIMARY_BACKBONE) -> float:
    det = backbone_sensitivity_detail(df, primary)
    if not det:
        return 0.0
    return float(sum(v["mean_abs_delta"] for v in det.values()) / len(det))
