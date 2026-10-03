"""Leakage rates from pre-scored DataFrame."""
import pandas as pd


def severe_leakage_rate(df: pd.DataFrame) -> float:
    return float((df["leakage_level"] == 2).mean())


def any_leakage_rate(df: pd.DataFrame) -> float:
    return float((df["leakage_level"] >= 1).mean())
