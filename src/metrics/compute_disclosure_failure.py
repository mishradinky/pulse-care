"""Disclosure failure rate from pre-scored DataFrame."""
import pandas as pd


def disclosure_failure_rate(df: pd.DataFrame) -> float:
    return float(df["disclosure_failure"].mean())
