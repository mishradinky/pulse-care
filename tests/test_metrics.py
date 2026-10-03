"""Unit tests for core metrics functions."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import subsets as S
from src.metrics.bootstrap_ci import bootstrap_ci, bootstrap_many, resample_cases
from src.metrics.compute_disclosure_failure import disclosure_failure_rate
from src.metrics.compute_efi import CORE_KEYS, FULL_KEYS, efi_core, efi_full
from src.metrics.compute_escalation_metrics import escalation_metrics
from src.metrics.compute_fairness import cad, cad_metrics, fairness_pairs, null_pairs
from src.metrics.compute_leakage import any_leakage_rate, severe_leakage_rate
from src.metrics.compute_rank_flips import rank_flip_details, rank_flip_rate
from src.metrics.compute_rank_instability import normalized_kendall_distance, ranking_instability
from src.metrics.compute_score_sensitivity import score_sensitivity


def _make_df(n_cases: int = 5, seed: int = 42, seeds: int = 1, swaps: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cases = [f"case_{i:03d}" for i in range(n_cases)]
    sims = ["cooperative", "sparse", "verbose", "low_health_literacy"]
    models = ["claude_haiku_4_5", "gpt_4o", "mistral_small_3_2"]
    demos = ["baseline"] + (["minority_swap", "ses_swap"] if swaps else [])
    rows = []
    for case in cases:
        for sim in sims:
            for model in models:
                for so in range(seeds):
                    for demo in demos:
                        if demo != "baseline" and (sim != "cooperative" or so != 0):
                            continue
                        rows.append({
                            "trajectory_id": f"{case}__{sim}__single__{model}__{demo}__{so}",
                            "case_id": case, "category": "cardiology", "simulator": sim,
                            "agent_model": model, "agent_arch": "single",
                            "sim_backbone": "claude_haiku_4_5", "demographic_variant": demo,
                            "seed_offset": so, "normalized": float(rng.uniform(0.3, 0.9)),
                            "escalation_level": rng.choice(list(escalation_metrics.__globals__["ORD"])),
                            "correct_escalation": "emergency_911" if case == "case_000" else "ed_referral",
                            "leakage_level": int(rng.choice([0, 0, 0, 1, 2])),
                            "disclosure_failure": int(rng.choice([0, 0, 1])),
                            "disparity_severity": rng.choice(["none", "minor", "major"]) if demo != "baseline" else "none",
                        })
    return pd.DataFrame(rows)


def test_score_sensitivity_range():
    assert 0.0 <= score_sensitivity(_make_df()) <= 1.0


def test_ranking_instability_range():
    assert 0.0 <= ranking_instability(_make_df()) <= 1.0


def test_rank_flip_rate_margin_monotone():
    df = _make_df()
    r0, r5, r50 = rank_flip_rate(df, 0.0), rank_flip_rate(df, 0.05), rank_flip_rate(df, 0.5)
    assert 0.0 <= r50 <= r5 <= r0 <= 1.0
    det = rank_flip_details(df)
    assert len(det) == 3 * 6  # 3 model pairs x 6 simulator pairs


def test_escalation_metrics_keys_and_denominators():
    df = _make_df()
    m = escalation_metrics(df)
    for k in ("under_escalation_rate", "over_escalation_rate", "emergency_miss_rate",
              "escalation_instability", "escalation_dispersion"):
        assert 0.0 <= m[k] <= 1.0
    # EMR denominator is emergency-case rows only (case_000: 4 sims x 3 models)
    assert m["n_emergency_rows"] == 12
    assert m["n_instability_groups"] == 5 * 3
    # instability is an indicator: identical escalation everywhere -> 0
    same = df.copy(); same["escalation_level"] = "ed_referral"
    assert escalation_metrics(same)["escalation_instability"] == 0.0


def test_cad_pairs_and_denominator():
    df = _make_df(swaps=True)
    pairs = fairness_pairs(df)
    assert len(pairs) == 5 * 3 * 2          # cases x models x swap variants
    m = cad_metrics(df)
    assert m["n_pairs"] == 30
    assert 0.0 <= m["cad_escalation"] <= 1.0
    assert cad(df) == m["cad_escalation"]
    # identical escalation on both sides -> 0
    same = df.copy(); same["escalation_level"] = "ed_referral"
    assert cad(same) == 0.0


def test_null_pairs_need_seeds():
    assert null_pairs(_make_df(seeds=1)).empty
    npairs = null_pairs(_make_df(seeds=3))
    assert len(npairs) == 5 * 4 * 3 * 2     # cases x sims x models x (seed1, seed2)


def test_subsets_primary_excludes_reseeds_and_swaps():
    df = _make_df(seeds=2, swaps=True)
    prim = S.primary(df)
    assert (prim["seed_offset"] == 0).all() and (prim["demographic_variant"] == "baseline").all()
    assert len(S.repeat(df)) == len(df[df["demographic_variant"] == "baseline"])
    assert S.describe(df)["fairness"] > 0


def test_efi_composition_matches_paper():
    # published EFI composition
    assert CORE_KEYS == ["score_sensitivity", "ranking_instability", "escalation_instability",
                         "emergency_miss_rate", "severe_leakage_rate"]
    assert len(FULL_KEYS) == 9
    m = {k: 0.2 for k in FULL_KEYS}
    assert efi_core(m) == pytest.approx(0.2) and efi_full(m) == pytest.approx(0.2)


def test_leakage_rates():
    df = _make_df()
    assert 0.0 <= severe_leakage_rate(df) <= any_leakage_rate(df) <= 1.0


def test_disclosure_failure_rate():
    assert 0.0 <= disclosure_failure_rate(_make_df()) <= 1.0


def test_normalized_kendall_distance():
    r1 = np.array([1, 2, 3])
    assert normalized_kendall_distance(r1, r1) == pytest.approx(0.0, abs=1e-6)
    assert normalized_kendall_distance(r1, r1[::-1]) == pytest.approx(1.0, abs=1e-6)


def test_bootstrap_relabels_duplicate_cases():
    df = _make_df()
    sub = resample_cases(df, np.random.default_rng(1))
    assert len(sub) == len(df)
    assert sub["case_id"].nunique() == df["case_id"].nunique()  # every draw is distinct


def test_bootstrap_ci_bounds():
    df = _make_df()
    mean, lo, hi = bootstrap_ci(df, score_sensitivity, n=30, seed=99)
    assert lo <= mean <= hi
    many = bootstrap_many(df, {"": lambda d: {"a": score_sensitivity(d)}}, n=10)
    assert "a" in many
