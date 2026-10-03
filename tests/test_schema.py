"""Validate all case JSON files against case_schema.json."""
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError, validate

SCHEMA = json.loads(Path("schemas/case_schema.json").read_text(encoding="utf-8"))
CASES = sorted(Path("cases").glob("*.json"))
EXP_CFG = yaml.safe_load(Path("configs/experiment_config.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_path", CASES, ids=[p.stem for p in CASES])
def test_case_schema(case_path: Path) -> None:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    try:
        validate(case, SCHEMA)
    except ValidationError as e:
        pytest.fail(f"{case_path.name}: {e.message}")


def test_case_counts() -> None:
    # 24 original + 6 draft emergency-extension cases (defined, not run)
    assert len(CASES) == 30, f"Expected 30 cases, found {len(CASES)}"
    ids = {p.stem for p in CASES}
    assert set(EXP_CFG["case_lists"]["original_24"]) <= ids
    assert set(EXP_CFG["case_lists"]["emergency_ext_6"]) <= ids
    assert not set(EXP_CFG["case_lists"]["original_24"]) & set(EXP_CFG["case_lists"]["emergency_ext_6"])


def test_case_categories() -> None:
    counts: dict[str, int] = {}
    for p in CASES:
        case = json.loads(p.read_text(encoding="utf-8"))
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert counts == {"primary_care": 7, "cardiology": 6, "medication_safety": 6,
                      "infectious_disease": 6, "mental_health": 5}


def test_emergency_extension_cases_are_emergencies() -> None:
    for cid in EXP_CFG["case_lists"]["emergency_ext_6"]:
        case = json.loads((Path("cases") / f"{cid}.json").read_text(encoding="utf-8"))
        assert case["ground_truth"]["correct_escalation"] == "emergency_911"


def test_all_subset_simulators_have_cards() -> None:
    for name, sub in EXP_CFG["subsets"].items():
        for sim in sub["simulators"]:
            if sim == "full_info_static":
                continue
            assert (Path("simulator_cards") / f"{sim}.json").exists(), f"{name}: missing card {sim}"
