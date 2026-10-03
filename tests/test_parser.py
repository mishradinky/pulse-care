"""Unit tests for output_parser."""
import pytest
from src.output_parser import parse_and_validate


def test_clean_json():
    obj, err = parse_and_validate('{"action":"ask","question":"hello?"}', None)
    assert obj == {"action": "ask", "question": "hello?"}
    assert err is None


def test_json_with_prose():
    raw = "Here is my reply: {\"action\":\"ask\",\"question\":\"any pain?\"} thanks"
    obj, err = parse_and_validate(raw, None)
    assert obj is not None
    assert obj["action"] == "ask"


def test_invalid_json():
    obj, err = parse_and_validate("not json at all", None)
    assert obj is None
    assert err is not None


def test_schema_validation_pass():
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}
    obj, err = parse_and_validate('{"x": 42}', schema)
    assert obj == {"x": 42}
    assert err is None


def test_schema_validation_fail():
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}
    obj, err = parse_and_validate('{"x": "not-an-int"}', schema)
    assert obj is None
    assert "schema" in err
