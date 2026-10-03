"""JSON validation and retry helpers."""
from __future__ import annotations

import json

from jsonschema import ValidationError, validate


def parse_and_validate(
    raw_text: str, schema: dict | None
) -> tuple[dict | None, str | None]:
    """Return (parsed_obj, None) on success or (None, error_message) on failure."""
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start < 0 or end < 0:
            return None, "no JSON braces found"
        obj = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError as e:
        return None, f"json decode: {e}"
    if schema is not None:
        try:
            validate(obj, schema)
        except ValidationError as e:
            return None, f"schema: {e.message}"
    return obj, None


def parse_with_retry(
    client,
    messages: list[dict],
    system: str | None,
    schema: dict | None,
    temperature: float,
    max_tokens: int,
    retries: int = 2,
) -> tuple[dict, dict]:
    """Call client.chat() up to (1 + retries) times; return (parsed_obj, raw_response)."""
    last_err: str | None = None
    msgs = list(messages)
    for attempt in range(retries + 1):
        resp = client.chat(
            msgs,
            system=system,
            response_schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        obj, err = parse_and_validate(resp["text"], schema)
        if obj is not None:
            return obj, resp
        last_err = err
        msgs = msgs + [
            {"role": "assistant", "content": resp["text"]},
            {
                "role": "user",
                "content": (
                    f"Your previous output failed validation: {err}. "
                    "Reply again with ONLY a valid JSON object."
                ),
            },
        ]
    raise RuntimeError(f"JSON parse failed after {retries} retries: {last_err}")
