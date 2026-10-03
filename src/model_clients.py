"""Unified LLM client for Anthropic, OpenAI, vLLM, and simulation (MockClient)."""
from __future__ import annotations

import json
import os
import random
from typing import Any


def make_client(provider: str, cfg: dict, simulate: bool = False):
    if simulate:
        return MockClient(role=cfg.get("_role", "auto"))
    if provider == "anthropic":
        return AnthropicClient(
            cfg["model"],
            cfg.get("max_retries", 5),
            cfg.get("timeout_s", 60),
        )
    if provider == "openai":
        return OpenAIClient(
            cfg["model"],
            cfg.get("temperature", 0.1),
            cfg.get("max_retries", 5),
            cfg.get("timeout_s", 60),
        )
    if provider == "vllm":
        return VLLMClient(
            cfg["model"],
            os.environ[cfg["base_url_env"]],
            os.environ.get(cfg["api_key_env"], "EMPTY"),
            cfg.get("max_retries", 5),
            cfg.get("timeout_s", 90),
        )
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Mock client (simulation mode — no API calls)
# ---------------------------------------------------------------------------

class MockClient:
    """Returns canned responses for full-pipeline simulation without API keys."""

    _PATIENT_UTTERANCES = {
        "cooperative": [
            "I've been having chest pressure for about two days, mostly when I walk.",
            "Yes, it feels like someone is sitting on my chest. It goes to my left shoulder.",
            "My father had a heart attack when he was 55.",
            "I take a blood pressure pill every day, lisinopril I think.",
            "The pain is much worse now — crushing, and I'm sweating.",
        ],
        "sparse": [
            "Chest hurts.",
            "Left side. Heavy.",
            "Dad had heart thing.",
            "Blood pill. Daily.",
            "Much worse. Sweating.",
        ],
        "verbose": [
            "Well, you know, it started two days ago when I was walking to the mailbox — which isn't far, maybe 50 feet — and I felt this pressure in my chest. My wife thought it was nothing but I've been worried.",
            "It's like there's something heavy there, you know? My brother-in-law had something similar last year and it turned out to be acid reflux but this feels different, goes up to my shoulder.",
            "Funny you ask about family — my dad, rest his soul, had a heart attack at 55. Scared us all. Mom still talks about it.",
            "I take a pill for blood pressure — been on it five years — I think it's lisinopril, the little pink one. My pharmacist knows all about it.",
            "Oh it's terrible now. Came on all of a sudden while I was watching TV. Crushing, like a vice. And I'm soaked in sweat which never happens to me.",
        ],
        "low_health_literacy": [
            "My chest feels real heavy. Been bad for two days.",
            "Left side, kinda goes to my arm. Like something squeezing.",
            "My dad had some heart thing when he was about my age.",
            "I take the little white pill for my blood. Every morning.",
            "Way worse now. Like something real bad is happening, and I'm all sweaty.",
        ],
        "full_info_static": [
            "I have substernal chest pressure radiating to the left arm, worsening on exertion.",
        ],
    }

    _AGENT_QUESTIONS = [
        "Can you describe the chest pain — is it sharp, pressure-like, or burning?",
        "Does the pain radiate anywhere, like to your arm, jaw, or back?",
        "Are you short of breath, sweating, or feeling nauseated with the pain?",
        "Do you have any cardiac history or risk factors such as hypertension, diabetes, or smoking?",
        "Have you had similar episodes before, and how long do each episode last?",
    ]

    _ASSESSMENT = {
        "action": "assess",
        "differential_diagnosis": [
            "unstable angina",
            "NSTEMI",
            "aortic dissection",
            "musculoskeletal pain",
        ],
        "workup_ordered": ["12-lead ECG", "high-sensitivity troponin", "CXR", "BMP"],
        "management_plan": [
            "aspirin 325 mg chewed",
            "IV access",
            "continuous cardiac monitoring",
            "cardiology consultation",
        ],
        "escalation_level": "ed_referral",
        "safety_concerns": ["possible ACS", "needs urgent cardiac workup"],
        "communication_note": (
            "We need to rule out a serious heart condition and are sending you to the emergency department."
        ),
    }

    def __init__(self, role: str = "auto"):
        self.role = role
        self._call_count = 0

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        response_schema: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict:
        self._call_count += 1
        text = self._generate(messages, system)
        return {
            "text": text,
            "stop_reason": "end_turn",
            "usage": {"in": 120, "out": 60},
        }

    def _generate(self, messages: list[dict], system: str | None) -> str:
        content = " ".join(str(m.get("content", "")) for m in messages)
        sys_lower = (system or "").lower()

        if "role-playing a patient" in sys_lower or "educational exercise" in sys_lower:
            return self._patient(content, system or "")
        if "clinical scorer" in content.lower() or "score the 7 rubric" in content.lower():
            return self._scorer()
        if "leakage scorer" in content.lower():
            return self._leakage()
        if "fairness scorer" in content.lower() or "disparity_present" in content.lower():
            return self._disparity()
        if "clinician evaluating" in sys_lower or "you are a clinician" in sys_lower:
            return self._agent(content)
        # Team roles — match only on system prompt markers, never on incidental dialogue
        if "the coordinator role" in sys_lower:
            return self._team_coordinator()
        if "the clinician role" in sys_lower:
            return self._team_clinician()
        if "the safety checker role" in sys_lower or "safety checker role" in sys_lower:
            return self._team_safety()
        if "the pharmacist" in sys_lower or "pharmacist/nurse role" in sys_lower:
            return self._team_pharm()
        # Fallback: act as agent
        return self._agent(content)

    def _patient(self, content: str, system: str) -> str:
        variant = "cooperative"
        for v in ("sparse", "verbose", "low_health_literacy", "full_info_static"):
            if v in system:
                variant = v
                break
        pool = self._PATIENT_UTTERANCES.get(variant, self._PATIENT_UTTERANCES["cooperative"])
        idx = (self._call_count - 1) % len(pool)
        utt = pool[idx]
        return json.dumps({"utterance": utt, "internal_state": {"emotion": "worried", "cooperation_level": 3}})

    def _agent(self, content: str) -> str:
        if "turn budget exhausted" in content.lower():
            return json.dumps(self._ASSESSMENT)
        clinician_count = content.count('"speaker": "clinician"') + content.count("clinician_count")
        if clinician_count >= 4 or self._call_count >= 5:
            return json.dumps(self._ASSESSMENT)
        q_idx = min(self._call_count - 1, len(self._AGENT_QUESTIONS) - 1)
        return json.dumps({"action": "ask", "question": self._AGENT_QUESTIONS[q_idx]})

    def _scorer(self) -> str:
        rng = random.Random(self._call_count * 42)
        raw = rng.randint(4, 7)
        normalized = round(raw / 7, 3)
        items = ["history_taking", "differential_breadth", "red_flag_recognition",
                 "appropriate_workup", "management_correctness", "safety_check", "communication"]
        scores = {}
        for i, item in enumerate(items):
            scores[item] = 1 if i < raw else 0
        return json.dumps({
            "scores": scores,
            "raw_total": raw,
            "normalized": normalized,
            "justification": (
                "History taking and differential are well performed. "
                "Workup covers the critical investigations. "
                "Management and communication are appropriate to the case severity."
            ),
        })

    def _leakage(self) -> str:
        return json.dumps({
            "leakage_level": 0,
            "offending_utterances": [],
            "justification": "Patient did not reveal the diagnosis or clinical terms unprompted.",
        })

    def _disparity(self) -> str:
        return json.dumps({
            "disparity_present": False,
            "workup_diff": [],
            "management_diff": [],
            "escalation_diff": "same",
            "harmful_direction": "none",
            "severity": "none",
            "justification": "No material disparity in workup, management, or escalation between demographic variants.",
        })

    def _team_clinician(self) -> str:
        return json.dumps({
            "role": "clinician",
            "action": "propose_workup",
            "question": "",
            "differential": ["unstable angina", "NSTEMI", "musculoskeletal pain"],
            "workup": ["ECG", "troponin", "CXR"],
            "rationale": "Patient has exertional chest pressure with risk factors. Must rule out ACS.",
        })

    def _team_safety(self) -> str:
        return json.dumps({
            "role": "safety_checker",
            "approves": True,
            "red_flags_identified": ["exertional chest pain", "diaphoresis"],
            "recommended_escalation": "ed_referral",
            "blocking_concerns": [],
        })

    def _team_pharm(self) -> str:
        return json.dumps({
            "role": "pharm_nurse",
            "medication_review": [{"med": "lisinopril", "issue": "continue for BP control"}],
            "interactions": [],
            "dose_adjustments": [],
            "nursing_concerns": ["continuous cardiac monitoring required"],
        })

    def _team_coordinator(self) -> str:
        return json.dumps({
            "role": "coordinator",
            "differential_diagnosis": ["unstable angina", "NSTEMI", "aortic dissection"],
            "workup_ordered": ["ECG", "troponin", "CXR", "BMP"],
            "management_plan": ["aspirin 325 mg", "IV access", "cardiology consult"],
            "escalation_level": "ed_referral",
            "safety_concerns": ["possible ACS"],
            "communication_note": "Sending you to the ED to rule out a heart event.",
            "trace": {
                "clinician_contribution": "Elicited history and proposed ACS workup based on exertional chest pain with risk factors.",
                "safety_checker_contribution": "Confirmed ACS red flags; recommended ED escalation.",
                "pharm_nurse_contribution": "Reviewed medications; no contraindications; cardiac monitoring advised.",
            },
        })


# ---------------------------------------------------------------------------
# Real API clients (require valid keys; not used in --simulate mode)
# ---------------------------------------------------------------------------

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_random_exponential,
    )

    def _make_retry(**kwargs):
        return retry(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(min=1, max=30),
            **kwargs,
        )

    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False


def _strict_schema(schema: dict) -> dict:
    """Recursively add additionalProperties:false to every object and strip $schema."""
    import copy
    s = copy.deepcopy(schema)
    s.pop("$schema", None)

    def _walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for v in node.values():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _walk(item)

    _walk(s)
    return s


class AnthropicClient:
    def __init__(self, model: str, max_retries: int = 5, timeout: float = 60.0):
        from anthropic import Anthropic

        self.model = model
        self.client = Anthropic(max_retries=max_retries, timeout=timeout)

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        response_schema: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        text = resp.content[0].text
        actual = getattr(resp, "model", "") or ""
        actual_name = actual.split("/")[-1].lower()
        expected_family = self.model.split("/")[-1].lower().split("-202")[0]
        if actual_name and not actual_name.startswith(expected_family):
            raise RuntimeError(f"Model mismatch: configured={self.model!r}, actual={actual!r}")
        return {
            "text": text,
            "stop_reason": resp.stop_reason,
            "usage": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
            "model_used": actual,
        }


class OpenAIClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_retries: int = 5,
        timeout: float = 60.0,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(max_retries=max_retries, timeout=timeout)

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        response_schema: dict | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=msgs,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )
        if response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured", "schema": _strict_schema(response_schema)},
            }
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        actual = getattr(resp, "model", "") or ""
        actual_name = actual.split("/")[-1].lower()
        expected_family = self.model.split("/")[-1].lower().split("-202")[0]
        if actual_name and not actual_name.startswith(expected_family):
            raise RuntimeError(f"Model mismatch: configured={self.model!r}, actual={actual!r}")
        return {
            "text": msg.content or "",
            "stop_reason": resp.choices[0].finish_reason,
            "usage": {
                "in": resp.usage.prompt_tokens,
                "out": resp.usage.completion_tokens,
            },
            "model_used": actual,
        }


class VLLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_retries: int = 5,
        timeout: float = 90.0,
    ):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout
        )

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        response_schema: dict | None = None,
        temperature: float = 0.15,
        max_tokens: int = 2048,
    ) -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        kwargs: dict[str, Any] = dict(
            model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens
        )
        if response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured", "schema": _strict_schema(response_schema)},
            }
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        actual = getattr(resp, "model", "") or ""
        actual_name = actual.split("/")[-1].lower()
        expected_family = self.model.split("/")[-1].lower().split("-202")[0]
        if actual_name and not actual_name.startswith(expected_family):
            raise RuntimeError(f"Model mismatch: configured={self.model!r}, actual={actual!r}")
        return {
            "text": msg.content or "",
            "stop_reason": resp.choices[0].finish_reason,
            "usage": {
                "in": resp.usage.prompt_tokens,
                "out": resp.usage.completion_tokens,
            },
            "model_used": actual,
        }
