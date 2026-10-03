"""Run one trajectory: one case × one simulator × one agent architecture × one model."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

from src.model_clients import make_client
from src.output_parser import parse_and_validate, parse_with_retry

MAX_FOLLOWUPS = 5
MAX_TURNS = 15


def _load_json(p: Path | str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _read_prompt(name: str) -> str:
    return (Path("prompts") / f"{name}.txt").read_text(encoding="utf-8")


def _patient_prompt(simulator_variant: str, sim_card: dict | None) -> str:
    """Named variants have their own prompt file; factorial cells (fx_*) and any other
    card without a dedicated prompt render prompts/patient_factorial.txt from the card."""
    p = Path("prompts") / f"patient_{simulator_variant}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    card = sim_card or {}
    rules = "\n".join(f"- {r}" for r in card.get("variant_rules", []))
    return (
        _read_prompt("patient_factorial")
        .replace("{VARIANT_NAME}", simulator_variant)
        .replace("{VARIANT_RULES}", rules)
        .replace("{VOICE}", str(card.get("voice", "")))
        .replace("{AFFECT}", str(card.get("emotional_affect", "")))
        .replace("{VOCAB}", str(card.get("vocabulary_constraint", "")))
        .replace("{MAX_WORDS}", str(card.get("max_words_per_turn", 80)))
    )


def _get_patient_utterance(
    sim_client, sim_system: str, sim_cfg: dict, user_msg: str
) -> str:
    """Call patient simulator and return plain utterance text."""
    resp = sim_client.chat(
        [{"role": "user", "content": user_msg}],
        system=sim_system,
        temperature=sim_cfg.get("temperature", 0.7),
        max_tokens=sim_cfg.get("max_tokens", 512),
    )
    text = resp["text"].strip()
    obj, _ = parse_and_validate(text, None)
    if obj and "utterance" in obj:
        return obj["utterance"]
    return text


def _run_team_visit(
    case: dict,
    visit: dict,
    dialogue: list[dict],
    agent_client,
    agent_cfg: dict,
    exp_cfg: dict,
    case_meta: dict,
    sim_client,
    sim_cfg: dict,
    sim_system: str,
) -> dict:
    """Run the 4-role team workflow for one visit and return coordinator assessment."""
    team_schema_coord = _load_json("schemas/agent_team_output.json")

    def call_role(prompt_name: str, extra_replacements: dict[str, str]) -> dict:
        tpl = _read_prompt(prompt_name)
        tpl = tpl.replace("{CASE_METADATA}", json.dumps(case_meta))
        tpl = tpl.replace("{VISIT_INDEX}", str(visit["visit_index"]))
        tpl = tpl.replace("{DIALOGUE_HISTORY}", json.dumps(dialogue))
        for k, v in extra_replacements.items():
            tpl = tpl.replace(k, v)
        obj, _ = parse_with_retry(
            agent_client,
            [{"role": "user", "content": tpl}],
            system=None,
            schema=None,
            temperature=agent_cfg.get("temperature", 0.1),
            max_tokens=agent_cfg.get("max_tokens", 2048),
            retries=exp_cfg["protocol"]["json_retries"],
        )
        return obj

    # Clinician proposes workup / asks question
    clin_out = call_role("team_clinician", {})
    clin_json = json.dumps(clin_out)

    # If clinician wants to ask a question, do one patient exchange
    if clin_out.get("action") == "ask" and clin_out.get("question"):
        question = clin_out["question"]
        dialogue.append({"speaker": "clinician", "text": question})
        utterance = _get_patient_utterance(
            sim_client, sim_system, sim_cfg, f"Clinician asks: {question}"
        )
        dialogue.append({"speaker": "patient", "text": utterance})

    # Safety checker reviews
    safety_out = call_role("team_safety_checker", {"{CLINICIAN_OUTPUT}": clin_json})
    safety_json = json.dumps(safety_out)

    # Pharm/nurse reviews
    pharm_out = call_role("team_pharmacist_nurse", {"{CLINICIAN_OUTPUT}": clin_json})
    pharm_json = json.dumps(pharm_out)

    # Coordinator synthesizes
    coord_obj, _ = parse_with_retry(
        agent_client,
        [{"role": "user", "content": _read_prompt("team_coordinator")
          .replace("{CLINICIAN_OUTPUT}", clin_json)
          .replace("{SAFETY_OUTPUT}", safety_json)
          .replace("{PHARM_OUTPUT}", pharm_json)
          .replace("{CASE_METADATA}", json.dumps(case_meta))
          .replace("{DIALOGUE_HISTORY}", json.dumps(dialogue))}],
        system=None,
        schema=team_schema_coord,
        temperature=agent_cfg.get("temperature", 0.1),
        max_tokens=agent_cfg.get("max_tokens", 2048),
        retries=exp_cfg["protocol"]["json_retries"],
    )
    return coord_obj


def _run_team_visit_static(
    case: dict,
    visit: dict,
    agent_client,
    agent_cfg: dict,
    exp_cfg: dict,
    case_meta: dict,
) -> dict:
    """Team workflow for full_info_static: all 4 roles use the vignette, no patient dialogue."""
    team_schema_coord = _load_json("schemas/agent_team_output.json")
    vignette_ctx = (
        f"Full case vignette: {case['static_vignette']}\n"
        f"Visit {visit['visit_index']} details: {json.dumps(visit)}"
    )
    dialogue = [{"speaker": "vignette", "text": vignette_ctx}]

    def call_role(prompt_name: str, extra_replacements: dict[str, str]) -> dict:
        tpl = _read_prompt(prompt_name)
        tpl = tpl.replace("{CASE_METADATA}", json.dumps(case_meta))
        tpl = tpl.replace("{VISIT_INDEX}", str(visit["visit_index"]))
        tpl = tpl.replace("{DIALOGUE_HISTORY}", json.dumps(dialogue))
        for k, v in extra_replacements.items():
            tpl = tpl.replace(k, v)
        obj, _ = parse_with_retry(
            agent_client,
            [{"role": "user", "content": tpl}],
            system=None,
            schema=None,
            temperature=agent_cfg.get("temperature", 0.1),
            max_tokens=agent_cfg.get("max_tokens", 2048),
            retries=exp_cfg["protocol"]["json_retries"],
        )
        return obj or {}

    clin_out = call_role("team_clinician", {})
    clin_json = json.dumps(clin_out)
    safety_out = call_role("team_safety_checker", {"{CLINICIAN_OUTPUT}": clin_json})
    safety_json = json.dumps(safety_out)
    pharm_out = call_role("team_pharmacist_nurse", {"{CLINICIAN_OUTPUT}": clin_json})
    pharm_json = json.dumps(pharm_out)

    coord_obj, _ = parse_with_retry(
        agent_client,
        [{"role": "user", "content": _read_prompt("team_coordinator")
          .replace("{CLINICIAN_OUTPUT}", clin_json)
          .replace("{SAFETY_OUTPUT}", safety_json)
          .replace("{PHARM_OUTPUT}", pharm_json)
          .replace("{CASE_METADATA}", json.dumps(case_meta))
          .replace("{DIALOGUE_HISTORY}", json.dumps(dialogue))}],
        system=None,
        schema=team_schema_coord,
        temperature=agent_cfg.get("temperature", 0.1),
        max_tokens=agent_cfg.get("max_tokens", 2048),
        retries=exp_cfg["protocol"]["json_retries"],
    )
    return coord_obj


def run_trajectory(
    case: dict,
    simulator_variant: str,
    agent_architecture: str,
    model_key: str,
    sim_backbone_key: str,
    model_cfg: dict,
    exp_cfg: dict,
    output_dir: Path,
    demographic_variant: str = "baseline",
    seed_offset: int = 0,
    simulate: bool = False,
) -> Path:
    agent_cfg = model_cfg["agents"][model_key]
    sim_cfg = model_cfg["patient_simulator_backbones"][sim_backbone_key]

    is_static = simulator_variant == "full_info_static"
    sim_card = None if is_static else _load_json(
        Path("simulator_cards") / f"{simulator_variant}.json"
    )

    agent_client = make_client(agent_cfg["provider"], agent_cfg, simulate=simulate)
    sim_client = None if is_static else make_client(
        sim_cfg["provider"], sim_cfg, simulate=simulate
    )

    agent_schema = _load_json(
        "schemas/agent_single_output.json"
        if agent_architecture == "single"
        else "schemas/agent_team_output.json"
    )
    agent_prompt = "agent_single" if agent_architecture == "single" else None
    agent_system = _read_prompt(agent_prompt) if agent_prompt else None

    # Build case metadata exposed to agent (no diagnosis)
    demo = case["fairness_fields"].get(demographic_variant, case["fairness_fields"]["baseline"])
    case_meta = {"category": case["category"], "demographics": demo}

    transcript: list[dict] = []
    total_turns = 0

    for visit in case["visits"]:
        visit_rec: dict = {"visit": visit["visit_index"]}

        # ── Static baseline: agent gets full vignette, no dialogue ──────────
        if is_static:
            if agent_architecture == "team":
                assessment = _run_team_visit_static(
                    case, visit, agent_client, agent_cfg, exp_cfg, case_meta
                )
            else:
                prompt_content = (
                    f"Full case vignette: {case['static_vignette']}\n"
                    f"Visit {visit['visit_index']} additional info: {json.dumps(visit)}\n"
                    "Produce a clinical assessment for this visit."
                )
                assessment, _ = parse_with_retry(
                    agent_client,
                    [{"role": "user", "content": prompt_content}],
                    system=agent_system,
                    schema=agent_schema,
                    temperature=agent_cfg.get("temperature", 0.1),
                    max_tokens=agent_cfg.get("max_tokens", 2048),
                    retries=exp_cfg["protocol"]["json_retries"],
                )
            visit_rec["assessment"] = assessment
            transcript.append(visit_rec)
            continue

        # ── Dynamic simulator: alternating dialogue ──────────────────────────
        sim_system = (
            _patient_prompt(simulator_variant, sim_card)
            .replace("{CASE_FILE_JSON}", json.dumps({**case, "visits": [visit]}))
            .replace("{SIMULATOR_CARD_JSON}", json.dumps(sim_card))
            .replace("{VISIT_INDEX}", str(visit["visit_index"]))
        )

        dialogue: list[dict] = []

        # Patient opens with chief complaint
        opening = _get_patient_utterance(
            sim_client,
            sim_system,
            sim_cfg,
            "The clinician has just entered. Give your chief complaint in-character.",
        )
        dialogue.append({"speaker": "patient", "text": opening})
        total_turns += 1

        if agent_architecture == "team":
            # Team architecture: delegate to team workflow per visit
            assessment = _run_team_visit(
                case, visit, dialogue, agent_client, agent_cfg,
                exp_cfg, case_meta, sim_client, sim_cfg, sim_system,
            )
            visit_rec["dialogue"] = dialogue
            visit_rec["assessment"] = assessment
            transcript.append(visit_rec)
            continue

        # Single agent alternating dialogue
        followups = 0
        concluded = False

        while followups < MAX_FOLLOWUPS and total_turns < MAX_TURNS:
            agent_content = (
                agent_system or ""
            ).replace("{CASE_METADATA}", json.dumps(case_meta)
             ).replace("{VISIT_INDEX}", str(visit["visit_index"])
             ).replace("{DIALOGUE_HISTORY}", json.dumps(dialogue))

            agent_resp, _ = parse_with_retry(
                agent_client,
                [{"role": "user",
                  "content": f"Patient said: \"{dialogue[-1]['text']}\"\nDialogue so far: {json.dumps(dialogue)}"}],
                system=agent_system
                    .replace("{CASE_METADATA}", json.dumps(case_meta))
                    .replace("{VISIT_INDEX}", str(visit["visit_index"]))
                    .replace("{DIALOGUE_HISTORY}", json.dumps(dialogue))
                    if agent_system else None,
                schema=agent_schema,
                temperature=agent_cfg.get("temperature", 0.1),
                max_tokens=agent_cfg.get("max_tokens", 2048),
                retries=exp_cfg["protocol"]["json_retries"],
            )
            total_turns += 1

            if agent_resp.get("action") == "assess":
                visit_rec["dialogue"] = dialogue
                visit_rec["assessment"] = agent_resp
                concluded = True
                break

            # Agent asks a question
            question = agent_resp.get("question", "Can you tell me more?")
            dialogue.append({"speaker": "clinician", "text": question})
            followups += 1

            pat_utterance = _get_patient_utterance(
                sim_client, sim_system, sim_cfg, f"Clinician asks: {question}"
            )
            dialogue.append({"speaker": "patient", "text": pat_utterance})
            total_turns += 1

        if not concluded:
            # Turn budget exhausted — force assessment
            forced, _ = parse_with_retry(
                agent_client,
                [{"role": "user", "content": "Turn budget exhausted. Emit final assessment now."}],
                system=agent_system
                    .replace("{CASE_METADATA}", json.dumps(case_meta))
                    .replace("{VISIT_INDEX}", str(visit["visit_index"]))
                    .replace("{DIALOGUE_HISTORY}", json.dumps(dialogue))
                    if agent_system else None,
                schema=agent_schema,
                temperature=agent_cfg.get("temperature", 0.1),
                max_tokens=agent_cfg.get("max_tokens", 2048),
                retries=exp_cfg["protocol"]["json_retries"],
            )
            visit_rec["dialogue"] = dialogue
            visit_rec["assessment"] = forced
            visit_rec["forced"] = True

        transcript.append(visit_rec)

    record = {
        "case_id": case["case_id"],
        "category": case["category"],
        "simulator": simulator_variant,
        "agent_arch": agent_architecture,
        "agent_model": model_key,
        "sim_backbone": sim_backbone_key,
        "demographic_variant": demographic_variant,
        "seed_offset": seed_offset,
        # exact API model ids, so a later model-version change (e.g. Mistral Small 3.2 -> 4)
        # is visible per trajectory and same-model replicate analyses can exclude mismatches
        "agent_model_id": agent_cfg.get("model", ""),
        "sim_backbone_id": sim_cfg.get("model", ""),
        "case_meta": case_meta,
        "transcript": transcript,
        "ground_truth": case["ground_truth"],
        "ts": time.time(),
    }

    key = hashlib.md5(
        json.dumps({
            k: record[k]
            for k in ("case_id", "simulator", "agent_arch", "agent_model",
                       "sim_backbone", "demographic_variant", "seed_offset")
        }).encode()
    ).hexdigest()[:10]

    fname = "__".join([
        case["case_id"], simulator_variant, agent_architecture,
        model_key, sim_backbone_key,
        demographic_variant, str(seed_offset), key,
    ]) + ".json"

    out = output_dir / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--simulator", required=True)
    ap.add_argument("--agent", required=True, choices=["single", "team"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--sim-backbone", required=True)
    ap.add_argument("--demographic-variant", default="baseline")
    ap.add_argument("--exp-cfg", default="configs/experiment_config.yaml")
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    ap.add_argument("--simulate", action="store_true", help="Use mock LLM client")
    args = ap.parse_args()

    exp = yaml.safe_load(Path(args.exp_cfg).read_text(encoding="utf-8"))
    mdl = yaml.safe_load(Path(args.model_cfg).read_text(encoding="utf-8"))
    case = _load_json(Path("cases") / f"{args.case}.json")

    out = run_trajectory(
        case,
        args.simulator,
        args.agent,
        args.model,
        args.sim_backbone,
        mdl,
        exp,
        Path(exp["output_dir"]) / "trajectories",
        demographic_variant=args.demographic_variant,
        simulate=args.simulate,
    )
    print(out)


if __name__ == "__main__":
    _main()
