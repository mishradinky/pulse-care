"""Send one tiny request to every configured model and report OK / error.

Run this BEFORE any batch so a wrong model id, missing key or dead endpoint is caught
with ~15 cheap calls instead of after hundreds of failed trajectories.

  python scripts/check_models.py            # all agents, backbones, scorers
  python scripts/check_models.py --only gpt_4_1 mistral_large
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

import yaml

from src.model_clients import make_client


def check(name: str, cfg: dict) -> tuple[str, str]:
    for env_key in ("base_url_env", "api_key_env"):
        if env_key in cfg and not os.environ.get(cfg[env_key]):
            return "FAIL", f"env var {cfg[env_key]} is not set"
    if cfg["provider"] == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "FAIL", "ANTHROPIC_API_KEY not set"
    if cfg["provider"] == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return "FAIL", "OPENAI_API_KEY not set"
    try:
        client = make_client(cfg["provider"], {**cfg, "max_retries": 1, "timeout_s": 45})
        t0 = time.time()
        resp = client.chat([{"role": "user", "content": "Reply with the single word OK."}],
                           system=None, response_schema=None, temperature=0.0, max_tokens=5)
        text = (resp.get("text") or "").strip()
        return "OK", f"{cfg['model']}  ->  '{text[:20]}'  ({time.time() - t0:.1f}s)"
    except Exception as e:  # noqa: BLE001
        return "FAIL", f"{cfg['model']}: {type(e).__name__}: {str(e)[:160]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cfg", default="configs/model_config.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    mc = yaml.safe_load(Path(a.model_cfg).read_text(encoding="utf-8"))
    seen, failures = set(), 0
    for section in ("agents", "patient_simulator_backbones", "scorers"):
        for name, cfg in mc.get(section, {}).items():
            if a.only and name not in a.only:
                continue
            key = (cfg["provider"], cfg["model"], cfg.get("base_url_env", ""))
            if key in seen:
                continue
            seen.add(key)
            status, msg = check(name, cfg)
            failures += status == "FAIL"
            print(f"[{status:4s}] {section:28s} {name:20s} {msg}")
    print(f"\n{len(seen)} distinct models checked, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
