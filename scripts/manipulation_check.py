"""Step 6: Manipulation check on simulator variants.

Computes NLP metrics per patient utterance:
  - words_per_turn: mean word count of patient turns
  - facts_per_turn: count of medical entity tokens (simple heuristic)
  - flesch_kincaid_grade: FK grade level of patient turns

Outputs: outputs/tables/T_manipulation_check.csv and a figure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.9)

MEDICAL_TERMS = {
    "pain", "pressure", "chest", "arm", "heart", "breath", "shortness",
    "sweating", "dizzy", "nausea", "fever", "cough", "blood", "medication",
    "pill", "diabetes", "hypertension", "cholesterol", "troponin", "ecg",
    "doctor", "hospital", "surgery", "stroke", "radiate", "crushing",
}


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and not word.endswith('le'):
        count = max(1, count - 1)
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0.0
    avg_words = len(words) / max(1, len(sentences))
    avg_syllables = sum(_count_syllables(w) for w in words) / max(1, len(words))
    return 0.39 * avg_words + 11.8 * avg_syllables - 15.59


def _count_medical_facts(text: str) -> int:
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    return len(tokens & MEDICAL_TERMS)


def extract_metrics(trajectories_dir: Path) -> pd.DataFrame:
    rows = []
    for tf in sorted(trajectories_dir.glob("*.json")):
        traj = json.loads(tf.read_text(encoding="utf-8"))
        sim = traj.get("simulator", "")
        if sim == "full_info_static":
            continue
        for visit in traj.get("transcript", []):
            for turn in visit.get("dialogue", []):
                if turn.get("speaker") != "patient":
                    continue
                text = turn.get("text", "")
                words = len(text.split())
                facts = _count_medical_facts(text)
                fk = flesch_kincaid_grade(text)
                rows.append({
                    "simulator": sim,
                    "case_id": traj["case_id"],
                    "words_per_turn": words,
                    "facts_per_turn": facts,
                    "fk_grade": fk,
                })
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["words_per_turn", "facts_per_turn", "fk_grade"]
    labels = ["Words / turn", "Medical facts / turn", "Flesch-Kincaid grade"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = ["cooperative", "sparse", "verbose", "low_health_literacy"]
    order = [s for s in order if s in df["simulator"].unique()]

    for ax, metric, label in zip(axes, metrics, labels):
        sns.boxplot(data=df, x="simulator", y=metric, order=order, ax=ax, palette="Set2")
        ax.set_xlabel("Simulator")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Manipulation check: patient utterance metrics by simulator variant")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "F_manipulation_check.png", dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / "F_manipulation_check.pdf", bbox_inches="tight")
    plt.close()
    print(f"[manipulation_check] Figure saved to {out_dir}/F_manipulation_check.png")


def run(trajectories_dir: Path, out_dir: Path) -> None:
    df = extract_metrics(trajectories_dir)
    print(f"[manipulation_check] {len(df)} patient turns extracted")

    summary = (
        df.groupby("simulator")[["words_per_turn", "facts_per_turn", "fk_grade"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    print(summary.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "T_manipulation_check.csv", index=False)
    make_figure(df, out_dir / ".." / "figures")
    print(f"[manipulation_check] Table saved to {out_dir}/T_manipulation_check.csv")


if __name__ == "__main__":
    run(
        trajectories_dir=Path("outputs/trajectories"),
        out_dir=Path("outputs/tables"),
    )
