"""Generate factorial patient-simulator cards.

Four binary factors, each manipulated independently:
  v  verbosity     0 = terse (<= 15 words/turn)          1 = long-winded (<= 180 words/turn)
  d  disclosure    0 = forthcoming (volunteers relevant  1 = withholding (answers only the literal
                       symptom on open questions)             question, downplays, never volunteers)
  l  literacy      0 = plain adult English               1 = <= grade-6, lay terms, misunderstands ~1/3
  x  distractors   0 = none                              1 = 1-2 irrelevant facts per turn

Card name: fx_v{v}_d{d}_l{l}_x{x}. The 2^(4-1) resolution-IV fraction (x = v XOR d XOR l)
gives 8 cells in which every main effect is unconfounded with any 2-way interaction; the
full 16-cell design is also written for a follow-up run.

The existing named variants map approximately to:
  cooperative ~ v=mid d0 l0 x0 | sparse ~ v0 d1 l0 x0 | verbose ~ v1 d0 l0 x1 | low_health_literacy ~ v0.5 d0 l1 x0
which is why they cannot separate the factors.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

COMMON = [
    "Never reveal the underlying diagnosis or use clinical terms unless the clinician introduces them first.",
    "Never invent facts not in the case file; say you are not sure instead.",
    "Stay consistent across all visits.",
    "Never break character or mention you are an AI.",
]

V = {
    0: dict(rules=["Answer in 1-10 words. Short clauses. No elaboration."], voice="terse, clipped",
            verbosity="minimal", max_words=15),
    1: dict(rules=["Wrap each answer in storytelling and background context; the actual answer "
                   "comes after the first sentence.", "Use 3-6 sentences per turn."],
            voice="long-winded, meandering", verbosity="high", max_words=180),
}
D = {
    0: dict(rules=["Answer every direct question accurately and completely from the case file.",
                   "If asked an open-ended question, volunteer the single most relevant symptom "
                   "you have not yet mentioned."]),
    1: dict(rules=["Answer only the literal question asked; never volunteer information.",
                   "Downplay severity on first mention ('it's probably nothing') but do not deny "
                   "facts when asked directly.",
                   "If asked an open-ended question, name only the chief complaint."]),
}
L = {
    0: dict(rules=["Use plain adult English; medical terms allowed only if the clinician used them first."],
            vocab="plain English"),
    1: dict(rules=["Use lay descriptions only ('my chest feels heavy' not 'angina'; 'pee' not 'urine').",
                   "Confuse medication names ('the white pill', 'the blood pill').",
                   "Misunderstand the question about 1 in 3 turns and answer an adjacent question instead.",
                   "If asked a yes/no question containing a medical term, reply 'what's that mean?'"],
            vocab="<= Grade 6 reading level; no medical terms"),
}
X = {
    0: dict(rules=["Do not mention anything unrelated to the question."]),
    1: dict(rules=["Add 1-2 IRRELEVANT facts per turn (what you ate, work stress, a relative's "
                   "unrelated illness) drawn only from hidden_history social_context or generic "
                   "daily life; keep them short if you are otherwise terse."]),
}


def card(v: int, d: int, l: int, x: int) -> dict:  # noqa: E741
    name = f"fx_v{v}_d{d}_l{l}_x{x}"
    rules = V[v]["rules"] + D[d]["rules"] + L[l]["rules"] + X[x]["rules"]
    return {
        "variant": name,
        "display_name": f"Factorial cell v{v} d{d} l{l} x{x}",
        "factors": {"verbosity": v, "withholding": d, "low_literacy": l, "distractors": x},
        "voice": V[v]["voice"],
        "common_rules": COMMON,
        "variant_rules": rules,
        "vocabulary_constraint": L[l]["vocab"],
        "verbosity": V[v]["verbosity"],
        "emotional_affect": "guarded" if d else "calm",
        "max_words_per_turn": V[v]["max_words"],
    }


def fraction_cells() -> list[tuple[int, int, int, int]]:
    return [(v, d, l, v ^ d ^ l) for v, d, l in product((0, 1), repeat=3)]


def main(out_dir: Path = Path("simulator_cards")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names_frac, names_full = [], []
    for v, d, l, x in product((0, 1), repeat=4):  # noqa: E741
        c = card(v, d, l, x)
        (out_dir / f"{c['variant']}.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
        names_full.append(c["variant"])
        if (v, d, l, x) in fraction_cells():
            names_frac.append(c["variant"])
    print("fraction (8):", names_frac)
    print("full (16):", names_full)


if __name__ == "__main__":
    main()
