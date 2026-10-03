"""Pick one real trajectory per phenomenon and render short appendix examples.

Phenomena:
  1. sparse under-escalation   (primary, sparse, emergency case, predicted < truth)
  2. verbose leakage           (primary, verbose, leakage_level == 2, else 1)
  3. demographic disparity     (fairness pair with different visit-3 escalation)
  4. scorer disagreement       (largest |primary - alt| with pass/fail flip)
  5. coordination trace loss   (team trajectory with a role contribution < 50 chars)

Writes outputs/tables/qualitative_examples.tex (LaTeX) and outputs/tables/qualitative_examples.md.
Dialogue is excerpted (last N turns of the decisive visit) and truncated per turn.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.metrics import subsets as S
from src.metrics.compute_escalation_metrics import ORD
from src.metrics.compute_fairness import fairness_pairs

MAX_TURNS = 6
MAX_CHARS = 220


def _tex(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def _trunc(s: str, n: int = MAX_CHARS) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _load(traj_dir: Path, tid: str) -> dict | None:
    p = traj_dir / f"{tid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _visit(traj: dict, v: int) -> dict:
    for rec in traj.get("transcript", []):
        if rec.get("visit") == v:
            return rec
    return {}


def _excerpt(traj: dict, v: int = 3) -> list[tuple[str, str]]:
    d = _visit(traj, v).get("dialogue", [])[-MAX_TURNS:]
    return [(t.get("speaker", "?"), _trunc(t.get("text", ""))) for t in d]


def _assess(traj: dict, v: int = 3) -> dict:
    return _visit(traj, v).get("assessment", {}) or {}


def pick(df: pd.DataFrame, traj_dir: Path, tables_dir: Path) -> list[dict]:
    prim = S.primary(df)
    ex = []

    # 1 sparse under-escalation
    p = prim.copy(); p["gap"] = p["escalation_level"].map(ORD) - p["correct_escalation"].map(ORD)
    c = p[(p["simulator"] == "sparse") & (p["gap"] < 0)].sort_values("gap")
    c_em = c[c["correct_escalation"] == "emergency_911"]
    r = (c_em if len(c_em) else c).iloc[0] if len(c) else None
    if r is not None:
        ex.append({"title": "Sparse-patient under-escalation", "row": r, "kind": "single"})

    # 2 verbose leakage
    c = prim[(prim["simulator"] == "verbose") & (prim["leakage_level"] >= 1)].sort_values("leakage_level", ascending=False)
    if len(c):
        ex.append({"title": "Simulator leakage under the verbose patient", "row": c.iloc[0], "kind": "leak"})

    # 3 demographic disparity
    pairs = fairness_pairs(S.fairness(df))
    if not pairs.empty:
        pos = pairs[pairs["escalation_differs"] == 1]
        if len(pos):
            pr = pos.iloc[0]
            ex.append({"title": "Demographic counterfactual changes escalation", "pair": pr, "kind": "pair"})

    # 4 scorer disagreement
    dp = tables_dir / "T_scorer_agreement_detail.csv"
    if dp.exists():
        sd = pd.read_csv(dp)
        alt = "alt_normalized" if "alt_normalized" in sd else next(
            (c for c in sd.columns if c.endswith("_normalized") and c != "primary_normalized"), None)
        if alt:
            sd["flip"] = (sd["primary_normalized"] >= 0.5) != (sd[alt] >= 0.5)
            sd["d"] = (sd["primary_normalized"] - sd[alt]).abs()
            f = sd[sd["flip"]].sort_values("d", ascending=False)
            if len(f):
                r = df[df["trajectory_id"] == f.iloc[0]["trajectory_id"]].iloc[0]
                ex.append({"title": "Scorer disagreement on pass/fail", "row": r, "kind": "scorer",
                           "alt": (alt.replace("_normalized", ""), float(f.iloc[0][alt]))})

    # 5 coordination trace loss
    for tf in sorted(traj_dir.glob("*__team__*.json")):
        t = json.loads(tf.read_text(encoding="utf-8"))
        for rec in t.get("transcript", []):
            tr = (rec.get("assessment") or {}).get("trace", {}) or {}
            short = [k for k in ("clinician_contribution", "safety_checker_contribution",
                                 "pharm_nurse_contribution") if len(str(tr.get(k, ""))) < 50]
            if tr and short:
                r = df[df["trajectory_id"] == tf.stem]
                if len(r):
                    ex.append({"title": "Coordination trace loss in the care team", "row": r.iloc[0],
                               "kind": "ctl", "visit": rec.get("visit"), "short": short, "trace": tr})
                break
        if any(e["kind"] == "ctl" for e in ex):
            break
    return ex


def render(ex: list[dict], traj_dir: Path) -> tuple[str, str]:
    tex, md = [], []
    tex.append("% Generated by scripts/qualitative_examples.py -- do not edit by hand\n"
               "\\section{Qualitative Examples}\n\\label{app:examples}\n"
               "Excerpts are the last turns of the decisive visit, truncated; full trajectories are in the released data.\n")
    md.append("# Qualitative examples (auto-generated)\n")
    for i, e in enumerate(ex, 1):
        if e["kind"] == "pair":
            pr = e["pair"]
            b, v = _load(traj_dir, pr["trajectory_id_base"]), _load(traj_dir, pr["trajectory_id_var"])
            head = (f"{e['title']} ({pr['case_id']}, {pr['agent_model']}, {pr['agent_arch']}, "
                    f"{pr['simulator']}; baseline vs {pr['demographic_variant']})")
            ab, av = _assess(b), _assess(v)
            body_md = [f"- baseline escalation: `{ab.get('escalation_level')}`; variant escalation: `{av.get('escalation_level')}`",
                       f"- baseline plan: {_trunc('; '.join(ab.get('management_plan', [])[:3]))}",
                       f"- variant plan: {_trunc('; '.join(av.get('management_plan', [])[:3]))}",
                       f"- LLM judge: `{pr.get('disparity_severity', '')}`"]
            body_tex = [f"\\item Baseline escalation: \\texttt{{{_tex(str(ab.get('escalation_level')))}}}; "
                        f"variant escalation: \\texttt{{{_tex(str(av.get('escalation_level')))}}}.",
                        f"\\item Baseline plan: {_tex(_trunc('; '.join(ab.get('management_plan', [])[:3])))}",
                        f"\\item Variant plan: {_tex(_trunc('; '.join(av.get('management_plan', [])[:3])))}",
                        f"\\item LLM judge: \\texttt{{{_tex(str(pr.get('disparity_severity', '')))}}}."]
            excerpt = []
        else:
            r = e["row"]
            t = _load(traj_dir, r["trajectory_id"])
            head = f"{e['title']} ({r['case_id']}, {r['agent_model']}, {r['agent_arch']}, {r['simulator']})"
            a = _assess(t)
            body_md = [f"- ground-truth escalation: `{r['correct_escalation']}`; agent visit-3 escalation: `{r['escalation_level']}`",
                       f"- differential: {_trunc(', '.join(a.get('differential_diagnosis', [])[:4]))}",
                       f"- task score (primary scorer): {r['normalized']}; leakage level: {r['leakage_level']}"]
            body_tex = [f"\\item Ground-truth escalation: \\texttt{{{_tex(str(r['correct_escalation']))}}}; "
                        f"agent visit-3 escalation: \\texttt{{{_tex(str(r['escalation_level']))}}}.",
                        f"\\item Differential: {_tex(_trunc(', '.join(a.get('differential_diagnosis', [])[:4])))}",
                        f"\\item Task score (primary scorer): {r['normalized']}; leakage level: {r['leakage_level']}."]
            if e["kind"] == "scorer":
                body_md.append(f"- alternative scorer ({e['alt'][0]}): {e['alt'][1]:.3f} (pass/fail flips at 0.5)")
                body_tex.append(f"\\item Alternative scorer ({_tex(e['alt'][0])}): {e['alt'][1]:.3f} (pass/fail flips at 0.5).")
            if e["kind"] == "ctl":
                for k in e["short"]:
                    body_md.append(f"- degenerate role contribution `{k}` (visit {e['visit']}): \"{_trunc(str(e['trace'].get(k, '')), 80)}\"")
                    body_tex.append(f"\\item Degenerate role contribution \\texttt{{{_tex(k)}}} (visit {e['visit']}): ``{_tex(_trunc(str(e['trace'].get(k, '')), 80))}''")
            excerpt = _excerpt(t, 3) if e["kind"] != "ctl" else []
        md.append(f"## Example {i}: {head}\n")
        md += body_md
        if excerpt:
            md.append("\nDialogue excerpt (visit 3):\n")
            md += [f"> **{s.capitalize()}:** {x}" for s, x in excerpt]
        md.append("")
        tex.append(f"\\paragraph{{Example {i}: {_tex(head)}}}\n\\begin{{itemize}}\\setlength\\itemsep{{0pt}}")
        tex += body_tex
        tex.append("\\end{itemize}")
        if excerpt:
            tex.append("\\begin{quote}\\small")
            tex += [f"\\textbf{{{s.capitalize()}:}} {_tex(x)}\\\\" for s, x in excerpt]
            tex.append("\\end{quote}")
    return "\n".join(tex) + "\n", "\n".join(md) + "\n"


def run(scored_csv: Path, traj_dir: Path, tables_dir: Path, tex_out: Path, md_out: Path) -> None:
    df = pd.read_csv(scored_csv)
    ex = pick(df, traj_dir, tables_dir)
    tex, md = render(ex, traj_dir)
    tex_out.write_text(tex, encoding="utf-8")
    md_out.write_text(md, encoding="utf-8")
    print(f"[examples] {len(ex)} examples -> {tex_out}, {md_out}")
    for e in ex:
        print("  -", e["title"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="outputs/scored/all_scored.csv")
    ap.add_argument("--trajectories", default="outputs/trajectories")
    ap.add_argument("--tables", default="outputs/tables")
    ap.add_argument("--tex-out", default="outputs/tables/qualitative_examples.tex")
    ap.add_argument("--md-out", default="outputs/tables/qualitative_examples.md")
    a = ap.parse_args()
    run(Path(a.scored_csv), Path(a.trajectories), Path(a.tables), Path(a.tex_out), Path(a.md_out))
