"""Run the evaluation and print/write a score table.

  python -m eval.run_eval                       # uses LLM_BACKEND (default: ollama)
  LLM_BACKEND=extractive python -m eval.run_eval   # no LLM; tests retrieval + guards only
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from src import config
from src.pipeline import answer
from src.predictor import predict

HERE = Path(__file__).parent
NOT_REFUSAL = {"answered"}


def norm(s):
    return re.sub(r"(?<=\d),(?=\d{3})", "", str(s)).lower()


def run_rag():
    rows = []
    for line in open(HERE / "questions.jsonl"):
        q = json.loads(line)
        t0 = time.time()
        r = answer(q["question"], log=False)
        dt = time.time() - t0
        exp = q["expect"]
        ok_status = (r.status == "answered") if exp == "answer" else (r.status in {"refused"}) if exp == "refuse" else (r.status == "limits")
        txt = norm(r.answer)
        contains = all(norm(x) in txt for x in q.get("must_contain", []))
        forbidden = not any(norm(x) in txt for x in q.get("must_not_contain", []))
        cite_ok = (len(r.citations) > 0) if exp == "answer" else True
        passed = ok_status and (contains and cite_ok if exp == "answer" else True) and forbidden
        rows.append({"id": q["id"], "expect": exp, "got": r.status, "reasons": ",".join(r.reasons), "pass": passed,
                     "contains": contains, "cites": cite_ok, "secs": round(dt, 2), "q": q["question"], "a": r.answer})
    return rows


def run_predictor():
    cases = [
        ("P1", "ok", dict(rank=6000, category="General", state="West Bengal"), "returns options, each with source rows"),
        ("P2", "refused", dict(rank=0, category="General", state="West Bengal"), "invalid rank"),
        ("P3", "refused", dict(rank=6000, category="General", state="West Bengal", year=2026), "asks for an unseen year: must not invent"),
        ("P4", "refused", dict(rank=6000, category="Martian", state="West Bengal"), "invalid category"),
        ("P5", "empty", dict(rank=1_900_000, category="General", state="West Bengal"), "rank beyond every cut-off: no guessing"),
        ("P6", "ok", dict(rank=60000, category="OBC", state="Kerala", course="BDS"), "BDS path"),
    ]
    rows = []
    for cid, exp, kw, note in cases:
        res = predict(**kw)
        if exp == "ok":
            good = res["status"] == "ok" and res["options"] and all(o["sources"] for o in res["options"])
        elif exp == "refused":
            good = res["status"] == "refused"
        else:
            good = res["status"] == "ok" and not res["options"]
        rows.append({"id": cid, "expect": exp, "got": res["status"] + f"({len(res['options'])})", "pass": bool(good), "note": note})
    return rows


def pct(n, d):
    return f"{n}/{d} = {100*n/d:.0f}%" if d else "n/a"


def main():
    rag, pred = run_rag(), run_predictor()
    ans = [r for r in rag if r["expect"] == "answer"]
    ref = [r for r in rag if r["expect"] == "refuse"]
    lim = [r for r in rag if r["expect"] == "limits"]
    refused = [r for r in rag if r["got"] == "refused"]
    out = []
    P = out.append
    P(f"# Evaluation results\n\nBackend: `{config.LLM_BACKEND}`" + (f"  Model: `{config.LLM_MODEL}`" if config.LLM_BACKEND == "ollama" else "  (no LLM: retrieval + guards only)"))
    P("\n## Score table\n")
    P("| Metric | Result |\n|---|---|")
    P(f"| Answerable questions answered correctly with citations | {pct(sum(r['pass'] for r in ans), len(ans))} |")
    P(f"| Should-refuse questions refused (refusal recall) | {pct(sum(r['pass'] for r in ref), len(ref))} |")
    P(f"| Refusals that were expected (refusal precision) | {pct(sum(r['expect']=='refuse' for r in refused), len(refused))} |")
    P(f"| 'Limits' questions handled without advice/guarantee | {pct(sum(r['pass'] for r in lim), len(lim))} |")
    P(f"| Answers whose citations are all valid | {pct(sum(r['cites'] for r in ans), len(ans))} |")
    P(f"| Predictor checks passed | {pct(sum(r['pass'] for r in pred), len(pred))} |")
    P(f"| Overall | {pct(sum(r['pass'] for r in rag+pred), len(rag+pred))} |")
    P("\n## Per-question detail\n")
    P("| ID | Expected | Got | Pass | Reasons | Question |\n|---|---|---|---|---|---|")
    for r in rag:
        P(f"| {r['id']} | {r['expect']} | {r['got']} | {'PASS' if r['pass'] else '**FAIL**'} | {r['reasons']} | {r['q'][:90]} |")
    for r in pred:
        P(f"| {r['id']} | {r['expect']} | {r['got']} | {'PASS' if r['pass'] else '**FAIL**'} | | predictor: {r['note']} |")
    fails = [r for r in rag if not r["pass"]]
    if fails:
        P("\n## Failures\n")
        for r in fails:
            P(f"- **{r['id']}** expected `{r['expect']}`, got `{r['got']}`: {r['a'][:200]}")
    text = "\n".join(out)
    (HERE / f"results_{config.LLM_BACKEND}.md").write_text(text + "\n")
    print(text)
    sys.exit(0)


if __name__ == "__main__":
    main()
