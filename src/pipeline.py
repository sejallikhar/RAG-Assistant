"""End-to-end question answering with layered refusal."""
import json
import re
import time
from dataclasses import dataclass, field, asdict

from . import config
from .generate import LLMUnavailable, call_ollama, extractive
from .guards import check_grounding, policy_check, policy_message
from .retrieve import Index

_index = None


def get_index():
    global _index
    if _index is None:
        _index = Index()
    return _index


@dataclass
class Result:
    status: str                      # answered | refused | limits | clarify | error
    answer: str
    reasons: list = field(default_factory=list)
    citations: list = field(default_factory=list)   # [{label, id, text}]
    backend: str = ""


def _log(question, res):
    config.LOG_DIR.mkdir(exist_ok=True)
    with open(config.LOG_DIR / "events.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": question, "status": res.status,
                            "reasons": res.reasons, "sources": [c["id"] for c in res.citations],
                            "backend": res.backend}) + "\n")


def _finish(q, res, log=True):
    if log:
        _log(q, res)
    return res


def answer(question, log=True, backend=None):
    backend = backend or config.LLM_BACKEND
    idx = get_index()
    mk = lambda status, ans, reasons, cites=(), : Result(status, ans, list(reasons), list(cites), backend)

    # 1. policy layer (injection, guarantees, forecasts, personal advice, out-of-range year)
    pol = policy_check(question, idx.years)
    if pol:
        kind, status = pol
        return _finish(question, mk(status, policy_message(kind, question, idx.years), [kind]), log)

    p = idx.parse(question)

    # 2. entity layer: things the index cannot contain
    if p["unknown_college"]:
        return _finish(question, mk("refused", f"Not in the sources: '{p['unknown_college']}' does not appear in the indexed data.",
                                    ["unknown_college"]), log)
    if p["unsupported_course"]:
        return _finish(question, mk("refused", f"Not in the sources: the data covers only MBBS and BDS, not '{p['unsupported_course'].upper()}'.",
                                    ["unsupported_course"]), log)

    # 3. relevance gate: content words that exist nowhere in the corpus
    unsupported, content = idx.unsupported_terms(question, p)
    if content and (len(unsupported) / len(content) >= 0.5 or (unsupported and not p["colleges"])):
        return _finish(question, mk("refused", "Not in the sources: nothing in the indexed data covers "
                                    + ", ".join(f"'{t}'" for t in unsupported) + ".", ["no_relevant_sources"]), log)
    if not content and not p["colleges"]:
        return _finish(question, mk("refused", "Not in the sources: I couldn't tell what to look up.", ["empty_query"]), log)

    # 4. retrieval
    wants_seats = bool(re.search(r"\bseats?\b|seat matrix|intake", question, re.I))
    wants_rank = bool(re.search(r"cut-?off|closing|rank", question, re.I))
    if p["colleges"]:
        rows = idx.structured(p, wants_seats and not wants_rank)
        if not rows:
            return _finish(question, mk("refused", "Not in the sources: no row matches that combination of college, course, category, year and quota.",
                                        ["no_matching_rows"]), log)
        if len(rows) > config.MAX_ROWS:
            return _finish(question, mk("clarify", f"That matches {len(rows)} rows. Please specify the course, category, year and/or quota so I can look up an exact figure.",
                                        ["too_many_rows"]), log)
        sources = rows
    else:
        if re.search(r"\b(which|list|top|lowest|highest|cheapest|compare|rank(ing)? of)\b.*\bcolleges?\b", question, re.I):
            return _finish(question, mk("limits", "I can't rank or list colleges across the whole table here, because retrieval returns only a few rows and could mislead you. "
                                        "Use the predictor with your rank, category and home state for a structured shortlist.", ["aggregate_query"]), log)
        hits = idx.hybrid(question, source="faq" if not (wants_rank or wants_seats) else None)
        hits = [(c, s) for c, s in hits if s >= config.MIN_BM25]
        if not hits:
            return _finish(question, mk("refused", "Not in the sources: retrieval found nothing relevant enough to answer from.", ["low_retrieval_score"]), log)
        sources = [c for c, _ in hits[:4]]

    # 5. generation
    try:
        raw = extractive(sources) if backend == "extractive" else call_ollama(question, sources)
    except LLMUnavailable as e:
        return _finish(question, mk("error", str(e), ["llm_unavailable"]), log)
    if raw.strip().upper().startswith("NOT_IN_SOURCES"):
        return _finish(question, mk("refused", "Not in the sources: the retrieved material does not support an answer.", ["model_said_not_in_sources"]), log)

    # 6. post-check: citations valid + numbers grounded
    cited, errors = check_grounding(raw, sources)
    if errors:
        return _finish(question, mk("refused", "I couldn't produce an answer that is fully backed by the sources, so I'm declining rather than guessing.",
                                    ["grounding_check_failed"] + errors), log)
    cites = [{"label": f"S{i}", "id": sources[i - 1].id, "text": sources[i - 1].text} for i in cited]
    return _finish(question, mk("answered", raw, [], cites), log)
