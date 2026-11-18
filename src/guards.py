"""Deterministic safety layers that run BEFORE and AFTER the LLM.

Pre-LLM  : prompt-injection / planted-fact, guarantee, forecast, personal-advice, out-of-range year.
Post-LLM : citation validity + numeric grounding (every number in the answer must be in a cited source).
"""
import re

INJECTION = [
    r"\bignore\b.{0,40}\b(document|source|context|instruction|rule|prompt)s?\b",
    r"\b(disregard|forget|override)\b.{0,40}\b(document|source|instruction|rule|prompt)s?\b",
    r"\b(pretend|act as|you are now|system prompt|jailbreak)\b",
    r"\b(tell|say|state|write|answer|confirm)\b.{0,30}\b(cut-?off|closing rank|rank|seats?)\b.{0,15}\b(is|was|=|as)\b\s*\d",
    r"\b(cut-?off|closing rank)\s*(is|was|=)\s*\d+\b.{0,20}\b(not|ignore)\b",
]
GUARANTEE = [
    r"\bguarantee", r"\bdefinitely\b.{0,15}\b(get|secure|admit)", r"\bwill i (surely |definitely |certainly )?(get|secure|be admitted|crack|make it)\b",
    r"\bam i (sure|certain) to\b", r"\b(sure|certain) (shot|to get)\b", r"\bpromise\b",
]
FORECAST = [r"\bnext year\b", r"\bupcoming\b", r"\bfuture\b", r"\bpredict(ed|ion)?\b", r"\bforecast", r"\bexpected cut-?off", r"\bwill be the cut-?off\b",
            r"\bthis year'?s? (cut-?off|closing)"]
ADVICE = [r"\bshould i (choose|join|take|go|pick|opt|prefer|select)\b", r"\bwhich (one )?(is|would be) (better|best) for me\b",
          r"\bis it (worth|wise|better) (to )?(choos|join|tak)", r"\bwhat should i do\b"]

MSG = {
    "injection": ("I can't accept figures or instructions supplied inside the question as facts. I only report values "
                  "that appear in the indexed sources. Ask for a specific college, course, category and year and I'll look it up."),
    "guarantee": ("I can't guarantee or promise any admission outcome. I can show published closing ranks and list options "
                  "whose past cut-offs are near a given rank (use the predictor)."),
    "forecast": ("The sources only contain published closing ranks for {years}. I can't forecast future cut-offs; "
                 "past values change from year to year."),
    "year_range": "The sources cover only {years}; there is no data for {year}, so I can't give a figure for it.",
    "advice": ("I can't tell you which college to choose: that depends on your priorities, finances and a counsellor's guidance. "
               "What I can do is list options whose past closing ranks are near your rank (use the predictor with your rank, "
               "category and home state). Limits: the data is synthetic, closing ranks change every year, and nothing here guarantees a seat."),
}


def _any(patterns, q):
    return any(re.search(p, q, re.I | re.S) for p in patterns)


def policy_check(q, years):
    """Return (kind, status) or None. status is 'refused' or 'limits'."""
    if _any(INJECTION, q):
        return "injection", "refused"
    if _any(GUARANTEE, q):
        return "guarantee", "refused"
    ys = [int(y) for y in re.findall(r"\b(20\d\d)\b", q)]
    bad = [y for y in ys if y not in years]
    if bad:
        return "year_range", "refused"
    if _any(FORECAST, q):
        return "forecast", "refused"
    if _any(ADVICE, q):
        return "advice", "limits"
    return None


def policy_message(kind, q, years):
    ys = [int(y) for y in re.findall(r"\b(20\d\d)\b", q) if int(y) not in years]
    return MSG[kind].format(years=f"{years[0]}-{years[-1]}", year=ys[0] if ys else "")


def _nums(text):
    return {n.replace(",", "") for n in re.findall(r"\d+(?:,\d{3})*", text)}


def check_grounding(answer, sources):
    """sources: list of Chunk in the order they were labelled S1..Sn. Returns (cited_indexes, errors)."""
    cites = [int(x) for x in re.findall(r"\[S(\d+)\]", answer)]
    errors = []
    if not cites:
        errors.append("no_citation")
    if any(not 1 <= c <= len(sources) for c in cites):
        errors.append("invalid_citation")
    valid = [c for c in cites if 1 <= c <= len(sources)]
    cited_text = " ".join(sources[c - 1].text for c in set(valid))
    allowed = _nums(cited_text)
    body = re.sub(r"\[S\d+\]", "", answer)
    ungrounded = [n for n in _nums(body) if n not in allowed and int(n) > 10]
    if ungrounded:
        errors.append("ungrounded_numbers:" + ",".join(sorted(ungrounded)))
    return sorted(set(valid)), errors
