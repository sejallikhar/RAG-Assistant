"""Track A: rank -> safe / target / reach options.

Deterministic on purpose: code filters and buckets rows from the cutoff table; no LLM chooses colleges.
Every option carries the row ids it was built from, so each line is checkable.
"""
from collections import defaultdict

from .pipeline import get_index

CANON = {"general": "General", "gen": "General", "open": "General", "ews": "EWS", "obc": "OBC", "sc": "SC", "st": "ST"}
SAFE, TARGET, REACH = 0.85, 1.00, 1.15   # rank / last-year closing rank
MAX_RANK = 2_000_000
DISCLAIMER = ("Based on SYNTHETIC past closing ranks. Not a forecast or guarantee: cut-offs change every year. "
              "Use a counsellor for real decisions.")


def predict(rank, category, state, course="MBBS", year=None, per_bucket=6):
    idx = get_index()
    latest = max(idx.years)
    year = year or latest
    refuse = lambda why: {"status": "refused", "reason": why, "options": [], "disclaimer": DISCLAIMER}

    if not isinstance(rank, int) or rank <= 0 or rank > MAX_RANK:
        return refuse(f"Rank must be a whole number between 1 and {MAX_RANK:,}.")
    cat = CANON.get(str(category).strip().lower())
    if not cat:
        return refuse(f"Unknown category '{category}'. Use one of: General, EWS, OBC, SC, ST.")
    course = str(course).upper()
    if course not in {"MBBS", "BDS"}:
        return refuse(f"Course '{course}' is not in the data (only MBBS and BDS).")
    if year not in idx.years:
        return refuse(f"The data covers {idx.years[0]}-{idx.years[-1]}. I can't predict or invent a cut-off for {year}.")

    series = defaultdict(dict)
    for c in idx.chunks:
        if c.source == "cutoffs" and c.meta["course"] == course and c.meta["category"] == cat:
            m = c.meta
            series[(m["college_id"], m["quota"])][m["year"]] = (m["closing_rank"], c.id)

    options = []
    for (cid, quota), yrs in series.items():
        info = idx.colleges[cid]
        if quota == "State" and info["state"].lower() != str(state).strip().lower():
            continue  # State quota only applies to home-state candidates
        if year not in yrs:
            continue
        closing, row_id = yrs[year]
        ratio = rank / closing
        bucket = "safe" if ratio <= SAFE else "target" if ratio <= TARGET else "reach" if ratio <= REACH else None
        if not bucket:
            continue
        trend = ", ".join(f"{y}: {yrs[y][0]}" for y in sorted(yrs))
        why = {"safe": f"your rank {rank} is well inside the {year} closing rank {closing}",
               "target": f"your rank {rank} is within the {year} closing rank {closing}, but close to it",
               "reach": f"your rank {rank} is slightly beyond the {year} closing rank {closing}; possible only if cut-offs loosen"}[bucket]
        options.append({"bucket": bucket, "college": info["name"], "college_id": cid, "state": info["state"],
                        "type": info["type"], "course": course, "quota": quota, "category": cat, "year": year,
                        "closing_rank": closing, "why": why, "trend": trend,
                        "sources": [yrs[y][1] for y in sorted(yrs)]})

    order = {"safe": 0, "target": 1, "reach": 2}
    options.sort(key=lambda o: (order[o["bucket"]], o["closing_rank"]))
    kept, counts = [], defaultdict(int)
    for o in options:
        if counts[o["bucket"]] < per_bucket:
            kept.append(o)
            counts[o["bucket"]] += 1
    note = None
    if not kept:
        note = "No option in the data is within range of this rank. I won't guess beyond the table."
    elif str(state).strip().lower() not in {s.lower() for s in idx.states}:
        note = f"'{state}' is not a state in the data, so only All India quota options are shown."
    return {"status": "ok", "options": kept, "note": note, "disclaimer": DISCLAIMER,
            "input": {"rank": rank, "category": cat, "state": state, "course": course, "year": year}}


def format_result(res):
    if res["status"] == "refused":
        return f"REFUSED: {res['reason']}"
    out = []
    for b in ("safe", "target", "reach"):
        rows = [o for o in res["options"] if o["bucket"] == b]
        if rows:
            out.append(f"\n== {b.upper()} ==")
            for o in rows:
                out.append(f"- {o['college']} ({o['state']}, {o['type']}) | {o['course']} | {o['quota']} quota\n"
                           f"    why: {o['why']}\n    trend: {o['trend']}\n    sources: {', '.join(o['sources'])}")
    if res.get("note"):
        out.append("\nNOTE: " + res["note"])
    out.append("\n" + res["disclaimer"])
    return "\n".join(out)
