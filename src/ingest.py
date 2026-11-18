"""Build the index: one chunk per CSV row, one chunk per FAQ section.

Usage: python -m src.ingest
Writes index/chunks.jsonl (+ index/embeddings.npy if USE_DENSE=1).
"""
import csv
import json
import re

from .config import DATA_DIR, INDEX_DIR, USE_DENSE, EMBED_MODEL


def cutoff_chunks():
    with open(DATA_DIR / "cutoffs.csv", newline="") as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):  # line 1 is the header
            text = (f"Closing rank | college: {r['college']} ({r['college_id']}) | state: {r['state']} | "
                    f"type: {r['type']} | course: {r['course']} | quota: {r['quota']} | "
                    f"category: {r['category']} | year: {r['year']} | closing rank: {r['closing_rank']}")
            yield {"id": f"cutoffs.csv#L{line_no}", "source": "cutoffs", "text": text,
                   "meta": {**r, "year": int(r["year"]), "closing_rank": int(r["closing_rank"])}}


def seat_chunks():
    with open(DATA_DIR / "seat_matrix.csv", newline="") as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):
            text = (f"Seat matrix | college: {r['college']} ({r['college_id']}) | state: {r['state']} | "
                    f"type: {r['type']} | course: {r['course']} | quota: {r['quota']} | "
                    f"year: {r['year']} | seats: {r['seats']}")
            yield {"id": f"seat_matrix.csv#L{line_no}", "source": "seat_matrix", "text": text,
                   "meta": {**r, "year": int(r["year"]), "seats": int(r["seats"])}}


def faq_chunks():
    md = (DATA_DIR / "counselling_faq.md").read_text()
    for sec in re.split(r"^## ", md, flags=re.M)[1:]:
        title, _, body = sec.partition("\n")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        yield {"id": f"counselling_faq.md#{slug}", "source": "faq",
               "text": f"{title.strip()}: {body.strip()}", "meta": {"title": title.strip()}}


def build():
    INDEX_DIR.mkdir(exist_ok=True)
    chunks = list(cutoff_chunks()) + list(seat_chunks()) + list(faq_chunks())
    with open(INDEX_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"indexed {len(chunks)} chunks -> {INDEX_DIR/'chunks.jsonl'}")
    if USE_DENSE:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
        np.save(INDEX_DIR / "embeddings.npy", emb)
        print(f"dense embeddings ({EMBED_MODEL}) -> {INDEX_DIR/'embeddings.npy'}")


if __name__ == "__main__":
    build()
