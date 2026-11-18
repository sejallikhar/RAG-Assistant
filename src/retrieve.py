"""Query parsing + retrieval.

Two paths:
  1. structured: when the question names a known college, filter the table rows exactly
     (college / course / category / year / quota). Exact rows beat fuzzy search for tables.
  2. hybrid: BM25 (+ optional dense embeddings, fused with RRF) for FAQ-style questions.
"""
import difflib
import json
import math
import re
from collections import Counter, defaultdict

from .config import INDEX_DIR, USE_DENSE, EMBED_MODEL, TOP_K

STOP = set("""a an the of in on at for to from is are was were be been am i me my we you your it its this that these
those and or but if then than so do does did can could should would will shall may might what whats which who whom
how when where why there here about tell please show give get with without by as any some all each every
many much more most also just only not no yes kindly want need know like help
mean means meaning difference between described describe explain define defined definition lowest highest cheapest
compare compared""".split())
GENERIC = set("""cut off cutoff cutoffs closing rank ranks college colleges course courses seat seats medical institute
sciences university data quota category year years published source sources table value number""".split())

CATS = {"general": "General", "ews": "EWS", "obc": "OBC", "sc": "SC", "st": "ST"}
COURSES = {"mbbs": "MBBS", "bds": "BDS"}
UNSUPPORTED_COURSES = {"md", "ms", "mds", "dm", "mch", "bams", "bhms", "bums", "nursing", "pharmacy",
                       "physiotherapy", "diploma"}
SUFFIX_RE = re.compile(r"((?:[\w'-]+\s+){1,2})(medical college|institute of medical sciences|medical university|"
                       r"dental college|medical institute|college of medicine|hospital)\b", re.I)
NAME_STOP = STOP | {"private", "government", "deemed", "top", "best", "other", "state", "central", "new", "good"}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class Chunk:
    def __init__(self, d):
        self.id, self.source, self.text, self.meta = d["id"], d["source"], d["text"], d["meta"]
        self.tokens = tokenize(self.text)


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, q_tokens):
        out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in set(q_tokens):
                f = self.tf[i].get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class Index:
    def __init__(self):
        path = INDEX_DIR / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError("Index not built. Run: python -m src.ingest")
        self.chunks = [Chunk(json.loads(l)) for l in open(path)]
        self.bm25 = BM25([c.tokens for c in self.chunks])
        self.vocab = set(t for c in self.chunks for t in c.tokens)
        # separate statistics for prose: table rows would otherwise swamp the FAQ's IDF
        self.faq_ids = [i for i, c in enumerate(self.chunks) if c.source == "faq"]
        self.bm25_faq = BM25([self.chunks[i].tokens for i in self.faq_ids])
        # catalog derived from the cutoff table
        self.colleges = {}
        for c in self.chunks:
            if c.source == "cutoffs":
                m = c.meta
                self.colleges.setdefault(m["college_id"], {"id": m["college_id"], "name": m["college"],
                                                           "state": m["state"], "type": m["type"]})
        self.first_words = {i["name"].split()[0].lower(): i["id"] for i in self.colleges.values()}
        self.years = sorted({c.meta["year"] for c in self.chunks if c.source == "cutoffs"})
        self.states = sorted({i["state"] for i in self.colleges.values()})
        self.emb = self.model = None
        if USE_DENSE and (INDEX_DIR / "embeddings.npy").exists():
            import numpy as np
            from sentence_transformers import SentenceTransformer
            self.emb = np.load(INDEX_DIR / "embeddings.npy")
            self.model = SentenceTransformer(EMBED_MODEL)

    # ---------- query understanding ----------
    def parse(self, q):
        ql = q.lower()
        toks = set(re.findall(r"[a-z]+", ql))
        p = {"colleges": [], "category": None, "years": [], "course": None, "quota": None,
             "unknown_college": None, "unsupported_course": None}
        for k, v in CATS.items():
            if k in toks:
                p["category"] = v
        for k, v in COURSES.items():
            if k in toks:
                p["course"] = v
        p["unsupported_course"] = next((t for t in toks if t in UNSUPPORTED_COURSES), None) if not p["course"] else None
        p["years"] = sorted({int(y) for y in re.findall(r"\b(20\d\d)\b", q)})
        if "all india" in ql or "aiq" in toks:
            p["quota"] = "All India"
        elif "state quota" in ql:
            p["quota"] = "State"
        found = set(re.findall(r"syn-\d{3}", ql))
        found = {f.upper() for f in found if f.upper() in self.colleges}
        for w, cid in self.first_words.items():
            if w in toks:
                found.add(cid)
        p["colleges"] = sorted(found)
        if not found:  # "X Medical College" where X is not in the index?
            for m in SUFFIX_RE.finditer(q):
                words = [w for w in re.findall(r"[\w'-]+", m.group(1).lower()) if w not in NAME_STOP]
                if words and not any(difflib.get_close_matches(w, self.first_words, n=1, cutoff=0.85) for w in words):
                    p["unknown_college"] = " ".join(w.title() for w in words) + " " + m.group(2).title()
                    break
        return p

    def unsupported_terms(self, q, parsed):
        """Content words that appear nowhere in the corpus (and are not explained by a parsed entity)."""
        content = [t for t in tokenize(q) if t not in GENERIC and not t.isdigit()]
        return [t for t in content if t not in self.vocab and not difflib.get_close_matches(t, self.first_words, 1, 0.85)], content

    # ---------- retrieval ----------
    def structured(self, p, wants_seats):
        src = "seat_matrix" if wants_seats else "cutoffs"
        rows = []
        for c in self.chunks:
            if c.source != src or c.meta["college_id"] not in p["colleges"]:
                continue
            m = c.meta
            if p["course"] and m["course"] != p["course"]:
                continue
            if p["quota"] and m["quota"] != p["quota"]:
                continue
            if src == "cutoffs":
                if p["category"] and m["category"] != p["category"]:
                    continue
                if p["years"] and m["year"] not in p["years"]:
                    continue
            rows.append(c)
        return rows

    def hybrid(self, q, k=TOP_K, source=None):
        qt = tokenize(q)
        if source == "faq":
            local = self.bm25_faq.scores(qt)
            s = [0.0] * len(self.chunks)
            for j, i in enumerate(self.faq_ids):
                s[i] = local[j]
        else:
            s = self.bm25.scores(qt)
        order = [i for i in sorted(range(len(s)), key=lambda i: -s[i])
                 if s[i] > 0 and (source is None or self.chunks[i].source == source)][:50]
        ranked = {i: r for r, i in enumerate(order)}
        if self.emb is not None:
            qe = self.model.encode([q], normalize_embeddings=True)[0]
            sims = self.emb @ qe
            dense = [i for i in sorted(range(len(sims)), key=lambda i: -sims[i])
                     if source is None or self.chunks[i].source == source][:50]
            fused = defaultdict(float)
            for r, i in enumerate(order):
                fused[i] += 1 / (60 + r)
            for r, i in enumerate(dense):
                fused[i] += 1 / (60 + r)
            order = sorted(fused, key=lambda i: -fused[i])
        top = order[:k]
        return [(self.chunks[i], s[i]) for i in top]
