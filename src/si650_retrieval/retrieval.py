"""Retrievers share ranked (doc_id, score) lists; no component reads qrels."""
import math
import re
from collections import Counter, defaultdict

import numpy as np


def tokenize(text):
    # Portable baseline only: deliberately not presented as Lucene analysis.
    return re.findall(r"\w+", text.casefold())


def bm25_build(docs):
    postings = defaultdict(list)
    lengths = []
    for i, doc in enumerate(docs):
        counts = Counter(tokenize(doc["contents"]))
        lengths.append(sum(counts.values()))
        for token, count in counts.items():
            postings[token].append([i, count])
    return {"ids": [d["doc_id"] for d in docs], "lengths": lengths, "postings": dict(postings)}


def bm25_search(index, text, top_k, k1=0.9, b=0.4):
    n = len(index["ids"])
    avg = sum(index["lengths"]) / n if n else 0
    if not avg:
        return []
    scores = defaultdict(float)
    for term in set(tokenize(text)):
        hits = index["postings"].get(term, [])
        idf = math.log(1 + (n - len(hits) + 0.5) / (len(hits) + 0.5))
        for i, tf in hits:
            norm = k1 * (1 - b + b * index["lengths"][i] / avg)
            scores[i] += idf * tf * (k1 + 1) / (tf + norm)
    return sorted(((index["ids"][i], score) for i, score in scores.items()),
                  key=lambda hit: (-hit[1], hit[0]))[:top_k]


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite 2D embedding matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding vector")
    return np.ascontiguousarray(vectors / norms)


def rrf(left, right, alpha=0.5, c=60, top_k=1000):
    if not 0 <= alpha <= 1 or c <= 0 or top_k <= 0:
        raise ValueError("Invalid RRF alpha/c/top_k")
    scores = defaultdict(float)
    for ranking, weight in ((left, alpha), (right, 1 - alpha)):
        if not weight:
            continue
        seen = set()
        rank = 0
        for doc_id, _ in ranking:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] += weight / (c + rank)
    return sorted(scores.items(), key=lambda hit: (-hit[1], hit[0]))[:top_k]


def load_encoder(model, revision=None, device=None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install the dense extra: pip install -e '.[dense]'") from exc
    return SentenceTransformer(model, revision=revision, device=device, trust_remote_code=False)
