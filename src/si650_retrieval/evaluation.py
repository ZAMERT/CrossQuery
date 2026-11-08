"""Binary duplicate-label metrics with all requested queries in the denominator."""
import math
from collections import defaultdict
from statistics import mean


def evaluate(queries, qrels, rankings):
    expected = {q["query_id"] for q in queries}
    if not expected:
        raise ValueError("No evaluation queries")
    if set(rankings) - expected:
        raise ValueError("Run contains queries outside the evaluation set")
    relevant = defaultdict(set)
    for qid, did, grade in qrels:
        if grade > 0:
            relevant[qid].add(did)
    rows = []
    for q in queries:
        qid = q["query_id"]
        positive = relevant[qid]
        if not positive:
            raise ValueError(f"No positive qrels for {qid}")
        ids = [hit[0] for hit in rankings.get(qid, [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate documents in run: {qid}")
        dcg = sum(1 / math.log2(i + 2) for i, did in enumerate(ids[:10]) if did in positive)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(positive))))
        rr = next((1 / (i + 1) for i, did in enumerate(ids[:10]) if did in positive), 0.0)
        rows.append({"query_id": qid, "forum": q["forum"], "nDCG@10": dcg / ideal,
                     "MRR@10": rr, "Recall@100": len(set(ids[:100]) & positive) / len(positive)})
    metrics = ("nDCG@10", "MRR@10", "Recall@100")
    per_forum = {forum: {m: mean(r[m] for r in rows if r["forum"] == forum) for m in metrics}
                 for forum in sorted({q["forum"] for q in queries})}
    return {"query_count": len(rows), "per_forum": per_forum,
            "macro": {m: mean(v[m] for v in per_forum.values()) for m in metrics}}, rows
