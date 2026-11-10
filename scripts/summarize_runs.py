"""Collect runs/dev/*/metrics.json into a table and run key paired comparisons.

python scripts/summarize_runs.py            # prints markdown, writes results/dev_summary.md
"""
import json
import random
from pathlib import Path
from statistics import mean

RUNS = Path("runs/dev")
VARIANTS = Path("data/query_variants.jsonl")
ORDER = ["random", "bm25_en", "dense_en", "bm25_mt_full", "bm25_mt_mixed", "dense_zh_full",
         "dense_zh_mixed", "dense_mt_full", "dense_mt_mixed", "hybrid_full", "hybrid_mixed"]
ROLE = {"random": "Baseline (mean of 5 seeds)", "bm25_en": "Reference (EN)", "dense_en": "Reference (EN)",
        "bm25_mt_full": "Translate + BM25", "bm25_mt_mixed": "Translate + BM25",
        "dense_zh_full": "Direct CLIR", "dense_zh_mixed": "Direct CLIR",
        "dense_mt_full": "Translate + dense", "dense_mt_mixed": "Translate + dense",
        "hybrid_full": "Hybrid RRF", "hybrid_mixed": "Hybrid RRF"}
# (A, B, question, restrict to queries whose zh_full != zh_mixed)
PAIRS = [("dense_zh_full", "dense_mt_full", "RQ1: direct vs translate (dense, ZH-full)", False),
         ("dense_zh_mixed", "dense_mt_mixed", "RQ1: direct vs translate (dense, ZH-mixed)", False),
         ("dense_zh_full", "dense_zh_mixed", "RQ2: ZH-full vs ZH-mixed (dense)", True),
         ("bm25_mt_full", "bm25_mt_mixed", "RQ2: ZH-full vs ZH-mixed (translate + BM25)", True),
         ("hybrid_full", "dense_zh_full", "Hybrid vs direct dense (ZH-full)", False),
         ("hybrid_full", "bm25_mt_full", "Hybrid vs translate + BM25 (ZH-full)", False)]
METRICS = ("nDCG@10", "MRR@10", "Recall@100")


def load(name):
    if name == "random":
        seeds = sorted(RUNS.glob("random_s*/metrics.json"))
        if not seeds:
            return None
        ms = [json.loads(p.read_text()) for p in seeds]
        return {"macro": {m: mean(x["macro"][m] for x in ms) for m in METRICS},
                "per_forum": {f: {m: mean(x["per_forum"][f][m] for x in ms) for m in METRICS}
                              for f in ms[0]["per_forum"]}}
    p = RUNS / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def per_query(name):
    p = RUNS / name / "per_query.jsonl"
    if not p.exists():
        return None
    return {r["query_id"]: r["nDCG@10"] for r in map(json.loads, p.read_text().splitlines()) if r}


def paired_test(a, b, trials=10000, seed=650):
    """Two-sided paired randomization (sign-flip) test on per-query differences."""
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    hits = sum(abs(mean(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12
               for _ in range(trials))
    return mean(diffs), (hits + 1) / (trials + 1)


def main():
    lines = ["| Run | Role | nDCG@10 | MRR@10 | R@100 | android | programmers | unix |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        m = load(name)
        if m is None:
            continue
        pf = m["per_forum"]
        lines.append(f"| {name} | {ROLE[name]} | " + " | ".join(f"{m['macro'][k]:.3f}" for k in METRICS)
                     + " | " + " | ".join(f"{pf[f]['nDCG@10']:.3f}" for f in ("android", "programmers", "unix")) + " |")
    lines.append("\nPer-forum columns are nDCG@10. Macro = mean over the three forums.\n")

    differ = None
    if VARIANTS.exists():
        rows = [json.loads(l) for l in VARIANTS.read_text(encoding="utf-8").splitlines() if l.strip()]
        differ = {r["query_id"] for r in rows if r["zh_full"] != r["zh_mixed"]}
        lines.append(f"zh_mixed differs from zh_full for {len(differ)}/{len(rows)} dev queries.\n")

    lines += ["| Comparison | Queries | mean ΔnDCG@10 (A−B) | p (paired randomization) |", "|---|---|---|---|"]
    for a, b, label, restrict in PAIRS:
        qa, qb = per_query(a), per_query(b)
        if qa is None or qb is None:
            continue
        ids = sorted(set(qa) & set(qb))
        if restrict and differ is not None:
            ids = [i for i in ids if i in differ]
        if not ids:
            continue
        d, p = paired_test([qa[i] for i in ids], [qb[i] for i in ids])
        lines.append(f"| {label}: {a} vs {b} | {len(ids)} | {d:+.3f} | {p:.4f} |")
    lines.append("\nQuery-level, pooled over forums; p-values not yet Holm-corrected.")

    text = "\n".join(lines)
    print(text)
    Path("results").mkdir(exist_ok=True)
    Path("results/dev_summary.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
