"""Write one config per dev experiment into configs/dev/ (skips existing files)."""
import json
from pathlib import Path

OUT = Path("configs/dev")
BASE = {"data": "../../data/cqadupstack", "split": "dev", "top_k": 1000, "seed": 650}
VARIANTS = "../../data/query_variants.jsonl"
BM25 = "../../indexes/cqadupstack_pyserini"
DENSE = "../../indexes/cqadupstack_dense"

RUNS = {
    # name: extra config
    "bm25_en":        {"method": "bm25", "lexical_index": BM25, "lexical_query": "en_orig"},
    "bm25_mt_full":   {"method": "bm25", "lexical_index": BM25, "lexical_query": "mt_plain_full"},
    "bm25_mt_mixed":  {"method": "bm25", "lexical_index": BM25, "lexical_query": "mt_plain_mixed"},
    "dense_en":       {"method": "dense", "dense_index": DENSE, "dense_query": "en_orig"},
    "dense_zh_full":  {"method": "dense", "dense_index": DENSE, "dense_query": "zh_full"},
    "dense_zh_mixed": {"method": "dense", "dense_index": DENSE, "dense_query": "zh_mixed"},
    "dense_mt_full":  {"method": "dense", "dense_index": DENSE, "dense_query": "mt_plain_full"},
    "dense_mt_mixed": {"method": "dense", "dense_index": DENSE, "dense_query": "mt_plain_mixed"},
    "hybrid_full":    {"method": "hybrid", "lexical_index": BM25, "dense_index": DENSE,
                       "lexical_query": "mt_plain_full", "dense_query": "zh_full", "alpha": 0.5, "rrf_c": 60},
    "hybrid_mixed":   {"method": "hybrid", "lexical_index": BM25, "dense_index": DENSE,
                       "lexical_query": "mt_plain_mixed", "dense_query": "zh_mixed", "alpha": 0.5, "rrf_c": 60},
}
for seed in range(1, 6):
    RUNS[f"random_s{seed}"] = {"method": "random", "seed": seed}

OUT.mkdir(parents=True, exist_ok=True)
for name, extra in RUNS.items():
    cfg = {"run_id": f"dev_{name}", **BASE, "output": f"../../runs/dev/{name}", **extra}
    if "dense_index" in extra:
        cfg.update(query_max_length=256, device="cuda")
    if any(f in json.dumps(extra) for f in ("zh_", "mt_plain")):
        cfg["variants"] = VARIANTS
    path = OUT / f"{name}.json"
    if path.exists():
        print(f"skip (exists): {path}")
        continue
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
