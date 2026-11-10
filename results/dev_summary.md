| Run | Role | nDCG@10 | MRR@10 | R@100 | android | programmers | unix |
|---|---|---|---|---|---|---|---|
| random | Baseline (mean of 5 seeds) | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 |
| bm25_en | Reference (EN) | 0.331 | 0.333 | 0.624 | 0.383 | 0.362 | 0.249 |
| dense_en | Reference (EN) | 0.457 | 0.451 | 0.798 | 0.511 | 0.446 | 0.413 |
| bm25_mt_full | Translate + BM25 | 0.294 | 0.295 | 0.590 | 0.322 | 0.310 | 0.251 |
| bm25_mt_mixed | Translate + BM25 | 0.302 | 0.302 | 0.601 | 0.341 | 0.309 | 0.256 |
| dense_zh_full | Direct CLIR | 0.378 | 0.369 | 0.751 | 0.450 | 0.362 | 0.322 |
| dense_zh_mixed | Direct CLIR | 0.403 | 0.390 | 0.784 | 0.474 | 0.391 | 0.344 |
| dense_mt_full | Translate + dense | 0.417 | 0.402 | 0.793 | 0.466 | 0.414 | 0.372 |
| dense_mt_mixed | Translate + dense | 0.426 | 0.412 | 0.798 | 0.493 | 0.414 | 0.371 |
| hybrid_full | Hybrid RRF | 0.381 | 0.372 | 0.773 | 0.437 | 0.391 | 0.315 |
| hybrid_mixed | Hybrid RRF | 0.397 | 0.395 | 0.787 | 0.468 | 0.406 | 0.317 |

Per-forum columns are nDCG@10. Macro = mean over the three forums.

zh_mixed differs from zh_full for 293/528 dev queries.

| Comparison | Queries | mean ΔnDCG@10 (A−B) | p (paired randomization) |
|---|---|---|---|
| RQ1: direct vs translate (dense, ZH-full): dense_zh_full vs dense_mt_full | 528 | -0.042 | 0.0003 |
| RQ1: direct vs translate (dense, ZH-mixed): dense_zh_mixed vs dense_mt_mixed | 528 | -0.024 | 0.0066 |
| RQ2: ZH-full vs ZH-mixed (dense): dense_zh_full vs dense_zh_mixed | 293 | -0.045 | 0.0006 |
| RQ2: ZH-full vs ZH-mixed (translate + BM25): bm25_mt_full vs bm25_mt_mixed | 293 | -0.014 | 0.0829 |
| Hybrid vs direct dense (ZH-full): hybrid_full vs dense_zh_full | 528 | +0.003 | 0.7651 |
| Hybrid vs translate + BM25 (ZH-full): hybrid_full vs bm25_mt_full | 528 | +0.083 | 0.0001 |

Query-level, pooled over forums; p-values not yet Holm-corrected.
