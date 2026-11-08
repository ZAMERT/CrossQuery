import json
import random
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean

import numpy as np

from .data import read_qrels, validate
from .evaluation import evaluate
from .indexing import Index
from .io import file_hash, object_hash, read_jsonl, unique, write_json, write_jsonl
from .retrieval import rrf


def execute(config_path, allow_test=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent

    def path(value):
        return (root / value).resolve()

    split = config.get("split", "dev")
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if split == "test" and not allow_test:
        raise ValueError("Test evaluation requires --allow-test after freezing settings")
    run_id = config["run_id"]
    if not run_id or any(c.isspace() for c in run_id):
        raise ValueError("run_id must be a nonempty TREC token")
    output = path(config["output"])
    if output.exists():
        raise ValueError(f"Run output exists; choose a new run_id/output: {output}")
    data = path(config["data"])
    corpus_path = data / "corpus.jsonl"
    corpus = read_jsonl(corpus_path)
    all_queries = read_jsonl(data / "queries.jsonl")
    qrels = read_qrels(data / "qrels.tsv")
    validate(corpus, all_queries, qrels)
    hashes = {name: file_hash(data / name) for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv")}
    if config.get("variants"):
        variants_path = path(config["variants"])
        variants = unique(read_jsonl(variants_path), "query_id")
        original = unique(all_queries, "query_id")
        for qid, fields in variants.items():
            if qid not in original:
                raise ValueError(f"Unknown variant query: {qid}")
            if set(fields) & {"en_orig", "forum", "split", "source_query_id"}:
                raise ValueError("Variants cannot override original query metadata")
            original[qid].update(fields)
        hashes["variants"] = file_hash(variants_path)
    queries = [q for q in all_queries if q["split"] == split]
    if not queries:
        raise ValueError(f"No queries in split {split}")
    method = config["method"]
    if method not in {"random", "bm25", "dense", "hybrid"}:
        raise ValueError(f"Unknown method: {method}")
    top_k = config.get("top_k", 1000)
    if not isinstance(top_k, int) or top_k < 100:
        raise ValueError("top_k must be an integer >=100 for Recall@100")
    required_fields = []
    if method in {"bm25", "hybrid"}:
        required_fields.append(config["lexical_query"])
    if method in {"dense", "hybrid"}:
        required_fields.append(config["dense_query"])
    reranker = config.get("reranker")
    if reranker:
        required_fields.append(reranker["query"])
        if not 1 <= reranker.get("depth", 100) <= top_k:
            raise ValueError("Rerank depth must be between 1 and top_k")
    for q in queries:
        for field in required_fields:
            if not isinstance(q.get(field), str) or not q[field].strip():
                raise ValueError(f"Missing/non-text query variant {field}: {q['query_id']}")
    alpha, c = config.get("alpha", 0.5), config.get("rrf_c", 60)
    if method == "hybrid":
        rrf([], [], alpha, c, top_k)  # Validate before loading models.
    indexes = {}
    for key in ("lexical", "dense"):
        if (key == "lexical" and method in {"bm25", "hybrid"}) or (key == "dense" and method in {"dense", "hybrid"}):
            idx = Index(path(config[f"{key}_index"]), corpus_path, device=config.get("device"),
                        query_max_length=config.get("query_max_length", 256))
            if (key == "dense") != (idx.backend == "dense"):
                raise ValueError(f"Wrong backend for {key} index: {idx.backend}")
            if not {q["forum"] for q in queries} <= set(idx.manifest["forums"]):
                raise ValueError("Index does not cover all query forums")
            indexes[key] = idx
    model = None
    if reranker:
        from sentence_transformers import CrossEncoder
        if not reranker.get("revision"):
            raise ValueError("Pin reranker.revision to an immutable model revision")
        model = CrossEncoder(reranker.get("model", "BAAI/bge-reranker-v2-m3"),
                             revision=reranker["revision"], device=config.get("device"),
                             max_length=reranker.get("max_length", 1024), trust_remote_code=False)
    docs = unique(corpus, "doc_id")
    forum_ids = {f: sorted(d["doc_id"] for d in corpus if d["forum"] == f)
                 for f in {d["forum"] for d in corpus}}
    rankings, details, timings = {}, [], []
    for query in queries:
        started = time.perf_counter()
        qid, forum = query["query_id"], query["forum"]
        stage_times = {}
        lexical, dense = [], []
        if "lexical" in indexes:
            t = time.perf_counter()
            lexical = indexes["lexical"].search(query[config["lexical_query"]], forum, top_k)
            stage_times["lexical_ms"] = (time.perf_counter() - t) * 1000
        if "dense" in indexes:
            t = time.perf_counter()
            dense = indexes["dense"].search(query[config["dense_query"]], forum, top_k)
            stage_times["dense_encode_search_ms"] = (time.perf_counter() - t) * 1000
        if method == "random":
            pool = forum_ids[forum]
            chosen = random.Random(f"{config.get('seed', 650)}:{qid}").sample(pool, min(top_k, len(pool)))
            hits = [(did, float(-rank)) for rank, did in enumerate(chosen, 1)]
        elif method == "hybrid":
            t = time.perf_counter()
            hits = rrf(lexical, dense, alpha, c, top_k)
            stage_times["fusion_ms"] = (time.perf_counter() - t) * 1000
        else:
            hits = lexical if method == "bm25" else dense
        first_stage = list(hits)
        rerank_scores = {}
        if model is not None and hits:
            t = time.perf_counter()
            head = hits[:reranker.get("depth", 100)]
            pairs = [(query[reranker["query"]], docs[did]["contents"]) for did, _ in head]
            scores = np.asarray(model.predict(pairs, batch_size=reranker.get("batch_size", 8))).reshape(-1)
            if len(scores) != len(head) or not np.isfinite(scores).all():
                raise ValueError("Invalid reranker scores")
            rerank_scores = {did: float(score) for (did, _), score in zip(head, scores)}
            head = sorted(head, key=lambda h: -rerank_scores[h[0]])  # Stable ties.
            hits = head + hits[len(head):]
            stage_times["reranking_ms"] = (time.perf_counter() - t) * 1000
        if any(did not in docs or docs[did]["forum"] != forum for did, _ in hits):
            raise ValueError(f"Retriever returned invalid document for {qid}")
        rankings[qid] = hits
        timings.append({"query_id": qid, "total_ms": (time.perf_counter() - started) * 1000, **stage_times})
        details.append({"query_id": qid, "first_stage": first_stage, "lexical": lexical,
                        "dense": dense, "rerank_scores": rerank_scores, "final": hits})
    summary, per_query = evaluate(queries, qrels, rankings)
    summary["latency"] = {"mean_ms": mean(t["total_ms"] for t in timings),
                          "p95_ms": float(np.percentile([t["total_ms"] for t in timings], 95)),
                          "scope": "sequential query processing; excludes model/index loading and offline translation; no warmup"}
    output.mkdir(parents=True)
    with (output / "run.trec").open("w", encoding="utf-8") as stream:
        for qid, hits in rankings.items():
            for rank, (did, _) in enumerate(hits, 1):
                stream.write(f"{qid} Q0 {did} {rank} {-rank} {run_id}\n")
    write_json(output / "metrics.json", summary)
    write_jsonl(output / "per_query.jsonl", per_query)
    write_jsonl(output / "scores.jsonl", details)
    write_jsonl(output / "timings.jsonl", timings)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config_path.parent,
                                capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        commit = None
    packages = {}
    for package in ("numpy", "sentence-transformers", "faiss-cpu", "pyserini", "ir-datasets"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    source_files = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    write_json(output / "manifest.json", {
        "config": config, "config_sha256": object_hash(config), "git_commit": commit,
        "source_hashes": source_files, "input_hashes": hashes, "packages": packages,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "indexes": {key: idx.manifest for key, idx in indexes.items()},
        "evaluation": "binary positive qrels; linear gain; macro across forums",
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    return summary
