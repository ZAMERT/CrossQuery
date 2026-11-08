import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from si650_retrieval.data import clean_text, prepare, read_qrels, validate
from si650_retrieval.evaluation import evaluate
from si650_retrieval.indexing import Index, build_index
from si650_retrieval.io import read_jsonl, write_json, write_jsonl
from si650_retrieval.pipeline import execute
from si650_retrieval.retrieval import bm25_build, bm25_search, normalize, rrf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "data"
    prepare(ROOT / "examples/toy", data)
    index = tmp_path / "index"
    build_index(data / "corpus.jsonl", index)
    return data, index


def configuration(tmp_path, data, index, **updates):
    config = {"run_id": "test", "data": str(data), "output": str(tmp_path / "run"),
              "method": "bm25", "lexical_index": str(index), "lexical_query": "en_orig"}
    config.update(updates)
    file = tmp_path / "config.json"
    write_json(file, config)
    return file


def test_html_preserves_code_and_plain_angle_brackets():
    assert clean_text("vector<T> --flag\n  a < b") == "vector<T> --flag\n  a < b"
    assert clean_text("<p>Hello</p><pre>if (a &lt; b)\n  run --flag</pre>", "html") == "Hello\n\nif (a < b)\n  run --flag"


def test_bm25_hand_calculation_and_no_match():
    index = bm25_build([{"doc_id": "a", "contents": "term term"},
                       {"doc_id": "b", "contents": "other"}])
    hits = bm25_search(index, "term", 10)
    expected = math.log(2) * 2 * 1.9 / (2 + 0.9 * (0.6 + 0.4 * 2 / 1.5))
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(expected)
    assert bm25_search(index, "missing", 10) == []


def test_rrf_dedup_ties_and_endpoints():
    left = [("a", 8), ("a", 7), ("b", 6)]
    right = [("b", 9), ("c", 8)]
    result = rrf(left, right)
    assert [d for d, _ in result] == ["b", "a", "c"]
    assert result[0][1] == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert [d for d, _ in rrf(left, right, alpha=0)] == ["b", "c"]
    assert [d for d, _ in rrf(left, right, alpha=1)] == ["a", "b"]
    assert rrf([("b", 1)], [("a", 1)])[0][0] == "a"


def test_normalization_and_invalid_vectors():
    x = normalize([[3, 4], [0, 2]])
    np.testing.assert_allclose(x @ x.T, [[1, 0.8], [0.8, 1]], atol=1e-6)
    with pytest.raises(ValueError):
        normalize([[0, 0]])
    with pytest.raises(ValueError):
        normalize([[float("nan"), 1]])


def test_metrics_missing_queries_and_macro():
    queries = [{"query_id": q, "forum": f} for q, f in [("a", "x"), ("b", "x"), ("c", "y")]]
    qrels = [("a", "d1", 1), ("a", "d2", 1), ("b", "d3", 1), ("c", "d4", 1)]
    summary, rows = evaluate(queries, qrels, {"a": [("noise", 2), ("d1", 1)], "c": [("d4", 1)]})
    assert rows[0]["nDCG@10"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert rows[0]["MRR@10"] == 0.5
    assert rows[0]["Recall@100"] == 0.5
    assert rows[1]["MRR@10"] == 0
    assert summary["macro"]["MRR@10"] == 0.625  # Not the micro average 0.5.
    with pytest.raises(ValueError, match="Duplicate documents"):
        evaluate(queries, qrels, {"a": [("d1", 1), ("d1", 0)]})


def test_prepare_split_is_order_independent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_jsonl(source / "corpus.jsonl", [{"doc_id": "d", "forum": "unix", "title": "hello"}])
    queries = [{"query_id": f"q{i}", "forum": "unix", "en_orig": f"query {i}"} for i in range(10)]
    write_jsonl(source / "queries.jsonl", queries)
    (source / "qrels.tsv").write_text("".join(f"q{i} 0 d 1\n" for i in range(10)))
    prepare(source, tmp_path / "a")
    write_jsonl(source / "queries.jsonl", reversed(queries))
    prepare(source, tmp_path / "b")
    a = read_jsonl(tmp_path / "a/queries.jsonl")
    assert a == read_jsonl(tmp_path / "b/queries.jsonl")
    assert sum(q["split"] == "dev" for q in a) == 2


def test_integrity_checks(prepared):
    data, _ = prepared
    docs = read_jsonl(data / "corpus.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    qrels = read_qrels(data / "qrels.tsv")
    with pytest.raises(ValueError, match="Duplicate doc_id"):
        validate(docs + [docs[0]], queries, qrels)
    with pytest.raises(ValueError, match="Dangling"):
        validate(docs, queries, qrels + [(queries[0]["query_id"], "missing", 1)])
    with pytest.raises(ValueError, match="without positive"):
        validate(docs, queries, [])


def test_index_rejects_stale_corpus_and_corrupt_artifact(prepared):
    data, index = prepared
    artifact = index / "unix.json"
    artifact.write_text(artifact.read_text() + " ")
    with pytest.raises(ValueError, match="artifact has changed"):
        Index(index, data / "corpus.jsonl")
    corpus = data / "corpus.jsonl"
    corpus.write_text(corpus.read_text() + "\n")
    with pytest.raises(ValueError, match="Corpus hash"):
        Index(index, corpus)


def test_end_to_end_and_no_overwrite(tmp_path, prepared):
    data, index = prepared
    config = configuration(tmp_path, data, index)
    summary = execute(config)
    assert summary["macro"] == {"nDCG@10": 1, "MRR@10": 1, "Recall@100": 1}
    rows = (tmp_path / "run/run.trec").read_text().splitlines()
    assert rows and all(float(r.split()[4]) == -int(r.split()[3]) for r in rows)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["input_hashes"]["corpus.jsonl"]
    assert manifest["source_hashes"]["pipeline.py"]
    with pytest.raises(ValueError, match="output exists"):
        execute(config)


def test_test_guard_and_missing_variant(tmp_path, prepared):
    data, index = prepared
    config = configuration(tmp_path, data, index, split="test")
    with pytest.raises(ValueError, match="--allow-test"):
        execute(config)
    config = configuration(tmp_path, data, index, lexical_query="not_translated")
    with pytest.raises(ValueError, match="Missing/non-text"):
        execute(config)
    assert not (tmp_path / "run").exists()


def test_external_variants_and_metadata_protection(tmp_path, prepared):
    data, index = prepared
    variants = tmp_path / "variants.jsonl"
    write_jsonl(variants, [{"query_id": q["query_id"], "mt_custom": q["en_orig"]}
                          for q in read_jsonl(data / "queries.jsonl")])
    config = configuration(tmp_path, data, index, variants=str(variants), lexical_query="mt_custom")
    assert execute(config)["macro"]["MRR@10"] == 1
    write_jsonl(variants, [{"query_id": "unix:q1", "split": "test"}])
    config = configuration(tmp_path, data, index, variants=str(variants), output=str(tmp_path / "other"))
    with pytest.raises(ValueError, match="metadata"):
        execute(config)


def test_random_reproducible(tmp_path, prepared):
    data, index = prepared
    config = configuration(tmp_path, data, index, method="random", seed=650)
    execute(config)
    config = configuration(tmp_path, data, index, method="random", seed=650, output=str(tmp_path / "second"))
    execute(config)
    assert (tmp_path / "run/run.trec").read_bytes() == (tmp_path / "second/run.trec").read_bytes()


def test_hybrid_orchestration_with_controlled_retriever(tmp_path, prepared, monkeypatch):
    data, index = prepared

    class ControlledIndex:
        def __init__(self, path, *args, **kwargs):
            self.backend = "dense" if path.name == "dense" else "bm25"
            self.manifest = {"forums": ["android", "unix"], "backend": self.backend}

        def search(self, text, forum, top_k):
            assert text
            if self.backend == "dense":
                return [(f"{forum}:d1", 0.9), (f"{forum}:d2", 0.8)]
            return [(f"{forum}:d2", 5), (f"{forum}:d1", 4)]

    monkeypatch.setattr("si650_retrieval.pipeline.Index", ControlledIndex)
    config = configuration(tmp_path, data, index, method="hybrid", dense_index=str(tmp_path / "dense"),
                           dense_query="zh_full", lexical_query="mt_plain_full", alpha=0.25)
    assert execute(config)["macro"]["MRR@10"] == 1
    scores = read_jsonl(tmp_path / "run/scores.jsonl")
    assert all(row["lexical"][0][0] != row["dense"][0][0] for row in scores)


def test_cli_help():
    result = subprocess.run([sys.executable, "-m", "si650_retrieval", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "prepare" in result.stdout


def test_reranker_preserves_tail_and_trec_order(tmp_path, prepared, monkeypatch):
    import types

    data, index = prepared

    class ControlledIndex:
        backend = "bm25"
        manifest = {"forums": ["android", "unix"]}

        def __init__(self, *args, **kwargs):
            pass

        def search(self, text, forum, top_k):
            return [(f"{forum}:d2", 30), (f"{forum}:d1", 20), (f"{forum}:d3", 10)]

    class ControlledReranker:
        def __init__(self, *args, **kwargs):
            assert kwargs["revision"] == "test-commit"

        def predict(self, pairs, **kwargs):
            assert len(pairs) == 2
            return [0.1, 0.9]

    monkeypatch.setattr("si650_retrieval.pipeline.Index", ControlledIndex)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=ControlledReranker))
    config = configuration(tmp_path, data, index, reranker={"revision": "test-commit", "query": "zh_full", "depth": 2})
    assert execute(config)["macro"]["MRR@10"] == 1
    for row in read_jsonl(tmp_path / "run/scores.jsonl"):
        assert [did.split(":")[1] for did, _ in row["final"]] == ["d1", "d2", "d3"]
        assert set(did for did, _ in row["final"]) == set(did for did, _ in row["first_stage"])
    for line in (tmp_path / "run/run.trec").read_text().splitlines():
        fields = line.split()
        assert float(fields[4]) == -int(fields[3])


def test_dense_adapter_roundtrip_with_controlled_dependencies(tmp_path, prepared, monkeypatch):
    import types

    data, _ = prepared

    class Encoder:
        def __getitem__(self, i):
            return types.SimpleNamespace(auto_model=types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="pinned")))

        def encode(self, texts, **kwargs):
            return np.array([[1, 0] if "Wireshark" in text else [0, 1] for text in texts], dtype=np.float32)

    class FlatIndex:
        def __init__(self, dim):
            self.dim = dim

        def add(self, vectors):
            self.vectors = vectors
            self.ntotal = len(vectors)

        def search(self, query, k):
            scores = query @ self.vectors.T
            order = np.argsort(-scores, axis=1)[:, :k]
            return np.take_along_axis(scores, order, axis=1), order

    def save(index, path):
        with open(path, "wb") as stream:
            np.save(stream, index.vectors)

    def load(path):
        with open(path, "rb") as stream:
            vectors = np.load(stream)
        index = FlatIndex(vectors.shape[1])
        index.add(vectors)
        return index

    revisions = []

    def encoder(model, revision=None, device=None):
        revisions.append(revision)
        return Encoder()

    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace(IndexFlatIP=FlatIndex, write_index=save, read_index=load))
    monkeypatch.setattr("si650_retrieval.indexing.load_encoder", encoder)
    path = tmp_path / "dense"
    build_index(data / "corpus.jsonl", path, backend="dense")
    index = Index(path, data / "corpus.jsonl")
    hits = index.search("Wireshark", "unix", 2)
    assert hits[0] == ("unix:d1", 1.0)
    assert revisions == [None, "pinned"]


def test_metrics_against_ir_measures_when_installed():
    ir = pytest.importorskip("ir_measures")
    queries = [{"query_id": "q", "forum": "unix"}, {"query_id": "empty", "forum": "unix"}]
    qrels = [("q", "a", 1), ("q", "b", 1), ("empty", "c", 1)]
    ranking = {"q": [("noise", 3), ("b", 2), ("a", 1)]}
    summary, _ = evaluate(queries, qrels, ranking)
    external = ir.calc_aggregate([ir.nDCG@10, ir.RR@10, ir.R@100],
                                [ir.Qrel(q, d, r) for q, d, r in qrels],
                                [ir.ScoredDoc("q", d, s) for d, s in ranking["q"]])
    for name, metric in [("nDCG@10", ir.nDCG@10), ("MRR@10", ir.RR@10), ("Recall@100", ir.R@100)]:
        assert summary["macro"][name] == pytest.approx(external[metric])
