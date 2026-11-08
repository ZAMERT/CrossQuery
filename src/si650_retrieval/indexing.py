import json
import subprocess
import sys
from pathlib import Path

from .io import file_hash, read_jsonl, unique, write_json, write_jsonl
from .retrieval import bm25_build, bm25_search, load_encoder, normalize


def build_index(corpus_path, output, backend="bm25", model="BAAI/bge-m3", revision=None,
                device=None, max_length=2048, batch_size=8):
    if backend not in {"bm25", "pyserini", "dense"}:
        raise ValueError(f"Unknown backend: {backend}")
    if max_length <= 0 or batch_size <= 0:
        raise ValueError("Lengths and batch sizes must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Index exists; choose a new path: {output}")
    docs = read_jsonl(corpus_path)
    unique(docs, "doc_id")
    if not docs or any(not d.get("contents", "").strip() for d in docs):
        raise ValueError("Index requires nonempty prepared documents")
    forums = sorted({d["forum"] for d in docs})
    if any(not f or not all(ch.isalnum() or ch in "_-" for ch in f) for f in forums):
        raise ValueError("Invalid forum name")
    encoder = None
    manifest = {"backend": backend, "corpus_sha256": file_hash(corpus_path), "forums": forums,
                "k1": 0.9, "b": 0.4, "max_length": max_length}
    if backend == "dense":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install the dense extra: pip install -e '.[dense]'") from exc
        encoder = load_encoder(model, revision, device)
        encoder.max_seq_length = max_length
        resolved = getattr(encoder[0].auto_model.config, "_commit_hash", None)
        if not resolved and not revision:
            raise ValueError("Could not resolve model revision; supply an immutable --revision")
        manifest.update(model=model, revision=resolved or revision, batch_size=batch_size,
                        normalized=True, index_type="IndexFlatIP")
    output.mkdir(parents=True)
    for forum in forums:
        subset = sorted((d for d in docs if d["forum"] == forum), key=lambda d: d["doc_id"])
        if backend == "bm25":
            write_json(output / f"{forum}.json", bm25_build(subset))
        elif backend == "pyserini":
            collection = output / f"{forum}_collection"
            write_jsonl(collection / "docs.jsonl", ({"id": d["doc_id"], "contents": d["contents"]} for d in subset))
            subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                            "--input", str(collection), "--index", str(output / forum),
                            "--generator", "DefaultLuceneDocumentGenerator", "--threads", "1",
                            "--storePositions", "--storeDocvectors", "--storeRaw"], check=True)
        else:
            vectors = normalize(encoder.encode([d["contents"] for d in subset], batch_size=batch_size,
                                                convert_to_numpy=True, normalize_embeddings=True,
                                                show_progress_bar=True, prompt=""))
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(output / f"{forum}.faiss"))
            write_json(output / f"{forum}.ids.json", [d["doc_id"] for d in subset])
    manifest["artifacts"] = {str(p.relative_to(output)): file_hash(p)
                             for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


class Index:
    def __init__(self, path, corpus_path, device=None, query_max_length=256):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest["corpus_sha256"] != file_hash(corpus_path):
            raise ValueError("Corpus hash differs from index; rebuild the index")
        for name, expected in self.manifest["artifacts"].items():
            if file_hash(self.path / name) != expected:
                raise ValueError(f"Index artifact has changed: {name}")
        self.backend = self.manifest["backend"]
        self.indexes = {}
        self.ids = {}
        self.encoder = None
        if query_max_length <= 0:
            raise ValueError("query_max_length must be positive")
        if self.backend == "dense":
            import faiss
            self.encoder = load_encoder(self.manifest["model"], self.manifest["revision"], device)
            self.encoder.max_seq_length = query_max_length
        for forum in self.manifest["forums"]:
            if self.backend == "bm25":
                self.indexes[forum] = json.loads((self.path / f"{forum}.json").read_text())
            elif self.backend == "pyserini":
                from pyserini.search.lucene import LuceneSearcher
                searcher = LuceneSearcher(str(self.path / forum))
                searcher.set_bm25(self.manifest["k1"], self.manifest["b"])
                self.indexes[forum] = searcher
            elif self.backend == "dense":
                self.indexes[forum] = faiss.read_index(str(self.path / f"{forum}.faiss"))
                self.ids[forum] = json.loads((self.path / f"{forum}.ids.json").read_text())
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

    def search(self, text, forum, top_k):
        if top_k <= 0 or not text.strip():
            raise ValueError("Search requires positive top_k and nonempty text")
        index = self.indexes[forum]
        if self.backend == "bm25":
            return bm25_search(index, text, top_k, self.manifest["k1"], self.manifest["b"])
        if self.backend == "pyserini":
            return [(h.docid, float(h.score)) for h in index.search(text, k=top_k)]
        vector = normalize(self.encoder.encode([text], convert_to_numpy=True,
                                               normalize_embeddings=True, prompt=""))
        # Retrieve all exact scores to make boundary ties deterministic by doc ID.
        scores, positions = index.search(vector, index.ntotal)
        return sorted(((self.ids[forum][int(i)], float(s)) for i, s in zip(positions[0], scores[0]) if i >= 0),
                      key=lambda h: (-h[1], h[0]))[:top_k]
