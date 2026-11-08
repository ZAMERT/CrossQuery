"""Import, deterministic query splits, and relevance integrity checks."""
import random
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

from .io import file_hash, read_jsonl, unique, write_json, write_jsonl

FORUMS = ("android", "programmers", "unix")


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "br", "pre", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "pre", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def clean_text(text, text_format="plain"):
    if text_format == "plain":
        return text.strip()
    if text_format != "html":
        raise ValueError(f"Unknown text_format: {text_format}")
    parser = HTMLText()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts).strip()


def read_qrels(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            qid, _, did, rel = line.split()
            pair = (qid, did)
            if pair in seen:
                raise ValueError(f"Duplicate qrel: {pair}")
            seen.add(pair)
            relevance = int(rel)
            if relevance < 0:
                raise ValueError("Negative relevance is not supported")
            rows.append((qid, did, relevance))
    return rows


def validate(corpus, queries, qrels):
    docs = unique(corpus, "doc_id")
    qs = unique(queries, "query_id")
    if not docs or not qs:
        raise ValueError("Corpus and queries must not be empty")
    for d in corpus:
        if not d.get("forum") or not d.get("contents", "").strip():
            raise ValueError(f"Missing forum or empty document: {d['doc_id']}")
    for q in queries:
        if not q.get("forum") or not q.get("en_orig", "").strip():
            raise ValueError(f"Missing forum or empty query: {q['query_id']}")
        if q["split"] not in {"dev", "test"}:
            raise ValueError(f"Invalid split: {q['query_id']}")
    positives = set()
    for qid, did, rel in qrels:
        if qid not in qs or did not in docs:
            raise ValueError(f"Dangling qrel: {qid}, {did}")
        if qs[qid]["forum"] != docs[did]["forum"]:
            raise ValueError(f"Cross-forum qrel: {qid}, {did}")
        if rel > 0:
            positives.add(qid)
    if positives != set(qs):
        raise ValueError(f"Queries without positive qrels: {sorted(set(qs) - positives)[:5]}")


def prepare(source, output, seed=650, dev_fraction=0.2):
    source, output = Path(source), Path(output)
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    if output.exists():
        raise ValueError(f"Output exists; choose a new dataset version: {output}")
    corpus = read_jsonl(source / "corpus.jsonl")
    queries = read_jsonl(source / "queries.jsonl")
    qrels = read_qrels(source / "qrels.tsv")
    for doc in corpus:
        fmt = doc.get("text_format", "plain")
        doc["title"] = clean_text(doc.get("title", ""), fmt)
        doc["body"] = clean_text(doc.get("body", ""), fmt)
        doc["contents"] = (doc["title"] + "\n\n" + doc["body"]).strip()
    # Either preserve an entirely explicit split or create all splits together.
    explicit = ["split" in q for q in queries]
    if any(explicit) and not all(explicit):
        raise ValueError("Provide split for all queries or none")
    if not any(explicit):
        groups = defaultdict(list)
        for q in queries:
            groups[q["forum"]].append(q)
        for forum, group in sorted(groups.items()):
            group.sort(key=lambda q: q["query_id"])
            random.Random(f"{seed}:{forum}").shuffle(group)
            cutoff = int(len(group) * dev_fraction)
            if cutoff == 0 or cutoff == len(group):
                raise ValueError(f"Too few queries to split {forum}; supply explicit splits")
            for i, query in enumerate(group):
                query["split"] = "dev" if i < cutoff else "test"
    validate(corpus, queries, qrels)
    output.mkdir(parents=True)
    write_jsonl(output / "corpus.jsonl", sorted(corpus, key=lambda d: d["doc_id"]))
    write_jsonl(output / "queries.jsonl", sorted(queries, key=lambda q: q["query_id"]))
    (output / "qrels.tsv").write_text(
        "".join(f"{q}\t0\t{d}\t{r}\n" for q, d, r in qrels), encoding="utf-8")
    counts = {forum: {"docs": sum(d["forum"] == forum for d in corpus),
                      **{split: sum(q["forum"] == forum and q["split"] == split for q in queries)
                         for split in ("dev", "test")}} for forum in sorted({q["forum"] for q in queries})}
    write_json(output / "manifest.json", {
        "seed": seed, "dev_fraction": dev_fraction, "explicit_split": all(explicit),
        "counts": counts, "preprocessing": "html-opt-in-v1; duplicate banners retained",
        "files": {name: file_hash(output / name) for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv")}})
    return counts


def export_beir(output, forums=FORUMS):
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    if not forums or not set(forums) <= set(FORUMS) or len(set(forums)) != len(forums):
        raise ValueError(f"Select unique forums from {FORUMS}")
    output.mkdir(parents=True)
    datasets = [(forum, ir_datasets.load(f"beir/cqadupstack/{forum}")) for forum in forums]
    write_jsonl(output / "corpus.jsonl", (
        {"doc_id": f"{forum}:{d.doc_id}", "source_doc_id": d.doc_id, "forum": forum,
         "title": d.title, "body": d.text, "text_format": "plain"}
        for forum, dataset in datasets for d in dataset.docs_iter()))
    write_jsonl(output / "queries.jsonl", (
        {"query_id": f"{forum}:{q.query_id}", "source_query_id": q.query_id,
         "forum": forum, "en_orig": q.text}
        for forum, dataset in datasets for q in dataset.queries_iter()))
    with (output / "qrels.tsv").open("w", encoding="utf-8") as stream:
        for forum, dataset in datasets:
            for rel in dataset.qrels_iter():
                stream.write(f"{forum}:{rel.query_id}\t0\t{forum}:{rel.doc_id}\t{rel.relevance}\n")
    write_json(output / "source.json", {"datasets": [f"beir/cqadupstack/{f}" for f in forums]})
