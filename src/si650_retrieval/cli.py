import argparse
import json
import sys

from .data import FORUMS, export_beir, prepare
from .indexing import build_index
from .pipeline import execute


def main():
    parser = argparse.ArgumentParser(description="SI650 reproducible retrieval pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Download/export CQADupStack with ir_datasets")
    export.add_argument("--output", required=True)
    export.add_argument("--forums", nargs="+", choices=FORUMS, default=list(FORUMS))
    prep = sub.add_parser("prepare", help="Validate, clean, and split a local JSONL dataset")
    prep.add_argument("--source", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=650)
    prep.add_argument("--dev-fraction", type=float, default=0.2)
    index = sub.add_parser("index", help="Build a corpus-bound retrieval index")
    index.add_argument("--corpus", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--backend", choices=("bm25", "pyserini", "dense"), default="bm25")
    index.add_argument("--model", default="BAAI/bge-m3")
    index.add_argument("--revision")
    index.add_argument("--device")
    index.add_argument("--max-length", type=int, default=2048)
    index.add_argument("--batch-size", type=int, default=8)
    run = sub.add_parser("run", help="Retrieve, optionally fuse/rerank, and evaluate")
    run.add_argument("--config", required=True)
    run.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "export":
            export_beir(args.output, args.forums)
            result = {"output": args.output}
        elif args.command == "prepare":
            result = prepare(args.source, args.output, args.seed, args.dev_fraction)
        elif args.command == "index":
            result = build_index(args.corpus, args.output, args.backend, args.model, args.revision,
                                 args.device, args.max_length, args.batch_size)
        else:
            result = execute(args.config, args.allow_test)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, ImportError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
