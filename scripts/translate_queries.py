"""Generate Chinese query variants and back-translations for the retrieval pipeline.

Step "fwd"  (EN -> ZH, model A): en_orig -> zh_full, zh_mixed
Step "back" (ZH -> EN, model B): zh_full -> mt_plain_full, zh_mixed -> mt_plain_mixed
The proposal requires model B != model A to limit back-translation bias.

Works with any OpenAI-compatible API (OpenAI, DeepSeek, Qwen/DashScope, ...).
Results are cached per query, so the script can be interrupted and re-run.

Example:
  export FWD_API_KEY=...   BACK_API_KEY=...
  python scripts/translate_queries.py --split dev --limit 5 \
      --fwd-model gpt-4o --back-model deepseek-chat \
      --back-base-url https://api.deepseek.com
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT_VERSION = "v2"

FWD_PROMPT = """You are preparing Chinese versions of English Stack Exchange question titles for a cross-lingual retrieval experiment. Translate the title below into Chinese twice.

Rules:
1. Keep the meaning exactly: the same goal or operation, environment, and constraints. Do not add, drop, or answer anything. If the title is a statement, do not turn it into a new request (e.g. do not add "what is the reason").
2. In BOTH versions, copy these unchanged: commands, flags, file paths, code identifiers, API names, error messages, any text inside quotation marks, version numbers, and software/product names (e.g. WSL 2, wireshark, Android, CentOS, bash).
3. "zh_full": translate all other technical terms into standard Chinese terms (memory -> 内存, permissions -> 权限, bug -> 缺陷, test case -> 测试用例, SMS -> 短信, server -> 服务器).
4. "zh_mixed": the same sentence as zh_full, but keep the technical terms in English, the way Chinese developers often write (e.g. memory, permissions, SMS, server, contact groups, requests). zh_mixed should differ from zh_full whenever the title contains any technical term.
5. Write natural Chinese.

Example
Title: Turn away a bug if no reproducible test case exists?
{{"zh_full": "没有可复现的测试用例时，是否可以拒绝处理该缺陷？", "zh_mixed": "没有可复现的 test case 时，是否可以拒绝处理该 bug？"}}

Answer with only a JSON object with keys "zh_full" and "zh_mixed".
Title: {title}"""

BACK_PROMPT = """Translate the following Chinese question into English. Output only the translation.

{text}"""

# Literal technical strings that must survive translation (for the QA report only).
LITERAL = re.compile(r"[\w./\\-]*\d[\w./\\-]*|[\w-]*[_/\\.][\w./\\-]+|--?[A-Za-z][\w-]*|`[^`]+`|\"[^\"]+\"")


def literals(text):
    return sorted({m.group(0).strip("`.,?\"") for m in LITERAL.finditer(text) if len(m.group(0).strip("`.,?\"")) > 1})


def make_client(base_url, key_env):
    from openai import OpenAI
    key = os.environ.get(key_env)
    if not key:
        sys.exit(f"Set the {key_env} environment variable")
    return OpenAI(api_key=key, base_url=base_url)


class Throttle:
    """Keep at most `rpm` requests per minute across all worker threads."""
    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm else 0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_time)
            self.next_time = start + self.interval
        time.sleep(max(0.0, start - now))


THROTTLE = Throttle(0)


def clean_output(text):
    # Reasoning models (e.g. Qwen3) may prepend <think>...</think>.
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def chat(client, model, prompt, temperature, retries=8):
    for attempt in range(retries):
        try:
            THROTTLE.wait()
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            return clean_output(resp.choices[0].message.content)
        except Exception as exc:  # rate limits, timeouts
            if attempt == retries - 1:
                raise
            wait = min(2 ** attempt, 60)
            print(f"  retry in {wait}s: {exc}", file=sys.stderr)
            time.sleep(wait)


def parse_fwd(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    repaired = False
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Common failure: unescaped quotation marks inside a value.
        m = re.search(r'"zh_full"\s*:\s*"(.*?)"\s*,\s*"zh_mixed"\s*:\s*"(.*)"\s*}', text, flags=re.S)
        if not m:
            raise
        obj, repaired = {"zh_full": m.group(1), "zh_mixed": m.group(2)}, True
    out = {k: obj[k].strip() for k in ("zh_full", "zh_mixed")}
    if not all(out.values()):
        raise ValueError("empty translation")
    if repaired:
        out["json_repaired"] = True
    return out


def load_cache(path):
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["query_id"]] = row
    return cache


def run_step(name, todo, work, cache_path, workers):
    lock = threading.Lock()
    done = failed = 0
    with cache_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(work, q): q["query_id"] for q in todo}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                failed += 1
                print(f"[{name}] FAILED {qid}: {exc}", file=sys.stderr)
                continue
            with lock:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            done += 1
            if done % 25 == 0:
                print(f"[{name}] {done}/{len(todo)}")
    print(f"[{name}] finished: {done} ok, {failed} failed (re-run to retry failures)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/cqadupstack/queries.jsonl")
    p.add_argument("--output", default="data/query_variants.jsonl")
    p.add_argument("--cache-dir", default="data/translation_cache")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--limit", type=int, help="only the first N queries (for a trial run)")
    p.add_argument("--fwd-model", required=True)
    p.add_argument("--fwd-base-url")
    p.add_argument("--fwd-key-env", default="FWD_API_KEY")
    p.add_argument("--back-model", required=True)
    p.add_argument("--back-base-url")
    p.add_argument("--back-key-env", default="BACK_API_KEY")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rpm", type=float, default=0, help="max requests per minute (0 = no limit)")
    args = p.parse_args()
    global THROTTLE
    THROTTLE = Throttle(args.rpm)
    if args.fwd_model == args.back_model and args.fwd_base_url == args.back_base_url:
        sys.exit("Use a different model for ZH->EN than for EN->ZH (proposal §5.3)")

    queries = [json.loads(l) for l in Path(args.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.split != "all":
        queries = [q for q in queries if q["split"] == args.split]
    queries.sort(key=lambda q: q["query_id"])
    if args.limit:
        queries = queries[:args.limit]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fwd_path, back_path = cache_dir / "fwd.jsonl", cache_dir / "back.jsonl"

    # Step 1: EN -> ZH
    fwd_client = make_client(args.fwd_base_url, args.fwd_key_env)
    fwd_cache = load_cache(fwd_path)

    def fwd_work(q):
        raw = chat(fwd_client, args.fwd_model, FWD_PROMPT.format(title=q["en_orig"]), args.temperature)
        return {"query_id": q["query_id"], **parse_fwd(raw), "model": args.fwd_model,
                "prompt_version": PROMPT_VERSION}

    run_step("en->zh", [q for q in queries if q["query_id"] not in fwd_cache], fwd_work, fwd_path, args.workers)
    fwd_cache = load_cache(fwd_path)

    # Step 2: ZH -> EN (plain MT, never sees the English original)
    back_client = make_client(args.back_base_url, args.back_key_env)
    back_cache = load_cache(back_path)

    def back_work(q):
        zh = fwd_cache[q["query_id"]]
        return {"query_id": q["query_id"],
                "mt_plain_full": chat(back_client, args.back_model, BACK_PROMPT.format(text=zh["zh_full"]), args.temperature),
                "mt_plain_mixed": chat(back_client, args.back_model, BACK_PROMPT.format(text=zh["zh_mixed"]), args.temperature),
                "model": args.back_model, "prompt_version": PROMPT_VERSION}

    run_step("zh->en", [q for q in queries if q["query_id"] in fwd_cache and q["query_id"] not in back_cache],
             back_work, back_path, args.workers)
    back_cache = load_cache(back_path)

    # Assemble the variants file and a QA report
    variants, report = [], []
    for q in queries:
        qid = q["query_id"]
        if qid not in fwd_cache or qid not in back_cache:
            continue
        f, b = fwd_cache[qid], back_cache[qid]
        variants.append({"query_id": qid, "zh_full": f["zh_full"], "zh_mixed": f["zh_mixed"],
                         "mt_plain_full": b["mt_plain_full"], "mt_plain_mixed": b["mt_plain_mixed"]})
        lits = literals(q["en_orig"])
        missing = sorted({l for l in lits for v in ("zh_full", "zh_mixed") if l.lower() not in f[v].lower()})
        report.append({"query_id": qid, "en_orig": q["en_orig"], **variants[-1], "missing_literals": missing})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in variants), encoding="utf-8")
    report_path = out.with_name(out.stem + "_report.jsonl")
    report_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report), encoding="utf-8")
    flagged = sum(bool(r["missing_literals"]) for r in report)
    same = sum(r["zh_full"] == r["zh_mixed"] for r in report)
    print(f"Wrote {len(variants)}/{len(queries)} variants to {out}")
    print(f"QA report: {report_path} ({flagged} queries with possibly altered literals, "
          f"{same} with zh_mixed identical to zh_full)")
    if len(variants) < len(queries):
        print("Some queries are missing; re-run the same command to retry them.")


if __name__ == "__main__":
    main()