# Chinese-to-English Technical Question Retrieval

SI650 基础检索 pipeline：数据准备 → BM25 / BGE-M3 → RRF → 可选 reranker → 评估。任务来自 [proposal](SI650_Project_Proposal.pdf)，完整研究设计见 [设计文档](docs/retrieval_pipeline.md)。

已实现 JSONL/BEIR 数据导入、论坛内固定 dev/test 划分、query variant 导入、random/BM25/dense/hybrid 检索、可选 cross-encoder 重排、TREC run、逐查询与论坛 macro 指标，以及配置和产物哈希记录。中文 query 和回译由 `scripts/translate_queries.py` 生成后从文件读取，检索代码本身不调用翻译服务。

**当前进度（2026-11-09）**：dev 集（528 条 query）上的英文参照、translate + BM25、direct CLIR、translate + dense、hybrid 和 random 已全部跑完，结果见下文「当前 dev 结果」。

## Great Lakes 环境

正式实验在 Great Lakes 上运行（账号 `si650f26_class`，`gpu` 分区为 V100 16GB，每人同时最多 1 块 GPU、单个任务最长 8 小时）。

```bash
module load python3.11-anaconda
eval "$(conda shell.bash hook)"          # 不加这一行时 conda activate 可能不生效
conda create -n si650 python=3.11 -y
conda activate si650
conda install -c conda-forge openjdk=21 -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126   # V100 需要 cu126 版本
python -m pip install -e '.[data,dense,lucene,eval,dev]'
```

已知问题：

- **导入 Pyserini 时报 `OpenAIError: Missing credentials`**：Pyserini 会在 import 时创建 OpenAI client。BM25 用不到 OpenAI，设一个假的 key 即可：`export OPENAI_API_KEY=dummy`。
- **建 dense 索引时报 `Could not resolve model revision`**：较新的 transformers 读不到模型 commit，需显式传 `--revision`。模型下载后可用 `cat ~/.cache/huggingface/hub/models--BAAI--bge-m3/refs/main` 查到。
- 用 `salloc` 申请的交互式节点会一直占着 GPU，用完要 `exit`；否则后续 `sbatch` 任务会一直排队。

## 先跑一个无需下载模型的例子

在本目录使用 Python 3.10+ 和 NumPy：

```bash
export PYTHONPATH="$PWD/src"
python -m si650_retrieval prepare --source examples/toy --output data/toy
python -m si650_retrieval index --corpus data/toy/corpus.jsonl --output indexes/toy_bm25 --backend bm25
python -m si650_retrieval run --config configs/toy_bm25.json
```

也可在自己的虚拟环境中安装 `python -m pip install -e '.[dev]'`，之后用 `si650` 替代 `python -m si650_retrieval`。

自带 toy 数据是人工构造的 6 篇文档、4 个查询，仅验证流程。示例命令在 dev 的 2 个查询上运行，不接触 test 指标。上述数据、索引和运行结果已在当前工作区生成；再次运行应使用新的输出路径并调整配置，命令会拒绝覆盖已有产物。

`bm25` 是内置 Python BM25（Unicode word tokenizer、无 stemming），用于离线开发与 smoke test；正式 proposal baseline 请使用 `pyserini`。两者分词和计分细节不同，不能混用后声称复现了 BEIR。

## 导入真实 CQADupStack 并跑英文基线

数据下载需要网络和 `data` extra。正式 Lucene 后端还需要 `lucene` extra 及兼容的 Java；当前 Pyserini 官方以 Python 3.12、Java 21 为环境基础，建议为正式实验单独建环境。[官方安装说明](https://github.com/castorini/pyserini#-installation)

```bash
python -m pip install -e '.[data,lucene]'
python -m si650_retrieval export --output data/raw_cqadupstack
python -m si650_retrieval prepare --source data/raw_cqadupstack --output data/cqadupstack
python -m si650_retrieval index --corpus data/cqadupstack/corpus.jsonl --output indexes/cqadupstack_pyserini --backend pyserini
python -m si650_retrieval run --config configs/cqadupstack_bm25.json
```

`export --forums unix` 可先下载一个论坛验证。默认导出 android/programmers/unix，ID 格式为 `forum:original_id`。`prepare` 按论坛分层、seed=650、20% dev；若输入所有查询已带 `split`，则保留原划分。文档默认是纯文本，只有显式 `text_format: "html"` 才去 HTML 标签，避免破坏 `vector<T>` 和代码。当前保留 duplicate banners，尚未实现研究设计中的横幅审计和删除。

## Dense 和 hybrid

```bash
python -m pip install -e '.[dense]'
python -m si650_retrieval index --corpus data/cqadupstack/corpus.jsonl --output indexes/cqadupstack_dense --backend dense --device cuda
```

**本项目实际使用的设置**（之后所有 dense 实验必须与此一致）：

```bash
python -m si650_retrieval index --corpus data/cqadupstack/corpus.jsonl \
  --output indexes/cqadupstack_dense --backend dense --device cuda \
  --max-length 512 --batch-size 32 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181
```

帖子正文中位数仅 83–140 词，截断到 512 token 几乎无损，速度比默认的 2048 快数倍；在 V100 上最大的 unix 子论坛约 12 分钟编码完。

默认模型 `BAAI/bge-m3`，doc max length=2048、query=256、batch=8；可用 `--device cpu`。模型接口使用 Sentence Transformers，显式关闭 query prompt，使用归一化向量及 FAISS `IndexFlatIP`。构建时记录实际模型 commit，查询时复用该 revision；也可显式传 `--revision <commit>`。[模型说明](https://huggingface.co/BAAI/bge-m3)、[编码接口](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)

`data/query_variants.jsonl` 由下一节的翻译脚本生成，每个待运行查询一行，例如：

```json
{"query_id":"unix:111650","zh_full":"如何为 wireshark 设置正确的权限？","zh_mixed":"如何为 wireshark 设置正确的 permissions？","mt_plain_full":"How to set up wireshark with correct permissions?","mt_plain_mixed":"How to set up the correct permissions for wireshark?"}
```

这里仅展示字段格式。缺失或空变体会报错，不会静默回退为英文原标题。变体文件不能修改 `en_orig`、`forum`、`split` 等原始元数据。

## 翻译：生成 query variants

```bash
python scripts/translate_queries.py --split dev \
  --fwd-model qwen2.5:14b --fwd-base-url http://localhost:11434/v1 \
  --back-model gemma3:12b --back-base-url http://localhost:11434/v1 --workers 4
```

- **EN→ZH（模型 A）**：`en_orig` → `zh_full`（技术术语译成中文）和 `zh_mixed`（技术术语保留英文）。两版都要求原样保留命令、路径、标识符、报错信息、引号内文字、版本号和软件名。
- **ZH→EN（模型 B）**：`zh_full` → `mt_plain_full`，`zh_mixed` → `mt_plain_mixed`。这一步看不到英文原题。按 proposal §5.3，两步必须用不同模型，以减轻回译偏差；脚本会拒绝两步使用同一模型。
- 支持任何 OpenAI 兼容接口（`--*-base-url`，key 从 `FWD_API_KEY` / `BACK_API_KEY` 读取），`--rpm` 可限速，`--limit N` 可先试几条。
- 每条结果写入 `data/translation_cache/` 后才继续，中断后重跑同一命令只会补缺失的条目。**改 prompt 或换模型时要先删掉缓存，并修改脚本中的 `PROMPT_VERSION`。**
- 同时输出 QA 报告 `data/query_variants_report.jsonl`：标记原文中的字面字符串在译文里可能被改动的条目（`missing_literals`），并统计 `zh_mixed` 与 `zh_full` 完全相同的条数。模型输出的 JSON 引号未转义时会自动修复，修复过的条目在缓存中带 `json_repaired: true`。

**当前使用的翻译设置**：Ollama 0.40.15 在 Great Lakes GPU 节点上本地运行，EN→ZH 用 `qwen2.5:14b`，ZH→EN 用 `gemma3:12b`，temperature 0，prompt `v2`。dev 与 test 必须用同一组模型和 prompt。Ollama 的安装方式（无需 sudo）：

```bash
mkdir -p ~/ollama && cd ~/ollama
curl -L -o ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst
tar --use-compress-program=zstd -xf ollama.tar.zst -C ~/ollama      # 缺 zstd 时：conda install -c conda-forge zstd
export PATH=$HOME/ollama/bin:$PATH LD_LIBRARY_PATH=$HOME/ollama/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=$HOME/ollama_models     # 可软链接到 /scratch 以节省 home 空间
ollama serve > ~/ollama/serve.log 2>&1 &
ollama pull qwen2.5:14b && ollama pull gemma3:12b
```

dev 翻译结果：528/528 条完成；52 条被 QA 标记（多为引号内报错信息被翻译）；235 条 `zh_mixed` 与 `zh_full` 完全相同。两类问题均待 W5 人工检查和修正。

## 运行实验

单个实验：

```bash
python -m si650_retrieval run --config configs/cqadupstack_hybrid.json
```

批量运行 dev 实验：

```bash
python scripts/make_dev_configs.py          # 在 configs/dev/ 下生成 15 个 config（已存在的跳过）
sbatch scripts/run_dev_gpu.sbatch           # 补翻缺失的 query → 依次跑所有 config（已完成的跳过）→ 汇总
python scripts/summarize_runs.py            # 汇总 runs/dev/*，输出 results/dev_summary.md
```

`summarize_runs.py` 输出各系统的分数表，并对 RQ1、RQ2 和 hybrid 的关键对比做配对随机化检验（sign-flip，10,000 次）。RQ2 的对比只用 `zh_mixed` 与 `zh_full` 确实不同的 query，因为两者相同时检索结果也相同，差值恒为 0。

Hybrid 用 `mt_plain_full` 检索英文 BM25，用 `zh_full` 做 direct dense，再以 alpha=0.5、c=60 融合。可复制配置调整：

| 实验 | 配置变化 |
| --- | --- |
| EN dense | `method: "dense"`, `dense_query: "en_orig"` |
| Direct CLIR | `method: "dense"`, `dense_query: "zh_full"` 或 `"zh_mixed"` |
| Translate + dense | `method: "dense"`, `dense_query: "mt_plain_full"` |
| Translate + BM25 | `method: "bm25"`, `lexical_query: "mt_plain_full"` |
| Random | `method: "random"`，改变 `seed`，分别运行 5 次 |

每次修改 `run_id` 和 `output`，保留所需 index 字段。**配置中的相对路径以配置文件所在目录为基准**。中文、译文共用同一英文语料索引，索引与 corpus hash 不匹配时拒绝运行。

可选重排：在配置加入以下对象，并将 revision 占位符换为真实模型 commit：

```json
{
  "reranker": {
    "model": "BAAI/bge-reranker-v2-m3",
    "revision": "REPLACE_WITH_MODEL_COMMIT",
    "query": "zh_full",
    "depth": 100,
    "max_length": 1024,
    "batch_size": 8
  }
}
```

仅重排头部候选；其余文档保持原序，同分保留一阶段顺序。`scores.jsonl` 保留真实分数，TREC 文件使用 `-rank` 保证评估顺序。模型输入采用库默认 pair truncation；精细的 query/title 保留策略留待正式实验前完善。

## 当前 dev 结果

528 条 dev query（android 139、programmers 175、unix 214），macro 为三个子论坛的平均值。完整结果见 `results/dev_summary.md`。

| Run | 角色 | nDCG@10 | MRR@10 | R@100 |
| --- | --- | --- | --- | --- |
| random（5 个 seed 平均） | Baseline | 0.000 | 0.000 | 0.005 |
| bm25_en | Reference (EN) | 0.331 | 0.333 | 0.624 |
| dense_en | Reference (EN) | 0.457 | 0.451 | 0.798 |
| bm25_mt_full | Translate + BM25 | 0.294 | 0.295 | 0.590 |
| bm25_mt_mixed | Translate + BM25 | 0.302 | 0.302 | 0.601 |
| dense_zh_full | Direct CLIR | 0.378 | 0.369 | 0.751 |
| dense_zh_mixed | Direct CLIR | 0.403 | 0.390 | 0.784 |
| dense_mt_full | Translate + dense | 0.417 | 0.402 | 0.793 |
| dense_mt_mixed | Translate + dense | 0.426 | 0.412 | 0.798 |
| hybrid_full（α=0.5） | Hybrid RRF | 0.381 | 0.372 | 0.773 |
| hybrid_mixed（α=0.5） | Hybrid RRF | 0.397 | 0.395 | 0.787 |

| 对比 | Query 数 | ΔnDCG@10 (A−B) | p |
| --- | --- | --- | --- |
| RQ1：dense_zh_full vs dense_mt_full | 528 | −0.042 | 0.0003 |
| RQ1：dense_zh_mixed vs dense_mt_mixed | 528 | −0.024 | 0.0066 |
| RQ2：dense_zh_full vs dense_zh_mixed | 293 | −0.045 | 0.0006 |
| RQ2：bm25_mt_full vs bm25_mt_mixed | 293 | −0.014 | 0.0829 |
| hybrid_full vs dense_zh_full | 528 | +0.003 | 0.7651 |
| hybrid_full vs bm25_mt_full | 528 | +0.083 | 0.0001 |

初步观察：

- 英文 BM25 在 android 上为 0.383，与 BEIR/Pyserini 公布的该子论坛 BM25 结果（约 0.38）接近，说明流程正确（公布值待在报告中核对引用）；其他子论坛的差异主要来自 dev 子集较小带来的波动。
- BGE-M3 在所有 query 版本上都明显优于 BM25；中文 query 的 dense 检索已超过英文 BM25。
- RQ1：同一 BGE-M3 下，先翻译成英文再检索优于直接用中文检索。这与通用领域文献的结论相反，但中文 query 由英文机器翻译而来，回译可能接近还原原题，**需用人工写的自然中文提问验证**。
- RQ2：保留英文术语（ZH-mixed）对 dense 检索有显著帮助（+0.045）。
- Hybrid 在 α=0.5 时未优于单一路线，可能因为 BM25 分支明显较弱；α 尚未调参。

局限：仅为 dev 结果；p 值未做 Holm 校正；翻译未经人工检查；hybrid 未调参。

## 输出与评估

每个 run 目录包含：

- `run.trec`：标准六列检索结果。
- `metrics.json`：各论坛、论坛等权 macro 的 nDCG@10 / MRR@10 / Recall@100，以及延迟。
- `per_query.jsonl`：逐查询指标，未命中查询计零。
- `scores.jsonl`：两路结果、一阶段候选、重排分数和最终顺序。
- `timings.jsonl`：查询级、阶段级耗时。
- `manifest.json`：配置、输入/输出/源码哈希、模型 revision、依赖版本和 Git commit（如有）。

当前内置指标将所有 `relevance > 0` 视为二元正例，已用手算样例测试；不适用于 0/1/2 人工 graded labels。`eval` extra 安装 `ir-measures`，用于可选指标交叉校验测试。计时包含逐查询处理，排除模型/索引加载与离线翻译，不做 warmup；这些时间不能直接解释成线上系统延迟。

默认只运行 dev。冻结配置后将 `split` 改为 `test`，并显式使用 `run --config ... --allow-test`。此开关防止误跑，不代替研究中的冻结记录。

## 验证与当前边界

```bash
PYTHONPATH=src python -m pytest -q
```

测试涵盖数据完整性、划分稳定性、BM25 数值、RRF、归一化、指标、索引哈希、query variants、random 可复现性、hybrid 和 reranker 编排及 CLI。toy BM25 已完成真实命令行端到端运行；模型相关编排测试使用受控替身，不证明 BGE-M3、FAISS 或 Java 环境已经可用。

已完成：真实数据导入与划分、Pyserini BM25 和 BGE-M3 索引、dev 集机器翻译、dev 上的全部基线与 hybrid 实验、配对随机化检验。

尚未完成：RRF α 调参、reranker 实验、术语保护翻译和 HyDE 两项改进、翻译人工检查（约 150 条）与 `zh_mixed` 的补充生成、30–50 条自然中文提问、test 集翻译与最终评估、Holm 校正、30 条 query 的人工补充标注、按 query 特征的失败分析，以及 mMARCO 扩展实验。依赖范围是安装约束，不是已验证的锁文件；正式环境建好后应记录精确版本。原始数据、模型、索引和运行结果由 `.gitignore` 排除，代码、配置、示例与测试纳入 Git。
