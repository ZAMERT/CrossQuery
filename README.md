# Chinese-to-English Technical Question Retrieval

SI650 基础检索 pipeline：数据准备 → BM25 / BGE-M3 → RRF → 可选 reranker → 评估。任务来自 [proposal](SI650_Project_Proposal.pdf)，完整研究设计见 [设计文档](docs/retrieval_pipeline.md)。

已实现 JSONL/BEIR 数据导入、论坛内固定 dev/test 划分、query variant 导入、random/BM25/dense/hybrid 检索、可选 cross-encoder 重排、TREC run、逐查询与论坛 macro 指标，以及配置和产物哈希记录。当前版本不自动调用翻译服务；中文和英文译文从文件读取。

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

默认模型 `BAAI/bge-m3`，doc max length=2048、query=256、batch=8；可用 `--device cpu`。模型接口使用 Sentence Transformers，显式关闭 query prompt，使用归一化向量及 FAISS `IndexFlatIP`。构建时记录实际模型 commit，查询时复用该 revision；也可显式传 `--revision <commit>`。[模型说明](https://huggingface.co/BAAI/bge-m3)、[编码接口](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)

准备 `data/query_variants.jsonl`，每个待运行查询一行，例如：

```json
{"query_id":"unix:111650","zh_full":"如何为 wireshark 设置正确的权限？","zh_mixed":"如何为 wireshark 设置正确的 permissions？","mt_plain_full":"How to set up wireshark with correct permissions?"}
```

这里仅展示字段格式，需补全真实 query ID 和所有 dev 查询的变体。缺失或空变体会报错，不会静默回退为英文原标题。变体文件不能修改 `en_orig`、`forum`、`split` 等原始元数据。

```bash
python -m si650_retrieval run --config configs/cqadupstack_hybrid.json
```

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

尚未下载真实数据和权重，未跑正式 Pyserini/BGE-M3/reranker 实验。自动翻译、术语保护、HyDE、显著性检验和人工错误分析尚未实现。依赖范围是安装约束，不是已验证的锁文件；正式环境建好后应记录精确版本。原始数据、模型、索引和运行结果由 `.gitignore` 排除，代码、配置、示例与测试纳入 Git。
