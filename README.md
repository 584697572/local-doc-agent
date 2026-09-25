# LocalDoc-Agent

面向本地知识库的 **Hybrid RAG / Tool-Calling Agent**。

项目从轻量 Tool-Calling Demo 演进而来，目前核心目标是：让 Agent 能够从本地 `.txt / .md / .pdf` 文档中进行可追溯的混合检索，并基于检索证据回答问题。

## Core Pipeline

```text
User Query
    |
    v
Tool-Calling Agent
    |
    v
search_documents
    |
    v
RetrievalEngine
    |
    +-------------------+
    |                   |
    v                   v
BM25 Retrieval     Dense Retrieval
(jieba + BM25)     (BGE + FAISS)
    |                   |
    +---------+---------+
              |
              v
          RRF Fusion
              |
              v
      Cross-Encoder Reranker
              |
              v
       Top-K Evidence + Source
              |
              v
          LLM Answer
```

## Features

- **Document Pipeline**：统一加载 TXT、Markdown、PDF，并切分为带 metadata 的 `DocumentChunk`
- **Sparse Retrieval**：BM25 + jieba，保留关键词、术语、标识符的精确召回能力
- **Dense Retrieval**：BGE Embedding + FAISS，支持语义相似检索
- **Hybrid Search**：RRF 融合 BM25 与 Dense 的排名结果
- **Reranking**：BGE Cross-Encoder 对候选证据进行二阶段精排
- **Citation Metadata**：保留 filename、page、section、chunk_id 等来源信息
- **Tool Calling**：LLM 通过 `search_documents` 工具访问本地知识库
- **Testability**：Embedding / Reranker 支持依赖注入，单元测试无需加载真实模型

## Project Structure

```text
local-doc-agent/
├── agent.py              # Tool-Calling Agent Loop
├── main.py               # CLI
├── config.py             # 统一配置
├── llm_client.py         # LLM Client
│
├── retrieval/
│   ├── document.py       # Document / DocumentChunk
│   ├── loader.py         # txt / md / pdf
│   ├── chunker.py        # fixed-window chunking
│   ├── bm25.py           # sparse retrieval
│   ├── dense.py          # embedding + FAISS
│   ├── fusion.py         # RRF
│   ├── reranker.py       # cross-encoder reranking
│   ├── hybrid.py         # hybrid pipeline
│   ├── engine.py         # knowledge-base build/search
│   └── result.py         # unified RetrievalResult
│
├── tools/
│   ├── registry.py       # Tool runtime
│   ├── schemas.py        # LLM tool schemas
│   └── retrieval.py      # search_documents tool
│
├── tests/
└── data/                 # local knowledge base; contents ignored by Git
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure LLM

Copy `.env.example` to `.env`:

```text
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. Add local documents

Put supported documents under:

```text
data/
```

Supported formats:

- `.txt`
- `.md`
- `.pdf`

The `data/` contents are ignored by Git by default to avoid accidentally publishing private documents.

### 4. Run

```bash
python main.py
```

Then ask a question directly, for example:

```text
这个知识库里是怎么解释 RAG 检索流程的？
```

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the full suite:

```bash
python -m pytest -v
```

Tests cover document loading, chunking, BM25, dense retrieval, RRF, reranking, hybrid retrieval, retrieval engine, and tool runtime.

## Design Decisions

### Why Hybrid Retrieval?

Dense retrieval is strong at semantic matching, while BM25 remains useful for exact terms such as function names, error codes, identifiers, and rare technical vocabulary. RRF combines both rankings without directly mixing incompatible score scales.

### Why a Reranker?

BM25 and Dense focus on candidate recall. A Cross-Encoder only evaluates the smaller fused candidate set, trading extra computation for better final ranking quality.

### Why custom Agent Loop instead of hiding everything behind a framework?

The project keeps Tool Calling, retrieval orchestration, execution limits, and error paths visible so each design choice can be explained and evaluated independently.

## Current Status

Completed:

- Document Pipeline
- BM25 Retrieval
- Dense Retrieval + FAISS
- RRF Fusion
- Cross-Encoder Reranker
- Hybrid Retriever
- Retrieval Engine
- Tool-Calling integration
- Unit tests for the retrieval stack

Next:

- Evidence sufficiency + query rewrite
- structured Agent state / execution budget / trace
- retrieval benchmark and ablation
- persistent / incremental indexing
- FastAPI + Docker
- benchmark-driven README results

> Benchmark numbers are intentionally not claimed yet. They will be added only after a reproducible evaluation set and ablation pipeline are complete.
