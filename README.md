# LocalDoc-Agent

**面向本地文档的 Agentic RAG：混合检索、证据判断、查询改写、增量索引与 HTTP 服务。**

LocalDoc-Agent 将 TXT、Markdown、PDF 文档组织为可检索知识库。面对依赖文档的问题，Agent 先检索、再判断证据是否充分；必要时围绕信息缺口改写查询，在搜索预算内补充证据，最后生成带来源说明的回答。

项目使用显式的 Python Agent 循环，保留工具调用、证据判断、搜索预算和执行轨迹，便于测试、分析与解释，而不是只实现一次“向量检索 + LLM 回答”。

**已保存的实验结果：**在 MultiHop-RAG 的 100 条非空问题样本上，项目严格答案匹配率由单轮 RAG 的 **63% 提升至 77%**，17 条由错转对、3 条由对转错。该结果是特定样本与配置下的对照实验，不是全量官方榜单成绩。详见[评测与结果](#evaluation)。

> **隐私边界：**“本地”指文档存储、切块、向量生成和检索在本机或容器内执行。默认 LLM 使用 DeepSeek API，问题、传入的对话历史以及检索片段会在相应调用中发送给外部模型服务。项目不是完全离线系统，请勿未经授权导入或发送敏感文档。

[快速启动](#quick-start) · [系统架构](#architecture) · [API](#api) · [索引管理](#indexing) · [评测与结果](#evaluation) · [测试与复现](#reproduction) · [已知边界](#limitations)

---

## 核心能力

| 模块 | 已实现能力 |
| --- | --- |
| 文档处理 | 递归加载 `.txt`、`.md`、`.pdf`；固定字符窗口切块；保留文件名、页码、章节及 `chunk_id` 等元数据 |
| 混合检索 | BM25 + jieba 与 BGE Dense Retrieval 分别提供候选排名，使用 FAISS `IndexFlatIP` 检索向量，RRF 融合排名 |
| 精排与融合 | BGE Cross-Encoder 对候选精排，再以 Safe Rank Fusion 综合原始 RRF 排名与精排排名 |
| Agent 控制 | Router、Evidence Judge、Query Rewrite、查询去重、搜索预算与执行轨迹 |
| 索引生命周期 | 磁盘持久化、文件级新增/修改/删除、未变化向量复用、配置变更失效与损坏缓存重建 |
| 服务与部署 | 命令行、FastAPI、Swagger、健康与状态接口、Docker Compose；文档、索引和模型缓存分离 |
| 评测与回归 | BEIR/SciFact 检索消融、MultiHop-RAG 单轮/多轮对照、内部脚本回归，以及模型依赖注入测试 |

“Safe”表示融合时保留原始检索排名信息，**不表示安全认证、无幻觉保证或始终优于普通精排**。

<a id="quick-start"></a>
## 快速启动

以下命令以 **Windows PowerShell** 为例，均在仓库根目录执行。Docker 方式不要求在宿主机安装项目的 Python 依赖。

### 方式一：Docker Compose

需要已启动的 Docker Engine，以及可用的 Docker Compose。首次构建和模型首次加载需要访问依赖源及模型仓库；真实问答需要可用的 DeepSeek API Key。

#### 1. 获取代码并配置密钥

```powershell
git clone https://github.com/584697572/local-doc-agent.git
cd local-doc-agent

# 仅首次配置时复制；已有 .env 时不要覆盖。
Copy-Item .env.example .env
```

编辑 `.env`，将占位符换成自己的密钥：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

`.env` 不应提交到 Git，也不会复制进运行镜像。当前 Compose 只注入所需的 `DEEPSEEK_API_KEY`；模型名称等配置仍由 [`config.py`](config.py) 管理。

#### 2. 准备文档

将文档放入 `data/`，支持子目录。首次体验可以复制仓库自带的虚构测试资料：

```powershell
Copy-Item .\evaluation\agent_cases\knowledge_base\project_orion.md .\data\demo_orion.md
```

Orion 是演示用虚构项目，文档中的技术栈和日期不是 LocalDoc-Agent 自身的配置。使用自己的文档时可跳过这一步。

#### 3. 构建并启动

```powershell
docker compose up --build -d
docker compose ps
```

当前 Compose 在宿主机仅发布 `127.0.0.1:8000`，容器内 Uvicorn 监听 `0.0.0.0:8000`。正常情况下容器最终显示 `healthy`。

打开浏览器：

```text
http://127.0.0.1:8000/docs
```

在 `POST /v1/ask` 中点击 **Try it out**，输入以下 JSON，再点击 **Execute**：

```json
{
  "query": "根据本地文档，Orion 项目的负责人代号和演示发布日期分别是什么？请注明来源。",
  "history": [],
  "include_trace": false
}
```

示例资料中的对应事实是负责人代号 **O17**、演示发布日期 **2026-03-18**。回答措辞可能不同，应核对事实与来源，而不是要求字符串完全一致。

#### 常用容器命令

```powershell
# 查看日志
docker compose logs --tail=200 local-doc-agent

# 查看索引状态；此接口不会主动建库
Invoke-RestMethod http://127.0.0.1:8000/v1/index/status | ConvertTo-Json -Depth 10

# 文档变化后重启服务，下一次需要检索的请求会重新检查索引
docker compose restart local-doc-agent

# 停止并移除容器，保留持久数据
docker compose down

# 使用已有镜像重新创建容器
docker compose up -d
```

修改应用源码后，使用 `docker compose up --build -d` 更新镜像和容器；仅重启容器不会把宿主机新源码复制进镜像。

**普通停止/重建不要使用 `docker compose down -v`**：它会删除本项目的 Hugging Face named volume，导致模型缓存丢失；宿主机 bind mount 中的文档和索引目录与该 named volume 不是同一种存储。

### 方式二：本地 Python

本项目开发环境使用过 Python 3.13，Docker 基础镜像为 `python:3.13-slim`。先完成上面的 `.env` 和文档准备，再安装开发依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# HTTP 服务
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

也可以运行命令行入口：

```powershell
.\.venv\Scripts\python.exe main.py
```

输入 `exit`、`quit` 或 `退出` 结束命令行会话。不要让本地 Uvicorn 与 Docker 服务同时占用宿主机的 8000 端口。

依赖分为两层：[`requirements-runtime.txt`](requirements-runtime.txt) 用于服务运行；[`requirements.txt`](requirements.txt) 复用运行依赖，并增加 pytest、HTTP 测试客户端和 BEIR。

<a id="architecture"></a>
## 系统架构

```text
CLI / HTTP 客户端
        │
        ▼
      Router ── 不需要知识库 ──► 直接回答
        │
        │ 需要知识库
        ▼
  LLM 生成搜索工具调用
        │
        ▼
  查询去重与搜索预算检查
        │
        ▼
  search_documents / RetrievalEngine
        │
        ├── BM25 + jieba
        └── BGE Embedding + FAISS
                    │
                    ▼
                RRF 融合
                    │
                    ▼
          Cross-Encoder 精排
                    │
                    ▼
             Safe Rank Fusion
                    │
                    ▼
             Top-K 证据与来源
                    │
                    ▼
              Evidence Judge
                    │
       ┌────────────┴──────────────────┐
       │ 证据足够                      │ 证据不足且仍有预算
       ▼                               ▼
    最终回答                        Query Rewrite
                                       │
                                       └──► 预算检查 → 再次检索
```

证据仍不足且无法继续检索时，最终回答路径被提示只回答可确认的部分并说明信息缺口；达到主循环轮数上限时会停止执行。证据判断由 LLM 完成，因此它是控制信号，不是答案正确性的证明。

### 混合检索与 Safe Rank Fusion

[`retrieval/hybrid.py`](retrieval/hybrid.py) 将检索分成三个层次：候选召回、Cross-Encoder 精排、排名融合。运行时默认 BM25 和 Dense 各取最多 20 个结果，经 RRF 选出最多 20 个候选，再精排并返回 Top-5。

RRF 使用排名，而不是直接相加不同检索器的原始分数：

```text
RRF(d) = Σ 1 / (60 + rank_i(d))
```

随后，运行时的 Safe Rank Fusion 使用：

```text
score(d) = 1 / (60 + rrf_rank(d))
         + 2 / (60 + reranker_rank(d))
```

这里的 `1:2` 是**原始 RRF 排名与 Reranker 排名**的权重，不是 BM25 与 Dense 的权重，也不是再运行一个精排模型。公式实现见 [`retrieval/fusion.py`](retrieval/fusion.py)。

重排只能调整候选顺序，不能找回候选池外的文档。证据缺失时，Query Rewrite 的作用是构造新的检索请求，而不是反复重排同一批结果。

### Agent 控制与可追溯性

[`agent.py`](agent.py) 负责工具调用主循环；[`harness/`](harness/) 管理路由、证据判断、改写、预算与轨迹。每次问题默认最多接受 3 次搜索，重复查询被拒绝，只有成功工具结果进入证据集合。

`run_agent()` 保留命令行的字符串返回接口；`run_agent_with_trace()` 返回答案、状态、耗时与逻辑 LLM 调用计数，供 API 和内部评测使用。当前 `search_documents` 只检索文档，不提供执行任意 Shell 命令或修改原始文件的工具。

<a id="api"></a>
## HTTP API

| 方法 | 路径 | 含义 |
| --- | --- | --- |
| GET | `/health` | HTTP 服务存活检查，不加载模型或调用 LLM |
| GET | `/ready` | 检查 API Key 非空与 `data/` 目录可用；未满足时返回 503 |
| GET | `/v1/index/status` | 读取运行时索引状态或磁盘 manifest 快照，不触发索引构建 |
| POST | `/v1/ask` | 执行一次 Agent 问答 |
| GET | `/docs` | 交互式接口文档 |

`/ready` 是**轻量配置检查**，不验证密钥是否有效、外部 API 是否可达、模型是否已加载，也不保证问题一定能回答。`healthy` 同样不等价于真实问答成功。

### 请求与响应

请求字段由 [`api/schemas.py`](api/schemas.py) 定义：`query` 最长 4000 个字符，拒绝纯空白；`history` 只允许 `user` / `assistant`，最多 8 条消息，单条内容最长 12000 个字符；`include_trace` 默认关闭。

API 不保存服务器端聊天会话。需要多轮对话时，由客户端将历史消息传入 `history`。

下面是**结构示意，不是某次实测结果**：

```json
{
  "answer": "Orion 的负责人代号为 O17，演示发布日期为 2026-03-18。来源：demo_orion.md。",
  "latency_ms": 25000.0,
  "usage": {
    "llm_calls": 4
  },
  "retrieval": {
    "search_count": 1,
    "max_search_calls": 3,
    "evidence_count": 1,
    "evidence_sufficient": true,
    "missing_aspects": []
  },
  "trace": null
}
```

`evidence_count` 是累计成功检索结果块的数量，**不是唯一文件数或 Chunk 数**；一次检索结果块可包含多个 Chunk。`llm_calls` 是应用层逻辑调用计数，不是 Token 用量，也不包含 SDK 内部重试。`latency_ms` 统计 Agent 函数内部耗时，不等同于包含锁等待、首次导入和网络传输的完整 HTTP 延迟。

`include_trace=true` 会返回查询、工具结果、证据判断及相关元数据，可能包含文档原文，仅用于受控调试。请求格式错误返回 422；当前 Agent 执行异常统一返回不含异常原文的 500 响应，详细异常记录在服务端日志。

实现见 [`api/app.py`](api/app.py)。

<a id="indexing"></a>
## 持久化与增量索引

索引目录与原始文档分开保存：

```text
.localdoc_index/
├── manifest.json    # 配置、源文件 SHA256、文件对应的 Chunk 范围、缓存校验值
├── chunks.json      # 文本块与来源元数据
├── embeddings.npy   # 与 Chunk 顺序一一对应的原始向量矩阵
└── dense.faiss      # 搜索用 FAISS 索引；空知识库时可不存在
```

`chunks[i]` 与 `embeddings[i]` 一一对应。manifest 记录每个源文件的 `chunk_start` 和 `chunk_count`，便于定位可复用的行。

| 场景 | 构建方式 | 文档 Embedding 行为 |
| --- | --- | --- |
| 首次构建、配置不兼容或缓存损坏 | `full_rebuild` | 重新处理全部有效文本块 |
| 源文件和配置均未变化 | `cache_hit` | 复用已有向量，不重新编码文档 |
| 新增或修改文件 | `incremental_update` | 仅重新解析、切块并编码受影响文件 |
| 删除文件 | `incremental_update` | 移除对应文本块和向量；其余文件不重新编码 |

**增量粒度是文件，不是句子或单个 Chunk。** 修改某个文件后，该文件的所有新 Chunk 都会重新编码。BM25 会重新构造；源文件变化时，FAISS `IndexFlatIP` 也会使用合并后的向量重新构造。节省的是未变化文档的解析与模型编码，不代表整个索引更新是常数时间操作。

缓存使用文件校验值与数量/维度检查；缓存不可用时退回重建。各缓存文件分别写入临时文件并替换，manifest 最后写入；这不是多进程并发写入事务，不应让多个写入进程同时维护同一索引目录。

### 文档更新如何生效

当前 Engine 在进程内复用，**没有文件监听或自动热更新接口**。修改 `data/` 后，重启服务；下一次触发检索时，`build()` 比较文件状态并选择增量或全量构建：

```powershell
docker compose restart local-doc-agent
```

在 Python 中显式调用 `engine.build()` 也会重新检查；`engine.build(force_rebuild=True)` 可强制全量重建。

`cache_status=rebuilt` 与 `update_mode=incremental_update` 不矛盾：前者说明缓存被重新生成，后者说明生成时复用了未变化的向量。

### 延迟加载与容器存储

Embedding / Reranker 使用延迟导入和模型延迟加载。缓存命中时不需要重新编码文档，但第一次真实搜索仍要加载模型，并计算查询向量。**首次请求的模型冷启动成本并没有消失。**

当前 Compose 的存储关系如下：

| 内容 | 宿主机或 Volume | 容器路径 | 权限 |
| --- | --- | --- | --- |
| 用户文档 | `./data` bind mount | `/app/data` | 只读 |
| 索引与原始向量 | `./.localdoc_index` bind mount | `/app/.localdoc_index` | 可写 |
| 模型缓存 | `huggingface-cache` named volume | `/root/.cache/huggingface` | 可写 |

容器被删除后，以上持久数据可继续挂载到新容器。镜像只复制运行所需源码；私人文档、密钥、索引、评测数据和测试目录不作为 Runtime 镜像内容。配置见 [`Dockerfile`](Dockerfile)、[`compose.yaml`](compose.yaml) 与 [`.dockerignore`](.dockerignore)。

<a id="configuration"></a>
## 默认配置

| 配置 | 当前运行时默认值 |
| --- | --- |
| LLM | `deepseek-flash`，通过兼容 OpenAI 的客户端调用 DeepSeek API |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-base` |
| Chunk 大小 / 重叠 | 500 / 100，单位为字符，不是 Token |
| 最终检索 Top-K / 候选数 | 5 / 20 |
| RRF / Safe Fusion 平滑常数 | 60 / 60 |
| Safe Fusion 权重 | RRF : Reranker = 1 : 2 |
| 每个问题搜索上限 / 主循环上限 | 3 次搜索 / 5 轮 |
| 对话历史 | 最多 4 对消息，由 CLI 保留或客户端传入 |

通用配置见 [`config.py`](config.py)，检索模型与融合参数见 [`retrieval/dense.py`](retrieval/dense.py)、[`retrieval/reranker.py`](retrieval/reranker.py)、[`retrieval/hybrid.py`](retrieval/hybrid.py)。并非所有配置都支持环境变量覆盖；修改已复制到镜像中的源码配置后，需要重新构建镜像。

英文公开评测显式使用 `BAAI/bge-small-en-v1.5`，不要把英文实验配置与中文运行时默认值混为一谈。

<a id="evaluation"></a>
## 评测与结果

项目将三类问题分开评测：**检索能否找到相关材料、迭代检索能否改善回答、控制流与异常路径是否符合约束**。下面的数据来自仓库保存的历史报告，不表示每次运行都会得到相同结果。

### 1. BEIR / SciFact：检索消融

保存结果使用 SciFact `test` 的 300 个查询，一个 BEIR 文档对应一个评测 Chunk，不采用运行时的 500 字符切分。Embedding 为 `bge-small-en-v1.5`，Reranker 为 `bge-reranker-base`；两路检索深度 100，精排候选 20。

| 方法 | Hit@5 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: |
| BM25 | 0.7533 | 0.7268 | 0.6126 |
| Dense | 0.7833 | 0.7653 | 0.6743 |
| RRF | 0.7733 | 0.7567 | 0.6707 |
| Rerank | 0.7967 | 0.7799 | 0.6800 |
| Safe Rerank 1:2 | **0.8167** | **0.7942** | **0.6922** |

Hit@5 表示前 5 个结果至少命中一个相关文档，**不是问答正确率**。在这份实验中，Safe Rerank 的 Hit@5 比普通 Rerank 高 2 个百分点；它是配置选择的实验依据，不是未经调参的独立泛化证明，也不表示所有数据集上都更优。

数据与实现：[完整结果](evaluation/results/scifact_bge-small-en-v1.5_safe-rerank_1-2_all.json) · [BEIR Adapter](evaluation/beir_adapter.py) · [评测 Runner](evaluation/beir_runner.py)。

### 2. MultiHop-RAG：单轮 RAG 与迭代检索对照

样本使用 `seed=42` 按题型分层选择 100 条非 `null_query` 问题：34 条 inference、33 条 comparison、33 条 temporal。语料为 609 篇文档，切分为 16,200 个 Chunk；切块参数 500/100，英文 Embedding、Reranker 及 Safe 1:2 配置固定。

两组复用相同的第一次检索结果：Single-shot 取 Top-5 后回答；Agentic 先做 Evidence Judge，需要时改写并再次检索，最多总计 3 次搜索，每次 Top-5，按 Chunk ID 去重合并证据后回答。

| 指标 | Single-shot RAG | Agentic RAG |
| --- | ---: | ---: |
| 严格答案匹配率 `strict_answer_match` | 63.00% | **77.00%** |
| 平均 Gold 父文档 Recall | 65.42% | 80.75% |
| 全部 Gold 父文档覆盖率 | 38.00% | 53.00% |
| 平均 Gold Fact Recall（字符串匹配） | 33.92% | 44.25% |
| 全部 Gold Facts 覆盖率（字符串匹配） | 14.00% | 20.00% |

配对结果为：**60 条保持正确、17 条由错转对、3 条由对转错、20 条保持错误**，严格评分净提升 **14 个百分点**。

收益同时伴随成本：该报告中 Agentic 平均搜索 2.15 次、改写 1.15 次，37% 的样本达到预算耗尽条件。报告记录的组件累计平均在线耗时约为 Single-shot 7.89 秒、Agentic 40.73 秒；这些是评测机器上的特定运行记录，排除单独统计的索引构建时间，不是当前 HTTP 服务的延迟承诺。

**如何解释这组结果：**

- 主指标是项目自动规则：Yes/No 检查标准化答案的首个 Token；其他答案检查完整 Gold Token 序列是否出现。简称会被单独标记为语义复核候选，自动匹配并不等于人工语义判断。
- Gold Fact Recall 使用规范化后的字符串包含匹配；缺少精确文本不必然表示缺少同义证据，父文档齐全也不表示必要段落已被检索到。失败分类属于诊断信号，不能直接证明某一组件是唯一原因。
- Agentic 最多可累计 15 个去重前的 Chunk，而 Single-shot 只使用 5 个。因此这是**带额外搜索、证据和 LLM 预算的整体策略对照**，不是等上下文长度、等计算成本的严格消融。
- MultiHop Agentic Runner 复用 Judge、Rewrite 等组件，但不包含 HTTP 入口和生产 Agent 的完整 Router/Tool-Selection 流程，不能把其调用数直接当作 `/v1/ask` 的调用数。
- 这里只覆盖 100 条英文非空问题；没有据此报告全量 MultiHop-RAG 成绩、中文通用准确率或 `null_query` 拒答能力。

**模型版本说明：**保存的 Single-shot 与 Agentic 报告均记录 `deepseek-v4-flash`；当前运行时配置为 `deepseek-flash`。重新运行会使用当前配置，不能把历史结果当作当前模型重新测得的成绩。

数据与评分：[Retrieval](evaluation/results/multihop_retrieval_100.json) · [Single-shot](evaluation/results/multihop_single_shot_100.json) · [Agentic](evaluation/results/multihop_agentic_100.json) · [Context Analysis](evaluation/results/multihop_context_analysis_100.json) · [QA Metrics](evaluation/multihop_qa_metrics.py)。

### 3. 内部诊断回归：验证行为，不替代公开 Benchmark

内部套件包含 40 条用例和 8 份虚构中文资料，覆盖路由、单跳、多跳、改写、不可回答问题、预算、证据约束与引用。`scripted` 模式使用固定 LLM/搜索响应，不调用真实模型；`live` 模式用于真实链路检查，具体规则见[内部用例说明](evaluation/agent_cases/README.md)。

保存的 [`baseline_summary.json`](evaluation/results/agent/baseline_summary.json) 中，scripted 任务成功为 **38/40**，硬约束检查通过；`budget_004`、`budget_005` 保留了连续工具空结果/错误后的行为问题。**38/40 不是 pytest 通过数，更不是模型真实准确率；脚本耗时也不是模型推理耗时。** 该基线记录了旧提交与 dirty 工作区状态，应作为历史诊断记录使用。

<a id="reproduction"></a>
## 测试与评测复现

以下命令在安装了 `requirements.txt` 的本地 Python 环境执行；使用上面的虚拟环境但未激活时，将 `python` 替换成 `.\.venv\Scripts\python.exe`。Runtime Docker 镜像不包含 `evaluation/` 和 `tests/`，不在该镜像内运行评测。

### 单元测试与内部回归

```powershell
python -m pytest -v

# 固定脚本回归，不发起真实 LLM 请求
python -m evaluation.agent_runner --mode scripted --output evaluation/results/agent/local_check.json

# 真实模型增量索引检查：使用临时目录，不修改用户 data/
python -m evaluation.verify_incremental_index
```

单元测试使用 FakeEncoder、FakeReranker、模拟客户端等隔离外部依赖，覆盖检索、预算、改写、持久化、增量更新和 API。真实增量验证脚本需要加载 Embedding 模型，首次可能下载模型；它打印各场景，离线专项测试位于 [`tests/test_incremental_index.py`](tests/test_incremental_index.py)。

### BEIR / SciFact

```powershell
python -m evaluation.beir_runner `
  --dataset scifact --split test --limit 0 `
  --top-k 5 --retrieval-depth 100 --rerank-candidate-k 20 `
  --embedding-model BAAI/bge-small-en-v1.5 `
  --reranker-model BAAI/bge-reranker-base `
  --rrf-k 60 --safe-fusion-k 60 `
  --safe-rrf-weight 1.0 --safe-reranker-weight 2.0 `
  --output evaluation/results/local_scifact.json
```

`--limit 0` 使用该 split 的全部查询；首次运行需要取得公开数据和检索模型，不调用问答 LLM。

<details>
<summary><strong>MultiHop-RAG 100 条完整复现命令</strong></summary>

这组命令会加载检索模型，Single-shot / Agentic 阶段还会产生真实 LLM API 调用与费用。输出采用 `local_` 前缀，避免覆盖已提交的历史结果。

```powershell
# 下载并适配公开数据
python -m evaluation.multihop_adapter

# 第一轮检索结果
python -m evaluation.multihop_retrieval_runner `
  --limit 100 --seed 42 --chunk-size 500 --overlap 100 `
  --retrieval-depth 100 --rerank-candidate-k 20 `
  --embedding-model BAAI/bge-small-en-v1.5 `
  --reranker-model BAAI/bge-reranker-base `
  --rrf-k 60 --safe-fusion-k 60 `
  --safe-rrf-weight 1.0 --safe-reranker-weight 2.0 `
  --output evaluation/results/local_multihop_retrieval_100.json

# Single-shot
python -m evaluation.multihop_single_shot_runner `
  --retrieval-report evaluation/results/local_multihop_retrieval_100.json `
  --method safe_rerank --top-k 5 `
  --output evaluation/results/local_multihop_single_shot_100.json `
  --checkpoint evaluation/results/checkpoints/local_multihop_single_shot_100.json

# Agentic
python -m evaluation.multihop_agentic_runner `
  --retrieval-report evaluation/results/local_multihop_retrieval_100.json `
  --single-shot-report evaluation/results/local_multihop_single_shot_100.json `
  --method safe_rerank --top-k 5 `
  --output evaluation/results/local_multihop_agentic_100.json `
  --checkpoint evaluation/results/checkpoints/local_multihop_agentic_100.json

# 离线证据覆盖诊断，不再次调用 LLM
python -m evaluation.multihop_context_analysis `
  --retrieval-report evaluation/results/local_multihop_retrieval_100.json `
  --single-shot-report evaluation/results/local_multihop_single_shot_100.json `
  --output evaluation/results/local_multihop_context_analysis_100.json
```

Single-shot / Agentic 中断后，在原命令末尾追加 `--resume`，保持输入报告、配置与 checkpoint 路径不变。检索 Runner 不提供这两个 QA Runner 相同的断点续跑机制。

复现时同时记录代码提交、依赖版本、模型名称、数据与参数。当前依赖未完全锁定，远端模型及运行环境也可能变化，因此命令复现不等于分数逐位一致。

</details>

### Docker 最小验收

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/index/status | ConvertTo-Json -Depth 10
```

通过 Swagger 执行一次需要文档检索的问题，检查 HTTP 200、答案与引用，再查看索引状态。随后执行 `docker compose down`、`docker compose up -d`，重新问答并检查 `update_mode=cache_hit`、`encoded_chunks=0`。计数以自己的知识库为准，不应固定要求 186 个 Chunk。

<a id="structure"></a>
## 项目结构

```text
local-doc-agent/
├── agent.py                    # Agent 主循环、工具调用与 Trace 返回
├── main.py                     # 命令行入口
├── config.py                   # 模型、目录与预算配置
├── llm_client.py               # DeepSeek API 客户端
├── api/                        # FastAPI 路由与请求/响应模型
├── harness/                    # Router、Judge、Rewrite、AgentState
├── retrieval/                  # Loader、Chunker、BM25、Dense、融合、精排、索引引擎
├── tools/                      # 工具注册、Schema、ToolResult、检索适配
├── evaluation/
│   ├── agent_cases/            # 内部用例与虚构知识库
│   ├── agent_runner.py         # 内部 scripted/live 回归
│   ├── beir_runner.py          # BEIR 检索消融
│   ├── multihop_*              # MultiHop 数据适配、检索、QA 与诊断
│   ├── verify_incremental_index.py
│   └── results/                # 已保存的实验与基线
├── tests/                      # 自动化测试
├── data/                       # 用户文档；内容默认不提交
├── .localdoc_index/             # 运行时生成的索引；不提交
├── requirements.txt            # 开发、测试与评测依赖
├── requirements-runtime.txt    # 服务运行依赖
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
└── .env.example
```

<a id="limitations"></a>
## 已知边界

本项目面向单机本地知识库和可解释的 Agent 工程实践，当前有以下边界：

- **不是完整文档管理产品。** 没有文件上传、OCR、全文摘要专用流程、聊天数据库或前端界面。PDF 依赖可提取文本；普通 Top-K 检索摘要不保证覆盖全文。
- **文档标识需要唯一。** 当前 Loader 使用不含扩展名的文件名构造 `document_id`；请避免在不同目录或扩展名下使用相同的基础文件名，以免 Chunk 标识冲突。
- **证据与回答仍需审查。** Judge 可能误判，改写可能无收益，答案可能遗漏或误解证据。引用元数据可追溯，不等于逐条结论已经被独立核验。
- **当前不是高并发生产服务。** Ask 请求使用进程内锁串行执行；未提供用户鉴权、限流、跨进程索引写锁或请求级超时治理。默认只绑定本机端口，不应直接公开到互联网。
- **状态与数据有明确时间点。** `/v1/index/status` 的未加载快照没有校验当前源文件；已加载状态反映最近一次构建，不代表实时监听到文档变化。
- **模型与缓存有外部依赖。** 第一次可能下载模型，后续仍需将权重加载到内存；模型文件可缓存不等于问答完全离线。当前 CPU 容器镜像及 Python 依赖未做完全锁版。

当前实现优先保证控制逻辑可见、结果可检查和部署路径完整。后续优化应以新的评测或真实使用需求为依据，而不是以增加组件数量为目标。
