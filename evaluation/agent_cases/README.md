# 内部 Diagnostic Regression Suite v1.0.0

这套 case 用来定位 Agent 控制流回归，不是公开 Benchmark，也不能把通过率作为行业权威准确率。BEIR / MultiHop-RAG 的结果仍使用项目原有评测目录与报告。

## 内容与模式

固定知识库位于 knowledge_base/，包含八份约 300～1000 中文字的文档。Orion、Lyra、Cedar 及相关日期、代号均为虚构测试事实。Runner 不读取用户 data/，Live 期间会替换并恢复检索引擎缓存与知识库路径。

cases.jsonl 共 40 条：router、singlehop、rewrite_required、multihop、unanswerable、budget_control、groundedness、citation，各五条。

- scripted：40 条全部执行，不访问模型 API，也不下载检索模型。LLM 响应和搜索结果由固定脚本提供，Router / Judge / Rewrite 的解析、ToolRegistry、搜索预算和 Agent 主循环仍使用项目代码。脚本只读取 script 和 fixture，不读取 Gold 生成答案。
- live：真实 Router、Retrieval v1、Judge、Rewrite、回答和独立语义评审。执行 34 条；5 条预算故障 case 和 citation_004 仅用于 scripted，会明确列入 skipped。
- citation_004 的第 7 页是工具结果中注入的模拟元数据，用于验证页码保留，不代表 Markdown 本身有 PDF 页码。
- Rewrite 类的 Live Gold 不规定第一次证据不足，也不强制必须 Rewrite。只有脚本固定首轮证据时，scripted_gold 才要求这一控制流。

Retrieval v1 与 Safe Rank Fusion 的 1:2 权重保持原样。评测不修改生产 Prompt、Retriever 或 Judge 来迎合这些 case。

## 运行

在项目根目录、已安装 requirements.txt 和 pytest 的 Python 环境下执行：

~~~powershell
# 全量内部回归；可离线运行，但仍需安装项目依赖
python -m evaluation.agent_runner --mode scripted --output evaluation/results/agent/scripted.json

# 单例、类别和限量
python -m evaluation.agent_runner --mode scripted --case-id rewrite_001 --output evaluation/results/agent/rewrite.json
python -m evaluation.agent_runner --mode scripted --category budget_control --limit 5 --output evaluation/results/agent/budget.json

# 第一次建立基线；明确执行该命令才会重写精选 baseline
python -m evaluation.agent_runner --mode scripted --output evaluation/results/agent/scripted.json --baseline-output evaluation/results/agent/baseline_summary.json

# 真实模型小样本，需要 .env 中的 DEEPSEEK_API_KEY 和本地模型/下载条件
python -m evaluation.agent_runner --mode live --case-id singlehop_003 --output evaluation/results/agent/live_smoke.json

# 真实模型全量；需承担生成与语义评审两部分 API 调用成本
python -m evaluation.agent_runner --mode live --output evaluation/results/agent/live.json

# 完整代码回归
python -m pytest -v
~~~

--case-id 和 --category 可重复传入。--limit 必须为正整数；未知 ID、未知类别和空筛选都会报错。--cases 与 --knowledge-base 可指定另一个内部版本，不能用公开 Benchmark 替换本套标签后仍沿用内部基线名称。

## 报告与判定

输出 JSON 包含 summary、records、failure_counts、failure_analysis、configuration 和 skipped。逐例保存答案、实际 Evidence、独立工具执行记录、原始 Trace、预算、路由、Judge、改写和回答评价。

- 硬约束：预算超限、重复查询被接受或执行、执行与计数不一致、拒绝后仍执行、非成功工具内容进入 Evidence、引用不属于本次证据、应拒答时出现无依据断言、单例执行异常。任一违规 CLI 返回 1。
- 软指标：任务成功率、Groundedness、引用、拒答、Router/Judge/Rewrite，以及调用次数与延迟。v1 只报告，不设置 90% 之类门槛。
- 评审失败或筛选模式下没有可执行 case 返回 2，不会把“未评价”算成通过。错误 case 会保留记录，后面的 case 继续运行。
- target_evidence_sufficient 是任务期望；expected_evidence_sufficient 是对实际检索证据的独立评价。这样可以区分“检索没找到”和“Judge 把已有完整证据误判成不足”。
- first_evidence_sufficient 仅在可控脚本下作为额外路径断言。第一次/最终判断都在 records 中保留。
- Rewrite Success 要求改写后出现不同的成功证据、最终 Judge 确认充分、并保留所需实体/约束。重复返回相同证据不会算作改写收益。
- Groundedness 的脚本检查要求事实句来自真实返回的 fixture，拒答及控制停止提示单独识别。Live 使用语义评审，允许同义表达。
- 引用检查核对文件名和与文件绑定的页码；Live 另由语义评审判断引用是否支持所述事实。单纯出现正确文件名不能保证语义引用正确。
- 每条引用来自本次成功工具返回的 metadata.sources，绝不从 Gold 反推已经查到的来源。

失败分析提供主因、伴随错误、典型 case_id、搜索/改写次数分布和事件统计。预期且正确处理的 duplicate_query 等事件不会仅因为发生过就算任务失败。

## 如何理解基线

baseline_summary.json 是首次完整脚本运行的精选结果，configuration.simulated=true。它验证固定输入下的工程行为，不测真实模型的路由能力、语义理解、检索质量或真实推理耗时。完整逐例记录默认被 Git 忽略，精选 baseline 可提交。

当前 Agent 在连续 empty/error 后可能只输出“达到最大工具调用轮数”，而没有解释证据缺口。budget_004、budget_005 用于保留并暴露这类问题，不修改 Gold 来隐藏它们。后续修复控制流后重新运行，并与保存的基线比较。

Live 的自动语义评审也不是人工真值。评审模型与回答模型默认相同，可能有共同偏差；重要失败需要人工查看原始答案和证据。评审异常计入 assessment_errors，缺失结果保持 null。不要在同一套 case 上反复调 Prompt 直到 100%。

## 成本和可追溯性

llm_calls 在公共 create 入口计数，包含 Router、工具选择、Judge、Rewrite、最终回答，不包含 SDK 内部重试。assessment_llm_calls 单独统计，token_usage 只统计 Agent 请求返回的 token 用量。脚本模式的调用数是模拟请求次数。

每例 latency_ms 包含该例首次检索可能触发的初始化，排除后续语义评审；首例冷启动会影响 P95。评审耗时单独保存。脚本延迟不可与 Live 推理延迟直接比较。

配置记录 suite 版本、模型名、检索参数、预算、commit SHA、dirty 状态、代码 SHA256、case SHA256、逐份 fixture SHA256 和 UTC 时间。未提交代码也有内容指纹。runner 全程串行，临时替换不适合作为并发服务接口使用。

## 版本规则

此版本为初始 40 条，无删除记录。修正文案错字可以保留 ID；修改 Gold、fixture 中影响结论的事实或 case 任务语义时，应升级版本或新增 ID，并记录原因。对外报告须明确区分 Unit Tests、Internal Cases 和 Public Benchmark。
