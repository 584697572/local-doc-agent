"""Agent 主控制流测试。"""

from types import SimpleNamespace

import agent
from harness.evidence import EvidenceDecision
from harness.rewrite import RewriteResult
from harness.router import RouteDecision
from tools.result import ToolResult


def make_tool_call(
    query: str,
    call_id: str = "call_1",
):
    """
    构造假的 search_documents Tool Call。
    """

    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="search_documents",
            arguments=f'{{"query": "{query}"}}',
        ),
    )


def make_response(
    content=None,
    tool_calls=None,
):
    """
    构造假的 LLM Response。
    """

    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
            )
        ]
    )


def force_retrieval(monkeypatch):
    """
    测试 Agentic Retrieval 时，
    固定让 Router 判定必须查询知识库。
    """

    monkeypatch.setattr(
        agent,
        "route_query",
        lambda user_query: RouteDecision(
            needs_retrieval=True,
            reason="测试中强制进入 Retrieval。",
        ),
    )


def test_agent_rewrite_search_then_answer(
    monkeypatch,
):
    """
    验证：

    Router=True
    → Search
    → Evidence 不足
    → Rewrite
    → Search Again
    → Evidence 足够
    → Final Answer
    """

    force_retrieval(monkeypatch)

    searched_queries = []

    def fake_run_tool(tool_name, arguments):
        query = arguments["query"]
        searched_queries.append(query)

        if len(searched_queries) == 1:
            return ToolResult.success(
                content=(
                    "初始证据：只有 "
                    "Cluster-State Graph 的定义。"
                )
            )

        return ToolResult.success(
            content=(
                "补充证据：包含节点、边和构造方法。"
            )
        )

    monkeypatch.setattr(
        agent,
        "run_tool",
        fake_run_tool,
    )

    # 第一次证据不足，第二次证据足够
    def fake_evaluate_evidence(
        user_query,
        evidence,
    ):
        if len(evidence) == 1:
            return EvidenceDecision(
                sufficient=False,
                reason="缺少节点、边和构造方式。",
                missing_aspects=[
                    "节点定义",
                    "边定义",
                    "构造方式",
                ],
            )

        return EvidenceDecision(
            sufficient=True,
            reason="当前证据已经覆盖主要信息需求。",
            missing_aspects=[],
        )

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        fake_evaluate_evidence,
    )

    # Evidence 不足后生成定向 Rewrite Query
    monkeypatch.setattr(
        agent,
        "rewrite_query",
        lambda original_query, missing_aspects, used_queries: (
            RewriteResult(
                query=(
                    "Cluster-State Graph "
                    "节点 边 构造方式"
                ),
                reason="补充当前缺失的图结构和构造信息。",
            )
        ),
    )

    # 第一次调用：主 Agent 产生 Tool Call
    # 第二次调用：生成最终答案
    responses = [
        make_response(
            tool_calls=[
                make_tool_call(
                    "Cluster-State Graph"
                )
            ]
        ),
        make_response(
            content="这是最终回答。",
            tool_calls=None,
        ),
    ]

    def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        fake_create,
    )

    reply = agent.run_agent(
        user_input=(
            "Cluster-State Graph 是什么，"
            "怎么构造？"
        ),
        chat_history=[],
    )

    assert reply == "这是最终回答。"

    # 应该先搜初始 Query，再搜 Rewrite Query
    assert searched_queries == [
        "Cluster-State Graph",
        "Cluster-State Graph 节点 边 构造方式",
    ]


def test_agent_answers_without_rewrite_when_evidence_is_sufficient(
    monkeypatch,
):
    """
    验证：

    Router=True
    → Search
    → 第一次 Evidence 就足够
    → 不 Rewrite
    → Final Answer
    """

    force_retrieval(monkeypatch)

    searched_queries = []

    def fake_run_tool(tool_name, arguments):
        query = arguments["query"]
        searched_queries.append(query)

        return ToolResult.success(
            content=(
                "完整证据：包含定义、节点、"
                "边和构造方式。"
            )
        )

    monkeypatch.setattr(
        agent,
        "run_tool",
        fake_run_tool,
    )

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda user_query, evidence: EvidenceDecision(
            sufficient=True,
            reason="证据已经覆盖用户问题。",
            missing_aspects=[],
        ),
    )

    # 如果 Evidence 已经足够，却仍然 Rewrite，
    # 测试直接失败。
    def should_not_rewrite(*args, **kwargs):
        raise AssertionError(
            "Evidence 已经足够，不应该执行 Query Rewrite"
        )

    monkeypatch.setattr(
        agent,
        "rewrite_query",
        should_not_rewrite,
    )

    responses = [
        make_response(
            tool_calls=[
                make_tool_call(
                    "Cluster-State Graph"
                )
            ]
        ),
        make_response(
            content="这是基于完整证据的最终回答。",
            tool_calls=None,
        ),
    ]

    def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        fake_create,
    )

    reply = agent.run_agent(
        user_input="Cluster-State Graph 是什么？",
        chat_history=[],
    )

    assert reply == (
        "这是基于完整证据的最终回答。"
    )

    # Evidence 第一次就够，只应该搜索一次
    assert searched_queries == [
        "Cluster-State Graph"
    ]


def test_agent_stops_when_search_budget_is_exhausted(
    monkeypatch,
):
    """
    验证：

    Router=True
    → Evidence 一直不足
    → Rewrite + Search
    → 达到 MAX_SEARCH_CALLS
    → 强制停止
    → 生成保守回答
    """

    force_retrieval(monkeypatch)

    searched_queries = []
    judge_call_count = 0

    def fake_run_tool(tool_name, arguments):
        query = arguments["query"]
        searched_queries.append(query)

        return ToolResult.success(
            content=f"来自 {query} 的部分证据。"
        )

    monkeypatch.setattr(
        agent,
        "run_tool",
        fake_run_tool,
    )

    # 无论获得多少 Evidence，都判定不足
    def fake_evaluate_evidence(
        user_query,
        evidence,
    ):
        nonlocal judge_call_count

        judge_call_count += 1

        return EvidenceDecision(
            sufficient=False,
            reason="当前证据仍然缺少完整构造算法。",
            missing_aspects=[
                "完整构造算法",
            ],
        )

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        fake_evaluate_evidence,
    )

    # 每次产生不同 Query，避免被去重
    def fake_rewrite_query(
        original_query,
        missing_aspects,
        used_queries,
    ):
        next_number = len(used_queries) + 1

        return RewriteResult(
            query=f"补充检索 Query {next_number}",
            reason="继续搜索当前缺失的信息。",
        )

    monkeypatch.setattr(
        agent,
        "rewrite_query",
        fake_rewrite_query,
    )

    responses = [
        make_response(
            tool_calls=[
                make_tool_call(
                    "初始 Query"
                )
            ]
        ),
        make_response(
            content=(
                "现有证据不足，"
                "以下只回答能够确认的部分。"
            ),
            tool_calls=None,
        ),
    ]

    def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        fake_create,
    )

    reply = agent.run_agent(
        user_input="请给出完整构造算法。",
        chat_history=[],
    )

    assert reply == (
        "现有证据不足，"
        "以下只回答能够确认的部分。"
    )

    # 真正搜索次数不能超过预算
    assert len(searched_queries) == (
        agent.MAX_SEARCH_CALLS
    )

    assert searched_queries == [
        "初始 Query",
        "补充检索 Query 2",
        "补充检索 Query 3",
    ]

    # 每获得一次新 Evidence，
    # 都应该重新执行 Evidence Judge
    assert judge_call_count == (
        agent.MAX_SEARCH_CALLS
    )


def test_agent_answers_directly_when_retrieval_is_not_needed(
    monkeypatch,
):
    """
    验证：

    Router=False
    → 不执行 Retrieval
    → 直接回答。
    """

    monkeypatch.setattr(
        agent,
        "route_query",
        lambda user_query: RouteDecision(
            needs_retrieval=False,
            reason="简单寒暄。",
        ),
    )

    # Router=False 时如果执行 Tool，
    # 说明控制流错误。
    def should_not_run_tool(*args, **kwargs):
        raise AssertionError(
            "Router=False 时不应该执行 Retrieval"
        )

    monkeypatch.setattr(
        agent,
        "run_tool",
        should_not_run_tool,
    )

    response = make_response(
        content="你好！有什么可以帮你的？",
        tool_calls=None,
    )

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        lambda **kwargs: response,
    )

    reply = agent.run_agent(
        user_input="你好",
        chat_history=[],
    )

    assert reply == (
        "你好！有什么可以帮你的？"
    )


def test_agent_does_not_accept_direct_answer_when_retrieval_is_required(
    monkeypatch,
):
    """
    验证：

    Router=True
    → 主模型第一次试图直接回答
    → Harness 不接受
    → 下一 Step 必须 Retrieval
    → 最终基于 Evidence 回答。
    """

    monkeypatch.setattr(
        agent,
        "route_query",
        lambda user_query: RouteDecision(
            needs_retrieval=True,
            reason="该问题依赖本地知识库。",
        ),
    )

    responses = [
        # Step 1：
        # 主模型错误地直接回答，没有 Tool Call
        make_response(
            content="这是一个没有证据的直接回答。",
            tool_calls=None,
        ),

        # Step 2：
        # 主模型正确调用 Retrieval
        make_response(
            tool_calls=[
                make_tool_call(
                    "Cluster-State Graph"
                )
            ]
        ),

        # Final Answer
        make_response(
            content="这是基于知识库证据的回答。",
            tool_calls=None,
        ),
    ]

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        lambda **kwargs: responses.pop(0),
    )

    monkeypatch.setattr(
        agent,
        "run_tool",
        lambda tool_name, arguments: ToolResult.success(
            content=(
                "Cluster-State Graph 的知识库证据。"
            )
        ),
    )

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda user_query, evidence: EvidenceDecision(
            sufficient=True,
            reason="证据足够。",
            missing_aspects=[],
        ),
    )

    # Evidence 已经足够，因此不应该 Rewrite
    def should_not_rewrite(*args, **kwargs):
        raise AssertionError(
            "Evidence 已经足够，不应该执行 Rewrite"
        )

    monkeypatch.setattr(
        agent,
        "rewrite_query",
        should_not_rewrite,
    )

    reply = agent.run_agent(
        user_input="Cluster-State Graph 是什么？",
        chat_history=[],
    )

    # 第一次没有证据的回答不能被接受，
    # 最终必须返回 Retrieval 后的答案。
    assert reply == (
        "这是基于知识库证据的回答。"
    )