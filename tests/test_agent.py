"""Agent 主控制流测试。"""

from types import SimpleNamespace

import agent
from harness.evidence import EvidenceDecision
from harness.rewrite import RewriteResult
from tools.result import ToolResult


def make_tool_call(
    query: str,
    call_id: str = "call_1",
):
    """
    构造一个假的 search_documents Tool Call。
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
    构造一个假的 LLM Response。
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


def test_agent_rewrite_search_then_answer(
    monkeypatch,
):
    """
    验证：

    Search
    → Evidence 不足
    → Rewrite
    → Search Again
    → Evidence 足够
    → Final Answer
    """

    searched_queries = []

    def fake_run_tool(tool_name, arguments):
        query = arguments["query"]
        searched_queries.append(query)

        if len(searched_queries) == 1:
            return ToolResult.success(
                content="初始证据：只有 Cluster-State Graph 的定义。"
            )

        return ToolResult.success(
            content="补充证据：包含节点、边和构造方法。"
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

    # Evidence 不够时，生成定向 Rewrite Query
    monkeypatch.setattr(
        agent,
        "rewrite_query",
        lambda original_query, missing_aspects, used_queries: (
            RewriteResult(
                query="Cluster-State Graph 节点 边 构造方式",
                reason="补充当前缺失的图结构和构造信息。",
            )
        ),
    )

    # 第一次：主 Agent 要求搜索
    # 第二次：generate_final_answer() 返回最终答案
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
        user_input="Cluster-State Graph 是什么，怎么构造？",
        chat_history=[],
    )

    assert reply == "这是最终回答。"

    # 应该先搜索初始 Query，再搜索 Rewrite Query
    assert searched_queries == [
        "Cluster-State Graph",
        "Cluster-State Graph 节点 边 构造方式",
    ]


def test_agent_answers_without_rewrite_when_evidence_is_sufficient(
    monkeypatch,
):
    """
    验证：

    Search
    → 第一次 Evidence 就足够
    → 不执行 Rewrite
    → 直接 Final Answer
    """

    searched_queries = []

    def fake_run_tool(tool_name, arguments):
        query = arguments["query"]
        searched_queries.append(query)

        return ToolResult.success(
            content="完整证据：包含定义、节点、边和构造方式。"
        )

    monkeypatch.setattr(
        agent,
        "run_tool",
        fake_run_tool,
    )

    # 第一次 Judge 就认为证据足够
    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda user_query, evidence: EvidenceDecision(
            sufficient=True,
            reason="证据已经覆盖用户问题。",
            missing_aspects=[],
        ),
    )

    # 如果真的进入 Rewrite，测试直接失败
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

    assert reply == "这是基于完整证据的最终回答。"

    # 只应该搜索一次
    assert searched_queries == [
        "Cluster-State Graph"
    ]


def test_agent_stops_when_search_budget_is_exhausted(
    monkeypatch,
):
    """
    验证：

    Evidence 一直不足
    → Rewrite + Search
    → 达到搜索预算
    → 停止继续搜索
    → 根据已有证据生成保守回答
    """

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

    # 无论搜索多少次，都认为证据不够
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

    # 每次根据 used_queries 数量生成新的 Query，
    # 保证不会因为重复 Query 被 Harness 拒绝。
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
            content="现有证据不足，以下只回答能够确认的部分。",
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
        "现有证据不足，以下只回答能够确认的部分。"
    )

    # 真正执行的搜索次数不能超过预算
    assert len(searched_queries) == agent.MAX_SEARCH_CALLS

    # 当前 MAX_SEARCH_CALLS = 3
    assert searched_queries == [
        "初始 Query",
        "补充检索 Query 2",
        "补充检索 Query 3",
    ]

    # 每得到一次新 Evidence，都应重新 Judge
    assert judge_call_count == agent.MAX_SEARCH_CALLS