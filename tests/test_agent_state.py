"""AgentState 的状态与搜索预算测试。"""

from harness.state import AgentState


def test_first_search_is_allowed():
    state = AgentState(
        original_query="什么是 RAG？",
        max_search_calls=3,
    )

    allowed, reason = state.reserve_search(
        "RAG 定义",
        step=1,
    )

    assert allowed is True
    assert reason == "ok"
    assert state.search_count == 1
    assert state.last_query == "RAG 定义"


def test_duplicate_query_is_rejected():
    state = AgentState(
        original_query="什么是 RAG？",
        max_search_calls=3,
    )

    state.reserve_search(
        "RAG 定义",
        step=1,
    )

    allowed, reason = state.reserve_search(
        "  rag   定义  ",
        step=2,
    )

    assert allowed is False
    assert reason == "duplicate_query"
    assert state.search_count == 1


def test_distinct_queries_are_allowed_in_same_step():
    state = AgentState(
        original_query="介绍 Cluster-State Graph",
        max_search_calls=3,
    )

    state.reserve_search(
        "Cluster-State Graph",
        step=1,
    )

    allowed, reason = state.reserve_search(
        "cluster state graph 定义 构造",
        step=1,
    )

    assert allowed is True
    assert reason == "ok"
    assert state.search_count == 2


def test_search_budget_is_enforced():
    state = AgentState(
        original_query="复杂问题",
        max_search_calls=2,
    )

    state.reserve_search(
        "query one",
        step=1,
    )

    state.reserve_search(
        "query two",
        step=2,
    )

    allowed, reason = state.reserve_search(
        "query three",
        step=3,
    )

    assert allowed is False
    assert reason == "budget_exhausted"
    assert state.search_count == 2


def test_search_budget_is_enforced_within_same_step():
    state = AgentState(
        original_query="比较两个概念",
        max_search_calls=2,
    )

    assert state.reserve_search("概念 A", step=1) == (True, "ok")
    assert state.reserve_search("概念 B", step=1) == (True, "ok")
    assert state.reserve_search("概念 C", step=1) == (
        False,
        "budget_exhausted",
    )
    assert state.search_count == 2


def test_evidence_and_trace_are_recorded():
    state = AgentState(
        original_query="什么是 RAG？",
        max_search_calls=3,
    )

    state.reserve_search(
        "RAG",
        step=1,
    )

    state.record_evidence(
        "证据内容"
    )

    assert state.evidence == [
        "证据内容"
    ]

    assert state.trace[0]["event"] == "search_accepted"
    assert state.trace[0]["query"] == "RAG"
