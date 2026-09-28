"""Query Router 测试。"""

import harness.router as router_module


def test_knowledge_query_needs_retrieval(
    monkeypatch,
):
    """技术知识问题应该进入 Retrieval。"""

    monkeypatch.setattr(
        router_module,
        "_call_router",
        lambda user_query: (
            '{'
            '"needs_retrieval": true,'
            '"reason": "该问题涉及本地技术知识。"'
            '}'
        ),
    )

    decision = router_module.route_query(
        "Cluster-State Graph 是什么？"
    )

    assert decision.needs_retrieval is True
    assert "技术知识" in decision.reason


def test_greeting_does_not_need_retrieval(
    monkeypatch,
):
    """简单寒暄不应该浪费一次知识库搜索。"""

    monkeypatch.setattr(
        router_module,
        "_call_router",
        lambda user_query: (
            '{'
            '"needs_retrieval": false,'
            '"reason": "这只是简单寒暄。"'
            '}'
        ),
    )

    decision = router_module.route_query(
        "你好"
    )

    assert decision.needs_retrieval is False


def test_invalid_json_defaults_to_retrieval(
    monkeypatch,
):
    """Router 输出损坏时，默认进入 Retrieval。"""

    monkeypatch.setattr(
        router_module,
        "_call_router",
        lambda user_query: "这不是 JSON",
    )

    decision = router_module.route_query(
        "解释一下 RAG"
    )

    assert decision.needs_retrieval is True
    assert "默认进入 Retrieval" in decision.reason


def test_invalid_boolean_type_defaults_to_retrieval(
    monkeypatch,
):
    """needs_retrieval 类型错误时采用保守策略。"""

    monkeypatch.setattr(
        router_module,
        "_call_router",
        lambda user_query: (
            '{'
            '"needs_retrieval": "maybe",'
            '"reason": "不确定"'
            '}'
        ),
    )

    decision = router_module.route_query(
        "介绍一下 GNN"
    )

    assert decision.needs_retrieval is True


def test_empty_query_does_not_need_retrieval():
    """空输入没有必要搜索。"""

    decision = router_module.route_query(
        "   "
    )

    assert decision.needs_retrieval is False