"""Query Rewrite 测试。"""

import harness.rewrite as rewrite_module


def test_rewrite_query_success(monkeypatch):
    # Fake LLM 返回固定 JSON，不调用真实 API
    monkeypatch.setattr(
        rewrite_module,
        "_call_rewriter",
        lambda original_query, missing_aspects, used_queries: (
            '{'
            '"query": "Cluster-State Graph 构造步骤 节点 边",'
            '"reason": "当前缺少图的构造过程。"'
            '}'
        ),
    )

    result = rewrite_module.rewrite_query(
        original_query="Cluster-State Graph 是什么，怎么构造？",
        missing_aspects=[
            "图的构造步骤",
            "节点和边的定义",
        ],
        used_queries={
            "cluster-state graph"
        },
    )

    assert result.query == (
        "Cluster-State Graph 构造步骤 节点 边"
    )

    assert result.reason == (
        "当前缺少图的构造过程。"
    )


def test_no_missing_aspects_returns_empty_query():
    result = rewrite_module.rewrite_query(
        original_query="什么是 RAG？",
        missing_aspects=[],
        used_queries={"rag"},
    )

    assert result.query == ""
    assert "没有明确的证据缺口" in result.reason


def test_invalid_json_returns_empty_query(monkeypatch):
    monkeypatch.setattr(
        rewrite_module,
        "_call_rewriter",
        lambda original_query, missing_aspects, used_queries: (
            "这不是合法 JSON"
        ),
    )

    result = rewrite_module.rewrite_query(
        original_query="什么是 RAG？",
        missing_aspects=[
            "RAG 的实现流程"
        ],
        used_queries={
            "rag 定义"
        },
    )

    assert result.query == ""
    assert "Query Rewrite 失败" in result.reason


def test_invalid_query_type_is_rejected(monkeypatch):
    # JSON 合法，但 query 类型错误
    monkeypatch.setattr(
        rewrite_module,
        "_call_rewriter",
        lambda original_query, missing_aspects, used_queries: (
            '{'
            '"query": ["错误类型"],'
            '"reason": "测试"'
            '}'
        ),
    )

    result = rewrite_module.rewrite_query(
        original_query="什么是 RAG？",
        missing_aspects=[
            "实现流程"
        ],
        used_queries=set(),
    )

    assert result.query == ""