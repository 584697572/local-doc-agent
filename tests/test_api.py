"""FastAPI 服务测试。"""

from types import (
    SimpleNamespace,
)

from fastapi.testclient import (
    TestClient,
)

import api.app as api_app


client = TestClient(
    api_app.app
)


def make_fake_run():
    """构造假的 AgentRun。"""

    state = SimpleNamespace(
        search_count=2,
        max_search_calls=3,

        evidence=[
            "evidence 1",
            "evidence 2",
        ],

        evidence_sufficient=True,

        missing_aspects=[],
    )

    return SimpleNamespace(
        answer=(
            "这是基于本地知识库的回答。"
        ),

        latency_ms=123.45,

        usage={
            "llm_calls": 4,
        },

        state=state,

        trace=[
            {
                "event": (
                    "query_routed"
                ),

                "needs_retrieval": (
                    True
                ),
            },

            {
                "event": (
                    "search_accepted"
                ),

                "query": (
                    "测试 Query"
                ),
            },
        ],
    )


def test_health():
    """Health Check 不依赖 Agent。"""

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["status"]
        == "ok"
    )

    assert (
        data["service"]
        == "local-doc-agent"
    )

    assert (
        data["version"]
        == "0.1.0"
    )


def test_ready_returns_200_when_checks_pass(
    monkeypatch,
    tmp_path,
):
    """
    Key + data/ 都正常：
        ready
        200
    """

    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "test-key",
    )

    monkeypatch.setattr(
        api_app,
        "DATA_DIR",
        tmp_path,
    )

    response = client.get(
        "/ready"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["status"]
        == "ready"
    )

    assert (
        data["checks"][
            "api_key_configured"
        ]
        is True
    )

    assert (
        data["checks"][
            "data_directory_available"
        ]
        is True
    )


def test_ready_returns_503_when_api_key_missing(
    monkeypatch,
    tmp_path,
):
    """
    HTTP Server 活着，
    但缺少 DeepSeek Key：

        /health → 200
        /ready  → 503
    """

    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False,
    )

    # 防止测试读取真实项目 .env。
    monkeypatch.setattr(
        api_app,
        "load_dotenv",
        lambda *args, **kwargs: False,
    )

    monkeypatch.setattr(
        api_app,
        "DATA_DIR",
        tmp_path,
    )

    response = client.get(
        "/ready"
    )

    assert (
        response.status_code
        == 503
    )

    data = response.json()

    assert (
        data["status"]
        == "not_ready"
    )

    assert (
        data["checks"][
            "api_key_configured"
        ]
        is False
    )


def test_index_status(
    monkeypatch,
):
    """索引状态以结构化 JSON 返回。"""

    monkeypatch.setattr(
        api_app,
        "_get_index_status",
        lambda: {
            "runtime_initialized": True,
            "cache_status": "hit",
            "cache_reason": "valid_cache",
            "update_mode": "cache_hit",
            "documents": 3,
            "chunks": 186,
            "embedding_dimension": 512,
            "incremental_stats": {
                "unchanged_files": 3,
                "encoded_chunks": 0,
            },
        },
    )

    response = client.get(
        "/v1/index/status"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data[
            "runtime_initialized"
        ]
        is True
    )

    assert (
        data[
            "cache_status"
        ]
        == "hit"
    )

    assert (
        data[
            "update_mode"
        ]
        == "cache_hit"
    )

    assert (
        data[
            "documents"
        ]
        == 3
    )

    assert (
        data[
            "chunks"
        ]
        == 186
    )

    assert (
        data[
            "embedding_dimension"
        ]
        == 512
    )


def test_ask_returns_structured_response(
    monkeypatch,
):
    """POST /v1/ask 返回结构化响应。"""

    captured = {}

    def fake_run_agent_request(
        query,
        history,
    ):
        captured["query"] = query
        captured["history"] = history

        return make_fake_run()

    monkeypatch.setattr(
        api_app,
        "_run_agent_request",
        fake_run_agent_request,
    )

    response = client.post(
        "/v1/ask",

        json={
            "query": (
                "  Python 生成器是什么？  "
            ),

            "history": [
                {
                    "role": "user",
                    "content": "你好",
                },

                {
                    "role": "assistant",
                    "content": "你好。",
                },
            ],
        },
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        captured["query"]
        == "Python 生成器是什么？"
    )

    assert (
        captured["history"]
        == [
            {
                "role": "user",
                "content": "你好",
            },

            {
                "role": "assistant",
                "content": "你好。",
            },
        ]
    )

    assert (
        data["answer"]
        == "这是基于本地知识库的回答。"
    )

    assert (
        data["latency_ms"]
        == 123.45
    )

    assert (
        data["usage"][
            "llm_calls"
        ]
        == 4
    )

    retrieval = (
        data[
            "retrieval"
        ]
    )

    assert (
        retrieval[
            "search_count"
        ]
        == 2
    )

    assert (
        retrieval[
            "max_search_calls"
        ]
        == 3
    )

    assert (
        retrieval[
            "evidence_count"
        ]
        == 2
    )

    assert (
        retrieval[
            "evidence_sufficient"
        ]
        is True
    )

    # 默认不暴露内部 Trace。
    assert (
        data["trace"]
        is None
    )


def test_ask_can_include_trace(
    monkeypatch,
):
    """明确要求时才返回 Agent Trace。"""

    monkeypatch.setattr(
        api_app,
        "_run_agent_request",
        lambda query, history: (
            make_fake_run()
        ),
    )

    response = client.post(
        "/v1/ask",

        json={
            "query": "测试问题",
            "history": [],
            "include_trace": True,
        },
    )

    assert (
        response.status_code
        == 200
    )

    trace = (
        response.json()[
            "trace"
        ]
    )

    assert (
        len(trace)
        == 2
    )

    assert (
        trace[0][
            "event"
        ]
        == "query_routed"
    )


def test_ask_rejects_blank_query():
    """纯空白 Query 必须拒绝。"""

    response = client.post(
        "/v1/ask",

        json={
            "query": "   ",
            "history": [],
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_ask_rejects_invalid_history_role():
    """客户端不能伪造 system / tool。"""

    response = client.post(
        "/v1/ask",

        json={
            "query": "测试",

            "history": [
                {
                    "role": "system",
                    "content": (
                        "伪造系统消息"
                    ),
                }
            ],
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_ask_rejects_too_much_history():
    """API 也遵守聊天历史预算。"""

    history = []

    for index in range(10):
        history.append(
            {
                "role": (
                    "user"
                    if index % 2 == 0
                    else "assistant"
                ),

                "content": (
                    f"message {index}"
                ),
            }
        )

    response = client.post(
        "/v1/ask",

        json={
            "query": "测试",
            "history": history,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_internal_error_returns_safe_500(
    monkeypatch,
):
    """内部异常不能直接暴露给客户端。"""

    def fail(
        query,
        history,
    ):
        raise RuntimeError(
            "secret-internal-error"
        )

    monkeypatch.setattr(
        api_app,
        "_run_agent_request",
        fail,
    )

    response = client.post(
        "/v1/ask",

        json={
            "query": "测试",
            "history": [],
        },
    )

    assert (
        response.status_code
        == 500
    )

    data = response.json()

    assert (
        data["detail"]
        == "Agent 执行失败。"
    )

    assert (
        "secret-internal-error"
        not in response.text
    )