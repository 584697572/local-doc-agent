"""LocalDoc-Agent FastAPI 服务。"""

import logging
import os
from threading import Lock

from fastapi import (
    FastAPI,
    HTTPException,
    Response,
)

from dotenv import (
    load_dotenv,
)

from config import (
    DATA_DIR,
    MAX_HISTORY_PAIRS,
    PROJECT_DIR,
)
from api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IndexStatusResponse,
    ReadyResponse,
    RetrievalResponse,
    UsageResponse,
)


logger = logging.getLogger(
    __name__
)


# Application

app = FastAPI(
    title="LocalDoc-Agent API",

    description=(
        "本地文档 Agentic RAG HTTP API"
    ),

    version="0.1.0",
)


# Agent Execution

# Agent / Retriever 共享进程内状态，串行执行以避免并发初始化模型和索引。
_AGENT_LOCK = Lock()


def _run_agent_request(
    query: str,
    history: list[dict],
):
    """
    延迟加载 Agent，使健康检查不依赖模型初始化。

    测试可替换此适配函数，避免调用真实 LLM。
    """

    from agent import (
        run_agent_with_trace,
    )

    with _AGENT_LOCK:
        return (
            run_agent_with_trace(
                query,
                history,
            )
        )


def _readiness_checks() -> dict[
    str,
    bool,
]:
    """
    执行轻量 Readiness 检查。

    不调用外部 API，
    不加载 Embedding / Reranker。
    """

    # /ready 也应该识别项目 .env，
    # 不能要求先调用 Agent 才加载环境变量。
    load_dotenv(
        PROJECT_DIR
        / ".env"
    )

    return {
        "api_key_configured": bool(
            os.getenv(
                "DEEPSEEK_API_KEY",
                "",
            ).strip()
        ),

        "data_directory_available": (
            DATA_DIR.exists()
            and DATA_DIR.is_dir()
        ),
    }


def _get_index_status() -> dict:
    """
    Lazy Import Retrieval Status。

    保证单纯启动 FastAPI 时，
    /health 不被 Retrieval 模块拖慢。
    """

    from tools.retrieval import (
        get_retrieval_status,
    )

    return (
        get_retrieval_status()
    )

# Health

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    检查 HTTP 服务存活，不加载 LLM、检索模型或索引。
    """

    return HealthResponse(
        status="ok",
        service="local-doc-agent",
        version="0.1.0",
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
)
def ready(
    response: Response,
):
    """
    Readiness Endpoint。

    ready:
        服务具备接受真实 Agent 请求的基本条件。

    not_ready:
        HTTP Server 活着，
        但配置还不完整。
    """

    checks = (
        _readiness_checks()
    )

    is_ready = all(
        checks.values()
    )

    if not is_ready:
        response.status_code = 503

    return ReadyResponse(
        status=(
            "ready"
            if is_ready
            else "not_ready"
        ),

        checks=checks,
    )


@app.get(
    "/v1/index/status",
    response_model=(
        IndexStatusResponse
    ),
)
def index_status():
    """
    返回本地知识库索引状态。

    只读接口。

    不主动：
        - 建索引
        - 重新 Embedding
        - 调用 LLM
    """

    return IndexStatusResponse(
        **_get_index_status()
    )

# Ask

@app.post(
    "/v1/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
):
    """
    执行一次完整 LocalDoc-Agent 请求。

    API 本身不保存 Session。

    聊天历史由调用方通过 history 传入。
    """

    query = (
        request.query.strip()
    )

    # 长度校验不能识别纯空白输入，需要在 strip 后再次检查。
    if not query:
        raise HTTPException(
            status_code=422,
            detail=(
                "query 不能为空。"
            ),
        )

    # 与 CLI 保持同样的上下文预算。
    max_history_messages = (
        MAX_HISTORY_PAIRS
        * 2
    )

    if (
        len(
            request.history
        )
        > max_history_messages
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "history 超过最大长度："
                f"{max_history_messages} 条消息。"
            ),
        )

    # 将请求模型转换为 Agent 使用的消息字典。
    history = [
        message.model_dump()
        for message
        in request.history
    ]

    try:
        run = (
            _run_agent_request(
                query,
                history,
            )
        )

    except Exception:
        # 完整异常仅写入服务端日志，HTTP 响应不暴露密钥、路径或 SDK 细节。
        logger.exception(
            "Agent request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Agent 执行失败。"
            ),
        )

    state = run.state

    trace = None

    if request.include_trace:
        trace = run.trace

    return AskResponse(
        answer=(
            run.answer
        ),

        latency_ms=(
            run.latency_ms
        ),

        usage=UsageResponse(
            llm_calls=(
                run.usage[
                    "llm_calls"
                ]
            ),
        ),

        retrieval=RetrievalResponse(
            search_count=(
                state.search_count
            ),

            max_search_calls=(
                state.max_search_calls
            ),

            evidence_count=(
                len(
                    state.evidence
                )
            ),

            evidence_sufficient=(
                state.evidence_sufficient
            ),

            missing_aspects=(
                list(
                    state.missing_aspects
                )
            ),
        ),

        trace=trace,
    )
