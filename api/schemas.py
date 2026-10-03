"""FastAPI 请求 / 响应数据结构。"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    聊天历史仅接受 user / assistant，禁止客户端伪造 system / tool 消息。
    """

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
        max_length=12000,
    )


class AskRequest(BaseModel):
    """POST /v1/ask 请求。"""

    query: str = Field(
        min_length=1,
        max_length=4000,
    )

    history: list[
        ChatMessage
    ] = Field(
        default_factory=list
    )

    # Trace 可能包含检索证据，默认关闭，仅在调用方明确请求时返回。
    include_trace: bool = False


class UsageResponse(BaseModel):
    """Agent 调用统计。"""

    llm_calls: int


class RetrievalResponse(BaseModel):
    """一次 Agent Run 的 Retrieval 状态。"""

    search_count: int

    max_search_calls: int

    evidence_count: int

    evidence_sufficient: (
        bool
        | None
    )

    missing_aspects: list[str]


class AskResponse(BaseModel):
    """POST /v1/ask 响应。"""

    answer: str

    latency_ms: float

    usage: UsageResponse

    retrieval: RetrievalResponse

    # 默认不返回内部 Trace。
    trace: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None


class HealthResponse(BaseModel):
    """GET /health 响应。"""

    status: str

    service: str

    version: str


class ReadyResponse(BaseModel):
    """
    服务是否具备处理真实 Agent 请求的基本条件。
    """

    status: Literal[
        "ready",
        "not_ready",
    ]

    checks: dict[
        str,
        bool,
    ]


class IndexStatusResponse(BaseModel):
    """
    Persistent / Incremental Index 状态。
    """

    runtime_initialized: bool

    cache_status: str

    cache_reason: (
        str
        | None
    )

    update_mode: str

    documents: int

    chunks: int

    embedding_dimension: (
        int
        | None
    )

    incremental_stats: dict[
        str,
        Any,
    ]
