"""把 RetrievalEngine 包装成 Agent 可调用的检索工具。"""
import json
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    INDEX_DIR,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from retrieval.engine import (
    RetrievalEngine,
)
from tools.result import ToolResult


# ======================================================
# Process-local Engine Cache
# ======================================================

# Persistent Index 解决的是：
#
#     程序关闭以后
#     下一次启动仍然可以复用 Dense Index。
#
# _engine 解决的是：
#
#     同一个 Python 进程中
#     不要每次搜索都重新创建 RetrievalEngine。
_engine = None


def _get_retrieval_engine() -> RetrievalEngine:
    """
    获取 RetrievalEngine。

    第一次调用：

        创建 Engine
            ↓
        build()
            ↓
        Persistent Cache Hit
            或
        全量 Build + 保存 Cache

    同一进程后续调用：

        直接复用 _engine。
    """

    global _engine

    if _engine is None:
        engine = RetrievalEngine(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,

            # 正式启用 Persistent Index。
            cache_dir=INDEX_DIR,
        )

        engine.build()

        # build 成功以后再赋值。
        #
        # 如果初始化期间抛异常，
        # 不会把一个半初始化 Engine
        # 留在全局缓存中。
        _engine = engine

    return _engine

# 文件：tools/retrieval.py
# 位置：_get_retrieval_engine() 后
# 操作：新增


def get_retrieval_status() -> dict:
    """
    获取 Retrieval / Persistent Index 状态。

    重要：

    这个函数只是查看状态，
    不会主动调用 _get_retrieval_engine()。

    因此：
        查看 /v1/index/status
    不会导致模型加载或索引重建。
    """

    # ==================================================
    # Runtime Engine 已经初始化
    # ==================================================

    if _engine is not None:
        chunks = getattr(
            _engine,
            "chunks",
            [],
        )

        source_files = getattr(
            _engine,
            "source_files",
            None,
        )

        if source_files is not None:
            document_count = len(
                source_files
            )

        else:
            # 兼容旧 Engine / 测试 Double。
            document_count = len(
                {
                    chunk.filename
                    for chunk
                    in chunks
                }
            )

        embedding_dimension = None

        retriever = getattr(
            _engine,
            "retriever",
            None,
        )

        dense = getattr(
            retriever,
            "dense_retriever",
            None,
        )

        embeddings = getattr(
            dense,
            "embeddings",
            None,
        )

        if (
            embeddings is not None
            and getattr(
                embeddings,
                "ndim",
                0,
            )
            == 2
        ):
            embedding_dimension = int(
                embeddings.shape[1]
            )

        elif (
            dense is not None
            and getattr(
                dense,
                "index",
                None,
            )
            is not None
        ):
            embedding_dimension = int(
                dense.index.d
            )

        return {
            "runtime_initialized": True,

            "cache_status": getattr(
                _engine,
                "cache_status",
                "unknown",
            ),

            "cache_reason": getattr(
                _engine,
                "cache_reason",
                None,
            ),

            "update_mode": getattr(
                _engine,
                "index_update_mode",
                "unknown",
            ),

            "documents": (
                document_count
            ),

            "chunks": len(
                chunks
            ),

            "embedding_dimension": (
                embedding_dimension
            ),

            "incremental_stats": dict(
                getattr(
                    _engine,
                    "incremental_stats",
                    {},
                )
            ),
        }

    # ==================================================
    # Runtime 尚未初始化
    #
    # 只查看磁盘 Manifest。
    # 注意：
    # 这只是上一次缓存快照，
    # 并不代表它已经与当前 data/ 校验过。
    # ==================================================

    manifest_path = (
        INDEX_DIR
        / "manifest.json"
    )

    if not manifest_path.exists():
        return {
            "runtime_initialized": False,
            "cache_status": "missing",
            "cache_reason": None,
            "update_mode": "not_loaded",
            "documents": 0,
            "chunks": 0,
            "embedding_dimension": None,
            "incremental_stats": {},
        }

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        source_files = manifest.get(
            "source_files",
            [],
        )

        return {
            "runtime_initialized": False,

            # 这里只说明磁盘上存在快照，
            # 还没有与当前 data/ 做完整校验。
            "cache_status": (
                "snapshot_available"
            ),

            "cache_reason": None,

            "update_mode": (
                "not_loaded"
            ),

            "documents": (
                len(source_files)
                if isinstance(
                    source_files,
                    list,
                )
                else 0
            ),

            "chunks": int(
                manifest.get(
                    "chunk_count",
                    0,
                )
                or 0
            ),

            "embedding_dimension": (
                manifest.get(
                    "embedding_dimension"
                )
            ),

            "incremental_stats": {},
        }

    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        # Status Endpoint 不暴露内部文件路径或异常。
        return {
            "runtime_initialized": False,
            "cache_status": (
                "snapshot_unreadable"
            ),
            "cache_reason": (
                "manifest_unreadable"
            ),
            "update_mode": (
                "not_loaded"
            ),
            "documents": 0,
            "chunks": 0,
            "embedding_dimension": None,
            "incremental_stats": {},
        }

def _get_index_cache_status(
    engine,
) -> str:
    """
    安全读取 Persistent Index 状态。

    真实 RetrievalEngine 会提供：

        engine.cache_status

    可能值例如：

        hit
        rebuilt
        disabled
        save_failed

    但是：

        - 单元测试 FakeEngine
        - 第三方测试 Double
        - 以后可能替换的 Retriever Backend

    不一定实现这个调试字段。

    Cache Status 只是 Observability Metadata，
    绝不能因为缺少它导致正常检索失败。
    """

    return getattr(
        engine,
        "cache_status",
        "unknown",
    )


def search_documents(
    query: str,
) -> ToolResult:
    """
    检索本地知识库，
    返回 LLM 可以直接阅读的 Evidence。
    """

    query = query.strip()

    if not query:
        return ToolResult.failure(
            error="query 不能为空。",
        )

    # ==================================================
    # Retrieval
    # ==================================================

    try:
        engine = (
            _get_retrieval_engine()
        )

        results = engine.search(
            query=query,
            top_k=(
                RETRIEVAL_TOP_K
            ),
            candidate_k=(
                RETRIEVAL_CANDIDATE_K
            ),
        )

    except Exception as exc:
        return ToolResult.failure(
            error=str(
                exc
            ),
            metadata={
                "query": query,
            },
        )

    # ==================================================
    # Empty Result
    # ==================================================

    if not results:
        return ToolResult.empty(
            content=(
                "知识库中没有找到"
                "与该问题相关的证据。"
            ),
            metadata={
                "query": query,
                "result_count": 0,

                # 调试字段不能影响核心功能。
                "index_cache_status": (
                    _get_index_cache_status(
                        engine
                    )
                ),
            },
        )

    # ==================================================
    # Format Evidence
    # ==================================================

    output_parts = []

    for result in results:
        chunk = result.chunk

        source_parts = [
            chunk.filename
        ]

        if chunk.page is not None:
            source_parts.append(
                f"第 {chunk.page} 页"
            )

        if chunk.section:
            source_parts.append(
                f"章节：{chunk.section}"
            )

        source = "，".join(
            source_parts
        )

        output_parts.append(
            f"[证据 {result.rank}]\n"
            f"来源：{source}\n"
            f"chunk_id：{chunk.chunk_id}\n"
            f"内容：\n{chunk.content}"
        )

    content = "\n\n".join(
        output_parts
    )

    # ==================================================
    # Tool Result
    # ==================================================

    return ToolResult.success(
        content=content,
        metadata={
            "query": query,

            "result_count": (
                len(
                    results
                )
            ),

            # Persistent Index 状态属于
            # Observability Metadata。
            #
            # FakeEngine 没有该属性时：
            #
            #     unknown
            #
            # 而不是让整个 Retrieval Tool 崩掉。
            "index_cache_status": (
                _get_index_cache_status(
                    engine
                )
            ),

            # 必须记录本次真实检索命中的来源。
            #
            # Citation / Evaluation 使用的是
            # 真正返回的 Evidence，
            # 不能从 Gold Answer 反推来源。
            "sources": [
                {
                    "filename": (
                        result.chunk.filename
                    ),

                    "page": (
                        result.chunk.page
                    ),

                    "section": (
                        result.chunk.section
                    ),

                    "chunk_id": (
                        result.chunk.chunk_id
                    ),
                }

                for result
                in results
            ],
        },
    )