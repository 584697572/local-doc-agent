"""把 RetrievalEngine 包装成 Agent 可调用的检索工具。"""

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from retrieval.engine import RetrievalEngine


# 同一进程内复用一个已构建的 Engine。
_engine = None


def _get_retrieval_engine() -> RetrievalEngine:
    """第一次调用时 build，后续直接复用。"""
    global _engine

    if _engine is None:
        engine = RetrievalEngine(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        engine.build()

        # build 成功后再赋值，避免缓存半构建状态。
        _engine = engine

    return _engine


def search_documents(query: str) -> str:
    """检索本地知识库，并返回 LLM 可直接阅读的证据文本。"""
    query = query.strip()
    if not query:
        return "检索失败：query 不能为空。"

    try:
        results = _get_retrieval_engine().search(
            query=query,
            top_k=RETRIEVAL_TOP_K,
            candidate_k=RETRIEVAL_CANDIDATE_K,
        )
    except Exception as exc:
        return f"检索失败：{exc}"

    if not results:
        return "知识库中没有找到与该问题相关的证据。"

    output_parts = []

    for result in results:
        chunk = result.chunk
        source_parts = [chunk.filename]

        if chunk.page is not None:
            source_parts.append(f"第 {chunk.page} 页")
        if chunk.section:
            source_parts.append(f"章节：{chunk.section}")

        source = "，".join(source_parts)

        output_parts.append(
            f"[证据 {result.rank}]\n"
            f"来源：{source}\n"
            f"chunk_id：{chunk.chunk_id}\n"
            f"内容：\n{chunk.content}"
        )

    return "\n\n".join(output_parts)
