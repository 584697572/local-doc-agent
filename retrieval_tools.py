"""
把 RetrievalEngine 包装成 Agent 可以调用的工具。
"""

from config import DATA_DIR
from retrieval.engine import RetrievalEngine


# 全局只保留一个 Engine，避免每次搜索都重新建立索引
_engine = None
_engine_built = False


def _get_retrieval_engine() -> RetrievalEngine:
    """
    第一次调用时创建并 build，
    后续直接复用已有索引。
    """
    global _engine
    global _engine_built

    if _engine is None:
        _engine = RetrievalEngine(DATA_DIR)

    if not _engine_built:
        _engine.build()
        _engine_built = True

    return _engine


def search_documents(query: str) -> str:
    """
    在本地知识库中搜索相关文档片段，
    并转换成适合直接交给 LLM 阅读的文本。
    """

    query = query.strip()

    if not query:
        return "检索失败：query 不能为空。"

    try:
        engine = _get_retrieval_engine()

        results = engine.search(
            query=query,
            top_k=5,
            candidate_k=20,
        )

    except Exception as e:
        # 第一版先保证检索失败时 Agent 不会直接崩溃
        return f"检索失败：{e}"

    if not results:
        return "知识库中没有找到与该问题相关的证据。"

    output_parts = []

    for result in results:
        chunk = result.chunk

        # 组织来源信息
        source_parts = [chunk.filename]

        if chunk.page is not None:
            source_parts.append(f"第 {chunk.page} 页")

        if chunk.section:
            source_parts.append(f"章节：{chunk.section}")

        source = "，".join(source_parts)

        output_parts.append(
            f"[结果 {result.rank}]\n"
            f"来源：{source}\n"
            f"chunk_id：{chunk.chunk_id}\n"
            f"相关性分数：{result.score:.4f}\n"
            f"内容：\n{chunk.content}"
        )

    # 每条结果之间空一行
    return "\n\n".join(output_parts)