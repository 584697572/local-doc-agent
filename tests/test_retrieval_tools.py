"""
测试 search_documents 工具的输出格式和异常情况。
"""

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult

import retrieval_tools


class FakeEngine:
    """
    测试用假 Engine，不加载真实模型。
    """

    def search(
        self,
        query,
        top_k=5,
        candidate_k=20,
    ):
        chunk = DocumentChunk(
            chunk_id="doc_001_chunk_0001",
            document_id="doc_001",
            content="Python 生成器采用惰性求值，可以减少内存占用。",
            filename="python.pdf",
            page=12,
        )

        return [
            RetrievalResult(
                chunk=chunk,
                score=0.95,
                rank=1,
            )
        ]


class EmptyFakeEngine:
    """
    模拟没有检索结果。
    """

    def search(
        self,
        query,
        top_k=5,
        candidate_k=20,
    ):
        return []


def test_search_documents_formats_evidence(monkeypatch):
    # 用 FakeEngine 替换真实 Engine
    monkeypatch.setattr(
        retrieval_tools,
        "_engine",
        FakeEngine(),
    )

    monkeypatch.setattr(
        retrieval_tools,
        "_engine_built",
        True,
    )

    result = retrieval_tools.search_documents(
        "生成器为什么省内存？"
    )

    assert "python.pdf" in result
    assert "第 12 页" in result
    assert "doc_001_chunk_0001" in result
    assert "生成器采用惰性求值" in result


def test_search_documents_empty_query():
    result = retrieval_tools.search_documents("   ")

    assert "query 不能为空" in result


def test_search_documents_no_results(monkeypatch):
    monkeypatch.setattr(
        retrieval_tools,
        "_engine",
        EmptyFakeEngine(),
    )

    monkeypatch.setattr(
        retrieval_tools,
        "_engine_built",
        True,
    )

    result = retrieval_tools.search_documents(
        "不存在的问题"
    )

    assert "没有找到" in result