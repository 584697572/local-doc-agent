"""tools.retrieval 的单元测试。"""

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult
import tools.retrieval as retrieval_tool


class FakeEngine:
    def search(self, query, top_k=5, candidate_k=20):
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
    def search(self, query, top_k=5, candidate_k=20):
        return []


def test_search_documents_formats_evidence(monkeypatch):
    monkeypatch.setattr(retrieval_tool, "_engine", FakeEngine())
    monkeypatch.setattr(retrieval_tool, "_engine_built", True)

    result = retrieval_tool.search_documents("生成器为什么省内存？")

    assert "python.pdf" in result
    assert "第 12 页" in result
    assert "doc_001_chunk_0001" in result
    assert "生成器采用惰性求值" in result


def test_search_documents_empty_query():
    result = retrieval_tool.search_documents("   ")
    assert "query 不能为空" in result


def test_search_documents_no_results(monkeypatch):
    monkeypatch.setattr(retrieval_tool, "_engine", EmptyFakeEngine())
    monkeypatch.setattr(retrieval_tool, "_engine_built", True)

    result = retrieval_tool.search_documents("不存在的问题")
    assert "没有找到" in result
