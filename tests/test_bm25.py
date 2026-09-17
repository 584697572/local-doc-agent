import pytest

from retrieval.document import DocumentChunk
from retrieval.bm25 import BM25Retriever, tokenize


def create_test_chunks():
    """
    创建 BM25 测试用的固定 DocumentChunk 数据。

    这三个 chunk 分别讨论：
    1. Python 生成器与内存
    2. Java 与 JVM
    3. Python 列表与内存

    这样可以测试 BM25 是否真的能把和 Query
    更相关的 chunk 排到前面。
    """
    return [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="doc_python",
            content="Python 生成器采用惰性求值，可以减少内存占用。",
            filename="python.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_id="doc_java",
            content="Java 程序运行在 JVM 虚拟机上。",
            filename="java.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            document_id="doc_python",
            content="Python 列表会一次性将所有元素保存在内存中。",
            filename="python.txt",
        ),
    ]


def test_tokenize_chinese_and_english():
    """
    测试中英文混合文本是否能够正常分词。
    """
    tokens = tokenize("Python生成器可以节省内存")

    assert "python" in tokens
    assert "生成器" in tokens
    assert "内存" in tokens


def test_bm25_search_returns_relevant_chunk():
    """
    测试 BM25 是否能把最相关的 chunk 排到第一名。
    """
    chunks = create_test_chunks()

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "Python 生成器 内存",
        top_k=2,
    )

    assert len(results) == 2

    # 第一个结果应该是专门讨论 Python 生成器的 chunk
    assert results[0].chunk.chunk_id == "chunk_001"

    # 第一名 rank 应该是 1
    assert results[0].rank == 1

    # 分数应该按照从高到低排列
    assert results[0].score >= results[1].score


def test_bm25_empty_query():
    """
    空 Query 不应该进行检索。
    """
    chunks = create_test_chunks()

    retriever = BM25Retriever(chunks)

    results = retriever.search("")

    assert results == []


def test_bm25_empty_corpus():
    """
    没有任何 DocumentChunk 时，搜索应该返回空列表，
    而不是报错。
    """
    retriever = BM25Retriever([])

    results = retriever.search("Python")

    assert results == []


def test_bm25_invalid_top_k():
    """
    top_k 必须大于 0。
    """
    chunks = create_test_chunks()

    retriever = BM25Retriever(chunks)

    with pytest.raises(ValueError):
        retriever.search(
            "Python",
            top_k=0,
        )


def test_bm25_no_match():
    """
    Query 与所有文档完全无关时，不应该返回无意义结果。
    """
    chunks = create_test_chunks()

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "量子纠缠超导芯片",
        top_k=3,
    )

    assert results == []


def test_bm25_top_k_limit():
    """
    返回结果数量不能超过 top_k。
    """
    chunks = create_test_chunks()

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "Python 内存",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1