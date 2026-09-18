import numpy as np
import pytest

from retrieval.document import DocumentChunk
from retrieval.hybrid import HybridRetriever


class FakeEncoder:
    """
    测试专用 Embedding 模型。
    """

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        vectors = []

        for text in texts:
            if "生成器" in text:
                vector = [1.0, 0.0, 0.0]

            elif "数据库" in text or "查询" in text:
                vector = [0.0, 1.0, 0.0]

            elif "Java" in text or "JVM" in text:
                vector = [0.0, 0.0, 1.0]

            else:
                vector = [0.577, 0.577, 0.577]

            vectors.append(vector)

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


class FakeReranker:
    """
    测试专用 Reranker。
    """

    def predict(
        self,
        pairs,
        show_progress_bar=False,
    ):
        scores = []

        for query, text in pairs:
            if (
                "生成器" in text
                and "惰性" in text
            ):
                score = 10.0

            elif (
                "列表" in text
                and "内存" in text
            ):
                score = 5.0

            elif "Java" in text:
                score = 1.0

            else:
                score = 0.0

            scores.append(score)

        return np.asarray(
            scores,
            dtype=np.float32,
        )


def create_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="doc_python",
            content=(
                "Python 列表会一次性将所有元素"
                "保存在内存中。"
            ),
            filename="python.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_id="doc_python",
            content=(
                "Python 生成器采用惰性求值，"
                "可以减少内存占用。"
            ),
            filename="python.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            document_id="doc_java",
            content=(
                "Java 程序运行在 JVM 虚拟机上。"
            ),
            filename="java.txt",
        ),
    ]


def create_retriever():
    return HybridRetriever(
        create_chunks(),
        dense_encoder=FakeEncoder(),
        reranker_model=FakeReranker(),
    )


def test_hybrid_search_returns_relevant_result():
    retriever = create_retriever()

    results = retriever.search(
        "Python 生成器为什么节省内存？",
        top_k=2,
        candidate_k=3,
    )

    assert len(results) == 2

    assert (
        results[0].chunk.chunk_id
        == "chunk_002"
    )

    assert results[0].rank == 1

    assert (
        results[0].score
        >= results[1].score
    )


def test_hybrid_empty_query():
    retriever = create_retriever()

    results = retriever.search(
        "",
        top_k=2,
        candidate_k=3,
    )

    assert results == []


def test_hybrid_empty_corpus():
    retriever = HybridRetriever(
        [],
        dense_encoder=FakeEncoder(),
        reranker_model=FakeReranker(),
    )

    results = retriever.search(
        "Python",
        top_k=1,
        candidate_k=1,
    )

    assert results == []


def test_hybrid_invalid_top_k():
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.search(
            "Python",
            top_k=0,
            candidate_k=3,
        )


def test_hybrid_invalid_candidate_k():
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.search(
            "Python",
            top_k=1,
            candidate_k=0,
        )


def test_hybrid_candidate_k_must_cover_top_k():
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.search(
            "Python",
            top_k=3,
            candidate_k=2,
        )


def test_hybrid_top_k_limit():
    retriever = create_retriever()

    results = retriever.search(
        "Python 生成器",
        top_k=1,
        candidate_k=3,
    )

    assert len(results) == 1
    assert results[0].rank == 1