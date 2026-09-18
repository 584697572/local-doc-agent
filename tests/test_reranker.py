import numpy as np
import pytest

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult
from retrieval.reranker import Reranker


class FakeReranker:
    """
    测试专用假 Reranker。

    根据 Chunk 内容返回固定分数，
    避免 pytest 加载真实模型。
    """

    def predict(
        self,
        pairs,
        show_progress_bar=False,
    ):
        scores = []

        for query, text in pairs:
            if "生成器" in text and "惰性" in text:
                score = 10.0

            elif "列表" in text and "内存" in text:
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


def create_candidates():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="doc_python",
            content="Python 列表会一次性将所有元素保存在内存中。",
            filename="python.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_id="doc_python",
            content="Python 生成器采用惰性求值，可以减少内存占用。",
            filename="python.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            document_id="doc_java",
            content="Java 程序运行在 JVM 虚拟机上。",
            filename="java.txt",
        ),
    ]

    # 模拟 RRF 阶段输出的候选。
    # 注意：这里故意把“列表”放在第一，
    # 测试 Reranker 是否能重新把生成器排到第一。
    return [
        RetrievalResult(
            chunk=chunks[0],
            score=0.032,
            rank=1,
        ),
        RetrievalResult(
            chunk=chunks[1],
            score=0.030,
            rank=2,
        ),
        RetrievalResult(
            chunk=chunks[2],
            score=0.015,
            rank=3,
        ),
    ]


def test_reranker_reorders_candidates():
    candidates = create_candidates()

    reranker = Reranker(
        model=FakeReranker(),
    )

    results = reranker.rerank(
        "Python 生成器为什么节省内存？",
        candidates,
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].chunk.chunk_id == "chunk_002"
    assert results[1].chunk.chunk_id == "chunk_001"
    assert results[2].chunk.chunk_id == "chunk_003"

    assert results[0].rank == 1

    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score


def test_reranker_empty_query():
    candidates = create_candidates()

    reranker = Reranker(
        model=FakeReranker(),
    )

    results = reranker.rerank(
        "",
        candidates,
    )

    assert results == []


def test_reranker_empty_candidates():
    reranker = Reranker(
        model=FakeReranker(),
    )

    results = reranker.rerank(
        "生成器为什么节省内存？",
        [],
    )

    assert results == []


def test_reranker_invalid_top_k():
    candidates = create_candidates()

    reranker = Reranker(
        model=FakeReranker(),
    )

    with pytest.raises(ValueError):
        reranker.rerank(
            "生成器为什么节省内存？",
            candidates,
            top_k=0,
        )


def test_reranker_top_k_limit():
    candidates = create_candidates()

    reranker = Reranker(
        model=FakeReranker(),
    )

    results = reranker.rerank(
        "生成器为什么节省内存？",
        candidates,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk_002"
    assert results[0].rank == 1


def test_reranker_top_k_larger_than_candidates():
    candidates = create_candidates()

    reranker = Reranker(
        model=FakeReranker(),
    )

    results = reranker.rerank(
        "生成器为什么节省内存？",
        candidates,
        top_k=100,
    )

    assert len(results) == len(candidates)


class BrokenFakeReranker:
    """
    故意返回错误数量的 score，
    用来验证异常保护。
    """

    def predict(
        self,
        pairs,
        show_progress_bar=False,
    ):
        return np.asarray(
            [1.0],
            dtype=np.float32,
        )


def test_reranker_rejects_wrong_score_count():
    candidates = create_candidates()

    reranker = Reranker(
        model=BrokenFakeReranker(),
    )

    with pytest.raises(ValueError):
        reranker.rerank(
            "生成器为什么节省内存？",
            candidates,
        )