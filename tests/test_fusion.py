import pytest

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult
from retrieval.fusion import (
    rank_preserving_rerank_fusion,
    reciprocal_rank_fusion,
)


def create_chunks():
    return {
        "A": DocumentChunk(
            chunk_id="A",
            document_id="doc",
            content="Chunk A",
            filename="demo.txt",
        ),
        "B": DocumentChunk(
            chunk_id="B",
            document_id="doc",
            content="Chunk B",
            filename="demo.txt",
        ),
        "C": DocumentChunk(
            chunk_id="C",
            document_id="doc",
            content="Chunk C",
            filename="demo.txt",
        ),
        "D": DocumentChunk(
            chunk_id="D",
            document_id="doc",
            content="Chunk D",
            filename="demo.txt",
        ),
        "E": DocumentChunk(
            chunk_id="E",
            document_id="doc",
            content="Chunk E",
            filename="demo.txt",
        ),
        "F": DocumentChunk(
            chunk_id="F",
            document_id="doc",
            content="Chunk F",
            filename="demo.txt",
        ),
    }


def create_ranked_results():
    chunks = create_chunks()

    bm25_results = [
        RetrievalResult(
            chunk=chunks["A"],
            score=8.5,
            rank=1,
        ),
        RetrievalResult(
            chunk=chunks["B"],
            score=6.1,
            rank=2,
        ),
        RetrievalResult(
            chunk=chunks["C"],
            score=4.0,
            rank=3,
        ),
    ]

    dense_results = [
        RetrievalResult(
            chunk=chunks["C"],
            score=0.95,
            rank=1,
        ),
        RetrievalResult(
            chunk=chunks["A"],
            score=0.90,
            rank=2,
        ),
        RetrievalResult(
            chunk=chunks["D"],
            score=0.80,
            rank=3,
        ),
    ]

    return (
        bm25_results,
        dense_results,
    )


def test_rrf_fuses_rankings():
    (
        bm25_results,
        dense_results,
    ) = create_ranked_results()

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=4,
    )

    assert len(results) == 4

    # A 在 BM25 第1、Dense 第2，
    # 因此综合排名应该最高。
    assert (
        results[0].chunk.chunk_id
        == "A"
    )

    # C 在 BM25 第3、Dense 第1，
    # 综合排名应该紧随 A。
    assert (
        results[1].chunk.chunk_id
        == "C"
    )

    assert results[0].rank == 1
    assert results[1].rank == 2

    assert (
        results[0].score
        >= results[1].score
    )


def test_rrf_merges_same_chunk_once():
    (
        bm25_results,
        dense_results,
    ) = create_ranked_results()

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=10,
    )

    chunk_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    assert (
        chunk_ids.count("A")
        == 1
    )

    assert (
        chunk_ids.count("C")
        == 1
    )


def test_rrf_empty_results():
    results = (
        reciprocal_rank_fusion(
            [],
            [],
        )
    )

    assert results == []


def test_rrf_invalid_top_k():
    (
        bm25_results,
        dense_results,
    ) = create_ranked_results()

    with pytest.raises(
        ValueError
    ):
        reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k=0,
        )


def test_rrf_invalid_rrf_k():
    (
        bm25_results,
        dense_results,
    ) = create_ranked_results()

    with pytest.raises(
        ValueError
    ):
        reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            rrf_k=-1,
        )


def test_rrf_top_k_limit():
    (
        bm25_results,
        dense_results,
    ) = create_ranked_results()

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_rrf_ignores_original_score_scale():
    chunks = create_chunks()

    bm25_results = [
        RetrievalResult(
            chunk=chunks["A"],
            score=1000.0,
            rank=2,
        ),
    ]

    dense_results = [
        RetrievalResult(
            chunk=chunks["B"],
            score=0.001,
            rank=1,
        ),
    ]

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=2,
    )

    # RRF 只看 rank，
    # 不比较不同 Retriever 的原始 score。
    assert (
        results[0].chunk.chunk_id
        == "B"
    )

    assert (
        results[1].chunk.chunk_id
        == "A"
    )


# ======================================================
# Rank-preserving Rerank Fusion Tests
# ======================================================


def create_safe_rerank_results():
    chunks = create_chunks()

    # 原始 RRF 排名。
    rrf_results = [
        RetrievalResult(
            chunk=chunks["A"],
            score=0.032,
            rank=1,
        ),
        RetrievalResult(
            chunk=chunks["B"],
            score=0.031,
            rank=2,
        ),
        RetrievalResult(
            chunk=chunks["C"],
            score=0.030,
            rank=3,
        ),
    ]

    # Reranker 把 B 提到了第一。
    reranked_results = [
        RetrievalResult(
            chunk=chunks["B"],
            score=10.0,
            rank=1,
        ),
        RetrievalResult(
            chunk=chunks["A"],
            score=9.0,
            rank=2,
        ),
        RetrievalResult(
            chunk=chunks["C"],
            score=1.0,
            rank=3,
        ),
    ]

    return (
        rrf_results,
        reranked_results,
    )


def test_rank_preserving_fusion_merges_same_candidates():
    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    results = (
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            top_k=3,
        )
    )

    assert len(results) == 3

    chunk_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    assert set(chunk_ids) == {
        "A",
        "B",
        "C",
    }

    assert len(
        chunk_ids
    ) == len(
        set(chunk_ids)
    )


def test_rank_preserving_fusion_prefers_rrf_on_exact_tie():
    """
    A:
        RRF rank 1
        Reranker rank 2

    B:
        RRF rank 2
        Reranker rank 1

    等权融合时二者分数完全相同。

    Safe Rerank 的 tie-break 策略是：
        优先保留原始 RRF 排名。

    所以 A 应该排在 B 前面。
    """

    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    results = (
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            top_k=3,
        )
    )

    assert (
        results[0].chunk.chunk_id
        == "A"
    )

    assert (
        results[1].chunk.chunk_id
        == "B"
    )


def test_rank_preserving_fusion_can_weight_reranker_more():
    """
    当 Reranker 权重提高时，
    B 应该可以超过原始 RRF 第一名 A。

    这里只验证函数支持权重，
    暂时不代表项目最终会使用 2.0。
    """

    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    results = (
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            top_k=3,
            rrf_weight=1.0,
            reranker_weight=2.0,
        )
    )

    assert (
        results[0].chunk.chunk_id
        == "B"
    )


def test_rank_preserving_fusion_empty_results():
    results = (
        rank_preserving_rerank_fusion(
            rrf_results=[],
            reranked_results=[],
        )
    )

    assert results == []


def test_rank_preserving_fusion_top_k_limit():
    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    results = (
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            top_k=2,
        )
    )

    assert len(results) == 2

    assert results[0].rank == 1
    assert results[1].rank == 2


def test_rank_preserving_fusion_invalid_top_k():
    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    with pytest.raises(
        ValueError
    ):
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            top_k=0,
        )


def test_rank_preserving_fusion_invalid_fusion_k():
    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    with pytest.raises(
        ValueError
    ):
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            fusion_k=-1,
        )


def test_rank_preserving_fusion_invalid_negative_weight():
    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    with pytest.raises(
        ValueError
    ):
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            rrf_weight=-1.0,
        )


def test_rank_preserving_fusion_rejects_zero_weights():
    (
        rrf_results,
        reranked_results,
    ) = create_safe_rerank_results()

    with pytest.raises(
        ValueError
    ):
        rank_preserving_rerank_fusion(
            rrf_results=rrf_results,
            reranked_results=(
                reranked_results
            ),
            rrf_weight=0.0,
            reranker_weight=0.0,
        )