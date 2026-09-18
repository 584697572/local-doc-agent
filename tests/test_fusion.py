import pytest

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult
from retrieval.fusion import reciprocal_rank_fusion


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

    return bm25_results, dense_results


def test_rrf_fuses_rankings():
    bm25_results, dense_results = (
        create_ranked_results()
    )

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=4,
    )

    assert len(results) == 4

    # A 在 BM25 第1、Dense 第2，
    # 因此综合排名应该最高。
    assert results[0].chunk.chunk_id == "A"

    # C 在 BM25 第3、Dense 第1，
    # 综合排名应该紧随 A。
    assert results[1].chunk.chunk_id == "C"

    assert results[0].rank == 1
    assert results[1].rank == 2

    assert results[0].score >= results[1].score


def test_rrf_merges_same_chunk_once():
    bm25_results, dense_results = (
        create_ranked_results()
    )

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=10,
    )

    chunk_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    # A 和 C 同时出现在两套 Retriever 中，
    # 但最终只能各出现一次。
    assert chunk_ids.count("A") == 1
    assert chunk_ids.count("C") == 1


def test_rrf_empty_results():
    results = reciprocal_rank_fusion(
        [],
        [],
    )

    assert results == []


def test_rrf_invalid_top_k():
    bm25_results, dense_results = (
        create_ranked_results()
    )

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k=0,
        )


def test_rrf_invalid_rrf_k():
    bm25_results, dense_results = (
        create_ranked_results()
    )

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            rrf_k=-1,
        )


def test_rrf_top_k_limit():
    bm25_results, dense_results = (
        create_ranked_results()
    )

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

    # RRF 看 rank，不看原始 score 的绝对大小。
    # 所以 Dense 的 B 虽然原始 score 只有 0.001，
    # 但因为 rank=1，仍然应该排在 rank=2 的 A 前面。
    assert results[0].chunk.chunk_id == "B"
    assert results[1].chunk.chunk_id == "A"