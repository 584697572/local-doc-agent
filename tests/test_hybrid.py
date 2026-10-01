import numpy as np
import pytest

import retrieval.hybrid as hybrid_module

from retrieval.document import DocumentChunk
from retrieval.hybrid import HybridRetriever
from retrieval.result import RetrievalResult


class FakeEncoder:
    """
    测试专用 Embedding 模型。

    根据文本内容返回固定向量，
    避免 pytest 加载真实 Embedding 模型。
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
                vector = [
                    1.0,
                    0.0,
                    0.0,
                ]

            elif (
                "数据库" in text
                or "查询" in text
            ):
                vector = [
                    0.0,
                    1.0,
                    0.0,
                ]

            elif (
                "Java" in text
                or "JVM" in text
            ):
                vector = [
                    0.0,
                    0.0,
                    1.0,
                ]

            else:
                vector = [
                    0.577,
                    0.577,
                    0.577,
                ]

            vectors.append(
                vector
            )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


class FakeReranker:
    """
    测试专用 Cross-Encoder。

    根据 Chunk 文本返回固定 score，
    避免 pytest 加载真实 Reranker 模型。
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

            scores.append(
                score
            )

        return np.asarray(
            scores,
            dtype=np.float32,
        )


def create_chunks():
    """
    创建一组固定的测试 Chunk。
    """

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
    """
    创建使用 Fake 模型的 HybridRetriever。
    """

    return HybridRetriever(
        create_chunks(),
        dense_encoder=FakeEncoder(),
        reranker_model=FakeReranker(),
    )


def test_hybrid_search_returns_relevant_result():
    """
    完整 Hybrid Retrieval 应该把
    生成器相关 Chunk 排到前面。
    """

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

    assert (
        results[0].rank
        == 1
    )

    assert (
        results[0].score
        >= results[1].score
    )


def test_hybrid_empty_query():
    """
    空 Query 不应该进行检索。
    """

    retriever = create_retriever()

    results = retriever.search(
        "",
        top_k=2,
        candidate_k=3,
    )

    assert results == []


def test_hybrid_empty_corpus():
    """
    空知识库应该返回空结果。
    """

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
    """
    top_k 必须大于 0。
    """

    retriever = create_retriever()

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "Python",
            top_k=0,
            candidate_k=3,
        )


def test_hybrid_invalid_candidate_k():
    """
    candidate_k 必须大于 0。
    """

    retriever = create_retriever()

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "Python",
            top_k=1,
            candidate_k=0,
        )


def test_hybrid_candidate_k_must_cover_top_k():
    """
    Candidate Pool 不能比最终 Top-K 更小。
    """

    retriever = create_retriever()

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "Python",
            top_k=3,
            candidate_k=2,
        )


def test_hybrid_top_k_limit():
    """
    最终返回数量必须受 top_k 限制。
    """

    retriever = create_retriever()

    results = retriever.search(
        "Python 生成器",
        top_k=1,
        candidate_k=3,
    )

    assert len(results) == 1

    assert (
        results[0].rank
        == 1
    )


def test_hybrid_reranks_full_candidate_pool(
    monkeypatch,
):
    """
    Safe Fusion 需要完整的 Reranker 排名。

    因此：

        candidate_k = 3
        final top_k = 2

    时，Cross-Encoder 仍然必须产生
    完整的 Top-3 排名，然后再进行 Safe Fusion。
    """

    retriever = create_retriever()

    captured_top_k = []

    original_rerank = (
        retriever.reranker.rerank
    )

    def capture_rerank(
        query,
        candidates,
        top_k,
    ):
        captured_top_k.append(
            top_k
        )

        return original_rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    monkeypatch.setattr(
        retriever.reranker,
        "rerank",
        capture_rerank,
    )

    results = retriever.search(
        "Python 生成器为什么节省内存？",
        top_k=2,
        candidate_k=3,
    )

    assert len(results) == 2

    # 虽然最终只需要 Top-2，
    # Reranker 仍然应该输出完整 Top-3。
    assert captured_top_k == [
        3
    ]


def test_hybrid_runtime_uses_safe_rank_fusion(
    monkeypatch,
):
    """
    验证 Runtime 最终确实经过 Safe Rank Fusion，
    而不是仍然直接采用 Cross-Encoder 排名。

    构造：

        RRF:
            A #1
            B #2
            C #3
            D #4

        Reranker:
            B #1
            C #2
            D #3
            A #4

    如果完全采用 Reranker：

        B, C, D

    使用当前 Safe Fusion 1:2：

        B, C, A

    A 会因为 RRF 原始排名较高，
    被重新保护回最终 Top-3。
    """

    chunks = {
        name: DocumentChunk(
            chunk_id=name,
            document_id="doc",
            content=f"Chunk {name}",
            filename="demo.txt",
        )
        for name in (
            "A",
            "B",
            "C",
            "D",
        )
    }

    retriever = HybridRetriever(
        list(
            chunks.values()
        ),
        dense_encoder=FakeEncoder(),
        reranker_model=FakeReranker(),
    )

    # 这个测试不关心真正的 BM25 结果，
    # 因为下面会直接固定 RRF 排名。
    monkeypatch.setattr(
        retriever.bm25_retriever,
        "search",
        lambda query, top_k: [],
    )

    # 同理，不关心真正的 Dense 结果。
    monkeypatch.setattr(
        retriever.dense_retriever,
        "search",
        lambda query, top_k: [],
    )

    rrf_results = [
        RetrievalResult(
            chunk=chunks["A"],
            score=0.04,
            rank=1,
        ),

        RetrievalResult(
            chunk=chunks["B"],
            score=0.03,
            rank=2,
        ),

        RetrievalResult(
            chunk=chunks["C"],
            score=0.02,
            rank=3,
        ),

        RetrievalResult(
            chunk=chunks["D"],
            score=0.01,
            rank=4,
        ),
    ]

    def fake_rrf(
        bm25_results,
        dense_results,
        top_k,
    ):
        """
        固定 Stage-3 的 RRF 排名。
        """

        return rrf_results[
            :top_k
        ]

    monkeypatch.setattr(
        hybrid_module,
        "reciprocal_rank_fusion",
        fake_rrf,
    )

    reranked_results = [
        RetrievalResult(
            chunk=chunks["B"],
            score=10.0,
            rank=1,
        ),

        RetrievalResult(
            chunk=chunks["C"],
            score=9.0,
            rank=2,
        ),

        RetrievalResult(
            chunk=chunks["D"],
            score=8.0,
            rank=3,
        ),

        RetrievalResult(
            chunk=chunks["A"],
            score=7.0,
            rank=4,
        ),
    ]

    def fake_rerank(
        query,
        candidates,
        top_k,
    ):
        """
        固定 Cross-Encoder 的排名。
        """

        return reranked_results[
            :top_k
        ]

    monkeypatch.setattr(
        retriever.reranker,
        "rerank",
        fake_rerank,
    )

    results = retriever.search(
        "test query",
        top_k=3,
        candidate_k=4,
    )

    result_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    # 如果这里是 B,C,D，
    # 说明 Runtime 仍然直接使用 Reranker。
    #
    # B,C,A 才说明 Safe Fusion 真正生效。
    assert result_ids == [
        "B",
        "C",
        "A",
    ]


def test_hybrid_default_safe_weights():
    """
    Retrieval v1 当前冻结参数：

        RRF       = 1
        Reranker  = 2
        fusion_k  = 60
    """

    retriever = create_retriever()

    assert (
        retriever.safe_rrf_weight
        == 1.0
    )

    assert (
        retriever.safe_reranker_weight
        == 2.0
    )

    assert (
        retriever.safe_fusion_k
        == 60
    )