import numpy as np
import pytest

from retrieval.dense import DenseRetriever
from retrieval.document import DocumentChunk


class FakeEncoder:
    """
    测试专用的假 Embedding 模型。

    不下载真实模型，
    根据文本内容返回固定向量。
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
            if "猫" in text:
                vector = [1.0, 0.0, 0.0]

            elif "数据库" in text or "查询" in text:
                vector = [0.0, 1.0, 0.0]

            elif "深度学习" in text or "训练数据" in text:
                vector = [0.0, 0.0, 1.0]

            else:
                vector = [0.577, 0.577, 0.577]

            vectors.append(vector)

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


def create_test_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            content="猫喜欢趴在阳光下晒太阳。",
            filename="animals.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_id="doc_002",
            content="数据库索引可以显著加速查询。",
            filename="database.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            document_id="doc_003",
            content="深度学习模型需要大量训练数据。",
            filename="ai.txt",
        ),
    ]


def test_dense_search_returns_relevant_chunk():
    chunks = create_test_chunks()

    retriever = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
    )

    results = retriever.search(
        "怎样提高数据库查询速度？",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].chunk.chunk_id == "chunk_002"

    assert results[0].rank == 1

    assert results[0].score >= results[1].score


def test_dense_empty_query():
    chunks = create_test_chunks()

    retriever = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
    )

    results = retriever.search("")

    assert results == []


def test_dense_empty_corpus():
    retriever = DenseRetriever(
        [],
        encoder=FakeEncoder(),
    )

    results = retriever.search(
        "数据库查询"
    )

    assert results == []


def test_dense_invalid_top_k():
    chunks = create_test_chunks()

    retriever = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
    )

    with pytest.raises(ValueError):
        retriever.search(
            "数据库",
            top_k=0,
        )


def test_dense_top_k_limit():
    chunks = create_test_chunks()

    retriever = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
    )

    results = retriever.search(
        "数据库查询",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1


def test_dense_top_k_larger_than_corpus():
    chunks = create_test_chunks()

    retriever = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
    )

    results = retriever.search(
        "数据库查询",
        top_k=100,
    )

    assert len(results) == len(chunks)