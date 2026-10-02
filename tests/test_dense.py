import numpy as np
import pytest

from retrieval.dense import (
    DenseRetriever,
)
from retrieval.document import (
    DocumentChunk,
)


class FakeEncoder:
    """
    测试专用 Embedding 模型。

    不下载真正的模型，
    根据文本内容返回固定向量。
    """

    def __init__(
        self,
    ):
        self.encode_calls = []

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.encode_calls.append(
            list(
                texts
            )
        )

        vectors = []

        for text in texts:
            if "猫" in text:
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
                "深度学习" in text
                or "训练数据" in text
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


def create_test_chunks():
    """创建固定测试 Chunk。"""

    return [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            content=(
                "猫喜欢趴在阳光下晒太阳。"
            ),
            filename="animals.txt",
        ),

        DocumentChunk(
            chunk_id="chunk_002",
            document_id="doc_002",
            content=(
                "数据库索引可以显著加速查询。"
            ),
            filename="database.txt",
        ),

        DocumentChunk(
            chunk_id="chunk_003",
            document_id="doc_003",
            content=(
                "深度学习模型需要大量训练数据。"
            ),
            filename="ai.txt",
        ),
    ]


def test_dense_search_returns_relevant_chunk():
    chunks = (
        create_test_chunks()
    )

    retriever = (
        DenseRetriever(
            chunks,
            encoder=FakeEncoder(),
        )
    )

    results = (
        retriever.search(
            "怎样提高数据库查询速度？",
            top_k=2,
        )
    )

    assert len(
        results
    ) == 2

    assert (
        results[0]
        .chunk
        .chunk_id
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


def test_dense_empty_query():
    retriever = (
        DenseRetriever(
            create_test_chunks(),
            encoder=FakeEncoder(),
        )
    )

    assert (
        retriever.search("")
        == []
    )


def test_dense_empty_corpus():
    retriever = (
        DenseRetriever(
            [],
            encoder=FakeEncoder(),
        )
    )

    assert (
        retriever.search(
            "数据库查询"
        )
        == []
    )


def test_dense_invalid_top_k():
    retriever = (
        DenseRetriever(
            create_test_chunks(),
            encoder=FakeEncoder(),
        )
    )

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "数据库",
            top_k=0,
        )


def test_dense_top_k_limit():
    retriever = (
        DenseRetriever(
            create_test_chunks(),
            encoder=FakeEncoder(),
        )
    )

    results = (
        retriever.search(
            "数据库查询",
            top_k=1,
        )
    )

    assert len(
        results
    ) == 1

    assert (
        results[0].rank
        == 1
    )


def test_dense_top_k_larger_than_corpus():
    chunks = (
        create_test_chunks()
    )

    retriever = (
        DenseRetriever(
            chunks,
            encoder=FakeEncoder(),
        )
    )

    results = (
        retriever.search(
            "数据库查询",
            top_k=100,
        )
    )

    assert (
        len(results)
        == len(chunks)
    )


def test_dense_persistent_index_round_trip(
    tmp_path,
):
    """
    FAISS Index：

        Build
        → Save
        → Load
        → Search

    结果必须保持一致。
    """

    chunks = (
        create_test_chunks()
    )

    first_encoder = (
        FakeEncoder()
    )

    first = DenseRetriever(
        chunks,
        encoder=first_encoder,
    )

    index_path = (
        tmp_path
        / "dense.faiss"
    )

    first.save_index(
        index_path
    )

    assert (
        index_path.exists()
    )

    loaded_index = (
        DenseRetriever.load_index(
            index_path
        )
    )

    second_encoder = (
        FakeEncoder()
    )

    second = DenseRetriever(
        chunks,
        encoder=second_encoder,
        index=loaded_index,
    )

    # 从 Persistent Index 创建 Retriever 时，
    # 不应该重新计算全部 Chunk Embedding。
    assert (
        second_encoder.encode_calls
        == []
    )

    results = second.search(
        "怎样提高数据库查询速度？",
        top_k=1,
    )

    # Search 时只应该编码 Query。
    assert (
        second_encoder.encode_calls
        == [
            [
                "怎样提高数据库查询速度？"
            ]
        ]
    )

    assert (
        results[0]
        .chunk
        .chunk_id
        == "chunk_002"
    )


def test_dense_persistent_index_supports_unicode_path(
    tmp_path,
):
    """
    Windows 用户目录可能包含中文。

    Persistent Index 必须支持：
        中文文件夹
        中文路径
    """

    unicode_dir = (
        tmp_path
        / "中文缓存目录"
    )

    unicode_dir.mkdir()

    index_path = (
        unicode_dir
        / "向量索引.faiss"
    )

    chunks = (
        create_test_chunks()
    )

    first = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
    )

    first.save_index(
        index_path
    )

    assert (
        index_path.exists()
    )

    loaded_index = (
        DenseRetriever.load_index(
            index_path
        )
    )

    second = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
        index=loaded_index,
    )

    assert (
        int(
            second.index.ntotal
        )
        == len(
            chunks
        )
    )


def test_dense_rejects_index_chunk_count_mismatch():
    """
    防止：

        FAISS 是旧版本
        chunks.json 是新版本

    两者错位以后返回错误文档。
    """

    chunks = (
        create_test_chunks()
    )

    original = DenseRetriever(
        chunks,
        encoder=FakeEncoder(),
    )

    with pytest.raises(
        ValueError
    ):
        DenseRetriever(
            chunks[
                :2
            ],
            encoder=FakeEncoder(),
            index=original.index,
        )