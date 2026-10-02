"""Incremental Indexing 测试。"""

import json

import numpy as np

from retrieval.engine import (
    EMBEDDINGS_FILENAME,
    RetrievalEngine,
)


class FakeEncoder:
    """
    记录真正执行过的 Embedding 文本。

    这样可以证明：
        修改一个文件
        ≠
        所有文件重新 Embedding。
    """

    def __init__(self):
        self.encode_calls = []

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        texts = list(texts)

        self.encode_calls.append(
            texts
        )

        vectors = []

        for text in texts:
            if "苹果" in text:
                vector = [
                    1.0,
                    0.0,
                    0.0,
                ]

            elif "数据库" in text:
                vector = [
                    0.0,
                    1.0,
                    0.0,
                ]

            elif "网络" in text:
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
    """测试用 CrossEncoder。"""

    def predict(
        self,
        pairs,
        show_progress_bar=False,
    ):
        scores = []

        for query, text in pairs:
            if (
                "网络" in query
                and "网络" in text
            ):
                score = 10.0

            elif (
                "数据库" in query
                and "数据库" in text
            ):
                score = 10.0

            elif (
                "苹果" in query
                and "苹果" in text
            ):
                score = 10.0

            else:
                score = 0.0

            scores.append(
                score
            )

        return np.asarray(
            scores,
            dtype=np.float32,
        )


def create_engine(
    data_dir,
    cache_dir,
    encoder,
    *,
    chunk_size=1000,
):
    """
    测试中让每个小文件只有一个 Chunk，
    更容易直接观察增量行为。
    """

    return RetrievalEngine(
        data_dir=data_dir,
        chunk_size=chunk_size,
        overlap=0,
        dense_encoder=encoder,
        reranker_model=FakeReranker(),
        embedding_model_name=(
            "fake-incremental-encoder-v1"
        ),
        cache_dir=cache_dir,
    )


def test_unchanged_corpus_is_cache_hit(
    tmp_path,
):
    """
    文件完全没变化：
        第二次启动不做文档 Embedding。
    """

    data_dir = (
        tmp_path
        / "data"
    )

    cache_dir = (
        tmp_path
        / "cache"
    )

    data_dir.mkdir()

    (
        data_dir
        / "a.txt"
    ).write_text(
        "苹果是一种水果。",
        encoding="utf-8",
    )

    first_encoder = (
        FakeEncoder()
    )

    first = create_engine(
        data_dir,
        cache_dir,
        first_encoder,
    )

    first.build()

    assert (
        len(
            first_encoder.encode_calls
        )
        == 1
    )

    second_encoder = (
        FakeEncoder()
    )

    second = create_engine(
        data_dir,
        cache_dir,
        second_encoder,
    )

    second.build()

    assert (
        second.cache_status
        == "hit"
    )

    assert (
        second.index_update_mode
        == "cache_hit"
    )

    assert (
        second_encoder.encode_calls
        == []
    )


def test_modified_file_only_reencodes_modified_file(
    tmp_path,
):
    """
    A 不变，B 修改。

    第二次 Build：
        A 不允许重新 Embedding。
        B 必须重新 Embedding。
    """

    data_dir = (
        tmp_path
        / "data"
    )

    cache_dir = (
        tmp_path
        / "cache"
    )

    data_dir.mkdir()

    a_file = (
        data_dir
        / "a.txt"
    )

    b_file = (
        data_dir
        / "b.txt"
    )

    a_file.write_text(
        "苹果是一种水果。",
        encoding="utf-8",
    )

    b_file.write_text(
        "数据库索引可以提高查询速度。",
        encoding="utf-8",
    )

    first = create_engine(
        data_dir,
        cache_dir,
        FakeEncoder(),
    )

    first.build()

    # 只修改 B。
    b_file.write_text(
        "网络协议负责设备之间的通信。",
        encoding="utf-8",
    )

    second_encoder = (
        FakeEncoder()
    )

    second = create_engine(
        data_dir,
        cache_dir,
        second_encoder,
    )

    second.build()

    assert (
        second.cache_status
        == "rebuilt"
    )

    assert (
        second.index_update_mode
        == "incremental_update"
    )

    # 真正送入 Encoder 的只有 B。
    assert (
        len(
            second_encoder.encode_calls
        )
        == 1
    )

    encoded_texts = (
        second_encoder.encode_calls[0]
    )

    assert len(
        encoded_texts
    ) == 1

    assert (
        "网络"
        in encoded_texts[0]
    )

    assert all(
        "苹果" not in text
        for text in encoded_texts
    )

    stats = (
        second.incremental_stats
    )

    assert (
        stats[
            "unchanged_files"
        ]
        == 1
    )

    assert (
        stats[
            "modified_files"
        ]
        == 1
    )

    assert (
        stats[
            "added_files"
        ]
        == 0
    )

    assert (
        stats[
            "deleted_files"
        ]
        == 0
    )

    assert (
        stats[
            "reused_chunks"
        ]
        == 1
    )

    assert (
        stats[
            "encoded_chunks"
        ]
        == 1
    )

    # 新索引应该能搜到修改后的内容。
    results = second.search(
        "网络通信",
        top_k=1,
        candidate_k=2,
    )

    assert (
        results[0]
        .chunk
        .filename
        == "b.txt"
    )

    assert (
        "网络"
        in results[0]
        .chunk
        .content
    )


def test_added_file_only_encodes_new_file(
    tmp_path,
):
    """
    新增 B：
        A 复用；
        只计算 B。
    """

    data_dir = (
        tmp_path
        / "data"
    )

    cache_dir = (
        tmp_path
        / "cache"
    )

    data_dir.mkdir()

    (
        data_dir
        / "a.txt"
    ).write_text(
        "苹果是一种水果。",
        encoding="utf-8",
    )

    first = create_engine(
        data_dir,
        cache_dir,
        FakeEncoder(),
    )

    first.build()

    (
        data_dir
        / "b.txt"
    ).write_text(
        "数据库索引可以提高查询速度。",
        encoding="utf-8",
    )

    second_encoder = (
        FakeEncoder()
    )

    second = create_engine(
        data_dir,
        cache_dir,
        second_encoder,
    )

    second.build()

    assert (
        second.index_update_mode
        == "incremental_update"
    )

    assert (
        second.incremental_stats[
            "added_files"
        ]
        == 1
    )

    assert (
        second.incremental_stats[
            "unchanged_files"
        ]
        == 1
    )

    assert (
        len(
            second_encoder.encode_calls
        )
        == 1
    )

    encoded_texts = (
        second_encoder.encode_calls[0]
    )

    assert len(
        encoded_texts
    ) == 1

    assert (
        "数据库"
        in encoded_texts[0]
    )

    assert all(
        "苹果" not in text
        for text in encoded_texts
    )


def test_deleted_file_requires_no_reembedding(
    tmp_path,
):
    """
    删除 B：

        A 的旧向量直接复用；
        B 直接丢弃；
        不需要任何新 Embedding。
    """

    data_dir = (
        tmp_path
        / "data"
    )

    cache_dir = (
        tmp_path
        / "cache"
    )

    data_dir.mkdir()

    (
        data_dir
        / "a.txt"
    ).write_text(
        "苹果是一种水果。",
        encoding="utf-8",
    )

    b_file = (
        data_dir
        / "b.txt"
    )

    b_file.write_text(
        "数据库索引可以提高查询速度。",
        encoding="utf-8",
    )

    first = create_engine(
        data_dir,
        cache_dir,
        FakeEncoder(),
    )

    first.build()

    b_file.unlink()

    second_encoder = (
        FakeEncoder()
    )

    second = create_engine(
        data_dir,
        cache_dir,
        second_encoder,
    )

    second.build()

    assert (
        second.index_update_mode
        == "incremental_update"
    )

    assert (
        second.incremental_stats[
            "deleted_files"
        ]
        == 1
    )

    assert (
        second.incremental_stats[
            "unchanged_files"
        ]
        == 1
    )

    assert (
        second.incremental_stats[
            "encoded_chunks"
        ]
        == 0
    )

    # 删除文件根本不需要 Encoder。
    assert (
        second_encoder.encode_calls
        == []
    )

    filenames = {
        chunk.filename
        for chunk in second.chunks
    }

    assert filenames == {
        "a.txt"
    }


def test_pipeline_change_forces_full_rebuild(
    tmp_path,
):
    """
    Chunk 参数改变时不能复用旧向量。

    这种情况必须 Full Rebuild。
    """

    data_dir = (
        tmp_path
        / "data"
    )

    cache_dir = (
        tmp_path
        / "cache"
    )

    data_dir.mkdir()

    (
        data_dir
        / "a.txt"
    ).write_text(
        (
            "苹果是一种水果。"
            "苹果也可以用于制作果汁。"
            "苹果含有多种营养成分。"
        ),
        encoding="utf-8",
    )

    first = create_engine(
        data_dir,
        cache_dir,
        FakeEncoder(),
        chunk_size=1000,
    )

    first.build()

    second_encoder = (
        FakeEncoder()
    )

    second = create_engine(
        data_dir,
        cache_dir,
        second_encoder,
        chunk_size=10,
    )

    second.build()

    assert (
        second.index_update_mode
        == "full_rebuild"
    )

    assert (
        second.cache_reason
        == "cache_stale"
    )

    assert (
        len(
            second_encoder.encode_calls
        )
        == 1
    )


def test_incremental_manifest_contains_file_ranges(
    tmp_path,
):
    """
    Manifest 必须记录：

        source file
        sha256
        chunk_start
        chunk_count

    否则下次无法知道旧向量属于哪个文件。
    """

    data_dir = (
        tmp_path
        / "data"
    )

    cache_dir = (
        tmp_path
        / "cache"
    )

    data_dir.mkdir()

    (
        data_dir
        / "a.txt"
    ).write_text(
        "苹果是一种水果。",
        encoding="utf-8",
    )

    (
        data_dir
        / "b.txt"
    ).write_text(
        "数据库索引可以提高查询速度。",
        encoding="utf-8",
    )

    engine = create_engine(
        data_dir,
        cache_dir,
        FakeEncoder(),
    )

    engine.build()

    manifest = json.loads(
        (
            cache_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "schema_version"
        ]
        == 2
    )

    sources = (
        manifest[
            "source_files"
        ]
    )

    assert len(
        sources
    ) == 2

    assert (
        sources[0][
            "chunk_start"
        ]
        == 0
    )

    assert (
        sources[0][
            "chunk_count"
        ]
        == 1
    )

    assert (
        sources[1][
            "chunk_start"
        ]
        == 1
    )

    assert (
        sources[1][
            "chunk_count"
        ]
        == 1
    )

    assert (
        cache_dir
        / EMBEDDINGS_FILENAME
    ).exists()