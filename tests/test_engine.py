import numpy as np
import pytest

from retrieval.engine import (
    DENSE_INDEX_FILENAME,
    RetrievalEngine,
)


class FakeEncoder:
    """
    测试专用 Embedding 模型。

    encode_calls 可以检查：

        第一次 Build
            → 应该计算文档 Embedding。

        第二次 Cache Hit
            → Build 阶段不应该再调用 encode。
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
        # 保存本次真正被编码的文本。
        self.encode_calls.append(
            list(
                texts
            )
        )

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

            else:
                vector = [
                    0.0,
                    0.0,
                    1.0,
                ]

            vectors.append(
                vector
            )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


class FakeReranker:
    """测试专用 Reranker。"""

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

            elif "数据库" in text:
                score = 5.0

            else:
                score = 1.0

            scores.append(
                score
            )

        return np.asarray(
            scores,
            dtype=np.float32,
        )


def create_engine(
    data_dir,
    *,
    cache_dir=None,
    encoder=None,
    chunk_size=100,
    overlap=20,
):
    """
    统一创建测试 Engine。

    测试 Persistent Index 时使用固定的
    embedding_model_name，
    防止 Cache Identity 不稳定。
    """

    return RetrievalEngine(
        data_dir=data_dir,

        chunk_size=(
            chunk_size
        ),

        overlap=(
            overlap
        ),

        dense_encoder=(
            encoder
            or FakeEncoder()
        ),

        reranker_model=(
            FakeReranker()
        ),

        embedding_model_name=(
            "test-fake-encoder-v1"
        ),

        cache_dir=(
            cache_dir
        ),
    )


def test_engine_builds_documents(
    tmp_path,
):
    file_path = (
        tmp_path
        / "python.txt"
    )

    file_path.write_text(
        (
            "Python 生成器采用惰性求值，"
            "可以减少内存占用。"
        ),
        encoding="utf-8",
    )

    engine = create_engine(
        tmp_path
    )

    engine.build()

    assert (
        len(
            engine.chunks
        )
        >= 1
    )

    assert (
        engine.chunks[0].filename
        == "python.txt"
    )

    # 没配置 cache_dir，
    # 保持原来的纯内存行为。
    assert (
        engine.cache_status
        == "disabled"
    )


def test_engine_search_returns_relevant_result(
    tmp_path,
):
    python_file = (
        tmp_path
        / "python.txt"
    )

    database_file = (
        tmp_path
        / "database.txt"
    )

    python_file.write_text(
        (
            "Python 生成器采用惰性求值，"
            "可以减少内存占用。"
        ),
        encoding="utf-8",
    )

    database_file.write_text(
        "数据库索引可以提高查询速度。",
        encoding="utf-8",
    )

    engine = create_engine(
        tmp_path
    )

    engine.build()

    results = engine.search(
        "生成器为什么节省内存？",
        top_k=1,
        candidate_k=2,
    )

    assert len(
        results
    ) == 1

    assert (
        results[0].chunk.filename
        == "python.txt"
    )

    assert (
        "生成器"
        in results[0]
        .chunk
        .content
    )


def test_engine_ignores_unsupported_files(
    tmp_path,
):
    txt_file = (
        tmp_path
        / "demo.txt"
    )

    jpg_file = (
        tmp_path
        / "photo.jpg"
    )

    txt_file.write_text(
        "这是正常文档。",
        encoding="utf-8",
    )

    jpg_file.write_text(
        (
            "这不是真正图片，"
            "只用于测试扩展名过滤。"
        ),
        encoding="utf-8",
    )

    engine = create_engine(
        tmp_path
    )

    engine.build()

    filenames = [
        chunk.filename
        for chunk
        in engine.chunks
    ]

    assert (
        "demo.txt"
        in filenames
    )

    assert (
        "photo.jpg"
        not in filenames
    )


def test_engine_missing_directory(
    tmp_path,
):
    missing_dir = (
        tmp_path
        / "missing"
    )

    engine = create_engine(
        missing_dir
    )

    with pytest.raises(
        FileNotFoundError
    ):
        engine.build()


def test_engine_requires_build_before_search(
    tmp_path,
):
    engine = create_engine(
        tmp_path
    )

    with pytest.raises(
        RuntimeError
    ):
        engine.search(
            "测试问题"
        )


def test_engine_empty_directory(
    tmp_path,
):
    engine = create_engine(
        tmp_path
    )

    engine.build()

    assert (
        engine.chunks
        == []
    )

    results = engine.search(
        "任何问题",
        top_k=1,
        candidate_k=1,
    )

    assert results == []


def test_engine_scans_subdirectories(
    tmp_path,
):
    sub_dir = (
        tmp_path
        / "docs"
    )

    sub_dir.mkdir()

    file_path = (
        sub_dir
        / "nested.txt"
    )

    file_path.write_text(
        "Python 生成器采用惰性求值。",
        encoding="utf-8",
    )

    engine = create_engine(
        tmp_path
    )

    engine.build()

    filenames = [
        chunk.filename
        for chunk
        in engine.chunks
    ]

    assert (
        "nested.txt"
        in filenames
    )


# ======================================================
# Persistent Index v1
# ======================================================


def test_engine_creates_persistent_cache(
    tmp_path,
):
    """
    第一次 Build 应该：

        计算文档 Embedding
        +
        写 Persistent Index。
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
        / "python.txt"
    ).write_text(
        (
            "Python 生成器采用惰性求值，"
            "可以减少内存占用。"
        ),
        encoding="utf-8",
    )

    encoder = FakeEncoder()

    engine = create_engine(
        data_dir,
        cache_dir=cache_dir,
        encoder=encoder,
    )

    engine.build()

    assert (
        engine.cache_status
        == "rebuilt"
    )

    # Build 时应该编码一次整个文档集合。
    assert len(
        encoder.encode_calls
    ) == 1

    assert (
        cache_dir
        / "manifest.json"
    ).exists()

    assert (
        cache_dir
        / "chunks.json"
    ).exists()

    assert (
        cache_dir
        / DENSE_INDEX_FILENAME
    ).exists()


def test_engine_cache_hit_skips_document_embedding(
    tmp_path,
):
    """
    最关键测试。

    第一次进程：
        Build → 文档 Embedding。

    第二次 Engine：
        Cache Hit → 不重新 Embedding 文档。
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
        / "python.txt"
    ).write_text(
        (
            "Python 生成器采用惰性求值，"
            "可以减少内存占用。"
        ),
        encoding="utf-8",
    )

    first_encoder = (
        FakeEncoder()
    )

    first_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=first_encoder,
        )
    )

    first_engine.build()

    assert len(
        first_encoder.encode_calls
    ) == 1

    # ----------------------------------------------
    # 模拟下一次程序启动。
    # ----------------------------------------------

    second_encoder = (
        FakeEncoder()
    )

    second_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=second_encoder,
        )
    )

    second_engine.build()

    assert (
        second_engine.cache_status
        == "hit"
    )

    # Cache Hit 的核心：
    # Build 阶段完全没有执行 encoder.encode。
    assert (
        second_encoder.encode_calls
        == []
    )

    # 真正 Search 时，
    # Query 自己仍然需要一次 Embedding。
    results = (
        second_engine.search(
            "生成器为什么节省内存？",
            top_k=1,
            candidate_k=1,
        )
    )

    assert len(
        second_encoder.encode_calls
    ) == 1

    assert (
        second_encoder
        .encode_calls[0]
        == [
            "生成器为什么节省内存？"
        ]
    )

    assert (
        results[0]
        .chunk
        .filename
        == "python.txt"
    )


def test_engine_invalidates_cache_when_document_changes(
    tmp_path,
):
    """
    源文档内容变化以后，
    Persistent Index 必须自动失效。
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

    file_path = (
        data_dir
        / "demo.txt"
    )

    file_path.write_text(
        "第一版文档内容。",
        encoding="utf-8",
    )

    first_encoder = (
        FakeEncoder()
    )

    first_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=first_encoder,
        )
    )

    first_engine.build()

    # 修改源文件。
    file_path.write_text(
        "第二版文档内容，已经发生变化。",
        encoding="utf-8",
    )

    second_encoder = (
        FakeEncoder()
    )

    second_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=second_encoder,
        )
    )

    second_engine.build()

    assert (
        second_engine.cache_status
        == "rebuilt"
    )

    assert (
        second_engine.cache_reason
        == "cache_stale"
    )

    # 文档发生变化，
    # 必须重新计算文档 Embedding。
    assert len(
        second_encoder.encode_calls
    ) == 1


def test_engine_invalidates_cache_when_chunk_config_changes(
    tmp_path,
):
    """
    即使源文档不变，
    chunk_size 改变也必须重建。
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
        / "demo.txt"
    ).write_text(
        (
            "这是一段用于测试 Chunk 参数"
            "发生变化以后 Cache 失效的文本。"
            * 5
        ),
        encoding="utf-8",
    )

    first_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=FakeEncoder(),
            chunk_size=100,
            overlap=20,
        )
    )

    first_engine.build()

    second_encoder = (
        FakeEncoder()
    )

    second_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=second_encoder,
            chunk_size=50,
            overlap=10,
        )
    )

    second_engine.build()

    assert (
        second_engine.cache_status
        == "rebuilt"
    )

    assert (
        second_engine.cache_reason
        == "cache_stale"
    )

    assert len(
        second_encoder.encode_calls
    ) == 1


def test_engine_corrupt_cache_falls_back_to_rebuild(
    tmp_path,
):
    """
    dense.faiss 损坏时：

        不能整个程序崩掉。

    应该：
        检测 checksum 失败
        → 自动全量重建。
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
        / "demo.txt"
    ).write_text(
        "数据库索引可以提高查询速度。",
        encoding="utf-8",
    )

    first_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=FakeEncoder(),
        )
    )

    first_engine.build()

    # 人为破坏 FAISS 文件。
    (
        cache_dir
        / DENSE_INDEX_FILENAME
    ).write_bytes(
        b"broken-faiss-index"
    )

    second_encoder = (
        FakeEncoder()
    )

    second_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=second_encoder,
        )
    )

    # 不应该抛异常。
    second_engine.build()

    assert (
        second_engine.cache_status
        == "rebuilt"
    )

    assert (
        second_engine.cache_reason
        .startswith(
            "cache_corrupt:"
        )
    )

    # 损坏以后重新 Embedding。
    assert len(
        second_encoder.encode_calls
    ) == 1


def test_engine_force_rebuild_ignores_valid_cache(
    tmp_path,
):
    """
    运维场景：

        Cache 明明有效，
        但用户明确要求强制重建。
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
        / "demo.txt"
    ).write_text(
        "数据库索引测试。",
        encoding="utf-8",
    )

    first_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=FakeEncoder(),
        )
    )

    first_engine.build()

    second_encoder = (
        FakeEncoder()
    )

    second_engine = (
        create_engine(
            data_dir,
            cache_dir=cache_dir,
            encoder=second_encoder,
        )
    )

    second_engine.build(
        force_rebuild=True
    )

    assert (
        second_engine.cache_status
        == "rebuilt"
    )

    assert (
        second_engine.cache_reason
        == "forced_rebuild"
    )

    assert len(
        second_encoder.encode_calls
    ) == 1