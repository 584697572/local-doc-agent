import numpy as np
import pytest

from retrieval.engine import RetrievalEngine


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

            else:
                vector = [0.0, 0.0, 1.0]

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

            elif "数据库" in text:
                score = 5.0

            else:
                score = 1.0

            scores.append(score)

        return np.asarray(
            scores,
            dtype=np.float32,
        )


def create_engine(data_dir):
    return RetrievalEngine(
        data_dir=data_dir,
        chunk_size=100,
        overlap=20,
        dense_encoder=FakeEncoder(),
        reranker_model=FakeReranker(),
    )


def test_engine_builds_documents(tmp_path):
    file_path = tmp_path / "python.txt"

    file_path.write_text(
        "Python 生成器采用惰性求值，可以减少内存占用。",
        encoding="utf-8",
    )

    engine = create_engine(tmp_path)

    engine.build()

    assert len(engine.chunks) >= 1

    assert (
        engine.chunks[0].filename
        == "python.txt"
    )


def test_engine_search_returns_relevant_result(
    tmp_path,
):
    python_file = tmp_path / "python.txt"
    database_file = tmp_path / "database.txt"

    python_file.write_text(
        "Python 生成器采用惰性求值，可以减少内存占用。",
        encoding="utf-8",
    )

    database_file.write_text(
        "数据库索引可以提高查询速度。",
        encoding="utf-8",
    )

    engine = create_engine(tmp_path)

    engine.build()

    results = engine.search(
        "生成器为什么节省内存？",
        top_k=1,
        candidate_k=2,
    )

    assert len(results) == 1

    assert (
        results[0].chunk.filename
        == "python.txt"
    )

    assert (
        "生成器"
        in results[0].chunk.content
    )


def test_engine_ignores_unsupported_files(
    tmp_path,
):
    txt_file = tmp_path / "demo.txt"
    jpg_file = tmp_path / "photo.jpg"

    txt_file.write_text(
        "这是正常文档。",
        encoding="utf-8",
    )

    jpg_file.write_text(
        "这不是真正图片，只用于测试扩展名过滤。",
        encoding="utf-8",
    )

    engine = create_engine(tmp_path)

    engine.build()

    filenames = [
        chunk.filename
        for chunk in engine.chunks
    ]

    assert "demo.txt" in filenames
    assert "photo.jpg" not in filenames


def test_engine_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    engine = create_engine(missing_dir)

    with pytest.raises(FileNotFoundError):
        engine.build()


def test_engine_requires_build_before_search(
    tmp_path,
):
    engine = create_engine(tmp_path)

    with pytest.raises(RuntimeError):
        engine.search(
            "测试问题"
        )


def test_engine_empty_directory(tmp_path):
    engine = create_engine(tmp_path)

    engine.build()

    assert engine.chunks == []

    results = engine.search(
        "任何问题",
        top_k=1,
        candidate_k=1,
    )

    assert results == []


def test_engine_scans_subdirectories(
    tmp_path,
):
    sub_dir = tmp_path / "docs"
    sub_dir.mkdir()

    file_path = sub_dir / "nested.txt"

    file_path.write_text(
        "Python 生成器采用惰性求值。",
        encoding="utf-8",
    )

    engine = create_engine(tmp_path)

    engine.build()

    filenames = [
        chunk.filename
        for chunk in engine.chunks
    ]

    assert "nested.txt" in filenames