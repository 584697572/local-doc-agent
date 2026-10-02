"""Retrieval Status 测试。"""

import json
from types import (
    SimpleNamespace,
)

import numpy as np

import tools.retrieval as retrieval_tool


def test_status_without_runtime_or_cache(
    monkeypatch,
    tmp_path,
):
    """既没 Engine，也没磁盘缓存。"""

    monkeypatch.setattr(
        retrieval_tool,
        "_engine",
        None,
    )

    monkeypatch.setattr(
        retrieval_tool,
        "INDEX_DIR",
        tmp_path,
    )

    status = (
        retrieval_tool
        .get_retrieval_status()
    )

    assert (
        status[
            "runtime_initialized"
        ]
        is False
    )

    assert (
        status[
            "cache_status"
        ]
        == "missing"
    )

    assert (
        status[
            "chunks"
        ]
        == 0
    )


def test_status_reads_disk_snapshot_without_building(
    monkeypatch,
    tmp_path,
):
    """
    Runtime 尚未初始化时，
    只读取 manifest，不触发 Engine Build。
    """

    monkeypatch.setattr(
        retrieval_tool,
        "_engine",
        None,
    )

    monkeypatch.setattr(
        retrieval_tool,
        "INDEX_DIR",
        tmp_path,
    )

    manifest = {
        "source_files": [
            {
                "path": "a.txt",
            },
            {
                "path": "b.txt",
            },
        ],

        "chunk_count": 25,

        "embedding_dimension": 512,
    }

    (
        tmp_path
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )

    status = (
        retrieval_tool
        .get_retrieval_status()
    )

    assert (
        status[
            "runtime_initialized"
        ]
        is False
    )

    assert (
        status[
            "cache_status"
        ]
        == "snapshot_available"
    )

    assert (
        status[
            "documents"
        ]
        == 2
    )

    assert (
        status[
            "chunks"
        ]
        == 25
    )

    assert (
        status[
            "embedding_dimension"
        ]
        == 512
    )


def test_status_uses_live_engine_when_initialized(
    monkeypatch,
):
    """Engine 已加载时优先报告真实 Runtime 状态。"""

    embeddings = np.zeros(
        (
            12,
            512,
        ),
        dtype=np.float32,
    )

    dense = SimpleNamespace(
        embeddings=embeddings,
        index=None,
    )

    retriever = SimpleNamespace(
        dense_retriever=dense,
    )

    fake_engine = SimpleNamespace(
        chunks=[
            SimpleNamespace(
                filename="a.txt"
            )
            for _ in range(12)
        ],

        source_files=[
            {
                "path": "a.txt",
            },
            {
                "path": "b.txt",
            },
        ],

        retriever=retriever,

        cache_status="hit",

        cache_reason=(
            "valid_cache"
        ),

        index_update_mode=(
            "cache_hit"
        ),

        incremental_stats={
            "unchanged_files": 2,
            "encoded_chunks": 0,
        },
    )

    monkeypatch.setattr(
        retrieval_tool,
        "_engine",
        fake_engine,
    )

    status = (
        retrieval_tool
        .get_retrieval_status()
    )

    assert (
        status[
            "runtime_initialized"
        ]
        is True
    )

    assert (
        status[
            "cache_status"
        ]
        == "hit"
    )

    assert (
        status[
            "update_mode"
        ]
        == "cache_hit"
    )

    assert (
        status[
            "documents"
        ]
        == 2
    )

    assert (
        status[
            "chunks"
        ]
        == 12
    )

    assert (
        status[
            "embedding_dimension"
        ]
        == 512
    )

    assert (
        status[
            "incremental_stats"
        ][
            "encoded_chunks"
        ]
        == 0
    )