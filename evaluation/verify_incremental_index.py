"""
真实验证 Incremental Indexing。

验证四种情况：

1. 第一次建立三个文件；
2. 完全不修改 → Cache Hit；
3. 只修改 B → 只重新编码 B；
4. 新增 D → 只编码 D；
5. 删除 C → 不重新编码任何文件。

使用系统临时目录，不修改项目真实 data/。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from sentence_transformers import SentenceTransformer

from retrieval.dense import (
    DEFAULT_EMBEDDING_MODEL,
)
from retrieval.engine import RetrievalEngine


def create_engine(
    data_dir: Path,
    cache_dir: Path,
    encoder,
) -> RetrievalEngine:
    """
    所有 Engine 共用同一个真实 Embedding 模型。

    这样测的是 Incremental Index 行为，
    不重复承担模型加载成本。
    """

    return RetrievalEngine(
        data_dir=data_dir,

        # 每个测试文件只产生一个 Chunk，
        # 方便直接观察复用 / 重算数量。
        chunk_size=1000,
        overlap=0,

        dense_encoder=encoder,

        embedding_model_name=(
            DEFAULT_EMBEDDING_MODEL
        ),

        cache_dir=cache_dir,
    )


def show_result(
    title: str,
    engine: RetrievalEngine,
) -> None:
    """打印一次 Build 的关键状态。"""

    print()
    print(
        "=" * 60
    )

    print(title)

    print(
        "=" * 60
    )

    print(
        "cache_status      =",
        engine.cache_status,
    )

    print(
        "cache_reason      =",
        engine.cache_reason,
    )

    print(
        "index_update_mode =",
        engine.index_update_mode,
    )

    print(
        "incremental_stats =",
        engine.incremental_stats,
    )

    print(
        "chunk_count       =",
        len(
            engine.chunks
        ),
    )

    dense = (
        engine
        .retriever
        .dense_retriever
    )

    print(
        "embedding_shape   =",
        dense.embeddings.shape,
    )


def main() -> None:
    print(
        "Loading shared embedding model..."
    )

    # 整个验证过程只加载一次真实模型。
    encoder = SentenceTransformer(
        DEFAULT_EMBEDDING_MODEL
    )

    with TemporaryDirectory(
        prefix="localdoc_incremental_"
    ) as temporary:
        root = Path(
            temporary
        )

        data_dir = (
            root
            / "data"
        )

        cache_dir = (
            root
            / "cache"
        )

        data_dir.mkdir()

        # ==================================================
        # Initial Corpus
        # ==================================================

        a_file = (
            data_dir
            / "a.txt"
        )

        b_file = (
            data_dir
            / "b.txt"
        )

        c_file = (
            data_dir
            / "c.txt"
        )

        a_file.write_text(
            "苹果是一种水果。",
            encoding="utf-8",
        )

        b_file.write_text(
            "数据库索引可以提高查询速度。",
            encoding="utf-8",
        )

        c_file.write_text(
            "Python 生成器采用惰性求值。",
            encoding="utf-8",
        )

        # ==================================================
        # 1. First Build
        # ==================================================

        engine = create_engine(
            data_dir,
            cache_dir,
            encoder,
        )

        engine.build()

        show_result(
            "1. INITIAL FULL BUILD",
            engine,
        )

        # ==================================================
        # 2. No Changes
        # ==================================================

        engine = create_engine(
            data_dir,
            cache_dir,
            encoder,
        )

        engine.build()

        show_result(
            "2. NO CHANGES",
            engine,
        )

        # ==================================================
        # 3. Modify B Only
        # ==================================================

        b_file.write_text(
            "网络协议负责设备之间的通信。",
            encoding="utf-8",
        )

        engine = create_engine(
            data_dir,
            cache_dir,
            encoder,
        )

        engine.build()

        show_result(
            "3. MODIFY B ONLY",
            engine,
        )

        # ==================================================
        # 4. Add D Only
        # ==================================================

        d_file = (
            data_dir
            / "d.txt"
        )

        d_file.write_text(
            "向量数据库用于相似度搜索。",
            encoding="utf-8",
        )

        engine = create_engine(
            data_dir,
            cache_dir,
            encoder,
        )

        engine.build()

        show_result(
            "4. ADD D ONLY",
            engine,
        )

        # ==================================================
        # 5. Delete C Only
        # ==================================================

        c_file.unlink()

        engine = create_engine(
            data_dir,
            cache_dir,
            encoder,
        )

        engine.build()

        show_result(
            "5. DELETE C ONLY",
            engine,
        )

        # ==================================================
        # Hard Assertions
        #
        # 如果实现并非真正增量，
        # 脚本应该直接失败，而不是只打印漂亮数字。
        # ==================================================

        stats = (
            engine.incremental_stats
        )

        assert (
            engine.index_update_mode
            == "incremental_update"
        )

        assert (
            stats[
                "deleted_files"
            ]
            == 1
        )

        assert (
            stats[
                "encoded_chunks"
            ]
            == 0
        )

        print()
        print(
            "Incremental Index verification PASSED."
        )


if __name__ == "__main__":
    main()