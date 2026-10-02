"""本地知识库 RetrievalEngine。"""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from retrieval.chunker import chunk_document
from retrieval.dense import (
    DEFAULT_EMBEDDING_MODEL,
    DenseRetriever,
)
from retrieval.document import DocumentChunk
from retrieval.hybrid import HybridRetriever
from retrieval.loader import load_document
from retrieval.result import RetrievalResult


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}


# ======================================================
# Persistent / Incremental Index v2
# ======================================================

CACHE_SCHEMA_VERSION = 2
CHUNKING_VERSION = 1
DENSE_INDEX_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
CHUNKS_FILENAME = "chunks.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
DENSE_INDEX_FILENAME = "dense.faiss"


def _sha256_file(
    path: Path,
) -> str:
    """计算文件 SHA256。"""

    digest = sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def _sha256_json(
    value,
) -> str:
    """稳定 JSON SHA256。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256(
        encoded
    ).hexdigest()


class RetrievalEngine:
    """
    本地知识库检索总入口。

    支持：

    1. Full Build
    2. Persistent Cache Hit
    3. Incremental Update

    Incremental Update：

        unchanged
            → 复用旧 Chunk + Embedding

        modified
            → 重新 Load / Chunk / Embedding

        added
            → 只处理新文件

        deleted
            → 不再加入新索引
    """

    def __init__(
        self,
        data_dir: str | Path,
        chunk_size: int = 500,
        overlap: int = 100,
        dense_encoder=None,
        reranker_model=None,
        embedding_model_name: str = (
            DEFAULT_EMBEDDING_MODEL
        ),
        cache_dir: str | Path | None = None,
    ):
        self.data_dir = Path(
            data_dir
        )

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.dense_encoder = (
            dense_encoder
        )

        self.reranker_model = (
            reranker_model
        )

        self.embedding_model_name = (
            embedding_model_name
        )

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else None
        )

        self.chunks: list[
            DocumentChunk
        ] = []

        self.source_files = []

        self.retriever = None

        self.cache_status = (
            "not_built"
        )

        self.cache_reason = None

        # 比 cache_status 更细：
        #
        # cache_hit
        # full_rebuild
        # incremental_update
        self.index_update_mode = (
            "not_built"
        )

        self.incremental_stats = {}

    # ==================================================
    # Source Files
    # ==================================================

    def _list_source_files(
        self,
    ) -> list[Path]:
        """扫描支持的知识库文件。"""

        return sorted(
            path
            for path
            in self.data_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

    def _describe_source_files(
        self,
        file_paths: list[Path],
    ) -> list[dict]:
        """记录当前每个源文件的身份。"""

        result = []

        for path in file_paths:
            result.append(
                {
                    "path": (
                        path.relative_to(
                            self.data_dir
                        ).as_posix()
                    ),
                    "size": (
                        path.stat().st_size
                    ),
                    "sha256": (
                        _sha256_file(
                            path
                        )
                    ),
                }
            )

        return result

    @staticmethod
    def _source_signature(
        source_files: list[dict],
    ) -> list[dict]:
        """
        去掉 chunk_start/chunk_count，
        只比较原始文件本身。
        """

        return [
            {
                "path": item["path"],
                "size": item["size"],
                "sha256": item["sha256"],
            }
            for item in source_files
        ]

    def _build_pipeline_identity(
        self,
    ) -> dict:
        """
        Pipeline Identity。

        这些参数变化时，
        旧 Embedding 不能安全复用。
        """

        return {
            "schema_version": (
                CACHE_SCHEMA_VERSION
            ),
            "chunking_version": (
                CHUNKING_VERSION
            ),
            "dense_index_version": (
                DENSE_INDEX_VERSION
            ),
            "chunk_size": (
                self.chunk_size
            ),
            "overlap": (
                self.overlap
            ),
            "embedding_model": (
                self.embedding_model_name
            ),
        }

    # ==================================================
    # Paths
    # ==================================================

    def _manifest_path(self):
        return (
            self.cache_dir
            / MANIFEST_FILENAME
        )

    def _chunks_path(self):
        return (
            self.cache_dir
            / CHUNKS_FILENAME
        )

    def _embeddings_path(self):
        return (
            self.cache_dir
            / EMBEDDINGS_FILENAME
        )

    def _dense_index_path(self):
        return (
            self.cache_dir
            / DENSE_INDEX_FILENAME
        )

    # ==================================================
    # JSON
    # ==================================================

    @staticmethod
    def _write_json_atomic(
        path: Path,
        data,
    ) -> None:
        """先写临时文件，再原子替换。"""

        temporary = path.with_name(
            path.name + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary,
            path,
        )

    def _load_chunks_from_cache(
        self,
        path: Path,
    ) -> list[DocumentChunk]:
        """从 JSON 恢复 Chunk。"""

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, list):
            raise ValueError(
                "chunks.json 必须是数组"
            )

        return [
            DocumentChunk(
                **item
            )
            for item in data
        ]

    # ==================================================
    # Cache Validation
    # ==================================================

    @staticmethod
    def _pipeline_matches(
        manifest: dict,
        expected: dict,
    ) -> bool:
        """判断旧向量是否属于同一个 Pipeline。"""

        for key, value in (
            expected.items()
        ):
            if (
                manifest.get(key)
                != value
            ):
                return False

        return True

    @staticmethod
    def _validate_source_layout(
        source_files: list[dict],
        chunk_count: int,
    ) -> None:
        """
        检查每个文件对应的 Chunk 范围。

        要求所有范围连续、无重叠、无缺口。
        """

        expected_start = 0
        seen_paths = set()

        for item in source_files:
            path = item.get("path")

            if (
                not isinstance(path, str)
                or not path
            ):
                raise ValueError(
                    "Manifest Source Path 非法"
                )

            if path in seen_paths:
                raise ValueError(
                    "Manifest 中存在重复 Source Path"
                )

            seen_paths.add(path)

            start = item.get(
                "chunk_start"
            )

            count = item.get(
                "chunk_count"
            )

            if (
                not isinstance(start, int)
                or not isinstance(count, int)
                or start < 0
                or count < 0
            ):
                raise ValueError(
                    "Manifest Chunk Range 非法"
                )

            if start != expected_start:
                raise ValueError(
                    "Manifest Chunk Range 不连续"
                )

            expected_start += count

        if expected_start != chunk_count:
            raise ValueError(
                "Manifest Chunk Range "
                "与总 Chunk 数不一致"
            )

    def _load_cache_bundle(
        self,
        pipeline_identity: dict,
    ):
        """
        加载完整旧缓存。

        注意：
        Source 文件变化不会在这里判为失效。

        因为 Source 变化正是 Incremental Update
        需要处理的情况。
        """

        if self.cache_dir is None:
            return None, "cache_disabled"

        manifest_path = (
            self._manifest_path()
        )

        chunks_path = (
            self._chunks_path()
        )

        embeddings_path = (
            self._embeddings_path()
        )

        if (
            not manifest_path.exists()
            or not chunks_path.exists()
            or not embeddings_path.exists()
        ):
            return None, "cache_missing"

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(
                manifest,
                dict,
            ):
                raise ValueError(
                    "manifest.json 非法"
                )

            # Chunk 参数 / 模型等变化：
            # 不能复用旧向量。
            if not self._pipeline_matches(
                manifest,
                pipeline_identity,
            ):
                return None, "cache_stale"

            # ------------------------------------------
            # chunks.json
            # ------------------------------------------

            expected = manifest.get(
                "chunks_file_sha256"
            )

            if not expected:
                raise ValueError(
                    "Manifest 缺少 chunks checksum"
                )

            if (
                _sha256_file(
                    chunks_path
                )
                != expected
            ):
                raise ValueError(
                    "chunks.json checksum 不匹配"
                )

            chunks = (
                self._load_chunks_from_cache(
                    chunks_path
                )
            )

            if (
                len(chunks)
                != manifest.get(
                    "chunk_count"
                )
            ):
                raise ValueError(
                    "Chunk 数量不一致"
                )

            # ------------------------------------------
            # embeddings.npy
            # ------------------------------------------

            expected = manifest.get(
                "embeddings_file_sha256"
            )

            if not expected:
                raise ValueError(
                    "Manifest 缺少 embeddings checksum"
                )

            if (
                _sha256_file(
                    embeddings_path
                )
                != expected
            ):
                raise ValueError(
                    "embeddings.npy checksum 不匹配"
                )

            embeddings = (
                DenseRetriever
                .load_embeddings(
                    embeddings_path
                )
            )

            if (
                embeddings.shape[0]
                != len(chunks)
            ):
                raise ValueError(
                    "Embedding 数量与 Chunk 数量不一致"
                )

            # ------------------------------------------
            # Source Layout
            # ------------------------------------------

            source_files = manifest.get(
                "source_files"
            )

            if not isinstance(
                source_files,
                list,
            ):
                raise ValueError(
                    "Manifest 缺少 source_files"
                )

            self._validate_source_layout(
                source_files,
                len(chunks),
            )

            # ------------------------------------------
            # dense.faiss
            # ------------------------------------------

            dense_index = None

            if chunks:
                index_path = (
                    self._dense_index_path()
                )

                if not index_path.exists():
                    raise ValueError(
                        "缓存缺少 dense.faiss"
                    )

                expected = manifest.get(
                    "dense_index_file_sha256"
                )

                if not expected:
                    raise ValueError(
                        "Manifest 缺少 FAISS checksum"
                    )

                if (
                    _sha256_file(
                        index_path
                    )
                    != expected
                ):
                    raise ValueError(
                        "dense.faiss checksum 不匹配"
                    )

                dense_index = (
                    DenseRetriever
                    .load_index(
                        index_path
                    )
                )

                if (
                    int(
                        dense_index.ntotal
                    )
                    != len(chunks)
                ):
                    raise ValueError(
                        "FAISS 向量数量不一致"
                    )

                if (
                    int(
                        dense_index.d
                    )
                    != embeddings.shape[1]
                ):
                    raise ValueError(
                        "FAISS / Embedding 维度不一致"
                    )

            return {
                "manifest": manifest,
                "chunks": chunks,
                "embeddings": embeddings,
                "dense_index": dense_index,
            }, "valid_cache"

        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            return (
                None,
                f"cache_corrupt: {exc}",
            )

    # ==================================================
    # Runtime Restore
    # ==================================================

    def _restore_cache_hit(
        self,
        bundle,
        current_sources,
    ) -> None:
        """完整 Cache Hit。"""

        self.chunks = (
            bundle["chunks"]
        )

        self.source_files = (
            bundle["manifest"][
                "source_files"
            ]
        )

        self.retriever = (
            HybridRetriever(
                self.chunks,
                dense_encoder=(
                    self.dense_encoder
                ),
                reranker_model=(
                    self.reranker_model
                ),
                dense_model_name=(
                    self.embedding_model_name
                ),
                dense_index=(
                    bundle[
                        "dense_index"
                    ]
                ),
                dense_embeddings=(
                    bundle[
                        "embeddings"
                    ]
                ),
            )
        )

        self.index_update_mode = (
            "cache_hit"
        )

        self.incremental_stats = {
            "mode": "cache_hit",
            "unchanged_files": (
                len(current_sources)
            ),
            "added_files": 0,
            "modified_files": 0,
            "deleted_files": 0,
            "reused_chunks": (
                len(self.chunks)
            ),
            "encoded_chunks": 0,
        }

    # ==================================================
    # Full Build
    # ==================================================

    def _full_rebuild(
        self,
        file_paths,
        source_descriptions,
    ) -> None:
        """重新处理整个知识库。"""

        chunks = []
        source_layout = []

        for file_path, description in zip(
            file_paths,
            source_descriptions,
        ):
            start = len(chunks)

            document = load_document(
                file_path
            )

            file_chunks = (
                chunk_document(
                    document,
                    chunk_size=(
                        self.chunk_size
                    ),
                    overlap=(
                        self.overlap
                    ),
                )
            )

            chunks.extend(
                file_chunks
            )

            source_layout.append(
                {
                    **description,
                    "chunk_start": start,
                    "chunk_count": (
                        len(file_chunks)
                    ),
                }
            )

        self.chunks = chunks
        self.source_files = source_layout

        self.retriever = (
            HybridRetriever(
                self.chunks,
                dense_encoder=(
                    self.dense_encoder
                ),
                reranker_model=(
                    self.reranker_model
                ),
                dense_model_name=(
                    self.embedding_model_name
                ),
            )
        )

        self.index_update_mode = (
            "full_rebuild"
        )

        self.incremental_stats = {
            "mode": "full_rebuild",
            "total_files": (
                len(source_descriptions)
            ),
            "reused_chunks": 0,
            "encoded_chunks": (
                len(chunks)
            ),
        }

    # ==================================================
    # Incremental Update
    # ==================================================

    def _incremental_rebuild(
        self,
        file_paths,
        source_descriptions,
        bundle,
    ) -> None:
        """
        只重新处理新增 / 修改文件。

        未变化文件直接复用：
            old_chunks
            old_embeddings
        """

        old_manifest = (
            bundle["manifest"]
        )

        old_chunks = (
            bundle["chunks"]
        )

        old_embeddings = (
            bundle["embeddings"]
        )

        old_sources = (
            old_manifest[
                "source_files"
            ]
        )

        old_by_path = {
            item["path"]: item
            for item in old_sources
        }

        new_by_path = {
            item["path"]: item
            for item
            in source_descriptions
        }

        old_paths = set(
            old_by_path
        )

        new_paths = set(
            new_by_path
        )

        added = (
            new_paths
            - old_paths
        )

        deleted = (
            old_paths
            - new_paths
        )

        common = (
            new_paths
            & old_paths
        )

        modified = {
            path
            for path in common
            if (
                old_by_path[path][
                    "sha256"
                ]
                != new_by_path[path][
                    "sha256"
                ]
            )
        }

        unchanged = (
            common
            - modified
        )

        path_map = {
            path.relative_to(
                self.data_dir
            ).as_posix(): path
            for path in file_paths
        }

        new_chunks = []
        embedding_parts = []
        source_layout = []

        reused_chunks = 0
        encoded_chunks = 0

        old_dimension = (
            old_embeddings.shape[1]
            if old_embeddings.ndim == 2
            else 0
        )

        current_dimension = (
            old_dimension
        )

        # 只创建一次 Encoder Helper。
        # 多个变化文件共享同一个模型实例。
        encoder_helper = DenseRetriever(
            [],
            model_name=(
                self.embedding_model_name
            ),
            encoder=(
                self.dense_encoder
            ),
        )

        for description in (
            source_descriptions
        ):
            relative_path = (
                description["path"]
            )

            new_start = len(
                new_chunks
            )

            # ------------------------------------------
            # Unchanged
            # ------------------------------------------

            if relative_path in unchanged:
                old_entry = (
                    old_by_path[
                        relative_path
                    ]
                )

                old_start = (
                    old_entry[
                        "chunk_start"
                    ]
                )

                old_count = (
                    old_entry[
                        "chunk_count"
                    ]
                )

                file_chunks = (
                    old_chunks[
                        old_start:
                        old_start
                        + old_count
                    ]
                )

                file_embeddings = (
                    old_embeddings[
                        old_start:
                        old_start
                        + old_count
                    ]
                )

                new_chunks.extend(
                    file_chunks
                )

                if old_count:
                    embedding_parts.append(
                        file_embeddings
                    )

                reused_chunks += (
                    old_count
                )

            # ------------------------------------------
            # Added / Modified
            # ------------------------------------------

            else:
                file_path = (
                    path_map[
                        relative_path
                    ]
                )

                document = (
                    load_document(
                        file_path
                    )
                )

                file_chunks = (
                    chunk_document(
                        document,
                        chunk_size=(
                            self.chunk_size
                        ),
                        overlap=(
                            self.overlap
                        ),
                    )
                )

                new_chunks.extend(
                    file_chunks
                )

                if file_chunks:
                    file_embeddings = (
                        encoder_helper
                        .encode_chunks(
                            file_chunks
                        )
                    )

                    dimension = (
                        file_embeddings
                        .shape[1]
                    )

                    if (
                        current_dimension
                        not in (
                            0,
                            dimension,
                        )
                    ):
                        raise ValueError(
                            "新增 Embedding "
                            "与旧 Embedding 维度不一致"
                        )

                    current_dimension = (
                        dimension
                    )

                    embedding_parts.append(
                        file_embeddings
                    )

                encoded_chunks += (
                    len(file_chunks)
                )

            source_layout.append(
                {
                    **description,
                    "chunk_start": (
                        new_start
                    ),
                    "chunk_count": (
                        len(file_chunks)
                    ),
                }
            )

        # ----------------------------------------------
        # Merge Embeddings
        # ----------------------------------------------

        if embedding_parts:
            embeddings = (
                np.ascontiguousarray(
                    np.concatenate(
                        embedding_parts,
                        axis=0,
                    )
                )
            )

        else:
            embeddings = np.empty(
                (
                    0,
                    current_dimension,
                ),
                dtype=np.float32,
            )

        if (
            embeddings.shape[0]
            != len(new_chunks)
        ):
            raise ValueError(
                "增量合并后 Embedding "
                "与 Chunk 数量不一致"
            )

        # ----------------------------------------------
        # Rebuild Cheap FAISS
        #
        # 注意：
        # 这里重建的是 IndexFlatIP 数据结构，
        # 没有重新跑旧文档的神经网络 Embedding。
        # ----------------------------------------------

        self.chunks = (
            new_chunks
        )

        self.source_files = (
            source_layout
        )

        self.retriever = (
            HybridRetriever(
                self.chunks,
                dense_encoder=(
                    self.dense_encoder
                ),
                reranker_model=(
                    self.reranker_model
                ),
                dense_model_name=(
                    self.embedding_model_name
                ),
                dense_embeddings=(
                    embeddings
                ),
            )
        )

        self.index_update_mode = (
            "incremental_update"
        )

        self.incremental_stats = {
            "mode": (
                "incremental_update"
            ),
            "unchanged_files": (
                len(unchanged)
            ),
            "added_files": (
                len(added)
            ),
            "modified_files": (
                len(modified)
            ),
            "deleted_files": (
                len(deleted)
            ),
            "reused_chunks": (
                reused_chunks
            ),
            "encoded_chunks": (
                encoded_chunks
            ),
        }

    # ==================================================
    # Cache Save
    # ==================================================

    def _save_cache(
        self,
        pipeline_identity,
    ) -> None:
        """保存 Incremental Cache v2。"""

        if self.cache_dir is None:
            return

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunks_path = (
            self._chunks_path()
        )

        embeddings_path = (
            self._embeddings_path()
        )

        index_path = (
            self._dense_index_path()
        )

        # ----------------------------------------------
        # chunks.json
        # ----------------------------------------------

        chunks_data = [
            asdict(chunk)
            for chunk in self.chunks
        ]

        chunks_tmp = (
            chunks_path.with_name(
                chunks_path.name
                + ".tmp"
            )
        )

        chunks_tmp.write_text(
            json.dumps(
                chunks_data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        chunks_hash = (
            _sha256_file(
                chunks_tmp
            )
        )

        os.replace(
            chunks_tmp,
            chunks_path,
        )

        # ----------------------------------------------
        # embeddings.npy
        # ----------------------------------------------

        dense = (
            self.retriever
            .dense_retriever
        )

        embeddings = (
            dense.embeddings
        )

        if embeddings is None:
            embeddings = np.empty(
                (0, 0),
                dtype=np.float32,
            )

        embeddings_tmp = (
            embeddings_path.with_name(
                embeddings_path.name
                + ".tmp"
            )
        )

        DenseRetriever.save_embeddings(
            embeddings_tmp,
            embeddings,
        )

        embeddings_hash = (
            _sha256_file(
                embeddings_tmp
            )
        )

        os.replace(
            embeddings_tmp,
            embeddings_path,
        )

        # ----------------------------------------------
        # dense.faiss
        # ----------------------------------------------

        dense_index_hash = None

        if dense.index is not None:
            index_tmp = (
                index_path.with_name(
                    index_path.name
                    + ".tmp"
                )
            )

            if index_tmp.exists():
                index_tmp.unlink()

            dense.save_index(
                index_tmp
            )

            dense_index_hash = (
                _sha256_file(
                    index_tmp
                )
            )

            os.replace(
                index_tmp,
                index_path,
            )

        elif index_path.exists():
            index_path.unlink()

        # ----------------------------------------------
        # manifest.json
        #
        # 最后写 Manifest，
        # 相当于提交整个 Cache。
        # ----------------------------------------------

        source_signature = (
            self._source_signature(
                self.source_files
            )
        )

        manifest = {
            **pipeline_identity,

            "source_fingerprint": (
                _sha256_json(
                    source_signature
                )
            ),

            "source_files": (
                self.source_files
            ),

            "chunk_count": (
                len(self.chunks)
            ),

            "embedding_dimension": (
                embeddings.shape[1]
            ),

            "chunks_file_sha256": (
                chunks_hash
            ),

            "embeddings_file_sha256": (
                embeddings_hash
            ),

            "dense_index_file_sha256": (
                dense_index_hash
            ),

            "created_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        self._write_json_atomic(
            self._manifest_path(),
            manifest,
        )

    # ==================================================
    # Public Build
    # ==================================================

    def build(
        self,
        force_rebuild: bool = False,
    ) -> None:
        """建立、加载或增量更新索引。"""

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"知识库目录不存在: "
                f"{self.data_dir}"
            )

        if not self.data_dir.is_dir():
            raise ValueError(
                f"知识库路径不是目录: "
                f"{self.data_dir}"
            )

        file_paths = (
            self._list_source_files()
        )

        source_descriptions = (
            self._describe_source_files(
                file_paths
            )
        )

        pipeline_identity = (
            self._build_pipeline_identity()
        )

        cache_reason = (
            "forced_rebuild"
            if force_rebuild
            else None
        )

        # ==================================================
        # Try Persistent / Incremental
        # ==================================================

        if (
            self.cache_dir is not None
            and not force_rebuild
        ):
            bundle, cache_reason = (
                self._load_cache_bundle(
                    pipeline_identity
                )
            )

            if bundle is not None:
                old_sources = (
                    bundle["manifest"][
                        "source_files"
                    ]
                )

                old_signature = (
                    self._source_signature(
                        old_sources
                    )
                )

                new_signature = (
                    self._source_signature(
                        source_descriptions
                    )
                )

                # --------------------------------------
                # 完全没变化
                # --------------------------------------

                if (
                    old_signature
                    == new_signature
                ):
                    self._restore_cache_hit(
                        bundle,
                        source_descriptions,
                    )

                    self.cache_status = (
                        "hit"
                    )

                    self.cache_reason = (
                        "valid_cache"
                    )

                    return

                # --------------------------------------
                # Source 发生变化
                # → Incremental Update
                # --------------------------------------

                try:
                    self._incremental_rebuild(
                        file_paths,
                        source_descriptions,
                        bundle,
                    )

                    self._save_cache(
                        pipeline_identity
                    )

                    # 保留原有 status 语义：
                    # 索引确实发生了重新生成。
                    self.cache_status = (
                        "rebuilt"
                    )

                    self.cache_reason = (
                        "cache_stale"
                    )

                    return

                except Exception as exc:
                    # Incremental 本身失败时，
                    # 不能让知识库不可用。
                    #
                    # 自动降级为 Full Rebuild。
                    incremental_error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    cache_reason = (
                        "cache_stale"
                    )

                    self._full_rebuild(
                        file_paths,
                        source_descriptions,
                    )

                    self.incremental_stats[
                        "incremental_fallback"
                    ] = incremental_error

                    self.index_update_mode = (
                        "full_rebuild"
                    )

                    try:
                        self._save_cache(
                            pipeline_identity
                        )

                        self.cache_status = (
                            "rebuilt"
                        )

                        self.cache_reason = (
                            cache_reason
                        )

                    except Exception as save_exc:
                        self.cache_status = (
                            "save_failed"
                        )

                        self.cache_reason = (
                            "cache_save_failed: "
                            f"{save_exc}"
                        )

                    return

        # ==================================================
        # Full Rebuild
        #
        # 情况：
        # - 第一次运行
        # - v1 Cache
        # - Pipeline 参数变化
        # - Cache 损坏
        # - force_rebuild
        # ==================================================

        self._full_rebuild(
            file_paths,
            source_descriptions,
        )

        if self.cache_dir is None:
            self.cache_status = (
                "disabled"
            )

            self.cache_reason = (
                "cache_disabled"
            )

            return

        try:
            self._save_cache(
                pipeline_identity
            )

            self.cache_status = (
                "rebuilt"
            )

            self.cache_reason = (
                cache_reason
                or "cache_missing"
            )

        except Exception as exc:
            self.cache_status = (
                "save_failed"
            )

            self.cache_reason = (
                f"cache_save_failed: {exc}"
            )

    # ==================================================
    # Search
    # ==================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievalResult]:
        """使用已经建立的索引搜索。"""

        if self.retriever is None:
            raise RuntimeError(
                "RetrievalEngine 尚未 build()，"
                "请先建立知识库索引。"
            )

        return self.retriever.search(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
        )