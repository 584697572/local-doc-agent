"""
Dense Retrieval。

支持：
1. 全量计算 Chunk Embedding；
2. Persistent FAISS Index；
3. Persistent Embedding Matrix；
4. Lazy Load SentenceTransformer；
5. 从已有 Embedding 快速重建 FAISS。
"""

from pathlib import Path

import faiss
import numpy as np

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class DenseRetriever:
    """
    基于 Embedding + FAISS 的语义检索器。

    chunks 与 embeddings 按位置严格对应：

        chunks[0] <-> embeddings[0]
        chunks[1] <-> embeddings[1]
        ...
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        encoder=None,
        index=None,
        embeddings=None,
    ):
        self.chunks = list(chunks)
        self.model_name = model_name
        self.encoder = encoder
        self.index = index

        self.embeddings = None

        if embeddings is not None:
            self.embeddings = self._prepare_embeddings(
                embeddings
            )

        # 空知识库。
        if not self.chunks:
            if (
                self.index is not None
                and int(self.index.ntotal) != 0
            ):
                raise ValueError(
                    "空 Chunk 列表不能绑定非空 FAISS Index"
                )

            if (
                self.embeddings is not None
                and self.embeddings.shape[0] != 0
            ):
                raise ValueError(
                    "空 Chunk 列表不能绑定非空 Embedding Matrix"
                )

            return

        # Embedding 数量必须和 Chunk 一致。
        if (
            self.embeddings is not None
            and self.embeddings.shape[0]
            != len(self.chunks)
        ):
            raise ValueError(
                "Embedding 数量与 Chunk 数量不一致"
            )

        # 已经有 FAISS Index。
        if self.index is not None:
            if (
                int(self.index.ntotal)
                != len(self.chunks)
            ):
                raise ValueError(
                    "FAISS Index 中的向量数量"
                    "与 Chunk 数量不一致"
                )

            if (
                self.embeddings is not None
                and self.embeddings.shape[1]
                != int(self.index.d)
            ):
                raise ValueError(
                    "Embedding 维度与 FAISS Index 不一致"
                )

            return

        # 已经有 Embedding，但没有 FAISS。
        #
        # Incremental Index 会走这条路径：
        #
        # 旧 Embedding + 新 Embedding
        # → 快速重建 FAISS。
        if self.embeddings is not None:
            self.index = self._create_index(
                self.embeddings
            )
            return

        # 什么缓存都没有：
        # 全量计算 Embedding。
        self._build_index()

    @staticmethod
    def _prepare_embeddings(
        embeddings,
    ) -> np.ndarray:
        """统一为连续的 float32 二维数组。"""

        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "Embedding 必须是二维数组"
            )

        return np.ascontiguousarray(
            array
        )

    def _get_encoder(self):
        """
        Lazy Load。

        Cache Hit 时不会 import sentence_transformers。
        """

        if self.encoder is None:
            from sentence_transformers import (
                SentenceTransformer,
            )

            self.encoder = SentenceTransformer(
                self.model_name
            )

        return self.encoder

    def encode_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> np.ndarray:
        """
        只计算指定 Chunk。

        Incremental Update 会只把发生变化的
        Chunk 传到这里。
        """

        chunks = list(chunks)

        if not chunks:
            dimension = 0

            if self.embeddings is not None:
                dimension = (
                    self.embeddings.shape[1]
                )

            elif self.index is not None:
                dimension = int(
                    self.index.d
                )

            return np.empty(
                (0, dimension),
                dtype=np.float32,
            )

        encoder = self._get_encoder()

        texts = [
            chunk.content
            for chunk in chunks
        ]

        embeddings = encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = self._prepare_embeddings(
            embeddings
        )

        if (
            embeddings.shape[0]
            != len(chunks)
        ):
            raise ValueError(
                "Embedding 数量与 Chunk 数量不一致"
            )

        return embeddings

    @staticmethod
    def _create_index(
        embeddings: np.ndarray,
    ):
        """
        从已经算好的 Embedding
        快速创建 FAISS。

        这里没有神经网络推理。
        """

        embeddings = (
            DenseRetriever
            ._prepare_embeddings(
                embeddings
            )
        )

        if embeddings.shape[0] == 0:
            return None

        if embeddings.shape[1] <= 0:
            raise ValueError(
                "Embedding 维度必须大于 0"
            )

        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(
            embeddings
        )

        return index

    def _build_index(self) -> None:
        """全量建立 Dense Index。"""

        if not self.chunks:
            self.embeddings = None
            self.index = None
            return

        self.embeddings = self.encode_chunks(
            self.chunks
        )

        self.index = self._create_index(
            self.embeddings
        )

    # ==================================================
    # FAISS Persistence
    # ==================================================

    def save_index(
        self,
        path: str | Path,
    ) -> None:
        """保存 FAISS，文件 I/O 由 Python 处理。"""

        if self.index is None:
            raise RuntimeError(
                "当前没有可保存的 FAISS Index"
            )

        path = Path(path)

        serialized = faiss.serialize_index(
            self.index
        )

        serialized = np.asarray(
            serialized,
            dtype=np.uint8,
        )

        path.write_bytes(
            serialized.tobytes()
        )

    @staticmethod
    def load_index(
        path: str | Path,
    ):
        """加载 FAISS Index。"""

        path = Path(path)

        raw_bytes = path.read_bytes()

        if not raw_bytes:
            raise ValueError(
                "FAISS Index 文件为空"
            )

        serialized = np.frombuffer(
            raw_bytes,
            dtype=np.uint8,
        ).copy()

        return faiss.deserialize_index(
            serialized
        )

    # ==================================================
    # Embedding Persistence
    # ==================================================

    @staticmethod
    def save_embeddings(
        path: str | Path,
        embeddings: np.ndarray,
    ) -> None:
        """
        保存原始向量矩阵。

        使用 .npy；
        禁止 Pickle。
        """

        path = Path(path)

        embeddings = (
            DenseRetriever
            ._prepare_embeddings(
                embeddings
            )
        )

        with path.open("wb") as file:
            np.save(
                file,
                embeddings,
                allow_pickle=False,
            )

    @staticmethod
    def load_embeddings(
        path: str | Path,
    ) -> np.ndarray:
        """加载原始向量矩阵。"""

        path = Path(path)

        with path.open("rb") as file:
            embeddings = np.load(
                file,
                allow_pickle=False,
            )

        return (
            DenseRetriever
            ._prepare_embeddings(
                embeddings
            )
        )

    # ==================================================
    # Search
    # ==================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """执行 Dense Retrieval。"""

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        if not query.strip():
            return []

        if self.index is None:
            return []

        encoder = self._get_encoder()

        query_embedding = encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_embedding = (
            self._prepare_embeddings(
                query_embedding
            )
        )

        k = min(
            top_k,
            len(self.chunks),
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                k,
            )
        )

        results = []

        for rank, (
            score,
            index,
        ) in enumerate(
            zip(
                scores[0],
                indices[0],
            ),
            start=1,
        ):
            if index < 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[
                        index
                    ],
                    score=float(score),
                    rank=rank,
                )
            )

        return results