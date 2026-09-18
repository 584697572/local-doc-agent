import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class DenseRetriever:
    """
    基于 Embedding + FAISS 的语义检索器。
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        encoder=None,
    ):
        # 保存原始 chunk。
        # FAISS 最后返回的是向量位置，
        # 我们需要根据位置重新找到对应的 DocumentChunk。
        self.chunks = list(chunks)

        self.model_name = model_name
        self.encoder = encoder
        self.index = None

        # 没有任何 chunk 时，不需要加载模型或建立索引。
        if not self.chunks:
            return

        # 正常运行时加载真正的 SentenceTransformer。
        # 测试时可以传入假的 encoder，避免下载真实模型。
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_name)

        texts = [
            chunk.content
            for chunk in self.chunks
        ]

        # 把所有 chunk 文本转换成向量。
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # FAISS 使用 float32。
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # 保证内存连续，方便 FAISS 处理。
        embeddings = np.ascontiguousarray(embeddings)

        # embeddings 的形状类似：
        #
        # (chunk数量, embedding维度)
        #
        # 例如：
        # (100, 512)
        dimension = embeddings.shape[1]

        # Inner Product 索引。
        #
        # 因为上面已经 normalize_embeddings=True，
        # 所以内积就等价于 cosine similarity。
        self.index = faiss.IndexFlatIP(dimension)

        # 把所有 chunk 向量加入 FAISS。
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        对 Query 进行语义检索，
        返回最相似的 Top-K DocumentChunk。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not query.strip():
            return []

        if self.index is None:
            return []

        # Query 也必须使用和 DocumentChunk 相同的
        # Embedding 模型和归一化方式。
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        query_embedding = np.ascontiguousarray(
            query_embedding
        )

        # 如果 top_k 比整个语料库还大，
        # 最多只能返回现有 chunk 数量。
        k = min(top_k, len(self.chunks))

        # FAISS 搜索。
        #
        # scores:
        #     每个结果的相似度
        #
        # indices:
        #     每个结果对应原始 chunks 的位置
        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        # Query 只有一条，所以取 [0]。
        for rank, (score, index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            # FAISS 某些情况下可能用 -1 表示无结果。
            if index < 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=float(score),
                    rank=rank,
                )
            )

        return results