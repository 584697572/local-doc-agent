from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from retrieval.result import RetrievalResult
from retrieval.document import DocumentChunk


class HybridRetriever:
    """
    Hybrid Retrieval 总入口。

    检索流程：

        Query
          ↓
        BM25 + Dense
          ↓
        RRF Fusion
          ↓
        Reranker
          ↓
        Final Top-K
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        dense_encoder=None,
        reranker_model=None,
    ):
        # 保存原始 Chunk
        self.chunks = list(chunks)

        # Sparse / lexical retrieval
        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

        # Dense / semantic retrieval
        self.dense_retriever = DenseRetriever(
            self.chunks,
            encoder=dense_encoder,
        )

        # Cross-Encoder reranking
        self.reranker = Reranker(
            model=reranker_model,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        执行完整 Hybrid Retrieval。

        参数：
            query:
                用户查询。

            top_k:
                最终 Reranker 返回多少条结果。

            candidate_k:
                BM25 / Dense / RRF 阶段保留多少候选，
                再交给 Reranker 精排。

        返回：
            最终按 Reranker score 排序的 RetrievalResult。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if candidate_k <= 0:
            raise ValueError("candidate_k 必须大于 0")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k 必须大于等于 top_k"
            )

        if not query.strip():
            return []

        if not self.chunks:
            return []

        # 第一条召回通道：BM25
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=candidate_k,
        )

        # 第二条召回通道：Dense
        dense_results = self.dense_retriever.search(
            query,
            top_k=candidate_k,
        )

        # Rank-level Fusion
        fused_results = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k=candidate_k,
        )

        if not fused_results:
            return []

        # 最终 Cross-Encoder 精排
        final_results = self.reranker.rerank(
            query,
            fused_results,
            top_k=top_k,
        )

        return final_results