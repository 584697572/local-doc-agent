from retrieval.bm25 import BM25Retriever
from retrieval.dense import (
    DEFAULT_EMBEDDING_MODEL,
    DenseRetriever,
)
from retrieval.fusion import (
    DEFAULT_RRF_K,
    rank_preserving_rerank_fusion,
    reciprocal_rank_fusion,
)
from retrieval.reranker import Reranker
from retrieval.result import RetrievalResult
from retrieval.document import DocumentChunk


DEFAULT_SAFE_FUSION_K = DEFAULT_RRF_K
DEFAULT_SAFE_RRF_WEIGHT = 1.0
DEFAULT_SAFE_RERANKER_WEIGHT = 2.0


class HybridRetriever:
    """
    Hybrid Retrieval：

        BM25 + Dense
            ↓
        RRF
            ↓
        Cross-Encoder
            ↓
        Safe Rank Fusion

    Dense 部分可以直接接收：
        - Persistent FAISS
        - Persistent Embeddings
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        dense_encoder=None,
        reranker_model=None,
        dense_model_name: str = (
            DEFAULT_EMBEDDING_MODEL
        ),
        dense_index=None,
        dense_embeddings=None,
        safe_fusion_k: int = (
            DEFAULT_SAFE_FUSION_K
        ),
        safe_rrf_weight: float = (
            DEFAULT_SAFE_RRF_WEIGHT
        ),
        safe_reranker_weight: float = (
            DEFAULT_SAFE_RERANKER_WEIGHT
        ),
    ):
        self.chunks = list(chunks)

        self.bm25_retriever = (
            BM25Retriever(
                self.chunks
            )
        )

        self.dense_retriever = (
            DenseRetriever(
                self.chunks,
                model_name=(
                    dense_model_name
                ),
                encoder=(
                    dense_encoder
                ),
                index=(
                    dense_index
                ),
                embeddings=(
                    dense_embeddings
                ),
            )
        )

        self.reranker = Reranker(
            model=reranker_model,
        )

        self.safe_fusion_k = (
            safe_fusion_k
        )

        self.safe_rrf_weight = (
            safe_rrf_weight
        )

        self.safe_reranker_weight = (
            safe_reranker_weight
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievalResult]:
        """执行完整 Retrieval v1。"""

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k 必须大于 0"
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k 必须大于等于 top_k"
            )

        if not query.strip():
            return []

        if not self.chunks:
            return []

        # Stage 1
        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=candidate_k,
            )
        )

        # Stage 2
        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=candidate_k,
            )
        )

        # Stage 3
        fused_results = (
            reciprocal_rank_fusion(
                bm25_results,
                dense_results,
                top_k=candidate_k,
            )
        )

        if not fused_results:
            return []

        # Stage 4
        reranked_results = (
            self.reranker.rerank(
                query=query,
                candidates=fused_results,
                top_k=len(
                    fused_results
                ),
            )
        )

        # Stage 5
        return (
            rank_preserving_rerank_fusion(
                rrf_results=(
                    fused_results
                ),
                reranked_results=(
                    reranked_results
                ),
                top_k=top_k,
                fusion_k=(
                    self.safe_fusion_k
                ),
                rrf_weight=(
                    self.safe_rrf_weight
                ),
                reranker_weight=(
                    self.safe_reranker_weight
                ),
            )
        )