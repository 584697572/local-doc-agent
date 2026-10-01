from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.fusion import (
    DEFAULT_RRF_K,
    rank_preserving_rerank_fusion,
    reciprocal_rank_fusion,
)
from retrieval.reranker import Reranker
from retrieval.result import RetrievalResult
from retrieval.document import DocumentChunk


# ======================================================
# Runtime Retrieval v1 配置
#
# 这组参数来自 SciFact 300-query 全量实验：
#
# 普通 Reranker Hit@5:
#     79.67%
#
# Safe Rerank 1:2 Hit@5:
#     81.67%
#
# 因此当前 Runtime v1 固定采用：
#
#     RRF weight      = 1.0
#     Reranker weight = 2.0
#
# 暂时不继续在 test set 上调参，避免过拟合。
# ======================================================

DEFAULT_SAFE_FUSION_K = DEFAULT_RRF_K
DEFAULT_SAFE_RRF_WEIGHT = 1.0
DEFAULT_SAFE_RERANKER_WEIGHT = 2.0


class HybridRetriever:
    """
    Hybrid Retrieval 总入口。

    当前 Retrieval v1 流程：

        Query
          ↓
        BM25 + Dense
          ↓
        RRF Fusion
          ↓
        Candidate Top-K
          ↓
        Cross-Encoder Reranker
          ↓
        Safe Rank Fusion
        (RRF : Reranker = 1 : 2)
          ↓
        Final Top-K

    Safe Rank Fusion 的目的不是替代 Reranker，
    而是在保留 Cross-Encoder 语义判断能力的同时，
    减少 Reranker 对原始优质 Retrieval 排名的误伤。
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        dense_encoder=None,
        reranker_model=None,
        safe_fusion_k: int = DEFAULT_SAFE_FUSION_K,
        safe_rrf_weight: float = DEFAULT_SAFE_RRF_WEIGHT,
        safe_reranker_weight: float = DEFAULT_SAFE_RERANKER_WEIGHT,
    ):
        # 保存原始 Chunk。
        self.chunks = list(
            chunks
        )

        # Sparse / lexical retrieval。
        self.bm25_retriever = (
            BM25Retriever(
                self.chunks
            )
        )

        # Dense / semantic retrieval。
        self.dense_retriever = (
            DenseRetriever(
                self.chunks,
                encoder=dense_encoder,
            )
        )

        # Cross-Encoder reranking。
        self.reranker = Reranker(
            model=reranker_model,
        )

        # Safe Rank Fusion 参数。
        #
        # 当前默认值来自已经完成的
        # 300-query controlled experiment。
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
        """
        执行完整 Retrieval v1。

        参数：
            query:
                用户查询。

            top_k:
                最终返回多少条结果。

            candidate_k:
                BM25 / Dense / RRF 阶段保留多少候选，
                再交给 Cross-Encoder。

        返回：
            Safe Rank Fusion 后的最终 RetrievalResult。
        """

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

        # ==================================================
        # Stage 1: BM25
        # ==================================================

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=candidate_k,
            )
        )

        # ==================================================
        # Stage 2: Dense Retrieval
        # ==================================================

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=candidate_k,
            )
        )

        # ==================================================
        # Stage 3: RRF
        #
        # 得到第一份 Top-N 排名。
        # 这份排名后面不会被直接丢弃，
        # Safe Fusion 还会再次使用它。
        # ==================================================

        fused_results = (
            reciprocal_rank_fusion(
                bm25_results,
                dense_results,
                top_k=candidate_k,
            )
        )

        if not fused_results:
            return []

        # ==================================================
        # Stage 4: Cross-Encoder Reranker
        #
        # 注意：
        #
        # 这里必须返回完整 candidate_k 排名，
        # 不能像以前一样直接 top_k=5。
        #
        # 因为 Safe Fusion 需要知道
        # 每一个候选在 Reranker 中的 rank。
        #
        # 这不会增加 CrossEncoder 的模型推理次数。
        # 原来的 Reranker 本来就会对所有 candidates
        # 计算 score，只是最后提前截成 Top-5。
        # ==================================================

        reranked_results = (
            self.reranker.rerank(
                query=query,
                candidates=fused_results,
                top_k=len(
                    fused_results
                ),
            )
        )

        # ==================================================
        # Stage 5: Safe Rank Fusion
        #
        # FinalScore(d)
        #
        # =
        #
        # 1 / (k + RRF_rank)
        #
        # +
        #
        # 2 / (k + Reranker_rank)
        #
        # 默认权重：
        #
        #     RRF       = 1
        #     Reranker  = 2
        #
        # 最终才截取真正的 Top-K。
        # ==================================================

        final_results = (
            rank_preserving_rerank_fusion(
                rrf_results=fused_results,

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

        return final_results