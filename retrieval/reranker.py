import numpy as np
from sentence_transformers import CrossEncoder

from retrieval.result import RetrievalResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """使用 Cross-Encoder 对候选检索结果进行精排。"""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model=None,
    ):
        self.model_name = model_name
        self.model = model

    def _get_model(self):
        """第一次真正 rerank 时再加载模型。"""
        if self.model is None:
            self.model = CrossEncoder(self.model_name)
        return self.model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """根据 query 对候选结果重新排序。"""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not query.strip() or not candidates:
            return []

        pairs = [
            [query, result.chunk.content]
            for result in candidates
        ]

        scores = self._get_model().predict(
            pairs,
            show_progress_bar=False,
        )

        scores = np.asarray(
            scores,
            dtype=float,
        ).reshape(-1)

        if len(scores) != len(candidates):
            raise ValueError(
                "Reranker 返回的 score 数量与候选数量不一致"
            )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        reranked_results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):
            reranked_results.append(
                RetrievalResult(
                    chunk=candidates[index].chunk,
                    score=float(scores[index]),
                    rank=rank,
                )
            )

        return reranked_results
