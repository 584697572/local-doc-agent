import numpy as np
from sentence_transformers import CrossEncoder

from retrieval.result import RetrievalResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """
    使用 Cross-Encoder 对候选检索结果进行精排。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model=None,
    ):
        self.model_name = model_name
        self.model = model

        # 正常运行时加载真实模型。
        # 测试时可以注入 FakeReranker，避免下载和加载模型。
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        根据 query 对候选 DocumentChunk 重新排序。

        参数：
            query:
                用户查询。

            candidates:
                BM25 / Dense / RRF 等阶段产生的候选结果。

            top_k:
                最终最多返回多少条结果。

        返回：
            按 Reranker score 从高到低排列的 RetrievalResult。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not query.strip():
            return []

        if not candidates:
            return []

        # Cross-Encoder 不是分别编码 Query 和 Chunk，
        # 而是把二者组成 pair 一起输入模型。
        pairs = [
            [query, result.chunk.content]
            for result in candidates
        ]

        # 每一个 (query, chunk) pair 得到一个相关性分数。
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        # 转成一维 NumPy 数组，方便统一排序。
        scores = np.asarray(
            scores,
            dtype=float,
        ).reshape(-1)

        if len(scores) != len(candidates):
            raise ValueError(
                "Reranker 返回的 score 数量与候选数量不一致"
            )

        # 根据 reranker score 从高到低排序候选位置。
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