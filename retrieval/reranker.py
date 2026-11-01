"""
Cross-Encoder Reranker。

模型使用 Lazy Load：

    import retrieval.reranker
        → 不加载 sentence_transformers

    第一次真正调用 rerank()
        → 才加载 CrossEncoder

这样 Persistent Index Cache Hit
不会因为 Reranker 的存在而拖慢启动。
"""

import numpy as np

from retrieval.result import RetrievalResult

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """
    使用 Cross-Encoder
    对候选检索结果进行精排。
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, model=None):
        self.model_name = model_name

        # 测试时可以注入 FakeReranker。
        self.model = model

    def _get_model(self):
        """
        第一次真正 Rerank 时
        才加载 CrossEncoder。

        sentence_transformers 使用 Lazy Import，
        避免拖慢：
            import retrieval.engine
            Persistent Cache Hit
        """

        if self.model is None:
            # 重型依赖只在真正需要时导入。
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)

        return self.model

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        根据 Query 对候选结果重新排序。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not query.strip() or not candidates:
            return []

        # Cross-Encoder 每次比较一对文本：问题和一个候选片段。
        pairs = []
        for result in candidates:
            text_pair = [query, result.chunk.content]
            pairs.append(text_pair)

        model = self._get_model()
        scores = model.predict(pairs, show_progress_bar=False)

        scores = np.asarray(scores, dtype=float).reshape(-1)

        if len(scores) != len(candidates):
            raise ValueError("Reranker 返回的 score 数量与候选数量不一致")

        def get_score(index):
            return scores[index]

        ranked_indices = sorted(range(len(scores)), key=get_score, reverse=True)

        reranked_results = []

        top_indices = ranked_indices[:top_k]
        for rank, index in enumerate(top_indices, start=1):
            result = RetrievalResult(
                chunk=candidates[index].chunk, score=float(scores[index]), rank=rank
            )
            reranked_results.append(result)

        return reranked_results
