from retrieval.result import RetrievalResult


DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """
    使用 Reciprocal Rank Fusion (RRF)
    融合 BM25 和 Dense Retrieval 的排名结果。

    RRF 不直接比较两种 Retriever 的原始 score，
    而是根据每个 chunk 在不同结果列表中的 rank 计算融合分数。

    公式：

        RRF(d) = Σ 1 / (rrf_k + rank)

    参数：
        bm25_results:
            BM25Retriever 返回的结果。

        dense_results:
            DenseRetriever 返回的结果。

        top_k:
            最终最多返回多少个融合结果。

        rrf_k:
            RRF 平滑常数，默认 60。

    返回：
        按 RRF score 从高到低排列的 RetrievalResult。
    """

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if rrf_k < 0:
        raise ValueError("rrf_k 不能小于 0")

    # 保存每个 chunk 累积得到的 RRF 分数。
    #
    # 结构类似：
    #
    # {
    #     "chunk_001": 0.0325,
    #     "chunk_002": 0.0161,
    # }
    fused_scores = {}

    # 保存 chunk_id → DocumentChunk 的映射。
    #
    # 因为最后排序完成以后，
    # 还需要把 chunk_id 重新转换成真正的 DocumentChunk。
    chunks_by_id = {}

    # 依次处理 BM25 和 Dense 两套排名。
    for results in (bm25_results, dense_results):

        for result in results:
            chunk_id = result.chunk.chunk_id

            # 保存真正的 DocumentChunk。
            chunks_by_id[chunk_id] = result.chunk

            # 当前排名对 RRF 的贡献。
            contribution = 1.0 / (
                rrf_k + result.rank
            )

            # 如果这个 chunk 之前已经被另一个 Retriever 命中过，
            # 就在原有分数基础上继续累加。
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + contribution
            )

    # 没有任何 Retriever 返回结果。
    if not fused_scores:
        return []

    # 按 RRF score 从高到低排列 chunk_id。
    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            chunk_id,
        ),
    )

    results = []

    for rank, chunk_id in enumerate(
        ranked_chunk_ids[:top_k],
        start=1,
    ):
        results.append(
            RetrievalResult(
                chunk=chunks_by_id[chunk_id],
                score=float(
                    fused_scores[chunk_id]
                ),
                rank=rank,
            )
        )

    return results