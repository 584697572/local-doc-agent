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
    而是根据每个 chunk 在不同结果列表中的 rank
    计算融合分数。

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
        raise ValueError(
            "top_k 必须大于 0"
        )

    if rrf_k < 0:
        raise ValueError(
            "rrf_k 不能小于 0"
        )

    # 保存每个 chunk 累积得到的 RRF 分数。
    fused_scores = {}

    # chunk_id → DocumentChunk。
    chunks_by_id = {}

    # 依次处理 BM25 和 Dense 两套排名。
    for results in (
        bm25_results,
        dense_results,
    ):
        for result in results:
            chunk_id = (
                result.chunk.chunk_id
            )

            chunks_by_id[
                chunk_id
            ] = result.chunk

            contribution = (
                1.0
                / (
                    rrf_k
                    + result.rank
                )
            )

            fused_scores[
                chunk_id
            ] = (
                fused_scores.get(
                    chunk_id,
                    0.0,
                )
                + contribution
            )

    if not fused_scores:
        return []

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[
                chunk_id
            ],
            chunk_id,
        ),
    )

    results = []

    for rank, chunk_id in enumerate(
        ranked_chunk_ids[
            :top_k
        ],
        start=1,
    ):
        results.append(
            RetrievalResult(
                chunk=chunks_by_id[
                    chunk_id
                ],
                score=float(
                    fused_scores[
                        chunk_id
                    ]
                ),
                rank=rank,
            )
        )

    return results


def rank_preserving_rerank_fusion(
    rrf_results: list[RetrievalResult],
    reranked_results: list[RetrievalResult],
    top_k: int = 5,
    fusion_k: int = DEFAULT_RRF_K,
    rrf_weight: float = 1.0,
    reranker_weight: float = 1.0,
) -> list[RetrievalResult]:
    """
    融合原始 RRF 排名和 Reranker 排名。

    目的：
        不让 Reranker 完全覆盖 RRF 的排序结果，
        而是同时保留：

            1. 第一阶段 Hybrid Retrieval 的排名信息；
            2. Cross-Encoder Reranker 的语义判断。

    公式：

        score(d)
        =
        rrf_weight
        / (fusion_k + rrf_rank)

        +

        reranker_weight
        / (fusion_k + reranker_rank)

    默认：
        RRF 和 Reranker 权重都为 1.0，
        即先做一个完全中性的 1:1 对照实验。

    注意：
        这不是说该方案一定比纯 Reranker 更好。

        它是针对当前 Failure Analysis 中
        reranker_harm 较多的问题，
        提出的一个可验证实验假设。

    参数：
        rrf_results:
            RRF 阶段产生的候选排名。

        reranked_results:
            对同一批候选进行 Cross-Encoder
            重排后的结果。

        top_k:
            最终返回数量。

        fusion_k:
            Rank Fusion 的平滑常数。

        rrf_weight:
            原始 RRF 排名的权重。

        reranker_weight:
            Reranker 排名的权重。
    """

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    if fusion_k < 0:
        raise ValueError(
            "fusion_k 不能小于 0"
        )

    if rrf_weight < 0:
        raise ValueError(
            "rrf_weight 不能小于 0"
        )

    if reranker_weight < 0:
        raise ValueError(
            "reranker_weight 不能小于 0"
        )

    if (
        rrf_weight == 0
        and reranker_weight == 0
    ):
        raise ValueError(
            "rrf_weight 和 "
            "reranker_weight "
            "不能同时为 0"
        )

    if (
        not rrf_results
        and not reranked_results
    ):
        return []

    fused_scores = {}

    chunks_by_id = {}

    # 保存两套原始排名。
    #
    # 除了计算分数，
    # 后面发生完全相同的融合分数时，
    # 我们优先保留原始 RRF 排名。
    rrf_rank_by_id = {}

    reranker_rank_by_id = {}

    # ==================================================
    # 原始 RRF 排名贡献
    # ==================================================

    for result in rrf_results:
        chunk_id = (
            result.chunk.chunk_id
        )

        chunks_by_id[
            chunk_id
        ] = result.chunk

        rrf_rank_by_id[
            chunk_id
        ] = result.rank

        contribution = (
            rrf_weight
            / (
                fusion_k
                + result.rank
            )
        )

        fused_scores[
            chunk_id
        ] = (
            fused_scores.get(
                chunk_id,
                0.0,
            )
            + contribution
        )

    # ==================================================
    # Reranker 排名贡献
    # ==================================================

    for result in reranked_results:
        chunk_id = (
            result.chunk.chunk_id
        )

        chunks_by_id[
            chunk_id
        ] = result.chunk

        reranker_rank_by_id[
            chunk_id
        ] = result.rank

        contribution = (
            reranker_weight
            / (
                fusion_k
                + result.rank
            )
        )

        fused_scores[
            chunk_id
        ] = (
            fused_scores.get(
                chunk_id,
                0.0,
            )
            + contribution
        )

    # ==================================================
    # Final Ranking
    # ==================================================

    infinity = float(
        "inf"
    )

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            # 第一优先级：
            # 融合分数越高越好。
            -fused_scores[
                chunk_id
            ],

            # 第二优先级：
            # 如果融合分数完全一样，
            # 优先保留原始 RRF 的靠前结果。
            rrf_rank_by_id.get(
                chunk_id,
                infinity,
            ),

            # 第三优先级：
            # 再参考 Reranker 排名。
            reranker_rank_by_id.get(
                chunk_id,
                infinity,
            ),

            # 最后保证排序确定性。
            chunk_id,
        ),
    )

    results = []

    for rank, chunk_id in enumerate(
        ranked_chunk_ids[
            :top_k
        ],
        start=1,
    ):
        results.append(
            RetrievalResult(
                chunk=chunks_by_id[
                    chunk_id
                ],

                # 这里保存的是新的融合 score，
                # 而不是 Cross-Encoder 原始 score。
                score=float(
                    fused_scores[
                        chunk_id
                    ]
                ),

                rank=rank,
            )
        )

    return results