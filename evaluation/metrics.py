"""Retrieval Eval 的基础客观指标。"""


def hit_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """
    Hit@K：
    Top-K 中只要出现至少一个 Gold Chunk，就返回 1.0。
    """

    if k <= 0:
        raise ValueError("k 必须大于 0")

    relevant_set = set(relevant_chunk_ids)
    top_k = retrieved_chunk_ids[:k]

    return float(
        any(
            chunk_id in relevant_set
            for chunk_id in top_k
        )
    )


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """
    Recall@K：
    所有 Gold Chunk 中，有多少比例出现在 Top-K。
    """

    if k <= 0:
        raise ValueError("k 必须大于 0")

    if not relevant_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    retrieved_set = set(
        retrieved_chunk_ids[:k]
    )

    found_count = len(
        relevant_set & retrieved_set
    )

    return (
        found_count
        / len(relevant_set)
    )


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> float:
    """
    Reciprocal Rank：
    看第一个 Gold Chunk 排在第几名。

    Rank 1 → 1.0
    Rank 2 → 0.5
    Rank 3 → 0.333...
    没找到 → 0.0
    """

    relevant_set = set(
        relevant_chunk_ids
    )

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0