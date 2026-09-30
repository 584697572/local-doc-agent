"""运行正式的 BEIR Retrieval Benchmark。"""

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean

from beir.retrieval.evaluation import EvaluateRetrieval

from evaluation.beir_adapter import (
    beir_corpus_to_chunks,
    load_beir_dataset,
)
from evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
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
from retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    Reranker,
)
from retrieval.result import RetrievalResult


METHOD_NAMES = (
    "bm25",
    "dense",
    "rrf",
    "rerank",
    "safe_rerank",
)

# 最终排序阶段重点看的指标。
BEIR_FINAL_K_VALUES = [
    1,
    3,
    5,
    10,
]

# Recall 阶段重点看的深度。
BEIR_RECALL_K_VALUES = [
    5,
    10,
    20,
    100,
]


def result_chunk_ids(
    results,
) -> list[str]:
    """提取 RetrievalResult 中的 chunk_id。"""

    return [
        str(result.chunk.chunk_id)
        for result in results
    ]


def get_relevant_ids(
    query_qrels: dict,
) -> list[str]:
    """从 qrels 中取出 relevance > 0 的 Gold 文档。"""

    return [
        str(document_id)
        for document_id, relevance
        in query_qrels.items()
        if float(relevance) > 0
    ]


def get_relevant_ranks(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> list[int]:
    """找出所有 Gold 文档在当前排名中的位置。"""

    relevant_set = set(
        relevant_ids
    )

    return [
        rank
        for rank, document_id
        in enumerate(
            retrieved_ids,
            start=1,
        )
        if document_id in relevant_set
    ]


def ranking_to_beir_scores(
    results,
) -> dict[str, float]:
    """
    把 RetrievalResult 排名转换成 BEIR results 格式。

    使用严格递减的 rank score，
    保证 BEIR 使用的排序和 Retriever 一致。
    """

    result_count = len(
        results
    )

    return {
        str(result.chunk.chunk_id): float(
            result_count - index
        )
        for index, result
        in enumerate(results)
    }


def normalize_qrels(
    qrels: dict,
) -> dict[str, dict[str, int]]:
    """把 qrels 中的 ID 全部统一成字符串。"""

    return {
        str(query_id): {
            str(document_id): int(relevance)
            for document_id, relevance
            in document_qrels.items()
        }
        for query_id, document_qrels
        in qrels.items()
    }


def score_project_ranking(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> dict:
    """计算项目自己的 Hit / Recall / RR 指标。"""

    top_k_ids = (
        retrieved_ids[:k]
    )

    return {
        f"hit@{k}": hit_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),

        f"recall@{k}": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),

        f"rr@{k}": reciprocal_rank(
            top_k_ids,
            relevant_ids,
        ),
    }


def summarize_records(
    records: list[dict],
    method_name: str,
    k: int,
) -> dict:
    """汇总项目指标和平均查询延迟。"""

    method_records = [
        record["methods"][
            method_name
        ]
        for record in records
    ]

    if not method_records:
        return {
            f"hit@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"mrr@{k}": 0.0,
            "avg_latency_ms": 0.0,
        }

    return {
        f"hit@{k}": mean(
            item[f"hit@{k}"]
            for item in method_records
        ),

        f"recall@{k}": mean(
            item[f"recall@{k}"]
            for item in method_records
        ),

        f"mrr@{k}": mean(
            item[f"rr@{k}"]
            for item in method_records
        ),

        "avg_latency_ms": mean(
            item["latency_ms"]
            for item in method_records
        ),
    }


def summarize_candidate_pool(
    records: list[dict],
) -> dict:
    """汇总 RRF Candidate Pool 覆盖能力。"""

    if not records:
        return {
            "rrf_recall@5": 0.0,
            "rrf_recall@10": 0.0,
            "rrf_recall@20": 0.0,
            "rrf_recall@100": 0.0,
            "rrf_hit@20": 0.0,
            "queries_with_gold_in_top20": 0,
            "queries_without_gold_in_top20": 0,
        }

    return {
        "rrf_recall@5": mean(
            record[
                "candidate_analysis"
            ]["rrf_recall@5"]
            for record in records
        ),

        "rrf_recall@10": mean(
            record[
                "candidate_analysis"
            ]["rrf_recall@10"]
            for record in records
        ),

        "rrf_recall@20": mean(
            record[
                "candidate_analysis"
            ]["rrf_recall@20"]
            for record in records
        ),

        "rrf_recall@100": mean(
            record[
                "candidate_analysis"
            ]["rrf_recall@100"]
            for record in records
        ),

        "rrf_hit@20": mean(
            float(
                record[
                    "candidate_analysis"
                ]["gold_in_rerank_pool"]
            )
            for record in records
        ),

        "queries_with_gold_in_top20": sum(
            1
            for record in records
            if record[
                "candidate_analysis"
            ]["gold_in_rerank_pool"]
        ),

        "queries_without_gold_in_top20": sum(
            1
            for record in records
            if not record[
                "candidate_analysis"
            ]["gold_in_rerank_pool"]
        ),
    }


def select_query_items(
    queries: dict,
    qrels: dict,
    limit: int,
    seed: int,
) -> list[tuple[str, str]]:
    """选择参与评测的 Query。"""

    valid_items = []

    for query_id, query_text in queries.items():
        query_id = str(
            query_id
        )

        query_qrels = qrels.get(
            query_id,
            {},
        )

        if not get_relevant_ids(
            query_qrels
        ):
            continue

        valid_items.append(
            (
                query_id,
                query_text,
            )
        )

    valid_items.sort(
        key=lambda item: item[0]
    )

    if (
        limit <= 0
        or limit >= len(valid_items)
    ):
        return valid_items

    random_generator = random.Random(
        seed
    )

    return random_generator.sample(
        valid_items,
        limit,
    )


def evaluate_beir_metrics(
    qrels: dict,
    results: dict,
    k_values: list[int],
) -> dict:
    """使用 BEIR 官方 Evaluator 计算标准指标。"""

    filtered_results = {}

    for (
        query_id,
        document_scores,
    ) in results.items():

        query_id = str(
            query_id
        )

        filtered_results[
            query_id
        ] = {
            str(document_id): float(score)
            for document_id, score
            in document_scores.items()
            if str(document_id)
            != query_id
        }

    standard_results = {
        query_id: dict(
            document_scores
        )
        for query_id, document_scores
        in filtered_results.items()
    }

    (
        ndcg,
        map_scores,
        recall,
        precision,
    ) = EvaluateRetrieval.evaluate(
        qrels=qrels,
        results=standard_results,
        k_values=k_values,
        ignore_identical_ids=False,
    )

    mrr_results = {
        query_id: dict(
            document_scores
        )
        for query_id, document_scores
        in filtered_results.items()
    }

    mrr = (
        EvaluateRetrieval.evaluate_custom(
            qrels=qrels,
            results=mrr_results,
            k_values=k_values,
            metric="mrr",
        )
    )

    return {
        "ndcg": ndcg,
        "map": map_scores,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


def warmup_retrievers(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    reranker: Reranker,
    chunks: list,
    warmup_query: str,
) -> dict:
    """正式计时前 Warm-up。"""

    warmup_times = {}

    start = time.perf_counter()

    bm25.search(
        warmup_query,
        top_k=1,
    )

    warmup_times[
        "bm25_seconds"
    ] = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()

    dense.search(
        warmup_query,
        top_k=1,
    )

    warmup_times[
        "dense_seconds"
    ] = (
        time.perf_counter()
        - start
    )

    if chunks:
        warmup_candidate = (
            RetrievalResult(
                chunk=chunks[0],
                score=0.0,
                rank=1,
            )
        )

        start = time.perf_counter()

        reranker.rerank(
            query=warmup_query,
            candidates=[
                warmup_candidate
            ],
            top_k=1,
        )

        warmup_times[
            "reranker_seconds"
        ] = (
            time.perf_counter()
            - start
        )

    else:
        warmup_times[
            "reranker_seconds"
        ] = 0.0

    return warmup_times


def run_benchmark(
    dataset_name: str = "scifact",
    split: str = "test",
    top_k: int = 5,
    retrieval_depth: int = 100,
    rerank_candidate_k: int = 20,
    limit: int = 20,
    seed: int = 42,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rrf_k: int = DEFAULT_RRF_K,
    safe_fusion_k: int = DEFAULT_RRF_K,
    safe_rrf_weight: float = 1.0,
    safe_reranker_weight: float = 1.0,
) -> dict:
    """
    运行正式 BEIR Retrieval Benchmark。

    比较：
        1. BM25
        2. Dense
        3. BM25 + Dense + RRF
        4. RRF + Cross-Encoder Reranker
        5. RRF + Reranker Rank Fusion
    """

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    if retrieval_depth < 100:
        raise ValueError(
            "retrieval_depth 至少为 100"
        )

    if rerank_candidate_k < 10:
        raise ValueError(
            "rerank_candidate_k 至少为 10"
        )

    if (
        rerank_candidate_k
        > retrieval_depth
    ):
        raise ValueError(
            "rerank_candidate_k 不能大于 "
            "retrieval_depth"
        )

    if (
        top_k
        > rerank_candidate_k
    ):
        raise ValueError(
            "top_k 不能大于 "
            "rerank_candidate_k"
        )

    if rrf_k < 0:
        raise ValueError(
            "rrf_k 不能小于 0"
        )

    if safe_fusion_k < 0:
        raise ValueError(
            "safe_fusion_k 不能小于 0"
        )

    if (
        safe_rrf_weight < 0
        or safe_reranker_weight < 0
    ):
        raise ValueError(
            "Safe Rerank 权重不能小于 0"
        )

    if (
        safe_rrf_weight == 0
        and safe_reranker_weight == 0
    ):
        raise ValueError(
            "Safe Rerank 两个权重不能同时为 0"
        )

    print(
        f"[Eval] Dataset: "
        f"{dataset_name}"
    )

    print(
        f"[Eval] Embedding: "
        f"{embedding_model}"
    )

    print(
        f"[Eval] Reranker: "
        f"{reranker_model}"
    )

    print(
        f"[Eval] RRF k: "
        f"{rrf_k}"
    )

    print(
        "[Eval] Safe Rerank: "
        f"fusion_k={safe_fusion_k}, "
        f"rrf_weight={safe_rrf_weight}, "
        f"reranker_weight={safe_reranker_weight}"
    )

    # ==================================================
    # Load Dataset
    # ==================================================

    (
        corpus,
        queries,
        raw_qrels,
    ) = load_beir_dataset(
        dataset_name=dataset_name,
        split=split,
    )

    qrels = normalize_qrels(
        raw_qrels
    )

    queries = {
        str(query_id): query_text
        for query_id, query_text
        in queries.items()
    }

    chunks = beir_corpus_to_chunks(
        corpus=corpus,
        dataset_name=dataset_name,
    )

    print(
        f"[Eval] Corpus="
        f"{len(chunks)}, "
        f"Queries={len(queries)}"
    )

    # ==================================================
    # Build BM25
    # ==================================================

    print(
        "[Eval] Building BM25..."
    )

    start = time.perf_counter()

    bm25 = BM25Retriever(
        chunks
    )

    bm25_build_seconds = (
        time.perf_counter()
        - start
    )

    # ==================================================
    # Build Dense
    # ==================================================

    print(
        "[Eval] Building Dense + FAISS..."
    )

    start = time.perf_counter()

    dense = DenseRetriever(
        chunks=chunks,
        model_name=embedding_model,
    )

    dense_build_seconds = (
        time.perf_counter()
        - start
    )

    # ==================================================
    # Reranker
    # ==================================================

    reranker = Reranker(
        model_name=reranker_model,
    )

    # ==================================================
    # Query Selection
    # ==================================================

    query_items = select_query_items(
        queries=queries,
        qrels=qrels,
        limit=limit,
        seed=seed,
    )

    if not query_items:
        raise ValueError(
            "没有可用于评测的 Query"
        )

    print(
        f"[Eval] Evaluated queries="
        f"{len(query_items)}"
    )

    # ==================================================
    # Warm-up
    # ==================================================

    print(
        "[Eval] Warming up models..."
    )

    warmup_times = warmup_retrievers(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        chunks=chunks,
        warmup_query=query_items[0][1],
    )

    # ==================================================
    # BEIR Results
    # ==================================================

    beir_results = {
        method_name: {}
        for method_name
        in METHOD_NAMES
    }

    records = []

    safe_fusion_only_values = []

    total_queries = len(
        query_items
    )

    # ==================================================
    # Main Loop
    # ==================================================

    for index, (
        query_id,
        query_text,
    ) in enumerate(
        query_items,
        start=1,
    ):
        relevant_ids = get_relevant_ids(
            qrels.get(
                query_id,
                {},
            )
        )

        print(
            f"[Eval] "
            f"{index}/{total_queries} "
            f"query_id={query_id}"
        )

        # ==============================================
        # BM25
        # ==============================================

        start = time.perf_counter()

        bm25_results = bm25.search(
            query=query_text,
            top_k=retrieval_depth,
        )

        bm25_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ==============================================
        # Dense
        # ==============================================

        start = time.perf_counter()

        dense_results = dense.search(
            query=query_text,
            top_k=retrieval_depth,
        )

        dense_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ==============================================
        # RRF
        # ==============================================

        start = time.perf_counter()

        fused_results = (
            reciprocal_rank_fusion(
                bm25_results=bm25_results,
                dense_results=dense_results,
                top_k=retrieval_depth,
                rrf_k=rrf_k,
            )
        )

        rrf_only_ms = (
            time.perf_counter()
            - start
        ) * 1000

        rrf_total_ms = (
            bm25_ms
            + dense_ms
            + rrf_only_ms
        )

        rrf_ids = (
            result_chunk_ids(
                fused_results
            )
        )

        rrf_relevant_ranks = (
            get_relevant_ranks(
                retrieved_ids=rrf_ids,
                relevant_ids=relevant_ids,
            )
        )

        rrf_first_relevant_rank = (
            rrf_relevant_ranks[0]
            if rrf_relevant_ranks
            else None
        )

        gold_in_rerank_pool = any(
            rank <= rerank_candidate_k
            for rank
            in rrf_relevant_ranks
        )

        # ==============================================
        # Reranker
        # ==============================================

        rerank_candidates = (
            fused_results[
                :rerank_candidate_k
            ]
        )

        if rerank_candidates:
            start = time.perf_counter()

            # 必须重排完整 Top-20，
            # Safe Rerank 才能使用完整排名。
            reranked_results = (
                reranker.rerank(
                    query=query_text,
                    candidates=(
                        rerank_candidates
                    ),
                    top_k=len(
                        rerank_candidates
                    ),
                )
            )

            rerank_only_ms = (
                time.perf_counter()
                - start
            ) * 1000

        else:
            reranked_results = []
            rerank_only_ms = 0.0

        rerank_total_ms = (
            rrf_total_ms
            + rerank_only_ms
        )

        # ==============================================
        # Safe Rerank
        #
        # 只融合完全相同的 Top-20 Candidate Pool。
        #
        # 不允许 RRF Top-100 中未进入 Reranker 的
        # 文档突然进入最终结果。
        # ==============================================

        start = time.perf_counter()

        safe_reranked_results = (
            rank_preserving_rerank_fusion(
                rrf_results=(
                    rerank_candidates
                ),
                reranked_results=(
                    reranked_results
                ),
                top_k=len(
                    rerank_candidates
                ),
                fusion_k=(
                    safe_fusion_k
                ),
                rrf_weight=(
                    safe_rrf_weight
                ),
                reranker_weight=(
                    safe_reranker_weight
                ),
            )
        )

        safe_fusion_only_ms = (
            time.perf_counter()
            - start
        ) * 1000

        safe_fusion_only_values.append(
            safe_fusion_only_ms
        )

        safe_rerank_total_ms = (
            rerank_total_ms
            + safe_fusion_only_ms
        )

        # ==============================================
        # BEIR Results
        # ==============================================

        method_results = {
            "bm25": bm25_results,
            "dense": dense_results,
            "rrf": fused_results,
            "rerank": reranked_results,
            "safe_rerank": (
                safe_reranked_results
            ),
        }

        for (
            method_name,
            ranking_results,
        ) in method_results.items():

            beir_results[
                method_name
            ][query_id] = (
                ranking_to_beir_scores(
                    ranking_results
                )
            )

        # ==============================================
        # Project Metrics
        # ==============================================

        method_rankings = {
            "bm25": (
                bm25_results,
                bm25_ms,
            ),

            "dense": (
                dense_results,
                dense_ms,
            ),

            "rrf": (
                fused_results,
                rrf_total_ms,
            ),

            "rerank": (
                reranked_results,
                rerank_total_ms,
            ),

            "safe_rerank": (
                safe_reranked_results,
                safe_rerank_total_ms,
            ),
        }

        method_scores = {}

        for (
            method_name,
            (
                ranking_results,
                latency_ms,
            ),
        ) in method_rankings.items():

            retrieved_ids = (
                result_chunk_ids(
                    ranking_results
                )
            )

            scores = score_project_ranking(
                retrieved_ids=(
                    retrieved_ids
                ),
                relevant_ids=(
                    relevant_ids
                ),
                k=top_k,
            )

            scores[
                "latency_ms"
            ] = latency_ms

            scores[
                "retrieved_chunk_ids"
            ] = (
                retrieved_ids[:top_k]
            )

            if (
                method_name
                == "rerank"
            ):
                scores[
                    "rerank_only_ms"
                ] = rerank_only_ms

            if (
                method_name
                == "safe_rerank"
            ):
                scores[
                    "safe_fusion_only_ms"
                ] = (
                    safe_fusion_only_ms
                )

            method_scores[
                method_name
            ] = scores

        # ==============================================
        # Record
        # ==============================================

        records.append(
            {
                "query_id": query_id,

                "query": query_text,

                "relevant_chunk_ids": (
                    relevant_ids
                ),

                "candidate_analysis": {
                    "rrf_recall@5": (
                        recall_at_k(
                            rrf_ids,
                            relevant_ids,
                            5,
                        )
                    ),

                    "rrf_recall@10": (
                        recall_at_k(
                            rrf_ids,
                            relevant_ids,
                            10,
                        )
                    ),

                    "rrf_recall@20": (
                        recall_at_k(
                            rrf_ids,
                            relevant_ids,
                            20,
                        )
                    ),

                    "rrf_recall@100": (
                        recall_at_k(
                            rrf_ids,
                            relevant_ids,
                            100,
                        )
                    ),

                    "rrf_relevant_ranks": (
                        rrf_relevant_ranks
                    ),

                    "rrf_first_relevant_rank": (
                        rrf_first_relevant_rank
                    ),

                    "gold_in_rerank_pool": (
                        gold_in_rerank_pool
                    ),
                },

                "methods": (
                    method_scores
                ),
            }
        )

    # ==================================================
    # Qrels
    # ==================================================

    evaluated_query_ids = {
        record["query_id"]
        for record in records
    }

    eval_qrels = {
        query_id: qrels[
            query_id
        ]
        for query_id
        in evaluated_query_ids
    }

    # ==================================================
    # Project Summary
    # ==================================================

    project_summary = {}

    for method_name in METHOD_NAMES:
        project_summary[
            method_name
        ] = summarize_records(
            records=records,
            method_name=method_name,
            k=top_k,
        )

    candidate_pool_summary = (
        summarize_candidate_pool(
            records
        )
    )

    # ==================================================
    # BEIR Final Metrics
    # ==================================================

    beir_final_metrics = {}

    for method_name in METHOD_NAMES:
        beir_final_metrics[
            method_name
        ] = evaluate_beir_metrics(
            qrels=eval_qrels,
            results=beir_results[
                method_name
            ],
            k_values=(
                BEIR_FINAL_K_VALUES
            ),
        )

    # ==================================================
    # Recall-stage
    # ==================================================

    beir_recall_metrics = {}

    for method_name in (
        "bm25",
        "dense",
        "rrf",
    ):
        beir_recall_metrics[
            method_name
        ] = evaluate_beir_metrics(
            qrels=eval_qrels,
            results=beir_results[
                method_name
            ],
            k_values=(
                BEIR_RECALL_K_VALUES
            ),
        )

    # ==================================================
    # Timing
    # ==================================================

    rerank_only_values = [
        record[
            "methods"
        ]["rerank"][
            "rerank_only_ms"
        ]
        for record in records
    ]

    avg_rerank_only_ms = (
        mean(
            rerank_only_values
        )
        if rerank_only_values
        else 0.0
    )

    avg_safe_fusion_only_ms = (
        mean(
            safe_fusion_only_values
        )
        if safe_fusion_only_values
        else 0.0
    )

    # ==================================================
    # Final Report
    # ==================================================

    return {
        "dataset": dataset_name,

        "split": split,

        "models": {
            "embedding": (
                embedding_model
            ),

            "reranker": (
                reranker_model
            ),
        },

        "retrieval": {
            "rrf_k": rrf_k,
        },

        "safe_rerank": {
            "fusion_k": (
                safe_fusion_k
            ),

            "rrf_weight": (
                safe_rrf_weight
            ),

            "reranker_weight": (
                safe_reranker_weight
            ),
        },

        "configuration": {
            "project_top_k": top_k,

            "retrieval_depth": (
                retrieval_depth
            ),

            "rerank_candidate_k": (
                rerank_candidate_k
            ),

            "limit": limit,

            "seed": seed,

            "beir_final_k_values": (
                BEIR_FINAL_K_VALUES
            ),

            "beir_recall_k_values": (
                BEIR_RECALL_K_VALUES
            ),
        },

        "evaluated_queries": (
            len(records)
        ),

        "build_time_seconds": {
            "bm25": (
                bm25_build_seconds
            ),

            "dense": (
                dense_build_seconds
            ),
        },

        "warmup_time_seconds": (
            warmup_times
        ),

        "timing": {
            "avg_rerank_only_ms": (
                avg_rerank_only_ms
            ),

            "avg_safe_fusion_only_ms": (
                avg_safe_fusion_only_ms
            ),
        },

        "project_summary": (
            project_summary
        ),

        "candidate_pool_summary": (
            candidate_pool_summary
        ),

        "beir_final_metrics": (
            beir_final_metrics
        ),

        "beir_recall_metrics": (
            beir_recall_metrics
        ),

        "records": records,
    }


def save_report(
    report: dict,
    output_path: str | Path,
) -> None:
    """保存完整 Benchmark JSON。"""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_project_summary(
    report: dict,
) -> None:
    """打印项目指标。"""

    top_k = report[
        "configuration"
    ]["project_top_k"]

    print()
    print("=" * 82)

    print(
        f"Project Metrics @ {top_k}"
    )

    print("=" * 82)

    print(
        f"{'Method':<15}"
        f"{'Hit':>10}"
        f"{'Recall':>12}"
        f"{'MRR':>10}"
        f"{'Latency(ms)':>18}"
    )

    print("-" * 82)

    for method_name in METHOD_NAMES:
        metrics = report[
            "project_summary"
        ][method_name]

        print(
            f"{method_name:<15}"
            f"{metrics[f'hit@{top_k}']:>10.4f}"
            f"{metrics[f'recall@{top_k}']:>12.4f}"
            f"{metrics[f'mrr@{top_k}']:>10.4f}"
            f"{metrics['avg_latency_ms']:>18.2f}"
        )

    print("=" * 82)


def print_beir_final_summary(
    report: dict,
) -> None:
    """打印 BEIR Top-10 指标。"""

    print()
    print("=" * 92)
    print(
        "BEIR Final Ranking Metrics"
    )
    print("=" * 92)

    print(
        f"{'Method':<15}"
        f"{'NDCG@10':>12}"
        f"{'MAP@10':>12}"
        f"{'Recall@10':>14}"
        f"{'P@10':>10}"
        f"{'MRR@10':>12}"
    )

    print("-" * 92)

    for method_name in METHOD_NAMES:
        metrics = report[
            "beir_final_metrics"
        ][method_name]

        print(
            f"{method_name:<15}"
            f"{metrics['ndcg']['NDCG@10']:>12.4f}"
            f"{metrics['map']['MAP@10']:>12.4f}"
            f"{metrics['recall']['Recall@10']:>14.4f}"
            f"{metrics['precision']['P@10']:>10.4f}"
            f"{metrics['mrr']['MRR@10']:>12.4f}"
        )

    print("=" * 92)


def print_beir_recall_summary(
    report: dict,
) -> None:
    """打印 Retrieval Recall-stage 指标。"""

    print()
    print("=" * 78)
    print(
        "BEIR Recall-stage Metrics"
    )
    print("=" * 78)

    print(
        f"{'Method':<12}"
        f"{'Recall@5':>12}"
        f"{'Recall@10':>12}"
        f"{'Recall@20':>12}"
        f"{'Recall@100':>14}"
    )

    print("-" * 78)

    for method_name in (
        "bm25",
        "dense",
        "rrf",
    ):
        metrics = report[
            "beir_recall_metrics"
        ][method_name]

        print(
            f"{method_name:<12}"
            f"{metrics['recall']['Recall@5']:>12.4f}"
            f"{metrics['recall']['Recall@10']:>12.4f}"
            f"{metrics['recall']['Recall@20']:>12.4f}"
            f"{metrics['recall']['Recall@100']:>14.4f}"
        )

    print("=" * 78)


def print_candidate_pool_summary(
    report: dict,
) -> None:
    """打印 Candidate Pool 情况。"""

    summary = report[
        "candidate_pool_summary"
    ]

    print()
    print("=" * 62)
    print(
        "RRF Candidate Pool Analysis"
    )
    print("=" * 62)

    print(
        f"Recall@5:   "
        f"{summary['rrf_recall@5']:.4f}"
    )

    print(
        f"Recall@10:  "
        f"{summary['rrf_recall@10']:.4f}"
    )

    print(
        f"Recall@20:  "
        f"{summary['rrf_recall@20']:.4f}"
    )

    print(
        f"Recall@100: "
        f"{summary['rrf_recall@100']:.4f}"
    )

    print(
        f"Hit@20:     "
        f"{summary['rrf_hit@20']:.4f}"
    )

    print("=" * 62)


def print_summary(
    report: dict,
) -> None:
    """打印完整 Summary。"""

    print()

    print(
        f"Dataset: "
        f"{report['dataset']}"
    )

    print(
        "Embedding Model: "
        f"{report['models']['embedding']}"
    )

    print(
        "Reranker Model: "
        f"{report['models']['reranker']}"
    )

    print(
        "Safe Rerank: "
        f"RRF weight="
        f"{report['safe_rerank']['rrf_weight']}, "
        f"Reranker weight="
        f"{report['safe_rerank']['reranker_weight']}"
    )

    print_project_summary(
        report
    )

    print_beir_final_summary(
        report
    )

    print_beir_recall_summary(
        report
    )

    print_candidate_pool_summary(
        report
    )

    print()

    print(
        "[Timing] Dense build: "
        f"{report['build_time_seconds']['dense']:.2f}s"
    )

    print(
        "[Timing] Avg reranker-only: "
        f"{report['timing']['avg_rerank_only_ms']:.2f}ms"
    )

    print(
        "[Timing] Avg Safe Fusion-only: "
        f"{report['timing']['avg_safe_fusion_only_ms']:.4f}ms"
    )


def model_name_to_slug(
    model_name: str,
) -> str:
    """把模型名转换成安全文件名。"""

    return (
        model_name
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run formal BEIR benchmark "
            "for LocalDoc-Agent."
        )
    )

    parser.add_argument(
        "--dataset",
        default="scifact",
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--retrieval-depth",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--embedding-model",
        default=(
            DEFAULT_EMBEDDING_MODEL
        ),
    )

    parser.add_argument(
        "--reranker-model",
        default=(
            DEFAULT_RERANKER_MODEL
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
    )

    parser.add_argument(
        "--safe-fusion-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "Safe Rerank Rank Fusion "
            "平滑常数。"
        ),
    )

    parser.add_argument(
        "--safe-rrf-weight",
        type=float,
        default=1.0,
        help=(
            "Safe Rerank 中原始 RRF "
            "排名权重。"
        ),
    )

    parser.add_argument(
        "--safe-reranker-weight",
        type=float,
        default=1.0,
        help=(
            "Safe Rerank 中 Cross-Encoder "
            "排名权重。"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    report = run_benchmark(
        dataset_name=args.dataset,
        split=args.split,
        top_k=args.top_k,
        retrieval_depth=(
            args.retrieval_depth
        ),
        rerank_candidate_k=(
            args.rerank_candidate_k
        ),
        limit=args.limit,
        seed=args.seed,
        embedding_model=(
            args.embedding_model
        ),
        reranker_model=(
            args.reranker_model
        ),
        rrf_k=args.rrf_k,
        safe_fusion_k=(
            args.safe_fusion_k
        ),
        safe_rrf_weight=(
            args.safe_rrf_weight
        ),
        safe_reranker_weight=(
            args.safe_reranker_weight
        ),
    )

    if args.output:
        output_path = (
            args.output
        )

    else:
        suffix = (
            str(args.limit)
            if args.limit > 0
            else "all"
        )

        embedding_slug = (
            model_name_to_slug(
                args.embedding_model
            )
        )

        output_path = (
            "evaluation/results/"
            f"{args.dataset}_"
            f"{embedding_slug}_"
            f"{suffix}_formal.json"
        )

    save_report(
        report,
        output_path,
    )

    print_summary(
        report
    )

    print()

    print(
        f"[Eval] Report saved: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()