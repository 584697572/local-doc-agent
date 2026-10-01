"""
MultiHop-RAG Retrieval Benchmark。

第一阶段目标：

比较 LocalDoc-Agent 当前各 Retrieval Stage：

    BM25
    Dense
    RRF
    Cross-Encoder Reranker
    Safe Rerank

同时计算两类指标：

1. MultiHop-RAG 官方风格指标
   - Hits@4
   - Hits@10
   - MAP@10
   - MRR@10

2. LocalDoc-Agent Multi-Hop 指标
   - Evidence Document Hit@K
   - Evidence Document Recall@K
   - All-Evidence Hit@K

其中：

All-Evidence Hit@K

是 Multi-Hop 场景特别重要的指标：

    只有当所有 supporting documents
    都出现在 Top-K Chunk 中，
    才记为 1。
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from evaluation.multihop_adapter import (
    MultiHopCase,
    load_multihop_benchmark,
)
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
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


# ======================================================
# Benchmark Defaults
# ======================================================

# MultiHop-RAG 是英文新闻语料，
# 因此这里显式使用英文 Embedding。
DEFAULT_MULTIHOP_EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

DEFAULT_RETRIEVAL_DEPTH = 100
DEFAULT_RERANK_CANDIDATE_K = 20

DEFAULT_SAFE_FUSION_K = 60
DEFAULT_SAFE_RRF_WEIGHT = 1.0
DEFAULT_SAFE_RERANKER_WEIGHT = 2.0

METHOD_NAMES = (
    "bm25",
    "dense",
    "rrf",
    "rerank",
    "safe_rerank",
)

NON_NULL_TYPES = (
    "inference_query",
    "comparison_query",
    "temporal_query",
)

OFFICIAL_METRIC_NAMES = (
    "hits@4",
    "hits@10",
    "map@10",
    "mrr@10",
)

DOCUMENT_METRIC_NAMES = (
    "doc_hit@4",
    "doc_recall@4",
    "all_evidence_hit@4",

    "doc_hit@10",
    "doc_recall@10",
    "all_evidence_hit@10",

    "doc_hit@20",
    "doc_recall@20",
    "all_evidence_hit@20",
)


# ======================================================
# Official-like Fact Metrics
# ======================================================


def normalize_official_text(
    text: str,
) -> str:
    """
    对齐 MultiHop-RAG 官方 Retrieval Eval
    的文本匹配方式。

    官方做法会去掉：
        空格
        换行

    然后判断 Gold Fact
    是否为 Retrieved Chunk 的子串。
    """

    return (
        str(text)
        .replace(" ", "")
        .replace("\n", "")
    )


def score_official_facts(
    retrieved_texts: list[str],
    gold_facts: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """
    计算 MultiHop-RAG 官方风格 Retrieval 指标。

    注意：

    官方 Benchmark 的 Hits 指标只要求
    Top-K 中出现至少一个 Gold Fact。

    因此它不代表所有 Multi-Hop Evidence
    已经全部找齐。
    """

    if not gold_facts:
        return {
            "hits@4": 0.0,
            "hits@10": 0.0,
            "map@10": 0.0,
            "mrr@10": 0.0,
        }

    normalized_gold = [
        normalize_official_text(
            fact
        )
        for fact in gold_facts
        if str(fact).strip()
    ]

    normalized_retrieved = [
        normalize_official_text(
            text
        )
        for text in retrieved_texts[
            :10
        ]
    ]

    if not normalized_gold:
        return {
            "hits@4": 0.0,
            "hits@10": 0.0,
            "map@10": 0.0,
            "mrr@10": 0.0,
        }

    hits_at_4 = False
    hits_at_10 = False

    first_relevant_rank = None

    # 防止同一个 Gold Fact
    # 被多个重叠 Chunk 重复计分。
    found_gold = set()

    average_precision_sum = 0.0

    for rank, retrieved_text in enumerate(
        normalized_retrieved,
        start=1,
    ):
        matched_indices = [
            index
            for index, gold_fact
            in enumerate(
                normalized_gold
            )
            if (
                gold_fact
                and gold_fact
                in retrieved_text
            )
        ]

        if not matched_indices:
            continue

        if rank <= 10:
            hits_at_10 = True

        if rank <= 4:
            hits_at_4 = True

        if first_relevant_rank is None:
            first_relevant_rank = rank

        # 当前 Rank 第一次新找到几个 Gold Fact。
        new_matches = [
            index
            for index in matched_indices
            if index not in found_gold
        ]

        for index in new_matches:
            found_gold.add(
                index
            )

        if new_matches:
            # 这里保持和官方脚本的思想一致：
            # 新 Gold 数量 / 当前 Rank。
            average_precision_sum += (
                len(new_matches)
                / rank
            )

    denominator = min(
        len(normalized_gold),
        10,
    )

    map_at_10 = (
        average_precision_sum
        / denominator
    )

    mrr_at_10 = (
        1.0
        / first_relevant_rank
        if first_relevant_rank
        is not None
        else 0.0
    )

    return {
        "hits@4": float(
            hits_at_4
        ),
        "hits@10": float(
            hits_at_10
        ),
        "map@10": float(
            map_at_10
        ),
        "mrr@10": float(
            mrr_at_10
        ),
    }


# ======================================================
# Multi-Hop Document Metrics
# ======================================================


def score_evidence_documents(
    results: list[RetrievalResult],
    gold_document_ids: (
        list[str]
        | tuple[str, ...]
    ),
    k: int,
) -> dict[str, float]:
    """
    根据父文档计算 Multi-Hop Evidence Coverage。

    注意：

    K 仍然表示 Top-K Chunk。

    例如：
        Top-10 Chunk 中，
        是否覆盖了全部 Gold Parent Documents。

    这比“取前 10 个唯一文档”更符合真实 RAG，
    因为最终送给 LLM 的通常就是 Top-K Chunks。
    """

    if k <= 0:
        raise ValueError(
            "k 必须大于 0"
        )

    gold_set = set(
        gold_document_ids
    )

    if not gold_set:
        return {
            f"doc_hit@{k}": 0.0,
            f"doc_recall@{k}": 0.0,
            f"all_evidence_hit@{k}": 0.0,
        }

    retrieved_document_ids = {
        result.chunk.document_id
        for result in results[:k]
    }

    found = (
        gold_set
        & retrieved_document_ids
    )

    doc_hit = float(
        bool(found)
    )

    doc_recall = (
        len(found)
        / len(gold_set)
    )

    all_evidence_hit = float(
        gold_set.issubset(
            retrieved_document_ids
        )
    )

    return {
        f"doc_hit@{k}": (
            doc_hit
        ),

        f"doc_recall@{k}": (
            doc_recall
        ),

        f"all_evidence_hit@{k}": (
            all_evidence_hit
        ),
    }


def score_method(
    results: list[RetrievalResult],
    case: MultiHopCase,
) -> dict[str, float]:
    """
    对一个 Method 的结果计算完整指标。
    """

    retrieved_texts = [
        result.chunk.content
        for result in results
    ]

    metrics = (
        score_official_facts(
            retrieved_texts=(
                retrieved_texts
            ),
            gold_facts=(
                case.evidence_facts
            ),
        )
    )

    for k in (
        4,
        10,
        20,
    ):
        metrics.update(
            score_evidence_documents(
                results=results,
                gold_document_ids=(
                    case.evidence_document_ids
                ),
                k=k,
            )
        )

    return metrics


# ======================================================
# Case Selection
# ======================================================


def select_benchmark_cases(
    cases: list[MultiHopCase],
    limit: int,
    seed: int,
) -> list[MultiHopCase]:
    """
    选择参与 Retrieval Benchmark 的 Case。

    Retrieval 阶段排除 null_query。

    limit > 0：
        做固定随机种子的分层采样。

    例如 limit=20：

        inference   7
        comparison  7
        temporal    6

    避免随机 20 条恰好集中在某一种 Query Type。
    """

    groups = {
        question_type: []
        for question_type
        in NON_NULL_TYPES
    }

    for case in cases:
        if (
            case.question_type
            in groups
        ):
            groups[
                case.question_type
            ].append(
                case
            )

    for question_type in (
        NON_NULL_TYPES
    ):
        groups[
            question_type
        ].sort(
            key=lambda case: (
                case.case_id
            )
        )

    all_non_null = [
        case
        for question_type
        in NON_NULL_TYPES
        for case
        in groups[
            question_type
        ]
    ]

    if (
        limit <= 0
        or limit
        >= len(all_non_null)
    ):
        return sorted(
            all_non_null,
            key=lambda case: (
                case.case_id
            ),
        )

    random_generator = (
        random.Random(
            seed
        )
    )

    type_count = len(
        NON_NULL_TYPES
    )

    base_quota = (
        limit
        // type_count
    )

    remainder = (
        limit
        % type_count
    )

    selected = []

    for index, question_type in enumerate(
        NON_NULL_TYPES
    ):
        quota = (
            base_quota
            + (
                1
                if index < remainder
                else 0
            )
        )

        group = groups[
            question_type
        ]

        quota = min(
            quota,
            len(group),
        )

        selected.extend(
            random_generator.sample(
                group,
                quota,
            )
        )

    return sorted(
        selected,
        key=lambda case: (
            case.case_id
        ),
    )


# ======================================================
# Summary
# ======================================================


def summarize_method_records(
    records: list[dict],
    method_name: str,
) -> dict:
    """
    汇总一个 Retrieval Method 的平均指标。
    """

    if not records:
        return {
            metric_name: 0.0
            for metric_name in (
                OFFICIAL_METRIC_NAMES
                + DOCUMENT_METRIC_NAMES
            )
        }

    method_records = [
        record["methods"][
            method_name
        ]
        for record in records
    ]

    summary = {}

    for metric_name in (
        OFFICIAL_METRIC_NAMES
        + DOCUMENT_METRIC_NAMES
    ):
        summary[
            metric_name
        ] = mean(
            item["metrics"][
                metric_name
            ]
            for item
            in method_records
        )

    summary[
        "avg_latency_ms"
    ] = mean(
        item[
            "latency_ms"
        ]
        for item
        in method_records
    )

    return summary


def summarize_records(
    records: list[dict],
) -> dict:
    """
    同时输出：

        overall
        by_question_type
    """

    overall = {
        method_name: (
            summarize_method_records(
                records,
                method_name,
            )
        )
        for method_name
        in METHOD_NAMES
    }

    question_types = sorted(
        {
            record[
                "question_type"
            ]
            for record in records
        }
    )

    by_question_type = {}

    for question_type in (
        question_types
    ):
        type_records = [
            record
            for record in records
            if (
                record[
                    "question_type"
                ]
                == question_type
            )
        ]

        by_question_type[
            question_type
        ] = {
            method_name: (
                summarize_method_records(
                    type_records,
                    method_name,
                )
            )
            for method_name
            in METHOD_NAMES
        }

    return {
        "overall": overall,
        "by_question_type": (
            by_question_type
        ),
    }


# ======================================================
# Warm-up
# ======================================================


def warmup_reranker(
    reranker: Reranker,
    chunks,
    query: str,
) -> float:
    """
    正式计时前加载 Cross-Encoder，
    避免第一条 Query 单独承担模型加载成本。
    """

    if not chunks:
        return 0.0

    candidate = RetrievalResult(
        chunk=chunks[0],
        score=0.0,
        rank=1,
    )

    start = time.perf_counter()

    reranker.rerank(
        query=query,
        candidates=[
            candidate
        ],
        top_k=1,
    )

    return (
        time.perf_counter()
        - start
    )


# ======================================================
# Benchmark
# ======================================================


def run_benchmark(
    limit: int = 20,
    seed: int = 42,
    chunk_size: int = 500,
    overlap: int = 100,
    retrieval_depth: int = (
        DEFAULT_RETRIEVAL_DEPTH
    ),
    rerank_candidate_k: int = (
        DEFAULT_RERANK_CANDIDATE_K
    ),
    embedding_model: str = (
        DEFAULT_MULTIHOP_EMBEDDING_MODEL
    ),
    reranker_model: str = (
        DEFAULT_RERANKER_MODEL
    ),
    rrf_k: int = DEFAULT_RRF_K,
    safe_fusion_k: int = (
        DEFAULT_SAFE_FUSION_K
    ),
    safe_rrf_weight: float = (
        DEFAULT_SAFE_RRF_WEIGHT
    ),
    safe_reranker_weight: float = (
        DEFAULT_SAFE_RERANKER_WEIGHT
    ),
) -> dict:
    """
    运行 MultiHop-RAG Retrieval Benchmark。
    """

    if retrieval_depth < 20:
        raise ValueError(
            "retrieval_depth 至少为 20"
        )

    if (
        rerank_candidate_k < 10
    ):
        raise ValueError(
            "rerank_candidate_k "
            "至少为 10"
        )

    if (
        rerank_candidate_k
        > retrieval_depth
    ):
        raise ValueError(
            "rerank_candidate_k 不能大于 "
            "retrieval_depth"
        )

    print(
        "[MultiHop Eval] Loading dataset..."
    )

    (
        documents,
        chunks,
        cases,
    ) = load_multihop_benchmark(
        chunk_size=chunk_size,
        overlap=overlap,
        download=False,
    )

    selected_cases = (
        select_benchmark_cases(
            cases=cases,
            limit=limit,
            seed=seed,
        )
    )

    if not selected_cases:
        raise ValueError(
            "没有可评测的 non-null Case"
        )

    type_counts = Counter(
        case.question_type
        for case in selected_cases
    )

    print(
        f"[MultiHop Eval] Documents="
        f"{len(documents)}"
    )

    print(
        f"[MultiHop Eval] Chunks="
        f"{len(chunks)}"
    )

    print(
        f"[MultiHop Eval] Cases="
        f"{len(selected_cases)}"
    )

    print(
        f"[MultiHop Eval] Types="
        f"{dict(type_counts)}"
    )

    print(
        f"[MultiHop Eval] Embedding="
        f"{embedding_model}"
    )

    print(
        f"[MultiHop Eval] Reranker="
        f"{reranker_model}"
    )

    # ==================================================
    # Build BM25
    # ==================================================

    print(
        "[MultiHop Eval] Building BM25..."
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
        "[MultiHop Eval] Building Dense + FAISS..."
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

    print(
        "[MultiHop Eval] Warming up Reranker..."
    )

    reranker_warmup_seconds = (
        warmup_reranker(
            reranker=reranker,
            chunks=chunks,
            query=(
                selected_cases[0]
                .query
            ),
        )
    )

    records = []

    # ==================================================
    # Query Loop
    # ==================================================

    total_cases = len(
        selected_cases
    )

    for case_index, case in enumerate(
        selected_cases,
        start=1,
    ):
        print(
            f"[MultiHop Eval] "
            f"{case_index}/{total_cases} "
            f"{case.case_id} "
            f"({case.question_type})"
        )

        # ==============================================
        # BM25
        # ==============================================

        start = time.perf_counter()

        bm25_results = bm25.search(
            case.query,
            top_k=retrieval_depth,
        )

        bm25_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # ==============================================
        # Dense
        # ==============================================

        start = time.perf_counter()

        dense_results = (
            dense.search(
                case.query,
                top_k=retrieval_depth,
            )
        )

        dense_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # ==============================================
        # RRF
        # ==============================================

        start = time.perf_counter()

        rrf_results = (
            reciprocal_rank_fusion(
                bm25_results=(
                    bm25_results
                ),
                dense_results=(
                    dense_results
                ),
                top_k=(
                    retrieval_depth
                ),
                rrf_k=rrf_k,
            )
        )

        rrf_fusion_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        rrf_pipeline_ms = (
            bm25_ms
            + dense_ms
            + rrf_fusion_ms
        )

        # ==============================================
        # Candidate Pool
        # ==============================================

        rerank_candidates = (
            rrf_results[
                :rerank_candidate_k
            ]
        )

        # ==============================================
        # Reranker
        # ==============================================

        start = time.perf_counter()

        reranked_results = (
            reranker.rerank(
                query=case.query,
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
        ) * 1000.0

        rerank_pipeline_ms = (
            rrf_pipeline_ms
            + rerank_only_ms
        )

        # ==============================================
        # Safe Rerank
        # ==============================================

        start = time.perf_counter()

        safe_results = (
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
        ) * 1000.0

        safe_pipeline_ms = (
            rerank_pipeline_ms
            + safe_fusion_only_ms
        )

        method_results = {
            "bm25": (
                bm25_results
            ),
            "dense": (
                dense_results
            ),
            "rrf": (
                rrf_results
            ),
            "rerank": (
                reranked_results
            ),
            "safe_rerank": (
                safe_results
            ),
        }

        method_latencies = {
            "bm25": (
                bm25_ms
            ),

            "dense": (
                dense_ms
            ),

            "rrf": (
                rrf_pipeline_ms
            ),

            "rerank": (
                rerank_pipeline_ms
            ),

            "safe_rerank": (
                safe_pipeline_ms
            ),
        }

        methods = {}

        for method_name in (
            METHOD_NAMES
        ):
            results = (
                method_results[
                    method_name
                ]
            )

            methods[
                method_name
            ] = {
                "metrics": (
                    score_method(
                        results=results,
                        case=case,
                    )
                ),

                "latency_ms": (
                    method_latencies[
                        method_name
                    ]
                ),

                "retrieved_chunk_ids": [
                    result.chunk.chunk_id
                    for result in results[
                        :20
                    ]
                ],

                "retrieved_document_ids": [
                    result.chunk.document_id
                    for result in results[
                        :20
                    ]
                ],
            }

        records.append(
            {
                "case_id": (
                    case.case_id
                ),

                "question_type": (
                    case.question_type
                ),

                "query": (
                    case.query
                ),

                "answer": (
                    case.answer
                ),

                "gold_evidence_facts": (
                    list(
                        case.evidence_facts
                    )
                ),

                "gold_document_ids": (
                    list(
                        case.evidence_document_ids
                    )
                ),

                "gold_document_count": (
                    len(
                        case.evidence_document_ids
                    )
                ),

                "methods": (
                    methods
                ),

                "timing": {
                    "rerank_only_ms": (
                        rerank_only_ms
                    ),

                    "safe_fusion_only_ms": (
                        safe_fusion_only_ms
                    ),
                },
            }
        )

    summary = (
        summarize_records(
            records
        )
    )

    return {
        "dataset": (
            "MultiHop-RAG"
        ),

        "configuration": {
            "limit": limit,
            "seed": seed,

            "chunk_size": (
                chunk_size
            ),

            "overlap": (
                overlap
            ),

            "retrieval_depth": (
                retrieval_depth
            ),

            "rerank_candidate_k": (
                rerank_candidate_k
            ),

            "embedding_model": (
                embedding_model
            ),

            "reranker_model": (
                reranker_model
            ),

            "rrf_k": (
                rrf_k
            ),

            "safe_fusion_k": (
                safe_fusion_k
            ),

            "safe_rrf_weight": (
                safe_rrf_weight
            ),

            "safe_reranker_weight": (
                safe_reranker_weight
            ),
        },

        "dataset_summary": {
            "documents": (
                len(documents)
            ),

            "chunks": (
                len(chunks)
            ),

            "evaluated_cases": (
                len(selected_cases)
            ),

            "question_types": (
                dict(type_counts)
            ),
        },

        "build_time_seconds": {
            "bm25": (
                bm25_build_seconds
            ),

            "dense": (
                dense_build_seconds
            ),

            "reranker_warmup": (
                reranker_warmup_seconds
            ),
        },

        "summary": summary,

        "records": records,
    }


# ======================================================
# CLI
# ======================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MultiHop-RAG "
            "Retrieval Benchmark."
        )
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
        "--chunk-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--retrieval-depth",
        type=int,
        default=(
            DEFAULT_RETRIEVAL_DEPTH
        ),
    )

    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=(
            DEFAULT_RERANK_CANDIDATE_K
        ),
    )

    parser.add_argument(
        "--embedding-model",
        default=(
            DEFAULT_MULTIHOP_EMBEDDING_MODEL
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
        default=(
            DEFAULT_SAFE_FUSION_K
        ),
    )

    parser.add_argument(
        "--safe-rrf-weight",
        type=float,
        default=(
            DEFAULT_SAFE_RRF_WEIGHT
        ),
    )

    parser.add_argument(
        "--safe-reranker-weight",
        type=float,
        default=(
            DEFAULT_SAFE_RERANKER_WEIGHT
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "evaluation/results/"
            "multihop_retrieval_smoke_20.json"
        ),
    )

    args = parser.parse_args()

    report = run_benchmark(
        limit=args.limit,
        seed=args.seed,

        chunk_size=(
            args.chunk_size
        ),

        overlap=(
            args.overlap
        ),

        retrieval_depth=(
            args.retrieval_depth
        ),

        rerank_candidate_k=(
            args.rerank_candidate_k
        ),

        embedding_model=(
            args.embedding_model
        ),

        reranker_model=(
            args.reranker_model
        ),

        rrf_k=(
            args.rrf_k
        ),

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

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "[MultiHop Eval] "
        f"Report saved to: "
        f"{output_path}"
    )

    print()

    overall = report[
        "summary"
    ][
        "overall"
    ]

    for method_name in (
        METHOD_NAMES
    ):
        metrics = overall[
            method_name
        ]

        print(
            f"[{method_name}] "
            f"Hits@10="
            f"{metrics['hits@10']:.4f} "
            f"MAP@10="
            f"{metrics['map@10']:.4f} "
            f"MRR@10="
            f"{metrics['mrr@10']:.4f} "
            f"DocRecall@10="
            f"{metrics['doc_recall@10']:.4f} "
            f"AllEvidence@10="
            f"{metrics['all_evidence_hit@10']:.4f}"
        )


if __name__ == "__main__":
    main()