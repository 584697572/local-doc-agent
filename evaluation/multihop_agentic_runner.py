"""
MultiHop-RAG Agentic Retrieval Benchmark。

Controlled Experiment：

Single-shot:
    First Retrieval
        ↓
    Final Answer

Agentic:
    First Retrieval
        ↓
    Evidence Judge
        ↓
    Query Rewrite
        ↓
    Search Again
        ↓
    Final Answer

支持：
    - Strict Answer Accuracy
    - Offline / Online Latency 分离
    - Checkpoint
    - Resume
"""

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, median

from config import (
    MAX_SEARCH_CALLS,
    MODEL_NAME,
)

from evaluation.multihop_adapter import (
    load_multihop_benchmark,
)

from evaluation.multihop_context_analysis import (
    score_fact_coverage,
)

from evaluation.multihop_qa_metrics import (
    score_qa_answer,
    strict_answer_correct,
)

from evaluation.multihop_single_shot_runner import (
    build_evidence_context,
    compute_context_diagnostics,
    generate_single_shot_answer,
    load_checkpoint,
    load_retrieval_report,
    save_checkpoint,
)

from harness.evidence import (
    evaluate_evidence,
)

from harness.rewrite import (
    rewrite_query,
)

from retrieval.bm25 import (
    BM25Retriever,
)

from retrieval.dense import (
    DenseRetriever,
)

from retrieval.fusion import (
    rank_preserving_rerank_fusion,
    reciprocal_rank_fusion,
)

from retrieval.reranker import (
    Reranker,
)


DEFAULT_RETRIEVAL_REPORT = (
    "evaluation/results/"
    "multihop_retrieval_smoke_20.json"
)

DEFAULT_SINGLE_SHOT_REPORT = (
    "evaluation/results/"
    "multihop_single_shot_smoke_20.json"
)

DEFAULT_OUTPUT = (
    "evaluation/results/"
    "multihop_agentic_smoke_20.json"
)

DEFAULT_METHOD = "safe_rerank"
DEFAULT_TOP_K = 5


def load_json(
    path: str | Path,
) -> dict:
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def merge_unique_chunk_ids(
    existing_ids: list[str],
    new_ids: list[str],
) -> list[str]:
    """多轮 Retrieval Evidence 去重合并。"""

    result = list(
        existing_ids
    )

    seen = set(
        existing_ids
    )

    for chunk_id in new_ids:
        if chunk_id in seen:
            continue

        seen.add(
            chunk_id
        )

        result.append(
            chunk_id
        )

    return result


def compare_answer_result(
    baseline_correct: bool,
    agentic_correct: bool,
) -> str:
    """比较 Single-shot 和 Agentic 结果。"""

    if (
        not baseline_correct
        and agentic_correct
    ):
        return "rescued"

    if (
        baseline_correct
        and not agentic_correct
    ):
        return "regressed"

    if (
        baseline_correct
        and agentic_correct
    ):
        return "stable_success"

    return "stable_failure"


def nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """nearest-rank percentile。"""

    if not values:
        return 0.0

    sorted_values = sorted(
        values
    )

    rank = math.ceil(
        percentile
        * len(sorted_values)
    )

    return sorted_values[
        rank - 1
    ]


class MultiHopSearchBackend:
    """
    Rewrite Search Backend。

    Dense Index 第一次真正需要二次搜索时
    才建立。

    Build Time 单独统计，
    不计入 Online Case Latency。
    """

    def __init__(
        self,
        chunks,
        configuration: dict,
        top_k: int,
    ):
        self.chunks = list(
            chunks
        )

        self.configuration = (
            configuration
        )

        self.top_k = top_k

        self.bm25 = None
        self.dense = None
        self.reranker = None

        self.build_seconds = 0.0

    def _ensure_ready(
        self,
    ) -> None:
        if self.dense is not None:
            return

        print(
            "[Agentic] "
            "Building rewrite-search index..."
        )

        start = time.perf_counter()

        self.bm25 = BM25Retriever(
            self.chunks
        )

        self.dense = DenseRetriever(
            chunks=self.chunks,

            model_name=(
                self.configuration[
                    "embedding_model"
                ]
            ),
        )

        self.reranker = Reranker(
            model_name=(
                self.configuration[
                    "reranker_model"
                ]
            ),
        )

        self.build_seconds = (
            time.perf_counter()
            - start
        )

        print(
            "[Agentic] "
            "Rewrite-search index ready. "
            f"BuildSeconds="
            f"{self.build_seconds:.2f}"
        )

    def search(
        self,
        query: str,
    ) -> tuple[
        list,
        float,
    ]:
        """
        Search Latency 不包含 Index Build。

        _ensure_ready 在计时开始前执行。
        """

        self._ensure_ready()

        retrieval_depth = (
            self.configuration[
                "retrieval_depth"
            ]
        )

        candidate_k = (
            self.configuration[
                "rerank_candidate_k"
            ]
        )

        start = time.perf_counter()

        bm25_results = (
            self.bm25.search(
                query,
                top_k=retrieval_depth,
            )
        )

        dense_results = (
            self.dense.search(
                query,
                top_k=retrieval_depth,
            )
        )

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

                rrf_k=(
                    self.configuration[
                        "rrf_k"
                    ]
                ),
            )
        )

        candidates = (
            rrf_results[
                :candidate_k
            ]
        )

        reranked_results = (
            self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=len(
                    candidates
                ),
            )
        )

        safe_results = (
            rank_preserving_rerank_fusion(
                rrf_results=candidates,

                reranked_results=(
                    reranked_results
                ),

                top_k=self.top_k,

                fusion_k=(
                    self.configuration[
                        "safe_fusion_k"
                    ]
                ),

                rrf_weight=(
                    self.configuration[
                        "safe_rrf_weight"
                    ]
                ),

                reranker_weight=(
                    self.configuration[
                        "safe_reranker_weight"
                    ]
                ),
            )
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            safe_results,
            latency_ms,
        )


def summarize_agentic_records(
    records: list[dict],
) -> dict:
    """汇总 Agentic Benchmark。"""

    if not records:
        return {
            "total_cases": 0,
        }

    total = len(
        records
    )

    baseline_correct_count = sum(
        1
        for record in records
        if record[
            "baseline_correct"
        ]
    )

    agentic_correct_count = sum(
        1
        for record in records
        if record[
            "agentic_correct"
        ]
    )

    comparison_counts = {}

    for record in records:
        name = record[
            "comparison"
        ]

        comparison_counts[
            name
        ] = (
            comparison_counts.get(
                name,
                0,
            )
            + 1
        )

    online_latencies = [
        record[
            "online_pipeline_latency_ms"
        ]
        for record in records
    ]

    semantic_review_count = sum(
        1
        for record in records
        if record[
            "qa_metrics"
        ][
            "semantic_review_candidate"
        ]
    )

    return {
        "total_cases": total,

        "baseline_accuracy": (
            baseline_correct_count
            / total
        ),

        "agentic_accuracy": (
            agentic_correct_count
            / total
        ),

        "accuracy_delta": (
            (
                agentic_correct_count
                - baseline_correct_count
            )
            / total
        ),

        "comparison_counts": (
            comparison_counts
        ),

        "semantic_review_candidate_count": (
            semantic_review_count
        ),

        "avg_search_calls": mean(
            record[
                "search_calls"
            ]
            for record in records
        ),

        "avg_rewrite_calls": mean(
            record[
                "rewrite_calls"
            ]
            for record in records
        ),

        "avg_llm_calls": mean(
            record[
                "llm_calls"
            ]
            for record in records
        ),

        "rewrite_trigger_rate": (
            sum(
                1
                for record in records
                if (
                    record[
                        "rewrite_calls"
                    ]
                    > 0
                )
            )
            / total
        ),

        "budget_exhaustion_rate": (
            sum(
                1
                for record in records
                if record[
                    "budget_exhausted"
                ]
            )
            / total
        ),

        "avg_final_gold_doc_recall": mean(
            record[
                "final_context"
            ][
                "gold_doc_recall"
            ]
            for record in records
        ),

        "all_gold_docs_present_rate": (
            sum(
                1
                for record in records
                if record[
                    "final_context"
                ][
                    "all_gold_docs_present"
                ]
            )
            / total
        ),

        "avg_final_gold_fact_recall": mean(
            record[
                "final_fact_coverage"
            ][
                "fact_recall"
            ]
            for record in records
        ),

        "all_gold_facts_present_rate": (
            sum(
                1
                for record in records
                if record[
                    "final_fact_coverage"
                ][
                    "all_gold_facts_present"
                ]
            )
            / total
        ),

        # 在线请求耗时：
        # 不包含一次性的 Dense Index Build。
        "avg_online_pipeline_latency_ms": mean(
            online_latencies
        ),

        "median_online_pipeline_latency_ms": median(
            online_latencies
        ),

        "p95_online_pipeline_latency_ms": (
            nearest_rank_percentile(
                online_latencies,
                0.95,
            )
        ),
    }


def run_agentic(
    retrieval_report_path: str | Path,
    single_shot_report_path: str | Path,
    method: str = DEFAULT_METHOD,
    top_k: int = DEFAULT_TOP_K,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> dict:
    """运行 Agentic MultiHop-RAG。"""

    retrieval_report = (
        load_retrieval_report(
            retrieval_report_path
        )
    )

    single_shot_report = (
        load_json(
            single_shot_report_path
        )
    )

    configuration = (
        retrieval_report[
            "configuration"
        ]
    )

    (
        _documents,
        chunks,
        _cases,
    ) = load_multihop_benchmark(
        chunk_size=(
            configuration[
                "chunk_size"
            ]
        ),

        overlap=(
            configuration[
                "overlap"
            ]
        ),

        download=False,
    )

    chunk_lookup = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    baseline_lookup = {
        record[
            "case_id"
        ]: record

        for record
        in single_shot_report[
            "records"
        ]
    }

    search_backend = (
        MultiHopSearchBackend(
            chunks=chunks,
            configuration=configuration,
            top_k=top_k,
        )
    )

    checkpoint_metadata = {
        "experiment": (
            "multihop_agentic"
        ),

        "model": MODEL_NAME,

        "retrieval_report": str(
            retrieval_report_path
        ),

        "single_shot_report": str(
            single_shot_report_path
        ),

        "method": method,

        "top_k": top_k,

        "max_search_calls": (
            MAX_SEARCH_CALLS
        ),
    }

    output_by_case = {}

    if (
        resume
        and checkpoint_path
    ):
        checkpoint_records = (
            load_checkpoint(
                checkpoint_path,
                checkpoint_metadata,
            )
        )

        output_by_case = {
            record[
                "case_id"
            ]: record
            for record
            in checkpoint_records
        }

        print(
            "[Agentic] "
            f"Loaded checkpoint: "
            f"{len(output_by_case)} records"
        )

    retrieval_records = (
        retrieval_report[
            "records"
        ]
    )

    total_cases = len(
        retrieval_records
    )

    for case_index, retrieval_record in enumerate(
        retrieval_records,
        start=1,
    ):
        case_id = (
            retrieval_record[
                "case_id"
            ]
        )

        existing = (
            output_by_case.get(
                case_id
            )
        )

        if (
            existing is not None
            and existing.get(
                "error"
            ) is None
        ):
            print(
                f"[Agentic] "
                f"{case_index}/{total_cases} "
                f"{case_id} "
                "[resume: skipped]"
            )

            continue

        execution_start = (
            time.perf_counter()
        )

        query = (
            retrieval_record[
                "query"
            ]
        )

        gold_answer = (
            retrieval_record[
                "answer"
            ]
        )

        gold_document_ids = (
            retrieval_record[
                "gold_document_ids"
            ]
        )

        gold_facts = (
            retrieval_record[
                "gold_evidence_facts"
            ]
        )

        baseline_record = (
            baseline_lookup[
                case_id
            ]
        )

        baseline_correct = (
            strict_answer_correct(
                baseline_record[
                    "prediction"
                ],
                gold_answer,
            )
        )

        first_result = (
            retrieval_record[
                "methods"
            ][
                method
            ]
        )

        accumulated_chunk_ids = list(
            first_result[
                "retrieved_chunk_ids"
            ][
                :top_k
            ]
        )

        search_queries = [
            query
        ]

        normalized_used_queries = {
            query
            .strip()
            .casefold()
        }

        search_calls = 1
        rewrite_calls = 0
        judge_calls = 0

        # 第一次 Retrieval 虽然来自缓存 Report，
        # 但这里使用它真实测得的 latency，
        # 以模拟真实线上 Pipeline。
        search_latencies = [
            first_result.get(
                "latency_ms",
                0.0,
            )
        ]

        judge_latency_ms = 0.0
        rewrite_latency_ms = 0.0

        budget_exhausted = False
        trace = []
        final_decision = None

        print()
        print(
            f"[Agentic] "
            f"{case_index}/{total_cases} "
            f"{case_id}"
        )

        # ==================================================
        # Agent Loop
        # ==================================================

        while True:
            evidence_context = (
                build_evidence_context(
                    accumulated_chunk_ids,
                    chunk_lookup,
                )
            )

            start = time.perf_counter()

            decision = (
                evaluate_evidence(
                    user_query=query,

                    evidence=[
                        evidence_context
                    ],
                )
            )

            judge_latency_ms += (
                time.perf_counter()
                - start
            ) * 1000.0

            judge_calls += 1

            final_decision = (
                decision
            )

            trace.append(
                {
                    "event": (
                        "evidence_judged"
                    ),

                    "search_count": (
                        search_calls
                    ),

                    "sufficient": (
                        decision.sufficient
                    ),

                    "reason": (
                        decision.reason
                    ),

                    "missing_aspects": (
                        decision.missing_aspects
                    ),
                }
            )

            if decision.sufficient:
                break

            if (
                search_calls
                >= MAX_SEARCH_CALLS
            ):
                budget_exhausted = True
                break

            # ==============================================
            # Rewrite
            # ==============================================

            start = time.perf_counter()

            rewrite_result = (
                rewrite_query(
                    original_query=query,

                    missing_aspects=(
                        decision.missing_aspects
                    ),

                    used_queries=set(
                        search_queries
                    ),
                )
            )

            rewrite_latency_ms += (
                time.perf_counter()
                - start
            ) * 1000.0

            rewrite_calls += 1

            rewritten_query = (
                rewrite_result.query
                .strip()
            )

            trace.append(
                {
                    "event": (
                        "query_rewritten"
                    ),

                    "query": (
                        rewritten_query
                    ),

                    "reason": (
                        rewrite_result.reason
                    ),
                }
            )

            if not rewritten_query:
                break

            normalized_rewrite = (
                rewritten_query
                .casefold()
            )

            if (
                normalized_rewrite
                in normalized_used_queries
            ):
                trace.append(
                    {
                        "event": (
                            "duplicate_query_rejected"
                        ),

                        "query": (
                            rewritten_query
                        ),
                    }
                )

                break

            normalized_used_queries.add(
                normalized_rewrite
            )

            search_queries.append(
                rewritten_query
            )

            # ==============================================
            # Search Again
            # ==============================================

            try:
                (
                    new_results,
                    search_latency_ms,
                ) = (
                    search_backend.search(
                        rewritten_query
                    )
                )

            except Exception as exc:
                trace.append(
                    {
                        "event": (
                            "search_failed"
                        ),

                        "query": (
                            rewritten_query
                        ),

                        "error": str(
                            exc
                        ),
                    }
                )

                break

            search_calls += 1

            search_latencies.append(
                search_latency_ms
            )

            new_chunk_ids = [
                result.chunk.chunk_id
                for result
                in new_results
            ]

            accumulated_chunk_ids = (
                merge_unique_chunk_ids(
                    accumulated_chunk_ids,
                    new_chunk_ids,
                )
            )

            trace.append(
                {
                    "event": (
                        "search_completed"
                    ),

                    "query": (
                        rewritten_query
                    ),

                    "new_chunk_ids": (
                        new_chunk_ids
                    ),

                    "total_evidence_chunks": (
                        len(
                            accumulated_chunk_ids
                        )
                    ),
                }
            )

        # ==================================================
        # Final Answer
        # ==================================================

        final_evidence = (
            build_evidence_context(
                accumulated_chunk_ids,
                chunk_lookup,
            )
        )

        try:
            (
                prediction,
                final_answer_latency_ms,
                token_usage,
            ) = (
                generate_single_shot_answer(
                    query=query,

                    evidence_context=(
                        final_evidence
                    ),
                )
            )

            final_error = None

        except Exception as exc:
            prediction = ""

            final_answer_latency_ms = (
                0.0
            )

            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            final_error = str(
                exc
            )

        qa_metrics = (
            score_qa_answer(
                prediction,
                gold_answer,
            )
        )

        agentic_correct = (
            qa_metrics[
                "strict_answer_match"
            ]
            == 1.0
        )

        comparison = (
            compare_answer_result(
                baseline_correct,
                agentic_correct,
            )
        )

        final_document_ids = [
            chunk_lookup[
                chunk_id
            ].document_id
            for chunk_id
            in accumulated_chunk_ids
        ]

        final_context = (
            compute_context_diagnostics(
                final_document_ids,
                gold_document_ids,
                len(
                    final_document_ids
                ),
            )
        )

        final_texts = [
            chunk_lookup[
                chunk_id
            ].content
            for chunk_id
            in accumulated_chunk_ids
        ]

        final_fact_coverage = (
            score_fact_coverage(
                final_texts,
                gold_facts,
            )
        )

        llm_calls = (
            judge_calls
            + rewrite_calls
            + 1
        )

        # 核心修复：
        #
        # Online Latency 使用各在线组件相加，
        # 不包含一次性的 Dense Index Build。
        online_pipeline_latency_ms = (
            sum(
                search_latencies
            )
            + judge_latency_ms
            + rewrite_latency_ms
            + final_answer_latency_ms
        )

        # Benchmark 程序真实墙钟时间，
        # 仅用于 Debug，不用于线上性能声明。
        execution_wall_time_ms = (
            (
                time.perf_counter()
                - execution_start
            )
            * 1000.0
        )

        record = {
            "case_id": (
                case_id
            ),

            "question_type": (
                retrieval_record[
                    "question_type"
                ]
            ),

            "query": query,

            "gold_answer": (
                gold_answer
            ),

            "baseline_prediction": (
                baseline_record[
                    "prediction"
                ]
            ),

            "agentic_prediction": (
                prediction
            ),

            "baseline_correct": (
                baseline_correct
            ),

            "agentic_correct": (
                agentic_correct
            ),

            "comparison": (
                comparison
            ),

            "qa_metrics": (
                qa_metrics
            ),

            "search_calls": (
                search_calls
            ),

            "rewrite_calls": (
                rewrite_calls
            ),

            "judge_calls": (
                judge_calls
            ),

            "llm_calls": (
                llm_calls
            ),

            "search_queries": (
                search_queries
            ),

            "final_evidence_sufficient": (
                final_decision.sufficient
                if final_decision
                else False
            ),

            "budget_exhausted": (
                budget_exhausted
            ),

            "final_chunk_ids": (
                accumulated_chunk_ids
            ),

            "final_context": (
                final_context
            ),

            "final_fact_coverage": (
                final_fact_coverage
            ),

            "timing": {
                "search_latency_ms": (
                    sum(
                        search_latencies
                    )
                ),

                "judge_latency_ms": (
                    judge_latency_ms
                ),

                "rewrite_latency_ms": (
                    rewrite_latency_ms
                ),

                "final_answer_latency_ms": (
                    final_answer_latency_ms
                ),
            },

            "online_pipeline_latency_ms": (
                online_pipeline_latency_ms
            ),

            "execution_wall_time_ms": (
                execution_wall_time_ms
            ),

            # 注意：
            # 目前只统计 Final Answer Token。
            "token_usage": (
                token_usage
            ),

            "trace": (
                trace
            ),

            "error": (
                final_error
            ),
        }

        output_by_case[
            case_id
        ] = record

        print(
            "[Agentic] "
            f"Gold={gold_answer!r} "
            f"Pred={prediction!r}"
        )

        print(
            "[Agentic] "
            f"SearchCalls={search_calls} "
            f"Comparison={comparison}"
        )

        if checkpoint_path:
            save_checkpoint(
                checkpoint_path,
                checkpoint_metadata,
                list(
                    output_by_case.values()
                ),
            )

    ordered_records = [
        output_by_case[
            record[
                "case_id"
            ]
        ]

        for record
        in retrieval_records

        if (
            record[
                "case_id"
            ]
            in output_by_case
        )
    ]

    return {
        "dataset": (
            "MultiHop-RAG"
        ),

        "experiment": (
            "agentic_rag"
        ),

        "model": MODEL_NAME,

        "max_search_calls": (
            MAX_SEARCH_CALLS
        ),

        "retrieval_method": (
            method
        ),

        "retrieval_top_k": (
            top_k
        ),

        # Offline preprocessing cost。
        "rewrite_search_index_build_seconds": (
            search_backend.build_seconds
        ),

        "latency_definition": {
            "offline": (
                "rewrite_search_index_build_seconds"
            ),

            "online": (
                "retrieval + judge + rewrite "
                "+ final answer; "
                "index build excluded"
            ),
        },

        "scoring": {
            "primary_metric": (
                "strict_answer_match"
            ),

            "semantic_review_candidates": (
                "reported separately"
            ),
        },

        "summary": (
            summarize_agentic_records(
                ordered_records
            )
        ),

        "records": (
            ordered_records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retrieval-report",
        default=(
            DEFAULT_RETRIEVAL_REPORT
        ),
    )

    parser.add_argument(
        "--single-shot-report",
        default=(
            DEFAULT_SINGLE_SHOT_REPORT
        ),
    )

    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    checkpoint_path = (
        args.checkpoint
        or (
            args.output
            + ".checkpoint.json"
        )
    )

    report = run_agentic(
        retrieval_report_path=(
            args.retrieval_report
        ),

        single_shot_report_path=(
            args.single_shot_report
        ),

        method=args.method,

        top_k=args.top_k,

        checkpoint_path=(
            checkpoint_path
        ),

        resume=args.resume,
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

    summary = report[
        "summary"
    ]

    print()
    print(
        "[Agentic Summary] "
        f"BaselineAccuracy="
        f"{summary['baseline_accuracy']:.4f}"
    )

    print(
        "[Agentic Summary] "
        f"AgenticAccuracy="
        f"{summary['agentic_accuracy']:.4f}"
    )

    print(
        "[Agentic Summary] "
        f"Delta="
        f"{summary['accuracy_delta']:+.4f}"
    )

    print(
        "[Agentic Summary] "
        f"Comparison="
        f"{summary['comparison_counts']}"
    )

    print(
        "[Agentic Summary] "
        f"AvgSearchCalls="
        f"{summary['avg_search_calls']:.3f}"
    )

    print(
        "[Agentic Summary] "
        f"BudgetExhaustion="
        f"{summary['budget_exhaustion_rate']:.3f}"
    )

    print(
        "[Agentic Summary] "
        f"AvgOnlineLatencyMs="
        f"{summary['avg_online_pipeline_latency_ms']:.1f}"
    )

    print(
        "[Agentic Summary] "
        f"P95OnlineLatencyMs="
        f"{summary['p95_online_pipeline_latency_ms']:.1f}"
    )

    print(
        "[Agentic Summary] "
        f"OfflineIndexBuildSeconds="
        f"{report['rewrite_search_index_build_seconds']:.2f}"
    )

    print(
        "[Agentic Summary] "
        f"SavedTo={output_path}"
    )


if __name__ == "__main__":
    main()