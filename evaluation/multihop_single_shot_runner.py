"""
MultiHop-RAG Single-shot RAG Baseline。

流程：

    已保存的 Retrieval Report
            ↓
    指定 Retrieval Method Top-K
            ↓
    LLM 一次性回答
            ↓
    QA Metrics

不执行：

    Evidence Judge
    Query Rewrite
    Second Search

支持：

    Checkpoint
    Resume
    Retry failed API cases
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from config import MODEL_NAME

from evaluation.multihop_adapter import (
    load_multihop_benchmark,
)

from evaluation.multihop_qa_metrics import (
    score_qa_answer,
)

from llm_client import client


DEFAULT_RETRIEVAL_REPORT = (
    "evaluation/results/"
    "multihop_retrieval_smoke_20.json"
)

DEFAULT_OUTPUT_PATH = (
    "evaluation/results/"
    "multihop_single_shot_smoke_20.json"
)

DEFAULT_METHOD = "safe_rerank"
DEFAULT_TOP_K = 5


QA_NUMERIC_METRICS = (
    "official_match",
    "exact_match",
    "gold_containment",
    "token_f1",
    "strict_answer_match",
)


def load_retrieval_report(
    path: str | Path,
) -> dict:
    """加载 Retrieval Benchmark Report。"""

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"找不到 Retrieval Report：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def build_chunk_lookup(
    chunk_size: int,
    overlap: int,
):
    """
    使用完全相同的 Chunk 参数
    重建 MultiHop-RAG Chunk。

    不重新加载 Embedding / Reranker。
    """

    (
        _documents,
        chunks,
        _cases,
    ) = load_multihop_benchmark(
        chunk_size=chunk_size,
        overlap=overlap,
        download=False,
    )

    return {
        chunk.chunk_id: chunk
        for chunk in chunks
    }


def build_evidence_context(
    chunk_ids: list[str],
    chunk_lookup: dict,
) -> str:
    """构造提供给 LLM 的 Evidence Context。"""

    evidence_blocks = []

    for index, chunk_id in enumerate(
        chunk_ids,
        start=1,
    ):
        chunk = chunk_lookup.get(
            chunk_id
        )

        if chunk is None:
            raise ValueError(
                f"无法找到 Chunk：{chunk_id}"
            )

        title = (
            chunk.metadata.get(
                "title",
                "",
            )
            or ""
        )

        source = (
            chunk.metadata.get(
                "source",
                "",
            )
            or ""
        )

        published_at = (
            chunk.metadata.get(
                "published_at",
                "",
            )
            or ""
        )

        evidence_blocks.append(
            (
                f"[Evidence {index}]\n"
                f"chunk_id: {chunk.chunk_id}\n"
                f"document_id: {chunk.document_id}\n"
                f"title: {title}\n"
                f"source: {source}\n"
                f"published_at: {published_at}\n"
                f"content:\n{chunk.content}"
            )
        )

    return "\n\n".join(
        evidence_blocks
    )


def extract_answer(
    content: str,
) -> str:
    """
    从 JSON Response 中提取 answer。

    非 JSON 时保留原始文本。
    """

    content = (
        content
        or ""
    ).strip()

    if not content:
        return ""

    try:
        data = json.loads(
            content
        )

        answer = data.get(
            "answer",
            "",
        )

        if isinstance(
            answer,
            str,
        ):
            return answer.strip()

    except json.JSONDecodeError:
        pass

    return content


def generate_single_shot_answer(
    query: str,
    evidence_context: str,
) -> tuple[
    str,
    float,
    dict,
]:
    """
    一次性调用 LLM 回答。

    token_usage 这里只统计：
        Final Answer 这一次 LLM Call。
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document-grounded "
                "question answering system. "

                "Answer the user's question using "
                "ONLY the provided evidence. "

                "Do not use outside knowledge. "

                "If the evidence is insufficient, "
                "return "
                "\"Insufficient information\". "

                "Keep the answer concise. "

                "Return valid JSON only, "
                "using exactly this format: "
                '{"answer": "your answer"}'
            ),
        },

        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Evidence:\n{evidence_context}"
            ),
        },
    ]

    start = time.perf_counter()

    response = (
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,

            response_format={
                "type": "json_object"
            },

            temperature=0,
        )
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    raw_content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    answer = extract_answer(
        raw_content
    )

    usage = getattr(
        response,
        "usage",
        None,
    )

    token_usage = {
        "prompt_tokens": (
            getattr(
                usage,
                "prompt_tokens",
                0,
            )
            if usage
            else 0
        ),

        "completion_tokens": (
            getattr(
                usage,
                "completion_tokens",
                0,
            )
            if usage
            else 0
        ),

        "total_tokens": (
            getattr(
                usage,
                "total_tokens",
                0,
            )
            if usage
            else 0
        ),
    }

    return (
        answer,
        latency_ms,
        token_usage,
    )


def compute_context_diagnostics(
    retrieved_document_ids: list[str],
    gold_document_ids: list[str],
    top_k: int,
) -> dict:
    """计算父文档层 Evidence Coverage。"""

    retrieved_top_k = (
        retrieved_document_ids[
            :top_k
        ]
    )

    retrieved_set = set(
        retrieved_top_k
    )

    gold_set = set(
        gold_document_ids
    )

    found = (
        retrieved_set
        & gold_set
    )

    gold_doc_recall = (
        len(found)
        / len(gold_set)
        if gold_set
        else 0.0
    )

    return {
        "gold_doc_recall": (
            gold_doc_recall
        ),

        "all_gold_docs_present": (
            bool(gold_set)
            and gold_set.issubset(
                retrieved_set
            )
        ),

        "unique_context_documents": (
            len(
                retrieved_set
            )
        ),
    }


# ======================================================
# Checkpoint
# ======================================================


def save_checkpoint(
    path: str | Path,
    metadata: dict,
    records: list[dict],
) -> None:
    """
    原子写入 Checkpoint。

    先写 .tmp，
    再 replace，
    避免中途终止留下损坏 JSON。
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def load_checkpoint(
    path: str | Path,
    expected_metadata: dict,
) -> list[dict]:
    """
    读取并验证 Checkpoint。

    防止把另一组实验的结果
    错误混入当前 Benchmark。
    """

    path = Path(
        path
    )

    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    metadata = payload.get(
        "metadata",
        {},
    )

    if metadata != expected_metadata:
        raise ValueError(
            "Checkpoint 配置与当前实验不一致，"
            "请删除旧 Checkpoint 或更换路径。"
        )

    records = payload.get(
        "records",
        [],
    )

    if not isinstance(
        records,
        list,
    ):
        raise ValueError(
            "Checkpoint records 必须是 list"
        )

    return records


# ======================================================
# Summary
# ======================================================


def average_qa_metrics(
    records: list[dict],
) -> dict:
    """计算平均 QA 指标。"""

    if not records:
        return {
            metric_name: 0.0
            for metric_name
            in QA_NUMERIC_METRICS
        }

    return {
        metric_name: mean(
            record[
                "qa_metrics"
            ][
                metric_name
            ]
            for record in records
        )
        for metric_name
        in QA_NUMERIC_METRICS
    }


def summarize_records(
    records: list[dict],
) -> dict:
    """汇总 Single-shot 实验。"""

    if not records:
        return {
            "total_cases": 0,
        }

    overall = (
        average_qa_metrics(
            records
        )
    )

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
        ] = (
            average_qa_metrics(
                type_records
            )
        )

    full_evidence_records = [
        record
        for record in records
        if record[
            "context_diagnostics"
        ][
            "all_gold_docs_present"
        ]
    ]

    incomplete_records = [
        record
        for record in records
        if not record[
            "context_diagnostics"
        ][
            "all_gold_docs_present"
        ]
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
        "total_cases": len(
            records
        ),

        "overall_qa": (
            overall
        ),

        "strict_answer_accuracy": (
            overall[
                "strict_answer_match"
            ]
        ),

        "semantic_review_candidate_count": (
            semantic_review_count
        ),

        "by_question_type": (
            by_question_type
        ),

        "context": {
            "avg_gold_doc_recall": mean(
                record[
                    "context_diagnostics"
                ][
                    "gold_doc_recall"
                ]
                for record in records
            ),

            "all_gold_docs_present_rate": (
                len(
                    full_evidence_records
                )
                / len(records)
            ),

            "avg_unique_context_documents": mean(
                record[
                    "context_diagnostics"
                ][
                    "unique_context_documents"
                ]
                for record in records
            ),
        },

        "qa_when_all_gold_docs_present": (
            average_qa_metrics(
                full_evidence_records
            )
        ),

        "qa_when_evidence_incomplete": (
            average_qa_metrics(
                incomplete_records
            )
        ),

        "full_evidence_case_count": (
            len(
                full_evidence_records
            )
        ),

        "incomplete_evidence_case_count": (
            len(
                incomplete_records
            )
        ),

        "latency": {
            "avg_llm_latency_ms": mean(
                record[
                    "llm_latency_ms"
                ]
                for record in records
            ),

            "avg_retrieval_latency_ms": mean(
                record[
                    "retrieval_latency_ms"
                ]
                for record in records
            ),

            "avg_online_pipeline_latency_ms": mean(
                (
                    record[
                        "retrieval_latency_ms"
                    ]
                    + record[
                        "llm_latency_ms"
                    ]
                )
                for record in records
            ),
        },

        # 这里只统计 Final Answer LLM Call。
        "final_answer_tokens": {
            "avg_prompt_tokens": mean(
                record[
                    "token_usage"
                ][
                    "prompt_tokens"
                ]
                for record in records
            ),

            "avg_completion_tokens": mean(
                record[
                    "token_usage"
                ][
                    "completion_tokens"
                ]
                for record in records
            ),

            "avg_total_tokens": mean(
                record[
                    "token_usage"
                ][
                    "total_tokens"
                ]
                for record in records
            ),
        },
    }


# ======================================================
# Runner
# ======================================================


def run_single_shot(
    retrieval_report_path: str | Path,
    method: str = DEFAULT_METHOD,
    top_k: int = DEFAULT_TOP_K,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> dict:
    """运行 Single-shot RAG。"""

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    if top_k > 20:
        raise ValueError(
            "当前 Retrieval Report "
            "只保存了 Top-20"
        )

    retrieval_report = (
        load_retrieval_report(
            retrieval_report_path
        )
    )

    configuration = (
        retrieval_report[
            "configuration"
        ]
    )

    print(
        "[Single-shot] "
        "Rebuilding Chunk lookup..."
    )

    chunk_lookup = (
        build_chunk_lookup(
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
        )
    )

    retrieval_records = (
        retrieval_report[
            "records"
        ]
    )

    checkpoint_metadata = {
        "experiment": (
            "multihop_single_shot"
        ),

        "model": MODEL_NAME,

        "retrieval_report": str(
            retrieval_report_path
        ),

        "method": method,

        "top_k": top_k,
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
            "[Single-shot] "
            f"Loaded checkpoint: "
            f"{len(output_by_case)} records"
        )

    total_cases = len(
        retrieval_records
    )

    for index, retrieval_record in enumerate(
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

        # 成功完成过的 Case 直接跳过。
        #
        # error != None 的 Case 会自动重试。
        if (
            existing is not None
            and existing.get(
                "error"
            ) is None
        ):
            print(
                f"[Single-shot] "
                f"{index}/{total_cases} "
                f"{case_id} "
                "[resume: skipped]"
            )

            continue

        question_type = (
            retrieval_record[
                "question_type"
            ]
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

        method_result = (
            retrieval_record[
                "methods"
            ].get(
                method
            )
        )

        if method_result is None:
            raise ValueError(
                f"{case_id} 不存在 Method={method}"
            )

        retrieved_chunk_ids = (
            method_result[
                "retrieved_chunk_ids"
            ]
        )

        retrieved_document_ids = (
            method_result[
                "retrieved_document_ids"
            ]
        )

        selected_chunk_ids = (
            retrieved_chunk_ids[
                :top_k
            ]
        )

        evidence_context = (
            build_evidence_context(
                selected_chunk_ids,
                chunk_lookup,
            )
        )

        diagnostics = (
            compute_context_diagnostics(
                retrieved_document_ids,
                gold_document_ids,
                top_k,
            )
        )

        print(
            f"[Single-shot] "
            f"{index}/{total_cases} "
            f"{case_id} "
            f"({question_type})"
        )

        error = None

        try:
            (
                prediction,
                llm_latency_ms,
                token_usage,
            ) = generate_single_shot_answer(
                query=query,
                evidence_context=(
                    evidence_context
                ),
            )

        except Exception as exc:
            prediction = ""
            llm_latency_ms = 0.0

            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            error = str(
                exc
            )

        qa_metrics = (
            score_qa_answer(
                prediction,
                gold_answer,
            )
        )

        record = {
            "case_id": (
                case_id
            ),

            "question_type": (
                question_type
            ),

            "query": query,

            "gold_answer": (
                gold_answer
            ),

            "prediction": (
                prediction
            ),

            "method": method,

            "top_k": top_k,

            "selected_chunk_ids": (
                selected_chunk_ids
            ),

            "gold_document_ids": (
                gold_document_ids
            ),

            "context_diagnostics": (
                diagnostics
            ),

            "qa_metrics": (
                qa_metrics
            ),

            "retrieval_latency_ms": (
                method_result.get(
                    "latency_ms",
                    0.0,
                )
            ),

            "llm_latency_ms": (
                llm_latency_ms
            ),

            # 仅 Final Answer Call。
            "token_usage": (
                token_usage
            ),

            "error": (
                error
            ),
        }

        output_by_case[
            case_id
        ] = record

        print(
            "[Single-shot] "
            f"Gold={gold_answer!r} "
            f"Pred={prediction!r} "
            f"Strict="
            f"{qa_metrics['strict_answer_match']:.0f}"
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
        for record in retrieval_records
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
            "single_shot_rag"
        ),

        "model": MODEL_NAME,

        "retrieval_report": str(
            retrieval_report_path
        ),

        "retrieval_method": (
            method
        ),

        "top_k": top_k,

        "scoring": {
            "primary_metric": (
                "strict_answer_match"
            ),

            "semantic_review_candidates": (
                "not automatically counted "
                "as correct"
            ),
        },

        "summary": (
            summarize_records(
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
        default=(
            DEFAULT_OUTPUT_PATH
        ),
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

    report = run_single_shot(
        retrieval_report_path=(
            args.retrieval_report
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
        "[Single-shot Summary] "
        f"StrictAccuracy="
        f"{summary['strict_answer_accuracy']:.4f}"
    )

    print(
        "[Single-shot Summary] "
        f"SemanticReviewCandidates="
        f"{summary['semantic_review_candidate_count']}"
    )

    print(
        "[Single-shot Summary] "
        f"AllGoldDocs="
        f"{summary['context']['all_gold_docs_present_rate']:.4f}"
    )

    print(
        "[Single-shot Summary] "
        f"SavedTo={output_path}"
    )


if __name__ == "__main__":
    main()