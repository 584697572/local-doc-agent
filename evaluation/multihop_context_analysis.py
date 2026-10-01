"""
MultiHop-RAG Context / Failure Analysis。

不调用 LLM。
不重新运行 Retrieval。

分析：

    Parent Document Coverage
    Gold Fact Coverage
    Strict Answer Correctness

Failure Types：

    success
    reasoning_limited
    retrieval_limited
    partial_evidence_success
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from evaluation.multihop_adapter import (
    load_multihop_benchmark,
)

from evaluation.multihop_qa_metrics import (
    semantic_review_candidate,
    strict_answer_correct,
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
    "multihop_context_analysis_20.json"
)


def normalize_match_text(
    text: str,
) -> str:
    """
    Gold Fact 严格字符串匹配标准化。

    注意：
    这不是语义匹配。
    """

    return (
        str(text)
        .replace(" ", "")
        .replace("\n", "")
    )


def score_fact_coverage(
    retrieved_texts: list[str],
    gold_facts: list[str],
) -> dict:
    """计算 Gold Fact Coverage。"""

    normalized_chunks = [
        normalize_match_text(
            text
        )
        for text
        in retrieved_texts
    ]

    normalized_facts = [
        normalize_match_text(
            fact
        )
        for fact
        in gold_facts
        if str(
            fact
        ).strip()
    ]

    if not normalized_facts:
        return {
            "matched_fact_count": 0,
            "gold_fact_count": 0,
            "fact_recall": 0.0,
            "any_gold_fact_present": False,
            "all_gold_facts_present": False,
        }

    matched_count = 0

    for gold_fact in (
        normalized_facts
    ):
        found = any(
            gold_fact
            in chunk_text

            for chunk_text
            in normalized_chunks
        )

        if found:
            matched_count += 1

    return {
        "matched_fact_count": (
            matched_count
        ),

        "gold_fact_count": (
            len(
                normalized_facts
            )
        ),

        "fact_recall": (
            matched_count
            / len(
                normalized_facts
            )
        ),

        "any_gold_fact_present": (
            matched_count > 0
        ),

        "all_gold_facts_present": (
            matched_count
            == len(
                normalized_facts
            )
        ),
    }


def classify_failure(
    answer_correct: bool,
    all_gold_facts_present: bool,
) -> str:
    """基础 Failure Classification。"""

    if (
        all_gold_facts_present
        and answer_correct
    ):
        return "success"

    if (
        all_gold_facts_present
        and not answer_correct
    ):
        return "reasoning_limited"

    if (
        not all_gold_facts_present
        and answer_correct
    ):
        return (
            "partial_evidence_success"
        )

    return "retrieval_limited"


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


def run_analysis(
    retrieval_report_path: str | Path,
    single_shot_report_path: str | Path,
) -> dict:
    """执行离线 Failure Analysis。"""

    retrieval_report = (
        load_json(
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

    single_lookup = {
        record[
            "case_id"
        ]: record

        for record
        in single_shot_report[
            "records"
        ]
    }

    analysis_records = []

    for retrieval_record in (
        retrieval_report[
            "records"
        ]
    ):
        case_id = (
            retrieval_record[
                "case_id"
            ]
        )

        single_record = (
            single_lookup.get(
                case_id
            )
        )

        if single_record is None:
            raise ValueError(
                f"Single-shot 缺少 "
                f"Case={case_id}"
            )

        retrieved_texts = []

        for chunk_id in (
            single_record[
                "selected_chunk_ids"
            ]
        ):
            chunk = (
                chunk_lookup.get(
                    chunk_id
                )
            )

            if chunk is None:
                raise ValueError(
                    f"找不到 Chunk：{chunk_id}"
                )

            retrieved_texts.append(
                chunk.content
            )

        fact_metrics = (
            score_fact_coverage(
                retrieved_texts,
                retrieval_record[
                    "gold_evidence_facts"
                ],
            )
        )

        gold_answer = (
            retrieval_record[
                "answer"
            ]
        )

        prediction = (
            single_record[
                "prediction"
            ]
        )

        answer_correct = (
            strict_answer_correct(
                prediction,
                gold_answer,
            )
        )

        review_candidate = (
            semantic_review_candidate(
                prediction,
                gold_answer,
            )
        )

        failure_type = (
            classify_failure(
                answer_correct=(
                    answer_correct
                ),

                all_gold_facts_present=(
                    fact_metrics[
                        "all_gold_facts_present"
                    ]
                ),
            )
        )

        analysis_records.append(
            {
                "case_id": (
                    case_id
                ),

                "question_type": (
                    retrieval_record[
                        "question_type"
                    ]
                ),

                "query": (
                    retrieval_record[
                        "query"
                    ]
                ),

                "gold_answer": (
                    gold_answer
                ),

                "prediction": (
                    prediction
                ),

                "document_coverage": (
                    single_record[
                        "context_diagnostics"
                    ]
                ),

                "fact_coverage": (
                    fact_metrics
                ),

                "strict_answer_correct": (
                    answer_correct
                ),

                "semantic_review_candidate": (
                    review_candidate
                ),

                "failure_type": (
                    failure_type
                ),
            }
        )

    total = len(
        analysis_records
    )

    failure_counts = {}

    for record in (
        analysis_records
    ):
        name = record[
            "failure_type"
        ]

        failure_counts[
            name
        ] = (
            failure_counts.get(
                name,
                0,
            )
            + 1
        )

    parent_doc_complete_count = sum(
        1
        for record
        in analysis_records

        if record[
            "document_coverage"
        ][
            "all_gold_docs_present"
        ]
    )

    all_fact_count = sum(
        1
        for record
        in analysis_records

        if record[
            "fact_coverage"
        ][
            "all_gold_facts_present"
        ]
    )

    doc_complete_fact_incomplete = sum(
        1
        for record
        in analysis_records

        if (
            record[
                "document_coverage"
            ][
                "all_gold_docs_present"
            ]

            and not record[
                "fact_coverage"
            ][
                "all_gold_facts_present"
            ]
        )
    )

    strict_correct_count = sum(
        1
        for record
        in analysis_records

        if record[
            "strict_answer_correct"
        ]
    )

    review_count = sum(
        1
        for record
        in analysis_records

        if record[
            "semantic_review_candidate"
        ]
    )

    return {
        "dataset": (
            "MultiHop-RAG"
        ),

        "experiment": (
            "context_failure_analysis"
        ),

        "total_cases": (
            total
        ),

        "summary": {
            "strict_answer_accuracy": (
                strict_correct_count
                / total
                if total
                else 0.0
            ),

            "semantic_review_candidate_count": (
                review_count
            ),

            "avg_gold_fact_recall": (
                mean(
                    record[
                        "fact_coverage"
                    ][
                        "fact_recall"
                    ]

                    for record
                    in analysis_records
                )
                if total
                else 0.0
            ),

            "all_gold_facts_present_rate": (
                all_fact_count
                / total
                if total
                else 0.0
            ),

            "all_gold_docs_present_rate": (
                parent_doc_complete_count
                / total
                if total
                else 0.0
            ),

            "doc_complete_but_fact_incomplete": (
                doc_complete_fact_incomplete
            ),

            "failure_counts": (
                failure_counts
            ),
        },

        "records": (
            analysis_records
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
        "--output",
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    report = run_analysis(
        retrieval_report_path=(
            args.retrieval_report
        ),

        single_shot_report_path=(
            args.single_shot_report
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

    summary = report[
        "summary"
    ]

    print()
    print(
        "[Context Analysis] "
        f"Cases={report['total_cases']}"
    )

    print(
        "[Context Analysis] "
        f"StrictAccuracy="
        f"{summary['strict_answer_accuracy']:.4f}"
    )

    print(
        "[Context Analysis] "
        f"SemanticReviewCandidates="
        f"{summary['semantic_review_candidate_count']}"
    )

    print(
        "[Context Analysis] "
        f"GoldFactRecall="
        f"{summary['avg_gold_fact_recall']:.4f}"
    )

    print(
        "[Context Analysis] "
        f"AllGoldFacts="
        f"{summary['all_gold_facts_present_rate']:.4f}"
    )

    print(
        "[Context Analysis] "
        f"AllGoldDocs="
        f"{summary['all_gold_docs_present_rate']:.4f}"
    )

    print(
        "[Context Analysis] "
        "DocCompleteFactIncomplete="
        f"{summary['doc_complete_but_fact_incomplete']}"
    )

    print(
        "[Context Analysis] "
        f"Failures="
        f"{summary['failure_counts']}"
    )

    print(
        "[Context Analysis] "
        f"SavedTo={output_path}"
    )


if __name__ == "__main__":
    main()