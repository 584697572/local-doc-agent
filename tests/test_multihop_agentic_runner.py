"""MultiHop Agentic Runner 测试。"""

import pytest

from evaluation.multihop_agentic_runner import (
    compare_answer_result,
    merge_unique_chunk_ids,
    nearest_rank_percentile,
    summarize_agentic_records,
)


def make_record(
    *,
    baseline_correct: bool,
    agentic_correct: bool,
    comparison: str,
    search_calls: int,
    rewrite_calls: int,
    budget_exhausted: bool,
    online_latency_ms: float,
):
    return {
        "baseline_correct": (
            baseline_correct
        ),

        "agentic_correct": (
            agentic_correct
        ),

        "comparison": (
            comparison
        ),

        "search_calls": (
            search_calls
        ),

        "rewrite_calls": (
            rewrite_calls
        ),

        "judge_calls": (
            search_calls
        ),

        "llm_calls": (
            search_calls
            + rewrite_calls
            + 1
        ),

        "budget_exhausted": (
            budget_exhausted
        ),

        "final_context": {
            "gold_doc_recall": 1.0,
            "all_gold_docs_present": True,
        },

        "final_fact_coverage": {
            "fact_recall": 0.5,
            "all_gold_facts_present": False,
        },

        "qa_metrics": {
            "semantic_review_candidate": False,
        },

        "online_pipeline_latency_ms": (
            online_latency_ms
        ),
    }


def test_merge_unique_chunk_ids():
    assert (
        merge_unique_chunk_ids(
            [
                "a",
                "b",
            ],

            [
                "b",
                "c",
            ],
        )
        == [
            "a",
            "b",
            "c",
        ]
    )


def test_compare_answer_result():
    assert (
        compare_answer_result(
            False,
            True,
        )
        == "rescued"
    )

    assert (
        compare_answer_result(
            True,
            False,
        )
        == "regressed"
    )

    assert (
        compare_answer_result(
            True,
            True,
        )
        == "stable_success"
    )

    assert (
        compare_answer_result(
            False,
            False,
        )
        == "stable_failure"
    )


def test_nearest_rank_percentile():
    assert (
        nearest_rank_percentile(
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ],

            0.95,
        )
        == 40.0
    )


def test_summarize_agentic_records():
    records = [
        make_record(
            baseline_correct=True,
            agentic_correct=True,
            comparison=(
                "stable_success"
            ),

            search_calls=1,
            rewrite_calls=0,

            budget_exhausted=False,

            online_latency_ms=1000.0,
        ),

        make_record(
            baseline_correct=False,
            agentic_correct=True,
            comparison="rescued",

            search_calls=2,
            rewrite_calls=1,

            budget_exhausted=False,

            online_latency_ms=2000.0,
        ),

        make_record(
            baseline_correct=False,
            agentic_correct=False,
            comparison=(
                "stable_failure"
            ),

            search_calls=3,
            rewrite_calls=2,

            budget_exhausted=True,

            online_latency_ms=3000.0,
        ),
    ]

    summary = (
        summarize_agentic_records(
            records
        )
    )

    assert (
        summary[
            "baseline_accuracy"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        summary[
            "agentic_accuracy"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        summary[
            "accuracy_delta"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        summary[
            "avg_search_calls"
        ]
        == pytest.approx(
            2.0
        )
    )

    assert (
        summary[
            "budget_exhaustion_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        summary[
            "avg_online_pipeline_latency_ms"
        ]
        == pytest.approx(
            2000.0
        )
    )

    assert (
        summary[
            "median_online_pipeline_latency_ms"
        ]
        == pytest.approx(
            2000.0
        )
    )

    assert (
        summary[
            "p95_online_pipeline_latency_ms"
        ]
        == pytest.approx(
            3000.0
        )
    )