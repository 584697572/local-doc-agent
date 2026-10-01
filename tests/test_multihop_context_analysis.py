"""MultiHop Context Analysis 测试。"""

import pytest

from evaluation.multihop_context_analysis import (
    classify_failure,
    normalize_match_text,
    score_fact_coverage,
)


def test_normalize_match_text():
    assert (
        normalize_match_text(
            "Hello world\nTest"
        )
        == "HelloworldTest"
    )


def test_fact_coverage_full():
    metrics = (
        score_fact_coverage(
            retrieved_texts=[
                "This contains Fact A.",
                "This contains Fact B.",
            ],

            gold_facts=[
                "Fact A",
                "Fact B",
            ],
        )
    )

    assert (
        metrics[
            "fact_recall"
        ]
        == 1.0
    )

    assert (
        metrics[
            "all_gold_facts_present"
        ]
        is True
    )


def test_fact_coverage_partial():
    metrics = (
        score_fact_coverage(
            retrieved_texts=[
                "This contains Fact A."
            ],

            gold_facts=[
                "Fact A",
                "Fact B",
            ],
        )
    )

    assert (
        metrics[
            "fact_recall"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "all_gold_facts_present"
        ]
        is False
    )


def test_failure_classification():
    assert (
        classify_failure(
            True,
            True,
        )
        == "success"
    )

    assert (
        classify_failure(
            False,
            True,
        )
        == "reasoning_limited"
    )

    assert (
        classify_failure(
            True,
            False,
        )
        == "partial_evidence_success"
    )

    assert (
        classify_failure(
            False,
            False,
        )
        == "retrieval_limited"
    )