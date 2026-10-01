"""MultiHop-RAG QA Metrics 测试。"""

import pytest

from evaluation.multihop_qa_metrics import (
    binary_answer_match,
    exact_match,
    gold_containment,
    normalize_answer,
    official_match,
    prediction_contained_in_gold,
    score_qa_answer,
    semantic_review_candidate,
    strict_answer_correct,
    token_f1,
)


def test_normalize_answer():
    assert (
        normalize_answer(
            "  Google, Inc.!  "
        )
        == "google inc"
    )


def test_official_match_is_intentionally_loose():
    """
    官方风格指标非常宽松。

    Sam Altman
    和
    Sam Bankman-Fried

    因为共享 Sam，
    仍然返回成功。
    """

    assert (
        official_match(
            "Sam Altman",
            "Sam Bankman-Fried",
        )
        == 1.0
    )


def test_exact_match():
    assert (
        exact_match(
            "Google!",
            "google",
        )
        == 1.0
    )

    assert (
        exact_match(
            "The answer is Google",
            "Google",
        )
        == 0.0
    )


def test_gold_containment_uses_token_boundaries():
    assert (
        gold_containment(
            "The company is Google.",
            "Google",
        )
        == 1.0
    )

    # "no" 不能错误匹配 "not"。
    assert (
        gold_containment(
            "This is not supported.",
            "no",
        )
        == 0.0
    )


def test_binary_answer_match():
    assert (
        binary_answer_match(
            "Yes, the reports agree.",
            "Yes",
        )
        == 1.0
    )

    assert (
        binary_answer_match(
            "No. They disagree.",
            "Yes",
        )
        == 0.0
    )

    assert (
        binary_answer_match(
            "Google",
            "Google",
        )
        is None
    )


def test_strict_answer_correct_yes_no():
    assert (
        strict_answer_correct(
            "No. The amounts are equal.",
            "no",
        )
        is True
    )

    assert (
        strict_answer_correct(
            "Not enough evidence.",
            "no",
        )
        is False
    )


def test_strict_answer_correct_entity():
    assert (
        strict_answer_correct(
            "The company is Google.",
            "Google",
        )
        is True
    )

    assert (
        strict_answer_correct(
            "Sam Altman",
            "Sam Bankman-Fried",
        )
        is False
    )


def test_prediction_contained_in_gold():
    assert (
        prediction_contained_in_gold(
            "Everton",
            "Everton Football Club",
        )
        == 1.0
    )


def test_semantic_review_candidate():
    """
    Everton 是合理简称候选，
    但 strict 不自动判对。
    """

    assert (
        strict_answer_correct(
            "Everton",
            "Everton Football Club",
        )
        is False
    )

    assert (
        semantic_review_candidate(
            "Everton",
            "Everton Football Club",
        )
        is True
    )

    assert (
        semantic_review_candidate(
            "Sam Altman",
            "Sam Bankman-Fried",
        )
        is False
    )


def test_token_f1():
    assert (
        token_f1(
            "Sam Bankman-Fried",
            "Sam Bankman-Fried",
        )
        == 1.0
    )

    assert (
        token_f1(
            "Sam Altman",
            "Sam Bankman-Fried",
        )
        == pytest.approx(
            0.5
        )
    )


def test_score_qa_answer():
    metrics = score_qa_answer(
        prediction=(
            "Yes, the reports are consistent."
        ),
        gold="Yes",
    )

    assert (
        metrics[
            "strict_answer_match"
        ]
        == 1.0
    )

    assert (
        metrics[
            "binary_answer_match"
        ]
        == 1.0
    )

    assert (
        metrics[
            "semantic_review_candidate"
        ]
        is False
    )