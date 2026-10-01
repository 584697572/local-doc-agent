"""Agent Eval Metrics 测试。"""

import pytest

from evaluation.agent_metrics import (
    AgentEvalRecord,
    binary_classification_metrics,
    nearest_rank_percentile,
    optional_binary_metrics,
    summarize_agent_eval,
    summarize_efficiency,
    summarize_final_answer,
)


def make_record(
    *,
    case_id: str,

    expected_retrieval: bool,
    actual_retrieval: bool,

    expected_evidence=None,
    actual_evidence=None,

    expected_rewrite=None,
    actual_rewrite=None,

    task_success=False,

    grounded_answer=None,
    citation_correct=None,

    search_calls=0,
    llm_calls=0,
    latency_ms=0.0,

    budget_exhausted=False,
):
    """
    创建测试用 AgentEvalRecord。

    避免每个测试重复填写大量字段。
    """

    return AgentEvalRecord(
        case_id=case_id,

        expected_needs_retrieval=(
            expected_retrieval
        ),

        actual_needs_retrieval=(
            actual_retrieval
        ),

        expected_evidence_sufficient=(
            expected_evidence
        ),

        actual_evidence_sufficient=(
            actual_evidence
        ),

        expected_should_rewrite=(
            expected_rewrite
        ),

        actual_rewrite_used=(
            actual_rewrite
        ),

        task_success=(
            task_success
        ),

        grounded_answer=(
            grounded_answer
        ),

        citation_correct=(
            citation_correct
        ),

        search_calls=(
            search_calls
        ),

        llm_calls=(
            llm_calls
        ),

        latency_ms=(
            latency_ms
        ),

        budget_exhausted=(
            budget_exhausted
        ),
    )


def test_binary_classification_metrics():
    """
    构造：

        TP = 2
        TN = 1
        FP = 1
        FN = 1
    """

    metrics = (
        binary_classification_metrics(
            expected_values=[
                True,
                True,
                False,
                False,
                True,
            ],

            actual_values=[
                True,
                False,
                False,
                True,
                True,
            ],
        )
    )

    assert metrics["total"] == 5

    assert metrics["tp"] == 2
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1

    assert (
        metrics["accuracy"]
        == pytest.approx(
            3 / 5
        )
    )

    assert (
        metrics["precision"]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        metrics["recall"]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        metrics["f1"]
        == pytest.approx(
            2 / 3
        )
    )


def test_binary_classification_empty():
    """
    空输入不能报错，
    所有指标返回 0。
    """

    metrics = (
        binary_classification_metrics(
            expected_values=[],
            actual_values=[],
        )
    )

    assert metrics["total"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_binary_classification_rejects_length_mismatch():
    """
    Gold 和实际结果数量必须一致。
    """

    with pytest.raises(
        ValueError
    ):
        binary_classification_metrics(
            expected_values=[
                True,
            ],
            actual_values=[],
        )


def test_optional_binary_metrics_tracks_coverage():
    """
    有三个应该执行 Evidence Judge 的 Case。

    其中：
        Case 1：执行且正确
        Case 2：执行但判断错误
        Case 3：没有真正执行

    因此：
        eligible = 3
        evaluated = 2
        coverage = 2/3
    """

    records = [
        make_record(
            case_id="1",

            expected_retrieval=True,
            actual_retrieval=True,

            expected_evidence=True,
            actual_evidence=True,
        ),

        make_record(
            case_id="2",

            expected_retrieval=True,
            actual_retrieval=True,

            expected_evidence=False,
            actual_evidence=True,
        ),

        make_record(
            case_id="3",

            expected_retrieval=True,
            actual_retrieval=False,

            expected_evidence=False,
            actual_evidence=None,
        ),

        # 这个 Case 根本不要求 Evidence Judge，
        # 不应该计入 eligible。
        make_record(
            case_id="4",

            expected_retrieval=False,
            actual_retrieval=False,

            expected_evidence=None,
            actual_evidence=None,
        ),
    ]

    metrics = (
        optional_binary_metrics(
            records=records,

            expected_field=(
                "expected_evidence_sufficient"
            ),

            actual_field=(
                "actual_evidence_sufficient"
            ),
        )
    )

    assert (
        metrics["eligible_cases"]
        == 3
    )

    assert (
        metrics["evaluated_cases"]
        == 2
    )

    assert (
        metrics["coverage"]
        == pytest.approx(
            2 / 3
        )
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1


def test_nearest_rank_percentile():
    """
    nearest-rank：

        P50 of 10 samples
            = 第 5 个

        P95 of 10 samples
            = ceil(9.5)
            = 第 10 个
    """

    values = [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
    ]

    assert (
        nearest_rank_percentile(
            values,
            0.50,
        )
        == 50.0
    )

    assert (
        nearest_rank_percentile(
            values,
            0.95,
        )
        == 100.0
    )

    assert (
        nearest_rank_percentile(
            values,
            1.0,
        )
        == 100.0
    )


def test_nearest_rank_percentile_empty():
    assert (
        nearest_rank_percentile(
            [],
            0.95,
        )
        == 0.0
    )


def test_nearest_rank_percentile_invalid_value():
    with pytest.raises(
        ValueError
    ):
        nearest_rank_percentile(
            [1.0],
            0.0,
        )

    with pytest.raises(
        ValueError
    ):
        nearest_rank_percentile(
            [1.0],
            1.1,
        )


def test_summarize_final_answer():
    records = [
        make_record(
            case_id="1",

            expected_retrieval=True,
            actual_retrieval=True,

            task_success=True,

            grounded_answer=True,
            citation_correct=True,
        ),

        make_record(
            case_id="2",

            expected_retrieval=True,
            actual_retrieval=True,

            task_success=True,

            grounded_answer=True,
            citation_correct=False,
        ),

        make_record(
            case_id="3",

            expected_retrieval=False,
            actual_retrieval=False,

            task_success=False,

            grounded_answer=False,

            # 该 Case 不要求引用。
            citation_correct=None,
        ),
    ]

    summary = (
        summarize_final_answer(
            records
        )
    )

    assert (
        summary[
            "task_success_rate"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        summary[
            "grounded_answer_rate"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        summary[
            "grounded_answer_evaluated"
        ]
        == 3
    )

    assert (
        summary[
            "citation_accuracy"
        ]
        == pytest.approx(
            1 / 2
        )
    )

    assert (
        summary[
            "citation_evaluated"
        ]
        == 2
    )


def test_summarize_efficiency():
    records = [
        make_record(
            case_id="1",

            expected_retrieval=True,
            actual_retrieval=True,

            search_calls=1,
            llm_calls=2,
            latency_ms=100.0,
        ),

        make_record(
            case_id="2",

            expected_retrieval=True,
            actual_retrieval=True,

            search_calls=3,
            llm_calls=4,
            latency_ms=300.0,

            budget_exhausted=True,
        ),
    ]

    summary = (
        summarize_efficiency(
            records
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
            "avg_llm_calls"
        ]
        == pytest.approx(
            3.0
        )
    )

    assert (
        summary[
            "avg_latency_ms"
        ]
        == pytest.approx(
            200.0
        )
    )

    # 两个值：
    # 100, 300
    #
    # P95 nearest-rank
    # = 第 2 个
    # = 300
    assert (
        summary[
            "p95_latency_ms"
        ]
        == 300.0
    )

    assert (
        summary[
            "budget_exhaustion_rate"
        ]
        == pytest.approx(
            0.5
        )
    )


def test_summarize_agent_eval():
    """
    模拟三个完整 Agent Case。
    """

    records = [
        # Case 1：
        # 正常走 Retrieval，
        # Evidence 第一次就够。
        make_record(
            case_id="1",

            expected_retrieval=True,
            actual_retrieval=True,

            expected_evidence=True,
            actual_evidence=True,

            expected_rewrite=False,
            actual_rewrite=False,

            task_success=True,

            grounded_answer=True,
            citation_correct=True,

            search_calls=1,
            llm_calls=3,
            latency_ms=100.0,
        ),

        # Case 2：
        # 本来应该 Retrieval，
        # Router 却直接回答。
        make_record(
            case_id="2",

            expected_retrieval=True,
            actual_retrieval=False,

            expected_evidence=False,
            actual_evidence=None,

            expected_rewrite=True,
            actual_rewrite=None,

            task_success=False,

            grounded_answer=False,

            search_calls=0,
            llm_calls=2,
            latency_ms=50.0,
        ),

        # Case 3：
        # 简单寒暄，本来不需要 Retrieval，
        # Agent 也正确直接回答。
        make_record(
            case_id="3",

            expected_retrieval=False,
            actual_retrieval=False,

            expected_evidence=None,
            actual_evidence=None,

            expected_rewrite=None,
            actual_rewrite=None,

            task_success=True,

            grounded_answer=None,
            citation_correct=None,

            search_calls=0,
            llm_calls=2,
            latency_ms=30.0,
        ),
    ]

    summary = (
        summarize_agent_eval(
            records
        )
    )

    assert (
        summary["total_cases"]
        == 3
    )

    router = summary[
        "router"
    ]

    assert router["tp"] == 1
    assert router["tn"] == 1
    assert router["fn"] == 1
    assert router["fp"] == 0

    assert (
        router["accuracy"]
        == pytest.approx(
            2 / 3
        )
    )

    # Evidence：
    # 两条理论上应该评价，
    # 实际只有 Case 1 真正执行。
    evidence = summary[
        "evidence_judge"
    ]

    assert (
        evidence["eligible_cases"]
        == 2
    )

    assert (
        evidence["evaluated_cases"]
        == 1
    )

    assert (
        evidence["coverage"]
        == pytest.approx(
            1 / 2
        )
    )

    final_answer = summary[
        "final_answer"
    ]

    assert (
        final_answer[
            "task_success_rate"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    efficiency = summary[
        "efficiency"
    ]

    assert (
        efficiency[
            "avg_search_calls"
        ]
        == pytest.approx(
            1 / 3
        )
    )


def test_agent_eval_record_rejects_empty_case_id():
    with pytest.raises(
        ValueError
    ):
        make_record(
            case_id="   ",

            expected_retrieval=True,
            actual_retrieval=True,
        )


def test_agent_eval_record_rejects_negative_values():
    with pytest.raises(
        ValueError
    ):
        make_record(
            case_id="bad-search",

            expected_retrieval=True,
            actual_retrieval=True,

            search_calls=-1,
        )

    with pytest.raises(
        ValueError
    ):
        make_record(
            case_id="bad-llm",

            expected_retrieval=True,
            actual_retrieval=True,

            llm_calls=-1,
        )

    with pytest.raises(
        ValueError
    ):
        make_record(
            case_id="bad-latency",

            expected_retrieval=True,
            actual_retrieval=True,

            latency_ms=-1.0,
        )


def test_abstention_rewrite_outcome_and_router_coverage():
    records = [
        AgentEvalRecord(case_id="a", expected_needs_retrieval=True,
                        actual_needs_retrieval=True, expected_abstain=True,
                        actual_abstain=True, rewrite_success=True),
        AgentEvalRecord(case_id="b", expected_needs_retrieval=True,
                        actual_needs_retrieval=None, expected_abstain=True,
                        actual_abstain=None, rewrite_success=False),
    ]
    summary = summarize_agent_eval(records)
    assert summary["router"]["coverage"] == 0.5
    assert summary["rewrite"]["success_rate"] == 0.5
    assert summary["rewrite"]["success_evaluated"] == 2
    assert summary["final_answer"]["abstention_accuracy"] == 1.0
    assert summary["final_answer"]["abstention_coverage"] == 0.5
