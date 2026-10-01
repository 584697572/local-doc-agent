"""
Agent Eval 的基础指标与数据结构。

Retrieval Eval 关注：
    “正确文档有没有被检索出来？”

Agent Eval 关注：
    “Agent 在整个决策链上有没有做对？”

例如：
    Router 是否正确决定 Retrieval；
    Evidence Judge 是否正确判断证据充分性；
    是否在需要时执行 Rewrite；
    最终任务是否成功；
    回答是否有证据支撑；
    工具和 LLM 调用了多少次。
"""

from dataclasses import dataclass
from math import ceil
from statistics import mean


@dataclass
class AgentEvalRecord:
    """
    一条 Agent Benchmark Case 的执行结果。

    expected_*：
        测试集中的人工 Gold 行为。

    actual_*：
        Agent 实际执行出的行为。

    None：
        表示这个指标对当前 Case 不适用，
        或该阶段没有真正执行。
    """

    case_id: str

    # ==================================================
    # Router
    # ==================================================

    expected_needs_retrieval: bool
    actual_needs_retrieval: bool | None

    # ==================================================
    # Evidence Sufficiency Judge
    # ==================================================

    expected_evidence_sufficient: bool | None = None
    actual_evidence_sufficient: bool | None = None

    # ==================================================
    # Query Rewrite
    # ==================================================

    expected_should_rewrite: bool | None = None
    actual_rewrite_used: bool | None = None

    # ==================================================
    # Final Answer
    # ==================================================

    task_success: bool = False

    # 只有需要依据知识库回答的 Case
    # 才需要评价 Groundedness。
    grounded_answer: bool | None = None

    # 只有要求引用的 Case
    # 才需要评价 Citation。
    citation_correct: bool | None = None

    # ==================================================
    # Efficiency
    # ==================================================

    search_calls: int = 0
    llm_calls: int = 0
    latency_ms: float = 0.0

    # 是否因为搜索预算耗尽而停止。
    budget_exhausted: bool = False

    # None 表示未评价，不能把缺失结果当成成功。
    expected_abstain: bool | None = None
    actual_abstain: bool | None = None
    rewrite_success: bool | None = None

    def __post_init__(self):
        """防止错误的 Eval 数据污染最终指标。"""

        if not self.case_id.strip():
            raise ValueError(
                "case_id 不能为空"
            )

        if self.search_calls < 0:
            raise ValueError(
                "search_calls 不能小于 0"
            )

        if self.llm_calls < 0:
            raise ValueError(
                "llm_calls 不能小于 0"
            )

        if self.latency_ms < 0:
            raise ValueError(
                "latency_ms 不能小于 0"
            )


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """安全除法，避免分母为 0。"""

    if denominator == 0:
        return 0.0

    return float(
        numerator
    ) / float(
        denominator
    )


def binary_classification_metrics(
    expected_values: list[bool],
    actual_values: list[bool],
) -> dict:
    """
    计算二分类指标。

    对 Router 来说：

        Positive = needs_retrieval=True

    因此：

        TP：
            应该 Retrieval，
            实际也 Retrieval。

        FN：
            应该 Retrieval，
            实际却没有 Retrieval。

    对 LocalDoc-Agent 来说，
    Router FN 通常尤其危险：

        应该查知识库
        ↓
        Agent 却直接回答
        ↓
        可能产生无证据回答
    """

    if len(expected_values) != len(
        actual_values
    ):
        raise ValueError(
            "expected_values 和 "
            "actual_values 长度必须一致"
        )

    total = len(
        expected_values
    )

    if total == 0:
        return {
            "total": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for expected, actual in zip(
        expected_values,
        actual_values,
    ):
        if expected and actual:
            tp += 1

        elif (
            not expected
            and not actual
        ):
            tn += 1

        elif (
            not expected
            and actual
        ):
            fp += 1

        else:
            fn += 1

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "total": total,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def optional_binary_metrics(
    records: list[AgentEvalRecord],
    expected_field: str,
    actual_field: str,
) -> dict:
    """
    对 Evidence / Rewrite 这类可选阶段计算指标。

    和普通 binary metrics 相比，
    这里额外记录 coverage。

    例如某条 Case 理论上应该执行 Evidence Judge，
    但因为前面的 Router 就失败了，
    Evidence Judge 根本没有执行。

    不能简单把这条 Case 从世界上“消失”，
    所以需要同时记录：

        eligible_cases
        evaluated_cases
        coverage
    """

    expected_values = []
    actual_values = []

    eligible_cases = 0

    for record in records:
        expected = getattr(
            record,
            expected_field,
        )

        actual = getattr(
            record,
            actual_field,
        )

        # Gold 为 None：
        # 这个阶段本来就不适用于该 Case。
        if expected is None:
            continue

        eligible_cases += 1

        # Gold 有值但 actual=None：
        # 说明本来应该评价，
        # 但 Agent 没有真正走到这一阶段。
        if actual is None:
            continue

        expected_values.append(
            bool(expected)
        )

        actual_values.append(
            bool(actual)
        )

    metrics = (
        binary_classification_metrics(
            expected_values,
            actual_values,
        )
    )

    evaluated_cases = len(
        expected_values
    )

    metrics.update(
        {
            "eligible_cases": (
                eligible_cases
            ),

            "evaluated_cases": (
                evaluated_cases
            ),

            "coverage": safe_divide(
                evaluated_cases,
                eligible_cases,
            ),
        }
    )

    return metrics


def nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """
    使用 nearest-rank 方法计算 Percentile。

    例如：

        P95 latency

    nearest-rank 定义：

        rank = ceil(P * N)

    10 个样本的 P95：
        ceil(0.95 * 10)
        = 第 10 个值。
    """

    if not values:
        return 0.0

    if (
        percentile <= 0
        or percentile > 1
    ):
        raise ValueError(
            "percentile 必须在 (0, 1] 范围内"
        )

    sorted_values = sorted(
        float(value)
        for value in values
    )

    rank = ceil(
        percentile
        * len(sorted_values)
    )

    # Python 下标从 0 开始。
    index = rank - 1

    return sorted_values[
        index
    ]


def summarize_final_answer(
    records: list[AgentEvalRecord],
) -> dict:
    """
    汇总最终任务表现。

    task_success：
        所有 Case 都参与。

    groundedness / citation：
        只统计实际有 Gold 标注的 Case。
    """

    if not records:
        return {
            "task_success_rate": 0.0,

            "grounded_answer_rate": 0.0,
            "grounded_answer_evaluated": 0,

            "citation_accuracy": 0.0,
            "citation_evaluated": 0,
        }

    task_success_count = sum(
        1
        for record in records
        if record.task_success
    )

    grounded_values = [
        record.grounded_answer
        for record in records
        if (
            record.grounded_answer
            is not None
        )
    ]

    citation_values = [
        record.citation_correct
        for record in records
        if (
            record.citation_correct
            is not None
        )
    ]

    grounded_success = sum(
        1
        for value in grounded_values
        if value
    )

    citation_success = sum(
        1
        for value in citation_values
        if value
    )

    return {
        "task_success_rate": (
            safe_divide(
                task_success_count,
                len(records),
            )
        ),

        "grounded_answer_rate": (
            safe_divide(
                grounded_success,
                len(grounded_values),
            )
        ),

        "grounded_answer_evaluated": (
            len(grounded_values)
        ),

        "citation_accuracy": (
            safe_divide(
                citation_success,
                len(citation_values),
            )
        ),

        "citation_evaluated": (
            len(citation_values)
        ),
    }


def summarize_efficiency(
    records: list[AgentEvalRecord],
) -> dict:
    """
    汇总 Agent 的执行成本。

    当前第一版关注：

        Search 调用次数
        LLM 调用次数
        平均延迟
        P95 延迟
        Budget Exhaustion

    后面还可以继续增加：
        token usage
        monetary cost
    """

    if not records:
        return {
            "avg_search_calls": 0.0,
            "avg_llm_calls": 0.0,

            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,

            "budget_exhaustion_rate": 0.0,
        }

    search_calls = [
        record.search_calls
        for record in records
    ]

    llm_calls = [
        record.llm_calls
        for record in records
    ]

    latencies = [
        record.latency_ms
        for record in records
    ]

    exhausted_count = sum(
        1
        for record in records
        if record.budget_exhausted
    )

    return {
        "avg_search_calls": mean(
            search_calls
        ),

        "avg_llm_calls": mean(
            llm_calls
        ),

        "avg_latency_ms": mean(
            latencies
        ),

        "p95_latency_ms": (
            nearest_rank_percentile(
                latencies,
                0.95,
            )
        ),

        "budget_exhaustion_rate": (
            safe_divide(
                exhausted_count,
                len(records),
            )
        ),
    }


def summarize_agent_eval(
    records: list[AgentEvalRecord],
) -> dict:
    """
    汇总完整 Agent Eval。

    Agent Runner 后面只需要负责：

        运行测试 Case
        ↓
        构造 AgentEvalRecord

    本模块负责：

        AgentEvalRecord[]
        ↓
        汇总成最终 Benchmark 指标
    """

    # ==================================================
    # Router
    # ==================================================

    # 路由阶段本身未执行时，也报告 coverage，避免把未知结果当成 False。
    router_metrics = optional_binary_metrics(
        records, "expected_needs_retrieval", "actual_needs_retrieval"
    )

    # ==================================================
    # Evidence Judge
    # ==================================================

    evidence_metrics = (
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

    # ==================================================
    # Rewrite Decision
    # ==================================================

    rewrite_metrics = (
        optional_binary_metrics(
            records=records,

            expected_field=(
                "expected_should_rewrite"
            ),

            actual_field=(
                "actual_rewrite_used"
            ),
        )
    )

    # ==================================================
    # Final Answer
    # ==================================================

    final_answer_metrics = (
        summarize_final_answer(
            records
        )
    )

    # ==================================================
    # Efficiency
    # ==================================================

    efficiency_metrics = (
        summarize_efficiency(
            records
        )
    )

    abstention = optional_binary_metrics(
        records, "expected_abstain", "actual_abstain"
    )
    rewrite_outcomes = [r.rewrite_success for r in records if r.rewrite_success is not None]
    rewrite_metrics["success_rate"] = safe_divide(sum(rewrite_outcomes), len(rewrite_outcomes))
    rewrite_metrics["success_evaluated"] = len(rewrite_outcomes)
    final_answer_metrics["abstention_accuracy"] = abstention["accuracy"]
    final_answer_metrics["abstention_coverage"] = abstention["coverage"]

    return {
        "total_cases": len(
            records
        ),

        "router": router_metrics,

        "evidence_judge": (
            evidence_metrics
        ),

        "rewrite": (
            rewrite_metrics
        ),

        "final_answer": (
            final_answer_metrics
        ),

        "efficiency": (
            efficiency_metrics
        ),
    }
