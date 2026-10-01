"""
MultiHop-RAG QA 指标。

我们同时保留：

1. official_match
   尽量对齐 MultiHop-RAG 官方宽松评分。

2. exact_match
   标准化后必须完全一致。

3. gold_containment
   Gold Token 序列完整出现在 Prediction 中。

4. token_f1
   衡量部分答案重叠。

5. strict_answer_match
   本项目主要自动评分指标：

       Yes / No：
           判断回答首个 Token 是否与 Gold 一致。

       其他答案：
           要求完整 Gold Token 序列出现在 Prediction 中。

6. semantic_review_candidate
   标记类似：

       Gold:
           Everton Football Club

       Prediction:
           Everton

   这种“严格自动评分没通过，
   但预测可能是合理简称”的情况。

   它只用于后续人工 / LLM Judge，
   不自动判正确。
"""

import re
from collections import Counter


BINARY_ANSWERS = {
    "yes",
    "no",
}


def normalize_answer(
    text: str,
) -> str:
    """
    标准化答案：

    - lowercase
    - 去掉大部分标点
    - 保留连字符
    - 合并多余空格
    """

    text = str(
        text
    ).lower()

    text = re.sub(
        r"[^\w\s-]",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def answer_tokens(
    text: str,
) -> list[str]:
    """取得标准化后的答案 Token。"""

    normalized = normalize_answer(
        text
    )

    if not normalized:
        return []

    return normalized.split()


def _contains_token_sequence(
    haystack: list[str],
    needle: list[str],
) -> bool:
    """
    判断 needle 是否作为连续 Token 序列
    出现在 haystack 中。

    使用 Token，而不是字符串 substring。

    因此：

        Gold = "no"
        Prediction = "not enough"

    不会错误认为 no 出现在 not 中。
    """

    if not needle:
        return False

    if len(
        needle
    ) > len(
        haystack
    ):
        return False

    window_size = len(
        needle
    )

    for start in range(
        len(haystack)
        - window_size
        + 1
    ):
        if (
            haystack[
                start:
                start + window_size
            ]
            == needle
        ):
            return True

    return False


def official_match(
    prediction: str,
    gold: str,
) -> float:
    """
    MultiHop-RAG 官方风格宽松指标。

    Prediction 与 Gold
    只要存在任意一个空格分隔 Token 交集，
    就记为成功。

    这个指标很宽松，
    因此不能单独作为最终 Accuracy。
    """

    prediction_tokens = set(
        str(
            prediction
        ).lower().split()
    )

    gold_tokens = set(
        str(
            gold
        ).lower().split()
    )

    if not gold_tokens:
        return 0.0

    return float(
        bool(
            prediction_tokens
            & gold_tokens
        )
    )


def exact_match(
    prediction: str,
    gold: str,
) -> float:
    """标准化后完全相同。"""

    prediction_normalized = (
        normalize_answer(
            prediction
        )
    )

    gold_normalized = (
        normalize_answer(
            gold
        )
    )

    if not gold_normalized:
        return 0.0

    return float(
        prediction_normalized
        == gold_normalized
    )


def gold_containment(
    prediction: str,
    gold: str,
) -> float:
    """
    完整 Gold Token 序列
    是否出现在 Prediction 中。

    例如：

        Gold:
            Google

        Prediction:
            The company is Google.

    → 成功。
    """

    prediction_tokens = (
        answer_tokens(
            prediction
        )
    )

    gold_tokens = (
        answer_tokens(
            gold
        )
    )

    return float(
        _contains_token_sequence(
            prediction_tokens,
            gold_tokens,
        )
    )


def prediction_contained_in_gold(
    prediction: str,
    gold: str,
) -> float:
    """
    Prediction 是否是 Gold 的连续子序列。

    主要用来发现简称：

        Prediction:
            Everton

        Gold:
            Everton Football Club

    注意：

    这里只标记“值得语义复核”，
    绝不自动判断正确。
    """

    prediction_tokens = (
        answer_tokens(
            prediction
        )
    )

    gold_tokens = (
        answer_tokens(
            gold
        )
    )

    return float(
        _contains_token_sequence(
            gold_tokens,
            prediction_tokens,
        )
    )


def binary_answer_match(
    prediction: str,
    gold: str,
) -> float | None:
    """
    专门评价 Yes / No。

    返回：
        1.0 / 0.0：
            Gold 是 Yes / No。

        None：
            这不是二分类答案。

    只读取 Prediction 的第一个标准化 Token。

    因此：

        Gold = Yes
        Prediction = "Yes, because ..."

    → 正确。
    """

    gold_normalized = (
        normalize_answer(
            gold
        )
    )

    if (
        gold_normalized
        not in BINARY_ANSWERS
    ):
        return None

    prediction_tokens = (
        answer_tokens(
            prediction
        )
    )

    if not prediction_tokens:
        return 0.0

    return float(
        prediction_tokens[0]
        == gold_normalized
    )


def strict_answer_correct(
    prediction: str,
    gold: str,
) -> bool:
    """
    LocalDoc-Agent 当前主要自动正确性规则。

    Yes / No：
        使用 binary_answer_match。

    其他答案：
        Gold 必须完整出现在 Prediction 中。

    不使用模糊实体匹配，
    避免把：

        Sam

    自动当成：

        Sam Bankman-Fried

    的正确答案。
    """

    binary_result = (
        binary_answer_match(
            prediction,
            gold,
        )
    )

    if binary_result is not None:
        return (
            binary_result
            == 1.0
        )

    return (
        gold_containment(
            prediction,
            gold,
        )
        == 1.0
    )


def semantic_review_candidate(
    prediction: str,
    gold: str,
) -> bool:
    """
    找出值得进一步做语义 Judge 的答案。

    当前规则：

        strict 已正确
            → 不需要复核。

        Prediction 是 Gold 的一部分
            → 标记复核。

    例如：
        Everton
        vs
        Everton Football Club
    """

    if strict_answer_correct(
        prediction,
        gold,
    ):
        return False

    prediction_tokens = (
        answer_tokens(
            prediction
        )
    )

    if not prediction_tokens:
        return False

    return (
        prediction_contained_in_gold(
            prediction,
            gold,
        )
        == 1.0
    )


def token_f1(
    prediction: str,
    gold: str,
) -> float:
    """
    Prediction / Gold Token F1。

    使用 multiset overlap，
    防止重复 Token 被错误重复计分。
    """

    prediction_tokens = (
        answer_tokens(
            prediction
        )
    )

    gold_tokens = (
        answer_tokens(
            gold
        )
    )

    if (
        not prediction_tokens
        or not gold_tokens
    ):
        return 0.0

    prediction_counter = Counter(
        prediction_tokens
    )

    gold_counter = Counter(
        gold_tokens
    )

    overlap = sum(
        (
            prediction_counter
            & gold_counter
        ).values()
    )

    if overlap == 0:
        return 0.0

    precision = (
        overlap
        / len(
            prediction_tokens
        )
    )

    recall = (
        overlap
        / len(
            gold_tokens
        )
    )

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def score_qa_answer(
    prediction: str,
    gold: str,
) -> dict:
    """
    一次性计算所有 QA 指标。
    """

    binary_result = (
        binary_answer_match(
            prediction,
            gold,
        )
    )

    strict_correct = (
        strict_answer_correct(
            prediction,
            gold,
        )
    )

    return {
        "official_match": (
            official_match(
                prediction,
                gold,
            )
        ),

        "exact_match": (
            exact_match(
                prediction,
                gold,
            )
        ),

        "gold_containment": (
            gold_containment(
                prediction,
                gold,
            )
        ),

        "token_f1": (
            token_f1(
                prediction,
                gold,
            )
        ),

        "binary_answer_match": (
            binary_result
        ),

        "strict_answer_match": float(
            strict_correct
        ),

        "semantic_review_candidate": (
            semantic_review_candidate(
                prediction,
                gold,
            )
        ),
    }