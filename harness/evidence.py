"""判断当前检索证据是否足以回答用户问题。"""

import json
from dataclasses import dataclass, field

from config import MODEL_NAME
from llm_client import client


@dataclass
class EvidenceDecision:
    """Evidence Judge 的判断结果。"""

    # 当前证据是否已经足够回答
    sufficient: bool

    # 为什么够 / 为什么不够
    reason: str

    # 如果不够，具体缺少哪些信息
    missing_aspects: list[str] = field(default_factory=list)


def _call_judge(
    user_query: str,
    evidence_text: str,
) -> str:
    """
    调用 LLM 判断证据是否充分。
    这个 LLM 只负责判断，不负责回答用户问题。
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 Evidence Sufficiency Judge。"
                    "你的任务只是判断当前证据是否足以回答用户问题，"
                    "不要直接回答用户问题。"

                    "只有当证据能够覆盖用户问题中的主要信息需求时，"
                    "sufficient 才能为 true。"

                    "不要使用证据之外的常识补全缺失信息。"

                    "必须返回 JSON，格式如下："
                    "{"
                    '"sufficient": true或false,'
                    '"reason": "判断原因",'
                    '"missing_aspects": ["缺失信息1", "缺失信息2"]'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{user_query}\n\n"
                    f"当前检索证据：\n{evidence_text}"
                ),
            },
        ],

        # 要求模型返回 JSON
        response_format={
            "type": "json_object"
        },

        # Judge 尽量稳定，不需要创造性
        temperature=0,
    )

    return response.choices[0].message.content or "{}"


def evaluate_evidence(
    user_query: str,
    evidence: list[str],
) -> EvidenceDecision:
    """
    判断当前所有 Evidence 是否足够回答用户问题。
    """

    # 完全没有证据时，不需要调用 LLM
    if not evidence:
        return EvidenceDecision(
            sufficient=False,
            reason="当前没有任何检索证据。",
            missing_aspects=[
                "需要先检索与用户问题相关的证据"
            ],
        )

    # 多次搜索得到的 Evidence 合并给 Judge
    evidence_text = "\n\n".join(evidence)

    try:
        raw_result = _call_judge(
            user_query=user_query,
            evidence_text=evidence_text,
        )

        data = json.loads(raw_result)

    except Exception as exc:
        return EvidenceDecision(
            sufficient=False,
            reason=f"Evidence Judge 执行失败：{exc}",
            missing_aspects=[],
        )

    missing_aspects = data.get(
        "missing_aspects",
        [],
    )

    # 防止模型偶尔返回错误的数据类型
    if not isinstance(missing_aspects, list):
        missing_aspects = []

    return EvidenceDecision(
        sufficient=data.get("sufficient") is True,
        reason=str(
            data.get("reason", "未提供判断原因")
        ),
        missing_aspects=missing_aspects,
    )