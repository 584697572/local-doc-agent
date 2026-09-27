"""Evidence Sufficiency Judge 测试。"""

import harness.evidence as evidence_module


def test_no_evidence_is_insufficient():
    decision = evidence_module.evaluate_evidence(
        user_query="什么是 RAG？",
        evidence=[],
    )

    assert decision.sufficient is False
    assert decision.missing_aspects


def test_sufficient_evidence(monkeypatch):
    # 用假的 Judge 输出替代真正 LLM
    monkeypatch.setattr(
        evidence_module,
        "_call_judge",
        lambda user_query, evidence_text: (
            '{'
            '"sufficient": true,'
            '"reason": "证据已经覆盖定义和作用。",'
            '"missing_aspects": []'
            '}'
        ),
    )

    decision = evidence_module.evaluate_evidence(
        user_query="什么是 RAG？",
        evidence=[
            "RAG 是检索增强生成方法。",
            "它会先检索外部知识，再生成答案。",
        ],
    )

    assert decision.sufficient is True
    assert decision.missing_aspects == []


def test_insufficient_evidence_reports_missing_aspects(
    monkeypatch,
):
    monkeypatch.setattr(
        evidence_module,
        "_call_judge",
        lambda user_query, evidence_text: (
            '{'
            '"sufficient": false,'
            '"reason": "只有定义，没有构造方法。",'
            '"missing_aspects": ['
            '"图的构造步骤",'
            '"节点和边的定义"'
            ']'
            '}'
        ),
    )

    decision = evidence_module.evaluate_evidence(
        user_query="Cluster-State Graph 是什么，怎么构造？",
        evidence=[
            "Cluster-State Graph 是一种聚类状态表示。"
        ],
    )

    assert decision.sufficient is False

    assert "图的构造步骤" in decision.missing_aspects
    assert "节点和边的定义" in decision.missing_aspects


def test_invalid_judge_output_is_insufficient(
    monkeypatch,
):
    monkeypatch.setattr(
        evidence_module,
        "_call_judge",
        lambda user_query, evidence_text: "这不是 JSON",
    )

    decision = evidence_module.evaluate_evidence(
        user_query="什么是 RAG？",
        evidence=["一些证据"],
    )

    # Judge 自己出错时不能假装证据充分
    assert decision.sufficient is False