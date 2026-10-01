"""主动注入错误，确认硬约束和失败归因能够检测退化。"""

from dataclasses import replace

from evaluation.agent_dataset import load_cases
from evaluation.agent_failure_analysis import (
    ABSTENTION, analyze_failures, check_citations, check_hard_invariants,
    classify_failure, review_scripted,
)


def base_record():
    return {
        "case_id": "demo_001", "executions": [], "trace": [], "used_queries": [],
        "search_calls": 0, "max_search_calls": 3, "actual_needs_retrieval": False,
        "expected_needs_retrieval": False, "expected_abstain": False,
        "unknown_citations": [], "wrong_pages": [], "missing_citations": [],
        "missing_pages": [], "citation_correct": None, "unsupported_claims": [],
        "task_success": True, "hard_invariants": [], "error": None,
        "expected_evidence_sufficient": None, "actual_evidence_sufficient": None,
        "expected_should_rewrite": False, "actual_rewrite_used": False,
        "budget_exhausted": False, "facts_covered": [], "rewrite_calls": 0,
    }


def test_all_hard_invariants_detect_corruption():
    record = base_record()
    record.update(
        executions=[{"query": "A", "status": "success"}] * 4,
        trace=[{"event": "search_accepted", "query": "a"}] * 4,
        search_calls=4, used_queries=["a"], unknown_citations=["ghost.md"],
        expected_abstain=True, unsupported_claims=["未经支持的事实"],
    )
    violations = check_hard_invariants(record)
    assert {"search_budget_exceeded", "duplicate_query_accepted", "duplicate_query_executed",
            "citation_outside_evidence", "unsupported_claim_when_abstaining"} <= set(violations)


def test_budget_exhaustion_is_not_itself_budget_violation():
    record = base_record()
    record.update(
        search_calls=3, used_queries=["a", "b", "c"], budget_exhausted=True,
        executions=[{"query": q, "status": "empty"} for q in "abc"],
        trace=[{"event": "search_accepted", "query": q} for q in "abc"],
    )
    assert check_hard_invariants(record) == []


def test_rejected_query_and_untracked_evidence_are_detected():
    record = base_record()
    record["trace"] = [
        {"event": "rewrite_search_rejected", "query": "a", "reason": "duplicate_query"},
        {"event": "tool_executed", "query": "a"},
    ]
    record["evidence"] = ["不属于任何成功工具结果"]
    violations = check_hard_invariants(record)
    assert "rejected_duplicate_executed" in violations
    assert "evidence_not_from_successful_tool" in violations


def test_router_and_judge_failure_precede_answer_error():
    record = base_record()
    record.update(task_success=False, expected_needs_retrieval=True, facts_covered=[False])
    assert classify_failure(record)["primary_failure"] == "router_false_negative"
    record.update(
        actual_needs_retrieval=True, expected_evidence_sufficient=False,
        actual_evidence_sufficient=True,
    )
    result = classify_failure(record)
    assert result["primary_failure"] == "evidence_false_positive"
    assert "answer_incomplete" in result["secondary_failures"]


def test_missing_retrieval_is_not_mislabeled_as_wrong_judge():
    record = base_record()
    record.update(task_success=False, expected_needs_retrieval=True, actual_needs_retrieval=True,
                  expected_evidence_sufficient=False, actual_evidence_sufficient=False,
                  missing_evidence_sources=["project_orion.md"])
    assert classify_failure(record)["primary_failure"] == "retrieval_miss"


def test_citations_require_actual_source_and_associated_page():
    case = next(c for c in load_cases() if c.case_id == "citation_004")
    executions = [{"status": "success", "metadata": {"sources": [
        {"filename": "deployment_notes.md", "page": 7},
        {"filename": "project_orion.md", "page": None},
    ]}}]
    assert check_citations("[deployment_notes.md，第7页] 8081", executions, case.gold)["citation_correct"]
    result = check_citations("[deployment_notes.md，第8页] 8081", executions, case.gold)
    assert not result["citation_correct"] and result["wrong_pages"]
    result = check_citations("[deployment_notes.md] [project_orion.md，第7页]", executions, case.gold)
    assert result["missing_pages"] == ["deployment_notes.md"]
    assert check_citations("[ghost.md] 8081", executions, case.gold)["unknown_citations"] == ["ghost.md"]


def test_grounding_checks_actual_answer_not_gold_or_script():
    gold = next(c.gold for c in load_cases() if c.case_id == "citation_001")
    evidence = "Orion 负责人代号为 O17。"
    good = review_scripted("[project_orion.md] Orion 负责人代号为 O17。", evidence, gold)
    bad = review_scripted("[project_orion.md] Orion 负责人代号为 O17。收入为一亿元。", evidence, gold)
    assert good["grounded_answer"]
    assert not bad["grounded_answer"]
    assert bad["unsupported_claims"] == ["收入为一亿元"]


def test_cross_reference_in_document_is_not_a_claimed_citation():
    gold = next(c.gold for c in load_cases() if c.case_id == "citation_001")
    executions = [{"status": "success", "metadata": {"sources": [
        {"filename": "project_orion.md", "page": None},
    ]}}]
    result = check_citations(
        "[project_orion.md] O17。部署内容另见 deployment_notes.md。",
        executions, gold,
    )
    assert result["citation_correct"]
    assert result["cited_sources"] == ["project_orion.md"]
    assert check_citations("来源：ghost.md", executions, gold)["unknown_citations"] == ["ghost.md"]


def test_negated_forbidden_claim_is_not_an_affirmation():
    gold = next(c.gold for c in load_cases() if c.case_id == "grounded_001")
    review = review_scripted("不能断言Orion一定运行在Linux上。", "", gold)
    assert review["forbidden_claims_present"] == []


def test_control_fallback_is_not_fabrication_but_not_abstention():
    gold = next(c.gold for c in load_cases() if c.case_id == "budget_004")
    review = review_scripted("达到最大工具调用轮数，已停止执行。", "", gold)
    assert review["unsupported_claims"] == []
    assert not review["abstained"]
    assert review_scripted(ABSTENTION, "", gold)["abstained"]


def test_successful_fault_handling_is_only_counted_as_event():
    record = base_record()
    record["trace"] = [{"event": "rewrite_search_rejected", "reason": "duplicate_query"}]
    record.update(classify_failure(record))
    report = analyze_failures([record])
    assert report["failure_counts"] == {}
    assert report["event_counts"]["rewrite_search_rejected"] == 1
