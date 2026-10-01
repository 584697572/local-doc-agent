"""脚本全量、Live 依赖接线、隔离和 CLI 的回归测试。"""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from evaluation.agent_dataset import KNOWLEDGE_BASE, load_cases, read_excerpt
from evaluation.agent_runner import (
    ScriptedBackend, exit_status, load_agent_modules, main, run_suite, select_cases,
)
from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult


@pytest.fixture(scope="module")
def scripted_report():
    return run_suite(load_cases(), "scripted")


def test_scripted_suite_runs_all_forty_without_network(scripted_report):
    report = scripted_report
    assert len(report["records"]) == report["summary"]["total_cases"] == 40
    assert report["configuration"]["simulated"] is True
    assert report["summary"]["hard_gate_passed"]
    assert all(r["assessment_llm_calls"] == 0 and not r["error"] for r in report["records"])
    assert report["configuration"]["safe_rerank_weights"] == [1.0, 2.0]
    assert len(report["configuration"]["knowledge_base_sha256"]) == 8


def test_trace_rewrite_and_duplicate_rejection(scripted_report):
    records = {r["case_id"]: r for r in scripted_report["records"]}
    rewrite = records["rewrite_001"]
    assert rewrite["search_calls"] == 2
    assert rewrite["actual_first_evidence_sufficient"] is False
    assert rewrite["actual_evidence_sufficient"] is True
    assert rewrite["rewrite_success"]
    duplicate = records["budget_002"]
    assert len(duplicate["executions"]) == 1
    assert any(e.get("reason") == "duplicate_query" for e in duplicate["trace"])
    assert duplicate["actual_abstain"]
    assert records["multihop_003"]["search_calls"] == 2
    assert records["multihop_003"]["rewrite_calls"] == 0


def test_errors_and_empty_searches_do_not_crash_runner(scripted_report):
    records = {r["case_id"]: r for r in scripted_report["records"]}
    for case_id, status in (("budget_004", "empty"), ("budget_005", "error")):
        record = records[case_id]
        assert record["error"] is None
        assert record["search_calls"] <= 3
        assert all(c["status"] == status for c in record["executions"])
        assert record["evidence"] == []
        assert record["actual_evidence_sufficient"] is None


def test_llm_usage_counts_all_stages(scripted_report):
    records = {r["case_id"]: r for r in scripted_report["records"]}
    assert records["router_001"]["llm_calls"] == 2
    assert records["singlehop_001"]["llm_calls"] == 4
    assert records["rewrite_001"]["llm_calls"] == 6
    assert all(r["llm_calls"] == r["logical_llm_calls"] for r in scripted_report["records"])


def test_gold_is_not_used_to_generate_scripted_answer():
    case = next(c for c in load_cases() if c.case_id == "citation_001")
    altered = replace(case, gold=replace(case.gold, required_facts=["错误日期"], fact_patterns=[["2099"]]))
    report = run_suite([altered])
    record = report["records"][0]
    assert "O17" in record["final_answer"]
    assert record["facts_covered"] == [False]
    assert not record["task_success"]


def test_rewrite_terms_do_not_force_unnecessary_rewrite():
    case = next(c for c in load_cases() if c.case_id == "rewrite_001")
    altered = replace(
        case, scripted_gold={},
        script={**case.script,
                "searches": [{"status": "success", "excerpts": case.script["answer_sections"]}],
                "judges": [{"sufficient": True, "missing_aspects": []}], "rewrites": []},
    )
    record = run_suite([altered])["records"][0]
    assert record["task_success"]
    assert not record["actual_rewrite_used"]


def test_unknown_citation_fails_hard_gate():
    case = next(c for c in load_cases() if c.case_id == "citation_001")
    altered = replace(case, script={**case.script, "answer": "[ghost.md] Orion 负责人代号为 O17。"})
    report = run_suite([altered])
    assert "citation_outside_evidence" in report["records"][0]["hard_invariants"]
    assert exit_status(report) == 1


def test_missing_tool_call_is_not_accepted():
    case = next(c for c in load_cases() if c.case_id == "singlehop_003")
    altered = replace(case, script={**case.script, "missing_tool_calls": 1})
    record = run_suite([altered])["records"][0]
    assert any(e["event"] == "missing_required_tool_call" for e in record["trace"])
    assert record["search_calls"] == 1
    assert "未检索回答" not in record["final_answer"]


def test_one_case_exception_does_not_abort_rest(monkeypatch):
    modules = load_agent_modules()
    original = modules["agent"].run_agent_with_trace
    cases = load_cases()[:2]

    def fail_first(question, history, **kwargs):
        if question == cases[0].question:
            raise RuntimeError("injected runner failure")
        return original(question, history, **kwargs)

    monkeypatch.setattr(modules["agent"], "run_agent_with_trace", fail_first)
    report = run_suite(cases)
    assert report["records"][0]["error"]
    assert report["records"][1]["task_success"]
    assert report["summary"]["router"]["coverage"] == 0.5
    assert exit_status(report) == 1


def test_live_mode_uses_fixture_engine_and_restores_globals(monkeypatch):
    modules = load_agent_modules()
    retrieval = modules["retrieval"]
    sentinel = object()
    monkeypatch.setattr(retrieval, "_engine", sentinel)
    original_dir = retrieval.DATA_DIR
    case = next(c for c in load_cases() if c.case_id == "singlehop_003")
    backend = ScriptedBackend(case, KNOWLEDGE_BASE)
    ref = case.script["searches"][0]["excerpts"][0]

    class FakeEngine:
        def search(self, query, **kwargs):
            assert retrieval.DATA_DIR == KNOWLEDGE_BASE
            assert retrieval._engine is None
            return [RetrievalResult(DocumentChunk(
                chunk_id="orion:release", document_id="orion",
                content=read_excerpt(ref), filename=ref["source"],
            ), score=1.0, rank=1)]

    monkeypatch.setattr(retrieval, "_get_retrieval_engine", lambda: FakeEngine())

    def create(**kwargs):
        if "内部回归评审员" in kwargs["messages"][0]["content"]:
            payload = json.loads(kwargs["messages"][1]["content"])
            assert payload["question"] == case.question
            data = {"facts_covered": [True], "grounded_answer": True, "unsupported_claims": [],
                    "forbidden_claims_present": [], "abstained": False,
                    "evidence_sufficient": True, "citation_supported": True}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
        return backend.create(**kwargs)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    report = run_suite([case], "live", live_client=client)
    record = report["records"][0]
    assert record["task_success"]
    assert record["assessment_llm_calls"] == 1 and record["llm_calls"] == 4
    assert retrieval._engine is sentinel and retrieval.DATA_DIR == original_dir
    assert modules["agent"].client is not client


def test_live_review_failure_is_not_counted_as_success(monkeypatch):
    import evaluation.agent_runner as runner
    modules = load_agent_modules()
    case = load_cases()[0]
    backend = ScriptedBackend(case, KNOWLEDGE_BASE)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=backend.create)))
    monkeypatch.setattr(runner, "review_live", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad review")))
    report = run_suite([case], "live", live_client=client)
    assert report["records"][0]["assessment_error"]
    assert not report["records"][0]["task_success"]
    assert exit_status(report) == 2


def test_case_filtering_and_mode_skips():
    cases = load_cases()
    assert len(select_cases(cases, categories=["citation"], limit=2)) == 2
    assert select_cases(cases, case_ids=["rewrite_001"])[0].case_id == "rewrite_001"
    with pytest.raises(ValueError):
        select_cases(cases, case_ids=["missing_999"])
    with pytest.raises(ValueError):
        select_cases(cases, limit=0)
    with pytest.raises(ValueError):
        run_suite([cases[0], cases[0]])


def test_live_skips_script_only_cases_without_claiming_results():
    cases = [c for c in load_cases() if c.case_id in {"budget_001", "citation_004"}]
    # 所有 case 都跳过时，此客户端不应收到任何请求。
    report = run_suite(cases, "live", live_client=object())
    assert report["records"] == []
    assert len(report["skipped"]) == 2
    assert exit_status(report) == 2


def test_cli_writes_report_and_compact_baseline(tmp_path):
    output, baseline = tmp_path / "run.json", tmp_path / "baseline.json"
    status = main(["--mode", "scripted", "--case-id", "router_001",
                   "--output", str(output), "--baseline-output", str(baseline)])
    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(baseline.read_text(encoding="utf-8"))
    assert len(report["records"]) == 1
    assert "records" not in summary and "configuration" in summary
