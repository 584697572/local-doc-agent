"""内部 fixture 的完整性、字段校验和路径隔离测试。"""

from collections import Counter
import copy
import json

import pytest

from evaluation.agent_dataset import (
    CATEGORIES, DEFAULT_CASES, KNOWLEDGE_BASE, case_from_dict, load_cases, read_excerpt,
)


def sample():
    return json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[5])


def test_suite_has_forty_cases_and_eight_documents():
    cases = load_cases()
    assert len(cases) == len({c.case_id for c in cases}) == 40
    assert Counter(c.category for c in cases) == {name: 5 for name in CATEGORIES}
    docs = list(KNOWLEDGE_BASE.glob("*.md"))
    assert len(docs) == 8
    assert all(300 <= len(p.read_text(encoding="utf-8")) <= 1800 for p in docs)
    assert all(c.gold.first_evidence_sufficient is None for c in cases if "live" in c.modes)


@pytest.mark.parametrize("field,value", [
    ("needs_retrieval", "true"), ("citation_required", 1),
    ("min_search_calls", -1), ("max_search_calls", False),
    ("min_search_calls", 4), ("required_facts", "bad"),
    ("fact_patterns", []), ("required_pages", {"ghost.md": 7}),
    ("required_sources", ["../secret.md"]),
])
def test_rejects_invalid_gold(field, value):
    raw = sample()
    raw["gold"][field] = value
    with pytest.raises(ValueError):
        case_from_dict(raw)


def test_missing_field_and_unknown_field():
    raw = sample()
    del raw["question"]
    with pytest.raises(ValueError):
        case_from_dict(raw)
    raw = sample()
    raw["typo"] = True
    with pytest.raises(ValueError):
        case_from_dict(raw)


def test_duplicate_id_and_bad_json_have_line_numbers(tmp_path):
    path = tmp_path / "cases.jsonl"
    line = json.dumps(sample(), ensure_ascii=False)
    path.write_text(line + "\n" + line, encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:.*case_id"):
        load_cases(path)
    path.write_text(line + "\n{", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        load_cases(path)


def test_empty_dataset_and_missing_source(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)
    raw = sample()
    raw["gold"]["required_sources"] = ["missing.md"]
    with pytest.raises(ValueError):
        case_from_dict(raw)


def test_scripted_gold_does_not_change_live_gold():
    case = next(c for c in load_cases() if c.case_id == "rewrite_001")
    assert case.expectations("scripted").first_evidence_sufficient is False
    assert case.expectations("scripted").min_search_calls == 2
    assert case.expectations("live").first_evidence_sufficient is None
    assert case.expectations("live").should_rewrite is None
    assert case.expectations("live").min_search_calls == 1


def test_script_source_and_section_must_exist():
    raw = sample()
    raw["script"]["searches"][0]["excerpts"][0]["section"] = "不存在"
    with pytest.raises(ValueError):
        case_from_dict(raw)
    with pytest.raises(ValueError):
        read_excerpt({"source": "../private.md", "section": "发布"})
    with pytest.raises(ValueError):
        read_excerpt({"source": "project_orion.md", "section": "发布", "page": 0})
