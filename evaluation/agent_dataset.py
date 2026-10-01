"""内部诊断 Case 的数据契约；读取时先检查数据，避免带错 Gold 运行。"""

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
import re

SUITE_VERSION = "1.0.0"
CASE_DIR = Path(__file__).parent / "agent_cases"
DEFAULT_CASES = CASE_DIR / "cases.jsonl"
KNOWLEDGE_BASE = CASE_DIR / "knowledge_base"
CATEGORIES = (
    "router", "singlehop", "rewrite_required", "multihop",
    "unanswerable", "budget_control", "groundedness", "citation",
)


@dataclass(frozen=True)
class Gold:
    needs_retrieval: bool
    first_evidence_sufficient: bool | None
    should_rewrite: bool | None
    final_evidence_sufficient: bool | None
    min_search_calls: int
    max_search_calls: int
    required_facts: list[str]
    required_sources: list[str]
    forbidden_claims: list[str]
    citation_required: bool
    should_abstain: bool
    # 每个信息点允许多个关键表达，不要求整句与标准答案完全相同。
    fact_patterns: list[list[str]] = field(default_factory=list)
    required_pages: dict[str, int] = field(default_factory=dict)
    rewrite_terms: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name in ("needs_retrieval", "citation_required", "should_abstain"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须是 bool")
        for name in ("first_evidence_sufficient", "should_rewrite", "final_evidence_sufficient"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须是 bool 或 null")
        if any(type(n) is not int for n in (self.min_search_calls, self.max_search_calls)):
            raise ValueError("搜索次数必须是整数")
        if not 0 <= self.min_search_calls <= self.max_search_calls:
            raise ValueError("搜索次数必须满足 0 <= min <= max")
        if not self.needs_retrieval and self.max_search_calls:
            raise ValueError("Direct Answer 不应要求搜索")
        for name in ("required_facts", "required_sources", "forbidden_claims", "rewrite_terms"):
            value = getattr(self, name)
            if not isinstance(value, list) or any(not isinstance(s, str) or not s.strip() for s in value):
                raise ValueError(f"{name} 必须是非空字符串组成的列表")
        for term in self.rewrite_terms:
            re.compile(term)
        if len(set(self.required_sources)) != len(self.required_sources):
            raise ValueError("required_sources 不得重复")
        if not isinstance(self.fact_patterns, list) or len(self.fact_patterns) != len(self.required_facts):
            raise ValueError("每个 required_fact 都需要一组 fact_patterns")
        for patterns in self.fact_patterns:
            if not isinstance(patterns, list) or not patterns:
                raise ValueError("fact_patterns 每组不能为空")
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern:
                    raise ValueError("匹配式不能为空")
                re.compile(pattern)
        if not isinstance(self.required_pages, dict):
            raise ValueError("required_pages 必须是字典")
        for source, page in self.required_pages.items():
            if source not in self.required_sources or type(page) is not int or page < 1:
                raise ValueError("页码必须为 required_sources 中来源的正整数页码")
        for source in self.required_sources:
            if Path(source).name != source or "/" in source or "\\" in source:
                raise ValueError("来源必须是固定知识库内的文件名")


@dataclass(frozen=True)
class AgentCase:
    case_id: str
    category: str
    question: str
    gold: Gold
    tags: list[str]
    notes: str
    modes: list[str] = field(default_factory=lambda: ["scripted", "live"])
    history: list[dict] = field(default_factory=list)
    script: dict = field(default_factory=dict)
    # 只有固定首轮证据时才指定 first_evidence_sufficient。
    scripted_gold: dict = field(default_factory=dict)

    def expectations(self, mode: str) -> Gold:
        return replace(self.gold, **self.scripted_gold) if mode == "scripted" else self.gold


def case_from_dict(data: dict, kb_dir: Path = KNOWLEDGE_BASE) -> AgentCase:
    """严格检查字段和来源；script 是固定输入，不能由执行结果反推 Gold。"""
    if not isinstance(data, dict):
        raise ValueError("Case 必须是 JSON 对象")
    unknown = set(data) - {f.name for f in fields(AgentCase)}
    if unknown:
        raise ValueError(f"未知 Case 字段: {sorted(unknown)}")
    try:
        case = AgentCase(**{**data, "gold": Gold(**data["gold"])})
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Case 必填字段或 Gold 格式错误: {exc}") from exc
    if not isinstance(case.case_id, str) or not re.fullmatch(r"[a-z]+_\d{3}", case.case_id):
        raise ValueError("case_id 必须形如 router_001")
    if case.category not in CATEGORIES:
        raise ValueError("未知 category")
    for value in (case.question, case.notes):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question 和 notes 不能为空")
    if not isinstance(case.tags, list) or not case.tags or any(not isinstance(t, str) or not t for t in case.tags):
        raise ValueError("tags 必须是非空字符串列表")
    if not isinstance(case.modes, list) or not case.modes or not set(case.modes) <= {"live", "scripted"}:
        raise ValueError("modes 只能包含 live/scripted")
    if not isinstance(case.history, list) or any(
        not isinstance(m, dict) or m.get("role") not in {"user", "assistant"}
        or not isinstance(m.get("content"), str) for m in case.history
    ):
        raise ValueError("history 必须是 user/assistant 聊天记录")
    if not isinstance(case.scripted_gold, dict) or not isinstance(case.script, dict):
        raise ValueError("script/scripted_gold 必须是对象")
    try:
        scripted_gold = case.expectations("scripted")
    except TypeError as exc:
        raise ValueError(f"scripted_gold 格式错误: {exc}") from exc
    if "live" in case.modes and case.gold.first_evidence_sufficient is not None:
        raise ValueError("真实检索的首轮证据不可预先固定，请用 null")
    for gold in (case.gold, scripted_gold):
        for source in gold.required_sources:
            if not (kb_dir / source).is_file():
                raise ValueError(f"固定知识库缺少来源: {source}")
    script = case.script
    if type(script.get("route")) is not bool:
        raise ValueError("script.route 必须是 bool")
    allowed = {"route", "initial_queries", "searches", "judges", "rewrites",
               "answer_sections", "answer", "abstain", "missing_tool_calls"}
    if set(script) - allowed:
        raise ValueError("script 存在未知字段")
    for name in ("initial_queries", "rewrites"):
        values = script.get(name, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"script.{name} 必须是字符串列表")
    for name in ("searches", "judges", "answer_sections"):
        values = script.get(name, [])
        if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
            raise ValueError(f"script.{name} 必须是对象列表")
    if "answer" in script and not isinstance(script["answer"], str):
        raise ValueError("script.answer 必须是字符串")
    if "abstain" in script and type(script["abstain"]) is not bool:
        raise ValueError("script.abstain 必须是 bool")
    count = script.get("missing_tool_calls", 0)
    if type(count) is not int or count < 0:
        raise ValueError("missing_tool_calls 必须是非负整数")
    for search in script.get("searches", []):
        if search.get("status") not in {"success", "empty", "error"}:
            raise ValueError("script 搜索状态无效")
        if search["status"] == "success" and not search.get("excerpts"):
            raise ValueError("成功的脚本搜索必须包含实际证据")
        for ref in search.get("excerpts", []):
            read_excerpt(ref, kb_dir)
    for ref in script.get("answer_sections", []):
        read_excerpt(ref, kb_dir)
    for decision in script.get("judges", []):
        if type(decision.get("sufficient")) is not bool:
            raise ValueError("script Judge sufficient 必须是 bool")
    return case


def read_excerpt(ref: dict, kb_dir: Path = KNOWLEDGE_BASE) -> str:
    """脚本证据按标题从真实 fixture 读取，避免另存一套会漂移的答案文本。"""
    source, section = ref["source"], ref["section"]
    if Path(source).name != source or "/" in source or "\\" in source:
        raise ValueError("非法 fixture 路径")
    text = (kb_dir / source).read_text(encoding="utf-8")
    heading = f"## {section}\n"
    if heading not in text:
        raise ValueError(f"{source} 缺少章节 {section}")
    if "page" in ref and (type(ref["page"]) is not int or ref["page"] < 1):
        raise ValueError("脚本页码必须是正整数")
    return text.split(heading, 1)[1].split("\n## ", 1)[0].strip()


def load_cases(path: str | Path = DEFAULT_CASES, kb_dir: Path = KNOWLEDGE_BASE) -> list[AgentCase]:
    """读取 JSONL；报错带行号，重复 ID 立即失败。"""
    cases, ids = [], set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = case_from_dict(json.loads(line), kb_dir)
            if case.case_id in ids:
                raise ValueError(f"重复 case_id: {case.case_id}")
        except (ValueError, KeyError, TypeError, OSError, re.error) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Case 集不能为空")
    return cases
