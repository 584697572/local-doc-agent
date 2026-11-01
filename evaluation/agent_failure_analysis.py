"""内部 case 的回答检查、硬约束和失败分类，不依赖真实模型。"""

from collections import Counter
import re

# 阶段顺序用于解释原因；没有发生回归的预期故障只进入事件统计。
FAILURE_ORDER = (
    "router_false_negative",
    "router_false_positive",
    "tool_error",
    "tool_empty",
    "retrieval_miss",
    "evidence_false_positive",
    "evidence_false_negative",
    "rewrite_not_triggered",
    "rewrite_duplicate",
    "rewrite_no_gain",
    "search_budget_exhausted",
    "unsupported_claim",
    "wrong_citation",
    "missing_citation",
    "answer_incomplete",
    "answer_incorrect",
)
ABSTENTION = "知识库没有提供所需信息，无法确认缺失部分。"
FILE_PATTERN = r"[\w.\-]+\.(?:md|txt|pdf)"
ABSTAIN_PATTERN = r"无法确认|无法支持|证据不足|资料不足|未提供|没有提供|不知道"


def citation_matches(answer):
    """区分正式来源标注与正文中的文件提及，避免把交叉阅读建议当成引用。"""
    matches = []
    for match in re.finditer(FILE_PATTERN, answer, re.I):
        line_start = answer.rfind("\n", 0, match.start()) + 1
        prefix = answer[line_start : match.start()]
        opened = max(prefix.rfind("["), prefix.rfind("【"))
        closed = max(prefix.rfind("]"), prefix.rfind("】"))
        labelled = re.search(
            r"(?:来源|出处|引用|依据|根据|参考文献)\s*[:：]?\s*[^。！？\n]*$", prefix
        )
        standalone = not prefix.strip(" -*" + chr(96))
        if opened > closed or labelled or standalone:
            matches.append(match)
    return matches


def matched_facts(text, gold):
    """每个事实允许若干表达；此规则仅用于可控脚本，Live 使用语义评审。"""
    fact_matches = []
    for alternatives in gold.fact_patterns:
        fact_found = False
        for pattern in alternatives:
            if re.search(pattern, text, re.I | re.S):
                fact_found = True
                break
        fact_matches.append(fact_found)
    return fact_matches


def review_scripted(answer, evidence, gold):
    """脚本回答为 fixture 摘录：逐句核对，额外编造的句子无法蒙混过关。"""
    facts = matched_facts(answer, gold)
    clean = re.sub(r"\[[^\]]*\]", "", answer)
    clean = clean.replace(ABSTENTION, "")
    # 停止提示不是知识性断言，但也不能冒充“已解释资料缺口”。
    clean = clean.replace("达到最大工具调用轮数，已停止执行。", "")
    claims = []
    for sentence in re.split(r"[。！？\n]+", clean):
        sentence = sentence.strip()
        if sentence:
            claims.append(sentence)
    unsupported = []
    if gold.needs_retrieval:
        for claim in claims:
            if claim not in evidence:
                unsupported.append(claim)
    normalized = re.sub(r"\s+", "", answer)
    forbidden = []
    for claim in gold.forbidden_claims:
        target = re.sub(r"\s+", "", claim)
        for match in re.finditer(re.escape(target), normalized):
            prefix = normalized[max(0, match.start() - 12) : match.start()]
            if not re.search(r"不能|无法|并非|不是|不应|没有依据", prefix):
                forbidden.append(claim)
                break
    grounded_answer = None
    evidence_sufficient = None
    if gold.needs_retrieval:
        grounded_answer = not unsupported and not forbidden
        evidence_sufficient = (
            bool(evidence)
            and not gold.should_abstain
            and all(matched_facts(evidence, gold))
        )

    return {
        "facts_covered": facts,
        "grounded_answer": grounded_answer,
        "unsupported_claims": unsupported + forbidden,
        "forbidden_claims_present": forbidden,
        "abstained": ABSTENTION in answer
        or bool(re.search(ABSTAIN_PATTERN, "\n".join(unsupported))),
        "evidence_sufficient": evidence_sufficient,
        "citation_supported": not unsupported,
        "assessment_method": "extractive_script_check",
    }


def check_citations(answer, executions, gold):
    """来源只能来自成功执行的工具结果；页码和文件绑定，不能全局凑数。"""
    sources = []
    for call in executions:
        if call["status"] == "success":
            metadata = call.get("metadata", {})
            sources.extend(metadata.get("sources", []))
    available = set()
    for source in sources:
        available.add(source["filename"])
    matches = citation_matches(answer)
    cited = set()
    for match in matches:
        cited.add(match.group())
    unknown = sorted(cited - available)
    pages = {}
    wrong_pages = []
    for i, match in enumerate(matches):
        # 下一文件出现后就停止，以免把另一来源的页码串过来。
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(answer)
        tail = re.split(r"[\]\n。]", answer[match.end() : end], maxsplit=1)[0]
        page = re.search(r"(?:第\s*(\d+)\s*页|p(?:age)?\.?\s*(\d+))", tail, re.I)
        if page:
            number = int(page.group(1) or page.group(2))
            pages.setdefault(match.group(), []).append(number)
            page_found = False
            for source in sources:
                if source["filename"] == match.group() and source.get("page") == number:
                    page_found = True
                    break
            if not page_found:
                wrong_pages.append({"filename": match.group(), "page": number})
    missing = sorted(set(gold.required_sources) - cited)
    missing_pages = []
    for name, page in gold.required_pages.items():
        if page not in pages.get(name, []):
            missing_pages.append(name)
    correct = None
    if gold.citation_required:
        correct = bool(cited) and not (
            unknown or missing or missing_pages or wrong_pages
        )
    return {
        "citation_correct": correct,
        "cited_sources": sorted(cited),
        "retrieved_sources": sorted(available),
        "unknown_citations": unknown,
        "missing_citations": missing,
        "missing_pages": missing_pages,
        "wrong_pages": wrong_pages,
    }


def check_hard_invariants(record):
    """返回所有硬约束违规，供 CLI 以非零状态退出；不被平均分掩盖。"""
    errors = []
    calls = record["executions"]
    trace = record["trace"]

    def normalize(query):
        words = query.lower().split()
        return " ".join(words)

    executed_queries = []
    for call in calls:
        executed_queries.append(normalize(call["query"]))
    accepted = []
    accepted_queries = []
    for event in trace:
        if event["event"] == "search_accepted":
            accepted.append(event)
            accepted_queries.append(normalize(event["query"]))
    if (
        max(record["search_calls"], len(calls), len(accepted))
        > record["max_search_calls"]
    ):
        errors.append("search_budget_exceeded")
    if len(set(accepted_queries)) != len(accepted_queries):
        errors.append("duplicate_query_accepted")
    if len(set(executed_queries)) != len(executed_queries):
        errors.append("duplicate_query_executed")
    if record["search_calls"] != len(calls) or len(accepted) != len(calls):
        errors.append("search_accounting_mismatch")
    if set(record["used_queries"]) != set(accepted_queries):
        errors.append("used_queries_mismatch")
    blocked = set()
    for event in trace:
        if (
            event["event"] in {"search_rejected", "rewrite_search_rejected"}
            and event.get("reason") == "duplicate_query"
        ):
            blocked.add(normalize(event["query"]))
        elif event["event"] == "tool_executed" and normalize(event["query"]) in blocked:
            errors.append("rejected_duplicate_executed")
            break
    valid_contents = set()
    has_successful_call = False
    for call in calls:
        if call["status"] == "success":
            has_successful_call = True
            valid_contents.add(call.get("content", ""))
    for evidence in record.get("evidence", []):
        if evidence not in valid_contents:
            errors.append("evidence_not_from_successful_tool")
            break
    # 预约和独立执行记录均要检查，避免只看 set 永远看不到重复。
    if record["unknown_citations"] or record["wrong_pages"]:
        errors.append("citation_outside_evidence")
    if (
        record["actual_needs_retrieval"] is True
        and not has_successful_call
        and record.get("unsupported_claims")
    ):
        errors.append("ungrounded_answer_without_retrieval")
    if record["expected_abstain"] and record.get("unsupported_claims"):
        errors.append("unsupported_claim_when_abstaining")
    if record.get("error"):
        errors.append("case_execution_error")
    return errors


def classify_failure(record):
    """依据最早已观察到的错误阶段排序；不要把正确的不足判断算成 Judge 错误。"""
    if record["task_success"] and not record["hard_invariants"]:
        return {
            "primary_failure": None,
            "secondary_failures": [],
            "failure_category": None,
        }
    failures = set()
    expected, actual = (
        record["expected_needs_retrieval"],
        record["actual_needs_retrieval"],
    )
    if expected and actual is False:
        failures.add("router_false_negative")
    elif not expected and actual is True:
        failures.add("router_false_positive")
    events = set()
    for event in record["trace"]:
        events.add(event["event"])
    if events & {"search_error", "rewrite_search_error"}:
        failures.add("tool_error")
    if events & {"search_empty", "rewrite_search_empty"}:
        failures.add("tool_empty")
    if record.get("missing_evidence_sources"):
        failures.add("retrieval_miss")
    judged = [
        (record["expected_evidence_sufficient"], record["actual_evidence_sufficient"])
    ]
    if record.get("expected_first_evidence_sufficient") is not None:
        judged.append(
            (
                record["expected_first_evidence_sufficient"],
                record["actual_first_evidence_sufficient"],
            )
        )
    for expected_judge, actual_judge in judged:
        if expected_judge is False and actual_judge is True:
            failures.add("evidence_false_positive")
        if expected_judge is True and actual_judge is False:
            failures.add("evidence_false_negative")
    if record["expected_should_rewrite"] is True and not record["actual_rewrite_used"]:
        failures.add("rewrite_not_triggered")
    for event in record["trace"]:
        if event.get("reason") == "duplicate_query":
            failures.add("rewrite_duplicate")
            break
    if record.get("rewrite_success") is False:
        failures.add("rewrite_no_gain")
    if record["budget_exhausted"]:
        failures.add("search_budget_exhausted")
    if record.get("unsupported_claims"):
        failures.add("unsupported_claim")
    if (
        record["unknown_citations"]
        or record["wrong_pages"]
        or record.get("citation_correct") is False
        and not record["missing_citations"]
        and not record["missing_pages"]
    ):
        failures.add("wrong_citation")
    if record["missing_citations"] or record["missing_pages"]:
        failures.add("missing_citation")
    if not all(record["facts_covered"]):
        failures.add("answer_incomplete")
    if not failures:
        failures.add("answer_incorrect")
    ordered = []
    for name in FAILURE_ORDER:
        if name in failures:
            ordered.append(name)
    return {
        "primary_failure": ordered[0],
        "secondary_failures": ordered[1:],
        "failure_category": ordered[0],
    }


def analyze_failures(records):
    """同时提供主因、伴随失败、典型样例与调用次数分布。"""
    primary = Counter()
    all_failures = Counter()
    examples = {}
    search_distribution = Counter()
    rewrite_distribution = Counter()
    event_counts = Counter()
    budget_exhaustion_cases = []
    unsupported_claim_cases = []
    citation_error_cases = []
    hard_invariant_cases = {}

    # 每条记录分别更新次数分布和失败样例，便于查看某个计数的来源。
    for record in records:
        if record["primary_failure"]:
            primary[record["primary_failure"]] += 1
        failure_names = [record["primary_failure"]]
        failure_names.extend(record["secondary_failures"])
        for name in failure_names:
            if name:
                all_failures[name] += 1
                examples.setdefault(name, []).append(record["case_id"])
        search_distribution[str(record["search_calls"])] += 1
        rewrite_distribution[str(record["rewrite_calls"])] += 1
        if record["budget_exhausted"]:
            budget_exhaustion_cases.append(record["case_id"])
        if record.get("unsupported_claims"):
            unsupported_claim_cases.append(record["case_id"])
        if (
            record["citation_correct"] is False
            or record["unknown_citations"]
            or record["wrong_pages"]
        ):
            citation_error_cases.append(record["case_id"])
        if record["hard_invariants"]:
            hard_invariant_cases[record["case_id"]] = record["hard_invariants"]
        for event in record["trace"]:
            event_counts[event["event"]] += 1

    limited_examples = {}
    for name, case_ids in examples.items():
        limited_examples[name] = case_ids[:5]
    return {
        "failure_counts": dict(primary),
        "all_failure_counts": dict(all_failures),
        "examples": limited_examples,
        "search_distribution": dict(search_distribution),
        "rewrite_distribution": dict(rewrite_distribution),
        "budget_exhaustion_cases": budget_exhaustion_cases,
        "unsupported_claim_cases": unsupported_claim_cases,
        "citation_error_cases": citation_error_cases,
        "hard_invariant_cases": hard_invariant_cases,
        "event_counts": dict(event_counts),
    }
