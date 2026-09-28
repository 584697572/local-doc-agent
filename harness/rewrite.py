"""根据 Evidence 缺口生成新的检索 Query。"""

import json
from dataclasses import dataclass

from config import MODEL_NAME
from llm_client import client


@dataclass
class RewriteResult:
    """一次 Query Rewrite 的结果。"""

    # 新生成的检索 Query
    query: str

    # 为什么这样改写
    reason: str


def _call_rewriter(
    original_query: str,
    missing_aspects: list[str],
    used_queries: set[str],
) -> str:
    """
    调用 LLM，根据证据缺口生成一条新的检索 Query。
    """

    # 把缺失信息整理成人类和 LLM 都容易读的文本
    missing_text = "\n".join(
        f"- {aspect}"
        for aspect in missing_aspects
    )

    # 把已经执行过的 Query 告诉 Rewriter，减少重复搜索
    used_query_text = "\n".join(
        f"- {query}"
        for query in sorted(used_queries)
    )

    if not used_query_text:
        used_query_text = "无"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 Query Rewriter。"
                    "你的任务不是回答用户问题，"
                    "而是生成下一条知识库检索 Query。"

                    "新 Query 应重点搜索当前缺失的信息，"
                    "同时保留原问题中的关键实体和专业术语。"

                    "不要生成已经搜索过的 Query。"
                    "只生成一条简洁、适合检索的 Query。"

                    "必须返回 JSON，格式如下："
                    "{"
                    '"query": "新的检索 Query",'
                    '"reason": "这样改写的原因"'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始问题：\n{original_query}\n\n"
                    f"当前缺失的信息：\n{missing_text}\n\n"
                    f"已经搜索过的 Query：\n{used_query_text}"
                ),
            },
        ],
        response_format={
            "type": "json_object"
        },
        temperature=0,
    )

    return response.choices[0].message.content or "{}"


def rewrite_query(
    original_query: str,
    missing_aspects: list[str],
    used_queries: set[str],
) -> RewriteResult:
    """
    根据缺失信息生成下一条检索 Query。
    """

    # 连缺什么都不知道时，不进行盲目改写
    if not missing_aspects:
        return RewriteResult(
            query="",
            reason="当前没有明确的证据缺口，无法进行定向 Query Rewrite。",
        )

    try:
        raw_result = _call_rewriter(
            original_query=original_query,
            missing_aspects=missing_aspects,
            used_queries=used_queries,
        )

        data = json.loads(raw_result)

    except Exception as exc:
        # Rewriter 出错时，不让整个 Agent 崩溃
        return RewriteResult(
            query="",
            reason=f"Query Rewrite 失败：{exc}",
        )

    query = data.get("query", "")
    reason = data.get("reason", "")

    # 防止模型虽然返回 JSON，但字段类型不正确
    if not isinstance(query, str):
        query = ""

    if not isinstance(reason, str):
        reason = ""

    return RewriteResult(
        query=query.strip(),
        reason=reason.strip() or "未提供改写原因。",
    )