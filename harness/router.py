"""判断用户问题是否需要访问本地知识库。"""

import json
from dataclasses import dataclass

from config import MODEL_NAME
from llm_client import client


@dataclass
class RouteDecision:
    """Query Router 的判断结果。"""

    # True：必须查询本地知识库
    # False：可以直接回答
    needs_retrieval: bool

    # 为什么这样路由
    reason: str


def _call_router(user_query: str) -> str:
    """调用 LLM 判断当前问题是否需要 Retrieval。"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 LocalDoc-Agent 的 Query Router。"
                    "你的任务只是判断用户问题是否需要查询本地知识库，"
                    "不要回答用户问题。"

                    "如果用户询问事实、概念、技术内容、文档内容，"
                    "或者问题可能依赖本地知识库中的信息，"
                    "needs_retrieval 应为 true。"

                    "只有简单寒暄、感谢等不依赖知识库的问题，"
                    "needs_retrieval 才为 false。"

                    "如果无法确定，优先返回 true，"
                    "避免在没有知识库证据的情况下直接回答。"

                    "必须返回 JSON："
                    "{"
                    '"needs_retrieval": true或false,'
                    '"reason": "判断原因"'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        response_format={
            "type": "json_object"
        },
        temperature=0,
    )

    return response.choices[0].message.content or "{}"


def route_query(user_query: str) -> RouteDecision:
    """判断当前用户问题应该走 Retrieval 还是直接回答。"""

    user_query = user_query.strip()

    if not user_query:
        return RouteDecision(
            needs_retrieval=False,
            reason="用户输入为空。",
        )

    try:
        raw_result = _call_router(user_query)
        data = json.loads(raw_result)

    except Exception as exc:
        # Router 出错时采用保守策略：
        # 宁可多搜索一次，也不要直接无证据回答。
        return RouteDecision(
            needs_retrieval=True,
            reason=f"Router 执行失败，默认进入 Retrieval：{exc}",
        )

    needs_retrieval = data.get(
        "needs_retrieval"
    )

    reason = data.get(
        "reason",
        "未提供路由原因。",
    )

    # 模型返回类型不正确时，也采用保守策略
    if not isinstance(needs_retrieval, bool):
        return RouteDecision(
            needs_retrieval=True,
            reason="Router 返回格式异常，默认进入 Retrieval。",
        )

    if not isinstance(reason, str):
        reason = "未提供路由原因。"

    return RouteDecision(
        needs_retrieval=needs_retrieval,
        reason=reason,
    )