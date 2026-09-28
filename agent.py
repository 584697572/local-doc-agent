"""LocalDoc-Agent 的 Agent 主循环。"""

import json

from config import (
    MAX_AGENT_STEPS,
    MAX_SEARCH_CALLS,
    MODEL_NAME,
)
from harness.state import AgentState
from harness.evidence import evaluate_evidence
from harness.rewrite import rewrite_query

from llm_client import client
from tools.registry import ToolRegistry
from tools.retrieval import search_documents
from tools.schemas import TOOLS



SYSTEM_PROMPT = (
    "你是 LocalDoc-Agent，一个面向本地知识库的检索问答助手。"
    "当用户的问题依赖本地文档中的事实、概念或技术内容时，调用 search_documents。"
    "得到检索证据后，只能基于证据回答，不要编造知识库中不存在的信息。"
    "回答时尽量注明来源文件；如果证据包含页码，也注明页码。"
    "如果没有足够证据，明确告诉用户当前知识库无法支持该结论。"
)


tool_registry = ToolRegistry()
tool_registry.register(
    "search_documents",
    search_documents,
    ["query"],
)


def run_tool(tool_name, arguments):
    """根据模型返回的工具名执行对应 Python 函数。"""
    return tool_registry.run(tool_name, arguments)

def generate_final_answer(messages, state):
    """
    根据最终 Evidence 状态生成回答。
    此时不再提供 tools，防止模型继续搜索。
    """

    if state.evidence_sufficient:
        instruction = (
            "Evidence Judge 已确认当前证据足够。"
            "请直接根据已有证据回答用户问题。"
            "不要使用证据之外的信息补全答案。"
            "回答时尽量注明来源文件和页码。"
        )

    else:
        missing_text = "；".join(
            state.missing_aspects
        )

        if not missing_text:
            missing_text = "无法明确确定具体缺失信息"

        instruction = (
            "当前搜索已经结束，但 Evidence Judge 认为证据仍不充分。"
            "请只根据已经获得的证据回答能够确认的部分，"
            "不要猜测缺失内容。"
            f"当前缺失信息：{missing_text}。"
            "回答中应明确说明哪些部分缺少知识库证据。"
        )

    final_messages = messages + [
        {
            "role": "system",
            "content": instruction,
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=final_messages,
    )

    return response.choices[0].message.content


def run_agent(user_input, chat_history):
    """执行多轮 Tool-Calling Agent Loop。"""

    # 每个用户问题都创建一个新的运行状态
    state = AgentState(
        original_query=user_input,
        max_search_calls=MAX_SEARCH_CALLS,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *chat_history,
        {
            "role": "user",
            "content": user_input,
        },
    ]

    for step in range(MAX_AGENT_STEPS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # 没有 tool call，说明模型已经生成最终答案。
        if not message.tool_calls:
            return message.content

        messages.append(message)

        search_executed = False

        for tool_index, tool_call in enumerate(message.tool_calls, start=1):
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            print(f"[Agent] step={step + 1} tool={tool_index} name={tool_name}")
            print(f"[Agent] arguments={arguments}")

            # search_documents 需要受到搜索预算和去重控制
            if tool_name == "search_documents":
                query = arguments.get("query", "")

                allowed, reason = state.reserve_search(
                    query=query,
                    step=step + 1,
                )

                if not allowed:
                    if reason == "duplicate_query":
                        tool_result = (
                            "该 Query 已经搜索过，请不要重复检索。"
                        )

                    elif reason == "budget_exhausted":
                        tool_result = (
                            "搜索预算已用完，请基于已有证据回答。"
                        )

                    else:
                        tool_result = "当前检索请求无效。"

                    state.add_trace(
                        "search_rejected",
                        step=step + 1,
                        query=query,
                        reason=reason,
                    )

                else:
                    tool_result = run_tool(
                        tool_name,
                        arguments,
                    )
                    #保存本次检索得到的证据
                    state.record_evidence(tool_result)

                    # 这一轮确实拿到了新的检索结果
                    search_executed = True

            else:
                tool_result = run_tool(
                    tool_name,
                    arguments,
                )

            print(
                f"[Agent] search_count="
                f"{state.search_count}/{state.max_search_calls}"
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        # 这一轮获得新证据后，进入 Evidence → Rewrite 循环
        if search_executed:
            decision = evaluate_evidence(
                user_query=state.original_query,
                evidence=state.evidence,
            )

            state.record_evidence_decision(
                sufficient=decision.sufficient,
                reason=decision.reason,
                missing_aspects=decision.missing_aspects,
            )

            print(
                f"[Agent] evidence_sufficient="
                f"{decision.sufficient}"
            )
            print(
                f"[Agent] evidence_reason="
                f"{decision.reason}"
            )

            if not decision.sufficient:
                print(
                    f"[Agent] missing_aspects="
                    f"{decision.missing_aspects}"
                )

            # 证据不足并且还有搜索预算时，自动 Rewrite + Search
            while (
                not decision.sufficient
                and state.search_count < state.max_search_calls
            ):
                rewrite_result = rewrite_query(
                    original_query=state.original_query,
                    missing_aspects=decision.missing_aspects,
                    used_queries=state.used_queries,
                )

                print(
                    f"[Agent] rewritten_query="
                    f"{rewrite_result.query}"
                )
                print(
                    f"[Agent] rewrite_reason="
                    f"{rewrite_result.reason}"
                )

                state.add_trace(
                    "query_rewritten",
                    query=rewrite_result.query,
                    reason=rewrite_result.reason,
                )

                # Rewriter 没有生成有效 Query，就无法继续搜索
                if not rewrite_result.query:
                    break

                # 新 Query 仍然要经过 Harness 的去重和预算检查
                allowed, reason = state.reserve_search(
                    query=rewrite_result.query,
                    step=step + 1,
                )

                if not allowed:
                    state.add_trace(
                        "rewrite_search_rejected",
                        query=rewrite_result.query,
                        reason=reason,
                    )

                    print(
                        f"[Agent] rewrite_search_rejected="
                        f"{reason}"
                    )

                    break

                # Harness 直接执行 Rewrite 后的新 Query
                rewritten_evidence = run_tool(
                    "search_documents",
                    {
                        "query": rewrite_result.query,
                    },
                )

                state.record_evidence(
                    rewritten_evidence
                )

                print(
                    f"[Agent] search_count="
                    f"{state.search_count}/"
                    f"{state.max_search_calls}"
                )

                # 把自动补充搜索得到的 Evidence 也加入上下文
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Harness 根据 Evidence 缺口进行了补充检索。\n"
                            f"检索 Query：{rewrite_result.query}\n\n"
                            f"检索结果：\n{rewritten_evidence}"
                        ),
                    }
                )

                # 有了新 Evidence，再重新判断一次
                decision = evaluate_evidence(
                    user_query=state.original_query,
                    evidence=state.evidence,
                )

                state.record_evidence_decision(
                    sufficient=decision.sufficient,
                    reason=decision.reason,
                    missing_aspects=decision.missing_aspects,
                )

                print(
                    f"[Agent] evidence_sufficient="
                    f"{decision.sufficient}"
                )
                print(
                    f"[Agent] evidence_reason="
                    f"{decision.reason}"
                )

                if not decision.sufficient:
                    print(
                        f"[Agent] missing_aspects="
                        f"{decision.missing_aspects}"
                    )

            # 到这里：
            # 1. 要么证据已经足够
            # 2. 要么搜索预算耗尽
            # 3. 要么 Rewriter 无法产生有效 Query
            return generate_final_answer(
                messages,
                state,
            )             

    return "达到最大工具调用轮数，已停止执行。"
