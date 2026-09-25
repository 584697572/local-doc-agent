"""LocalDoc-Agent 的 Agent 主循环。"""

import json

from config import MAX_AGENT_STEPS, MODEL_NAME
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


def run_agent(user_input, chat_history):
    """执行多轮 Tool-Calling Agent Loop。"""
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

        for tool_index, tool_call in enumerate(message.tool_calls, start=1):
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            print(f"[Agent] step={step + 1} tool={tool_index} name={tool_name}")
            print(f"[Agent] arguments={arguments}")

            tool_result = run_tool(tool_name, arguments)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    return "达到最大工具调用轮数，已停止执行。"
