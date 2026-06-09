import json

from config import MODEL_NAME
from llm_client import client
from tool_schemas import TOOLS
from memory import build_memory_message, update_memory_from_tool_call
from document_tools import (
    get_current_time,
    calculate,
    get_today_date,
    read_text_file,
    list_txt_files,
    search_text_file,
    extract_text_section,
)

# =========================
# 3. 根据模型的决定，真正执行对应工具
# =========================

def run_tool(tool_name, arguments):
    """
    根据工具名和参数，执行对应的 Python 函数。

    tool_name:
        模型想调用的工具名，例如 "get_current_time"

    arguments:
        模型传给工具的参数，是一个字典
        例如 {"expression": "25*17"}
    """

    if tool_name == "get_current_time":
        return get_current_time()

    elif tool_name == "calculate":
        # 从 arguments 中取出 expression
        expression = arguments.get("expression", "")
        return calculate(expression)
    elif tool_name == "get_today_date":
        return get_today_date()
    elif tool_name=="read_text_file":
        filename=arguments.get("filename","")
        return read_text_file(filename)
    elif tool_name == "list_txt_files":
        return list_txt_files()
    elif tool_name == "search_text_file":
        filename = arguments.get("filename", "")
        keyword = arguments.get("keyword", "")
        return search_text_file(filename, keyword)
    elif tool_name == "extract_text_section":
        filename = arguments.get("filename", "")
        section_name = arguments.get("section_name", "")
        return extract_text_section(filename, section_name)
    else:
        return f"未知工具：{tool_name}"


# =========================
# 4. agent 主流程
# =========================

def run_agent(user_input,chat_history,memory_state):
    """
    多轮工具调用版 agent。

    核心思想：
    模型可以连续调用多个工具。
    每次工具调用结束后，把工具结果加入 messages，
    再让模型继续判断下一步。
    """

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会使用工具的中文助手。"

                "如果用户询问当前时间，必须调用 get_current_time 工具。"
                "如果用户询问今天日期，必须调用 get_today_date 工具。"
                "如果用户要求数学计算，必须调用 calculate 工具。"

                "如果用户想查看当前目录有哪些 txt 文件，必须调用 list_txt_files 工具。"
                "如果用户的问题包含“章节”、“section”、“部分”、“一节”、“某一章”，并且是在问 txt 文件内容，优先调用 extract_text_section。"
                "如果用户只是想找某个词在哪里出现、搜索关键词、查看关键词附近上下文，优先调用 search_text_file。"
                "如果用户只是想查看、读取、总结整个小 txt 文件，才调用 read_text_file。"
                "对于超过大小限制的大文件，不要用 read_text_file 获取章节内容。"

                "如果用户没有给出精确文件名，你可以先调用 list_txt_files 查看文件列表，"
                "然后根据文件列表选择最相关的文件，再调用 read_text_file。"

                "不要编造文件内容。"
                "不要输出 DSML。"
                "不要输出伪造的 tool_calls 文本。"
                "工具名必须使用已提供的工具名，不要自造 read_file、read_local 之类的工具名。"
            )
        }

    ]
    memory_message = build_memory_message(memory_state)
    if memory_message:
        messages.append(memory_message)

    messages.extend(chat_history)

    messages.append({
        "role": "user",
        "content": user_input
    })

    # 最多允许连续调用 5 轮工具，防止死循环
    max_steps = 5

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # 如果模型没有要求调用工具，说明它已经给出最终回答
        if not message.tool_calls:
            return message.content

        # 把模型的工具调用请求加入历史
        messages.append(message)

        # 处理本轮所有工具调用
        for tool_index, tool_call in enumerate(message.tool_calls, start=1):
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            update_memory_from_tool_call(tool_name, arguments, memory_state)

            print(f"[调试] 第 {step + 1} 轮，第 {tool_index} 个工具调用")
            print(f"[调试] 模型决定调用工具：{tool_name}")
            print(f"[调试] 工具参数：{arguments}")

            tool_result = run_tool(tool_name, arguments)

            print(f"[调试] 工具返回结果：{tool_result}")

            # 把工具结果加入 messages，让模型下一轮能看到
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result
            })

    return "工具调用轮数过多，已停止执行，避免陷入循环。"