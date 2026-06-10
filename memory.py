"""
会话记忆模块。

这里的“记忆”不是把所有历史消息永久塞给模型，而是保存少量关键事实：
例如当前正在看的文件、当前章节。

这样用户说“这个文件”“刚才那个章节”“继续看 Extract”时，
Agent 可以根据这些结构化信息补全用户省略的上下文。
"""

import json
from config import PROJECT_DIR, MEMORY_FILE
from document_tools import find_matching_txt_file

def build_memory_message(memory_state):
    """
    把 memory_state 字典转换成一条 system 消息。

    memory_state 是程序内部使用的结构化字典，例如：
    {
        "current_file": "tech35格式.txt",
        "current_section": "Drc"
    }

    模型看不懂 Python 字典本身，所以这里把它翻译成自然语言提示。
    """

    parts = []

    if memory_state.get("current_file"):
        parts.append(f"当前文件 = {memory_state['current_file']}")

    if memory_state.get("current_section"):
        parts.append(f"当前章节 = {memory_state['current_section']}")

    if not parts:
        return None

    # 返回一条 system 消息，让模型在回答当前问题前先知道这些记忆。
    return {
        "role": "system",
        "content": (
            "以下是当前会话记忆：\n"
            + "\n".join(parts)
            + "\n如果用户说“这个文件”、“刚才那个文件”、“继续”等，优先参考这些记忆。"
        )
    }


def update_memory_from_tool_call(tool_name, arguments, memory_state):
    """
    根据工具调用参数更新结构化记忆。

    为什么从工具调用里更新？
    因为工具参数通常已经被模型整理成了明确字段，例如：
    {"filename": "tech35", "section_name": "Drc"}

    这比从自然语言回答里猜“当前文件是什么”更可靠。
    """

    filename = arguments.get("filename", "")

    if filename:
        current_dir = PROJECT_DIR
        match_result = find_matching_txt_file(current_dir, filename)

        # 如果能匹配到唯一真实文件，就把真实文件名记下来。
        # 例如用户说 tech35，最终记忆里保存 tech35格式.txt。
        if match_result is not None and not isinstance(match_result, list):
            memory_state["current_file"] = match_result.name
        else:
            # 如果没匹配到唯一文件，也先保存用户传入的原始文件名。
            memory_state["current_file"] = filename

    if tool_name == "extract_text_section":
        section_name = arguments.get("section_name", "")
        if section_name:
            memory_state["current_section"] = section_name

def load_memory_state():
    """
    从 memory.json 读取长期记忆。
    如果文件不存在，就返回默认空记忆。
    """

    default_memory = {
        "current_file": None,
        "current_section": None,
    }

    if not MEMORY_FILE.exists():
        return default_memory

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return default_memory

        return {
            "current_file": data.get("current_file"),
            "current_section": data.get("current_section"),
        }

    except Exception:
        return default_memory


def save_memory_state(memory_state):
    """
    把当前记忆写入 memory.json。
    """

    data = {
        "current_file": memory_state.get("current_file"),
        "current_section": memory_state.get("current_section"),
    }

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)