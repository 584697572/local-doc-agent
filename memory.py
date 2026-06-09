from config import PROJECT_DIR
from document_tools import find_matching_txt_file

def build_memory_message(memory_state):
    parts = []

    if memory_state.get("current_file"):
        parts.append(f"当前文件 = {memory_state['current_file']}")

    if memory_state.get("current_section"):
        parts.append(f"当前章节 = {memory_state['current_section']}")

    if not parts:
        return None

    return {
        "role": "system",
        "content": (
            "以下是当前会话记忆：\n"
            + "\n".join(parts)
            + "\n如果用户说“这个文件”、“刚才那个文件”、“继续”等，优先参考这些记忆。"
        )
    }


def update_memory_from_tool_call(tool_name, arguments, memory_state):
    filename = arguments.get("filename", "")

    if filename:
        current_dir = PROJECT_DIR
        match_result = find_matching_txt_file(current_dir, filename)

        if match_result is not None and not isinstance(match_result, list):
            memory_state["current_file"] = match_result.name
        else:
            memory_state["current_file"] = filename

    if tool_name == "extract_text_section":
        section_name = arguments.get("section_name", "")
        if section_name:
            memory_state["current_section"] = section_name