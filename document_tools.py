"""
真实工具函数模块。

这个文件里的函数才是真正会被执行的“工具”：
时间查询、计算、列文件、读文件、搜索关键词、提取章节。

大模型不会直接读取本地文件，也不会直接执行 Python。
它只会通过 tool_calls 告诉 agent.py：“我想调用某个工具，并传入这些参数”。
agent.py 再通过 ToolRegistry 找到这里的函数并执行。
"""

import re

from datetime import datetime

from config import (
    PROJECT_DIR,
    MAX_FILE_SIZE,
    PREVIEW_CHARS,
    SEARCH_CONTEXT_CHARS,
    MAX_SEARCH_MATCHES,
    MAX_SECTION_CHARS,
)

# =========================
# 1. 定义真正会被执行的工具函数
# =========================

def get_current_time():
    """
    工具函数 1：获取当前时间

    注意：
    这只是普通 Python 函数。
    大模型自己不会真的执行它。
    模型只会告诉我们：“我想调用 get_current_time”。
    真正执行函数的是我们的 Python 程序。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression):
    """
    工具函数 2：计算数学表达式

    expression 是字符串，例如：
    "25*17"
    "100+200"
    "3.14*2"
    """
    try:
        # eval 可以把字符串当作 Python 表达式执行
        # 例如 eval("25*17") 得到 425
        #
        # 这里禁用了 builtins，避免它执行一些危险内置函数
        # 但真实项目里仍然不建议随便用 eval
        result = eval(expression, {"__builtins__": {}}, {})

        # 工具返回值最好转成字符串，方便后面传给模型
        return str(result)

    except Exception as e:
        # 如果表达式有问题，不让程序崩溃，而是返回错误原因
        return f"计算失败：{e}"

def get_today_date():
    """
    工具函数 3：获取今天的日期

    只返回年月日，不返回具体几点几分。
    例如：2026-05-12
    """
    return datetime.now().strftime("%Y-%m-%d")

def read_text_file(filename):
    """
    工具函数 4：读取当前项目目录下的 txt 文本文件。

    参数：
    filename：文件名，例如 "demo.txt"

    注意：
    为了安全起见，我们只允许读取当前项目目录下的 .txt 文件。
    不允许读取任意路径，比如 C 盘、D 盘其他目录里的文件。
    """

    try:
        # 获取当前 main.py 所在的文件夹
        current_dir = PROJECT_DIR
        
        # 复用统一的 txt 文件匹配逻辑
        # 这个函数会负责：
        # 1. 自动补 .txt 后缀
        # 2. 先精确匹配文件名
        # 3. 精确匹配失败后再模糊匹配文件名
        # 4. 返回唯一文件 / 多个候选文件 / 找不到
        match_result = find_matching_txt_file(current_dir, filename)

        # 没有找到任何匹配文件
        if match_result is None:
            return f"读取失败：没有找到文件 {filename}"

        # 匹配到多个文件，说明用户说得不够具体
        if isinstance(match_result, list):
            return (
                "读取失败：找到多个可能匹配的 txt 文件，请说得更具体一些：\n"
                + "\n".join(match_result)
            )

        # 匹配到唯一文件
        file_path = match_result
        filename = file_path.name

        # 安全检查：确保最终匹配到的文件仍然在当前项目目录下
        if current_dir.resolve() not in file_path.parents and file_path != current_dir.resolve():
            return "读取失败：只能读取当前项目目录下的 txt 文件。"

        # 安全检查：只允许读取 .txt 文件
        if file_path.suffix.lower() != ".txt":
            return "读取失败：目前只允许读取 .txt 文件。"

        
        # 安全检查 ：
        # 检查文件大小，防止超大文件一次性塞给模型
        file_size = file_path.stat().st_size

        
        is_large_file = file_size > MAX_FILE_SIZE
    

        content: str = ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if is_large_file:
                    content = f.read(PREVIEW_CHARS)
                else:
                    content = f.read()

        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                if is_large_file:
                    content = f.read(PREVIEW_CHARS)
                else:
                    content = f.read()

        if is_large_file:
            #preview = content[:PREVIEW_CHARS]

            return (
                f"注意：该文件较大，当前大小约为 {file_size / 1024:.2f} KB，"
                f"超过完整读取限制 {MAX_FILE_SIZE / 1024:.2f} KB。\n"
                f"以下只提供文件前 {PREVIEW_CHARS} 个字符作为预览：\n\n"
                f"{content}"
            )

        return content

    except Exception as e:
        return f"读取失败：{e}"

def list_txt_files():
    """
    工具函数5：列出当前项目目录下所有 txt 文件。

    作用：
    当用户不知道准确文件名，或者想查看有哪些 txt 文件时，
    agent 可以调用这个工具。
    """

    try:
        # 获取当前 main.py 所在目录
        current_dir = PROJECT_DIR

        # 找出当前目录下所有 .txt 文件
        txt_files = list(current_dir.glob("*.txt"))

        # 如果没有 txt 文件
        if not txt_files:
            return "当前项目目录下没有找到 txt 文件。"

        # 把文件名整理成字符串
        file_names = [file.name for file in txt_files]

        return "当前项目目录下的 txt 文件有：\n" + "\n".join(file_names)

    except Exception as e:
        return f"列出 txt 文件失败：{e}"
    
def find_matching_txt_file(current_dir,filename):
    """
    根据用户传入的文件名，在当前目录下寻找匹配的 txt 文件。

    返回值有三种：
    1. 找到唯一文件：返回 Path 对象
    2. 找到多个可能文件：返回文件名列表
    3. 没找到文件：返回 None
    """

    # 如果用户输入的文件名没有 .txt 后缀，就自动补上
    # 例如：用户输入 "tech35"，这里会变成 "tech35.txt"
    if filename and not filename.lower().endswith(".txt"):
        filename=filename+".txt"

    # 先进行精确匹配
    # 例如用户输入 "tech35格式.txt"，就直接找当前目录下有没有这个文件
    file_path=(current_dir/filename).resolve()

    # 如果精确文件名存在，直接返回这个文件路径
    if file_path.exists():
        return file_path
    
    # 如果精确匹配失败，就改成模糊匹配
    # 例如 "tech35.txt" 找不到，就提取出 "tech35" 作为关键词   
    keyword=filename.lower().replace(".txt","")

    # 获取当前目录下所有 .txt 文件
    txt_files=list(current_dir.glob("*.txt"))

    # 从所有 txt 文件里筛选文件名包含 keyword 的文件
    # 例如 keyword 是 "tech35"，可以匹配到 "tech35格式.txt"
    matched_files=[
        file for file in txt_files
        if keyword in file.stem.lower()
    ]
     # 如果只匹配到一个文件，说明结果明确，直接返回它
    if len(matched_files)==1:
        return matched_files[0].resolve()
    
    # 如果匹配到多个文件，说明不够明确
    # 返回文件名列表，让上层函数提示用户说得更具体一点
    if len(matched_files)>1:
        return [file.name for file in matched_files]
    # 如果一个文件都没匹配到，返回 None
    # 上层 search_text_file 会根据 None 返回“没有找到文件”的提示
    return None

def search_text_file(filename, keyword):
    """
    工具函数：在 txt 文件中搜索关键词，并返回关键词附近的上下文。

    参数：
    filename：文件名或文件名关键词，例如 "tech35"、"tech35格式.txt"
    keyword：要搜索的关键词，例如 "Drc section"、"extract"、"contact"

    返回：
    找到关键词时：返回关键词附近的一段上下文
    找不到关键词时：返回提示信息
    """

    try:
        # 获取当前 main.py 所在目录
        current_dir = PROJECT_DIR

        # 复用之前的文件名匹配逻辑
        match_result = find_matching_txt_file(current_dir, filename)

        # 没找到文件
        if match_result is None:
            return f"搜索失败：没有找到与“{filename}”匹配的 txt 文件。"

        # 匹配到多个文件
        if isinstance(match_result, list):
            return (
                f"搜索失败：找到多个可能匹配的 txt 文件，请说得更具体一些：\n"
                + "\n".join(match_result)
            )

        # 找到唯一文件
        file_path = match_result
        filename = file_path.name

        # 安全检查：确保文件仍在当前项目目录下
        if current_dir.resolve() not in file_path.parents and file_path != current_dir.resolve():
            return "搜索失败：只能搜索当前项目目录下的 txt 文件。"

        # 安全检查：只允许 txt 文件
        if file_path.suffix.lower() != ".txt":
            return "搜索失败：目前只允许搜索 .txt 文件。"

        # keyword 不能为空
        keyword = keyword.strip()
        if not keyword:
            return "搜索失败：关键词不能为空。"

        # 读取文件内容，先尝试 utf-8，再尝试 gbk
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()

        # 为了做大小写不敏感搜索，统一转成小写
        lower_content = content.lower()
        lower_keyword = keyword.lower()



        #进行了优化，从返回关键词首次出现位置到返回多个位置

        matches = []
        search_start = 0

        while True:
            index = lower_content.find(lower_keyword, search_start)

            # 找不到更多匹配了，就结束循环
            if index == -1:
                break

            matches.append(index)

            # 从当前关键词后面继续找，避免一直找到同一个位置
            search_start = index + len(lower_keyword)

            # 最多只返回前 MAX_SEARCH_MATCHES 个结果
            if len(matches) >= MAX_SEARCH_MATCHES:
                break

        if not matches:
            return f"在文件 {filename} 中没有找到关键词：{keyword}"

        result_parts = []

        for i, index in enumerate(matches, start=1):
            start = max(0, index - SEARCH_CONTEXT_CHARS)
            end = min(len(content), index + len(keyword) + SEARCH_CONTEXT_CHARS)

            context = content[start:end]

            # 计算大概行号，方便判断匹配位置
            line_number = content.count("\n", 0, index) + 1

            result_parts.append(
                f"【匹配 {i}】字符位置：{index}，大约行号：{line_number}\n"
                f"{context}"
            )

        return (
            f"已在文件 {filename} 中找到关键词：{keyword}\n"
            f"共返回前 {len(matches)} 个匹配位置：\n\n"
            + "\n\n" + "=" * 40 + "\n\n".join(result_parts)
        )

    except Exception as e:
        return f"搜索失败：{e}"
    

def extract_text_section(filename, section_name):
    """
    提取 txt 文件中的某个章节内容。
    例如：filename="tech35格式.txt", section_name="Drc"

    与 search_text_file 不同：
    search_text_file 是找关键词附近的片段；
    extract_text_section 是找章节标题，并提取到下一个同级/更高级标题之前。
    """

    try:
        current_dir = PROJECT_DIR

        match_result = find_matching_txt_file(current_dir, filename)

        if match_result is None:
            return f"章节提取失败：没有找到文件 {filename}"

        if isinstance(match_result, list):
            return (
                "章节提取失败：找到多个可能匹配的 txt 文件，请说得更具体一些：\n"
                + "\n".join(match_result)
            )

        file_path = match_result
        filename = file_path.name

        if current_dir.resolve() not in file_path.parents and file_path != current_dir.resolve():
            return "章节提取失败：只能读取当前项目目录下的 txt 文件。"

        if file_path.suffix.lower() != ".txt":
            return "章节提取失败：目前只允许读取 .txt 文件。"

        section_name = section_name.strip()
        if not section_name:
            return "章节提取失败：章节名不能为空。"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()

        lines = content.splitlines(keepends=True)

        # 1. 把用户给出的章节名整理成多个可能写法。
        # 例如用户可能说：
        #   DRC章节、Drc section、section Drc、绪论部分
        # 程序会尽量提取出核心名，并补充常见英文标题写法。
        title = section_name.lower()
        title_aliases = {title}

        # 去掉常见的描述性后缀，得到更核心的章节名。
        # 例如 "drc章节" -> "drc"，"绪论部分" -> "绪论"。
        for suffix in ["章节部分", "章节", "小节", "部分", "section", "chapter", "章", "节"]:
            if title.endswith(suffix):
                title_aliases.add(title[:-len(suffix)].strip())

        # 处理英文前缀写法。
        # 例如 "section drc" -> "drc"。
        for prefix in ["section ", "chapter "]:
            if title.startswith(prefix):
                title_aliases.add(title[len(prefix):].strip())

        # 把核心名扩展成常见标题形式。
        # 例如核心名 drc 会得到：
        # drc、drc section、section drc、drc chapter、chapter drc。
        target_titles = set()
        for alias in title_aliases:
            if not alias:
                continue
            target_titles.add(alias)
            target_titles.add(alias + " section")
            target_titles.add("section " + alias)
            target_titles.add(alias + " chapter")
            target_titles.add("chapter " + alias)

        # 2. 定义“什么样的行算章节标题”。
        # 每个元素是 (正则表达式, 标题层级规则)。
        #
        # 标题层级用于判断章节在哪里结束：
        # - 一级标题遇到下一个一级标题才结束
        # - 二级标题遇到下一个二级或一级标题结束
        #
        # 这些规则不是针对某个具体文件的章节名，而是识别通用标题格式。
        heading_patterns = [
            (r"^(#{1,6})\s+\S+", "markdown"),
            (r"^第[一二三四五六七八九十百千万两0-9]+([章节篇]|部分)\s*[：:、.\-]?\s*\S+", "chinese_chapter"),
            (r"^[一二三四五六七八九十]+[、.．]\s*\S+", 1),
            (r"^[（(][一二三四五六七八九十0-9]+[）)]\s*\S+", 2),
            (r"^(\d+(?:\.\d+)+)[、.．\s]+\S+", "dotted_number"),
            (r"^\d+[、.．]\s*\S+", 1),
            (r"^\d+\s+\S+", 1),
            (r"^[A-Za-z][A-Za-z0-9 _/\-]{0,80}\s+(section|chapter)$", 1),
            (r"^(section|chapter)\s+[A-Za-z0-9 _/\-]{1,80}$", 1),
        ]

        start_line = None
        start_level = 1

        # 3. 扫描全文，寻找目标章节的开始行。
        for line_no, line in enumerate(lines):
            line_title = line.strip()
            lower_line_title = line_title.lower()
            heading_level = None

            # 先判断当前行是不是标题；如果是，算出它的标题层级。
            for pattern, level_rule in heading_patterns:
                match = re.match(pattern, line_title, re.IGNORECASE)
                if not match:
                    continue

                if level_rule == "markdown":
                    heading_level = len(match.group(1))
                elif level_rule == "chinese_chapter":
                    heading_level = 2 if match.group(1) == "节" else 1
                elif level_rule == "dotted_number":
                    heading_level = match.group(1).count(".") + 1
                else:
                    heading_level = level_rule
                break

            # 第一种匹配：标题行完全等于某个候选标题。
            # 例如 "drc section" 完全匹配 target_titles 里的 "drc section"。
            is_target_line = lower_line_title in target_titles

            # 第二种匹配：如果这一行是标题，再做包含匹配。
            # 例如用户要 "绪论"，文档标题是 "第一章 绪论"，也应该认为匹配。
            if not is_target_line and heading_level is not None:
                compact_line_title = re.sub(r"\s+", "", lower_line_title)
                for target_title in target_titles:
                    compact_target = re.sub(r"\s+", "", target_title)
                    if compact_target and compact_target in compact_line_title:
                        is_target_line = True
                        break

            if is_target_line:
                # 不 break 是有意的：
                # 如果目录里出现一次标题，正文里又出现一次标题，
                # 这里会保留最后一次匹配，更容易跳过目录定位到正文。
                start_line = line_no
                start_level = heading_level or 1

        if start_line is None:
            return f"章节提取失败：在文件 {filename} 中没有找到章节：{section_name}"

        end_line = len(lines)

        # 4. 从章节开始行往后找结束行。
        # 遇到同级或更高级标题，说明当前章节结束。
        for line_no in range(start_line + 1, len(lines)):
            line_title = lines[line_no].strip()
            lower_line_title = line_title.lower()
            heading_level = None

            # 结束判断和开始判断使用同一套标题格式规则，
            # 这样英文 section、中文章节、Markdown、数字标题都按同一逻辑处理。
            for pattern, level_rule in heading_patterns:
                match = re.match(pattern, line_title, re.IGNORECASE)
                if not match:
                    continue

                if level_rule == "markdown":
                    heading_level = len(match.group(1))
                elif level_rule == "chinese_chapter":
                    heading_level = 2 if match.group(1) == "节" else 1
                elif level_rule == "dotted_number":
                    heading_level = match.group(1).count(".") + 1
                else:
                    heading_level = level_rule
                break

            if heading_level is not None and heading_level <= start_level:
                end_line = line_no
                break

        start_char = sum(len(line) for line in lines[:start_line])
        end_char = sum(len(line) for line in lines[:end_line])

        # 5. 根据字符位置截取章节正文。
        section_text = content[start_char:end_char].strip()

        # 6. 如果章节太长，只返回前 MAX_SECTION_CHARS 个字符。
        is_truncated = len(section_text) > MAX_SECTION_CHARS
        if is_truncated:
            section_text = section_text[:MAX_SECTION_CHARS]

        result = (
            f"已从文件 {filename} 中提取章节：{section_name}\n\n"
            f"{section_text}"
        )

        if is_truncated:
            result += f"\n\n注意：章节内容较长，只返回前 {MAX_SECTION_CHARS} 个字符。"

        return result

    except Exception as e:
        return f"章节提取失败：{e}"
    
