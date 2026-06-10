"""
项目配置文件。

把所有“可能经常调整的值”集中放在这里，其他模块直接 import。
这样以后要改模型名、文件大小限制、历史轮数时，不需要到处翻代码。
"""

from pathlib import Path



# 项目根目录。
# 这里的 __file__ 指 config.py 自己，parent 就是 D:\agent_demo。
PROJECT_DIR = Path(__file__).parent


MEMORY_FILE = PROJECT_DIR / "memory.json"

# 直接读取 txt 全文的最大大小。
# 超过这个大小时，read_text_file 只返回前 PREVIEW_CHARS 个字符预览。
MAX_FILE_SIZE = 10 * 1024
PREVIEW_CHARS = 3000

# 搜索关键词时，返回关键词前后各多少字符作为上下文。
SEARCH_CONTEXT_CHARS = 800

# 搜索结果最多返回几个匹配位置，避免一个常见词返回太多内容。
MAX_SEARCH_MATCHES = 5

# 章节提取最多返回多少字符，避免大章节一次性塞满模型上下文。
MAX_SECTION_CHARS = 12000

# 命令行会话里最多保留最近几组“用户-助手”对话。
MAX_HISTORY_PAIRS = 4

# 模型名和 DeepSeek 兼容 OpenAI SDK 的 API 地址。
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
