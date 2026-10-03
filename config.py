"""项目统一配置。"""

from pathlib import Path


PROJECT_DIR = Path(__file__).parent

# 用户本地知识库。
DATA_DIR = PROJECT_DIR / "data"

# 可重新生成的 manifest、文本块、FAISS 索引和原始向量，不提交到 Git。
INDEX_DIR = PROJECT_DIR / ".localdoc_index"


# Agent

MAX_HISTORY_PAIRS = 4
MAX_AGENT_STEPS = 5

# 单个问题最多真正执行几次知识库搜索。
MAX_SEARCH_CALLS = 3


# Document Pipeline

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# Retrieval

RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATE_K = 20


# LLM

MODEL_NAME = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
