"""项目统一配置。"""

from pathlib import Path


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

# Agent 上下文与执行预算
MAX_HISTORY_PAIRS = 4
MAX_AGENT_STEPS = 5

# Document Pipeline
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATE_K = 20

# LLM
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
