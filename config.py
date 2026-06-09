from pathlib import Path

PROJECT_DIR = Path(__file__).parent

MAX_FILE_SIZE = 10 * 1024
PREVIEW_CHARS = 3000
SEARCH_CONTEXT_CHARS = 800
MAX_SEARCH_MATCHES = 5
MAX_SECTION_CHARS = 12000
MAX_HISTORY_PAIRS = 4

MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"