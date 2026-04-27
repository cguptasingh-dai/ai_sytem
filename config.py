import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Gemini 2.5 Pro — stable Pro tier with much higher free-tier quotas than the
# 3.1 preview models. Each react agent fires many LLM calls per task (one per
# tool-use iteration), so a 14-agent pipeline easily hits 100+ calls/minute,
# which the 3.1 preview free tier (~2 RPM) cannot serve.
# To switch back to the preview, set:
#   GEMINI_MODEL_PRO       = "gemini-3.1-pro-preview"
#   GEMINI_MODEL_PRO_TOOLS = "gemini-3.1-pro-preview-customtools"
GEMINI_MODEL_PRO = "gemini-2.5-pro"
GEMINI_MODEL_PRO_TOOLS = "gemini-2.5-pro"

BASE_DIR = os.environ.get("PROJECT_PATH", "./generated_project")

def get_dirs():
    return {
        "CODE_DIR": f"{BASE_DIR}/src",
        "TESTS_DIR": f"{BASE_DIR}/tests",
        "DOCS_DIR": f"{BASE_DIR}/docs",
        "LOGS_DIR": f"{BASE_DIR}/logs",
    }

CODE_DIR = f"{BASE_DIR}/src"
TESTS_DIR = f"{BASE_DIR}/tests"
DOCS_DIR = f"{BASE_DIR}/docs"
LOGS_DIR = f"{BASE_DIR}/logs"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

MAX_PARALLEL_AGENTS = 4
RECURSION_LIMIT = 50

# Lead Engineer review loop: how many rework cycles before forcing forward progress
MAX_REVIEW_ITERATIONS = 2

# Keywords that trigger the AI/ML specialist track (data_engineer, ml_researcher,
# prompt_engineer, mlops, ai_dev). For non-AI projects we skip these to save time.
AI_KEYWORDS = [
    "ai", "ml ", "llm", "gemini", "openai", "anthropic", "claude",
    "rag", "embedding", "vector", "model", "nlp", "machine learning",
    "neural", "gpt", "fine-tun", "transformer", "chatbot", "agent",
    "summariz", "classif", "recommend", "generate", "completion",
    "prompt", "inference", "ml-",
]