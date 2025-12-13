"""
Configuration for God-Level LLM Quiz Solver Agent
"""
import os

# ============================================================
# AUTHENTICATION
# ============================================================
MY_SECRET = os.getenv("MY_SECRET", "")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ============================================================
# API SETTINGS
# ============================================================
OPENAI_BASE_URL = "https://aipipe.org/openai/v1"
LLM_MODEL = "gpt-5-mini"

# ============================================================
# USER SETTINGS
# ============================================================
USER_EMAIL = "23f3003663@ds.study.iitm.ac.in"
EMAIL_LENGTH = len(USER_EMAIL)  # 30

# ============================================================
# AGENT SETTINGS
# ============================================================
MAX_STEPS = 10
TIMEOUT = 60
BROWSER_TIMEOUT = 30000
