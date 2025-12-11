"""
Configuration for LLM Quiz Solver
"""
import os

# Your secret string for verification (use env var in production)
MY_SECRET = os.getenv("MY_SECRET", "TDS2025_Q7x9Kp2mNvL8wR4jF6hB3yZ")

# AI Pipe configuration (OpenAI-compatible API)
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
OPENAI_BASE_URL = "https://aipipe.org/openai/v1"

# LLM Model - using best available
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Timeout settings
BROWSER_TIMEOUT = 30000  # 30 seconds for page load
TASK_TIMEOUT = 180  # 3 minutes total for solving
