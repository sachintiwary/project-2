"""
Configuration for LLM Quiz Solver Agent
Using gpt-4o-mini for cost-efficient tool calling
"""
import os

# Authentication
MY_SECRET = os.getenv("MY_SECRET", "TDS2025_Q7x9Kp2mNvL8wR4jF6hB3yZ")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")

# AI Pipe API settings
OPENAI_BASE_URL = "https://aipipe.org/openai/v1"

# Model - using gpt-4o for best function calling
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Browser settings
BROWSER_TIMEOUT = 30000  # 30 seconds

# Agent settings
MAX_AGENT_STEPS = 10  # Max reasoning steps per question
TASK_TIMEOUT = 180  # 3 minutes per question

# User email for personalized questions
USER_EMAIL = "23f3003663@ds.study.iitm.ac.in"
