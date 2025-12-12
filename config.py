"""
Configuration for LLM Quiz Solver Agent
Using gpt-5-mini for reasoning + Gemini 2.5 Flash for audio
"""
import os

# Authentication
MY_SECRET = os.getenv("MY_SECRET", "TDS2025_Q7x9Kp2mNvL8wR4jF6hB3yZ")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")

# Gemini API for audio transcription
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# AI Pipe API settings
OPENAI_BASE_URL = "https://aipipe.org/openai/v1"

# Model - using gpt-5-mini for advanced reasoning
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")

# Browser settings
BROWSER_TIMEOUT = 30000  # 30 seconds

# Agent settings
MAX_AGENT_STEPS = 10  # Max reasoning steps per question
TASK_TIMEOUT = 180  # 3 minutes per question

# User email for personalized questions
USER_EMAIL = "23f3003663@ds.study.iitm.ac.in"

