"""
Configuration for LLM Quiz Solver Agent
Using DeepSeek-V3 via GitHub Models + Gemini 2.5 Flash for audio
"""
import os

# Authentication
MY_SECRET = os.getenv("MY_SECRET", "TDS2025_Q7x9Kp2mNvL8wR4jF6hB3yZ")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")

# GitHub Models API (for DeepSeek-V3)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Gemini API for audio transcription
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# GitHub Models API settings (OpenAI-compatible)
OPENAI_BASE_URL = "https://models.github.ai/inference"

# Model - using DeepSeek-V3 for advanced reasoning
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/DeepSeek-V3-0324")
BROWSER_TIMEOUT = 30000  # 30 seconds

# Agent settings
MAX_AGENT_STEPS = 10  # Max reasoning steps per question
TASK_TIMEOUT = 180  # 3 minutes per question

# User email for personalized questions
USER_EMAIL = "23f3003663@ds.study.iitm.ac.in"

