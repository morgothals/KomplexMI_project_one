"""Configuration helpers for LLM integrations."""
import os

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv(
    "LLM_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))


def is_configured() -> bool:
    return bool(LLM_API_KEY)
