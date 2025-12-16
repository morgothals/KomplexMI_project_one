"""Lightweight OpenAI-compatible chat client with graceful fallbacks."""
from __future__ import annotations

import json
from typing import Iterable, List, Mapping
import requests

from . import config


class LLMNotConfigured(RuntimeError):
    """Raised when the API key is missing."""


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = (base_url or config.LLM_API_BASE).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS

    def _headers(self) -> Mapping[str, str]:
        if not self.api_key:
            raise LLMNotConfigured("Missing LLM_API_KEY environment variable")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _as_prompt(self, messages: Iterable[Mapping[str, str]]) -> str:
        # Gemini SDK-hoz egyszerű, determinisztikus prompt stringet állítunk össze.
        parts: list[str] = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            content = m.get("content") or ""
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts).strip()

    def chat(self, messages: Iterable[Mapping[str, str]]) -> str:
        # Gemini (google-genai) útvonal – a user által kért integráció.
        if (config.LLM_PROVIDER or "").lower() == "gemini":
            if not self.api_key:
                raise LLMNotConfigured("Missing LLM_API_KEY environment variable")
            try:
                from google import genai

                client = genai.Client(api_key=self.api_key)
                prompt = self._as_prompt(messages)
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return (getattr(response, "text", None) or "").strip()
            except LLMNotConfigured:
                raise
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"LLM request failed (gemini): {exc}") from exc

        # OpenAI-kompatibilis HTTP útvonal (ha más providert használsz)
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except LLMNotConfigured:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM request failed: {exc}") from exc


def render_messages(system: str, user: str) -> List[Mapping[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
