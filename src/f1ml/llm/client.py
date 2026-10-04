"""Minimal Ollama chat client (http://localhost:11434/api/chat)."""
from __future__ import annotations

import json

import requests

from ..config import settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None, timeout: int = 180):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    def check(self) -> None:
        """Fail early with a clear message if Ollama is not running or the model is missing."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Start the Ollama app (or `ollama serve`)."
            ) from e
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(
                f"Model {self.model!r} is not pulled. Run: ollama pull {self.model}"
            )

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 260) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            r = requests.post(f"{self.host}/api/chat", data=json.dumps(payload), timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"Ollama request failed: {e}") from e
        return r.json()["message"]["content"].strip()
