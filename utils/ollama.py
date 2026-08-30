"""Optional Ollama client used only for bonus narrative explanations."""

from __future__ import annotations

import os
from typing import Any

import requests


def ollama_chat(
    prompt: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 45.0,
) -> str | None:
    endpoint = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    selected_model = model or os.getenv("OLLAMA_MODEL", "gemma3:4b")
    try:
        response = requests.post(
            f"{endpoint}/api/chat",
            json={
                "model": selected_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        content = payload.get("message", {}).get("content")
        return str(content).strip() if content else None
    except (requests.RequestException, ValueError, TypeError):
        return None


__all__ = ["ollama_chat"]
