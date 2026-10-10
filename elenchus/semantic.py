"""Local model availability checks + optional semantic narration.

Everything deterministic works without this module. It answers two
questions: is Ollama up, and is the model pulled — plus an optional
plain-language narration layer for the web UI.
"""
from __future__ import annotations

import json
import os
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("ELENCHUS_MODEL", "gemma3:4b")


def ollama_available(host: str = OLLAMA_HOST, timeout: int = 3) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            json.loads(r.read())
        return True
    except Exception:
        return False


def model_pulled(model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST,
                 timeout: int = 3) -> bool:
    """True if a model with the same base name is pulled (tag-agnostic)."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read())
        base = model.split(":")[0]
        return any(m.get("name", "").split(":")[0] == base
                   for m in data.get("models", []))
    except Exception:
        return False


def list_models(host: str = OLLAMA_HOST, timeout: int = 3) -> list:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read())
        return [m.get("name") for m in data.get("models", [])]
    except Exception:
        return []
