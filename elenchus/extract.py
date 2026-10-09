"""Extraction via a local Ollama model.

The ONLY file that touches a model. Everything downstream is
deterministic, which is why the demo can cache this layer's output and
survive any model flakiness.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

from .schemas import EXTRACTION_SCHEMA, Extraction

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("ELENCHUS_MODEL", "gemma3:4b")

PROMPT = (
    "Extract every BILLED line item (charges only — NOT metadata rows "
    "like consumer number, address, dates, rate slabs) and the stated "
    "total from this bill. Use the flat fields provided."
)


def extract_from_image(image_path: str, model: str = DEFAULT_MODEL,
                       host: str = OLLAMA_HOST, timeout: int = 180) -> Extraction:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "format": EXTRACTION_SCHEMA,
        "images": [img_b64],
        "options": {"temperature": 0, "seed": 42},
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return Extraction.from_dict(json.loads(data["response"]))


def extract_from_text(text: str, model: str = DEFAULT_MODEL,
                      host: str = OLLAMA_HOST, timeout: int = 180) -> Extraction:
    body = json.dumps({
        "model": model,
        "prompt": f"{PROMPT}\n\n--- DOCUMENT ---\n{text}",
        "stream": False,
        "format": EXTRACTION_SCHEMA,
        "options": {"temperature": 0, "seed": 42},
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return Extraction.from_dict(json.loads(data["response"]))


def cache_path_for(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + ".extraction.json"


def extract_with_cache(image_path: str, **kw) -> Extraction:
    """Cache the model's output next to the fixture. The demo replays
    the cache; the live call refreshes it."""
    cp = cache_path_for(image_path)
    if os.path.isfile(cp):
        with open(cp, encoding="utf-8") as f:
            return Extraction.from_dict(json.load(f))
    ext = extract_from_image(image_path, **kw)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(ext.to_dict(), f, indent=2)
    return ext
