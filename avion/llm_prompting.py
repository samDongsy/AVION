"""LLM-based domain prompting (offline).

Two modes:
- ``load_class_descriptions(json_path)``: read pre-generated descriptions.
- ``generate_with_gemini``: optional online generation through Gemini 2.5 Flash.
  Requires the env var ``GEMINI_API_KEY`` and the ``google-generativeai``
  package. The training pipeline only uses the JSON cache, so this is purely
  for users who want to reproduce the prompt-generation step.

The query template comes from Appendix C of AVION:

    "Generate N overhead-view descriptions of [CLASS] from satellite imagery,
     highlighting class-specific scene elements while avoiding any
     ground-level terms."
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROMPT_TEMPLATE = (
    "Generate {n} overhead-view descriptions of {class_name} from satellite "
    "imagery, highlighting class-specific scene elements while avoiding any "
    "ground-level terms. Return each description on its own line, no "
    "bullet points or numbering."
)


def load_class_descriptions(json_path: str | os.PathLike) -> Dict[str, List[str]]:
    """Load cached class -> [descriptions] mapping from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_class_descriptions(mapping: Dict[str, List[str]], json_path: str | os.PathLike) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def build_query(class_name: str, n: int = 30) -> str:
    return PROMPT_TEMPLATE.format(n=n, class_name=class_name.replace("_", " "))


def generate_with_gemini(
    class_names: Sequence[str],
    n_per_class: int = 30,
    model: str = "gemini-2.5-flash",
    sleep_sec: float = 0.5,
    api_key: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Call Gemini 2.5 Flash to generate ``n_per_class`` descriptions for each class.

    Falls back to ImportError if ``google-generativeai`` isn't installed.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "google-generativeai is required for online prompt generation. "
            "Install with `pip install google-generativeai`."
        ) from e

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY env var or pass api_key.")
    genai.configure(api_key=key)
    llm = genai.GenerativeModel(model)

    out: Dict[str, List[str]] = {}
    for cls in class_names:
        prompt = build_query(cls, n_per_class)
        resp = llm.generate_content(prompt)
        text = resp.text or ""
        lines = [ln.strip(" -*\t") for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        out[cls] = lines
        time.sleep(sleep_sec)
    return out
