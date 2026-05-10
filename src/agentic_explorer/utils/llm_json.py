"""Helpers for extracting JSON payloads from LLM responses."""

from __future__ import annotations

import json
from typing import Any


def normalize_llm_text(content: Any) -> str:
    """Normalize LangChain/Gemini response content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
        return "".join(text_parts)
    return str(content)


def extract_json_text(content: Any) -> str:
    """Extract raw JSON text from plain text or fenced Markdown blocks."""
    text = normalize_llm_text(content).strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def parse_json_from_llm(content: Any) -> Any:
    """Parse JSON content returned by an LLM response."""
    return json.loads(extract_json_text(content))


def extract_yaml_text(content: Any) -> str:
    """Extract raw YAML text from plain text or fenced Markdown blocks."""
    text = normalize_llm_text(content).strip()
    if "```yaml" in text:
        return text.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text

