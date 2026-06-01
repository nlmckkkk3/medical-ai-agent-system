"""Shared utility functions for JSON parsing and common operations."""

import json
import re


def parse_json_response(raw: str) -> dict | None:
    """Robust JSON parser that handles markdown code blocks and whitespace.
    
    Attempts direct parsing first; if that fails, tries to extract JSON
    from markdown code blocks. Returns None if all parsing attempts fail.
    """
    if not raw or not raw.strip():
        return None
    
    text = raw.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extract from markdown code block (```json ... ```)
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object by scanning for { and }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    
    return None
