#!/usr/bin/env python3
"""Parse Qwen2.5-VL output to extract tool call information."""

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedPrediction:
    tool_called: str | None
    tool_arguments: dict | None
    raw_output: str
    parse_error: str | None = None


def parse_model_output(raw_output: str) -> ParsedPrediction:
    """Parse Qwen2.5-VL output to extract the first tool call.

    Qwen2.5-VL-Instruct uses a Hermes-style format for tool calls:

        <tool_call>
        {"name": "function_name", "arguments": {...}}
        </tool_call>

    The model may also produce plain text with no tool call, or
    multiple tool calls (we take the first).
    """
    tool_call = _extract_tool_call_tags(raw_output)
    if tool_call:
        return tool_call

    tool_call = _extract_function_call_json(raw_output)
    if tool_call:
        return tool_call

    return ParsedPrediction(
        tool_called=None,
        tool_arguments=None,
        raw_output=raw_output,
    )


def _extract_tool_call_tags(raw_output: str) -> ParsedPrediction | None:
    """Extract tool call from <tool_call>...</tool_call> tags.

    Also handles truncated output where the closing tag is missing
    (e.g. model hit max_new_tokens before finishing).
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    match = re.search(pattern, raw_output, re.DOTALL)

    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            return ParsedPrediction(
                tool_called=None,
                tool_arguments=None,
                raw_output=raw_output,
                parse_error=f"JSON decode error in <tool_call>: {e}",
            )

        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}

        return ParsedPrediction(
            tool_called=name,
            tool_arguments=arguments,
            raw_output=raw_output,
        )

    open_match = re.search(r"<tool_call>\s*", raw_output)
    if not open_match:
        return None
    fragment = raw_output[open_match.end():]
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', fragment)
    if not name_match:
        return None
    return ParsedPrediction(
        tool_called=name_match.group(1),
        tool_arguments=None,
        raw_output=raw_output,
        parse_error="truncated output, extracted tool name only",
    )


def _extract_function_call_json(raw_output: str) -> ParsedPrediction | None:
    """Fallback: look for JSON objects containing a function/tool name field."""
    patterns = [
        r'\{"name"\s*:\s*"([^"]+)".*?\}',
        r'\{"function"\s*:\s*\{"name"\s*:\s*"([^"]+)".*?\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            name = match.group(1)
            try:
                json_str = match.group(0)
                payload = json.loads(json_str)
                arguments = payload.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                return ParsedPrediction(
                    tool_called=name,
                    tool_arguments=arguments,
                    raw_output=raw_output,
                )
            except json.JSONDecodeError:
                return ParsedPrediction(
                    tool_called=name,
                    tool_arguments=None,
                    raw_output=raw_output,
                )

    return None
