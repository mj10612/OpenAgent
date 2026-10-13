"""Render external tool metadata as bounded escaped data."""

from __future__ import annotations

import json
from html import escape

from ..core.types import ToolSpec


def render_tool_metadata(spec: ToolSpec) -> str:
    data = {
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.to_openai_schema()["function"]["parameters"],
    }
    serialized = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    # Markdown fences are escaped too, so metadata cannot create protocol blocks.
    return (
        "<tool_description>\n" + escape(serialized).replace("`", "&#96;") + "\n</tool_description>"
    )
