# app_agents.py
from __future__ import annotations
from google.adk.tools.tool_context import ToolContext

def mark_quality_done(tool_context: ToolContext) -> dict:
    """Encerra o loop de qualidade."""
    tool_context.state["quality_done"] = True
    tool_context.actions.skip_summarization = True
    return {}