from __future__ import annotations
from google.adk.tools.tool_context import ToolContext

def mark_flow_done(tool_context: ToolContext) -> dict:
    """Encerra o fluxo inteiro."""
    tool_context.state["flow_done"] = True
    tool_context.actions.skip_summarization = True
    return {}
