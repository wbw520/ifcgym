"""Vendored from ifc-bonsai-mcp src/blender_mcp/mcp_instance.py (@ 6215493). The original was

    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("BlenderMCP", instructions="Blender integration through the Model Context Protocol")

Only the instructions string is needed here (it is the first line of our system prompt); the registry lives in runtime.py."""

NAME = "BlenderMCP"
INSTRUCTIONS = "Blender integration through the Model Context Protocol"
