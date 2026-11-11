"""Route-2 port of the ifc-bonsai-mcp Blender add-on (Show2Instruct/ifc-bonsai-mcp @ 6215493, MIT).

The original package root registered a Blender add-on (bl_info, bpy UI panel, socket server in core.py). None of that is needed
when IfcOpenShell is used directly: this file only makes `blender_addon.api` and `blender_addon.commands` importable, which keeps
the sandbox's import whitelist (`blender_addon.api.*`, code.py) and the ICL prelude of execute_ifc_code_tool (`from
blender_addon.api.ifc_utils import ...`) valid without changes.
"""
