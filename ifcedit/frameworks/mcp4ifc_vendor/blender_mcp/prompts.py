'''
Prompts
'''
# Vendored from ifc-bonsai-mcp src/blender_mcp/mcp_functions/prompts.py (@ 6215493); the `@mcp.prompt()` decorator and the
# `from ..mcp_instance import mcp` import were removed, the text is verbatim (including the "#ToDo:" item).


def ifc_building_element_creation_strategy() -> str:
    """Defines the preferred strategy for creating IFC building elements"""
    return """
    When creating IFC building elements in Bonsai (formerly BlenderBIM), follow these guidelines:

    1. IFC Structure and Hierarchy:
       - Respect the IFC hierarchy: Project → Site → Building → Building Story → Building Elements
       - Every element must belong to a proper container (typically a Building Story)
       - Use proper IFC entity types (IfcWall, IfcSlab, IfcDoor, etc.) that match the element's function

    2. #ToDo:

    Following these guidelines will ensure a well-structured, standards-compliant IFC model.
    """
