"""
API package for IFC Bonsai MCP addon.
Provides a registry system for command handlers.
"""

_command_registry = {}

def register_command(command_name, description=None):
    """
    Register a command handler function. This function is a decorator factory.

    Args:
        command_name: The name of the command as it will be called via MCP
        description: Optional description of what the command does
    """
    def decorator(handler_function):
        _command_registry[command_name] = {
            'handler': handler_function,
            'description': description or getattr(handler_function, '__doc__', None) or "No description provided"
        }
        return handler_function
    return decorator

def get_command(command_name):
    """
    Get a command handler by name.

    Args:
        command_name: The name of the command

    Returns:
        The handler function or None if not found
    """
    if command_name in _command_registry:
        return _command_registry[command_name]['handler']
    return None

def get_all_commands():
    """
    Get all registered commands.

    Returns:
        Dictionary of command names to their handlers and descriptions
    """
    return _command_registry

from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import scene
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import wall
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import slab
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import window
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import roof
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import door
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import feature
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import style
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import mesh_ifc
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import mesh_trimesh
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import root
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import system
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import code
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import stairs

from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.wall import create_wall, update_wall, get_wall_properties
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.window import create_window, update_window, get_window_properties, get_window_partition_types
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.slab import create_slab, update_slab, get_slab_properties
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.door import create_door, update_door, get_door_properties, get_door_operation_types
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.roof import create_roof, update_roof, delete_roof, get_roof_types
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.style import create_surface_style, create_pbr_style, apply_style_to_object, list_styles, update_style, remove_style
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.mesh_ifc import create_mesh_ifc, list_ifc_entities, get_mesh_examples
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.mesh_trimesh import create_trimesh_ifc, get_trimesh_examples
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.scene import get_scene_info, get_object_info, get_ifc_scene_overview  # Route-2 port: get_blender_object_info, get_selected_objects dropped
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.system import list_commands
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.code import execute_code, ping, execute_ifc_code
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.root import copy_class, reassign_class, delete_ifc_objects
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.feature import get_opening_types, create_opening, fill_opening, remove_opening, remove_filling, get_element_openings, get_opening_info
