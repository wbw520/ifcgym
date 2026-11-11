"""
Scene-related functions for the IFC Bonsai MCP addon.
Simple Blender-only operations for scene information.

Model-friendly improvements:
- Optional inclusion of world-space AABB and transform data in listings.
- Consistent geometry block in detailed object info.

Route-2 port (IfcEdit, 2026-11-03): this module read Blender scene objects (bpy.context.scene.objects, mathutils, bonsai.tool).
Without Blender, get_scene_info lists the IfcProducts of the open file with the same fields (name = "<IfcClass>/<Name>" as Bonsai
names its objects, location = the product's world placement in metres, type MESH if it has a representation else EMPTY, visible
True, selected False; include_bbox / include_transform are computed from the IFC geometry and placement, `detailed` = a get_blender_object_info-like block
from placement / body / container / material instead of the Blender object), get_object_info and
get_ifc_scene_overview take the file from ifc_utils.get_ifc_file() and the "geometry" block / selection summaries come from the IFC
placement instead of a Blender object. get_blender_object_info and get_selected_objects (Blender-only) are not ported.
"""

import math
import traceback
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api import register_command
from typing import List, Optional, Dict, Any, Union
from ifcedit.frameworks.mcp4ifc_vendor.blender_addon.api.ifc_utils import get_ifc_file, calculate_unit_scale
import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement
try:
    import ifcopenshell.util.unit as ifc_unit
except Exception:
    ifc_unit = None


def _world_matrix(element, scale: float):
    """4x4 world placement of a product (translation in metres)."""
    try:
        m = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
    except Exception:
        return None
    m = m.copy()
    m[0:3, 3] *= scale
    return m


def _euler_xyz(m) -> List[float]:
    """XYZ Euler angles (radians) of the rotation part, as Blender's rotation_euler would report them."""
    r = m[0:3, 0:3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-9:
        return [math.atan2(r[2, 1], r[2, 2]), math.atan2(-r[2, 0], sy), math.atan2(r[1, 0], r[0, 0])]
    return [math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0]


def _shape_info(element):
    """(world AABB min, max, vertex count, face count) in metres from the element's body geometry, or None."""
    try:
        import ifcopenshell.geom
        import numpy as np
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        if len(verts) == 0:
            return None
        return verts.min(axis=0).tolist(), verts.max(axis=0).tolist(), len(verts), len(shape.geometry.faces) // 3
    except Exception:
        return None


def _world_bbox(element):
    """World-space axis-aligned bounding box (metres), or None."""
    si = _shape_info(element)
    return (si[0], si[1]) if si else None


def _blender_like_object_info(element, scale: float, si=None) -> Dict[str, Any]:
    """Stand-in for get_blender_object_info(obj.name) (the `detailed` block of get_scene_info): the same keys the Blender object
    carried - name, type, location, rotation, scale, visible, selected, parent, geometry, materials, mesh_info, bounding_box -
    filled from the IFC placement, body geometry, container and material."""
    mw = _world_matrix(element, scale)
    loc = [float(v) for v in mw[0:3, 3]] if mw is not None else [0.0, 0.0, 0.0]
    rot = _euler_xyz(mw) if mw is not None else [0.0, 0.0, 0.0]
    if si is None and getattr(element, "Representation", None):
        si = _shape_info(element)
    dims = [float(si[1][i] - si[0][i]) for i in range(3)] if si else [0.0, 0.0, 0.0]
    try:
        container = ifcopenshell.util.element.get_container(element)
    except Exception:
        container = None
    info: Dict[str, Any] = {
        "name": f"{element.is_a()}/{getattr(element, 'Name', None) or 'Unnamed'}",
        "type": "MESH" if getattr(element, "Representation", None) else "EMPTY",
        "location": loc, "rotation": rot, "scale": [1.0, 1.0, 1.0], "visible": True, "selected": False,
        "parent": f"{container.is_a()}/{getattr(container, 'Name', None) or 'Unnamed'}" if container else None,
        "geometry": {"location": loc, "rotation": rot, "scale": [1.0, 1.0, 1.0], "dimensions": dims,
                     "matrix_world": [float(v) for row in (mw if mw is not None else [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for v in row]},
        "materials": [],
    }
    try:
        material = ifcopenshell.util.element.get_material(element)
        if material:
            info["materials"] = [getattr(material, "Name", None) or material.is_a()]
    except Exception:
        pass
    if si:
        info["mesh_info"] = {"vertex_count": si[2], "edge_count": None, "face_count": si[3]}
        info["bounding_box"] = {"min": [float(v) for v in si[0]], "max": [float(v) for v in si[1]], "dimensions": dims}
    return info


def _geometry_block(element, scale: float, with_bbox: bool = True) -> Dict[str, Any]:
    m = _world_matrix(element, scale)
    g: Dict[str, Any] = {"location": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "dimensions": [0.0, 0.0, 0.0]}
    if m is not None:
        g["location"] = [float(v) for v in m[0:3, 3]]
        g["rotation"] = _euler_xyz(m)
    if with_bbox:
        bb = _world_bbox(element)
        if bb:
            g["dimensions"] = [float(bb[1][i] - bb[0][i]) for i in range(3)]
    return g


@register_command('get_scene_info', description="Get basic information about the current Blender scene.")
def get_scene_info(
    limit: int = -1,
    offset: int = 0,
    obj_type: Optional[str] = None,
    include_bbox: bool = False,
    include_transform: bool = False,
    round_decimals: int = 3,
    detailed: bool = False
):
    """Get list of Blender objects with basic information including IFC GUIDs.

    Args:
        limit: Max number of objects to return; -1 returns all from offset.
        offset: Start index for pagination.
        obj_type: Filter by Blender object type (e.g., 'MESH').
        include_bbox: When True, include world AABB min/max/dimensions.
        include_transform: When True, include rotation, scale, dimensions, matrix_world.
        round_decimals: Rounding for floats in compact listings.
        detailed: When True, include detailed object information.
    
    Returns:
        Each object includes 'guid' (IFC GlobalId) and 'ifc_class' fields.
    """
    try:
        ifc_file = get_ifc_file()
        scale = calculate_unit_scale(ifc_file)
        all_objects = list(ifc_file.by_type("IfcProduct"))
        if obj_type:
            all_objects = [el for el in all_objects
                           if (obj_type.upper() == "MESH") == bool(getattr(el, "Representation", None))]
        
        selected_objects = all_objects[offset:offset + limit] if limit > 0 else all_objects[offset:]
        
        objects = []
        for element in selected_objects:
            r = round_decimals
            mw = _world_matrix(element, scale)
            loc = [float(v) for v in mw[0:3, 3]] if mw is not None else [0.0, 0.0, 0.0]
            obj_info: Dict[str, Any] = {
                "name": f"{element.is_a()}/{getattr(element, 'Name', None) or 'Unnamed'}",
                "type": "MESH" if getattr(element, "Representation", None) else "EMPTY",
                "location": [round(loc[0], r), round(loc[1], r), round(loc[2], r)],
                "visible": True,
                "selected": False
            }
            
            obj_info["guid"] = getattr(element, 'GlobalId', None)
            obj_info["ifc_class"] = element.is_a()

            bb = si = None
            if include_transform or include_bbox or detailed:
                si = _shape_info(element) if getattr(element, "Representation", None) else None
                bb = (si[0], si[1]) if si else None

            if include_transform:
                rot = _euler_xyz(mw) if mw is not None else [0.0, 0.0, 0.0]
                obj_info["rotation"] = [round(float(v), r) for v in rot]
                obj_info["scale"] = [1.0, 1.0, 1.0]
                dims = [bb[1][i] - bb[0][i] for i in range(3)] if bb else [0.0, 0.0, 0.0]
                obj_info["dimensions"] = [round(float(v), r) for v in dims]
                obj_info["matrix_world"] = [round(float(v), r) for row in (mw if mw is not None else [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for v in row]

            if include_bbox:
                if bb:
                    mins, maxs = bb
                    obj_info["bounding_box"] = {
                        "min": [round(float(mins[0]), r), round(float(mins[1]), r), round(float(mins[2]), r)],
                        "max": [round(float(maxs[0]), r), round(float(maxs[1]), r), round(float(maxs[2]), r)],
                        "dimensions": [round(float(maxs[0]-mins[0]), r),
                                        round(float(maxs[1]-mins[1]), r),
                                        round(float(maxs[2]-mins[2]), r)],
                    }
                else:
                    obj_info["bounding_box"] = None
                
            if detailed:
                obj_info["detailed_info"] = _blender_like_object_info(element, scale, si)
            
            objects.append(obj_info)
        
        return {
            "count": len(objects),
            "total": len(all_objects),
            "objects": objects
        }
    except Exception as e:
        print(f"Error in get_blender_objects: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}
    

# Route-2 port: get_blender_object_info and get_selected_objects (Blender objects / viewport selection) are not ported.


@register_command('get_object_info', description="Get IFC object information")
def get_object_info(
    guids: Optional[Union[str, List[str]]] = None,
    use_selection: bool = False,
    detailed: bool = False
) -> Dict[str, Any]:
    """Get IFC object information from GUIDs or selection."""
    if isinstance(guids, str):
        guids = [guids]
    
    try:
        ifc_file = get_ifc_file()
    except Exception:
        ifc_file = None
    if not ifc_file:
        return {"success": False, "error": "No IFC file loaded"}
    
    objects_info = []
    errors = []
    
    def extract_object_info(element: ifcopenshell.entity_instance, blender_obj: Optional[Any], detailed: bool) -> Dict[str, Any]:
        info = {
            "guid": getattr(element, 'GlobalId', None),
            "ifc_class": element.is_a(),
            "name": getattr(element, 'Name', None),
            "description": getattr(element, 'Description', None),
            "tag": getattr(element, 'Tag', None),
            "id": element.id(),
            "blender_name": blender_obj.name if blender_obj else None
        }
        
        try:
            info["predefined_type"] = ifcopenshell.util.element.get_predefined_type(element)
            element_type = ifcopenshell.util.element.get_type(element)
            info["type"] = {
                "name": element_type.Name,
                "guid": element_type.GlobalId,
                "class": element_type.is_a()
            } if element_type else None
            
            container = ifcopenshell.util.element.get_container(element)
            info["container"] = {
                "name": container.Name,
                "guid": container.GlobalId,
                "class": container.is_a()
            } if container else None
        except:
            info["predefined_type"] = None
            info["type"] = None
            info["container"] = None
        
        if detailed:
            # Get property sets
            try:
                info["property_sets"] = ifcopenshell.util.element.get_psets(element, qtos_only=False) or {}
            except:
                info["property_sets"] = {}
            
            # Get quantities
            try:
                info["quantities"] = ifcopenshell.util.element.get_psets(element, qtos_only=True) or {}
            except:
                info["quantities"] = {}
            
            # Get material info
            try:
                material = ifcopenshell.util.element.get_material(element)
                info["material"] = {
                    "name": getattr(material, 'Name', str(material)),
                    "class": material.is_a()
                } if material else None
            except:
                info["material"] = None
            
            # Get relationships
            relationships = {
                "decomposes": [],
                "decomposed_by": [],
                "associates": [],
                "connected_to": [],
                "connected_from": [],
                "contained_in": None,
                "fills": None
            }
            
            try:
                def make_ref(el):
                    return {
                        "name": getattr(el, 'Name', None),
                        "guid": getattr(el, 'GlobalId', None),
                        "class": el.is_a()
                    }
                
                for rel in getattr(element, 'Decomposes', []):
                    parent = getattr(rel, 'RelatingObject', None)
                    if parent:
                        relationships["decomposes"].append(make_ref(parent))
                
                for rel in getattr(element, 'IsDecomposedBy', []):
                    for child in getattr(rel, 'RelatedObjects', []):
                        relationships["decomposed_by"].append(make_ref(child))
                
                for rel in getattr(element, 'HasAssociations', []):
                    relationships["associates"].append({
                        "type": rel.is_a(),
                        "name": getattr(rel, 'Name', None)
                    })
                
                for rel in getattr(element, 'ConnectedTo', []):
                    connected = getattr(rel, 'RelatedElement', None)
                    if connected:
                        relationships["connected_to"].append(make_ref(connected))
                
                for rel in getattr(element, 'ConnectedFrom', []):
                    connecting = getattr(rel, 'RelatingElement', None)
                    if connecting:
                        relationships["connected_from"].append(make_ref(connecting))
                
                for rel in getattr(element, 'ContainedInStructure', []):
                    structure = getattr(rel, 'RelatingStructure', None)
                    if structure:
                        relationships["contained_in"] = make_ref(structure)
                        break
                
                for rel in getattr(element, 'FillsVoids', []):
                    opening = getattr(rel, 'RelatingOpeningElement', None)
                    if opening:
                        relationships["fills"] = {
                            "opening_guid": getattr(opening, 'GlobalId', None),
                            "class": opening.is_a()
                        }
                        break
            except:
                pass
            
            info["relationships"] = relationships
            
            if hasattr(element, "ObjectPlacement"):   # Route-2: geometry from the IFC placement / body instead of the Blender object
                info["geometry"] = _geometry_block(element, calculate_unit_scale(ifc_file))
        
        return info
    
    if use_selection or not guids:
        # Route-2: no viewport selection exists
        return {"success": False, "error": "No objects selected", "objects": []}
    
    elif guids:
        for guid in guids:
            try:
                element = ifc_file.by_guid(guid)
                if element:
                    objects_info.append(extract_object_info(element, None, detailed))
                else:
                    errors.append(f"GUID {guid} not found")
            except Exception as e:
                errors.append(f"Error processing GUID {guid}: {str(e)}")
    
    return {
        "success": bool(objects_info),
        "objects": objects_info,
        "errors": errors,
        "count": len(objects_info)
    }


@register_command('get_ifc_scene_overview', description='Get comprehensive IFC scene overview')
def get_ifc_scene_overview(include_selection_summary: bool = False) -> Dict[str, Any]:
    """Return consolidated overview of the loaded IFC scene."""
    try:
        ifc_file = get_ifc_file()
    except Exception:
        ifc_file = None
    if not ifc_file:
        return {"success": False, "error": "No IFC file loaded"}

    overview = {"success": True}

    result = {"success": True, "units": {}, "project": {}}
    try:
        project = next(iter(ifc_file.by_type('IfcProject')), None)
        if project:
            result["project"] = {
                "name": getattr(project, 'Name', None),
                "global_id": getattr(project, 'GlobalId', None),
                "schema": getattr(ifc_file, 'schema', None),
            }
    except:
        pass

    if ifc_unit:
        try:
            result["units"] = {
                "length": ifc_unit.calculate_length_unit(ifc_file),
                "area": ifc_unit.calculate_area_unit(ifc_file),
                "volume": ifc_unit.calculate_volume_unit(ifc_file),
                "angle": ifc_unit.calculate_plane_angle_unit(ifc_file),
            }
        except:
            pass
    
    if not result["units"]:
        try:
            project = next(iter(ifc_file.by_type('IfcProject')), None)
            uctx = getattr(project, 'UnitsInContext', None)
            if uctx and getattr(uctx, 'Units', None):
                units = {}
                for u in uctx.Units:
                    try:
                        unit_type = getattr(u, 'UnitType', None)
                        name = getattr(u, 'Name', None) or getattr(u, 'Prefix', None)
                        if unit_type:
                            units[str(unit_type)] = str(name)
                    except:
                        continue
                if units:
                    result["units"] = units
        except:
            pass
    
    if result.get("success"):
        overview.update(result)
    else:
        overview.update({"project": {}, "units": {}})

    counts = {}
    for el in ifc_file.by_type('IfcElement'):
        cls = el.is_a()
        counts[cls] = counts.get(cls, 0) + 1
    
    overview["class_counts"] = counts
    overview["class_total"] = sum(counts.values())

    def build_tree(element):
        node = {
            "guid": getattr(element, 'GlobalId', None),
            "ifc_class": element.is_a(),
            "name": getattr(element, 'Name', None),
            "contained_elements": 0,
            "children": []
        }
        
        for rel in getattr(element, 'ContainsElements', []):
            node["contained_elements"] += len(getattr(rel, 'RelatedElements', []))
        
        for rel in getattr(element, 'IsDecomposedBy', []):
            for child in getattr(rel, 'RelatedObjects', []):
                node["children"].append(build_tree(child))
        
        return node

    projects = [build_tree(p) for p in ifc_file.by_type('IfcProject')]
    overview["spatial"] = projects
    
    counts = {"projects": 0, "sites": 0, "buildings": 0, "storeys": 0, "spaces": 0}
    
    def walk(tree):
        yield tree
        for child in tree.get("children", []):
            yield from walk(child)

    for project in projects:
        for node in walk(project):
            cls = node.get("ifc_class", "").lower()
            if cls == "ifcproject":
                counts["projects"] += 1
            elif cls == "ifcsite":
                counts["sites"] += 1
            elif cls == "ifcbuilding":
                counts["buildings"] += 1
            elif cls == "ifcbuildingstorey":
                counts["storeys"] += 1
            elif cls == "ifcspace":
                counts["spaces"] += 1
    
    overview["summary"] = counts

    if include_selection_summary:
        # Route-2: no viewport selection exists
        overview["selection_summary"] = {"success": False, "error": "No objects selected", "count": 0}

    return overview