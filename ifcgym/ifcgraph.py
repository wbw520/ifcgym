"""Semantic property graph of an IFC file and the edit graph between two files (BIM-Edit Sec. 3.2: G=(N,R); G_delta = added /
removed / modified nodes and relations). Nodes = IfcProduct instances with a GlobalId (elements, spaces, openings, storeys...);
relations = IfcRelationship instances, expressed as typed edges between node GlobalIds. Geometry is sampled lazily.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import ifcopenshell
import ifcopenshell.util.element as eu

PROPERTY_IGNORE = {"Tag", "Description", "LongName"}


def _scalar(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return tuple(_scalar(x) for x in v)
    if isinstance(v, ifcopenshell.entity_instance):
        return ("#", v.is_a(), getattr(v, "GlobalId", None) or v.id())
    return str(v)


@dataclass
class Node:
    guid: str
    cls: str
    predefined: str | None
    name: str | None
    attrs: dict                      # direct attributes (ObjectType, PredefinedType, Name, Tag ...) + property sets flattened
    geom_hash: str | None            # hash of the representation subtree (cheap change detector)
    placement_hash: str | None


@dataclass
class Edge:
    cls: str                         # IfcRelVoidsElement, IfcRelFillsElement, IfcRelContainedInSpatialStructure, IfcRelConnectsPathElements, IfcRelSpaceBoundary ...
    src: str                         # relating object guid
    dst: str                         # related object guid
    key: tuple = field(default_factory=tuple)

    def ident(self) -> tuple:
        return (self.cls, self.src, self.dst) + tuple(self.key)


def _hash_subtree(inst, depth: int = 6, _seen=None) -> str:
    """Stable content hash of an entity subtree (ignores STEP ids and GlobalIds)."""
    h = hashlib.sha1()
    stack = [(inst, 0)]
    seen = set()
    while stack:
        e, d = stack.pop()
        if e.id() in seen or d > depth:
            continue
        seen.add(e.id())
        h.update(e.is_a().encode())
        for i, v in enumerate(e):
            if e.attribute_name(i) in ("GlobalId", "OwnerHistory"):
                continue
            if isinstance(v, ifcopenshell.entity_instance):
                stack.append((v, d + 1))
            elif isinstance(v, (tuple, list)):
                for x in v:
                    if isinstance(x, ifcopenshell.entity_instance):
                        stack.append((x, d + 1))
                    else:
                        h.update(repr(_scalar(x)).encode())
            else:
                h.update(repr(_scalar(v)).encode())
    return h.hexdigest()


def rel_edges(rel) -> list[Edge]:
    cls = rel.is_a()
    out: list[Edge] = []
    def g(x):
        return getattr(x, "GlobalId", None)
    if cls == "IfcRelVoidsElement":
        out.append(Edge(cls, g(rel.RelatingBuildingElement), g(rel.RelatedOpeningElement)))
    elif cls == "IfcRelFillsElement":
        out.append(Edge(cls, g(rel.RelatingOpeningElement), g(rel.RelatedBuildingElement)))
    elif cls == "IfcRelContainedInSpatialStructure":
        for e in rel.RelatedElements or []:
            out.append(Edge(cls, g(rel.RelatingStructure), g(e)))
    elif cls == "IfcRelAggregates":
        for e in rel.RelatedObjects or []:
            out.append(Edge(cls, g(rel.RelatingObject), g(e)))
    elif cls == "IfcRelConnectsPathElements" or cls == "IfcRelConnectsElements":
        out.append(Edge(cls, g(rel.RelatingElement), g(rel.RelatedElement)))
    elif cls == "IfcRelSpaceBoundary" or cls.startswith("IfcRelSpaceBoundary"):
        out.append(Edge("IfcRelSpaceBoundary", g(rel.RelatingSpace), g(rel.RelatedBuildingElement) if rel.RelatedBuildingElement else None))
    elif cls == "IfcRelDefinesByType":
        t = rel.RelatingType
        for e in rel.RelatedObjects or []:
            out.append(Edge(cls, g(e), None, (t.is_a(), t.Name)))
    elif cls == "IfcRelAssociatesMaterial":
        m = rel.RelatingMaterial
        for e in rel.RelatedObjects or []:
            out.append(Edge(cls, g(e), None, (m.is_a() if m else None, getattr(m, "Name", None))))
    elif cls == "IfcRelConnectsPorts" or cls == "IfcRelNests":
        for e in getattr(rel, "RelatedObjects", []) or []:
            out.append(Edge(cls, g(getattr(rel, "RelatingObject", None)), g(e)))
    return [e for e in out if e.src is not None or e.dst is not None]


def build_graph(f: ifcopenshell.file) -> tuple[dict[str, Node], dict[tuple, Edge]]:
    nodes: dict[str, Node] = {}
    for p in f.by_type("IfcProduct"):
        gid = getattr(p, "GlobalId", None)
        if not gid:
            continue
        attrs = {}
        for i, v in enumerate(p):
            n = p.attribute_name(i)
            if n in ("GlobalId", "OwnerHistory", "ObjectPlacement", "Representation") or n in PROPERTY_IGNORE:
                continue
            if isinstance(v, ifcopenshell.entity_instance) or isinstance(v, (tuple, list)):
                continue
            attrs[n] = v
        try:
            psets = eu.get_psets(p, psets_only=True)
            for pname, props in psets.items():
                for k, v in props.items():
                    if k == "id" or k in PROPERTY_IGNORE:
                        continue
                    attrs[f"{pname}.{k}"] = _scalar(v)
        except Exception:  # noqa: BLE001
            pass
        rep = getattr(p, "Representation", None)
        pl = getattr(p, "ObjectPlacement", None)
        nodes[gid] = Node(gid, p.is_a(), attrs.get("PredefinedType"), attrs.get("Name"), attrs,
                          _hash_subtree(rep, depth=24) if rep else None, _hash_subtree(pl, depth=24) if pl else None)
    edges: dict[tuple, Edge] = {}
    for r in f.by_type("IfcRelationship"):
        for e in rel_edges(r):
            edges[e.ident()] = e
    return nodes, edges


@dataclass
class EditGraph:
    added: dict[str, Node]
    removed: dict[str, Node]
    modified: dict[str, tuple[Node, Node]]        # guid -> (before, after)
    rel_added: dict[tuple, Edge]
    rel_removed: dict[tuple, Edge]

    @property
    def node_ids(self) -> set[str]:
        return set(self.added) | set(self.removed) | set(self.modified)

    def summary(self) -> dict:
        return {"added": len(self.added), "removed": len(self.removed), "modified": len(self.modified),
                "rel_added": len(self.rel_added), "rel_removed": len(self.rel_removed)}


def edit_graph(before: ifcopenshell.file, after: ifcopenshell.file, drop_edges_of_edited_nodes: bool = True) -> EditGraph:
    return edit_graph_from(build_graph(before), after)


def edit_graph_from(before_graph: tuple, after: ifcopenshell.file) -> EditGraph:
    """Edit graph from a (cached) graph of the original model and the edited file."""
    return diff_graphs(before_graph, build_graph(after))


def diff_graphs(before_graph: tuple, after_graph: tuple) -> EditGraph:
    """Edit graph from two prebuilt graphs (2026-10-26: lets the BIM-Edit feature cache reuse the graphs of 349 MB models)."""
    nb, eb = before_graph; na, ea = after_graph
    added = {g: n for g, n in na.items() if g not in nb}
    removed = {g: n for g, n in nb.items() if g not in na}
    modified = {}
    for g in nb.keys() & na.keys():
        x, y = nb[g], na[g]
        if x.cls != y.cls or x.attrs != y.attrs or x.geom_hash != y.geom_hash or x.placement_hash != y.placement_hash:
            modified[g] = (x, y)
    rel_added = {k: e for k, e in ea.items() if k not in eb}
    rel_removed = {k: e for k, e in eb.items() if k not in ea}
    return EditGraph(added, removed, modified, rel_added, rel_removed)


def dump(eg: EditGraph) -> str:
    return json.dumps({"added": sorted(eg.added), "removed": sorted(eg.removed), "modified": sorted(eg.modified),
                       "rel_added": [list(map(str, k)) for k in eg.rel_added], "rel_removed": [list(map(str, k)) for k in eg.rel_removed]}, indent=1)


def shape_matrix(shp) -> "np.ndarray":
    """4x4 world transform of an ifcopenshell.geom shape. ifcopenshell 0.8 returns 16 floats COLUMN-major (translation in the last
    four), older versions 12 floats (3 axes + origin). Verified 2026-10-19 on wbdg_office wall #76 (A-E case outputs)."""
    import numpy as np
    mat = shp.transformation.matrix
    m = np.array(getattr(mat, "data", mat), dtype=float).ravel()
    M = np.eye(4)
    if m.size == 12:
        M[:3, :3] = m.reshape(4, 3)[:3].T; M[:3, 3] = m.reshape(4, 3)[3]
    elif m.size == 16:
        M = m.reshape(4, 4).T          # column-major -> row-major
        if not np.allclose(M[3, :3], 0) and np.allclose(M[:3, 3], 0):
            M = M.T                    # it was row-major already
    return M


def world_verts(shp) -> "np.ndarray":
    import numpy as np
    v = np.array(shp.geometry.verts, dtype=float).reshape(-1, 3)
    M = shape_matrix(shp)
    return (M[:3, :3] @ v.T).T + M[:3, 3]
