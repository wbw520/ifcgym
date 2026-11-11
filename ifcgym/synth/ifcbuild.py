"""Synthetic buildings (H001, user 2026-10-21: 'floor plans -> whole buildings'): turn an abstract floor plan into a complete IFC
model with the entities our tasks and checkers rely on - standard walls (axis + extruded profile), wall joints
(IfcRelConnectsPathElements), openings and doors/windows placed relative to the wall, spaces with footprints that meet the wall
faces, space boundaries on the walls, storeys stacked with slabs, property sets. Export 'styles' imitate real exports: units in
metres or millimetres, IFC4 or IFC2X3, rectangle or polyline wall profiles, Revit- or ArchiCAD-like names.

Plan model (metres, storey-local XY, walls axis-aligned):
    Plan(walls=[WallSeg(p0, p1, thickness, external)], doors=[Opening(wall, t, width, kind)], rooms=[Room(name, polygon)])
    t = position of the opening centre along the wall (0..1); polygon = the room's interior footprint (list of (x, y)).
`demo_plan()` is a two-room flat used by the self-check; `cubicasa.py` produces plans from CubiCasa5K SVGs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.owner
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import ifcopenshell.guid


@dataclass
class WallSeg:
    p0: tuple[float, float]
    p1: tuple[float, float]
    thickness: float = 0.2
    external: bool = False


@dataclass
class Opening:
    wall: int          # index into Plan.walls
    t: float           # centre position along the wall, 0..1
    width: float = 0.9
    kind: str = "door"  # door | window
    height: float = 2.1
    sill: float = 0.0


@dataclass
class Room:
    name: str
    polygon: list[tuple[float, float]]
    long_name: str | None = None


@dataclass
class Plan:
    walls: list[WallSeg]
    doors: list[Opening]
    rooms: list[Room]
    height: float = 2.8
    outline: list[tuple[float, float]] | None = None
    core: object | None = None        # tiling.Core: elevator shaft / stair well rectangles (slab openings, stair, elevator)


@dataclass
class Style:
    units: str = "m"            # m | mm
    schema: str = "IFC4"        # IFC4 | IFC2X3
    profile: str = "rect"       # rect | polyline   (wall Body profile)
    body: str = "extrusion"     # extrusion | brep | clipped | faceset   (wall / space bodies: faceted Brep like ArchiCAD exports; an extrusion
                                # wrapped in an IfcBooleanClippingResult like Revit walls under roofs - the clip plane here never cuts;
                                # IfcPolygonalFaceSet like Revit 2022+ tessellated exports, IFC4 only - IFC2X3 falls back to brep)
    sparse: bool = False        # Revit-style sparse export: no IfcRelConnectsPathElements and no IfcRelSpaceBoundary (fantasy_* models)
    # door / window / opening styles (E014; survey of the 7 non-evaluation ifc-bench models, outputs/E014/survey_fills.txt: every real
    # wall opening is an elevation profile extruded ACROSS the wall - rectangle with x along the wall, rectangle turned 90 degrees
    # (Revit), polyline (ArchiCAD hitos) or Brep / face set - and doors / windows carry shared type geometry (IfcMappedItem)).
    # The defaults reproduce the original synthetic buildings.
    opening_body: str = "vertical"  # vertical (plan profile extruded up) | through | through_rot | through_poly | brep | faceset
    opening_mix: float = 0.0        # share of openings that take another opening_body at random (exports mix families)
    fill_body: str = "extrusion"    # extrusion | mapped_extrusion | mapped_brep | mapped_faceset   (mapped: one IfcRepresentationMap per type)
    dense: bool = False             # extra windows along external walls, clear of the existing openings and wall junctions
    naming: str = "revit"       # revit | archicad
    storeys: int = 1
    storey_height: float = 3.0
    seed: int = 0

    @property
    def scale(self) -> float:    # model units per metre
        return 1000.0 if self.units == "mm" else 1.0


# ------------------------------------------------------------------------------------------ helpers
def _pt(f, xy, scale):
    return f.createIfcCartesianPoint([float(c * scale) for c in xy])


def _dir(f, v):
    return f.createIfcDirection([float(c) for c in v])


def _a2p3(f, origin, scale, xdir=(1.0, 0.0, 0.0)):
    return f.createIfcAxis2Placement3D(_pt(f, (origin[0], origin[1], origin[2] if len(origin) > 2 else 0.0), scale), _dir(f, (0.0, 0.0, 1.0)), _dir(f, xdir))


def _placement(f, rel_to, origin, scale, xdir=(1.0, 0.0, 0.0)):
    return f.createIfcLocalPlacement(rel_to, _a2p3(f, origin, scale, xdir))


def _brep_prism(f, poly, z0: float, height: float, scale: float):
    """IfcFacetedBrep of a vertical prism over a CCW polygon (local coordinates, metres): bottom, top and one face per edge."""
    n = len(poly)
    bot = [f.createIfcCartesianPoint([float(x * scale), float(y * scale), float(z0 * scale)]) for x, y in poly]
    top = [f.createIfcCartesianPoint([float(x * scale), float(y * scale), float((z0 + height) * scale)]) for x, y in poly]
    def face(pts):
        return f.createIfcFace([f.createIfcFaceOuterBound(f.createIfcPolyLoop(pts), True)])
    faces = [face(list(reversed(bot))), face(top)]
    for i in range(n):
        j = (i + 1) % n
        faces.append(face([bot[i], bot[j], top[j], top[i]]))
    return f.createIfcFacetedBrep(f.createIfcClosedShell(faces))


def _faceset_prism(f, poly, z0: float, height: float, scale: float):
    """IfcPolygonalFaceSet of a vertical prism (IFC4+): one point list, indexed faces (1-based)."""
    n = len(poly)
    coords = [[float(x * scale), float(y * scale), float(z0 * scale)] for x, y in poly] + [[float(x * scale), float(y * scale), float((z0 + height) * scale)] for x, y in poly]
    pl = f.createIfcCartesianPointList3D(coords)
    faces = [f.createIfcIndexedPolygonalFace(list(reversed(range(1, n + 1)))), f.createIfcIndexedPolygonalFace(list(range(n + 1, 2 * n + 1)))]
    for i in range(n):
        j = (i + 1) % n
        faces.append(f.createIfcIndexedPolygonalFace([i + 1, j + 1, n + j + 1, n + i + 1]))
    return f.createIfcPolygonalFaceSet(pl, True, faces, None)


def _clipped(f, solid, z_plane: float, scale: float):
    """IfcBooleanClippingResult(solid minus the half-space above z_plane): the plane lies above the body, so geometry is unchanged
    but the structure is the one Revit writes for walls under roofs."""
    plane = f.createIfcPlane(f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([0.0, 0.0, float(z_plane * scale)]), f.createIfcDirection([0.0, 0.0, 1.0]), f.createIfcDirection([1.0, 0.0, 0.0])))
    return f.createIfcBooleanClippingResult("DIFFERENCE", solid, f.createIfcHalfSpaceSolid(plane, False))


OPENING_BODIES = ("vertical", "through", "through_rot", "through_poly", "brep", "faceset")


def _poly_area(poly):
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return a / 2


class Builder:
    def __init__(self, style: Style, name: str = "synthetic"):
        self.s = style; self.rng = random.Random(style.seed); self.name = name
        self.f = ifcopenshell.api.project.create_file(version=style.schema)
        f = self.f
        # owner history: register a user/application OF THIS FILE for the api (IFC2X3 requires one; the settings are module-level,
        # so they must be re-bound per file - a stale binding from a previous build made every IfcOwnerHistory reference a
        # foreign entity, 1000+ schema errors per file)
        import ifcopenshell.api.owner.settings as owner_settings
        person = f.createIfcPerson(FamilyName="Synthetic", GivenName="IfcEdit"); org = f.createIfcOrganization(Name="IfcEdit synthetic buildings")
        user = f.createIfcPersonAndOrganization(ThePerson=person, TheOrganization=org)
        app = f.createIfcApplication(ApplicationDeveloper=org, Version="1", ApplicationFullName="IfcEdit synth", ApplicationIdentifier="ifcedit-synth")
        owner_settings.get_user = lambda ifc, _u=user, _f=f: _u if ifc is _f else None
        owner_settings.get_application = lambda ifc, _a=app, _f=f: _a if ifc is _f else None
        self.project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name=name)
        length = ifcopenshell.api.unit.add_si_unit(f, unit_type="LENGTHUNIT", prefix=("MILLI" if style.units == "mm" else None))
        area = ifcopenshell.api.unit.add_si_unit(f, unit_type="AREAUNIT"); vol = ifcopenshell.api.unit.add_si_unit(f, unit_type="VOLUMEUNIT")
        ifcopenshell.api.unit.assign_unit(f, units=[length, area, vol])
        self.model_ctx = ifcopenshell.api.context.add_context(f, context_type="Model")
        self.body = ifcopenshell.api.context.add_context(f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=self.model_ctx)
        self.plan_ctx = ifcopenshell.api.context.add_context(f, context_type="Plan")
        self.axis = ifcopenshell.api.context.add_context(f, context_type="Plan", context_identifier="Axis", target_view="GRAPH_VIEW", parent=self.plan_ctx)
        self.site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Site")
        self.building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name=name)
        ifcopenshell.api.aggregate.assign_object(f, relating_object=self.project, products=[self.site])
        ifcopenshell.api.aggregate.assign_object(f, relating_object=self.site, products=[self.building])
        self.site.ObjectPlacement = _placement(f, None, (0, 0, 0), 1.0)
        self.building.ObjectPlacement = _placement(f, self.site.ObjectPlacement, (0, 0, 0), 1.0)
        self.wall_counter = 0; self.counters = {}; self.types = {}

    # ---- naming styles
    def _name(self, kind: str, external: bool = False) -> str:
        self.counters[kind] = self.counters.get(kind, 0) + 1; n = self.counters[kind]
        if self.s.naming == "revit":
            fam = {"wall": ("Basic Wall:Exterior - Brick on CMU" if external else "Basic Wall:Interior - Partition (92mm Stud)"),
                   "door": "Single-Flush:0915 x 2134mm", "window": "Fixed:0915 x 1220mm", "space": None, "slab": "Floor:Generic 150mm", "opening": None, "stair": "Stair:Monolithic"}[kind]
            if kind == "space":
                return f"{100 + n}"
            if kind == "opening":
                return f"Opening {n}"
            return f"{fam}:{self.rng.randint(100000, 999999)}"
        de = {"wall": "Wand", "door": "Tür", "window": "Fenster", "space": "Raum", "slab": "Decke", "opening": "Öffnung", "stair": "Treppe"}[kind]
        return f"{de}-{n:03d}"

    # ---- one storey
    def add_storey(self, plan: Plan, index: int, elevation: float):
        f, s, sc = self.f, self.s, self.s.scale
        storey = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingStorey", name=f"Level {index + 1}" if s.naming == "revit" else f"{index}. OG" if index else "EG")
        storey.Elevation = float(elevation * sc)
        ifcopenshell.api.aggregate.assign_object(f, relating_object=self.building, products=[storey])
        storey.ObjectPlacement = _placement(f, self.building.ObjectPlacement, (0, 0, elevation), sc)
        H = plan.height
        walls = []
        if plan.rooms:      # exterior flag from geometry (review 2026-10-24: the bounding-box rule mislabelled facades of non-rectangular plans)
            from shapely.geometry import Polygon as _Poly
            from ifcgym.geomext import external_flags
            for w, e in zip(plan.walls, external_flags([(w.p0, w.p1, w.thickness) for w in plan.walls], [_Poly(r.polygon).buffer(0) for r in plan.rooms if not (r.long_name and any(k in r.long_name.lower() for k in ("outdoor", "balcony", "terrace", "garden", "yard")))])):   # outdoor rooms are outside
                w.external = bool(e)
        # walls: placement at p0, local x along the wall; axis polyline; body extrusion; corner extension so joints overlap
        ext = self._corner_extensions(plan)
        for i, w in enumerate(plan.walls):
            p0, p1 = np.array(w.p0), np.array(w.p1); u = (p1 - p0) / np.linalg.norm(p1 - p0)
            e0, e1 = ext.get((i, 0), 0.0), ext.get((i, 1), 0.0)
            q0, q1 = p0 - u * e0, p1 + u * e1; L = float(np.linalg.norm(q1 - q0))
            cls = "IfcWallStandardCase" if s.schema == "IFC2X3" else "IfcWall"
            wall = ifcopenshell.api.root.create_entity(f, ifc_class=cls, name=self._name("wall", w.external), predefined_type=None if s.schema == "IFC2X3" else "STANDARD")
            wall.ObjectPlacement = _placement(f, storey.ObjectPlacement, (q0[0], q0[1], 0.0), sc, xdir=(float(u[0]), float(u[1]), 0.0))
            axis = f.createIfcPolyline([_pt(f, (0.0, 0.0), sc), _pt(f, (L, 0.0), sc)])
            axis_rep = f.createIfcShapeRepresentation(self.axis, "Axis", "Curve2D", [axis])
            if s.profile == "rect":
                prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, (L / 2, 0.0), sc), None), float(L * sc), float(w.thickness * sc))
            else:
                t2 = w.thickness / 2
                pts = [(0.0, -t2), (L, -t2), (L, t2), (0.0, t2)]
                prof = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([_pt(f, p, sc) for p in pts] + [_pt(f, pts[0], sc)]))
            if s.body == "faceset" and s.schema != "IFC2X3":
                t2 = w.thickness / 2
                solid = _faceset_prism(f, [(0.0, -t2), (L, -t2), (L, t2), (0.0, t2)], 0.0, H, sc)
                body_rep = f.createIfcShapeRepresentation(self.body, "Body", "Tessellation", [solid])
            elif s.body in ("brep", "faceset"):
                t2 = w.thickness / 2
                solid = _brep_prism(f, [(0.0, -t2), (L, -t2), (L, t2), (0.0, t2)], 0.0, H, sc)
                body_rep = f.createIfcShapeRepresentation(self.body, "Body", "Brep", [solid])
            else:
                solid = f.createIfcExtrudedAreaSolid(prof, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(H * sc))
                if s.body == "clipped":
                    solid = _clipped(f, solid, H + 2.0, sc); body_rep = f.createIfcShapeRepresentation(self.body, "Body", "Clipping", [solid])
                else:
                    body_rep = f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])
            wall.Representation = f.createIfcProductDefinitionShape(None, None, [axis_rep, body_rep])
            ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[wall])
            ifcopenshell.api.pset.edit_pset(f, pset=ifcopenshell.api.pset.add_pset(f, product=wall, name="Pset_WallCommon"),
                                            properties={"IsExternal": bool(w.external), "LoadBearing": bool(w.external),
                                                        **({"FireRating": self.rng.choice(["REI 30", "REI 60", "REI 90"])} if self.rng.random() < 0.5 else {})})
            walls.append((wall, q0, u, L, w))
        # joints
        if not s.sparse:
            self._connect(plan, walls)
        # openings and fills
        if s.dense:
            self._densify(plan)
        for op in plan.doors:
            wall, q0, u, L, w = walls[op.wall]
            p0 = np.array(w.p0); along = float(np.dot(p0 - q0, u)) + op.t * float(np.linalg.norm(np.array(w.p1) - p0))
            self._opening(storey, wall, w, along, op, H)
        # spaces + boundaries
        for r in plan.rooms:
            self._space(storey, r, walls, H)
        # slab
        outline = plan.outline or self._outline(plan)
        slab = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSlab", name=self._name("slab"), predefined_type="FLOOR")
        slab.ObjectPlacement = _placement(f, storey.ObjectPlacement, (0, 0, -0.15), sc)
        prof = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([_pt(f, p, sc) for p in outline] + [_pt(f, outline[0], sc)]))
        solid = f.createIfcExtrudedAreaSolid(prof, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(0.15 * sc))
        slab.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])])
        ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[slab])
        # vertical core: slab openings for the shafts, a stair flight per storey, one elevator
        core = getattr(plan, "core", None)
        if core is not None and getattr(core, "slab_openings", True):
            # the FLOOR slab of this storey is voided over the shafts; for an open house stair the void is in the slab above the
            # stair, i.e. the floor slab of storeys 1..n-1 (never the ground floor), and slightly larger than the flight
            for name, rect in (("elevator", core.elevator), ("stair", core.stair)):
                if rect is None:
                    continue
                if getattr(core, "open_stair", False):
                    if index == 0:
                        continue
                    x0, y0, x1, y1 = rect; rect = (x0 - 0.2, y0 - 0.2, x1 + 0.2, y1 + 0.2)
                x0, y0, x1, y1 = rect
                op = ifcopenshell.api.root.create_entity(f, ifc_class="IfcOpeningElement", name=self._name("opening"))
                op.ObjectPlacement = _placement(f, slab.ObjectPlacement, (x0, y0, -0.05), sc)
                prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, ((x1 - x0) / 2, (y1 - y0) / 2), sc), None), float((x1 - x0) * sc), float((y1 - y0) * sc))
                solid = f.createIfcExtrudedAreaSolid(prof, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(0.25 * sc))
                op.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])])
                f.create_entity("IfcRelVoidsElement", GlobalId=ifcopenshell.guid.new(), OwnerHistory=ifcopenshell.api.owner.create_owner_history(f), RelatingBuildingElement=slab, RelatedOpeningElement=op)
            if core.stair is not None and index < s.storeys - 1:
                x0, y0, x1, y1 = core.stair; open_stair = getattr(core, "open_stair", False)
                inset = 0.0 if open_stair else 0.3
                along_y = (y1 - y0) >= (x1 - x0)
                L = ((y1 - y0) if along_y else (x1 - x0)) - 2 * inset; W = ((x1 - x0) if along_y else (y1 - y0)) - 2 * inset
                n_steps = max(8, int(round(s.storey_height / 0.18))); rise, run = s.storey_height / n_steps, L / n_steps
                pts = [(0.0, 0.0)]
                for k in range(n_steps):
                    pts += [(k * run, (k + 1) * rise), ((k + 1) * run, (k + 1) * rise)]
                pts += [(L, 0.0)]
                prof = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([_pt(f, q, sc) for q in pts] + [_pt(f, pts[0], sc)]))
                origin = (x0 + inset, y0 + inset, 0.0)
                xdir = (0.0, 1.0, 0.0) if along_y else (1.0, 0.0, 0.0)
                # flight solid: profile in the (run, rise) plane of a frame whose x = flight direction; extruded across the width
                def flight_solid():
                    place = f.createIfcAxis2Placement3D(_pt(f, (0.0, 0.0, 0.0), sc), _dir(f, (0.0, -1.0, 0.0)), _dir(f, (1.0, 0.0, 0.0)))
                    return f.createIfcExtrudedAreaSolid(prof, place, _dir(f, (0, 0, 1)), float(W * sc))
                if s.naming == "revit":
                    # Revit pattern (wbdg_office, fantasy_*): IfcStair aggregating an IfcStairFlight and railings
                    stair = ifcopenshell.api.root.create_entity(f, ifc_class="IfcStair", name=f"Stair:Monolithic:{self.rng.randint(100000, 999999)}", predefined_type="STRAIGHT_RUN_STAIR")
                    stair.ObjectPlacement = _placement(f, storey.ObjectPlacement, origin, sc, xdir=xdir)
                    flight = ifcopenshell.api.root.create_entity(f, ifc_class="IfcStairFlight", name=f"Monolithic Run:{self.rng.randint(100000, 999999)}", predefined_type="STRAIGHT")
                    flight.ObjectPlacement = _placement(f, stair.ObjectPlacement, (0.0, 0.0, 0.0), sc)
                    flight.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [flight_solid()])])
                    rails = []
                    for side in (0.0, W - 0.05):
                        rail = ifcopenshell.api.root.create_entity(f, ifc_class="IfcRailing", name=f"Railing:Handrail:{self.rng.randint(100000, 999999)}", predefined_type="HANDRAIL")
                        rail.ObjectPlacement = _placement(f, stair.ObjectPlacement, (0.0, side, 0.9), sc)
                        rp = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, (L / 2, 0.025), sc), None), float(L * sc), float(0.05 * sc))
                        rail.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [f.createIfcExtrudedAreaSolid(rp, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(0.05 * sc))])])
                        rails.append(rail)
                    if s.schema == "IFC2X3":
                        stair.ShapeType = "STRAIGHT_RUN_STAIR"
                    ifcopenshell.api.aggregate.assign_object(f, relating_object=stair, products=[flight] + rails)
                    ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[stair])
                else:
                    # ArchiCAD pattern (ac20, hitos): one IfcStair solid
                    stair = ifcopenshell.api.root.create_entity(f, ifc_class="IfcStair", name=self._name("stair"), predefined_type="STRAIGHT_RUN_STAIR")
                    stair.ObjectPlacement = _placement(f, storey.ObjectPlacement, origin, sc, xdir=xdir)
                    if s.schema == "IFC2X3":
                        stair.ShapeType = "STRAIGHT_RUN_STAIR"
                    stair.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [flight_solid()])])
                    ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[stair])
            if core.elevator is not None and index == 0:
                x0, y0, x1, y1 = core.elevator
                lift = ifcopenshell.api.root.create_entity(f, ifc_class="IfcTransportElement", name="Elevator" if s.naming == "revit" else "Aufzug-001", predefined_type="ELEVATOR")
                lift.ObjectPlacement = _placement(f, storey.ObjectPlacement, (x0 + 0.2, y0 + 0.2, 0.0), sc)
                prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, ((x1 - x0 - 0.4) / 2, (y1 - y0 - 0.4) / 2), sc), None), float((x1 - x0 - 0.4) * sc), float((y1 - y0 - 0.4) * sc))
                solid = f.createIfcExtrudedAreaSolid(prof, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(2.4 * sc))
                lift.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])])
                ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[lift])
        return storey

    def _corner_extensions(self, plan: Plan) -> dict:
        """Extend a wall end by the other wall's half thickness when its end meets the other wall's end (L corner) or side
        (T junction) so the solids overlap at the joint; T stems are extended to the host's centreline."""
        ext = {}; tol = 1e-6
        for i, w in enumerate(plan.walls):
            for end, p in ((0, np.array(w.p0)), (1, np.array(w.p1))):
                best = 0.0
                for j, o in enumerate(plan.walls):
                    if i == j:
                        continue
                    a, b = np.array(o.p0), np.array(o.p1); ab = b - a; L2 = float(ab @ ab)
                    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0)); d = float(np.linalg.norm(p - (a + t * ab)))
                    if d <= tol + 1e-3:
                        best = max(best, o.thickness / 2)
                if best:
                    ext[(i, end)] = best
        return ext

    def _connect(self, plan: Plan, walls):
        f = self.f; tol = 1e-3
        for i, (wi, q0i, ui, Li, w) in enumerate(walls):
            for j, (wj, q0j, uj, Lj, o) in enumerate(walls):
                if j <= i:
                    continue
                def end_on(p, a, b):
                    ab = b - a; L2 = float(ab @ ab); t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0)); return float(np.linalg.norm(p - (a + t * ab))) <= max(w.thickness, o.thickness) + tol, t
                pi0, pi1 = np.array(w.p0), np.array(w.p1); pj0, pj1 = np.array(o.p0), np.array(o.p1)
                ci = None; cj = None
                for name, p in (("ATSTART", pi0), ("ATEND", pi1)):
                    on, t = end_on(p, pj0, pj1)
                    if on:
                        ci = name; cj = "ATSTART" if t < 0.02 else "ATEND" if t > 0.98 else "ATPATH"
                if ci is None:
                    for name, p in (("ATSTART", pj0), ("ATEND", pj1)):
                        on, t = end_on(p, pi0, pi1)
                        if on:
                            cj = name; ci = "ATSTART" if t < 0.02 else "ATEND" if t > 0.98 else "ATPATH"
                if ci is None:
                    continue
                f.create_entity("IfcRelConnectsPathElements", GlobalId=ifcopenshell.guid.new(), OwnerHistory=ifcopenshell.api.owner.create_owner_history(f),
                                RelatingElement=wi, RelatedElement=wj, RelatingConnectionType=ci, RelatedConnectionType=cj, RelatingPriorities=[], RelatedPriorities=[])

    def _opening(self, storey, wall, w: WallSeg, along: float, op: Opening, H: float):
        f, s, sc = self.f, self.s, self.s.scale
        t = w.thickness; ob = s.opening_body
        if s.opening_mix and self.rng.random() < s.opening_mix:
            ob = self.rng.choice(OPENING_BODIES)
        if ob == "faceset" and s.schema == "IFC2X3":
            ob = "brep"
        opening = ifcopenshell.api.root.create_entity(f, ifc_class="IfcOpeningElement", name=self._name("opening"))
        if ob == "vertical":      # the original synthetic style: plan profile (width x thickness) extruded up by the height
            opening.ObjectPlacement = _placement(f, wall.ObjectPlacement, (along - op.width / 2, -t / 2 - 0.05, op.sill), sc)
            prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, (op.width / 2, (t + 0.1) / 2), sc), None), float(op.width * sc), float((t + 0.1) * sc))
            solid = f.createIfcExtrudedAreaSolid(prof, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(op.height * sc)); rtype = "SweptSolid"
            fill_origin = (0.0, 0.05, 0.0)
        else:                     # real exports: the cut through the wall; opening origin on the wall axis at the opening's left edge
            opening.ObjectPlacement = _placement(f, wall.ObjectPlacement, (along - op.width / 2, 0.0, op.sill), sc)
            d = t / 2 + 0.05
            if ob in ("brep", "faceset"):
                poly = [(0.0, -d), (op.width, -d), (op.width, d), (0.0, d)]
                solid = (_brep_prism if ob == "brep" else _faceset_prism)(f, poly, 0.0, op.height, sc); rtype = "Brep" if ob == "brep" else "Tessellation"
            else:
                # profile plane = the wall face: x along the wall, y up; extruded across the wall (local z = -wall y)
                pos = f.createIfcAxis2Placement3D(_pt(f, (0.0, d, 0.0), sc), _dir(f, (0.0, -1.0, 0.0)), _dir(f, (1.0, 0.0, 0.0)))
                if ob == "through":
                    prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, (op.width / 2, op.height / 2), sc), None), float(op.width * sc), float(op.height * sc))
                elif ob == "through_rot":     # Revit: the profile's x axis points up, so XDim is the height and YDim the width
                    prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, (op.width / 2, op.height / 2), sc), _dir(f, (0.0, 1.0))), float(op.height * sc), float(op.width * sc))
                else:
                    pts = [(0.0, 0.0), (op.width, 0.0), (op.width, op.height), (0.0, op.height)]
                    prof = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([_pt(f, p, sc) for p in pts] + [_pt(f, pts[0], sc)]))
                solid = f.createIfcExtrudedAreaSolid(prof, pos, _dir(f, (0, 0, 1)), float(2 * d * sc)); rtype = "SweptSolid"
            fill_origin = (0.0, -t / 2, 0.0)
        opening.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", rtype, [solid])])
        f.create_entity("IfcRelVoidsElement", GlobalId=ifcopenshell.guid.new(), OwnerHistory=ifcopenshell.api.owner.create_owner_history(f), RelatingBuildingElement=wall, RelatedOpeningElement=opening)
        cls = "IfcDoor" if op.kind == "door" else "IfcWindow"
        fill = ifcopenshell.api.root.create_entity(f, ifc_class=cls, name=self._name(op.kind))
        fill.OverallHeight = float(op.height * sc); fill.OverallWidth = float(op.width * sc)
        fill.ObjectPlacement = _placement(f, opening.ObjectPlacement, fill_origin, sc)
        fill.Representation = f.createIfcProductDefinitionShape(None, None, [self._fill_rep(op, t)])
        f.create_entity("IfcRelFillsElement", GlobalId=ifcopenshell.guid.new(), OwnerHistory=ifcopenshell.api.owner.create_owner_history(f), RelatingOpeningElement=opening, RelatedBuildingElement=fill)
        ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[fill])
        if s.fill_body != "extrusion":
            self._type_of(op, t)[1].RelatedObjects = self._type_of(op, t)[1].RelatedObjects + (fill,)

    def _leaf(self, op: Opening, t: float, body: str):
        """The door / window leaf in fill-local coordinates (origin at the opening's left edge on the wall's -y face): a slab of
        0.05 m on the wall axis, width x height."""
        f, sc = self.f, self.s.scale; leaf = 0.05
        if body == "faceset" and self.s.schema == "IFC2X3":
            body = "brep"
        if body in ("brep", "faceset"):
            poly = [(0.0, t / 2 - leaf / 2), (op.width, t / 2 - leaf / 2), (op.width, t / 2 + leaf / 2), (0.0, t / 2 + leaf / 2)]
            return (_brep_prism if body == "brep" else _faceset_prism)(f, poly, 0.0, op.height, sc), ("Brep" if body == "brep" else "Tessellation")
        prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(_pt(f, (op.width / 2, t / 2), sc), None), float(op.width * sc), float(leaf * sc))
        return f.createIfcExtrudedAreaSolid(prof, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(op.height * sc)), "SweptSolid"

    def _fill_rep(self, op: Opening, t: float):
        f = self.f
        if self.s.fill_body == "extrusion":
            item, rtype = self._leaf(op, t, "extrusion")
            return f.createIfcShapeRepresentation(self.body, "Body", rtype, [item])
        rmap = self._type_of(op, t)[2]
        target = f.createIfcCartesianTransformationOperator3D(None, None, f.createIfcCartesianPoint([0.0, 0.0, 0.0]), 1.0, None)
        return f.createIfcShapeRepresentation(self.body, "Body", "MappedRepresentation", [f.createIfcMappedItem(rmap, target)])

    def _type_of(self, op: Opening, t: float):
        """(type, IfcRelDefinesByType, IfcRepresentationMap) shared by all fills of one kind and size, as Revit / ArchiCAD export them."""
        key = (op.kind, round(op.width, 3), round(op.height, 3), round(t, 3))
        if key in self.types:
            return self.types[key]
        f, s = self.f, self.s
        item, rtype = self._leaf(op, t, s.fill_body.split("_", 1)[1])
        rmap = f.createIfcRepresentationMap(_a2p3(f, (0, 0, 0), s.scale), f.createIfcShapeRepresentation(self.body, "Body", rtype, [item]))
        name = f"{'Single-Flush' if op.kind == 'door' else 'Fixed'}:{round(op.width * 1000)} x {round(op.height * 1000)}mm" if s.naming == "revit" else f"{'Tür' if op.kind == 'door' else 'Fenster'} {round(op.width * 100)}x{round(op.height * 100)}"
        oh = ifcopenshell.api.owner.create_owner_history(f)
        if s.schema == "IFC2X3":
            if op.kind == "door":
                typ = f.create_entity("IfcDoorStyle", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, Name=name, RepresentationMaps=[rmap], OperationType="SINGLE_SWING_LEFT", ConstructionType="NOTDEFINED", ParameterTakesPrecedence=False, Sizeable=False)
            else:
                typ = f.create_entity("IfcWindowStyle", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, Name=name, RepresentationMaps=[rmap], ConstructionType="NOTDEFINED", OperationType="SINGLE_PANEL", ParameterTakesPrecedence=False, Sizeable=False)
        elif op.kind == "door":
            typ = f.create_entity("IfcDoorType", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, Name=name, RepresentationMaps=[rmap], PredefinedType="DOOR", OperationType="SINGLE_SWING_LEFT")
        else:
            typ = f.create_entity("IfcWindowType", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, Name=name, RepresentationMaps=[rmap], PredefinedType="WINDOW", PartitioningType="SINGLE_PANEL")
        rel = f.create_entity("IfcRelDefinesByType", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, RelatedObjects=[], RelatingType=typ)
        self.types[key] = (typ, rel, rmap)
        return self.types[key]

    def _densify(self, plan: Plan):
        """Extra windows on the external walls (real facades: 1-4 openings per wall, up to 12+), placed in the free stretches between
        the existing openings, the wall ends and the T junctions of other walls; once per plan (plans repeat over storeys)."""
        if getattr(plan, "_dense", False):
            return
        plan._dense = True
        rng = self.rng
        for i, w in enumerate(plan.walls):
            if not w.external:
                continue
            p0, p1 = np.array(w.p0, float), np.array(w.p1, float); L = float(np.linalg.norm(p1 - p0)); u = (p1 - p0) / L
            busy = [(0.0, 0.6), (L - 0.6, L)]
            for o in plan.doors:
                if o.wall == i:
                    c = o.t * L; busy.append((c - o.width / 2 - 0.4, c + o.width / 2 + 0.4))
            for j, ow in enumerate(plan.walls):
                if j == i:
                    continue
                for q in (np.array(ow.p0, float), np.array(ow.p1, float)):
                    a = float((q - p0) @ u)
                    if -0.01 <= a <= L + 0.01 and float(np.linalg.norm(q - (p0 + a * u))) <= w.thickness / 2 + ow.thickness / 2 + 0.01:
                        busy.append((a - ow.thickness / 2 - 0.4, a + ow.thickness / 2 + 0.4))
            busy.sort(); x = 0.0
            for lo, hi in busy + [(L, L)]:
                while lo - x >= 1.0:
                    width = rng.choice([0.6, 0.9, 1.2, 1.5]); height, sill = rng.choice([(1.2, 0.9), (1.5, 0.6), (1.0, 1.1)])
                    if width > lo - x - 0.2 or sill + height > plan.height - 0.2:
                        break
                    if rng.random() < 0.7:
                        plan.doors.append(Opening(i, (x + 0.1 + width / 2) / L, width, "window", height=height, sill=sill))
                    x += 0.1 + width + rng.choice([0.6, 0.9, 1.5])
                x = max(x, hi)

    def _space(self, storey, room: Room, walls, H: float):
        f, s, sc = self.f, self.s, self.s.scale
        poly = list(room.polygon)
        if _poly_area(poly) < 0:
            poly = poly[::-1]
        space = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSpace", name=room.name)
        if room.long_name:
            space.LongName = room.long_name
        if s.schema == "IFC2X3":
            space.InteriorOrExteriorSpace = "INTERNAL"
        space.ObjectPlacement = _placement(f, storey.ObjectPlacement, (0, 0, 0), sc)
        if s.body == "brep":      # ArchiCAD-style space bodies (ac20, hitos): faceted Brep instead of an extruded footprint
            space.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "Brep", [_brep_prism(f, poly, 0.0, H, sc)])])
        else:
            prof = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline([_pt(f, p, sc) for p in poly] + [_pt(f, poly[0], sc)]))
            solid = f.createIfcExtrudedAreaSolid(prof, _a2p3(f, (0, 0, 0), sc), _dir(f, (0, 0, 1)), float(H * sc))
            space.Representation = f.createIfcProductDefinitionShape(None, None, [f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])])
        ifcopenshell.api.aggregate.assign_object(f, relating_object=storey, products=[space])
        # boundaries: each polygon edge that lies on a wall face -> IfcRelSpaceBoundary with a surface of linear extrusion on the wall
        for k in (range(len(poly)) if not s.sparse else []):
            a, b = np.array(poly[k]), np.array(poly[(k + 1) % len(poly)])
            for wall, q0, u, L, w in walls:
                n = np.array([-u[1], u[0]]); t2 = w.thickness / 2
                da, db = float(np.dot(a - q0, n)), float(np.dot(b - q0, n))
                if abs(abs(da) - t2) > 0.02 or abs(abs(db) - t2) > 0.02:
                    continue
                ta, tb = float(np.dot(a - q0, u)), float(np.dot(b - q0, u))
                if max(ta, tb) < -0.02 or min(ta, tb) > L + 0.02:
                    continue
                lo, hi = max(min(ta, tb), 0.0), min(max(ta, tb), L)
                if hi - lo < 0.05:
                    continue
                pa, pb = q0 + u * lo + n * da, q0 + u * hi + n * da
                curve = f.createIfcPolyline([_pt(f, (0.0, 0.0), sc), _pt(f, (float(np.linalg.norm(pb - pa)), 0.0), sc)])
                prof2 = f.createIfcArbitraryOpenProfileDef("CURVE", None, curve)
                d = (pb - pa) / np.linalg.norm(pb - pa)
                surf = f.createIfcSurfaceOfLinearExtrusion(prof2, _a2p3(f, (float(pa[0]), float(pa[1]), 0.0), sc, xdir=(float(d[0]), float(d[1]), 0.0)), _dir(f, (0, 0, 1)), float(H * sc))
                geom = f.createIfcConnectionSurfaceGeometry(surf, None)
                f.create_entity("IfcRelSpaceBoundary", GlobalId=ifcopenshell.guid.new(), OwnerHistory=ifcopenshell.api.owner.create_owner_history(f),
                                Name="2ndLevel", Description="2a", RelatingSpace=space, RelatedBuildingElement=wall, ConnectionGeometry=geom,
                                PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="EXTERNAL" if w.external else "INTERNAL")

    def _outline(self, plan: Plan):
        xs = [p[0] for w in plan.walls for p in (w.p0, w.p1)]; ys = [p[1] for w in plan.walls for p in (w.p0, w.p1)]
        m = max(w.thickness for w in plan.walls) / 2
        return [(min(xs) - m, min(ys) - m), (max(xs) + m, min(ys) - m), (max(xs) + m, max(ys) + m), (min(xs) - m, max(ys) + m)]

    def build(self, plans: list[Plan]) -> ifcopenshell.file:
        z = 0.0
        for i in range(self.s.storeys):
            plan = plans[i % len(plans)]
            self.add_storey(plan, i, z); z += self.s.storey_height
        return self.f


# ------------------------------------------------------------------------------------------ demo plan: two rooms with a door
def demo_plan(w=10.0, d=6.0, split=4.5, t_ext=0.3, t_int=0.12, height=2.8) -> Plan:
    walls = [WallSeg((0, 0), (w, 0), t_ext, True), WallSeg((w, 0), (w, d), t_ext, True), WallSeg((w, d), (0, d), t_ext, True), WallSeg((0, d), (0, 0), t_ext, True),
             WallSeg((split, 0), (split, d), t_int, False)]
    doors = [Opening(4, 0.5, 0.9, "door"), Opening(0, 0.25, 1.2, "window", height=1.2, sill=0.9), Opening(2, 0.3, 1.2, "window", height=1.2, sill=0.9), Opening(0, 0.75, 0.9, "door")]
    e, i2 = t_ext / 2, t_int / 2
    rooms = [Room("101", [(e, e), (split - i2, e), (split - i2, d - e), (e, d - e)], "Office"),
             Room("102", [(split + i2, e), (w - e, e), (w - e, d - e), (split + i2, d - e)], "Meeting")]
    return Plan(walls, doors, rooms, height)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--units", default="m"); ap.add_argument("--schema", default="IFC4")
    ap.add_argument("--profile", default="rect"); ap.add_argument("--naming", default="revit"); ap.add_argument("--storeys", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    st = Style(units=a.units, schema=a.schema, profile=a.profile, naming=a.naming, storeys=a.storeys, seed=a.seed)
    f = Builder(st, name="demo").build([demo_plan()])
    f.write(a.out)
    print(f"wrote {a.out}: walls={len(f.by_type('IfcWall'))} doors={len(f.by_type('IfcDoor'))} windows={len(f.by_type('IfcWindow'))} spaces={len(f.by_type('IfcSpace'))} boundaries={len(f.by_type('IfcRelSpaceBoundary'))} joints={len(f.by_type('IfcRelConnectsPathElements'))}")


if __name__ == "__main__":
    main()
