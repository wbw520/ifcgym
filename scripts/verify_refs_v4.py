"""Independent check of generator-v4 references and relational answers: every task whose target is described topologically (-T),
spatially (-S) or along a path (-P) is re-resolved here from MESHES (ifcopenshell.geom, metres) and raw inverse relationships -
not from the axis / placement / profile code the generator uses - and compared with the generated target. Prints mismatches and
counts per template.    python scripts/verify_refs_v4.py <tasks.jsonl> [--limit N]"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import ifcopenshell, ifcopenshell.util.element as ue
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from ifcgym.checkers import mesh, _tri_mesh

args = [a for a in sys.argv[1:] if not a.startswith("--")]; lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
tasks = [json.loads(l) for l in open(args[0]) if l.strip()]
DIRV = {"north": np.array([0, 1.0]), "south": np.array([0, -1.0]), "east": np.array([1.0, 0]), "west": np.array([-1.0, 0])}


class M:
    def __init__(self, f):
        self.f = f; self._m = {}; self._fp = {}
    def m(self, e):
        if e.GlobalId not in self._m:
            self._m[e.GlobalId] = mesh(e)
        return self._m[e.GlobalId]
    def c(self, e):
        v = self.m(e); return None if v is None else v.mean(0)
    def bb(self, e):
        v = self.m(e); return None if v is None else (v.min(0), v.max(0))
    def wdir(self, w):
        """Unit direction of a wall from its mesh: principal axis of the xy vertices."""
        v = self.m(w)[:, :2]; v = v - v.mean(0); u, s, vt = np.linalg.svd(v, full_matrices=False); d = vt[0]
        return d / (np.linalg.norm(d) or 1.0)
    def footprint(self, sp):
        """Space footprint polygon from the mesh triangles lying on the bottom face."""
        if sp.GlobalId not in self._fp:
            r = _tri_mesh(sp)
            if r is None:
                self._fp[sp.GlobalId] = None
            else:
                v, fc = r; z0 = v[:, 2].min(); tris = [Polygon(v[t][:, :2]) for t in fc if np.all(np.abs(v[t][:, 2] - z0) < 1e-3)]
                self._fp[sp.GlobalId] = unary_union(tris).buffer(1e-4) if tris else None
        return self._fp[sp.GlobalId]
    def storey(self, e):
        c = ue.get_container(e)
        if c is not None and c.is_a("IfcBuildingStorey"):
            return c
        a = ue.get_aggregate(e); return a if a is not None and a.is_a("IfcBuildingStorey") else None
    def conn(self, w):
        out = set()
        for r in list(w.ConnectedTo or []) + list(w.ConnectedFrom or []):
            o = r.RelatedElement if r.RelatingElement == w else r.RelatingElement
            if o is not None and o.is_a("IfcWall"):
                out.add(o.GlobalId)
        return out
    def bspaces(self, w):
        return {r.RelatingSpace.GlobalId for r in (w.ProvidesBoundaries or []) if r.RelatingSpace is not None}
    def swalls(self, sp):
        return {r.RelatedBuildingElement.GlobalId for r in (sp.BoundedBy or []) if r.RelatedBuildingElement is not None and r.RelatedBuildingElement.is_a("IfcWall")}
    def fills(self, w, cls=None):
        out = []
        for r in w.HasOpenings or []:
            for q in r.RelatedOpeningElement.HasFillings or []:
                x = q.RelatedBuildingElement
                if cls is None or x.is_a(cls):
                    out.append(x)
        return out
    def host(self, d):
        return d.FillsVoids[0].RelatingOpeningElement.VoidsElements[0].RelatingBuildingElement
    def axis_pts(self, w):
        """World end points (metres) of the wall's 'Axis' polyline, straight from the IFC data."""
        import ifcopenshell.util.placement as up, ifcopenshell.util.unit as uu
        sc = uu.calculate_unit_scale(self.f)
        for r in w.Representation.Representations:
            if r.RepresentationIdentifier == "Axis":
                for it in r.Items:
                    if it.is_a("IfcPolyline"):
                        M = up.get_local_placement(w.ObjectPlacement); pts = []
                        for pt in (it.Points[0], it.Points[-1]):
                            c = list(pt.Coordinates) + [0.0] * (3 - len(pt.Coordinates)); pts.append((M @ np.array(c + [1.0]))[:2] * sc)
                        return pts
        return None
    def ref_space(self, p, tpl):
        """The space a prompt describes (D: 'space with GlobalId', T: 'bounded by exactly', S: 'contains the point')."""
        m = re.search(r"the space with GlobalId ([0-9A-Za-z_$]{22})", p)
        if m:
            return {m.group(1)}
        return None
    def geo_exterior(self, w):
        """Mesh-based exterior test (independent of ifcgym.geomext): at 25/50/75 % along the wall, exactly one of the probe points
        0.25 m beyond its faces lies inside the storey's union of space footprints (closed by 0.3 m)."""
        st = self.storey(w); key = ("outline", st.GlobalId if st is not None else None)
        if key not in self._fp:
            fps = [self.footprint(s_) for s_ in self.f.by_type("IfcSpace") if self.storey(s_) == st]
            fps = [x for x in fps if x is not None]
            if fps:   # wall footprints close areas without a space that walls enclose (then filled as holes)
                fps += [x for x in (self.footprint(w_) for w_ in self.f.by_type("IfcWall") if self.m(w_) is not None and self.storey(w_) == st) if x is not None]
            U_ = unary_union(fps).buffer(0.3, join_style=2).buffer(-0.3, join_style=2) if fps else None
            self._fp[key] = unary_union([Polygon(P.exterior) for P in getattr(U_, "geoms", [U_]) if P.geom_type == "Polygon"]) if U_ is not None else None   # enclosed holes = inside
        U = self._fp[key]
        if U is None:
            return False
        u = self.wdir(w); n = np.array([-u[1], u[0]]); v = self.m(w)[:, :2]; pu = v @ u; pn = v @ n
        half = float(pn.max() - pn.min()) / 2; mid = float(pn.max() + pn.min()) / 2; votes = 0
        for s_ in (0.25, 0.5, 0.75):
            q = u * (pu.min() + s_ * (pu.max() - pu.min())) + n * mid
            votes += U.contains(Point(q + n * (half + 0.25))) != U.contains(Point(q - n * (half + 0.25)))
        return votes >= 2

    def door_sides(self, d):
        """Spaces on either side of the door's wall at the door position, by probing: a point 0.15 m beyond each face of the wall
        (wall half-thickness from its mesh) at the door centre; the bounded space whose footprint contains it is that side's
        space (review 2026-10-24: centroids of L-shaped rooms can lie outside the room, on the wrong side of the wall)."""
        w = self.host(d); u = self.wdir(w); n = np.array([-u[1], u[0]]); cd = self.c(d)[:2]
        vw = self.m(w)[:, :2]; pn = vw @ n; half = float(pn.max() - pn.min()) / 2; mid = float((pn.max() + pn.min()) / 2)
        base = cd + n * (mid - float(cd @ n)); res = {}
        for side in (1, -1):
            for gap in (0.15, 0.3, 0.5):       # exporters may leave a gap between the space boundary and the wall face
                pt = Point(base + side * n * (half + gap))
                hit = [sg for sg in self.bspaces(w) if (fp := self.footprint(self.f.by_guid(sg))) is not None and fp.contains(pt)]
                if hit:
                    break
            if len(hit) != 1:
                return None
            res[side] = hit[0]
        return (res[-1], res[1])


def flat_slabs_of(f, mm):
    """Flat floor slabs, independent of the generator's profile test: not ROOF / LANDING, mesh vertices on two levels only (bottom / top)."""
    out = []
    for x in f.by_type("IfcSlab"):
        if (x.PredefinedType or "NOTDEFINED") in ("ROOF", "LANDING"):
            continue
        v = mm.m(x)
        if v is None:
            continue
        z = v[:, 2]; z0, z1 = float(z.min()), float(z.max())
        if z1 - z0 < 0.6 and np.all((np.abs(z - z0) < 0.01) | (np.abs(z - z1) < 0.01)):
            out.append(x)
    return out


def resolve(mm, t):
    """Return the set of GlobalIds the prompt's description denotes (independent method), or None if not handled."""
    f = mm.f; tpl = t["template"]; p = t.get("prompt_template", t["prompt"]); meta = t.get("meta", {}); style = meta.get("style"); anchors = meta.get("anchors", [])   # paraphrased lists keep the template wording in prompt_template
    g = lambda x: f.by_guid(x)
    walls = [w for w in f.by_type("IfcWall") if mm.m(w) is not None]
    if tpl.startswith("U-merge-space-T"):      # E011 cover4: the extended room = the first wall list ("bounded by the walls ... so that")
        ids = set(re.findall(r"[0-9A-Za-z_$]{22}", p.split("room that is bounded by the walls")[1].split(" so that")[0]))
        return {sp.GlobalId for sp in f.by_type("IfcSpace") if ids <= mm.swalls(sp)}
    if tpl.startswith("U-storey-length"):      # E011 cover4: the storey's floor slab = the IfcSlab(s) contained in it
        st = g(re.search(r"building storey with ID ([0-9A-Za-z_$]{22})", p).group(1))
        return {sl.GlobalId for sl in f.by_type("IfcSlab") if mm.storey(sl) == st}
    m = re.search(r"slab that is directly above the walls of the storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p)
    if m:      # the lowest slab whose underside is within 0.3 m above the storey's highest wall top, overlapping the walls by > 1 m in X and Y
        st = g(m.group(1)); ws = [w for w in walls if mm.storey(w) == st]
        if not ws:
            return set()
        ztop = max(float(mm.bb(w)[1][2]) for w in ws); lo = np.min([mm.bb(w)[0][:2] for w in ws], axis=0); hi = np.max([mm.bb(w)[1][:2] for w in ws], axis=0)
        zs = sorted((float(mm.bb(sl)[0][2]), sl.GlobalId) for sl in f.by_type("IfcSlab") if mm.m(sl) is not None and ztop - 0.02 <= float(mm.bb(sl)[0][2]) <= ztop + 0.3
                    and min(mm.bb(sl)[1][0], hi[0]) - max(mm.bb(sl)[0][0], lo[0]) > 1.0 and min(mm.bb(sl)[1][1], hi[1]) - max(mm.bb(sl)[0][1], lo[1]) > 1.0)
        return {zs[0][1]} if zs and (len(zs) == 1 or zs[1][0] - zs[0][0] >= 0.1) else set()
    m = re.search(r"windows that are in the (north|south|east|west) walls", p)
    if m:      # exterior walls (mesh probes against the storey outline) whose outside faces the direction; their windows
        d = DIRV[m.group(1)]; ms = re.search(r"walls of the storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p); st_only = g(ms.group(1)) if ms else None; out = set()
        for w in walls:
            if (st_only is not None and mm.storey(w) != st_only) or not mm.geo_exterior(w):
                continue
            st = mm.storey(w); U = mm._fp[("outline", st.GlobalId if st is not None else None)]
            u = mm.wdir(w); n = np.array([-u[1], u[0]])
            if abs(float(n @ d)) < 0.9:
                continue
            v = mm.m(w)[:, :2]; pn = v @ n; half = float(pn.max() - pn.min()) / 2; q = mm.c(w)[:2] + n * (float((pn.max() + pn.min()) / 2) - float(mm.c(w)[:2] @ n))
            ins = U.contains(Point(q + n * (half + 0.25)))
            outward = -n if ins else n
            if float(outward @ d) > 0.9:
                out |= {x.GlobalId for x in mm.fills(w, "IfcWindow")}
        return out
    m = re.search(r"window that is below the window with ID ([0-9A-Za-z_$]{22})", p)
    if m:      # the wall right below the window's wall (same plan box, top at its base), its window nearest along the wall
        X = g(m.group(1)); wu = mm.host(X); bo = mm.bb(wu); best = []
        for w in walls:
            b = mm.bb(w)
            if w.GlobalId != wu.GlobalId and np.linalg.norm(b[0][:2] - bo[0][:2]) < 0.1 and np.linalg.norm(b[1][:2] - bo[1][:2]) < 0.1 and b[1][2] <= bo[0][2] + 0.3:
                best.append((-float(b[1][2]), w))
        if not best:
            return set()
        wl = sorted(best, key=lambda r_: r_[0])[0][1]; u = mm.wdir(wl); cx = float(mm.c(X)[:2] @ u)
        ds = sorted((abs(float(mm.c(y)[:2] @ u) - cx), y.GlobalId) for y in mm.fills(wl, "IfcWindow"))
        return {ds[0][1]} if ds and (len(ds) == 1 or ds[1][0] - ds[0][0] >= 0.5) else set()
    if tpl.startswith("R-adjacent-spaces"):   # answer = neighbours of the DESCRIBED space: resolve the reference first, then compute
        ref = resolve(mm, {**t, "template": "X-ref", "meta": {**meta}}) if style in ("T", "S") else {t["target"][0]}
        if not ref or len(ref) != 1:
            return set()
        sp = g(next(iter(ref))); return {o for wg in mm.swalls(sp) for o in mm.bspaces(g(wg))} - {sp.GlobalId}
    if "connected to both" in p:
        A, B = anchors; return {w.GlobalId for w in walls if A in mm.conn(w) and B in mm.conn(w)}
    if "separates the space" in p or "separates the room" in p or tpl.startswith("R-wall-separating"):
        A, B = anchors if anchors else re.findall(r"GlobalId ([0-9A-Za-z_$]{22})", p)[:2]
        return {w.GlobalId for w in walls if {A, B} <= mm.bspaces(w)}
    m = re.search(r"wall (?:lies directly |directly |that lies directly )?(north|south|east|west) \(\+?-?[XY](?: direction)?\) of the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        d = DIRV[m.group(1)]; O = g(m.group(2)); u = mm.wdir(O); co = mm.c(O); st = mm.storey(O); vo = mm.m(O)[:, :2] @ u; lo, hi = vo.min(), vo.max()
        best = []
        for w in walls:
            if w.GlobalId == O.GlobalId or mm.storey(w) != st or abs(float(mm.wdir(w) @ u)) < 0.95:
                continue
            vw = mm.m(w)[:, :2] @ u; ov = min(hi, vw.max()) - max(lo, vw.min()); off = float((mm.c(w)[:2] - co[:2]) @ d)
            if ov > -0.3 and off > 0.05:
                best.append((off, w.GlobalId, ov > 0.5 * (hi - lo) and ov > 0.5 * (vw.max() - vw.min())))
        best.sort()
        if not best or not best[0][2]:       # nearest overlapping wall must be the half-overlap one, else the description is arguable
            return set()
        return {best[0][1]}
    m = re.search(r"wall directly above the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        O = g(m.group(1)); bo = mm.bb(O); best = []
        for w in walls:
            b = mm.bb(w)
            if w.GlobalId == O.GlobalId or np.linalg.norm(b[0][:2] - bo[0][:2]) > 0.1 or np.linalg.norm(b[1][:2] - bo[1][:2]) > 0.1 or b[0][2] < bo[1][2] - 0.3:
                continue
            best.append((b[0][2], w.GlobalId))
        best.sort(); return {best[0][1]} if best else set()
    m = re.search(r"the wall facing you when you look through the window with GlobalId ([0-9A-Za-z_$]{22}) into the building", p)
    if m:
        win = g(m.group(1)); host = mm.host(win); c = mm.c(win)[:2]; u = mm.wdir(host); n = np.array([-u[1], u[0]]); st = mm.storey(win)
        ins = [s_ for s_ in mm.bspaces(host) if mm.footprint(g(s_)) is not None]
        if not ins:
            return {"NO-INSIDE-SPACE"}
        sides = {float((np.array(mm.footprint(g(s_)).representative_point().coords[0]) - c) @ n) > 0 for s_ in ins}
        if len(sides) != 1:
            return {"SPACES-ON-BOTH-SIDES"}
        e = n if sides.pop() else -n
        from shapely.geometry import LineString as _LS
        ray = _LS([c, c + e * 60.0]); hits = []
        for w in walls:
            if w.GlobalId == host.GlobalId or mm.storey(w) != st:
                continue
            fp_ = mm.footprint(w)
            if fp_ is None:
                continue
            inter = ray.intersection(fp_)
            if not inter.is_empty:
                pt = inter if inter.geom_type == "Point" else inter.representative_point() if inter.geom_type != "MultiLineString" and inter.geom_type != "LineString" else Point(list(inter.coords)[0]) if inter.geom_type == "LineString" else Point(list(inter.geoms[0].coords)[0])
                hits.append((float(ray.project(pt)), w.GlobalId))
        hits.sort(); return {hits[0][1]} if hits and (len(hits) == 1 or hits[1][0] - hits[0][0] > 0.3) else set()
    m = re.search(r"after leaving the space with GlobalId ([0-9A-Za-z_$]{22}) through the door with GlobalId ([0-9A-Za-z_$]{22}) to the outside", p)
    if m:
        A, D = m.group(1), m.group(2); d = g(D); w = mm.host(d); c = mm.c(d)[:2]; u = mm.wdir(w); n = np.array([-u[1], u[0]])
        pn = mm.m(w)[:, :2] @ n; half = float(pn.max() - pn.min()) / 2; base = c + n * (float((pn.max() + pn.min()) / 2) - float(c @ n)); fpA = mm.footprint(g(A))
        inA = {sd: fpA.contains(Point(base + sd * n * (half + 0.3))) for sd in (1, -1)}
        if inA[1] == inA[-1]:
            return {"DOOR-NOT-ON-A"}
        e = -n if inA[1] else n; r = np.array([e[1], -e[0]]); side = r if "to your right" in p else -r
        m2 = re.search(r"the (door|window) farthest to your", p); cls = "IfcDoor" if m2.group(1) == "door" else "IfcWindow"; st = mm.storey(d)
        def external(wall):
            try:
                v = (ue.get_psets(wall).get("Pset_WallCommon") or {}).get("IsExternal")
            except Exception:  # noqa: BLE001
                v = None
            return bool(v) if v is not None else mm.geo_exterior(wall)
        pool = [x for x in f.by_type(cls) if x.GlobalId != d.GlobalId and mm.m(x) is not None and mm.storey(x) == st and external(mm.host(x)) and np.linalg.norm(mm.c(x)[:2] - c) <= 8.0]
        ds = sorted((float((mm.c(x)[:2] - c) @ side), x.GlobalId) for x in pool)
        return {ds[-1][1]} if ds and ds[-1][0] > 0.3 and (len(ds) == 1 or ds[-1][0] - ds[-2][0] > 0.3) else set()
    if "after entering the space" in p:
        A, B, D = anchors; d = g(D); w = mm.host(d); cd = mm.c(d)[:2]; cb = np.array(mm.footprint(g(B)).centroid.coords[0]); u = mm.wdir(w); n = np.array([-u[1], u[0]])
        if mm.door_sides(d) is None or set(mm.door_sides(d)) != {A, B}:
            return {"ENTRY-NOT-BETWEEN-A-AND-B"}
        e = n if float((cb - cd) @ n) > 0 else -n; r = np.array([e[1], -e[0]])
        if "to your right" in p or "on your right-hand side" in p:
            side = r
        elif "to your left" in p or "on your left-hand side" in p:
            side = -r
        else:
            side = None
        if tpl.split("-")[1] in ("move", "set", "shorten", "rotate", "delete", "create", "wall") and ("the wall on your" in p or "the wall facing you" in p):
            ws = [g(x) for x in mm.swalls(g(B))]
            if "facing you" in p:
                return {x.GlobalId for x in ws if abs(float(mm.wdir(x) @ e)) < 0.1 and float((mm.c(x)[:2] - cd) @ e) > 0.3}
            if side is None:
                return {"NO-SIDE-WORD"}
            return {x.GlobalId for x in ws if abs(float(mm.wdir(x) @ e)) > 0.9 and float((mm.c(x)[:2] - cd) @ side) > 0.3}
        m2 = re.search(r"the (door|window|column) farthest to your", p); cls = {"door": "IfcDoor", "window": "IfcWindow", "column": "IfcColumn"}[m2.group(1)]
        if cls == "IfcColumn":
            fp = mm.footprint(g(B)); st = mm.storey(g(B)); pool = [x for x in f.by_type("IfcColumn") if mm.storey(x) == st and fp.contains(Point(mm.c(x)[:2]))]
        else:
            fpB = mm.footprint(g(B)).buffer(0.3); pool = [x for wg in mm.swalls(g(B)) for x in mm.fills(g(wg), cls) if x.GlobalId != d.GlobalId and mm.m(x) is not None and fpB.contains(Point(mm.c(x)[:2]))]
        if side is None:
            return {"NO-SIDE-WORD"}
        ds = sorted((float((mm.c(x)[:2] - cd) @ side), x.GlobalId) for x in pool)
        return {ds[-1][1]} if ds and ds[-1][0] > 0.3 and (len(ds) == 1 or ds[-1][0] - ds[-2][0] > 0.3) else set()
    m = re.search(r"the (door|window) directly (above|below) the (door|window) with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        X = g(m.group(4)); cls = "IfcDoor" if m.group(1) == "door" else "IfcWindow"; cx = mm.c(X)[:2]; zx = float(mm.bb(X)[0][2]); up = m.group(2) == "above"
        col = sorted((float(mm.bb(x)[0][2]), x.GlobalId) for x in f.by_type(cls) if mm.m(x) is not None and np.linalg.norm(mm.c(x)[:2] - cx) <= 0.05)
        i = [k_ for k_, (z_, g_) in enumerate(col) if g_ == X.GlobalId][0]
        j = i + 1 if up else i - 1
        return {col[j][1]} if 0 <= j < len(col) and 0.5 < abs(col[j][0] - zx) < 6.0 else set()
    m = re.search(r"wall directly below the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        O = g(m.group(1)); bo = mm.bb(O); best = []
        for w in walls:
            b = mm.bb(w)
            if w.GlobalId == O.GlobalId or np.linalg.norm(b[0][:2] - bo[0][:2]) > 0.1 or np.linalg.norm(b[1][:2] - bo[1][:2]) > 0.1 or b[1][2] > bo[0][2] + 0.3:
                continue
            best.append((-b[0][2], w.GlobalId))
        best.sort(); return {best[0][1]} if best else set()
    m = re.search(r"the (door|window) hosted by the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        return {x.GlobalId for x in mm.fills(g(m.group(2)), "IfcDoor" if m.group(1) == "door" else "IfcWindow")}
    m = re.search(r"the door connecting the space with GlobalId ([0-9A-Za-z_$]{22}) and the space with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        return {x.GlobalId for w in walls if {m.group(1), m.group(2)} <= mm.bspaces(w) for x in mm.fills(w, "IfcDoor") if mm.m(x) is not None and mm.door_sides(x) is not None and set(mm.door_sides(x)) == {m.group(1), m.group(2)}}
    m = re.search(r"the (door|window) closest to the corner where the wall with GlobalId ([0-9A-Za-z_$]{22}) meets the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        W, V = g(m.group(2)), g(m.group(3)); cls = "IfcDoor" if m.group(1) == "door" else "IfcWindow"
        uw, uv = mm.wdir(W), mm.wdir(V); cw, cv = mm.c(W)[:2], mm.c(V)[:2]   # corner = intersection of the two centre lines
        A = np.array([uw, -uv]).T; s_ = np.linalg.lstsq(A, cv - cw, rcond=None)[0]; corner = cw + uw * s_[0]
        stW = mm.storey(W)   # 'among all doors/windows on that storey' (review 2026-10-24)
        ds = sorted((float(np.linalg.norm(mm.c(x)[:2] - corner)), x.GlobalId) for x in f.by_type(cls) if mm.m(x) is not None and mm.storey(x) == stW)
        return {ds[0][1]} if ds and (len(ds) == 1 or ds[1][0] - ds[0][0] > 0.3) else set()
    m = re.search(r"the (westernmost|easternmost|northernmost|southernmost) (door|window) on the storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p)
    if m:
        d = {"westernmost": DIRV["west"], "easternmost": DIRV["east"], "northernmost": DIRV["north"], "southernmost": DIRV["south"]}[m.group(1)]; cls = "IfcDoor" if m.group(2) == "door" else "IfcWindow"; st = g(m.group(3))
        ds = sorted((-float(mm.c(x)[:2] @ d), x.GlobalId) for x in f.by_type(cls) if mm.storey(x) == st and mm.m(x) is not None)
        return {ds[0][1]} if ds and (len(ds) == 1 or ds[1][0] - ds[0][0] > 0.3) else set()
    m = re.search(r"the most central (door|window) of the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        W = g(m.group(2)); ds = sorted((float(np.linalg.norm(mm.c(x)[:2] - mm.c(W)[:2])), x.GlobalId) for x in mm.fills(W, "IfcDoor" if m.group(1) == "door" else "IfcWindow"))
        return {ds[0][1]} if ds and (len(ds) == 1 or ds[1][0] - ds[0][0] > 0.3) else set()
    m = re.search(r"the space bounded by the walls with GlobalIds (.+?) on |the space bounded by the walls with GlobalIds (.+?)(?: \(|,| \-|\.)", p)
    if "space bounded by exactly the walls" in p and not tpl.startswith("C-") and not tpl.startswith("R-adjacent"):
        ids = set(re.findall(r"[0-9A-Za-z_$]{22}", p.split("bounded by exactly the walls with GlobalIds")[1].split(" on the storey")[0].split(" (an IfcSpace)")[0].split(" (the IfcSpace)")[0].split(", keeping")[0].split(" (footprint")[0]))
        ids = {x for x in ids if any(x == w.GlobalId for w in walls)}
        return {sp.GlobalId for sp in f.by_type("IfcSpace") if mm.swalls(sp) == ids}
    m = re.search(r"contains the point \(([-\d.]+), ([-\d.]+)\)", p)
    if m and not tpl.startswith("C-") and not tpl.startswith("R-adjacent"):
        st = g(re.search(r"storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p).group(1)); pt = Point(float(m.group(1)), float(m.group(2)))
        return {sp.GlobalId for sp in f.by_type("IfcSpace") if mm.storey(sp) == st and (fp := mm.footprint(sp)) is not None and fp.contains(pt)}
    def flat_slabs():
        return flat_slabs_of(f, mm)
    def _unused():
        out = []
        for x in f.by_type("IfcSlab"):
            if (x.PredefinedType or "NOTDEFINED") in ("ROOF", "LANDING"):
                continue
            v = mm.m(x)
            if v is None:
                continue
            z = v[:, 2]; z0, z1 = float(z.min()), float(z.max())
            if z1 - z0 < 0.6 and np.all((np.abs(z - z0) < 0.01) | (np.abs(z - z1) < 0.01)):
                out.append(x)
        return out
    m = re.search(r"the floor slab \(IfcSlab, not a roof or landing\) contained in the storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p)
    if m:
        st = g(m.group(1)); return {x.GlobalId for x in flat_slabs() if mm.storey(x) == st}
    m = re.search(r"the (lowest|highest) floor slab of the building", p)
    if m:
        zs = sorted((float(mm.m(x)[:, 2].min()), x.GlobalId) for x in flat_slabs())
        if not zs:
            return set()
        if m.group(1) == "lowest":
            return {zs[0][1]} if len(zs) == 1 or zs[1][0] - zs[0][0] > 0.3 else set()
        return {zs[-1][1]} if len(zs) == 1 or zs[-1][0] - zs[-2][0] > 0.3 else set()
    m = re.search(r"the column standing inside the space with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        sp = g(m.group(1)); fp = mm.footprint(sp); st = mm.storey(sp)
        return {x.GlobalId for x in f.by_type("IfcColumn") if mm.storey(x) == st and fp.contains(Point(mm.c(x)[:2]))}
    m = re.search(r"the column on the storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\) closest to the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
    if m:
        W = g(m.group(2)); st = g(m.group(1)); ax = mm.axis_pts(W)
        def seg_d(p_):
            a_, b_ = ax; ab = b_ - a_; tt = np.clip(np.dot(p_ - a_, ab) / (np.dot(ab, ab) or 1e-12), 0, 1); return float(np.linalg.norm(p_ - (a_ + tt * ab)))
        ds = sorted((seg_d(mm.c(x)[:2]), x.GlobalId) for x in f.by_type("IfcColumn") if mm.storey(x) == st and mm.m(x) is not None)
        return {ds[0][1]} if ds and (len(ds) == 1 or ds[1][0] - ds[0][0] > 0.3) else set()
    m = re.search(r"the floor slab of the storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p)
    if m:
        return {x.GlobalId for x in f.by_type("IfcSlab") if mm.storey(x) == g(m.group(1))}
    if "the lowest slab of the building" in p or "the highest slab of the building" in p:
        zs = sorted((float(mm.bb(x)[0][2]), x.GlobalId) for x in f.by_type("IfcSlab") if mm.m(x) is not None)
        return {zs[0][1]} if "lowest" in p else {zs[-1][1]}
    if tpl.startswith("R-connected-walls"):
        return mm.conn(g(anchors[0]))
    if tpl.startswith("A-corridor") or tpl.startswith("A-hub"):
        st = g(re.search(r"storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p).group(1)); sp = [s for s in f.by_type("IfcSpace") if mm.storey(s) == st]
        if tpl.startswith("A-corridor"):
            sc_ = {s.GlobalId: sum(len(mm.fills(g(w), "IfcDoor")) for w in mm.swalls(s)) for s in sp}
        else:
            sc_ = {s.GlobalId: len({o for w in mm.swalls(s) for o in mm.bspaces(g(w))} - {s.GlobalId}) for s in sp}
        top = sorted(sc_.items(), key=lambda kv: -kv[1]); return {top[0][0]} if top[0][1] - top[1][1] >= 1 else set()
    if tpl.startswith(("R-shortest-wall", "R-longest-wall")):
        st = g(re.search(r"storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p).group(1))
        ls = sorted((float(np.linalg.norm(mm.axis_pts(w)[1] - mm.axis_pts(w)[0])), w.GlobalId) for w in f.by_type("IfcWall") if mm.storey(w) == st and mm.axis_pts(w) is not None)
        pick, nxt = (ls[0], ls[1]) if "shortest" in tpl else (ls[-1], ls[-2])
        return {pick[1]} if abs(pick[0] - nxt[0]) > 0.05 else {"TIE"}
    if tpl.startswith("A-door-path"):
        src, dst_ = anchors if anchors else t["target"][:2]; st = mm.storey(g(src)); edges = {}
        for x in f.by_type("IfcDoor"):
            if mm.m(x) is None or mm.storey(x) != st:
                continue
            sd = mm.door_sides(x)
            if sd:
                edges.setdefault(sd[0], []).append((sd[1], x.GlobalId)); edges.setdefault(sd[1], []).append((sd[0], x.GlobalId))
        from collections import deque
        dist = {src: 0}; paths = {src: [[]]}; dq = deque([src])
        while dq:
            u_ = dq.popleft()
            for v, door in edges.get(u_, []):
                if v not in dist:
                    dist[v] = dist[u_] + 1; paths[v] = [p_ + [door] for p_ in paths[u_]]; dq.append(v)
                elif dist[v] == dist[u_] + 1:
                    paths[v] += [p_ + [door] for p_ in paths[u_]]
        return set(paths[dst_][0]) if dst_ in paths and len(paths[dst_]) == 1 else {"NO-UNIQUE-PATH"}
    if t["intent"].get("kind") == "number":
        it = t["intent"]; val = None
        def axlen(w):
            a_ = mm.axis_pts(w); return None if a_ is None else float(np.linalg.norm(a_[1] - a_[0]))
        def ref_wall_of_prompt():
            r_ = resolve(mm, {**t, "template": "X-ref", "intent": {"type": "x"}})
            m_ = re.search(r"the wall with GlobalId ([0-9A-Za-z_$]{22})", p)
            return r_ if (r_ and len(r_) == 1) else ({m_.group(1)} if (m_ and style == "D") else None)
        def ref_space_of_prompt():
            m_ = re.search(r"the space with GlobalId ([0-9A-Za-z_$]{22})", p)
            if m_ and style in (None, "D"):
                return {m_.group(1)}
            r_ = resolve(mm, {**t, "template": "X-ref", "intent": {"type": "x"}}); return r_ if (r_ and len(r_) == 1) else None
        st_m = re.search(r"storey named '[^']*' \(GlobalId ([0-9A-Za-z_$]{22})\)", p)
        if tpl.startswith("R-wall-length"):
            r_ = ref_wall_of_prompt(); val = axlen(g(next(iter(r_)))) if r_ else None
        elif tpl.startswith("R-storey-wall-length"):
            st = g(st_m.group(1)); ls = [axlen(w) for w in f.by_type("IfcWall") if mm.storey(w) == st]; val = None if any(x is None for x in ls) else sum(ls)
        elif tpl.startswith("R-count-windows"):
            st = g(st_m.group(1)); val = sum(1 for x in f.by_type("IfcWindow") if mm.storey(x) == st)
        elif tpl.startswith("R-count-doors"):
            st = g(st_m.group(1)); val = sum(1 for x in f.by_type("IfcDoor") if mm.storey(x) == st)
        elif tpl.startswith("R-count-large-fills"):
            import ifcopenshell.util.unit as uu; sc = uu.calculate_unit_scale(f); st = g(st_m.group(1))
            val = sum(1 for x in f.by_type("IfcDoor") + f.by_type("IfcWindow") if mm.storey(x) == st and x.OverallWidth and x.OverallHeight and float(x.OverallWidth) * float(x.OverallHeight) * sc * sc > 2.0)
        elif tpl.startswith("R-space-area"):
            r_ = ref_space_of_prompt(); fp = mm.footprint(g(next(iter(r_)))) if r_ else None; val = fp.area if fp is not None else None
        elif tpl.startswith("R-space-volume"):
            r_ = ref_space_of_prompt(); sp = g(next(iter(r_))) if r_ else None; fp = mm.footprint(sp) if sp else None
            val = fp.area * float(mm.bb(sp)[1][2] - mm.bb(sp)[0][2]) if fp is not None else None
        elif tpl.startswith("R-column-distance"):
            a_, b_ = re.findall(r"column with GlobalId ([0-9A-Za-z_$]{22})", p)[:2]; val = float(np.linalg.norm(mm.c(g(a_))[:2] - mm.c(g(b_))[:2]))
        elif tpl.startswith("R-storey-gross-area") and st_m:
            st = g(st_m.group(1)); sl = [x for x in flat_slabs_of(f, mm) if mm.storey(x) == st and mm.footprint(x) is not None]
            if len(sl) == 1:
                fp = mm.footprint(sl[0]); val = sum(Polygon(P.exterior).area for P in getattr(fp, "geoms", [fp]))   # outer boundary: openings included
        elif tpl.startswith("R-space-perimeter"):
            r_ = ref_space_of_prompt(); fp = mm.footprint(g(next(iter(r_)))) if r_ else None; val = float(fp.exterior.length) if fp is not None else None
        elif tpl.startswith("R-wall-net-area"):
            r_ = ref_wall_of_prompt()
            if r_:
                w_ = g(next(iter(r_))); L_ = axlen(w_); bb = mm.bb(w_); H_ = float(bb[1][2] - bb[0][2]); u_ = mm.wdir(w_); op = 0.0
                for rel in w_.HasOpenings or []:
                    o_ = rel.RelatedOpeningElement; v_ = mm.m(o_)
                    if v_ is not None:
                        pu_ = v_[:, :2] @ u_; op += float(pu_.max() - pu_.min()) * float(v_[:, 2].max() - v_[:, 2].min())
                val = L_ * H_ - op if L_ is not None else None
        elif tpl.startswith("R-door-width"):
            r_ = resolve(mm, {**t, "template": "X-ref", "intent": {"type": "x"}}) if style != "D" else {re.search(r"the door with GlobalId ([0-9A-Za-z_$]{22})", p).group(1)}
            if r_ and len(r_) == 1:
                d_ = g(next(iter(r_))); w_ = mm.host(d_); u_ = mm.wdir(w_); v_ = mm.m(d_)[:, :2] @ u_; val = float(v_.max() - v_.min())
        if val is None:
            return None
        tol = max(it.get("tol", 0.0), it.get("rel_tol", 0.0) * abs(it["value"]))
        return set(t["target"][:1]) if abs(val - it["value"]) <= tol + 1e-9 else {f"VALUE {val:.3f} vs {it['value']}"}
    if tpl.startswith("R-non-adjacent-space"):
        A, STg = anchors; st = g(STg); sp = g(A); adj = {o for wg in mm.swalls(sp) for o in mm.bspaces(g(wg))} - {A}
        return {o.GlobalId for o in f.by_type("IfcSpace") if mm.storey(o) == st and o.GlobalId != A and o.GlobalId not in adj}
    if tpl.startswith("A-vertical-connector"):
        st = g(re.search(r"\(GlobalId ([0-9A-Za-z_$]{22})\)", p).group(1)); return {x.GlobalId for x in f.by_type("IfcStair") if mm.storey(x) == st}
    if tpl.startswith("C-create-wall-S"):     # intent p0/p1 must be the anchor wall's Axis shifted by the stated offset in the stated direction
        m2 = re.search(r"parallel to the wall with GlobalId ([0-9A-Za-z_$]{22}), ([\d.]+) m away from it towards (north|south|east|west)", p)
        W = g(m2.group(1)); off = float(m2.group(2)); d = DIRV[m2.group(3)]; ax = mm.axis_pts(W)
        if ax is None:
            return {"NO-AXIS"}
        want_pts = [ax[0] + d * off, ax[1] + d * off]; it = t["intent"]; got = [np.array(it["p0"]), np.array(it["p1"])]
        err = min(np.linalg.norm(got[0] - want_pts[0]) + np.linalg.norm(got[1] - want_pts[1]), np.linalg.norm(got[0] - want_pts[1]) + np.linalg.norm(got[1] - want_pts[0]))
        return set(t["target"]) if err < 0.03 else {f"AXIS-ERR-{err:.3f}"}
    if tpl.startswith(("C-create-column-", "C-create-slab-")) and not tpl.endswith("-S"):   # geometry of the intent vs the described space's footprint
        if "bounded by exactly the walls" in p:
            ids = set(re.findall(r"[0-9A-Za-z_$]{22}", p.split("bounded by exactly the walls with GlobalIds")[1].split(" (compute")[0].split(", with its top")[0].split(", contained")[0]))
            ids = {x for x in ids if any(x == w.GlobalId for w in walls)}; ref = {sp.GlobalId for sp in f.by_type("IfcSpace") if mm.swalls(sp) == ids}
        else:
            ref = mm.ref_space(p, tpl)
        if not ref or len(ref) != 1:
            return set()
        sp = g(next(iter(ref))); fp = mm.footprint(sp); it = t["intent"]
        if tpl.startswith("C-create-column"):
            cen = np.array(fp.centroid.coords[0]); return set(t["target"]) if np.linalg.norm(cen - np.array(it["xy"])) < 0.03 else {f"CENTROID-ERR-{np.linalg.norm(cen - np.array(it['xy'])):.3f}"}
        P = Polygon(it["polygon"]); return set(t["target"]) if P.symmetric_difference(fp).area < 0.05 * fp.area + 0.05 else {f"POLY-ERR-{P.symmetric_difference(fp).area:.3f}"}
    if tpl.startswith("C-create-space-") and not tpl.endswith("-D"):   # the recorded polygon must be closed by the named walls / surround the point
        it = t["intent"]; P = np.array(it["polygon"]); st = g(it["storey"])
        if tpl.endswith("-T"):
            ids = set(re.findall(r"[0-9A-Za-z_$]{22}", p.split("enclosed by the walls with GlobalIds")[1].split(" on the storey")[0])); wl = [g(x) for x in ids]
        else:
            wl = [w for w in walls if mm.storey(w) == st]
            m2 = re.search(r"around the point \(([-\d.]+), ([-\d.]+)\)", p)
            if not Polygon(P).contains(Point(float(m2.group(1)), float(m2.group(2)))):
                return {"POINT-OUTSIDE"}
        fps = [mm.footprint(w) for w in wl]; fps = [x for x in fps if x is not None]
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]; mid = (a + b) / 2
            if not any(fp_.distance(Point(mid)) < 0.06 and fp_.distance(Point(a)) < 0.12 and fp_.distance(Point(b)) < 0.12 for fp_ in fps):
                return {f"EDGE-{i}-NOT-ON-WALL"}
        return set(t["target"])
    if tpl.startswith("A-entrance"):
        st = g(re.search(r"\(GlobalId ([0-9A-Za-z_$]{22})\)", p).group(1))
        ext = {d.GlobalId for d in f.by_type("IfcDoor") if mm.storey(d) == st and (ue.get_psets(mm.host(d)).get("Pset_WallCommon") or {}).get("IsExternal")}
        if "leads outside" in p:     # v8 definition: a space beyond one face of the wall at the door, none beyond the other (meshes)
            fps = [fp for sp in f.by_type("IfcSpace") if mm.storey(sp) == st and (fp := mm.footprint(sp)) is not None]
            def leads_out(d):
                w = mm.host(d); u = mm.wdir(w); n = np.array([-u[1], u[0]]); cd = mm.c(d)[:2]; pn = mm.m(w)[:, :2] @ n
                half = float(pn.max() - pn.min()) / 2; base = cd + n * (float((pn.max() + pn.min()) / 2) - float(cd @ n))
                hit = {sd: any(fp.contains(Point(base + sd * n * (half + gap))) for gap in (0.15, 0.3, 0.5) for fp in fps) for sd in (1, -1)}
                return hit[1] != hit[-1]
            ext = {x for x in ext if leads_out(g(x))}
        return ext if len(ext) == 1 else set()
    return None


by = collections.defaultdict(lambda: [0, 0, 0]); bad = 0
sel = [t for t in tasks if t.get("meta", {}).get("style") in ("T", "S", "P", "V") or t["template"].split("-")[0] in ("A", "R") or t["template"].startswith("S-")][:lim]
cache = {}
for t in sel:
    if t["ifc"] not in cache:
        cache.clear()      # one building at a time (sort the list by building for large runs)
        cache[t["ifc"]] = M(ifcopenshell.open(t["ifc"]))
    mm = cache[t["ifc"]]
    try:
        got = resolve(mm, t)
    except Exception as e:  # noqa: BLE001
        got = f"EXC {type(e).__name__}: {str(e)[:100]}"
    want = set(t["intent"]["value"]) if t["intent"].get("kind") == "guids" else ({t["intent"]["value"]} if t["intent"].get("kind") == "guid" else set(t["target"][:1]))
    if t["template"].startswith("C-"):   # create tasks: the handler returns the task's own target on success (host wall / described space / storey)
        want = set(t["target"]) if t["intent"]["type"] in ("create_space", "create_column", "create_slab", "create_wall") else {t["intent"]["host"]}
    if t["template"].startswith("D-delete-windows-facing"):   # E011 cover4: the description denotes the whole window set
        want = set(t["intent"]["guids"])
    key = t["template"]
    if got is None or want is None:
        by[key][2] += 1; continue
    if isinstance(got, str) or got != want:
        by[key][1] += 1; bad += 1; print(f"MISMATCH {t['task_id']}: got {got if isinstance(got, str) else sorted(got)} want {sorted(want)} | {t['prompt'][:160]}")
    else:
        by[key][0] += 1
for k, (ok, ko, skip) in sorted(by.items()):
    print(f"{k:32s} ok {ok:3d} mismatch {ko:3d} unhandled {skip:3d}")
print(f"TOTAL checked {sum(v[0] + v[1] for v in by.values())} mismatches {bad} unhandled {sum(v[2] for v in by.values())}")
