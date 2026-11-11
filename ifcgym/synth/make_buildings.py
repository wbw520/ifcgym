"""Synthetic building factory: CubiCasa5K plans -> IFC buildings with randomised export styles -> generator-v2 tasks.

    python -m ifcgym.synth.make_buildings --root data/cubicasa5k/cubicasa5k --out data/synth/buildings --n 50 --seed 0

Per building: one accepted plan (optionally the same plan on every storey), a random Style (units m/mm, IFC4/IFC2X3,
rect/polyline profiles, revit/archicad names, 1-4 storeys), the IFC at <out>/<id>/arc.ifc, a manifest line with the source
sample and style. Then tasks: outputs/tasks_synth/<id>.jsonl via ifcgym.tasks.generate --perpendicular (per-template 20).
Buildings stay out of evaluation: they are training material only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from ifcgym.synth.cubicasa import load_plan
from ifcgym.synth.ifcbuild import Builder, Style
from ifcgym.synth.tiling import add_core, add_house_stair, tile_plans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tasks-out", default=None); ap.add_argument("--per-template", type=int, default=20)
    ap.add_argument("--start", type=int, default=0, help="skip this many candidate samples first (parallel runs)")
    ap.add_argument("--tiled", type=float, default=0.0, help="fraction of buildings made by tiling 4-8 plans along a corridor with a vertical core")
    ap.add_argument("--families", nargs="*", default=["create", "retrieve", "spatial"], help="extra template families passed to the task generator")
    ap.add_argument("--body", default=None, choices=["extrusion", "brep", "clipped", "faceset"], help="force the wall/space body style (default: random 45/20/15/20 from a survey of 33 real models)")
    ap.add_argument("--sparse", default=None, choices=["0", "1"], help="force sparse relations (no connections / space boundaries; default random 20 percent)")
    ap.add_argument("--fills", default="legacy", choices=["legacy", "survey"], help="door / window / opening styles: legacy = the original "
                    "(vertical plan-profile openings, direct extruded leaves); survey = random per building with the shares of a survey "
                    "of the 7 non-evaluation ifc-bench models, plus extra facade windows in half of them")
    ap.add_argument("--schema", default=None, choices=["IFC4", "IFC4X3", "IFC2X3"], help="force the schema (default random IFC4 40 / IFC4X3 40 / IFC2X3 20; both external benchmarks ship IFC4X3)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    svgs = sorted(glob.glob(str(Path(a.root) / "*" / "*" / "model.svg")))
    rng.shuffle(svgs)
    os.makedirs(a.out, exist_ok=True)
    manifest = open(os.path.join(a.out, "manifest.jsonl"), "a")
    made = 0; pool: list = []
    it = iter(svgs[a.start:])
    while made < a.n:
        svg = next(it, None)
        if svg is None:
            break
        plan = load_plan(str(Path(svg).parent))
        if plan is None:
            continue
        tiled = rng.random() < a.tiled
        if tiled:
            pool.append(plan)
            k = rng.choice([4, 5, 6, 8])
            if len(pool) < k:
                continue
            plan = add_core(tile_plans(pool[:k], corridor=rng.choice([1.8, 2.0, 2.4]))); pool = pool[k:]
        style = Style(units=rng.choice(["m", "m", "mm"]), schema=a.schema or rng.choice(["IFC4", "IFC4", "IFC4X3", "IFC4X3", "IFC2X3"]), profile=rng.choice(["rect", "polyline"]),
                      naming=rng.choice(["revit", "archicad"]), storeys=(rng.choice([2, 3, 4, 5, 6, 8]) if tiled else rng.choice([1, 1, 2, 2, 3, 4])), storey_height=rng.choice([2.8, 3.0, 3.2, 3.5]), seed=rng.randint(0, 10**6),
                      body=a.body or rng.choice(["extrusion"] * 9 + ["brep"] * 4 + ["clipped"] * 3 + ["faceset"] * 4), sparse=(a.sparse == "1") if a.sparse else (rng.random() < 0.2))
        if a.fills == "survey":   # drawn after the legacy fields, so a legacy run's random stream is unchanged
            style.opening_body = rng.choices(["through", "through_rot", "through_poly", "brep", "faceset", "vertical"], [45, 20, 15, 5, 5, 10])[0]
            style.opening_mix = rng.choice([0.0, 0.0, 0.2])
            style.fill_body = rng.choices(["mapped_extrusion", "mapped_brep", "mapped_faceset", "extrusion"], [35, 30, 20, 15])[0]
            style.dense = rng.random() < 0.5
        plan.height = style.storey_height - 0.2
        if not tiled and style.storeys >= 2:
            plan = add_house_stair(plan, rng)
        bid = f"synth_{'tiled' if tiled else Path(svg).parent.parent.name}_{Path(svg).parent.name}_{style.units}_{style.schema.lower()}" + ("" if style.body == "extrusion" else f"_{style.body}") + ("_sparse" if style.sparse else "") + (f"_o{style.opening_body}_f{style.fill_body}" + ("_dense" if style.dense else "") if a.fills == "survey" else "")
        bdir = os.path.join(a.out, bid); os.makedirs(bdir, exist_ok=True)
        try:
            f = Builder(style, name=bid).build([plan]); f.write(os.path.join(bdir, "arc.ifc"))
        except Exception as e:  # noqa: BLE001
            print("build failed", bid, type(e).__name__, str(e)[:100]); continue
        rec = {"id": bid, "source": str(Path(svg).parent), "tiled": tiled, "units": style.units, "schema": style.schema, "profile": style.profile, "body": style.body, "sparse": style.sparse, "naming": style.naming,
               "storeys": style.storeys, **({"opening_body": style.opening_body, "opening_mix": style.opening_mix, "fill_body": style.fill_body, "dense": style.dense} if a.fills == "survey" else {}), "walls": len(plan.walls), "doors": sum(1 for o in plan.doors if o.kind == "door"), "windows": sum(1 for o in plan.doors if o.kind == "window"), "rooms": len(plan.rooms)}
        manifest.write(json.dumps(rec) + "\n"); manifest.flush(); made += 1
        print(rec, flush=True)
        if a.tasks_out:
            os.makedirs(a.tasks_out, exist_ok=True)
            subprocess.run([sys.executable, "-B", "-m", "ifcgym.tasks.generate", "--ifc", os.path.join(bdir, "arc.ifc"), "--project", bid, "--out", os.path.join(a.tasks_out, bid + ".jsonl"),
                            "--per-template", str(a.per_template), "--seed", "1", "--perpendicular"] + (["--families", *a.families] if a.families else []), check=False, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
    print(f"made {made} buildings")


if __name__ == "__main__":
    main()
