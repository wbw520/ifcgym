"""Locations used by IfcGym and IfcEdit, all overridable through environment variables.

    IFCGYM_ROOT     repository root (default: the current working directory); outputs/ and configs/ live below it
    IFCGYM_DATA     dataset root (default: <IFCGYM_ROOT>/data), see docs/DATA.md for the expected layout
    IFCGYM_CACHE    cache directory for checker meshes and graphs (default: <IFCGYM_ROOT>/outputs/cache)
    IFCGYM_MODELS   directory holding model weights for the training and serving scripts (default: <IFCGYM_ROOT>/models)

BIM-Edit task manifest paths (`data/realistic/...`, `data/artificial/...`) map onto the released IFC repository directories
`complex/` and `simple/` (basenames are unique: 648/648 resolve, 0 collisions)."""
from __future__ import annotations

import os

ROOT = os.environ.get("IFCGYM_ROOT", os.getcwd())
DATA = os.environ.get("IFCGYM_DATA", os.path.join(ROOT, "data"))
CACHE = os.environ.get("IFCGYM_CACHE", os.path.join(ROOT, "outputs", "cache"))
MODELS = os.environ.get("IFCGYM_MODELS", os.path.join(ROOT, "models"))

IFCBENCH = os.path.join(DATA, "ifc-bench")                      # real projects: <project>/arc.ifc
SYNTH = os.path.join(DATA, "synth")                             # synthetic buildings: <set>/<building>/arc.ifc
BIMEDIT_TASKS = os.path.join(DATA, "bimedit-runs", "tasks.jsonl")
BIMEDIT_IFC = os.path.join(DATA, "bimedit")
BIBIMBAP = os.environ.get("IFCGYM_BIBIMBAP", os.path.join(DATA, "bibimbap", "code"))
_MAP = {"data/realistic": "complex", "data/artificial": "simple"}


def resolve_bimedit_ifc(path: str) -> str:
    d, b = os.path.split(path)
    sub = _MAP.get(d)
    cand = os.path.join(BIMEDIT_IFC, sub, b) if sub else os.path.join(BIMEDIT_IFC, path)
    if os.path.exists(cand):
        return cand
    for sub in ("complex", "simple"):
        p = os.path.join(BIMEDIT_IFC, sub, b)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"BIM-Edit IFC not found for {path} under {BIMEDIT_IFC}")
