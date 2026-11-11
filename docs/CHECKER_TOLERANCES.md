# Checker tolerances and their provenance

The four checkers (`ifcgym/checkers.py`) compare meshes and attributes in metres. The numeric tolerances below were set as
engineering judgement when the checkers were written and then checked against construction tolerance standards and against the
tolerances of the two external benchmarks.

| check | tolerance | origin |
|---|---|---|
| translate (wall, door/window, column), heights, wall thickness, slab thickness | 0.02 m | ours; same order as BIBIMBAP's geometry tests (`TOLERANCE = 0.02` in several of its tests, 0.05-0.1 in others) |
| shorten / rotate: axis end points | 0.03 m per end | construction tolerance standards: DIN 18202 ±30 mm maximum, ACI 117 ±25 mm, USIBD LOA20 ceiling 50 mm |
| create door/window/wall/column: dimensions, end points, centre | 0.03 m | same anchors; BIBIMBAP uses 0.05-0.1 m |
| create slab / space footprint | symmetric difference ≤ 5 % of the area + 0.05 m² (space: + 0.1 m²) | ours |
| joined_to (wall joint) | surface gap ≤ 0.02 m or solids overlap | ours |
| numeric answers (area, volume, length, distance) | relative 1 % + absolute 0.1 | aligned to BIBIMBAP's numeric tests (`rel_tol = 0.001, abs_tol = 0.1`), slightly looser relatively because our truths come from meshes |
| door width answer | relative 1 % + absolute 0.02 m | ours (0.1 m would not separate 0.8 / 0.9 m doors) |
| counts, GlobalId answers | exact | both benchmarks |
| BIM-Edit external scoring | bbox IoU threshold 0.05 | BIM-Edit paper / our calibrated re-implementation (`ifcedit/bimedit_eval.py`) |

A perfectly executed prompt never fails on rounding: prompts carry two to three decimals, well inside every tolerance above.
Standards consulted: DIN 18202, ACI 117, BS 5606, USIBD Level of Accuracy, GSA/NIST BIM guides, Bonduel et al. (2017),
Bassier et al. (2020), and the BIM-Edit and BIBIMBAP test suites. Whether practitioners accept these thresholds as a correct edit
was examined in the paper's expert evaluation.
