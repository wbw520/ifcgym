"""Rule checks that a paraphrase keeps the meaning-critical tokens of a generated prompt (review 2026-10-24: DeepSeek turned 'farthest
to your left' into 'right' in 60-70 % of such prompts; used by paraphrase.check as a guard and by scripts/audit_paraphrase.py).
Semantic shifts these rules cannot see (an id moved to another element) are left to scripts/judge_paraphrase.py."""
import re

GUID = re.compile(r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])")
NUM = re.compile(r"(?<![0-9A-Za-z_$.])[-+]?\d+(?:\.\d+)?(?![0-9.])")
HINT = re.compile(r"\(\+X is east, \+Y is north\)|\+X is east, \+Y is north", re.I)
OP = re.compile(r"(?:to|on|at) your (left|right)\b", re.I)
PAIRS = [
    (["counter-clockwise", "counterclockwise", "anticlockwise", "anti-clockwise"], ["clockwise"]),
    (["north", "northern", "northernmost", "northward"], ["south", "southern", "southernmost", "southward"]),
    (["east", "eastern", "easternmost", "eastward"], ["west", "western", "westernmost", "westward"]),
    (["above"], ["below", "beneath", "under"]),
    (["lowest"], ["highest", "top"]), (["farthest", "furthest"], ["closest", "nearest"]),
    (["longest", "greatest length", "maximum length"], ["shortest", "minimum length"]),
    (["inwards", "inward", "into the building"], ["outwards", "outward", "to the outside"]),
]


def clean(text):
    for ch in "‐‑‒–—−":
        text = text.replace(ch, "-")
    return HINT.sub(" ", text)


def _hits(text, words):
    t = text.lower(); out = []
    for w in words:
        out += [m.start() for m in re.finditer(r"(?<![a-z-])" + re.escape(w) + r"(?![a-z])", t)]
    return out


def _sides(text, A, B):
    a, b = _hits(text, A), _hits(text, B)
    if B[0] == "clockwise":
        b = [i for i in b if not any(j <= i < j + 20 for j in a)]
    return bool(a), bool(b)


def norm_num(x):
    x = x.lstrip("+-")
    return x.rstrip("0").rstrip(".") if "." in x else x


def problems(orig: str, new: str) -> list:
    """Meaning-critical differences between the generated prompt and its rewrite (empty = passes the rules)."""
    out = []; o, n = clean(orig), clean(new)
    if set(GUID.findall(orig)) != set(GUID.findall(new)):
        out.append("identifier set differs")
    miss = {norm_num(x) for x in NUM.findall(GUID.sub(" ", o))} - {norm_num(x) for x in NUM.findall(GUID.sub(" ", n))}
    if miss:
        out.append(f"missing number(s) {sorted(miss)}")
    s0 = {x.lower() for x in OP.findall(o)}
    if s0:
        side = s0.pop(); other = "right" if side == "left" else "left"
        if not re.search(rf"\b{side}\b", n, re.I) or re.search(rf"(?:farthest|furthest|offset|on|to) (?:your |the )?{other}\b", n, re.I):
            out.append(f"operative side '{side}' not kept")
    for A, B in PAIRS:
        a0, b0 = _sides(o, A, B); a1, b1 = _sides(n, A, B)
        if (a0 and not b0 and (not a1 or b1)) or (b0 and not a0 and (not b1 or a1)):
            out.append(f"polarity {A[0]}/{B[0]} changed or dropped")
    return out
