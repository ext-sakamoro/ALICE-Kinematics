#!/usr/bin/env python3
"""scripts/nan_boundary_scan.py — every public function that returns an f32 value has a NaN test

ALICE-Law G2 (src/nan.rs): a NaN leaving this crate is the canonical quiet NaN 0x7fc0_0000,
whatever the codegen did inside (the release build flipped the sign of the IK's NaN angles).
The canonicalisation is applied at the outputs, so a public function added later without it
would expose a NaN of unspecified bits again. This scan lists the outputs and fails unless
each one has its own test in tests/nan_boundaries.rs.

A boundary is:
  - a `pub fn` in src/ whose return type names f32, f64 or Vec3k (arrays, tuples and Options
    of them included, and a module-level `type` alias of any of them),
  - a function returning `Self` inside `impl Vec3k` (Vec3k is the crate's float value type),
  - a function of a trait implemented for Vec3k (`impl Add for Vec3k` and the like).
Constructors returning `Self` of other types are not outputs: their float fields are read
back through the accessors, which are boundaries themselves.

The test of boundary `Type::name` is `fn boundary_<type>_<name>` (lower case; a free function
`name` is `fn boundary_<name>`). Fails on: a boundary with no test, a `boundary_` test that
names no boundary, a boundary listed as exempt or debt that is not one (or that also has a
test), a new boundary that is neither tested, exempt nor debt, and a scan that found no boundary.
The struct-carrier outputs still to convert are DEBT, counted in the summary line.

usage: scripts/nan_boundary_scan.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: boundaries that need no test, each with its reason (exact names)
EXEMPT = {
    "Vec3k::new": "a const constructor that stores the caller's three values; it computes nothing, "
                  "and every operation on the value is a boundary",
    "alice_lint_probe": "compiled only with --cfg alice_lint_probe for the clippy step's lint probe; "
                        "no build of the crate contains it",
    "Intent::decode": "decodes Q8.8 integers (q8_to_f32 of an i16): every float it returns is a finite number, so no NaN can leave it",
    "ExtendedIntent::decode": "decodes Q8.8 integers (q8_to_f32 of an i16): every float it returns is a finite number, so no NaN can leave it",
    "intents": "an iterator of Intent::decode over the packets: decodes Q8.8 integers (q8_to_f32 of an i16): every float it returns is a finite number, so no NaN can leave it",
    "Skeleton::default_humanoid": "Skeleton::with_height(1.70): no input reaches it, and with_height's "
                                  "output is canonical",
    "distance_based_skinning": "every weight is 0, 1 or 1 / (d² + 1e-6) over the sum of such terms: "
                               "a NaN distance fails `< f32::MAX` and weighs 0, so no NaN can leave it",
    "Skeleton::find_joint": "returns a reference to a joint as stored; every crate computation that "
                            "stores into a joint canonicalises (with_height, rescale, hypothesis_to_skeleton, "
                            "auto_rig_on_sphere), the rest is the caller's own write through the pub fields",
    "extended_intents": "an iterator of ExtendedIntent::decode over the packets: decodes Q8.8 integers (q8_to_f32 of an i16): every float it returns is a finite number, so no NaN can leave it",
}

#: struct-carrier outputs (a returned crate type with public float fields) not converted yet:
#: tracked debt, ordered by exposure (what crosses the wire first). Each entry leaves this list
#: in the branch that canonicalises its float fields and adds its `boundary_` test; the scan
#: fails when an entry is no longer an output, or when a new struct-carrier output is not listed
DEBT = [
    # constructors, accessors returning stored values, defaults
    "Intent::reach", "Intent::grasp", "ExtendedIntent::new", "JointConstraint::new",
    "JointConstraint::free", "Joint::new", "ArmChain::right_arm", "Finger::new",
    "HandModel::right_hand", "HandModel::left_hand", "HandModel::finger", "HandModel::finger_mut",
    "Predictor::new", "Predictor::default", "MeshView::from_vertices", "AutoRigConfig::default",
    "InfluenceRadii::default", "SkeletonHypothesis::default",
]

IMPL_RE = re.compile(r"^impl(?:<[^>]*>)?\s+(?:(?P<trait>[\w:<>, ]+?)\s+for\s+)?(?P<ty>\w+)(?:<[^>]*>)?\s*\{")
FN_RE = re.compile(r"^\s*(?:(?P<pub>pub\s+)|pub\([^)]*\)\s+)?(?:const\s+)?fn\s+(?P<name>\w+)\s*(?:<[^>]*>)?\(")
STRUCT_RE = re.compile(r"^\s*pub\s+struct\s+(?P<name>\w+)")
FIELD_RE = re.compile(r"^\s*pub\s+\w+\s*:\s*(?P<ty>[^,]+),?\s*$")
# a module-level type alias (any visibility): `pub type Angle = f32;` makes `-> Angle` an f32 output
ALIAS_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?type\s+(?P<name>\w+)\s*(?:<[^=]*>)?\s*=\s*(?P<ty>[^;]+);")


def float_types(src: Path) -> set[str]:
    """the crate's public types that carry a float a caller can read: a public field whose type
    names f32, f64 or another such type, and a module-level type alias of such a type (to a
    fixpoint; an associated `type Output = ..` inside an impl or trait is not an alias)"""
    fields: dict[str, list[str]] = {}
    for path in sorted(src.rglob("*.rs")):
        lines = path.read_text(encoding="utf-8").splitlines()
        cur: str | None = None
        depth = 0
        nest = 0  # brace depth of the file
        assoc: list[int] = []  # depths at which an impl / trait block opened
        for line in lines:
            code = line.split("//", 1)[0]
            a = ALIAS_RE.match(code)
            if a and not assoc:
                fields.setdefault(a.group("name"), []).append(a.group("ty"))
            if re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:impl|trait)\b", code) and "{" in code:
                assoc.append(nest)
            nest += code.count("{") - code.count("}")
            while assoc and nest <= assoc[-1]:
                assoc.pop()
            m = STRUCT_RE.match(code)
            if m and cur is None:
                cur = m.group("name")
                fields.setdefault(cur, [])
                depth = 0
                if "(" in code and ";" in code:  # tuple struct on one line
                    fields[cur] += re.findall(r"pub\s+([^,)]+)", code)
                    cur = None
                    continue
            if cur is not None:
                f = FIELD_RE.match(code)
                if f:
                    fields[cur].append(f.group("ty"))
                depth += code.count("{") - code.count("}")
                if depth <= 0 and "}" in code:
                    cur = None
    found = {"f32", "f64"}
    while True:
        grown = {t for t, tys in fields.items()
                 if any(re.search(rf"\b{re.escape(x)}\b", ty) for ty in tys for x in found)}
        if grown <= found:
            return found - {"f32", "f64"}
        found |= grown


def boundaries(src: Path) -> list[str]:
    """`Type::name` / `name` of every output function of the crate's source"""
    carriers = float_types(src)
    names = ["f32", "f64", *sorted(carriers)]
    carries = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b")
    out: list[str] = []
    for path in sorted(src.rglob("*.rs")):
        lines = path.read_text(encoding="utf-8").splitlines()
        impl_ty: str | None = None
        impl_trait: str | None = None
        depth = 0
        in_test = False
        pending_test = False
        for i, line in enumerate(lines):
            code = line.split("//", 1)[0]
            if depth == 0:
                stripped = code.strip()
                if stripped.startswith("#[cfg(test)]"):
                    pending_test = True  # applies to the next item
                elif stripped and not stripped.startswith("#["):
                    in_test = pending_test  # this item, until its block closes
                    pending_test = False
                m = IMPL_RE.match(code)
                impl_ty, impl_trait = (m.group("ty"), m.group("trait")) if m else (None, None)
            m = FN_RE.match(code)
            if m and not in_test:
                sig = code
                j = i
                while "{" not in sig and ";" not in sig and j + 1 < len(lines):
                    j += 1
                    sig += " " + lines[j].split("//", 1)[0]
                ret = sig.split("->", 1)[1].split("{", 1)[0] if "->" in sig else ""
                if impl_ty:
                    ret = re.sub(r"\bSelf\b", impl_ty, ret)
                name = m.group("name")
                public = bool(m.group("pub"))
                trait_on_carrier = impl_trait is not None and impl_ty in carriers
                if ret.strip() and carries.search(ret) and (public or trait_on_carrier):
                    out.append(f"{impl_ty}::{name}" if impl_ty else name)
            depth += code.count("{") - code.count("}")
            if depth == 0:
                impl_ty, impl_trait = None, None
    return out


def test_name(boundary: str) -> str:
    return "boundary_" + boundary.lower().replace("::", "_")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args(argv)
    found = boundaries(a.root / "src")
    tests_file = a.root / "tests" / "nan_boundaries.rs"
    tests = set(re.findall(r"^\s*fn (boundary_\w+)\s*\(", tests_file.read_text(encoding="utf-8"), re.M)) \
        if tests_file.exists() else set()
    problems: list[str] = []
    if not found:
        problems.append("no output function found in src/ (the scan compared nothing)")
    if len(set(DEBT)) != len(DEBT):
        problems.append("DEBT lists an entry twice")
    for b in [*EXEMPT, *DEBT]:
        if b not in found:
            problems.append(f"{b!r} is listed (exempt or debt) but is no output function")
    for b in set(EXEMPT) & set(DEBT):
        problems.append(f"{b!r} is both exempt and debt")
    tested = {test_name(b): b for b in found}
    if len(tested) != len(set(found)):
        problems.append("two output functions map to one test name")
    for b in found:
        has_test = test_name(b) in tests
        listed = b in EXEMPT or b in DEBT
        if not has_test and not listed:
            problems.append(f"{b}: no test `fn {test_name(b)}` in tests/nan_boundaries.rs (nor exempt, nor debt)")
        if has_test and listed:
            problems.append(f"{b}: has a test and is still listed as exempt or debt (remove it from the list)")
    for t in sorted(tests - set(tested)):
        problems.append(f"tests/nan_boundaries.rs: `fn {t}` names no output function")
    for p in problems:
        print(f"nan-boundary: {p}")
    n_test = sum(1 for b in found if test_name(b) in tests)
    print(f"nan-boundary: {len(found)} output function(s): {n_test} tested, {len(EXEMPT)} exempt, "
          f"{len(DEBT)} debt (struct carriers not yet canonical); {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
