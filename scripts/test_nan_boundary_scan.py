#!/usr/bin/env python3
"""tests of scripts/nan_boundary_scan.py on fixture crates (python3 scripts/test_nan_boundary_scan.py)"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nan_boundary_scan as scan  # noqa: E402

LIB = """
pub struct Vec3k {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Vec3k {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x
    }
    pub(crate) fn private_len(self) -> f32 {
        self.x
    }
}
impl core::ops::Neg for Vec3k {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}
pub struct Fit {
    pub error: f32,
}
pub struct Wrapper {
    pub fit: Fit,
}
pub fn fit() -> Option<Wrapper> {
    None
}
pub fn count() -> usize {
    0
}
#[cfg(test)]
mod tests {
    pub fn helper() -> f32 {
        0.0
    }
}
"""


class Scan(unittest.TestCase):
    def run_on(self, lib: str, tests: str, exempt=None, debt=None) -> tuple[int, str]:
        saved = (scan.EXEMPT, scan.DEBT)
        scan.EXEMPT = {"Vec3k::new": "store"} if exempt is None else exempt
        scan.DEBT = ["fit"] if debt is None else debt
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "src").mkdir()
                (root / "tests").mkdir()
                (root / "src" / "lib.rs").write_text(lib, encoding="utf-8")
                (root / "tests" / "nan_boundaries.rs").write_text(tests, encoding="utf-8")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = scan.main(["--root", str(root)])
                return rc, out.getvalue()
        finally:
            scan.EXEMPT, scan.DEBT = saved

    ALL = "fn boundary_vec3k_dot() {}\nfn boundary_vec3k_neg() {}\n"

    def test_the_outputs_are_found(self):
        found = None
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lib.rs").write_text(LIB, encoding="utf-8")
            found = scan.boundaries(Path(d))
        # pub f32 return, a trait impl on Vec3k, Self of Vec3k, a struct carrier through a field
        # of a struct carrier (Wrapper -> Fit); not pub(crate), not usize, not a cfg(test) module
        self.assertEqual(sorted(found), ["Vec3k::dot", "Vec3k::neg", "Vec3k::new", "fit"])

    def test_a_type_alias_of_a_float_is_an_output(self):
        # a module-level alias (pub or not, nested aliases to a fixpoint) names a float; an
        # associated `type Item = f32` inside an impl is not an alias of the module
        lib = LIB + """
pub type Angle = f32;
type Turn = Angle;
pub fn angle() -> Angle {
    0.0
}
pub fn turn() -> Option<Turn> {
    None
}
pub struct Counter;
impl Iterator for Counter {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
pub fn item() -> Item {
    Item
}
pub struct Item;
"""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lib.rs").write_text(lib, encoding="utf-8")
            found = scan.boundaries(Path(d))
        self.assertIn("angle", found)
        self.assertIn("turn", found)
        self.assertNotIn("item", found)

    def test_tested_exempt_and_debt_pass(self):
        rc, out = self.run_on(LIB, self.ALL)
        self.assertEqual(rc, 0, out)
        self.assertIn("4 output function(s): 2 tested, 1 exempt, 1 debt", out)

    def test_an_untested_output_fails(self):
        rc, out = self.run_on(LIB, "fn boundary_vec3k_dot() {}\n")
        self.assertEqual(rc, 1)
        self.assertIn("Vec3k::neg: no test `fn boundary_vec3k_neg`", out)

    def test_a_test_naming_no_output_fails(self):
        rc, out = self.run_on(LIB, self.ALL + "fn boundary_gone() {}\n")
        self.assertEqual(rc, 1)
        self.assertIn("`fn boundary_gone` names no output function", out)

    def test_a_new_struct_output_not_in_the_debt_fails(self):
        rc, out = self.run_on(LIB, self.ALL, debt=[])
        self.assertEqual(rc, 1)
        self.assertIn("fit: no test `fn boundary_fit`", out)

    def test_a_debt_entry_that_is_no_output_fails(self):
        rc, out = self.run_on(LIB, self.ALL, debt=["fit", "gone"])
        self.assertEqual(rc, 1)
        self.assertIn("'gone' is listed (exempt or debt) but is no output function", out)

    def test_a_tested_output_still_in_the_debt_fails(self):
        rc, out = self.run_on(LIB, self.ALL + "fn boundary_fit() {}\n")
        self.assertEqual(rc, 1)
        self.assertIn("fit: has a test and is still listed", out)

    def test_nothing_found_fails(self):
        rc, out = self.run_on("pub fn count() -> usize { 0 }\n", "", exempt={}, debt=[])
        self.assertEqual(rc, 1)
        self.assertIn("compared nothing", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
