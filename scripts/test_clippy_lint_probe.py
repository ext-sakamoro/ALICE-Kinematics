#!/usr/bin/env python3
"""Tests for scripts/clippy_lint_probe.py (no cargo run: synthetic JSON lines)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clippy_lint_probe as p  # noqa: E402

SRC = "#![no_std]\n\n#[cfg(all(alice_lint_probe, feature = \"std\"))]\npub fn alice_lint_probe(x: f32) -> f32 {\n    x.sin()\n}\n"


def diag(file: str, line: int, method: str, code: str = p.LINT) -> str:
    return json.dumps({"reason": "compiler-message", "message": {
        "code": {"code": code}, "message": f"use of a disallowed method `{method}`",
        "spans": [{"is_primary": True, "file_name": file, "line_start": line}]}})


class ProbeLines(unittest.TestCase):
    def test_finds_the_function_body(self):
        self.assertEqual(list(p.probe_lines(SRC)), [4, 5, 6])

    def test_missing_probe_fails(self):
        with self.assertRaises(SystemExit):
            p.probe_lines("pub fn other() {}\n")


class Classify(unittest.TestCase):
    probe = range(4, 7)

    def test_hit_at_the_probe_is_counted(self):
        self.assertEqual(p.classify([diag("src/lib.rs", 5, "f32::sin")] * 2, self.probe), (2, []))

    def test_nothing_linted_is_zero(self):
        self.assertEqual(p.classify(['{"reason":"build-finished","success":true}', "not json"], self.probe), (0, []))

    def test_other_disallowed_call_is_reported(self):
        n, others = p.classify([diag("src/lib.rs", 5, "f32::sin"), diag("src/joint.rs", 9, "f32::cos")], self.probe)
        self.assertEqual(n, 1)
        self.assertEqual(len(others), 1)

    def test_other_method_at_the_probe_line_is_not_the_probe(self):
        n, others = p.classify([diag("src/lib.rs", 5, "f32::cos")], self.probe)
        self.assertEqual((n, len(others)), (0, 1))

    def test_other_lints_are_ignored(self):
        self.assertEqual(p.classify([diag("src/lib.rs", 5, "f32::sin", code="clippy::pedantic")], self.probe), (0, []))


class WithJson(unittest.TestCase):
    def test_inserts_after_clippy(self):
        self.assertEqual(p.with_json(["cargo", "clippy", "--lib", "--", "-W", "x"]),
                         ["cargo", "clippy", "--message-format=json", "--lib", "--", "-W", "x"])

    def test_refuses_other_commands(self):
        with self.assertRaises(SystemExit):
            p.with_json(["cargo", "build"])


if __name__ == "__main__":
    unittest.main()
