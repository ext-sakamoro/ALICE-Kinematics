#!/usr/bin/env python3
"""Check that a clippy step lints the build where the G1 ban list can fire.

The crate is no_std by default, so f32 / f64 transcendentals exist only in the
std build (feature `std`, or `lol`, which enables it). A clippy step that never
reaches that build passes whatever its source holds. This runs the given
`cargo clippy ...` command again with `--cfg alice_lint_probe` added to
RUSTFLAGS, which compiles the probe at the end of src/lib.rs (one call of
f32::sin), and with `--message-format=json`.

It fails unless

  (i)   at least one `clippy::disallowed_methods` diagnostic names f32::sin at
        the probe (src/lib.rs, inside `alice_lint_probe`), and
  (ii)  no other `clippy::disallowed_methods` diagnostic appears (the step itself
        must be clean; the probe is the only planted call).

Zero diagnostics at the probe means the step compiled nothing that clippy
linted with the std build, which is the failure this exists to catch.

usage: scripts/clippy_lint_probe.py -- cargo clippy <args> [-- <clippy args>]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_FILE = "src/lib.rs"
PROBE_FN = "alice_lint_probe"
LINT = "clippy::disallowed_methods"
CFG = "--cfg alice_lint_probe"


def probe_lines(text: str) -> range:
    """Line range (1-based, inclusive) of the probe function in src/lib.rs."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"\s*pub fn {PROBE_FN}\b", line):
            for j in range(i, len(lines)):
                if lines[j].startswith("}"):
                    return range(i + 1, j + 2)
    raise SystemExit(f"clippy_lint_probe: no `pub fn {PROBE_FN}` in {PROBE_FILE}")


def with_json(cmd: list[str]) -> list[str]:
    """Insert --message-format=json after `clippy` (before any `--`)."""
    if cmd[:2] != ["cargo", "clippy"]:
        raise SystemExit("clippy_lint_probe: the command must start with `cargo clippy`")
    return cmd[:2] + ["--message-format=json"] + cmd[2:]


def classify(stream, probe: range) -> tuple[int, list[str]]:
    """(diagnostics at the probe, other disallowed_methods diagnostics)."""
    at_probe, others = 0, []
    for raw in stream:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        diag = msg["message"]
        if (diag.get("code") or {}).get("code") != LINT:
            continue
        spans = [s for s in diag["spans"] if s.get("is_primary")]
        span = spans[0] if spans else None
        if (span and span["file_name"] == PROBE_FILE and span["line_start"] in probe
                and "`f32::sin`" in diag["message"]):
            at_probe += 1
        else:
            where = f"{span['file_name']}:{span['line_start']}" if span else "?"
            others.append(f"{where}: {diag['message']}")
    return at_probe, others


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = with_json(argv[argv.index("--") + 1:])
    probe = probe_lines((ROOT / PROBE_FILE).read_text(encoding="utf-8"))
    env = dict(os.environ)
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} {CFG}".strip()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", check=False)
    at_probe, others = classify(proc.stdout.splitlines(), probe)
    for o in others:
        print(f"clippy_lint_probe: unexpected {LINT}: {o}", file=sys.stderr)
    if at_probe == 0:
        print(f"clippy_lint_probe: 0 diagnostics at the probe: `{' '.join(argv[argv.index('--') + 1:])}` "
              "does not lint the std build", file=sys.stderr)
        return 1
    if others:
        return 1
    print(f"clippy_lint_probe: {at_probe} diagnostic(s) at the probe, 0 others")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
