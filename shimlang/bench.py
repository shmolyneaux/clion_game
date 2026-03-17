#!/usr/bin/env python3
"""
Benchmark for Shimlang method calls.

Profiles the AttrCall optimization (commit cd83c37) which avoids allocating a
temporary BoundMethod object for every `obj.method(args)` call. Run this script
to observe timing and memory-usage improvements for method-call-heavy code.

Usage:
    cd shimlang/
    python3 bench.py

Output includes:
  - Execution time for a method-call-intensive script (lower is better after opt)
  - Memory words used immediately before GC   (lower = less temporary garbage created)
  - Memory words freed by GC                  (lower = less garbage to collect)
  - Memory words used after  GC               (should be the same baseline either way)
"""
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[33m"
RESET  = "\033[0m"

# ── build ──────────────────────────────────────────────────────────────────────

print("Building shm...")
result = subprocess.run("cargo build --bin shm --release", shell=True)
if result.returncode:
    print(f"{RED}Build failed{RESET}")
    sys.exit(1)
print("Build succeeded.\n")

if sys.platform == "win32":
    exe = "target\\release\\shm.exe"
else:
    exe = "target/release/shm"

# ── benchmark scripts ──────────────────────────────────────────────────────────

# Heavy method-call script: 1 000 iterations of .add() + .scale() + .dot()
# Uses the AttrCall fused opcode (obj.method(args) form).
METHOD_CALL_SCRIPT = b"""
struct Point {
    x,
    y,
    fn add(self, other) {
        return Point(self.x + other.x, self.y + other.y);
    }
    fn scale(self, factor) {
        return Point(self.x * factor, self.y * factor);
    }
    fn dot(self, other) {
        return self.x * other.x + self.y * other.y;
    }
}

let p = Point(0, 0);
let i = 0;
while i < 1000 {
    p = p.add(Point(1, 2));
    p = p.scale(1);
    i = i + 1;
}
print(p.dot(Point(1, 0)));
"""

# Equivalent script that stores the bound method in a local before calling —
# this follows the BoundMethod allocation path that the AttrCall optimization
# bypasses.  Each iteration allocates two temporary BoundMethod objects that
# become garbage and must be collected by GC.
BOUND_METHOD_SCRIPT = b"""
struct Point {
    x,
    y,
    fn add(self, other) {
        return Point(self.x + other.x, self.y + other.y);
    }
    fn scale(self, factor) {
        return Point(self.x * factor, self.y * factor);
    }
    fn dot(self, other) {
        return self.x * other.x + self.y * other.y;
    }
}

let p = Point(0, 0);
let i = 0;
while i < 1000 {
    let add_fn   = p.add;
    p = add_fn(Point(1, 2));
    let scale_fn = p.scale;
    p = scale_fn(1);
    i = i + 1;
}
let dot_fn = p.dot;
print(dot_fn(Point(1, 0)));
"""

# ── helpers ────────────────────────────────────────────────────────────────────

GC_STATS_RE = re.compile(
    r"\[gc_stats\] used_before=(\d+) used_after=(\d+) freed=(\d+)"
)

REPEATS = 5


def bench(label: str, script: bytes, repeats: int = REPEATS):
    """Write *script* to a temp file, run with --gc, return timing & gc stats."""
    times = []
    before_list, after_list, freed_list = [], [], []

    with tempfile.NamedTemporaryFile(suffix=".shm", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            proc = subprocess.run(
                [exe, "--gc", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)

            m = GC_STATS_RE.search(proc.stderr.decode())
            if m:
                before_list.append(int(m.group(1)))
                after_list.append(int(m.group(2)))
                freed_list.append(int(m.group(3)))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    avg_ms   = (sum(times) / repeats) * 1000
    min_ms   = min(times) * 1000
    before   = before_list[0]  if before_list else "n/a"
    after    = after_list[0]   if after_list  else "n/a"
    freed    = freed_list[0]   if freed_list  else "n/a"

    print(f"  {label}")
    print(f"    Time  : avg={avg_ms:.1f} ms  min={min_ms:.1f} ms  (n={repeats})")
    print(f"    Memory: used_before_gc={before} words  "
          f"used_after_gc={after} words  freed={freed} words")
    return avg_ms, before


def run_gc_script(script_path: str):
    """Run an on-disk .shm file with --gc and parse its [gc_stats] line."""
    proc = subprocess.run(
        [exe, "--gc", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    m = GC_STATS_RE.search(proc.stderr.decode())
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


# ── main ───────────────────────────────────────────────────────────────────────

print("=" * 64)
print("Shimlang method-call benchmark (AttrCall optimization)")
print("=" * 64)
print()

print("[ Timing benchmark — 1 000 method call iterations ]")
print()

t_opt,  mem_opt  = bench("AttrCall path  (obj.method(args) — optimised)",  METHOD_CALL_SCRIPT)
print()
t_bm,   mem_bm   = bench("BoundMethod path (let f = obj.method; f(args))", BOUND_METHOD_SCRIPT)
print()

speedup = t_bm / t_opt if t_opt > 0 else float("inf")
if isinstance(mem_bm, int) and isinstance(mem_opt, int):
    mem_savings = mem_bm - mem_opt
    print(f"  Speedup         : {speedup:.2f}x  (AttrCall vs BoundMethod path)")
    print(f"  Memory savings  : {mem_savings} fewer words in use before GC")
else:
    print(f"  Speedup         : {speedup:.2f}x  (AttrCall vs BoundMethod path)")
print()

print("[ Per-test GC memory stats ]")
print()
gc_tests = sorted(Path("test_scripts/07_gc").glob("*.shm"))
for shm in gc_tests:
    before, after, freed = run_gc_script(str(shm))
    if before is not None:
        print(f"  {shm.stem:<30}  before={before:>6}  after={after:>6}  freed={freed:>6}")

print()
print("Done.")
