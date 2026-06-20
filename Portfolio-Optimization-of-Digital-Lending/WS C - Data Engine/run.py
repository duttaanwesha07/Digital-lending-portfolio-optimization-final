"""
run.py — the fixed-order pipeline runner (spec §11.1).

Runs the build end to end, in the one order determinism depends on:

    customers -> loans -> engine(repayments+behaviour) -> outcomes -> assemble

Each step regenerates from the master seed, so a full run reproduces the
dataset byte-for-byte. The internal z exists during the engine/outcomes steps
and is dropped only at assemble, so the delivered tables never carry it.

Usage:  python run.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "generate"

STEPS = [
    ("Step 2  customers",  GEN / "customers.py"),
    ("Step 3  loans",      GEN / "loans.py"),
    ("Step 4  engine",     GEN / "engine.py"),
    ("Step 5  outcomes",   GEN / "outcomes.py"),
    ("Step 6  assemble",   GEN / "assemble.py"),
]


def main():
    for label, script in STEPS:
        print(f"\n{'='*70}\n>>> {label}\n{'='*70}")
        r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\n!! {label} FAILED (exit {r.returncode}); stopping.")
            sys.exit(r.returncode)
    print(f"\n{'='*70}\nPipeline complete. Delivered files in out/data/ ; "
          f"dictionary + manifest in out/meta/.\nNext: Step 8 Gate-1 harness.\n{'='*70}")


if __name__ == "__main__":
    main()
