#!/usr/bin/env python3
"""Insert `#![allow(non_snake_case)]` after the file-level doc block."""
import sys, pathlib, re

ATTR = "#![allow(non_snake_case)]\n"

def patch(path: pathlib.Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if ATTR in text:
        return False
    lines = text.splitlines(keepends=True)
    out = []
    inserted = False
    i = 0
    # Skip leading doc lines (`//!` or `///`) plus blanks
    while i < len(lines) and (lines[i].startswith("//!") or lines[i].startswith("///") or lines[i].strip() == ""):
        out.append(lines[i])
        i += 1
        if i < len(lines) and not (lines[i].startswith("//!") or lines[i].startswith("///") or lines[i].strip() == ""):
            break
    out.append(ATTR)
    out.append("\n")
    out.extend(lines[i:])
    path.write_text("".join(out), encoding="utf-8")
    return True

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        ok = patch(pathlib.Path(arg))
        print(f"{'patched' if ok else 'already done'}: {arg}")
