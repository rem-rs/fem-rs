#!/bin/bash
set -euo pipefail

# MFEM_DIR = MFEM source root (with libmfem built and examples/ present)
MFEM_DIR="${MFEM_DIR:-/tmp/mfem}"

cd "$MFEM_DIR"
for ex in 2 3 4 5 9 14; do
  echo "Compiling ex${ex}..."
  g++ -O3 -std=c++17 -I. "examples/ex${ex}.cpp" -o "examples/ex${ex}" -L. -lmfem -lrt
  echo "  done (exit $?)"
done
