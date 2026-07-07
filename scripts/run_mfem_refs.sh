#!/bin/bash
set -euo pipefail

# MFEM_DIR = directory with compiled MFEM examples (examples/ex${ex} binaries)
MFEM_DIR="${MFEM_DIR:-/tmp/mfem}"
MESH="${MESH:-${MFEM_DIR}/../mfem_bench/tri32_v2.mesh}"

for ex in 1 2 3 4 5 9; do
  echo "=== MFEM ex${ex} ==="
  "${MFEM_DIR}/examples/ex${ex}" -m "$MESH" -o 1 --no-visualization 2>&1 | grep -E "Number of|unknown|norm|reduction|Iteration" | tail -3
  echo ""
done
