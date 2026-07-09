#!/usr/bin/env bash
# Run all MFEM C++ serial examples and extract key data for Rust comparison
# Usage: bash run_mfem_cpp.sh <mfem_root> <output_dir>

set -euo pipefail

MFEM_DIR="${1:-/tmp/mfem}"
OUT_DIR="${2:-/tmp/mfem_cpp_results}"
mkdir -p "$OUT_DIR"
cd "$MFEM_DIR/examples"

RESULTS="$OUT_DIR/results.json"
echo '{' > "$RESULTS"
first=true

run_one() {
    local ex=$1       # e.g. "ex3"
    local label=$2    # e.g. "mfem_ex3"
    shift 2
    local args=("$@")

    echo "  Running $label ..." >&2

    # Build if needed
    make "$ex" 2>/dev/null || { echo "    BUILD FAILED" >&2; return 1; }

    # Run with timeout, capture stdout+stderr
    local outfile="$OUT_DIR/${label}.out"
    if ! timeout 300 ./"$ex" "${args[@]}" > "$outfile" 2>&1; then
        echo "    RUN FAILED" >&2
        return 1
    fi

    # Extract key metrics
    local n_dofs=$(grep -Eo '[0-9]+ unknowns' "$outfile" | grep -Eo '[0-9]+' | tail -1)
    local n_dofs2=$(grep -Eo 'Number of.*[kK]nowns: [0-9]+' "$outfile" | grep -Eo '[0-9]+' | tail -1)
    local n_sys=$(grep -Eo 'Size of linear system: [0-9]+' "$outfile" | grep -Eo '[0-9]+' | tail -1)
    local n_sys2=$(grep -Eo 'Number of.*[tT]ype [0-9]+' "$outfile" | grep -Eo '[0-9]+' | tail -1)
    local l2_err=$(grep -Eo '\|\|.*\|\|_\{L\^2\} = [0-9.e+-]+' "$outfile" | grep -Eo '[0-9.e+-]+$' | tail -1)
    local l2_err2=$(grep -Eo 'L2 error: [0-9.e+-]+' "$outfile" | grep -Eo '[0-9.e+-]+' | tail -1)
    local iters=$(grep -Eo '[0-9]+ iteration' "$outfile" | grep -Eo '[0-9]+' | tail -1)
    local iters2=$(grep -Eo '[0-9]+ iters' "$outfile" | grep -Eo '[0-9]+' | tail -1)
    local residual=$(grep -Eo '\|\|r\|\|/\|\|b\|\| = [0-9.e+-]+' "$outfile" | grep -Eo '[0-9.e+-]+' | tail -1)
    local eig=$(grep -Eo 'Eigenmode [0-9]+:.*lambda = [0-9.e+-]+' "$outfile" | head -3)
    local neigs=$(echo "$eig" | grep -c "lambda" || true)

    # Write JSON
    $first || echo "," >> "$RESULTS"
    first=false
    cat >> "$RESULTS" <<EOF
  "$label": {
    "dofs": ${n_dofs:-${n_dofs2:-null}},
    "system_size": ${n_sys:-${n_sys2:-null}},
    "l2_error": ${l2_err:-${l2_err2:-null}},
    "iterations": ${iters:-${iters2:-null}},
    "final_residual": ${residual:-null},
    "neigs": ${neigs:-null},
    "eigenvalues": [$(echo "$eig" | sed "s/.*lambda = //" | tr '\n' ',' | sed 's/,$//')]
  }
EOF
    echo "    DOFs=${n_dofs:-N/A} L²=${l2_err:-N/A}" >&2
}

# ── Serial examples ─────────────────────────────────────────
# Match the fem-rs mfem_ex* naming

run_one ex0  mfem_ex0  -no-vis
run_one ex1  mfem_ex1  -no-vis
run_one ex2  mfem_ex2  -no-vis
run_one ex3  mfem_ex3  -no-vis -m ../../data/star.mesh
run_one ex3  mfem_ex3_unit  -no-vis
run_one ex4  mfem_ex4  -no-vis -m ../../data/star.mesh
run_one ex5  mfem_ex5  -no-vis
run_one ex6  mfem_ex6  -no-vis
run_one ex7  mfem_ex7  -no-vis
run_one ex8  mfem_ex8  -no-vis
run_one ex9  mfem_ex9  -no-vis
run_one ex10 mfem_ex10 -no-vis
run_one ex11 mfem_ex11 -no-vis
run_one ex14 mfem_ex14 -no-vis
run_one ex15 mfem_ex15 -no-vis
run_one ex16 mfem_ex16 -no-vis
run_one ex22 mfem_ex22 -no-vis
run_one ex25 mfem_ex25 -no-vis -m ../../data/star.mesh
run_one ex31 mfem_ex31 -no-vis
run_one ex32 mfem_ex32 -no-vis
run_one ex34 mfem_ex34 -no-vis
run_one ex41 mfem_ex41 -no-vis

# ── Default-mesh variants ────────────────────────────────────
run_one ex4  mfem_ex4_default -no-vis

echo '}' >> "$RESULTS"
echo "Done! Results: $RESULTS" >&2
