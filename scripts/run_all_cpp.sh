#!/bin/bash
set -euo pipefail

BUILD_DIR="$1"
RESULTS_DIR="${2:-/tmp/mfem_cpp_results}"
DATA_DIR="$BUILD_DIR/../data"
mkdir -p "$RESULTS_DIR"

RESULTS="$RESULTS_DIR/results.json"
echo "{" > "$RESULTS"
first=true

log()  { echo "  $@" >&2; }
extract_json_val() {
    local pat="$1" out="$2"
    grep -Eo "$pat" "$out" | grep -Eo '[0-9.eE+-]+$' | tail -1
}

run_ex() {
    local ex="$1" tag="$2"
    shift 2
    local out="$RESULTS_DIR/${tag}.out"
    log "Running $tag ($ex)"
    if ! "./$ex" "$@" > "$out" 2>&1; then
        log "  FAILED"
        return
    fi

    local dofs=$(extract_json_val "(Number of finite element unknowns|Number of unknowns|Number of.*[Dd]o[fF]): [0-9]+" "$out")
    local syssz=$(extract_json_val "(Size of linear system|Number of.*[Tt]ype [0-9]+): [0-9]+" "$out")
    local l2=$(extract_json_val "\|\|.*\|\|_\{L.2\} = [0-9.eE+-]+" "$out")
    local l2b=$(extract_json_val "L2 error: [0-9.eE+-]+" "$out")
    [ -z "$l2" ] && l2="$l2b"
    local iters=$(extract_json_val "(PCG|CG|GMRES|LOBPCG).*: [0-9]+" "$out")
    local resid=$(extract_json_val "\|\|r\|\|./.b. = [0-9.eE+-]+" "$out")
    local eigenvals=$(grep -oP 'Eigenmode [0-9]+:  lambda = \K[0-9.eE+-]+' "$out" | head -5 | paste -sd,)

    $first || echo "," >> "$RESULTS"
    first=false
    cat >> "$RESULTS" <<EOS
  "$tag": {
    "dofs": ${dofs:-null},
    "system_size": ${syssz:-null},
    "l2_error": ${l2:-null},
    "iterations": ${iters:-null},
    "residual": ${resid:-null},
    "eigenvalues": [${eigenvals:-}]
  }
EOS
    log "  DOFs=${dofs:-N/A} L2=${l2:-N/A}"
}

cd "$BUILD_DIR/examples"

run_ex ex0  mfem_ex0  -no-vis
run_ex ex1  mfem_ex1  -no-vis
run_ex ex2  mfem_ex2  -no-vis
run_ex ex3  mfem_ex3  -no-vis
run_ex ex4  mfem_ex4  -m "$DATA_DIR/star.mesh" -no-vis
run_ex ex5  mfem_ex5  -no-vis
run_ex ex6  mfem_ex6  -no-vis
run_ex ex7  mfem_ex7  -no-vis
run_ex ex8  mfem_ex8  -no-vis
run_ex ex9  mfem_ex9  -no-vis
run_ex ex10 mfem_ex10 -no-vis
run_ex ex11 mfem_ex11 -no-vis
run_ex ex14 mfem_ex14 -no-vis
run_ex ex15 mfem_ex15 -no-vis
run_ex ex16 mfem_ex16 -no-vis
run_ex ex22 mfem_ex22 -no-vis
run_ex ex25 mfem_ex25 -m "$DATA_DIR/star.mesh" -no-vis
run_ex ex31 mfem_ex31 -no-vis
run_ex ex32 mfem_ex32 -no-vis
run_ex ex34 mfem_ex34 -no-vis
run_ex ex41 mfem_ex41 -no-vis

echo "}" >> "$RESULTS"
echo "=== Done! Results: $RESULTS ===" >&2
