#!/bin/bash
# Round 61 C-route pass 2: re-runs with standard args (default mesh + -no-vis) for
# examples whose pass-1 no-arg run hit missing -no-vis / ../data resolution / long runtime.
cd /c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir || exit 1
LOGS=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/logs
EC=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/exit_codes_pass2.txt
: > "$EC"
run() { # name timeout args...
  local name=$1; local tmo=$2; shift 2
  timeout "$tmo" ../../../target/release/examples/"$name".exe "$@" > "$LOGS/$name.log" 2>&1
  echo "$? $name [$*]" >> "$EC"
}
run mfem_ex2_elasticity            240 -m data/beam-tri.mesh -no-vis
run mfem_ex4_darcy                 240 -m data/star.mesh -no-vis
run mfem_ex5_mixed_darcy           240 -m data/star.mesh -no-vis
run mfem_ex8_dpg_2x2               240 -m data/star.mesh -no-vis
run mfem_ex31_anisotropic_maxwell  240 -m data/inline-quad.mesh -no-vis
run mfem_ex31_dump                 240 -m data/inline-quad.mesh -r 2 -o 1
run mfem_ex9_dg_advection          240 -m data/periodic-square.mesh -no-vis
run mfem_ex15_dynamic_amr          240 -m data/star.mesh -no-vis
run mfem_pex15_parallel_dynamic_amr 240 -m data/star.mesh -no-vis
run mfem_pex30_amr_preprocess      600 -m data/star.mesh -no-vis
run mfem_pex5_hdiv_darcy           300 -m data/star.mesh -no-vis
run mfem_pex40_eikonal             300 -m data/star.mesh -no-vis
run mfem_ex15_dump_A_true          120
run mfem_ex15_dump_T002            120
run mfem_ex15_dump_flow            120
run mfem_ex15_dump_it2_coords      120
run mfem_ex15_dump_p1              120
run mfem_ex15_dump_p1_it3          120
echo "PASS2 DONE" >> "$EC"
