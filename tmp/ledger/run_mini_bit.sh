#!/bin/bash
# Round 63 C-route batch B: BIT re-runs (Rust side). C++ side runs in WSL (wsl_bit.sh).
cd /c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir || exit 1
LOGS=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/logs
EC=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/mini_exit_codes_bit.txt
EXE=../../../target/release/examples
: > "$EC"
r() { local tag=$1 t=$2 exe=$3; shift 3
  timeout "$t" "$EXE/$exe.exe" "$@" > "$LOGS/mini_$tag.log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then echo "$rc $tag (TIMEOUT ${t}s)" >> "$EC"; else echo "$rc $tag" >> "$EC"; fi
}
r nurbs_ex1_bh     300 mini_nurbs_ex1 -m data/beam-hex-nurbs.mesh -no-vis
r nurbs_ex3_bh     300 mini_nurbs_ex3 -m data/beam-hex-nurbs.mesh -no-vis
r nurbs_ex24_r1p0  300 mini_nurbs_ex24 -r 1 -p 0 -no-vis
r printfunc        60  mini_nurbs_printfunc
r curveint_uw9     120 nurbs_curveint -uw -n 9
r mesh_quality_iq  120 mesh_quality -m data/inline-quad.mesh -size -aspr -skew
r hpref_iq100      240 mesh_hpref -m data/inline-quad.mesh -pref -n 100 -no-vis
r phpref_n100      240 mesh_phpref -n 100 -no-vis
r ref321_mm        240 mesh_ref321 -mm -dim 2 -r 100 -no-vis
r extruder_iq      60  mesh_extruder -m data/inline-quad.mesh -nz 4 -hz 2.0
r toroid_o1        120 mesh_toroid -o 1
r field_interp_star 240 gslib_field_interp -m2 data/star.mesh -no-vis
r findpts_rt2d     120 gslib_findpts -m data/rt-2d-q3.mesh -o 8 -mo 4 -no-vis
r findpts_hex      120 gslib_findpts -m data/inline-hex.mesh -o 3 -random 1 -npt 4 -no-vis
r field_diff       120 gslib_field_diff -no-vis
r maxwell_novis    600 miniapp_maxwell -no-vis
r absl1_default    300 diag_abs_l1_jacobi
r gridfn_bounds    120 miniapp_gridfunction_bounds -m data/star.mesh
r nurbs_surface_e1 120 mini_nurbs_surface -e 1
echo "BIT BATCH DONE" >> "$EC"
cat "$EC"