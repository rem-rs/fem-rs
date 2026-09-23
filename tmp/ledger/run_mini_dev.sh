#!/bin/bash
# Round 63 C-route batch A: DEV (exit(3)-declared) confirmations + declared-behavior checks
# Runs from tmp/ledger/rundir; logs -> tmp/ledger/logs/mini_<tag>.log; rc -> mini_exit_codes_dev.txt
cd /c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir || exit 1
LOGS=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/logs
EC=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/mini_exit_codes_dev.txt
EXE=../../../target/release/examples
: > "$EC"
r() { # tag timeout exe args...
  local tag=$1 t=$2 exe=$3; shift 3
  timeout "$t" "$EXE/$exe.exe" "$@" > "$LOGS/mini_$tag.log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then echo "$rc $tag (TIMEOUT ${t}s)" >> "$EC"; else echo "$rc $tag" >> "$EC"; fi
}

r tesla            60  miniapp_tesla
r lorentz          60  miniapp_lorentz
r solenoidal       180 mini_nurbs_solenoidal
r tmop_check       60  tmop_check_metric
r tmop_mag_mid7    60  tmop_metric_magnitude -mid 7 -pv 2.0 -par 0.5 -ps 4.0
r tmop_mag_noarg   60  tmop_metric_magnitude
r lissajous        60  toys_lissajous -no-vis
r mandel           240 toys_mandel -m data/inline-quad.mesh -no-vis
r mondrian         240 toys_mondrian -i australia.pgm -m data/inline-quad.mesh
r polar_nc         60  mesh_polar_nc
r twist_default    60  mesh_twist
r twist_o1_nopm    60  mesh_twist -o 1 -no-pm
r trimmer_default  60  mesh_trimmer
r trimmer_vtk      120 mesh_trimmer -m beam-tet.vtk
r joule            300 miniapp_joule
r navier_cht       300 navier_cht
r pconvdiff        120 pconvection_diffusion
r reflector_def    60  mesh_reflector
r reflector_fich   60  mesh_reflector -m data/fichera.mesh -o "1 0 0" -n "1 0 0"
r volta_def        300 miniapp_volta
r volta_r2         60  miniapp_volta --ranks 2
r nurbs_ex10       60  nurbs_ex10
r nurbs_surface    60  mini_nurbs_surface
r shaper           300 mesh_shaper -m data/inline-quad.mesh
r pacoustics_p2    120 pacoustics -prob 2
r lorstub_fe_l     120 miniapp_lor_solvers -fe l -m data/inline-quad.mesh
echo "DEV BATCH DONE" >> "$EC"
cat "$EC"