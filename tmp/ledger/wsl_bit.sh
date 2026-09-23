#!/bin/bash
# Round 63 C-route batch B (C++ side in WSL): reference runs + fresh compiles.
# Outputs -> /mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref/cpp_<tag>.out (and meshes next to them)
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
R=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir
M=/home/quan/mfem410_ser
mkdir -p "$W"
WORK=$HOME/work
# /tmp/data symlink so binaries with ../data/<mesh> defaults resolve
ln -sfn "$M/data" /tmp/data 2>/dev/null || { mkdir -p /tmp/data; }
mkdir -p /tmp/r63bit && cd /tmp/r63bit

run() { local tag=$1; shift; "$@" > "$W/cpp_$tag.out" 2>&1; echo "rc=$? $tag"; }

# 1) nurbs_ex1 / ex3 (historical binaries)
run nurbs_ex1 "$WORK/nurbs_ex1" -m "$M/data/beam-hex-nurbs.mesh" -no-vis
run nurbs_ex3 "$WORK/nurbs_ex3" -m "$M/data/beam-hex-nurbs.mesh" -no-vis
# 2) nurbs_ex24 (historical binary)
run nurbs_ex24 "$WORK/nurbs_ex24_ser/nex24" -r 1 -p 0 -no-vis
# 3) printfunc fresh compile
g++ -std=c++17 -O2 -I"$M" -o printfunc "$M/miniapps/nurbs/printfunc.cpp" -L"$M" -lmfem && run printfunc ./printfunc
# 4) mesh-quality fresh compile
g++ -std=c++17 -O2 -I"$M" -o meshquality "$M/miniapps/meshing/mesh-quality.cpp" -L"$M" -lmfem && \
  run mesh_quality "$PWD/meshquality" -m "$M/data/inline-quad.mesh" -size -aspr -skew
# 5) hpref historical binary
run hpref "$WORK/hpref_cpp" -m "$M/data/inline-quad.mesh" -n 100 -no-vis
# 6) phpref serial harness fresh compile
g++ -std=c++17 -O2 -I"$M" -o phpref_ser "$WORK/phpref_serial.cpp" -L"$M" -lmfem && run phpref "$PWD/phpref_ser" -n 100 -no-vis
# 7) ref321 fresh compile
g++ -std=c++17 -O2 -I"$M" -o ref321 "$M/miniapps/meshing/ref321.cpp" -L"$M" -lmfem && \
  run ref321 "$PWD/ref321" -mm -dim 2 -r 100 -no-vis
# 8) extruder historical binary
run extruder "$WORK/extruder" -m "$M/data/inline-quad.mesh" -nz 4 -hz 2.0
[ -f extruded.mesh ] && cp extruded.mesh "$W/cpp_extruded.mesh" && rm -f extruded.mesh
# 9) toroid historical binary (-o 1)
run toroid_o1 "$WORK/toroid_cpp" -o 1
ls toroid-* 2>/dev/null && cp toroid-*.mesh "$W/" && rm -f toroid-*.mesh
# 10) nurbs_curveint fresh compile
g++ -std=c++17 -O2 -I"$M" -o curveint "$M/miniapps/nurbs/curveint.cpp" -L"$M" -lmfem && run curveint_uw9 "$PWD/curveint" -uw -n 9
# 11) nurbs_surface fresh compile
g++ -std=c++17 -O2 -I"$M" -o nurbs_surface "$M/miniapps/nurbs/nurbs_surface.cpp" -L"$M" -lmfem && run nurbs_surface_e1 "$PWD/nurbs_surface" -e 1 -no-vis
# 12) field-interp fresh compile (needs gslib dir on include path)
g++ -std=c++17 -O2 -I"$M" -o fieldinterp "$M/miniapps/gslib/field-interp.cpp" -L"$M" -lmfem && \
  run field_interp_star "$PWD/fieldinterp" -m2 "$M/data/star.mesh" -no-vis
[ -f interpolated.gf ] && cp interpolated.gf "$W/cpp_interpolated.gf" && rm -f interpolated.gf
# 13) field-diff fresh compile (triple-pt fixtures from miniapps/gslib)
g++ -std=c++17 -O2 -I"$M" -o fielddiff "$M/miniapps/gslib/field-diff.cpp" -L"$M" -lmfem && \
  cp "$M/miniapps/gslib/triple-pt-1.mesh" "$M/miniapps/gslib/triple-pt-1.gf" \
     "$M/miniapps/gslib/triple-pt-2.mesh" "$M/miniapps/gslib/triple-pt-2.gf" . 2>/dev/null
run field_diff "$PWD/fielddiff" -no-vis
# 14) maxwell serial (historical binary if runnable)
run maxwell "$WORK/maxwell_ser" -no-vis
# 15) abs-l1-jacobi fresh compile (serial)
g++ -std=c++17 -O2 -I"$M" -o absl1 "$M/miniapps/diag-smoothers/abs-l1-jacobi.cpp" -L"$M" -lmfem && run absl1 "$PWD/absl1"
# 16) twist fresh compile (default gear = periodic o3)
g++ -std=c++17 -O2 -I"$M" -o twist "$M/miniapps/meshing/twist.cpp" -L"$M" -lmfem && run twist_def "$PWD/twist"
[ -f twist-hex-o3-s2-p.mesh ] && cp twist-hex-o3-s2-p.mesh "$W/cpp_twist-hex-o3-s2-p.mesh" && rm -f twist-hex-o3-s2-p.mesh
run twist_o1 "$PWD/twist" -o 1 -no-pm
[ -f twist-hex-o1-s2-c.mesh ] && cp twist-hex-o1-s2-c.mesh "$W/cpp_twist-hex-o1-s2-c.mesh" && rm -f twist-hex-o1-s2-c.mesh
# 17) reflector fresh compile (fichera gear)
g++ -std=c++17 -O2 -I"$M" -o reflector "$M/miniapps/meshing/reflector.cpp" -L"$M" -lmfem && \
  run reflector_fich "$PWD/reflector" -m "$M/data/fichera.mesh" -o "1 0 0" -n "1 0 0"
[ -f reflected.mesh ] && cp reflected.mesh "$W/cpp_reflected.mesh" && rm -f reflected.mesh
echo WSL_BIT_DONE
ls "$W" | grep cpp_ | head -40