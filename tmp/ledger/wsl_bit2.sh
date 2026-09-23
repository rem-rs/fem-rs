#!/bin/bash
# Round 63 C-route: WSL C++ references round 2 (single session, persistent home)
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
M=/home/quan/mfem410_ser
B=/home/quan/work/r63bit
mkdir -p "$B" && cd "$B"
g++ -std=c++17 -O2 -I"$M" -o printfunc "$M/miniapps/nurbs/printfunc.cpp" -L"$M" -lmfem 2>pf.err
echo "pf_compile=$?" && ./printfunc > "$W/cpp_printfunc.out" 2>&1; echo "pf_run=$?"
g++ -std=c++17 -O2 -I"$M" -o meshquality "$M/miniapps/meshing/mesh-quality.cpp" -L"$M" -lmfem 2>mq.err
echo "mq_compile=$?" && ./meshquality -m "$M/data/inline-quad.mesh" -size -aspr -skew > "$W/cpp_mesh_quality.out" 2>&1; echo "mq_run=$?"
g++ -std=c++17 -O2 -I"$M" -o curveint "$M/miniapps/nurbs/curveint.cpp" -L"$M" -lmfem 2>ci.err
echo "ci_compile=$?" && ./curveint -uw -n 9 > "$W/cpp_curveint_uw9.out" 2>&1; echo "ci_run=$?"
g++ -std=c++17 -O2 -I"$M" -o fieldinterp "$M/miniapps/gslib/field-interp.cpp" -L"$M" -lmfem 2>fi.err
echo "fi_compile=$?"
cp "$M/miniapps/gslib/triple-pt-1.mesh" "$M/miniapps/gslib/triple-pt-1.gf" "$M/miniapps/gslib/triple-pt-2.mesh" "$M/miniapps/gslib/triple-pt-2.gf" .
./fieldinterp -m2 "$M/data/star.mesh" -no-vis > "$W/cpp_field_interp_star.out" 2>&1; echo "fi_run=$?"
[ -f interpolated.gf ] && cp interpolated.gf "$W/cpp_interpolated.gf"
./fielddiff 2>/dev/null || true
g++ -std=c++17 -O2 -I"$M" -o fielddiff "$M/miniapps/gslib/field-diff.cpp" -L"$M" -lmfem 2>fd.err
echo "fd_compile=$?" && ./fielddiff -no-vis > "$W/cpp_field_diff2.out" 2>&1; echo "fd_run=$?"
echo SINGLE_SESSION_DONE