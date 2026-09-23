#!/bin/bash
# Round 63 C-route: WSL C++ references round 3 (fixed filenames + common lib)
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
M=/home/quan/mfem410_ser
B=/home/quan/work/r63bit
cd "$B" || exit 1
g++ -std=c++17 -O2 -I"$M" -o printfunc "$M/miniapps/nurbs/nurbs_printfunc.cpp" -L"$M" -lmfem 2>pf.err
echo "pf_compile=$?"; ./printfunc > "$W/cpp_printfunc.out" 2>&1; echo "pf_run=$?"
g++ -std=c++17 -O2 -I"$M" -I"$M/miniapps/common" -o meshquality "$M/miniapps/meshing/mesh-quality.cpp" -L"$M/miniapps/common" -lmfem-common -L"$M" -lmfem 2>mq.err
echo "mq_compile=$?"; ./meshquality -m "$M/data/inline-quad.mesh" -size -aspr -skew > "$W/cpp_mesh_quality.out" 2>&1; echo "mq_run=$?"
g++ -std=c++17 -O2 -I"$M" -I"$M/miniapps/common" -o curveint "$M/miniapps/nurbs/nurbs_curveint.cpp" -L"$M/miniapps/common" -lmfem-common -L"$M" -lmfem 2>ci.err
echo "ci_compile=$?"; ./curveint -uw -n 9 > "$W/cpp_curveint_uw9.out" 2>&1; echo "ci_run=$?"
g++ -std=c++17 -O2 -I"$M" -I"$M/miniapps/common" -o fieldinterp "$M/miniapps/gslib/field-interp.cpp" -L"$M/miniapps/common" -lmfem-common -L"$M" -lmfem 2>fi.err
echo "fi_compile=$?"
cp -f "$M/miniapps/gslib/triple-pt-1.mesh" "$M/miniapps/gslib/triple-pt-1.gf" "$M/miniapps/gslib/triple-pt-2.mesh" "$M/miniapps/gslib/triple-pt-2.gf" .
./fieldinterp -m2 "$M/data/star.mesh" -no-vis > "$W/cpp_field_interp_star.out" 2>&1; echo "fi_run=$?"
[ -f interpolated.gf ] && cp interpolated.gf "$W/cpp_interpolated.gf" && rm -f interpolated.gf
g++ -std=c++17 -O2 -I"$M" -I"$M/miniapps/common" -o fielddiff "$M/miniapps/gslib/field-diff.cpp" -L"$M/miniapps/common" -lmfem-common -L"$M" -lmfem 2>fd.err
echo "fd_compile=$?"; ./fielddiff -no-vis > "$W/cpp_field_diff2.out" 2>&1; echo "fd_run=$?"
echo SESSION3_DONE