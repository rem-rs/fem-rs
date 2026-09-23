#!/bin/bash
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
M=/home/quan/mfem410_ser
B=/home/quan/work/r63bit
cd "$B" || exit 1
g++ -std=c++17 -O2 -I"$M" -o nurbssurface "$M/miniapps/nurbs/nurbs_surface.cpp" -L"$M" -lmfem 2>ns.err
echo "ns_compile=$?"
./nurbssurface -ex 1 -no-vis > "$W/cpp_nurbs_surface_ex1.out" 2>&1
echo "ns_run=$?"
wc -l "$W/cpp_nurbs_surface_ex1.out"
tail -5 "$W/cpp_nurbs_surface_ex1.out"