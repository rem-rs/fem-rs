#!/bin/bash
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
M=/home/quan/mfem410_ser
B=/home/quan/work/r63bit
cd "$B" || exit 1
g++ -std=c++17 -O2 -I"$M" -o hpref410 "$M/miniapps/meshing/hpref.cpp" -L"$M" -lmfem 2>hp.err
echo "hp_compile=$?"
./hpref410 -m "$M/data/inline-quad.mesh" -pref -n 100 -no-vis > "$W/cpp_hpref_pref.out" 2>&1
echo "hp_pref_run=$?"
./hpref410 -m "$M/data/inline-quad.mesh" -n 100 -no-vis > "$W/cpp_hpref_hp.out" 2>&1
echo "hp_hp_run=$?"
g++ -std=c++17 -O2 -I"$M" -o phpref410 "$M/miniapps/meshing/phpref.cpp" -L"$M" -lmfem 2>pp.err
echo "pp_compile=$? (expect serial link ok?)"
tail -12 "$W/cpp_phpref.out"