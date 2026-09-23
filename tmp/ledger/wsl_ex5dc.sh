#!/bin/bash
set -x
M=/home/quan/mfem410_ser
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir
B=/home/quan/work/r63bit
cd "$B" || exit 1
g++ -std=c++17 -O2 -I"$M" -o ex5 "$M/examples/ex5.cpp" -L"$M" -lmfem 2>ex5.err
echo "ex5_compile=$?"
mkdir -p dcgen && cd dcgen
timeout 120 ../ex5 -m "$M/data/star.mesh" -no-vis > ex5_run.log 2>&1
echo "ex5_run=$?"
ls | head
# ex5 saves Example5 DC by default when not visualizing? check files
[ -f Example5_000000.mfem_root ] || { echo "no DC from ex5"; tail -5 ex5_run.log; }