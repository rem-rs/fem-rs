#!/bin/bash
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
M=/home/quan/mfem410_ser
B=/home/quan/work/r63bit
cd "$B" || exit 1
g++ -std=c++17 -O2 -I"$M" -o maxwell410 "$M/miniapps/electromagnetics/maxwell.cpp" -L"$M" -lmfem 2>mw.err
echo "mw_compile=$?"
head -3 mw.err
timeout 500 ./maxwell410 -m "$M/data/beam-hex.mesh" -no-vis > "$W/cpp_maxwell410_beamhex.out" 2>&1
echo "mw_run=$?"
grep -c Energy "$W/cpp_maxwell410_beamhex.out"
tail -3 "$W/cpp_maxwell410_beamhex.out"