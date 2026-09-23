#!/bin/bash
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
N=/home/quan/mfem410_ser/miniapps/nurbs
cd "$N" || exit 1
/home/quan/work/nurbs_ex24_ser/nex24 -r 1 -p 0 -no-vis > "$W/cpp_nurbs_ex24.out" 2>&1
echo "ex24=$?"
/home/quan/work/nurbs_ex3 -no-vis > "$W/cpp_nurbs_ex3.out" 2>&1
echo "ex3=$?"
tail -5 "$W/cpp_nurbs_ex24.out"
tail -6 "$W/cpp_nurbs_ex3.out"