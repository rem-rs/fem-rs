#!/bin/bash
set -x
W=/mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ref
E=/home/quan/mfem410_ser/miniapps/electromagnetics
cd "$E" || exit 1
timeout 500 /home/quan/work/maxwell_ser -m /home/quan/mfem410_ser/data/beam-hex.mesh -no-vis > "$W/cpp_maxwell_beamhex.out" 2>&1
echo "maxwell_beamhex=$?"
tail -4 "$W/cpp_maxwell_beamhex.out"