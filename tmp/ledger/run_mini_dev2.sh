#!/bin/bash
# Round 63 C-route batch A2: re-runs with junction -m for ../../data defaults
cd /c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir || exit 1
LOGS=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/logs
EC=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/mini_exit_codes_dev.txt
EXE=../../../target/release/examples
r() { local tag=$1 t=$2 exe=$3; shift 3
  timeout "$t" "$EXE/$exe.exe" "$@" > "$LOGS/mini_$tag.log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then echo "$rc $tag (TIMEOUT ${t}s)" >> "$EC"; else echo "$rc $tag" >> "$EC"; fi
}
r solenoidal_m   180 mini_nurbs_solenoidal -m data/square-nurbs.mesh
r reflector_nurbs 60 mesh_reflector -m data/pipe-nurbs.mesh
r volta_m1       300 miniapp_volta -maxit 1
echo "A2 DONE" >> "$EC"
tail -4 "$EC"