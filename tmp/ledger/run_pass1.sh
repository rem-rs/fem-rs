#!/bin/bash
# Round 61 C-route: batch run all examples with default args, pass 1 (timeout 240s)
# Runs from tmp/ledger/rundir so output files (refined.mesh, sol.gf, ...) land there.
cd /c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir || exit 1
LIST=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/ex86_list.txt
LOGS=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/logs
EC=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/exit_codes_pass1.txt
: > "$EC"
while read -r name; do
  timeout 240 ../../../target/release/examples/"$name".exe > "$LOGS/$name.log" 2>&1
  rc=$?
  echo "$rc $name" >> "$EC"
done < "$LIST"
echo "PASS1 DONE" >> "$EC"
