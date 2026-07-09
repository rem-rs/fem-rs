#!/bin/bash
OUT=/tmp/mfem_cpp_results
echo "{" > $OUT/results.json
first=true
for f in mfem_ex0 mfem_ex1 mfem_ex2 mfem_ex3 mfem_ex4 mfem_ex5 mfem_ex6 mfem_ex10 mfem_ex22 mfem_ex25 mfem_ex31 mfem_ex34; do
  fout=$OUT/$f.out
  [ -f "$fout" ] || continue
  dof=$(grep -oP '(Number of finite element unknowns|Number of unknowns)[^0-9]*\K[0-9]+' "$fout" | tail -1)
  [ -z "$dof" ] && dof=$(grep -oP 'Number of.*[Dd]of[^0-9]*\K[0-9]+' "$fout" | tail -1)
  sys=$(grep -oP 'Size of linear system: \K[0-9]+' "$fout" | tail -1)
  l2=$(grep -oP '\|\|.*\|\|_\{L\^2\} = \K[0-9.]+[eE]?[+-]?[0-9]*' "$fout" | tail -1)
  [ -z "$l2" ] && l2=$(grep -oP 'L2 error: \K[0-9.]+[eE]?[+-]?[0-9]*' "$fout" | tail -1)
  it=$(grep -oP '(PCG|CG|GMRES)[^0-9]*\K[0-9]+' "$fout" | tail -1)
  eig=$(grep -oP 'Eigenmode [0-9]+:.*lambda = \K[0-9.]+[eE]?[+-]?[0-9]*' "$fout" | head -5 | paste -sd,)

  $first || echo "," >> $OUT/results.json
  first=false
  cat >> $OUT/results.json <<INNER
  "$f": {
    "dofs": ${dof:-null},
    "system_size": ${sys:-null},
    "l2_error": ${l2:-null},
    "iterations": ${it:-null},
    "eigenvalues": [${eig:-}]
  }
INNER
done
echo "}" >> $OUT/results.json
echo "--- results ---"
cat $OUT/results.json
