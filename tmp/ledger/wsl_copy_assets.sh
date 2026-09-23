#!/bin/bash
# Round 63 C-route: copy asset files from WSL MFEM tree into rundir
set -x
cd /mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir || exit 1
S=/home/quan/mfem410_ser/data
for f in ref-cube.mesh beam-tet.vtk australia.pgm triple-pt-1.mesh triple-pt-2.mesh; do
  if [ -f "$S/$f" ]; then cp "$S/$f" .; else echo "MISS-IN-MFEM $f"; fi
done
ls "$HOME/mfem410_ser/miniapps/gslib/" | grep -i triple
cp "$HOME/mfem410_ser/miniapps/gslib/triple-pt-1.gf" . 2>/dev/null || echo "no gf1 in miniapps/gslib"
cp "$HOME/mfem410_ser/miniapps/gslib/triple-pt-2.gf" . 2>/dev/null || echo "no gf2 in miniapps/gslib"
# VisIt DC sample roots from historical work dir
for n in Example5 Example3; do
  if [ -f "$HOME/work/${n}_000000.mfem_root" ]; then cp "$HOME/work/${n}_000000.mfem_root" .; else echo "MISS root $n"; fi
  if [ -d "$HOME/work/${n}_000000" ]; then cp -r "$HOME/work/${n}_000000" .; else echo "MISS dir $n"; fi
done
echo ==== rundir assets ====
ls | grep -E "triple|australia|ref-cube|beam-tet|Example"