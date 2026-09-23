#!/bin/bash
# Round 63 C-route batch C: RUN deep checks (default gears / README gears).
cd /c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/rundir || exit 1
LOGS=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/logs
EC=/c/Users/lilu/works/fem-pro/fem-rs/tmp/ledger/mini_exit_codes_run.txt
EXE=../../../target/release/examples
: > "$EC"
r() { local tag=$1 t=$2 exe=$3; shift 3
  timeout "$t" "$EXE/$exe.exe" "$@" > "$LOGS/mini_$tag.log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then echo "$rc $tag (TIMEOUT ${t}s)" >> "$EC"; else echo "$rc $tag" >> "$EC"; fi
}
# tools
r display_basis     120 tools_display_basis -e 2 -b 3 -o 3
r lor_transfer_iq   120 tools_lor_transfer -m data/inline-quad.mesh -o 2 -no-vis
r get_values_ex5    60  tools_get_values -r Example5_000000.mfem_root -p "0.5 0.5 0.1 0.1" -fn pressure
r load_dc_ex5       60  miniapp_load_dc -r Example5_000000.mfem_root
r compare_dc_ex     60  miniapp_compare_dc -r Example5_000000.mfem_root -w Example3_000000.mfem_root
r nodal_transfer    120 tools_nodal_transfer
# toys / autodiff
r automata          120 toys_automata
r life_sp           120 toys_life -nx 10 -ny 10 -sp "3 3 0 1 1 1 2 1 1 1"
r autodiff          300 autodiff_example -no-vis
# multidomain / shifted / hooke / dfem
r multidomain       300 multidomain
r multidomain_nd    300 multidomain_nd
r multidomain_rt    300 multidomain_rt
r shifted_distance  120 shifted_distance
r shifted_diffusion 120 shifted_diffusion
r shifted_extrap    120 shifted_extrapolate
r hooke_def         300 hooke
r dfem_min_surf     300 dfem_minimal_surface -no-vis
# solvers
r lor_elast_def     300 lor_elast
r plor_solvers_st   300 plor_solvers -m data/star.mesh -o 3 -rs 1 -rp 1 -no-vis --ranks 1
# adjoint
r adjoint_roberts   300 adjoint_cvodes_roberts
r adjoint_advdif    300 adjoint_advection_diffusion -fd 1
# dpg serial
r dpg_helm_1d       300 dpg_helmholtz_1d
r dpg_poisson_2d    300 dpg_poisson_2d
r dpg_acoustics_2d  300 dpg_acoustics_2d
r dpg_maxwell_2d    300 dpg_maxwell_2d
r dpg_acoustics_3d  300 dpg_acoustics_3d
r dpg_maxwell_3d    300 dpg_maxwell_3d
# dpg parallel
r pdiffusion        300 pdiffusion
r pacoustics        300 pacoustics
r pmaxwell          300 pmaxwell
# fluids
r schrodinger       300 schrodinger_flow
r navier_kov        300 navier_kovasznay
r navier_mms        300 navier_mms
r navier_shear      300 navier_shear
r navier_kov_vs     300 navier_kovasznay_vs
r navier_tgv        300 navier_tgv
r navier_3dfoc      600 navier_3dfoc
r navier_turbchan   600 navier_turbchan
r navier_bifurc     600 navier_bifurcation
# hdiv
r hdiv_darcy        300 hdiv_darcy
r hdiv_grad_div     300 hdiv_grad_div
# spde
r spde_grf          300 spde_generate_random_field -m ref-cube.mesh -r 2 -rp 1 -no-vis -no-rs
# meshing leftovers
r mesh_explorer     60  mesh_explorer
r mesh_optimizer_st 300 mesh_optimizer -m data/star-mixed.mesh -no-vis
r fit_node_pos      300 mesh_fit_node_position
r mobius            120 mesh_mobius_strip
r klein             120 mesh_klein_bottle
r toroid_def        300 mesh_toroid
echo "RUN BATCH DONE" >> "$EC"
cat "$EC"