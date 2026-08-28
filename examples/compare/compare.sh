#!/bin/bash
# MFEM 示例 1:1 比对工具 (Git Bash)
# 用法: bash compare.sh ex1 ex2 ex3
#       bash compare.sh --all

# Go to project root
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /c/Users/lilu/works/fem-pro/fem-rs)"

DATA_DIR="data"
CPP_DATA="/home/quan/mfem49/data"
OUT_DIR="tmp/cmp"
mkdir -p "$OUT_DIR"

extract_dof() {
    grep -oE "Number of (finite element )?unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || true
}

extract_ev() {
    grep -oE "(eigenvalue|lambda)[^=]*=[ ]*[-+0-9.eE]+" "$1" 2>/dev/null | grep -oE "[-+0-9.eE]+$" | head -1 || true
}

extract_iter() {
    grep -oE "(CG|MINRES|GMRES|FGMRES) converged at iter [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || true
}

extract_newton() {
    grep -oE "Number of Newton iterations = [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || true
}

extract_marked() {
    grep -oE "Marked [0-9]+ elements?" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || true
}

extract_conv_avg() {
    grep -oE "Average reduction factor = [-+0-9.eE]+" "$1" 2>/dev/null | grep -oE "[-+0-9.eE]+$" || true
}

# 示例名称 -> Rust 二进制文件名
declare -A EXE_MAP
EXE_MAP[ex1]="mfem_ex1_poisson"
EXE_MAP[ex2]="mfem_ex2_elasticity"
EXE_MAP[ex3]="mfem_ex3_maxwell_cavity"
EXE_MAP[ex4]="mfem_ex4_darcy"
EXE_MAP[ex5]="mfem_ex5_mixed_darcy"
EXE_MAP[ex6]="mfem_ex6_flux_recovery"
EXE_MAP[ex7]="mfem_ex7_surface_poisson"
EXE_MAP[ex8]="mfem_ex8_dpg_2x2"
EXE_MAP[ex9]="mfem_ex9_dg_advection"
EXE_MAP[ex10]="mfem_ex10_hyperelastic_dyn"
EXE_MAP[ex11]="mfem_ex11_eigenvalue"
EXE_MAP[ex12]="mfem_ex12_elastic_eigen"
EXE_MAP[ex13]="mfem_ex13_eigenvalue"
EXE_MAP[ex14]="mfem_ex14_dg_poisson"
EXE_MAP[ex15]="mfem_ex15_dynamic_amr"
EXE_MAP[ex16]="mfem_ex16_nonlinear_heat"
EXE_MAP[ex17]="mfem_ex17_dg_elasticity"
EXE_MAP[ex18]="mfem_ex18_euler"
EXE_MAP[ex19]="mfem_ex19_hyperelastic_incomp"
EXE_MAP[ex20]="mfem_ex20_symplectic"
EXE_MAP[ex21]="mfem_ex21_amr_elasticity"
EXE_MAP[ex22]="mfem_ex22_complex_helmholtz"
EXE_MAP[ex23]="mfem_ex23_wave_equation"
EXE_MAP[ex24]="mfem_ex24_discrete_ops"
EXE_MAP[ex25]="mfem_ex25_pml_maxwell"
EXE_MAP[ex26]="mfem_ex26_geom_mg"
EXE_MAP[ex27]="mfem_ex27_robin_bc"
EXE_MAP[ex28]="mfem_ex28_sliding_elasticity"
EXE_MAP[ex29]="mfem_ex29_curved_poisson"
EXE_MAP[ex30]="mfem_ex30_aniso_amr"
EXE_MAP[ex31]="mfem_ex31_anisotropic_maxwell"
EXE_MAP[ex32]="mfem_ex32_maxwell_eigenvalue"
EXE_MAP[ex33]="mfem_ex33_fractional_diffusion"
EXE_MAP[ex34]="mfem_ex34_magnetostatics"
EXE_MAP[ex35]="mfem_ex35_complex_oscillator"
EXE_MAP[ex36]="mfem_ex36_obstacle"
EXE_MAP[ex37]="mfem_ex37_topology_optimization"
EXE_MAP[ex38]="mfem_ex38_implicit_integration"
EXE_MAP[ex39]="mfem_ex39_compass"
EXE_MAP[ex40]="mfem_ex40_eikonal"
EXE_MAP[ex41]="mfem_ex41_imex"

run_one() {
    local name="$1" mesh="$2" ra="$3" ca="$4" modes="$5"
    local rout="$OUT_DIR/${name}_rust.log"
    local cout="$OUT_DIR/${name}_cpp.log"
    local exe_name="${EXE_MAP[$name]:-$name}"

    echo "=== $name ==="

    local rc=0
    # Git Bash can run .exe directly
    "target/release/examples/${exe_name}.exe" -m "data/${mesh}" $ra > "$rout" 2>&1 || rc=$?
    if [ $rc -ne 0 ] || grep -q "panic" "$rout"; then
        echo "  FAIL (rust exit=$rc)"
        return
    fi

    wsl -e bash -c "timeout 150 ~/bin/${name}_cpp -m ${CPP_DATA}/${mesh} ${ca}" > "$cout" 2>&1

    local rd=$(extract_dof "$rout")
    local cd=$(extract_dof "$cout")

    if [ -z "$rd" ] && [ -z "$cd" ]; then
        echo "  NO_DOF"
    elif [ -z "$rd" ]; then
        echo "  NO_RUST_DOF"
    elif [ -z "$cd" ]; then
        echo "  NO_CPP_DOF"
    elif [ "$rd" = "$cd" ]; then
        echo "  OK (dof=$rd)"
    else
        echo "  DIFF (rust=$rd cpp=$cd)"
    fi

    for mode in $(echo "$modes" | tr '+' ' '); do
        case "$mode" in
            eigenvalue) a=$(extract_ev "$rout"); b=$(extract_ev "$cout"); [ -n "$a" ] && [ -n "$b" ] && [ "$a" = "$b" ] && echo "  eigenvalue: =$a" || echo "  eigenvalue: rust=$a cpp=$b" ;;
            iter)       a=$(extract_iter "$rout"); b=$(extract_iter "$cout"); [ -n "$a" ] && [ -n "$b" ] && echo "  iter: rust=$a cpp=$b" ;;
            newton)     a=$(extract_newton "$rout"); b=$(extract_newton "$cout"); [ -n "$a" ] && [ -n "$b" ] && [ "$a" = "$b" ] && echo "  newton: =$a" || echo "  newton: rust=$a cpp=$b" ;;
            marked)     a=$(extract_marked "$rout"); b=$(extract_marked "$cout"); [ -n "$a" ] && [ -n "$b" ] && [ "$a" = "$b" ] && echo "  marked: =$a" || echo "  marked: rust=$a cpp=$b" ;;
            conv_avg)   a=$(extract_conv_avg "$rout"); b=$(extract_conv_avg "$cout"); [ -n "$a" ] && [ -n "$b" ] && echo "  conv_avg: rust=$a cpp=$b" ;;
        esac
    done
}

declare -A MESH_ARGS
MESH_ARGS[ex1]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex2]="beam-tri.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex3]="beam-tet.mesh|-no-vis|-no-vis|dof+cg"
MESH_ARGS[ex4]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex5]="star.mesh|-no-vis|-no-vis|dof+minres"
MESH_ARGS[ex6]="square-disc.mesh|-o 1 -no-vis|-o 1 -no-vis|dof"
MESH_ARGS[ex7]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex8]="star.mesh|-no-vis|-no-vis|dof+conv_avg"
MESH_ARGS[ex9]="star.mesh|-no-vis|-no-vis|dof"
MESH_ARGS[ex10]="beam-quad.mesh|-r 2 -o 2 -dt 3 -no-vis|-r 2 -o 2 -dt 3 -no-vis|dof+newton"
MESH_ARGS[ex11]="star.mesh|-no-vis|-rs 0|eigenvalue"
MESH_ARGS[ex12]="beam-tri.mesh|-n 5 -no-vis|-rs 0 -n 5|eigenvalue"
MESH_ARGS[ex13]="beam-tet.mesh|--ame -no-vis|-rs 0 -rp 0|eigenvalue"
MESH_ARGS[ex14]="star.mesh|-r 4 -o 2 -no-vis|-r 4 -o 2 -no-vis|dof+iter"
MESH_ARGS[ex15]="star.mesh|-no-vis|-no-vis|dof+marked"
MESH_ARGS[ex16]="star.mesh|-r 2 -o 2 -no-vis|-r 2 -o 2 -no-vis|dof+iter"
MESH_ARGS[ex17]="beam-tri.mesh|-no-vis|-no-vis|dof"
MESH_ARGS[ex18]="periodic-square.mesh|-no-vis|-no-vis|dof"
MESH_ARGS[ex19]="beam-quad.mesh|-o 2 -r 0 -no-vis|-o 2 -r 0 -no-vis|dof+newton"
MESH_ARGS[ex20]="star.mesh|-no-vis|-no-vis|dof+energy"
MESH_ARGS[ex21]="beam-tri.mesh|-o 2 -no-vis|-o 2 -no-vis|dof"
MESH_ARGS[ex22]="inline-quad.mesh|-p 0 -no-vis|-p 0 -no-vis|dof+iter"
MESH_ARGS[ex23]="star.mesh|-o 4 -tf 2 -no-vis|-o 4 -tf 2 -no-vis|dof"
MESH_ARGS[ex24]="star.mesh|-p 2 -o 2 -no-vis|-p 2 -o 2 -no-vis|dof"
MESH_ARGS[ex25]="inline-quad.mesh|-o 2 -f 5.0 -ref 3 -prob 4 -no-vis|-o 2 -f 5.0 -ref 3 -prob 4 -no-vis|dof"
MESH_ARGS[ex26]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex27]="inline-quad.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex28]="inline-quad.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex29]="disc-nurbs.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex30]="star.mesh|-o 1 -no-vis|-o 1 -no-vis|dof+marked"
MESH_ARGS[ex31]="beam-tri.mesh|-o 1 -r 1 -no-vis|-o 1 -r 1 -no-vis|dof+iter"
MESH_ARGS[ex32]="fichera.mesh|-no-vis|-rs 0|eigenvalue"
MESH_ARGS[ex33]="square-disc.mesh|-alpha 0.33 -o 2 -no-vis|-alpha 0.33 -o 2 -no-vis|dof+iter"
MESH_ARGS[ex34]="fichera-mixed.mesh|-no-vis|-no-vis|dof+cg"
MESH_ARGS[ex35]="fichera-mixed.mesh|-p 0 -o 1 -no-vis|-p 0 -o 1 -no-vis -rs 0|dof+fgmres"
MESH_ARGS[ex36]="disc-nurbs.mesh|-no-vis|-no-vis|dof+newton"
MESH_ARGS[ex37]="star.mesh|-no-vis|-no-vis|dof+objective"
MESH_ARGS[ex38]="inline-segment.mesh|-no-vis|-no-vis|dof"
MESH_ARGS[ex39]="compass.msh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex40]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex41]="periodic-square.mesh|-p 0 -r 2 -o 3 -no-vis|-p 0 -r 2 -o 3 -no-vis|dof+iter"

if [ "$1" = "--all" ]; then
    for name in $(echo "${!MESH_ARGS[@]}" | tr ' ' '\n' | sort -V); do
        IFS='|' read -r mesh ra ca modes <<< "${MESH_ARGS[$name]}"
        run_one "$name" "$mesh" "$ra" "$ca" "$modes" || true
    done
else
    for name in "$@"; do
        if [ -n "${MESH_ARGS[$name]}" ]; then
            IFS='|' read -r mesh ra ca modes <<< "${MESH_ARGS[$name]}"
            run_one "$name" "$mesh" "$ra" "$ca" "$modes" || true
        else
            echo "$name: 未配置"
        fi
    done
fi
