#!/bin/bash
# MFEM 示例 1:1 比对工具 (Git Bash)
# 用法: bash compare.sh ex1 ex2 ex3
#       bash compare.sh --all

# Go to project root
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="/c/Users/lilu/works/fem-pro/fem-rs"
fi
cd "$PROJECT_ROOT"

DATA_DIR="$PROJECT_ROOT/data"
CPP_DATA="/home/quan/mfem49/data"
OUT_DIR="$PROJECT_ROOT/tmp/cmp"
mkdir -p "$OUT_DIR"

extract_dof() {
    local val=""
    val=$(grep -oE "Number of finite element unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Number of unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Unknowns: *[0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "NDoFs: *[0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "dim\(R\+W\) = [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    echo ""
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

run_one() {
    local name="$1" mesh="$2" ra="$3" ca="$4" modes="$5"
    local rout="$OUT_DIR/${name}_rust.log"
    local cout="$OUT_DIR/${name}_cpp.log"

    echo "=== $name ==="

    local rc=0
    # Map example name to Rust binary name
    local exe_name="${name}"
    case "$name" in
        ex1) exe_name="mfem_ex1_poisson" ;;
        ex2) exe_name="mfem_ex2_elasticity" ;;
        ex3) exe_name="mfem_ex3_maxwell_cavity" ;;
        ex4) exe_name="mfem_ex4_darcy" ;;
        ex5) exe_name="mfem_ex5_mixed_darcy" ;;
        ex6) exe_name="mfem_ex6_flux_recovery" ;;
        ex7) exe_name="mfem_ex7_surface_poisson" ;;
        ex8) exe_name="mfem_ex8_dpg_2x2" ;;
        ex9) exe_name="mfem_ex9_dg_advection" ;;
        ex10) exe_name="mfem_ex10_hyperelastic_dyn" ;;
        ex11) exe_name="mfem_ex11_eigenvalue" ;;
        ex12) exe_name="mfem_ex12_elastic_eigen" ;;
        ex13) exe_name="mfem_ex13_eigenvalue" ;;
        ex14) exe_name="mfem_ex14_dg_poisson" ;;
        ex15) exe_name="mfem_ex15_dynamic_amr" ;;
        ex16) exe_name="mfem_ex16_nonlinear_heat" ;;
        ex17) exe_name="mfem_ex17_dg_elasticity" ;;
        ex18) exe_name="mfem_ex18_euler" ;;
        ex19) exe_name="mfem_ex19_hyperelastic_incomp" ;;
        ex20) exe_name="mfem_ex20_symplectic" ;;
        ex21) exe_name="mfem_ex21_amr_elasticity" ;;
        ex22) exe_name="mfem_ex22_complex_helmholtz" ;;
        ex23) exe_name="mfem_ex23_wave_equation" ;;
        ex24) exe_name="mfem_ex24_discrete_ops" ;;
        ex25) exe_name="mfem_ex25_pml_maxwell" ;;
        ex26) exe_name="mfem_ex26_geom_mg" ;;
        ex27) exe_name="mfem_ex27_robin_bc" ;;
        ex28) exe_name="mfem_ex28_sliding_elasticity" ;;
        ex29) exe_name="mfem_ex29_curved_poisson" ;;
        ex30) exe_name="mfem_ex30_aniso_amr" ;;
        ex31) exe_name="mfem_ex31_anisotropic_maxwell" ;;
        ex32) exe_name="mfem_ex32_maxwell_eigenvalue" ;;
        ex33) exe_name="mfem_ex33_fractional_diffusion" ;;
        ex34) exe_name="mfem_ex34_magnetostatics" ;;
        ex35) exe_name="mfem_ex35_complex_oscillator" ;;
        ex36) exe_name="mfem_ex36_obstacle" ;;
        ex37) exe_name="mfem_ex37_topology_optimization" ;;
        ex38) exe_name="mfem_ex38_implicit_integration" ;;
        ex39) exe_name="mfem_ex39_compass" ;;
        ex40) exe_name="mfem_ex40_eikonal" ;;
        ex41) exe_name="mfem_ex41_imex" ;;
    esac
    # Git Bash can run .exe directly
    "target/release/examples/${exe_name}.exe" -m "data/${mesh}" $ra > "$rout" 2>&1 || rc=$?
    if [ $rc -ne 0 ] || grep -q "panic" "$rout"; then
        echo "  FAIL (rust exit=$rc)"
        return
    fi

    # Try serial C++ binary first, then parallel with -rs 0
    local cpp_bin="${name}_cpp"
    if ! wsl -e bash -c "test -x ~/bin/${cpp_bin}" 2>/dev/null; then
        cpp_bin="${name}p_cpp"
        ca="-rs 0 $ca"
    fi
    wsl -e bash -c "timeout 300 ~/bin/${cpp_bin} -m ${CPP_DATA}/${mesh} ${ca}" > "$cout" 2>&1

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
