#!/bin/bash
# MFEM 并行示例 1:1 比对工具 (Git Bash)
# 用法: bash pex_compare.sh pex1 pex2 pex3
#       bash pex_compare.sh --all
# 策略: Rust np=1/2/4 一致 + Rust np1 vs C++ np1 DOF 一致

cd /c/Users/lilu/works/fem-pro/fem-rs

DATA_DIR="data"
CPP_DATA="/home/quan/mfem49/data"
OUT_DIR="tmp/cmp"
mkdir -p "$OUT_DIR"

# 示例名 -> Rust 二进制名
pex_bin() {
    local name="$1"
    case "$name" in
        pex1) echo "mfem_pex1_parallel_poisson" ;;
        pex2) echo "mfem_pex2_parallel_elasticity" ;;
        pex3) echo "mfem_pex3_maxwell_cavity" ;;
        pex4) echo "mfem_pex4_parallel_hdiv_diffusion" ;;
        pex5) echo "mfem_pex5_hdiv_darcy" ;;
        pex6) echo "mfem_pex6_parallel_amr" ;;
        pex7) echo "mfem_pex7_parallel_surface" ;;
        pex8) echo "mfem_pex8_parallel_dpg" ;;
        pex9) echo "mfem_pex9_parallel_dg_advection" ;;
        pex10) echo "mfem_pex10_parallel_hyperelastic" ;;
        pex12) echo "mfem_pex12_parallel_elastic_eigen" ;;
        pex13) echo "mfem_pex13_parallel_eigenvalue" ;;
        pex14) echo "mfem_pex14_parallel_dg_poisson" ;;
        pex15) echo "mfem_pex15_parallel_dynamic_amr" ;;
        pex16) echo "mfem_pex16_parallel_nonlinear_heat" ;;
        pex17) echo "mfem_pex17_parallel_dg_elasticity" ;;
        pex18) echo "mfem_pex18_parallel_euler" ;;
        pex19) echo "mfem_pex19_parallel_incomp_hyperelastic" ;;
        pex20) echo "mfem_pex20_parallel_symplectic" ;;
        pex21) echo "mfem_pex21_parallel_amr_elasticity" ;;
        pex22) echo "mfem_pex22_parallel_complex_helmholtz" ;;
        pex24) echo "mfem_pex24_parallel_discrete_ops" ;;
        pex25) echo "mfem_pex25_pml_maxwell" ;;
        pex26) echo "mfem_pex26_parallel_geom_mg" ;;
        pex27) echo "mfem_pex27_parallel_robin_bc" ;;
        pex28) echo "mfem_pex28_parallel_sliding_elasticity" ;;
        pex29) echo "mfem_pex29_surface_poisson" ;;
        pex30) echo "mfem_pex30_amr_preprocess" ;;
        pex31) echo "mfem_pex31_restricted_hcurl" ;;
        pex32) echo "mfem_pex32_maxwell_eigenvalue" ;;
        pex33) echo "mfem_pex33_fractional_laplacian" ;;
        pex34) echo "mfem_pex34_magnetostatics" ;;
        pex35) echo "mfem_pex35_complex_oscillator" ;;
        pex36) echo "mfem_pex36_obstacle" ;;
        pex37) echo "mfem_pex37_topology_optimization" ;;
        pex39) echo "mfem_pex39_named_attributes" ;;
        pex40) echo "mfem_pex40_eikonal" ;;
        pex41) echo "mfem_pex41_imex" ;;
    esac
}

extract_dof() {
    local val=""
    val=$(grep -oE "Number of finite element unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Number of unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Unknowns: *[0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "dim\(R\+W\) = [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Number of [a-zA-Z/]+ unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Number of velocity/deformation unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "unknowns = [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    # DPG format: "Trial space,     X0   : 5281 (order 1)"
    val=$(grep "Trial space" "$1" 2>/dev/null | head -1 | sed 's/.*: *\([0-9]*\).*/\1/')
    [ -n "$val" ] && [ "$val" != "0" ] && echo "$val" && return
    echo ""
}

run_pex() {
    local name="$1" mesh="$2" ra="$3"
    local bin=$(pex_bin "$name")
    local r1="$OUT_DIR/${name}_r1.log"
    local r2="$OUT_DIR/${name}_r2.log"
    local r4="$OUT_DIR/${name}_r4.log"
    local c1="$OUT_DIR/${name}_c1.log"

    echo "=== $name ==="

    # Rust np=1/2/4
    local rc1=0 rc2=0 rc4=0
    local mesh_arg=""
    # mesh 为 "default" 时不传 -m（示例内部用默认网格+加密）
    if [ "$mesh" != "default" ]; then
        mesh_arg="-m $DATA_DIR/$mesh"
    fi
    "./target/release/examples/${bin}.exe" --ranks 1 $mesh_arg $ra > "$r1" 2>&1 || rc1=$?
    "./target/release/examples/${bin}.exe" --ranks 2 $mesh_arg $ra > "$r2" 2>&1 || rc2=$?
    if [ -n "$SKIP_NP4" ]; then
        cp "$r2" "$r4"  # np4 跳过时复用 np2
    else
        "./target/release/examples/${bin}.exe" --ranks 4 $mesh_arg $ra > "$r4" 2>&1 || rc4=$?
    fi

    if [ $rc1 -ne 0 ] || grep -q "panic" "$r1"; then
        echo "  FAIL (rust np1 exit=$rc1)"
        return
    fi
    if [ $rc2 -ne 0 ] || grep -q "panic" "$r2"; then
        echo "  FAIL (rust np2 exit=$rc2)"
        return
    fi
    if [ $rc4 -ne 0 ] || grep -q "panic" "$r4"; then
        echo "  FAIL (rust np4 exit=$rc4)"
        return
    fi

    # C++ np=1 (mpirun -np 1, WSL 中 np>1 段错误)
    local cpp_bin="ex${name#pex}p_cpp"
    local cpp_mesh_arg="-m ${CPP_DATA}/${mesh}"
    if [ "$mesh" = "default" ]; then
        cpp_mesh_arg="-m ${CPP_DATA}/star.mesh"
    fi
    # Rust 用 -r，C++ 用 -rs/-rp（转换；-rs 不转换）
    local cpp_ra="$ra"
    if [[ "$cpp_ra" == *" -r "* ]] || [[ "$cpp_ra" == -*" -r "* ]] || [[ "$cpp_ra" == "-r "* ]]; then
        cpp_ra=$(echo "$cpp_ra" | sed 's/ -r \([0-9]*\)/ -rs \1 -rp 0/; s/^-r \([0-9]*\)/-rs \1 -rp 0/')
    fi
    # 特殊处理：pex7 C++ 用 -e -o -r（不用 -m）
    if [ "$name" = "pex7" ]; then
        cpp_mesh_arg=""
        cpp_ra="-e 0 -o 1 -r 0 -no-vis"
    fi
    wsl -e bash -c "timeout 300 mpirun --allow-run-as-root -np 1 ~/bin/${cpp_bin} ${cpp_mesh_arg} ${cpp_ra} 2>&1" > "$c1" 2>&1

    # DOF 提取
    local d1=$(extract_dof "$r1")
    local d2=$(extract_dof "$r2")
    local d4=$(extract_dof "$r4")
    local dc=$(extract_dof "$c1")

    echo "  Rust DOFs: np1=$d1 np2=$d2 np4=$d4"
    echo "  C++  DOF:  np1=$dc"

    # 判定: np1=np2=np4 一致 + np1 与 C++ 一致
    if [ -n "$d1" ] && [ "$d1" = "$d2" ] && [ "$d1" = "$d4" ]; then
        if [ -n "$dc" ] && [ "$d1" = "$dc" ]; then
            echo "  OK (dof=$d1, np1=np2=np4=C++)"
        else
            echo "  DIFF_CPP (rust np1=$d1 cpp=$dc)"
        fi
    elif [ -z "$d1" ] && [ -z "$dc" ]; then
        echo "  NO_DOF"
    else
        echo "  DIFF_NP (np1=$d1 np2=$d2 np4=$d4)"
    fi
}

declare -A PEX_MESH
PEX_MESH[pex1]="star.mesh|-no-vis"
PEX_MESH[pex2]="beam-tri.mesh|-no-vis"
PEX_MESH[pex3]="default|-no-vis"
PEX_MESH[pex4]="star.mesh|-no-vis"
PEX_MESH[pex5]="star.mesh|-r 1 -no-vis"
PEX_MESH[pex6]="star.mesh|-no-vis"
PEX_MESH[pex7]="star.mesh|-no-vis"
PEX_MESH[pex8]="star.mesh|-r 5 -no-vis"
PEX_MESH[pex9]="star.mesh|-no-vis"
PEX_MESH[pex10]="beam-quad.mesh|-r 2 -o 2 -dt 3 -no-vis"
PEX_MESH[pex12]="beam-tri.mesh|-n 5 -no-vis"
PEX_MESH[pex13]="beam-tri.mesh|-rs 3 -rp 0 -no-vis"
PEX_MESH[pex14]="star.mesh|-r 4 -o 2 -no-vis"
PEX_MESH[pex15]="star.mesh|-no-vis"
PEX_MESH[pex16]="star.mesh|-r 2 -o 2 -no-vis"
PEX_MESH[pex17]="beam-tri.mesh|-no-vis"
PEX_MESH[pex18]="star.mesh|-no-vis"
PEX_MESH[pex19]="beam-quad.mesh|-o 2 -r 0 -no-vis"
PEX_MESH[pex20]="star.mesh|-no-vis"
PEX_MESH[pex21]="beam-tri.mesh|-o 2 -no-vis"
PEX_MESH[pex22]="inline-quad.mesh|-p 0 -no-vis"
PEX_MESH[pex24]="star.mesh|-p 2 -o 2 -no-vis"
PEX_MESH[pex25]="inline-quad.mesh|-o 2 -f 5.0 -ref 3 -prob 4 -no-vis"
PEX_MESH[pex26]="star.mesh|-no-vis"
PEX_MESH[pex27]="inline-quad.mesh|-no-vis"
PEX_MESH[pex28]="inline-quad.mesh|-no-vis"
PEX_MESH[pex29]="disc-nurbs.mesh|-no-vis"
PEX_MESH[pex30]="star.mesh|-no-vis"
PEX_MESH[pex31]="beam-tri.mesh|-o 1 -r 1 -no-vis"
PEX_MESH[pex32]="fichera.mesh|-no-vis"
PEX_MESH[pex33]="square-disc.mesh|-alpha 0.33 -o 2 -no-vis"
PEX_MESH[pex34]="fichera-mixed.mesh|-no-vis"
PEX_MESH[pex35]="fichera-mixed.mesh|-p 0 -o 1 -no-vis"
PEX_MESH[pex36]="disc-nurbs.mesh|-no-vis"
PEX_MESH[pex37]="star.mesh|-no-vis"
PEX_MESH[pex39]="compass.msh|-no-vis"
PEX_MESH[pex40]="star.mesh|-no-vis"
PEX_MESH[pex41]="periodic-square.mesh|-p 0 -r 2 -o 3 -no-vis"

if [ "$1" = "--all" ]; then
    for name in $(echo "${!PEX_MESH[@]}" | tr ' ' '\n' | sort -V); do
        IFS='|' read -r mesh ra <<< "${PEX_MESH[$name]}"
        run_pex "$name" "$mesh" "$ra" || true
    done
else
    for name in "$@"; do
        if [ -n "${PEX_MESH[$name]}" ]; then
            IFS='|' read -r mesh ra <<< "${PEX_MESH[$name]}"
            run_pex "$name" "$mesh" "$ra" || true
        else
            echo "$name: 未配置"
        fi
    done
fi
