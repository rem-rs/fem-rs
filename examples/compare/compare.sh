#!/bin/bash
# MFEM 示例 1:1 比对工具 (Git Bash)
# 用法: bash compare.sh ex1 ex2 ex3 或 bash compare.sh --all

cd /c/Users/lilu/works/fem-pro/fem-rs

DATA_DIR="data"
CPP_DATA="/home/quan/mfem49/data"
OUT_DIR="tmp/cmp"
mkdir -p "$OUT_DIR"

# DOF 提取函数 - 支持多种格式
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
    val=$(grep "Trial space" "$1" 2>/dev/null | head -1 | sed 's/.*: *\([0-9]*\).*/\1/')
    [ -n "$val" ] && [ "$val" != "0" ] && echo "$val" && return
    val=$(grep -oE "Number of [a-zA-Z/]+ unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Number of Raviart-Thomas finite element unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    val=$(grep -oE "Number of L2 finite element unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    # H(curl) format: "DOFs: H(Curl)=114 H¹(z)=51 total=165"
    val=$(grep -oE "total=[0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    # H(Curl) format: "Number of H(Curl) unknowns: 1752"
    val=$(grep -oE "Number of H\(Curl\) unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    # Fractional format: "Number of degrees of freedom: 20096"
    val=$(grep -oE "Number of degrees of freedom: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    # SubMesh format: "SubMesh H1 DOFs: 34, RT DOFs: 110"
    val=$(grep -oE "SubMesh H1 DOFs: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    # Element count: "Number of Elements 590"
    val=$(grep -oE "Number of Elements [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    [ -n "$val" ] && echo "$val" && return
    echo ""
}

# 查找 C++ 二进制文件
find_cpp_bin() {
    local name="$1"
    local bin=""
    
    # 尝试各种可能的二进制文件名
    for suffix in "" "p" "_cpp" "p_cpp"; do
        bin="${name}${suffix}"
        if wsl -e bash -c "test -x ~/bin/${bin}" 2>/dev/null; then
            echo "$bin"
            return
        fi
    done
    echo ""
}

# 检查 C++ 二进制是否支持某个选项
cpp_supports_option() {
    local bin="$1"
    local opt="$2"
    wsl -e bash -c "~/bin/${bin} --help 2>&1" | grep -q -- "$opt"
}

run_one() {
    local name="$1" mesh="$2" ra="$3" ca="$4" modes="$5"
    local rout="$OUT_DIR/${name}_rust.log"
    local cout="$OUT_DIR/${name}_cpp.log"

    echo "=== $name ==="

    # 查找 Rust 二进制
    local rust_bin=$(find_cpp_bin "mfem_${name}_" 2>/dev/null || find_cpp_bin "mfem_${name}" 2>/dev/null || echo "${name}.exe")
    if [ ! -f "target/release/${rust_bin}" ] && [ ! -f "target/release/examples/${rust_bin}" ]; then
        # 尝试直接作为示例名
        rust_bin="examples/mfem_${name}.exe"
        case "$name" in
            ex1) rust_bin="examples/mfem_ex1_poisson.exe" ;;
            ex2) rust_bin="examples/mfem_ex2_elasticity.exe" ;;
            ex3) rust_bin="examples/mfem_ex3_maxwell_cavity.exe" ;;
            ex4) rust_bin="examples/mfem_ex4_darcy.exe" ;;
            ex5) rust_bin="examples/mfem_ex5_mixed_darcy.exe" ;;
            ex6) rust_bin="examples/mfem_ex6_flux_recovery.exe" ;;
            ex7) rust_bin="examples/mfem_ex7_surface_poisson.exe" ;;
            ex8) rust_bin="examples/mfem_ex8_dpg_2x2.exe" ;;
            ex9) rust_bin="examples/mfem_ex9_dg_advection.exe" ;;
            ex10) rust_bin="examples/mfem_ex10_hyperelastic_dyn.exe" ;;
            ex11) rust_bin="examples/mfem_ex11_eigenvalue.exe" ;;
            ex12) rust_bin="examples/mfem_ex12_elastic_eigen.exe" ;;
            ex13) rust_bin="examples/mfem_ex13_eigenvalue.exe" ;;
            ex14) rust_bin="examples/mfem_ex14_dg_poisson.exe" ;;
            ex15) rust_bin="examples/mfem_ex15_dynamic_amr.exe" ;;
            ex16) rust_bin="examples/mfem_ex16_nonlinear_heat.exe" ;;
            ex17) rust_bin="examples/mfem_ex17_dg_elasticity.exe" ;;
            ex18) rust_bin="examples/mfem_ex18_euler.exe" ;;
            ex19) rust_bin="examples/mfem_ex19_hyperelastic_incomp.exe" ;;
            ex20) rust_bin="examples/mfem_ex20_symplectic.exe" ;;
            ex21) rust_bin="examples/mfem_ex21_amr_elasticity.exe" ;;
            ex22) rust_bin="examples/mfem_ex22_complex_helmholtz.exe" ;;
            ex23) rust_bin="examples/mfem_ex23_wave_equation.exe" ;;
            ex24) rust_bin="examples/mfem_ex24_discrete_ops.exe" ;;
            ex25) rust_bin="examples/mfem_ex25_pml_maxwell.exe" ;;
            ex26) rust_bin="examples/mfem_ex26_geom_mg.exe" ;;
            ex27) rust_bin="examples/mfem_ex27_robin_bc.exe" ;;
            ex28) rust_bin="examples/mfem_ex28_sliding_elasticity.exe" ;;
            ex29) rust_bin="examples/mfem_ex29_curved_poisson.exe" ;;
            ex30) rust_bin="examples/mfem_ex30_aniso_amr.exe" ;;
            ex31) rust_bin="examples/mfem_ex31_anisotropic_maxwell.exe" ;;
            ex32) rust_bin="examples/mfem_ex32_maxwell_eigenvalue.exe" ;;
            ex33) rust_bin="examples/mfem_ex33_fractional_diffusion.exe" ;;
            ex34) rust_bin="examples/mfem_ex34_magnetostatics.exe" ;;
            ex35) rust_bin="examples/mfem_ex35_complex_oscillator.exe" ;;
            ex36) rust_bin="examples/mfem_ex36_obstacle.exe" ;;
            ex37) rust_bin="examples/mfem_ex37_topology_optimization.exe" ;;
            ex38) rust_bin="examples/mfem_ex38_implicit_integration.exe" ;;
            ex39) rust_bin="examples/mfem_ex39_compass.exe" ;;
            ex40) rust_bin="examples/mfem_ex40_eikonal.exe" ;;
            ex41) rust_bin="examples/mfem_ex41_imex.exe" ;;
        esac
    fi

    # 运行 Rust
    local rc=0
    "./target/release/${rust_bin}" -m "data/${mesh}" $ra > "$rout" 2>&1 || rc=$?
    if [ $rc -ne 0 ] || grep -q "panic" "$rout"; then
        echo "  FAIL (rust exit=$rc)"
        return
    fi

    # 查找 C++ 二进制
    local cpp_bin=$(find_cpp_bin "$name")
    if [ -z "$cpp_bin" ]; then
        echo "  NO_CPP_BIN"
        return
    fi

    # 构建 C++ 参数
    local cpp_ca="$ca"
    local cpp_mesh_arg="-m ${CPP_DATA}/${mesh}"
    
    # 检查是否支持 -m
    if ! cpp_supports_option "$cpp_bin" "--mesh"; then
        cpp_mesh_arg=""
    fi
    
    # 检查是否支持 -rs
    if [ -n "$cpp_ca" ] && ! cpp_supports_option "$cpp_bin" "--refine-serial"; then
        cpp_ca=$(echo "$cpp_ca" | sed 's/-rs [0-9]*//g')
    fi
    
    # 特殊处理：ex29 C++ 用 -mt 和 -mo 而不是 -m
    if [ "$name" = "ex29" ]; then
        cpp_mesh_arg="-mt 4 -mo 3"
        # 移除 ca 中已有的 -mt/-mo（避免重复）
        cpp_ca=$(echo "$cpp_ca" | sed 's/-mt [0-9]*//g; s/-mo [0-9]*//g')
    fi

    wsl -e bash -c "timeout 300 ~/bin/${cpp_bin} ${cpp_mesh_arg} ${cpp_ca}" > "$cout" 2>&1

    # 比对 DOF（若模式包含 dof）
    if [[ "$modes" == *"dof"* ]]; then
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
    fi

    # 比对 conv_avg（取最后一个值）
    if [[ "$modes" == *"conv_avg"* ]]; then
        local ra_=$(grep -oE "Average reduction factor = [-+0-9.eE]+" "$rout" 2>/dev/null | tail -1 | grep -oE "[-+0-9.eE]+$")
        local ca_=$(grep -oE "Average reduction factor = [-+0-9.eE]+" "$cout" 2>/dev/null | tail -1 | grep -oE "[-+0-9.eE]+$")
        if [ -n "$ra_" ] && [ -n "$ca_" ]; then
            echo "  conv_avg: rust=$ra_ cpp=$ca_"
        elif [ -n "$ra_" ] || [ -n "$ca_" ]; then
            echo "  conv_avg: rust=$ra_ cpp=$ca_"
        fi
    fi
}

# 示例配置
declare -A MESH_ARGS
MESH_ARGS[ex1]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex2]="beam-tri.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex3]="beam-tet.mesh|-no-vis|-no-vis|dof+cg"
MESH_ARGS[ex4]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex5]="star.mesh|-no-vis|-no-vis|dof+minres"
MESH_ARGS[ex6]="square-disc.mesh|-o 1 -no-vis|-o 1 -no-vis|dof"
MESH_ARGS[ex7]="star.mesh|-no-vis|-e 0 -o 2 -r 2 -no-vis|dof+iter"
MESH_ARGS[ex8]="star.mesh|-no-vis|-no-vis|dof+conv_avg"
MESH_ARGS[ex9]="star.mesh|-no-vis|-no-vis|dof"
MESH_ARGS[ex10]="beam-quad.mesh|-r 2 -o 2 -dt 3 -no-vis|-r 2 -o 2 -dt 3 -no-vis|dof+newton"
MESH_ARGS[ex11]="star.mesh|-no-vis||eigenvalue"
MESH_ARGS[ex12]="beam-tri.mesh|-n 5 -no-vis|-n 5|eigenvalue"
MESH_ARGS[ex13]="beam-tet.mesh|--ame -no-vis|-rp 0|eigenvalue"
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
MESH_ARGS[ex24]="star.mesh|-p 2 -o 1 -no-vis|-p 2 -o 1 -no-vis|dof"
MESH_ARGS[ex25]="inline-quad.mesh|-o 2 -f 5.0 -ref 3 -prob 4 -no-vis|-o 2 -f 5.0 -ref 3 -prob 4 -no-vis|dof"
MESH_ARGS[ex26]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex27]="inline-quad.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex28]="inline-quad.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex29]="disc-nurbs.mesh|-no-vis|-mt 4 -mo 3 -rs 0 -rp 0 -no-vis|dof+iter"
MESH_ARGS[ex30]="star.mesh|-o 1 -no-vis|-o 1 -no-vis|dof+marked"
MESH_ARGS[ex31]="beam-tri.mesh|-o 1 -r 1 -no-vis|-o 1 -r 1 -no-vis|dof+iter"
MESH_ARGS[ex32]="fichera.mesh|-rs 2 -no-vis|-rs 2 -rp 0 -no-vis|eigenvalue"
MESH_ARGS[ex33]="square-disc.mesh|-alpha 0.33 -o 2 -no-vis|-alpha 0.33 -o 2 -no-vis|dof+iter"
MESH_ARGS[ex34]="fichera-mixed.mesh|-no-vis|-no-vis|conv_avg"
MESH_ARGS[ex35]="fichera-mixed.mesh|-p 0 -o 1 -no-vis|-p 0 -o 1 -no-vis|dof+fgmres"
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
