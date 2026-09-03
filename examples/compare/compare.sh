#!/bin/bash
# MFEM 示例 1:1 比对工具 (Git Bash)
# 核心目标：解的一致性 + 迭代轨迹比对
# 用法: bash compare.sh ex1 ex2 ex3 或 bash compare.sh --all

cd /c/Users/lilu/works/fem-pro/fem-rs

DATA_DIR="data"
CPP_DATA="/home/quan/mfem49/data"
OUT_DIR="tmp/cmp"
mkdir -p "$OUT_DIR"

# 比对两个 sol.gf 文件
compare_sol() {
    local rust_sol="$1" cpp_sol="$2"
    
    if [ ! -f "$rust_sol" ] || [ ! -f "$cpp_sol" ]; then
        echo "MISSING_SOL"
        return
    fi
    
    wsl -e bash -c "python3 << 'PYEOF'
import math, sys

def read_sol(path):
    \"\"\"读取 MFEM sol.gf 格式：头部(字母行) + 空行 + 每行一个值(按顶点序)\"\"\"
    vals = {}
    try:
        with open(path) as f:
            lines = f.readlines()
        data_started = False
        vid = 0
        for line in lines:
            line = line.strip()
            if not line:
                data_started = True
                continue
            if not data_started:
                continue
            try:
                val = float(line)
                vals[vid] = val
                vid += 1
            except ValueError:
                pass
    except Exception as e:
        print(f'ERROR reading {path}: {e}')
    return vals

rust = read_sol('$rust_sol')
cpp = read_sol('$cpp_sol')

if not rust and not cpp:
    print('BOTH_EMPTY')
    exit(0)
if not rust:
    print('NO_RUST_SOL')
    exit(0)
if not cpp:
    print('NO_CPP_SOL')
    exit(0)

common = set(rust.keys()) & set(cpp.keys())
if not common:
    print('NO_COMMON_VERTICES')
    exit(0)

max_diff = 0.0
l2_diff = 0.0
rust_l2 = 0.0
cpp_l2 = 0.0

for vid in common:
    rv = rust[vid]
    cv = cpp[vid]
    diff = abs(rv - cv)
    max_diff = max(max_diff, diff)
    l2_diff += diff * diff
    rust_l2 += rv * rv
    cpp_l2 += cv * cv

n = len(common)
l2_diff = math.sqrt(l2_diff / n)
rust_l2 = math.sqrt(rust_l2 / n)
cpp_l2 = math.sqrt(cpp_l2 / n)

rel_l2 = l2_diff / rust_l2 if rust_l2 > 0 else 0
rel_max = max_diff / max(abs(v) for v in rust.values()) if rust else 0

print(f'n_verts={n} rust_l2={rust_l2:.10e} cpp_l2={cpp_l2:.10e} l2_diff={l2_diff:.10e} max_diff={max_diff:.10e} rel_l2={rel_l2:.2e} rel_max={rel_max:.2e}')
PYEOF"
}

# 比对迭代轨迹
compare_iterations() {
    local rust_log="$1" cpp_log="$2"
    
    local rust_iter=$(grep -c "Iteration" "$rust_log" 2>/dev/null || echo 0)
    local cpp_iter=$(grep -c "Iteration" "$cpp_log" 2>/dev/null || echo 0)
    
    echo "  [ITER] rust=$rust_iter cpp=$cpp_iter"
    
    # 比较最终残差
    local rust_final=$(grep "Iteration" "$rust_log" | tail -1 | grep -oE "[-+0-9.eE]+$" | tail -1)
    local cpp_final=$(grep "Iteration" "$cpp_log" | tail -1 | grep -oE "[-+0-9.eE]+$" | tail -1)
    
    if [ -n "$rust_final" ] && [ -n "$cpp_final" ]; then
        wsl -e python3 -c "
r=float('$rust_final'); c=float('$cpp_final')
diff=abs(r-c)
rel=diff/abs(r) if r!=0 else 0
print(f'  [RES] rust_final={r:.10e} cpp_final={c:.10e} diff={diff:.2e} rel={rel:.2e}')
"
    fi
}

# 运行单个示例
run_one() {
    local name="$1" mesh="$2" ra="$3" ca="$4"
    local rout="$OUT_DIR/${name}_rust.log"
    local cout="$OUT_DIR/${name}_cpp.log"
    
    echo "=== $name ==="
    
    # 查找 Rust 二进制
    local rust_bin="examples/mfem_ex${name#ex}.exe"
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
        ex39) rust_bin="examples/mfem_ex39_compass.exe" ;;
        ex40) rust_bin="examples/mfem_ex40_eikonal.exe" ;;
        ex41) rust_bin="examples/mfem_ex41_imex.exe" ;;
    esac
    
    if [ ! -f "target/release/${rust_bin}" ]; then
        echo "  NO_RUST_BIN ($rust_bin)"
        return
    fi
    
    # 运行 Rust
    local rc=0
    "./target/release/${rust_bin}" -m "data/${mesh}" $ra > "$rout" 2>&1 || rc=$?
    if [ $rc -ne 0 ] || grep -q "panic" "$rout"; then
        echo "  RUST_FAIL (exit=$rc)"
        tail -5 "$rout"
        return
    fi
    
    # 查找 C++ 二进制
    local cpp_bin=""
    for suffix in "" "_cpp" "p_cpp" "p"; do
        if wsl -e bash -c "test -x ~/bin/${name}${suffix}" 2>/dev/null; then
            cpp_bin="${name}${suffix}"
            break
        fi
    done
    
    if [ -z "$cpp_bin" ]; then
        echo "  NO_CPP_BIN"
        return
    fi
    
    # 运行 C++
    local cpp_ca="$ca"
    local cpp_mesh_arg="-m ${CPP_DATA}/${mesh}"
    
    if [ "$name" = "ex29" ]; then
        cpp_mesh_arg="-mt 4 -mo 3"
        cpp_ca=$(echo "$cpp_ca" | sed 's/-mt [0-9]*//g; s/-mo [0-9]*//g')
    fi
    
    wsl -e bash -c "timeout 300 ~/bin/${cpp_bin} ${cpp_mesh_arg} ${cpp_ca}" > "$cout" 2>&1
    
    # ─── 比对解 ───
    local rust_sol="sol.gf"
    local cpp_sol="sol.gf"
    
    if [ -f "$rust_sol" ] && wsl -e bash -c "test -f $cpp_sol" 2>/dev/null; then
        echo "  [SOL] Comparing sol.gf..."
        compare_sol "$rust_sol" "$cpp_sol"
    else
        echo "  [SOL] Rust: ${rust_sol:-MISSING} C++: ${cpp_sol:-MISSING}"
    fi
    
    # ─── 比对迭代轨迹 ───
    compare_iterations "$rout" "$cout"
}

# 示例配置
declare -A MESH_ARGS
MESH_ARGS[ex1]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex2]="beam-tri.mesh|-no-vis|-no-vis"
MESH_ARGS[ex3]="beam-tet.mesh|-no-vis|-no-vis"
MESH_ARGS[ex4]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex5]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex6]="square-disc.mesh|-o 1 -no-vis|-o 1 -no-vis"
MESH_ARGS[ex7]="star.mesh|-no-vis|-e 0 -o 2 -r 2 -no-vis"
MESH_ARGS[ex8]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex9]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex10]="beam-quad.mesh|-r 2 -o 2 -dt 3 -no-vis|-r 2 -o 2 -dt 3 -no-vis"
MESH_ARGS[ex11]="star.mesh|-no-vis|"
MESH_ARGS[ex12]="beam-tri.mesh|-n 5 -no-vis|-n 5"
MESH_ARGS[ex13]="beam-tet.mesh|--ame -no-vis|-rp 0"
MESH_ARGS[ex14]="star.mesh|-r 4 -o 2 -no-vis|-r 4 -o 2 -no-vis"
MESH_ARGS[ex15]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex16]="star.mesh|-r 2 -o 2 -no-vis|-r 2 -o 2 -no-vis"
MESH_ARGS[ex17]="beam-tri.mesh|-no-vis|-no-vis"
MESH_ARGS[ex18]="periodic-square.mesh|-o 1 -no-vis|-o 1 -no-vis"
MESH_ARGS[ex19]="beam-quad.mesh|-o 2 -r 0 -no-vis|-o 2 -r 0 -no-vis"
MESH_ARGS[ex20]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex21]="beam-tri.mesh|-o 2 -no-vis|-o 2 -no-vis"
MESH_ARGS[ex22]="inline-quad.mesh|-p 0 -no-vis|-p 0 -no-vis"
MESH_ARGS[ex23]="star.mesh|-o 4 -tf 2 -no-vis|-o 4 -tf 2 -no-vis"
MESH_ARGS[ex24]="star.mesh|-p 2 -o 1 -no-vis|-p 2 -o 1 -no-vis"
MESH_ARGS[ex25]="inline-quad.mesh|-o 2 -f 5.0 -ref 3 -prob 4 -no-vis|-o 2 -f 5.0 -ref 3 -prob 4 -no-vis"
MESH_ARGS[ex26]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex27]="inline-quad.mesh|-no-vis|-no-vis"
MESH_ARGS[ex28]="inline-quad.mesh|-no-vis|-no-vis"
MESH_ARGS[ex29]="disc-nurbs.mesh|-no-vis|-mt 4 -mo 3 -r 0 -no-vis"
MESH_ARGS[ex30]="star.mesh|-o 1 -no-vis|-o 1 -no-vis"
MESH_ARGS[ex31]="beam-tri.mesh|-o 1 -r 1 -no-vis|-o 1 -r 1 -no-vis"
MESH_ARGS[ex32]="fichera.mesh|-rs 2 -no-vis|-rs 2 -rp 0 -no-vis"
MESH_ARGS[ex33]="square-disc.mesh|-alpha 0.33 -o 2 -no-vis|-alpha 0.33 -o 2 -no-vis"
MESH_ARGS[ex34]="fichera-mixed.mesh|-no-vis|-no-vis"
MESH_ARGS[ex35]="fichera-mixed.mesh|-p 0 -o 1 -no-vis|-p 0 -o 1 -no-vis"
MESH_ARGS[ex36]="disc-nurbs.mesh|-no-vis|-no-vis"
MESH_ARGS[ex37]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex39]="compass.msh|-no-vis|-no-vis"
MESH_ARGS[ex40]="star.mesh|-no-vis|-no-vis"
MESH_ARGS[ex41]="periodic-square.mesh|-p 0 -r 2 -o 3 -no-vis|-p 0 -r 2 -o 3 -no-vis"

if [ "$1" = "--all" ]; then
    for name in $(echo "${!MESH_ARGS[@]}" | tr ' ' '\n' | sort -V); do
        IFS='|' read -r mesh ra ca <<< "${MESH_ARGS[$name]}"
        run_one "$name" "$mesh" "$ra" "$ca" || true
    done
else
    for name in "$@"; do
        if [ -n "${MESH_ARGS[$name]}" ]; then
            IFS='|' read -r mesh ra ca <<< "${MESH_ARGS[$name]}"
            run_one "$name" "$mesh" "$ra" "$ca" || true
        else
            echo "$name: 未配置"
        fi
    done
fi
