#!/bin/bash
# MFEM 示例 1:1 比对工具 (Git Bash)
# 用法: bash compare.sh ex1 ex2 ex3
#       bash compare.sh --all

set -e
cd /c/Users/lilu/works/fem-pro/fem-rs

DATA_DIR="data"
CPP_DATA="/home/quan/mfem49/data"
OUT_DIR="tmp/cmp"
mkdir -p "$OUT_DIR"

extract_dof() {
    grep -oE "Number of (finite element )?unknowns: [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || echo ""
}

extract_ev() {
    grep -oE "(eigenvalue|lambda)[^=]*=[ ]*[-+0-9.eE]+" "$1" 2>/dev/null | grep -oE "[-+0-9.eE]+$" | head -1 || echo ""
}

extract_iter() {
    grep -oE "(CG|MINRES|GMRES|FGMRES) converged at iter [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || echo ""
}

extract_newton() {
    grep -oE "Number of Newton iterations = [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || echo ""
}

extract_marked() {
    grep -oE "Marked [0-9]+ elements?" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1 || echo ""
}

extract_conv_avg() {
    grep -oE "Average reduction factor = [-+0-9.eE]+" "$1" 2>/dev/null | grep -oE "[-+0-9.eE]+$" || echo ""
}

run_one() {
    local name="$1" mesh="$2" ra="$3" ca="$4" modes="$5"
    local rout="$OUT_DIR/${name}_rust.log"
    local cout="$OUT_DIR/${name}_cpp.log"

    echo "=== $name ==="

    # Rust (use cmd.exe to run .exe from Git Bash)
    local rc=0
    cmd.exe /c "target\\release\\examples\\${name}.exe" -m "$DATA_DIR\\$mesh" $ra > "$rout" 2>&1 || rc=$?
    if [ $rc -ne 0 ] || grep -q "panic" "$rout"; then
        echo "  FAIL (rust exit=$rc)"
        return
    fi

    # C++
    wsl -e bash -c "timeout 150 ~/bin/${name}_cpp -m $CPP_DATA/$mesh $ca" > "$cout" 2>&1

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

    # 比对各指标
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

# 示例配置
declare -A MESH_ARGS
MESH_ARGS[ex1]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex2]="beam-tri.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex3]="beam-tet.mesh|-no-vis|-no-vis|dof+cg"
MESH_ARGS[ex4]="star.mesh|-no-vis|-no-vis|dof+iter"
MESH_ARGS[ex5]="star.mesh|-no-vis|-no-vis|dof+minres"
MESH_ARGS[ex13]="beam-tet.mesh|--ame -no-vis|-rs 0 -rp 0|eigenvalue"
MESH_ARGS[ex21]="beam-tri.mesh|-o 2 -no-vis|-o 2 -no-vis|dof"

if [ "$1" = "--all" ]; then
    for name in $(echo "${!MESH_ARGS[@]}" | tr ' ' '\n' | sort -V); do
        IFS='|' read -r mesh ra ca modes <<< "${MESH_ARGS[$name]}"
        run_one "$name" "$mesh" "$ra" "$ca" "$modes" 2>/dev/null
    done
else
    for name in "$@"; do
        if [ -n "${MESH_ARGS[$name]}" ]; then
            IFS='|' read -r mesh ra ca modes <<< "${MESH_ARGS[$name]}"
            run_one "$name" "$mesh" "$ra" "$ca" "$modes" 2>/dev/null
        else
            echo "$name: 未配置"
        fi
    done
fi
