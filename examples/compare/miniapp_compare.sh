#!/bin/bash
# Miniapp 1:1 比对工具
# 用法: bash miniapp_compare.sh mini_nurbs_solenoidal mini_joule ...

cd /c/Users/lilu/works/fem-pro/fem-rs

DATA_DIR="data"
OUT_DIR="tmp/miniapp_cmp"
mkdir -p "$OUT_DIR"

# C++ miniapp 二进制目录
CPP_BIN="$HOME/bin"

run_miniapp() {
    local name="$1"
    local rust_args="$2"
    local cpp_args="$3"
    local cpp_bin="$4"
    local rout="$OUT_DIR/${name}_rust.log"
    local cout="$OUT_DIR/${name}_cpp.log"

    echo "=== $name ==="

    # 查找 Rust 二进制
    local rust_bin="examples/${name}.exe"
    local rust_bin_path="target/release/${rust_bin}"
    if [ ! -f "$rust_bin_path" ]; then
        rust_bin_path="target/debug/${rust_bin}"
    fi
    if [ ! -f "$rust_bin_path" ]; then
        echo "  NO_RUST_BIN ($rust_bin)"
        return
    fi

    # 运行 Rust
    local rc=0
    if [[ "$name" == mini_* ]]; then
        # MPI miniapp - 通过 wsl 运行
        local win_path=$(echo "$rust_bin_path" | sed 's|/|\\|g')
        wsl -e bash -c "mpirun --allow-run-as-root -n 1 $win_path $rust_args" > "$rout" 2>&1 || rc=$?
    else
        "./$rust_bin_path" $rust_args > "$rout" 2>&1 || rc=$?
    fi
    if [ $rc -ne 0 ] || grep -q "panic" "$rout"; then
        echo "  RUST_FAIL (exit=$rc)"
        tail -5 "$rout"
        return
    fi
    echo "  RUST_OK"

    # 运行 C++
    if [ -n "$cpp_bin" ] && wsl -e bash -c "test -x $CPP_BIN/$cpp_bin" 2>/dev/null; then
        wsl -e bash -c "mpirun --allow-run-as-root -n 1 $CPP_BIN/$cpp_bin $cpp_args" > "$cout" 2>&1
        echo "  CPP_OK"
    else
        echo "  NO_CPP_BIN ($cpp_bin)"
        return
    fi

    # 比对 DOF 数量
    local rust_dof=$(grep -oP 'dim\(R\)\s*=\s*\K\d+' "$rout" | head -1)
    local cpp_dof=$(grep -oP 'dim\(R\)\s*=\s*\K\d+' "$cout" | head -1)
    if [ -n "$rust_dof" ] && [ -n "$cpp_dof" ]; then
        if [ "$rust_dof" = "$cpp_dof" ]; then
            echo "  [DOF] R: rust=$rust_dof cpp=$cpp_dof ✓"
        else
            echo "  [DOF] R: rust=$rust_dof cpp=$cpp_dof ✗ MISMATCH"
        fi
    fi

    local rust_dof_w=$(grep -oP 'dim\(W\)\s*=\s*\K\d+' "$rout" | head -1)
    local cpp_dof_w=$(grep -oP 'dim\(W\)\s*=\s*\K\d+' "$cout" | head -1)
    if [ -n "$rust_dof_w" ] && [ -n "$cpp_dof_w" ]; then
        if [ "$rust_dof_w" = "$cpp_dof_w" ]; then
            echo "  [DOF] W: rust=$rust_dof_w cpp=$cpp_dof_w ✓"
        else
            echo "  [DOF] W: rust=$rust_dof_w cpp=$cpp_dof_w ✗ MISMATCH"
        fi
    fi

    # 比对 MINRES 迭代数
    local rust_iter=$(grep -oP 'MINRES converged in \K\d+' "$rout" | head -1)
    local cpp_iter=$(grep -oP 'MINRES converged in \K\d+' "$cout" | head -1)
    if [ -n "$rust_iter" ] && [ -n "$cpp_iter" ]; then
        echo "  [ITER] rust=$rust_iter cpp=$cpp_iter"
    fi

    # 比对残差
    local rust_res=$(grep -oP 'residual norm of \K[0-9.eE+-]+' "$rout" | head -1)
    local cpp_res=$(grep -oP 'residual norm of \K[0-9.eE+-]+' "$cout" | head -1)
    if [ -n "$rust_res" ] && [ -n "$cpp_res" ]; then
        echo "  [RES] rust=$rust_res cpp=$cpp_res"
    fi

    # 比对误差
    local rust_err=$(grep -oP '\|\| u_h - u_ex \|\|\s*=\s*\K[0-9.eE+-]+' "$rout" | head -1)
    local cpp_err=$(grep -oP '\|\| u_h - u_ex \|\|\s*=\s*\K[0-9.eE+-]+' "$cout" | head -1)
    if [ -n "$rust_err" ] && [ -n "$cpp_err" ]; then
        echo "  [ERR] rust=$rust_err cpp=$cpp_err"
    fi
}

# 解析参数
if [ "$1" = "--all" ]; then
    miniapps="mini_nurbs_solenoidal mini_joule mini_lorentz mini_maxwell mini_tesla mini_volta mini_nurbs_ex1 mini_nurbs_ex3 mini_nurbs_ex5 mini_nurbs_ex24"
else
    miniapps="$@"
fi

for name in $miniapps; do
    case "$name" in
        mini_nurbs_solenoidal)
            run_miniapp "$name" "-m data/square-nurbs.mesh -o 2 -r 0 -no-vis" \
                "-m \$HOME/mfem49_mpi/data/square-nurbs.mesh -o 2 -r 0 -no-vis" \
                "nurbs_solenoidal"
            ;;
        mini_joule)
            run_miniapp "$name" "-m data/cylinder-hex.mesh -p rod -tf 3 -dt 0.5 -no-vis" \
                "-m \$HOME/mfem49_mpi/miniapps/electromagnetics/cylinder-hex.mesh -p rod -tf 3 -dt 0.5 -no-vis" \
                "joule"
            ;;
        mini_lorentz)
            run_miniapp "$name" "-m data/inline-hex.mesh -no-vis" \
                "-m \$HOME/mfem49_mpi/miniapps/electromagnetics/inline-hex.mesh -no-vis" \
                "lorentz"
            ;;
        mini_maxwell)
            run_miniapp "$name" "-m data/inline-hex.mesh -no-vis" \
                "-m \$HOME/mfem49_mpi/miniapps/electromagnetics/inline-hex.mesh -no-vis" \
                "maxwell"
            ;;
        mini_tesla)
            run_miniapp "$name" "-m data/inline-hex.mesh -no-vis" \
                "-m \$HOME/mfem49_mpi/miniapps/electromagnetics/inline-hex.mesh -no-vis" \
                "tesla"
            ;;
        mini_volta)
            run_miniapp "$name" "-m data/inline-hex.mesh -no-vis" \
                "-m \$HOME/mfem49_mpi/miniapps/electromagnetics/inline-hex.mesh -no-vis" \
                "volta"
            ;;
        mini_nurbs_ex1)
            run_miniapp "$name" "-m data/square-nurbs.mesh -o 2 -no-vis" \
                "-m \$HOME/mfem49_mpi/data/square-nurbs.mesh -o 2 -no-vis" \
                "nurbs_ex1"
            ;;
        mini_nurbs_ex3)
            run_miniapp "$name" "-m data/square-nurbs.mesh -o 2 -r 1 -no-vis" \
                "-m \$HOME/mfem49_mpi/data/square-nurbs.mesh -o 2 -r 1 -no-vis" \
                "nurbs_ex3"
            ;;
        mini_nurbs_ex5)
            run_miniapp "$name" "-m data/square-nurbs.mesh -o 2 -r 1 -no-vis" \
                "-m \$HOME/mfem49_mpi/data/square-nurbs.mesh -o 2 -r 1 -no-vis" \
                "nurbs_ex5"
            ;;
        mini_nurbs_ex24)
            run_miniapp "$name" "-m data/pipe-nurbs-2d.mesh -o 2 -r 1 -p 0 -no-vis" \
                "-m \$HOME/mfem49_mpi/data/pipe-nurbs-2d.mesh -o 2 -r 1 -p 0 -no-vis" \
                "nurbs_ex24"
            ;;
        *)
            echo "Unknown miniapp: $name"
            ;;
    esac
done
