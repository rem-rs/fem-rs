#!/usr/bin/env python3
"""MFEM 示例 1:1 比对工具

用法:
    python3 compare.py --all              # 跑全部示例
    python3 compare.py ex1 ex11 ex27      # 跑指定示例
    python3 compare.py --list             # 列出所有示例
    python3 compare.py --summary          # 汇总报告

运行环境: WSL (python3 可用)
"""
import argparse
import os
import platform
import re
import subprocess
import sys
import toml

# ─── 路径配置 ─────────────────────────────────────────────────────────────────

def detect_paths():
    """自动检测运行环境，返回路径配置"""
    # 检测是否在 WSL 中
    try:
        with open("/proc/version", "r") as f:
            version = f.read().lower()
        in_wsl = "microsoft" in version or "wsl" in version
    except:
        in_wsl = False
    
    if in_wsl:
        # WSL 环境：通过 /mnt/c 访问 Windows 文件
        win_dir = "/mnt/c/Users/lilu/works/fem-pro/fem-rs"
    else:
        # Windows 原生环境
        win_dir = r"C:\Users\lilu\works\fem-pro\fem-rs"
    
    data_win = os.path.join(win_dir, "data")
    data_cpp = "/home/quan/mfem49/data"
    
    return {
        "win_dir": win_dir,
        "data_win": data_win,
        "data_cpp": data_cpp,
    }


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.toml")
    with open(config_path, "r", encoding="utf-8") as f:
        return toml.load(f)


def run_rust(paths, exe, mesh, args, ranks=None):
    """运行 Rust .exe，返回 (stdout+stderr, returncode)"""
    mesh_path = os.path.join(paths["data_win"], mesh)
    if ranks is not None:
        args = args.replace("{ranks}", str(ranks))
    exe_path = os.path.join(paths["win_dir"], "target", "release", "examples", f"{exe}.exe")
    cmd = [exe_path, "-m", mesh_path] + args.split()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=150, cwd=paths["win_dir"])
        return r.stdout + "\n" + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1


def run_cpp(paths, bin_name, mesh, args):
    """运行 C++ 参考（通过 WSL），返回 (stdout+stderr, returncode)"""
    mesh_cpp = os.path.join(paths["data_cpp"], mesh)
    cmd = f"timeout 150 ~/bin/{bin_name} -m {mesh_cpp} {args}"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
        return r.stdout + "\n" + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1


# ─── 提取器 ───────────────────────────────────────────────────────────────────

def extract_dof(text):
    """提取 DOF 数"""
    patterns = [
        r"Number of finite element unknowns:\s*(\d+)",
        r"Number of unknowns:\s*(\d+)",
        r"Number of Unknowns:\s*(\d+)",
        r"unknowns:\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_iter(text, solver="CG"):
    """提取迭代数"""
    patterns = {
        "CG": [r"CG converged at iter\s*(\d+)", r"CG converged in (\d+) iterations"],
        "MINRES": [r"MINRES converged iter\s*(\d+)", r"MINRES converged in (\d+) iterations"],
        "GMRES": [r"GMRES converged at iter\s*(\d+)", r"GMRES converged in (\d+) iterations"],
        "FGMRES": [r"FGMRES converged at iter\s*(\d+)", r"FGMRES converged in (\d+) iterations"],
        "Newton": [r"Newton iteration\s+(\d+)\s*:"],
    }
    for pat in patterns.get(solver, []):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_eigenvalues(text):
    """提取特征值"""
    patterns = [
        r"eigenvalue\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"lambda\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"EV\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"Eigenmode\s+\d+/\d+,\s*Lambda\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"Eigenvalue\s+lambda\s+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
    ]
    for pat in patterns:
        evs = re.findall(pat, text, re.IGNORECASE)
        if evs:
            return [float(x) for x in evs[:5]]
    return None


def extract_conv_avg(text):
    """提取平均缩减因子"""
    m = re.search(r"Average reduction factor\s*=\s*([-\d.eE]+)", text)
    if m:
        return float(m.group(1))
    return None


def extract_marked(text):
    """提取标记元素数"""
    m = re.search(r"Marked\s*(\d+)\s*elements?", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def extract_objective(text):
    """提取目标函数值"""
    patterns = [r"objective\s*=\s*([-\d.eE]+)", r"J\s*=\s*([-\d.eE]+)"]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def extract_residual(text):
    """提取残差"""
    patterns = [
        r"\|\|r\|\|/\|\|b\|\|\s*=\s*([-\d.eE]+)",
        r"residual\s*=\s*([-\d.eE]+)",
        r"Final L2-error\s*\(.*?\)\s*=\s*([-\d.eE]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def extract_energy(text):
    """提取能量"""
    patterns = [r"EE\s*=\s*([-\d.eE]+)", r"KE\s*=\s*([-\d.eE]+)", r"ΔTE\s*=\s*([-\d.eE]+)"]
    result = {}
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            result[pat.split("\\")[0].strip()] = float(m.group(1))
    return result if result else None


# ─── 比对器 ───────────────────────────────────────────────────────────────────

def compare_values(rust_val, cpp_val, tolerance=1e-6, label=""):
    """比对两个值，返回 (是否一致, 差异描述)"""
    if rust_val is None and cpp_val is None:
        return True, "N/A"
    if rust_val is None:
        return False, f"rust=None cpp={cpp_val}"
    if cpp_val is None:
        return False, f"rust={rust_val} cpp=None"
    
    if isinstance(rust_val, (int, float)) and isinstance(cpp_val, (int, float)):
        if rust_val == cpp_val:
            return True, f"={rust_val}"
        if cpp_val != 0:
            rel_diff = abs(rust_val - cpp_val) / abs(cpp_val)
            if rel_diff <= tolerance:
                return True, f"rust={rust_val} cpp={cpp_val} (rel={rel_diff:.2e})"
            return False, f"rust={rust_val} cpp={cpp_val} (rel={rel_diff:.2e})"
        else:
            if abs(rust_val) <= tolerance:
                return True, f"rust={rust_val} cpp=0"
            return False, f"rust={rust_val} cpp=0"
    
    if isinstance(rust_val, list) and isinstance(cpp_val, list):
        if len(rust_val) != len(cpp_val):
            return False, f"len(rust)={len(rust_val)} len(cpp)={len(cpp_val)}"
        all_ok = True
        diffs = []
        for i, (rv, cv) in enumerate(zip(rust_val, cpp_val)):
            ok, diff = compare_values(rv, cv, tolerance, f"[{i}]")
            if not ok:
                all_ok = False
            diffs.append(diff)
        return all_ok, "; ".join(diffs)
    
    return str(rust_val) == str(cpp_val), f"rust={rust_val} cpp={cpp_val}"


# ─── 主比对逻辑 ───────────────────────────────────────────────────────────────

def run_comparison(name, config, verbose=True):
    """运行单个示例的比对，返回结果字典"""
    mesh = config.get("mesh", "star.mesh")
    rust_args = config.get("args", "-no-vis")
    cpp_args = config.get("cpp_args", rust_args)
    compare_modes = config.get("compare", "dof").split("+")
    tolerance = config.get("tolerance", 1e-6)
    is_parallel = "np" in config
    
    result = {
        "name": name,
        "mesh": mesh,
        "status": "UNKNOWN",
        "details": [],
        "rust_fail": False,
    }
    
    if is_parallel:
        # 并行模式：跑 np=1,2,4
        np_list = config.get("np", [1, 2, 4])
        rust_outputs = {}
        rust_dofs = {}
        all_ok = True
        
        for np in np_list:
            out, rc = run_rust(name, mesh, rust_args, ranks=np)
            rust_outputs[np] = out
            if rc != 0 or "panic" in out.lower():
                result["rust_fail"] = True
                result["status"] = "RUST_FAIL"
                result["details"].append(f"np={np}: panic or exit={rc}")
                all_ok = False
                continue
            rust_dofs[np] = extract_dof(out)
        
        if not result["rust_fail"]:
            # 检查 np1=np2=np4 一致性
            dof_set = set(v for v in rust_dofs.values() if v is not None)
            if len(dof_set) == 1:
                result["status"] = "OK"
                result["details"].append(f"np1=np2=np4 DOF={dof_set.pop()}")
            elif len(dof_set) == 0:
                result["status"] = "NO_DOF"
            else:
                result["status"] = "MISMATCH"
                result["details"].append(f"DOFs differ: {rust_dofs}")
        
        # C++ 比对（np=1）
        cpp_out, cpp_rc = run_cpp(name.replace("pex", "ex") + "p_cpp", mesh, cpp_args)
        cpp_dof = extract_dof(cpp_out)
        if cpp_dof is not None and 1 in rust_dofs and rust_dofs[1] is not None:
            ok, diff = compare_values(rust_dofs[1], cpp_dof, tolerance, "dof")
            result["details"].append(f"np1 vs C++ DOF: {diff}")
            if not ok and result["status"] == "OK":
                result["status"] = "MISMATCH_CPP"
    else:
        # 串行模式
        rust_out, rust_rc = run_rust(name, mesh, rust_args)
        cpp_out, cpp_rc = run_cpp(name + "_cpp", mesh, cpp_args)
        
        if rust_rc != 0 or "panic" in rust_out.lower():
            result["rust_fail"] = True
            result["status"] = "RUST_FAIL"
            return result
        
        # 提取 DOF
        rust_dof = extract_dof(rust_out)
        cpp_dof = extract_dof(cpp_out)
        
        if rust_dof is None and cpp_dof is None:
            result["status"] = "NO_DOF"
        elif rust_dof is None:
            result["status"] = "NO_DOF_RUST"
        elif cpp_dof is None:
            result["status"] = "NO_DOF_CPP"
        elif rust_dof == cpp_dof:
            result["status"] = "OK"
            result["details"].append(f"DOF={rust_dof}")
        else:
            result["status"] = "MISMATCH"
            result["details"].append(f"DOF rust={rust_dof} cpp={cpp_dof}")
        
        # 比对其他指标
        for mode in compare_modes:
            if mode == "dof":
                continue
            elif mode == "iter":
                rv = extract_iter(rust_out, "CG")
                cv = extract_iter(cpp_out, "CG")
                if rv is None:
                    rv = extract_iter(rust_out, "MINRES")
                    cv = extract_iter(cpp_out, "MINRES")
                ok, diff = compare_values(rv, cv, tolerance, "iter")
                result["details"].append(f"iter: {diff}")
            elif mode == "eigenvalue":
                rv = extract_eigenvalues(rust_out)
                cv = extract_eigenvalues(cpp_out)
                ok, diff = compare_values(rv, cv, tolerance, "eigenvalue")
                result["details"].append(f"eigenvalue: {diff}")
            elif mode == "conv_avg":
                rv = extract_conv_avg(rust_out)
                cv = extract_conv_avg(cpp_out)
                ok, diff = compare_values(rv, cv, tolerance, "conv_avg")
                result["details"].append(f"conv_avg: {diff}")
            elif mode == "marked":
                rv = extract_marked(rust_out)
                cv = extract_marked(cpp_out)
                ok, diff = compare_values(rv, cv, tolerance, "marked")
                result["details"].append(f"marked: {diff}")
            elif mode == "objective":
                rv = extract_objective(rust_out)
                cv = extract_objective(cpp_out)
                ok, diff = compare_values(rv, cv, tolerance, "objective")
                result["details"].append(f"objective: {diff}")
            elif mode == "newton":
                rv = extract_iter(rust_out, "Newton")
                cv = extract_iter(cpp_out, "Newton")
                ok, diff = compare_values(rv, cv, tolerance, "newton")
                result["details"].append(f"newton: {diff}")
            elif mode == "cg":
                rv = extract_iter(rust_out, "CG")
                cv = extract_iter(cpp_out, "CG")
                ok, diff = compare_values(rv, cv, tolerance, "cg")
                result["details"].append(f"cg: {diff}")
            elif mode == "minres":
                rv = extract_iter(rust_out, "MINRES")
                cv = extract_iter(cpp_out, "MINRES")
                ok, diff = compare_values(rv, cv, tolerance, "minres")
                result["details"].append(f"minres: {diff}")
            elif mode == "fgmres":
                rv = extract_iter(rust_out, "FGMRES")
                cv = extract_iter(cpp_out, "FGMRES")
                ok, diff = compare_values(rv, cv, tolerance, "fgmres")
                result["details"].append(f"fgmres: {diff}")
            elif mode == "sol_gf":
                result["details"].append("sol_gf: 需手动运行专用比对脚本")
    
    return result


# ─── 报告 ─────────────────────────────────────────────────────────────────────

def print_report(results):
    """打印汇总报告"""
    print("\n" + "=" * 80)
    print(f"{'示例':<25} {'状态':<15} {'详情'}")
    print("=" * 80)
    
    ok_count = sum(1 for r in results if r["status"] in ("OK",))
    fail_count = len(results) - ok_count
    
    for r in results:
        status = r["status"]
        details = "; ".join(r["details"]) if r["details"] else ""
        print(f"{r['name']:<25} {status:<15} {details}")
    
    print("=" * 80)
    print(f"总计: {len(results)} | OK: {ok_count} | 其他: {fail_count}")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MFEM 示例 1:1 比对工具")
    parser.add_argument("examples", nargs="*", help="要比对的示例名称（如 ex1 ex11）")
    parser.add_argument("--all", action="store_true", help="比对全部示例")
    parser.add_argument("--list", action="store_true", help="列出所有示例")
    parser.add_argument("--summary", action="store_true", help="汇总报告")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    config = load_config()
    general = config.pop("general", {})
    
    if args.list:
        print("可用示例：")
        for name in sorted(config.keys()):
            print(f"  {name}")
        return
    
    if args.all:
        examples = list(config.keys())
    elif args.examples:
        examples = args.examples
    else:
        parser.print_help()
        return
    
    results = []
    for name in examples:
        if name not in config:
            print(f"警告: {name} 不在配置中，跳过")
            continue
        print(f"\n--- {name} ---")
        result = run_comparison(name, config[name], verbose=args.verbose)
        results.append(result)
        print(f"  状态: {result['status']}")
        for d in result["details"]:
            print(f"    {d}")
    
    print_report(results)


if __name__ == "__main__":
    main()
