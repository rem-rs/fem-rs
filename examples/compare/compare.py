#!/usr/bin/env python3
"""MFEM 示例 1:1 比对工具"""
import argparse, json, os, re, subprocess, sys

def detect_paths():
    try:
        with open("/proc/version") as f:
            in_wsl = "microsoft" in f.read().lower()
    except:
        in_wsl = False
    win_dir = "/mnt/c/Users/lilu/works/fem-pro/fem-rs" if in_wsl else r"C:\Users\lilu\works\fem-pro\fem-rs"
    return {"win_dir": win_dir, "data_win": os.path.join(win_dir, "data"), "data_cpp": "/home/quan/mfem49/data"}

def load_config():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    toml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.toml")
    if os.path.exists(toml_path):
        try:
            import toml
            with open(toml_path) as f:
                return toml.load(f)
        except ImportError:
            pass
    sys.exit("ERROR: No examples.json found and toml module not available (pip install toml)")

def run_rust(paths, exe, mesh, args, ranks=None):
    mesh_path = os.path.join(paths["data_win"], mesh)
    if ranks is not None:
        args = args.replace("{ranks}", str(ranks))
    cmd = [os.path.join(paths["win_dir"], "target/release/examples", f"{exe}.exe"), "-m", mesh_path] + args.split()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=150, cwd=paths["win_dir"])
        return r.stdout + "\n" + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def run_cpp(paths, bin_name, mesh, args):
    cmd = f"timeout 150 ~/bin/{bin_name} -m {paths['data_cpp']}/{mesh} {args}"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
        return r.stdout + "\n" + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def extract_dof(text):
    for pat in [r"Number of finite element unknowns:\s*(\d+)", r"Number of unknowns:\s*(\d+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return None

def extract_eigenvalues(text):
    for pat in [r"eigenvalue\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", r"lambda\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", r"Eigenmode\s+\d+/\d+,\s*Lambda\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"]:
        evs = re.findall(pat, text, re.IGNORECASE)
        if evs: return [float(x) for x in evs[:5]]
    return None

def compare_values(rv, cv, tol=1e-6):
    if rv is None and cv is None: return True, "N/A"
    if rv is None: return False, f"rust=None cpp={cv}"
    if cv is None: return False, f"rust={rv} cpp=None"
    if isinstance(rv, (int,float)) and isinstance(cv, (int,float)):
        if rv == cv: return True, f"={rv}"
        if cv != 0 and abs(rv-cv)/abs(cv) <= tol: return True, f"rel={abs(rv-cv)/abs(cv):.2e}"
        return False, f"rust={rv} cpp={cv}"
    if isinstance(rv, list) and isinstance(cv, list):
        if len(rv) != len(cv): return False, f"len {len(rv)} vs {len(cv)}"
        ok = all(compare_values(a,b,tol)[0] for a,b in zip(rv,cv))
        return ok, f"{'OK' if ok else 'MISMATCH'}"
    return str(rv)==str(cv), f"rust={rv} cpp={cv}"

def run_comparison(name, cfg):
    mesh = cfg.get("mesh", "star.mesh")
    rust_args = cfg.get("args", "-no-vis")
    cpp_args = cfg.get("cpp_args", rust_args)
    modes = cfg.get("compare", "dof").split("+")
    tol = cfg.get("tolerance", 1e-6)
    is_par = "np" in cfg
    paths = detect_paths()
    
    result = {"name": name, "status": "UNKNOWN", "details": []}
    
    if is_par:
        np_list = cfg.get("np", [1,2,4])
        rust_dofs = {}
        for np in np_list:
            out, rc = run_rust(paths, name, mesh, rust_args, ranks=np)
            if rc != 0 or "panic" in out.lower():
                result["status"] = "RUST_FAIL"
                result["details"].append(f"np={np}: exit={rc}")
                return result
            rust_dofs[np] = extract_dof(out)
        dof_set = set(v for v in rust_dofs.values() if v is not None)
        result["status"] = "OK" if len(dof_set) == 1 else "MISMATCH"
        result["details"].append(f"DOFs={rust_dofs}")
    else:
        rust_out, rc = run_rust(paths, name, mesh, rust_args)
        cpp_out, _ = run_cpp(paths, name + "_cpp", mesh, cpp_args)
        if rc != 0 or "panic" in rust_out.lower():
            result["status"] = "RUST_FAIL"
            return result
        rd, cd = extract_dof(rust_out), extract_dof(cpp_out)
        if rd is None and cd is None: result["status"] = "NO_DOF"
        elif rd is None: result["status"] = "NO_DOF_RUST"
        elif cd is None: result["status"] = "NO_DOF_CPP"
        elif rd == cd: result["status"] = "OK"
        else: result["status"] = "MISMATCH"
        result["details"].append(f"DOF rust={rd} cpp={cd}")
        if "eigenvalue" in modes:
            rv, cv = extract_eigenvalues(rust_out), extract_eigenvalues(cpp_out)
            ok, diff = compare_values(rv, cv, tol)
            result["details"].append(f"eigenvalue: {diff}")
    return result

def main():
    parser = argparse.ArgumentParser(description="MFEM 比对工具")
    parser.add_argument("examples", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.list:
        print("\n".join(sorted(config.keys())))
        return
    keys = list(config.keys()) if args.all else args.examples
    results = [run_comparison(k, config[k]) for k in keys if k in config]
    print(f"\n{'示例':<25} {'状态':<15} {'详情'}")
    print("="*80)
    for r in results:
        print(f"{r['name']:<25} {r['status']:<15} {'; '.join(r['details'])}")
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n总计: {len(results)} | OK: {ok}")

if __name__ == "__main__":
    main()
