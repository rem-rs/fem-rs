#!/usr/bin/env python3
"""MFEM 示例 1:1 比对工具（Git Bash / Windows 原生）"""
import argparse, json, os, re, subprocess, sys

def paths():
    d = r"C:\Users\lilu\works\fem-pro\fem-rs"
    return {"win": d, "data": os.path.join(d, "data"), "cpp": "/home/quan/mfem49/data"}

def load():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.json")) as f:
        return json.load(f)

def run_rust(p, exe, mesh, args, ranks=None):
    mp = os.path.join(p["data"], mesh)
    if ranks is not None:
        args = args.replace("{ranks}", str(ranks))
    cmd = [os.path.join(p["win"], "target", "release", "examples", exe + ".exe"),
           "-m", mp] + args.split()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=150, cwd=p["win"])
        return r.stdout + "\n" + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def run_cpp(p, name, mesh, args):
    cmd = f"wsl -e bash -c \"timeout 150 ~/bin/{name} -m {p['cpp']}/{mesh} {args} 2>&1\""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
        return r.stdout + "\n" + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def dof(text):
    for pat in [r"Number of finite element unknowns:\s*(\d+)", r"Number of unknowns:\s*(\d+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return None

def evs(text):
    for pat in [r"eigenvalue\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                r"lambda\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                r"Eigenmode\s+\d+/\d+,\s*Lambda\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"]:
        e = re.findall(pat, text, re.IGNORECASE)
        if e: return [float(x) for x in e[:5]]
    return None

def iters(text, solver="CG"):
    for pat in [rf"{solver} converged at iter\s*(\d+)", rf"{solver} converged in (\d+) iterations"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return None

def newton(text):
    m = re.search(r"Number of Newton iterations\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def obj(text):
    m = re.search(r"(?:objective|J)\s*=\s*([-\d.eE]+)", text, re.IGNORECASE)
    return float(m.group(1)) if m else None

def marked(text):
    m = re.search(r"Marked\s*(\d+)\s*elements?", text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def conv_avg(text):
    m = re.search(r"Average reduction factor\s*=\s*([-\d.eE]+)", text)
    return float(m.group(1)) if m else None

def cmp_val(a, b, tol=1e-6):
    if a is None and b is None: return "  ✓ N/A", True
    if a is None: return f"  ✗ rust=None cpp={b}", False
    if b is None: return f"  ✗ rust={a} cpp=None", False
    if isinstance(a, (int,float)) and isinstance(b, (int,float)):
        if a == b: return f"  ✓ ={a}", True
        if b != 0 and abs(a-b)/abs(b) <= tol: return f"  ≈ rel={abs(a-b)/abs(b):.2e}", True
        return f"  ✗ rust={a} cpp={b}", False
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return f"  ✗ len {len(a)} vs {len(b)}", False
        ok = all(cmp_val(x,y,tol)[1] for x,y in zip(a,b))
        return f"  {'✓' if ok else '✗'} {len(a)} values", ok
    return (f"  ✓ ={a}", True) if str(a)==str(b) else (f"  ✗ rust={a} cpp={b}", False)

def compare(name, cfg, p):
    mesh = cfg.get("mesh", "star.mesh")
    ra = cfg.get("args", "-no-vis")
    ca = cfg.get("cpp_args", ra)
    modes = cfg.get("compare", "dof").split("+")
    tol = cfg.get("tolerance", 1e-6)
    par = "np" in cfg

    r = {"name": name, "status": "?", "lines": []}

    if par:
        nps = cfg.get("np", [1,2,4])
        dofs = {}
        for np in nps:
            out, rc = run_rust(p, name, mesh, ra, ranks=np)
            if rc != 0 or "panic" in out.lower():
                r["status"] = "FAIL"
                r["lines"].append(f"  np={np}: exit={rc}")
                return r
            dofs[np] = dof(out)
        s = set(v for v in dofs.values() if v is not None)
        r["status"] = "OK" if len(s)==1 else "DIFF"
        r["lines"].append(f"  DOFs: {dofs}")
        return r

    ro, rc = run_rust(p, name, mesh, ra)
    co, _ = run_cpp(p, name + "_cpp", mesh, ca)

    if rc != 0 or "panic" in ro.lower():
        r["status"] = "FAIL"
        r["lines"].append(f"  rust exit={rc}")
        return r

    rd, cd = dof(ro), dof(co)
    if rd is None and cd is None: r["status"] = "NODATA"
    elif rd is None: r["status"] = "NO_RUST_DOF"
    elif cd is None: r["status"] = "NO_CPP_DOF"
    elif rd == cd:
        r["status"] = "OK"
        r["lines"].append(f"  DOF: {rd}")
    else:
        r["status"] = "DIFF"
        r["lines"].append(f"  DOF: rust={rd} cpp={cd}")

    for m in modes:
        if m == "dof": continue
        if m == "eigenvalue":
            a, b = evs(ro), evs(co)
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  eigenvalue{line}")
        elif m == "iter":
            a, b = iters(ro, "CG"), iters(co, "CG")
            if a is None: a = iters(ro, "MINRES")
            if b is None: b = iters(co, "MINRES")
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  iter{line}")
        elif m == "minres":
            a, b = iters(ro, "MINRES"), iters(co, "MINRES")
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  minres{line}")
        elif m == "fgmres":
            a, b = iters(ro, "FGMRES"), iters(co, "FGMRES")
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  fgmres{line}")
        elif m == "newton":
            a, b = newton(ro), newton(co)
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  newton{line}")
        elif m == "objective":
            a, b = obj(ro), obj(co)
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  objective{line}")
        elif m == "marked":
            a, b = marked(ro), marked(co)
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  marked{line}")
        elif m == "conv_avg":
            a, b = conv_avg(ro), conv_avg(co)
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  conv_avg{line}")
        elif m == "cg":
            a, b = iters(ro, "CG"), iters(co, "CG")
            line, _ = cmp_val(a, b, tol)
            r["lines"].append(f"  cg{line}")

    return r

def report(results):
    print("\n" + "="*80)
    print(f"{'示例':<25} {'状态':<12} {'详情'}")
    print("="*80)
    for r in results:
        status = r["status"]
        icon = {"OK": "✓", "DIFF": "✗", "FAIL": "✗", "NODATA": "?"}.get(status, " ")
        print(f"{r['name']:<25} [{icon}] {status:<10}")
        for l in r["lines"]:
            print(f"  {l}")
    print("="*80)
    ok = sum(1 for r in results if r["status"] == "OK")
    diff = sum(1 for r in results if r["status"] == "DIFF")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"总计: {len(results)} | OK: {ok} | DIFF: {diff} | FAIL: {fail}")

def main():
    ap = argparse.ArgumentParser(description="MFEM 示例 1:1 比对工具")
    ap.add_argument("examples", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cfg = load()
    p = paths()

    if args.list:
        print("\n".join(sorted(k for k in cfg if not k.startswith("_"))))
        return

    keys = [k for k in cfg if not k.startswith("_")] if args.all else args.examples
    results = [compare(k, cfg[k], p) for k in keys if k in cfg]
    report(results)

if __name__ == "__main__":
    main()
