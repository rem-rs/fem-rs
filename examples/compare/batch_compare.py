#!/usr/bin/env python3
"""Batch compare all MFEM examples: Rust (Windows exe) vs C++ (WSL)."""
import json, os, re, subprocess, sys

BASE = "/mnt/c/Users/lilu/works/fem-pro"
DATA_WIN = "C:\\Users\\lilu\\works\\fem-pro\\data"
DATA_CPP = "/home/quan/mfem410/data"
CPP_BIN = "/home/quan/bin/410"
EXE_DIR = "C:\\Users\\lilu\\works\\fem-pro\\fem-rs\\target\\debug\\examples"

with open(BASE + "/fem-rs/examples/compare/examples.json") as f:
    CFG = json.load(f)

RMAP = {"ex0":"mfem_ex0_mesh_intro","ex1":"mfem_ex1_poisson","ex2":"mfem_ex2_elasticity","ex3":"mfem_ex3_maxwell_cavity","ex4":"mfem_ex4_darcy","ex5":"mfem_ex5_mixed_darcy","ex6":"mfem_ex6_flux_recovery","ex7":"mfem_ex7_surface_poisson","ex8":"mfem_ex8_dpg_2x2","ex9":"mfem_ex9_dg_advection","ex10":"mfem_ex10_hyperelastic_dyn","ex11":"mfem_ex11_eigenvalue","ex12":"mfem_ex12_elastic_eigen","ex13":"mfem_ex13_eigenvalue","ex14":"mfem_ex14_dg_poisson","ex15":"mfem_ex15_dynamic_amr","ex16":"mfem_ex16_nonlinear_heat","ex17":"mfem_ex17_dg_elasticity","ex18":"mfem_ex18_euler","ex19":"mfem_ex19_hyperelastic_incomp","ex20":"mfem_ex20_symplectic","ex21":"mfem_ex21_amr_elasticity","ex22":"mfem_ex22_complex_helmholtz","ex23":"mfem_ex23_wave_equation","ex24":"mfem_ex24_discrete_ops","ex25":"mfem_ex25_pml_maxwell","ex26":"mfem_ex26_geom_mg","ex27":"mfem_ex27_robin_bc","ex28":"mfem_ex28_sliding_elasticity","ex29":"mfem_ex29_curved_poisson","ex30":"mfem_ex30_aniso_amr","ex31":"mfem_ex31_anisotropic_maxwell","ex32":"mfem_ex32_maxwell_eigenvalue","ex33":"mfem_ex33_fractional_diffusion","ex34":"mfem_ex34_magnetostatics","ex35":"mfem_ex35_complex_oscillator","ex36":"mfem_ex36_obstacle","ex37":"mfem_ex37_topology_optimization","ex38":"mfem_ex38_implicit_integration","ex39":"mfem_ex39_compass","ex40":"mfem_ex40_eikonal","ex41":"mfem_ex41_imex"}
PMAP = {"pex1":"mfem_pex1_parallel_poisson","pex2":"mfem_pex2_parallel_elasticity","pex3":"mfem_pex3_maxwell_cavity","pex4":"mfem_pex4_parallel_hdiv_diffusion","pex5":"mfem_pex5_hdiv_darcy","pex6":"mfem_pex6_parallel_amr","pex7":"mfem_pex7_parallel_surface","pex8":"mfem_pex8_parallel_dpg","pex9":"mfem_pex9_parallel_dg_advection","pex10":"mfem_pex10_parallel_hyperelastic","pex11":"mfem_pex11_parallel_eigenvalue","pex12":"mfem_pex12_parallel_elastic_eigen","pex13":"mfem_pex13_parallel_eigenvalue","pex14":"mfem_pex14_parallel_dg_poisson","pex15":"mfem_pex15_parallel_dynamic_amr","pex16":"mfem_pex16_parallel_nonlinear_heat","pex17":"mfem_pex17_parallel_dg_elasticity","pex18":"mfem_pex18_parallel_euler","pex19":"mfem_pex19_parallel_incomp_hyperelastic","pex20":"mfem_pex20_parallel_symplectic","pex21":"mfem_pex21_parallel_amr_elasticity","pex22":"mfem_pex22_parallel_complex_helmholtz","pex24":"mfem_pex24_parallel_discrete_ops","pex25":"mfem_pex25_pml_maxwell","pex26":"mfem_pex26_parallel_geom_mg","pex27":"mfem_pex27_parallel_robin_bc","pex28":"mfem_pex28_parallel_sliding_elasticity","pex29":"mfem_pex29_surface_poisson","pex30":"mfem_pex30_amr_preprocess","pex31":"mfem_pex31_restricted_hcurl","pex32":"mfem_pex32_maxwell_eigenvalue","pex33":"mfem_pex33_fractional_laplacian","pex34":"mfem_pex34_magnetostatics","pex35":"mfem_pex35_complex_oscillator","pex36":"mfem_pex36_obstacle","pex37":"mfem_pex37_topology_optimization","pex39":"mfem_pex39_named_attributes","pex40":"mfem_pex40_eikonal","pex41":"mfem_pex41_imex"}

def rust_exe(n):
    return PMAP.get(n, RMAP.get(n, f"mfem_{n}"))

def dof(t):
    for p in [r"Number of finite element unknowns:\s*(\d+)", r"Number of unknowns:\s*(\d+)"]:
        m = re.search(p, t, re.I)
        if m: return int(m.group(1))
    return None

def conv(t):
    m = re.search(r"Average reduction factor\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", t)
    return float(m.group(1)) if m else None

def iters(t):
    n = re.findall(r"Iteration\s*:\s*(\d+)", t)
    return int(n[-1])+1 if n else None

def run_rust(exe, mesh, args, ranks=None):
    ep = os.path.join(EXE_DIR, exe + ".exe")
    if not os.path.isfile(ep): return f"MISSING: {exe}", -1
    mp = os.path.join(DATA_WIN, mesh)
    # Use cmd.exe to run Windows exe with Windows paths
    cmd = f'cmd.exe /c "{ep}" -m "{mp}" {args}'
    if ranks: cmd += f" --ranks {ranks}"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or ""), r.returncode
    except subprocess.TimeoutExpired: return "TIMEOUT", -1

def run_cpp(name, mesh, args):
    cpp = os.path.join(CPP_BIN, name + "_cpp")
    if not os.path.isfile(cpp): return f"MISSING: {cpp}", -1
    try:
        cmd = f"timeout 60 {cpp} -m {DATA_CPP}/{mesh} {args} 2>&1"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or ""), r.returncode
    except subprocess.TimeoutExpired: return "TIMEOUT", -1

def cmp_one(name, cfg):
    mesh = cfg.get("mesh", "star.mesh")
    ra = cfg.get("args", "")
    ca = cfg.get("cpp_args", ra)
    modes = cfg.get("compare", "dof").split("+")
    par = "np" in cfg
    exe = rust_exe(name)
    r = {"name": name, "status": "?", "lines": []}

    if par:
        dofs = {}
        for np in cfg.get("np", [1,2,4]):
            out, rc = run_rust(exe, mesh, ra, ranks=np)
            if rc != 0 or "panic" in out.lower():
                r["status"] = "FAIL"
                r["lines"].append(f"np={np} exit={rc}")
                for l in out.split("\n"):
                    if any(x in l.lower() for x in ["panic","error","failed"]):
                        r["lines"].append(f"  >> {l.strip()[:80]}")
                return r
            dofs[np] = dof(out)
        s = set(v for v in dofs.values() if v is not None)
        r["status"] = "OK" if len(s)==1 else "DIFF"
        r["lines"].append(f"DOFs: {dofs}")
        return r

    ro, rc = run_rust(exe, mesh, ra)
    if rc != 0 or "panic" in ro.lower():
        r["status"] = "FAIL"
        r["lines"].append(f"exit={rc}")
        for l in ro.split("\n"):
            if any(x in l.lower() for x in ["panic","error","failed","assertion"]):
                r["lines"].append(f"  >> {l.strip()[:80]}")
        return r
    co, _ = run_cpp(name, mesh, ca)
    rd = dof(ro); cd = dof(co)

    if "dof" in modes:
        r["lines"].append(f"DOF: {rd} =={cd}" if rd==cd else f"DOF DIFF: r={rd} c={cd}")
    if "conv_avg" in modes:
        rv = conv(ro); cv = conv(co)
        if rv and cv: r["lines"].append(f"conv: {'OK' if abs(rv-cv)<1e-5 else 'DIFF'} r={rv} c={cv}")
    if "iter" in modes:
        ri = iters(ro); ci = iters(co)
        if ri and ci: r["lines"].append(f"iter: {'OK' if abs(ri-ci)<=2 else 'DIFF'} r={ri} c={ci}")

    st = []
    for l in r["lines"]:
        if "DIFF" in l or "FAIL" in l: st.append("D")
        elif "OK" in l or re.search(r'\d+ ==\d+', l): st.append("O")
    r["status"] = "OK" if st and all(s=="O" for s in st) else ("DIFF" if st else "N/A")
    if not r["lines"]:
        r["lines"].append(f"rd={rd} cd={cd}")
        r["status"] = "OK" if rd==cd else "DIFF"
    return r

def main():
    keys = [k for k in CFG if not k.startswith("_") and k != "general"]
    res = []
    for k in keys:
        r = cmp_one(k, CFG[k])
        res.append(r)
        ic = {"OK":"OK","DIFF":"DIFF","FAIL":"FAIL"}.get(r["status"],"?")
        print(f"{k:25s} [{ic}] {' | '.join(r['lines'][:3])}", flush=True)
    ok = sum(1 for r in res if r["status"]=="OK")
    df = sum(1 for r in res if r["status"]=="DIFF")
    fl = sum(1 for r in res if r["status"]=="FAIL")
    print(f"\n总计: {len(res)} | OK: {ok} | DIFF: {df} | FAIL: {fl}")

if __name__ == "__main__":
    main()
