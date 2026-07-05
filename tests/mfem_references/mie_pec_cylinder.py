#!/usr/bin/env python3
"""IEEE 1597 §5.3.1 — Mie series reference for PEC cylinder scattering (TM_z).

Computes the analytical scattered field from a PEC cylinder of radius `a`
for TM_z polarization (E-field parallel to cylinder axis).

Requires: scipy >= 1.0
"""
import json, sys, math
import numpy as np
from scipy.special import jv, yv, hankel1

def mie_pec_cylinder_tm(k: float, a: float, theta: np.ndarray, r: float) -> np.ndarray:
    """Scattered field E_z at (r, theta) for a PEC cylinder of radius a.
    
    Incident: E_inc = exp(-ikx) = exp(-ik·r·cos(θ))
    
    Mie series: E_scat(r,θ) = Σ_{n=-∞}^{∞} a_n · H_n⁽¹⁾(kr) · e^{inθ}
    where a_n = -J_n(ka) / H_n⁽¹⁾(ka) for TM_z PEC cylinder.
    """
    ka = k * a
    kr = k * r
    n_max = int(ka + 4 * (ka ** (1/3)) + 10)  # Wiscombe's criterion
    
    e = np.zeros_like(theta, dtype=complex)
    for n in range(-n_max, n_max + 1):
        jn_ka = jv(abs(n), ka) if n >= 0 else ((-1) ** abs(n)) * jv(abs(n), ka)
        hn_kr = hankel1(abs(n), kr) if n >= 0 else ((-1) ** abs(n)) * hankel1(abs(n), kr)
        hn_ka = hankel1(abs(n), ka) if n >= 0 else ((-1) ** abs(n)) * hankel1(abs(n), ka)
        
        jn_ka_actual = jv(n, ka)
        hn_kr_actual = hankel1(n, kr)
        hn_ka_actual = hankel1(n, ka)
        
        a_n = -jn_ka_actual / hn_ka_actual
        e += a_n * hn_kr_actual * np.exp(1j * n * theta)
    return e


def run_case(k: float, a: float, r_outer: float):
    """Compute field values for a test case."""
    thetas = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    
    # Field on outer boundary
    e_scat = mie_pec_cylinder_tm(k, a, thetas, r_outer)
    incident = np.exp(-1j * k * r_outer * np.cos(thetas))
    total = e_scat + incident
    
    results = {
        "case": f"PEC cylinder: ka={k*a:.2f}, a={a}, k={k}, r_outer={r_outer}",
        "ka": k*a,
        "a": a,
        "k": k,
        "positions": {
            "theta_rad": [float(t) for t in thetas],
            "theta_deg": [float(t*180/np.pi) for t in thetas],
            "r": [float(r_outer)] * len(thetas),
        },
        "scattered_field": {
            "re": [float(z.real) for z in e_scat],
            "im": [float(z.imag) for z in e_scat],
            "mag": [float(abs(z)) for z in e_scat],
        },
        "total_field": {
            "re": [float(z.real) for z in total],
            "im": [float(z.imag) for z in total],
            "mag": [float(abs(z)) for z in total],
        },
    }
    return results


if __name__ == "__main__":
    # Case: ka = 2.0 (k=4, a=0.5), outer measurement at r=2.0
    result = run_case(k=4.0, a=0.5, r_outer=2.0)
    print(json.dumps(result, indent=2))
    
    # Also print a brief summary
    print(f"\nMie series reference for {result['case']}:", file=sys.stderr)
    print(f"  Position θ(deg)  |E_scat|  |E_total|", file=sys.stderr)
    for i, th in enumerate(result['positions']['theta_deg']):
        print(f"  {th:8.1f}      {result['scattered_field']['mag'][i]:8.4f}  {result['total_field']['mag'][i]:8.4f}", file=sys.stderr)
