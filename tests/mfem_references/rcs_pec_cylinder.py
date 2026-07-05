#!/usr/bin/env python3
"""RCS verification for 2D PEC cylinder — Mie series + NFFFT reference.

Computes:
  1. Mie series RCS for a PEC cylinder (TM_z polarization)
  2. Backscatter RCS (monostatic) at ka = 0.5, 1.0, 2.0
  3. Bistatic RCS pattern at selected ka
  4. Reference values for regression baselines

Usage:
  uv run python tests/mfem_references/rcs_pec_cylinder.py
"""
import json, sys, math
import numpy as np
from scipy.special import hankel1, jv


def mie_coefficient(n: int, ka: float) -> complex:
    """Mie coefficient a_n for PEC cylinder, TM_z polarization."""
    jn = jv(n, ka)
    hn = hankel1(n, ka)
    if abs(hn) < 1e-300:
        return 0j
    return -jn / hn


def rcs_2d_mie(theta: np.ndarray, ka: float, n_max: int | None = None) -> np.ndarray:
    """2D RCS / λ for a PEC cylinder at observation angles theta (radians).
    
    The 2D RCS per unit length normalized by wavelength:
    σ_2D/λ = (2/π) · |Σ_{n=-∞}^{∞} a_n · e^{inθ}|²
    """
    if n_max is None:
        n_max = int(ka + 4 * (ka ** (1/3)) + 10)  # Wiscombe criterion
    
    s = np.zeros_like(theta, dtype=complex)
    for n in range(-n_max, n_max + 1):
        an = mie_coefficient(n, ka)
        s += an * np.exp(1j * n * theta)
    
    return (2.0 / math.pi) * np.abs(s) ** 2


def backscatter_rcs(ka: float) -> float:
    """Monostatic RCS (θ=π) / λ for PEC cylinder at given ka."""
    return float(rcs_2d_mie(np.array([math.pi]), ka)[0])


def main():
    results = {}
    
    # Monostatic RCS at various ka values
    print("Monostatic (backscatter) RCS / λ for PEC cylinder TM_z:", file=sys.stderr)
    print("  ka    σ/λ", file=sys.stderr)
    print("  ----  ------", file=sys.stderr)
    
    monostatic = {}
    for ka in [0.5, 1.0, 2.0, 3.0, 5.0]:
        rcs = backscatter_rcs(ka)
        monostatic[f"ka_{ka}"] = rcs
        print(f"  {ka:<5.1f} {rcs:.6f}", file=sys.stderr)
    
    results["monostatic_rcs"] = monostatic
    
    # Bistatic RCS pattern for ka = 2.0 (matching our FEM test)
    thetas = np.linspace(0, 2 * math.pi, 361)
    rcs_pattern = rcs_2d_mie(thetas, 2.0)
    
    # Sample at key angles
    key_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180]
    bistatic = {}
    for deg in key_angles:
        rad = deg * math.pi / 180
        idx = int(deg)
        bistatic[f"theta_{deg}"] = float(rcs_pattern[idx])
    
    # Average RCS
    avg_rcs = float(np.mean(rcs_pattern))
    bistatic["average"] = avg_rcs
    results["bistatic_rcs_ka2"] = bistatic
    
    print(f"\nBistatic RCS/λ at ka=2:", file=sys.stderr)
    for deg in key_angles:
        print(f"  θ={deg}°   σ/λ = {bistatic[f'theta_{deg}']:.6f}", file=sys.stderr)
    print(f"  average σ/λ = {avg_rcs:.6f}", file=sys.stderr)
    
    # Export for Rust test reference
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
