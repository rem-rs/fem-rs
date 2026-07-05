#!/usr/bin/env python3
"""CEM benchmark: Mie series reference for dielectric cylinder scattering (TMz).

Computes the analytical scattered field from a dielectric cylinder of radius a
with relative permittivity eps_r. Incident: plane wave E_inc = exp(-ikx).

For a dielectric cylinder, the Mie coefficient a_n is:

    a_n = [k2 * J_n(k1*a) * J'_n(k2*a) - k1 * J'_n(k1*a) * J_n(k2*a)]
        / [k1 * Hn'(k1*a) * J_n(k2*a) - k2 * Hn(k1*a) * J'_n(k2*a)]

where k1 = k, k2 = k * sqrt(eps_r).

Requires: scipy >= 1.0, numpy
"""
import json, sys, math
import numpy as np
from scipy.special import jv, hankel1


def jvp(n, z):
    """Derivative J'_n(z) = (J_{n-1}(z) - J_{n+1}(z))/2"""
    return 0.5 * (jv(n - 1, z) - jv(n + 1, z))


def h1vp(n, z):
    """Derivative H_n^{(1)}'(z)"""
    return 0.5 * (hankel1(n - 1, z) - hankel1(n + 1, z))


def mie_dielectric_cylinder(k, a, eps_r, theta, r):
    """Scattered field E_z at (r, theta) for a dielectric cylinder.

    Args:
        k: wavenumber in free space
        a: cylinder radius
        eps_r: relative permittivity of the cylinder
        theta: array of observation angles (radians)
        r: observation radius

    Returns:
        e_scat: complex scattered field array
    """
    k1 = k
    k2 = k * math.sqrt(eps_r)
    ka = k1 * a
    k2a = k2 * a
    kr = k1 * r

    n_max = int(ka + 4.0 * (ka ** (1.0 / 3.0)) + 15)

    e_scat = np.zeros_like(theta, dtype=complex)
    for n in range(-n_max, n_max + 1):
        jn_k1a = jv(n, ka)
        jn_k2a = jv(n, k2a)
        hn_k1a = hankel1(n, ka)
        jnp_k1a = jvp(n, ka)
        jnp_k2a = jvp(n, k2a)
        hnp_k1a = h1vp(n, ka)

        num = k2 * jn_k1a * jnp_k2a - k1 * jnp_k1a * jn_k2a
        den = k1 * hnp_k1a * jn_k2a - k2 * hn_k1a * jnp_k2a
        a_n = num / den

        hn_kr = hankel1(n, kr)
        e_scat += a_n * hn_kr * np.exp(1j * n * theta)
    return e_scat


def run_case(k, a, eps_r, r_outer):
    """Compute fields at 5 angles on the outer boundary."""
    thetas = np.array([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi])
    e_scat = mie_dielectric_cylinder(k, a, eps_r, thetas, r_outer)
    incident = np.exp(-1j * k * r_outer * np.cos(thetas))
    total = e_scat + incident

    results = {
        "case": f"Dielectric cylinder: ka={k*a:.2f}, eps_r={eps_r}",
        "ka": k * a,
        "eps_r": eps_r,
        "a": a,
        "k": k,
        "r_outer": r_outer,
        "positions": {
            "theta_deg": [float(round(t * 180 / math.pi)) for t in thetas],
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
    # Default case: ka=2.0, k=4, a=0.5, eps_r=4, r=2.0
    result = run_case(k=4.0, a=0.5, eps_r=4.0, r_outer=2.0)
    print(json.dumps(result, indent=2))

    print(f"\nMie series reference for {result['case']}:", file=sys.stderr)
    print(f"  theta(deg)  |E_scat|    |E_total|", file=sys.stderr)
    for i, th in enumerate(result['positions']['theta_deg']):
        print(f"  {th:8.1f}    {result['scattered_field']['mag'][i]:8.6f}  {result['total_field']['mag'][i]:8.6f}", file=sys.stderr)
