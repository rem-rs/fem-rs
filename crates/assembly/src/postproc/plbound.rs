//! Piecewise-linear bounds of a grid function — 1:1 port of MFEM 4.10
//! `fem/bounds.cpp` (`PLBound`) plus the bounds half of `fem/gridfunc.cpp`
//! (`GetElementBounds*`, `EstimateFunctionMinimum/Maximum`, and the
//! `GridFunction::ProjectGridFunction` / `NodalFiniteElement::Project` pair
//! needed by `miniapps/tools/gridfunction-bounds.cpp`'s `-bt` option).
//!
//! # Public API surface
//!
//! * [`PLBound`] — the piecewise-linear bounding basis machinery
//!   (`mfem::PLBound`, `cp_type = 0` GL+end-points control points, `tol = 0`):
//!   - [`PLBound::n_control_points`] (`GetNControlPoints`),
//!   - the 1-D control-point positions via [`PLBound::control_points`]
//!     (`GetControlPoints`).
//!   Constructors mirror `PLBound(const FiniteElementSpace *, ncp_i,
//!   cp_type_i)`'s FEC-name dispatch:
//!   - [`PLBound::from_h1_order`] — `"H1_*"` → GLL bases, nodes are the H1
//!     reference element's own 1-D nodes;
//!   - [`PLBound::from_l2`] — `"L2_*"` → GL nodes (`BoundsBasis::
//!     GaussLegendre`) or `"L2_T1_*"` → GLL nodes, nodes/weights from the
//!     MFEM 1-D rules ([`mfem_gauss_legendre_01`] / [`mfem_gauss_lobatto_01`]).
//!   The Bernstein/positive bases (`"H1Pos_*"`, `"L2_T2_*"`, `b_type = 2`)
//!   are **not** ported — fem-rs has no positive basis; the miniapp rejects
//!   them up front (documented gap).
//! * [`get_element_bounds`] / [`estimate_function_minimum`] /
//!   [`estimate_function_maximum`] — H1 (GLL) entry points, byte-compatible
//!   with the round-41 miniapp `mod plbound`.
//! * [`get_element_bounds_in`] / [`estimate_function_minimum_in`] /
//!   [`estimate_function_maximum_in`] — the same machinery for an explicit
//!   [`BoundsSpace`] (H1 GLL or L2 GL/GLL), used by the `-l2`/`-bt` tiers.
//! * [`project_h1_to_l2`] — `GridFunction::ProjectGridFunction(src)` for a
//!   discontinuous tensor target: per-element nodal interpolation of the H1
//!   (GLL) source onto the target's 1-D node positions
//!   (`NodalFiniteElement::Project`, including its `< 1e-12 → 0` snap), with
//!   the target DOFs laid out element-major lexicographic (fem-rs
//!   `L2Space`'s element-continuous numbering).
//! * [`h1_tensor_nodes`] — the H1 tensor dof map + 1-D nodes of a reference
//!   quad/hex (MFEM `TensorBasisElement::GetDofMap`), exported for tests.
//! * [`mfem_gauss_legendre_01`] / [`mfem_gauss_lobatto_01`] — bit-exact ports
//!   of `QuadratureFunctions1D::GaussLegendre` / `GaussLobatto` (`fem/
//!   intrules.cpp` L620–706 / L708–838).  These replace fem-element's
//!   generic Newton solvers inside this module: D259 measured fem-rs'
//!   `gauss_legendre_01_arbitrary` ~1 ulp off MFEM's for `n ≥ 6` (it maps
//!   `[-1,1]` results through `0.5*(x+1)`, while MFEM solves directly on
//!   `[0,1]` with the `xi = ((1-z)+dz)/2` round-off-safe mapping), and the
//!   lobatto variant differs in residual/derivative formulation.  Everything
//!   the PLBound consumes (control points, projection weights, L2 basis
//!   nodes) now comes from these bit-exact rules.
//!
//! # Provenance / promotion note (D255, round 42)
//!
//! This module lived inline in `miniapps/tools/gridfunction_bounds.rs`
//! (`mod plbound`, round 41, D159) and was promoted here verbatim, then
//! extended for D256 (`-bt` projection), D257 (`-l2`), and D259 (ulp-level
//! GL node alignment).  C++ ground truth: `$HOME/work/d274/` (round-42
//! rebuild of `mpirun -np 1 gfb_cpp`), mirrored in `tmp/d274/`.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, btree_map::Entry};

use crate::postproc::grid_function::GridFunction;
use fem_element::lagrange::factory::{HexQk, QuadQk};
use fem_element::lagrange::hex::hex_tensor_layout;
use fem_element::lagrange::quad::quad_tensor_layout;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
// Only the tests cross-check the MFEM port against fem-element's tables.
#[cfg(test)]
use fem_element::quadrature::gauss_legendre_01;

// ── MFEM 1-D quadrature rules (D259: bit-exact ports) ────────────────────────

/// `QuadratureFunctions1D::GaussLegendre(np)` (`fem/intrules.cpp` L620–706):
/// the `np`-point Gauss-Legendre rule on `[0,1]`.  `np ≤ 3` are hard-coded
/// table values; `np ≥ 4` Newton-iterates the Legendre roots on `[-1,1]` and
/// maps through `xi = ((1 - z) + dz)/2` (the round-off-safe form — the
/// naive `(1 - (z - dz))/2` has bad round-off, per the C++ comment), with
/// mirrored nodes `1 - xi` and weights `1/(4·xi·(1-xi)·pp²)`.
pub fn mfem_gauss_legendre_01(np: usize) -> (Vec<f64>, Vec<f64>) {
    match np {
        1 => return (vec![0.5], vec![1.0]),
        2 => {
            return (
                vec![0.21132486540518711775, 0.78867513459481288225],
                vec![0.5, 0.5],
            );
        }
        3 => {
            return (
                vec![0.11270166537925831148, 0.5, 0.88729833462074168852],
                vec![5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0],
            );
        }
        _ => {}
    }
    let n = np;
    let m = (n + 1) / 2;
    let mut x = vec![0.0_f64; np];
    let mut w = vec![0.0_f64; np];
    for i in 1..=m {
        let mut z = (std::f64::consts::PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        let pp;
        let mut xi = 0.0_f64;
        let mut done = false;
        loop {
            let mut p2 = 1.0_f64;
            let mut p1 = z;
            for j in 2..=n {
                let p3 = p2;
                p2 = p1;
                // `((2*j - 1)*z*p2 - (j-1)*p3) / j` with the int operands
                // promoted exactly as C++ does.
                p1 = ((2 * j - 1) as f64 * z * p2 - (j - 1) as f64 * p3) / j as f64;
            }
            // p1 is the Legendre polynomial P_n(z), p2 = P_{n-1}(z).
            let pp_here = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            if done {
                pp = pp_here;
                break;
            }
            let dz = p1 / pp_here;
            if dz.abs() < 1e-16 {
                done = true;
                // Map the new point (z-dz) to (0,1); continue one more pass to
                // re-evaluate pp at the converged point, then exit.
                xi = ((1.0 - z) + dz) / 2.0;
            }
            z -= dz;
        }
        x[i - 1] = xi;
        x[n - i] = 1.0 - xi;
        let wt = 1.0 / (4.0 * xi * (1.0 - xi) * pp * pp);
        w[i - 1] = wt;
        w[n - i] = wt;
    }
    (x, w)
}

/// `QuadratureFunctions1D::GaussLobatto(np)` (`fem/intrules.cpp` L708–838):
/// the `np`-point Gauss-Lobatto rule on `[0,1]`.  Endpoints `0`/`1` carry
/// weight `1/(np·(np-1))`; interior points are Newton solves for the zeros of
/// `P'_{np-1}` started from `x_i = sin(π(i/(np-1) - 0.5))`, converged with the
/// residual `(x·P_l - P_{l-1})/(np·P_l)` and the same `done`-flag structure as
/// the C++ (one extra pass after `|dx| < 1e-16` to evaluate the weight at the
/// converged point).
pub fn mfem_gauss_lobatto_01(np: usize) -> (Vec<f64>, Vec<f64>) {
    if np == 1 {
        return (vec![0.5], vec![1.0]);
    }
    let mut x = vec![0.0_f64; np];
    let mut w = vec![0.0_f64; np];
    // Endpoints and their weights: 1/(np*(np-1)) on [0,1].
    x[0] = 0.0;
    x[np - 1] = 1.0;
    let w_end = 1.0 / (np * (np - 1)) as f64;
    w[0] = w_end;
    w[np - 1] = w_end;

    // Interior points, using symmetry: i = 1..=(np-1)/2.
    for i in 1..=(np - 1) / 2 {
        // Initial guess: the Chebyshev point x_i = sin(π(i/(np-1) - 0.5)).
        let mut x_i =
            (std::f64::consts::PI * ((i as f64) / ((np - 1) as f64) - 0.5)).sin();
        let mut z_i = 0.0_f64;
        let p_l_final;
        let mut done = false;
        let mut iter = 0_i32;
        loop {
            // Legendre polynomials up to P_{np-1}(x_i).
            let mut p_lm1 = 1.0_f64;
            let mut p_l = x_i;
            for l in 1..(np - 1) {
                // P_{l+1}(x) = ((2l+1)·x·P_l(x) - l·P_{l-1}(x)) / (l+1)
                let p_lp1 =
                    ((2 * l + 1) as f64 * x_i * p_l - l as f64 * p_lm1) / (l + 1) as f64;
                p_lm1 = p_l;
                p_l = p_lp1;
            }
            if done {
                p_l_final = p_l;
                break;
            }
            // dx = resid/deriv with resid = (x²-1)P'_{np-1}(x)
            //                  = (np-1)·(x·P_{np-1} - P_{np-2})
            // and deriv = np·(np-1)·P_{np-1}  ⇒ dx = (x·p_l - p_lm1)/(np·p_l).
            let dx = (x_i * p_l - p_lm1) / (np as f64 * p_l);
            if dx.abs() < 1e-16 {
                done = true;
                // Map the converged point to [0,1].
                z_i = ((1.0 + x_i) - dx) / 2.0;
            }
            // If the iteration does not converge fast, something is wrong
            // (MFEM_VERIFY(iter < 8, ...)).
            assert!(iter < 8, "GaussLobatto Newton did not converge (np {np})");
            iter += 1;
            x_i -= dx;
        }
        // w_i = 1/(np·(np-1)·P_{np-1}(x_i)²) on [0,1].
        let wt = 1.0 / (np * (np - 1)) as f64 / (p_l_final * p_l_final);
        x[i] = z_i;
        w[i] = wt;
        // Symmetric point.
        x[np - 1 - i] = 1.0 - z_i;
        w[np - 1 - i] = wt;
    }
    (x, w)
}

// ── 1-D barycentric Lagrange basis on arbitrary nodes in [0, 1] ──────────────
// (MFEM `Poly_1D::Basis::Eval` for the GaussLobatto/GaussLegendre basis types.)

/// Barycentric nodal Lagrange basis through the given 1-D nodes.
struct BaryLagrange1D {
    nodes: Vec<f64>,
    /// Barycentric weights `w_j = 1 / Π_{k!=j} (x_j - x_k)`.
    w: Vec<f64>,
}

impl BaryLagrange1D {
    fn from_nodes(nodes: Vec<f64>) -> Self {
        let n = nodes.len();
        let mut w = vec![1.0_f64; n];
        for j in 0..n {
            for k in 0..n {
                if k != j {
                    w[j] /= nodes[j] - nodes[k];
                }
            }
        }
        BaryLagrange1D { nodes, w }
    }

    /// All mode values and first derivatives at `x`.
    fn eval(&self, x: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.nodes.len();
        let mut vals = vec![0.0_f64; n];
        let mut ders = vec![0.0_f64; n];
        // Exact evaluation at a node (the barycentric formula is singular).
        for m in 0..n {
            if x == self.nodes[m] {
                vals[m] = 1.0;
                // l'_m(x_m) = Σ_{k!=m} 1/(x_m - x_k)
                let a0: f64 = (0..n)
                    .filter(|&k| k != m)
                    .map(|k| 1.0 / (self.nodes[m] - self.nodes[k]))
                    .sum();
                ders[m] = a0;
                // l'_j(x_m) = (w_j / w_m) / (x_m - x_j)
                for j in 0..n {
                    if j != m {
                        ders[j] = (self.w[j] / self.w[m]) / (self.nodes[m] - self.nodes[j]);
                    }
                }
                return (vals, ders);
            }
        }
        let u: Vec<f64> = self.nodes.iter().map(|&xj| 1.0 / (x - xj)).collect();
        let s: f64 = self.w.iter().zip(u.iter()).map(|(wj, uj)| wj * uj).sum();
        let mut sum_lu = 0.0;
        for j in 0..n {
            vals[j] = self.w[j] * u[j] / s;
            sum_lu += vals[j] * u[j];
        }
        // l'_j = l_j * Σ_k l_k (u_k - u_j)
        for j in 0..n {
            ders[j] = vals[j] * (sum_lu - u[j]);
        }
        (vals, ders)
    }
}

// ── PLBound (fem/bounds.cpp) ─────────────────────────────────────────────────

/// `min_ncp_gl_x[cp_type][nb-2]` — minimum control points that bound GL-node
/// Lagrange bases (bounds.hpp), `cp_type = 0` (GL+ends) column only; the
/// Chebyshev column (`cp_type = 1`) is not ported.
const MIN_NCP_GL_X: [usize; 11] = [3, 5, 6, 8, 9, 10, 11, 11, 12, 13, 14];

/// `min_ncp_gll_x[cp_type][nb-2]` — minimum control points that bound GLL
/// bases (bounds.hpp), `cp_type = 0` column.
const MIN_NCP_GLL_X: [usize; 11] = [3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16];

/// Node placement of the bounding bases (`PLBound::b_type`).
///
/// The positive/Bernstein bases (`b_type = 2`, `ClosedUniform` nodes,
/// `basisMatNodes`/LU machinery in bounds.cpp) are not ported — fem-rs has no
/// positive basis, and `gridfunction-bounds`' `-bt 2` tier stays a documented
/// gap in the miniapp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundsBasis {
    /// `b_type = 0`: Lagrange interpolants on Gauss-Legendre nodes
    /// (`"L2_*"` collections).
    GaussLegendre,
    /// `b_type = 1`: Lagrange interpolants on Gauss-Lobatto-Legendre nodes
    /// (`"H1_*"`, `"L2_T1_*"`).
    GaussLobatto,
}

/// The tensor-product space whose coefficients the bounds machinery consumes
/// (which FEC the grid function lives in — it fixes both the dof map and the
/// 1-D basis nodes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundsSpace {
    /// Continuous H1, GLL nodes, MFEM `H1_DOF_MAP` slot↔lex map.
    H1GaussLobatto,
    /// Discontinuous L2 with the given node placement: GL or GLL nodes,
    /// lexicographic (identity) dof map — MFEM `L2_DOF_MAP`.
    L2(BoundsBasis),
}

/// Piecewise-linear bounds of a tensor-product basis
/// (`mfem::PLBound`, `cp_type = 0`, `tol = 0`, Bernstein paths elided —
/// see [`BoundsBasis`] for the supported basis set).
pub struct PLBound {
    nb: usize,
    ncp: usize,
    /// 1-D quadrature/basis nodes on `[0,1]`: for [`BoundsBasis::GaussLobatto`]
    /// H1 these are the reference element's own nodes (bit-identical to
    /// `Poly_1D::GetBasis`); for the L2 bases they are the MFEM 1-D rule
    /// nodes.
    nodes: Vec<f64>,
    /// 1-D rule weights on `[0,1]` (sum 1) — `GaussLobatto(nb)` /
    /// `GaussLegendre(nb)` from bounds.cpp `Setup`.
    weights: Vec<f64>,
    /// `ncp` control points on `[0,1]` (GL + end points).
    control_points: Vec<f64>,
    /// `ncp x nb`, row-major `[j*nb + i]` — bounds of basis `i` over the
    /// interval starting at control point `j`.
    lbound: Vec<f64>,
    ubound: Vec<f64>,
}

impl PLBound {
    /// `PLBound(fes, ncp_i, 0)` for an `"H1_*"` collection: `nb = order + 1`,
    /// `ncp = max(min_ncp_gll, ncp_i)`, nodes from the reference element.
    fn from_h1_order(order: usize, ncp_i: usize, nodes1d: Vec<f64>) -> PLBound {
        let nb = order + 1;
        let minncp = if nb > 12 { 2 * nb } else { MIN_NCP_GLL_X[nb - 2] };
        let ncp = minncp.max(ncp_i);
        PLBound::setup(nb, ncp, BoundsBasis::GaussLobatto, nodes1d)
    }

    /// `PLBound(fes, ncp_i, 0)` for an `"L2_*"` / `"L2_T1_*"` collection.
    fn from_l2(order: usize, ncp_i: usize, basis: BoundsBasis) -> PLBound {
        let nb = order + 1;
        let minncp = match basis {
            BoundsBasis::GaussLegendre => {
                if nb > 12 { 2 * nb } else { MIN_NCP_GL_X[nb - 2] }
            }
            BoundsBasis::GaussLobatto => {
                if nb > 12 { 2 * nb } else { MIN_NCP_GLL_X[nb - 2] }
            }
        };
        let ncp = minncp.max(ncp_i);
        // The 1-D basis nodes are the MFEM rule nodes (Poly_1D::GetBasis(p,
        // btype) = the GL/GLL rule with np = nb points).
        let (nodes1d, _) = match basis {
            BoundsBasis::GaussLegendre => mfem_gauss_legendre_01(nb),
            BoundsBasis::GaussLobatto => mfem_gauss_lobatto_01(nb),
        };
        PLBound::setup(nb, ncp, basis, nodes1d)
    }

    fn setup(nb: usize, ncp: usize, basis: BoundsBasis, nodes1d: Vec<f64>) -> PLBound {
        assert!(ncp >= 2, "At least 2 control points are required.");
        // cp_type 0: GL + end points — `poly1d.GetPoints(ncp-3, 0)`; the GL
        // rule here is D259's bit-exact port (ncp-2 points).
        let (gl_cp, _) = mfem_gauss_legendre_01(ncp - 2);
        let mut control_points = vec![0.0_f64; ncp];
        control_points[0] = 0.0;
        control_points[ncp - 1] = 1.0;
        if ncp > 2 {
            control_points[1..ncp - 1].copy_from_slice(&gl_cp);
        }
        // `scalenodes(control_points, 0, 1)` is the identity here (the end
        // points 0 and 1 are already in the array).

        // `QuadratureFunctions1D::GaussLobatto/GaussLegendre(nb)` weights on
        // [0,1] (bounds.cpp Setup, b_type branch) — D259 bit-exact rules.
        let (_, qw) = match basis {
            BoundsBasis::GaussLegendre => mfem_gauss_legendre_01(nb),
            BoundsBasis::GaussLobatto => mfem_gauss_lobatto_01(nb),
        };
        let weights = qw;

        let basis1d = BaryLagrange1D::from_nodes(nodes1d.clone());

        // Bounding matrices (Section 3.1.1 of arXiv:2501.12349); tol = 0.
        let mut lbound = vec![0.0_f64; ncp * nb];
        let mut ubound = vec![0.0_f64; ncp * nb];
        for j in 0..ncp {
            let x = control_points[j];
            let xm = if j != 0 { 0.5 * (control_points[j - 1] + control_points[j]) } else { x };
            let xp = if j != ncp - 1 {
                0.5 * (control_points[j] + control_points[j + 1])
            } else {
                x
            };
            let (bmv, bdmv) = basis1d.eval(xm);
            let (bpv, bdpv) = basis1d.eval(xp);
            let (bv, _) = basis1d.eval(x);
            let dm = x - xm;
            let dp = x - xp;
            for i in 0..nb {
                if j == 0 || j == ncp - 1 {
                    lbound[j * nb + i] = bv[i];
                    ubound[j * nb + i] = bv[i];
                } else {
                    let v0 = bv[i];
                    let v1 = bmv[i] + dm * bdmv[i];
                    let v2 = bpv[i] + dp * bdpv[i];
                    lbound[j * nb + i] = v0.min(v1).min(v2);
                    ubound[j * nb + i] = v0.max(v1).max(v2);
                }
            }
        }

        PLBound {
            nb,
            ncp,
            nodes: nodes1d,
            weights,
            control_points,
            lbound,
            ubound,
        }
    }

    /// `GetNControlPoints()`.
    pub fn n_control_points(&self) -> usize {
        self.ncp
    }

    /// `GetControlPoints()` — the 1-D control-point positions in `[0,1]`.
    pub fn control_points(&self) -> &[f64] {
        &self.control_points
    }

    /// `Get1DBounds`: bounds of a 1-D tensor slice of coefficients.
    fn get_1d_bounds(&self, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nb = self.nb;
        let ncp = self.ncp;
        debug_assert_eq!(coeff.len(), nb);
        // Linear L2 projection (proj = true, b_type != 2).
        let mut a0 = 0.0_f64;
        let mut a1 = 0.0_f64;
        for i in 0..nb {
            let x = 2.0 * self.nodes[i] - 1.0;
            let w = 2.0 * self.weights[i];
            a0 += 0.5 * coeff[i] * w;
            a1 += 1.5 * coeff[i] * w * x;
        }
        let mut coeffm = vec![0.0_f64; nb];
        for i in 0..nb {
            let x = 2.0 * self.nodes[i] - 1.0;
            coeffm[i] = coeff[i] - a0 - a1 * x;
        }
        let mut intmin = vec![0.0_f64; ncp];
        let mut intmax = vec![0.0_f64; ncp];
        for j in 0..ncp {
            let x = 2.0 * self.control_points[j] - 1.0;
            intmin[j] = a0 + a1 * x;
            intmax[j] = intmin[j];
        }
        for (i, c) in coeffm.iter().enumerate() {
            for j in 0..ncp {
                intmin[j] += (self.lbound[j * nb + i] * c).min(self.ubound[j * nb + i] * c);
                intmax[j] += (self.lbound[j * nb + i] * c).max(self.ubound[j * nb + i] * c);
            }
        }
        (intmin, intmax)
    }

    /// `Get2DBounds`: lexicographic coefficients (x fastest) → bounds at
    /// the `ncp²` control points (x fastest).
    fn get_2d_bounds(&self, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nb = self.nb;
        let ncp = self.ncp;
        let mut intmin = vec![0.0_f64; ncp * ncp];
        let mut intmax = vec![0.0_f64; ncp * ncp];
        let mut intmin_t = vec![0.0_f64; ncp * nb];
        let mut intmax_t = vec![0.0_f64; ncp * nb];
        // Bounds for each row of the solution.
        for i in 0..nb {
            let (rmin, rmax) = self.get_1d_bounds(&coeff[i * nb..(i + 1) * nb]);
            intmin_t[i * ncp..(i + 1) * ncp].copy_from_slice(&rmin);
            intmax_t[i * ncp..(i + 1) * ncp].copy_from_slice(&rmax);
        }
        // Linear fit along each column of nodes; offset it from the
        // bounds on the coefficient.
        let mut a0v = vec![0.0_f64; ncp];
        let mut a1v = vec![0.0_f64; ncp];
        for j in 0..nb {
            let x = 2.0 * self.nodes[j] - 1.0;
            let w = 2.0 * self.weights[j];
            for i in 0..ncp {
                let t = 0.5 * (intmin_t[j * ncp + i] + intmax_t[j * ncp + i]);
                a0v[i] += 0.5 * t * w;
                a1v[i] += 1.5 * t * w * x;
            }
        }
        for j in 0..nb {
            let x = 2.0 * self.nodes[j] - 1.0;
            for i in 0..ncp {
                let t = a0v[i] + a1v[i] * x;
                intmin_t[j * ncp + i] -= t;
                intmax_t[j * ncp + i] -= t;
            }
        }
        // Initialize bounds using the a0/a1 values.
        for j in 0..ncp {
            let x = 2.0 * self.control_points[j] - 1.0;
            for i in 0..ncp {
                intmin[j * ncp + i] = a0v[i] + a1v[i] * x;
                intmax[j * ncp + i] = intmin[j * ncp + i];
            }
        }
        // Tensor combination.
        let mut id1 = 0usize;
        let mut id2 = 0usize;
        for j in 0..nb {
            for i in 0..ncp {
                let w0 = intmin_t[id1];
                id1 += 1;
                let w1 = intmax_t[id2];
                id2 += 1;
                for k in 0..ncp {
                    let lbk = self.lbound[k * nb + j];
                    let ubk = self.ubound[k * nb + j];
                    let v0 = w0 * lbk;
                    let v1 = w0 * ubk;
                    let v2 = w1 * lbk;
                    let v3 = w1 * ubk;
                    intmin[k * ncp + i] += v0.min(v1).min(v2).min(v3);
                    intmax[k * ncp + i] += v0.max(v1).max(v2).max(v3);
                }
            }
        }
        (intmin, intmax)
    }

    /// `Get3DBounds`: lexicographic coefficients (x fastest, z slowest) →
    /// bounds at the `ncp³` control points.
    fn get_3d_bounds(&self, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nb = self.nb;
        let ncp = self.ncp;
        let nb2 = nb * nb;
        let ncp2 = ncp * ncp;
        let mut intmin = vec![0.0_f64; ncp2 * ncp];
        let mut intmax = vec![0.0_f64; ncp2 * ncp];
        let mut intmin_t = vec![0.0_f64; ncp2 * nb];
        let mut intmax_t = vec![0.0_f64; ncp2 * nb];
        // Bounds for each slice of the solution.
        for i in 0..nb {
            let (smin, smax) = self.get_2d_bounds(&coeff[i * nb2..(i + 1) * nb2]);
            intmin_t[i * ncp2..(i + 1) * ncp2].copy_from_slice(&smin);
            intmax_t[i * ncp2..(i + 1) * ncp2].copy_from_slice(&smax);
        }
        // Linear fit along each tower of nodes.
        let mut a0v = vec![0.0_f64; ncp2];
        let mut a1v = vec![0.0_f64; ncp2];
        for j in 0..nb {
            let x = 2.0 * self.nodes[j] - 1.0;
            let w = 2.0 * self.weights[j];
            for i in 0..ncp2 {
                let t = 0.5 * (intmin_t[j * ncp2 + i] + intmax_t[j * ncp2 + i]);
                a0v[i] += 0.5 * t * w;
                a1v[i] += 1.5 * t * w * x;
            }
        }
        for j in 0..nb {
            let x = 2.0 * self.nodes[j] - 1.0;
            for i in 0..ncp2 {
                let t = a0v[i] + a1v[i] * x;
                intmin_t[j * ncp2 + i] -= t;
                intmax_t[j * ncp2 + i] -= t;
            }
        }
        // Initialize bounds using the a0/a1 values.
        for j in 0..ncp {
            let x = 2.0 * self.control_points[j] - 1.0;
            for i in 0..ncp2 {
                intmin[j * ncp2 + i] = a0v[i] + a1v[i] * x;
                intmax[j * ncp2 + i] = a0v[i] + a1v[i] * x;
            }
        }
        // Tensor combination.
        let mut id1 = 0usize;
        let mut id2 = 0usize;
        for j in 0..nb {
            for i in 0..ncp2 {
                let w0 = intmin_t[id1];
                id1 += 1;
                let w1 = intmax_t[id2];
                id2 += 1;
                for k in 0..ncp {
                    let lbk = self.lbound[k * nb + j];
                    let ubk = self.ubound[k * nb + j];
                    let v0 = w0 * lbk;
                    let v1 = w0 * ubk;
                    let v2 = w1 * lbk;
                    let v3 = w1 * ubk;
                    intmin[k * ncp2 + i] += v0.min(v1).min(v2).min(v3);
                    intmax[k * ncp2 + i] += v0.max(v1).max(v2).max(v3);
                }
            }
        }
        (intmin, intmax)
    }

    /// `GetNDBounds(rdim, coeff, intmin, intmax)`.
    fn get_nd_bounds(&self, rdim: usize, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match rdim {
            1 => self.get_1d_bounds(coeff),
            2 => self.get_2d_bounds(coeff),
            3 => self.get_3d_bounds(coeff),
            _ => unreachable!("PLBound supports rdim 1..=3 only"),
        }
    }
}

// ── Tensor-space info (dof maps + 1-D nodes per BoundsSpace) ─────────────────

/// 1-D nodes (in `[0,1]` convention) + H1 tensor dof map of the space's
/// reference element — see [`h1_tensor_nodes`].
pub fn h1_tensor_nodes(
    elem_type: ElementType,
    p: usize,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    let (nodes, slots): (Vec<f64>, Vec<Vec<usize>>) = match elem_type {
        ElementType::Quad4 => {
            let (n, s) = quad_tensor_layout(&QuadQk::new(p));
            (n, s.iter().map(|i| vec![i[0], i[1]]).collect())
        }
        ElementType::Hex8 => {
            let (n, s) = hex_tensor_layout(&HexQk::new(p));
            (
                n.iter().map(|&x| 0.5 * (x + 1.0)).collect(),
                s.iter().map(|i| vec![i[0], i[1], i[2]]).collect(),
            )
        }
        other => {
            return Err(format!(
                "TensorBasis FiniteElement expected (PLBound supports quad/hex elements), got \
                 {other:?}."
            ))
        }
    };
    let nb = nodes.len();
    let stride: Vec<usize> = match slots[0].len() {
        2 => vec![1, nb],
        _ => vec![1, nb, nb * nb],
    };
    let mut map = vec![0usize; nodes.len().pow(slots[0].len() as u32)];
    for (slot, idx) in slots.iter().enumerate() {
        let j: usize = idx.iter().zip(stride.iter()).map(|(&i, &s)| i * s).sum();
        map[j] = slot;
    }
    Ok((nodes, map))
}

/// The H1 tensor dof map: entry `j` is the element-local dof slot holding
/// the lexicographic tensor node `j` (x fastest), matching MFEM's
/// `TensorBasisElement::GetDofMap`.
fn h1_tensor_dof_map(elem_type: ElementType, p: usize) -> Result<Vec<usize>, String> {
    h1_tensor_nodes(elem_type, p).map(|(_, map)| map)
}

/// Per-space tensor info: the 1-D basis nodes (in `[0,1]`) and, for H1, the
/// lex→slot dof map (`None` = identity, the L2 `L2_DOF_MAP` convention).
struct TensorInfo {
    nodes1d: Vec<f64>,
    dof_map: Option<Vec<usize>>,
}

fn tensor_info(elem_type: ElementType, order: usize, space: BoundsSpace) -> Result<TensorInfo, String> {
    match space {
        BoundsSpace::H1GaussLobatto => {
            let (nodes, map) = h1_tensor_nodes(elem_type, order)?;
            Ok(TensorInfo { nodes1d: nodes, dof_map: Some(map) })
        }
        BoundsSpace::L2(basis) => {
            if !matches!(elem_type, ElementType::Quad4 | ElementType::Hex8) {
                return Err(format!(
                    "TensorBasis FiniteElement expected (L2 bounds support quad/hex elements), \
                     got {elem_type:?}."
                ));
            }
            // MFEM L2 elements' 1-D nodes are the GL/GLL rule nodes
            // (Poly_1D::GetPoints(p, btype)).
            let (nodes1d, _) = match basis {
                BoundsBasis::GaussLegendre => mfem_gauss_legendre_01(order + 1),
                BoundsBasis::GaussLobatto => mfem_gauss_lobatto_01(order + 1),
            };
            Ok(TensorInfo { nodes1d, dof_map: None })
        }
    }
}

/// Element-local DOF values in lexicographic order
/// (`GetSubVector` + `dof_map` in `GetElementBoundsAtControlPoints`).
fn element_lex_data<S: FESpace>(
    gf: &GridFunction<'_, S>,
    elem: u32,
    info: &TensorInfo,
) -> Vec<f64> {
    let elem_dofs = gf.space().element_dofs(elem);
    match &info.dof_map {
        Some(map) => map.iter().map(|&slot| gf.dofs()[elem_dofs[slot] as usize]).collect(),
        // L2: the element dofs are already lexicographic.
        None => elem_dofs.iter().map(|&d| gf.dofs()[d as usize]).collect(),
    }
}

// ── Projection (GridFunction::ProjectGridFunction, -bt tier) ─────────────────

/// `pfunc_proj->ProjectGridFunction(pfunc)` for a discontinuous tensor target:
/// MFEM assembles, per element, the nodal-interpolation matrix
/// `I(k,j) = src.CalcShape(target_node_k)[j]` (`NodalFiniteElement::Project`,
/// `fem/fe/fe_base.cpp` L877 — entries with `|shape| < 1e-12` snap to `0`),
/// then sets each target element's coefficients to `I·src`
/// (`GridFunction::ProjectGridFunction`, `fem/gridfunc.cpp` L1764).
///
/// The source must be a scalar H1 (GLL) field on quad/hex elements; the
/// result holds the projected DOFs in the target `L2Space` layout
/// (element-major, lexicographic per element, `(order+1)^dim` per element),
/// i.e. `project_h1_to_l2(...)[i]` is the value of target DOF `i`.  The
/// target node positions are the MFEM 1-D rule nodes for `basis`
/// ([`mfem_gauss_legendre_01`] / [`mfem_gauss_lobatto_01`]), exactly what the
/// C++ `L2_FECollection(order, dim, btype)` elements carry.
pub fn project_h1_to_l2<S: FESpace>(
    h1: &GridFunction<'_, S>,
    order: usize,
    basis: BoundsBasis,
) -> Result<Vec<f64>, String> {
    let mesh = h1.space().mesh();
    let nel = mesh.n_elements();
    if nel == 0 {
        return Err("ProjectGridFunction: mesh has no elements".to_string());
    }
    let etype = mesh.element_type(0);
    for e in 1..nel as u32 {
        if mesh.element_type(e) != etype {
            return Err(format!(
                "ProjectGridFunction: mixed element types are not supported (C++ aborts on \
                 non-TensorBasis elements); first mismatch at element {e}"
            ));
        }
    }
    let dim = mesh.topological_dim() as usize;
    let (gll_nodes, dof_map) = h1_tensor_nodes(etype, order)?;
    let nb = order + 1;
    let (target_nodes, _) = match basis {
        BoundsBasis::GaussLegendre => mfem_gauss_legendre_01(nb),
        BoundsBasis::GaussLobatto => mfem_gauss_lobatto_01(nb),
    };
    // 1-D GLL basis of the source, evaluated at every target node position.
    // `h1_tensor_nodes` rescales the hex ([-1,1]) nodes into the uniform
    // [0,1] convention, so both the basis nodes and the target positions
    // (the MFEM [0,1] rule nodes) live on [0,1] — no further mapping.
    let src_basis = BaryLagrange1D::from_nodes(gll_nodes);
    let shape1d: Vec<Vec<f64>> = target_nodes
        .iter()
        .map(|&t| {
            let (v, _) = src_basis.eval(t);
            v
        })
        .collect();

    let nb2 = nb * nb;
    let nb_dim = nb.pow(dim as u32);
    let mut out = vec![0.0_f64; nel as usize * nb_dim];
    for e in 0..nel as u32 {
        let elem_dofs = h1.space().element_dofs(e);
        // Source values in slot order (the FES/FE convention CalcShape uses).
        let src: Vec<f64> = elem_dofs.iter().map(|&d| h1.dofs()[d as usize]).collect();
        let mut dest = vec![0.0_f64; nb_dim];
        // For every lexicographic target node k: I(k,·) = src basis scattered
        // through dof_map (H1_QuadrilateralElement::CalcShape semantics),
        // then the row-dot in slot order (DenseMatrix::Mult).
        match dim {
            1 => unreachable!("1-D meshes have no fem-rs Mesh type"),
            2 => {
                for (iy, shape_y) in shape1d.iter().enumerate() {
                    for (ix, shape_x) in shape1d.iter().enumerate() {
                        // shape(slot) via the lex→slot map: for source lex
                        // node (jx,jy), shape(dof_map[jx + jy*nb]) =
                        // shape_x[jx]*shape_y[jy] (H1_QuadrilateralElement::
                        // CalcShape semantics) — the scatter key is the source
                        // lex index, while the target node (ix,iy) only
                        // selects the evaluated 1-D shapes.
                        let mut row = vec![0.0_f64; nb2];
                        for (jy, &sy) in shape_y.iter().enumerate() {
                            for (jx, &sx) in shape_x.iter().enumerate() {
                                row[dof_map[jx + jy * nb]] = sx * sy;
                            }
                        }
                        // Threshold |I| < 1e-12 → 0, accumulate in slot order.
                        let mut acc = 0.0_f64;
                        for (j, r) in row.iter().enumerate() {
                            let i_kj = if r.abs() < 1e-12 { 0.0 } else { *r };
                            acc += i_kj * src[j];
                        }
                        dest[ix + iy * nb] = acc;
                    }
                }
            }
            _ => {
                for (iz, shape_z) in shape1d.iter().enumerate() {
                    for (iy, shape_y) in shape1d.iter().enumerate() {
                        for (ix, shape_x) in shape1d.iter().enumerate() {
                            let mut row = vec![0.0_f64; nb2 * nb];
                            for (jz, &sz) in shape_z.iter().enumerate() {
                                for (jy, &sy) in shape_y.iter().enumerate() {
                                    for (jx, &sx) in shape_x.iter().enumerate() {
                                        row[dof_map[jx + jy * nb + jz * nb2]] = sx * sy * sz;
                                    }
                                }
                            }
                            let mut acc = 0.0_f64;
                            for (j, r) in row.iter().enumerate() {
                                let i_kj = if r.abs() < 1e-12 { 0.0 } else { *r };
                                acc += i_kj * src[j];
                            }
                            dest[ix + iy * nb + iz * nb2] = acc;
                        }
                    }
                }
            }
        }
        let base = e as usize * dest.len();
        out[base..base + dest.len()].copy_from_slice(&dest);
    }
    Ok(out)
}

// ── GetElementBounds (gridfunc.cpp) ──────────────────────────────────────────

/// `PLBound GridFunction::GetElementBounds(lower, upper, ref_factor, vdim)`
/// on the H1 (GLL) path — builds the PLBound with
/// `ncp = max(min_ncp, ref_factor*(order+1))` and returns the per-element
/// lower/upper bounds (length = #elements for `vdim == 1`).
pub fn get_element_bounds<S: FESpace>(
    gf: &GridFunction<'_, S>,
    ref_factor: i32,
    vdim: usize,
) -> Result<(PLBound, Vec<f64>, Vec<f64>), String> {
    get_element_bounds_in(gf, ref_factor, vdim, BoundsSpace::H1GaussLobatto)
}

/// [`get_element_bounds`] for an explicit [`BoundsSpace`] (the `-l2`/`-bt`
/// tiers: L2 GL/GLL coefficient layout with identity dof maps).
pub fn get_element_bounds_in<S: FESpace>(
    gf: &GridFunction<'_, S>,
    ref_factor: i32,
    vdim: usize,
    space: BoundsSpace,
) -> Result<(PLBound, Vec<f64>, Vec<f64>), String> {
    if vdim != 1 {
        return Err(format!("GetElementBounds: vdim {vdim} not supported (port is scalar)"));
    }
    let mesh = gf.space().mesh();
    let nel = mesh.n_elements();
    if nel == 0 {
        return Err("GetElementBounds: mesh has no elements".to_string());
    }
    // One representative element type: the C++ MFEM_VERIFYs (TensorBasis
    // expected) on the first non-tensor element, so any mix aborts there.
    let etype = mesh.element_type(0);
    for e in 1..nel as u32 {
        if mesh.element_type(e) != etype {
            return Err(format!(
                "GetElementBounds: mixed element types are not supported (C++ aborts on \
                 non-TensorBasis elements); first mismatch at element {e}"
            ));
        }
    }
    let dim = mesh.topological_dim() as usize;
    let order = gf.space().order() as usize;
    let info = tensor_info(etype, order, space)?;
    let plb = match space {
        BoundsSpace::H1GaussLobatto => {
            PLBound::from_h1_order(order, (ref_factor * (order as i32 + 1)) as usize, info.nodes1d.clone())
        }
        BoundsSpace::L2(basis) => PLBound::from_l2(order, (ref_factor * (order as i32 + 1)) as usize, basis),
    };

    let mut lower = vec![0.0_f64; nel];
    let mut upper = vec![0.0_f64; nel];
    for e in 0..nel as u32 {
        let lex = element_lex_data(gf, e, &info);
        let (lo, up) = plb.get_nd_bounds(dim, &lex);
        // `GetElementBounds(elem, ...)`: min/max over the control points.
        lower[e as usize] = lo.iter().cloned().reduce(f64::min).unwrap();
        upper[e as usize] = up.iter().cloned().reduce(f64::max).unwrap();
    }
    Ok((plb, lower, upper))
}

// ── EstimateFunctionMinimum / Maximum (gridfunc.cpp) ─────────────────────────

/// Total-ordered f64 for the priority queues / leaf maps.
#[derive(Clone, Copy, Debug)]
struct OrdF64(f64);

impl PartialEq for OrdF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrdF64 {}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Best-first search item (`SearchInterval`): `key` is `val_min` (min
/// search) or `val_max` (max search); `seq` breaks ties by insertion
/// order (the C++ heap's tie order is internal, and the final result is
/// tie-order independent — min/max over the leaf set).
struct SearchItem {
    key: OrdF64,
    seq: u64,
    depth: usize,
    range: Vec<f64>,
}

impl PartialEq for SearchItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.seq == other.seq
    }
}

impl Eq for SearchItem {}

impl PartialOrd for SearchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key).then(self.seq.cmp(&other.seq))
    }
}

/// Leaf-set bookkeeping (`IntervalNode::GetChildMinLower` /
/// `GetChildMaxUpper` reduce to a min/max over the created-but-never-
/// expanded intervals — the leaves of the C++ interval tree).
fn bump_leaf(leaves: &mut BTreeMap<OrdF64, usize>, v: f64) {
    *leaves.entry(OrdF64(v)).or_insert(0) += 1;
}

fn drop_leaf(leaves: &mut BTreeMap<OrdF64, usize>, v: f64) {
    match leaves.entry(OrdF64(v)) {
        Entry::Occupied(mut e) => {
            let c = e.get_mut();
            *c -= 1;
            if *c == 0 {
                e.remove();
            }
        }
        Entry::Vacant(_) => unreachable!("leaf bookkeeping: dropped a non-leaf value"),
    }
}

/// One bounds computation on a reference sub-range
/// (`GetElementBoundsAtControlPoints(elem, plb, ref_range, vdim, ...)`):
/// evaluate the function at the FE node positions mapped into the
/// sub-range, then bound those values as GLL coefficients.  Also returns
/// the mapped control-point positions.
fn bounds_on_range(
    plb: &PLBound,
    basis: &BaryLagrange1D,
    lex: &[f64],
    range: &[f64],
    dim: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nb = basis.nodes.len();
    let ncp = plb.n_control_points();
    // 1-D mode values at every sub-range-scaled node position:
    // `ip_coord(d) = ref_range(d) + (ref_range(dim+d) - ref_range(d)) * x`;
    // `axes[d][m][i] = l_m(x_i^(d))`.
    let mut axes: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; nb]; nb]; dim];
    for d in 0..dim {
        let lo = range[d];
        let scale = range[dim + d] - range[d];
        for (i, &node) in basis.nodes.iter().enumerate() {
            let (vi, _) = basis.eval(lo + scale * node);
            for (m, v) in vi.iter().enumerate() {
                axes[d][m][i] = *v;
            }
        }
    }
    // u at mapped node `i` = Σ_lex c[j] Π_d l_{j_d}(x_i^(d))
    // (`GridFunction::GetValues` at `ir_new`; for L2 the C++ dot runs over the
    // identity (lex) dof map, for H1 over the scattered slot order — the
    // round-41 note documents the ulp-level difference the lex order makes).
    let mut vals = vec![0.0_f64; lex.len()];
    match dim {
        1 => {
            for i in 0..nb {
                let mut s = 0.0;
                for m in 0..nb {
                    s += lex[m] * axes[0][m][i];
                }
                vals[i] = s;
            }
        }
        2 => {
            for iy in 0..nb {
                for ix in 0..nb {
                    let mut s = 0.0;
                    for jy in 0..nb {
                        for jx in 0..nb {
                            s += lex[jx + jy * nb] * axes[0][jx][ix] * axes[1][jy][iy];
                        }
                    }
                    vals[ix + iy * nb] = s;
                }
            }
        }
        _ => {
            for iz in 0..nb {
                for iy in 0..nb {
                    for ix in 0..nb {
                        let mut s = 0.0;
                        for jz in 0..nb {
                            for jy in 0..nb {
                                for jx in 0..nb {
                                    let j = jx + jy * nb + jz * nb * nb;
                                    s += lex[j]
                                        * axes[0][jx][ix]
                                        * axes[1][jy][iy]
                                        * axes[2][jz][iz];
                                }
                            }
                        }
                        vals[ix + iy * nb + iz * nb * nb] = s;
                    }
                }
            }
        }
    }
    let (lower, upper) = plb.get_nd_bounds(dim, &vals);
    // Control point positions in the sub-range.
    let mut cp_ref_loc = vec![0.0_f64; dim * ncp];
    for i in 0..ncp {
        for d in 0..dim {
            cp_ref_loc[i + d * ncp] =
                range[d] + (range[dim + d] - range[d]) * plb.control_points[i];
        }
    }
    (lower, upper, cp_ref_loc)
}

/// `EstimateFunctionMinimum(elem, plb, vdim, max_depth, tol, min_threshold)`
/// — returns `(min_lower, min_upper)` for the element.
fn estimate_function_minimum_elem(
    plb: &PLBound,
    basis: &BaryLagrange1D,
    lex: &[f64],
    dim: usize,
    max_depth: i32,
    tol: f64,
    min_threshold: &mut f64,
) -> (f64, f64) {
    let ncp = plb.n_control_points();
    let mut pos_range = vec![0.0_f64; 2 * dim];
    for d in 0..dim {
        pos_range[d + dim] = 1.0;
    }

    let (lower, upper) = plb.get_nd_bounds(dim, lex);
    let val_min = lower.iter().cloned().reduce(f64::min).unwrap();
    let val_max = upper.iter().cloned().reduce(f64::min).unwrap();

    *min_threshold = (*min_threshold).min(val_max);
    if val_min >= *min_threshold {
        return (val_min, val_max);
    }
    if val_min == val_max || max_depth == 0 {
        *min_threshold = (*min_threshold).min(val_min);
        return (val_min, val_max);
    }
    let abs_tol = tol * (val_max - val_min);

    // Leaf-set bookkeeping = `IntervalNode::GetChildMinLower` over the
    // whole tree: the minimum `val_min` over all created-but-never-
    // expanded intervals.
    let mut leaves: BTreeMap<OrdF64, usize> = BTreeMap::new();
    bump_leaf(&mut leaves, val_min);

    let mut seq: u64 = 0;
    // Best-first min-heap on the interval's lower bound (`val_min`).
    let mut pq: BinaryHeap<Reverse<SearchItem>> = BinaryHeap::new();
    pq.push(Reverse(SearchItem { key: OrdF64(val_min), seq: 0, depth: 0, range: pos_range.clone() }));

    let mut min_upper_bound = val_max;
    // Recomputed from the leaf set before any read (as in the C++ code,
    // where `GetChildMinLower` overwrites the initial `lower.Min()`).
    #[allow(unused_assignments)] // mirrors gridfunc.cpp initialization
    let mut min_lower_bound = val_min;

    while let Some(Reverse(item)) = pq.pop() {
        let SearchItem { key, depth, range, .. } = &item;
        let key = *key;
        let depth = *depth;
        let range = range.as_slice();
        // Reached max depth or this interval cannot contain the minimum.
        if key.0 >= *min_threshold || (depth as i32) >= max_depth {
            continue; // stays a leaf
        }
        min_lower_bound = leaves.keys().next().map(|k| k.0).unwrap();
        if min_upper_bound - min_lower_bound < abs_tol {
            break;
        }
        // Expand: subdivide and bound each child interval.
        drop_leaf(&mut leaves, key.0);
        let (lower, upper, cp_ref_loc) = bounds_on_range(plb, basis, lex, &range, dim);
        for k in 0..if dim == 3 { ncp - 1 } else { 1 } {
            for j in 0..if dim >= 2 { ncp - 1 } else { 1 } {
                for i in 0..ncp - 1 {
                    let (lv, uv) = match dim {
                        1 => (
                            lower[i].min(lower[i + 1]),
                            upper[i].min(upper[i + 1]),
                        ),
                        2 => (
                            (lower[i + j * ncp])
                                .min(lower[(i + 1) + j * ncp])
                                .min(lower[i + (j + 1) * ncp])
                                .min(lower[(i + 1) + (j + 1) * ncp]),
                            (upper[i + j * ncp])
                                .min(upper[(i + 1) + j * ncp])
                                .min(upper[i + (j + 1) * ncp])
                                .min(upper[(i + 1) + (j + 1) * ncp]),
                        ),
                        _ => {
                            let n2 = ncp * ncp;
                            (
                                (lower[i + j * ncp + k * n2])
                                    .min(lower[(i + 1) + j * ncp + k * n2])
                                    .min(lower[i + (j + 1) * ncp + k * n2])
                                    .min(lower[(i + 1) + (j + 1) * ncp + k * n2])
                                    .min(lower[i + j * ncp + (k + 1) * n2])
                                    .min(lower[(i + 1) + j * ncp + (k + 1) * n2])
                                    .min(lower[i + (j + 1) * ncp + (k + 1) * n2])
                                    .min(lower[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                                (upper[i + j * ncp + k * n2])
                                    .min(upper[(i + 1) + j * ncp + k * n2])
                                    .min(upper[i + (j + 1) * ncp + k * n2])
                                    .min(upper[(i + 1) + (j + 1) * ncp + k * n2])
                                    .min(upper[i + j * ncp + (k + 1) * n2])
                                    .min(upper[(i + 1) + j * ncp + (k + 1) * n2])
                                    .min(upper[i + (j + 1) * ncp + (k + 1) * n2])
                                    .min(upper[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                            )
                        }
                    };
                    // Child interval (always recorded in the tree).
                    bump_leaf(&mut leaves, lv);
                    if lv < *min_threshold {
                        min_upper_bound = min_upper_bound.min(uv);
                        *min_threshold = (*min_threshold).min(uv);
                        if (depth as i32) < max_depth {
                            let mut child_range = range.to_vec();
                            child_range[0] = cp_ref_loc[i];
                            child_range[0 + dim] = cp_ref_loc[i + 1];
                            if dim >= 2 {
                                child_range[1] = cp_ref_loc[ncp + j];
                                child_range[1 + dim] = cp_ref_loc[ncp + j + 1];
                            }
                            if dim == 3 {
                                child_range[2] = cp_ref_loc[2 * ncp + k];
                                child_range[2 + dim] = cp_ref_loc[2 * ncp + k + 1];
                            }
                            seq += 1;
                            pq.push(Reverse(SearchItem {
                                key: OrdF64(lv),
                                seq,
                                depth: depth + 1,
                                range: child_range,
                            }));
                        }
                    }
                }
            }
        }
    }

    min_lower_bound = leaves.keys().next().map(|k| k.0).unwrap();
    *min_threshold = (*min_threshold).min(min_lower_bound);
    (min_lower_bound, min_upper_bound)
}

/// `EstimateFunctionMaximum(elem, ...)` — mirror of the minimum search.
fn estimate_function_maximum_elem(
    plb: &PLBound,
    basis: &BaryLagrange1D,
    lex: &[f64],
    dim: usize,
    max_depth: i32,
    tol: f64,
    max_threshold: &mut f64,
) -> (f64, f64) {
    let ncp = plb.n_control_points();
    let mut pos_range = vec![0.0_f64; 2 * dim];
    for d in 0..dim {
        pos_range[d + dim] = 1.0;
    }

    let (lower, upper) = plb.get_nd_bounds(dim, lex);
    let val_min = lower.iter().cloned().reduce(f64::max).unwrap();
    let val_max = upper.iter().cloned().reduce(f64::max).unwrap();

    *max_threshold = (*max_threshold).max(val_min);
    if val_max <= *max_threshold {
        return (val_min, val_max);
    }
    if val_min == val_max || max_depth == 0 {
        *max_threshold = (*max_threshold).max(val_max);
        return (val_min, val_max);
    }
    let abs_tol = tol * (val_max - val_min);

    // Leaf set tracks `val_max` of unexpanded intervals; query = max.
    let mut leaves: BTreeMap<OrdF64, usize> = BTreeMap::new();
    bump_leaf(&mut leaves, val_max);

    let mut seq: u64 = 0;
    // Best-first max-heap on the interval's upper bound (`val_max`).
    let mut pq: BinaryHeap<SearchItem> = BinaryHeap::new();
    pq.push(SearchItem { key: OrdF64(val_max), seq: 0, depth: 0, range: pos_range.clone() });

    let mut max_lower_bound = val_min;
    // Recomputed from the leaf set before any read (as in the C++ code,
    // where `GetChildMaxUpper` overwrites the initial `upper.Max()`).
    #[allow(unused_assignments)] // mirrors gridfunc.cpp initialization
    let mut max_upper_bound = val_max;

    while let Some(item) = pq.pop() {
        let SearchItem { key, depth, range, .. } = &item;
        let key = *key;
        let depth = *depth;
        let range = range.as_slice();
        if key.0 <= *max_threshold || (depth as i32) >= max_depth {
            continue;
        }
        max_upper_bound = leaves.keys().next_back().map(|k| k.0).unwrap();
        if max_upper_bound - max_lower_bound < abs_tol {
            break;
        }
        drop_leaf(&mut leaves, key.0);
        let (lower, upper, cp_ref_loc) = bounds_on_range(plb, basis, lex, &range, dim);
        for k in 0..if dim == 3 { ncp - 1 } else { 1 } {
            for j in 0..if dim >= 2 { ncp - 1 } else { 1 } {
                for i in 0..ncp - 1 {
                    let (lv, uv) = match dim {
                        1 => (
                            lower[i].max(lower[i + 1]),
                            upper[i].max(upper[i + 1]),
                        ),
                        2 => (
                            (lower[i + j * ncp])
                                .max(lower[(i + 1) + j * ncp])
                                .max(lower[i + (j + 1) * ncp])
                                .max(lower[(i + 1) + (j + 1) * ncp]),
                            (upper[i + j * ncp])
                                .max(upper[(i + 1) + j * ncp])
                                .max(upper[i + (j + 1) * ncp])
                                .max(upper[(i + 1) + (j + 1) * ncp]),
                        ),
                        _ => {
                            let n2 = ncp * ncp;
                            (
                                (lower[i + j * ncp + k * n2])
                                    .max(lower[(i + 1) + j * ncp + k * n2])
                                    .max(lower[i + (j + 1) * ncp + k * n2])
                                    .max(lower[(i + 1) + (j + 1) * ncp + k * n2])
                                    .max(lower[i + j * ncp + (k + 1) * n2])
                                    .max(lower[(i + 1) + j * ncp + (k + 1) * n2])
                                    .max(lower[i + (j + 1) * ncp + (k + 1) * n2])
                                    .max(lower[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                                (upper[i + j * ncp + k * n2])
                                    .max(upper[(i + 1) + j * ncp + k * n2])
                                    .max(upper[i + (j + 1) * ncp + k * n2])
                                    .max(upper[(i + 1) + (j + 1) * ncp + k * n2])
                                    .max(upper[i + j * ncp + (k + 1) * n2])
                                    .max(upper[(i + 1) + j * ncp + (k + 1) * n2])
                                    .max(upper[i + (j + 1) * ncp + (k + 1) * n2])
                                    .max(upper[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                            )
                        }
                    };
                    bump_leaf(&mut leaves, uv);
                    if uv > *max_threshold {
                        max_lower_bound = max_lower_bound.max(lv);
                        *max_threshold = (*max_threshold).max(lv);
                        if (depth as i32) < max_depth {
                            let mut child_range = range.to_vec();
                            child_range[0] = cp_ref_loc[i];
                            child_range[0 + dim] = cp_ref_loc[i + 1];
                            if dim >= 2 {
                                child_range[1] = cp_ref_loc[ncp + j];
                                child_range[1 + dim] = cp_ref_loc[ncp + j + 1];
                            }
                            if dim == 3 {
                                child_range[2] = cp_ref_loc[2 * ncp + k];
                                child_range[2 + dim] = cp_ref_loc[2 * ncp + k + 1];
                            }
                            seq += 1;
                            pq.push(SearchItem {
                                key: OrdF64(uv),
                                seq,
                                depth: depth + 1,
                                range: child_range,
                            });
                        }
                    }
                }
            }
        }
    }

    max_upper_bound = leaves.keys().next_back().map(|k| k.0).unwrap();
    *max_threshold = (*max_threshold).max(max_upper_bound);
    (max_lower_bound, max_upper_bound)
}

/// `EstimateFunctionMinimum(vdim, plb, max_depth, tol)` — global loop
/// carrying `global_min_lower` as the running prune threshold (H1 GLL path).
pub fn estimate_function_minimum<S: FESpace>(
    gf: &GridFunction<'_, S>,
    plb: &PLBound,
    vdim: usize,
    max_depth: i32,
    tol: f64,
) -> (f64, f64) {
    estimate_function_minimum_in(gf, plb, BoundsSpace::H1GaussLobatto, vdim, max_depth, tol)
}

/// [`estimate_function_minimum`] for an explicit [`BoundsSpace`].
pub fn estimate_function_minimum_in<S: FESpace>(
    gf: &GridFunction<'_, S>,
    plb: &PLBound,
    space: BoundsSpace,
    vdim: usize,
    max_depth: i32,
    tol: f64,
) -> (f64, f64) {
    assert_eq!(vdim, 1, "EstimateFunctionMinimum: scalar port");
    let mesh = gf.space().mesh();
    let dim = mesh.topological_dim() as usize;
    let order = gf.space().order() as usize;
    let info = tensor_info(mesh.element_type(0), order, space).expect("tensor dof map");
    let basis = BaryLagrange1D::from_nodes(info.nodes1d.clone());

    let mut global_min_lower = f64::MAX;
    let mut global_min_upper = f64::MAX;
    for e in 0..mesh.n_elements() as u32 {
        let lex = element_lex_data(gf, e, &info);
        let pair = estimate_function_minimum_elem(
            plb,
            &basis,
            &lex,
            dim,
            max_depth,
            tol,
            &mut global_min_lower,
        );
        global_min_upper = global_min_upper.min(pair.1);
    }
    (global_min_lower, global_min_upper)
}

/// `EstimateFunctionMaximum(vdim, plb, max_depth, tol)` — global loop
/// carrying `global_max_upper` as the running prune threshold (H1 GLL path).
pub fn estimate_function_maximum<S: FESpace>(
    gf: &GridFunction<'_, S>,
    plb: &PLBound,
    vdim: usize,
    max_depth: i32,
    tol: f64,
) -> (f64, f64) {
    estimate_function_maximum_in(gf, plb, BoundsSpace::H1GaussLobatto, vdim, max_depth, tol)
}

/// [`estimate_function_maximum`] for an explicit [`BoundsSpace`].
pub fn estimate_function_maximum_in<S: FESpace>(
    gf: &GridFunction<'_, S>,
    plb: &PLBound,
    space: BoundsSpace,
    vdim: usize,
    max_depth: i32,
    tol: f64,
) -> (f64, f64) {
    assert_eq!(vdim, 1, "EstimateFunctionMaximum: scalar port");
    let mesh = gf.space().mesh();
    let dim = mesh.topological_dim() as usize;
    let order = gf.space().order() as usize;
    let info = tensor_info(mesh.element_type(0), order, space).expect("tensor dof map");
    let basis = BaryLagrange1D::from_nodes(info.nodes1d.clone());

    let mut global_max_lower = f64::MIN;
    let mut global_max_upper = f64::MIN;
    for e in 0..mesh.n_elements() as u32 {
        let lex = element_lex_data(gf, e, &info);
        let pair = estimate_function_maximum_elem(
            plb,
            &basis,
            &lex,
            dim,
            max_depth,
            tol,
            &mut global_max_upper,
        );
        global_max_lower = global_max_lower.max(pair.0);
    }
    (global_max_lower, global_max_upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    use fem_element::lagrange::factory::QuadQk;
    use fem_element::lagrange::QuadL2GL;
    use fem_element::ReferenceElement;
    use fem_mesh::Mesh;
    use fem_space::{L2Basis, L2Space};

    // ── D259: bit-exact quadrature pins against the MFEM 4.10 dump ─────────
    // (tmp/d274/quad_dump.txt — `mpirun -np 1 ./quad_dump` in $HOME/work/d274,
    // %.17g printf of the very values `poly1d.GetPoints` / `PLBound::Setup`
    // consume; last-bit-exact comparisons, no tolerance.)

    #[test]
    fn mfem_gauss_legendre_matches_cpp_dump() {
        let (x, w) = mfem_gauss_legendre_01(6);
        // GL np=6 from quad_dump.txt.
        let x_cpp = [
            0.033765242898423996,
            0.16939530676686773,
            0.38069040695840156,
            0.61930959304159838,
            0.83060469323313224,
            0.96623475710157603,
        ];
        let w_cpp = [
            0.085662246189585234,
            0.1803807865240693,
            0.23395696728634552,
            0.23395696728634552,
            0.1803807865240693,
            0.085662246189585234,
        ];
        for i in 0..6 {
            assert_eq!(x[i], x_cpp[i], "GL np=6 node {i}");
            assert_eq!(w[i], w_cpp[i], "GL np=6 weight {i}");
        }
    }

    #[test]
    fn mfem_gauss_lobatto_matches_cpp_dump() {
        let (x, w) = mfem_gauss_lobatto_01(6);
        // GLL np=6 from quad_dump.txt.
        let x_cpp = [
            0.0,
            0.11747233803526766,
            0.35738424175967742,
            0.64261575824032258,
            0.88252766196473231,
            1.0,
        ];
        let w_cpp = [
            0.033333333333333333,
            0.18923747814892347,
            0.27742918851774323,
            0.27742918851774323,
            0.18923747814892347,
            0.033333333333333333,
        ];
        for i in 0..6 {
            assert_eq!(x[i], x_cpp[i], "GLL np=6 node {i}");
            assert_eq!(w[i], w_cpp[i], "GLL np=6 weight {i}");
        }
        // GLL np=8 exercises the (np-1)/2 symmetry loop with three interiors.
        let (x8, _) = mfem_gauss_lobatto_01(8);
        assert_eq!(x8[0], 0.0);
        assert_eq!(x8[7], 1.0);
        for i in 0..4 {
            assert_eq!(x8[i] + x8[7 - i], 1.0, "GLL np=8 symmetry {i}");
        }
    }

    /// The D259 port must agree bit-for-bit with fem-element's hard-coded
    /// `[0,1]` tables where those were verified against MFEM (n ≤ 5), so the
    /// default path's control points stay untouched.
    #[test]
    fn mfem_gauss_legendre_agrees_with_fem_element_tables_up_to_five() {
        for n in 1..=5 {
            let (a, wa) = mfem_gauss_legendre_01(n);
            let (b, wb) = gauss_legendre_01(n);
            assert_eq!(a, b, "GL node tables diverge at n={n}");
            assert_eq!(wa, wb, "GL weight tables diverge at n={n}");
        }
    }

    /// The PLBound control points are `poly1d.GetPoints(ncp-3, 0)` = the GL
    /// rule with `ncp-2` points — checked against the C++ dump at n = 10
    /// (`f_quad5 -l2 -bt 0` runs with ncp = 12 ⇒ GL(10) control points),
    /// where the old generic Newton path used to be 1 ulp off.
    #[test]
    fn control_points_gl10_match_cpp_dump() {
        let (x, _) = mfem_gauss_legendre_01(10);
        let first_cpp = 0.013046735741414158;
        assert_eq!(x[0], first_cpp);
        assert_eq!(x[9], 1.0 - first_cpp);
    }

    // ── H1 tensor dof map (moved from the round-41 miniapp tests) ───────────

    /// Pin: the H1 tensor dof map is a permutation with the MFEM H1 slot
    /// convention (lex node 0 ↔ slot 0 for every order; quad and hex).
    #[test]
    fn h1_tensor_dof_map_is_permutation() {
        let (map, nb) = match h1_tensor_nodes(ElementType::Quad4, 3) {
            Ok((nodes, map)) => (map, nodes.len()),
            Err(e) => panic!("{e}"),
        };
        assert_eq!(nb * nb, map.len());
        let mut seen = vec![false; map.len()];
        for &s in &map {
            seen[s] = true;
        }
        assert!(seen.iter().all(|&s| s), "quad p3 dof map must be a permutation");
        assert_eq!(map[0], 0, "lex (0,0) is the first H1 vertex dof");
        // Hex: same property at order 2.
        let (map3, nb3) = match h1_tensor_nodes(ElementType::Hex8, 2) {
            Ok((nodes, map)) => (map, nodes.len()),
            Err(e) => panic!("{e}"),
        };
        assert_eq!(nb3 * nb3 * nb3, map3.len());
        let mut seen3 = vec![false; map3.len()];
        for &s in &map3 {
            seen3[s] = true;
        }
        assert!(seen3.iter().all(|&s| s), "hex p2 dof map must be a permutation");
        // Hex 1-D nodes must be rescaled into [0,1].
        let (nodes3, _) = h1_tensor_nodes(ElementType::Hex8, 2).ok().unwrap();
        assert!(
            nodes3.iter().all(|&x| (0.0..=1.0).contains(&x)),
            "hex nodes must be in [0,1], got {nodes3:?}"
        );
    }

    // ── D256/D257: projection + L2 (GL) bounds on a unit-quad P2 field ──────
    //
    // Semantics ground truth is the C++ pair (`NodalFiniteElement::Project` +
    // `GridFunction::ProjectGridFunction`): the projected L2 field interpolates
    // the H1 field exactly at the GL node positions, and the bounds sandwich
    // every projected coefficient.  End-to-end 6-digit parity with
    // `mpirun -np 1 gfb_cpp -l2 -bt 0` is pinned in the miniapp tests
    // (`tmp/d274/l2bt0_fhex2.out`: the hex fixture PL min 0.27827 vs the node-exact H1 0.309007).

    fn analytic(x: f64, y: f64) -> f64 {
        2.0 + (std::f64::consts::TAU * x + 1.0).sin() * (3.0 * std::f64::consts::PI * y).cos()
    }

    #[test]
    fn l2_gl_projection_and_bounds_on_unit_quad() {
        // 2×2 quad elements on the unit square, P2.
        let mesh: Mesh<2> = Mesh::unit_square_quad(2);
        let order = 2usize;
        let nb = order + 1;

        // Sample the analytic field at the H1 GLL nodes (per element slot).
        let h1 = fem_space::H1Space::new(mesh.clone(), order as u8);
        let mut h1_dofs = vec![0.0_f64; h1.n_dofs()];
        let qk = QuadQk::new(order);
        let gll_xy = qk.dof_coords();
        for e in 0..mesh.n_elements() as u32 {
            let ex = (e % 2) as f64 * 0.5;
            let ey = (e / 2) as f64 * 0.5;
            for (slot, xy) in gll_xy.iter().enumerate() {
                let g = h1.element_dofs(e)[slot];
                h1_dofs[g as usize] = analytic(ex + xy[0] * 0.5, ey + xy[1] * 0.5);
            }
        }
        let gf_h1 = GridFunction::new(&h1, h1_dofs);

        // D256 projection onto the L2 GL space.
        let projected =
            project_h1_to_l2(&gf_h1, order, BoundsBasis::GaussLegendre).expect("projection");
        let l2 = L2Space::new_with_basis(mesh.clone(), order as u8, L2Basis::GaussLegendre);
        assert_eq!(projected.len(), l2.n_dofs(), "projected dof count");

        // Interpolation property: coefficient i of element e equals the H1
        // interpolant evaluated at that element's GL node (up to evaluation
        // round-off) — `NodalFiniteElement::Project` interpolates the source
        // field, which coincides with the analytic function only at the GLL
        // nodes.
        let gl_xy = QuadL2GL::new(order).dof_coords();
        for e in 0..mesh.n_elements() as u32 {
            for k in 0..nb * nb {
                let g = l2.element_dofs(e)[k] as usize;
                let want = gf_h1.evaluate_at_element(e, &gl_xy[k]);
                assert!(
                    (projected[g] - want).abs() < 1e-12,
                    "projected coefficient (e{e} k{k}): {} vs {want}",
                    projected[g]
                );
            }
        }

        // D257 bounds on the projected field.
        let gf_l2 = GridFunction::new(&l2, projected.clone());
        let (plb, lower, upper) =
            get_element_bounds_in(&gf_l2, 2, 1, BoundsSpace::L2(BoundsBasis::GaussLegendre))
                .expect("L2 bounds");
        // GL basis, nb = 3: min_ncp_gl = 6, ncp = max(6, ref*(2+1)) = 6.
        assert_eq!(plb.n_control_points(), 6);
        let gmin = lower.iter().cloned().reduce(f64::min).unwrap();
        let gmax = upper.iter().cloned().reduce(f64::max).unwrap();
        let (rec_min, _ru) = estimate_function_minimum_in(
            &gf_l2,
            &plb,
            BoundsSpace::L2(BoundsBasis::GaussLegendre),
            1,
            4,
            1e-4,
        );
        let (_rl, rec_max) = estimate_function_maximum_in(
            &gf_l2,
            &plb,
            BoundsSpace::L2(BoundsBasis::GaussLegendre),
            1,
            4,
            1e-4,
        );
        // Tightening and sandwich invariants.
        assert!(rec_min >= gmin - 1e-12, "recursion min {rec_min} below PL min {gmin}");
        assert!(rec_max <= gmax + 1e-12, "recursion max {rec_max} above PL max {gmax}");
        for &c in &projected {
            assert!(c >= gmin - 1e-12 && c <= gmax + 1e-12, "coefficient {c} outside bounds");
        }
    }
}
