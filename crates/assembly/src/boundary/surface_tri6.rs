//! Tri6 (P2) surface finite element integrators for 3-D embedded surfaces.
//!
//! Supports the same operations as [`super::SurfaceAssembler`] but for
//! 6-node quadratic triangles (P2) on a 2-D manifold in 3-D space.
//!
//! D1274: the element kernels are a 1:1 port of MFEM 4.10's evaluation chain
//! for `H1_FECollection(2)` on `Geometry::TRIANGLE` with an isoparametric
//! (curved) transformation, so that the assembled surface system is bitwise
//! identical to MFEM's (required for ex7 stdout BIT):
//!
//! * shapes/dshapes come from MFEM's `H1_TriangleElement` product basis
//!   (`fem/fe/fe_h1.cpp:533/555`): the hierarchical Chebyshev 1-D basis
//!   (`Poly_1D::CalcBasis` → `CalcChebyshev`, `fem/fe/fe_base.cpp:2391`)
//!   combined via `T(o,k) = C_i(x_k)·C_j(y_k)·C_k(1-x_k-y_k)` and mapped
//!   through the LU inverse `Ti` (`LUFactors::Factor/Solve`,
//!   `linalg/densemat.cpp`) — NOT the closed-form P2 barycentric
//!   polynomials (last-ulp differences);
//! * the geometry chain is `IsoparametricTransformation` on the Tri6 rows:
//!   `J = PointMat·dshape` (`kernels::AddMult` accumulation), `Weight() =
//!   sqrt(E·G − F·F)` for a 3×2 Jacobian (`DenseMatrix::Weight`,
//!   `linalg/densemat.cpp:553`), `AdjugateJacobian = CalcAdjugate`
//!   (`linalg/densemat.cpp:2565`);
//! * the per-integrator quadrature rules follow MFEM's selection formulas
//!   (see each integrator); the tables themselves are
//!   [`fem_element::quadrature::tri_rule_mfem_order`] — the pinned 1:1 port
//!   of `IntegrationRules::TriangleIntegrationRule` (D578/D603).

use std::sync::OnceLock;

use fem_element::quadrature::tri_rule_mfem_order;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use super::surface::get_coord3;
use super::surface::{SurfaceTri6BilinearIntegrator, SurfaceTri6LinearIntegrator};

// ─── MFEM 4.10 H1(2)-triangle evaluation machinery ──────────────────────────

/// `Poly_1D::CalcChebyshev(p=2, x, u, d)` (`fem/fe/fe_base.cpp:2391`) — the
/// hierarchical 1-D basis used by `Poly_1D::CalcBasis` (`fe_base.hpp:1220`,
/// `CalcChebyshev` branch): `z = 2x−1`, `u0 = 1`, `u1 = z`,
/// `u_{n+1} = 2z·u_n − u_{n−1}`; derivatives via
/// `d_{n+1} = (n+1)·(z·d_n/n + 2·u_n)`.
///
/// NOTE: `H1_TriangleElement` builds its shape functions from THIS
/// hierarchical basis (not the GaussLobatto nodal basis — that is only the
/// dof-point set), so the values feed the T matrix / `CalcBasis` products
/// below.
fn calc_chebyshev(x: f64) -> ([f64; 3], [f64; 3]) {
    let z = 2.0 * x - 1.0;
    let mut u = [1.0f64, z, 0.0];
    let mut d = [0.0f64, 2.0, 0.0];
    // n = 1 (p = 2)
    u[2] = 2.0 * z * u[1] - u[0];
    d[2] = 2.0 * (z * d[1] / 1.0 + 2.0 * u[1]);
    (u, d)
}

/// The 6 dof points of `H1_TriangleElement(2)` in MFEM's Nodes order
/// (`fem/fe/fe_h1.cpp:477-508`): vertices, then edge i=1..p-1 of the three
/// edges (0→1, 1→2, 2→0).  GL2 = {0, 1/2, 1}.
const TRI2_DOF_PTS: [(f64, f64); 6] =
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)];

/// `H1_TriangleElement(2)`'s dof-basis matrix T (`fem/fe/fe_h1.cpp:513`)
/// factored exactly like `LUFactors::Factor` without LAPACK
/// (`linalg/densemat.cpp:3420`): partial pivoting, column-major data.
fn factor_tri2() -> ([f64; 36], [i32; 6]) {
    // T(o,k) = sx(i)·sy(j)·sl(2-i-j), (i,j) lex with j outer — column k.
    let mut t = [0.0f64; 36];
    for (k, &(dx, dy)) in TRI2_DOF_PTS.iter().enumerate() {
        let (sx, _) = calc_chebyshev(dx);
        let (sy, _) = calc_chebyshev(dy);
        let (sl, _) = calc_chebyshev(1.0 - dx - dy);
        let mut o = 0;
        for j in 0..3 {
            for i in 0..=(2 - j) {
                t[o + k * 6] = sx[i] * sy[j] * sl[2 - i - j];
                o += 1;
            }
        }
    }
    let mut lu = t;
    let mut ipiv = [0i32; 6];
    for i in 0..6 {
        // pivoting
        let mut piv = i;
        let mut a = lu[piv + i * 6].abs();
        for j in i + 1..6 {
            let b = lu[j + i * 6].abs();
            if b > a {
                a = b;
                piv = j;
            }
        }
        ipiv[i] = (piv + 1) as i32;
        if piv != i {
            for j in 0..6 {
                lu.swap(i + j * 6, piv + j * 6);
            }
        }
        let a_ii_inv = 1.0 / lu[i + i * 6];
        for j in i + 1..6 {
            lu[j + i * 6] *= a_ii_inv;
        }
        for k in i + 1..6 {
            let a_ik = lu[i + k * 6];
            for j in i + 1..6 {
                lu[j + k * 6] -= a_ik * lu[j + i * 6];
            }
        }
    }
    (lu, ipiv)
}

/// The factorized `Ti` of the P2 triangle — process-wide constant.
fn tri2_lu() -> &'static ([f64; 36], [i32; 6]) {
    static LU: OnceLock<([f64; 36], [i32; 6])> = OnceLock::new();
    LU.get_or_init(factor_tri2)
}

/// `LUFactors::LSolve` + `USolve` (`linalg/kernels.hpp:1760/1785`) applied to
/// the first `ncols` columns of a column-major 6×n block, in place.
fn lu_solve_in_place(lu: &[f64; 36], ipiv: &[i32; 6], x: &mut [f64], ncols: usize) {
    for c in 0..ncols {
        let col = &mut x[c * 6..c * 6 + 6];
        // X <- P X, X <- L^{-1} X
        for i in 0..6 {
            col.swap(i, ipiv[i] as usize - 1);
        }
        for j in 0..6 {
            let xj = col[j];
            for i in j + 1..6 {
                col[i] -= lu[i + j * 6] * xj;
            }
        }
        // X <- U^{-1} X
        for j in (0..6).rev() {
            col[j] /= lu[j + j * 6];
            let xj = col[j];
            for i in 0..j {
                col[i] -= lu[i + j * 6] * xj;
            }
        }
    }
}

/// `H1_TriangleElement::CalcShape` (`fem/fe/fe_h1.cpp:533`): product basis
/// mapped through `Ti.Mult` (LU solve).
fn h1_tri2_shape(ipx: f64, ipy: f64) -> [f64; 6] {
    let (sx, _) = calc_chebyshev(ipx);
    let (sy, _) = calc_chebyshev(ipy);
    let (sl, _) = calc_chebyshev(1.0 - ipx - ipy);
    let mut u = [0.0f64; 6];
    let mut o = 0;
    for j in 0..3 {
        for i in 0..=(2 - j) {
            u[o] = sx[i] * sy[j] * sl[2 - i - j];
            o += 1;
        }
    }
    let (lu, ipiv) = tri2_lu();
    lu_solve_in_place(lu, ipiv, &mut u, 1);
    u
}

/// `H1_TriangleElement::CalcDShape` (`fem/fe/fe_h1.cpp:555`): 6×2 reference
/// derivative matrix, column-major (`du(o, c)` at `o + c*6`).
fn h1_tri2_dshape(ipx: f64, ipy: f64) -> [f64; 12] {
    let (sx, dsx) = calc_chebyshev(ipx);
    let (sy, dsy) = calc_chebyshev(ipy);
    let (sl, dsl) = calc_chebyshev(1.0 - ipx - ipy);
    let mut du = [0.0f64; 12];
    let mut o = 0;
    for j in 0..3 {
        for i in 0..=(2 - j) {
            let k = 2 - i - j;
            du[o] = ((dsx[i] * sl[k]) - (sx[i] * dsl[k])) * sy[j];
            du[o + 6] = ((dsy[j] * sl[k]) - (sy[j] * dsl[k])) * sx[i];
            o += 1;
        }
    }
    let (lu, ipiv) = tri2_lu();
    lu_solve_in_place(lu, ipiv, &mut du, 2);
    du
}

/// `IsoparametricTransformation::EvalJacobian`
/// (`fem/eltrans.cpp:444`): `dFdx = PointMat·dshape` with the
/// `kernels::AddMult` accumulation (k ascending, `+= val·B`).  `pm` is the
/// 3×6 point matrix in column-major layout (`pm[i + k*3]` = coordinate i of
/// node k); the result is a 3×2 Jacobian, column-major.
fn eval_jacobian(pm: &[f64; 18], dshape: &[f64; 12]) -> [f64; 6] {
    let mut dfdx = [0.0f64; 6];
    for j in 0..2 {
        for k in 0..6 {
            let val = dshape[k + j * 6];
            for i in 0..3 {
                dfdx[i + j * 3] += val * pm[i + k * 3];
            }
        }
    }
    dfdx
}

/// `DenseMatrix::Weight()` for a 3×2 Jacobian (`linalg/densemat.cpp:591`):
/// `sqrt(E·G − F·F)` — this is `Trans.Weight()`, i.e. `sqrt(det(JᵀJ))`.
fn weight_3x2(d: &[f64; 6]) -> f64 {
    let e = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let g = d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
    let f = d[0] * d[3] + d[1] * d[4] + d[2] * d[5];
    (e * g - f * f).sqrt()
}

/// `CalcAdjugate` for a 3×2 matrix (`linalg/densemat.cpp:2606`): the
/// 2×3 `adj(JᵀJ)·Jᵀ`, column-major — MFEM's `Trans.AdjugateJacobian()`.
fn calc_adjugate_3x2(d: &[f64; 6]) -> [f64; 6] {
    let e = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let g = d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
    let f = d[0] * d[3] + d[1] * d[4] + d[2] * d[5];
    [
        d[0] * g - d[3] * f,
        d[3] * e - d[0] * f,
        d[1] * g - d[4] * f,
        d[4] * e - d[1] * f,
        d[2] * g - d[5] * f,
        d[5] * e - d[2] * f,
    ]
}

/// `Mult(dshape, adjJ, dshapedxt)` via the `kernels::AddMult` accumulation:
/// 6×2 · 2×3 → 6×3 column-major.
fn mult_dshape_adj(dshape: &[f64; 12], adj: &[f64; 6]) -> [f64; 18] {
    let mut dxt = [0.0f64; 18];
    for j in 0..3 {
        for k in 0..2 {
            let val = adj[k + j * 2];
            for i in 0..6 {
                dxt[i + j * 6] += val * dshape[i + k * 6];
            }
        }
    }
    dxt
}

/// `AddMult_a_AAt` (`linalg/densemat.cpp:3241`): `AAt += a·A·Aᵀ` with the
/// exact loop/rounding structure (off-diagonal `d·=a` added to both halves,
/// diagonal `a·d`).
fn add_mult_a_aat(a: f64, a_mat: &[f64; 18], aat: &mut [f64; 36]) {
    for i in 0..6 {
        for j in 0..i {
            let mut d = 0.0f64;
            for k in 0..3 {
                d += a_mat[i + k * 6] * a_mat[j + k * 6];
            }
            d *= a;
            aat[i * 6 + j] += d;
            aat[j * 6 + i] += d;
        }
        let mut d = 0.0f64;
        for k in 0..3 {
            d += a_mat[i + k * 6] * a_mat[i + k * 6];
        }
        aat[i * 6 + i] += a * d;
    }
}

/// `AddMult_a_VVt` (`linalg/densemat.cpp:3379`): `VVt += a·v·vᵀ`.
fn add_mult_a_vvt(a: f64, v: &[f64; 6], vvt: &mut [f64; 36]) {
    for i in 0..6 {
        let avi = a * v[i];
        for j in 0..i {
            let avivj = avi * v[j];
            vvt[i * 6 + j] += avivj;
            vvt[j * 6 + i] += avivj;
        }
        vvt[i * 6 + i] += avi * v[i];
    }
}

/// `IsoparametricTransformation::Transform(ip)` (`fem/eltrans.cpp:532`):
/// `x = PointMat·shape` with the `kernels::Mult` accumulation.
fn transform_point(pm: &[f64; 18], shape: &[f64; 6]) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for c in 0..3 {
        out[c] = shape[0] * pm[c];
    }
    for k in 1..6 {
        for c in 0..3 {
            out[c] += shape[k] * pm[c + k * 3];
        }
    }
    out
}

/// Element point matrix (column-major 3×6) from the six row coordinates.
fn point_matrix(elem_nodes: &[[f64; 3]; 6]) -> [f64; 18] {
    let mut pm = [0.0f64; 18];
    for (k, node) in elem_nodes.iter().enumerate() {
        pm[k * 3] = node[0];
        pm[k * 3 + 1] = node[1];
        pm[k * 3 + 2] = node[2];
    }
    pm
}

// ─── P2 surface integrators ───────────────────────────────────────────────

/// Surface diffusion (Laplace-Beltrami) bilinear form for Tri6: `∫_Γ ∇_Γ u · ∇_Γ v dS`
///
/// D1274: MFEM `DiffusionIntegrator::GetRule` (`fem/bilininteg.cpp:1347`) —
/// for `FunctionSpace::Pk` the order is `p_trial + p_test − 2` (no
/// `Trans.OrderW()` term), i.e. **2** for P2 → the 3-point rule
/// `IntRules.Get(TRIANGLE, 2)`.  The kernel is the non-square (2-D in 3-D)
/// path of `DiffusionIntegrator::AssembleElementMatrix`
/// (`fem/bilininteg.cpp:934`): `w = ip.weight / W³` with
/// `dshapedxt = dshape·AdjugateJacobian` and `elmat += w·dshapedxt·dshapedxtᵀ`.
pub struct SurfaceTri6DiffusionIntegrator;

impl SurfaceTri6DiffusionIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        let ir = tri_rule_mfem_order(2);
        let pm = point_matrix(elem_nodes);
        for q in 0..ir.points.len() {
            let ipx = ir.points[q][0];
            let ipy = ir.points[q][1];
            let dshape = h1_tri2_dshape(ipx, ipy);
            let dfdx = eval_jacobian(&pm, &dshape);
            let wt = weight_3x2(&dfdx);
            let w = ir.weights[q] / (wt * wt * wt);
            let adj = calc_adjugate_3x2(&dfdx);
            let dxt = mult_dshape_adj(&dshape, &adj);
            add_mult_a_aat(w, &dxt, k_elem);
        }
    }
}

/// Surface mass bilinear form for Tri6: `∫_Γ u v dS`
///
/// D1274: MFEM `MassIntegrator::GetRule` (`fem/bilininteg.cpp:1459`) —
/// `order = p_trial + p_test + Trans.OrderW()` with
/// `IsoparametricTransformation::OrderW() = (p−1)·dim = 2` for the P2
/// triangle geometry (`fem/eltrans.cpp:493`), i.e. **6** → the 12-point
/// rule `IntRules.Get(TRIANGLE, 6)`.  Kernel:
/// `w = Weight·ip.weight`, `elmat += w·shape·shapeᵀ`
/// (`AddMult_a_VVt`).
pub struct SurfaceTri6MassIntegrator;

impl SurfaceTri6MassIntegrator {
    pub fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        let ir = tri_rule_mfem_order(6);
        let pm = point_matrix(elem_nodes);
        for q in 0..ir.points.len() {
            let ipx = ir.points[q][0];
            let ipy = ir.points[q][1];
            let dshape = h1_tri2_dshape(ipx, ipy);
            let dfdx = eval_jacobian(&pm, &dshape);
            let w = weight_3x2(&dfdx) * ir.weights[q];
            let shape = h1_tri2_shape(ipx, ipy);
            add_mult_a_vvt(w, &shape, k_elem);
        }
    }
}

/// Surface domain source linear form for Tri6: `∫_Γ f(x) v(x) dS`
///
/// D1274: MFEM `DomainLFIntegrator` (default `oa=2, ob=0`,
/// `fem/lininteg.hpp:116`) selects `IntRules.Get(geom, oa·p + ob)`
/// (`fem/lininteg.cpp:64`) — order **4** → the 6-point rule.  Kernel:
/// `val = Weight·f(Transform(ip))`, `elvect += ip.weight·val·shape`
/// (the `add(v1, c, v2, v)` accumulate).
pub struct SurfaceTri6DomainSourceIntegrator<'a> {
    pub f: &'a dyn Fn(&[f64; 3]) -> f64,
}

impl SurfaceTri6DomainSourceIntegrator<'_> {
    pub fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 6], f_elem: &mut [f64; 6]) {
        let ir = tri_rule_mfem_order(4);
        let pm = point_matrix(elem_nodes);
        for q in 0..ir.points.len() {
            let ipx = ir.points[q][0];
            let ipy = ir.points[q][1];
            let dshape = h1_tri2_dshape(ipx, ipy);
            let dfdx = eval_jacobian(&pm, &dshape);
            let wt = weight_3x2(&dfdx);
            let shape_t = h1_tri2_shape(ipx, ipy);
            let x_phys = transform_point(&pm, &shape_t);
            let val = wt * (self.f)(&x_phys);
            let shape = h1_tri2_shape(ipx, ipy);
            let c = ir.weights[q] * val;
            for (i, fe_i) in f_elem.iter_mut().enumerate() {
                *fe_i += c * shape[i];
            }
        }
    }
}

// ─── Surface Assembler for Tri6 ───────────────────────────────────────────

/// Assemble a surface bilinear form using a Tri6 surface integrator.
pub struct SurfaceTri6Assembler;

impl SurfaceTri6Assembler {
    pub fn assemble_bilinear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceTri6BilinearIntegrator],
    ) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut coo = CooMatrix::new(n_dofs, n_dofs);

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 6 { continue; }
            let x: [[f64; 3]; 6] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
                get_coord3(mesh, nodes[3]),
                get_coord3(mesh, nodes[4]),
                get_coord3(mesh, nodes[5]),
            ];
            let mut ke = [0.0; 36];
            for integ in integrators {
                integ.add_to_element_matrix(&x, &mut ke);
            }
            for i in 0..6 {
                for j in 0..6 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, ke[i * 6 + j]);
                }
            }
        }
        coo.into_csr()
    }

    pub fn assemble_linear<S: FESpace>(
        space: &S,
        integrators: &[&dyn SurfaceTri6LinearIntegrator],
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let n_dofs = space.n_dofs();
        let ne = mesh.n_elements() as u32;
        let mut rhs = vec![0.0; n_dofs];

        for e in 0..ne {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 6 { continue; }
            let x: [[f64; 3]; 6] = [
                get_coord3(mesh, nodes[0]),
                get_coord3(mesh, nodes[1]),
                get_coord3(mesh, nodes[2]),
                get_coord3(mesh, nodes[3]),
                get_coord3(mesh, nodes[4]),
                get_coord3(mesh, nodes[5]),
            ];
            let mut fe = [0.0; 6];
            for integ in integrators {
                integ.add_to_element_vector(&x, &mut fe);
            }
            for i in 0..6 {
                rhs[dofs[i] as usize] += fe[i];
            }
        }
        rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tri6_surface_diffusion_runs() {
        let x = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let integ = SurfaceTri6DiffusionIntegrator;
        let mut ke = [0.0; 36];
        integ.add_to_element_matrix(&x, &mut ke);
        let trace: f64 = (0..6).map(|i| ke[i * 6 + i]).sum();
        assert!(trace > 0.0, "diffusion matrix trace should be positive");
    }

    #[test]
    fn tri6_surface_mass_runs() {
        let x = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let integ = SurfaceTri6MassIntegrator;
        let mut ke = [0.0; 36];
        integ.add_to_element_matrix(&x, &mut ke);
        let trace: f64 = (0..6).map(|i| ke[i * 6 + i]).sum();
        assert!(trace > 0.0, "mass matrix trace should be positive");
    }
}



impl SurfaceTri6BilinearIntegrator for SurfaceTri6DiffusionIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

impl SurfaceTri6BilinearIntegrator for SurfaceTri6MassIntegrator {
    fn add_to_element_matrix(&self, elem_nodes: &[[f64; 3]; 6], k_elem: &mut [f64; 36]) {
        self.add_to_element_matrix(elem_nodes, k_elem);
    }
}

impl SurfaceTri6LinearIntegrator for SurfaceTri6DomainSourceIntegrator<'_> {
    fn add_to_element_vector(&self, elem_nodes: &[[f64; 3]; 6], f_elem: &mut [f64; 6]) {
        self.add_to_element_vector(elem_nodes, f_elem);
    }
}
