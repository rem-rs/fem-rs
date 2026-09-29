//! Bit-exact port of MFEM 4.10's Neo-Hookean hyperelastic element kernels
//! (D842-2).
//!
//! MFEM's `examples/ex10.cpp` assembles the hyperelastic operator through
//! `HyperelasticNLFIntegrator` + `NeoHookeanModel` (`fem/nonlininteg.cpp`)
//! on top of a chain of small primitives whose *floating-point evaluation
//! order* is observable in the assembled Newton tangent:
//!
//! * `QuadratureFunctions1D::GaussLobatto` (Newton-iterated GL points,
//!   `fem/intrules.cpp:708`) and the tensor Gauss-Legendre rule
//!   `IntRules.Get(Geometry::SQUARE, 2*order+3)` — 4x4 for the P2 beam,
//!   *not* the `2*order+1` rule the generic Rust form used;
//! * `Poly_1D::Basis` in Barycentric mode (`fem/fe/fe_base.cpp:1936`);
//! * `BiLinear2DFiniteElement::CalcDShape`
//!   (`fem/fe/fe_fixed_order.cpp:113`) for the straight-quad geometry
//!   transformation, `IsoparametricTransformation::EvalJacobian`
//!   (`fem/eltrans.cpp:444`), `kernels::CalcInverse<2>` and
//!   `DenseMatrix::Det/Weight`;
//! * `EvalW` / `EvalP` / `AssembleH` statement-for-statement, with MFEM's
//!   closed-form 2x2 determinant (not a general LU), the `Z *= (1.0/dJ)`
//!   reciprocal scaling, `MultABt`/`AddMultABt` k-outer accumulation and the
//!   exact two-loop tangent split of `AssembleH`.
//!
//! The positions entering the kernels are the **deformed node positions**
//! (MFEM treats `elfun` as the position field, not a displacement), which is
//! what makes MFEM's `EvalW(F=I)` return `−7.9e-19`-style round-off noise
//! instead of an exact zero (ex10's EE0 line).
//!
//! Scope: straight quadrilateral elements (the ex10 default `beam-quad`
//! family), any field order.  [`MfemHyperelasticQuad`] exposes the three
//! element kernels; the caller supplies per-slot reference dof coordinates
//! and per-slot deformed positions (interleaved `x,y`), matching the slot
//! order of `VectorH1Space::element_dofs`.

use std::f64::consts::PI;

use fem_element::quadrature::{gauss_legendre_01, gauss_legendre_01_newton_mfem};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::ref_elem::h1_field_element;

use fem_element::lagrange::PyramidBasisType;

/// Bit-exact MFEM Neo-Hookean hyperelastic element kernels on straight quads.
pub struct MfemHyperelasticQuad {
    /// Field polynomial order `p` (MFEM `H1_FECollection(p, 2)`).
    order: u8,
    /// Shear modulus (MFEM `NeoHookeanModel(mu, K)`).
    mu: f64,
    /// Bulk modulus (MFEM `K`).
    bulk: f64,
    dim: usize,
    /// Field dof count per element, `(p+1)^2`.
    dof: usize,
    /// GL points on `[0,1]` of the field basis (`poly1d.ClosedPoints(p, …)`).
    field_nodes: Vec<f64>,
    /// Barycentric weights of the field basis.
    field_bw: Vec<f64>,
    /// Quadrature points `[0,1]^2` and weights: tensor of GL `n = order/2+1`
    /// with `x` varying fastest (MFEM `IntegrationRule(seg, seg)` layout).
    quad_pts: Vec<[f64; 2]>,
    quad_w: Vec<f64>,
}

impl MfemHyperelasticQuad {
    /// Build the kernels for field order `p` on the straight-quad mesh
    /// `mesh`.  Verifies the mesh is made of straight quads.
    pub fn new(mesh: &Mesh<2>, order: u8, mu: f64, bulk: f64) -> Self {
        debug_assert_eq!(mesh.element_type(0), ElementType::Quad4,
            "MfemHyperelasticQuad: only straight quads are supported (ex10 default)");
        let dim = 2;
        let dof = (order as usize + 1) * (order as usize + 1);
        let field_nodes = gauss_lobatto_points(order as usize + 1);
        let field_bw = barycentric_weights(&field_nodes);
        // IntRules.Get(SQUARE, 2*p+3): odd order → n = order/2 + 1 Gauss points.
        let ir_order = 2 * order as usize + 3;
        let n = ir_order / 2 + 1;
        let (xs, ws) = gauss_legendre_01(n);
        let mut quad_pts = Vec::with_capacity(n * n);
        let mut quad_w = Vec::with_capacity(n * n);
        // MFEM IntegrationRule(a, b): first (x) index fastest.
        for yj in 0..n {
            for xi in 0..n {
                quad_pts.push([xs[xi], xs[yj]]);
                quad_w.push(ws[xi] * ws[yj]);
            }
        }
        Self { order, mu, bulk, dim, dof, field_nodes, field_bw, quad_pts, quad_w }
    }

    /// Field order.
    pub fn order(&self) -> u8 { self.order }

    /// Number of slot positions expected per element (`(p+1)^2`).
    pub fn dof(&self) -> usize { self.dof }

    /// Quadrature order (`2*p+3`, MFEM `HyperelasticNLFIntegrator`).
    pub fn integration_order(&self) -> usize { 2 * self.order as usize + 3 }

    /// The quadrature rule (points `[0,1]^2`, weights) — probe/diagnostic.
    pub fn quadrature(&self) -> (&[[f64; 2]], &[f64]) {
        (&self.quad_pts, &self.quad_w)
    }

    /// Element elastic energy `GetElementEnergy` (MFEM `nonlininteg.cpp:393`).
    ///
    /// Note the energy path is **not** the residual path: MFEM forms
    /// `Jpr = PMatIᵀ·DSh` first (`MultAtB`) and then `Jpt = Jpr·Jrt`
    /// (`Mult`), while `AssembleElementVector/Grad` form `DS = DSh·Jrt`
    /// first and then `Jpt = PMatIᵀ·DS`.  The two associativities round
    /// differently, so this method mirrors `GetElementEnergy` statement for
    /// statement.
    ///
    /// `corners` — the four geometry vertex coordinates (element vertex
    /// order); `ref_xy[k]` — slot `k`'s reference dof coordinates in `[0,1]^2`
    /// (the `h1_field_element(Quad4, order)` slot order);
    /// `pos[k*2 + c]` — the deformed position dofs (interleaved).
    pub fn element_energy(
        &self,
        corners: &[[f64; 2]],
        ref_xy: &[[f64; 2]],
        pos: &[f64],
    ) -> f64 {
        let dof = self.dof;
        let mut energy = 0.0_f64;
        for (q, ip) in self.quad_pts.iter().enumerate() {
            let jrt = self.jrt_at_ip(corners, *ip);
            let dsh = self.dsh_at_ip(ref_xy, *ip);
            // Ttr.Weight(): DenseMatrix::Det of the geometry Jacobian.
            let t_weight = self.geom_det_at_ip(corners, *ip);
            // nonlininteg.cpp:417: MultAtB(PMatI, DSh, Jpr) — kernels::
            // AddMultAtB with beta = 0: Jpr(i,j) = Σ_k PMatI(k,i)·DSh(k,j),
            // local `val`, k ascending.
            let mut jpr = vec![0.0_f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut val = 0.0_f64;
                    for k in 0..dof {
                        val += pos[k * 2 + i] * dsh[k * 2 + j];
                    }
                    jpr[i * 2 + j] += val;
                }
            }
            // nonlininteg.cpp:418: Mult(Jpr, Jrt, Jpt) — kernels::AddMult
            // with beta = 0: Jpt(i,j) = Σ_k Jpr(i,k)·Jrt(k,j), k ascending.
            let mut jpt = vec![0.0_f64; 4];
            for j in 0..2 {
                for k in 0..2 {
                    let val = jrt[k * 2 + j];
                    for i in 0..2 {
                        jpt[i * 2 + j] += val * jpr[i * 2 + k];
                    }
                }
            }
            // nonlininteg.cpp:420: energy += ip.weight * Ttr.Weight()
            //                        * model->EvalW(Jpt);
            energy += self.quad_w[q] * t_weight * self.eval_w(&jpt);
        }
        energy
    }

    /// Element internal-force vector `AssembleElementVector`
    /// (MFEM `nonlininteg.cpp:426`).  Returns `dof*2` entries laid out
    /// `(slot, comp) → slot*2 + comp`.
    pub fn assemble_element_vector(
        &self,
        corners: &[[f64; 2]],
        ref_xy: &[[f64; 2]],
        pos: &[f64],
    ) -> Vec<f64> {
        let (dof, dim) = (self.dof, self.dim);
        let n_vec = dof * dim;
        // nonlininteg.cpp:447: elvect = 0.0
        let mut pmat_o = vec![0.0_f64; n_vec];
        for (q, ip) in self.quad_pts.iter().enumerate() {
            let (ds, jpt, weight) = self.chain_at_ip(corners, ref_xy, pos, *ip);
            // NeoHookeanModel::EvalP (nonlininteg.cpp:297)
            let mut p = self.eval_p(&jpt);
            // nonlininteg.cpp:461: P *= ip.weight * Ttr.Weight();
            let w = self.quad_w[q] * weight;
            for v in p.iter_mut() { *v *= w; }
            // nonlininteg.cpp:462: AddMultABt(DS, P, PMatO);
            add_mult_abt(&ds, &p, &mut pmat_o, dof, dim);
        }
        pmat_o
    }

    /// Element tangent `AssembleElementGrad` (MFEM `nonlininteg.cpp:466`).
    /// Returns the `(dof*2)^2` row-major matrix with entry
    /// `(row, col) → row*n_vec + col`, `row = slot*2 + comp`.
    pub fn assemble_element_grad(
        &self,
        corners: &[[f64; 2]],
        ref_xy: &[[f64; 2]],
        pos: &[f64],
    ) -> Vec<f64> {
        let (dof, dim) = (self.dof, self.dim);
        let n_vec = dof * dim;
        // nonlininteg.cpp:486: elmat = 0.0
        let mut elmat = vec![0.0_f64; n_vec * n_vec];
        for (q, ip) in self.quad_pts.iter().enumerate() {
            let (ds, jpt, weight) = self.chain_at_ip(corners, ref_xy, pos, *ip);
            // nonlininteg.cpp:498:
            //   model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
            self.assemble_h(&jpt, &ds, self.quad_w[q] * weight, &mut elmat);
        }
        elmat
    }

    // ── Per-IP chain ────────────────────────────────────────────────────────

    /// Probe accessor: the per-integration-point chain in MFEM order —
    /// `(Jrt, DSh, DS, Jpt, Ttr.Weight())` for quadrature point `q` (the
    /// oracle pin dumps exactly these, `tmp/d92b/tangent_probe.cpp`).
    pub fn probe_chain(
        &self,
        corners: &[[f64; 2]],
        ref_xy: &[[f64; 2]],
        pos: &[f64],
        q: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let (ds, jpt, w) = self.chain_at_ip(corners, ref_xy, pos, self.quad_pts[q]);
        let jrt = self.jrt_at_ip(corners, self.quad_pts[q]);
        let dsh = self.dsh_at_ip(ref_xy, self.quad_pts[q]);
        (jrt, dsh, ds, jpt, w)
    }

    /// Geometry Jacobian inverse `CalcInverse(Ttr.Jacobian())` at ip `q`.
    fn jrt_at_ip(&self, corners: &[[f64; 2]], ip: [f64; 2]) -> Vec<f64> {
        let jgeom = self.geom_jacobian_at_ip(corners, ip);
        let det = jgeom[0] * jgeom[3] - jgeom[1] * jgeom[2];
        let recip = 1.0 / det;
        // MFEM `CalcInverse(Ttr.Jacobian(), Jrt)` lands in
        // `kernels::CalcInverse<2>` = adjugate (NOT adjugate-transpose):
        // Jrt(0,1) = −Jg(0,1)/det, Jrt(1,0) = −Jg(1,0)/det.  (Row-major
        // jgeom: [0]=(0,0), [1]=(0,1), [2]=(1,0), [3]=(1,1).)  The two
        // conventions agree only when the skew round-off Jg(0,1) == Jg(1,0)
        // — e.g. the pin's unit square, where both are exact ±0.0 — which is
        // why the element pin alone could not catch this (D842-2).
        vec![
            jgeom[3] * recip,
            -(jgeom[1]) * recip,
            -(jgeom[2]) * recip,
            jgeom[0] * recip,
        ]
    }

    /// `Ttr.Jacobian()` = `Mult(PointMat, dshape, dFdx)` (row-major 2×2),
    /// the per-entry accumulation being `kernels::AddMult`: k ascending,
    /// i innermost.
    fn geom_jacobian_at_ip(&self, corners: &[[f64; 2]], ip: [f64; 2]) -> [f64; 4] {
        geom_jacobian(corners, ip)
    }

    /// `DenseMatrix::Det` of the geometry Jacobian — `Ttr.Weight()` for the
    /// straight-quad transformation.
    fn geom_det_at_ip(&self, corners: &[[f64; 2]], ip: [f64; 2]) -> f64 {
        let jgeom = geom_jacobian(corners, ip);
        jgeom[0] * jgeom[3] - jgeom[1] * jgeom[2]
    }

    /// Field `CalcDShape` at ip `q` (slot-major, 2 columns).
    fn dsh_at_ip(&self, ref_xy: &[[f64; 2]], ip: [f64; 2]) -> Vec<f64> {
        let dof = self.dof;
        let (ux, dux) = barycentric_eval(ip[0], &self.field_nodes, &self.field_bw);
        let (uy, duy) = barycentric_eval(ip[1], &self.field_nodes, &self.field_bw);
        let mut dsh = vec![0.0_f64; dof * 2];
        for (slot, rc) in ref_xy.iter().enumerate().take(dof) {
            let (i, j) = slot_ij(*rc, &self.field_nodes);
            dsh[slot * 2] = dux[i] * uy[j];
            dsh[slot * 2 + 1] = ux[i] * duy[j];
        }
        dsh
    }

    /// `Jpt = ∇x·Jrt⁻¹`, plus `DS = DSh·Jrt` and `Ttr.Weight()`.
    /// Returns `(DS, Jpt, Ttr.Weight())`.
    fn chain_at_ip(
        &self,
        corners: &[[f64; 2]],
        ref_xy: &[[f64; 2]],
        pos: &[f64],
        ip: [f64; 2],
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let dof = self.dof;

        let jrt = self.jrt_at_ip(corners, ip);
        let t_weight = self.geom_det_at_ip(corners, ip); // ElementTransformation::Weight() → DenseMatrix::Det

        // Field CalcDShape (fe_h1.cpp:170 H1_QuadrilateralElement):
        //   dshape(slot,0) = dshape_x(i)*shape_y(j);
        //   dshape(slot,1) =  shape_x(i)*dshape_y(j);
        let dsh = self.dsh_at_ip(ref_xy, ip);

        // nonlininteg.cpp:495: Mult(DSh, Jrt, DS) — kernels::AddMult:
        //   DS(i,j) = Σ_k Jrt(k,j)·DSh(i,k), k ascending.
        let mut ds = vec![0.0_f64; dof * 2];
        for j in 0..2 {
            for k in 0..2 {
                let val = jrt[k * 2 + j];
                for i in 0..dof {
                    ds[i * 2 + j] += val * dsh[i * 2 + k];
                }
            }
        }

        // nonlininteg.cpp:496: MultAtB(PMatI, DS, Jpt) — kernels::AddMultAtB:
        //   Jpt(i,j) = Σ_k PMatI(k,i)·DS(k,j) accumulated in a local `val`,
        //   stored into the (beta=0) zeroed matrix — `0 + (−0.0) == +0.0`.
        // PMatI(k,c) = pos[k*2 + c] (the column-major view of the blocked
        // elfun has the same values as the interleaved slot layout).
        let mut jpt = vec![0.0_f64; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0_f64;
                for k in 0..dof {
                    val += pos[k * 2 + i] * ds[k * 2 + j];
                }
                jpt[i * 2 + j] += val;
            }
        }
        (ds, jpt, t_weight)
    }

    // ── NeoHookeanModel (nonlininteg.cpp:281-384) ───────────────────────────

    /// `NeoHookeanModel::EvalW` (nonlininteg.cpp:281).
    fn eval_w(&self, jpt: &[f64]) -> f64 {
        let dim = self.dim as f64;
        let d_j = det2x2(jpt);
        // g == 1.0 (ex10 scalar-constant model): sJ = dJ/g.
        let s_j = d_j / 1.0;
        let b_i1 = crate::physics::glibc_pow::glibc_pow(d_j, -2.0 / dim) * frobenius2(jpt);
        // 0.5*(mu*(bI1 - dim) + K*(sJ - 1.0)*(sJ - 1.0));
        0.5 * (self.mu * (b_i1 - dim) + self.bulk * (s_j - 1.0) * (s_j - 1.0))
    }

    /// `NeoHookeanModel::EvalP` (nonlininteg.cpp:297).
    fn eval_p(&self, jpt: &[f64]) -> Vec<f64> {
        let dim_f = self.dim as f64;
        let d_j = det2x2(jpt);
        // CalcAdjugateTranspose(J, Z) — unscaled.  Row-major:
        //   Z(0,0)=J(1,1); Z(1,0)=−J(0,1); Z(0,1)=−J(1,0); Z(1,1)=J(0,0).
        let z = [jpt[3], -jpt[2], -jpt[1], jpt[0]];
        let a = self.mu * crate::physics::glibc_pow::glibc_pow(d_j, -2.0 / dim_f);
        // b = K*(dJ/g - 1.0)/g - a*(J*J)/(dim*dJ);
        let b = self.bulk * (d_j / 1.0 - 1.0) / 1.0 - a * frobenius2(jpt) / (dim_f * d_j);
        // P = 0.0; P.Add(a, J); P.Add(b, Z);
        let mut p = vec![0.0_f64; 4];
        for (pi, &jv) in p.iter_mut().zip(jpt.iter()) {
            *pi = a * jv;
        }
        for (pi, &zv) in p.iter_mut().zip(z.iter()) {
            *pi += b * zv;
        }
        p
    }

    /// `NeoHookeanModel::AssembleH` (nonlininteg.cpp:318) into `elmat`
    /// (row-major `n_vec × n_vec`; MFEM's entry `(i + j*dof, k + l*dof)` in
    /// the component-blocked slot indexing == our `slot*2 + comp` layout).
    fn assemble_h(&self, jpt: &[f64], ds: &[f64], weight: f64, elmat: &mut [f64]) {
        let dof = self.dof;
        let dim = self.dim;
        let dim_f = dim as f64;
        let d_j = det2x2(jpt);
        let s_j = d_j / 1.0;
        let mut a = self.mu * crate::physics::glibc_pow::glibc_pow(d_j, -2.0 / dim_f);
        let bc = a * frobenius2(jpt) / dim_f;
        let mut b = bc - self.bulk * s_j * (s_j - 1.0);
        let mut c = 2.0 * bc / dim_f + self.bulk * s_j * (2.0 * s_j - 1.0);

        // CalcAdjugateTranspose(J, Z); Z *= (1.0/dJ);
        //   Z(0,0)=J(1,1); Z(1,0)=−J(0,1); Z(0,1)=−J(1,0); Z(1,1)=J(0,0).
        let mut z = [jpt[3], -jpt[2], -jpt[1], jpt[0]];
        let recip = 1.0 / d_j;
        for v in z.iter_mut() { *v *= recip; }

        // MultABt(DS, J, C): C(i,j) = Σ_k DS(i,k)·J(j,k), ascending k.
        let cm = mult_abt(ds, jpt, dof, dim);
        // MultABt(DS, Z, G): G = DS J^{-1}.
        let g = mult_abt(ds, &z, dof, dim);

        a *= weight;
        b *= weight;
        c *= weight;

        let n_vec = dof * dim;
        // Part 1 (nonlininteg.cpp:350-370).
        for i in 0..dof {
            for k in 0..=i {
                let mut s = 0.0_f64;
                for d in 0..dim {
                    s += ds[i * 2 + d] * ds[k * 2 + d];
                }
                s *= a;
                for d in 0..dim {
                    let row = (i * dim + d) * n_vec + (k * dim + d);
                    elmat[row] += s;
                }
                if k != i {
                    for d in 0..dim {
                        let row = (k * dim + d) * n_vec + (i * dim + d);
                        elmat[row] += s;
                    }
                }
            }
        }

        a *= -2.0 / dim_f;

        // Part 2 (nonlininteg.cpp:375-383).
        for i in 0..dof {
            for j in 0..dim {
                for k in 0..dof {
                    for l in 0..dim {
                        let c_ij = cm[i * 2 + j];
                        let g_kl = g[k * 2 + l];
                        let g_ij = g[i * 2 + j];
                        let c_kl = cm[k * 2 + l];
                        let g_il = g[i * 2 + l];
                        let g_kj = g[k * 2 + j];
                        let row = i * dim + j;
                        let col = k * dim + l;
                        elmat[row * n_vec + col] +=
                            a * (c_ij * g_kl + g_ij * c_kl) + b * g_il * g_kj + c * g_ij * g_kl;
                    }
                }
            }
        }
    }
}

// ── Slot helpers ─────────────────────────────────────────────────────────────

/// Slot (i, j) on the `p+1` GL grid: nearest node lookup (the reference dof
/// coordinates and the basis nodes describe the same GL lattice; nearest
/// matching is exact for p ≤ 2, where ex10's default lives).
#[inline]
fn slot_ij(ref_xy: [f64; 2], nodes: &[f64]) -> (usize, usize) {
    let nearest = |v: f64| -> usize {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (idx, &n) in nodes.iter().enumerate() {
            let d = (v - n).abs();
            if d < best_d {
                best_d = d;
                best = idx;
            }
        }
        debug_assert!(best_d < 1e-10, "dof coord {v} not on the GL node grid");
        best
    };
    (nearest(ref_xy[0]), nearest(ref_xy[1]))
}

#[inline]
fn det2x2(j: &[f64]) -> f64 {
    j[0] * j[3] - j[1] * j[2]
}

/// MFEM `(J*J)`: `DenseMatrix::operator*` sums `data[i]*data[i]` over the
/// linear (column-major) storage — for 2x2: J00, J10, J01, J11.
#[inline]
fn frobenius2(j: &[f64]) -> f64 {
    let mut a = 0.0_f64;
    a += j[0] * j[0];
    a += j[2] * j[2];
    a += j[1] * j[1];
    a += j[3] * j[3];
    a
}

/// `MultABt(A, B)`: `C(i,j) = Σ_k A(i,k)·B(j,k)` (kernels zero then `+=`).
fn mult_abt(a: &[f64], b: &[f64], dof: usize, dim: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; dof * dim];
    for k in 0..dim {
        for j in 0..dim {
            let b_jk = b[j * 2 + k];
            for i in 0..dof {
                c[i * 2 + j] += a[i * 2 + k] * b_jk;
            }
        }
    }
    c
}

/// `AddMultABt(A, B, C)`: `C(i,j) += Σ_k A(i,k)·B(j,k)` accumulating directly
/// into `C` (kernels do not zero and do not stage a local accumulator).
fn add_mult_abt(a: &[f64], b: &[f64], c: &mut [f64], dof: usize, dim: usize) {
    for k in 0..dim {
        for j in 0..dim {
            let b_jk = b[j * 2 + k];
            for i in 0..dof {
                c[i * 2 + j] += a[i * 2 + k] * b_jk;
            }
        }
    }
}

// ── Poly_1D barycentric basis (fe_base.cpp:1811/1936) ────────────────────────

/// `Poly_1D::Basis` Barycentric constructor weights.
fn barycentric_weights(nodes: &[f64]) -> Vec<f64> {
    let p = nodes.len() - 1;
    let mut w = vec![1.0_f64; p + 1];
    for i in 0..=p {
        for j in 0..i {
            let xij = nodes[i] - nodes[j];
            w[i] *= xij;
            w[j] *= -xij;
        }
    }
    for v in w.iter_mut() { *v = 1.0 / *v; }
    w
}

/// `Poly_1D::Basis::Eval(y, u)` — the **values-only** Barycentric overload
/// (fe_base.cpp:1875).  Note it is a DIFFERENT expression from the
/// values+derivatives overload below: `u(i) = l·w(i)/(y−x(i))` here versus
/// `u(i) = l·(1/(y−x(i)))·w(i)` there — the rounding association differs by
/// 1 ulp at irrational points, which is observable in `CalcShape`-based
/// kernels (mass) but not in `CalcDShape`-based ones (diffusion/hyperelastic).
fn barycentric_eval_u(y: f64, x: &[f64], w: &[f64]) -> Vec<f64> {
    let p = x.len() - 1;
    let mut u = vec![0.0_f64; p + 1];
    if p == 0 {
        u[0] = 1.0;
        return u;
    }
    let mut lk = 1.0_f64;
    let mut k = p;
    for kk in 0..p {
        if y >= (x[kk] + x[kk + 1]) / 2.0 {
            lk *= y - x[kk];
        } else {
            for i in kk + 1..=p {
                lk *= y - x[i];
            }
            k = kk;
            break;
        }
    }
    let l = lk * (y - x[k]);
    for i in 0..k {
        u[i] = l * w[i] / (y - x[i]);
    }
    u[k] = lk * w[k];
    for i in k + 1..=p {
        u[i] = l * w[i] / (y - x[i]);
    }
    u
}

/// `Poly_1D::Basis::Eval(y, u, d)` Barycentric path (fe_base.cpp:1936).
fn barycentric_eval(y: f64, x: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let p = x.len() - 1;
    let mut u = vec![0.0_f64; p + 1];
    let mut d = vec![0.0_f64; p + 1];
    if p == 0 {
        u[0] = 1.0;
        d[0] = 0.0;
        return (u, d);
    }
    let mut lk = 1.0_f64;
    // The C loop leaves k == p after normal completion, k == break index else.
    let mut k = p;
    for kk in 0..p {
        if y >= (x[kk] + x[kk + 1]) / 2.0 {
            lk *= y - x[kk];
        } else {
            for i in kk + 1..=p {
                lk *= y - x[i];
            }
            k = kk;
            break;
        }
    }
    let l = lk * (y - x[k]);

    let mut sk = 0.0_f64;
    for i in 0..k {
        let si = 1.0 / (y - x[i]);
        sk += si;
        u[i] = l * si * w[i];
    }
    u[k] = lk * w[k];
    for i in k + 1..=p {
        let si = 1.0 / (y - x[i]);
        sk += si;
        u[i] = l * si * w[i];
    }
    let lp = l * sk + lk;

    for i in 0..k {
        d[i] = (lp * w[i] - u[i]) / (y - x[i]);
    }
    d[k] = sk * u[k];
    for i in k + 1..=p {
        d[i] = (lp * w[i] - u[i]) / (y - x[i]);
    }
    (u, d)
}

/// `QuadratureFunctions1D::GaussLobatto` points (intrules.cpp:708), `[0,1]`.
fn gauss_lobatto_points(np: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; np];
    if np == 1 {
        x[0] = 0.5;
        return x;
    }
    x[0] = 0.0;
    x[np - 1] = 1.0;
    for i in 1..=(np - 1) / 2 {
        let mut x_i = (PI * (i as f64 / (np - 1) as f64 - 0.5)).sin();
        let mut z_i = 0.0_f64;
        let mut p_l;
        let mut done = false;
        loop {
            // Legendre up to P_{np-1} at x_i.
            let mut p_lm1 = 1.0_f64;
            p_l = x_i;
            for l in 1..np - 1 {
                let p_lp1 =
                    ((2 * l + 1) as f64 * x_i * p_l - l as f64 * p_lm1) / (l + 1) as f64;
                p_lm1 = p_l;
                p_l = p_lp1;
            }
            if done {
                break;
            }
            let dx = (x_i * p_l - p_lm1) / (np as f64 * p_l);
            if dx.abs() < 1e-16 {
                done = true;
                z_i = ((1.0 + x_i) - dx) / 2.0;
            }
            x_i -= dx;
        }
        x[i] = z_i;
        x[np - 1 - i] = 1.0 - z_i;
    }
    x
}

// ── Whole-mesh assembly (MFEM NonlinearForm element loop) ────────────────────

/// Whole-mesh MFEM-exact hyperelastic assembler on a straight-quad mesh.
///
/// MFEM semantics: the field entering the kernels is the **deformed position
/// field** `x_ref + displacement` (MFEM's `x` GridFunction), blocked per
/// component.  The assembler holds the reference positions (the scalar space's
/// dof coordinates) and the per-element dof maps supplied by the caller.
pub struct MfemHyperelasticAssembler {
    kernels: MfemHyperelasticQuad,
    mesh: Mesh<2>,
    /// Reference dof coordinates per slot (`h1_field_element(Quad4, order)`).
    ref_xy: Vec<[f64; 2]>,
    /// Reference positions per component (physical scalar dof coordinates).
    x_ref: [Vec<f64>; 2],
    /// Per-element scalar dof ids.
    scalar_dofs: Vec<Vec<u32>>,
    /// Per-element interleaved vector dofs (`VectorH1Space::element_dofs`).
    elem_dofs: Vec<Vec<usize>>,
}

impl MfemHyperelasticAssembler {
    /// Build for field order `p` with Neo-Hookean `(mu, K)` on the refined
    /// mesh, using the scalar space's dof coordinates and element dof maps.
    pub fn new(
        mesh: Mesh<2>,
        order: u8,
        mu: f64,
        bulk: f64,
        x_ref: [Vec<f64>; 2],
        scalar_dofs: Vec<Vec<u32>>,
        elem_dofs: Vec<Vec<usize>>,
    ) -> Self {
        let kernels = MfemHyperelasticQuad::new(&mesh, order, mu, bulk);
        let re = h1_field_element(ElementType::Quad4, order, PyramidBasisType::default());
        let ref_xy: Vec<[f64; 2]> =
            re.dof_coords().iter().map(|c| [c[0], c[1]]).collect();
        Self { kernels, mesh, ref_xy, x_ref, scalar_dofs, elem_dofs }
    }

    pub fn kernels(&self) -> &MfemHyperelasticQuad { &self.kernels }

    /// The reference dof coordinates per slot (probe/diagnostic).
    pub fn ref_xy(&self) -> &[[f64; 2]] { &self.ref_xy }

    fn element_buffers(&self, field: &[Vec<f64>], elem_scalar_dofs: &[u32]) -> Vec<f64> {
        let dof = self.kernels.dof();
        let mut pos = vec![0.0_f64; dof * 2];
        for (k, &s) in elem_scalar_dofs.iter().enumerate().take(dof) {
            pos[k * 2] = field[0][s as usize];
            pos[k * 2 + 1] = field[1][s as usize];
        }
        pos
    }

    fn element_corners(&self, e: u32) -> Vec<[f64; 2]> {
        self.mesh.element_nodes(e).iter().map(|&n| {
            let c = self.mesh.node_coords(n);
            [c[0], c[1]]
        }).collect()
    }

    /// `H(x)` internal-force assembly: `y[dof] += element vector`, element
    /// order ascending (MFEM `NonlinearForm::Mult` accumulation).
    pub fn assemble_residual(
        &self,
        field: &[Vec<f64>],
        elem_scalar_dofs_all: &[Vec<u32>],
        y: &mut [f64],
    ) {
        // MFEM `NonlinearForm::Mult` zeroes the output before the element
        // loop (`py = 0.0`, nonlinearform.cpp:270) — the accumulation below
        // (`py.AddElementVector`) is an ADD onto the zeroed vector, so this
        // entry point carries the same contract.
        for v in y.iter_mut() {
            *v = 0.0;
        }
        for (e, dofs) in elem_scalar_dofs_all.iter().enumerate() {
            let corners = self.element_corners(e as u32);
            let pos = self.element_buffers(field, dofs);
            let ev = self.kernels.assemble_element_vector(&corners, &self.ref_xy, &pos);
            let n_scalar = field[0].len();
            for (k, &s) in dofs.iter().enumerate() {
                y[s as usize] += ev[k * 2];
                y[s as usize + n_scalar] += ev[k * 2 + 1];
            }
        }
    }

    /// MFEM `NonlinearForm::GetGradient` element phase on `H(x_ref + disp)`:
    /// one `Grad->AddSubMatrix(vdofs, vdofs, elmat, 0)` per element
    /// (ascending) into an **open LIL** — the caller mirrors the tail
    /// (`Finalize(0)`, then the ess `EliminateRowCol(·, DIAG_ONE)` loop).
    pub fn grad_lil(&self, disp: &[f64]) -> MfemLilMatrix {
        let field = self.displaced_field(disp);
        let n_scalar = field[0].len();
        let mut lil = MfemLilMatrix::new(n_scalar * 2);
        let dof = self.kernels.dof();
        for (e, sd) in self.scalar_dofs.iter().enumerate() {
            let corners = self.element_corners(e as u32);
            let mut pos = vec![0.0_f64; sd.len() * 2];
            for (k, &s) in sd.iter().enumerate() {
                pos[2 * k] = field[0][s as usize];
                pos[2 * k + 1] = field[1][s as usize];
            }
            let em = self.kernels.assemble_element_grad(&corners, &self.ref_xy, &pos);
            let em_b = blocked_from_interleaved(&em, dof);
            // byNODES vdofs: [component-0 dofs; component-1 dofs (+ n_scalar)].
            let vdofs: Vec<u32> = sd
                .iter()
                .copied()
                .chain(sd.iter().map(|&d| d + n_scalar as u32))
                .collect();
            lil.add_sub_matrix(&vdofs, &em_b);
        }
        lil
    }

    /// Total elastic energy `Σ elements GetElementEnergy` (element order).
    pub fn total_energy(
        &self,
        field: &[Vec<f64>],
        elem_scalar_dofs_all: &[Vec<u32>],
    ) -> f64 {
        let mut energy = 0.0;
        for (e, dofs) in elem_scalar_dofs_all.iter().enumerate() {
            let corners = self.element_corners(e as u32);
            let pos = self.element_buffers(field, dofs);
            energy += self.kernels.element_energy(&corners, &self.ref_xy, &pos);
        }
        energy
    }

    /// Deformed position field `x_ref + displacement` (blocked per component;
    /// the displacement layout matches the ex10 `x` block).
    pub fn displaced_field(&self, disp: &[f64]) -> [Vec<f64>; 2] {
        let n_scalar = self.x_ref[0].len();
        let mut p = [self.x_ref[0].clone(), self.x_ref[1].clone()];
        for i in 0..n_scalar {
            p[0][i] += disp[i];
            p[1][i] += disp[n_scalar + i];
        }
        p
    }

    /// `H(x_ref + disp)` internal force (MFEM `NonlinearForm::Mult` on the
    /// deformed positions).
    pub fn residual_displaced(&self, disp: &[f64], y: &mut [f64]) {
        let field = self.displaced_field(disp);
        self.assemble_residual(&field, &self.scalar_dofs, y);
    }

    /// `H(pos)` internal force with `pos` already holding the **deformed
    /// positions** in the blocked `[x-dofs; y-dofs]` layout — the exact
    /// operand of MFEM's `H->Mult(pos, y)` when the caller's state vector is
    /// a position field (ex10's `x_gf`).
    pub fn residual_positions(&self, pos: &[f64], y: &mut [f64]) {
        let n = self.x_ref[0].len();
        let field = [pos[..n].to_vec(), pos[n..].to_vec()];
        self.assemble_residual(&field, &self.scalar_dofs, y);
    }

    /// Elastic energy at the given **deformed positions** (blocked layout).
    pub fn energy_positions(&self, pos: &[f64]) -> f64 {
        let n = self.x_ref[0].len();
        let field = [pos[..n].to_vec(), pos[n..].to_vec()];
        self.total_energy(&field, &self.scalar_dofs)
    }

    /// `GetGradient` element phase at the given **deformed positions**
    /// (blocked layout) — see [`Self::grad_lil`].
    pub fn grad_lil_positions(&self, pos: &[f64]) -> MfemLilMatrix {
        let n = self.x_ref[0].len();
        let field = [&pos[..n], &pos[n..]];
        let mut lil = MfemLilMatrix::new(n * 2);
        let dof = self.kernels.dof();
        for (e, sd) in self.scalar_dofs.iter().enumerate() {
            let corners = self.element_corners(e as u32);
            let mut pos_e = vec![0.0_f64; sd.len() * 2];
            for (k, &s) in sd.iter().enumerate() {
                pos_e[2 * k] = field[0][s as usize];
                pos_e[2 * k + 1] = field[1][s as usize];
            }
            let em = self.kernels.assemble_element_grad(&corners, &self.ref_xy, &pos_e);
            let em_b = blocked_from_interleaved(&em, dof);
            let vdofs: Vec<u32> = sd
                .iter()
                .copied()
                .chain(sd.iter().map(|&d| d + n as u32))
                .collect();
            lil.add_sub_matrix(&vdofs, &em_b);
        }
        lil
    }

    /// Probe: all element internal-force vectors at the displaced state,
    /// element order ascending — the D842-2 bitwise oracle harness.
    #[doc(hidden)]
    pub fn debug_all_element_vectors(&self, disp: &[f64]) -> Vec<Vec<f64>> {
        let field = self.displaced_field(disp);
        let mut out = Vec::with_capacity(self.scalar_dofs.len());
        for (e, sd) in self.scalar_dofs.iter().enumerate() {
            let mut pos = vec![0.0_f64; sd.len() * 2];
            for (k, &s) in sd.iter().enumerate() {
                pos[2 * k] = field[0][s as usize];
                pos[2 * k + 1] = field[1][s as usize];
            }
            let corners = self.element_corners(e as u32);
            out.push(self.kernels.assemble_element_vector(&corners, &self.ref_xy, &pos));
        }
        out
    }

    /// Probe: element `e`'s (positions, element internal-force vector) at the
    /// displaced state — the D842-2 bitwise oracle harness.
    #[doc(hidden)]
    pub fn debug_element_vector(&self, disp: &[f64], e: usize) -> (Vec<f64>, Vec<f64>) {
        let field = self.displaced_field(disp);
        let sd = &self.scalar_dofs[e];
        let mut pos = vec![0.0_f64; sd.len() * 2];
        for (k, &s) in sd.iter().enumerate() {
            pos[2 * k] = field[0][s as usize];
            pos[2 * k + 1] = field[1][s as usize];
        }
        let corners = self.element_corners(e as u32);
        let ev = self.kernels.assemble_element_vector(&corners, &self.ref_xy, &pos);
        (pos, ev)
    }

    /// Probe variant taking **deformed positions** directly (blocked layout).
    #[doc(hidden)]
    pub fn debug_element_vector_positions(&self, pos: &[f64], e: usize) -> (Vec<f64>, Vec<f64>) {
        let n = self.x_ref[0].len();
        let field = [&pos[..n], &pos[n..]];
        let sd = &self.scalar_dofs[e];
        let mut pos_e = vec![0.0_f64; sd.len() * 2];
        for (k, &s) in sd.iter().enumerate() {
            pos_e[2 * k] = field[0][s as usize];
            pos_e[2 * k + 1] = field[1][s as usize];
        }
        let corners = self.element_corners(e as u32);
        let ev = self.kernels.assemble_element_vector(&corners, &self.ref_xy, &pos_e);
        (pos_e, ev)
    }

    /// Probe variant taking **deformed positions** directly (blocked layout).
    #[doc(hidden)]
    pub fn debug_chain_positions(
        &self,
        pos: &[f64],
        e: usize,
        q: usize,
    ) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.x_ref[0].len();
        let field = [&pos[..n], &pos[n..]];
        let sd = &self.scalar_dofs[e];
        let mut pos_e = vec![0.0_f64; sd.len() * 2];
        for (k, &s) in sd.iter().enumerate() {
            pos_e[2 * k] = field[0][s as usize];
            pos_e[2 * k + 1] = field[1][s as usize];
        }
        let corners = self.element_corners(e as u32);
        let ip = self.kernels.quadrature().0[q];
        let (ds, jpt, w) = self.kernels.chain_at_ip(&corners, &self.ref_xy, &pos_e, ip);
        let jr = self.kernels.jrt_at_ip(&corners, ip);
        let p = self.kernels.eval_p(&jpt);
        (w, jr, ds, jpt, p)
    }

    /// Probe: element `e`'s per-IP chain at the displaced state —
    /// `(weight, Jrt, DS, Jpt, P)` for quadrature point `q`.
    #[doc(hidden)]
    pub fn debug_chain(
        &self,
        disp: &[f64],
        e: usize,
        q: usize,
    ) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let field = self.displaced_field(disp);
        let sd = &self.scalar_dofs[e];
        let mut pos = vec![0.0_f64; sd.len() * 2];
        for (k, &s) in sd.iter().enumerate() {
            pos[2 * k] = field[0][s as usize];
            pos[2 * k + 1] = field[1][s as usize];
        }
        let corners = self.element_corners(e as u32);
        let ip = self.kernels.quadrature().0[q];
        let (ds, jpt, w) = self.kernels.chain_at_ip(&corners, &self.ref_xy, &pos, ip);
        let jr = self.kernels.jrt_at_ip(&corners, ip);
        let p = self.kernels.eval_p(&jpt);
        (w, jr, ds, jpt, p)
    }

    /// Total elastic energy at the deformed positions `x_ref + disp`.
    pub fn energy_displaced(&self, disp: &[f64]) -> f64 {
        let field = self.displaced_field(disp);
        self.total_energy(&field, &self.scalar_dofs)
    }

    /// `grad_H(x_ref + disp)` as a CSR matrix; element matrices are scattered
    /// with the interleaved element dofs in element order (MFEM
    /// `NonlinearForm::GetGradient`).
    pub fn jacobian_displaced(&self, disp: &[f64]) -> fem_linalg::CsrMatrix<f64> {
        let field = self.displaced_field(disp);
        let n = field[0].len() * 2;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for (e, dofs) in self.elem_dofs.iter().enumerate() {
            let corners = self.element_corners(e as u32);
            let sd = &self.scalar_dofs[e];
            let mut pos = vec![0.0_f64; sd.len() * 2];
            for (k, &s) in sd.iter().enumerate() {
                pos[2 * k] = field[0][s as usize];
                pos[2 * k + 1] = field[1][s as usize];
            }
            let em = self.kernels.assemble_element_grad(&corners, &self.ref_xy, &pos);
            coo.add_element_matrix(dofs, &em);
        }
        coo.into_csr()
    }
}

// ── MFEM SparseMatrix storage pipeline (D842-2, round-92 continuation) ───────
//
// MFEM's `BilinearForm::Assemble` / `NonlinearForm::GetGradient` build their
// CSR through an open linked-list phase whose *row-internal entry order* is
// observable in any matrix-vector product: `AddSubMatrix` inserts per (row,
// col) through `SearchRow` — a walk from the row head with find-or-PREPEND —
// and `Finalize(0)` emits each row **head→tail** (i.e. reverse insertion
// order) without sorting or merging.  ex10's Newton operator is then
// `SparseMatrix::Add(1.0, M, dt, S)` (order-preserving CSR merge: M's row
// entries first, then S's new columns in S's order) followed by the member
// `Add(dt², G)` (in-place entrywise adds — no reorder).  The MINRES inner
// solve of the default run does not converge to its 1e-8 tolerance within its
// 300 iterations on this tangent, so the round-off of every `y += Σ
// (stored-order) A(i,k)·x(k)` is macroscopically amplified into the Newton
// trajectory: bit-exact reproduction therefore needs the storage order, not
// just the values.

/// MFEM open `SparseMatrix` (LIL) semantics: per-row prepend chain with
/// `SearchRow` find-or-prepend accumulation and `Finalize(0)` head→tail
/// emission.
pub struct MfemLilMatrix {
    n: usize,
    /// `rows[i]` holds the row's nodes with the **head last** (`push` =
    /// prepend; head→tail walk = `.rev()`), mirroring MFEM's `RowNode::Prev`
    /// chain.
    rows: Vec<Vec<(u32, f64)>>,
}

impl MfemLilMatrix {
    /// An open `n × n` matrix (all rows empty).
    pub fn new(n: usize) -> Self {
        MfemLilMatrix { n, rows: vec![Vec::new(); n] }
    }

    /// `SparseMatrix::SearchRow(col)`: walk the chain from the head; found →
    /// mutable value, missing → new node **prepended** with `Value = 0.0`.
    fn search_row_add(&mut self, row: usize, col: u32, val: f64) {
        let nodes = &mut self.rows[row];
        let mut hit = None;
        for node in nodes.iter_mut().rev() {
            if node.0 == col {
                hit = Some(node);
                break;
            }
        }
        match hit {
            // `_Add_(col, val)`: node value += val.
            Some(node) => node.1 += val,
            None => {
                // SearchRow creates the node with Value = 0.0 and `_Add_`
                // then does `SearchRow(col) += val`, i.e. `0.0 + val` (which
                // maps −0.0 to +0.0).
                nodes.push((col, 0.0 + val));
            }
        }
    }

    /// `SparseMatrix::AddSubMatrix(rows, cols, subm, skip_zeros = 0)`:
    /// per matrix row `i` (vdofs order) a `SetColPtr(rows[i])` pass that
    /// `_Add_`s `subm(i, j)` at column `cols[j]`, j in vdofs order.  `subm`
    /// is row-major in the same (vdofs) layout — MFEM's component-blocked
    /// element matrix.
    pub fn add_sub_matrix(&mut self, vdofs: &[u32], subm: &[f64]) {
        let nv = vdofs.len();
        for (i, &gi) in vdofs.iter().enumerate() {
            for (j, &gj) in vdofs.iter().enumerate() {
                let a = subm[i * nv + j];
                self.search_row_add(gi as usize, gj, a);
            }
        }
    }

    /// `SparseMatrix::Finalize(0)`: each row is written **head→tail** (the
    /// reverse of the insertion sequence), no sorting, no merging.
    pub fn finalize(self) -> CsrMatrix<f64> {
        let n = self.n;
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for (i, nodes) in self.rows.iter().enumerate() {
            row_ptr[i] = col_idx.len();
            for &(c, v) in nodes.iter().rev() {
                col_idx.push(c);
                values.push(v);
            }
        }
        row_ptr[n] = col_idx.len();
        CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values }
    }
}

/// `IsoparametricTransformation::EvalJacobian` on a straight quad:
/// `Mult(PointMat, dshape_geom, dFdx)` — `kernels::AddMult`, k ascending,
/// i innermost.  Row-major 2×2.
pub(crate) fn geom_jacobian(corners: &[[f64; 2]], ip: [f64; 2]) -> [f64; 4] {
    let x = ip[0];
    let y = ip[1];
    // `BiLinear2DFiniteElement::CalcDShape` (fe_fixed_order.cpp:124).
    let dg: [[f64; 2]; 4] = [
        [-1. + y, -1. + x],
        [1. - y, -x],
        [y, x],
        [-y, 1. - x],
    ];
    let mut jgeom = [0.0_f64; 4];
    for j in 0..2 {
        for k in 0..4 {
            let val = dg[k][j];
            for i in 0..2 {
                jgeom[i * 2 + j] += val * corners[k][i];
            }
        }
    }
    jgeom
}

/// `DenseMatrix::Det` of the geometry Jacobian — `Ttr.Weight()` on this
/// transformation (pinned bitwise via the chain dumps).
pub(crate) fn geom_det(corners: &[[f64; 2]], ip: [f64; 2]) -> f64 {
    let jgeom = geom_jacobian(corners, ip);
    jgeom[0] * jgeom[3] - jgeom[1] * jgeom[2]
}

/// Field `CalcDShape` at `ip` (slot-major, 2 columns) — shared by the
/// bilinear kernels.
fn dshape_at(ref_xy: &[[f64; 2]], ip: [f64; 2], nodes: &[f64], bw: &[f64]) -> Vec<f64> {
    let dof = ref_xy.len();
    let (ux, dux) = barycentric_eval(ip[0], nodes, bw);
    let (uy, duy) = barycentric_eval(ip[1], nodes, bw);
    let mut dsh = vec![0.0_f64; dof * 2];
    for (slot, rc) in ref_xy.iter().enumerate().take(dof) {
        let (i, j) = slot_ij(*rc, nodes);
        dsh[slot * 2] = dux[i] * uy[j];
        dsh[slot * 2 + 1] = ux[i] * duy[j];
    }
    dsh
}

/// MFEM-exact bilinear element kernels for ex10's mass/viscosity operators —
/// `VectorMassIntegrator` / `VectorDiffusionIntegrator` (bilininteg.cpp:1641 /
/// :3049) on straight quads, statement for statement:
///
/// * mass: `norm = ip.weight·Ttr.Weight()` (`·ρ`), `MultVVt`, `partelmat *=
///   norm`, one `AddMatrix(partelmat, nd·k, nd·k)` per component;
/// * diffusion: `w = ip.weight / Ttr.Weight()`, `Mult(dshape, AdjJ, dxt)`
///   (`kernels::AddMult`, `CalcAdjugate` plain adjugate), `w *= κ`,
///   `Mult_a_AAt(w, dxt, pelmat)` (local accumulator, `a·d` LAST), one
///   `AddMatrix` per component;
/// * both on the `IntRules.Get(SQUARE, 2p+1)` tensor Gauss-Legendre rule
///   (`p+1` points per direction; mass `2p + OrderW()` with `OrderW = 1`,
///   diffusion `GetRule` = `p + p + dim − 1` — the same order).
///
/// Element matrices are returned in MFEM's component-**blocked** layout
/// `(slot + comp·nd)` — the `vdofs` layout of a byNODES vector space.
pub struct MfemBilinearQuad {
    dof: usize,
    field_nodes: Vec<f64>,
    field_bw: Vec<f64>,
    quad_pts: Vec<[f64; 2]>,
    quad_w: Vec<f64>,
}

impl MfemBilinearQuad {
    /// Build the kernels for field order `p`.
    pub fn new(order: u8) -> Self {
        let dof = (order as usize + 1) * (order as usize + 1);
        let field_nodes = gauss_lobatto_points(order as usize + 1);
        let field_bw = barycentric_weights(&field_nodes);
        let rule_order = 2 * order as usize + 1;
        let n = rule_order / 2 + 1;
        // MFEM `IntRules.Get(SEGMENT, rule_order)` runs
        // `QuadratureFunctions1D::GaussLegendre` (Newton) — the D339 port;
        // the hard-coded `gauss_legendre_01` table differs at 1 ulp for
        // n = 2, 3.
        let (xs, ws) = gauss_legendre_01_newton_mfem(n);
        let mut quad_pts = Vec::with_capacity(n * n);
        let mut quad_w = Vec::with_capacity(n * n);
        for yj in 0..n {
            for xi in 0..n {
                quad_pts.push([xs[xi], xs[yj]]);
                quad_w.push(ws[xi] * ws[yj]);
            }
        }
        MfemBilinearQuad { dof, field_nodes, field_bw, quad_pts, quad_w }
    }

    /// The quadrature rule (probe/diagnostic).
    pub fn quadrature(&self) -> (&[[f64; 2]], &[f64]) {
        (&self.quad_pts, &self.quad_w)
    }

    fn shapes_at(&self, ref_xy: &[[f64; 2]], ip: [f64; 2]) -> Vec<f64> {
        let ux = barycentric_eval_u(ip[0], &self.field_nodes, &self.field_bw);
        let uy = barycentric_eval_u(ip[1], &self.field_nodes, &self.field_bw);
        let nd = self.dof;
        let mut shape = vec![0.0_f64; nd];
        for (slot, rc) in ref_xy.iter().enumerate().take(nd) {
            let (i, j) = slot_ij(*rc, &self.field_nodes);
            shape[slot] = ux[i] * uy[j];
        }
        shape
    }

    /// `VectorMassIntegrator::AssembleElementMatrix` with density `rho`
    /// (ex10: `ConstantCoefficient(1.0)`), blocked layout `(slot + k·nd)`.
    pub fn mass_element(&self, corners: &[[f64; 2]], ref_xy: &[[f64; 2]], rho: f64) -> Vec<f64> {
        let nd = self.dof;
        let vdim = 2usize;
        let n = nd * vdim;
        let mut elmat = vec![0.0_f64; n * n];
        let mut partelmat = vec![0.0_f64; nd * nd];
        for (q, ip) in self.quad_pts.iter().enumerate() {
            let shape = self.shapes_at(ref_xy, *ip);
            // norm = ip.weight * Trans.Weight(); norm *= Q->Eval(...) (=ρ).
            let mut norm = self.quad_w[q] * geom_det(corners, *ip);
            norm *= rho;
            // MultVVt(shape, partelmat): i outer, j ≤ i,
            // `vvt(i,j) = vvt(j,i) = v(i)·v(j)`.
            for i in 0..nd {
                for j in 0..=i {
                    let v = shape[i] * shape[j];
                    partelmat[i * nd + j] = v;
                    partelmat[j * nd + i] = v;
                }
            }
            // partelmat *= norm.
            for v in partelmat.iter_mut() {
                *v *= norm;
            }
            // AddMatrix(partelmat, nd·k, nd·k) per component.
            for k in 0..vdim {
                for i in 0..nd {
                    for j in 0..nd {
                        elmat[(i + k * nd) * n + (j + k * nd)] += partelmat[i * nd + j];
                    }
                }
            }
        }
        elmat
    }

    /// `VectorDiffusionIntegrator::AssembleElementMatrix` with conductivity
    /// `kappa` (ex10: `ConstantCoefficient(viscosity)`), blocked layout.
    pub fn diffusion_element(
        &self,
        corners: &[[f64; 2]],
        ref_xy: &[[f64; 2]],
        kappa: f64,
    ) -> Vec<f64> {
        let nd = self.dof;
        let vdim = 2usize;
        let n = nd * vdim;
        let mut elmat = vec![0.0_f64; n * n];
        let mut dxt = vec![0.0_f64; nd * 2];
        let mut pelmat = vec![0.0_f64; nd * nd];
        for (q, ip) in self.quad_pts.iter().enumerate() {
            // el.CalcDShape(ip, dshape) — slot-major (dof × 2).
            let dsh = dshape_at(ref_xy, *ip, &self.field_nodes, &self.field_bw);
            // w = Trans.Weight(); w = ip.weight / w  (square).
            let wgt = geom_det(corners, *ip);
            let mut w = self.quad_w[q] / wgt;
            // CalcAdjugate(J) — plain adjugate (densemat.cpp:2620).
            let jgeom = geom_jacobian(corners, *ip);
            let adj = [jgeom[3], -jgeom[1], -jgeom[2], jgeom[0]];
            // Mult(dshape, AdjJ, dshapedxt) — kernels::AddMult, beta = 0:
            // the kernel ZEROES `A` first, then dxt(i,j) =
            // Σ_k dshape(i,k)·adj(k,j), k ascending.
            for v in dxt.iter_mut() {
                *v = 0.0;
            }
            for j in 0..2 {
                for k in 0..2 {
                    let val = adj[k * 2 + j];
                    for i in 0..nd {
                        dxt[i * 2 + j] += val * dsh[i * 2 + k];
                    }
                }
            }
            // w *= Q->Eval(...) (=κ).
            w *= kappa;
            // Mult_a_AAt(w, dxt, pelmat): i outer, j ≤ i, local accumulator,
            // `AAt(i,j) = AAt(j,i) = a·d` (multiply LAST).
            for i in 0..nd {
                for j in 0..=i {
                    let mut d = 0.0_f64;
                    for k in 0..2 {
                        d += dxt[i * 2 + k] * dxt[j * 2 + k];
                    }
                    pelmat[i * nd + j] = w * d;
                    pelmat[j * nd + i] = w * d;
                }
            }
            // AddMatrix(pelmat, nd·k, nd·k) per component.
            for k in 0..vdim {
                for i in 0..nd {
                    for j in 0..nd {
                        elmat[(i + k * nd) * n + (j + k * nd)] += pelmat[i * nd + j];
                    }
                }
            }
        }
        elmat
    }
}

/// `SparseMatrix *Add(a, A, b, B)` (sparsemat.cpp:4110) — order-preserving
/// CSR merge: each output row is `A`'s row entries (A's stored order) with
/// `a·value`, followed by `B`'s row entries at columns `A` lacks (in B's
/// stored order) with `b·value`; shared columns accumulate `C += b·B` in
/// place.
pub fn mfem_add(a: f64, am: &CsrMatrix<f64>, b: f64, bm: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = am.nrows;
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        row_ptr[row] = col_idx.len();
        for k in am.row_ptr[row]..am.row_ptr[row + 1] {
            col_idx.push(am.col_idx[k]);
            values.push(a * am.values[k]);
        }
        let base = row_ptr[row];
        let len = col_idx.len() - base;
        for k in bm.row_ptr[row]..bm.row_ptr[row + 1] {
            let col = bm.col_idx[k];
            let add = b * bm.values[k];
            match col_idx[base..base + len].iter().position(|&c| c == col) {
                Some(p) => values[base + p] += add,
                None => {
                    col_idx.push(col);
                    values.push(add);
                }
            }
        }
    }
    row_ptr[n] = col_idx.len();
    CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values }
}

/// `SparseMatrix::Add(a, B)` member (sparsemat.cpp:3231) — in-place
/// entrywise `A(i,j) += a·B(i,j)` over **A's** stored entries (no structural
/// change; ex10's `M + dt·S` sparsity ⊇ grad_H's).
pub fn mfem_add_in_place(j: &mut CsrMatrix<f64>, a: f64, bm: &CsrMatrix<f64>) {
    for row in 0..j.nrows {
        for k in j.row_ptr[row]..j.row_ptr[row + 1] {
            let col = j.col_idx[k];
            let bv = bm.get(row, col as usize);
            j.values[k] += a * bv;
        }
    }
}

/// `MfemHyperelasticQuad::assemble_element_grad` returns the interleaved
/// `(slot·2 + comp)` layout; MFEM's `vdofs`/`elmat` are component-**blocked**
/// `(slot + comp·nd)`.  Bitwise permutation.
pub fn blocked_from_interleaved(em: &[f64], dof: usize) -> Vec<f64> {
    let n = dof * 2;
    let mut out = vec![0.0_f64; n * n];
    for s in 0..dof {
        for c in 0..2 {
            for t in 0..dof {
                for d in 0..2 {
                    out[(s + c * dof) * n + (t + d * dof)] =
                        em[(s * 2 + c) * n + (t * 2 + d)];
                }
            }
        }
    }
    out
}
