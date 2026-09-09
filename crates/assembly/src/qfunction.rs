//! Quadrature functions — 1:1 port of MFEM `fem/qfunction.hpp` /
//! `fem/qfunction.cpp`.
//!
//! A [`QuadratureFunction`] represents values (or vectors of values) at the
//! quadrature points of a [`QuadratureSpace`](crate::qspace::QuadratureSpace):
//! a flat value array plus a vector dimension, with the point numbering taken
//! from the space's element→QP offsets.
//!
//! # Deviations from the C++ original (fem-rs architecture)
//!
//! - C++ `QuadratureFunction` derives from `Vector` and optionally *owns* its
//!   `QuadratureSpaceBase` (`own_qspace`).  In Rust the function borrows the
//!   space (`&dyn QuadratureSpaceBase`) and the borrow checker replaces the
//!   ownership flag, so `SetSpace` / `OwnsSpace` / the default (null-space)
//!   constructor have no equivalent.
//! - `QuadratureFunction(mesh, istream)` (stream construction with owned
//!   space) is provided as [`QuadratureFunction::load`], which borrows a
//!   caller-built space and parses the `VDim:` + values part of the format.
//! - `SaveVTU` is not ported (fem-rs has no VTK writer in `fem-assembly`).
//! - `ProjectGridFunction` uses the scalar reference-evaluation fallback
//!   (MFEM `ProjectGridFunctionFallback`), which produces the same values as
//!   MFEM's tensor-product fast path.

use std::io::{BufRead, Write};

use nalgebra::DMatrix;

use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use crate::qspace::QuadratureSpaceBase;
use crate::postproc::grid_function::GridFunction;

// ─── QuadratureFunction ──────────────────────────────────────────────────────

/// Values or vectors of values at quadrature points on a mesh (MFEM
/// `QuadratureFunction`).
///
/// The value layout is component-fastest (MFEM `QVectorLayout::byVDIM`):
/// entry `i + vdim * j` is component `i` at global quadrature point `j`.
pub struct QuadratureFunction<'a> {
    qspace: &'a dyn QuadratureSpaceBase,
    /// Vector dimension (MFEM `vdim`).
    vdim: usize,
    /// Global values, length `vdim * qspace.get_size()` (the C++ `Vector`
    /// base).
    values: Vec<f64>,
}

impl<'a> QuadratureFunction<'a> {
    /// Create a zero-filled quadrature function of unit vector dimension on
    /// `qspace` (MFEM `QuadratureFunction(QuadratureSpaceBase&, vdim = 1)`).
    pub fn new(qspace: &'a dyn QuadratureSpaceBase) -> Self {
        Self::with_vdim(qspace, 1)
    }

    /// Create a zero-filled quadrature function with the given vector
    /// dimension (MFEM `QuadratureFunction(QuadratureSpaceBase&, int vdim)`).
    pub fn with_vdim(qspace: &'a dyn QuadratureSpaceBase, vdim: usize) -> Self {
        Self {
            qspace,
            vdim,
            values: vec![0.0; vdim * qspace.get_size()],
        }
    }

    /// Read a quadrature function from a stream produced by [`Self::save`],
    /// i.e. the `VDim:` header and the value block that follows the
    /// [`QuadratureSpace`](crate::qspace::QuadratureSpace) header (MFEM
    /// `QuadratureFunction(Mesh*, std::istream&)`).
    ///
    /// The C++ constructor also reads the space itself and takes ownership;
    /// here the caller builds the space (fem-rs borrows instead of owning).
    pub fn load(
        qspace: &'a dyn QuadratureSpaceBase,
        input: &mut dyn BufRead,
    ) -> std::io::Result<Self> {
        let msg = "invalid input stream";
        let ident = crate::qspace::read_token(input)?;
        assert_eq!(ident, "VDim:", "{msg}");
        let vdim: usize = crate::qspace::read_token(input)?
            .parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, msg))?;
        let mut qf = Self::with_vdim(qspace, vdim);
        for v in qf.values.iter_mut() {
            *v = crate::qspace::read_token(input)?
                .parse::<f64>()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, msg))?;
        }
        Ok(qf)
    }

    /// The vector dimension (MFEM `GetVDim()`).
    pub fn get_vdim(&self) -> usize {
        self.vdim
    }

    /// Set the vector dimension, re-sizing (and zeroing) the value array
    /// (MFEM `SetVDim`).
    pub fn set_vdim(&mut self, vdim: usize) {
        self.vdim = vdim;
        self.values = vec![0.0; vdim * self.qspace.get_size()];
    }

    /// The associated quadrature space (MFEM `GetSpace()`).
    pub fn space(&self) -> &dyn QuadratureSpaceBase {
        self.qspace
    }

    /// Total number of stored values, `vdim * qspace.get_size()` (MFEM
    /// `Vector::Size()`).
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Borrow the global values (MFEM implicit `Vector` conversion).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutably borrow the global values.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Set all entries to `value` (MFEM `operator=(real_t)`).
    pub fn fill(&mut self, value: f64) {
        self.values.iter_mut().for_each(|v| *v = value);
    }

    /// Copy the data from `v` (MFEM `operator=(const Vector&)`).
    ///
    /// # Panics
    /// Panics unless `v.len() == self.size()`.
    pub fn assign(&mut self, v: &[f64]) {
        assert_eq!(
            v.len(),
            self.size(),
            "QuadratureFunction::assign: size mismatch"
        );
        self.values.copy_from_slice(v);
    }

    /// Evaluate a scalar grid function at each quadrature point (MFEM
    /// `ProjectGridFunction`).
    ///
    /// Sets the vector dimension to the grid function's and fills the values
    /// `u_h(x_q) = Σ_i c_i φ_i(ξ_q)` element by element, using the same
    /// solution reference elements as the rest of fem-rs assembly.
    pub fn project_grid_function<S: FESpace>(&mut self, gf: &GridFunction<S>) {
        self.set_vdim(1);
        let space = gf.space();
        let mesh: &dyn MeshTopology = space.mesh();
        let dofs = gf.dofs();
        for e in 0..self.qspace.get_ne() {
            let elem_type = mesh.element_type(e as u32);
            let fe = crate::assembler::ref_elem_vol_for_space(
                space,
                elem_type,
                space.element_order(e as u32),
            );
            let ir = self.qspace.get_int_rule(e);
            let elem_dofs = space.element_dofs(e as u32);
            let n_ldofs = fe.n_dofs();
            debug_assert_eq!(n_ldofs, elem_dofs.len());
            let mut phi = vec![0.0_f64; n_ldofs];
            let base = self.qspace.offset(e);
            for (q, xi) in ir.points.iter().enumerate() {
                fe.eval_basis(xi, &mut phi);
                let mut val = 0.0_f64;
                for (i, &p) in phi.iter().enumerate() {
                    val += p * dofs[elem_dofs[i] as usize];
                }
                self.values[base + q] = val;
            }
        }
    }

    /// All values of entity `idx`, component-fastest (MFEM
    /// `GetValues(idx, Vector&)` reference version): entry `i + vdim * j` is
    /// component `i` at the entity's `j`-th quadrature point.
    pub fn get_values(&self, idx: usize) -> &[f64] {
        let s = self.qspace.offset(idx) * self.vdim;
        let n = (self.qspace.offset(idx + 1) - self.qspace.offset(idx)) * self.vdim;
        &self.values[s..s + n]
    }

    /// Mutable variant of [`Self::get_values`].
    pub fn get_values_mut(&mut self, idx: usize) -> &mut [f64] {
        let s = self.qspace.offset(idx) * self.vdim;
        let n = (self.qspace.offset(idx + 1) - self.qspace.offset(idx)) * self.vdim;
        &mut self.values[s..s + n]
    }

    /// The values at one integration point of entity `idx` (MFEM
    /// `GetValues(idx, ip_num, Vector&)`), length `vdim`.
    pub fn get_values_at_qp(&self, idx: usize, ip_num: usize) -> &[f64] {
        let s = self.qspace.offset(idx) * self.vdim + ip_num * self.vdim;
        &self.values[s..s + self.vdim]
    }

    /// All values of entity `idx` as a `vdim x n_qp` matrix (MFEM
    /// `GetValues(idx, DenseMatrix&)`): entry `(i, j)` is component `i` at
    /// the entity's `j`-th quadrature point.
    pub fn get_values_dense(&self, idx: usize) -> DMatrix<f64> {
        let vals = self.get_values(idx);
        let n_qp = vals.len() / self.vdim;
        let mut m = DMatrix::<f64>::zeros(self.vdim, n_qp);
        for j in 0..n_qp {
            for i in 0..self.vdim {
                m[(i, j)] = vals[i + self.vdim * j];
            }
        }
        m
    }

    /// The `IntegrationRule` of entity `idx` (MFEM `GetIntRule(idx)`).
    pub fn get_int_rule(&self, idx: usize) -> &fem_element::reference::QuadratureRule {
        self.qspace.get_int_rule(idx)
    }

    /// Integral of the quadrature function; only valid for `vdim == 1` (MFEM
    /// `Integrate()`): `Σ_q values[q] · weights[q]` with the space's
    /// geometric integration weights.
    ///
    /// # Panics
    /// Panics when `vdim != 1` (MFEM `MFEM_VERIFY`).
    pub fn integrate(&self) -> f64 {
        assert!(
            self.vdim == 1,
            "QuadratureFunction::integrate: only scalar functions are supported"
        );
        let weights = self.qspace.get_weights();
        self.values
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * w)
            .sum()
    }

    /// Integrate a (possibly vector-valued) quadrature function component by
    /// component (MFEM `Integrate(Vector&)`).
    pub fn integrate_vec(&self) -> Vec<f64> {
        let weights = self.qspace.get_weights();
        let mut integrals = vec![0.0_f64; self.vdim];
        for (i, &w) in weights.iter().enumerate() {
            for (vd, int) in integrals.iter_mut().enumerate() {
                *int += self.values[vd + i * self.vdim] * w;
            }
        }
        integrals
    }

    /// Write the quadrature function to `out` in MFEM's text format (MFEM
    /// `Save`): the space header, then `VDim:` and the values, `vdim` per
    /// line.
    pub fn save(&self, out: &mut dyn Write) -> std::io::Result<()> {
        self.qspace.save(out)?;
        writeln!(out, "VDim: {}", self.vdim)?;
        writeln!(out)?;
        for chunk in self.values.chunks(self.vdim) {
            for (k, v) in chunk.iter().enumerate() {
                if k > 0 {
                    write!(out, " ")?;
                }
                write!(out, "{v:.16e}")?;
            }
            writeln!(out)?;
        }
        out.flush()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qspace::QuadratureSpace;
    use fem_mesh::element_type::ElementType;
    use fem_mesh::Mesh;
    use fem_space::l2::L2Space;

    /// 1 quad + 1 tri mixed mesh (mirrors the qspace test fixture).
    fn mixed_mesh() -> Mesh<2> {
        let coords = vec![
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 2.0, 1.0,
        ];
        let mut mesh = Mesh::<2>::uniform(
            coords,
            vec![0, 1, 2, 3, 4, 5, 6],
            vec![1, 1],
            ElementType::Quad4,
            vec![],
            vec![],
            ElementType::Line2,
        );
        mesh.elem_types = Some(vec![ElementType::Quad4, ElementType::Tri3]);
        mesh.elem_offsets = Some(vec![0, 4, 7]);
        mesh
    }

    /// (c) offsets numbering: filling the QF with the global index makes the
    /// element view read back `offset(e) + q`.
    #[test]
    fn global_indexing_via_offsets() {
        // Uniform mesh → compressed offsets.
        let mesh = Mesh::<2>::unit_square_quad(3);
        let qs = QuadratureSpace::new(&mesh, 2);
        let mut qf = QuadratureFunction::new(&qs);
        assert_eq!(qf.size(), qs.get_size());
        for (i, v) in qf.values_mut().iter_mut().enumerate() {
            *v = i as f64;
        }
        for e in 0..qs.get_ne() {
            let vals = qf.get_values(e);
            assert_eq!(vals.len(), 4);
            for (q, &v) in vals.iter().enumerate() {
                assert_eq!(v as usize, qs.offset(e) + q);
            }
            // Single-point access.
            assert_eq!(qf.get_values_at_qp(e, 2)[0] as usize, qs.offset(e) + 2);
        }

        // Mixed mesh → full offsets; also checks the vector-dim layout
        // (component i + vdim * j inside get_values).
        let mmesh = mixed_mesh();
        let mqs = QuadratureSpace::new(&mmesh, 2);
        let mut mqf = QuadratureFunction::with_vdim(&mqs, 2);
        assert_eq!(mqf.size(), mqs.get_size() * 2);
        for (i, v) in mqf.values_mut().iter_mut().enumerate() {
            *v = i as f64;
        }
        assert_eq!(mqf.get_values(0).len(), 8); // 4 QP x 2 comps
        assert_eq!(mqf.get_values(1).len(), 6); // 3 QP x 2 comps
        // element 1, QP 1, component 0 → global QP index 4 + 1 = 5 → 2*5 + 0.
        assert_eq!(mqf.get_values_at_qp(1, 1)[0] as usize, 10);
        // Dense view: (comp, qp) layout.
        let dense = mqf.get_values_dense(1);
        assert_eq!((dense.nrows(), dense.ncols()), (2, 3));
        assert_eq!(dense[(0, 0)] as usize, 2 * 4); // elem 1 QP 0 comp 0
        assert_eq!(dense[(1, 2)] as usize, 2 * 6 + 1); // elem 1 QP 2 comp 1
    }

    /// (a) `QuadratureFunction::integrate` against the analytic integral of a
    /// known coefficient on tri / quad meshes (project the coefficient at the
    /// quadrature points exactly, as MFEM `Coefficient::Project(qf)` does).
    #[test]
    fn integrate_known_coefficient() {
        let f = |x: &[f64]| x[0] * x[0] * x[1] * x[1] * x[1] + 1.0;
        let expected = 13.0 / 12.0;

        let tri = Mesh::<2>::unit_square_tri(5);
        let qs_tri = QuadratureSpace::new(&tri, 6);
        let mut qf = QuadratureFunction::new(&qs_tri);
        for e in 0..qs_tri.get_ne() {
            let ir = qs_tri.get_int_rule(e);
            let mut xs = vec![0.0_f64; ir.n_points() * 2];
            let mut det = vec![0.0_f64; ir.n_points()];
            qs_tri.map_quadrature_points(e, &mut xs, &mut det);
            let vals = qf.get_values_mut(e);
            for q in 0..ir.n_points() {
                vals[q] = f(&xs[q * 2..q * 2 + 2]);
            }
        }
        assert!((qf.integrate() - expected).abs() < 1e-12);

        // Quad mesh: the same coefficient projected through the space.
        let quad = Mesh::<2>::unit_square_quad(4);
        let qs_q = QuadratureSpace::new(&quad, 6);
        let mut qf_q = QuadratureFunction::new(&qs_q);
        for e in 0..qs_q.get_ne() {
            let ir = qs_q.get_int_rule(e);
            let mut xs = vec![0.0_f64; ir.n_points() * 2];
            let mut det = vec![0.0_f64; ir.n_points()];
            qs_q.map_quadrature_points(e, &mut xs, &mut det);
            let vals = qf_q.get_values_mut(e);
            for q in 0..ir.n_points() {
                vals[q] = f(&xs[q * 2..q * 2 + 2]);
            }
        }
        assert!((qf_q.integrate() - expected).abs() < 1e-12);
        assert_eq!(qf_q.get_vdim(), 1);
        assert_eq!(qf_q.space().get_order(), 6);
    }

    /// (a, vector) `integrate_vec` matches the analytic component integrals.
    #[test]
    fn integrate_vec_components() {
        let quad = Mesh::<2>::unit_square_quad(3);
        let qs = QuadratureSpace::new(&quad, 4);
        let mut qf = QuadratureFunction::with_vdim(&qs, 2);
        qf.fill(1.0);
        for e in 0..qs.get_ne() {
            let ir = qs.get_int_rule(e);
            let mut xs = vec![0.0_f64; ir.n_points() * 2];
            let mut det = vec![0.0_f64; ir.n_points()];
            qs.map_quadrature_points(e, &mut xs, &mut det);
            let vals = qf.get_values_mut(e);
            for q in 0..ir.n_points() {
                let x = &xs[q * 2..q * 2 + 2];
                vals[q * 2] = x[0]; // component 0: f = x
                vals[q * 2 + 1] = x[1] * x[1]; // component 1: f = y^2
            }
        }
        let ints = qf.integrate_vec();
        assert!((ints[0] - 0.5).abs() < 1e-12, "∫x = {}", ints[0]);
        assert!((ints[1] - 1.0 / 3.0).abs() < 1e-12, "∫y² = {}", ints[1]);
    }

    /// (b) Per-element QF integrals match the VectorAssembler linear form:
    /// on an L² space the element DOFs are local and form a partition of
    /// unity, so Σ_{i∈e} F_i = ∫_e f.
    #[test]
    fn per_element_integrals_match_linear_form() {
        use crate::assembler::Assembler;
        use crate::standard::DomainSourceIntegrator;

        let f = |x: &[f64]| {
            let a = (std::f64::consts::PI * x[0]).sin();
            let b = (std::f64::consts::PI * x[1]).sin();
            a * b + 0.5 * x[0]
        };
        let quad_order = 4u8;

        for (label, mesh) in [
            ("tri", Mesh::<2>::unit_square_tri(5)),
            ("quad", Mesh::<2>::unit_square_quad(4)),
        ] {
            let space = L2Space::new(mesh.clone(), 2);
            let rhs = Assembler::assemble_linear(
                &space,
                &[&DomainSourceIntegrator::new(f)],
                quad_order,
            );
            // QF integral of f per element (project f exactly at the QPs).
            let mesh_ref: &dyn MeshTopology = &mesh;
            let qs = QuadratureSpace::new(mesh_ref, quad_order as i32);
            let mut qf = QuadratureFunction::new(&qs);
            for e in 0..qs.get_ne() {
                let ir = qs.get_int_rule(e);
                let mut xs = vec![0.0_f64; ir.n_points() * 2];
                let mut det = vec![0.0_f64; ir.n_points()];
                qs.map_quadrature_points(e, &mut xs, &mut det);
                let vals = qf.get_values_mut(e);
                for q in 0..ir.n_points() {
                    vals[q] = f(&xs[q * 2..q * 2 + 2]);
                }
            }
            // Compare per element.
            for e in 0..space.mesh().n_elements() as u32 {
                let elem_integral: f64 =
                    space.element_dofs(e).iter().map(|&d| rhs[d as usize]).sum();
                let qf_integral: f64 = {
                    let vals = qf.get_values(e as usize);
                    let w = &qs.get_weights()
                        [qs.offset(e as usize)..qs.offset(e as usize + 1)];
                    vals.iter().zip(w.iter()).map(|(&v, &wi)| v * wi).sum()
                };
                assert!(
                    (elem_integral - qf_integral).abs() < 1e-10,
                    "{label}: elem {e}: assembled {elem_integral} vs qf {qf_integral}"
                );
            }
        }
    }

    /// `project_grid_function` reproduces a polynomial H¹ grid function
    /// exactly at the quadrature points.
    #[test]
    fn project_grid_function_polynomial() {
        use fem_space::h1::H1Space;

        let mesh = Mesh::<2>::unit_square_quad(2);
        let space = H1Space::new(mesh.clone(), 2);
        // Per-axis degree ≤ 2: the Q2 nodal interpolant is exact.
        let f = |x: &[f64]| x[0] * x[0] * x[1] * x[1] - 2.0 * x[0] + 1.0;
        let dofs = space.interpolate(&f);
        let gf = GridFunction::new(&space, dofs.into_vec());

        let mesh_dyn: &dyn MeshTopology = &mesh;
        let qs = QuadratureSpace::new(mesh_dyn, 4);
        let mut qf = QuadratureFunction::new(&qs);
        qf.project_grid_function(&gf);
        assert_eq!(qf.get_vdim(), 1);

        // The projected QF integrates f exactly (degree-4 rule, f quadratic).
        let expected = qs.integrate(&f);
        assert!((qf.integrate() - expected).abs() < 1e-12);

        // Spot-check values at physical quadrature points.
        for e in 0..qs.get_ne() {
            let ir = qf.get_int_rule(e);
            let mut xs = vec![0.0_f64; ir.n_points() * 2];
            let mut det = vec![0.0_f64; ir.n_points()];
            qs.map_quadrature_points(e, &mut xs, &mut det);
            let vals = qf.get_values(e);
            for q in 0..ir.n_points() {
                let expect = f(&xs[q * 2..q * 2 + 2]);
                assert!(
                    (vals[q] - expect).abs() < 1e-12,
                    "elem {e} qp {q}: {} vs {expect}",
                    vals[q]
                );
            }
        }
    }

    /// Save / load round trip preserves values and vector dimension.
    #[test]
    fn save_load_round_trip() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let qs = QuadratureSpace::new(&mesh, 2);
        let mut qf = QuadratureFunction::with_vdim(&qs, 3);
        for (i, v) in qf.values_mut().iter_mut().enumerate() {
            *v = (i as f64).cosh() * 0.5;
        }
        let mut buf = Vec::new();
        qf.save(&mut buf).expect("save");

        let text = String::from_utf8(buf).expect("utf8");
        let (space_txt, qf_txt) = text.split_once("VDim:").expect("VDim header");
        let mut reader = space_txt.as_bytes();
        let qs2 = QuadratureSpace::from_stream(&mesh, &mut reader).expect("load space");
        let qf_txt = format!("VDim:{qf_txt}");
        let mut reader = qf_txt.as_bytes();
        let qf2 = QuadratureFunction::load(&qs2, &mut reader).expect("load qf");

        assert_eq!(qf2.get_vdim(), 3);
        assert_eq!(qf2.size(), qf.size());
        for (a, b) in qf.values().iter().zip(qf2.values().iter()) {
            assert!((a - b).abs() < 1e-13, "{a} vs {b}");
        }
    }
}
