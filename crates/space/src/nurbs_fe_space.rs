//! `NurbsFESpace` — an MFEM `FiniteElementSpace` over a `NURBSExtension`.
//!
//! 1:1 with MFEM's NURBS assembly path: `FiniteElementSpace::UpdateNURBS` +
//! `NURBSExtension::LoadFE` (`fem/fespace.cpp`, `mesh/nurbs.cpp`) and the
//! `NURBSFiniteElement` value evaluation of `fem/fe/fe_nurbs.cpp`.  Three
//! choices of MFEM's design drive this module:
//!
//! 1. **An element's DOF table is not a topology table.**  Every knot span owns
//!    `prod(orders + 1)` DOFs, listed by
//!    `NURBSExtension::GenerateElementDofTable` in the tensor order
//!    `o = i + (px+1)*(j + (py+1)*k)` — the order the FE's `CalcShape` uses, so
//!    `element_dofs(e)[o]` is the global DOF of the `o`-th local basis function.
//! 2. **An element's reference element is its knot span.**  `LoadFE` hands the
//!    FE the span index `ijk` (`el_to_IJK`) and `KnotVector::CalcShape(shape,
//!    ijk[d], xi)` is evaluated at the span-local `xi ∈ [0,1]^dim`; MFEM's
//!    integrators pick `IntRules.Get(SQUARE/CUBE, order)` on `[0,1]^dim`.
//! 3. **The geometry is the mesh's rational NURBS map, the space is not.**  The
//!    space extension built by `NURBSExtension(parent, order)` resets
//!    `weights = 1`, so the analysis basis is the polynomial B-spline basis of
//!    the space's knot vectors while the element transformation uses the mesh's
//!    weights (the mesh `Nodes` grid function).
//!
//! # Geometry after refinement
//!
//! `Mesh::UniformRefinement` on a NURBS mesh inserts knots into the patch knot
//! vectors *and* re-derives the control net
//! (`NURBSPatch::UniformRefinement`/`SetCoordsFromPatches`), leaving the
//! geometry invariant.  This module therefore keeps the mesh's original knot
//! vectors and control points and evaluates a refined element over its own
//! parameter interval `[U_ref[k], U_ref[k+1]]` — a sub-interval of an original
//! span — reusing the original basis there.  The map, its Jacobian and
//! `Weight() = det J` are the same as MFEM's refined-net evaluation because
//! knot insertion does not change the curve.
//!
//! # Why the assembly lives here
//!
//! `fem_assembly::Assembler` selects the solution reference element from the
//! space's `SpaceType` and the mesh's `ElementType` (`ref_elem_vol_for_space`)
//! and the geometry element from `mesh.element_type`/`geom_order`
//! (`geo_ref_elem`); `FESpace` has no hook to contribute a NURBS span element
//! (span index, per-element weights, `[0,1]^dim` reference domain).  Until that
//! hook exists the element kernels and the assembly loop for the true NURBS
//! space live in this module, and the miniapps call [`NurbsFESpace::assemble_diffusion`]
//! and friends instead of `Assembler::assemble_bilinear`.
//!
//! # Limitations
//!
//! * Dimensions 1, 2 and 3.  The 1-D case (`segment-nurbs.mesh`) uses MFEM's
//!   `NURBS1DFiniteElement` and `IntRules.Get(Geometry::SEGMENT, order)`; the
//!   element transformation is 1 x 1, so `Weight() = J(0,0)` and
//!   `AdjugateJacobian = [1]` (MFEM's `DenseMatrix::Weight` /
//!   `CalcAdjugate` for a `1 x 1` matrix).
//! * [`NurbsFESpace::boundary_dofs`] implements MFEM's `ess_bdr = 1` case
//!   (every mesh boundary attribute essential); [`NurbsFESpace::boundary_dofs_marked`]
//!   is the per-attribute form (`GetEssentialTrueDofs(ess_bdr)`), built from the
//!   boundary elements' attributes rather than from
//!   `NURBSExtension::GenerateBdrElementDofTable` (whose `bel_dof` rows the
//!   `ess_bdr = 1` union does not need).
//! * Scalar (H¹) space: the vector `NURBS_HDiv`/`NURBS_HCurl` paths are outside
//!   the delivered scope (they are dimensions 2 and 3 only, as in MFEM).

use fem_element::iga::KnotVector;
use fem_element::nurbs_fe_collection::{
    degree_elevate, knot_botella, knot_in_span, knot_order, knot_span_dparam, knot_span_shape,
    Nurbs1DFiniteElement, NurbsScalar2D, NurbsScalar3D,
};
use fem_element::nurbs_vector::{NurbsHCurl2D, NurbsHCurl3D, NurbsHDiv2D, NurbsHDiv3D};
use fem_element::quadrature::gauss_legendre_01;
use fem_element::reference::VectorReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};

use crate::nurbs_extension::{BdrDofMode, NurbsExtension, unsign_dof};

/// MFEM `Table(const Table &t1, const Table &t2, int offset)` /
/// `Table(t1, t2, o2, t3, o3)` — the merge `FiniteElementSpace::UpdateNURBS`
/// uses for the element *and* boundary DOF tables of a vector NURBS space:
/// component `c`'s entries are shifted by its DOF offset, and a negative entry
/// (MFEM's `-1 - dof` sign encoding) is shifted by `-offset`, which keeps the
/// encoding (`-1 - (dof + offset)`) intact.
fn merge_component_tables(parts: &[(usize, Vec<Vec<i64>>)]) -> Vec<Vec<i64>> {
    let n_rows = parts[0].1.len();
    let mut out = Vec::with_capacity(n_rows);
    for b in 0..n_rows {
        let mut row = Vec::new();
        for (off, table) in parts {
            for &e in &table[b] {
                row.push(if e < 0 { e - *off as i64 } else { e + *off as i64 });
            }
        }
        out.push(row);
    }
    out
}

/// A quadrature rule on `[0,1]^dim` with MFEM's tensor ordering (the first
/// direction varies fastest, so the summation order matches MFEM's).
#[derive(Debug, Clone)]
pub struct Rule {
    pub points: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
}

/// `IntRules.Get(Geometry::SEGMENT/SQUARE/CUBE, order)` — Gauss-Legendre with
/// `(order + 2)/2` points per direction on `[0,1]^dim`.  MFEM's three
/// `GetSegmentRealOrder`-style helpers all use `n = Order/2 + 1` points per
/// direction, which agrees with `(order + 2)/2` for every `order >= 0`.
pub fn nurbs_rule(dim: usize, order: u8) -> Rule {
    let n = ((order as usize + 2) / 2).max(1);
    if dim == 1 {
        // `IntRules.Get(Geometry::SEGMENT, order)`: the 1-D Gauss rule on the
        // reference segment `[0,1]` (`QuadratureFunctions1D::GaussLegendre`).
        let (xs, ws) = gauss_legendre_01(n);
        return Rule { points: xs.iter().map(|&x| vec![x]).collect(), weights: ws };
    }
    if dim == 2 {
        // MFEM's `IntRules.Get(Geometry::SQUARE, order)`: an `n x n` tensor
        // product of the 1-D Gauss rule, first index varying fastest.  Built
        // from `gauss_legendre_01` (bit-identical to `Poly_1D::GaussLegendre`
        // for `n <= 5`) rather than `quad_rule_01`, which routes `n = 5`
        // through the generic Newton solver.
        let (xs, ws) = gauss_legendre_01(n);
        let mut points = Vec::with_capacity(n * n);
        let mut weights = Vec::with_capacity(n * n);
        for (yk, wk) in xs.iter().zip(ws.iter()) {
            for (xk, wj) in xs.iter().zip(ws.iter()) {
                points.push(vec![*xk, *yk]);
                weights.push(wj * wk);
            }
        }
        return Rule { points, weights };
    }
    let (xs, ws) = gauss_legendre_01(n);
    let mut points = Vec::with_capacity(n * n * n);
    let mut weights = Vec::with_capacity(n * n * n);
    for (zk, wk) in xs.iter().zip(ws.iter()) {
        for (yj, wj) in xs.iter().zip(ws.iter()) {
            for (xi, wi) in xs.iter().zip(ws.iter()) {
                points.push(vec![*xi, *yj, *zk]);
                weights.push(wi * wj * wk);
            }
        }
    }
    Rule { points, weights }
}

/// The 1-D nodal Lagrange basis through `nodes` evaluated at `x`
/// (`l_i(x) = Π_{j≠i} (x−x_j)/(x_i−x_j)`), the values MFEM's
/// `Poly_1D::Basis::Eval` produces for the Gauss-Legendre points.
fn lagrange_1d(nodes: &[f64], x: f64) -> Vec<f64> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let mut v = 1.0;
            for (j, &xj) in nodes.iter().enumerate() {
                if j != i {
                    v *= (x - xj) / (xi - xj);
                }
            }
            v
        })
        .collect()
}

/// `CalcShape` of MFEM's `L2_QuadrilateralElement` / `L2_HexahedronElement`
/// with `BasisType::GaussLegendre`: the tensor product of the 1-D nodal
/// Lagrange basis [`lagrange_1d`] of the Gauss-Legendre points, in `L2_DOF_MAP`
/// order (`o = ix + (p+1)*(iy + (p+1)*iz)`, x fastest).
fn l2_gl_shape(dim: usize, nodes: &[f64], xi: &[f64], out: &mut [f64]) {
    let np = nodes.len();
    let lx = lagrange_1d(nodes, xi[0]);
    let ly = lagrange_1d(nodes, xi[1]);
    if dim == 2 {
        for ix in 0..np {
            for iy in 0..np {
                out[iy * np + ix] = lx[ix] * ly[iy];
            }
        }
    } else {
        let lz = lagrange_1d(nodes, xi[2]);
        for ix in 0..np {
            for iy in 0..np {
                for iz in 0..np {
                    out[iz * np * np + iy * np + ix] = lx[ix] * ly[iy] * lz[iz];
                }
            }
        }
    }
}

/// MFEM `LinearSolve(DenseMatrix&, real_t*, TOL)` — `LUFactors::Factor` with
/// partial pivoting followed by `LUFactors::Solve`, on a row-major `n x n`
/// matrix.  Returns `false` — leaving `rhs` untouched — when a pivot is
/// `<= tol`, exactly like MFEM's factorisation.
fn dense_lu_solve(mat: &mut [f64], n: usize, rhs: &mut [f64], tol: f64) -> bool {
    let mut piv = vec![0usize; n];
    for i in 0..n {
        let mut p = i;
        let mut amax = mat[i * n + i].abs();
        for j in i + 1..n {
            let b = mat[j * n + i].abs();
            if b > amax {
                amax = b;
                p = j;
            }
        }
        piv[i] = p;
        if p != i {
            for j in 0..n {
                mat.swap(i * n + j, p * n + j);
            }
        }
        if mat[i * n + i].abs() <= tol {
            return false;
        }
        let inv = 1.0 / mat[i * n + i];
        for j in i + 1..n {
            mat[j * n + i] *= inv;
        }
        for k in i + 1..n {
            let aik = mat[i * n + k];
            for j in i + 1..n {
                mat[j * n + k] -= aik * mat[j * n + i];
            }
        }
    }
    for i in 0..n {
        if piv[i] != i {
            rhs.swap(i, piv[i]);
        }
    }
    for i in 0..n {
        let mut s = rhs[i];
        for j in 0..i {
            s -= mat[i * n + j] * rhs[j];
        }
        rhs[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= mat[i * n + j] * rhs[j];
        }
        rhs[i] = s / mat[i * n + i];
    }
    true
}

/// The scalar NURBS element of one knot span: MFEM's `NURBS2DFiniteElement` /
/// `NURBS3DFiniteElement` bound to a patch's knot vectors, one span index
/// (`NURBSFiniteElement::ijk`) and one weight per local DOF (`LoadFE`).
#[derive(Debug, Clone)]
pub enum SpanElement {
    /// MFEM `NURBS1DFiniteElement`.
    One(Nurbs1DFiniteElement),
    /// MFEM `NURBS2DFiniteElement`.
    Two(NurbsScalar2D),
    /// MFEM `NURBS3DFiniteElement`.
    Three(NurbsScalar3D),
}

impl SpanElement {
    /// `FiniteElement::GetDim`.
    pub fn dim(&self) -> usize {
        match self {
            SpanElement::One(_) => 1,
            SpanElement::Two(_) => 2,
            SpanElement::Three(_) => 3,
        }
    }

    /// `FiniteElement::GetDof`.
    pub fn n_dofs(&self) -> usize {
        match self {
            SpanElement::One(fe) => fe.n_dofs(),
            SpanElement::Two(fe) => fe.n_dofs(),
            SpanElement::Three(fe) => fe.n_dofs(),
        }
    }

    /// `FiniteElement::GetOrder` (`max(orders)` after `SetOrder`).
    pub fn order(&self) -> usize {
        match self {
            SpanElement::One(fe) => fe.order(),
            SpanElement::Two(fe) => fe.order(),
            SpanElement::Three(fe) => fe.order(),
        }
    }

    /// `NURBSFiniteElement::SetIJK`.
    pub fn set_ijk(&mut self, ijk: &[usize; 3]) {
        match self {
            SpanElement::One(fe) => fe.set_ijk(ijk[0]),
            SpanElement::Two(fe) => fe.set_ijk([ijk[0], ijk[1]]),
            SpanElement::Three(fe) => fe.set_ijk(*ijk),
        }
    }

    /// `NURBS1D/2D/3DFiniteElement::CalcShape`.
    pub fn shape(&self, xi: &[f64], values: &mut [f64]) {
        match self {
            SpanElement::One(fe) => fe.calc_shape(xi[0], values),
            SpanElement::Two(fe) => fe.calc_shape(xi, values),
            SpanElement::Three(fe) => fe.calc_shape(xi, values),
        }
    }

    /// `NURBS1D/2D/3DFiniteElement::CalcDShape` (`dim` entries per DOF).
    pub fn grad(&self, xi: &[f64], grads: &mut [f64]) {
        match self {
            // `NURBS1DFiniteElement::CalcDShape` fills a `DenseMatrix` with one
            // column, i.e. exactly this DOF-major `dim = 1` layout.
            SpanElement::One(fe) => fe.calc_dshape(xi[0], grads),
            SpanElement::Two(fe) => fe.calc_grad(xi, grads),
            SpanElement::Three(fe) => fe.calc_grad(xi, grads),
        }
    }
}

/// `ElementTransformation` of one element at a span-local point: the physical
/// coordinate, the Jacobian `d x_i / d xi_j` (row-major) and MFEM's
/// `Weight() = det J`.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub x: [f64; 3],
    pub jac: [[f64; 3]; 3],
    pub det_j: f64,
}

/// `CalcAdjugate` (MFEM `linalg/densemat.cpp`) of a `dim x dim` Jacobian.
fn adjugate(j: &[[f64; 3]; 3], dim: usize) -> [[f64; 3]; 3] {
    let mut a = [[0.0_f64; 3]; 3];
    if dim == 1 {
        // `CalcAdjugate` of a `1 x 1` matrix: `adja(0,0) = 1.0`.
        a[0][0] = 1.0;
    } else if dim == 2 {
        a[0][0] = j[1][1];
        a[0][1] = -j[0][1];
        a[1][0] = -j[1][0];
        a[1][1] = j[0][0];
    } else {
        a[0][0] = j[1][1] * j[2][2] - j[1][2] * j[2][1];
        a[0][1] = j[0][2] * j[2][1] - j[0][1] * j[2][2];
        a[0][2] = j[0][1] * j[1][2] - j[0][2] * j[1][1];
        a[1][0] = j[1][2] * j[2][0] - j[1][0] * j[2][2];
        a[1][1] = j[0][0] * j[2][2] - j[0][2] * j[2][0];
        a[1][2] = j[0][2] * j[1][0] - j[0][0] * j[1][2];
        a[2][0] = j[1][0] * j[2][1] - j[1][1] * j[2][0];
        a[2][1] = j[0][1] * j[2][0] - j[0][0] * j[2][1];
        a[2][2] = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    }
    a
}

/// `DenseMatrix::Det` (MFEM `linalg/densemat.cpp`); for `dim = 1` this is also
/// `DenseMatrix::Weight()` (a `1 x 1` matrix is square, so `Weight` returns
/// `Det()` — MFEM's `fabs` is commented out there).
fn det(j: &[[f64; 3]; 3], dim: usize) -> f64 {
    if dim == 1 {
        j[0][0]
    } else if dim == 2 {
        j[0][0] * j[1][1] - j[0][1] * j[1][0]
    } else {
        j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            + j[1][0] * (j[0][2] * j[2][1] - j[0][1] * j[2][2])
            + j[2][0] * (j[0][1] * j[1][2] - j[0][2] * j[1][1])
    }
}

/// The local multi-index (first direction fastest) of the `o`-th tensor-product
/// entry of a span element: `o = i + n0*(j + n1*k)`.
fn multi_index(o: usize, lens: &[usize], d: usize) -> usize {
    let mut idx = o;
    for &n in lens.iter().take(d) {
        idx /= n;
    }
    idx % lens[d]
}

/// Split MFEM's *signed* `NURBSFiniteElement::ijk` entry (a knot-span index,
/// `FlipIndexSign(i) = -1 - i` when the boundary element runs against its
/// patch's direction) into the knot span to evaluate and the reference
/// coordinate to evaluate it at: `KnotVector::CalcShape(shape, i, xi)` uses the
/// span `ip = (i >= 0) ? i : -1 - i` (plus the order) with the coordinate
/// `(i >= 0) ? xi : 1 - xi`.
pub fn split_signed_span(signed: i64, xi: f64) -> (usize, f64) {
    if signed >= 0 {
        (signed as usize, xi)
    } else {
        ((-1 - signed) as usize, 1.0 - xi)
    }
}

/// A NURBS finite element space (`FiniteElementSpace` + `NURBSExtension`).
#[derive(Debug, Clone)]
pub struct NurbsFESpace {
    /// Geometry: the mesh file's knot vectors and control net, *before*
    /// refinement.
    geo: NurbsExtension,
    /// Control-point coordinates in `geo` DOF order.
    geo_coords: Vec<Vec<f64>>,
    /// Refinement levels applied to the mesh's knots.
    ref_levels: usize,
    /// The refined mesh-order extension: its knot vectors give each element's
    /// parameter interval.
    mesh_ext: NurbsExtension,
    /// The analysis extension (`NURBSExtension(mesh->NURBSext, orders)`).
    ext: NurbsExtension,
    /// Per-knot-vector orders of the analysis space.
    orders: Vec<usize>,
    /// Element → the unrefined extension's element id (geometry lookup).
    geo_element: Vec<usize>,
    dim: usize,
}

impl NurbsFESpace {
    /// Build the space MFEM's `nurbs_ex1`/`nurbs_ex3` construct: read the NURBS
    /// mesh, apply `ref_levels` uniform refinements
    /// (`Mesh::UniformRefinement`), then
    /// `NURBSExtension(mesh->NURBSext, orders)` + `NURBSFECollection`.
    ///
    /// `orders` may hold a single order (broadcast to every knot vector, as
    /// `nurbs_ex1` does with `order.SetSize(nkv); order = tmp`) or one per knot
    /// vector.
    pub fn from_mesh_str(text: &str, ref_levels: usize, orders: &[usize]) -> Result<Self, String> {
        let geo = NurbsExtension::from_mesh_str(text)?;
        let dim = geo.dim();
        if dim < 1 || dim > 3 {
            return Err(format!(
                "NurbsFESpace: dimension {dim} is not supported (only 1, 2 and 3)"
            ));
        }
        let nodes = NurbsExtension::parse_nodes(text, geo.n_dofs())?;
        if nodes.vdim < dim {
            return Err(format!(
                "NurbsFESpace: {} control point components for dimension {dim}",
                nodes.vdim
            ));
        }

        let mut mesh_ext = geo.clone();
        for _ in 0..ref_levels {
            mesh_ext.uniform_refinement(2)?;
        }

        let orders: Vec<usize> = if orders.len() == 1 {
            vec![orders[0]; mesh_ext.n_knot_vectors()]
        } else if orders.len() == mesh_ext.n_knot_vectors() {
            orders.to_vec()
        } else {
            return Err(format!(
                "NurbsFESpace: {} orders for {} knot vectors",
                orders.len(),
                mesh_ext.n_knot_vectors()
            ));
        };
        let ext = mesh_ext.with_orders(&orders)?;

        let mut geo_element = Vec::with_capacity(ext.n_elements());
        for e in 0..ext.n_elements() {
            geo_element.push(ijk_to_element(
                &geo,
                &mesh_ext,
                ext.element_patch(e),
                ext.element_ijk(e),
                ref_levels,
            )?);
        }

        Ok(Self {
            geo,
            geo_coords: nodes.coords,
            ref_levels,
            mesh_ext,
            ext,
            orders,
            geo_element,
            dim,
        })
    }

    /// Read a NURBS mesh file and build the space (see [`Self::from_mesh_str`]).
    pub fn from_mesh_file(
        path: impl AsRef<std::path::Path>,
        ref_levels: usize,
        orders: &[usize],
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("NurbsFESpace::from_mesh_file: {e}"))?;
        Self::from_mesh_str(&text, ref_levels, orders)
    }

    /// The analysis extension (`FiniteElementSpace::GetNURBSext`).
    pub fn extension(&self) -> &NurbsExtension {
        &self.ext
    }

    /// MFEM `FiniteElementSpace::GetBdrElementDofs` for this scalar space —
    /// `NURBSext->GetBdrElementDofTable()` (`Mode::H_1`), one row per mesh
    /// boundary element, in MFEM's signed encoding (see
    /// [`crate::nurbs_extension::unsign_dof`]).
    pub fn boundary_dof_table(&self) -> Vec<Vec<i64>> {
        self.ext.boundary_dof_table(BdrDofMode::H1)
    }

    /// The refined mesh extension (the `mesh->NURBSext` after
    /// `Mesh::UniformRefinement`; its knot vectors carry the *geometry* orders).
    pub fn mesh_extension(&self) -> &NurbsExtension {
        &self.mesh_ext
    }

    /// `FiniteElementSpace::GetNDofs`.
    pub fn n_dofs(&self) -> usize {
        self.ext.n_dofs()
    }

    /// `FiniteElementSpace::GetNE`.
    pub fn n_elements(&self) -> usize {
        self.ext.n_elements()
    }

    /// Mesh dimension (2 or 3).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Per-knot-vector orders of the analysis space.
    pub fn orders(&self) -> &[usize] {
        &self.orders
    }

    /// The geometry (mesh) order of the single-patch refinement — the scalar
    /// knot-vector order of the patch's first direction
    /// (`ElementTransformation::OrderW() = mesh_order * dim - 1`).
    pub fn mesh_order(&self) -> usize {
        let kvs = self.mesh_ext.patch_knot_vectors(0).expect("geometry patch");
        knot_order(kvs[0].knot_vector()).expect("validated knot vector")
    }

    /// `ComputeLpNorm(2., Coefficient, mesh, irs)`: `√(Σ_e Σ_q w_q W_q f(x_q)²)`
    /// — the exact scalar function's `L²` norm over the NURBS mesh, using the
    /// same refined-span geometry as the error norms.
    pub fn compute_exact_l2_norm(&self, f: &dyn Fn(&[f64]) -> f64, order_quad: u8) -> f64 {
        let dim = self.dim;
        let rule = nurbs_rule(dim, order_quad);
        let mut norm2 = 0.0_f64;
        for e in 0..self.n_elements() {
            for q in 0..rule.points.len() {
                let geo = self.geometry(e, &rule.points[q]);
                let v = f(&geo.x[..dim]);
                norm2 += rule.weights[q] * geo.det_j * v * v;
            }
        }
        norm2.sqrt()
    }

    /// `ComputeLpNorm(2., VectorCoefficient, mesh, irs)` — the vector version
    /// of [`Self::compute_exact_l2_norm`].
    pub fn compute_exact_l2_norm_vec(
        &self,
        f: &dyn Fn(&[f64]) -> Vec<f64>,
        order_quad: u8,
    ) -> f64 {
        let dim = self.dim;
        let rule = nurbs_rule(dim, order_quad);
        let mut norm2 = 0.0_f64;
        for e in 0..self.n_elements() {
            for q in 0..rule.points.len() {
                let geo = self.geometry(e, &rule.points[q]);
                let v = f(&geo.x[..dim]);
                norm2 += rule.weights[q] * geo.det_j * v.iter().map(|c| c * c).sum::<f64>();
            }
        }
        norm2.sqrt()
    }

    /// `NURBSExtension::GetElementDofTable`.
    pub fn element_dof_table(&self) -> &[Vec<usize>] {
        self.ext.element_dof_table()
    }

    /// Global DOFs of element `e`, in the local FE order.
    pub fn element_dofs(&self, e: usize) -> &[usize] {
        self.ext.element_dofs(e)
    }

    /// `NURBSExtension::LoadFE`'s weights for element `e` (`GetSubVector`).
    pub fn element_weights(&self, e: usize) -> Vec<f64> {
        let w = self.ext.weights();
        self.ext.element_dofs(e).iter().map(|&g| w[g]).collect()
    }

    /// The scalar FE of element `e` with `SetIJK` and `SetWeights` applied
    /// (`NURBSExtension::LoadFE`).
    pub fn element_fe(&self, e: usize) -> SpanElement {
        let patch = self.ext.element_patch(e);
        let kvs = self.ext.patch_knot_vectors(patch).expect("element patch");
        let ijk = self.ext.element_ijk(e);
        let weights = self.element_weights(e);
        match self.dim {
            1 => {
                let mut fe = Nurbs1DFiniteElement::new(kvs[0].knot_vector().clone())
                    .expect("Nurbs1DFiniteElement::new");
                fe.set_ijk(ijk[0]);
                fe.set_weights(weights).expect("Nurbs1DFiniteElement::set_weights");
                SpanElement::One(fe)
            }
            2 => {
                let mut fe = NurbsScalar2D::new(
                    kvs[0].knot_vector().clone(),
                    kvs[1].knot_vector().clone(),
                )
                .expect("NurbsScalar2D::new");
                fe.set_ijk([ijk[0], ijk[1]]);
                fe.set_weights(weights).expect("NurbsScalar2D::set_weights");
                SpanElement::Two(fe)
            }
            _ => {
                let mut fe = NurbsScalar3D::new(
                    kvs[0].knot_vector().clone(),
                    kvs[1].knot_vector().clone(),
                    kvs[2].knot_vector().clone(),
                )
                .expect("NurbsScalar3D::new");
                fe.set_ijk(ijk);
                fe.set_weights(weights).expect("NurbsScalar3D::set_weights");
                SpanElement::Three(fe)
            }
        }
    }

    /// `FiniteElement::GetDof` of element `e`.
    pub fn element_n_dofs(&self, e: usize) -> usize {
        self.ext.element_dofs(e).len()
    }

    /// `NURBS2D/3DFiniteElement::CalcShape` of element `e` at the span-local
    /// point `xi` (one value per local DOF), after `LoadFE`.
    pub fn fe_shape(&self, e: usize, xi: &[f64], values: &mut [f64]) {
        self.element_fe(e).shape(xi, values);
    }

    /// `NURBS2D/3DFiniteElement::CalcDShape` of element `e` at the span-local
    /// point `xi` (`dim` values per local DOF, DOF-major).
    pub fn fe_grad(&self, e: usize, xi: &[f64], grads: &mut [f64]) {
        self.element_fe(e).grad(xi, grads);
    }

    /// The mesh's **rational** NURBS element transformation of element `e` at
    /// the span-local point `xi` (see the module docs for why the original
    /// control net is evaluated over the refined parameter interval).
    pub fn geometry(&self, e: usize, xi: &[f64]) -> Geometry {
        let dim = self.dim;
        let patch = self.ext.element_patch(e);
        let ijk = self.ext.element_ijk(e);
        let old_e = self.geo_element[e];
        let old_ijk = self.geo.element_ijk(old_e);
        let geo_kv = self.geo.patch_knot_vectors(patch).expect("geometry patch");
        let ref_kv = self.mesh_ext.patch_knot_vectors(patch).expect("refined patch");

        // Per direction: the span-local values and the parametric derivative
        // `dN/du` of the *geometry* basis at the element's parameter, scaled to
        // `dN/dxi` over `[0,1]` of the element.
        let mut n1d: Vec<Vec<f64>> = Vec::with_capacity(dim);
        let mut dn1d: Vec<Vec<f64>> = Vec::with_capacity(dim);
        for d in 0..dim {
            let order = geo_kv[d].order();
            let knots = geo_kv[d].knot_vector().as_slice();
            let refined = ref_kv[d].knot_vector().as_slice();
            let a = refined[ijk[d] + order];
            let b = refined[ijk[d] + order + 1];
            let u = a + xi[d] * (b - a);
            // `KnotVector::CalcShape` takes the *reference* coordinate of the
            // unrefined span, so map the parameter `u` back onto it (the two
            // coincide only when the patch has a single span in this direction).
            let ga = knots[old_ijk[d] + order];
            let gb = knots[old_ijk[d] + order + 1];
            let mut n = vec![0.0; order + 1];
            let mut dn = vec![0.0; order + 1];
            knot_span_shape(knots, order, old_ijk[d], (u - ga) / (gb - ga), &mut n);
            knot_span_dparam(knots, order, old_ijk[d], u, &mut dn);
            for v in dn.iter_mut() {
                *v *= b - a;
            }
            n1d.push(n);
            dn1d.push(dn);
        }

        // The element's control points and weights, in the local tensor order.
        let lens: Vec<usize> = n1d.iter().map(|n| n.len()).collect();
        let n_local: usize = lens.iter().product();
        let mut coords = Vec::with_capacity(n_local);
        let mut wts = Vec::with_capacity(n_local);
        for o in 0..n_local {
            let mut midx = [0usize; 3];
            for d in 0..dim {
                midx[d] = old_ijk[d] + multi_index(o, &lens, d);
            }
            let g = self
                .geo
                .patch_dof(patch, &midx[..dim])
                .expect("geometry patch dof");
            coords.push(&self.geo_coords[g]);
            wts.push(self.geo.weights()[g]);
        }

        // Rational tensor product: W, dW/dxi_j, X_i and dX_i/dxi_j.
        let mut big_w = 0.0;
        let mut dw = [0.0_f64; 3];
        let mut big_x = [0.0_f64; 3];
        let mut dbuf = vec![0.0_f64; n_local * dim];
        for o in 0..n_local {
            let mut b = wts[o];
            for d in 0..dim {
                b *= n1d[d][multi_index(o, &lens, d)];
            }
            big_w += b;
            for d in 0..dim {
                let mut db = wts[o] * dn1d[d][multi_index(o, &lens, d)];
                for d2 in 0..dim {
                    if d2 != d {
                        db *= n1d[d2][multi_index(o, &lens, d2)];
                    }
                }
                dbuf[o * dim + d] = db;
                dw[d] += db;
            }
            for i in 0..dim {
                big_x[i] += b * coords[o][i];
            }
        }

        let mut x = [0.0_f64; 3];
        for i in 0..dim {
            x[i] = big_x[i] / big_w;
        }
        let mut jac = [[0.0_f64; 3]; 3];
        for i in 0..dim {
            for j in 0..dim {
                let mut dxi = 0.0;
                for o in 0..n_local {
                    dxi += dbuf[o * dim + j] * coords[o][i];
                }
                jac[i][j] = (dxi - x[i] * dw[j]) / big_w;
            }
        }

        Geometry { x, jac, det_j: det(&jac, dim) }
    }

    /// The DOF list of element `e` as `u32` ids for
    /// [`crate::constraints::form_linear_system`].
    pub fn element_dof_ids(&self, e: usize) -> Vec<u32> {
        self.ext.element_dofs(e).iter().map(|&d| d as u32).collect()
    }

    /// `mesh->GetBdrElementTransformation(i)` of a NURBS mesh at one boundary
    /// quadrature point: the physical point of the **rational** geometry on the
    /// patch boundary side `(patch, dir, low)`.
    ///
    /// MFEM builds that transformation from the mesh's `Nodes` grid function —
    /// `Nodes->FESpace()->GetBE(i)` (the mesh-order `NURBS1D/2D` boundary
    /// element of the *refined* mesh extension) with the point matrix
    /// `nodes(UnsignIndex(vdofs[n*k+j]))` (the refined net's boundary control
    /// points) — so it evaluates the boundary curve of the refined patch.  Knot
    /// insertion in the tangential direction leaves that curve unchanged, hence
    /// this port evaluates the **original** patch's boundary side over the
    /// refined span's parameter interval, exactly as [`Self::geometry`] does for
    /// a volume element, and fixes the normal direction at the patch boundary
    /// parameter (the first / last active original span, at its low / high end).
    ///
    /// `tang` holds one `(patch direction, signed span index, reference
    /// coordinate)` per *boundary* reference direction, in the boundary
    /// element's own order (see
    /// [`NurbsExtension::bdr_element_span`](crate::nurbs_extension::NurbsExtension::bdr_element_span)
    /// for the signed span, which mirrors the reference coordinate exactly as
    /// MFEM's `KnotVector::CalcShape(shape, i, xi)` does).
    pub fn bdr_geometry(
        &self,
        patch: usize,
        dir: usize,
        low: bool,
        tang: &[(usize, i64, f64)],
    ) -> [f64; 3] {
        let dim = self.dim;
        let geo_kv = self.geo.patch_knot_vectors(patch).expect("geometry patch");
        let ref_kv = self.mesh_ext.patch_knot_vectors(patch).expect("refined patch");
        let ref_spans = self.mesh_ext.patch_element_spans(patch).expect("refined spans");
        let geo_spans = self.geo.patch_element_spans(patch).expect("geometry spans");

        // Per direction: the original knot span and the span-local coordinate.
        let mut span = [0usize; 3];
        let mut xi = [0.0_f64; 3];
        for &(d, signed, x) in tang {
            let order = geo_kv[d].order();
            let (s, xs) = split_signed_span(signed, x);
            let refined = ref_kv[d].knot_vector().as_slice();
            let a = refined[s + order];
            let b = refined[s + order + 1];
            let u = a + xs * (b - a);
            // The refined span's ordinal reduces to the original span's ordinal
            // (uniform refinement splits every span in `2^ref_levels`), the same
            // correspondence `ijk_to_element` uses.
            let ordinal = ref_spans[d]
                .iter()
                .position(|&v| v == s)
                .expect("refined span is an element of the refined knot vector");
            let old = geo_spans[d][ordinal >> self.ref_levels];
            let knots = geo_kv[d].knot_vector().as_slice();
            let ga = knots[old + order];
            let gb = knots[old + order + 1];
            span[d] = old;
            xi[d] = (u - ga) / (gb - ga);
        }
        span[dir] = if low {
            geo_spans[dir][0]
        } else {
            *geo_spans[dir].last().expect("a patch direction has knot spans")
        };
        xi[dir] = if low { 0.0 } else { 1.0 };

        // The original (geometry) basis in every direction.
        let mut n1d: Vec<Vec<f64>> = Vec::with_capacity(dim);
        for d in 0..dim {
            let order = geo_kv[d].order();
            let mut n = vec![0.0; order + 1];
            knot_span_shape(geo_kv[d].knot_vector().as_slice(), order, span[d], xi[d], &mut n);
            n1d.push(n);
        }

        let lens: Vec<usize> = n1d.iter().map(|n| n.len()).collect();
        let n_local: usize = lens.iter().product();
        let mut num = [0.0_f64; 3];
        let mut den = 0.0_f64;
        for o in 0..n_local {
            let mut b = 1.0;
            let mut midx = [0usize; 3];
            for d in 0..dim {
                let i = multi_index(o, &lens, d);
                b *= n1d[d][i];
                midx[d] = span[d] + i;
            }
            let g = self.geo.patch_dof(patch, &midx[..dim]).expect("geometry patch dof");
            let w = self.geo.weights()[g] * b;
            den += w;
            for d in 0..dim {
                num[d] += w * self.geo_coords[g][d];
            }
        }
        for d in 0..dim {
            num[d] /= den;
        }
        num
    }

    /// MFEM `FiniteElementSpace::GetEssentialTrueDofs(ess_bdr = 1)`.
    ///
    /// A clamped NURBS patch's boundary is exactly the set of control points
    /// with an extreme parameter index (`0` or `NCP - 1`) in a direction whose
    /// boundary side is a mesh boundary, so the essential DOFs are the union
    /// over the *mesh boundary elements* of the control points of the
    /// patch-boundary entity each of them lies on
    /// ([`NurbsExtension::boundary_sides`]).  In 1-D a patch-boundary entity is
    /// a single point, i.e. the endpoint control point (`NCP - 1 = 0` for the
    /// "other" directions, so the loop degenerates to one DOF).
    ///
    /// `ess_bdr = 1` marks every mesh boundary attribute essential (the case
    /// `nurbs_ex1` uses); for a partial mask use
    /// [`Self::boundary_dofs_marked`], which is the same union restricted to
    /// the marked attributes.
    pub fn boundary_dofs(&self) -> Vec<u32> {
        let all = vec![true; self.ext.max_bdr_attribute().max(0) as usize];
        self.boundary_dofs_marked(&all)
    }

    /// MFEM `FiniteElementSpace::GetEssentialTrueDofs(ess_bdr)` — the essential
    /// DOFs of the *subset* of mesh boundary attributes marked in `ess_bdr`
    /// (index `a - 1` is the mesh attribute `a`, as MFEM's
    /// `bdr_attr_is_ess[GetBdrAttribute(i)-1]` test does; `ess_bdr` may be
    /// shorter than the mesh's attribute count, missing entries counting as
    /// not essential).
    ///
    /// This is the per-attribute form of [`Self::boundary_dofs`]: MFEM's
    /// `GetEssentialVDofs` loops over the boundary elements and unions
    /// `GetBdrElementDofs(i)` for the marked ones, so grouping by the boundary
    /// attribute of each side is exactly `GenerateBdrElementDofTable`'s
    /// `bel_dof` restricted to the marked rows.
    pub fn boundary_dofs_marked(&self, ess_bdr: &[bool]) -> Vec<u32> {
        let dim = self.dim;
        let mut mark = vec![false; self.n_dofs()];
        // A patch-boundary entity can be shared by several boundary elements
        // (one per knot span along it); each `(side, attribute)` is marked once.
        let mut sides = self.ext.boundary_sides().to_vec();
        sides.sort_unstable();
        sides.dedup();

        for side in &sides {
            let (patch, dir, low, attr) = (side.patch, side.dir, side.low, side.attr);
            // `bdr_attr_is_ess[attr-1]`, with attribute 0 (unnumbered) never
            // marked — MFEM's `bdr_attr_is_ess` is indexed the same way.
            if attr < 1 || !ess_bdr.get((attr - 1) as usize).copied().unwrap_or(false) {
                continue;
            }
            let kvs = self.ext.patch_knot_vectors(patch).expect("element patch");
            let ncp: Vec<usize> = kvs.iter().map(|k| k.ncp()).collect();
            let idx = if low { 0 } else { ncp[dir] - 1 };
            // Every control point of the patch entity: `multi[dir] = idx`, the
            // other directions run over the whole patch.
            let others: Vec<Vec<usize>> = (0..dim)
                .map(|d| if d == dir { vec![idx] } else { (0..ncp[d]).collect() })
                .collect();
            let total: usize = others.iter().map(|v| v.len()).product();
            for n in 0..total {
                let mut rest = n;
                let mut multi = vec![0usize; dim];
                for (d, o) in others.iter().enumerate() {
                    multi[d] = o[rest % o.len()];
                    rest /= o.len();
                }
                let g = self.ext.patch_dof(patch, &multi).expect("patch dof");
                mark[g] = true;
            }
        }
        (0..self.n_dofs())
            .filter(|&d| mark[d])
            .map(|d| d as u32)
            .collect()
    }

    /// `BilinearForm::Assemble` for `a(u,v) = ∫ kappa ∇u · ∇v`
    /// (`DiffusionIntegrator`, MFEM's quadrature order `2*p + dim - 1`).
    pub fn assemble_diffusion(&self, kappa: f64) -> CsrMatrix<f64> {
        self.assemble_bilinear(kappa, true)
    }

    /// `BilinearForm::Assemble` for `a(u,v) = ∫ alpha u v`
    /// (`MassIntegrator`, MFEM's quadrature order `2*p + dim - 1`).
    pub fn assemble_mass(&self, alpha: f64) -> CsrMatrix<f64> {
        self.assemble_bilinear(alpha, false)
    }

    /// The shared element loop of `DiffusionIntegrator`/`MassIntegrator`
    /// (`fem/bilininteg.cpp`), including MFEM's `Weight()`/`AdjugateJacobian`
    /// operation order.
    fn assemble_bilinear(&self, coeff: f64, diffusion: bool) -> CsrMatrix<f64> {
        let dim = self.dim;
        let n = self.n_dofs();
        let mut coo = CooMatrix::new(n, n);
        coo.reserve(n * 16);
        let mut shape = Vec::new();
        let mut grad = Vec::new();
        let mut dshapedxt = Vec::new();
        let mut k_elem = Vec::new();

        for e in 0..self.n_elements() {
            let nd = self.element_n_dofs(e);
            let fe = self.element_fe(e);
            let qo = 2 * fe.order() as u8 + dim as u8 - 1;
            let rule = nurbs_rule(dim, qo);
            k_elem.clear();
            k_elem.resize(nd * nd, 0.0);
            shape.clear();
            shape.resize(nd, 0.0);
            grad.clear();
            grad.resize(nd * dim, 0.0);
            dshapedxt.clear();
            dshapedxt.resize(nd * dim, 0.0);

            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.geometry(e, xi);
                let adj = adjugate(&geo.jac, dim);
                if diffusion {
                    // MFEM: w = ip.weight / Weight(); dshapedxt = dshape * AdjugateJacobian;
                    //       elmat += w * dshapedxt * dshapedxt^t.
                    let w = rule.weights[q] / geo.det_j;
                    fe.grad(xi, &mut grad);
                    for i in 0..nd {
                        for k in 0..dim {
                            let mut s = 0.0;
                            for j in 0..dim {
                                s += grad[i * dim + j] * adj[j][k];
                            }
                            dshapedxt[i * dim + k] = s;
                        }
                    }
                    for i in 0..nd {
                        for j in 0..nd {
                            let mut s = 0.0;
                            for k in 0..dim {
                                s += dshapedxt[i * dim + k] * dshapedxt[j * dim + k];
                            }
                            k_elem[i * nd + j] += w * s;
                        }
                    }
                } else {
                    // MFEM: w = Weight() * ip.weight; elmat += w * shape * shape^t.
                    let w = geo.det_j * rule.weights[q];
                    fe.shape(xi, &mut shape);
                    for i in 0..nd {
                        for j in 0..nd {
                            k_elem[i * nd + j] += w * shape[i] * shape[j];
                        }
                    }
                }
            }
            for v in k_elem.iter_mut() {
                *v *= coeff;
            }
            coo.add_element_matrix(self.ext.element_dofs(e), &k_elem);
        }
        coo.into_csr()
    }

    /// `GridFunction::ComputeL2Error(Coefficient, irs)` for the scalar NURBS
    /// space — per element, `u_h = Σ_j shape_j x_el` and each quadrature point
    /// contributes `ip.weight * Weight() * (u_h − u)²`.  `order_quad` is the
    /// caller's `irs` rule order (`nurbs_ex5`'s `max(2, 2*order+1)`).
    pub fn compute_l2_error(
        &self,
        x: &[f64],
        f: &dyn Fn(&[f64]) -> f64,
        order_quad: u8,
    ) -> f64 {
        let dim = self.dim;
        let mut error = 0.0_f64;
        let mut shape = Vec::new();
        for e in 0..self.n_elements() {
            let nd = self.element_dofs(e).len();
            let fe = self.element_fe(e);
            let rule = nurbs_rule(dim, order_quad);
            shape.clear();
            shape.resize(nd, 0.0);
            let mut elem_error = 0.0_f64;
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.geometry(e, xi);
                fe.shape(xi, &mut shape);
                let mut uh = 0.0_f64;
                for (o, &g) in self.ext.element_dofs(e).iter().enumerate() {
                    uh += shape[o] * x[g];
                }
                let d = uh - f(&geo.x[..dim]);
                elem_error += rule.weights[q] * geo.det_j * d * d;
            }
            error += elem_error.abs();
        }
        error.sqrt()
    }

    /// `LinearForm::Assemble` for `b(v) = ∫ f(x) v`
    /// (`DomainLFIntegrator`, MFEM's quadrature order `2*p`).
    pub fn assemble_domain_lf(&self, f: &dyn Fn(&[f64]) -> f64) -> Vec<f64> {
        let dim = self.dim;
        let mut rhs = vec![0.0_f64; self.n_dofs()];
        let mut shape = Vec::new();
        for e in 0..self.n_elements() {
            let nd = self.element_n_dofs(e);
            let fe = self.element_fe(e);
            let qo = 2 * fe.order() as u8;
            let rule = nurbs_rule(dim, qo);
            shape.clear();
            shape.resize(nd, 0.0);
            let mut elvec = vec![0.0_f64; nd];
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.geometry(e, xi);
                // MFEM: val = Weight() * Q.Eval(Trans, ip); elvect += ip.weight * val * shape.
                let val = geo.det_j * f(&geo.x[..dim]);
                let w = rule.weights[q] * val;
                fe.shape(xi, &mut shape);
                for i in 0..nd {
                    elvec[i] += w * shape[i];
                }
            }
            for (o, &g) in self.ext.element_dofs(e).iter().enumerate() {
                rhs[g] += elvec[o];
            }
        }
        rhs
    }
}

/// The `geometry` extension's element id for refined span `ijk` of patch `p`.
///
/// Uniform refinement subdivides every element span of a knot vector into
/// `2^ref_levels` sub-spans, so the *ordinal* of a refined span within its
/// patch's element list is `base_ordinal << ref_levels | sub`.  The ordinals are
/// used rather than the raw span indices because `NURBSFiniteElement::ijk`
/// stores raw knot-span indices, which are not consecutive when a knot vector
/// repeats an interior knot (`pipe-nurbs.mesh`).
fn ijk_to_element(
    geo: &NurbsExtension,
    mesh_ext: &NurbsExtension,
    patch: usize,
    ijk: [usize; 3],
    ref_levels: usize,
) -> Result<usize, String> {
    let base_spans = geo.patch_element_spans(patch)?;
    let ref_spans = mesh_ext.patch_element_spans(patch)?;
    let mut old = [0usize; 3];
    for d in 0..geo.dim() {
        let ordinal = ref_spans[d].iter().position(|&x| x == ijk[d]).ok_or_else(|| {
            format!("NurbsFESpace: patch {patch} has no span {} in direction {d}", ijk[d])
        })?;
        old[d] = base_spans[d][ordinal >> ref_levels];
    }
    for e in 0..geo.n_elements() {
        if geo.element_patch(e) == patch && geo.element_ijk(e) == old {
            return Ok(e);
        }
    }
    Err(format!(
        "NurbsFESpace: patch {patch} span {old:?} not found in the unrefined mesh extension"
    ))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NurbsHCurlSpace — the vector NURBS space of `NURBS_HCurlFECollection`
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The H(curl) NURBS element of one knot span: MFEM's
/// `NURBS_HCurl2DFiniteElement` / `NURBS_HCurl3DFiniteElement` bound to the
/// analysis space's knot vectors with `SetIJK` (`NURBSExtension::LoadFE`).
#[derive(Debug, Clone)]
pub enum HCurlSpanElement {
    /// MFEM `NURBS_HCurl2DFiniteElement`.
    Two(NurbsHCurl2D),
    /// MFEM `NURBS_HCurl3DFiniteElement`.
    Three(NurbsHCurl3D),
}

impl HCurlSpanElement {
    /// `FiniteElement::GetDof` (the number of *vector* basis functions).
    pub fn n_dofs(&self) -> usize {
        match self {
            HCurlSpanElement::Two(fe) => VectorReferenceElement::n_dofs(fe),
            HCurlSpanElement::Three(fe) => VectorReferenceElement::n_dofs(fe),
        }
    }

    /// `FiniteElement::GetOrder` — `max(orders) + 1`, the *elevated* degree
    /// MFEM's `SetOrder` reports (the quadrature orders of the curl-curl /
    /// vector-mass / L2-error integrators are `2*GetOrder()`-based).
    pub fn order(&self) -> usize {
        match self {
            HCurlSpanElement::Two(fe) => VectorReferenceElement::order(fe) as usize,
            HCurlSpanElement::Three(fe) => VectorReferenceElement::order(fe) as usize,
        }
    }

    /// `CalcVShape(ip, shape)` — the *reference-space* vector basis, `n_dofs`
    /// rows of `dim` components each (row-major, DOF-major).
    pub fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        match self {
            HCurlSpanElement::Two(fe) => fe.eval_basis_vec(xi, values),
            HCurlSpanElement::Three(fe) => fe.eval_basis_vec(xi, values),
        }
    }

    /// `CalcCurlShape(ip, curl_shape)` — the reference-space curl, `n_dofs`
    /// rows of `dimc` components (`dimc = 1` in 2D, `3` in 3D).
    pub fn eval_curl(&self, xi: &[f64], curl: &mut [f64]) {
        match self {
            HCurlSpanElement::Two(fe) => {
                fe.eval_curl(xi, curl);
            }
            HCurlSpanElement::Three(fe) => fe.eval_curl(xi, curl),
        }
    }

    /// `GetCurlDim`.
    pub fn curl_dim(&self) -> usize {
        match self {
            HCurlSpanElement::Two(_) => 1,
            HCurlSpanElement::Three(_) => 3,
        }
    }
}

/// The curl-extension knot vectors of the single patch: the base (analysis,
/// order `p`) vectors and the `DegreeElevate(1)` copies (`kv1`) that MFEM's
/// `NURBS_HCurl*FiniteElement::SetOrder` builds.  Both are evaluated at the
/// same span indices `ijk`, which degree elevation preserves.
#[derive(Debug, Clone)]
struct PatchKnots {
    base: Vec<KnotVector>,
    elevated: Vec<KnotVector>,
}

/// A vector-valued H(curl) NURBS finite element space — the
/// `NURBS_HCurlFECollection` path of MFEM's `nurbs_ex3`.
///
/// MFEM builds it (`FiniteElementSpace::UpdateNURBS`, `fem/fespace.cpp`) as
/// `VNURBSext[d] = NURBSext->GetCurlExtension(d)` for every component `d`:
/// each curl extension raises every knot-vector order of the *analysis*
/// extension by one and lowers the component's direction back, so its
/// element DOF table carries only that component's basis.  The space's DOFs
/// are the concatenation `ndofs = Σ_d VNURBSext[d]->GetNDof()` and the
/// per-element table is the offset-merged `Table(*t0, *t1, offset1, ...)`
/// — component-major, exactly the DOF order of `CalcVShape`.  The
/// `NURBSExtension(parent, newOrders)` constructors reset the analysis
/// weights to one, so (as in the scalar space) only the *geometry* is
/// rational; the vector elements never divide by a weight.
///
/// `GetCurlExtension` only works for single-patch meshes (MFEM raises an
/// error for `GetNP() > 1`), which this port enforces as well.
#[derive(Debug, Clone)]
pub struct NurbsHCurlSpace {
    /// The analysis space (order-`p` `NurbsFESpace`) — shared `NURBSext`:
    /// geometry evaluation, span indices and element numbering.
    base: NurbsFESpace,
    /// `VNURBSext[d]` — the per-component curl extensions.
    curl_ext: Vec<NurbsExtension>,
    /// Cumulative DOF offsets; component `d`'s global DOFs are
    /// `comp_offsets[d] + <local dof>`.
    comp_offsets: Vec<usize>,
    /// The merged element DOF table (`elem_dof`).
    elem_dof: Vec<Vec<usize>>,
    /// The patch knot vectors (base + elevated) of the single patch.
    patch_knots: PatchKnots,
    dim: usize,
    n_dofs: usize,
}

impl NurbsHCurlSpace {
    /// Build the space `nurbs_ex3` constructs: read the NURBS mesh, apply
    /// `ref_levels` uniform refinements, then `NURBSExtension(mesh->NURBSext,
    /// order)` and the `dim` curl extensions `GetCurlExtension(d)`.
    pub fn from_mesh_str(text: &str, ref_levels: usize, order: usize) -> Result<Self, String> {
        let base = NurbsFESpace::from_mesh_str(text, ref_levels, &[order])?;
        let dim = base.dim();
        let ext = base.extension();

        // `NURBSExtension::GetCurlExtension`: single patch only.
        if ext.n_patches() != 1 {
            return Err(format!(
                "NurbsHCurlSpace: GetCurlExtension only works for single patch NURBS meshes \
                 (this one has {} patches)",
                ext.n_patches()
            ));
        }

        // MFEM `GetCurlExtension(component)`: `newOrders = GetOrders(); for all
        // c: newOrders[c]++; newOrders[component]--;` — the orders are those of
        // the *analysis* extension, per patch direction.
        let dir_kv = ext.patch_direction_kv(0)?;
        let n_kv = ext.n_knot_vectors();
        let aorders: Vec<usize> = (0..n_kv).map(|i| ext.knot_vector(i).order()).collect();
        let mut curl_ext = Vec::with_capacity(dim);
        for c in 0..dim {
            let mut targets = aorders.clone();
            for t in targets.iter_mut() {
                *t += 1;
            }
            targets[dir_kv[c]] -= 1;
            curl_ext.push(ext.with_orders(&targets)?);
        }

        let mut comp_offsets = vec![0usize; dim + 1];
        for c in 0..dim {
            comp_offsets[c + 1] = comp_offsets[c] + curl_ext[c].n_dofs();
        }
        let n_dofs = comp_offsets[dim];

        // `FiniteElementSpace::UpdateNURBS`: merge the component tables with
        // one offset per component (`Table(*t0, *t1, offset1, ...)`).  All
        // extensions share the (order-independent) span enumeration, so the
        // rows merge element by element.
        let n_elem = ext.n_elements();
        let mut elem_dof = Vec::with_capacity(n_elem);
        for e in 0..n_elem {
            let mut row = Vec::new();
            for c in 0..dim {
                let r = curl_ext[c].element_dofs(e);
                if r.len() != curl_ext[c].element_dofs(0).len() {
                    return Err("NurbsHCurlSpace: component element tables disagree".to_string());
                }
                row.extend(r.iter().map(|&g| comp_offsets[c] + g));
            }
            elem_dof.push(row);
        }

        let patch_knots = {
            let kvs = ext.patch_knot_vectors(0)?;
            let kv_base: Vec<KnotVector> = kvs.iter().map(|k| k.knot_vector().clone()).collect();
            let kv_elev: Vec<KnotVector> = kv_base
                .iter()
                .map(|kv| degree_elevate(kv, 1))
                .collect::<Result<_, _>>()?;
            PatchKnots { base: kv_base, elevated: kv_elev }
        };

        Ok(Self { base, curl_ext, comp_offsets, elem_dof, patch_knots, dim, n_dofs })
    }

    /// Read a NURBS mesh file and build the space (see [`Self::from_mesh_str`]).
    pub fn from_mesh_file(
        path: impl AsRef<std::path::Path>,
        ref_levels: usize,
        order: usize,
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("NurbsHCurlSpace::from_mesh_file: {e}"))?;
        Self::from_mesh_str(&text, ref_levels, order)
    }

    /// `FiniteElementSpace::GetTrueVSize` — the merged DOF count.
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// `FiniteElementSpace::GetNE`.
    pub fn n_elements(&self) -> usize {
        self.base.n_elements()
    }

    /// Mesh dimension (2 or 3).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The merged element DOF table (`GetElementDofTable`), `n_elements` rows
    /// of `n_dofs_per_element` globally numbered DOFs (component-major).
    pub fn element_dofs(&self, e: usize) -> &[usize] {
        &self.elem_dof[e]
    }

    /// The analysis extension (MFEM `FiniteElementSpace::GetNURBSext`).
    pub fn extension(&self) -> &NurbsExtension {
        self.base.extension()
    }

    /// `FiniteElementSpace::GetBdrElementDofs` for this H(curl) space — the
    /// `Mode::H_CURL` boundary DOF table of every `VNURBSext[d]`, merged with
    /// the component offsets (all signs `+`, see
    /// [`NurbsExtension::boundary_dof_table`]).
    pub fn boundary_dof_table(&self) -> Vec<Vec<i64>> {
        let parts: Vec<(usize, Vec<Vec<i64>>)> = (0..self.dim)
            .map(|c| {
                (self.comp_offsets[c], self.curl_ext[c].boundary_dof_table(BdrDofMode::HCurl))
            })
            .collect();
        merge_component_tables(&parts)
    }

    /// The element's vector FE with `SetIJK` applied (`NURBSExtension::LoadFE`).
    pub fn element_fe(&self, e: usize) -> HCurlSpanElement {
        let ext = self.base.extension();
        let patch = ext.element_patch(e);
        let kvs = ext.patch_knot_vectors(patch).expect("element patch");
        let ijk = ext.element_ijk(e);
        match self.dim {
            2 => {
                let mut fe = NurbsHCurl2D::from_knot_vectors(
                    kvs[0].knot_vector().clone(),
                    kvs[1].knot_vector().clone(),
                )
                .expect("NurbsHCurl2D::from_knot_vectors");
                fe.set_ijk([ijk[0], ijk[1]]);
                HCurlSpanElement::Two(fe)
            }
            _ => {
                let mut fe = NurbsHCurl3D::from_knot_vectors(
                    kvs[0].knot_vector().clone(),
                    kvs[1].knot_vector().clone(),
                    kvs[2].knot_vector().clone(),
                )
                .expect("NurbsHCurl3D::from_knot_vectors");
                fe.set_ijk(ijk);
                HCurlSpanElement::Three(fe)
            }
        }
    }

    /// `FiniteElement::CalcPhysVShape` of a span element:
    /// `NURBS_HCurl*FiniteElement::CalcVShape(Trans, shape)` — the
    /// reference-space vector basis (`CalcVShape(ip, shape)`) mapped through
    /// `J⁻¹ = adj(J)/det(J)` (`Trans.InverseJacobian()`), which is the
    /// components' contravariant/covariant pairing the H(curl) trace uses.
    /// `ref_shape` is scratch of `n_dofs*dim` entries, `out` receives the
    /// `n_dofs x dim` physical values (row-major, DOF-major).
    fn phys_vshape(
        &self,
        fe: &HCurlSpanElement,
        xi: &[f64],
        geo: &Geometry,
        ref_shape: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        let dim = self.dim;
        let nd = fe.n_dofs();
        ref_shape.resize(nd * dim, 0.0);
        out.resize(nd * dim, 0.0);
        ref_shape.iter_mut().for_each(|v| *v = 0.0);
        out.iter_mut().for_each(|v| *v = 0.0);
        fe.eval_basis_vec(xi, ref_shape);
        let adj = adjugate(&geo.jac, dim);
        let inv_det = 1.0 / geo.det_j;
        for i in 0..nd {
            for c in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    s += ref_shape[i * dim + k] * (adj[k][c] * inv_det);
                }
                out[i * dim + c] = s;
            }
        }
    }

    /// MFEM `FiniteElementSpace::GetEssentialTrueDofs(ess_bdr = 1)` — the
    /// sorted unique DOF list of every mesh boundary element.
    ///
    /// `Generate{2,3}DBdrElementDofTable` (H_CURL mode) drops the component
    /// whose order is maximal on the boundary entity: a 2D edge only carries
    /// the component running *along* it, a 3D face only the two components
    /// lying *in* it.  The remaining components' boundary DOFs are all control
    /// points of the boundary entity, so the essential set is the union over
    /// the mesh boundary sides (one row per knot span collapses to the whole
    /// side) — then `MarkerToList` sorts and de-duplicates.
    pub fn essential_dofs(&self) -> Vec<u32> {
        let dim = self.dim;
        let ext = self.base.extension();
        let mut mark = vec![false; self.n_dofs];

        // One entry per boundary element; a side is shared by all spans along
        // it, so de-duplicate first (`activeBdrElem` enumeration order).
        let mut sides = ext.boundary_sides().to_vec();
        sides.sort_unstable();
        sides.dedup();

        for side in &sides {
            let (patch, dir, low) = (side.patch, side.dir, side.low);
            for c in 0..dim {
                // H_CURL `Generate{2,3}DBdrElementDofTable`: dofs exist iff the
                // entity's tangential knot-vector order differs from
                // `mOrders.Max()`, i.e. iff the lowered component `c` is
                // tangential to the entity (`c != dir`, with `dir` the
                // fixed/normal direction of the side).
                if c == dir {
                    continue;
                }
                let cext = &self.curl_ext[c];
                let kvs = cext.patch_knot_vectors(patch).expect("element patch");
                let ncp: Vec<usize> = kvs.iter().map(|k| k.ncp()).collect();
                let fixed = if low { 0 } else { ncp[dir] - 1 };
                let ranges: Vec<Vec<usize>> = (0..dim)
                    .map(|d| if d == dir { vec![fixed] } else { (0..ncp[d]).collect() })
                    .collect();
                let total: usize = ranges.iter().map(|v| v.len()).product();
                for n in 0..total {
                    let mut rest = n;
                    let mut multi = vec![0usize; dim];
                    for (d, r) in ranges.iter().enumerate() {
                        multi[d] = r[rest % r.len()];
                        rest /= r.len();
                    }
                    let g = cext.patch_dof(patch, &multi).expect("patch dof");
                    mark[self.comp_offsets[c] + g] = true;
                }
            }
        }
        (0..self.n_dofs)
            .filter(|&d| mark[d])
            .map(|d| d as u32)
            .collect()
    }

    /// `GridFunction::ProjectCoefficient(VectorCoefficient&)` — MFEM's
    /// **default** dispatch for a NURBS space.
    ///
    /// `GridFunction::ProjectCoefficient` routes `ProjectType::DEFAULT` to
    /// `ProjectCoefficientElementL2(vcoeff)` whenever `fes->GetNURBSext() !=
    /// NULL` (fem/gridfunc.cpp): the element-local L² projection followed by a
    /// least-squares fit back onto the NURBS basis, **not** the Botella-point
    /// interpolation of `ProjectType::ELEMENT`.  This is the call
    /// `nurbs_ex3.cpp` makes with its bare `x.ProjectCoefficient(E)`.
    pub fn project_coefficient(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        self.project_coefficient_element_l2(f)
    }

    /// `ProjectCoefficientElementL2_(VectorCoefficient&, x, Va)` followed by
    /// `(*this) /= Va` (`GridFunction::ProjectCoefficientElementL2`) — the
    /// NURBS branch, i.e. MFEM's default projection for a
    /// `NURBS_HCurlFECollection` space.
    ///
    /// Per element `e` (fem/gridfunc.cpp):
    ///
    /// 1. `el` = the span's `NURBS_HCurl2D/3DFiniteElement`, `dof = el.GetDof()`,
    ///    `dim = el.GetRangeDim()`, `p = el.GetOrder()` — the *elevated* degree
    ///    `max(orders)+1` reported by `NURBSFiniteElement::SetOrder`, not the
    ///    span's knot order (`NURBS_HCurlFECollection`'s own order `o` gives
    ///    `p = o+1`).
    /// 2. `el2` = `L2_FECollection(p, dim).FiniteElementForGeometry(geom)`, i.e.
    ///    `L2_QuadrilateralElement(p, GaussLegendre)` /
    ///    `L2_HexahedronElement(p, GaussLegendre)`: `dof2 = (p+1)^dim` nodal DOFs
    ///    at the Gauss-Legendre points of `[0,1]^dim`.
    /// 3. Quadrature `IntRules.Get(geom, 2*p+1)` — `p+1` Gauss points per
    ///    direction, the same 1-D rule that defines the L2 nodes.  Accumulate
    ///    the `dim` independent L² projections of the coefficient components
    ///    (`shape2`), the L² mass matrix (`elmat`: one identical `dof2 x dof2`
    ///    block per component) and the NURBS weight `elwght[j] += w * ‖vshape_j‖₂`
    ///    (`DenseMatrix::GetRowl2` of the *physical* vector shape functions —
    ///    the Buffa–Sangalli–Vázquez partition-of-unity normaliser).
    /// 4. `LinearSolve(elmat, elvect)`: the component-wise L² projection
    ///    coefficients in the L2 nodal basis.
    /// 5. `el2.Project(el, tr, I)`: `I` is `(dim*dof2) x dof` with
    ///    `I(d*dof2 + k, j) = vshape_j,d(`node `k)` — `NodalFiniteElement::Project`
    ///    evaluates the NURBS vector shape functions at the L2 nodes (physical
    ///    space; `L2_FECollection`'s default `VALUE` map type adds no
    ///    `Trans.Weight()` factor).  The NURBS DOFs are then the least-squares
    ///    solution of `I x ≈ elvect`, `(IᵀI) x = Iᵀ elvect`
    ///    (`I.Transpose(); I.Mult(elvect, vec); MultAAt(I, mat)`).
    /// 6. `elvect *= elwght`, accumulate `elvect` into `x` through the signed
    ///    `GetElementVDofs` entries and `elwght` into `Va` through their
    ///    `DecodeDof` component numbers; finally `x /= Va`.
    ///
    /// `dof2*dim >= dof` always holds for these elements, so the LSQ system is
    /// overdetermined — but `IᵀI` inherits the conditioning of the L² pairing
    /// of the two bases: MFEM itself warns "This project is not stable for
    /// NURBS VectorFE with order >= 5".
    pub fn project_coefficient_element_l2(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        let dim = self.dim;
        let mut x = vec![0.0_f64; self.n_dofs];
        let mut va = vec![0.0_f64; self.n_dofs];
        let mut shape2: Vec<f64> = Vec::new();
        let mut ref_shape: Vec<f64> = Vec::new();
        let mut shape: Vec<f64> = Vec::new();
        for e in 0..self.n_elements() {
            let fe = self.element_fe(e);
            let nd = self.elem_dof[e].len();
            // `el.GetOrder()` of the *loaded* NURBS element: `max(orders)+1`.
            let p = fe.order();
            let dof2 = (p + 1).pow(dim as u32);
            // `IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 1)`.
            let rule = nurbs_rule(dim, (2 * p + 1) as u8);
            // `Poly_1D::OpenPoints(p, GaussLegendre)` — the L2 element's 1-D
            // nodal points, i.e. the same Gauss-Legendre rule as the quadrature
            // above; evaluated once per element.
            let l2_nodes = gauss_legendre_01(p + 1).0;

            shape2.clear();
            shape2.resize(dof2, 0.0);
            ref_shape.clear();
            ref_shape.resize(nd * dim, 0.0);
            shape.clear();
            shape.resize(nd * dim, 0.0);

            let nblk = dof2 * dim;
            let mut elvect = vec![0.0_f64; nblk];
            let mut elmat = vec![0.0_f64; nblk * nblk];
            let mut elwght = vec![0.0_f64; nd];
            let mut partelmat = vec![0.0_f64; dof2 * dof2];

            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.base.geometry(e, xi);
                // `real_t wght = ip.weight*tr.Weight();`
                let w = rule.weights[q] * geo.det_j;
                // `vcoeff.Eval(val, tr, ip); val *= wght;`
                let val: Vec<f64> = f(&geo.x[..dim]).iter().map(|v| v * w).collect();
                // `el2.CalcPhysShape` (`VALUE` map type: the plain tensor
                // Lagrange basis of the Gauss-Legendre nodes).
                l2_gl_shape(dim, &l2_nodes, xi, &mut shape2);
                // `el.CalcPhysVShape`.
                self.phys_vshape(&fe, xi, &geo, &mut ref_shape, &mut shape);

                for c in 0..dim {
                    for s in 0..dof2 {
                        elvect[dof2 * c + s] += val[c] * shape2[s];
                    }
                }
                // `MultVVt(shape2, partelmat); partelmat *= wght;` followed by
                // `elmat.AddMatrix(partelmat, dof2*k, dof2*k)` per component.
                for r in 0..dof2 {
                    for s in 0..dof2 {
                        partelmat[r * dof2 + s] = shape2[r] * shape2[s] * w;
                    }
                }
                for c in 0..dim {
                    let off = dof2 * c;
                    for r in 0..dof2 {
                        for s in 0..dof2 {
                            elmat[(off + r) * nblk + off + s] += partelmat[r * dof2 + s];
                        }
                    }
                }
                // `shape.GetRowl2(shapel2); elwght.Add(wght, shapel2);`
                for j in 0..nd {
                    let mut s2 = 0.0;
                    for c in 0..dim {
                        s2 += shape[j * dim + c] * shape[j * dim + c];
                    }
                    elwght[j] += w * s2.sqrt();
                }
            }

            // `LinearSolve(elmat, elvect.GetData())` (default TOL = 1e-9).
            if !dense_lu_solve(&mut elmat, nblk, &mut elvect, 1e-9) {
                panic!(
                    "NurbsHCurlSpace::project_coefficient_element_l2: singular L2 mass matrix"
                );
            }

            // `el2.Project(el, tr, I)`, I(d*dof2 + k, j) = vshape_j,d(node k).
            let mut imat = vec![0.0_f64; nblk * nd];
            for k in 0..dof2 {
                let node: Vec<f64> = if dim == 2 {
                    vec![l2_nodes[k % (p + 1)], l2_nodes[k / (p + 1)]]
                } else {
                    vec![
                        l2_nodes[k % (p + 1)],
                        l2_nodes[(k / (p + 1)) % (p + 1)],
                        l2_nodes[k / ((p + 1) * (p + 1))],
                    ]
                };
                let geo = self.base.geometry(e, &node);
                self.phys_vshape(&fe, &node, &geo, &mut ref_shape, &mut shape);
                for j in 0..nd {
                    for c in 0..dim {
                        imat[(c * dof2 + k) * nd + j] = shape[j * dim + c];
                    }
                }
            }

            // `I.Transpose(); I.Mult(elvect, vec); MultAAt(I, mat);` then
            // `LinearSolve(mat, vec, 1e-24)`: the LSQ fit of the NURBS DOFs.
            let mut vec = vec![0.0_f64; nd];
            let mut mat = vec![0.0_f64; nd * nd];
            for j in 0..nd {
                let mut s = 0.0;
                for r in 0..nblk {
                    s += imat[r * nd + j] * elvect[r];
                }
                vec[j] = s;
                for j2 in 0..nd {
                    let mut t = 0.0;
                    for r in 0..nblk {
                        t += imat[r * nd + j] * imat[r * nd + j2];
                    }
                    mat[j * nd + j2] = t;
                }
            }
            if !dense_lu_solve(&mut mat, nd, &mut vec, 1e-24) {
                panic!("NurbsHCurlSpace::project_coefficient_element_l2: singular IᵀI");
            }

            // `elvect = vec; elvect *= elwght;` then `AddElementVector`.
            for (j, &g) in self.elem_dof[e].iter().enumerate() {
                x[g] += vec[j] * elwght[j];
                va[g] += elwght[j];
            }
        }
        for i in 0..self.n_dofs {
            x[i] /= va[i];
        }
        x
    }

    /// `GridFunction::ProjectCoefficient(VectorCoefficient&, ProjectType::ELEMENT)`
    /// for the H(curl) space — MFEM loops the elements and calls
    /// `NURBS_HCurl*FiniteElement::Project`, which assigns each local DOF the
    /// component-`c` value of `Jᵀ E(x_phys)` at the DOF's Botella abscissa
    /// (`KnotVector::GetBotella` Newton iteration) when it lies in the
    /// element's span; DOFs outside remain untouched (they are assigned by the
    /// element that owns their span).
    pub fn project_coefficient_element(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        let dim = self.dim;
        let mut x = vec![0.0_f64; self.n_dofs];
        for e in 0..self.n_elements() {
            let ext = self.base.extension();
            let ijk = ext.element_ijk(e);
            let kv = &self.patch_knots.base;
            let kv1 = &self.patch_knots.elevated;
            let orders: Vec<usize> =
                (0..dim).map(|d| knot_order(&kv[d]).expect("validated knot vector")).collect();
            let (ox, oy, oz) = match dim {
                2 => (orders[0], orders[1], 0),
                _ => (orders[0], orders[1], orders[2]),
            };

            let ref_point = |knots: &[f64], u: f64, ni: usize| {
                (u - knots[ni]) / (knots[ni + 1] - knots[ni])
            };

            let nd = self.elem_dof[e].len();
            // MFEM's `NURBS_HCurl*FiniteElement::Project` leaves DOFs it cannot
            // evaluate untouched ("Dofs that can not be evaluated will remain
            // unmodified"), and the NURBS branch of
            // `GridFunction::ProjectCoefficient` filters them out with an
            // `-infinity()` sentinel before `SetSubVector` — such a DOF keeps
            // the value written by the element that owns its span.
            let mut dofs = vec![f64::NEG_INFINITY; nd];
            match dim {
                2 => {
                    let (kx, ky) = (kv[0].as_slice(), kv[1].as_slice());
                    let (kx1, ky1) = (kv1[0].as_slice(), kv1[1].as_slice());
                    let mut o = 0usize;
                    for j in 0..=oy + 1 {
                        let kz = knot_botella(&kv1[1], ijk[1] + j);
                        if !knot_in_span(ky1, kz, ijk[1] + oy + 1) {
                            o += ox + 1;
                            continue;
                        }
                        let ipy = ref_point(ky1, kz, ijk[1] + oy + 1);
                        for i in 0..=ox {
                            // MFEM writes `for (int i = 0; i <= orders[0]; i++, o++)`:
                            // `o` advances for skipped DOFs too (they belong to the
                            // neighbouring span and must stay untouched).
                            let oi = o;
                            o += 1;
                            let kxv = knot_botella(&kv[0], ijk[0] + i);
                            if !knot_in_span(kx, kxv, ijk[0] + ox) {
                                continue;
                            }
                            let ipx = ref_point(kx, kxv, ijk[0] + ox);
                            let geo = self.base.geometry(e, &[ipx, ipy]);
                            let ev = f(&geo.x[..dim]);
                            // dofs(o) = (Jᵀ E)(0)
                            dofs[oi] = geo.jac[0][0] * ev[0] + geo.jac[1][0] * ev[1];
                        }
                    }
                    for j in 0..=oy {
                        let kz = knot_botella(&kv[1], ijk[1] + j);
                        if !knot_in_span(ky, kz, ijk[1] + oy) {
                            o += ox + 2;
                            continue;
                        }
                        let ipy = ref_point(ky, kz, ijk[1] + oy);
                        for i in 0..=ox + 1 {
                            let oi = o;
                            o += 1;
                            let kxv = knot_botella(&kv1[0], ijk[0] + i);
                            if !knot_in_span(kx1, kxv, ijk[0] + ox + 1) {
                                continue;
                            }
                            let ipx = ref_point(kx1, kxv, ijk[0] + ox + 1);
                            let geo = self.base.geometry(e, &[ipx, ipy]);
                            let ev = f(&geo.x[..dim]);
                            dofs[oi] = geo.jac[0][1] * ev[0] + geo.jac[1][1] * ev[1];
                        }
                    }
                    debug_assert_eq!(o, nd);
                }
                _ => {
                    let (kx, ky, kz0) =
                        (kv[0].as_slice(), kv[1].as_slice(), kv[2].as_slice());
                    let (kx1, ky1, kz1) = (
                        kv1[0].as_slice(),
                        kv1[1].as_slice(),
                        kv1[2].as_slice(),
                    );
                    let mut o = 0usize;
                    for k in 0..=oz + 1 {
                        let kzz = knot_botella(&kv1[2], ijk[2] + k);
                        if !knot_in_span(kz1, kzz, ijk[2] + oz + 1) {
                            o += (ox + 1) * (oy + 2);
                            continue;
                        }
                        let ipz = ref_point(kz1, kzz, ijk[2] + oz + 1);
                        for j in 0..=oy + 1 {
                            let kyy = knot_botella(&kv1[1], ijk[1] + j);
                            if !knot_in_span(ky1, kyy, ijk[1] + oy + 1) {
                                o += ox + 1;
                                continue;
                            }
                            let ipy = ref_point(ky1, kyy, ijk[1] + oy + 1);
                            for i in 0..=ox {
                                let oi = o;
                                o += 1;
                                let kxx = knot_botella(&kv[0], ijk[0] + i);
                                if !knot_in_span(kx, kxx, ijk[0] + ox) {
                                    continue;
                                }
                                let ipx = ref_point(kx, kxx, ijk[0] + ox);
                                let geo = self.base.geometry(e, &[ipx, ipy, ipz]);
                                let ev = f(&geo.x[..dim]);
                                dofs[oi] = geo.jac[0][0] * ev[0]
                                    + geo.jac[1][0] * ev[1]
                                    + geo.jac[2][0] * ev[2];
                            }
                        }
                    }
                    for k in 0..=oz + 1 {
                        let kzz = knot_botella(&kv1[2], ijk[2] + k);
                        if !knot_in_span(kz1, kzz, ijk[2] + oz + 1) {
                            o += (ox + 2) * (oy + 1);
                            continue;
                        }
                        let ipz = ref_point(kz1, kzz, ijk[2] + oz + 1);
                        for j in 0..=oy {
                            let kyy = knot_botella(&kv[1], ijk[1] + j);
                            if !knot_in_span(ky, kyy, ijk[1] + oy) {
                                o += ox + 2;
                                continue;
                            }
                            let ipy = ref_point(ky, kyy, ijk[1] + oy);
                            for i in 0..=ox + 1 {
                                let oi = o;
                                o += 1;
                                let kxx = knot_botella(&kv1[0], ijk[0] + i);
                                if !knot_in_span(kx1, kxx, ijk[0] + ox + 1) {
                                    continue;
                                }
                                let ipx = ref_point(kx1, kxx, ijk[0] + ox + 1);
                                let geo = self.base.geometry(e, &[ipx, ipy, ipz]);
                                let ev = f(&geo.x[..dim]);
                                dofs[oi] = geo.jac[0][1] * ev[0]
                                    + geo.jac[1][1] * ev[1]
                                    + geo.jac[2][1] * ev[2];
                            }
                        }
                    }
                    for k in 0..=oz {
                        let kzz = knot_botella(&kv[2], ijk[2] + k);
                        if !knot_in_span(kz0, kzz, ijk[2] + oz) {
                            o += (ox + 2) * (oy + 2);
                            continue;
                        }
                        let ipz = ref_point(kz0, kzz, ijk[2] + oz);
                        for j in 0..=oy + 1 {
                            let kyy = knot_botella(&kv1[1], ijk[1] + j);
                            if !knot_in_span(ky1, kyy, ijk[1] + oy + 1) {
                                o += ox + 2;
                                continue;
                            }
                            let ipy = ref_point(ky1, kyy, ijk[1] + oy + 1);
                            for i in 0..=ox + 1 {
                                let oi = o;
                                o += 1;
                                let kxx = knot_botella(&kv1[0], ijk[0] + i);
                                if !knot_in_span(kx1, kxx, ijk[0] + ox + 1) {
                                    continue;
                                }
                                let ipx = ref_point(kx1, kxx, ijk[0] + ox + 1);
                                let geo = self.base.geometry(e, &[ipx, ipy, ipz]);
                                let ev = f(&geo.x[..dim]);
                                dofs[oi] = geo.jac[0][2] * ev[0]
                                    + geo.jac[1][2] * ev[1]
                                    + geo.jac[2][2] * ev[2];
                            }
                        }
                    }
                    debug_assert_eq!(o, nd);
                }
            }
            // `GridFunction::ProjectCoefficient` (NURBS, `ProjectType::ELEMENT`):
            // only the defined DOFs of this element are written.
            for (o, &g) in self.elem_dof[e].iter().enumerate() {
                if dofs[o] != f64::NEG_INFINITY {
                    x[g] = dofs[o];
                }
            }
        }
        x
    }

    /// `BilinearForm::Assemble` of `curl curl E + sigma E` with the constant
    /// coefficients `muinv`/`sigma` (`CurlCurlIntegrator` +
    /// `VectorFEMassIntegrator`, in MFEM's elementwise accumulation order).
    pub fn assemble_system(&self, muinv: f64, sigma: f64) -> CsrMatrix<f64> {
        let dim = self.dim;
        let n = self.n_dofs;
        let mut coo = CooMatrix::new(n, n);
        coo.reserve(n * 16);
        let mut curl_ref: Vec<f64> = Vec::new();
        let mut vshape: Vec<f64> = Vec::new();
        let mut curl_phys: Vec<f64> = Vec::new();
        let mut k_cc: Vec<f64> = Vec::new();
        let mut k_m: Vec<f64> = Vec::new();

        // `Trans.OrderW()` of the NURBS mesh transformation: the scalar
        // geometry FE has `FunctionSpace::Qk` and the mesh's own order, so
        // `OrderW() = mesh_order * dim - 1`.
        let mesh_order = {
            let geo_ext = self.base.mesh_extension();
            let kvs = geo_ext.patch_knot_vectors(0).expect("geometry patch");
            knot_order(kvs[0].knot_vector()).expect("validated knot vector")
        };
        let order_w = mesh_order * dim - 1;

        for e in 0..self.n_elements() {
            let nd = self.elem_dof[e].len();
            let fe = self.element_fe(e);
            let dimc = fe.curl_dim();
            let el_order = fe.order();
            // CurlCurlIntegrator: order = 2*el.GetOrder() (Qk).
            let rule_cc = nurbs_rule(dim, (2 * el_order) as u8);
            // VectorFEMassIntegrator: order = OrderW() + 2*el.GetOrder().
            let rule_m = nurbs_rule(dim, (order_w + 2 * el_order) as u8);

            curl_phys.clear();
            curl_phys.resize(nd * dimc, 0.0);
            k_cc.clear();
            k_cc.resize(nd * nd, 0.0);
            k_m.clear();
            k_m.resize(nd * nd, 0.0);

            // CurlCurlIntegrator::AssembleElementMatrix.
            for q in 0..rule_cc.points.len() {
                let xi = &rule_cc.points[q];
                let geo = self.base.geometry(e, xi);
                let w = rule_cc.weights[q] * geo.det_j * muinv;
                curl_ref.clear();
                curl_ref.resize(nd * dimc, 0.0);
                fe.eval_curl(xi, &mut curl_ref);
                // `CalcPhysCurlShape`: 2D scales the reference curl by
                // `1/Weight()`; 3D maps it through `MultABt(., J, .)` first.
                for i in 0..nd {
                    for c in 0..dimc {
                        let v = if dim == 2 {
                            curl_ref[i] * (1.0 / geo.det_j)
                        } else {
                            let mut s = 0.0;
                            for k in 0..dim {
                                s += curl_ref[i * dimc + k] * geo.jac[c][k];
                            }
                            s * (1.0 / geo.det_j)
                        };
                        curl_phys[i * dimc + c] = v;
                    }
                }
                // `AddMult_a_AAt(w, curlshape_dFt, elmat)`.
                for i in 0..nd {
                    for j in 0..nd {
                        let mut s = 0.0;
                        for c in 0..dimc {
                            s += curl_phys[i * dimc + c] * curl_phys[j * dimc + c];
                        }
                        k_cc[i * nd + j] += w * s;
                    }
                }
            }

            // VectorFEMassIntegrator::AssembleElementMatrix.
            vshape.clear();
            vshape.resize(nd * dim, 0.0);
            for q in 0..rule_m.points.len() {
                let xi = &rule_m.points[q];
                let geo = self.base.geometry(e, xi);
                let w = rule_m.weights[q] * geo.det_j * sigma;
                // `CalcVShape(Trans, vshape)`: reference basis then `J⁻¹`.
                self.phys_vshape(&fe, xi, &geo, &mut curl_ref, &mut vshape);
                // `AddMult_a_AAt(w, trial_vshape, elmat)`.
                for i in 0..nd {
                    for j in 0..nd {
                        let mut s = 0.0;
                        for c in 0..dim {
                            s += vshape[i * dim + c] * vshape[j * dim + c];
                        }
                        k_m[i * nd + j] += w * s;
                    }
                }
            }

            // `elmat = elemmat(cc); elmat += elemmat(mass)`.
            let mut k_elem = vec![0.0_f64; nd * nd];
            for i in 0..nd * nd {
                k_elem[i] = k_cc[i] + k_m[i];
            }
            coo.add_element_matrix(&self.elem_dof[e], &k_elem);
        }
        coo.into_csr()
    }

    /// `LinearForm::Assemble` of `(f, v)` (`VectorFEDomainLFIntegrator`,
    /// quadrature order `2*GetOrder()`).
    pub fn assemble_vector_domain_lf(
        &self,
        f: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let dim = self.dim;
        let mut rhs = vec![0.0_f64; self.n_dofs];
        for e in 0..self.n_elements() {
            let nd = self.elem_dof[e].len();
            let fe = self.element_fe(e);
            let rule = nurbs_rule(dim, (2 * fe.order()) as u8);
            let mut elvec = vec![0.0_f64; nd];
            let mut vshape = vec![0.0_f64; nd * dim];
            let mut vec = vec![0.0_f64; dim];
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.base.geometry(e, xi);
                // `CalcVShape(Trans, vshape)`: reference basis then `J⁻ᵀ`.
                let mut vref = vec![0.0_f64; nd * dim];
                fe.eval_basis_vec(xi, &mut vref);
                let adj = adjugate(&geo.jac, dim);
                let inv_det = 1.0 / geo.det_j;
                for i in 0..nd {
                    for c in 0..dim {
                        let mut s = 0.0;
                        for k in 0..dim {
                            s += vref[i * dim + k] * (adj[k][c] * inv_det);
                        }
                        vshape[i * dim + c] = s;
                    }
                }
                let fv = f(&geo.x[..dim]);
                let w = rule.weights[q] * geo.det_j;
                for c in 0..dim {
                    vec[c] = fv[c] * w;
                }
                // `vshape.AddMult(vec, elvect)`.
                for o in 0..nd {
                    let mut s = 0.0;
                    for c in 0..dim {
                        s += vshape[o * dim + c] * vec[c];
                    }
                    elvec[o] += s;
                }
            }
            for (o, &g) in self.elem_dof[e].iter().enumerate() {
                rhs[g] += elvec[o];
            }
        }
        rhs
    }

    /// `GridFunction::ComputeL2Error(VectorCoefficient)` — per element,
    /// quadrature order `2*GetOrder() + 3`, the vector values are
    /// `CalcVShape(Trans)ᵀ x_el` and each point contributes
    /// `ip.weight * Weight() * ‖E_h − E‖²`.
    pub fn compute_l2_error(&self, x: &[f64], f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
        let dim = self.dim;
        let mut error = 0.0_f64;
        for e in 0..self.n_elements() {
            let nd = self.elem_dof[e].len();
            let fe = self.element_fe(e);
            let rule = nurbs_rule(dim, (2 * fe.order() + 3) as u8);
            let mut elem_error = 0.0_f64;
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.base.geometry(e, xi);
                let mut vref = vec![0.0_f64; nd * dim];
                fe.eval_basis_vec(xi, &mut vref);
                let adj = adjugate(&geo.jac, dim);
                let inv_det = 1.0 / geo.det_j;
                let fv = f(&geo.x[..dim]);
                // `vals -= exact_vals; vals.Norm2(...)` per point.
                let mut n2 = 0.0_f64;
                for c in 0..dim {
                    let mut val_c = 0.0_f64;
                    for o in 0..nd {
                        let mut s = 0.0;
                        for k in 0..dim {
                            s += vref[o * dim + k] * (adj[k][c] * inv_det);
                        }
                        val_c += s * x[self.elem_dof[e][o]];
                    }
                    let d = val_c - fv[c];
                    n2 += d * d;
                }
                // `loc_errs(j) = Norm2(vals)` then squared again.
                let nrm = n2.sqrt();
                elem_error += rule.weights[q] * geo.det_j * (nrm * nrm);
            }
            // Negative quadrature weights may cause the error to be negative.
            error += elem_error.abs();
        }
        error.sqrt()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NurbsHDivSpace — the vector NURBS space of `NURBS_HDivFECollection`
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The H(div) NURBS element of one knot span: MFEM's
/// `NURBS_HDiv2DFiniteElement` / `NURBS_HDiv3DFiniteElement` bound to the
/// analysis space's knot vectors with `SetIJK` (`NURBSExtension::LoadFE`).
#[derive(Debug, Clone)]
pub enum HDivSpanElement {
    /// MFEM `NURBS_HDiv2DFiniteElement`.
    Two(NurbsHDiv2D),
    /// MFEM `NURBS_HDiv3DFiniteElement`.
    Three(NurbsHDiv3D),
}

impl HDivSpanElement {
    /// `FiniteElement::GetDof` (the number of *vector* basis functions).
    ///
    /// Note MFEM's own `NURBS_HDiv2DFiniteElement::SetOrder` computes
    /// `dof = (o0+2)*(o1+1) + (o1+1)*(o1+2)` — the second block uses `o1` twice.
    /// That agrees with the space's merged table
    /// `(o0+2)*(o1+1) + (o0+1)*(o1+2)` only when `o0 == o1` (a uniform order,
    /// which is what both miniapps build), and the Rust element mirrors MFEM
    /// verbatim either way.
    pub fn n_dofs(&self) -> usize {
        match self {
            HDivSpanElement::Two(fe) => VectorReferenceElement::n_dofs(fe),
            HDivSpanElement::Three(fe) => VectorReferenceElement::n_dofs(fe),
        }
    }

    /// `FiniteElement::GetOrder` — `max(orders) + 1`, the *elevated* degree
    /// MFEM's `SetOrder` reports.
    pub fn order(&self) -> usize {
        match self {
            HDivSpanElement::Two(fe) => VectorReferenceElement::order(fe) as usize,
            HDivSpanElement::Three(fe) => VectorReferenceElement::order(fe) as usize,
        }
    }

    /// `FiniteElement::GetDim`.
    pub fn dim(&self) -> usize {
        match self {
            HDivSpanElement::Two(_) => 2,
            HDivSpanElement::Three(_) => 3,
        }
    }

    /// `CalcVShape(ip, shape)` — the *reference-space* vector basis, `n_dofs`
    /// rows of `dim` components each (row-major, DOF-major).
    pub fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        match self {
            HDivSpanElement::Two(fe) => fe.eval_basis_vec(xi, values),
            HDivSpanElement::Three(fe) => fe.eval_basis_vec(xi, values),
        }
    }

    /// `CalcDivShape(ip, divshape)` — the reference-space divergence,
    /// `n_dofs` values.
    pub fn eval_div(&self, xi: &[f64], values: &mut [f64]) {
        match self {
            HDivSpanElement::Two(fe) => fe.eval_div(xi, values),
            HDivSpanElement::Three(fe) => fe.eval_div(xi, values),
        }
    }
}

/// A vector-valued H(div) NURBS finite element space — the
/// `NURBS_HDivFECollection` path of MFEM's `nurbs_ex5` / `nurbs_ex24`.
///
/// MFEM builds it (`FiniteElementSpace::UpdateNURBS`) as
/// `VNURBSext[d] = NURBSext->GetDivExtension(d)` for every component `d`:
/// `GetDivExtension(component)` *raises* the component's own knot-vector order
/// by one (`newOrders[component] += 1`) and leaves the others alone, so the
/// extension's element DOF table carries only that component's basis.  As for
/// H(curl) the space's DOFs are the concatenation
/// `ndofs = Σ_d VNURBSext[d]->GetNDof()` and the per-element table is the
/// offset-merged `Table(*t0, *t1, offset1, …)`, component-major — exactly the
/// DOF order of `CalcVShape`.
///
/// `GetDivExtension` only works for single-patch meshes (MFEM raises an error
/// for `GetNP() > 1`), which this port enforces as well.
#[derive(Debug, Clone)]
pub struct NurbsHDivSpace {
    /// The analysis space (order-`p` `NurbsFESpace`): geometry evaluation,
    /// span indices and element numbering; also the scalar space MFEM's
    /// `nurbs_ex5` uses for the pressure (`NURBSFECollection(order)`).
    base: NurbsFESpace,
    /// `VNURBSext[d]` — the per-component divergence extensions.
    div_ext: Vec<NurbsExtension>,
    /// Cumulative DOF offsets; component `d`'s global DOFs are
    /// `comp_offsets[d] + <local dof>`.
    comp_offsets: Vec<usize>,
    /// The merged element DOF table (`elem_dof`).
    elem_dof: Vec<Vec<usize>>,
    dim: usize,
    n_dofs: usize,
}

impl NurbsHDivSpace {
    /// Build the space `nurbs_ex5`/`nurbs_ex24` construct: read the NURBS
    /// mesh, apply `ref_levels` uniform refinements, then
    /// `NURBSExtension(mesh->NURBSext, order)` and the `dim` divergence
    /// extensions `GetDivExtension(d)`.
    pub fn from_mesh_str(text: &str, ref_levels: usize, order: usize) -> Result<Self, String> {
        let base = NurbsFESpace::from_mesh_str(text, ref_levels, &[order])?;
        let dim = base.dim();
        if dim < 2 {
            return Err(format!("NurbsHDivSpace: dimension {dim} is not supported (2 or 3)"));
        }
        let ext = base.extension();

        // `NURBSExtension::GetDivExtension`: single patch only.
        if ext.n_patches() != 1 {
            return Err(format!(
                "NurbsHDivSpace: GetDivExtension only works for single patch NURBS meshes \
                 (this one has {} patches)",
                ext.n_patches()
            ));
        }

        // MFEM `GetDivExtension(component)`: `newOrders = GetOrders();
        // newOrders[component] += 1;`.
        let n_kv = ext.n_knot_vectors();
        let aorders: Vec<usize> = (0..n_kv).map(|i| ext.knot_vector(i).order()).collect();
        let mut div_ext = Vec::with_capacity(dim);
        for c in 0..dim {
            let mut targets = aorders.clone();
            targets[c] += 1;
            div_ext.push(ext.with_orders(&targets)?);
        }

        let mut comp_offsets = vec![0usize; dim + 1];
        for c in 0..dim {
            comp_offsets[c + 1] = comp_offsets[c] + div_ext[c].n_dofs();
        }
        let n_dofs = comp_offsets[dim];

        // `FiniteElementSpace::UpdateNURBS`: merge the component tables with
        // one offset per component (`Table(*t0, *t1, offset1, ...)`).  All
        // extensions share the (order-independent) span enumeration, so the
        // rows merge element by element.
        let n_elem = ext.n_elements();
        let mut elem_dof = Vec::with_capacity(n_elem);
        for e in 0..n_elem {
            let mut row = Vec::new();
            for c in 0..dim {
                let r = div_ext[c].element_dofs(e);
                row.extend(r.iter().map(|&g| comp_offsets[c] + g));
            }
            elem_dof.push(row);
        }

        Ok(Self { base, div_ext, comp_offsets, elem_dof, dim, n_dofs })
    }

    /// Read a NURBS mesh file and build the space (see [`Self::from_mesh_str`]).
    pub fn from_mesh_file(
        path: impl AsRef<std::path::Path>,
        ref_levels: usize,
        order: usize,
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("NurbsHDivSpace::from_mesh_file: {e}"))?;
        Self::from_mesh_str(&text, ref_levels, order)
    }

    /// `FiniteElementSpace::GetTrueVSize` — the merged DOF count
    /// ("Number of HDiv finite element unknowns" / `dim(R)`).
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// `FiniteElementSpace::GetNE`.
    pub fn n_elements(&self) -> usize {
        self.base.n_elements()
    }

    /// Mesh dimension (2 or 3).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The merged element DOF table (`GetElementDofTable`), `n_elements` rows
    /// of globally numbered DOFs (component-major).
    pub fn element_dofs(&self, e: usize) -> &[usize] {
        &self.elem_dof[e]
    }

    /// The analysis extension (MFEM `FiniteElementSpace::GetNURBSext`).
    pub fn extension(&self) -> &NurbsExtension {
        self.base.extension()
    }

    /// The scalar NURBS space that shares this mesh, extension and element
    /// numbering — MFEM's `W_space` in `nurbs_ex5`
    /// (`NURBSFECollection(order)` on the stolen `NURBSext`).
    pub fn scalar_space(&self) -> &NurbsFESpace {
        &self.base
    }

    /// Component `d`'s divergence extension (`VNURBSext[d]`).
    pub fn component_extension(&self, d: usize) -> &NurbsExtension {
        &self.div_ext[d]
    }

    /// `FiniteElementSpace::GetBdrElementDofs` for this H(div) space — the
    /// `Mode::H_DIV` boundary DOF table of every `VNURBSext[d]`, merged with the
    /// component offsets exactly like the element table
    /// (`bdr_elem_dof = Table(*t0, *t1, offset1, ...)`).  Rows are MFEM's signed
    /// encoding: a negative entry `e` denotes DOF `-1 - e` with the opposite
    /// sign, which is `Vector::AddElementVector`'s convention and what makes the
    /// natural boundary condition of `nurbs_ex5` come out right.
    pub fn boundary_dof_table(&self) -> Vec<Vec<i64>> {
        let parts: Vec<(usize, Vec<Vec<i64>>)> = (0..self.dim)
            .map(|c| {
                (self.comp_offsets[c], self.div_ext[c].boundary_dof_table(BdrDofMode::HDiv))
            })
            .collect();
        merge_component_tables(&parts)
    }

    /// The element's vector FE with `SetIJK` applied (`NURBSExtension::LoadFE`).
    pub fn element_fe(&self, e: usize) -> HDivSpanElement {
        let ext = self.base.extension();
        let patch = ext.element_patch(e);
        let kvs = ext.patch_knot_vectors(patch).expect("element patch");
        let ijk = ext.element_ijk(e);
        match self.dim {
            2 => {
                let mut fe = NurbsHDiv2D::from_knot_vectors(
                    kvs[0].knot_vector().clone(),
                    kvs[1].knot_vector().clone(),
                )
                .expect("NurbsHDiv2D::from_knot_vectors");
                fe.set_ijk([ijk[0], ijk[1]]);
                HDivSpanElement::Two(fe)
            }
            _ => {
                let mut fe = NurbsHDiv3D::from_knot_vectors(
                    kvs[0].knot_vector().clone(),
                    kvs[1].knot_vector().clone(),
                    kvs[2].knot_vector().clone(),
                )
                .expect("NurbsHDiv3D::from_knot_vectors");
                fe.set_ijk(ijk);
                HDivSpanElement::Three(fe)
            }
        }
    }

    /// `NURBS_HDiv*FiniteElement::CalcVShape(Trans, shape)` — the reference
    /// vector basis mapped by the **contravariant Piola** transformation
    /// `J / det(J)` (`shape(i,c) = Σ_k ref(i,k) J(c,k) / Weight()`), the
    /// H(div) counterpart of the H(curl) `J⁻¹ = adj(J)/det(J)`.
    fn phys_vshape(
        &self,
        fe: &HDivSpanElement,
        xi: &[f64],
        geo: &Geometry,
        ref_shape: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        let dim = self.dim;
        let nd = fe.n_dofs();
        ref_shape.resize(nd * dim, 0.0);
        out.resize(nd * dim, 0.0);
        ref_shape.iter_mut().for_each(|v| *v = 0.0);
        out.iter_mut().for_each(|v| *v = 0.0);
        fe.eval_basis_vec(xi, ref_shape);
        let inv_det = 1.0 / geo.det_j;
        for i in 0..nd {
            for c in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    s += ref_shape[i * dim + k] * geo.jac[c][k];
                }
                out[i * dim + c] = s * inv_det;
            }
        }
    }

    /// MFEM `FiniteElementSpace::GetEssentialTrueDofs(ess_bdr)` for the
    /// H(div) NURBS space.
    ///
    /// `Generate{2,3}DBdrElementDofTable` in `Mode::H_DIV` keeps a component
    /// `c`'s DOFs on a boundary entity exactly when the *tangential*
    /// knot-vector order of that entity still differs from the extension's
    /// maximal order: `GetDivExtension(c)` raised only direction `c`, so for
    /// the uniform analysis order `p` that is precisely "direction `c` is the
    /// entity's **normal** direction" — a 2-D edge only carries the DOF block
    /// of the component normal to it, a 3-D face only the block of the
    /// component normal to it (the two tangential blocks are dropped because
    /// `ord0 != ord1`).  The remaining DOFs are all control points of the
    /// boundary entity, so the essential set is the union over the mesh
    /// boundary sides — then `MarkerToList` sorts and de-duplicates.
    pub fn essential_dofs(&self) -> Vec<u32> {
        let dim = self.dim;
        let ext = self.base.extension();
        let mut mark = vec![false; self.n_dofs];

        // One entry per boundary element; a side is shared by all spans along
        // it, so de-duplicate first (`activeBdrElem` enumeration order).
        let mut sides = ext.boundary_sides().to_vec();
        sides.sort_unstable();
        sides.dedup();

        for side in &sides {
            let (patch, dir, low) = (side.patch, side.dir, side.low);
            for c in 0..dim {
                // `Mode::H_DIV`: `add_dofs` survives only for the component
                // whose raised direction is the entity's normal direction.
                if c != dir {
                    continue;
                }
                let cext = &self.div_ext[c];
                let kvs = cext.patch_knot_vectors(patch).expect("element patch");
                let ncp: Vec<usize> = kvs.iter().map(|k| k.ncp()).collect();
                let fixed = if low { 0 } else { ncp[dir] - 1 };
                let ranges: Vec<Vec<usize>> = (0..dim)
                    .map(|d| if d == dir { vec![fixed] } else { (0..ncp[d]).collect() })
                    .collect();
                let total: usize = ranges.iter().map(|v| v.len()).product();
                for n in 0..total {
                    let mut rest = n;
                    let mut multi = vec![0usize; dim];
                    for (d, r) in ranges.iter().enumerate() {
                        multi[d] = r[rest % r.len()];
                        rest /= r.len();
                    }
                    let g = cext.patch_dof(patch, &multi).expect("patch dof");
                    mark[self.comp_offsets[c] + g] = true;
                }
            }
        }
        (0..self.n_dofs)
            .filter(|&d| mark[d])
            .map(|d| d as u32)
            .collect()
    }

    /// `BilinearForm::Assemble` + `Finalize` of the vector mass matrix
    /// `∫ alpha u·v dΩ` (`VectorFEMassIntegrator(alpha)`, quadrature order
    /// `Trans.OrderW() + 2*GetOrder()`).
    pub fn assemble_mass(&self, alpha: f64) -> CsrMatrix<f64> {
        let dim = self.dim;
        let n = self.n_dofs;
        let mut coo = CooMatrix::new(n, n);
        coo.reserve(n * 16);
        let mut ref_shape: Vec<f64> = Vec::new();
        let mut vsh: Vec<f64> = Vec::new();
        let order_w = self.base.mesh_order() * dim - 1;
        for e in 0..self.n_elements() {
            let nd = self.elem_dof[e].len();
            let fe = self.element_fe(e);
            let rule = nurbs_rule(dim, (order_w + 2 * fe.order()) as u8);
            let mut elmat = vec![0.0_f64; nd * nd];
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.base.geometry(e, xi);
                // `CalcVShape(Trans)`: `J/W`; `w = ip.weight * Trans.Weight()`.
                self.phys_vshape(&fe, xi, &geo, &mut ref_shape, &mut vsh);
                let w = rule.weights[q] * geo.det_j;
                for i in 0..nd {
                    for j in 0..nd {
                        let mut s = 0.0;
                        for c in 0..dim {
                            s += vsh[i * dim + c] * vsh[j * dim + c];
                        }
                        elmat[i * nd + j] += w * alpha * s;
                    }
                }
            }
            coo.add_element_matrix(&self.elem_dof[e], &elmat);
        }
        coo.into_csr()
    }

    /// `LinearForm::Assemble` of `(f, v)` (`VectorFEDomainLFIntegrator`,
    /// quadrature order `2*GetOrder()`).
    pub fn assemble_vector_domain_lf(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        let dim = self.dim;
        let mut rhs = vec![0.0_f64; self.n_dofs];
        let mut ref_shape: Vec<f64> = Vec::new();
        let mut vsh: Vec<f64> = Vec::new();
        for e in 0..self.n_elements() {
            let nd = self.elem_dof[e].len();
            let fe = self.element_fe(e);
            let rule = nurbs_rule(dim, (2 * fe.order()) as u8);
            let mut elvec = vec![0.0_f64; nd];
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.base.geometry(e, xi);
                self.phys_vshape(&fe, xi, &geo, &mut ref_shape, &mut vsh);
                let fv = f(&geo.x[..dim]);
                let w = rule.weights[q] * geo.det_j;
                for o in 0..nd {
                    let mut s = 0.0;
                    for c in 0..dim {
                        s += vsh[o * dim + c] * fv[c];
                    }
                    elvec[o] += w * s;
                }
            }
            for (o, &g) in self.elem_dof[e].iter().enumerate() {
                rhs[g] += elvec[o];
            }
        }
        rhs
    }

    /// `LinearForm::Assemble` of `nurbs_ex5`'s natural-boundary term
    /// `∫_Γ g (v·n) ds` (`VectorFEBoundaryFluxLFIntegrator`), on the boundary
    /// elements of the H(div) NURBS space.
    ///
    /// MFEM's integrator (`fem/lininteg.cpp`) evaluates the *boundary* FE
    /// `fes->GetBE(i)` — for `NURBS_HDivFECollection` a `NURBS1DFiniteElement`
    /// in 2-D and a `NURBS2DFiniteElement` in 3-D, i.e. the *scalar* NURBS
    /// element of the analysis extension, whose weights are `1` — on
    /// `mesh->GetBdrElementTransformation(i)` with the default
    /// `oa = 2, ob = 0`, so
    ///
    /// ```text
    ///   elvect_j += ip.weight · g(x_q) · shape_j(ξ_q),   intorder = 2·GetOrder()
    /// ```
    ///
    /// with no `Trans.Weight()` and no explicit normal: the normal component and
    /// the surface measure live in the DOF row's signs, which
    /// [`Self::boundary_dof_table`] produces (`Mode::H_DIV` negates the low side
    /// of each patch direction) and `Vector::AddElementVector` applies as
    /// "`j < 0` ⇒ subtract `elvect[-1-j]`".  The signed weight vector is
    /// therefore uniform `±1` per row and cancels in the rational
    /// normalization, which is why the shape below is the plain B-spline basis.
    pub fn assemble_vector_boundary_flux(&self, g: &dyn Fn(&[f64]) -> f64) -> Vec<f64> {
        let dim = self.dim;
        let ext = self.base.extension();
        let rows = self.boundary_dof_table();
        let elements = self.boundary_element_spans();
        assert_eq!(elements.len(), rows.len(), "boundary element count vs bel_dof rows");
        let mut rhs = vec![0.0_f64; self.n_dofs];
        for (i, &(bp, ref spans)) in elements.iter().enumerate() {
            let side = ext.boundary_sides()[bp];
            let kvs = ext.bdr_patch_knot_vectors(bp);
            let order: Vec<usize> = kvs.iter().map(|k| k.order()).collect();
            assert!(
                order.iter().all(|&o| o == order[0]),
                "NurbsHDivSpace: a boundary patch with unequal knot-vector orders has no \
                 H(div) boundary DOFs (MFEM's `add_dofs` test)"
            );
            let local = ext.bdr_element_span(bp, spans);
            // `IntRules.Get(SEGMENT/SQUARE, oa*GetOrder() + ob)` with the
            // coefficient constructor's `oa = 2, ob = 0`.
            let rule = nurbs_rule(dim - 1, (2 * order[0]) as u8);
            let dof_shape: Vec<usize> = order.iter().map(|o| o + 1).collect();
            let nd: usize = dof_shape.iter().product();
            let row = &rows[i];
            assert_eq!(row.len(), nd, "boundary FE dofs vs bel_dof row");
            let mut elvec = vec![0.0_f64; nd];
            for q in 0..rule.points.len() {
                let xiq = &rule.points[q];
                // The `Order_j + 1` non-vanishing span-local basis values of
                // each boundary reference direction, evaluated at the signed
                // span's own coordinate.
                let mut n1d: Vec<Vec<f64>> = Vec::with_capacity(order.len());
                for (j, &(_, signed)) in local.iter().enumerate() {
                    let (s, xs) = split_signed_span(signed, xiq[j]);
                    let mut nb = vec![0.0; order[j] + 1];
                    knot_span_shape(kvs[j].knot_vector().as_slice(), order[j], s, xs, &mut nb);
                    n1d.push(nb);
                }
                let mut shape = vec![0.0_f64; nd];
                for (o, v) in shape.iter_mut().enumerate() {
                    let mut p = 1.0;
                    for (j, nb) in n1d.iter().enumerate() {
                        p *= nb[multi_index(o, &dof_shape, j)];
                    }
                    *v = p;
                }
                let tang: Vec<(usize, i64, f64)> = local
                    .iter()
                    .enumerate()
                    .map(|(j, &(d, signed))| (d, signed, xiq[j]))
                    .collect();
                let x = self.base.bdr_geometry(side.patch, side.dir, side.low, &tang);
                let w = rule.weights[q] * g(&x[..dim]);
                for (o, &s) in shape.iter().enumerate() {
                    elvec[o] += w * s;
                }
            }
            for (o, &dof) in row.iter().enumerate() {
                let (d, s) = unsign_dof(dof);
                rhs[d] += s as f64 * elvec[o];
            }
        }
        rhs
    }

    /// MFEM's `NURBSExtension` boundary-element enumeration for the H(div)
    /// space: one `(patch boundary entity, knot spans)` pair per mesh boundary
    /// element, in the order `Generate{2,3}DBdrElementDofTable` builds the
    /// `bel_dof` rows (`Generate3DBdrElementDofTable` runs the *second* local
    /// direction outer), with one knot-span index per boundary reference
    /// direction.
    pub fn boundary_element_spans(&self) -> Vec<(usize, Vec<usize>)> {
        let ext = self.base.extension();
        let n_loc = self.dim - 1;
        let mut out = Vec::new();
        for bp in 0..ext.n_bdr_patches() {
            let kvs = ext.bdr_patch_knot_vectors(bp);
            let spans0: Vec<usize> =
                (0..kvs[0].nks()).filter(|&i| kvs[0].is_element(i)).collect();
            let spans1: Vec<usize> = if n_loc == 2 {
                (0..kvs[1].nks()).filter(|&j| kvs[1].is_element(j)).collect()
            } else {
                vec![0]
            };
            for &s1 in &spans1 {
                for &s0 in &spans0 {
                    out.push((bp, if n_loc == 2 { vec![s0, s1] } else { vec![s0] }));
                }
            }
        }
        out
    }

    /// The descriptor of the `i`-th mesh boundary element that
    /// [`Self::assemble_vector_boundary_flux`] uses: `(patch, normal direction,
    /// low side, (patch direction, signed span index) per boundary reference
    /// direction)`.
    pub fn boundary_element(&self, i: usize) -> (usize, usize, bool, Vec<(usize, i64)>) {
        let ext = self.base.extension();
        let (bp, spans) = &self.boundary_element_spans()[i];
        let side = ext.boundary_sides()[*bp];
        (side.patch, side.dir, side.low, ext.bdr_element_span(*bp, spans))
    }

    /// `MixedBilinearForm(R_space, q_space).Assemble() + Finalize()` with
    /// `VectorFEDivergenceIntegrator` — MFEM's `B` of `nurbs_ex5`, i.e.
    /// `B(q_i, u_j) = Σ_q ip.weight * q_i(ξ_q) * div_ref(u_j)(ξ_q)`.
    ///
    /// Note the integrator uses the **reference** `CalcDivShape(ip)` and the
    /// test FE's `CalcPhysShape`, and multiplies only by `ip.weight` — no
    /// `Trans.Weight()` factor (MFEM's `VectorFEDivergenceIntegrator::
    /// AssembleElementMatrix2`, quadrature order `trial.order + test.order - 1`).
    /// This port reproduces that verbatim; it is what makes the C++ binary's
    /// iteration block reproducible.
    ///
    /// The returned matrix has `q_space.n_dofs()` rows and `self.n_dofs()`
    /// columns (the `MixedBilinearForm(trial_fes, test_fes)` orientation).
    pub fn assemble_mixed_divergence(&self, q_space: &NurbsFESpace) -> CsrMatrix<f64> {
        let dim = self.dim;
        let mut coo = CooMatrix::new(q_space.n_dofs(), self.n_dofs);
        coo.reserve(self.n_dofs() * 16);
        assert_eq!(q_space.n_elements(), self.n_elements(), "mixed form: element counts");
        for e in 0..self.n_elements() {
            let fe = self.element_fe(e);
            let nd_trial = self.elem_dof[e].len();
            let nd_test = q_space.element_dofs(e).len();
            let test_fe = q_space.element_fe(e);
            let order = fe.order() + test_fe.order() - 1;
            let rule = nurbs_rule(dim, order as u8);
            let mut elmat = vec![0.0_f64; nd_test * nd_trial];
            let mut div_ref = vec![0.0_f64; nd_trial];
            let mut shape = vec![0.0_f64; nd_test];
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                fe.eval_div(xi, &mut div_ref);
                test_fe.shape(xi, &mut shape);
                let w = rule.weights[q];
                for i in 0..nd_test {
                    for j in 0..nd_trial {
                        elmat[i * nd_trial + j] += w * shape[i] * div_ref[j];
                    }
                }
            }
            let test_dofs: Vec<usize> = q_space.element_dofs(e).to_vec();
            for i in 0..nd_test {
                for j in 0..nd_trial {
                    coo.add(test_dofs[i], self.elem_dof[e][j], elmat[i * nd_trial + j]);
                }
            }
        }
        coo.into_csr()
    }

    /// `GridFunction::ProjectCoefficient(VectorCoefficient&)` — MFEM's
    /// **default** dispatch for a NURBS space, `ProjectCoefficientElementL2`.
    pub fn project_coefficient(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        self.project_coefficient_element_l2(f)
    }

    /// `GridFunction::ProjectCoefficientElementL2_(VectorCoefficient&, x, Va)`
    /// followed by `(*this) /= Va` (`GridFunction::ProjectCoefficientElementL2`)
    /// — the NURBS branch, MFEM's default projection for a
    /// `NURBS_HDivFECollection` space (`nurbs_ex24`, `ProjectType::DEFAULT`).
    ///
    /// Structurally identical to [`NurbsHCurlSpace::project_coefficient_element_l2`]
    /// (see that method for the five steps), with the H(div)
    /// `CalcPhysVShape` (`J/W`) in place of the H(curl) one.
    pub fn project_coefficient_element_l2(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        let dim = self.dim;
        let mut x = vec![0.0_f64; self.n_dofs];
        let mut va = vec![0.0_f64; self.n_dofs];
        let mut shape2: Vec<f64> = Vec::new();
        let mut ref_shape: Vec<f64> = Vec::new();
        let mut shape: Vec<f64> = Vec::new();
        for e in 0..self.n_elements() {
            let fe = self.element_fe(e);
            let nd = self.elem_dof[e].len();
            let p = fe.order();
            let dof2 = (p + 1).pow(dim as u32);
            let rule = nurbs_rule(dim, (2 * p + 1) as u8);
            let l2_nodes = gauss_legendre_01(p + 1).0;

            shape2.clear();
            shape2.resize(dof2, 0.0);
            ref_shape.clear();
            ref_shape.resize(nd * dim, 0.0);
            shape.clear();
            shape.resize(nd * dim, 0.0);

            let nblk = dof2 * dim;
            let mut elvect = vec![0.0_f64; nblk];
            let mut elmat = vec![0.0_f64; nblk * nblk];
            let mut elwght = vec![0.0_f64; nd];
            let mut partelmat = vec![0.0_f64; dof2 * dof2];

            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.base.geometry(e, xi);
                let w = rule.weights[q] * geo.det_j;
                let val: Vec<f64> = f(&geo.x[..dim]).iter().map(|v| v * w).collect();
                l2_gl_shape(dim, &l2_nodes, xi, &mut shape2);
                self.phys_vshape(&fe, xi, &geo, &mut ref_shape, &mut shape);

                for c in 0..dim {
                    for s in 0..dof2 {
                        elvect[dof2 * c + s] += val[c] * shape2[s];
                    }
                }
                for r in 0..dof2 {
                    for s in 0..dof2 {
                        partelmat[r * dof2 + s] = shape2[r] * shape2[s] * w;
                    }
                }
                for c in 0..dim {
                    let off = dof2 * c;
                    for r in 0..dof2 {
                        for s in 0..dof2 {
                            elmat[(off + r) * nblk + off + s] += partelmat[r * dof2 + s];
                        }
                    }
                }
                for j in 0..nd {
                    let mut s2 = 0.0;
                    for c in 0..dim {
                        s2 += shape[j * dim + c] * shape[j * dim + c];
                    }
                    elwght[j] += w * s2.sqrt();
                }
            }

            if !dense_lu_solve(&mut elmat, nblk, &mut elvect, 1e-9) {
                panic!("NurbsHDivSpace::project_coefficient_element_l2: singular L2 mass matrix");
            }

            // `el2.Project(el, tr, I)`, I(d*dof2 + k, j) = vshape_j,d(node k).
            let mut imat = vec![0.0_f64; nblk * nd];
            for k in 0..dof2 {
                let node: Vec<f64> = if dim == 2 {
                    vec![l2_nodes[k % (p + 1)], l2_nodes[k / (p + 1)]]
                } else {
                    vec![
                        l2_nodes[k % (p + 1)],
                        l2_nodes[(k / (p + 1)) % (p + 1)],
                        l2_nodes[k / ((p + 1) * (p + 1))],
                    ]
                };
                let geo = self.base.geometry(e, &node);
                self.phys_vshape(&fe, &node, &geo, &mut ref_shape, &mut shape);
                for j in 0..nd {
                    for c in 0..dim {
                        imat[(c * dof2 + k) * nd + j] = shape[j * dim + c];
                    }
                }
            }

            let mut vec = vec![0.0_f64; nd];
            let mut mat = vec![0.0_f64; nd * nd];
            for j in 0..nd {
                let mut s = 0.0;
                for r in 0..nblk {
                    s += imat[r * nd + j] * elvect[r];
                }
                vec[j] = s;
                for j2 in 0..nd {
                    let mut t = 0.0;
                    for r in 0..nblk {
                        t += imat[r * nd + j] * imat[r * nd + j2];
                    }
                    mat[j * nd + j2] = t;
                }
            }
            if !dense_lu_solve(&mut mat, nd, &mut vec, 1e-24) {
                panic!("NurbsHDivSpace::project_coefficient_element_l2: singular IᵀI");
            }

            for (j, &g) in self.elem_dof[e].iter().enumerate() {
                x[g] += vec[j] * elwght[j];
                va[g] += elwght[j];
            }
        }
        for i in 0..self.n_dofs {
            x[i] /= va[i];
        }
        x
    }

    /// `GridFunction::ComputeL2Error(VectorCoefficient, irs)` — per element,
    /// the vector values are `CalcVShape(Trans)ᵀ x_el` and each point
    /// contributes `ip.weight * Weight() * ‖u_h − u‖²`.
    ///
    /// `order_quad` is MFEM's `IntRules.Get(Geometry::SQUARE/CUBE, order_quad)`
    /// order — `nurbs_ex5` passes `max(2, 2*order+1)` explicitly (it fills the
    /// whole `irs[]` array), which is not the `2*GetOrder() + 3` that
    /// `ComputeL2Error(VectorCoefficient)` uses when `irs == NULL`.
    pub fn compute_l2_error(
        &self,
        x: &[f64],
        f: &dyn Fn(&[f64]) -> Vec<f64>,
        order_quad: u8,
    ) -> f64 {
        let dim = self.dim;
        let mut error = 0.0_f64;
        let mut ref_shape: Vec<f64> = Vec::new();
        let mut vsh: Vec<f64> = Vec::new();
        for e in 0..self.n_elements() {
            let nd = self.elem_dof[e].len();
            let fe = self.element_fe(e);
            let rule = nurbs_rule(dim, order_quad);
            let mut elem_error = 0.0_f64;
            for q in 0..rule.points.len() {
                let xi = &rule.points[q];
                let geo = self.base.geometry(e, xi);
                self.phys_vshape(&fe, xi, &geo, &mut ref_shape, &mut vsh);
                let fv = f(&geo.x[..dim]);
                let mut n2 = 0.0_f64;
                for c in 0..dim {
                    let mut val_c = 0.0_f64;
                    for o in 0..nd {
                        val_c += vsh[o * dim + c] * x[self.elem_dof[e][o]];
                    }
                    let d = val_c - fv[c];
                    n2 += d * d;
                }
                elem_error += rule.weights[q] * geo.det_j * n2;
            }
            error += elem_error.abs();
        }
        error.sqrt()
    }

    /// `ComputeLpNorm(2., VectorCoefficient, mesh, irs)` — the *exact*
    /// function's `L²` norm over the mesh, with the same element geometry and
    /// the quadrature order the caller passes (`order_quad`).
    pub fn compute_exact_l2_norm(&self, f: &dyn Fn(&[f64]) -> Vec<f64>, order_quad: u8) -> f64 {
        self.base.compute_exact_l2_norm_vec(f, order_quad)
    }
}
