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
//! * Dimensions 2 and 3 (1-D `segment-nurbs.mesh` meshes are rejected).
//! * [`NurbsFESpace::boundary_dofs`] implements MFEM's `ess_bdr = 1` case
//!   (every mesh boundary attribute essential); the general boundary-attribute
//!   map needs `NURBSExtension::GenerateBdrElementDofTable`, which is not ported.
//! * Scalar (H¹) space: the vector `NURBS_HDiv`/`NURBS_HCurl` paths are outside
//!   the delivered scope.

use fem_element::nurbs_fe_collection::{
    knot_span_dparam, knot_span_shape, NurbsScalar2D, NurbsScalar3D,
};
use fem_element::quadrature::{gauss_legendre_01, quad_rule_01};
use fem_linalg::{CooMatrix, CsrMatrix};

use crate::nurbs_extension::NurbsExtension;

/// A quadrature rule on `[0,1]^dim` with MFEM's tensor ordering (the first
/// direction varies fastest, so the summation order matches MFEM's).
#[derive(Debug, Clone)]
pub struct Rule {
    pub points: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
}

/// `IntRules.Get(Geometry::SQUARE/CUBE, order)` — Gauss-Legendre with
/// `(order + 2)/2` points per direction on `[0,1]^dim` (`[0,1]²` reuses
/// [`fem_element::quadrature::quad_rule_01`]).
pub fn nurbs_rule(dim: usize, order: u8) -> Rule {
    if dim == 2 {
        let r = quad_rule_01(order);
        return Rule { points: r.points, weights: r.weights };
    }
    let n = ((order as usize + 2) / 2).max(1);
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

/// The scalar NURBS element of one knot span: MFEM's `NURBS2DFiniteElement` /
/// `NURBS3DFiniteElement` bound to a patch's knot vectors, one span index
/// (`NURBSFiniteElement::ijk`) and one weight per local DOF (`LoadFE`).
#[derive(Debug, Clone)]
pub enum SpanElement {
    /// MFEM `NURBS2DFiniteElement`.
    Two(NurbsScalar2D),
    /// MFEM `NURBS3DFiniteElement`.
    Three(NurbsScalar3D),
}

impl SpanElement {
    /// `FiniteElement::GetDim`.
    pub fn dim(&self) -> usize {
        match self {
            SpanElement::Two(_) => 2,
            SpanElement::Three(_) => 3,
        }
    }

    /// `FiniteElement::GetDof`.
    pub fn n_dofs(&self) -> usize {
        match self {
            SpanElement::Two(fe) => fe.n_dofs(),
            SpanElement::Three(fe) => fe.n_dofs(),
        }
    }

    /// `FiniteElement::GetOrder` (`max(orders)` after `SetOrder`).
    pub fn order(&self) -> usize {
        match self {
            SpanElement::Two(fe) => fe.order(),
            SpanElement::Three(fe) => fe.order(),
        }
    }

    /// `NURBSFiniteElement::SetIJK`.
    pub fn set_ijk(&mut self, ijk: &[usize; 3]) {
        match self {
            SpanElement::Two(fe) => fe.set_ijk([ijk[0], ijk[1]]),
            SpanElement::Three(fe) => fe.set_ijk(*ijk),
        }
    }

    /// `NURBS2D/3DFiniteElement::CalcShape`.
    pub fn shape(&self, xi: &[f64], values: &mut [f64]) {
        match self {
            SpanElement::Two(fe) => fe.calc_shape(xi, values),
            SpanElement::Three(fe) => fe.calc_shape(xi, values),
        }
    }

    /// `NURBS2D/3DFiniteElement::CalcDShape` (`dim` entries per DOF).
    pub fn grad(&self, xi: &[f64], grads: &mut [f64]) {
        match self {
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
    if dim == 2 {
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

/// `DenseMatrix::Det` (MFEM `linalg/densemat.cpp`).
fn det(j: &[[f64; 3]; 3], dim: usize) -> f64 {
    if dim == 2 {
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
        if dim < 2 || dim > 3 {
            return Err(format!(
                "NurbsFESpace: dimension {dim} is not supported (only 2 and 3)"
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

    /// MFEM `FiniteElementSpace::GetEssentialTrueDofs(ess_bdr = 1)`.
    ///
    /// A clamped NURBS patch's boundary is exactly the set of control points
    /// with an extreme parameter index (`0` or `NCP - 1`) in a direction whose
    /// boundary side is a mesh boundary, so the essential DOFs are the union
    /// over the *mesh boundary elements* of the control points of the
    /// patch-boundary entity each of them lies on
    /// ([`NurbsExtension::boundary_sides`]).
    ///
    /// `ess_bdr = 1` marks every mesh boundary attribute essential (the case
    /// `nurbs_ex1` uses); a partial `ess_bdr` needs
    /// `NURBSExtension::GenerateBdrElementDofTable`'s attribute per row, which
    /// is not ported.
    pub fn boundary_dofs(&self) -> Vec<u32> {
        let dim = self.dim;
        let mut mark = vec![false; self.n_dofs()];
        // A patch-boundary entity can be shared by several boundary elements
        // (one per knot span along it); each side is marked once.
        let mut sides: Vec<(usize, usize, bool)> = self.ext.boundary_sides().to_vec();
        sides.sort_unstable();
        sides.dedup();

        for &(patch, dir, low) in &sides {
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
