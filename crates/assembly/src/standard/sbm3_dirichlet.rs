//! Shifted Boundary Method (SBM) Dirichlet face integrators — 1:1 serial port
//! of the MFEM `miniapps/shifted` second-generation SBM kernels
//! (`sbm_solver.{hpp,cpp}`): [`Sbm3DirichletIntegrator`] (MFEM
//! `SBM2DirichletIntegrator`) and [`Sbm3DirichletLFIntegrator`] (MFEM
//! `SBM2DirichletLFIntegrator`).
//!
//! The true boundary is given implicitly by a level set.  Elements are marked
//! inside/outside/cut by [`crate::dist_solver::ShiftedFaceMarker`]; the faces
//! between inside and cut elements form the *surrogate boundary* Σ̂.  On each
//! surrogate face the Dirichlet data is imposed with a Nitsche-type form where
//! the boundary value is Taylor-shifted from the true boundary point
//! `x̂ = x̂̂ + D` (D = distance vector to the true boundary, obtained e.g. from
//! [`crate::dist_solver::HeatDistanceSolver::compute_vector_distance`]):
//!
//! ```text
//! A(u,w) = -⟨∇u·n, w⟩ - ⟨u + ∇u·d + h.o.t., ∇w·n⟩
//!          + ⟨α h⁻¹ (u + ∇u·d + h.o.t.), w + ∇w·d + h.o.t.⟩
//! L(w)   = -⟨u_D, ∇w·n⟩ + ⟨α h⁻¹ u_D, w + ∇w·d + h.o.t.⟩
//! ```
//!
//! where `n` is the surrogate-boundary normal (with the surface measure folded
//! in, MFEM `CalcOrtho`), `h⁻¹ = |n|²/det(J)` on the face point and `u_D` is
//! evaluated at the shifted point `x + D` (MFEM
//! `ShiftedFunctionCoefficient::Eval(T, ip, D)`).  The higher-order Taylor
//! terms (`ho_terms`, MFEM `-ho`) use the discrete second-derivative operators
//! `D_k D_c` (MFEM `FiniteElement::ProjectGrad` products) exactly as in the
//! C++ implementation.
//!
//! The C++ integrators are `ParMesh` interior/boundary-face integrators; the
//! serial port exposes the same mathematics as whole-form assemblers
//! ([`Sbm3DirichletIntegrator::assemble_bilinear`] /
//! [`Sbm3DirichletLFIntegrator::assemble_linear`]) plus the public face
//! enumeration ([`surrogate_face_list`]) used to drive them.

use std::collections::HashMap;

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use nalgebra::DMatrix;

use crate::assembler::{geo_ref_elem, ref_elem_vol_for_space};

/// SBM element classification (MFEM `ShiftedFaceMarker::SBElementType`), local
/// re-declaration to keep this module self-contained: 0 = inside, 1 = outside,
/// ≥ 2 = cut (by level set `value − 2`).
pub(crate) const SB_INSIDE: i32 = 0;
pub(crate) const SB_OUTSIDE: i32 = 1;
pub(crate) const SB_CUT: i32 = 2;

// ─── Surrogate face enumeration ──────────────────────────────────────────────

/// One surrogate-boundary face: the *active* (surrogate-domain) element and
/// the face nodes (MFEM: the `FaceElementTransformations` of an interior face
/// between an inside and a cut element, or a boundary face of a cut element
/// when cut cells are included).
#[derive(Debug, Clone)]
pub struct SbmFace {
    /// Element that belongs to the surrogate domain and receives the whole
    /// face contribution (MFEM `Trans.Elem1No` when `elem1f`, else `Elem2No`).
    pub elem: u32,
    /// Face nodes (geometry).
    pub nodes: Vec<u32>,
}

/// Local face node tables (MFEM `Mesh::local_face_nodes` equivalent), keyed by
/// element node count and dimension.  Only the element types used by the
/// shifted miniapps are supported (straight-sided tri/quad/tet/hex).
fn local_face_nodes(et: ElementType) -> &'static [&'static [usize]] {
    match et {
        ElementType::Tri3 => &[&[0, 1], &[1, 2], &[2, 0]],
        ElementType::Quad4 => &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]],
        ElementType::Tet4 => &[&[1, 2, 3], &[0, 2, 3], &[0, 1, 3], &[0, 1, 2]],
        ElementType::Hex8 => &[
            &[0, 1, 2, 3],
            &[4, 5, 6, 7],
            &[0, 1, 5, 4],
            &[1, 2, 6, 5],
            &[2, 3, 7, 6],
            &[3, 0, 4, 7],
        ],
        other => panic!(
            "sbm3: unsupported element type {other:?} for surrogate face enumeration \
             (supported: Tri3, Quad4, Tet4, Hex8)"
        ),
    }
}

/// Enumerate all surrogate-boundary faces (MFEM `ShiftedFaceMarker` semantics,
/// mirrored from the face-selection logic of `SBM2DirichletIntegrator::
/// AssembleFaceMatrix`):
///
/// * `include_cut_cell == false`: interior faces between an INSIDE element and
///   a CUT (or CUT+n) element; the active element is the INSIDE one.
/// * `include_cut_cell == true`: interior faces between a CUT (or CUT+n)
///   element and an OUTSIDE element (active = the cut one), plus boundary
///   faces whose owner element is cut.
pub(crate) fn surrogate_face_list<M: MeshTopology>(
    mesh: &M,
    elem_marker: &[i32],
    include_cut_cell: bool,
) -> Vec<SbmFace> {
    let dim = mesh.topological_dim() as usize;
    let is_cut = |m: i32| m >= SB_CUT;

    // All faces of the mesh by node-key matching (interior faces shared by two
    // elements; the leftovers are the mesh boundary, already stored in the
    // mesh as boundary faces).
    let mut face_map: HashMap<Vec<u32>, (u32, Vec<u32>)> = HashMap::new();
    let mut faces = Vec::new();

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        for lf in local_face_nodes(mesh.element_type(e)) {
            let fnodes: Vec<u32> = lf.iter().map(|&k| nodes[k]).collect();
            let mut key = fnodes.clone();
            key.sort_unstable();
            match face_map.remove(&key) {
                None => {
                    face_map.insert(key, (e, fnodes));
                }
                Some((other, _)) => {
                    // Interior face: select it if the marker pair matches.
                    let (m1, m2) = (elem_marker[other as usize], elem_marker[e as usize]);
                    let selected = if include_cut_cell {
                        (is_cut(m1) && m2 == SB_OUTSIDE) || (m1 == SB_OUTSIDE && is_cut(m2))
                    } else {
                        (m1 == SB_INSIDE && is_cut(m2)) || (is_cut(m1) && m2 == SB_INSIDE)
                    };
                    if selected {
                        let elem = if include_cut_cell {
                            if is_cut(m1) {
                                other
                            } else {
                                e
                            }
                        } else if m1 == SB_INSIDE {
                            other
                        } else {
                            e
                        };
                        faces.push(SbmFace { elem, nodes: fnodes });
                    }
                }
            }
        }
    }

    // Boundary faces modelled as SBM faces (MFEM: `AddBdrFaceIntegrator` with
    // the `ess_shift_bdr` marker; only cut elements qualify).
    if include_cut_cell {
        debug_assert_eq!(dim, mesh.dim() as usize);
        for f in 0..mesh.n_boundary_faces() as u32 {
            let (e1, _e2) = mesh.face_elements(f);
            if is_cut(elem_marker[e1 as usize]) {
                faces.push(SbmFace {
                    elem: e1,
                    nodes: mesh.face_nodes(f).to_vec(),
                });
            }
        }
    }

    faces
}

// ─── Element / face geometry (MFEM ElementTransformation + IntRules) ─────────

/// Geometry map of one element: reference basis (of the geometry order),
/// physical node coordinates, and Newton inversion for face-point location
/// (serial equivalent of `FaceElementTransformations` point transforms).
pub(crate) struct ElemGeometry<'a, M: MeshTopology> {
    mesh: &'a M,
    elem: u32,
    re: Box<dyn fem_element::ReferenceElement>,
    nodes: Vec<u32>,
    dim: usize,
}

impl<'a, M: MeshTopology> ElemGeometry<'a, M> {
    pub(crate) fn new(mesh: &'a M, elem: u32) -> Self {
        let dim = mesh.topological_dim() as usize;
        // Same dispatch as `assembler::elem_geometry`: the geometry element
        // when present (curved meshes), otherwise the reference element of the
        // geometry order (P1/Q1 for the straight-sided miniapp meshes).
        let re = match geo_ref_elem(mesh, elem) {
            Some(g) => g,
            None => {
                let go = mesh.geom_order().max(1);
                mesh.element_type(elem).ref_elem(go)
            }
        };
        let nodes = mesh.geometry_nodes(elem).to_vec();
        ElemGeometry { mesh, elem, re, nodes, dim }
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn n_dofs(&self) -> usize {
        self.re.n_dofs()
    }

    /// Map to physical coordinates.
    pub(crate) fn map(&self, xi: &[f64]) -> Vec<f64> {
        let nd = self.re.n_dofs();
        let mut phi = vec![0.0_f64; nd];
        self.re.eval_basis(xi, &mut phi);
        let mut x = vec![0.0_f64; self.dim];
        for (k, &n) in self.nodes.iter().enumerate() {
            let xk = self.mesh.node_coords(n);
            for (xi_, xij) in x.iter_mut().zip(xk.iter()) {
                *xi_ += phi[k] * xij;
            }
        }
        x
    }

    /// Jacobian `J`, its determinant and the inverse-transpose `J^{-T}` at `xi`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn jacobian(&self, xi: &[f64]) -> (DMatrix<f64>, f64, DMatrix<f64>) {
        let nd = self.re.n_dofs();
        let d = self.dim;
        let mut grad = vec![0.0_f64; nd * d];
        self.re.eval_grad_basis(xi, &mut grad);
        let mut j = DMatrix::<f64>::zeros(d, d);
        for (k, &n) in self.nodes.iter().enumerate() {
            let xk = self.mesh.node_coords(n);
            for (i, xij) in xk.iter().enumerate().take(d) {
                for c in 0..d {
                    j[(i, c)] += xij * grad[k * d + c];
                }
            }
        }
        let det = match d {
            1 => j[(0, 0)],
            2 => j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)],
            _ => j.determinant(),
        };
        let jit = j
            .clone()
            .try_inverse()
            .unwrap_or_else(|| panic!("sbm3: degenerate element {}", self.elem))
            .transpose();
        (j, det, jit)
    }

    /// Reference coordinates of the physical point `x` (MFEM
    /// `FaceElementTransformations` location of the face point inside the
    /// element): Newton iteration on the geometry map.
    pub(crate) fn invert(&self, x: &[f64]) -> Vec<f64> {
        let d = self.dim;
        // Seed candidates adapt to the reference-domain convention of the
        // element family: simplex bases live on [0,1]^d (positive corner) and
        // tensor-product bases (quad/hex) on [-1,1]^d (element centre 0).
        let tensor = self.re.dof_coords()[0].iter().any(|&c| c < 0.0);
        let seeds: Vec<Vec<f64>> = if tensor {
            let mut s = vec![0.0_f64; d];
            let mut corners = vec![s.clone()];
            s.iter_mut().for_each(|c| *c = 0.5);
            corners.push(s);
            corners
        } else {
            let mut s = vec![1.0 / (d as f64 + 1.0); d];
            let mut corners = vec![s.clone()];
            s.iter_mut().for_each(|c| *c = 0.25);
            corners.push(s);
            corners
        };
        for seed in &seeds {
            let mut xi = seed.clone();
            if self.newton(x, &mut xi) {
                return xi;
            }
        }
        panic!("sbm3: Newton inversion failed on element {}", self.elem);
    }

    fn newton(&self, x: &[f64], xi: &mut Vec<f64>) -> bool {
        let d = self.dim;
        for _ in 0..50 {
            let xc = self.map(xi);
            let mut r2 = 0.0_f64;
            let mut r = vec![0.0_f64; d];
            for i in 0..d {
                r[i] = x[i] - xc[i];
                r2 += r[i] * r[i];
            }
            if r2 < 1e-28 {
                return true;
            }
            let (_j, _det, jit) = self.jacobian(xi);
            // dx = J^{-T} r?? — J: x_i += J[i][c]·dxi_c → dxi = J^{-1} r;
            // jit = J^{-T} → J^{-1} = jit^T.
            for i in 0..d {
                let mut s = 0.0_f64;
                for c in 0..d {
                    s += jit[(c, i)] * r[c];
                }
                xi[i] += s;
            }
        }
        let xc = self.map(xi);
        r_norm(x, &xc) < 1e-11
    }
}

fn r_norm(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt()
}

/// MFEM `CalcOrtho(J, nor)`: face normal scaled by the surface measure —
/// 2-D: `(J[1], -J[0])` for the segment tangent `J`; 3-D: `t₀ × t₁`.
pub(crate) fn calc_ortho(j: &DMatrix<f64>, dim: usize) -> Vec<f64> {
    if dim == 2 {
        // 2×1 Jacobian (column) stored as d×1 DMatrix.
        vec![j[(1, 0)], -j[(0, 0)]]
    } else {
        let t0 = [j[(0, 0)], j[(1, 0)], j[(2, 0)]];
        let t1 = [j[(0, 1)], j[(1, 1)], j[(2, 1)]];
        vec![
            t0[1] * t1[2] - t0[2] * t1[1],
            t0[2] * t1[0] - t0[0] * t1[2],
            t0[0] * t1[1] - t0[1] * t1[0],
        ]
    }
}

/// Reference element + quadrature of a surrogate face (MFEM
/// `IntRules.Get(face geometry, 4·order)`).
pub(crate) struct FaceGeometry {
    pub(crate) re: Box<dyn fem_element::ReferenceElement>,
    pub(crate) fdim: usize,
}

impl FaceGeometry {
    pub(crate) fn new(n_face_nodes: usize, topological_dim: usize) -> Self {
        let (et, fdim) = match (topological_dim, n_face_nodes) {
            (2, 2) => (ElementType::Line2, 1),
            (3, 3) => (ElementType::Tri3, 2),
            (3, 4) => (ElementType::Quad4, 2),
            other => panic!("sbm3: unsupported face geometry {other:?}"),
        };
        FaceGeometry { re: et.ref_elem(1), fdim }
    }

    /// Physical coordinates of a face point.
    pub(crate) fn map<M: MeshTopology>(&self, mesh: &M, nodes: &[u32], xi: &[f64]) -> Vec<f64> {
        let nd = self.re.n_dofs();
        let mut phi = vec![0.0_f64; nd];
        self.re.eval_basis(xi, &mut phi);
        let dim = self.fdim + 1;
        let mut x = vec![0.0_f64; dim];
        for (k, &n) in nodes.iter().enumerate() {
            let xk = mesh.node_coords(n);
            for (xi_, xij) in x.iter_mut().zip(xk.iter()) {
                *xi_ += phi[k] * xij;
            }
        }
        x
    }

    /// Face Jacobian (dim × fdim).
    pub(crate) fn jacobian<M: MeshTopology>(
        &self,
        mesh: &M,
        nodes: &[u32],
        xi: &[f64],
    ) -> DMatrix<f64> {
        let nd = self.re.n_dofs();
        let fdim = self.fdim;
        let dim = fdim + 1;
        let mut grad = vec![0.0_f64; nd * fdim];
        self.re.eval_grad_basis(xi, &mut grad);
        let mut j = DMatrix::<f64>::zeros(dim, fdim);
        for (k, &n) in nodes.iter().enumerate() {
            let xk = mesh.node_coords(n);
            for (i, xij) in xk.iter().enumerate().take(dim) {
                for c in 0..fdim {
                    j[(i, c)] += xij * grad[k * fdim + c];
                }
            }
        }
        j
    }
}

// ─── Distance-vector field (MFEM VectorGridFunctionCoefficient) ─────────────

/// Vector distance field with component-major (MFEM byNODES) DOF layout:
/// `dist[c * n_scalar + s]`, sharing the scalar space DOF numbering.
pub(crate) struct DistVec<'a> {
    pub(crate) dofs: &'a [f64],
    pub(crate) n_scalar: usize,
}

impl DistVec<'_> {
    /// Evaluate at reference point `xi` of `elem` (MFEM `vD->Eval(D, Tr, ip)`).
    pub(crate) fn eval<S: FESpace>(&self, space: &S, elem: u32, xi: &[f64]) -> Vec<f64> {
        let dim = space.mesh().topological_dim() as usize;
        let re = ref_elem_vol_for_space(space, space.mesh().element_type(elem), space.element_order(elem));
        let nd = re.n_dofs();
        let mut phi = vec![0.0_f64; nd];
        re.eval_basis(xi, &mut phi);
        let edofs = space.element_dofs(elem);
        let mut d = vec![0.0_f64; dim];
        for (k, &dof) in edofs.iter().enumerate() {
            for (c, dc) in d.iter_mut().enumerate() {
                *dc += self.dofs[c * self.n_scalar + dof as usize] * phi[k];
            }
        }
        d
    }
}

// ─── Higher-order Taylor machinery (MFEM ProjectGrad products) ───────────────

/// Discrete gradient-at-DOF matrices `D_c[m][j] = ∂φ_j/∂x_c(dof m)`
/// (MFEM `el.ProjectGrad(el, T, grad_phys)` for the straight-sided elements
/// of the shifted miniapps, where the projection interpolates the physical
/// gradient at the element DOF points).
pub(crate) fn discrete_grad_matrices<M: MeshTopology>(
    geo: &ElemGeometry<'_, M>,
    space_re: &dyn fem_element::ReferenceElement,
) -> Vec<DMatrix<f64>> {
    let d = geo.dim();
    let nd = space_re.n_dofs();
    let mut out: Vec<DMatrix<f64>> = (0..d).map(|_| DMatrix::zeros(nd, nd)).collect();
    for (m, xi) in space_re.dof_coords().iter().enumerate() {
        let (_j, _det, jit) = geo.jacobian(xi);
        let mut gref = vec![0.0_f64; nd * d];
        space_re.eval_grad_basis(xi, &mut gref);
        for jj in 0..nd {
            for c in 0..d {
                let mut s = 0.0_f64;
                for r in 0..d {
                    s += jit[(c, r)] * gref[jj * d + r];
                }
                out[c][(m, jj)] = s;
            }
        }
    }
    out
}

/// `nd × (nd·dim)`: column `j·dim + c` is `D_c[m][j]` (MFEM reshapes the
/// `ProjectGrad` output to `(ndof, ndof·dim)` in-place).
fn grad_work_matrix(dmats: &[DMatrix<f64>], nd: usize, d: usize) -> DMatrix<f64> {
    let mut g = DMatrix::<f64>::zeros(nd, nd * d);
    for c in 0..d {
        for m in 0..nd {
            for k in 0..nd {
                g[(m, k * d + c)] = dmats[c][(m, k)];
            }
        }
    }
    g
}

/// Higher-order derivative tensors `dkphi_dxk[i]` (flat row-major: row `m`,
/// column `j·(sz1·dim) + k·sz1 + loc`) — mechanical port of the MFEM loop
/// building `D_k · D_{...}` products for the Taylor terms.
pub(crate) fn dkphi_dxk(dmats: &[DMatrix<f64>], nd: usize, d: usize, nterms: usize) -> Vec<DMatrix<f64>> {
    let gw = grad_work_matrix(dmats, nd, d);
    let mut out: Vec<DMatrix<f64>> = Vec::with_capacity(nterms);
    for i in 0..nterms {
        let sz1 = d.pow(i as u32 + 1);
        let loc = sz1;
        let tot = loc * d;
        // Current input block: i == 0 → grad_phys_work (nd × nd·sz1);
        // i > 0 → previous dkphi (nd × nd·sz1) re-blocked (already blocked).
        let input: &DMatrix<f64> = if i == 0 { &gw } else { out.last().expect("prev dkphi") };
        let mut blk = DMatrix::<f64>::zeros(nd, nd * tot);
        for k in 0..d {
            // grad_work = D_k · input  (nd × nd·sz1)
            let dk = &dmats[k];
            let mut gw_k = DMatrix::<f64>::zeros(nd, nd * loc);
            for m in 0..nd {
                for col in 0..nd * loc {
                    let j = col / loc;
                    let dd = col % loc;
                    let mut s = 0.0_f64;
                    for p in 0..nd {
                        s += dk[(m, p)] * input[(p, j * loc + dd)];
                    }
                    gw_k[(m, col)] = s;
                }
            }
            // Re-block columns: dof-major (j, k, d) layout.
            for j in 0..nd {
                for dd in 0..loc {
                    for m in 0..nd {
                        blk[(m, j * tot + k * loc + dd)] = gw_k[(m, j * loc + dd)];
                    }
                }
            }
        }
        out.push(blk);
    }
    out
}

/// MFEM `Factorial` vector for the Dirichlet integrators:
/// `F(0) = 2, F(i) = F(i-1)·(i+2)` → `F(i) = (i+2)!`.
fn factorial_dirichlet(nterms: usize) -> Vec<f64> {
    let mut f = Vec::with_capacity(nterms);
    if nterms > 0 {
        f.push(2.0);
        for i in 1..nterms {
            f.push(f[i - 1] * (i as f64 + 2.0));
        }
    }
    f
}

/// Contract every `dim`-block of `v` with `dir`: `out[c] = Σ_r v[c·dim+r]·dir[r]`
/// (MFEM `DenseMatrix(dim, nd·sz).MultTranspose(dir, out)` on the flat
/// column-major reinterpretation of the contraction vector).
fn contract(v: &[f64], dir: &[f64], d: usize) -> Vec<f64> {
    v.chunks_exact(d)
        .map(|blk| blk.iter().zip(dir.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// `q_hess = Σ_i D^{i+2}φ · D^{⊗(i+1)}·dir / F(i)` contracted with the shape
/// vector (MFEM `q_hess_dot_d` loop shared by the Dirichlet and Neumann
/// integrators; `dir` is the final contraction direction — `D` for Dirichlet,
/// `n̂` for Neumann).
pub(crate) fn q_hess_dot_d_generic(
    dkphi: &[DMatrix<f64>],
    factorial: &[f64],
    shape: &[f64],
    d_vec: &[f64],
    dir: &[f64],
    nd: usize,
    d: usize,
) -> Vec<f64> {
    let mut q = vec![0.0_f64; nd];
    for (i, blk) in dkphi.iter().enumerate() {
        // T = blkᵀ · shape  (len nd·sz1·dim, laid out dof-major).
        let mut t = vec![0.0_f64; blk.ncols()];
        for m in 0..nd {
            let sm = shape[m];
            if sm == 0.0 {
                continue;
            }
            for (col, tcol) in t.iter_mut().enumerate() {
                *tcol += blk[(m, col)] * sm;
            }
        }
        // Contract (i+1) times with D, then once with `dir` (MFEM loop).
        for _ in 0..i + 1 {
            t = contract(&t, d_vec, d);
        }
        t = contract(&t, dir, d);
        let f = 1.0 / factorial[i];
        for (qj, tj) in q.iter_mut().zip(t.iter()) {
            *qj += f * tj;
        }
    }
    q
}

/// Dirichlet variant: every contraction uses the distance vector `D`
/// (MFEM `q_hess_dot_d`).
fn q_hess_dot_d(
    dkphi: &[DMatrix<f64>],
    factorial: &[f64],
    shape: &[f64],
    d_vec: &[f64],
    nd: usize,
    d: usize,
) -> Vec<f64> {
    q_hess_dot_d_generic(dkphi, factorial, shape, d_vec, d_vec, nd, d)
}

// ─── Dirichlet bilinear form integrator ──────────────────────────────────────

/// SBM Dirichlet Nitsche form (MFEM `SBM2DirichletIntegrator`): assembles the
/// face contributions of every surrogate-boundary face onto the DOFs of the
/// active element.
pub struct Sbm3DirichletIntegrator<'a, M: MeshTopology, S: FESpace<Mesh = M>> {
    /// Solution H¹ space.
    pub space: &'a S,
    /// Nitsche penalty parameter (MFEM `-alpha`, ~1 in 2-D, ~10 in 3-D).
    pub alpha: f64,
    /// Distance vector to the true boundary (component-major DOFs, length
    /// `dim · n_dofs`; MFEM `VectorCoefficient vD`).
    pub dist: &'a [f64],
    /// Element markers from [`crate::dist_solver::ShiftedFaceMarker::mark_elements`].
    pub elem_marker: &'a [i32],
    /// Whether elements cut by the true boundary belong to the surrogate
    /// domain (MFEM `-cut`).
    pub include_cut_cell: bool,
    /// Extra Taylor terms beyond `∇u·d` (MFEM `-ho`, 0 by default).
    pub ho_terms: usize,
}

impl<M, S> Sbm3DirichletIntegrator<'_, M, S>
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    /// Assemble `A += Σ_faces E_activeᵀ K_face E_active` (MFEM
    /// `AssembleFaceMatrix` scattered to the active element only).
    pub fn assemble_bilinear(&self) -> CsrMatrix<f64> {
        let mesh = self.space.mesh();
        let n = self.space.n_dofs();
        let dim = mesh.topological_dim() as usize;
        let mut coo = CooMatrix::<f64>::new(n, n);

        let faces = surrogate_face_list(mesh, self.elem_marker, self.include_cut_cell);
        let dist = DistVec { dofs: self.dist, n_scalar: n };

        for face in &faces {
            let e = face.elem;
            let order = self.space.element_order(e);
            let sre = ref_elem_vol_for_space(self.space, mesh.element_type(e), order);
            let nd = sre.n_dofs();
            let edofs = self.space.element_dofs(e);

            let geo = ElemGeometry::new(mesh, e);
            let fgeo = FaceGeometry::new(face.nodes.len(), dim);
            let quad = fgeo.re.quadrature((4 * order).min(u8::MAX));

            // Higher-order Taylor operators (MFEM dkphi_dxk / Factorial).
            let (dkphi, factorial) = if self.ho_terms > 0 {
                let dm = discrete_grad_matrices(&geo, sre.as_ref());
                let dk = dkphi_dxk(&dm, nd, dim, self.ho_terms);
                let f = factorial_dirichlet(self.ho_terms);
                (Some(dk), Some(f))
            } else {
                (None, None)
            };

            let mut phi = vec![0.0_f64; nd];
            let mut gref = vec![0.0_f64; nd * dim];
            let mut gphys = vec![0.0_f64; nd * dim];
            let mut k_face = vec![0.0_f64; nd * nd];

            for (q, xi) in quad.points.iter().enumerate() {
                let ipw = quad.weights[q];
                let x = fgeo.map(mesh, &face.nodes, xi);
                let jf = fgeo.jacobian(mesh, &face.nodes, xi);
                let mut nor = calc_ortho(&jf, dim);
                // Outward w.r.t. the active element (MFEM orients the face
                // transformation w.r.t. Elem1 and flips for `!elem1f`).
                let xc = elem_centroid(mesh, mesh.element_nodes(e));
                if dot(&nor, &sub(&x, &xc)) < 0.0 {
                    for nv in nor.iter_mut() {
                        *nv = -*nv;
                    }
                }

                let xi_e = geo.invert(&x);
                let (_je, det_e, jit) = geo.jacobian(&xi_e);
                sre.eval_basis(&xi_e, &mut phi);
                sre.eval_grad_basis(&xi_e, &mut gref);
                transform_grads(&jit, &gref, &mut gphys, nd, dim);

                let d_vec = dist.eval(self.space, e, &xi_e);

                // dshapedn_i = ipw · (∇φ_i · nor)   [= ∇φ_i·n with measure]
                let dshapedn: Vec<f64> = (0..nd)
                    .map(|i| ipw * dot_row(&gphys, i, &nor, dim))
                    .collect();

                // Term 2: -⟨∇u·n, w⟩
                for i in 0..nd {
                    for j in 0..nd {
                        k_face[i * nd + j] -= shape_at(&phi, i) * dshapedn[j];
                    }
                }

                // wrk = φ + ∇φ·D + h.o.t.
                let mut wrk: Vec<f64> = phi.clone();
                for i in 0..nd {
                    wrk[i] += dot_row(&gphys, i, &d_vec, dim);
                }
                if let (Some(dk), Some(f)) = (&dkphi, &factorial) {
                    let q = q_hess_dot_d(dk, f, &phi, &d_vec, nd, dim);
                    for (wi, qi) in wrk.iter_mut().zip(q.iter()) {
                        *wi += qi;
                    }
                }

                // Term 3: -⟨u + ∇u·d + h.o.t., ∇w·n⟩
                for i in 0..nd {
                    for j in 0..nd {
                        k_face[i * nd + j] -= dshapedn[i] * wrk[j];
                    }
                }

                // Term 4: +⟨α h⁻¹ (...), (...), h⁻¹ = |nor|²/det J.
                let hinvdx: f64 = dot(&nor, &nor) / det_e;
                let w4 = ipw * self.alpha * hinvdx;
                for i in 0..nd {
                    for j in 0..nd {
                        k_face[i * nd + j] += w4 * wrk[i] * wrk[j];
                    }
                }
            }

            for (i, &di) in edofs.iter().enumerate() {
                for (j, &dj) in edofs.iter().enumerate() {
                    coo.add(di as usize, dj as usize, k_face[i * nd + j]);
                }
            }
        }

        coo.into_csr()
    }
}

// ─── Dirichlet linear form integrator ────────────────────────────────────────

/// SBM Dirichlet right-hand side (MFEM `SBM2DirichletLFIntegrator` with a
/// `ShiftedFunctionCoefficient u_D` evaluated at the true-boundary point
/// `x + D`).
pub struct Sbm3DirichletLFIntegrator<'a, M: MeshTopology, S: FESpace<Mesh = M>> {
    /// Solution H¹ space.
    pub space: &'a S,
    /// Nitsche penalty parameter.
    pub alpha: f64,
    /// Distance vector to the true boundary (component-major DOFs).
    pub dist: &'a [f64],
    /// Element markers.
    pub elem_marker: &'a [i32],
    /// Include cut cells in the surrogate domain.
    pub include_cut_cell: bool,
    /// Extra Taylor terms beyond `∇w·d` (MFEM `-ho`).
    pub ho_terms: usize,
    /// Dirichlet data `u_D` evaluated at the *surrogate* point `x`; the shift
    /// to the true boundary `x + D` is applied internally
    /// (MFEM `ShiftedFunctionCoefficient::Eval(T, ip, D)`).
    pub ubc: &'a (dyn Fn(&[f64]) -> f64 + Send + Sync),
}

impl<M, S> Sbm3DirichletLFIntegrator<'_, M, S>
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    /// Assemble `b += Σ_faces E_activeᵀ l_face`.
    pub fn assemble_linear(&self) -> Vec<f64> {
        let mesh = self.space.mesh();
        let n = self.space.n_dofs();
        let dim = mesh.topological_dim() as usize;
        let mut b = vec![0.0_f64; n];

        let faces = surrogate_face_list(mesh, self.elem_marker, self.include_cut_cell);
        let dist = DistVec { dofs: self.dist, n_scalar: n };

        for face in &faces {
            let e = face.elem;
            let order = self.space.element_order(e);
            let sre = ref_elem_vol_for_space(self.space, mesh.element_type(e), order);
            let nd = sre.n_dofs();
            let edofs = self.space.element_dofs(e);

            let geo = ElemGeometry::new(mesh, e);
            let fgeo = FaceGeometry::new(face.nodes.len(), dim);
            let quad = fgeo.re.quadrature((4 * order).min(u8::MAX));

            let (dkphi, factorial) = if self.ho_terms > 0 {
                let dm = discrete_grad_matrices(&geo, sre.as_ref());
                let dk = dkphi_dxk(&dm, nd, dim, self.ho_terms);
                let f = factorial_dirichlet(self.ho_terms);
                (Some(dk), Some(f))
            } else {
                (None, None)
            };

            let mut phi = vec![0.0_f64; nd];
            let mut gref = vec![0.0_f64; nd * dim];
            let mut gphys = vec![0.0_f64; nd * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let ipw = quad.weights[q];
                let x = fgeo.map(mesh, &face.nodes, xi);
                let jf = fgeo.jacobian(mesh, &face.nodes, xi);
                let mut nor = calc_ortho(&jf, dim);
                let xc = elem_centroid(mesh, mesh.element_nodes(e));
                if dot(&nor, &sub(&x, &xc)) < 0.0 {
                    for nv in nor.iter_mut() {
                        *nv = -*nv;
                    }
                }

                let xi_e = geo.invert(&x);
                let (_je, det_e, jit) = geo.jacobian(&xi_e);
                sre.eval_basis(&xi_e, &mut phi);
                sre.eval_grad_basis(&xi_e, &mut gref);
                transform_grads(&jit, &gref, &mut gphys, nd, dim);

                let d_vec = dist.eval(self.space, e, &xi_e);
                // Shifted Dirichlet data: u_D(x̂ + D).
                let mut x_true = x.clone();
                for (xt, dv) in x_true.iter_mut().zip(d_vec.iter()) {
                    *xt += dv;
                }
                let ud = (self.ubc)(&x_true);

                // T2: -⟨u_D, ∇w·n⟩.  The MFEM chain adjJ·(ipw·u_D/detJ)·nor
                // cancels detJ, leaving ipw·u_D·(∇φ_i·nor).
                for i in 0..nd {
                    b[edofs[i] as usize] -= ipw * ud * dot_row(&gphys, i, &nor, dim);
                }

                // T4: +⟨α h⁻¹ u_D, w + ∇w·d + h.o.t.⟩
                let hinvdx = dot(&nor, &nor) / det_e;
                let w4 = ipw * ud * self.alpha * hinvdx;
                let mut wrk: Vec<f64> = phi.clone();
                for i in 0..nd {
                    wrk[i] += dot_row(&gphys, i, &d_vec, dim);
                }
                if let (Some(dk), Some(f)) = (&dkphi, &factorial) {
                    let q = q_hess_dot_d(dk, f, &phi, &d_vec, nd, dim);
                    for (wi, qi) in wrk.iter_mut().zip(q.iter()) {
                        *wi += qi;
                    }
                }
                for i in 0..nd {
                    b[edofs[i] as usize] += w4 * wrk[i];
                }
            }
        }

        b
    }
}

// ─── Small helpers ───────────────────────────────────────────────────────────

/// Transform reference gradients to physical coordinates with `J^{-T}`:
/// `grad_phys[i·dim + d] = Σ_c (J^{-T})[d,c] · grad_ref[i·dim + c]`
/// (same convention as the crate-private `assembler::transform_grads`).
pub(crate) fn transform_grads(
    jit: &DMatrix<f64>,
    grad_ref: &[f64],
    grad_phys: &mut [f64],
    n: usize,
    dim: usize,
) {
    for i in 0..n {
        for d in 0..dim {
            let mut s = 0.0_f64;
            for c in 0..dim {
                s += jit[(d, c)] * grad_ref[i * dim + c];
            }
            grad_phys[i * dim + d] = s;
        }
    }
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(p, q)| p * q).sum()
}

pub(crate) fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(p, q)| p - q).collect()
}

pub(crate) fn dot_row(g: &[f64], i: usize, v: &[f64], dim: usize) -> f64 {
    (0..dim).map(|c| g[i * dim + c] * v[c]).sum()
}

fn shape_at(phi: &[f64], i: usize) -> f64 {
    phi[i]
}

pub(crate) fn elem_centroid<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> Vec<f64> {
    let dim = mesh.topological_dim() as usize;
    let mut c = vec![0.0_f64; dim];
    for &n in nodes {
        for (ci, xi) in c.iter_mut().zip(mesh.node_coords(n)) {
            *ci += xi;
        }
    }
    let cnt = nodes.len() as f64;
    for ci in c.iter_mut() {
        *ci /= cnt;
    }
    c
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// Element centroid marker: elements with centroid in {x < 0.5} are
    /// INSIDE, everything else CUT (the inactive side of the surrogate mesh).
    fn marker_split(mesh: &Mesh<2>) -> Vec<i32> {
        mesh.elem_iter()
            .map(|e| {
                let nodes = mesh.element_nodes(e);
                let cx: f64 = nodes.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>()
                    / nodes.len() as f64;
                if cx <= 0.5 {
                    SB_INSIDE
                } else {
                    SB_CUT
                }
            })
            .collect()
    }

    /// Distance vector D = (0.5 − x, 0) at the space DOFs (component-major);
    /// zero on the surrogate plane x = 0.5.
    fn planar_dist(space: &H1Space<Mesh<2>>) -> Vec<f64> {
        let nd = space.n_dofs();
        let dm = space.dof_manager();
        let mut dist = vec![0.0_f64; 2 * nd];
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            dist[dof as usize] = 0.5 - x[0];
        }
        dist
    }

    /// SBM bilinear + LF with the given ubc on the {x<0.5} strip.
    fn sbm_forms(
        space: &H1Space<Mesh<2>>,
        marker: &[i32],
        dist: &[f64],
        alpha: f64,
        ho_terms: usize,
        ubc: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    ) -> (CsrMatrix<f64>, Vec<f64>) {
        let a = Sbm3DirichletIntegrator {
            space,
            alpha,
            dist,
            elem_marker: marker,
            include_cut_cell: false,
            ho_terms,
        }
        .assemble_bilinear();
        let b = Sbm3DirichletLFIntegrator {
            space,
            alpha,
            dist,
            elem_marker: marker,
            include_cut_cell: false,
            ho_terms,
            ubc,
        }
        .assemble_linear();
        (a, b)
    }

    /// Volume Laplacian `∫∇u·∇v` over the INSIDE elements only, via the same
    /// geometry layer the SBM integrators use.
    fn diffusion_inside(space: &H1Space<Mesh<2>>, marker: &[i32]) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        let dim = 2;
        let nd_glob = space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(nd_glob, nd_glob);
        for e in mesh.elem_iter() {
            if marker[e as usize] != SB_INSIDE {
                continue;
            }
            let sre = ref_elem_vol_for_space(space, mesh.element_type(e), space.element_order(e));
            let nd = sre.n_dofs();
            let rule = sre.quadrature(2 * space.element_order(e) + 1);
            let geo = ElemGeometry::new(mesh, e);
            let mut phi = vec![0.0_f64; nd];
            let mut gref = vec![0.0_f64; nd * dim];
            let mut gphys = vec![0.0_f64; nd * dim];
            let mut ke = vec![0.0_f64; nd * nd];
            for (q, xi) in rule.points.iter().enumerate() {
                let (_j, det, jit) = geo.jacobian(xi);
                let w = rule.weights[q] * det.abs();
                sre.eval_basis(xi, &mut phi);
                sre.eval_grad_basis(xi, &mut gref);
                transform_grads(&jit, &gref, &mut gphys, nd, dim);
                for i in 0..nd {
                    for j in 0..nd {
                        ke[i * nd + j] += w * dot_row(&gphys, i, &gphys[j * dim..], dim);
                    }
                }
            }
            let dofs = space.element_dofs(e);
            for (i, &di) in dofs.iter().enumerate() {
                for (j, &dj) in dofs.iter().enumerate() {
                    coo.add(di as usize, dj as usize, ke[i * nd + j]);
                }
            }
        }
        coo.into_csr()
    }

    /// Constant consistency identity: for u ≡ 1 (which satisfies -Δu = 0 and
    /// natural zero-flux conditions everywhere), the assembled system must
    /// satisfy `A·1 = b` exactly, quadrature point by quadrature point.
    #[test]
    fn dirichlet_constant_consistency() {
        for tri in [false, true] {
            let mesh = if tri {
                Mesh::<2>::unit_square_tri(5)
            } else {
                Mesh::<2>::unit_square_quad(5)
            };
            let space = H1Space::new(mesh, 1);
            let marker = marker_split(space.mesh());
            let dist = planar_dist(&space);
            let (a, b) = sbm_forms(&space, &marker, &dist, 2.0, 0, &|_| 1.0);
            let ones = vec![1.0_f64; space.n_dofs()];
            let mut au = vec![0.0_f64; space.n_dofs()];
            a.spmv(&ones, &mut au);
            let mut max_diff = 0.0_f64;
            for (l, r) in au.iter().zip(b.iter()) {
                max_diff = max_diff.max((l - r).abs());
            }
            println!("SBM constant consistency (tri={tri}): {max_diff:.3e}");
            assert!(max_diff < 1e-12, "consistency identity failed: {max_diff}");
        }
    }

    /// Nitsche patch test: strip {x < 0.5}, surrogate boundary x = 0.5 with
    /// zero distance on it (level set x − 0.5, D = (0.5−x, 0)).  The exact
    /// solution u = x satisfies -Δu = 0 with u = x on x = 0.5; the artificial
    /// outer boundary of the strip carries the matching exact values so the
    /// solution is exact in the P1 space.  Residual ≈ round-off.
    fn patch_test_2d(mesh: Mesh<2>, alpha: f64, tol: f64) {
        let space = H1Space::new(mesh, 1);
        let marker = marker_split(space.mesh());
        let nd = space.n_dofs();
        let dm = space.dof_manager();
        let dist = planar_dist(&space);

        let a_vol = diffusion_inside(&space, &marker);
        let (a_sbm, b_sbm) = sbm_forms(&space, &marker, &dist, alpha, 0, &|x: &[f64]| x[0]);
        let mut a = a_vol.add(&a_sbm);
        let mut b = b_sbm.clone();

        // Strong exact values on the artificial outer boundary (x = 0 and
        // the y = 0, y = 1 edges); the SBM plane x = 0.5 stays free.
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            let on_outer = x[0] < 1e-12 || x[1] < 1e-12 || x[1] > 1.0 - 1e-12;
            if on_outer && x[0] < 0.5 + 1e-12 {
                a.apply_dirichlet_keep_diag(dof as usize, x[0], &mut b);
            }
        }

        let mut u = vec![0.0_f64; nd];
        let cfg = fem_linalg::SolverConfig {
            rtol: 1e-13,
            atol: 0.0,
            max_iter: 2000,
            ..Default::default()
        };
        let res = fem_solver::solve_cg(&a, &b, &mut u, &cfg)
            .unwrap_or_else(|e| panic!("CG failed: {e}"));
        assert!(res.converged, "CG did not converge");

        let mut max_err = 0.0_f64;
        let mut worst = 0_usize;
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            if x[0] <= 0.5 + 1e-12 {
                let e = (u[dof as usize] - x[0]).abs();
                if e > max_err {
                    max_err = e;
                    worst = dof as usize;
                }
            }
        }
        let wx = dm.dof_coord(worst as u32);
        println!(
            "SBM Dirichlet patch test: L-inf error = {max_err:.3e} at dof {worst} ({:?}) u={}",
            wx, u[worst]
        );
        assert!(max_err < tol, "patch test failed: {max_err}");
    }

    #[test]
    fn dirichlet_patch_test_quad() {
        patch_test_2d(Mesh::<2>::unit_square_quad(6), 1.0, 1e-9);
    }

    #[test]
    fn dirichlet_patch_test_tri() {
        patch_test_2d(Mesh::<2>::unit_square_tri(6), 4.0, 1e-9);
    }

    /// The assembled SBM Dirichlet face form is symmetric (Nitsche structure).
    #[test]
    fn face_form_is_symmetric() {
        let space = H1Space::new(Mesh::<2>::unit_square_quad(5), 2);
        let marker = marker_split(space.mesh());
        let dist = planar_dist(&space);
        let sbm = Sbm3DirichletIntegrator {
            space: &space,
            alpha: 2.0,
            dist: &dist,
            elem_marker: &marker,
            include_cut_cell: false,
            ho_terms: 0,
        };
        let a = sbm.assemble_bilinear().to_dense();
        let nd = space.n_dofs();
        let mut max_asym = 0.0_f64;
        for i in 0..nd {
            for j in 0..nd {
                max_asym = max_asym.max((a[i * nd + j] - a[j * nd + i]).abs());
            }
        }
        println!("SBM Dirichlet form asymmetry: {max_asym:.3e}");
        assert!(max_asym < 1e-12, "not symmetric: {max_asym}");
    }

    /// Higher-order Taylor terms, P1-simplex exactness: the discrete second
    /// derivatives vanish, so the shifted test functional
    /// `w + ∇w·D + h.o.t.` must reproduce `w_h(x̂ + D)` exactly.  The ho=1
    /// assembled LF is compared against a hand evaluation with the basis
    /// evaluated at the shifted points (constant D on a straight face).
    #[test]
    fn ho_terms_reproduce_shifted_evaluation_p1() {
        let space = H1Space::new(Mesh::<2>::unit_square_tri(4), 1);
        let marker = marker_split(space.mesh());
        let nd = space.n_dofs();
        let dm = space.dof_manager();

        let d0 = [0.1_f64, 0.05];
        let mut dist = vec![0.0_f64; 2 * nd];
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            dist[dof as usize] = d0[0];
            dist[nd + dof as usize] = d0[1];
        }

        let lf = Sbm3DirichletLFIntegrator {
            space: &space,
            alpha: 2.0,
            dist: &dist,
            elem_marker: &marker,
            include_cut_cell: false,
            ho_terms: 1,
            ubc: &|_x: &[f64]| 1.0_f64,
        };
        let b_ho = lf.assemble_linear();

        // Direct assembly of the same functional with wrk replaced by the
        // exact shifted evaluation φ_m(x̂ + D): T2 unchanged (the test
        // gradient carries no shift) and T4 = α h⁻¹ · φ_m(x̂ + D).
        let faces = surrogate_face_list(space.mesh(), &marker, false);
        assert!(!faces.is_empty());
        let mut b_direct = vec![0.0_f64; nd];
        for face in &faces {
            let e = face.elem;
            let sre = ref_elem_vol_for_space(&space, space.mesh().element_type(e), 1);
            let geo = ElemGeometry::new(space.mesh(), e);
            let fgeo = FaceGeometry::new(face.nodes.len(), 2);
            let quad = fgeo.re.quadrature(4);
            let edofs = space.element_dofs(e);
            let ndl = sre.n_dofs();
            let mut phi = vec![0.0_f64; ndl];
            let mut gref = vec![0.0_f64; ndl * 2];
            let mut gphys = vec![0.0_f64; ndl * 2];
            for (q, xi) in quad.points.iter().enumerate() {
                let ipw = quad.weights[q];
                let x = fgeo.map(space.mesh(), &face.nodes, xi);
                let jf = fgeo.jacobian(space.mesh(), &face.nodes, xi);
                let mut nor = calc_ortho(&jf, 2);
                let xc = elem_centroid(space.mesh(), space.mesh().element_nodes(e));
                if dot(&nor, &sub(&x, &xc)) < 0.0 {
                    for nv in nor.iter_mut() {
                        *nv = -*nv;
                    }
                }
                let xi_e = geo.invert(&x);
                let (_je, det_e, jit) = geo.jacobian(&xi_e);
                sre.eval_basis(&xi_e, &mut phi);
                sre.eval_grad_basis(&xi_e, &mut gref);
                transform_grads(&jit, &gref, &mut gphys, ndl, 2);
                // φ_m(x̂ + D): invert the affine map at the shifted point.
                let x_true = [x[0] + d0[0], x[1] + d0[1]];
                let n0 = space.mesh().element_nodes(e)[0];
                let x0 = space.mesh().node_coords(n0);
                let mut xi_t = [0.0_f64; 2];
                for i in 0..2 {
                    // d i = J^{-1},dx = jit^{T},dx: jit[(c, i)] = (J^{-1})[i][c].
                    for c in 0..2 {
                        xi_t[i] += jit[(c, i)] * (x_true[c] - x0[c]);
                    }
                }
                let mut phi_t = vec![0.0_f64; ndl];
                sre.eval_basis(&xi_t, &mut phi_t);
                let hinv = dot(&nor, &nor) / det_e;
                for (i, &dof) in edofs.iter().enumerate() {
                    let t2 = -ipw * dot_row(&gphys, i, &nor, 2);
                    let t4 = ipw * 2.0 * hinv * phi_t[i];
                    b_direct[dof as usize] += t2 + t4;
                }
            }
        }

        let mut max_diff = 0.0_f64;
        for (a, b) in b_ho.iter().zip(b_direct.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("SBM ho_terms=1 shifted-evaluation max diff: {max_diff:.3e}");
        assert!(max_diff < 1e-12, "h.o.t. Taylor mismatch: {max_diff}");
    }

    /// Surrogate face count sanity: the split quad mesh has exactly one row
    /// (n faces) of surrogate faces, each with an INSIDE active element.
    #[test]
    fn surrogate_face_list_counts() {
        for n in [3_usize, 5] {
            let space = H1Space::new(Mesh::<2>::unit_square_quad(n), 1);
            let marker = marker_split(space.mesh());
            let faces = surrogate_face_list(space.mesh(), &marker, false);
            assert_eq!(faces.len(), n, "n={n}");
            for f in &faces {
                assert_eq!(marker[f.elem as usize], SB_INSIDE);
                assert_eq!(f.nodes.len(), 2);
            }
        }
    }

    /// 3-D: Tet4 mesh split by the plane x = 0.5 — the SBM form is symmetric
    /// and the patch test (u = x, α = 10 as in the 3-D miniapp runs) holds.
    #[test]
    fn dirichlet_3d_tet_patch_test() {
        let space = H1Space::new(Mesh::<3>::unit_cube_tet(2), 1);
        let nd = space.n_dofs();
        let dm = space.dof_manager();
        let mesh = space.mesh();

        let marker: Vec<i32> = mesh
            .elem_iter()
            .map(|e| {
                let nodes = mesh.element_nodes(e);
                let cx: f64 = nodes.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>()
                    / nodes.len() as f64;
                if cx <= 0.5 {
                    SB_INSIDE
                } else {
                    SB_CUT
                }
            })
            .collect();

        let mut dist = vec![0.0_f64; 3 * nd];
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            dist[dof as usize] = 0.5 - x[0];
        }

        let sbm = Sbm3DirichletIntegrator {
            space: &space,
            alpha: 10.0,
            dist: &dist,
            elem_marker: &marker,
            include_cut_cell: false,
            ho_terms: 0,
        };

        let a_sbm = sbm.assemble_bilinear();
        let dense = a_sbm.to_dense();
        let mut max_asym = 0.0_f64;
        for i in 0..nd {
            for j in 0..nd {
                max_asym = max_asym.max((dense[i * nd + j] - dense[j * nd + i]).abs());
            }
        }
        println!("SBM 3-D face form asymmetry: {max_asym:.3e}");
        assert!(max_asym < 1e-12, "SBM 3-D form not symmetric: {max_asym}");

        // Patch test: -Δu = 0 in {x < 0.5}, u = x on the surrogate plane,
        // exact strong values on the artificial outer faces.
        let mut coo = CooMatrix::<f64>::new(nd, nd);
        for e in mesh.elem_iter() {
            if marker[e as usize] != SB_INSIDE {
                continue;
            }
            let sre = ref_elem_vol_for_space(&space, mesh.element_type(e), 1);
            let ndl = sre.n_dofs();
            let rule = sre.quadrature(5);
            let geo = ElemGeometry::new(mesh, e);
            let mut phi = vec![0.0_f64; ndl];
            let mut gref = vec![0.0_f64; ndl * 3];
            let mut gphys = vec![0.0_f64; ndl * 3];
            let mut ke = vec![0.0_f64; ndl * ndl];
            for (q, xi) in rule.points.iter().enumerate() {
                let (_j, det, jit) = geo.jacobian(xi);
                let w = rule.weights[q] * det.abs();
                sre.eval_basis(xi, &mut phi);
                sre.eval_grad_basis(xi, &mut gref);
                transform_grads(&jit, &gref, &mut gphys, ndl, 3);
                for i in 0..ndl {
                    for j in 0..ndl {
                        ke[i * ndl + j] += w * (gphys[i * 3] * gphys[j * 3]
                            + gphys[i * 3 + 1] * gphys[j * 3 + 1]
                            + gphys[i * 3 + 2] * gphys[j * 3 + 2]);
                    }
                }
            }
            let dofs = space.element_dofs(e);
            for (i, &di) in dofs.iter().enumerate() {
                for (j, &dj) in dofs.iter().enumerate() {
                    coo.add(di as usize, dj as usize, ke[i * ndl + j]);
                }
            }
        }
        let mut a = coo.into_csr().add(&a_sbm);
        let lf = Sbm3DirichletLFIntegrator {
            space: &space,
            alpha: 10.0,
            dist: &dist,
            elem_marker: &marker,
            include_cut_cell: false,
            ho_terms: 0,
            ubc: &|x: &[f64]| x[0],
        };
        let mut b = lf.assemble_linear();

        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            let on_outer = x[0] < 1e-12
                || x[1] < 1e-12
                || x[1] > 1.0 - 1e-12
                || x[2] < 1e-12
                || x[2] > 1.0 - 1e-12;
            if on_outer && x[0] < 0.5 + 1e-12 {
                a.apply_dirichlet_keep_diag(dof as usize, x[0], &mut b);
            }
        }

        let mut u = vec![0.0_f64; nd];
        let cfg = fem_linalg::SolverConfig {
            rtol: 1e-13,
            atol: 0.0,
            max_iter: 5000,
            ..Default::default()
        };
        let res = fem_solver::solve_cg(&a, &b, &mut u, &cfg)
            .unwrap_or_else(|e| panic!("CG failed: {e}"));
        assert!(res.converged, "3-D CG did not converge");
        let mut max_err = 0.0_f64;
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            if x[0] <= 0.5 + 1e-12 {
                max_err = max_err.max((u[dof as usize] - x[0]).abs());
            }
        }
        println!("SBM 3-D tet patch test: L-inf error = {max_err:.3e}");
        assert!(max_err < 1e-8, "3-D patch test failed: {max_err}");
    }
}
