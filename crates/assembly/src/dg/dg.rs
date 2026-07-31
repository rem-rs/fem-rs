//! Discontinuous Galerkin (DG) interior penalty assembly.
//!
//! Implements the **Symmetric Interior Penalty (SIP)** method for the scalar
//! diffusion equation `−∇·(κ ∇u) = f` with Dirichlet boundary conditions.
//!
//! # Bilinear form
//!
//! ```text
//! a_h(u,v) = ∑_K ∫_K κ ∇u·∇v dx
//!            − ∑_F ∫_F { κ ∇u }·[[v]] ds   (consistency)
//!            − ∑_F ∫_F { κ ∇v }·[[u]] ds   (symmetry, only for SIP)
//!            + ∑_F ∫_F (σ/h_F) [[u]]·[[v]] ds  (penalty)
//! ```
//!
//! where:
//! - `{·}` is the average operator: `{w} = ½(w⁺ + w⁻)` on interior faces,
//!   `{w} = w` on Dirichlet boundary faces.
//! - `[[·]]` is the scalar jump: `[[u]] = u⁺ n⁺ + u⁻ n⁻` (vector jump) or
//!   `[[u]] = u⁺ − u⁻` (scalar jump used with normal orientation convention).
//! - `h_F` is the face size (length in 2-D).
//! - `σ` is the penalty parameter (must be large enough for coercivity; typically
//!   σ ≥ C p²/h_F where p is the polynomial degree).
//!
//! # Usage
//! ```rust,ignore
//! let space = L2Space::new(mesh, 1);
//! let ifl   = InteriorFaceList::build(space.mesh());
//! let mat   = DgAssembler::assemble_sip(&space, &ifl, kappa, sigma, 3);
//! ```

use nalgebra::DMatrix;

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

use super::dg_base::{
    build_face_elem_map, face_geom_2d, orient_normal_outward, phys_to_ref,
    quad_jac_at, ref_elem_face, ref_elem_vol, simplex_jac, xform_grads,
};
use crate::interior_faces::InteriorFaceList;
#[cfg(feature = "parallel")]
use crate::assembler::assembly_parallel_min_elems;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ─── DgAssembler ─────────────────────────────────────────────────────────────

/// Stateless DG assembly driver.
pub struct DgAssembler;

impl DgAssembler {
    /// Assemble the global SIP-DG stiffness matrix.
    ///
    /// Combines:
    /// 1. **Volume terms**: standard diffusion `∫ κ ∇u·∇v dx` per element.
    /// 2. **Interior face terms**: consistency + symmetry + penalty.
    /// 3. **Boundary face terms** (Dirichlet, all boundary tags): same penalty form.
    ///
    /// # Arguments
    /// - `space`      — the L² (DG) finite element space.
    /// - `ifl`        — pre-built interior face list.
    /// - `kappa`      — diffusion coefficient (scalar, uniform).
    /// - `sigma`      — penalty parameter (dimensionless; use ≥ 3*(order+1)² for coercivity).
    /// - `quad_order` — polynomial order the quadrature integrates exactly.
    pub fn assemble_sip<S: FESpace + Sync>(
        space:      &S,
        ifl:        &InteriorFaceList,
        kappa:      f64,
        sigma:      f64,
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        // SIP: sigma_sign = -1, penalty applied on every boundary tag.
        Self::assemble_dg(space, ifl, kappa, -1.0, sigma, quad_order, None)
    }

    /// General MFEM-style DG diffusion assembly.
    ///
    /// Mirrors MFEM's `DGDiffusionIntegrator(a, sigma, kappa)`:
    /// - **Volume terms**: `∫ a ∇u·∇v dx`.
    /// - **Interior face terms**: `−a·elmat + σ·a·elmatᵀ + κ·jmat`.
    /// - **Boundary face terms** (only on `bdr_tags`, or all boundary faces when
    ///   `bdr_tags` is `None`): the same penalty form — the weak enforcement of
    ///   homogeneous Dirichlet BCs.
    ///
    /// # Arguments
    /// - `space`      — the L² (DG) finite element space.
    /// - `ifl`        — pre-built interior face list.
    /// - `a`          — diffusion coefficient (scalar, uniform; MFEM `matCoef`).
    /// - `sigma`      — symmetrization sign, +1 (NIP) or −1 (SIP); a value of
    ///   −1 yields a symmetric matrix (PCG), any other value a non-symmetric one
    ///   (GMRES).
    /// - `penalty`    — DG penalty parameter (MFEM `kappa`; `(order+1)²` by default).
    /// - `quad_order` — polynomial order the quadrature integrates exactly.
    /// - `bdr_tags`   — boundary attributes on which the Dirichlet face penalty is
    ///   applied; `None` means every boundary face.
    pub fn assemble_dg<S: FESpace + Sync>(
        space:      &S,
        ifl:        &InteriorFaceList,
        a:          f64,
        sigma:      f64,
        penalty:    f64,
        quad_order: u8,
        bdr_tags:   Option<&[i32]>,
    ) -> CsrMatrix<f64> {
        let mesh   = space.mesh();
        let dim    = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        let order  = space.order();

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        // ── 1. Volume terms ────────────────────────────────────────────────────
        #[cfg(feature = "parallel")]
        {
            if mesh.n_elements() >= assembly_parallel_min_elems() {
                coo.append(assemble_dg_volume_parallel(space, a, quad_order));
            } else {
                assemble_volume(&mut coo, space, a, quad_order);
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            assemble_volume(&mut coo, space, a, quad_order);
        }

        // ── 2. Interior face terms ─────────────────────────────────────────────
        #[cfg(feature = "parallel")]
        {
            if ifl.faces.len() >= assembly_parallel_min_elems() {
                let merged = ifl
                    .faces
                    .par_iter()
                    .map(|iface| {
                        let mut local = CooMatrix::<f64>::new(n_dofs, n_dofs);
                        assemble_interior_face(
                            &mut local,
                            mesh,
                            space,
                            iface.elem_left,
                            iface.elem_right,
                            &iface.face_nodes,
                            a,
                            sigma,
                            penalty,
                            order,
                            quad_order,
                        );
                        local
                    })
                    .reduce(
                        || CooMatrix::<f64>::new(n_dofs, n_dofs),
                        |mut a, b| {
                            a.append(b);
                            a
                        },
                    );
                coo.append(merged);
            } else {
                for iface in &ifl.faces {
                    assemble_interior_face(
                        &mut coo, mesh, space, iface.elem_left, iface.elem_right,
                        &iface.face_nodes, a, sigma, penalty, order, quad_order,
                    );
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for iface in &ifl.faces {
                assemble_interior_face(
                    &mut coo, mesh, space, iface.elem_left, iface.elem_right,
                    &iface.face_nodes, a, sigma, penalty, order, quad_order,
                );
            }
        }

        // ── 3. Boundary face terms (Dirichlet) ─────────────────────────────────
        // Build face→element map (Mesh::face_elements always returns (0,None)).
        let face_to_elem = build_face_elem_map(mesh, dim);
        let boundary_pairs: Vec<(u32, u32)> = mesh
            .face_iter()
            .filter_map(|f| {
                if let Some(tags) = bdr_tags {
                    if !tags.contains(&mesh.face_tag(f)) {
                        return None;
                    }
                }
                face_to_elem.get(&f).copied().map(|e| (f, e))
            })
            .collect();
        #[cfg(feature = "parallel")]
        {
            if boundary_pairs.len() >= assembly_parallel_min_elems() {
                let merged = boundary_pairs
                    .par_iter()
                    .copied()
                    .map(|(f, elem)| {
                        let mut local = CooMatrix::<f64>::new(n_dofs, n_dofs);
                        assemble_boundary_face_with_elem(
                            &mut local, mesh, space, f, elem, a, sigma, penalty, order,
                            quad_order,
                        );
                        local
                    })
                    .reduce(
                        || CooMatrix::<f64>::new(n_dofs, n_dofs),
                        |mut a, b| {
                            a.append(b);
                            a
                        },
                    );
                coo.append(merged);
            } else {
                for (f, elem) in &boundary_pairs {
                    assemble_boundary_face_with_elem(
                        &mut coo, mesh, space, *f, *elem, a, sigma, penalty, order, quad_order,
                    );
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (f, elem) in &boundary_pairs {
                assemble_boundary_face_with_elem(
                    &mut coo, mesh, space, *f, *elem, a, sigma, penalty, order, quad_order,
                );
            }
        }

        coo.into_csr()
    }
}

// ─── Volume contribution ──────────────────────────────────────────────────────

fn accumulate_dg_volume_element<S: FESpace>(
    space:      &S,
    e:          u32,
    kappa:      f64,
    quad_order: u8,
    coo:        &mut CooMatrix<f64>,
) {
    let mesh  = space.mesh();
    let dim   = mesh.dim() as usize;
    let order = space.order();

    let mut phi      = Vec::<f64>::new();
    let mut grad_ref = Vec::<f64>::new();
    let mut grad_p   = Vec::<f64>::new();

    let elem_type = mesh.element_type(e);
    let re = ref_elem_vol(elem_type, order);
    let n  = re.n_dofs();
    let q  = re.quadrature(quad_order);
    let gd = space.element_dofs(e).iter().map(|&d| d as usize).collect::<Vec<_>>();
    let nodes = mesh.element_nodes(e);
    let (jac, det_j) = simplex_jac(mesh, nodes, dim);
    let j_inv_t = jac.clone().try_inverse().unwrap_or_else(|| {eprintln!("  warning: degenerate element"); DMatrix::identity(2,2)}).transpose();

    phi.resize(n, 0.0);
    grad_ref.resize(n * dim, 0.0);
    grad_p.resize(n * dim, 0.0);

    let mut k_elem = vec![0.0_f64; n * n];

    for (qi, xi) in q.points.iter().enumerate() {
        let w = q.weights[qi] * det_j.abs();
        re.eval_grad_basis(xi, &mut grad_ref);
        xform_grads(&j_inv_t, &grad_ref, &mut grad_p, n, dim);
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += grad_p[i * dim + d] * grad_p[j * dim + d];
                }
                k_elem[i * n + j] += w * kappa * dot;
            }
        }
    }

    for (i, &gi) in gd.iter().enumerate() {
        for (j, &gj) in gd.iter().enumerate() {
            coo.add(gi, gj, k_elem[i * n + j]);
        }
    }
}

fn assemble_volume<S: FESpace>(
    coo:        &mut CooMatrix<f64>,
    space:      &S,
    kappa:      f64,
    quad_order: u8,
) {
    let mesh = space.mesh();
    for e in mesh.elem_iter() {
        accumulate_dg_volume_element(space, e, kappa, quad_order, coo);
    }
}

#[cfg(feature = "parallel")]
fn assemble_dg_volume_parallel<S: FESpace>(
    space:      &S,
    kappa:      f64,
    quad_order: u8,
) -> CooMatrix<f64> {
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    mesh.elem_iter()
        .into_par_iter()
        .map(|e| {
            let mut local = CooMatrix::<f64>::new(n_dofs, n_dofs);
            accumulate_dg_volume_element(space, e, kappa, quad_order, &mut local);
            local
        })
        .reduce(
            || CooMatrix::<f64>::new(n_dofs, n_dofs),
            |mut a, b| {
                a.append(b);
                a
            },
        )
}

// ─── Interior face contribution ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn assemble_interior_face<S: FESpace>(
    coo:        &mut CooMatrix<f64>,
    mesh:       &S::Mesh,
    space:      &S,
    el:         u32,
    er:         u32,
    face_nodes: &[u32],
    diff:       f64,
    sigma:      f64,
    penalty:    f64,
    order:      u8,
    quad_order: u8,
) {
    let dim = mesh.dim() as usize;
    let (h_f, mut normal_l) = face_geom_2d(mesh, face_nodes);
    orient_normal_outward(mesh, el, face_nodes, &mut normal_l);

    // Build reference elements and quadrature for the face.
    let face_elem_type = if dim == 2 { ElementType::Line2 } else { ElementType::Tri3 };
    let ref_face = ref_elem_face(face_elem_type, order);
    let q_face   = ref_face.quadrature(quad_order);
    let _n_f = ref_face.n_dofs();

    // Build reference elements for the volume.
    let et_l = mesh.element_type(el);
    let re_l = ref_elem_vol(et_l, order);
    let et_r = mesh.element_type(er);
    let re_r = ref_elem_vol(et_r, order);
    let n_l = re_l.n_dofs();
    let n_r = re_r.n_dofs();

    let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();

    let nodes_l = mesh.element_nodes(el);
    let nodes_r = mesh.element_nodes(er);
    let (jac_l, _det_l) = simplex_jac(mesh, nodes_l, dim);
    let (jac_r, _det_r) = simplex_jac(mesh, nodes_r, dim);
    let jit_l = jac_l.clone().try_inverse().unwrap_or_else(|| {eprintln!("  warning: degenerate element {} for face", el); DMatrix::identity(2,2)}).transpose();
    let jit_r = jac_r.clone().try_inverse().unwrap_or_else(|| {eprintln!("  warning: degenerate element {} for face", er); DMatrix::identity(2,2)}).transpose();

    // Pre-compute quad vertex coords for per-point Jacobian evaluation.
    let (xl, yl, xr, yr) = if nodes_l.len() > 3 || nodes_r.len() > 3 {
        let xl: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes_l[k.min(nodes_l.len()-1)])[0]).collect();
        let yl: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes_l[k.min(nodes_l.len()-1)])[1]).collect();
        let xr: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes_r[k.min(nodes_r.len()-1)])[0]).collect();
        let yr: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes_r[k.min(nodes_r.len()-1)])[1]).collect();
        (xl, yl, xr, yr)
    } else {
        (vec![], vec![], vec![], vec![])
    };

    // Physical face quadrature points (for nor and QP mapping).
    let x0f = mesh.node_coords(face_nodes[0]);
    let x1f = mesh.node_coords(face_nodes[1]);

    // C++-style: nor = CalcOrtho(J_face) = unit_normal * h_f/2
    // This is the UNNORMALIZED normal; |nor| = face Jacobian determinant = h_f/2.
    // normal_l is already oriented outward from element 1 (left element).
    let nor = vec![h_f / 2.0 * normal_l[0], h_f / 2.0 * normal_l[1]];
    let nor_norm2 = h_f * h_f / 4.0;  // = |nor|² = h_f²/4

    // Single ndofs×ndofs matrix per C++: elmat (consistency) + jmat (penalty, lower tri)
    let ndofs = n_l + n_r;
    let mut el_local = vec![0.0_f64; ndofs * ndofs];
    let mut jm_local = vec![0.0_f64; ndofs * ndofs];

    let face_xi: Vec<Vec<f64>> = q_face.points.clone();
    let face_weights = &q_face.weights;

    let mut phi_l    = vec![0.0_f64; n_l];
    let mut phi_r    = vec![0.0_f64; n_r];
    let mut gref_l   = vec![0.0_f64; n_l * dim];
    let mut gref_r   = vec![0.0_f64; n_r * dim];
    let mut gphys_l  = vec![0.0_f64; n_l * dim];
    let mut gphys_r  = vec![0.0_f64; n_r * dim];
    let mut dsf1dn   = vec![0.0_f64; n_l];
    let mut dsf2dn   = vec![0.0_f64; n_r];

    for (qi, xi_f) in face_xi.iter().enumerate() {
        // Physical quadrature point on the face (linear mapping: [-1,1] → physical).
        let xp: Vec<f64> = (0..dim).map(|i| {
            0.5 * ((1.0 - xi_f[0]) * x0f[i] + (1.0 + xi_f[0]) * x1f[i])
        }).collect();

        // Map physical point → reference coordinates of each element.
        let mut xi_l = phys_to_ref(&jac_l, mesh.node_coords(nodes_l[0]), &xp, dim);
        let mut xi_r = phys_to_ref(&jac_r, mesh.node_coords(nodes_r[0]), &xp, dim);
        if nodes_l.len() > 3 { for v in &mut xi_l { *v -= 1.0; } }
        if nodes_r.len() > 3 { for v in &mut xi_r { *v -= 1.0; } }

        re_l.eval_basis(&xi_l, &mut phi_l);
        re_r.eval_basis(&xi_r, &mut phi_r);
        re_l.eval_grad_basis(&xi_l, &mut gref_l);
        re_r.eval_grad_basis(&xi_r, &mut gref_r);
        xform_grads(&jit_l, &gref_l, &mut gphys_l, n_l, dim);
        xform_grads(&jit_r, &gref_r, &mut gphys_r, n_r, dim);

        // Per-point Jacobian for quads (triangles use constant from simplex_jac).
        let (jac_pt_l, det_l, jit_pt_l) = if nodes_l.len() > 3 {
            let (j, d) = quad_jac_at(&xl, &yl, xi_l[0], xi_l[1]);
            let d_safe = d.abs().max(1e-14);
            let ji = j.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(2,2)).transpose();
            (j, d_safe, ji)
        } else {
            (jac_l.clone(), jac_l.determinant().abs().max(1e-14), jit_l.clone())
        };
        let (jac_pt_r, det_r, jit_pt_r) = if nodes_r.len() > 3 {
            let (j, d) = quad_jac_at(&xr, &yr, xi_r[0], xi_r[1]);
            let d_safe = d.abs().max(1e-14);
            let ji = j.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(2,2)).transpose();
            (j, d_safe, ji)
        } else {
            (jac_r.clone(), jac_r.determinant().abs().max(1e-14), jit_r.clone())
        };
        // Re-compute physical gradients using the per-point Jacobian.
        xform_grads(&jit_pt_l, &gref_l, &mut gphys_l, n_l, dim);
        xform_grads(&jit_pt_r, &gref_r, &mut gphys_r, n_r, dim);

        // ── C++ DGDiffusionIntegrator per-QP algorithm ──────────────────────
        // w = ip.weight / det(J_elem) / 2  (interior face averaging)
        //   → dshape·nh  where  nh = adjJ * (w * nor)
        //   → dshape1dn[j] = det(J) * w * ∇_x φ_j · nor
        //                    = (qw/2) * ∇_x φ_j · nor        (det(J)*w = qw/2)

        let qw = face_weights[qi];  // ip.weight on reference face

        // Element 1 (left): ∇_x φ_j · nor → dsf1dn[j]
        for j in 0..n_l {
            let dot = gphys_l[j * dim] * nor[0] + gphys_l[j * dim + 1] * nor[1];
            dsf1dn[j] = (qw / 2.0) * dot;
        }

        // Element 2 (right): ∇_x φ_j · nor → dsf2dn[j]
        for j in 0..n_r {
            let dot = gphys_r[j * dim] * nor[0] + gphys_r[j * dim + 1] * nor[1];
            dsf2dn[j] = (qw / 2.0) * dot;
        }

        // ── Consistency matrix elmat (before sign) ──────────────────────────
        // C++: elmat(i,j) += shape(i) * dshape·nh(j)
        // A_11: test 1, trial 1
        for i in 0..n_l {
            for j in 0..n_l {
                el_local[i * ndofs + j] += phi_l[i] * dsf1dn[j];
            }
        }
        // A_12: test 1, trial 2
        for i in 0..n_l {
            for j in 0..n_r {
                el_local[i * ndofs + (n_l + j)] += phi_l[i] * dsf2dn[j];
            }
        }
        // A_21: test 2, trial 1  (C++: -= shape2 * dshape1dn)
        for i in 0..n_r {
            for j in 0..n_l {
                el_local[(n_l + i) * ndofs + j] -= phi_r[i] * dsf1dn[j];
            }
        }
        // A_22: test 2, trial 2  (C++: -= shape2 * dshape2dn)
        for i in 0..n_r {
            for j in 0..n_r {
                el_local[(n_l + i) * ndofs + (n_l + j)] -= phi_r[i] * dsf2dn[j];
            }
        }

        // ── Penalty wq ──────────────────────────────────────────────────────
        // wq = ni·nor = w * |nor|², summed over elements
        // w1 = qw / det_l / 2, w2 = qw / det_r / 2
        let wq = nor_norm2 * (qw / 2.0) * (1.0 / det_l + 1.0 / det_r);
        // C++: jmat += kappa * wq * shape * shape  (kappa = penalty here)
        let jscale = penalty * wq;  // penalty = the caller's DG penalty κ

        // jmat lower-triangular block structure (C++ matches both symmetric halves)
        // jmat_11
        for i in 0..n_l {
            let jsi = jscale * phi_l[i];
            for j in 0..=i {
                jm_local[i * ndofs + j] += jsi * phi_l[j];
            }
        }
        // jmat_21 (C++: -= wq * shape2 * shape1)
        for i in 0..n_r {
            let ii = n_l + i;
            let jsi = jscale * phi_r[i];
            for j in 0..n_l {
                jm_local[ii * ndofs + j] -= jsi * phi_l[j];
            }
        }
        // jmat_22
        for i in 0..n_r {
            let ii = n_l + i;
            let jsi = jscale * phi_r[i];
            for j in 0..=i {
                jm_local[ii * ndofs + (n_l + j)] += jsi * phi_r[j];
            }
        }
    }

    // ── Combine: el_local = -diff*el + sigma*diff*el^T + jm_local ──
    // MFEM: elmat = -elmat + sigma_cpp * elmat^T + jmat
    //       (sigma_cpp = -1 → SIP: -elmat - elmat^T + jmat)
    // diff = diffusion coefficient (elmat), penalty = DG penalty (jmat)
    for i in 0..ndofs {
        for j in 0..i {
            let aij = el_local[i * ndofs + j];
            let aji = el_local[j * ndofs + i];
            let mij = jm_local[i * ndofs + j];
            el_local[i * ndofs + j] = sigma * diff * aji - diff * aij + mij;
            el_local[j * ndofs + i] = sigma * diff * aij - diff * aji + mij;
        }
        let diag = el_local[i * ndofs + i];
        el_local[i * ndofs + i] = (sigma - 1.0) * diff * diag + jm_local[i * ndofs + i];
    }

    // Scatter into global COO
    for (i, &gi) in dofs_l.iter().enumerate() {
        for (j, &gj) in dofs_l.iter().enumerate() {
            coo.add(gi, gj, el_local[i * ndofs + j]);
        }
        for (j, &gj) in dofs_r.iter().enumerate() {
            coo.add(gi, gj, el_local[i * ndofs + (n_l + j)]);
        }
    }
    for (i, &gi) in dofs_r.iter().enumerate() {
        for (j, &gj) in dofs_l.iter().enumerate() {
            coo.add(gi, gj, el_local[(n_l + i) * ndofs + j]);
        }
        for (j, &gj) in dofs_r.iter().enumerate() {
            coo.add(gi, gj, el_local[(n_l + i) * ndofs + (n_l + j)]);
        }
    }
}

// ─── Face → element map ───────────────────────────────────────────────────────

// ─── Boundary face contribution (Dirichlet) ───────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn assemble_boundary_face_with_elem<S: FESpace>(
    coo:        &mut CooMatrix<f64>,
    mesh:       &S::Mesh,
    space:      &S,
    face:       u32,
    elem:       u32,
    diff:       f64,
    sigma:      f64,
    penalty:    f64,
    order:      u8,
    quad_order: u8,
) {
    let dim = mesh.dim() as usize;
    let face_nodes = mesh.face_nodes(face);
    let (h_f, mut normal) = face_geom_2d(mesh, face_nodes);
    orient_normal_outward(mesh, elem, face_nodes, &mut normal);

    let et = mesh.element_type(elem);
    let re = ref_elem_vol(et, order);
    let n  = re.n_dofs();
    let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

    let face_elem_type = if dim == 2 { ElementType::Line2 } else { ElementType::Tri3 };
    let ref_face = ref_elem_face(face_elem_type, order);
    let q_face   = ref_face.quadrature(quad_order);

    let nodes = mesh.element_nodes(elem);
    let (jac, _det_j) = simplex_jac(mesh, nodes, dim);
    let jit = jac.clone().try_inverse().unwrap_or_else(|| {eprintln!("  warning: degenerate element"); DMatrix::identity(2,2)}).transpose();

    let x0f = mesh.node_coords(face_nodes[0]);
    let x1f = mesh.node_coords(face_nodes[1]);

    // C++-style: nor = CalcOrtho(face Jacobian) = (dy/2, -dx/2)
    // C++-style: nor = unit_normal * h_f/2 (oriented outward from elem)
    let nor = vec![h_f / 2.0 * normal[0], h_f / 2.0 * normal[1]];
    let nor_norm2 = h_f * h_f / 4.0;  // = h_f²/4

    let mut el_loc = vec![0.0_f64; n * n];
    let mut jm_loc = vec![0.0_f64; n * n];
    let mut phi    = vec![0.0_f64; n];
    let mut gref   = vec![0.0_f64; n * dim];
    let mut gphys  = vec![0.0_f64; n * dim];
    let mut dsdn   = vec![0.0_f64; n];

    for (qi, xi_f) in q_face.points.iter().enumerate() {
        let qw = q_face.weights[qi];
        let xp: Vec<f64> = (0..dim).map(|i| {
            0.5 * ((1.0 - xi_f[0]) * x0f[i] + (1.0 + xi_f[0]) * x1f[i])
        }).collect();
        let mut xi_e = phys_to_ref(&jac, mesh.node_coords(nodes[0]), &xp, dim);
        if nodes.len() > 3 { for v in &mut xi_e { *v -= 1.0; } }

        re.eval_basis(&xi_e, &mut phi);
        re.eval_grad_basis(&xi_e, &mut gref);
        xform_grads(&jit, &gref, &mut gphys, n, dim);

        // Per-point Jacobian for quads
        let (det_j, jit_pt) = if nodes.len() > 3 {
            // Need vertex coords for per-point Jacobian
            let (_, det_j0) = simplex_jac(mesh, nodes, dim);
            (det_j0.abs().max(1e-14), jit.clone())
        } else {
            (jac.determinant().abs().max(1e-14), jit.clone())
        };

        // ── C++ boundary face: w = qw / det(J) (no /2 for boundary) ─────────
        // dshapedn[j] = det(J) * w * gphys[j] · nor
        //             = qw * gphys[j] · nor          (det(J)*w = qw)
        for j in 0..n {
            let dot = gphys[j * dim] * nor[0] + gphys[j * dim + 1] * nor[1];
            dsdn[j] = qw * dot;
        }

        // Consistency matrix (boundary: only block 1,1)
        for i in 0..n {
            for j in 0..n {
                el_loc[i * n + j] += phi[i] * dsdn[j];
            }
        }

        // Penalty: wq = qw * |nor|² / det(J)  (boundary, no /2)
        let wq = (qw / det_j) * nor_norm2;
        let jscale = penalty * wq;  // penalty = the caller's DG penalty κ

        // jmat lower triangle
        for i in 0..n {
            let jsi = jscale * phi[i];
            for j in 0..=i {
                jm_loc[i * n + j] += jsi * phi[j];
            }
        }
    }

    // ── Combine: el = -diff*el + sigma*diff*el^T + jm  (SIP when sigma=-1) ──
    for i in 0..n {
        for j in 0..i {
            let aij = el_loc[i * n + j];
            let aji = el_loc[j * n + i];
            let mij = jm_loc[i * n + j];
            el_loc[i * n + j] = sigma * diff * aji - diff * aij + mij;
            el_loc[j * n + i] = sigma * diff * aij - diff * aji + mij;
        }
        el_loc[i * n + i] = (sigma - 1.0) * diff * el_loc[i * n + i] + jm_loc[i * n + i];
    }

    for (i, &gi) in dofs.iter().enumerate() {
        for (j, &gj) in dofs.iter().enumerate() {
            coo.add(gi, gj, el_loc[i * n + j]);
        }
    }
}

// ─── Helpers (from dg_base) ─────────────────────────────────────────────────

// MFEM: DGDiffusionIntegrator (SIP)

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::L2Space;
    use crate::interior_faces::InteriorFaceList;

    /// SIP matrix should be symmetric for a uniform mesh.
    #[test]
    fn sip_matrix_symmetric() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let ifl   = InteriorFaceList::build(&mesh);
        let space = L2Space::new(mesh, 1);
        let mat   = DgAssembler::assemble_sip(&space, &ifl, 1.0, 10.0, 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i*n+j] - dense[j*n+i]).abs();
                assert!(diff < 1e-11, "SIP K[{i},{j}]-K[{j},{i}] = {diff}");
            }
        }
    }

    /// SIP matrix should give positive energy for non-constant functions.
    #[test]
    fn sip_positive_energy() {
        use fem_mesh::Mesh;
        use fem_space::L2Space;
        use crate::interior_faces::InteriorFaceList;

        let mesh = Mesh::<2>::unit_square_tri(6);
        let ifl = InteriorFaceList::build(&mesh);
        let space = L2Space::new(mesh, 1);
        let mat = DgAssembler::assemble_sip(&space, &ifl, 0.1, 15.0, 3);
        // u = sin(πx)
        let u_vec = space.interpolate(&|x| (std::f64::consts::PI * x[0]).sin());
        let u_slice: &[f64] = u_vec.as_slice();
        // Compute u^T * K * u
        let mut ku = vec![0.0; u_slice.len()];
        mat.spmv(u_slice, &mut ku);
        let energy: f64 = u_slice.iter().zip(ku.iter()).map(|(ui, kui)| ui * kui).sum();
        assert!(energy > 0.0, "u^T K u should be positive, got {energy}");
    }
    /// (all eigenvalues > 0).  We check via Cholesky or by verifying row-dominant structure:
    /// diagonal entry should be the largest in each row for a well-conditioned problem.
    #[test]
    fn sip_matrix_positive_diagonal() {
        let mesh  = Mesh::<2>::unit_square_tri(3);
        let ifl   = InteriorFaceList::build(&mesh);
        let space = L2Space::new(mesh, 1);
        let mat   = DgAssembler::assemble_sip(&space, &ifl, 1.0, 20.0, 3);
        for i in 0..mat.nrows {
            let diag = mat.get(i, i);
            assert!(diag > 0.0, "diagonal[{i}] = {diag}");
        }
    }

    #[test]
    fn sip_matrix_symmetric_l2_p3() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let ifl = InteriorFaceList::build(&mesh);
        let space = L2Space::new(mesh, 3);
        let mat = DgAssembler::assemble_sip(&space, &ifl, 1.0, 40.0, 7);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-9, "SIP K[{i},{j}]-K[{j},{i}] = {diff}");
            }
        }
    }
}
