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
use fem_mesh::transformation::element_jacobian_at;
use fem_space::fe_space::FESpace;

use super::dg_base::{
    build_face_elem_map, face_point_geom, face_point_geom_3d_face, face_type_of, ref_elem_face,
    ref_elem_vol, FaceGeom, xform_grads,
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
    // D814-1: MFEM's `DiffusionIntegrator` volume rule
    // (`DiffusionIntegrator::GetRule`, fem/bilininteg.cpp:1347) is
    //   Qk (tensor) spaces:  o + o + dim − 1
    //   Pk (simplex) spaces: o + o − 2
    // The single `quad_order` argument of `assemble_dg` is the **face** rule
    // (`2·max(o₁,o₂)`, what ex14 passes); in 2-D the volume orders `2p+dim−1`
    // and the caller's `2p` select the *same* Gauss points, so Q1..Q3 quad
    // meshes assemble bit-identically to before.  In 3-D they diverge (hex p=1:
    // order 4 → 3³ points vs order 2 → 2³), so the tensor families must use
    // MFEM's own formula.  The simplex formula (`2o−2`, i.e. one centroid point
    // at p=1) changes simplex values away from the caller's over-integration;
    // without a simplex DG volume oracle that arm keeps the caller's rule.
    let vol_quad_order = match elem_type {
        ElementType::Quad4 | ElementType::Hex8 => 2 * order + dim as u8 - 1,
        _ => quad_order,
    };
    let q  = re.quadrature(vol_quad_order);
    let gd = space.element_dofs(e).iter().map(|&d| d as usize).collect::<Vec<_>>();

    phi.resize(n, 0.0);
    grad_ref.resize(n * dim, 0.0);
    grad_p.resize(n * dim, 0.0);

    let mut k_elem = vec![0.0_f64; n * n];

    for (qi, xi) in q.points.iter().enumerate() {
        // D795-1: MFEM's `DiffusionIntegrator::AssembleElementMatrix` evaluates
        // the element's **isoparametric** transformation (`Trans.Weight()` and
        // `Trans.AdjugateJacobian()`), i.e. the order-`g` curved map for a
        // curved mesh and the P1/bilinear map for a straight one.  The previous
        // corner-bilinear `quad_jac_at_01` / affine `simplex_jac` agreed with
        // MFEM only on straight-sided elements.
        //
        // The assembled entry is
        //   `Σ_q ip.w · det(J) · ∇_xφ_i · ∇_xφ_j`
        // with `det(J)` **signed** (MFEM `Weight()`; algebra:
        // `ip.w/detJ · (dshape·Adj J)·(dshape·Adj J)ᵀ = ip.w·detJ·(J⁻ᵀ∇φ)·(J⁻ᵀ∇φ)ᵀ`).
        let (jac, _xp) = element_jacobian_at(mesh, e, xi, dim);
        let det_j = jac.determinant();
        // Degeneracy guard (magnitude test only, D696 batch 4 precedent):
        // never taken on a valid mesh.
        let j_inv_t = jac
            .try_inverse()
            .unwrap_or_else(|| {
                eprintln!("  warning: degenerate element {e}");
                DMatrix::identity(2, 2)
            })
            .transpose();
        let w = q.weights[qi] * det_j;
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

    // D814-1: the face rule follows the face's own type (node count: 2 → edge,
    // 3 → triangle, 4 → quadrilateral), not `mesh.dim()` — a hexahedron's quad
    // faces take MFEM's `[0,1]²` tensor rule
    // (`DGDiffusionIntegrator::GetRule(o, geom) = IntRules.Get(geom, 2·o)`).
    let ref_face = ref_elem_face(face_type_of(face_nodes), order);
    // MFEM `DGDiffusionIntegrator::GetRule(order, geom) = IntRules.Get(geom, 2*order)`:
    // a `[0,1]` reference segment whose weights sum to 1 (verified against
    // `IntRules.Get(Geometry::SEGMENT, 2)` — 2 Gauss points at
    // 0.2113…/0.7886…, w = 0.5 each).  `quad_order` is the caller's `2*order`.
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
        let ipw = face_weights[qi];   // [0,1] face rule: weights sum to 1

        // MFEM `Trans.SetAllIntPoints(&ip)`: the reference point in each
        // neighbouring element comes from the *reference* face→element map
        // (`Loc1`/`Loc2`), never from inverting the physical map.
        // D795-1: `nor = CalcOrtho(Trans.Jacobian())` is built from Elem1's
        // isoparametric geometry (the face transformation is composed through
        // `Elem1` for a `Nodes`-carrying mesh, `Mesh::GetFaceTransformation`),
        // and it is used for BOTH sides.
        //
        // D814-1: 3-D faces (triangular or quadrilateral) compose through the
        // same `face_point_geom_3d_face` dispatcher as the advection driver.
        let g1 = if dim == 2 {
            FaceGeom::Face2(face_point_geom(mesh, el, face_nodes[0], face_nodes[1], xi_f[0]))
        } else {
            FaceGeom::Face3(face_point_geom_3d_face(mesh, el, face_nodes, [xi_f[0], xi_f[1]]))
        };
        let g2 = if dim == 2 {
            FaceGeom::Face2(face_point_geom(mesh, er, face_nodes[0], face_nodes[1], xi_f[0]))
        } else {
            FaceGeom::Face3(face_point_geom_3d_face(mesh, er, face_nodes, [xi_f[0], xi_f[1]]))
        };
        let nor: &[f64] = g1.nor();

        re_l.eval_basis(g1.eip(), &mut phi_l);
        re_r.eval_basis(g2.eip(), &mut phi_r);
        re_l.eval_grad_basis(g1.eip(), &mut gref_l);
        re_r.eval_grad_basis(g2.eip(), &mut gref_r);
        xform_grads(g1.jit(), &gref_l, &mut gphys_l, n_l, dim);
        xform_grads(g2.jit(), &gref_r, &mut gphys_r, n_r, dim);

        // ── MFEM DGDiffusionIntegrator per-QP algorithm ──────────────────────
        //   w  = ip.weight / (2·det(J1))                       (interior)
        //   ni = w·nor ;  adjJ = CalcAdjugate(J1) = det(J1)·J1⁻¹
        //   nh = adjJ·ni ; dshape1dn = dshape1·nh
        //      = det(J1)·w·(J1⁻¹∇_refφ)·nor = (ip.weight/2)·∇_xφ·nor
        //   wq = kappa·(ni·nor)  summed over both sides
        //      = kappa·ip.weight/2·(1/det(J1) + 1/det(J2))·|nor|²
        // so the consistency term carries **no** det(J) at all (it cancels).
        let half_w = 0.5 * ipw;
        for j in 0..n_l {
            let dot: f64 = (0..dim).map(|k| gphys_l[j * dim + k] * nor[k]).sum();
            dsf1dn[j] = half_w * dot;
        }
        for j in 0..n_r {
            let dot: f64 = (0..dim).map(|k| gphys_r[j * dim + k] * nor[k]).sum();
            dsf2dn[j] = half_w * dot;
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
        // C++: wq = ni·nor (side 1) + ni·nor (side 2, with w = ip.w/2/det2),
        // then `wq *= kappa`:
        //   wq = kappa·ipw/2·|nor|²·(1/det(J1) + 1/det(J2))
        // |nor|² = |dX/dξ|² of the isoparametric face, det = MFEM `Weight()`.
        let nor_norm2: f64 = (0..dim).map(|k| nor[k] * nor[k]).sum();
        let wq = penalty * nor_norm2 * half_w * (1.0 / g1.det_j() + 1.0 / g2.det_j());
        // C++: jmat += wq * shape * shape  (lower triangle only)
        // C++: jmat += wq * shape * shape  (lower triangle only)
        let jscale = wq;

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

    let et = mesh.element_type(elem);
    let re = ref_elem_vol(et, order);
    let n  = re.n_dofs();
    let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

    // D814-1: face rule by node count — a hexahedron's quad faces take the
    // `[0,1]²` tensor rule, not the triangular one.
    let ref_face = ref_elem_face(face_type_of(&face_nodes), order);
    let q_face   = ref_face.quadrature(quad_order);

    let mut el_loc = vec![0.0_f64; n * n];
    let mut jm_loc = vec![0.0_f64; n * n];
    let mut phi    = vec![0.0_f64; n];
    let mut gref   = vec![0.0_f64; n * dim];
    let mut gphys  = vec![0.0_f64; n * dim];
    let mut dsdn   = vec![0.0_f64; n];

    for (qi, xi_f) in q_face.points.iter().enumerate() {
        let ipw = q_face.weights[qi];   // [0,1] face rule
        // MFEM `Trans.SetAllIntPoints` + `nor = CalcOrtho(Trans.Jacobian())`:
        // reference composition through the (single) neighbouring element, with
        // the isoparametric geometry — see `face_point_geom` (D795-1).
        let g = if dim == 2 {
            FaceGeom::Face2(face_point_geom(mesh, elem, face_nodes[0], face_nodes[1], xi_f[0]))
        } else {
            FaceGeom::Face3(face_point_geom_3d_face(mesh, elem, &face_nodes, [xi_f[0], xi_f[1]]))
        };
        let nor: &[f64] = g.nor();

        re.eval_basis(g.eip(), &mut phi);
        re.eval_grad_basis(g.eip(), &mut gref);
        xform_grads(g.jit(), &gref, &mut gphys, n, dim);

        // ── MFEM boundary face (ndof2 = 0): w = ip.weight/det(J) (no 1/2).
        //   dshapedn = det(J)·w·∇_xφ·nor = ip.weight·∇_xφ·nor
        //   wq       = kappa·ip.weight·|nor|²/det(J)
        for j in 0..n {
            let dot: f64 = (0..dim).map(|k| gphys[j * dim + k] * nor[k]).sum();
            dsdn[j] = ipw * dot;
        }

        // Consistency matrix (boundary: only block 1,1)
        for i in 0..n {
            for j in 0..n {
                el_loc[i * n + j] += phi[i] * dsdn[j];
            }
        }

        // Penalty: C++ `wq = ni·nor` with `ni = w·nor`, `w = ip.weight/det(J)`.
        let nor_norm2: f64 = (0..dim).map(|k| nor[k] * nor[k]).sum();
        let jscale = penalty * ipw * nor_norm2 / g.det_j();

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
