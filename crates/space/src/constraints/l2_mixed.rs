//! D942 — the resident heterogeneous (mixed-geometry) L² space.
//!
//! MFEM's `FiniteElementSpace` over an `L2_FECollection` puts a per-geometry
//! element on every cell and numbers all DOFs element-major consecutively (L²
//! DOFs are element-local, nothing is shared).  A *refined* mesh mixes
//! geometries (a pyramid parent spawns 6 pyramids + 4 tets, D472), and a
//! genuinely mixed coarse mesh (hex next to prism, …) does so a fortiori — the
//! per-geometry DOF counts `(p+1)³` / `(p+1)²(p+2)/2` / `(p+1)(p+2)(p+3)/6` /
//! … cannot be held by the homogeneous [`crate::l2::L2Space`] (one
//! `dofs_per_elem` for the whole mesh; the d340 fixture pins MFEM's
//! `6·(p+1)³ + 4·tet` count on the refined pyramid mesh).
//!
//! [`MixedL2Space`] is that space: per-element reference-element dispatch by
//! geometry and basis, element-major consecutive numbering, DOF nodes mapped
//! through the mesh's own geometry (the same authoritative
//! `element_jacobian_at` map every space builder uses).  It plugs into the L²
//! prolongation builder through the [`L2ProlongationSpace`] read interface
//! declared in [`crate::constraints::prolong`].

use fem_core::types::DofId;
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;

use crate::constraints::prolong::L2ProlongationSpace;
use crate::l2::L2Basis;

/// Per-geometry reference element of the mixed L² space — MFEM's
/// `L2_FECollection(p, dim, btype)` element table for the geometries fem-rs
/// meshes carry (fe_coll.cpp:2186: `b_type` is forwarded to every
/// per-geometry `L2_*Element(p, btype)` constructor; the pyramid keeps the
/// default `pyr_type = 1`, i.e. the Fuentes arm, exactly as
/// [`crate::l2::l2_pyramid_element`] pins it).
fn l2_mixed_ref_element(
    dim: usize,
    et: fem_mesh::ElementType,
    p: usize,
    basis: L2Basis,
) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::factory::{HexQk, QuadQk};
    use fem_element::lagrange::{HexL2GL, QuadL2GL, TetL2GL, TriL2GL, WedgeL2};
    match (dim, et) {
        (3, fem_mesh::ElementType::Hex8) => match basis {
            L2Basis::GaussLegendre => Box::new(HexL2GL::new(p)),
            L2Basis::GaussLobatto => Box::new(HexQk::new_lex(p)),
        },
        (3, fem_mesh::ElementType::Pyramid5) => crate::l2::l2_pyramid_element(p, basis),
        (3, fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10) => match basis {
            L2Basis::GaussLegendre => Box::new(TetL2GL::new(p)),
            L2Basis::GaussLobatto => Box::new(TetL2GL::new_gauss_lobatto(p)),
        },
        // D940: MFEM `L2_WedgeElement(p, btype)`.
        (3, fem_mesh::ElementType::Prism6) => match basis {
            L2Basis::GaussLegendre => Box::new(WedgeL2::new(p)),
            L2Basis::GaussLobatto => Box::new(WedgeL2::new_gauss_lobatto(p)),
        },
        (2, fem_mesh::ElementType::Quad4) => match basis {
            L2Basis::GaussLegendre => Box::new(QuadL2GL::new(p)),
            L2Basis::GaussLobatto => Box::new(QuadQk::new_lex(p)),
        },
        (2, fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6) => match basis {
            L2Basis::GaussLegendre => Box::new(TriL2GL::new(p)),
            L2Basis::GaussLobatto => Box::new(TriL2GL::new_gauss_lobatto(p)),
        },
        other => panic!(
            "MixedL2Space: unsupported geometry {other:?} (order {p}, basis {basis:?})"
        ),
    }
}

/// Heterogeneous (mixed-geometry) scalar L² space, order `p >= 0`, either
/// [`L2Basis`], over any 2-D/3-D fem-rs mesh (tet/hex/prism/pyramid, tri/quad,
/// homogeneous or mixed).  DOF numbering is element-major consecutive — MFEM's
/// layout (pinned against MFEM `GetVSize` by the d105 fixtures on the refined
/// pyramid mesh, `CSIZE/FSIZE`, and by d340 for the homogeneous-pyramid case).
pub struct MixedL2Space<M: MeshTopology> {
    mesh: M,
    order: u8,
    basis: L2Basis,
    n_dofs: usize,
    /// Element-major consecutive global DOF ids per element.
    elem_dofs: Vec<Vec<DofId>>,
    /// DOF node coordinates (flat, `n_dofs * dim`).
    dof_coords: Vec<f64>,
}

impl<M: MeshTopology> MixedL2Space<M> {
    /// The mixed L² space of `order` over `mesh` with the default
    /// Gauss-Legendre `L2_FECollection` basis.
    pub fn new(mesh: M, order: u8) -> Self {
        Self::new_with_basis(mesh, order, L2Basis::GaussLegendre)
    }

    /// The mixed L² space with an explicit basis node placement.
    pub fn new_with_basis(mesh: M, order: u8, basis: L2Basis) -> Self {
        let dim = mesh.dim() as usize;
        let n_elems = mesh.n_elements();

        if order == 0 {
            // P0: one DOF per element at the corner centroid — geometry
            // independent (MFEM's PointFiniteElement arm for order 0).
            let mut elem_dofs = Vec::with_capacity(n_elems);
            let mut dof_coords = vec![0.0_f64; n_elems * dim];
            for e in 0..n_elems as u32 {
                let nodes = mesh.element_nodes(e);
                let mut cen = [0.0_f64; 3];
                for &n in nodes {
                    let c = mesh.node_coords(n);
                    for d in 0..dim {
                        cen[d] += c[d];
                    }
                }
                let nv = nodes.len() as f64;
                for d in 0..dim {
                    dof_coords[e as usize * dim + d] = cen[d] / nv;
                }
                elem_dofs.push(vec![e as DofId]);
            }
            return MixedL2Space {
                mesh,
                order,
                basis,
                n_dofs: n_elems,
                elem_dofs,
                dof_coords,
            };
        }

        let p = order as usize;
        let mut elem_dofs = Vec::with_capacity(n_elems);
        let mut dof_coords = Vec::new();
        let mut next = 0usize;
        for e in 0..n_elems as u32 {
            let re = l2_mixed_ref_element(dim, mesh.element_type(e), p, basis);
            let mut dofs = Vec::with_capacity(re.n_dofs());
            for rc in re.dof_coords() {
                dofs.push(next as DofId);
                next += 1;
                // The geometry sampler speaks the *mesh's* reference frame per
                // geometry; the reference elements all speak MFEM's.  These
                // differ only on the prism: MFEM's `Geometry::PRISM` frame
                // (and [`WedgeL2`]'s) is (tri_x, tri_y, layer) while the mesh
                // geometry element is (layer, tri_eta, tri_zeta) — see the
                // axis-convention note in `fem_mesh::transformation`.
                let rc_mesh: Vec<f64> = if mesh.element_type(e) == fem_mesh::ElementType::Prism6 {
                    vec![rc[2], rc[0], rc[1]]
                } else {
                    rc
                };
                let (_, x) =
                    fem_mesh::transformation::element_jacobian_at(&mesh, e, &rc_mesh, dim);
                dof_coords.extend_from_slice(&x[..dim]);
            }
            elem_dofs.push(dofs);
        }
        MixedL2Space { mesh, order, basis, n_dofs: next, elem_dofs, dof_coords }
    }

    /// The basis the space was built with.
    pub fn basis(&self) -> L2Basis {
        self.basis
    }
}

impl<M: MeshTopology> L2ProlongationSpace<M> for MixedL2Space<M> {
    fn order(&self) -> u8 {
        self.order
    }
    fn l2_basis(&self) -> Option<L2Basis> {
        Some(self.basis)
    }
    fn mesh(&self) -> &M {
        &self.mesh
    }
    fn n_dofs(&self) -> usize {
        self.n_dofs
    }
    fn element_dofs(&self, e: u32) -> &[DofId] {
        &self.elem_dofs[e as usize]
    }
    fn dof_coords(&self) -> &[f64] {
        &self.dof_coords
    }
}
