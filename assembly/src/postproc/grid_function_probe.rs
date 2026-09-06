//! Grid function: a DOF coefficient vector paired with its finite element space.
//!
//! [`GridFunction`] wraps a DOF vector and provides field evaluation, error
//! norms (L², H¹ semi, full H¹), per-element gradient computation, and L²
//! projection from a coefficient.

use nalgebra::DMatrix;

use fem_element::lagrange::{TetP1, TetP2, TriP1};
use fem_element::lagrange::factory::{TriPk, TetPk};
use fem_element::quadrature::quad_rule_01;
use fem_element::{vec_ref_elem, VecFamily, ReferenceElement, QuadratureRule, VectorReferenceElement};
use fem_linalg::CsrMatrix;
use fem_mesh::element_jacobian_at;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::{EdgeKey, HCurlSpace, HDivSpace, L2Space};
use fem_mesh::Mesh;

use crate::assembler::Assembler;
use crate::standard::{DomainSourceIntegrator, MassIntegrator};
use crate::vector_assembler::{self, piola_hcurl_basis, piola_hcurl_curl, piola_hdiv_basis, piola_hdiv_div};

// ─── Reference element factory (mirrors assembler.rs) ──────────────────────

pub(crate) fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, _) | (ElementType::Tri6, _) => Box::new(TriPk::new(order as usize)),
        (ElementType::Quad4, 1) | (ElementType::Quad9, 1) => Box::new(fem_element::lagrange::quad::QuadQ1),
        (ElementType::Quad4, 2) | (ElementType::Quad9, 2) => Box::new(fem_element::lagrange::quad::QuadQ2),
        (ElementType::Quad4, _) | (ElementType::Quad9, _) => Box::new(fem_element::lagrange::quad::QuadQk::new(order as usize)),
        (ElementType::Tet4, 1) | (ElementType::Tet10, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) | (ElementType::Tet10, 2) => Box::new(TetP2),
        (ElementType::Tet4, _) | (ElementType::Tet10, _) => Box::new(TetPk::new(order as usize)),
        _ => panic!("ref_elem_vol: unsupported element type {:?} order {}", elem_type, order),
    }
}

// ─── Surface element factory ───────────────────────────────────────────────

pub(crate) fn ref_elem_surf(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, _) | (ElementType::Tri6, _) => Box::new(TriPk::new(order as usize)),
        (ElementType::Quad4, 1) | (ElementType::Quad9, 1) => Box::new(fem_element::lagrange::quad::QuadQ1),
        (ElementType::Quad4, 2) | (ElementType::Quad9, 2) => Box::new(fem_element::lagrange::quad::QuadQ2),
        (ElementType::Quad4, _) | (ElementType::Quad9, _) => Box::new(fem_element::lagrange::quad::QuadQk::new(order as usize)),
        _ => panic!("ref_elem_surf: unsupported element type {:?} order {}", elem_type, order),
    }
}

// ─