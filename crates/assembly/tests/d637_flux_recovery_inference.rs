//! D637 — flux-recovery `n_flux_dofs` inference + explicit rejection of the
//! unwired canonical-rotation cases.
//!
//! Two defects, one guard:
//!
//! 1. `zz_estimator_mfem_impl` sampled the flux at **element 0's** H¹ dof
//!    coordinates and sized every element's flux vector with element 0's H¹
//!    slot count.  On a mixed mesh (or any space whose `element_dofs(e).len()`
//!    is not the H¹ slot count — e.g. the D613 wildcard-numbered labels or an
//!    RT/HDiv space) the per-element flux rows and the element's dof table
//!    decouple: hex-first meshes silently mis-sample every tet, tet-first
//!    meshes index out of bounds.  MFEM semantics: the flux vector has one
//!    slot per *that element's* flux-space DOF.
//! 2. A space that reports canonical face-block rotations
//!    (`element_face_blocks`, non-empty) or one whose flux table is not the
//!    H¹ family's (RT/HDiv — whose hex p ≥ 1 shared-quad-face canonical
//!    rotation is unwired, d448 covered 2-D only) must be **rejected
//!    loudly**, never silently averaged in the element-local convention.
//!
//! Green contract established here:
//! * per-element η on a mixed hex+tet mesh is independent of the element
//!   numbering (each element sampled in its own family at its own dof coords);
//! * an HDiv (RT0 hex) space passed to `zz_estimator_mfem` panics with the
//!   D637 message instead of producing silently wrong η;
//! * a space claiming non-empty `element_face_blocks` is rejected the same way.

use fem_assembly::postproc::flux_recovery::zz_estimator_mfem;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::DiffusionIntegrator;
use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{hcurl::FaceDofBlock, hdiv::HDivSpace, H1Space};

/// Two face-sharing unit hexes `[0,1]³`, `[1,2]×[0,1]²` plus one disjoint
/// unit tet in `[4,5]×[0,1]²`, in the given element order.
fn mixed_mesh(hex_first: bool) -> Mesh<3> {
    #[rustfmt::skip]
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 1.0, 0.0,  0.0, 1.0, 0.0, // 0..3 hex A base
        0.0, 0.0, 1.0,  1.0, 0.0, 1.0,  1.0, 1.0, 1.0,  0.0, 1.0, 1.0, // 4..7 hex A top
        2.0, 0.0, 0.0,  2.0, 1.0, 0.0,  2.0, 1.0, 1.0,  2.0, 0.0, 1.0, // 8..11 hex B top (x=2)
        4.0, 0.0, 0.0,  5.0, 0.0, 0.0,  4.0, 1.0, 0.0,  4.0, 0.0, 1.0, // 12..15 tet
    ];
    // Hex A: unit cube.  Hex B: [1,2]×[0,1]², sharing nodes 1,2,5,6 on x=1.
    let hex_a = [0u32, 1, 2, 3, 4, 5, 6, 7];
    let hex_b = [1u32, 8, 9, 2, 5, 10, 11, 6];
    let tet = [12u32, 13, 14, 15];
    let hexes = [hex_a.as_slice(), hex_b.as_slice()].concat();
    let (types, offsets, conn): (Vec<ElementType>, Vec<usize>, Vec<u32>) = if hex_first {
        (
            vec![ElementType::Hex8, ElementType::Hex8, ElementType::Tet4],
            vec![0, 8, 16, 20],
            [hexes.as_slice(), tet.as_slice()].concat(),
        )
    } else {
        (
            vec![ElementType::Tet4, ElementType::Hex8, ElementType::Hex8],
            vec![0, 4, 12, 20],
            [tet.as_slice(), hexes.as_slice()].concat(),
        )
    };
    Mesh {
        coords,
        conn,
        elem_tags: vec![1, 1, 1],
        elem_type: ElementType::Hex8,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Tri3,
        elem_types: Some(types),
        elem_offsets: Some(offsets),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    }
}

/// η per element must not depend on the element numbering: each element is
/// sampled in its own family, at its own H¹ dof coordinates.
#[test]
fn d637_mixed_mesh_eta_is_order_independent() {
    let int = DiffusionIntegrator { kappa: 1.0 };

    let mesh_a = mixed_mesh(true); // e0,e1 = hexes, e2 = tet
    let s_a = H1Space::new(mesh_a.clone(), 1);
    let d_a = s_a.interpolate(&|x| x[0] * x[0]);
    let gf_a = GridFunction::new(&s_a, d_a.as_slice().to_vec());
    let eta_a = zz_estimator_mfem(&gf_a, &int).eta;

    let mesh_b = mixed_mesh(false); // e0 = tet, e1,e2 = hexes
    let s_b = H1Space::new(mesh_b.clone(), 1);
    let d_b = s_b.interpolate(&|x| x[0] * x[0]);
    let gf_b = GridFunction::new(&s_b, d_b.as_slice().to_vec());
    let eta_b = zz_estimator_mfem(&gf_b, &int).eta;

    // The two hexes share nodes, so a healthy recovery gives them strictly
    // positive η (the shared-face gradient average differs from each side).
    let (hex_a1, hex_a2, tet_a) = (eta_a[0], eta_a[1], eta_a[2]);
    let (tet_b, hex_b1, hex_b2) = (eta_b[0], eta_b[1], eta_b[2]);
    assert!(hex_a1 > 1e-8 && hex_a2 > 1e-8, "hex η {hex_a1} / {hex_a2}");
    assert!(
        (hex_a1 - hex_b1).abs() <= 1e-10 * hex_a1.max(1.0)
            && (hex_a2 - hex_b2).abs() <= 1e-10 * hex_a2.max(1.0),
        "hex η differs across orderings: {hex_a1}/{hex_a2} vs {hex_b1}/{hex_b2}"
    );
    assert!(
        (tet_a - tet_b).abs() <= 1e-10 * tet_a.abs().max(1.0),
        "tet η differs across orderings: {tet_a} vs {tet_b} — the tet was \
         sampled at element 0's (hex) flux dof coordinates"
    );
}

fn panic_message<T: std::fmt::Debug>(r: Result<T, Box<dyn std::any::Any + Send>>) -> String {
    let payload = r.expect_err("expected the D637 explicit rejection");
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        panic!("panic payload was not a string");
    }
}

/// An RT0-hex H(div) space (6 flux dofs/element against the 8-slot H¹ hex
/// flux family, plus the unwired shared-quad-face canonical rotations) must
/// be rejected loudly, not silently mis-indexed into a wrong η.
#[test]
fn d637_hdiv_flux_space_rejected_explicitly() {
    let m = Mesh::<3>::unit_cube_hex(1);
    let space = HDivSpace::new(m, 0);
    let dofs = vec![0.0; space.n_dofs()];
    let gf = GridFunction::new(&space, dofs);
    let int = DiffusionIntegrator { kappa: 1.0 };
    let msg = panic_message(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || zz_estimator_mfem(&gf, &int),
    )));
    assert!(
        msg.contains("D637"),
        "rejection must carry the D637 tag: {msg}"
    );
}

/// A space that reports canonical face-block rotations (`element_face_blocks`
/// non-empty) is averaged by this recovery in the element-local convention —
/// the rotation would be silently dropped.  Reject.
struct FaceBlockSpace(H1Space<Mesh<3>>);

impl FESpace for FaceBlockSpace {
    type Mesh = Mesh<3>;

    fn mesh(&self) -> &Mesh<3> { self.0.mesh() }
    fn n_dofs(&self) -> usize { self.0.n_dofs() }
    fn element_dofs(&self, elem: u32) -> &[DofId] { self.0.element_dofs(elem) }
    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> { self.0.interpolate(f) }
    fn space_type(&self) -> SpaceType { self.0.space_type() }
    fn order(&self) -> u8 { self.0.order() }
    fn element_face_blocks(&self, _elem: u32) -> &[FaceDofBlock] {
        // A non-empty table whose rotation this recovery cannot honour.
        const BLOCKS: &[FaceDofBlock] = &[FaceDofBlock {
            slot: 0,
            canon_dofs: [0, 1],
            s: [[1.0, 0.0], [0.0, 1.0]],
        }];
        BLOCKS
    }
}

#[test]
fn d637_face_block_space_rejected_explicitly() {
    let m = Mesh::<3>::unit_cube_hex(1);
    let space = FaceBlockSpace(H1Space::new(m, 1));
    let d = space.interpolate(&|x| x[0]);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());
    let int = DiffusionIntegrator { kappa: 1.0 };
    let msg = panic_message(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || zz_estimator_mfem(&gf, &int),
    )));
    assert!(
        msg.contains("D637") && msg.contains("face"),
        "rejection must name the unwired face-block rotation: {msg}"
    );
}
