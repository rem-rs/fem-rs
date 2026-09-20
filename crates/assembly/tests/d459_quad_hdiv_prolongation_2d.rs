//! D459 — the 2-D branch of `build_prolongation_hdiv` must walk each
//! element's *own* edges, not the triangle edge table.
//!
//! The 2-D walk used the hardcoded triangle edge list `[(0,1),(1,2),(0,2)]`
//! for every element.  On a quad element that third pair `(0,2)` is the
//! **diagonal**, and the real edges `(2,3)` (top) and `(3,0)` (left) are never
//! visited.  A boundary edge belongs to exactly one element, so on a quad
//! mesh the top-row/top and left-column/left fine boundary edges never reach
//! the parent search: their HDiv face dofs keep all-zero prolongation rows
//! and the mesh-boundary normal trace is silently dropped.  (Interior edges
//! survive only because a neighbouring element happens to list them as its
//! `(0,1)`/`(1,2)`.)
//!
//! Red observable: on a 2×2 unit-square quad mesh refined once (4×4) with
//! RT1, the fine boundary edge dofs on the top and left boundary have empty
//! prolongation rows.  The fix mirrors the 3-D `hdiv_element_faces_3d`
//! shape-driven walk as a 2-D edge table (`Quad4` = the four boundary edges,
//! matching `HDivSpace`'s `QUAD_FACES`).
//!
//! The triangle path is pinned bit-for-bit by the same coverage assertion
//! (it holds before and after) and by `transfer.rs`'s
//! `prolongation_hdiv_rt{0,1}_2d` tests.

use fem_assembly::build_prolongation_hdiv;
use fem_mesh::amr::refine_uniform;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::dof_manager::EdgeKey;
use fem_space::fe_space::FESpace;
use fem_space::HDivSpace;
use std::collections::HashMap;

/// Fine-mesh edges lying on exactly one element (the mesh boundary).
fn boundary_edges(mesh: &Mesh<2>) -> Vec<EdgeKey> {
    let mut incidence: HashMap<EdgeKey, u32> = HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let n = mesh.element_nodes(e);
        let edges: &[(usize, usize)] = match mesh.element_type(e) {
            ElementType::Quad4 => &[(0, 1), (1, 2), (2, 3), (3, 0)],
            _ => &[(0, 1), (1, 2), (0, 2)],
        };
        for &(i, j) in edges {
            *incidence.entry(EdgeKey::new(n[i], n[j])).or_insert(0) += 1;
        }
    }
    incidence
        .into_iter()
        .filter(|&(_, c)| c == 1)
        .map(|(k, _)| k)
        .collect()
}

/// Number of fine boundary-edge dofs whose prolongation row is empty.
fn missing_boundary_rows(mesh: Mesh<2>, order: u8) -> (usize, usize) {
    let coarse = HDivSpace::new(mesh, order);
    let fine_mesh = refine_uniform(&coarse.mesh().clone());
    let fine = HDivSpace::new(fine_mesh, order);
    let (p, _stats) = build_prolongation_hdiv(&coarse, &fine);

    let mut missing = 0usize;
    let mut total = 0usize;
    let fine_mesh = fine.mesh().clone();
    for ek in boundary_edges(&fine_mesh) {
        let dofs = fine.edge_face_dofs(ek).expect("boundary edge must carry dofs");
        for d in dofs {
            let r = d as usize;
            total += 1;
            if p.row_ptr[r] == p.row_ptr[r + 1] {
                missing += 1;
            }
        }
    }
    (missing, total)
}

/// D459: on a quad mesh every fine boundary-edge dof must receive a
/// prolongation row.  Before the fix the top/left boundary edges (2,3)/(3,0)
/// were never walked — 8 of the 16 boundary edges (16 dofs at RT1) had empty
/// rows.
#[test]
fn d459_quad_rt1_boundary_edge_dofs_are_prologated() {
    let (missing, total) = missing_boundary_rows(Mesh::<2>::unit_square_quad(2), 1);
    assert_eq!(total, 32, "4x4 quad mesh: 16 boundary edges x 2 RT1 dofs");
    assert_eq!(
        missing, 0,
        "{missing}/{} boundary-edge dofs have no prolongation row — the 2-D \
         walk used the triangle edge table on quads",
        total
    );
}

/// Same coverage at RT0 (1 dof per boundary edge).
#[test]
fn d459_quad_rt0_boundary_edge_dofs_are_prologated() {
    let (missing, total) = missing_boundary_rows(Mesh::<2>::unit_square_quad(2), 0);
    assert_eq!(total, 16, "4x4 quad mesh: 16 boundary edges x 1 RT0 dof");
    assert_eq!(missing, 0, "{missing}/{total} RT0 boundary dofs unreachable");
}

/// Triangle control: the tri edge walk already reaches every boundary
/// half-edge (as `(0,1)`/`(1,2)`/`(0,2)` of some child), so this holds before
/// *and* after the fix — the tri path is pinned bit-for-bit.
#[test]
fn d459_tri_control_boundary_coverage_unchanged() {
    let (missing, total) = missing_boundary_rows(Mesh::<2>::unit_square_tri(2), 1);
    assert_eq!(missing, 0, "{missing}/{total} tri boundary dofs unreachable");
}
