//! D394 — prism and pyramid HDiv spaces must size face blocks by the face
//! shape and match the assembly reference elements' per-element dof counts.
//!
//! Debt: `build_3d_prism`/`build_3d_pyramid` allocated `order + 1` dofs on
//! every face and no interiors, so RT1 prism/pyramid spaces had 10 slots per
//! element while `PrismRTk(1)`/`PyraRTk(1)` expose 18/17 basis functions —
//! the assembler would pair slots with the wrong basis functions.
//!
//! MFEM anchors:
//! - `RT_WedgeElement(p)` (`fem/fe/fe_rt.cpp:1082-1086`): per element
//!   `(p+1)(p+2)/2 · (p+2)` vertical + `(p+1)(p+2)/2 · (p+1)` horizontal
//!   dofs, i.e. the 2 triangular faces carry `(k+1)(k+2)/2` and the 3
//!   quadrilateral faces `(k+1)^2` face dofs, interior `p(p+1)(3p+4)/2`
//!   (= 7 at k=1; `RT_dof[PRISM]`, `fe_coll.cpp:2581-2583`).
//! - MFEM has no classical RT pyramid; `RT_FuentesPyramidElement(p)`
//!   (`fe_rt.cpp:1273-1275`) counts `(p+1)(3p(p+2)+5)` = 28 at k=1.  The
//!   fem-rs element layer (`PyraRTk`, `raviart_thomas/pyramid.rs:192-206`)
//!   is its own construction: 4 tri faces `(k+1)(k+2)/2` + 1 quad face
//!   `(k+1)^2` + interior {k=0→0, k=1→1}.  D394 aligns the SPACE with the
//!   fem-rs element (the count invariant); the gap to MFEM's Fuentes
//!   pyramid is element-layer debt (D437 family).
//!
//! Prism-stack oracle (MFEM `data/d394_prism_stack.mesh`, 2 stacked wedges,
//! 3 tri + 6 quad faces, 8 boundary faces): RT0 vsize=9/ess=8 (MFEM
//! `GetVSize`/`GetBoundaryTrueDofs`).  RT1: ess=30 = MFEM (only face dofs
//! are essential); fem-rs vsize target 33 = 3·3 + 6·4 face dofs, no
//! interiors (MFEM vsize 47 adds 2·7 wedge interiors — element-layer D436).

use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::{FESpace, HDivSpace};
use fem_space::dof_manager::FaceKey;

fn load(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

#[test]
fn prism_stack_rt0_matches_mfem() {
    let mesh = load("data/d394_prism_stack.mesh");
    let all_tags = mesh.unique_boundary_tags();
    let space = HDivSpace::new(mesh.clone(), 0);
    assert_eq!(space.n_dofs(), 9, "RT0: one dof per unique face");
    assert_eq!(
        boundary_dofs_hdiv(space.mesh(), &space, &all_tags).len(),
        8,
        "RT0: one essential dof per boundary face"
    );
    // RT0 bit-identity pin (D394): dof ids follow first-encounter face order.
    // Elem 0 (verts 0..5): bottom, top, q0, q1, q2 — all first-seen → 0..5.
    // Elem 1 (verts 3..8): bottom = shared top-of-elem0 (dof 1), then its own
    // top + 3 quads in face-table order.
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3, 4]);
    assert_eq!(space.element_dofs(1), &[1, 5, 6, 7, 8]);
}

#[test]
fn prism_stack_rt1_face_blocks_and_ess_match_mfem() {
    let mesh = load("data/d394_prism_stack.mesh");
    let all_tags = mesh.unique_boundary_tags();
    let space = HDivSpace::new(mesh.clone(), 1);
    // 3 tri faces x 3 + 6 quad faces x 4 = 33 face dofs; no interior at the
    // capped order (PrismRTk(1) has none — D436 notes the MFEM difference).
    assert_eq!(space.n_dofs(), 33, "RT1 prism-stack vsize");
    assert_eq!(
        boundary_dofs_hdiv(space.mesh(), &space, &all_tags).len(),
        30,
        "RT1 essential dofs (all face dofs of the 8 boundary faces) match MFEM"
    );

    // The two prisms (elem 0: verts 0..5, elem 1: verts 3..8) share tri
    // face {3,4,5}.
    let shared = FaceKey::new(3, 4, 5);
    let block = space.face_dofs(shared).expect("shared tri face missing");
    assert_eq!(block.len(), 3, "tri face carries (k+1)(k+2)/2 = 3 dofs");
    let quad = space
        .face_dofs(FaceKey::new(0, 1, 3)) // quad {0,1,4,3}: sorted-4 first-3 key
        .expect("quad face missing");
    assert_eq!(quad.len(), 4, "quad face carries (k+1)^2 = 4 dofs");
}

#[test]
fn prism_slot_count_matches_prismrtk() {
    use fem_element::raviart_thomas::PrismRTk;
    use fem_element::VectorReferenceElement;

    let mesh = load("data/d394_prism_stack.mesh");
    for k in 0..=1u8 {
        let space = HDivSpace::new(mesh.clone(), k);
        let n_ref = PrismRTk::new(k as usize).n_dofs();
        for e in 0..mesh.n_elements() as u32 {
            assert_eq!(
                space.element_dofs(e).len(),
                n_ref,
                "k={k} elem {e}: space slots must equal PrismRTk({k}).n_dofs()"
            );
        }
    }
}

#[test]
fn prism_shared_face_is_conforming() {
    let mesh = load("data/d394_prism_stack.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let block = space.face_dofs(FaceKey::new(3, 4, 5)).expect("shared face");
    let mut a: Vec<_> = space.element_dofs(0).to_vec();
    let mut b: Vec<_> = space.element_dofs(1).to_vec();
    a.sort_unstable();
    b.sort_unstable();
    let mut inter: Vec<_> = a.iter().copied().collect();
    inter.retain(|d| b.binary_search(d).is_ok());
    let mut block_sorted = block.clone();
    block_sorted.sort_unstable();
    assert_eq!(inter, block_sorted, "shared tri face dofs = exact common set");
}

#[test]
fn pyramid_matches_pyra_rtk_layout() {
    use fem_element::raviart_thomas::PyraRTk;
    use fem_element::VectorReferenceElement;

    let mesh = load("tmp/d392/one_pyramid.mesh");
    let all_tags = mesh.unique_boundary_tags();

    // RT0: 5 face dofs, all essential.
    let space0 = HDivSpace::new(mesh.clone(), 0);
    assert_eq!(space0.n_dofs(), 5);
    assert_eq!(boundary_dofs_hdiv(space0.mesh(), &space0, &all_tags).len(), 5);

    // RT1: 4 tri faces x 3 + base quad x 4 + 1 interior = 17 = PyraRTk(1);
    // the interior dof is not essential → ess = 16.
    let space = HDivSpace::new(mesh.clone(), 1);
    assert_eq!(space.n_dofs(), PyraRTk::new(1).n_dofs(), "RT1 pyramid vsize");
    assert_eq!(space.n_dofs(), 17);
    assert_eq!(
        boundary_dofs_hdiv(space.mesh(), &space, &all_tags).len(),
        16,
        "RT1 pyramid ess (all 5 faces are boundary; interior excluded)"
    );

    // Face block shapes.
    assert_eq!(space.face_dofs(FaceKey::new(0, 1, 4)).unwrap().len(), 3);
    assert_eq!(space.face_dofs(FaceKey::new(0, 1, 2)).unwrap().len(), 4);
    // Every element slot count equals the element n_dofs.
    assert_eq!(space.element_dofs(0).len(), PyraRTk::new(1).n_dofs());
}

/// D414: `dof_coords` fills each face block with the equispaced face-grid
/// anchors of the canonical frame (distinct, on the face), edges with
/// distinct edge points, and interior dofs with the element centroid.
#[test]
fn dof_coords_anchors_cover_every_block() {
    let mesh = load("tmp/d392/one_pyramid.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let coords = space.dof_coords();

    // Base quad block: 4 DISTINCT anchors on the base plane z = 0 — the
    // equispaced (k+1)^2 tensor lattice, at k=1 the four corners.
    let base = space.face_dofs(FaceKey::new(0, 1, 2)).unwrap();
    assert_eq!(base.len(), 4);
    assert!(
        base.iter().all(|&d| coords[d as usize][2] == 0.0),
        "base anchors must lie on z=0"
    );
    let mut pts: Vec<[f64; 2]> = base
        .iter()
        .map(|&d| [coords[d as usize][0], coords[d as usize][1]])
        .collect();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for p in &pts {
        assert!(
            (p[0] - 0.0).abs() < 1e-12 || (p[0] - 1.0).abs() < 1e-12,
            "anchor {p:?} not on the k=1 equispaced tensor lattice"
        );
        assert!(
            (p[1] - 0.0).abs() < 1e-12 || (p[1] - 1.0).abs() < 1e-12,
            "anchor {p:?} not on the k=1 equispaced tensor lattice"
        );
    }
    // Distinctness across all four anchors:
    assert!(pts.windows(2).all(|w| w[0] != w[1]), "base anchors coincide: {pts:?}");

    // Tri face block at k=1: the three anchors are the face's vertices.
    let tri = space.face_dofs(FaceKey::new(0, 1, 4)).unwrap();
    assert_eq!(tri.len(), 3);
    let mut got: Vec<[f64; 3]> = tri.iter().map(|&d| coords[d as usize]).collect();
    let mut mesh_verts: Vec<[f64; 3]> = [0, 1, 4]
        .iter()
        .map(|&v| {
            let c = mesh.node_coords(v);
            [c[0], c[1], c[2]]
        })
        .collect();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mesh_verts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got, mesh_verts, "k=1 tri anchors = the three vertices");

    // The single interior dof carries the element centroid — collect every
    // face block first; the interior is the one dof outside all of them.
    let mut face_dofs_all: Vec<u32> = Vec::new();
    for fk in [
        FaceKey::new(0, 1, 4),
        FaceKey::new(1, 2, 4),
        FaceKey::new(2, 3, 4),
        FaceKey::new(3, 0, 4),
        FaceKey::new(0, 1, 2),
    ] {
        face_dofs_all.extend(space.face_dofs(fk).unwrap());
    }
    face_dofs_all.sort_unstable();
    assert_eq!(face_dofs_all.len(), 16);
    let interior = (0..space.n_dofs() as u32)
        .find(|d| face_dofs_all.binary_search(d).is_err())
        .expect("pyramid RT1 must have exactly one interior dof");
    let cen = coords[interior as usize];
    assert!((cen[0] - 0.5).abs() < 1e-12 && (cen[1] - 0.5).abs() < 1e-12 && cen[2] > 0.0);

    // No dof left at the [0,0,0] default (every dof got an anchor).
    assert!(coords.iter().any(|p| *p != [0.0; 3]) && coords.iter().all(|p| p[2] >= 0.0));
}
