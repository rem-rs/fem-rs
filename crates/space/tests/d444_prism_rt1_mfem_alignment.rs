//! D444 — `PrismRTk` (k ≥ 1) and the prism HDiv space carry MFEM's
//! `RT_WedgeElement` slot layout.
//!
//! Round 50 registered: the `PrismRTk` k≥1 Vandermonde arm enumerated its
//! quad-face groups `[eta=0, zeta=0, diagonal]` against the space's (MFEM)
//! slot order `[bottom, top, q0=(0,1,4,3), q1=(1,2,5,4), q2=(2,0,3,5)]`, and
//! `vector_assembler.rs:146` pairs `PrismRTk(1)` basis functions with the
//! space slots — so prism RT1 assembly assigned quad slots 2–4 to the wrong
//! geometric faces.  The missing MFEM wedge interiors (`p(p+1)(3p+4)/2`,
//! `RT_dof[PRISM]`, `fe_coll.cpp:2581`) are D436, closed by the same fix.
//!
//! MFEM ground truth (4.10, probes archived under `tmp/d444/`):
//! - element: `RT_WedgeElement(1) dof=25` (`probe_d444.out`); node table
//!   pins the face slot order (bottom tri, top tri, q0 y=0, q1 diagonal,
//!   q2 x=0 quads in the canonical frames, interior last);
//! - space: `probe49 ess3d data/d394_prism_stack.mesh 1 0` →
//!   `vsize=47 ess=30` (33 face dofs + 2·7 wedge interiors).

use fem_element::raviart_thomas::PrismRTk;
use fem_element::VectorReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::dof_manager::FaceKey;
use fem_space::{FESpace, HDivSpace};

fn load(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

/// `PrismRTk(1)` matches MFEM `RT_WedgeElement(1)`: 25 dofs, slot groups
/// `[bottom(3), top(3), q0(4), q1(4), q2(4), interior(7)]`.
#[test]
fn prism_rtk1_matches_mfem_wedge_layout() {
    let e = PrismRTk::new(1);
    assert_eq!(e.n_dofs(), 25, "RT_WedgeElement(1) dof=25 (probe)");
    // Group boundaries via the space's slot ranges (same convention).
    let bounds: [usize; 6] = (0..6).map(|f| e.slot_range(f).start).collect::<Vec<_>>().try_into().unwrap();
    assert_eq!(bounds, [0, 3, 6, 10, 14, 18], "MFEM wedge slot group starts");
}

/// Prism stack (`data/d394_prism_stack.mesh`): RT1 space vsize 47 / ess 30 —
/// the MFEM probe49 numbers; RT0 bit-pins unchanged (9/8).
#[test]
fn prism_stack_rt1_vsize_ess_match_mfem() {
    let mesh = load("data/d394_prism_stack.mesh");
    let all_tags = mesh.unique_boundary_tags();

    let space0 = HDivSpace::new(mesh.clone(), 0);
    assert_eq!(space0.n_dofs(), 9, "RT0 vsize (MFEM, unchanged k=0 pin)");
    assert_eq!(boundary_dofs_hdiv(space0.mesh(), &space0, &all_tags).len(), 8);

    let space = HDivSpace::new(mesh.clone(), 1);
    assert_eq!(space.n_dofs(), 47, "RT1 vsize = 33 face + 2x7 interiors (MFEM)");
    assert_eq!(
        boundary_dofs_hdiv(space.mesh(), &space, &all_tags).len(),
        30,
        "RT1 ess (face dofs of the 8 boundary faces only)"
    );
    // Element slot count == PrismRTk(1).n_dofs() (the assembler pairing
    // invariant, vector_assembler.rs:146 pairs PrismRTk(1)).
    for e in 0..mesh.n_elements() as u32 {
        assert_eq!(space.element_dofs(e).len(), 25, "elem {e} slots");
    }
}

/// The D444 lesion, end to end: element 0 of the prism stack IS the
/// reference prism (verts 0..5 = unit wedge), so the space's slot groups can
/// be checked against the element's basis functions directly.  For each face
/// g, the basis functions in the space's slot group of g must carry
/// vanishing normal-flux moments on every other face — before D444 the quad
/// slots 2..4 were paired with permuted faces (eta=0/zeta=0/diagonal vs the
/// space's q0/q1/q2).
#[test]
fn prism_rt1_space_slots_agree_with_element_face_groups() {
    let mesh = load("data/d394_prism_stack.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let e = PrismRTk::new(1);

    // Face g slot ranges in the space's layout (element 0, first-seen
    // element → orientation 0 → slot ids == group positions).
    let ranges: [std::ops::Range<usize>; 5] =
        (0..5).map(|f| e.slot_range(f)).collect::<Vec<_>>().try_into().unwrap();
    let dofs = space.element_dofs(0);
    for (g, r) in ranges.iter().enumerate() {
        assert_eq!(
            dofs[r.start] as usize,
            r.start,
            "first-seen element must hold consecutive face-block ids (face {g})"
        );
    }

    // Reference-prism faces: parametric maps + outward normals + measures.
    let maps: [Box<dyn Fn(f64, f64) -> [f64; 3]>; 5] = [
        Box::new(|u, v| [0.0, u, v]),
        Box::new(|u, v| [1.0, u, v]),
        Box::new(|u, v| [u, v, 0.0]),
        Box::new(|u, v| [u, 1.0 - v, v]),
        Box::new(|u, v| [u, 0.0, v]),
    ];
    let normals: [[f64; 3]; 5] = [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 0.0],
    ];
    let dss = [1.0, 1.0, 1.0, std::f64::consts::SQRT_2, 1.0];
    // {1,u,v} is the RT1 dof functional set (foreign-face vanishing);
    // {u²,uv,v²} extend the own-face check to the full degree-2 trace
    // (corner-dual basis functions may have vanishing {1,u,v} moments).
    let tests: [fn(f64, f64) -> f64; 6] = [
        |_, _| 1.0,
        |u, _| u,
        |_, v| v,
        |u, _| u * u,
        |u, v| u * v,
        |_, v| v * v,
    ];

    let mut phi = vec![0.0_f64; e.n_dofs() * 3];
    for (g, r) in ranges.iter().enumerate() {
        for slot in r.clone() {
            for (h, map) in maps.iter().enumerate() {
                // ∫_h (φ_slot·n̂_h)·q dA (exact quadrature)
                let (rule, tri): (Vec<Vec<f64>>, Vec<f64>) = if h < 2 {
                    let q = fem_element::quadrature::tri_rule(12);
                    (q.points, q.weights)
                } else {
                    let q = fem_element::quadrature::quad_rule_01(12);
                    (q.points, q.weights)
                };
                let mut mom = [0.0_f64; 6];
                for (pt, &w) in rule.iter().zip(tri.iter()) {
                    let x = map(pt[0], pt[1]);
                    e.eval_basis_vec(&x, &mut phi);
                    for (t, tc) in tests.iter().enumerate() {
                        for c in 0..3 {
                            mom[t] += w * dss[h] * tc(pt[0], pt[1]) * phi[slot * 3 + c] * normals[h][c];
                        }
                    }
                }
                if h == g {
                    // own face: at least one degree-≤2 functional nonzero
                    let max = mom.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                    assert!(max > 1e-8, "space face {g} slot {slot}: own-face flux vanishes");
                } else {
                    for (t, &mv) in mom.iter().take(3).enumerate() {
                        assert!(
                            mv.abs() < 1e-6,
                            "space face {g} slot {slot}: moment {t} on foreign face {h} = {mv}"
                        );
                    }
                }
            }
        }
    }
    // Interiors vanish on every face.
    for slot in e.slot_range(5) {
        for (h, map) in maps.iter().enumerate() {
            let (rule, tri): (Vec<Vec<f64>>, Vec<f64>) = if h < 2 {
                let q = fem_element::quadrature::tri_rule(12);
                (q.points, q.weights)
            } else {
                let q = fem_element::quadrature::quad_rule_01(12);
                (q.points, q.weights)
            };
            let mut mom = [0.0_f64; 3];
            for (pt, &w) in rule.iter().zip(tri.iter()) {
                let x = map(pt[0], pt[1]);
                e.eval_basis_vec(&x, &mut phi);
                for c in 0..3 {
                    mom[0] += w * dss[h] * phi[slot * 3 + c] * normals[h][c];
                }
            }
            for (t, &mv) in mom.iter().enumerate() {
                assert!(mv.abs() < 1e-6, "interior slot {slot} leaks on face {h}: {mv} (moment {t})");
            }
        }
    }
}

/// k=0 numbering is untouched by the D444 rebuild: the first-seen element
/// holds consecutive face ids in face-table order and the shared face is
/// reused (historical bit-pins from D394).
#[test]
fn prism_rt0_numbering_unchanged() {
    let mesh = load("data/d394_prism_stack.mesh");
    let space = HDivSpace::new(mesh.clone(), 0);
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3, 4]);
    assert_eq!(space.element_dofs(1), &[1, 5, 6, 7, 8]);
}

/// Shared tri face {3,4,5} still forms one 3-dof conforming block.
#[test]
fn prism_rt1_shared_face_block() {
    let mesh = load("data/d394_prism_stack.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let block = space.face_dofs(FaceKey::new(3, 4, 5)).expect("shared face");
    assert_eq!(block.len(), 3);
    let quad = space.face_dofs(FaceKey::new(0, 1, 3)).expect("quad face");
    assert_eq!(quad.len(), 4);
}
