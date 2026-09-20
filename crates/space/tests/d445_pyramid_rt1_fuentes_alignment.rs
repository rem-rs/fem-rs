//! D445 — `PyraRTk` and the pyramid HDiv space carry MFEM's
//! `RT_FuentesPyramidElement` slot layout.
//!
//! Round 50 registered: the element's slot order `[base, x=0, y=0, x+z=1,
//! y+z=1]` contradicted the space's `PYRAMID_FACES` convention `[4 tris,
//! base]`, and the k=1 count 17 was not MFEM's Fuentes 28.  Round 51
//! (D445/D437) aligns both sides to MFEM:
//!
//! - element slot order = Fuentes construction order (`fe_rt.cpp:1304-1373`):
//!   `[base quad (3,2,1,0), tri (0,1,4), tri (1,2,4), tri (2,3,4),
//!   tri (3,0,4), interior x/y/z]`;
//! - counts `(p+1)(3p(p+2)+5)` = 28 at k=1 (`fe_rt.cpp:1273-1275`), interior
//!   `3p(p+1)^2` (`RT_dof[PYRAMID]`, `fe_coll.cpp:2586`);
//! - the space's `PYRAMID_FACES` uses the MFEM FaceVert order (base first).
//!
//! MFEM ground truth (4.10, probes archived under `tmp/d444/`):
//! - element: `RT_FuentesPyramidElement(1) dof=28` (`probe_d444.out`), with
//!   the full 28-node table;
//! - space on a single pyramid (in-code mesh, `pyr_incode.out`):
//!   k=0 vsize=5/ess=5, k=1 vsize=28/ess=16.
//!
//! Fixture: `data/d445_one_pyramid.mesh` (copy of the historical
//! `tmp/d392/one_pyramid.mesh`, promoted to `data/` per fixture discipline;
//! MFEM cannot read that file — its 1-pyramid reader segfaults — so the
//! space-level MFEM oracle comes from the in-code probe).

use fem_element::raviart_thomas::PyraRTk;
use fem_element::VectorReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::dof_manager::FaceKey;
use fem_space::{FESpace, HDivSpace};

fn load(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

/// `PyraRTk(1)` matches MFEM `RT_FuentesPyramidElement(1)`: 28 dofs with
/// slot groups `[base(4), tri(3), tri(3), tri(3), tri(3), interior(12)]`.
#[test]
fn pyra_rtk1_matches_mfem_fuentes_layout() {
    let e = PyraRTk::new(1);
    assert_eq!(e.n_dofs(), 28, "RT_FuentesPyramidElement(1) dof=28 (probe)");
    let starts: Vec<usize> = (0..6).map(|f| e.slot_range(f).start).collect();
    assert_eq!(starts, [0, 4, 7, 10, 13, 16], "Fuentes slot group starts");
}

/// Single pyramid RT1: vsize 28 / ess 16 — the MFEM in-code probe numbers;
/// RT0 unchanged (5/5).
#[test]
fn pyramid_rt1_vsize_ess_match_mfem() {
    let mesh = load("data/d445_one_pyramid.mesh");
    let all_tags = mesh.unique_boundary_tags();

    let space0 = HDivSpace::new(mesh.clone(), 0);
    assert_eq!(space0.n_dofs(), 5, "RT0 vsize (MFEM)");
    assert_eq!(boundary_dofs_hdiv(space0.mesh(), &space0, &all_tags).len(), 5);

    let space = HDivSpace::new(mesh.clone(), 1);
    assert_eq!(space.n_dofs(), 28, "RT1 vsize = 16 face + 12 interiors (MFEM)");
    assert_eq!(
        boundary_dofs_hdiv(space.mesh(), &space, &all_tags).len(),
        16,
        "RT1 ess (all 5 faces boundary; interiors excluded)"
    );
    assert_eq!(space.element_dofs(0).len(), 28, "element slots == PyraRTk(1)");
}

/// Slot layout: the base quad block occupies slots 0..4, the four tri
/// blocks 4..16 (MFEM FaceVert order), interiors 16..28.
#[test]
fn pyramid_rt1_slot_layout_is_fuentes() {
    let mesh = load("data/d445_one_pyramid.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let dofs = space.element_dofs(0);

    let base = space.face_dofs(FaceKey::new(0, 1, 2)).expect("base quad");
    assert_eq!(base.len(), 4);
    assert_eq!(&dofs[..4], base.as_slice(), "base quad block first (MFEM)");

    for tri in [
        FaceKey::new(0, 1, 4),
        FaceKey::new(1, 2, 4),
        FaceKey::new(2, 3, 4),
        FaceKey::new(3, 0, 4),
    ] {
        let block = space.face_dofs(tri).expect("tri face");
        assert_eq!(block.len(), 3);
        for &d in &block {
            assert!(
                dofs[4..16].contains(&d),
                "tri face dof {d} must live in slots 4..16"
            );
        }
    }
    // Interiors fill slots 16..28 exclusively.
    let mut face_dofs: Vec<u32> = vec![FaceKey::new(0, 1, 2)]
        .into_iter()
        .chain([FaceKey::new(0, 1, 4), FaceKey::new(1, 2, 4), FaceKey::new(2, 3, 4), FaceKey::new(3, 0, 4)])
        .flat_map(|f| space.face_dofs(f).unwrap())
        .collect();
    face_dofs.sort_unstable();
    for slot in 16..28usize {
        assert!(
            face_dofs.binary_search(&dofs[slot]).is_err(),
            "slot {slot} (dof {}) must be interior",
            dofs[slot]
        );
    }
}

/// D445 lesion check: element face groups agree with the space's slots —
/// each face group's basis functions carry vanishing normal-flux moments
/// (RT1 functional set) on all other faces.
#[test]
fn pyramid_rt1_slot_groups_are_face_conforming() {
    let mesh = load("data/d445_one_pyramid.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let e = PyraRTk::new(1);
    let dofs = space.element_dofs(0);

    // The fixture is the unit reference pyramid; face 0 of the space layout
    // (base quad, verts (3,2,1,0)) pairs with element slots 0..4, etc.
    let ranges: [std::ops::Range<usize>; 5] =
        (0..5).map(|f| e.slot_range(f)).collect::<Vec<_>>().try_into().unwrap();

    // Reference-pyramid faces: maps + normals + measures (Fuentes frames).
    let maps: [Box<dyn Fn(f64, f64) -> [f64; 3]>; 5] = [
        Box::new(|u, v| [u, 1.0 - v, 0.0]),
        Box::new(|u, v| [u, 0.0, v]),
        Box::new(|u, v| [1.0 - v, u, v]),
        Box::new(|u, v| [1.0 - u - v, 1.0 - v, v]),
        Box::new(|u, v| [0.0, 1.0 - u - v, v]),
    ];
    let r2 = std::f64::consts::SQRT_2;
    let normals: [[f64; 3]; 5] = [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 0.0],
    ];
    let dss = [1.0, 1.0, r2, r2, 1.0];
    // {1,u,v} = RT1 functional set (foreign-face vanishing); {u²,uv,v²}
    // complete the own-face degree-2 trace check (corner-dual basis
    // functions may have vanishing {1,u,v} moments).
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
                let (pts, wts): (Vec<Vec<f64>>, Vec<f64>) = if h == 0 {
                    let q = fem_element::quadrature::quad_rule_01(12);
                    (q.points, q.weights)
                } else {
                    let q = fem_element::quadrature::tri_rule(12);
                    (q.points, q.weights)
                };
                let mut mom = [0.0_f64; 6];
                for (pt, &w) in pts.iter().zip(wts.iter()) {
                    let x = map(pt[0], pt[1]);
                    e.eval_basis_vec(&x, &mut phi);
                    for (t, tc) in tests.iter().enumerate() {
                        for c in 0..3 {
                            mom[t] += w * dss[h] * tc(pt[0], pt[1]) * phi[slot * 3 + c] * normals[h][c];
                        }
                    }
                }
                if h == g {
                    let max = mom.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                    assert!(max > 1e-7, "space face {g} slot {slot}: own-face flux vanishes");
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
    // Interiors vanish on all five faces (constant-weight moment).
    let _ = dofs;
    for slot in e.slot_range(5) {
        for (h, map) in maps.iter().enumerate() {
            let (pts, wts): (Vec<Vec<f64>>, Vec<f64>) = if h == 0 {
                let q = fem_element::quadrature::quad_rule_01(12);
                (q.points, q.weights)
            } else {
                let q = fem_element::quadrature::tri_rule(12);
                (q.points, q.weights)
            };
            let mut m0 = 0.0_f64;
            for (pt, &w) in pts.iter().zip(wts.iter()) {
                let x = map(pt[0], pt[1]);
                e.eval_basis_vec(&x, &mut phi);
                for c in 0..3 {
                    m0 += w * dss[h] * phi[slot * 3 + c] * normals[h][c];
                }
            }
            assert!(m0.abs() < 1e-6, "interior slot {slot} leaks on face {h}: {m0}");
        }
    }
}
