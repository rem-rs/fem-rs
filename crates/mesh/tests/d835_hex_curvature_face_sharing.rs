//! D835-1 (round 90 Lane B): `set_curvature_hex8` must allocate **face**
//! geometry dofs once per mesh face — MFEM's continuous `H1` entity dof —
//! not one private copy per element.
//!
//! ## The defect and the corrected round-89 registration
//!
//! Round 89 registered the count mismatch (3×3×3 grid `geom_n_nodes` 397 vs
//! MFEM `H1_3D_P2` 343 = 64 V + 144 E + 108 F + 27 C) and claimed the
//! duplicated face/interior dofs were "bitwise identical on an affine mesh,
//! hence harmless".  The probe (`tmp/d90b/d835_probe_prefix.txt`) **refutes
//! the harmlessness**: on the plain unit 3×3×3 grid the two sides of a shared
//! face evaluate the trilinear map in different corner orders, and 6 of the
//! 54 duplicated face dofs at `p = 2` (48 of 216 at `p = 3`) already differ
//! by **1 ulp** (`0.5` vs `0.49999999999999994` on the *affine* mesh).  Each
//! element's isoparametric map then uses its own copy, so the face trace is
//! not single-valued — a geometric micro-crack along every affected interior
//! face, exactly what MFEM's single shared H1 dof cannot have.
//!
//! ## Fix
//!
//! `set_curvature_hex8` keys face dofs by the mesh face (sorted corner quad +
//! the dof's canonical in-face indices, the same rule
//! `amr::curved_hex::build_refined_hex_geometry` uses) and keeps interior
//! dofs per element — MFEM `H1` semantics: V/E/F shared, interior private.
//! Counts become exact MFEM `H1` counts on every grid below.

use fem_mesh::{ElementType, Mesh, MeshTopology};

/// MFEM `H1_<dim>D_Pp` dof counts on an n×n×n hex grid:
/// `NV + (p-1)·NE + (p-1)²·NF + (p-1)³·NC`.
fn h1_count(n: usize, p: usize) -> usize {
    let nv = (n + 1).pow(3);
    let ne = 3 * n * (n + 1) * (n + 1);
    let nf = 3 * n * n * (n + 1);
    let nc = n * n * n;
    nv + (p - 1) * ne + (p - 1) * (p - 1) * nf + (p - 1) * (p - 1) * (p - 1) * nc
}

#[test]
fn d835_affine_grid_face_dofs_match_mfem_h1_counts() {
    for &(n, p) in &[(1usize, 2usize), (1, 3), (2, 2), (3, 2), (3, 3)] {
        let mut m = Mesh::<3>::make_cartesian_3d(n, n, n, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        m.set_curvature(p as u8);
        let g = m.geometry.as_ref().expect("geometry");
        let want = h1_count(n, p);
        assert_eq!(
            g.n_nodes,
            want,
            "{n}³ grid p={p}: geom dofs {} must equal the MFEM H1 count {want} \
             (per-element face allocation duplicates interior-face dofs)",
            g.n_nodes
        );
    }
}

#[test]
fn d835_affine_grid_geometry_positions_are_single_valued() {
    // Sharing must remove every duplicate position: distinct geometry dofs
    // have distinct coordinates (the grid spacing is ≥ 1/n³ in every slot).
    let mut m = Mesh::<3>::make_cartesian_3d(3, 3, 3, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    m.set_curvature(2);
    let g = m.geometry.as_ref().expect("geometry");
    let mut seen = std::collections::BTreeSet::new();
    for d in 0..g.n_nodes {
        let k: [i64; 3] =
            std::array::from_fn(|a| (g.coords[d * 3 + a] * 1e9).round() as i64);
        assert!(
            seen.insert(k),
            "dof {d} duplicates an existing geometry position: {:?}",
            &g.coords[d * 3..d * 3 + 3]
        );
    }
}

#[test]
fn d835_warped_mesh_face_trace_is_single_valued() {
    // A trilinearly warped 2×1×1 pair: the shared x=0.5 face carries warped
    // bilinear geometry.  Both elements must reference the *same* face dof
    // ids (one shared dof, MFEM H1), and the two face traces must agree.
    let mut w = Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    for i in 0..w.n_nodes() {
        let c: [f64; 3] = w.coords_of(i as u32).try_into().unwrap();
        if (c[0] - 0.5).abs() < 1e-12 && c[1].abs() < 1e-12 && c[2].abs() < 1e-12 {
            w.coords[i * 3 + 2] += 0.3;
        }
        if (c[0] - 0.5).abs() < 1e-12 && (c[1] - 1.0).abs() < 1e-12 && c[2].abs() < 1e-12 {
            w.coords[i * 3 + 2] -= 0.2;
        }
    }
    w.set_curvature(2);
    // MFEM H1(2) on this mesh: 12 V + 20 E + 11 F + 2 C.
    let g = w.geometry.as_ref().expect("geometry");
    assert_eq!(g.n_nodes, 12 + 20 + 11 + 2, "warped pair p=2 H1(2) dofs");

    // The shared face's centre dof: element 0's +x face and element 1's -x
    // face must name the same geometry node.
    let row0: Vec<u32> = w.geometry_row(0).to_vec();
    let row1: Vec<u32> = w.geometry_row(1).to_vec();
    // The face-centre slot of an order-2 HexQk row is the (1, .5, .5)/(0, .5, .5)
    // dof — find it by position: the geometry node whose coordinates are
    // closest to (0.5, 0.5, 0.5) within each row, then check id equality.
    let face_center = |row: &[u32]| -> u32 {
        let mut best = (u32::MAX, f64::MAX);
        for &d in row {
            if (d as usize) < w.n_nodes() {
                continue; // vertex dofs sit on the mesh nodes
            }
            let c: [f64; 3] = g.coords[d as usize * 3..d as usize * 3 + 3].try_into().unwrap();
            let dist = (c[0] - 0.5).powi(2) + (c[1] - 0.5).powi(2) + (c[2] - 0.5).powi(2);
            if dist < best.1 {
                best = (d, dist);
            }
        }
        best.0
    };
    let d0 = face_center(&row0);
    let d1 = face_center(&row1);
    assert_ne!(d0, u32::MAX, "face-centre dof not found in row 0");
    assert_eq!(
        d0, d1,
        "the shared face must carry ONE geometry dof, not one copy per element"
    );

    // The two element maps agree on the shared face (up to evaluation noise).
    for (y, z) in [(0.25_f64, 0.25), (0.5, 0.5), (0.75, 0.75)] {
        let (_j0, _, x0) = w.element_jacobian(0, &[1.0, y, z]);
        let (_j1, _, x1) = w.element_jacobian(1, &[0.0, y, z]);
        for a in 0..3 {
            assert!(
                (x0[a] - x1[a]).abs() < 1e-12,
                "face trace ({y},{z}) component {a}: {} vs {}",
                x0[a],
                x1[a]
            );
        }
    }
}
