//! D32 regression tests: ND2 H(curl) cross-edge/face pairing conformity.
//!
//! The ND2 edge DOFs are MFEM point-value functionals `σ(Φ) = Φ(x_i)·t̂` at
//! reflection-symmetric Gauss points; `HCurlSpace` pairs reversed edges with a
//! signed anti-diagonal permutation (MFEM's own encoding).  These tests pin:
//!
//! 1. the anti-diagonal + (−1) reversal encoding on every shared edge,
//! 2. exact tangential-trace identity across a shared edge (≤1e-13),
//! 3. exact `interpolate_vector` reconstruction for fields inside the space,
//! 4. tet ND2 face-dof tangents relating across elements exactly through the
//!    MFEM `ND_DofTransformation` T(ori) 2×2 family.

use fem_element::nedelec::{QuadND2, TetND2, TriND2};
use fem_element::reference::VectorReferenceElement;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::dof_manager::EdgeKey;
use fem_space::fe_space::FESpace;
use fem_space::HCurlSpace;

/// Deterministic pseudo-random coefficient vector in [-1, 1].
fn random_coeffs(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            2.0 * r - 1.0
        })
        .collect()
}

/// Local edges per element type (matching HCurlSpace order>=2 tables).
fn tri_local_edges() -> [(usize, usize); 3] {
    [(0, 1), (1, 2), (2, 0)]
}

// ─── 1. Anti-diagonal reversal encoding on shared edges ─────────────────────

#[test]
fn d32_shared_edge_pairing_is_signed_anti_diagonal() {
    let mesh = Mesh::<2>::unit_square_tri(3);
    let space = HCurlSpace::new(mesh.clone(), 2);
    let k = 2usize;

    // Collect per-element edge claims: EdgeKey -> (element, [local slot; k],
    // [sign; k]) where local slot s satisfies element_dofs(e)[s] == global.
    let n_elem = mesh.n_elements();
    let mut claims: std::collections::HashMap<
        EdgeKey,
        Vec<(u32, [usize; 2], [f64; 2])>,
    > = std::collections::HashMap::new();
    for e in 0..n_elem as u32 {
        let verts = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        for &(li, lj) in tri_local_edges().iter() {
            let key = EdgeKey::new(verts[li], verts[lj]);
            let mut slots = [usize::MAX; 2];
            let mut sgn = [f64::NAN; 2];
            for m in 0..k {
                let g = space.edge_dofs(key).unwrap()[m];
                let pos = dofs
                    .iter()
                    .position(|&d| d == g)
                    .expect("edge global dof must appear in element dofs");
                slots[m] = pos;
                sgn[m] = signs[pos];
            }
            claims.entry(key).or_default().push((e, slots, sgn));
        }
    }

    let mut n_shared = 0usize;
    for (_key, users) in &claims {
        if users.len() < 2 {
            continue;
        }
        n_shared += 1;
        for (i, &(_ea, sa, ga)) in users.iter().enumerate() {
            for &(_eb, sb, gb) in users.iter().skip(i + 1) {
                for m in 0..k {
                    // opposite signs at the same canonical slot
                    assert_eq!(
                        ga[m] * gb[m],
                        -1.0,
                        "shared edge signs must be opposite"
                    );
                }
                // opposite local order: (pos(g1) − pos(g0)) flips sign
                let ord_a = (sa[1] as isize - sa[0] as isize).signum();
                let ord_b = (sb[1] as isize - sb[0] as isize).signum();
                assert_eq!(
                    ord_a * ord_b,
                    -1,
                    "shared edge dof order must be reversed (anti-diagonal)"
                );
            }
        }
    }
    assert!(n_shared > 0, "mesh must contain shared edges");
}

// ─── 2. Tangential trace identity across the shared diagonal ────────────────

/// Reference-coordinate point of parameter `s` along local edge (li, lj).
fn tri_edge_ref_point(li: usize, lj: usize, s: f64) -> [f64; 2] {
    match (li, lj) {
        (0, 1) => [s, 0.0],
        (1, 2) => [1.0 - s, s],
        (2, 0) => [0.0, 1.0 - s],
        _ => panic!("unexpected tri edge ({li},{lj})"),
    }
}

#[test]
fn d32_split_mesh_trace_matches_across_shared_edge() {
    let mesh = Mesh::<2>::unit_square_tri(1);
    assert_eq!(mesh.n_elements(), 2, "expected the 2-triangle split mesh");
    let space = HCurlSpace::new(mesh.clone(), 2);
    let x = random_coeffs(space.n_dofs(), 0xD32C0FFE);

    // Shared edge = the vertex pair claimed by both elements.
    let v: Vec<_> = (0..2u32)
        .map(|e| {
            let nv = mesh.element_nodes(e);
            let mut set = std::collections::BTreeSet::new();
            for i in 0..3 {
                for j in (i + 1)..3 {
                    set.insert(EdgeKey::new(nv[i], nv[j]));
                }
            }
            set
        })
        .collect();
    let shared: Vec<_> = v[0].intersection(&v[1]).copied().collect();
    assert_eq!(shared.len(), 1, "exactly one shared edge");
    let key = shared[0];
    let (pmin, pmax) = {
        let a = mesh.node_coords(key.0);
        let b = mesh.node_coords(key.1);
        (a.to_owned(), b.to_owned())
    };
    let tau = [pmax[0] - pmin[0], pmax[1] - pmin[1]];

    let mut mismatch = 0.0f64;
    for &s in &[0.13f64, 0.5, 0.87] {
        // physical point on the edge
        let y = [pmin[0] + s * tau[0], pmin[1] + s * tau[1]];
        // evaluate the tangential trace from each element
        let mut trace = [0.0f64; 2];
        for e in 0..2u32 {
            let verts = mesh.element_nodes(e);
            let dofs = space.element_dofs(e);
            let signs = space.element_signs(e);
            // find the local edge matching the shared pair
            let local = tri_local_edges()
                .iter()
                .position(|&(li, lj)| {
                    EdgeKey::new(verts[li], verts[lj]) == key
                })
                .expect("shared edge must be a local edge");
            let (li, lj) = tri_local_edges()[local];
            let pa = mesh.node_coords(verts[li]);
            let pb = mesh.node_coords(verts[lj]);
            // sample parameter along the LOCAL direction
            let denom = (pb[0] - pa[0]) * tau[0] + (pb[1] - pa[1]) * tau[1];
            let denom2 = denom / (tau[0] * tau[0] + tau[1] * tau[1]);
            let s_local = s * denom2;
            // reference point and affine Jacobian
            let xi = tri_edge_ref_point(li, lj, s_local);
            let x0 = mesh.node_coords(verts[0]);
            let x1 = mesh.node_coords(verts[1]);
            let x2 = mesh.node_coords(verts[2]);
            let j00 = x1[0] - x0[0];
            let j10 = x1[1] - x0[1];
            let j01 = x2[0] - x0[0];
            let j11 = x2[1] - x0[1];
            let det = j00 * j11 - j01 * j10;
            // invert the map: y = x0 + J ξ
            let rx = y[0] - x0[0];
            let ry = y[1] - x0[1];
            let ksi = (j11 * rx - j01 * ry) / det;
            let eta = (-j10 * rx + j00 * ry) / det;

            let mut vals = vec![0.0; TriND2.n_dofs() * 2];
            TriND2.eval_basis_vec(&[ksi, eta], &mut vals);
            let mut uh = [0.0f64; 2];
            for (i, (&g, &sg)) in dofs.iter().zip(signs.iter()).enumerate() {
                // covariant Piola: phi_phys = J^{-T} phi_ref
                let px = (j11 * vals[i * 2] - j10 * vals[i * 2 + 1]) / det;
                let py = (-j01 * vals[i * 2] + j00 * vals[i * 2 + 1]) / det;
                let c = x[g as usize] * sg;
                uh[0] += c * px;
                uh[1] += c * py;
            }
            trace[e as usize] = uh[0] * tau[0] + uh[1] * tau[1];
        }
        mismatch = mismatch.max((trace[0] - trace[1]).abs());
    }
    assert!(
        mismatch <= 1e-13,
        "shared-edge tangential trace mismatch {mismatch:.3e} > 1e-13"
    );
}

// ─── 3. interpolate_vector exactness for fields inside the space ────────────

/// Reconstruct u_h from interpolated dofs at reference point `xi` of element e
/// and compare against the exact field.
fn reconstruction_error_2d<M: MeshTopology + Clone>(
    mesh: &M,
    space: &HCurlSpace<M>,
    f: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    sample: &[f64],
) -> f64
where
    M: fem_mesh::topology::MeshTopology,
{
    let mut worst = 0.0f64;
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let (x0, x1, x2, j00, j10, j01, j11, det) = if matches!(
            mesh.element_type(e),
            ElementType::Quad4 | ElementType::Quad8
        ) {
            let x3 = mesh.node_coords(verts[3]);
            let a = mesh.node_coords(verts[0]);
            let b = mesh.node_coords(verts[1]);
            let d = mesh.node_coords(verts[3]);
            (
                a.to_owned(),
                b.to_owned(),
                x3.to_owned(),
                b[0] - a[0],
                b[1] - a[1],
                d[0] - a[0],
                d[1] - a[1],
                (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]),
            )
        } else {
            let a = mesh.node_coords(verts[0]);
            let b = mesh.node_coords(verts[1]);
            let c = mesh.node_coords(verts[2]);
            (
                a.to_owned(),
                b.to_owned(),
                c.to_owned(),
                b[0] - a[0],
                b[1] - a[1],
                c[0] - a[0],
                c[1] - a[1],
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
            )
        };
        let (j00, j10, j01, j11, det) = (j00, j10, j01, j11, det);
        let _ = (x1.clone(), x2.clone());
        // physical point of the reference sample (bilinear for quads — exact
        // on the axis-aligned unit-square mesh)
        let xp = if matches!(mesh.element_type(e), ElementType::Quad4 | ElementType::Quad8) {
            let x3 = mesh.node_coords(verts[3]);
            let (a, b, c, d) = (
                mesh.node_coords(verts[0]),
                mesh.node_coords(verts[1]),
                mesh.node_coords(verts[2]),
                x3,
            );
            let (u, v) = (sample[0], sample[1]);
            let w = [
                (1.0 - u) * (1.0 - v),
                u * (1.0 - v),
                u * v,
                (1.0 - u) * v,
            ];
            [
                w[0] * a[0] + w[1] * b[0] + w[2] * c[0] + w[3] * d[0],
                w[0] * a[1] + w[1] * b[1] + w[2] * c[1] + w[3] * d[1],
            ]
        } else {
            [
                x0[0] + j00 * sample[0] + j01 * sample[1],
                x0[1] + j10 * sample[0] + j11 * sample[1],
            ]
        };
        let exact = f(&xp);

        let elem: Box<dyn VectorReferenceElement> =
            if matches!(mesh.element_type(e), ElementType::Quad4 | ElementType::Quad8) {
                Box::new(QuadND2)
            } else {
                Box::new(TriND2)
            };
        let mut vals = vec![0.0; elem.n_dofs() * 2];
        elem.eval_basis_vec(sample, &mut vals);
        let mut uh = [0.0f64; 2];
        for (i, (&g, &sg)) in dofs.iter().zip(signs.iter()).enumerate() {
            let px = (j11 * vals[i * 2] - j10 * vals[i * 2 + 1]) / det;
            let py = (-j01 * vals[i * 2] + j00 * vals[i * 2 + 1]) / det;
            let c = x[g as usize] * sg;
            uh[0] += c * px;
            uh[1] += c * py;
        }
        worst = worst
            .max((uh[0] - exact[0]).abs())
            .max((uh[1] - exact[1]).abs());
    }
    worst
}

#[test]
fn d32_tri_nd2_interpolate_vector_reproduces_space_fields() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = HCurlSpace::new(mesh.clone(), 2);
    for f in [
        &(|_x: &[f64]| vec![1.0, 0.0]) as &dyn Fn(&[f64]) -> Vec<f64>,
        &(|_x: &[f64]| vec![0.0, 1.0]),
        &(|x: &[f64]| vec![x[0], 2.0 * x[1]]),
    ] {
        let x = space.interpolate_vector(f);
        let err = reconstruction_error_2d(
            &mesh,
            &space,
            f,
            x.as_slice(),
            &[0.371, 0.413],
        );
        assert!(err <= 1e-12, "tri ND2 interpolation error {err:.3e} > 1e-12");
    }
}

#[test]
fn d32_quad_nd2_interpolate_vector_reproduces_space_fields() {
    for n in [1usize, 2] {
        let mesh = Mesh::<2>::unit_square_quad(n);
        let space = HCurlSpace::new(mesh.clone(), 2);
        for (fname, f) in [
            ("const-x", &(|_x: &[f64]| vec![1.0, 0.0]) as &dyn Fn(&[f64]) -> Vec<f64>),
            ("const-y", &(|_x: &[f64]| vec![0.0, 1.0])),
            ("linear", &(|x: &[f64]| vec![x[0] + 3.0 * x[1], 1.0 - 2.0 * x[0]])),
        ] {
            let x = space.interpolate_vector(f);
            let err = reconstruction_error_2d(
                &mesh,
                &space,
                f,
                x.as_slice(),
                &[0.271, 0.639],
            );
            eprintln!("quad n={n} field {fname}: err {err:.3e}");
            assert!(err <= 1e-12, "quad ND2 ({fname}, n={n}) interpolation error {err:.3e} > 1e-12");
        }
    }
}

#[test]
fn d32_tet_nd2_interpolate_vector_reproduces_constant() {
    // Single tet: all 20 DOFs element-local, so the interpolation must be
    // exact.  (On multi-tet meshes the two shared-face DOFs need MFEM's 2×2
    // ND face rotations, which the scalar-sign pairing cannot express — see
    // `d32_tet_face_tangents_follow_mfem_rotation_family`.)
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let conn: Vec<u32> = vec![0, 1, 2, 3];
    let face_conn: Vec<u32> = vec![
        1, 3, 2, // (1,2,3)
        0, 2, 3, // (0,2,3)
        0, 1, 3, // (0,1,3)
        0, 1, 2, // (0,1,2)
    ];
    let face_tags = vec![1, 2, 3, 4];
    let mesh = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1],
        ElementType::Tet4,
        face_conn,
        face_tags,
        ElementType::Tri3,
    );
    let space = HCurlSpace::new(mesh.clone(), 2);
    assert_eq!(space.element_dofs(0).len(), 20);
    let f = |_x: &[f64]| vec![1.0, 0.0, 0.0];
    let x = space.interpolate_vector(&f);
    // reconstruct at an interior sample and compare
    let mut worst = 0.0f64;
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let a = mesh.node_coords(verts[0]);
        let b = mesh.node_coords(verts[1]);
        let c = mesh.node_coords(verts[2]);
        let d = mesh.node_coords(verts[3]);
        // affine Jacobian J = [b−a | c−a | d−a]
        let j = [
            [b[0] - a[0], c[0] - a[0], d[0] - a[0]],
            [b[1] - a[1], c[1] - a[1], d[1] - a[1]],
            [b[2] - a[2], c[2] - a[2], d[2] - a[2]],
        ];
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        let sample = [0.2f64, 0.3, 0.1];
        let xp = [
            a[0] + j[0][0] * sample[0] + j[0][1] * sample[1] + j[0][2] * sample[2],
            a[1] + j[1][0] * sample[0] + j[1][1] * sample[1] + j[1][2] * sample[2],
            a[2] + j[2][0] * sample[0] + j[2][1] * sample[1] + j[2][2] * sample[2],
        ];
        let exact = f(&xp);
        let mut vals = vec![0.0; TetND2.n_dofs() * 3];
        TetND2.eval_basis_vec(&sample, &mut vals);
        // J^{-T} rows: solve J^T y = phi for each basis (3x3 inverse)
        let mut jt_inv = [[0.0f64; 3]; 3];
        {
            let m = [
                [j[0][0], j[1][0], j[2][0]],
                [j[0][1], j[1][1], j[2][1]],
                [j[0][2], j[1][2], j[2][2]],
            ]; // J^T
            for i in 0..3 {
                for jj in 0..3 {
                    let (r1, r2) = ((i + 1) % 3, (i + 2) % 3);
                    let (c1, c2) = ((jj + 1) % 3, (jj + 2) % 3);
                    let cof = m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1];
                    jt_inv[jj][i] = cof / det; // adjugate/det
                }
            }
        }
        let mut uh = [0.0f64; 3];
        for (i, (&g, &sg)) in dofs.iter().zip(signs.iter()).enumerate() {
            let c = x[g as usize] * sg;
            for r in 0..3 {
                uh[r] += c * (jt_inv[r][0] * vals[i * 3]
                    + jt_inv[r][1] * vals[i * 3 + 1]
                    + jt_inv[r][2] * vals[i * 3 + 2]);
            }
        }
        for r in 0..3 {
            worst = worst.max((uh[r] - exact[r]).abs());
        }
    }
    assert!(worst <= 1e-12, "tet ND2 interpolation error {worst:.3e} > 1e-12");
}

// ─── 4. Tet ND2 face tangents relate through MFEM's T(ori) family ───────────

#[test]
fn d32_tet_face_tangents_follow_mfem_rotation_family() {
    // MFEM `ND_DofTransformation::T_data` (doftrans.cpp), 2×2 per orientation.
    let t_family: [[f64; 4]; 6] = [
        [1.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0, -1.0],
        [1.0, 0.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ];

    let mesh = Mesh::<3>::unit_cube_tet(1);
    let n_elem = mesh.n_elements();

    // Find a pair of tets sharing a face.
    use fem_space::dof_manager::FaceKey;
    let mut faces: std::collections::HashMap<
        FaceKey,
        Vec<(u32, [usize; 3])>,
    > = std::collections::HashMap::new();
    for e in 0..n_elem as u32 {
        let verts = mesh.element_nodes(e);
        for (fi, verts3) in [(0usize, [1usize, 2, 3]), (1, [0, 2, 3]), (2, [0, 1, 3]), (3, [0, 1, 2])]
        {
            let key = FaceKey::new(verts[verts3[0]], verts[verts3[1]], verts[verts3[2]]);
            faces.entry(key).or_default().push((e, verts3));
        }
    }
    let shared: Vec<_> = faces
        .iter()
        .filter(|(_, v)| v.len() == 2)
        .collect();
    assert!(!shared.is_empty(), "mesh must contain shared faces");

    for (key, users) in shared.iter().take(4) {
        let mut w: Vec<[[f64; 2]; 2]> = Vec::new();
        let mut centroid = [0.0f64; 3];
        for &(e, local3) in users.iter() {
            let verts = mesh.element_nodes(e);
            // TetND2 face tangent table (MFEM dof2tk pairs), per TET_FACES f.
            let tang3: [[f64; 3]; 2] = match local3 {
                [1, 2, 3] => [[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]],
                [0, 3, 2] | [0, 2, 3] => [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [0, 1, 3] => [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0, 2, 1] | [0, 1, 2] => [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                _ => panic!("unexpected face local verts {local3:?}"),
            };
            let p = |i: usize| mesh.node_coords(verts[i]);
            // physical image of a reference tangent t: J·t with
            // J = [P1−P0 | P2−P0 | P3−P0] (columns)
            let apply = |t: [f64; 3]| -> [f64; 3] {
                let mut v = [0.0f64; 3];
                for r in 0..3 {
                    v[r] = t[0] * (p(1)[r] - p(0)[r])
                        + t[1] * (p(2)[r] - p(0)[r])
                        + t[2] * (p(3)[r] - p(0)[r]);
                }
                v
            };
            let phys = [apply(tang3[0]), apply(tang3[1])];
            // project the two 3-vectors onto the 2-D parameterization of the
            // shared face given by `key` (sorted vertex triple)
            let kv = [key.0, key.1, key.2];
            let q = |i: usize| mesh.node_coords(kv[i]);
            let d0 = [
                q(1)[0] - q(0)[0],
                q(1)[1] - q(0)[1],
                q(1)[2] - q(0)[2],
            ];
            let d1 = [
                q(2)[0] - q(0)[0],
                q(2)[1] - q(0)[1],
                q(2)[2] - q(0)[2],
            ];
            let coords2 = |v: [f64; 3]| -> [f64; 2] {
                // solve [d0 d1] a = v in least squares (exact: v in span)
                let a11 = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2];
                let a12 = d0[0] * d1[0] + d0[1] * d1[1] + d0[2] * d1[2];
                let a22 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
                let b1 = v[0] * d0[0] + v[1] * d0[1] + v[2] * d0[2];
                let b2 = v[0] * d1[0] + v[1] * d1[1] + v[2] * d1[2];
                let det = a11 * a22 - a12 * a12;
                [(b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det]
            };
            w.push([coords2(phys[0]), coords2(phys[1])]);
            let a = p(local3[0]);
            let b = p(local3[1]);
            let c = p(local3[2]);
            for r in 0..3 {
                centroid[r] += (a[r] + b[r] + c[r]) / 3.0 / users.len() as f64;
            }
        }
        let _ = centroid;
        // both elements must anchor at the same centroid
        // (verified through the space's dof_coords in other tests)

        // change-of-basis: w_B = M w_A (2×2, columns = B tangents in A basis)
        let wa = w[0];
        let wb = w[1];
        let det = wa[0][0] * wa[1][1] - wa[0][1] * wa[1][0];
        assert!(
            det.abs() > 1e-12,
            "face tangents of adjacent tets must span the same plane"
        );
        let minv = [
            [wa[1][1] / det, -wa[0][1] / det],
            [-wa[1][0] / det, wa[0][0] / det],
        ];
        let m = [
            [
                minv[0][0] * wb[0][0] + minv[0][1] * wb[1][0],
                minv[0][0] * wb[0][1] + minv[0][1] * wb[1][1],
            ],
            [
                minv[1][0] * wb[0][0] + minv[1][1] * wb[1][0],
                minv[1][0] * wb[0][1] + minv[1][1] * wb[1][1],
            ],
        ];
        // M must be a member of MFEM's T(ori) family (entries in {0, ±1}).
        let in_family = t_family.iter().any(|t| {
            (m[0][0] - t[0]).abs() < 1e-12
                && (m[1][0] - t[1]).abs() < 1e-12
                && (m[0][1] - t[2]).abs() < 1e-12
                && (m[1][1] - t[3]).abs() < 1e-12
        });
        assert!(
            in_family,
            "face tangent change-of-basis {:?} not in MFEM T(ori) family",
            m
        );
    }
}
