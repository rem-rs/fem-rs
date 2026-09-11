//! D38 + D37 regression: NDk (k ≥ 3) nodal DOF semantics and the tet
//! face-DOF block transform.
//!
//! * **D38** — `TriNDk`/`TetNDk` (k ≥ 3) and `HCurlSpace`'s k ≥ 3 interior /
//!   face DOFs are MFEM point-value functionals `σ_i(Φ) = Φ(x_i)·t̂_i`; the
//!   tests pin the resulting cross-element trace identity (≤1e-13) and the
//!   exactness of `interpolate_vector` for fields inside the space.
//! * **D37** — a shared triangular face's two DOFs per point are related
//!   between the adjacent tets by a full 2×2 change of basis
//!   ([`FaceDofBlock`]), not a scalar sign.  The test reconstructs the
//!   element-local DOFs through that block transform and verifies the shared
//!   face trace identity for k = 2 and k = 3; the same comparison *without*
//!   the transform is asserted to be O(1e-1) — the defect D37 fixes.

use std::collections::HashMap;

use fem_element::nedelec::{TetND2, TetNDk, TriND2, TriNDk};
use fem_element::reference::VectorReferenceElement;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::{EdgeKey, FaceKey};

use fem_space::HCurlSpace;

/// Deterministic pseudo-random coefficients in [-1, 1].
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

/// Element-local (signed) DOF vector reconstructed from a canonical global
/// vector: edges take their ±1 orientation sign, each face pair is rotated by
/// its `FaceDofBlock` (`u_local = S·u_canon`).
fn local_coeffs<M: MeshTopology>(space: &HCurlSpace<M>, e: u32, u_canon: &[f64]) -> Vec<f64> {
    let dofs = space.element_dofs(e);
    let signs = space.element_signs(e);
    let mut u: Vec<f64> = dofs
        .iter()
        .zip(signs.iter())
        .map(|(&d, &s)| s * u_canon[d as usize])
        .collect();
    for b in space.element_face_blocks(e) {
        let c = [
            u_canon[b.canon_dofs[0] as usize],
            u_canon[b.canon_dofs[1] as usize],
        ];
        u[b.slot] = b.s[0][0] * c[0] + b.s[0][1] * c[1];
        u[b.slot + 1] = b.s[1][0] * c[0] + b.s[1][1] * c[1];
    }
    u
}

/// Same, but ignoring the face blocks (the pre-D37 scalar-sign convention).
fn local_coeffs_no_block<M: MeshTopology>(
    space: &HCurlSpace<M>,
    e: u32,
    u_canon: &[f64],
) -> Vec<f64> {
    let dofs = space.element_dofs(e);
    let signs = space.element_signs(e);
    dofs.iter()
        .zip(signs.iter())
        .map(|(&d, &s)| s * u_canon[d as usize])
        .collect()
}

// ─── 2-D: Tri ND3 edge trace + interpolation ────────────────────────────────

fn tri_ref_point(li: usize, lj: usize, s: f64) -> [f64; 2] {
    match (li, lj) {
        (0, 1) => [s, 0.0],
        (1, 2) => [1.0 - s, s],
        (2, 0) => [0.0, 1.0 - s],
        _ => panic!("unexpected tri edge ({li},{lj})"),
    }
}

/// Tangential trace of the reconstructed field at a physical point, from
/// element `e`.
fn tri_trace_at<M: MeshTopology>(
    mesh: &M,
    space: &HCurlSpace<M>,
    e: u32,
    u: &[f64],
    x: [f64; 2],
    dir: [f64; 2],
) -> f64 {
    let verts = mesh.element_nodes(e);
    let dofs = space.element_dofs(e);
    let x0 = mesh.node_coords(verts[0]);
    let x1 = mesh.node_coords(verts[1]);
    let x2 = mesh.node_coords(verts[2]);
    let j00 = x1[0] - x0[0];
    let j10 = x1[1] - x0[1];
    let j01 = x2[0] - x0[0];
    let j11 = x2[1] - x0[1];
    let det = j00 * j11 - j01 * j10;
    let rx = x[0] - x0[0];
    let ry = x[1] - x0[1];
    let xi = (j11 * rx - j01 * ry) / det;
    let eta = (-j10 * rx + j00 * ry) / det;
    let elem = TriNDk::new(space.order() as usize);
    let mut vals = vec![0.0; elem.n_dofs() * 2];
    elem.eval_basis_vec(&[xi, eta], &mut vals);
    let mut uh = [0.0_f64; 2];
    for (i, _g) in dofs.iter().enumerate() {
        let px = (j11 * vals[i * 2] - j10 * vals[i * 2 + 1]) / det;
        let py = (-j01 * vals[i * 2] + j00 * vals[i * 2 + 1]) / det;
        uh[0] += u[i] * px;
        uh[1] += u[i] * py;
    }
    uh[0] * dir[0] + uh[1] * dir[1]
}

#[test]
fn d38_tri_ndk3_shared_edge_trace_is_exact() {
    for n in [1usize, 2] {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = HCurlSpace::new(mesh.clone(), 3);
        assert_eq!(space.element_dofs(0).len(), 15);
        let u_canon = random_coeffs(space.n_dofs(), 0xD38);
        // per-element signed local coefficients (2-D: edges only)
        let locals: Vec<Vec<f64>> = (0..mesh.n_elements() as u32)
            .map(|e| local_coeffs_no_block(&space, e, &u_canon))
            .collect();

        // collect, per mesh edge, all (element, endpoints) claims
        let mut claims: HashMap<EdgeKey, Vec<(u32, usize, usize)>> = HashMap::new();
        for e in 0..mesh.n_elements() as u32 {
            let v = mesh.element_nodes(e);
            for (li, lj) in [(0usize, 1usize), (1, 2), (2, 0)] {
                claims
                    .entry(EdgeKey::new(v[li], v[lj]))
                    .or_default()
                    .push((e, li, lj));
            }
        }
        let mut worst = 0.0_f64;
        let mut n_shared = 0usize;
        for (key, users) in &claims {
            if users.len() < 2 {
                continue;
            }
            n_shared += 1;
            let pa = mesh.node_coords(key.0);
            let pb = mesh.node_coords(key.1);
            let dir = [pb[0] - pa[0], pb[1] - pa[1]];
            for &s in &[0.3f64, 0.55, 0.71] {
                let x = [pa[0] + s * dir[0], pa[1] + s * dir[1]];
                let mut tr = Vec::new();
                for &(e, _li, _lj) in users {
                    tr.push(tri_trace_at(&mesh, &space, e, &locals[e as usize], x, dir));
                }
                for t in &tr[1..] {
                    worst = worst.max((t - tr[0]).abs());
                }
            }
        }
        assert!(n_shared > 0, "mesh {n} must have shared edges");
        eprintln!("TriND3 worst shared-edge trace mismatch: {worst:.3e} (n={n})");
        assert!(
            worst <= 1e-13,
            "TriND3 shared-edge trace mismatch {worst:.3e} (n={n})"
        );
    }
}

/// Reconstruct the interpolant of `f` and compare with `f` itself.
fn tri_reconstruction_error<M: MeshTopology>(
    mesh: &M,
    space: &HCurlSpace<M>,
    u: &[f64],
    f: &dyn Fn(&[f64]) -> Vec<f64>,
    sample: [f64; 2],
) -> f64 {
    let mut worst = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let x0 = mesh.node_coords(verts[0]);
        let x1 = mesh.node_coords(verts[1]);
        let x2 = mesh.node_coords(verts[2]);
        let j00 = x1[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j01 = x2[0] - x0[0];
        let j11 = x2[1] - x0[1];
        let det = j00 * j11 - j01 * j10;
        let xp = [
            x0[0] + j00 * sample[0] + j01 * sample[1],
            x0[1] + j10 * sample[0] + j11 * sample[1],
        ];
        let exact = f(&xp);
        let signs = space.element_signs(e);
        let elem = TriNDk::new(space.order() as usize);
        let mut vals = vec![0.0; elem.n_dofs() * 2];
        elem.eval_basis_vec(&sample, &mut vals);
        let mut uh = [0.0_f64; 2];
        for (i, &g) in dofs.iter().enumerate() {
            let px = (j11 * vals[i * 2] - j10 * vals[i * 2 + 1]) / det;
            let py = (-j01 * vals[i * 2] + j00 * vals[i * 2 + 1]) / det;
            let c = signs[i] * u[g as usize];
            uh[0] += c * px;
            uh[1] += c * py;
        }
        for d in 0..2 {
            worst = worst.max((uh[d] - exact[d]).abs());
        }
    }
    worst
}

#[test]
fn d38_tri_ndk3_interpolate_vector_is_exact() {
    // N_3 ⊃ P_2², so constant / linear / quadratic fields are reproduced
    // exactly.
    for n in [1usize, 2] {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = HCurlSpace::new(mesh.clone(), 3);
        for (name, f) in [
            ("const", &(|_x: &[f64]| vec![1.0, 0.0]) as &dyn Fn(&[f64]) -> Vec<f64>),
            ("linear", &(|x: &[f64]| vec![1.0 + x[0] - 3.0 * x[1], 2.0 * x[0] + x[1]])),
            ("quad", &(|x: &[f64]| vec![x[0] * x[0], x[0] * x[1]])),
        ] {
            let u = space.interpolate_vector(f);
            let err = tri_reconstruction_error(&mesh, &space, u.as_slice(), f, [0.271, 0.639]);
            assert!(
                err <= 1e-11,
                "TriND3 ({name}, n={n}) interpolation error {err:.3e}"
            );
        }
    }
}

#[test]
fn d38_tri_ndk2_path_unchanged() {
    // k = 2 must stay the D32 nodal construction: same DOF sites, same
    // element basis as the explicit TriND2, and the same trace identity.
    for n in [1usize, 2] {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = HCurlSpace::new(mesh.clone(), 2);
        let t2 = TriND2;
        let tk = TriNDk::new(2);
        for xi in &[[0.137, 0.421], [0.33, 0.29]] {
            let mut a = vec![0.0; 16];
            let mut b = vec![0.0; 16];
            t2.eval_basis_vec(xi, &mut a);
            tk.eval_basis_vec(xi, &mut b);
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-12, "TriND2 vs TriNDk(2): {x} vs {y}");
            }
        }
        let u = random_coeffs(space.n_dofs(), 0xD32);
        let locals: Vec<Vec<f64>> = (0..mesh.n_elements() as u32)
            .map(|e| local_coeffs_no_block(&space, e, &u))
            .collect();
        let mut worst = 0.0_f64;
        for e0 in 0..mesh.n_elements() as u32 {
            for e1 in e0 + 1..mesh.n_elements() as u32 {
                let v0 = mesh.element_nodes(e0);
                let v1 = mesh.element_nodes(e1);
                let mut shared = Vec::new();
                for (li, lj) in [(0usize, 1usize), (1, 2), (2, 0)] {
                    let k = EdgeKey::new(v0[li], v0[lj]);
                    if (0..3).any(|i| {
                        (0..3).any(|j| i != j && EdgeKey::new(v1[i], v1[j]) == k)
                    }) {
                        shared.push(k);
                    }
                }
                for key in shared {
                    let pa = mesh.node_coords(key.0);
                    let pb = mesh.node_coords(key.1);
                    let dir = [pb[0] - pa[0], pb[1] - pa[1]];
                    for &s in &[0.3f64, 0.6] {
                        let x = [pa[0] + s * dir[0], pa[1] + s * dir[1]];
                        let t0 = tri_trace_at(&mesh, &space, e0, &locals[e0 as usize], x, dir);
                        let t1 = tri_trace_at(&mesh, &space, e1, &locals[e1 as usize], x, dir);
                        worst = worst.max((t0 - t1).abs());
                    }
                }
            }
        }
        eprintln!("TriND2 worst shared-edge trace mismatch: {worst:.3e}");
        assert!(worst <= 1e-13, "ND2 trace worst {worst:.3e}");
    }
}

// ─── 3-D: tet face trace through the D37 block transform ────────────────────

/// Inverse-transpose of a 3×3 matrix and its determinant.
fn inv_transpose3(j: [[f64; 3]; 3]) -> ([[f64; 3]; 3], f64) {
    let c = [
        [
            j[1][1] * j[2][2] - j[1][2] * j[2][1],
            j[1][2] * j[2][0] - j[1][0] * j[2][2],
            j[1][0] * j[2][1] - j[1][1] * j[2][0],
        ],
        [
            j[0][2] * j[2][1] - j[0][1] * j[2][2],
            j[0][0] * j[2][2] - j[0][2] * j[2][0],
            j[0][1] * j[2][0] - j[0][0] * j[2][1],
        ],
        [
            j[0][1] * j[1][2] - j[0][2] * j[1][1],
            j[0][2] * j[1][0] - j[0][0] * j[1][2],
            j[0][0] * j[1][1] - j[0][1] * j[1][0],
        ],
    ];
    let det = j[0][0] * c[0][0] + j[0][1] * c[0][1] + j[0][2] * c[0][2];
    // J⁻¹ = Cᵀ/det, hence J⁻ᵀ = C/det.
    let mut jit = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for s in 0..3 {
            jit[r][s] = c[r][s] / det;
        }
    }
    (jit, det)
}

/// Tangential trace component of the reconstructed field at physical point
/// `x` in element `e`, along `dir`.
fn tet_trace_at<M: MeshTopology>(
    mesh: &M,
    space: &HCurlSpace<M>,
    e: u32,
    u: &[f64],
    x: [f64; 3],
    dir: [f64; 3],
) -> f64 {
    let verts = mesh.element_nodes(e);
    let dofs = space.element_dofs(e);
    let p0 = mesh.node_coords(verts[0]);
    let mut j = [[0.0_f64; 3]; 3];
    for (c, lv) in [1usize, 2, 3].iter().enumerate() {
        let p = mesh.node_coords(verts[*lv]);
        for d in 0..3 {
            j[d][c] = p[d] - p0[d];
        }
    }
    let (jit, _det) = inv_transpose3(j);
    // ξ = J⁻¹ (x − p0)
    let rx = [x[0] - p0[0], x[1] - p0[1], x[2] - p0[2]];
    let mut xi = [0.0_f64; 3];
    for d in 0..3 {
        xi[d] = jit[0][d] * rx[0] + jit[1][d] * rx[1] + jit[2][d] * rx[2];
    }
    let elem = TetNDk::new(space.order() as usize);
    let n = elem.n_dofs();
    let mut vals = vec![0.0; n * 3];
    elem.eval_basis_vec(&xi, &mut vals);
    let mut uh = [0.0_f64; 3];
    for (i, _g) in dofs.iter().enumerate() {
        let phi = [vals[i * 3], vals[i * 3 + 1], vals[i * 3 + 2]];

        let c = u[i];
        for r in 0..3 {
            uh[r] += c * (jit[r][0] * phi[0] + jit[r][1] * phi[1] + jit[r][2] * phi[2]);
        }
    }
    uh[0] * dir[0] + uh[1] * dir[1] + uh[2] * dir[2]
}

/// Worst shared-face tangential-trace mismatch of the interpolant, evaluated
/// with (`use_blocks = true`) or without the D37 face block transform.
fn tet_face_trace_mismatch<M: MeshTopology>(
    mesh: &M,
    space: &HCurlSpace<M>,
    f: &dyn Fn(&[f64]) -> Vec<f64>,
    use_blocks: bool,
) -> f64 {
    let u_canon = space.interpolate_vector(f);
    let locals: Vec<Vec<f64>> = (0..mesh.n_elements() as u32)
        .map(|e| {
            if use_blocks {
                local_coeffs(space, e, u_canon.as_slice())
            } else {
                local_coeffs_no_block(space, e, u_canon.as_slice())
            }
        })
        .collect();

    let mut faces: HashMap<FaceKey, Vec<u32>> = HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let v = mesh.element_nodes(e);
        for (a, b, c) in [(1usize, 2usize, 3usize), (0, 2, 3), (0, 1, 3), (0, 1, 2)] {
            faces
                .entry(FaceKey::new(v[a], v[b], v[c]))
                .or_default()
                .push(e);
        }
    }
    let mut worst = 0.0_f64;
    for (key, users) in &faces {
        if users.len() < 2 {
            continue;
        }
        let pa = mesh.node_coords(key.0);
        let pb = mesh.node_coords(key.1);
        let pc = mesh.node_coords(key.2);
        let d0 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let d1 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        for (s, t) in [(0.3f64, 0.3f64), (0.5, 0.2), (0.2, 0.6)] {
            let x = [
                pa[0] + s * d0[0] + t * d1[0],
                pa[1] + s * d0[1] + t * d1[1],
                pa[2] + s * d0[2] + t * d1[2],
            ];
            for dir in [d0, d1] {
                let mut tr = Vec::new();
                for &e in users {
                    tr.push(tet_trace_at(mesh, space, e, &locals[e as usize], x, dir));
                }
                for q in &tr[1..] {
                    worst = worst.max((q - tr[0]).abs());
                }
            }
        }
    }
    worst
}

#[test]
fn d37_tet_nd2_face_trace_needs_block_transform() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let space = HCurlSpace::new(mesh.clone(), 2);
    let f = |x: &[f64]| vec![1.0 + 0.5 * x[1], 1.0 - 0.25 * x[2], 0.5 * x[0]];
    let with = tet_face_trace_mismatch(&mesh, &space, &f, true);
    let without = tet_face_trace_mismatch(&mesh, &space, &f, false);
    eprintln!("tet ND2 shared-face trace: with blocks {with:.3e}, without {without:.3e}");
    assert!(
        with <= 1e-12,
        "tet ND2 shared-face trace with D37 blocks {with:.3e} > 1e-12"
    );
    assert!(
        without > 1e-3,
        "the pre-D37 scalar-sign pairing was expected to be non-conforming, got {without:.3e}"
    );
}

#[test]
fn d38_tet_ndk3_face_trace_and_interpolation() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let space = HCurlSpace::new(mesh.clone(), 3);
    assert_eq!(space.element_dofs(0).len(), 45);
    // N_3 ⊃ P_2³: linear fields are reproduced exactly, so the interpolant's
    // trace must match the exact field's on every shared face.
    let f = |x: &[f64]| vec![1.0 + x[1], 1.0 - 2.0 * x[2], 0.5 * x[0]];
    let exact = |x: &[f64]| vec![1.0 + x[1], 1.0 - 2.0 * x[2], 0.5 * x[0]];
    let with = tet_face_trace_mismatch(&mesh, &space, &f, true);
    let without = tet_face_trace_mismatch(&mesh, &space, &f, false);
    eprintln!("tet ND3 shared-face trace: with blocks {with:.3e}, without {without:.3e}");
    assert!(with <= 1e-12, "tet ND3 shared-face trace {with:.3e} > 1e-12");
    assert!(without > 1e-3, "expected non-conforming without blocks, got {without:.3e}");

    // exact reconstruction of the linear field on every element
    let u = space.interpolate_vector(&f);
    let locals: Vec<Vec<f64>> = (0..mesh.n_elements() as u32)
        .map(|e| local_coeffs(&space, e, u.as_slice()))
        .collect();
    let mut worst = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let p0 = mesh.node_coords(verts[0]);
        let mut j = [[0.0_f64; 3]; 3];
        for (c, lv) in [1usize, 2, 3].iter().enumerate() {
            let p = mesh.node_coords(verts[*lv]);
            for d in 0..3 {
                j[d][c] = p[d] - p0[d];
            }
        }
        let (jit, _det) = inv_transpose3(j);
        let xi = [0.23, 0.31, 0.17];
        let xp = [
            p0[0] + j[0][0] * xi[0] + j[0][1] * xi[1] + j[0][2] * xi[2],
            p0[1] + j[1][0] * xi[0] + j[1][1] * xi[1] + j[1][2] * xi[2],
            p0[2] + j[2][0] * xi[0] + j[2][1] * xi[1] + j[2][2] * xi[2],
        ];
        let want = exact(&xp);
        let elem = TetNDk::new(3);
        let n = elem.n_dofs();
        let mut vals = vec![0.0; n * 3];
        elem.eval_basis_vec(&xi, &mut vals);
        let mut uh = [0.0_f64; 3];
        for (i, _g) in dofs.iter().enumerate() {
            let c = locals[e as usize][i];
            for r in 0..3 {
                uh[r] += c
                    * (jit[r][0] * vals[i * 3]
                        + jit[r][1] * vals[i * 3 + 1]
                        + jit[r][2] * vals[i * 3 + 2]);
            }
        }
        for d in 0..3 {
            worst = worst.max((uh[d] - want[d]).abs());
        }
    }
    assert!(worst <= 1e-11, "tet ND3 linear reconstruction {worst:.3e}");
}

#[test]
fn d38_tet_nd2_interpolation_and_element_unchanged() {
    // k = 2: the D32 guarantee (a constant field is reproduced exactly) holds
    // on every element of a multi-tet mesh once the D37 face blocks are
    // applied, and the generic TetNDk(2) has the same basis as TetND2.
    let mesh = Mesh::<3>::unit_cube_tet(1);
    let space = HCurlSpace::new(mesh.clone(), 2);
    let f = |_x: &[f64]| vec![1.0, 0.0, 0.0];
    let u = space.interpolate_vector(&f);
    let mut worst = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let locals = local_coeffs(&space, e, u.as_slice());
        for x in [[0.2f64, 0.3, 0.1], [0.35, 0.15, 0.05]] {
            let tr = tet_trace_at(&mesh, &space, e, &locals, x, [1.0, 0.0, 0.0]);
            worst = worst.max((tr - 1.0).abs());
        }
    }
    assert!(worst <= 1e-12, "ND2 constant-field reconstruction {worst:.3e}");
    // TetND2 and TetNDk(2) agree (same functionals ⟹ same dual basis)
    let a = TetND2;
    let b = TetNDk::new(2);
    let xi = [0.111, 0.253, 0.198];
    let mut va = vec![0.0; 60];
    let mut vb = vec![0.0; 60];
    a.eval_basis_vec(&xi, &mut va);
    b.eval_basis_vec(&xi, &mut vb);
    for (x, y) in va.iter().zip(vb.iter()) {
        assert!((x - y).abs() < 1e-12, "TetND2 vs TetNDk(2): {x} vs {y}");
    }
}
