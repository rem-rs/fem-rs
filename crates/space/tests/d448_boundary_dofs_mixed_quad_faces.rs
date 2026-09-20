//! D448 — pinning test for the D354 side effect: on a **mixed** 3-D mesh at
//! p = 2, `constraints::dirichlet::boundary_dofs` must return the FULL dof
//! set of a quadrilateral boundary face — vertex + edge dofs *and the face
//! centre dof* (looked up through `DofManager::quad_face_pk_map`, which
//! D354 filled for mixed meshes; before that the lookup found nothing, so
//! Dirichlet silently missed the face centre on every quad boundary face).
//!
//! The mesh is the d349/d354 `tinyzoo` (1 hex + 1 prism + 1 pyramid + 1 tet
//! glued into a 2×1×1 block, 12 vertices), extended with its 14-face boundary
//! table (6 quads + 8 tris, each face tagged).  Two probes:
//!
//! 1. the hex face `y = 0, x ∈ [0,1]` (tag 1): `boundary_dofs` must equal the
//!    independent geometric dof set of the closed face — 4 vertex dofs, 4
//!    edge dofs, 1 face centre = 9 at p = 2;
//! 2. the tet face `x = 2, y + z ≤ 1` (tag 2): a triangular face has no
//!    interior dof at p = 2, so the answer must be exactly vertices + edges
//!    (6) — and no face-centre dof exists to miss.
//!
//! End-to-end corroboration: build the dof-graph Laplacian of the zoo (SPD
//! after pinning), pin exactly the `boundary_dofs` result of each probe face
//! to per-dof values via `apply_dirichlet`, dense-solve, and require the
//! solution to equal the prescribed values at **every geometric dof of the
//! face**.  A boundary_dofs that misses the quad face centre fails this: the
//! centre dof is left free and solves to something else.

use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::constraints::apply_dirichlet;
use fem_space::DofManager;
/// `data/tinyzoo-3d.mesh` as a programmatic mesh (d349's vertex/element
/// tables, MFEM's own tet orientation) plus the full boundary table: 6 quad
/// + 8 tri faces; the two probe faces carry tags 1 (quad) and 2 (tri).
fn tinyzoo_with_boundary() -> Mesh<3> {
    let coords = vec![
        0., 0., 0., 1., 0., 0., 2., 0., 0., 0., 1., 0., 1., 1., 0., 2., 1., 0., //
        0., 0., 1., 1., 0., 1., 2., 0., 1., 0., 1., 1., 1., 1., 1., 2., 1., 1.,
    ];
    let conn = vec![
        0, 1, 4, 3, 6, 7, 10, 9, // hex
        4, 1, 5, 10, 7, 11, // prism
        11, 7, 1, 5, 8, // pyramid
        5, 8, 1, 2, // tet (MFEM's own vertex order)
    ];
    let mut mesh = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1, 1, 1],
        ElementType::Hex8,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    mesh.elem_types = Some(vec![
        ElementType::Hex8,
        ElementType::Prism6,
        ElementType::Pyramid5,
        ElementType::Tet4,
    ]);
    mesh.elem_offsets = Some(vec![0usize, 8, 14, 19, 23]);

    // Boundary faces of the 2×1×1 block.  The closed zoo surface splits into
    // 6 quads and 8 tris (pyramid/tet/prism faces fill x ∈ [1,2]).
    let quads: [[u32; 4]; 6] = [
        [0, 1, 7, 6],   // y=0, x∈[0,1]  (hex)      — D448 probe QUAD, tag 1
        [3, 4, 10, 9],  // y=1, x∈[0,1]  (hex)
        [4, 5, 11, 10], // y=1, x∈[1,2]  (prism)
        [0, 3, 9, 6],   // x=0           (hex)
        [0, 1, 4, 3],   // z=0, x∈[0,1]  (hex)
        [6, 7, 10, 9],  // z=1, x∈[0,1]  (hex)
    ];
    let tris: [[u32; 3]; 8] = [
        [1, 7, 8],   // y=0           (pyramid)
        [8, 1, 2],   // y=0, x∈[1,2]  (tet)
        [1, 4, 5],   // z=0, x∈[1,2]  (prism)
        [5, 1, 2],   // z=0, x∈[1,2]  (tet)
        [7, 10, 11], // z=1, x∈[1,2]  (prism)
        [7, 11, 8],  // z=1, x∈[1,2]  (pyramid)
        [5, 11, 8],  // x=2, y+z≥1    (pyramid)
        [5, 8, 2],   // x=2, y+z≤1    (tet)      — D448 probe TRI, tag 2
    ];
    let mut face_conn = Vec::new();
    let mut face_offsets = vec![0usize];
    let mut face_types = Vec::new();
    let mut face_tags = Vec::new();
    for (i, q) in quads.iter().enumerate() {
        face_conn.extend_from_slice(q);
        face_offsets.push(face_conn.len());
        face_types.push(ElementType::Quad4);
        face_tags.push(if i == 0 { 1 } else { 7 });
    }
    for (i, t) in tris.iter().enumerate() {
        face_conn.extend_from_slice(t);
        face_offsets.push(face_conn.len());
        face_types.push(ElementType::Tri3);
        face_tags.push(if i == 7 { 2 } else { 7 });
    }
    mesh.face_conn = face_conn;
    mesh.face_tags = face_tags;
    mesh.face_types = Some(face_types);
    mesh.face_offsets = Some(face_offsets);
    mesh
}

/// Independent geometric oracle: every dof whose coordinate lies in the
/// closed region `pred`, sorted.
fn geometric_face_dofs(dm: &DofManager, pred: impl Fn([f64; 3]) -> bool) -> Vec<u32> {
    let mut out: Vec<u32> = (0..dm.n_dofs as u32)
        .filter(|&d| {
            let c = dm.dof_coord(d);
            pred([c[0], c[1], c[2]])
        })
        .collect();
    out.sort_unstable();
    out
}

const TOL: f64 = 1e-12;

/// Dof-graph Laplacian of the whole space: one clique per element's dof set.
/// Connected ⇒ the only null vector is the global constant ⇒ SPD once any
/// dof is pinned — a valid stand-in for a stiffness matrix here (the pinned
/// components' values are what matters, not the PDE).
fn laplacian(mesh: &Mesh<3>, dm: &DofManager) -> Vec<Vec<f64>> {
    let n = dm.n_dofs;
    let mut a = vec![vec![0.0; n]; n];
    for e in 0..mesh.n_elements() as u32 {
        let dofs = dm.element_dofs(e);
        for &i in dofs {
            for &j in dofs {
                if i != j {
                    a[i as usize][j as usize] -= 1.0;
                    a[i as usize][i as usize] += 1.0;
                }
            }
        }
    }
    a
}

/// Dense Gauss solve with partial pivoting (the system is ≤ 46×46).
fn solve_dense(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for c in 0..n {
        let p = (c..n).fold(c, |best, r| if a[r][c].abs() > a[best][c].abs() { r } else { best });
        a.swap(c, p);
        b.swap(c, p);
        assert!(a[c][c].abs() > 1e-12, "singular system at column {c}");
        for r in (c + 1)..n {
            let f = a[r][c] / a[c][c];
            if f != 0.0 {
                for k in c..n {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = ((r + 1)..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x
}

/// Pin `ess` to per-dof values on the Laplacian system, dense-solve, return x.
fn pinned_solution(mesh: &Mesh<3>, dm: &DofManager, ess: &[u32]) -> Vec<f64> {
    let n = dm.n_dofs;
    let a = laplacian(mesh, dm);
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                coo.add(i, j, v);
            }
        }
    }
    let mut mat = coo.into_csr();
    // Distinct deterministic prescribed value per dof.
    let values: Vec<f64> = ess.iter().map(|&d| 3.0 + 0.25 * d as f64).collect();
    let mut rhs = vec![1.0; n];
    apply_dirichlet(&mut mat, &mut rhs, ess, &values);
    let dense_rows = mat.to_dense();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| dense_rows[i * n..(i + 1) * n].to_vec()).collect();
    solve_dense(&rows, &rhs)
}

/// Probe 1 (quad): the hex face y=0, x∈[0,1] must pin 4 vertex + 4 edge +
/// 1 face-centre dof — exactly the closed face's geometric dof set.  The
/// face-centre entry is the D354/D448 point: `quad_face_pk_map` lookup.
#[test]
fn quad_boundary_face_pins_its_face_centre() {
    let mesh = tinyzoo_with_boundary();
    let dm = DofManager::new(&mesh, 2);
    let ess = fem_space::constraints::boundary_dofs(&mesh, &dm, &[1]);
    let geo = geometric_face_dofs(&dm, |p| {
        p[1].abs() <= TOL
            && -TOL <= p[0]
            && p[0] <= 1.0 + TOL
            && -TOL <= p[2]
            && p[2] <= 1.0 + TOL
    });
    assert_eq!(geo.len(), 9, "closed quad face at p=2: 4 verts + 4 edges + 1 centre");
    assert_eq!(ess, geo, "boundary_dofs(tag 1) vs the geometric dof set of the quad face");
    // The face-centre dof is genuinely among them (an interior point of the face).
    let centre = geometric_face_dofs(&dm, |p| {
        p[1].abs() <= TOL && (0.0 + TOL) < p[0] && p[0] < (1.0 - TOL) && (0.0 + TOL) < p[2] && p[2] < (1.0 - TOL)
    });
    assert_eq!(centre.len(), 1, "exactly one interior (face-centre) dof on the quad face");
    assert!(
        ess.binary_search(&centre[0]).is_ok(),
        "quad face-centre dof {} missing from boundary_dofs — Dirichlet would silently skip it",
        centre[0]
    );
}

/// Probe 2 (tri): the tet face x=2, y+z≤1 has no interior dof at p=2 — the
/// answer must be exactly its 3 vertex + 3 edge dofs.
#[test]
fn tri_boundary_face_pins_verts_and_edges_only() {
    let mesh = tinyzoo_with_boundary();
    let dm = DofManager::new(&mesh, 2);
    let ess = fem_space::constraints::boundary_dofs(&mesh, &dm, &[2]);
    let geo = geometric_face_dofs(&dm, |p| {
        (p[0] - 2.0).abs() <= TOL
            && p[1] >= -TOL
            && p[2] >= -TOL
            && p[1] + p[2] <= 1.0 + TOL
    });
    assert_eq!(geo.len(), 6, "closed tri face at p=2: 3 verts + 3 edges, no interior");
    assert_eq!(ess, geo, "boundary_dofs(tag 2) vs the geometric dof set of the tri face");
    // No dof strictly inside the triangle (face-centre analogue must not exist).
    let interior = geometric_face_dofs(&dm, |p| {
        (p[0] - 2.0).abs() <= TOL && p[1] > TOL && p[2] > TOL && p[1] + p[2] < 1.0 - TOL
    });
    assert!(interior.is_empty(), "unexpected interior dof inside the p=2 tri face");
}

/// End-to-end: after pinning `boundary_dofs(tag)` on the Laplacian system and
/// solving, the solution equals the prescribed values at EVERY geometric dof
/// of the face.  If boundary_dofs missed the quad face centre (the pre-D354
/// behavior), that dof stays free and the last assertion fails.
#[test]
fn pinned_face_dofs_reach_the_solution() {
    let mesh = tinyzoo_with_boundary();
    let dm = DofManager::new(&mesh, 2);

    for (tag, geo) in [
        (
            1,
            geometric_face_dofs(&dm, |p| {
                p[1].abs() <= TOL
                    && -TOL <= p[0]
                    && p[0] <= 1.0 + TOL
                    && -TOL <= p[2]
                    && p[2] <= 1.0 + TOL
            }),
        ),
        (
            2,
            geometric_face_dofs(&dm, |p| {
                (p[0] - 2.0).abs() <= TOL
                    && p[1] >= -TOL
                    && p[2] >= -TOL
                    && p[1] + p[2] <= 1.0 + TOL
            }),
        ),
    ] {
        let ess = fem_space::constraints::boundary_dofs(&mesh, &dm, &[tag]);
        assert_eq!(ess, geo, "tag {tag}: boundary_dofs == geometric face dofs");
        let x = pinned_solution(&mesh, &dm, &ess);
        let values: Vec<f64> = ess.iter().map(|&d| 3.0 + 0.25 * d as f64).collect();
        for (&d, &w) in geo.iter().zip(values.iter()) {
            assert!(
                (x[d as usize] - w).abs() <= 1e-9,
                "tag {tag}: solution at face dof {d} = {}, prescribed {w} — \
                 the pin did not reach this dof",
                x[d as usize]
            );
        }
    }
}
