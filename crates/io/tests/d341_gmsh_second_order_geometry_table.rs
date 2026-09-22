//! D341 — functional acceptance for the Gmsh **second-order geometry table**.
//!
//! D319 put the file's node rows into fem-rs's canonical (evaluation-element)
//! order, but the reader still left `Mesh::geometry = None`, so
//! `Mesh::geom_order()` reported 1 for a quadratic import and
//! `Mesh::element_jacobian` evaluated an **order-1** factory element (3/4/8
//! dofs) against 6/9/10/18/27 connectivity entries — an index-out-of-bounds
//! panic at `crates/mesh/src/simplex.rs:383` (measured in round 45, see
//! `tmp/d339/EVIDENCE.md` §5.5).
//!
//! The fix attaches a `GeometryData` whose `conn` **is** the (already
//! permuted) element connectivity and whose `coords` **is** the mesh node
//! table: a Gmsh second-order file supplies every geometry node directly, so
//! nothing has to be invented or renumbered.
//!
//! Acceptance below is by *function*: the file is written with
//! `file_row[perm[m]] = g(ref[m])` for the quadratic map `g` of D319, and after
//! reading, `Mesh::element_jacobian` (not a test-local sampler) must reproduce
//! `g` and `∇g` — it is the very function that used to panic.
//!
//! `tmp/d343/EVIDENCE.md` records the numbers; `tmp/d339/EVIDENCE.md` has the
//! MFEM `GetNodeMap` derivation of the permutations.

use fem_mesh::element_type::ElementType;
use fem_mesh::MeshTopology;

fn write_temp(name: &str, text: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, text).expect("write temp msh");
    p
}

/// Gmsh v4.1 file with one 3-D element of `code` (same shape as the D319 and
/// `d297` fixtures).
fn gmsh_v41_single_3d(code: i32, coords: &[[f64; 3]]) -> String {
    let n = coords.len();
    let mut s = String::new();
    s.push_str("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
    s.push_str("$Entities\n0 0 0 1\n$EndEntities\n");
    s.push_str(&format!("$Nodes\n1 {n} 1 {n}\n"));
    s.push_str(&format!("3 1 0 {n}\n"));
    for i in 0..n {
        s.push_str(&format!("{}\n", i + 1));
    }
    for c in coords.iter() {
        s.push_str(&format!("{} {} {}\n", c[0], c[1], c[2]));
    }
    s.push_str("$EndNodes\n");
    s.push_str("$Elements\n1 1 1 1\n");
    s.push_str(&format!("3 1 {} 1\n", code));
    s.push_str("1 ");
    for i in 0..n {
        s.push_str(&format!("{} ", i + 1));
    }
    s.push('\n');
    s.push_str("$EndElements\n");
    s
}

/// Gmsh v4.1 file with one 2-D element of `code`.
fn gmsh_v41_single_2d(code: i32, coords: &[[f64; 3]]) -> String {
    let n = coords.len();
    let mut s = String::new();
    s.push_str("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
    s.push_str("$Entities\n0 0 1 0\n$EndEntities\n");
    s.push_str(&format!("$Nodes\n1 {n} 1 {n}\n"));
    s.push_str(&format!("2 1 0 {n}\n"));
    for i in 0..n {
        s.push_str(&format!("{}\n", i + 1));
    }
    for c in coords.iter() {
        s.push_str(&format!("{} {} {}\n", c[0], c[1], c[2]));
    }
    s.push_str("$EndNodes\n");
    s.push_str("$Elements\n1 1 1 1\n");
    s.push_str(&format!("2 1 {} 1\n", code));
    s.push_str("1 ");
    for i in 0..n {
        s.push_str(&format!("{} ", i + 1));
    }
    s.push('\n');
    s.push_str("$EndElements\n");
    s
}

/// The quadratic map of D319, `g`, and its Jacobian `∇g` (3 coordinates).
fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}
fn j3(x: [f64; 3]) -> [[f64; 3]; 3] {
    [
        [1.0 + 0.4 * x[0], 0.1 * x[2], 0.1 * x[1]],
        [0.05 * x[2], 1.0, 0.05 * x[0]],
        [0.1 * x[1], 0.1 * x[0], 1.0],
    ]
}

/// 2-D version (`Mesh<2>` cannot carry a z-component).
fn g2(x: [f64; 2]) -> [f64; 2] {
    [x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1], x[1] + 0.05 * x[0] * x[1]]
}
fn j2(x: [f64; 2]) -> [[f64; 2]; 2] {
    [
        [1.0 + 0.4 * x[0] + 0.1 * x[1], 0.1 * x[0]],
        [0.05 * x[1], 1.0 + 0.05 * x[0]],
    ]
}

/// `file[perm[m]] = g(ref_node m)`.
fn file_rows(el: &dyn fem_element::ReferenceElement, perm: &[usize]) -> Vec<[f64; 3]> {
    let rp = el.dof_coords();
    assert_eq!(rp.len(), perm.len());
    let mut file = vec![[0.0_f64; 3]; perm.len()];
    for m in 0..perm.len() {
        let p = &rp[m];
        file[perm[m]] = g3([p[0], p[1], p.get(2).copied().unwrap_or(0.0)]);
    }
    file
}

/// Read a single-element 3-D file, assert `geom_order() == 2` and that
/// `Mesh::element_jacobian` reproduces `g`/`∇g` at `pts`.  Returns the worst
/// deviations `(map, jacobian)`.
fn check_3d(name: &str, code: i32, el: &dyn fem_element::ReferenceElement, perm: &[usize], pts: &[[f64; 3]]) -> (f64, f64) {
    let coords = file_rows(el, perm);
    let path = write_temp(name, &gmsh_v41_single_3d(code, &coords));
    let msh = fem_io::gmsh::read_msh_file(&path).expect("read second-order msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh3d.expect("3-D mesh");

    assert_eq!(
        mesh.geom_order(),
        2,
        "{name}: a second-order import must report geom_order 2 (D341)"
    );
    assert_eq!(mesh.element_nodes(0).len(), el.n_dofs(), "{name}: connectivity length");

    let (mut wx, mut wj) = (0.0_f64, 0.0_f64);
    for xi in pts {
        let (j, _det, x) = mesh.element_jacobian(0, xi);
        let want = g3(*xi);
        let wjq = j3(*xi);
        for i in 0..3 {
            wx = wx.max((x[i] - want[i]).abs());
            for d in 0..3 {
                wj = wj.max((j[(i, d)] - wjq[i][d]).abs());
            }
        }
    }
    eprintln!("D341 {name}: max |x-g| = {wx:.3e}, max |J-dg| = {wj:.3e}");
    assert!(wx < 1e-12, "{name}: geometry map wrong ({wx:.3e})");
    assert!(wj < 1e-12, "{name}: geometry Jacobian wrong ({wj:.3e})");
    (wx, wj)
}

// ─── 3-D complete second-order families ──────────────────────────────────────

#[test]
fn d341_tet10_geometry_table() {
    let el = fem_element::lagrange::H1TetPk::new(2);
    let perm = [0usize, 1, 2, 3, 4, 6, 7, 5, 9, 8];
    check_3d(
        "d341_tet10.msh",
        11,
        &el,
        &perm,
        &[[0.1, 0.2, 0.3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], [0.25, 0.25, 0.25]],
    );
}

#[test]
fn d341_hex27_geometry_table() {
    let el = fem_element::lagrange::factory::HexQk::new(2);
    // Gmsh file order → fem-rs (= MFEM `H1_HexahedronElement(2)` order since
    // D31); matches `GMSH_PERM_HEX27` in `crates/io/src/gmsh.rs`.
    let perm = [
        0usize, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15, 20, 21, 23, 24,
        22, 25, 26,
    ];
    check_3d("d341_hex27.msh", 12, &el, &perm, &[[-0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [0.75, -0.75, 0.25]]);
}

#[test]
fn d341_prism18_geometry_table() {
    let el = fem_element::lagrange::PrismPk::new(2);
    let perm = [0usize, 1, 2, 6, 9, 7, 8, 10, 11, 15, 17, 16, 3, 4, 5, 12, 14, 13];
    check_3d(
        "d341_prism18.msh",
        13,
        &el,
        &perm,
        &[[0.3, 0.2, 0.2], [0.5, 1.0 / 3.0, 1.0 / 3.0], [0.9, 0.05, 0.05]],
    );
}

// ─── 2-D complete second-order families ──────────────────────────────────────

/// Read a single-element 2-D file and check `element_jacobian` against the 2-D
/// quadratic map (identity permutation — Gmsh's order already is fem-rs's).
fn check_2d(name: &str, code: i32, el: &dyn fem_element::ReferenceElement, pts: &[[f64; 2]]) {
    let rp = el.dof_coords();
    let coords: Vec<[f64; 3]> = rp
        .iter()
        .map(|p| {
            let v = g2([p[0], p[1]]);
            [v[0], v[1], 0.0]
        })
        .collect();
    let path = write_temp(name, &gmsh_v41_single_2d(code, &coords));
    let msh = fem_io::gmsh::read_msh_file(&path).expect("read second-order 2-D msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh2d.expect("2-D mesh");
    assert_eq!(mesh.geom_order(), 2, "{name}: geom_order (D341)");
    assert_eq!(mesh.element_nodes(0).len(), el.n_dofs(), "{name}: connectivity length");

    let (mut wx, mut wj) = (0.0_f64, 0.0_f64);
    for xi in pts {
        let (j, _det, x) = mesh.element_jacobian(0, xi);
        let want = g2(*xi);
        let wjq = j2(*xi);
        for i in 0..2 {
            wx = wx.max((x[i] - want[i]).abs());
            for d in 0..2 {
                wj = wj.max((j[(i, d)] - wjq[i][d]).abs());
            }
        }
    }
    eprintln!("D341 {name}: max |x-g| = {wx:.3e}, max |J-dg| = {wj:.3e}");
    assert!(wx < 1e-12, "{name}: geometry map wrong ({wx:.3e})");
    assert!(wj < 1e-12, "{name}: geometry Jacobian wrong ({wj:.3e})");
}

#[test]
fn d341_tri6_geometry_table() {
    let el = fem_element::lagrange::H1TriPk::new(2);
    check_2d("d341_tri6.msh", 9, &el, &[[0.2, 0.3], [1.0 / 3.0, 1.0 / 3.0]]);
}

#[test]
fn d341_quad9_geometry_table() {
    let el = fem_element::lagrange::factory::QuadQk::new(2);
    check_2d("d341_quad9.msh", 10, &el, &[[0.25, 0.75], [0.5, 0.5], [0.1, 0.9]]);
}

// ─── Deliberate exclusions ───────────────────────────────────────────────────

/// The incomplete serendipity families keep the linear view (D244): their
/// geometry comes from `findpts::incomplete`, which owns the Gmsh slot order
/// and is reached **before** `geom_order` is consulted.  `element_jacobian`
/// must therefore still work, and `geom_order()` must stay 1.
#[test]
fn d341_incomplete_families_keep_the_linear_view() {
    // Quad8 (16): 8 nodes, all distinct in x so a scramble would show.
    let coords: Vec<[f64; 3]> = (0..8).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
    let path = write_temp("d341_quad8.msh", &gmsh_v41_single_2d(16, &coords));
    let msh = fem_io::gmsh::read_msh_file(&path).expect("read quad8");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh2d.expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Quad8);
    assert_eq!(mesh.geom_order(), 1, "Quad8 must not get a geometry table");
    assert!(mesh.geometry.is_none());
    // The serendipity path still evaluates (no panic, no factory mismatch).
    let (j, det, _x) = mesh.element_jacobian(0, &[0.25, 0.25]);
    assert!(j.iter().all(|v| v.is_finite()) && det.is_finite());

    // Hex20 (17) / Prism15 (18) likewise.
    for (code, n, name) in [(17, 20usize, "d341_hex20.msh"), (18, 15, "d341_prism15.msh")] {
        let coords: Vec<[f64; 3]> = (0..n).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
        let path = write_temp(name, &gmsh_v41_single_3d(code, &coords));
        let msh = fem_io::gmsh::read_msh_file(&path).expect("read incomplete family");
        let _ = std::fs::remove_file(&path);
        let mesh = msh.mesh3d.expect("3-D mesh");
        assert_eq!(mesh.geom_order(), 1, "{name}: must not get a geometry table");
        assert!(mesh.geometry.is_none(), "{name}");
    }
}

/// A mesh that **mixes** linear and quadratic blocks (e.g. Tet4 + Tet10) cannot
/// carry a `GeometryData`: it has a single `nodes_per_elem` stride, so the
/// reader deliberately keeps the linear view.  The quadratic block therefore
/// still panics in `element_jacobian` — pinned here as the remaining D341
/// limitation (fixing it needs a per-element stride, i.e. a mesh-side change).
#[test]
fn d341_mixed_order_mesh_keeps_the_linear_view() {
    // v2 ASCII, 4 vertices + 6 Tet10 edge nodes.
    let mut data = String::from(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n10\n\
         1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\n4 0.0 0.0 1.0\n\
         5 0.5 0.0 0.0\n6 0.5 0.5 0.0\n7 0.0 0.5 0.0\n\
         8 0.0 0.0 0.5\n9 0.5 0.0 0.5\n10 0.0 0.5 0.5\n$EndNodes\n$Elements\n2\n",
    );
    data.push_str("1 4 2 1 1 1 2 3 4\n"); // Tet4
    data.push_str("2 11 2 1 1 1 2 3 4 5 6 7 8 9 10\n"); // Tet10
    data.push_str("$EndElements\n");

    let msh = fem_io::gmsh::read_msh(&data.as_bytes() as &[u8]).expect("read mixed v2 mesh");
    let mesh = msh.mesh3d.expect("3-D mesh");
    assert!(mesh.is_mixed(), "fixture must be mixed");
    assert_eq!(mesh.element_type(0), ElementType::Tet4);
    assert_eq!(
        mesh.geom_order(),
        1,
        "a mixed-order mesh keeps the linear view (single nodes_per_elem stride)"
    );
    assert!(mesh.geometry.is_none());
    // The linear block evaluates fine...
    let (j, det, _x) = mesh.element_jacobian(0, &[0.1, 0.2, 0.3]);
    assert!(j.iter().all(|v| v.is_finite()) && det.is_finite());
    // ...the quadratic one cannot (documented D341 limitation).
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mesh.element_jacobian(1, &[0.1, 0.2, 0.3])
    }));
    assert!(res.is_err(), "mixed-order quadratic block expected to still panic");
}

/// `Pyramid13` (Gmsh code 19) is **still** the open D341 gap: fem-rs maps code
/// 19 to `Pyramid13` (13 nodes) while the factory's quadratic pyramid has 14
/// dofs, and MFEM's 14-node pyramid (code 14) is not accepted by this reader at
/// all.  No geometry table can be attached until that convention is ruled on,
/// so `element_jacobian` still panics on the quadratic block — pinned here so
/// the gap cannot be mistaken for "fixed".
#[test]
fn d341_pyramid13_gap_is_still_open() {
    let coords: Vec<[f64; 3]> = (0..13).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
    let path = write_temp("d341_pyramid13.msh", &gmsh_v41_single_3d(19, &coords));
    let msh = fem_io::gmsh::read_msh_file(&path).expect("read pyramid13");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh3d.expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Pyramid13);
    assert_eq!(mesh.geom_order(), 1, "Pyramid13 stays linear (open D341 gap)");

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mesh.element_jacobian(0, &[0.1, 0.1, 0.2])
    }));
    assert!(
        res.is_err(),
        "Pyramid13 is expected to still panic — if this now succeeds, the \
         Pyramid13/14 convention has been settled and the geometry table should \
         be attached instead"
    );
}
