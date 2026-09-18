//! D319 — functional acceptance for the Gmsh second-order node permutations:
//! after reading, the element connectivity must be in the order fem-rs's own
//! canonical element for that type expects, so that the isoparametric map
//! built from it reproduces the geometry the file was generated from.
//!
//! The file is written with `file_row[k] = g(ref[perm[k]])` for a **quadratic**
//! map `g` (exactly reproducible by an order-2 nodal element) and the
//! permutation `perm` from MFEM's `GetNodeMap` (`tmp/d339/EVIDENCE.md`,
//! probe `tmp/d339/gmsh_nodemap_probe_d319.cpp`).  A reader that applies the
//! same permutation reconstructs `g` to roundoff at *any* sample point; a
//! reader that keeps the file order does not (the node/function pairing is
//! scrambled away from the vertices).  `Tri6` (9) / `Quad9` (10) are checked
//! with the identity, since Gmsh's order already is fem-rs's for those types.
//!
//! See `d319_gmsh_second_order_permutation.rs` for the tables and the
//! canonical-order derivation dump.

use fem_element::ReferenceElement;
use fem_io::gmsh::{read_msh, read_msh_file};
use fem_mesh::element_type::ElementType;
use fem_mesh::MeshTopology;

fn write_temp(name: &str, text: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, text).expect("write temp msh");
    p
}

/// A Gmsh v4.1 file with one 3-D element of `code` at `coords` (file order;
/// same format as `tests/d297_gmsh_high_order_types.rs`).
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

/// A Gmsh v4.1 file with one 2-D element of `code` at `coords`.
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

/// Sample the isoparametric map built from `mesh.element_nodes(0)` with `el`:
/// returns `(x(ξ), ∂x/∂ξ(ξ))`.
fn sample_map(el: &dyn ReferenceElement, nodes: &[u32], coords: &dyn Fn(u32) -> [f64; 3], xi: &[f64]) -> ([f64; 3], [[f64; 3]; 3]) {
    let n = el.n_dofs();
    let rdim = el.dim() as usize; // reference dimension: 2 for surface elements
    assert_eq!(nodes.len(), n, "connectivity length vs element dofs");
    let mut phi = vec![0.0_f64; n];
    let mut grad = vec![0.0_f64; n * rdim];
    el.eval_basis(xi, &mut phi);
    el.eval_grad_basis(xi, &mut grad);
    let mut x = [0.0_f64; 3];
    let mut j = [[0.0_f64; 3]; 3];
    for k in 0..n {
        let c = coords(nodes[k]);
        for i in 0..3 {
            x[i] += phi[k] * c[i];
            for d in 0..rdim {
                j[i][d] += c[i] * grad[k * rdim + d];
            }
        }
    }
    (x, j)
}

/// The quadratic map used by all checks below, `g` and `∂g/∂ξ`.
fn g_quad(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn jac_g_quad(x: [f64; 3]) -> [[f64; 3]; 3] {
    [
        [1.0 + 0.4 * x[0], 0.1 * x[2], 0.1 * x[1]],
        [0.05 * x[2], 1.0, 0.05 * x[0]],
        [0.1 * x[1], 0.1 * x[0], 1.0],
    ]
}

/// Build the file node list for a canonical element table `perm`:
/// `perm[m]` is the file row fem-rs row `m` must receive, so file row
/// `perm[m]` carries the geometry image of canonical node `m`.
fn file_from_canonical(el: &dyn ReferenceElement, perm: &[usize]) -> Vec<[f64; 3]> {
    let ref_pos = el.dof_coords();
    assert_eq!(perm.len(), ref_pos.len());
    let mut file = vec![[0.0_f64; 3]; perm.len()];
    for m in 0..perm.len() {
        let p = &ref_pos[m];
        file[perm[m]] = g_quad([p[0], p[1], p.get(2).copied().unwrap_or(0.0)]);
    }
    file
}

#[test]
fn d319_tet10_is_permuted_into_fem_rs_order() {
    let el = fem_element::lagrange::H1TetPk::new(2);
    let perm = [0usize, 1, 2, 3, 4, 6, 7, 5, 9, 8];
    let coords = file_from_canonical(&el, &perm);
    let path = write_temp("d319_curve_11.msh", &gmsh_v41_single_3d(11, &coords));
    let msh = read_msh_file(&path).expect("read tet10 msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh3d.expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Tet10);
    let nodes = mesh.element_nodes(0).to_vec();
    let coords_of = |n: u32| -> [f64; 3] {
        let c = mesh.geom_coords_of(n);
        [c[0], c[1], c[2]]
    };
    let (mut wx, mut wj) = (0.0_f64, 0.0_f64);
    for xi in [[0.1, 0.2, 0.3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], [0.25, 0.25, 0.25]] {
        let (x, j) = sample_map(&el, &nodes, &coords_of, &xi);
        let (want, wjq) = (g_quad(xi), jac_g_quad(xi));
        for i in 0..3 {
            wx = wx.max((x[i] - want[i]).abs());
            for d in 0..3 {
                wj = wj.max((j[i][d] - wjq[i][d]).abs());
            }
        }
    }
    eprintln!("D319 Tet10 (type 11): max |x-g| = {wx:.3e}, max |J-dg| = {wj:.3e}");
    assert!(wx < 1e-12, "Tet10 geometry map scrambled: {wx:.3e}");
    assert!(wj < 1e-12, "Tet10 Jacobian scrambled: {wj:.3e}");
}

#[test]
fn d319_hex27_is_permuted_into_fem_rs_order() {
    let el = fem_element::lagrange::factory::HexQk::new(2);
    let perm = [
        0usize, 1, 2, 3, 4, 5, 6, 7, 12, 14, 15, 10, 9, 11, 18, 17, 8, 13, 19, 16, 22, 23, 21, 24,
        20, 25, 26,
    ];
    let coords = file_from_canonical(&el, &perm);
    let path = write_temp("d319_curve_12.msh", &gmsh_v41_single_3d(12, &coords));
    let msh = read_msh_file(&path).expect("read hex27 msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh3d.expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Hex27);
    let nodes = mesh.element_nodes(0).to_vec();
    let coords_of = |n: u32| -> [f64; 3] {
        let c = mesh.geom_coords_of(n);
        [c[0], c[1], c[2]]
    };
    let (mut wx, mut wj) = (0.0_f64, 0.0_f64);
    for xi in [[-0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [0.75, -0.75, 0.25]] {
        let (x, j) = sample_map(&el, &nodes, &coords_of, &xi);
        let (want, wjq) = (g_quad(xi), jac_g_quad(xi));
        for i in 0..3 {
            wx = wx.max((x[i] - want[i]).abs());
            for d in 0..3 {
                wj = wj.max((j[i][d] - wjq[i][d]).abs());
            }
        }
    }
    eprintln!("D319 Hex27 (type 12): max |x-g| = {wx:.3e}, max |J-dg| = {wj:.3e}");
    assert!(wx < 1e-12, "Hex27 geometry map scrambled: {wx:.3e}");
    assert!(wj < 1e-12, "Hex27 Jacobian scrambled: {wj:.3e}");
}

#[test]
fn d319_prism18_is_permuted_into_fem_rs_order() {
    use fem_element::lagrange::PrismPk;
    let el = PrismPk::new(2);
    let perm = [0usize, 1, 2, 6, 9, 7, 8, 10, 11, 15, 17, 16, 3, 4, 5, 12, 14, 13];
    let coords = file_from_canonical(&el, &perm);
    let path = write_temp("d319_curve_13.msh", &gmsh_v41_single_3d(13, &coords));
    let msh = read_msh_file(&path).expect("read prism18 msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh3d.expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Prism18);
    let nodes = mesh.element_nodes(0).to_vec();
    let coords_of = |n: u32| -> [f64; 3] {
        let c = mesh.geom_coords_of(n);
        [c[0], c[1], c[2]]
    };
    let (mut wx, mut wj) = (0.0_f64, 0.0_f64);
    for xi in [[0.3, 0.2, 0.2], [0.5, 1.0 / 3.0, 1.0 / 3.0], [0.9, 0.05, 0.05]] {
        let (x, j) = sample_map(&el, &nodes, &coords_of, &xi);
        let (want, wjq) = (g_quad(xi), jac_g_quad(xi));
        for i in 0..3 {
            wx = wx.max((x[i] - want[i]).abs());
            for d in 0..3 {
                wj = wj.max((j[i][d] - wjq[i][d]).abs());
            }
        }
    }
    eprintln!("D319 Prism18 (type 13): max |x-g| = {wx:.3e}, max |J-dg| = {wj:.3e}");
    assert!(wx < 1e-12, "Prism18 geometry map scrambled: {wx:.3e}");
    assert!(wj < 1e-12, "Prism18 Jacobian scrambled: {wj:.3e}");
}

/// `Tri6` (type 9): Gmsh's order already is `H1TriPk`'s (vertices → the three
/// edges in `(0,1) (1,2) (2,0)` order → interior), so the file order must be
/// kept.
#[test]
fn d319_tri6_keeps_the_file_order() {
    // A purely 2-D quadratic map (`Mesh<2>` cannot carry a z-component).
    fn g2(x: [f64; 2]) -> [f64; 2] {
        [x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1], x[1] + 0.05 * x[0] * x[1]]
    }
    fn j2(x: [f64; 2]) -> [[f64; 2]; 2] {
        [
            [1.0 + 0.4 * x[0] + 0.1 * x[1], 0.1 * x[0]],
            [0.05 * x[1], 1.0 + 0.05 * x[0]],
        ]
    }
    let el = fem_element::lagrange::H1TriPk::new(2);
    let rp = el.dof_coords();
    let coords: Vec<[f64; 3]> = rp.iter().map(|p| {
        let v = g2([p[0], p[1]]);
        [v[0], v[1], 0.0]
    }).collect();
    let path = write_temp("d319_curve_9.msh", &gmsh_v41_single_2d(9, &coords));
    let msh = read_msh_file(&path).expect("read tri6 msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh2d.expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Tri6);
    let nodes = mesh.element_nodes(0).to_vec();
    let coords_of = |n: u32| -> [f64; 3] {
        let c = mesh.geom_coords_of(n);
        [c[0], c[1], 0.0]
    };
    let (mut wx, mut wj) = (0.0_f64, 0.0_f64);
    for xi in [[0.2, 0.3, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]] {
        let (x, j) = sample_map(&el, &nodes, &coords_of, &xi[..2]);
        let want = g2([xi[0], xi[1]]);
        let wjq = j2([xi[0], xi[1]]);
        for i in 0..2 {
            wx = wx.max((x[i] - want[i]).abs());
            for d in 0..2 {
                wj = wj.max((j[i][d] - wjq[i][d]).abs());
            }
        }
    }
    eprintln!("D319 Tri6 (type 9, identity): max |x-g| = {wx:.3e}, max |J-dg| = {wj:.3e}");
    assert!(wx < 1e-12 && wj < 1e-12, "Tri6 file order changed: {wx:.3e} / {wj:.3e}");
}

/// `Quad9` (type 10): Gmsh's order (corners, then the edge midpoints on
/// `(0,1) (1,2) (2,3) (3,0)`, then the center) already is `QuadQk`'s order, so
/// the file order must be kept — verified with the same 2-D quadratic map.
#[test]
fn d319_quad9_keeps_the_file_order() {
    fn g2(x: [f64; 2]) -> [f64; 2] {
        [x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1], x[1] + 0.05 * x[0] * x[1]]
    }
    fn j2(x: [f64; 2]) -> [[f64; 2]; 2] {
        [
            [1.0 + 0.4 * x[0] + 0.1 * x[1], 0.1 * x[0]],
            [0.05 * x[1], 1.0 + 0.05 * x[0]],
        ]
    }
    let el = fem_element::lagrange::factory::QuadQk::new(2);
    let rp = el.dof_coords();
    let coords: Vec<[f64; 3]> = rp
        .iter()
        .map(|p| {
            let v = g2([p[0], p[1]]);
            [v[0], v[1], 0.0]
        })
        .collect();
    let path = write_temp("d319_curve_10.msh", &gmsh_v41_single_2d(10, &coords));
    let msh = read_msh_file(&path).expect("read quad9 msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh2d.expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Quad9);
    let nodes = mesh.element_nodes(0).to_vec();
    let coords_of = |n: u32| -> [f64; 3] {
        let c = mesh.geom_coords_of(n);
        [c[0], c[1], 0.0]
    };
    let (mut wx, mut wj) = (0.0_f64, 0.0_f64);
    for xi in [[0.25, 0.75, 0.0], [0.5, 0.5, 0.0], [0.1, 0.9, 0.0]] {
        let (x, j) = sample_map(&el, &nodes, &coords_of, &xi[..2]);
        let want = g2([xi[0], xi[1]]);
        let wjq = j2([xi[0], xi[1]]);
        for i in 0..2 {
            wx = wx.max((x[i] - want[i]).abs());
            for d in 0..2 {
                wj = wj.max((j[i][d] - wjq[i][d]).abs());
            }
        }
    }
    eprintln!("D319 Quad9 (type 10, identity): max |x-g| = {wx:.3e}, max |J-dg| = {wj:.3e}");
    assert!(wx < 1e-12 && wj < 1e-12, "Quad9 file order changed: {wx:.3e} / {wj:.3e}");
}

/// The reader must not touch order-1 meshes or the incomplete serendipity
/// families (whose canonical order *is* the Gmsh order, D244).
#[test]
fn d319_low_order_and_serendipity_rows_are_untouched() {
    // A linear triangle mesh: connectivity as written.
    let coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let path = write_temp("d319_linear_tri.msh", &gmsh_v41_single_2d(2, &coords));
    let msh = read_msh_file(&path).expect("read tri3");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh2d.expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Tri3);
    let got: Vec<[f64; 3]> = mesh
        .element_nodes(0)
        .iter()
        .map(|&n| {
            let c = mesh.geom_coords_of(n);
            [c[0], c[1], 0.0]
        })
        .collect();
    assert_eq!(got, coords.to_vec());

    // Quad8 (type 16) is an incomplete family: its connectivity order must be
    // the Gmsh one (x-encoded node ids make any permutation visible).
    let quad8: Vec<[f64; 3]> = (0..8).map(|i| [i as f64, 0.0, 0.0]).collect();
    let path = write_temp("d319_quad8.msh", &gmsh_v41_single_2d(16, &quad8));
    let msh = read_msh_file(&path).expect("read quad8");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh2d.expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Quad8);
    let xs: Vec<f64> = mesh
        .element_nodes(0)
        .iter()
        .map(|&n| mesh.geom_coords_of(n)[0])
        .collect();
    assert_eq!(xs, (0..8).map(|i| i as f64).collect::<Vec<_>>());

    // Also exercise the `read_msh` (reader-based) entry point.
    let text = gmsh_v41_single_2d(16, &quad8);
    let msh = read_msh(text.as_bytes()).expect("read quad8 via read_msh");
    assert!(msh.mesh2d.is_some());
}
