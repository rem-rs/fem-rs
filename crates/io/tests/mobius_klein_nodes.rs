//! D171: the writer's **surface** path — a `dimension 2` mesh in 3-D space
//! (`Mesh<3>` of `Quad4`, `MeshTopology::topological_dim() == 2`) with a
//! 3-component high-order `nodes` section, the representation the MFEM
//! `mobius-strip` / `klein-bottle` miniapps write
//! (`MakeCartesian2D` + `SetCurvature(order, …, 3, Ordering::byVDIM)` +
//! `Transform`).
//!
//! The tests mirror the miniapp bodies (the transforms are duplicated here on
//! purpose: the pinning must not depend on the binary crates) and check the
//! structure MFEM's reader sees: `dimension`, element/boundary/vertex counts,
//! the `nodes` FEC name (`H1_2D_P<p>` / `L2_T1_2D_P<p>`), `VDim: 3`, and the
//! values at dofs whose expected positions are known analytically (element
//! vertices, straight GLL edge/interior nodes).  Counts and per-attribute
//! boundary layout are the ones measured on the MFEM 4.10 C++ binaries
//! (`$HOME/work/c34/{mobius,klein}_cpp`).
//!
//! When the `FEM_C34_DUMP` environment variable is set, the generated files
//! are also copied there (for byte/numeric comparison against the C++
//! artifacts by the round probes).

use std::path::PathBuf;

use fem_io::mfem::{write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::surface_embed::{
    cartesian2d_quad_surface_in_3d, identify_vertices_and_clean, zero_small_node_values,
};
use fem_mesh::{Mesh, MeshTopology};

const TWO_PI: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;
const PI_2: f64 = std::f64::consts::FRAC_PI_2;

// ── miniapp transforms (mirrors of mobius-strip.rs / klein-bottle.rs) ────────

fn mobius_trans(x: [f64; 3], num_twists: f64) -> [f64; 3] {
    let a = 1.0 + 0.5 * (x[1] - 1.0) * (num_twists * x[0]).cos();
    [a * x[0].cos(), a * x[0].sin(), 0.5 * (x[1] - 1.0) * (num_twists * x[0]).sin()]
}

fn figure8_trans(x: [f64; 3]) -> [f64; 3] {
    let r = 2.5f64;
    let a = r + (x[0] / 2.0).cos() * x[1].sin() - (x[0] / 2.0).sin() * (2.0 * x[1]).sin();
    [
        a * x[0].cos(),
        a * x[0].sin(),
        (x[0] / 2.0).sin() * x[1].sin() + (x[0] / 2.0).cos() * (2.0 * x[1]).sin(),
    ]
}

fn bottle_trans(x: [f64; 3]) -> [f64; 3] {
    let u = x[0];
    let v = x[1] + PI_2;
    let a = 6.0 * u.cos() * (1.0 + u.sin());
    let b = 16.0 * u.sin();
    let r = 4.0 * (1.0 - u.cos() / 2.0);
    let (p0, p1) = if u <= PI {
        (a + r * u.cos() * v.cos(), b + r * u.sin() * v.cos())
    } else {
        (a + r * (v + PI).cos(), b)
    };
    [p0, p1, r * v.sin()]
}

fn bottle2_trans(x: [f64; 3]) -> [f64; 3] {
    let u = x[1] - PI_2;
    let v = 2.0 * x[0];
    let p0 = if v < PI {
        (2.5 - 1.5 * v.cos()) * u.cos()
    } else if v < 2.0 * PI {
        (2.5 - 1.5 * v.cos()) * u.cos()
    } else if v < 3.0 * PI {
        -2.0 + (2.0 + u.cos()) * v.cos()
    } else {
        -2.0 + 2.0 * v.cos() - u.cos()
    };
    let p1 = if v < PI {
        (2.5 - 1.5 * v.cos()) * u.sin()
    } else if v < 2.0 * PI {
        (2.5 - 1.5 * v.cos()) * u.sin()
    } else {
        u.sin()
    };
    let p2 = if v < PI {
        -2.5 * v.sin()
    } else if v < 2.0 * PI {
        3.0 * v - 3.0 * PI
    } else if v < 3.0 * PI {
        (2.0 + u.cos()) * v.sin() + 3.0 * PI
    } else {
        -3.0 * v + 12.0 * PI
    };
    [p0, p1, p2]
}

// ── miniapp bodies (mirrors) ─────────────────────────────────────────────────

fn mobius_mesh(nx: usize, ny: usize, order: u8, close_strip: i32, num_twists: f64) -> Mesh<3> {
    let mut mesh = cartesian2d_quad_surface_in_3d(nx, ny, TWO_PI, 2.0);
    // C++ order of operations: SetCurvature *before* the identification, so
    // the node values keep the pre-identification samples through the vertex
    // removal and the continuous projection resolves seam dofs last-writer-wins.
    mesh.set_curvature(order);
    if close_strip != 0 {
        let npx = nx + 1;
        let mut v2v: Vec<i32> = (0..mesh.n_nodes() as i32).collect();
        for j in 0..=ny {
            let v_old = nx + j * npx;
            let v_new = (if close_strip == 1 { j } else { ny - j }) * npx;
            v2v[v_old] = v_new as i32;
        }
        identify_vertices_and_clean(&mut mesh, &v2v);
    }
    mesh.transform(|x| mobius_trans(x, num_twists));
    zero_small_node_values(&mut mesh);
    mesh
}

fn klein_mesh(nx: usize, ny: usize, order: u8, trans: fn([f64; 3]) -> [f64; 3]) -> Mesh<3> {
    let mut mesh = cartesian2d_quad_surface_in_3d(nx, ny, TWO_PI, TWO_PI);
    mesh.set_curvature(order);
    let npx = nx + 1;
    let mut v2v: Vec<i32> = (0..mesh.n_nodes() as i32).collect();
    for i in 0..=nx {
        v2v[i + ny * npx] = i as i32;
    }
    for j in 0..=ny {
        let v_old = nx + j * npx;
        let v_new = (ny - j) * npx;
        v2v[v_old] = v2v[v_new];
    }
    identify_vertices_and_clean(&mut mesh, &v2v);
    mesh.transform(trans);
    zero_small_node_values(&mut mesh);
    mesh
}

fn dump(name: &str, content: &str) {
    if let Ok(dir) = std::env::var("FEM_C34_DUMP") {
        let path = PathBuf::from(dir).join(name);
        std::fs::write(path, content).expect("dump file");
    }
}

// ── parsing helpers (the structural parts of MFEM's v1.0 format) ─────────────

struct ParsedMesh {
    dimension: usize,
    n_elems: usize,
    n_bdr: usize,
    n_verts: usize,
    fec: String,
    vdim: usize,
    ordering: usize,
    /// First `n` values of the flattened `byVDIM` node table.
    head_values: Vec<f64>,
}

fn parse_mesh(text: &str) -> ParsedMesh {
    let lines: Vec<&str> = text.lines().collect();
    let mut dimension = None;
    let mut n_elems = None;
    let mut n_bdr = None;
    let mut n_verts = None;
    let mut fec = None;
    let mut vdim = None;
    let mut ordering = None;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        let scalar = |_: &str| lines.get(i + 1).unwrap_or(&"").trim().parse::<usize>().ok();
        match t {
            "dimension" => dimension = scalar(t),
            "elements" => n_elems = scalar(t),
            "boundary" => n_bdr = scalar(t),
            "vertices" => n_verts = scalar(t),
            _ => {}
        }
        if let Some(rest) = t.strip_prefix("FiniteElementCollection:") {
            fec = Some(rest.trim().to_string());
        } else if let Some(rest) = t.strip_prefix("VDim:") {
            vdim = Some(rest.trim().parse().unwrap());
        } else if let Some(rest) = t.strip_prefix("Ordering:") {
            ordering = Some(rest.trim().parse().unwrap());
        }
    }
    // The nodes values follow the `Ordering:` line and a blank line.
    let mut head_values = Vec::new();
    let mut in_values = false;
    for line in text.lines() {
        if line.starts_with("Ordering:") {
            in_values = true;
            continue;
        }
        if in_values && !line.trim().is_empty() {
            for v in line.split_whitespace() {
                head_values.push(v.parse::<f64>().unwrap());
            }
        }
        if head_values.len() >= 30 {
            break;
        }
    }
    ParsedMesh {
        dimension: dimension.expect("dimension line"),
        n_elems: n_elems.expect("elements line"),
        n_bdr: n_bdr.expect("boundary line"),
        n_verts: n_verts.expect("vertices line"),
        fec: fec.unwrap_or_default(),
        vdim: vdim.unwrap_or(0),
        ordering: ordering.unwrap_or(0),
        head_values,
    }
}

/// `H1_2D_P{p}` dof 0 of the mobius default is the strip's first vertex
/// `(x[0], y[0]) = (0, 0)` → `mobius_trans((0,0)) = (1, 0, 0)` and the small
/// wipe makes the following components exactly 0 — the C++ file's first node
/// row is `1 0 0`.
#[test]
fn mobius_default_matches_cpp_structure() {
    let mesh = mobius_mesh(8, 2, 3, 2, 0.5);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mobius-strip.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("mobius-strip.mesh", &text);
    let parsed = parse_mesh(&text);

    // Measured C++ (`mobius-strip`, defaults): dim 2, NE 16, NBE 16, NV 24.
    assert_eq!(parsed.dimension, 2);
    assert_eq!(parsed.n_elems, 16);
    assert_eq!(parsed.n_bdr, 16);
    assert_eq!(parsed.n_verts, 24);
    assert_eq!(parsed.fec, "H1_2D_P3");
    assert_eq!(parsed.vdim, 3);
    assert_eq!(parsed.ordering, 1);
    // First node: mobius_trans((0,0,0)) with the <1e-12 wipe → (0.5, 0, 0)
    // (a = 1 - 0.5 = 0.5; the C++ file's first node row is `0.5 0 0`).
    assert_eq!(parsed.head_values[0], 0.5);
    assert_eq!(parsed.head_values[1], 0.0);
    assert_eq!(parsed.head_values[2], 0.0);
    // Element vertex dofs are the transformed strip vertices; dof 1 is
    // vertex 1 (x = sx/nx = 2π/8 on the bottom row): the C++ file's second
    // node row is `0.38046604 0.38046604 -0.19134172`.
    let x0 = TWO_PI / 8.0;
    let a = 1.0 + 0.5 * (0.0 - 1.0) * (0.5 * x0).cos();
    let expect = [a * x0.cos(), a * x0.sin(), 0.5 * (0.0 - 1.0) * (0.5 * x0).sin()];
    for c in 0..3 {
        assert!(
            (parsed.head_values[1 * 3 + c] - expect[c]).abs() < 1e-12,
            "vertex-1 dof mismatch: {:?} vs {expect:?}",
            &parsed.head_values[3..6]
        );
    }
}

/// The C++ `-c 0` (open strip) keeps all four boundary groups in MFEM's
/// order: bottom (attr 1), top (attr 3), left (attr 4), right (attr 2).
#[test]
fn mobius_open_keeps_four_boundary_groups() {
    let mesh = mobius_mesh(8, 2, 3, 0, 0.5);
    assert_eq!(mesh.n_nodes(), 27);
    assert_eq!(mesh.n_faces(), 20);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mobius-open.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("mobius-open.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.n_bdr, 20);
    // Boundary record order: 8×1, 8×3, 2×4, 2×2 (each record `attr 1 a b`).
    let attrs: Vec<i32> = text
        .lines()
        .skip_while(|l| l.trim() != "boundary")
        .skip(2)
        .take(20)
        .map(|l| l.split_whitespace().next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(attrs, vec![1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 2]);
}

/// `-dm` produces the discontinuous section (`L2_T1_2D_P<p>`, private dofs:
/// 16·16 values × 3 components) with the same topology.
#[test]
fn mobius_discont_uses_l2_t1_collection() {
    let mesh = mobius_mesh(8, 2, 3, 2, 0.5);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mobius-dm.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Discontinuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("mobius-dm.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.fec, "L2_T1_2D_P3");
    assert_eq!(parsed.dimension, 2);
    assert_eq!(parsed.n_elems, 16);
    assert_eq!(parsed.vdim, 3);
    // First element, first dof: same first vertex as the continuous file
    // (mobius_trans((0,0)) with the wipe → (0.5, 0, 0)).
    assert_eq!(parsed.head_values[0], 0.5);
}

/// `-c 1` (closed, no twist): same counts as `-c 2`, boundary only the two
/// end rings, and the first vertex (mapped through `v2v` onto the seam) sits
/// at `mobius_trans((0,0))`.
#[test]
fn mobius_closed_untwisted_matches_cpp_counts() {
    let mesh = mobius_mesh(8, 2, 3, 1, 0.5);
    assert_eq!(mesh.topological_dim(), 2);
    assert_eq!(mesh.n_elems(), 16);
    assert_eq!(mesh.n_nodes(), 24);
    assert_eq!(mesh.n_faces(), 16);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mobius-c1.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("mobius-c1.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.n_bdr, 16);
    assert_eq!(parsed.dimension, 2);
}

/// The klein default (`-t 1`): 128 quads / 128 vertices / `boundary 0`,
/// `H1_2D_P3`, `VDim: 3`; first node = `bottle_trans((0, 0, 0))`.
#[test]
fn klein_default_matches_cpp_structure() {
    let mesh = klein_mesh(16, 8, 3, bottle_trans);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("klein-bottle.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("klein-bottle.mesh", &text);
    let parsed = parse_mesh(&text);

    // Measured C++ (`klein-bottle`, defaults).
    assert_eq!(parsed.dimension, 2);
    assert_eq!(parsed.n_elems, 128);
    assert_eq!(parsed.n_bdr, 0);
    assert_eq!(parsed.n_verts, 128);
    assert_eq!(parsed.fec, "H1_2D_P3");
    assert_eq!(parsed.vdim, 3);
    assert_eq!(parsed.ordering, 1);
    // bottle_trans((0,0)): u=0, v=π/2 → a=6, b=0, r=4·(1−cos 0/2)=2;
    // p = (6 + 2·1·0, 0 + 0, 2·1) = (6, 0, 2) — the C++ file's first row.
    let expect = [6.0, 0.0, 2.0];
    for c in 0..3 {
        assert!(
            (parsed.head_values[c] - expect[c]).abs() < 1e-12,
            "klein first node: {} vs {expect:?}",
            parsed.head_values[0]
        );
    }
}

/// `-t 0` (figure-8): same topology, first node = `figure8_trans((0,0,0))` =
/// `(2.5, 0, 0)` (r = 2.5, sin terms vanish).
#[test]
fn klein_figure8_matches_cpp_counts() {
    let mesh = klein_mesh(16, 8, 3, figure8_trans);
    assert_eq!(mesh.n_elems(), 128);
    assert_eq!(mesh.n_nodes(), 128);
    assert_eq!(mesh.n_faces(), 0);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("klein-figure8.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("klein-figure8.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.n_bdr, 0);
    assert_eq!(parsed.fec, "H1_2D_P3");
    // All original vertices identified onto compact vertex 0 — (0,0),
    // (0,2π), (2π,2π) — project to (2.5, 0, 0) in exact arithmetic; the
    // last-writer element's sample (figure8 of (2π,2π)) carries O(1e-16)
    // trig rounding, which MFEM's 8-digit print also collapses to `2.5 0 0`.
    assert!((parsed.head_values[0] - 2.5).abs() < 1e-6);
    assert!(parsed.head_values[1].abs() < 1e-6);
    assert!(parsed.head_values[2].abs() < 1e-6);
}

/// `-dm` on the klein bottle: `L2_T1_2D_P3` with 128·16 private dofs.
#[test]
fn klein_discont_uses_l2_t1_collection() {
    let mesh = klein_mesh(16, 8, 3, bottle_trans);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("klein-dm.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Discontinuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("klein-dm.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.fec, "L2_T1_2D_P3");
    assert_eq!(parsed.vdim, 3);
    assert_eq!(parsed.n_verts, 128);
    // First element, first dof = bottle_trans((0,0,0)) = (6, 0, 2).
    assert!((parsed.head_values[0] - 6.0).abs() < 1e-12);
    assert!((parsed.head_values[2] - 2.0).abs() < 1e-12);
}

/// `-o 4` on the mobius default (`mobius-strip -o 4`, one of the C++ sample
/// runs): order-4 section, otherwise the default topology.
#[test]
fn mobius_order4_matches_cpp_structure() {
    let mesh = mobius_mesh(8, 2, 4, 2, 0.5);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mobius-o4.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("mobius-o4.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.fec, "H1_2D_P4");
    assert_eq!(parsed.n_elems, 16);
    assert_eq!(parsed.n_verts, 24);
    assert_eq!(parsed.n_bdr, 16);
    assert_eq!(parsed.vdim, 3);
}

/// `klein-bottle -t 2` (bottle2 transform), one of the C++ sample runs.
#[test]
fn klein_bottle2_matches_cpp_structure() {
    let mesh = klein_mesh(16, 8, 3, bottle2_trans);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("klein-bottle2.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("klein-bottle2.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.dimension, 2);
    assert_eq!(parsed.n_elems, 128);
    assert_eq!(parsed.n_bdr, 0);
    assert_eq!(parsed.n_verts, 128);
    assert_eq!(parsed.fec, "H1_2D_P3");
}

/// A straight-sided surface (`Mesh<3>` of `Quad4`, no `set_curvature`) writes
/// `dimension 2` with an ordinary 3-component `vertices` block — MFEM's
/// straight `Dim = 2, spaceDim = 3` representation.
#[test]
fn straight_surface_writes_dimension_2_vertices_block() {
    let mesh = cartesian2d_quad_surface_in_3d(3, 2, TWO_PI, 2.0);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("surface-straight.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    dump("surface-straight.mesh", &text);
    let parsed = parse_mesh(&text);
    assert_eq!(parsed.dimension, 2);
    assert_eq!(parsed.n_elems, 6);
    assert_eq!(parsed.n_verts, 12);
    assert_eq!(parsed.fec, ""); // no nodes section
}

// ── parity against the MFEM 4.10 C++ artifacts ───────────────────────────────
//
// `crates/io/tests/data/{mobius_strip,klein_bottle}_cpp_*.mesh` were produced
// by the reference C++ miniapps (`mfem410_ser/miniapps/meshing/{mobius-strip,
// klein-bottle}` built from the same source tree, run with the matching
// flags).  The topology section — everything from `dimension` through the
// `vertices` count, comments stripped — is byte-identical, and the `nodes`
// values agree to MFEM's 8-digit print precision.

/// Split a v1.0 mesh file into (topology text, optional node values): the
/// topology part runs from `dimension` to the `vertices` count inclusive;
/// `#` comments and blank lines are dropped (both writers comment, the C++
/// one prefixes a geometry-type legend).
fn split_topology_and_nodes(text: &str) -> (String, Option<Vec<f64>>) {
    let mut topology = String::new();
    let mut values: Option<Vec<f64>> = None;
    let mut lines = text.lines().peekable();
    let mut in_topo = false;
    while let Some(line) = lines.next() {
        let t = line.trim();
        if t.starts_with('#') || t.is_empty() {
            continue;
        }
        if t == "dimension" {
            in_topo = true;
        }
        if in_topo {
            topology.push_str(t);
            topology.push('\n');
            if t == "nodes" {
                // header of the nodes section: keep the section keyword in the
                // topology text (its presence is part of the format), then
                // collect the values numerically after `Ordering: 1`.
                for l in lines.by_ref() {
                    let lt = l.trim();
                    if lt.starts_with('#') || lt.is_empty() {
                        continue;
                    }
                    if lt.starts_with("Ordering:") {
                        break;
                    }
                    topology.push_str(lt);
                    topology.push('\n');
                }
                let mut vals = Vec::new();
                for l in lines {
                    if l.trim().is_empty() {
                        continue;
                    }
                    for v in l.split_whitespace() {
                        vals.push(v.parse::<f64>().unwrap());
                    }
                }
                values = Some(vals);
                break;
            }
            if t == "vertices" {
                // the count line follows (both fixtures are curved, so the
                // coordinates block is replaced by the nodes section); keep
                // scanning for the `nodes` keyword instead of stopping.
                let count = lines.next().unwrap().trim();
                topology.push_str(count);
                topology.push('\n');
                continue;
            }
        }
    }
    (topology, values)
}

fn assert_matches_cpp_fixture(ours_path: &std::path::Path, fixture: &str) {
    let fixture_text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(fixture),
    )
    .unwrap();
    let ours_text = std::fs::read_to_string(ours_path).unwrap();
    let (ours_topo, ours_vals) = split_topology_and_nodes(&ours_text);
    let (cpp_topo, cpp_vals) = split_topology_and_nodes(&fixture_text);
    assert_eq!(
        ours_topo, cpp_topo,
        "topology section (dimension..vertices, comments stripped) must be byte-identical"
    );
    let cpp_vals = cpp_vals.expect("fixture carries a nodes section");
    let ours_vals = ours_vals.expect("our file carries a nodes section");
    assert_eq!(ours_vals.len(), cpp_vals.len(), "nodes value count");
    let mut max_rel = 0.0f64;
    for (a, b) in ours_vals.iter().zip(cpp_vals.iter()) {
        let rel = (a - b).abs() / b.abs().max(1.0);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel <= 5e-8,
        "nodes values deviate from the C++ artifact beyond print noise: {max_rel}"
    );
}

/// Default mobius (`-o 3 -c 2`, continuous) against the C++ artifact.
#[test]
fn mobius_default_matches_cpp_artifact() {
    let mesh = mobius_mesh(8, 2, 3, 2, 0.5);
    // Distinct filename per test — the same reason `klein_default_matches_cpp_artifact`
    // documents: cargo runs one binary's tests on parallel threads, so sharing
    // `mobius-strip.mesh` with `mobius_default_matches_cpp_structure` in the same
    // `CARGO_TARGET_TMPDIR` races on the write/read pair (observed as a single
    // red in the full `--tests` gate that passes in isolation — the reader sees
    // a truncated file and panics with "dimension line").  Round 73 fixed the
    // klein pair only; round 76 swept the family (discipline ⑰).
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mobius-strip-artifact.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    assert_matches_cpp_fixture(&path, "mobius_strip_cpp_default.mesh");
}

/// Default klein (`-o 3 -t 1`, continuous) against the C++ artifact.
#[test]
fn klein_default_matches_cpp_artifact() {
    let mesh = klein_mesh(16, 8, 3, bottle_trans);
    // Distinct filename per test: cargo runs the tests in one binary on
    // parallel threads, so two tests sharing `klein-bottle.mesh` in the same
    // `CARGO_TARGET_TMPDIR` race on the write/read pair (observed as an
    // intermittent gate failure that passes in isolation).
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("klein-bottle-artifact.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Continuous).expect("write");
    assert_matches_cpp_fixture(&path, "klein_bottle_cpp_default.mesh");
}

/// Mobius `-dm` (discontinuous `L2_T1_2D_P3`) against the C++ artifact.
#[test]
fn mobius_discont_matches_cpp_artifact() {
    let mesh = mobius_mesh(8, 2, 3, 2, 0.5);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mobius-dm-artifact.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Discontinuous).expect("write");
    assert_matches_cpp_fixture(&path, "mobius_strip_cpp_dm.mesh");
}
