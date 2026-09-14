//! D151: the `.mesh` `nodes`-section writer's **2-D** element families.
//!
//! Round 32 landed `write_mfem_nodes` for the continuous (`H1`) spaces of
//! hexahedra and tetrahedra and the discontinuous (`L2`) spaces of hexahedra
//! and quadrilaterals.  The 2-D families the `mobius-strip` / `klein-bottle`
//! miniapps need (a `Quad4` surface, `H1_2D_P<p>` and `L2_T1_2D_P<p>`) were
//! refused with "no MFEM-faithful numbering … no `nodes` section was written",
//! which is why both miniapps exit 3 before writing anything.
//!
//! MFEM's numbering (`fem/fespace.cpp` `FiniteElementSpace::GetElementDofs`,
//! with the entity enumeration of `Mesh::FinalizeTopology`):
//!
//! ```text
//! H1_2D_P<p>:  [ vertices | edges | element interiors ]
//!   vertex v            -> dof v
//!   mesh edge E, slot t -> dof NV + E*(p-1) + t
//!   element e, slot o   -> dof NV + NE*(p-1) + e*(p-1)^2 + o
//! L2_T1_2D_P<p>: (p+1)^2 private dofs per element, lexicographic
//!   ix + iy*(p+1)  (`L2_QuadrilateralElement`)
//! ```
//!
//! The tests are pure Rust: the L2 numbering is a permutation of `QuadQk`'s /
//! `H1TriPk`'s own slot order (that lattice identity is already pinned against
//! MFEM's own output by the round-32 `nodes_writer.rs` fixtures), so a
//! read → write round trip through the writer and the reader must be the
//! identity.

use fem_io::mfem::{read_mfem, write_mfem_nodes, NodesSpace};
use fem_mesh::Mesh;

fn write_space(mesh: &Mesh<2>, space: NodesSpace) -> String {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut buf, mesh, None, space).expect("write_mfem_nodes");
    String::from_utf8(buf).expect("utf-8")
}

fn read_back(text: &str) -> Mesh<2> {
    read_mfem(text.as_bytes())
        .expect("read_mfem must accept the file write_mfem produced")
        .mesh2d
        .expect("2-D mesh")
}

fn curved_tri(order: u8) -> Mesh<2> {
    let mut mesh = Mesh::<2>::make_cartesian_2d_tri(2, 1, 1.0, 1.0);
    mesh.set_curvature(order);
    mesh.transform(|p| [p[0] + 0.09 * (1.3 * p[1]).sin(), p[1] + 0.05 * p[0] * p[0]]);
    mesh
}

/// `L2_T1_2D_P<p>` on triangles: MFEM's `L2_TriangleElement` enumerates its
/// `w`-normalised Gauss-Lobatto barycentric nodes as
/// `for (j) for (i <= p-j)` — the same point set as `H1TriPk`, in another
/// order — so a read → write round trip must be the identity.
#[test]
fn l2_tri3_roundtrip() {
    for order in 2..=4u8 {
        let mesh = curved_tri(order);
        let file1 = write_space(&mesh, NodesSpace::Discontinuous);
        assert!(
            file1.contains(&format!("FiniteElementCollection: L2_T1_2D_P{order}")),
            "order {order}:\n{file1}"
        );
        let back = read_back(&file1);
        assert_eq!(back.geom_order(), order, "order {order}: geom order");
        let file2 = write_space(&back, NodesSpace::Discontinuous);
        assert_eq!(file1, file2, "L2 tri P{order}: read → write is not the identity");
    }
}

// ─── 2-D quadrilateral: pinned against MFEM 4.10's own output ───────────────

/// The polynomial warp the C++ probe applies (`Transform`), reproduced exactly
/// — only `+` and `*`, so the two sides agree bit for bit.
fn warp(p: [f64; 2]) -> [f64; 2] {
    [p[0] + 0.13 * p[0] * p[1], p[1] + 0.07 * p[0] * p[0]]
}

/// A curved `beam-quad.mesh`, built the way the C++ probe builds it:
/// `Mesh(file, 1, 0)` → `SetCurvature(order, discont, 2, byVDIM)` →
/// `Transform(warp)`.
fn curved_beam_quad(order: u8) -> Mesh<2> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/beam-quad.mesh"
    ))
    .expect("data/beam-quad.mesh");
    let mut mesh = read_back(&text);
    mesh.set_curvature(order);
    mesh.transform(warp);
    mesh
}

/// The `L2_T1_2D_P<p>` numbering of **triangles** pinned against MFEM 4.10: the
/// same probe on `data/beam-tri.mesh` (16 curved triangles), so the artifact
/// covers MFEM's `L2_TriangleElement` node *lattice* and its DOF order, not just
/// the self-consistency of a round trip.
#[test]
fn l2_tri3_nodes_match_mfem_reference() {
    for order in [2u8, 3] {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/beam-tri.mesh"
        ))
        .expect("data/beam-tri.mesh");
        let mut mesh = read_back(&text);
        mesh.set_curvature(order);
        mesh.transform(warp);

        let got = write_space(&mesh, NodesSpace::Discontinuous);
        let want_path = format!(
            "{}/tests/data/l2_tri_p{order}_d1.mesh",
            env!("CARGO_MANIFEST_DIR")
        );
        let want = std::fs::read_to_string(&want_path).unwrap_or_else(|e| panic!("{want_path}: {e}"));
        assert!(want.contains(&format!("L2_T1_2D_P{order}")), "{want_path}");

        let a = nodes_values(&want);
        let b = nodes_values(&got);
        assert_eq!(a.len(), b.len(), "P{order}: node value count");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let scale = x.abs().max(y.abs()).max(1e-12);
            assert!(
                (x - y).abs() < 1e-7 * scale,
                "P{order}: value {i}: MFEM wrote {x}, we wrote {y}"
            );
        }
    }
}

/// The `nodes` section's coordinate rows, as flat values.
fn nodes_values(text: &str) -> Vec<f64> {
    let at = text.lines().position(|l| l == "nodes").expect("`nodes` section");
    text.lines()
        .skip(at + 6) // FiniteElementSpace / FEC / VDim / Ordering / blank
        .flat_map(|l| l.split_whitespace().filter_map(|v| v.parse::<f64>().ok()))
        .collect()
}

/// The C++ artifacts in `tests/data/` were produced by
///
/// ```text
///   Mesh m("data/beam-quad.mesh", 1, 0);
///   m.SetCurvature(order, discont, 2, Ordering::byVDIM);
///   m.Transform(warp);           // the polynomial warp above
///   ofs.precision(8); m.Print(ofs);
/// ```
///
/// so the *whole* `nodes` payload — the dof count, the ordering and every node
/// coordinate — is pinned against MFEM.  The reference is printed with
/// `precision(8)`, so the comparison is relative to `1e-7` (the same tolerance
/// the round-32 hex/tet fixtures use).
#[test]
fn quad2d_nodes_match_mfem_reference() {
    for (order, discont) in [(2u8, false), (3, false), (2, true), (3, true)] {
        let mesh = curved_beam_quad(order);
        let space = if discont {
            NodesSpace::Discontinuous
        } else {
            NodesSpace::Continuous
        };
        let got = write_space(&mesh, space);
        let family = if discont { "L2_T1" } else { "H1" };
        let want_path = format!(
            "{}/tests/data/h1_2d_quad_p{order}_d{}.mesh",
            env!("CARGO_MANIFEST_DIR"),
            usize::from(discont)
        );
        let want = std::fs::read_to_string(&want_path).unwrap_or_else(|e| panic!("{want_path}: {e}"));
        assert!(
            want.contains(&format!("FiniteElementCollection: {family}_2D_P{order}")),
            "{want_path}"
        );
        assert!(
            got.contains(&format!("FiniteElementCollection: {family}_2D_P{order}")),
            "{family} P{order}:\n{got}"
        );

        let a = nodes_values(&want);
        let b = nodes_values(&got);
        assert_eq!(
            a.len(),
            b.len(),
            "{family} P{order}: node value count (MFEM {}, ours {})",
            a.len() / 2,
            b.len() / 2
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let scale = x.abs().max(y.abs()).max(1e-12);
            assert!(
                (x - y).abs() < 1e-7 * scale,
                "{family} P{order}: value {i} (dof {} component {}): MFEM wrote {x}, we wrote {y}",
                i / 2,
                i % 2
            );
        }
    }
}
