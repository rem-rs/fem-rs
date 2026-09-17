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

use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem_nodes, NodesSpace};
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

// ─── D178: 2-D triangle `nodes`, whole-file fixtures ────────────────────────
//
// The `L2_T1_2D_P<p>` triangle numbering above is also pinned against
// MFEM-written *whole files* below (not just the `nodes` payload), together
// with the continuous `H1_2D_P<p>` triangle numbering.
//
// Reference generator: `tmp/r36/tri2d_nodes.cpp` (compiled against MFEM 4.10
// in WSL).  It builds the *straight-sided* triangulated grid
// `Mesh::MakeCartesian2D(nx, ny, Element::TRIANGLE, false, 1.0, 1.0, false)` —
// the exact grid `Mesh::make_cartesian_2d_tri` reproduces — promotes it with
// `SetCurvature(p, discont, 2, Ordering::byVDIM)` and prints the whole mesh
// with 17 digits.  A straight-sided grid is the strongest fixture: every node
// value is fixed by the numbering alone.  `mode "tables"` of the same probe
// printed `H1_TriangleElement(p)`'s and `L2_TriangleElement(p,
// BasisType::GaussLobatto)`'s `GetNodes()` for p = 2..4: both agree with
// `H1TriPk::dof_coords()` point-for-point (the same closed Gauss-Lobatto,
// `w`-normalised barycentric lattice), so both writers are *pure
// renumberings* of the mesh's own slots — no interpolation matrix.  (Note the
// default `L2_TriangleElement(p)` btype is `GaussLegendre` — open points; the
// `L2_T1` collection passes `GaussLobatto` explicitly.)
//
// Regenerate the fixtures with:
//   `./tri2d_nodes fixture <p> <discont> <nx> <ny> <out>.mesh`
// (p = 2..4, discont = 0/1, (nx, ny) = (2,1)/(3,2))

/// The non-comment lines of a `.mesh` file, each split into whitespace tokens.
fn tokens(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect()
}

/// Compare two `.mesh` files line by line: every token must be equal, or two
/// numbers agreeing to `tol` (MFEM writes its fixtures with 17 digits, but the
/// two sides evaluate the shared straight-sided geometry through different
/// affine combinations, so bit equality is only demanded for the round trip).
fn compare_mesh(ours: &str, theirs: &str, tol: f64, label: &str) {
    let (a, b) = (tokens(ours), tokens(theirs));
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: different number of non-comment lines ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(ra.len(), rb.len(), "{label}: line {} has {} vs {} tokens", i + 1, ra.len(), rb.len());
        for (ta, tb) in ra.iter().zip(rb.iter()) {
            if ta == tb {
                continue;
            }
            let (x, y) = (
                ta.parse::<f64>().unwrap_or_else(|_| panic!("{label}: line {} token '{ta}'", i + 1)),
                tb.parse::<f64>().unwrap_or_else(|_| panic!("{label}: line {} token '{tb}'", i + 1)),
            );
            assert!(
                (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
                "{label}: line {} ({}): {ta} vs {tb}",
                i + 1,
                a[i].join(" ")
            );
        }
    }
}

/// The inline `MakeCartesian2D(nx, ny, TRIANGLE)` grid of the C++ probe.
fn straight_tri_grid(nx: usize, ny: usize) -> Mesh<2> {
    Mesh::<2>::make_cartesian_2d_tri(nx, ny, 1.0, 1.0)
}

/// The written H1/L2 triangle file must match MFEM 4.10's own output for the
/// same grid, as a *whole file* — elements, boundary, vertices header and the
/// complete `nodes` payload at every order.
#[test]
fn tri3_nodes_whole_file_matches_mfem_reference() {
    for order in [2u8, 3, 4] {
        for (nx, ny) in [(2usize, 1usize), (3usize, 2usize)] {
            for discont in [false, true] {
                let label = format!(
                    "{}/{}/p={order} d={}",
                    nx,
                    ny,
                    usize::from(discont)
                );
                let mut mesh = straight_tri_grid(nx, ny);
                mesh.set_curvature(order);
                let space = if discont {
                    NodesSpace::Discontinuous
                } else {
                    NodesSpace::Continuous
                };
                let got = write_space(&mesh, space);
                let family = if discont { "L2_T1" } else { "H1" };
                assert!(
                    got.contains(&format!("FiniteElementCollection: {family}_2D_P{order}")),
                    "{label}:\n{got}"
                );
                let want_path = format!(
                    "{}/tests/data/tri2d_{}_p{}_g{nx}{ny}.mesh",
                    env!("CARGO_MANIFEST_DIR"),
                    usize::from(discont),
                    order
                );
                let want =
                    std::fs::read_to_string(&want_path).unwrap_or_else(|e| panic!("{want_path}: {e}"));
                assert!(
                    want.contains(&format!("FiniteElementCollection: {family}_2D_P{order}")),
                    "{want_path}"
                );
                compare_mesh(&got, &want, 1e-14, &label);
            }
        }
    }
}

/// MFEM's own `H1_2D_P<p>` triangle file must round-trip through the reader
/// and the writer with the numbering composition the identity permutation:
/// the reader attaches the file's dof values through the entity-wise
/// numbering (2-D triangles take the `DofManager` entity-blocked path) and
/// the writer re-numbers them through `tri2d_slot_map`, so apart from the
/// value *rendering* the text matches MFEM's re-save of the same file.
///
/// The comparison tolerance is one rounding quantum at the writer's stream
/// precision (`Mesh::Save`'s default `precision = 16`, D274), not 0: the
/// fixtures carry 17 digits (`0.16666666666666666`) and MFEM's own
/// `Mesh::Save(out, 16)` re-save of them truncates to 16
/// (`0.1666666666666667` — `$HOME/work/r31_save` on
/// `tri2d_0_p2_g32.mesh`, 40 changed lines, `tmp/d294/`), exactly like the
/// Rust writer now does.  A mis-numbered dof would still move the value by
/// O(1), four orders of magnitude above the tolerance.
#[test]
fn h1_tri3_fixture_round_trips_through_the_reader() {
    for order in [2u8, 3, 4] {
        for (nx, ny) in [(2usize, 1usize), (3usize, 2usize)] {
            let label = format!("{}/{}/p={order}", nx, ny);
            let want_path = format!(
                "{}/tests/data/tri2d_0_p{}_g{nx}{ny}.mesh",
                env!("CARGO_MANIFEST_DIR"),
                order
            );
            let fixture =
                std::fs::read_to_string(&want_path).unwrap_or_else(|e| panic!("{want_path}: {e}"));
            let mesh = read_mfem(fixture.as_bytes())
                .expect("read_mfem must accept MFEM's own H1_2D_P tri file")
                .mesh2d
                .expect("2-D mesh");
            assert_eq!(mesh.geom_order(), order, "{label}: wrong geometric order");
            let back = write_space(&mesh, NodesSpace::Continuous);
            compare_mesh(&back, &fixture, 5e-16, &label);
        }
    }
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

/// End to end: MFEM's own curved triangle mesh (`data/square-disc-p2.mesh`,
/// 154 curved `H1_2D_P2` triangles, refined once — the round-35 D173
/// fixtures), read, uniformly refined in fem-rs and **written with curvature**
/// — the operation D178 unlocked.  The topology sections must match MFEM's
/// own refine+save token-for-token, and the `nodes` payload must agree to the
/// reference's 8-significant-digit print quantum.  (The reference keeps its
/// original `Ordering: 0` — byNODES, one value per line — while our writer
/// always emits `Ordering: 1`/byVDIM, exactly what `GridFunction::Save`
/// produces for such a space; the two encodings carry the same dof vector, so
/// the comparison normalises both to dof-major.)
#[test]
fn curved_refined_tri_mesh_is_written_with_its_curvature() {
    let parent_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../mesh/tests/data/square_disc_p2.mesh");
    let refined_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../mesh/tests/data/square_disc_p2_r1.mesh");
    let parent = read_mfem_file(parent_path)
        .expect("read parent fixture")
        .mesh2d
        .expect("2-D parent");
    assert_eq!(parent.n_elems(), 154);
    assert_eq!(parent.geom_order(), 2);

    let fine = fem_mesh::refine_uniform(&parent);
    assert_eq!(fine.n_elems(), 616, "154 tris x 4 children");
    let got = write_space(&fine, NodesSpace::Continuous);
    assert!(
        got.contains("FiniteElementCollection: H1_2D_P2"),
        "no nodes section in\n{got}"
    );

    let want = std::fs::read_to_string(refined_path).expect("MFEM refined reference");

    // Topology (everything before the `nodes` keyword): token-for-token.
    let cut = |text: &str| {
        let at = text.lines().position(|l| l.trim() == "nodes").expect("`nodes` section");
        let head: Vec<Vec<String>> = tokens(&text.lines().take(at).collect::<Vec<_>>().join("\n"));
        let rest: Vec<&str> = text.lines().skip(at + 1).collect();
        (head, rest.join("\n"))
    };
    let (got_head, got_nodes) = cut(&got);
    let (want_head, want_nodes) = cut(&want);
    assert_eq!(
        got_head.iter().map(|r| r.join(" ")).collect::<Vec<_>>(),
        want_head.iter().map(|r| r.join(" ")).collect::<Vec<_>>(),
        "topology sections disagree"
    );

    // `nodes`: normalise both storage orders to dof-major and compare to the
    // 8-digit print quantum of the reference.
    let parse = |text: &str| -> (usize, usize, Vec<f64>) {
        let ls: Vec<&str> = text.lines().collect();
        assert_eq!(ls[0].trim(), "FiniteElementSpace", "{text}");
        let vdim: usize = ls[2].trim().strip_prefix("VDim:").unwrap().trim().parse().unwrap();
        let ordering: usize =
            ls[3].trim().strip_prefix("Ordering:").unwrap().trim().parse().unwrap();
        let values: Vec<f64> = ls[4..]
            .iter()
            .flat_map(|l| l.split_whitespace().filter_map(|v| v.parse::<f64>().ok()))
            .collect();
        (vdim, ordering, values)
    };
    let (vdim_g, ordering_g, values_g) = parse(&got_nodes);
    let (vdim_w, ordering_w, values_w) = parse(&want_nodes);
    assert_eq!((vdim_g, ordering_g), (2, 1), "we write byVDIM");
    assert_eq!((vdim_w, ordering_w), (2, 0), "the reference keeps MFEM's byNODES");
    assert_eq!(values_g.len(), values_w.len(), "node value count");
    let n = values_g.len() / 2;
    for d in 0..n {
        // ours: [x_d, y_d] at 2d, 2d+1.  theirs (byNODES): [x_0..x_n, y_0..y_n].
        for c in 0..2 {
            let x = values_g[2 * d + c];
            let y = values_w[c * n + d];
            let scale = x.abs().max(y.abs()).max(1e-12);
            assert!(
                (x - y).abs() < 1e-7 * scale,
                "dof {d} component {c}: wrote {x}, MFEM's refine+save has {y}"
            );
        }
    }
}
