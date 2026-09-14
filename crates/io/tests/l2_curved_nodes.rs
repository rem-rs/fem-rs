//! D153: `read_mfem`'s discontinuous (L2) `nodes` path must permute the file's
//! values into the mesh's own reference-element slot order.
//!
//! An `L2_T1_<dim>D_P<p>` `nodes` section (MFEM `Mesh::SetCurvature(order,
//! discont = true)`) stores, per element, the `(p+1)^dim` node values in MFEM's
//! **L2 element ordering**.  For the tensor-product families that order is
//! lexicographic (`ix + iy·(p+1) + iz·(p+1)²`, `ix` fastest —
//! `L2_QuadrilateralElement` / `L2_HexahedronElement` fill `Nodes.IntPoint(o++)`
//! over `for (j) for (i)`, and `TensorBasisElement`'s `L2_DOF_MAP` leaves
//! `dof_map` empty, so no re-ordering is applied: `fem/fe/fe_l2.cpp`,
//! `fem/fe/fe_base.cpp`).  fem-rs's geometry elements (`QuadQk` / `HexQk`) use
//! the *topological* H1 order instead, so the values cannot be handed over
//! unchanged.
//!
//! The reader used a hard-coded `[0, 1, 3, 2]` for the P1 quadrilateral and the
//! identity everywhere else, while the writer maps the mesh's slots through
//! `lex_slot_permutation(factory_slots, mfem_l2_slots)`.  A read → write round
//! trip therefore only closes when the reader applies the *inverse* of that
//! permutation: the bug is invisible for P1 quads (where the two happen to
//! coincide) and corrupts every other L2 family/order — most visibly the
//! hexahedra, whose L2 lex order differs from the H1 order already at P1.
//!
//! The tests below are round trips (write → read → write must be bit-identical),
//! which is the strongest check available without a second C++ artifact: the
//! writer is already pinned against MFEM's own `L2_T1_3D_P3` output by
//! `nodes_writer.rs` (`tests/data/twist_hex_o3_s2_p.mesh`), so an exact round
//! trip of the same mesh pins the reader to the same numbering.

use fem_io::mfem::{read_mfem, write_mfem_nodes, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

fn write_discont(mesh: &Mesh<3>) -> String {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(
        &mut buf,
        &Mesh::<2>::unit_square_tri(2),
        Some(mesh),
        NodesSpace::Discontinuous,
    )
    .expect("write_mfem_nodes(Discontinuous)");
    String::from_utf8(buf).expect("utf-8")
}

fn read_back(text: &str) -> Mesh<3> {
    read_mfem(text.as_bytes())
        .expect("read_mfem must accept the file write_mfem produced")
        .mesh3d
        .expect("3-D mesh")
}

fn write_discont_2d(mesh: &Mesh<2>) -> String {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut buf, mesh, None, NodesSpace::Discontinuous)
        .expect("write_mfem_nodes(Discontinuous)");
    String::from_utf8(buf).expect("utf-8")
}

fn read_back_2d(text: &str) -> Mesh<2> {
    read_mfem(text.as_bytes())
        .expect("read_mfem must accept the file write_mfem produced")
        .mesh2d
        .expect("2-D mesh")
}

/// The coordinate rows of a `nodes` section, as flat `f64`s in file order.
fn nodes_rows(text: &str) -> Vec<f64> {
    let at = text.lines().position(|l| l == "nodes").expect("nodes section");
    let body = text.lines().skip(at + 6); // header + blank line
    body.flat_map(|l| l.split_whitespace().filter_map(|v| v.parse::<f64>().ok()))
        .collect()
}

/// A genuinely curved hex mesh: `set_curvature` + a non-affine map, so no
/// accidental symmetry can hide a wrong slot permutation.
fn curved_hex(order: u8) -> Mesh<3> {
    let mut mesh =
        Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    mesh.set_curvature(order);
    mesh.transform(|p| {
        [
            p[0] + 0.10 * (1.7 * p[1] + 0.3).sin(),
            p[1] + 0.05 * (1.1 * p[2] * p[2]),
            p[2] + 0.07 * p[0] * p[1],
        ]
    });
    mesh
}

fn curved_quad(order: u8) -> Mesh<2> {
    let mut mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
    mesh.set_curvature(order);
    mesh.transform(|p| [p[0] + 0.11 * (0.9 + 1.3 * p[1]).sin(), p[1] + 0.04 * p[0] * p[0]]);
    mesh
}

/// `order >= 2` only: `set_curvature(1)` clears the geometry, so a P1 mesh has
/// no `nodes` section at all (both in MFEM and here).
#[test]
fn l2_curved_hex_roundtrip() {
    for order in 2..=3u8 {
        let mesh = curved_hex(order);
        let file1 = write_discont(&mesh);
        assert!(
            file1.contains(&format!("FiniteElementCollection: L2_T1_3D_P{order}")),
            "order {order}: unexpected section header:\n{file1}"
        );
        let back = read_back(&file1);
        assert_eq!(back.geom_order(), order, "order {order}: geom order");
        let file2 = write_discont(&back);
        assert_eq!(file1, file2, "L2 hex P{order}: read → write is not the identity");
    }
}

#[test]
fn l2_curved_quad_roundtrip() {
    for order in 2..=3u8 {
        let mesh = curved_quad(order);
        let file1 = write_discont_2d(&mesh);
        assert!(
            file1.contains(&format!("FiniteElementCollection: L2_T1_2D_P{order}")),
            "order {order}: unexpected section header:\n{file1}"
        );
        let back = read_back_2d(&file1);
        assert_eq!(back.geom_order(), order, "order {order}: geom order");
        let file2 = write_discont_2d(&back);
        assert_eq!(file1, file2, "L2 quad P{order}: read → write is not the identity");
    }
}

/// The P1 quadrilateral is the historical special case and must keep working:
/// it is the only L2 family actually present in `data/` (`periodic-square.mesh`,
/// `periodic-hexagon.mesh`, the meshes ex9/ex18/ex41 read).
///
/// `data/periodic-square.mesh` is MFEM's own output, so the reader's slot
/// assignment can be checked against it directly: for element `e` the geometry
/// slot `s` must hold the file's row `e*4 + [0,1,3,2][s]` (the L2 lexicographic
/// index of the quad's `LL, LR, UR, UL` vertex).  As an independent check the
/// recovered corner order must describe a positively oriented (non-folded)
/// quadrilateral: `element_jacobian_at` — which is exactly the code that reads
/// this table — must give a positive determinant.  A scrambled slot order folds
/// the element and flips it.
#[test]
fn l2_p1_quad_slots_match_mfems_own_file() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/periodic-square.mesh");
    let text = std::fs::read_to_string(path).expect("data/periodic-square.mesh");
    assert!(text.contains("L2_T1_2D_P1"), "fixture must be an L2 P1 file");

    let mesh = read_back_2d(&text);
    assert_eq!(mesh.geom_order(), 1);
    let geo = mesh.geometry.as_ref().expect("L2 P1 geometry table");
    assert_eq!(geo.nodes_per_elem, 4);

    // The file's per-element rows, in file order.
    let raw = nodes_rows(&text);
    assert_eq!(raw.len(), mesh.n_elems() as usize * 4 * 2);
    const P1_QUAD_LEX_OF_SLOT: [usize; 4] = [0, 1, 3, 2];
    for e in 0..mesh.n_elems() as usize {
        for (s, &lex) in P1_QUAD_LEX_OF_SLOT.iter().enumerate() {
            let n = geo.conn[e * 4 + s] as usize;
            for c in 0..2 {
                let want = raw[(e * 4 + lex) * 2 + c];
                let got = geo.coords[n * 2 + c];
                assert!(
                    (got - want).abs() < 1e-8 * (1.0 + want.abs()),
                    "elem {e} slot {s} component {c}: got {got}, file row {} has {want}",
                    e * 4 + lex
                );
            }
        }
    }

    for e in 0..mesh.n_elems() {
        let (jac, _x) = fem_mesh::transformation::element_jacobian_at(&mesh, e as u32, &[0.5, 0.5], 2);
        let det = jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)];
        assert!(det > 0.0, "element {e}: folded geometry (det J = {det})");
    }
}

/// The geometry must survive the round trip *as a map*: the per-element node
/// coordinates must be unchanged (same reference slot ↔ same physical point).
#[test]
fn l2_roundtrip_preserves_per_slot_coordinates() {
    let mesh = curved_hex(3);
    let back = read_back(&write_discont(&mesh));
    let (g1, g2) = (
        mesh.geometry.as_ref().expect("geometry"),
        back.geometry.as_ref().expect("geometry"),
    );
    assert_eq!(g1.nodes_per_elem, g2.nodes_per_elem);
    let npe = g2.nodes_per_elem;
    for e in 0..back.n_elems() as usize {
        for s in 0..npe {
            let n1 = g1.conn[e * npe + s] as usize;
            let n2 = g2.conn[e * npe + s] as usize;
            for c in 0..3 {
                let a = g1.coords[n1 * 3 + c];
                let b = g2.coords[n2 * 3 + c];
                assert!(
                    (a - b).abs() < 1e-12,
                    "elem {e} slot {s} component {c}: {a} vs {b}"
                );
            }
        }
    }
}
