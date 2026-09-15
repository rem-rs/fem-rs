//! D187: `Mesh::<3>::set_curvature` on a triangular **surface** mesh (Dim = 2,
//! spaceDim = 3) must use the same closed Gauss-Lobatto lattice as MFEM.
//!
//! Anchor: `crates/mesh/src/simplex.rs` `set_curvature_tri3` used to place its
//! geometry nodes on the *equispaced* `TriPk` lattice **and** radially project
//! every new node onto the unit sphere.  Both deviate from MFEM's
//! `Mesh::SetCurvature(p)` (`mesh/mesh.cpp` 7207-7241), which builds
//! `H1_FECollection(order, Dim, BasisType::GaussLobatto)` — i.e. the
//! `H1_TriangleElement(p)` lattice (`H1TriPk` here) — and evaluates the
//! mesh's *old, straight* transformation at the new nodes' reference points:
//! **no projection of any kind**.  On the octahedron below that leaves the
//! edge nodes at ~0.775 from the origin (on the chord), not on the sphere.
//!
//! Reference (MFEM 4.10 serial, `tmp/d187/octa_probe.cpp`): the ex7
//! octahedron (6 vertices, 8 triangles — the same connectivity
//! `Mesh::<3>::unit_sphere_octahedron()` mirrors), `SetCurvature(p)` for
//! p = 2, 3, 4, per-element H1 slot tables printed at 17 digits:
//!
//! ```text
//! wsl g++ -std=c++17 -O2 -I$HOME/mfem410_ser octa_probe.cpp \
//!     $HOME/mfem410_ser/libmfem.a -o octa_probe && ./octa_probe 3
//! ```
//!
//! The tables below are the probe's verbatim `slot` lines.  They pin, per
//! element: the vertex slots, the direction of every edge block (slots 3..3+n
//! walk v0→v1, v1→v2, v2→v0), and the interior slots — everything the old
//! equispaced-plus-sphere-snap code got wrong (worst deviation 2.5e-1 at
//! p = 3 edge nodes, measured in `tmp/d187/femrs_before_p3.txt`).

use fem_mesh::Mesh;

/// `./octa_probe 2` slot lines: p = 2 edge midpoints sit on the chords
/// (`0.5 0.5 0`, norm 0.7071 — the old code snapped them to 0.7071/0.7071/0,
/// norm 1).
const CPP_P2: &str = include_str!("d187/octa_p2_slots.txt");
/// `./octa_probe 3` slot lines: the GLL edge parameters 0.7236/0.2764 make
/// the equispaced-vs-GLL split visible for the first time.
const CPP_P3: &str = include_str!("d187/octa_p3_slots.txt");
/// `./octa_probe 4`: three edge dofs per edge plus the first interior lattice
/// (3 interior dofs at the GLL barycentric points).
const CPP_P4: &str = include_str!("d187/octa_p4_slots.txt");

/// The `Mesh::<3>::unit_sphere_octahedron()` connectivity, in probe order.
const OCTA_CONN: [[u32; 3]; 8] = [
    [0, 1, 4],
    [1, 2, 4],
    [2, 3, 4],
    [3, 0, 4],
    [1, 0, 5],
    [2, 1, 5],
    [3, 2, 5],
    [0, 3, 5],
];

/// Parse a probe's `slot` lines into `table[element][slot] = [x, y, z]`.
fn parse_cpp_slots(txt: &str) -> Vec<Vec<[f64; 3]>> {
    let mut table: Vec<Vec<[f64; 3]>> = Vec::new();
    for line in txt.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("slot") {
            // `slot  0 dof  0 : x y z` — the coordinates follow the colon.
            let vals: Vec<f64> = rest.split(':').nth(1).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
            assert_eq!(vals.len(), 3, "slot line must carry x y z: {t}");
            table.last_mut().expect("slot before its elem line").push([vals[0], vals[1], vals[2]]);
        } else if t.starts_with("elem ") {
            table.push(Vec::new());
        }
    }
    table
}

/// fem-rs's in-memory geometry table for the octahedron at order `p`, as
/// `coords[element][slot]` (slot = the H1TriPk table order).
fn femrs_octahedron_slots(p: u8) -> Vec<Vec<[f64; 3]>> {
    let mut m = Mesh::<3>::unit_sphere_octahedron();
    m.set_curvature(p);
    let g = m.geometry.as_ref().expect("set_curvature must build geometry");
    (0..8usize)
        .map(|e| {
            let base = e * g.nodes_per_elem;
            (0..g.nodes_per_elem)
                .map(|s| {
                    let n = g.conn[base + s] as usize;
                    [g.coords[3 * n], g.coords[3 * n + 1], g.coords[3 * n + 2]]
                })
                .collect()
        })
        .collect()
}

/// The order-p tables must match MFEM slot for slot, coordinate for
/// coordinate (probe prints 17 digits; the values agree to ~1 ulp).
#[test]
fn curvature_tables_match_mfem_octahedron_p2_p3_p4() {
    for (p, txt) in [(2u8, CPP_P2), (3, CPP_P3), (4, CPP_P4)] {
        let cpp = parse_cpp_slots(txt);
        assert_eq!(cpp.len(), 8, "probe must carry 8 elements at p = {p}");
        let ours = femrs_octahedron_slots(p);
        let mut worst = 0.0f64;
        for (e, (c, o)) in cpp.iter().zip(&ours).enumerate() {
            assert_eq!(c.len(), o.len(), "p = {p} element {e}: slot count");
            for (_s, (cv, ov)) in c.iter().zip(o).enumerate() {
                for d in 0..3 {
                    worst = worst.max((cv[d] - ov[d]).abs());
                }
            }
        }
        assert!(
            worst < 1e-14,
            "p = {p}: geometry deviates from the MFEM probe by {worst}"
        );
    }
}

/// The vertex slots must reuse the mesh vertices, not copies of them.
#[test]
fn vertex_slots_reuse_mesh_vertices() {
    let m = {
        let mut m = Mesh::<3>::unit_sphere_octahedron();
        m.set_curvature(3);
        m
    };
    let g = m.geometry.as_ref().unwrap();
    for (e, verts) in OCTA_CONN.iter().enumerate() {
        for (s, &vid) in verts.iter().enumerate() {
            assert_eq!(g.conn[e * g.nodes_per_elem + s], vid, "elem {e} slot {s}");
        }
    }
}

/// A straight-sided planar triangle in 3-D must keep its geometry ON THE
/// PLANE at the affine image of each lattice point: the old code's baked-in
/// unit-sphere projection moved nodes *within* the plane too (the p = 2 edge
/// midpoint (0.5, 0, 0) of this triangle was normalized onto (1, 0, 0) — the
/// vertex! — and MFEM's SetCurvature never projects).  Every order-p node
/// lies at `Σ λ_v·c_v`, so z stays exactly 0 and x, y are the GLL barycentric
/// combinations.
#[test]
fn planar_straight_tri_stays_on_its_plane() {
    use fem_element::lagrange::factory::H1TriPk;
    use fem_element::ReferenceElement;
    // Single z = 0 triangle with a vertex at the origin: radial projection
    // cannot change z here, so the pin is about the in-plane positions.
    let coords: Vec<f64> = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].into();
    let mut m = Mesh::<3>::uniform(
        coords,
        vec![0, 1, 2],
        vec![1],
        fem_mesh::ElementType::Tri3,
        vec![0, 1, 2],
        vec![1],
        fem_mesh::ElementType::Line2,
    );
    m.set_curvature(3);
    let g = m.geometry.as_ref().expect("geometry");
    let ref_nodes = H1TriPk::new(3).dof_coords();
    assert_eq!(g.nodes_per_elem, ref_nodes.len());
    let c = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    for (s, xi) in ref_nodes.iter().enumerate() {
        let lam = [1.0 - xi[0] - xi[1], xi[0], xi[1]];
        let n = g.conn[s] as usize;
        for d in 0..3 {
            let want: f64 = (0..3).map(|v| lam[v] * c[v][d]).sum();
            assert!(
                (g.coords[3 * n + d] - want).abs() < 1e-14,
                "slot {s} comp {d}: got {}, want the affine {want}",
                g.coords[3 * n + d]
            );
        }
    }
}

/// Edge blocks are shared direction-aware: two triangles meeting along an
/// edge list the shared GLL nodes in *opposite* slot order but with the
/// *same* coordinates and the same node ids.  (Octahedron elements 0/1 share
/// edge 1-4, i.e. MFEM dofs 8/9 in the probe tables.)
#[test]
fn shared_edge_nodes_are_deduplicated_direction_aware() {
    let ours = femrs_octahedron_slots(3);
    // Element 0 edge v1→v2 slots 5,6 (dofs 8,9) vs element 1 edge v2→v0
    // slots 8,7 (dofs 9,8): reversed order, identical coordinates.
    assert_eq!(ours[0][5], ours[1][8], "edge dof 8");
    assert_eq!(ours[0][6], ours[1][7], "edge dof 9");
    // And the ids really are shared, not duplicated per element.
    let m = {
        let mut m = Mesh::<3>::unit_sphere_octahedron();
        m.set_curvature(3);
        m
    };
    let g = m.geometry.as_ref().unwrap();
    assert_eq!(
        g.conn[0 * g.nodes_per_elem + 5],
        g.conn[1 * g.nodes_per_elem + 8],
        "the shared edge node must be one geometry node, not two"
    );
}
