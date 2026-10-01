//! D102 — `Mesh::Extrude1D` (1-D segment → 2-D quads), the `extruder` miniapp's
//! 1-D path, previously an honest `exit(3)` gap.
//!
//! Acceptance against the C++ 4.10 oracle (`$HOME/work/d102/extruder_cpp`,
//! built from `miniapps/meshing/extruder.cpp` + `$HOME/mfem410_ser`):
//! `extruder_cpp -m data/inline-segment.mesh -ny 8 -wy 2` writes a
//! `extruder.mesh` that is **byte-identical** to the Rust miniapp's output
//! (128 lines; also verified for the default `ny = 1` autoselect and the
//! chained `-ny 4 -wy 1 -nz 2 -hz 1` 1-D → 2-D → 3-D run).  This test pins
//! the structural values of that file so the mapping cannot drift.
//!
//! MFEM semantics (`mesh/mesh.cpp:15585`, `closed = false`): vertices are
//! point-major (`v*nvy + j`, `y = sy·j/ny`), elements are edge-major with
//! layer-inner loops, boundary = source POINTs extruded per layer (odd
//! attributes flip the segment orientation, MFEM `if (attr%2) Swap`) followed
//! by each element's bottom `(v0_0, v1_0)` and top `(v1_ny, v0_ny)` segments
//! with attribute `nba + elem attr`.
use fem_io::mfem::read_mfem_file;
use fem_mesh::extrusion::extrude_1d;

#[test]
fn d102_extrude_1d_matches_cpp_extruder() {
    let file = read_mfem_file("../../data/inline-segment.mesh").expect("read 1-D mesh");
    let m1 = file.mesh1d.expect("inline-segment.mesh must load as Mesh<1>");

    let m2 = extrude_1d(&m1, 8, 2.0);

    // Counts from the C++ oracle run (-ny 8 -wy 2): 32 quads, 24 boundary
    // segments, 45 vertices (2 source vertices × 9 layers).
    assert_eq!(m2.n_elems(), 32);
    assert_eq!(m2.n_faces(), 24);
    assert_eq!(m2.n_nodes(), 45);

    // Vertex layering: source vertex v owns nodes [v*9, v*9+9) with y =
    // 2·j/8 (inline-segment has 5 vertices at x = 0, 0.25, …, 1 — nx = 4).
    let y = |j: usize| 2.0 * (j as f64 / 8.0);
    assert_eq!(m2.coords_of(0)[0], 0.0);
    assert_eq!(m2.coords_of(0)[1], 0.0);
    assert_eq!(m2.coords_of(4)[1], y(4));
    assert_eq!(m2.coords_of(9)[0], m1.coords_of(1)[0]);
    assert_eq!(m2.coords_of(9)[1], 0.0);
    let last = m2.n_nodes() as u32 - 1;
    assert_eq!(
        (m2.coords_of(last)[0], m2.coords_of(last)[1]),
        (m1.coords_of(4)[0], 2.0)
    );

    // Element connectivity: elem 0 = {0,9,10,1}, elem 31 = {34,43,44,35}
    // (C++ oracle's first/last `1 3 …` lines).
    let e0: Vec<u32> = m2.elem_nodes(0).to_vec();
    assert_eq!(e0, vec![0, 9, 10, 1]);
    let e31: Vec<u32> = m2.elem_nodes(31).to_vec();
    assert_eq!(e31, vec![34, 43, 44, 35]);

    // Boundary: 8× attr 1 (x=0 wall, odd attr → swapped orientation), 8×
    // attr 2 (x=1 wall), 8× attr 3 (bottom+top of the 4 edges; attr =
    // nba + elem attr = 2 + 1).
    let mut hist = std::collections::BTreeMap::new();
    for t in &m2.face_tags {
        *hist.entry(*t).or_insert(0) += 1;
    }
    assert_eq!(hist, [(1, 8), (2, 8), (3, 8)].into_iter().collect());
    // Face 0: source POINT attr 1 (odd → swap): {1, 0} — C++ `1 1 1 0`.
    assert_eq!(m2.bface_nodes(0), &[1, 0]);
    // Face 8: source POINT attr 2 (even → as-is): {36, 37} — C++ `2 1 36 37`.
    assert_eq!(m2.bface_nodes(8), &[36, 37]);
    // Face 16: first bottom segment (0, 9) — C++ `3 1 0 9`.
    assert_eq!(m2.bface_nodes(16), &[0, 9]);
    // Face 17: elem 0's top segment (17, 8) — C++ `3 1 17 8`.
    assert_eq!(m2.bface_nodes(17), &[17, 8]);
}
