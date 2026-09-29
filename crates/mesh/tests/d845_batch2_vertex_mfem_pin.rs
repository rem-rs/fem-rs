//! D843-1: batch ≥ 2 vertex renumbering + leaf-state evolution vs MFEM 4.10.
//!
//! ex6's ZZ loop marks (probe `tmp/d92c/probe_d92c.cpp`, MFEM 4.10 serial):
//! batch 1 = 15 elements, batch 2 = 10, batch 3 = 45, batch 4 = 20.  This pin
//! replays those four batches through the stateless NC refinement and asserts
//! the resulting vertex table (ids + coordinates, bitwise) matches MFEM's
//! `UpdateVertices` output after batch 4 — deep enough that the per-leaf
//! Hilbert state carrier (`nc_leaf_states`) is load-bearing: the pre-D843-1
//! greedy recomputation reordered one leaf triplet at batch 4 and swapped two
//! center ids (ex6 iter4 X differed by 11 ulps, stdout diverged from iter6).

use fem_io::mfem::read_mfem_file;
use fem_mesh::amr::{refine_nonconforming_quad_aniso, QuadRefineDir};
use fem_mesh::Mesh;

const FIXTURE: &str = include_str!("data/d845_batch2_vertices_mfem.txt");


fn refine(mesh: &Mesh<2>, marked: &[(u32, QuadRefineDir)]) -> Mesh<2> {
    let (mesh, _cons) = refine_nonconforming_quad_aniso(mesh, marked, None);
    mesh
}

const BOTH1: &[(u32, QuadRefineDir)] = &[
    (0, QuadRefineDir::Both), (1, QuadRefineDir::Both), (2, QuadRefineDir::Both),
    (3, QuadRefineDir::Both), (4, QuadRefineDir::Both), (5, QuadRefineDir::Both),
    (7, QuadRefineDir::Both), (8, QuadRefineDir::Both), (10, QuadRefineDir::Both),
    (11, QuadRefineDir::Both), (13, QuadRefineDir::Both), (14, QuadRefineDir::Both),
    (16, QuadRefineDir::Both), (17, QuadRefineDir::Both), (19, QuadRefineDir::Both),
];
const BOTH2: &[(u32, QuadRefineDir)] = &[
    (21, QuadRefineDir::Both), (28, QuadRefineDir::Both), (29, QuadRefineDir::Both),
    (36, QuadRefineDir::Both), (39, QuadRefineDir::Both), (46, QuadRefineDir::Both),
    (47, QuadRefineDir::Both), (54, QuadRefineDir::Both), (57, QuadRefineDir::Both),
    (62, QuadRefineDir::Both),
];
const MIX3: &[(u32, QuadRefineDir)] = &[
    (0, QuadRefineDir::Both), (1, QuadRefineDir::Both), (2, QuadRefineDir::Both),
    (3, QuadRefineDir::Both), (4, QuadRefineDir::Both), (5, QuadRefineDir::Both),
    (6, QuadRefineDir::Both), (7, QuadRefineDir::Both), (8, QuadRefineDir::Both),
    (9, QuadRefineDir::Both), (10, QuadRefineDir::Both), (11, QuadRefineDir::Both),
    (12, QuadRefineDir::Both), (13, QuadRefineDir::Both), (14, QuadRefineDir::Both),
    (15, QuadRefineDir::Both), (16, QuadRefineDir::Both), (17, QuadRefineDir::Both),
    (18, QuadRefineDir::Both), (19, QuadRefineDir::Both), (22, QuadRefineDir::Both),
    (26, QuadRefineDir::Both), (27, QuadRefineDir::Both), (29, QuadRefineDir::Both),
    (34, QuadRefineDir::Both), (35, QuadRefineDir::Both), (40, QuadRefineDir::Both),
    (42, QuadRefineDir::Both), (43, QuadRefineDir::Both), (47, QuadRefineDir::Both),
    (52, QuadRefineDir::Both), (56, QuadRefineDir::Both), (57, QuadRefineDir::Both),
    (59, QuadRefineDir::Both), (64, QuadRefineDir::Both), (65, QuadRefineDir::Both),
    (70, QuadRefineDir::Both), (72, QuadRefineDir::Both), (73, QuadRefineDir::Both),
    (77, QuadRefineDir::Both), (82, QuadRefineDir::Both), (86, QuadRefineDir::Both),
    (87, QuadRefineDir::Both), (90, QuadRefineDir::Both), (94, QuadRefineDir::Both),
];
const ANISO4: &[(u32, QuadRefineDir)] = &[
    (80, QuadRefineDir::X), (88, QuadRefineDir::X), (97, QuadRefineDir::Y),
    (102, QuadRefineDir::Y), (117, QuadRefineDir::X), (122, QuadRefineDir::X),
    (131, QuadRefineDir::Y), (139, QuadRefineDir::Y), (140, QuadRefineDir::X),
    (148, QuadRefineDir::X), (157, QuadRefineDir::Y), (162, QuadRefineDir::Y),
    (177, QuadRefineDir::X), (182, QuadRefineDir::X), (191, QuadRefineDir::Y),
    (199, QuadRefineDir::Y), (200, QuadRefineDir::X), (208, QuadRefineDir::X),
    (217, QuadRefineDir::Y), (225, QuadRefineDir::Y),
];

#[test]
fn d845_batch4_vertex_table_matches_mfem() {
    let mesh0 = read_mfem_file("../../data/star.mesh")
        .expect("star.mesh")
        .mesh2d
        .expect("2-D");
    let mesh1 = refine(&mesh0, BOTH1);
    let mesh2 = refine(&mesh1, BOTH2);
    let mesh3 = refine(&mesh2, MIX3);
    let mesh4 = refine(&mesh3, ANISO4);

    // fixture: NV4 header + V4 lines (id, x, y) — bitwise vs MFEM
    let mut nv = None;
    let mut verts: Vec<(u32, f64, f64)> = Vec::new();
    for line in FIXTURE.lines() {
        if let Some(rest) = line.strip_prefix("NV4 ") {
            nv = Some(rest.split_whitespace().next().unwrap().parse::<usize>().unwrap());
        } else if let Some(rest) = line.strip_prefix("V4 ") {
            let mut it = rest.split_whitespace();
            let id: u32 = it.next().unwrap().parse().unwrap();
            let x: f64 = it.next().unwrap().parse().unwrap();
            let y: f64 = it.next().unwrap().parse().unwrap();
            verts.push((id, x, y));
        }
    }
    assert_eq!(nv, Some(mesh4.n_nodes()), "batch-4 vertex count");
    assert_eq!(verts.len(), mesh4.n_nodes(), "fixture vertex count");
    for &(id, x, y) in &verts {
        let p = mesh4.coords_of(id as _);
        assert_eq!(&[p[0], p[1]], &[x, y], "vertex {id} after batch 4");
    }
}

#[test]
fn d845_batch2_vertex_table_matches_mfem() {
    let mesh0 = read_mfem_file("../../data/star.mesh")
        .expect("star.mesh")
        .mesh2d
        .expect("2-D");
    let mesh1 = refine(&mesh0, BOTH1);
    let mesh2 = refine(&mesh1, BOTH2);

    let mut nv = None;
    let mut verts: Vec<(u32, f64, f64)> = Vec::new();
    for line in FIXTURE.lines() {
        if let Some(rest) = line.strip_prefix("NV2 ") {
            nv = Some(rest.split_whitespace().next().unwrap().parse::<usize>().unwrap());
        } else if let Some(rest) = line.strip_prefix("V2 ") {
            let mut it = rest.split_whitespace();
            let id: u32 = it.next().unwrap().parse().unwrap();
            let x: f64 = it.next().unwrap().parse().unwrap();
            let y: f64 = it.next().unwrap().parse().unwrap();
            verts.push((id, x, y));
        }
    }
    assert_eq!(nv, Some(mesh2.n_nodes()), "batch-2 vertex count");
    assert_eq!(verts.len(), mesh2.n_nodes(), "fixture vertex count");
    for &(id, x, y) in &verts {
        let p = mesh2.coords_of(id as _);
        assert_eq!(&[p[0], p[1]], &[x, y], "vertex {id} after batch 2");
    }

    let states = mesh2.nc_leaf_states.as_ref().expect("carried leaf states");
    assert_eq!(states.len(), mesh2.n_elems(), "one state per leaf");
}
