//! D151 acceptance tests: fem-rs's `.mesh` writer must reproduce MFEM 4.10's
//! `H1_WedgeElement` `nodes` section for prism (wedge) meshes.
//!
//! MFEM's `H1_WedgeElement` (`fem/fe/fe_h1.cpp:863`) places its dofs on the
//! **closed Gauss-Lobatto** points of `H1_TriangleElement ×
//! H1_SegmentElement`, while fem-rs's prism geometry element `PrismPk` is
//! equispaced: the two lattices agree for `p ≤ 2` and differ from `p = 3`
//! (`0.276393202250021` vs `1/3`).  The writer therefore evaluates the mesh's
//! own `PrismPk` geometry at MFEM's nodes (see `prism_nodes_dof_values`).
//! These tests pin that against files produced by MFEM 4.10 itself.
//!
//! Reference generator: `tmp/d151/flatprism.cpp` (compiled against MFEM 4.10
//! in WSL).  It builds a straight-sided stack of wedges — exactly the element
//! arrangement of `miniapps/meshing/toroid.cpp` before its transform — and
//! promotes it with `SetCurvature(order, false, 3, Ordering::byVDIM)`.  A
//! straight-sided mesh is the strongest possible fixture: the geometry is the
//! element's affine prism map, so *every* dof value is fixed by the numbering
//! alone (no interpolation freedom), and the file fem-rs writes must match
//! MFEM's to roundoff at every order.  `mode 1` rotates the last element's
//! vertex list, which rotates the shared quadrilateral faces and the shared
//! triangular face (exercising the `QuadDofOrd`/`TriDofOrd` paths).
//!
//! Regenerate the fixtures with:
//!   `./flatprism <p> <mode> <out>.mesh`  (p = 2..4, mode = 0/1)

use std::collections::HashMap;

use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

/// MFEM `Geometry::Constants<PRISM>::FaceVert` (vertex lists of the five local
/// faces of a wedge: bottom triangle, top triangle, three quadrilaterals).
const PRISM_FACES: [[usize; 4]; 5] = [
    [0, 2, 1, usize::MAX],
    [3, 4, 5, usize::MAX],
    [0, 1, 4, 3],
    [1, 2, 5, 4],
    [2, 0, 3, 5],
];

/// The wedge stack of `tmp/d151/flatprism.cpp` with `nphi` elements.
fn flat_prism_stack(nphi: usize, mode: i32) -> Mesh<3> {
    let mut mesh: Mesh<3> =
        Mesh::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 0.0, 0.0, 0.0, false);
    mesh.conn.clear();
    mesh.elem_tags.clear();
    mesh.face_conn.clear();
    mesh.face_tags.clear();
    mesh.coords.clear();
    mesh.elem_type = ElementType::Prism6;

    for i in 0..=nphi {
        let z = i as f64;
        mesh.add_vertex_3d(0.0, 0.0, z);
        mesh.add_vertex_3d(1.0, 0.0, z);
        mesh.add_vertex_3d(0.0, 1.0, z);
    }
    for i in 0..nphi {
        let v: [u32; 6] = [
            3 * i as u32,
            3 * i as u32 + 1,
            3 * i as u32 + 2,
            3 * (i + 1) as u32,
            3 * (i + 1) as u32 + 1,
            3 * (i + 1) as u32 + 2,
        ];
        let v = if mode == 1 && i == nphi - 1 {
            // Rotate the triangle numbering on both ends: still a valid wedge,
            // but the shared faces get a rotated orientation.
            [v[1], v[2], v[0], v[4], v[5], v[3]]
        } else {
            v
        };
        mesh.add_wedge(&v, 1);
    }
    finalize_prism_topology(&mut mesh);
    mesh
}

/// MFEM `Mesh::FinalizeTopology` for this wedge stack: faces are numbered by
/// element traversal × local `FaceVert` order (first encounter wins) and the
/// boundary elements are the faces owned by exactly one element.
fn finalize_prism_topology(mesh: &mut Mesh<3>) {
    let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut all: Vec<(Vec<u32>, ElementType, u32)> = Vec::new();
    for e in 0..mesh.n_elems() {
        let ns = mesh.elem_nodes(e as u32).to_vec();
        for fv in PRISM_FACES.iter() {
            let key: Vec<u32> = fv
                .iter()
                .filter(|&&v| v != usize::MAX)
                .map(|&v| ns[v])
                .collect();
            let mut sorted = key.clone();
            sorted.sort_unstable();
            match seen.get(&sorted) {
                Some(&fi) => all[fi].2 += 1,
                None => {
                    let ft = if fv[3] == usize::MAX {
                        ElementType::Tri3
                    } else {
                        ElementType::Quad4
                    };
                    seen.insert(sorted, all.len());
                    all.push((key, ft, 1));
                }
            }
        }
    }
    let mut conn = Vec::new();
    let mut tags = Vec::new();
    let mut types = Vec::new();
    for (key, ft, count) in all {
        if count == 1 {
            conn.extend_from_slice(&key);
            tags.push(1);
            types.push(ft);
        }
    }
    mesh.face_conn = conn;
    mesh.face_tags = tags.into_iter().collect();
    let uniform = types.windows(2).all(|w| w[0] == w[1]);
    if uniform {
        mesh.face_type = types[0];
        mesh.face_types = None;
        mesh.face_offsets = None;
    } else {
        // A prism mesh has triangular *and* quadrilateral boundary faces, so
        // `Mesh::n_faces` needs the per-face offsets to walk `face_conn`.
        let mut offsets = Vec::with_capacity(types.len() + 1);
        offsets.push(0);
        for t in &types {
            offsets.push(offsets.last().unwrap() + t.nodes_per_element());
        }
        mesh.face_type = types[0];
        mesh.face_types = Some(types);
        mesh.face_offsets = Some(offsets);
    }
}

/// The non-comment lines of a `.mesh` file, each split into whitespace tokens.
fn tokens(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect()
}

/// Compare two `.mesh` files line by line: every token must be equal, or two
/// numbers agreeing to `tol` (MFEM writes its fixtures with 17 digits).
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

fn write_to_temp(mesh: &Mesh<3>, name: &str) -> String {
    let path = std::env::temp_dir().join(name);
    write_mfem_file_3d(&path, mesh).expect("write mesh");
    std::fs::read_to_string(&path).expect("read back")
}

/// One fixture: order, generator mode, and the MFEM-produced file.
struct Case {
    order: u8,
    mode: i32,
    fixture: &'static str,
}

const CASES: &[Case] = &[
    Case { order: 2, mode: 0, fixture: "flatprism-p2-m0.mesh" },
    Case { order: 2, mode: 1, fixture: "flatprism-p2-m1.mesh" },
    Case { order: 3, mode: 0, fixture: "flatprism-p3-m0.mesh" },
    Case { order: 3, mode: 1, fixture: "flatprism-p3-m1.mesh" },
    Case { order: 4, mode: 0, fixture: "flatprism-p4-m0.mesh" },
    Case { order: 4, mode: 1, fixture: "flatprism-p4-m1.mesh" },
];

/// The written `nodes` section must match MFEM 4.10's own file for the same
/// (straight-sided) wedge stack — at *every* order, including the orders where
/// the `PrismPk` and `H1_WedgeElement` node lattices differ.
#[test]
fn prism_nodes_section_matches_mfem() {
    for case in CASES {
        let label = format!("p={} mode={}", case.order, case.mode);
        let mut mesh = flat_prism_stack(3, case.mode);
        mesh.set_curvature(case.order);
        let ours = write_to_temp(&mesh, &format!("d151_prism_{}_{}.mesh", case.order, case.mode));
        assert!(
            ours.contains(&format!("H1_3D_P{}", case.order)),
            "{label}: no H1 nodes section in\n{ours}"
        );
        let fixture = std::fs::read_to_string(format!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/{}"),
            case.fixture
        ))
        .expect("fixture");
        compare_mesh(&ours, &fixture, 1e-14, &label);
    }
}

/// The `nodes` section the writer emits for a given mesh must be a function of
/// the mesh alone (no hidden state, no filesystem dependence), so two writes
/// have to be byte-identical.
#[test]
fn prism_nodes_write_is_deterministic() {
    for case in CASES {
        let mut mesh = flat_prism_stack(3, case.mode);
        mesh.set_curvature(case.order);
        let first = write_to_temp(&mesh, &format!("d151_rt_a_{}_{}.mesh", case.order, case.mode));
        let second = write_to_temp(&mesh, &format!("d151_rt_b_{}_{}.mesh", case.order, case.mode));
        assert_eq!(first, second, "p={} mode={}: writer is not deterministic", case.order, case.mode);
    }
}
