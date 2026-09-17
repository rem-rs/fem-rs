//! D165 acceptance tests: fem-rs's `.mesh` writer must reproduce MFEM 4.10's
//! `L2_WedgeElement` `nodes` section for prism (wedge) meshes written with a
//! *discontinuous* geometry space (`Mesh::SetCurvature(order, discont = true)`
//! → `L2_T1_3D_P<p>`).
//!
//! MFEM's `L2_WedgeElement` (`fem/fe/fe_l2.cpp:839`) composes
//! `L2_TriangleElement × L2_SegmentElement` on the same closed Gauss-Lobatto
//! points `PrismPk` uses (classical GLL — `p = 3`: `{0, 0.276393…, 0.723607…,
//! 1}`), and enumerates its `T·(p+1)` dofs (`T = (p+1)(p+2)/2`) layer-major:
//! node `k·T + l` sits at triangle node `l` (`L2_TriangleElement`'s
//! `for (j) for (i+j<=p)` order) of layer `cp[k]` — a permutation of
//! `PrismPk`'s layer-major H1-triangle slot order, so the writer re-orders the
//! mesh's slots through `lex_slot_permutation` (`mfem_l2_slots`) instead of
//! re-evaluating an interpolation matrix.
//!
//! Reference generator: `tmp/r35/flatprism_l2.cpp` (compiled against MFEM 4.10
//! in WSL) — the `tmp/d151/flatprism.cpp` probe with
//! `SetCurvature(order, true, 3, Ordering::byVDIM)`.  A straight-sided mesh is
//! the strongest possible fixture: the geometry is the element's affine prism
//! map, so every dof value is fixed by the numbering alone, and the file
//! fem-rs writes must match MFEM's to roundoff at every order.  `mode 1`
//! rotates the last element's vertex list (rotated shared faces), which the
//! L2 numbering is oblivious to — every dof is element-private, so no face
//! orientation logic can intervene.
//!
//! Regenerate the fixtures with:
//!   `./flatprism_l2 <p> <mode> <out>.mesh`  (p = 2..4, mode = 0/1)

use fem_io::mfem::{read_mfem, write_mfem_file_3d_nodes, NodesSpace};
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

/// The wedge stack of `tmp/r35/flatprism_l2.cpp` with `nphi` elements.
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
    let mut seen: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
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

/// Canonical form of a boundary record for the read → write round trip:
/// `read_mfem` applies MFEM's `Mesh::MarkTetMeshForRefinement`
/// (`mesh/mesh.cpp:3166`), whose second loop runs `MarkEdge` on every
/// *triangular boundary element* as well — so even MFEM's own read → write
/// round trip re-rotates triangle boundary records (cyclically — same
/// orientation, same face).  Rotate each triangle's vertex list to start at
/// its smallest vertex id; every other record is already canonical.
fn canonical_boundary(a: &[Vec<String>]) -> Vec<String> {
    let mut out = Vec::with_capacity(a.len());
    let mut in_bdr = false;
    for row in a {
        let mut line = row.join(" ");
        if row.len() == 1 {
            if row[0] == "boundary" {
                in_bdr = true;
            } else if row[0].parse::<f64>().is_err() {
                in_bdr = false; // the next section keyword
            } // else: the boundary count line — still inside `boundary`
        }
        if in_bdr && row.len() == 5 && row[1] == "2" {
            // Triangle: attr, geom code, v0 v1 v2 → min-first cyclic form.
            let v: Vec<u32> = row[2..].iter().map(|s| s.parse().unwrap()).collect();
            let start = (0..3).min_by_key(|&i| v[i]).unwrap();
            line = format!(
                "{} 2 {} {} {}",
                row[0],
                v[start],
                v[(start + 1) % 3],
                v[(start + 2) % 3]
            );
        }
        out.push(line);
    }
    out
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

fn write_discont_to_temp(mesh: &Mesh<3>, name: &str) -> String {
    let path = std::env::temp_dir().join(name);
    write_mfem_file_3d_nodes(&path, mesh, NodesSpace::Discontinuous).expect("write mesh");
    std::fs::read_to_string(&path).expect("read back")
}

/// One fixture: order, generator mode, and the MFEM-produced file.
struct Case {
    order: u8,
    mode: i32,
    fixture: &'static str,
}

const CASES: &[Case] = &[
    Case { order: 2, mode: 0, fixture: "flatprism_l2-p2-m0.mesh" },
    Case { order: 2, mode: 1, fixture: "flatprism_l2-p2-m1.mesh" },
    Case { order: 3, mode: 0, fixture: "flatprism_l2-p3-m0.mesh" },
    Case { order: 3, mode: 1, fixture: "flatprism_l2-p3-m1.mesh" },
    Case { order: 4, mode: 0, fixture: "flatprism_l2-p4-m0.mesh" },
    Case { order: 4, mode: 1, fixture: "flatprism_l2-p4-m1.mesh" },
];

/// The written discontinuous `nodes` section must match MFEM 4.10's own
/// `L2_T1_3D_P<p>` file for the same (straight-sided) wedge stack.
#[test]
fn prism_l2_nodes_section_matches_mfem() {
    for case in CASES {
        let label = format!("p={} mode={}", case.order, case.mode);
        let mut mesh = flat_prism_stack(3, case.mode);
        mesh.set_curvature(case.order);
        let ours = write_discont_to_temp(
            &mesh,
            &format!("d165_prism_l2_{}_{}.mesh", case.order, case.mode),
        );
        assert!(
            ours.contains(&format!("L2_T1_3D_P{}", case.order)),
            "{label}: no L2_T1 nodes section in\n{ours}"
        );
        let fixture = std::fs::read_to_string(format!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/{}"),
            case.fixture
        ))
        .expect("fixture");
        compare_mesh(&ours, &fixture, 1e-14, &label);
    }
}

/// The written file must be a function of the mesh alone: two writes are
/// byte-identical.
#[test]
fn prism_l2_nodes_write_is_deterministic() {
    for case in CASES {
        let mut mesh = flat_prism_stack(3, case.mode);
        mesh.set_curvature(case.order);
        let first =
            write_discont_to_temp(&mesh, &format!("d165_rt_a_{}_{}.mesh", case.order, case.mode));
        let second =
            write_discont_to_temp(&mesh, &format!("d165_rt_b_{}_{}.mesh", case.order, case.mode));
        assert_eq!(first, second, "p={} mode={}: writer is not deterministic", case.order, case.mode);
    }
}

/// MFEM's own `L2_T1_3D_P<p>` wedge file must be *read* into the mesh's
/// `PrismPk` slot order: the reader applies the inverse of the writer's
/// permutation, so reading the fixture and writing it back must reproduce it
/// to the fixture's 17-digit print precision.
#[test]
fn prism_l2_fixture_round_trips_through_the_reader() {
    for case in CASES {
        let label = format!("p={} mode={}", case.order, case.mode);
        let fixture = std::fs::read_to_string(format!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/{}"),
            case.fixture
        ))
        .expect("fixture");
        let mesh = read_mfem(fixture.as_bytes())
            .expect("read_mfem must accept MFEM's own -dm wedge file")
            .mesh3d
            .expect("3-D mesh");
        assert_eq!(mesh.geom_order(), case.order, "{label}: wrong geometric order");
        let npe_expected = (case.order as usize + 1) * (case.order as usize + 1)
            * (case.order as usize + 2)
            / 2;
        let geo = mesh.geometry.as_ref().expect("L2 nodes geometry table");
        assert_eq!(
            geo.nodes_per_elem, npe_expected,
            "{label}: wrong per-element private dof count (MFEM L2_WedgeElement)"
        );
        let back = write_discont_to_temp(
            &mesh,
            &format!("d165_roundtrip_{}_{}.mesh", case.order, case.mode),
        );
        // The round trip must close to one rounding quantum at the writer's
        // stream precision: `compare_mesh` with `tol = 5e-16` (1 ulp at
        // `Mesh::Save`'s 16 significant digits, D274) still catches any
        // mis-numbering through the permutation, while tolerating that MFEM
        // prints its fixtures with 17 digits (`0.27639320225002106`) and both
        // writers now re-print at 16 (`0.2763932022500211`) — MFEM's own
        // `Mesh::Save(out, 16)` re-save of these very fixtures truncates the
        // same way (`$HOME/work/r31_save`, `tmp/d294/`).  The boundary
        // triangles are put in cyclic-canonical form first
        // (`canonical_boundary` above — `read_mfem`'s faithful
        // `MarkTetMeshForRefinement` re-rotation).
        let canon = |text: &str| canonical_boundary(&tokens(text)).join("\n");
        compare_mesh(&canon(&back), &canon(&fixture), 5e-16, &label);
    }
}
