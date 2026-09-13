//! Acceptance tests for uniform refinement of the *curved* hex mesh
//! `data/multidomain-hex.mesh` (MFEM `nodes` H1_3D_P2 geometry).
//!
//! Reference: serial MFEM 4.9 `Mesh::UniformRefinement()` on the same file,
//! dumping every refined child hex's 8 corner coordinates at `%.17e`
//! (sorted lexicographically within each element). Regenerate with:
//!
//! ```text
//! wsl g++ -std=c++17 -O2 -I$HOME/mfem49 hexdump17.cpp $HOME/mfem49/libmfem.a \
//!     -o hexdump17 && ./hexdump17 -m ../../data/multidomain-hex.mesh
//! ```
//!
//! (harness kept in `tmp/multidomain/hexdump17.cpp`; the fixture checked in
//! under `tests/data/` is its output).
//!
//! Root cause this guards against (defect D1): MFEM's `UniformRefinement`
//! on a curved mesh ends with `UpdateNodes()` → `SetVerticesFromNodes`, so
//! every fine vertex receives the coarse Qk geometry-dof value at its
//! reference point — plain straight averaging matches only on affine
//! regions (96/480 children on this mesh).
//!
//! Tolerance: MFEM's refinement operator evaluates the coarse nodal basis
//! through dense-matrix arithmetic, so its fine vertex coordinates carry
//! ~1 ulp noise (~1e-17 here) relative to the exact dof picks. The
//! acceptance criterion (≤1e-15 per coordinate) is checked with greedy
//! nearest-corner matching per element; the rust picks themselves are
//! exact, and most coordinates match the dump bit-for-bit.

use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_hex8_uniform, refine_uniform_3d, Mesh};

const PARENT_MESH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/multidomain-hex.mesh");
const CPP_DUMP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/multidomain_hex_refined_cpp.txt");
const TOL: f64 = 1e-15;

/// Load the curved parent mesh (60 Hex8, order-2 geometry).
fn load_parent() -> Mesh<3> {
    let f = read_mfem_file(PARENT_MESH).expect("read parent mesh");
    let mesh: Mesh<3> = f.mesh3d.expect("3D mesh");
    assert_eq!(mesh.n_elems(), 60);
    assert!(mesh.geometry.is_some(), "parent must carry curved geometry");
    mesh
}

/// Parse the MFEM reference dump: one line per child hex, 24 floats
/// (8 corners × xyz), corners sorted lexicographically within the line.
fn load_cpp_dump() -> Vec<Vec<[f64; 3]>> {
    let text = std::fs::read_to_string(CPP_DUMP).expect("read reference dump");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: Vec<f64> = line.split_whitespace().map(|t| t.parse().expect("float")).collect();
            assert_eq!(v.len(), 24, "reference row must hold 8 corners");
            (0..8)
                .map(|k| [v[3 * k], v[3 * k + 1], v[3 * k + 2]])
                .collect()
        })
        .collect()
}

/// Corner triples of fine element `e`.
fn corners(mesh: &Mesh<3>, e: u32) -> Vec<[f64; 3]> {
    mesh.elem_nodes(e)
        .iter()
        .map(|&n| {
            let c = mesh.coords_of(n);
            [c[0], c[1], c[2]]
        })
        .collect()
}

/// Compare all refined children against the MFEM reference dump and return
/// (coordinate-exact-match count, max |Δ|). Corners are matched by greedy
/// nearest pairing per element (MFEM's ulp noise can flip lexicographic
/// sort ties, so positional matching of sorted lists is not robust).
fn compare_with_dump(fine: &Mesh<3>, cpp: &[Vec<[f64; 3]>]) -> (usize, f64) {
    assert_eq!(fine.n_elems() as usize, cpp.len(), "child count must match reference");
    let mut exact = 0usize;
    let mut max_diff = 0.0_f64;
    for e in 0..fine.n_elems() {
        let got = corners(fine, e as u32);
        let want = &cpp[e as usize];
        let mut used = [false; 8];
        for g in &got {
            let mut best: Option<(usize, f64)> = None;
            for (k, (&u, &w)) in used.iter().zip(want.iter()).enumerate() {
                if u { continue; }
                let d = (g[0] - w[0]).abs().max((g[1] - w[1]).abs()).max((g[2] - w[2]).abs());
                if best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((k, d));
                }
            }
            let (k, d) = best.expect("more rust corners than reference corners");
            assert!(
                d <= TOL,
                "child {e}: no reference corner within {TOL:e} of {g:?} (best Δ={d:e})"
            );
            used[k] = true;
            if d == 0.0 { exact += 3; }
            max_diff = max_diff.max(d);
        }
    }
    (exact, max_diff)
}

#[test]
fn refine_hex8_uniform_matches_mfem_reference() {
    let parent = load_parent();
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&parent, &all);
    let cpp = load_cpp_dump();
    let (exact, max_diff) = compare_with_dump(&fine, &cpp);
    // 480/480 children, all 8 corners within 1e-15 of MFEM.
    assert_eq!(fine.n_elems() as usize, 480);
    assert!(max_diff <= TOL, "max coordinate deviation {max_diff:e} exceeds {TOL:e}");
    eprintln!("multidomain-hex refinement: 480/480 children match; \
               exact coordinates {exact}/{}; max |Δ| = {max_diff:e}", 480 * 24);
}

#[test]
fn refine_uniform_3d_handles_curved_hex_with_quad_boundary() {
    // Before the fix this panicked: rebuild_3d_boundary looked up the new
    // quad-face centers by exact-bit coordinate match computed with a
    // different accumulation order (and, for curved geometry, with values
    // that are not recomputable averages at all). Now the hex path builds
    // the boundary topologically from the refinement's own maps.
    let parent = load_parent();
    let fine = refine_uniform_3d(&parent);
    let cpp = load_cpp_dump();
    let (_, max_diff) = compare_with_dump(&fine, &cpp);
    assert!(max_diff <= TOL);
    // Boundary: every quad boundary face is split into 4 (MFEM order).
    assert_eq!(fine.n_faces(), 4 * parent.n_faces());
}

#[test]
fn curved_geometry_is_refined_and_survives_repeated_refinement() {
    let parent = load_parent();
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&parent, &all);

    let geo = fine.geometry.as_ref().expect("refined mesh must carry geometry");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 27);
    assert_eq!(geo.conn.len(), fine.n_elems() as usize * 27);
    assert!(geo.coords.iter().all(|v| v.is_finite()));

    // Second refinement: the child geometry must again refine consistently.
    let all2: Vec<u32> = (0..fine.n_elems() as u32).collect();
    let (fine2, _, _) = refine_hex8_uniform(&fine, &all2);
    assert_eq!(fine2.n_elems(), 8 * fine.n_elems());
    let geo2 = fine2.geometry.as_ref().expect("level-2 geometry");
    assert_eq!(geo2.conn.len(), fine2.n_elems() as usize * 27);
    assert_eq!(geo2.n_nodes, geo2.coords.len() / 3);
}

#[test]
fn partial_refinement_keeps_straight_neighbors_consistent() {
    // Mark only element 0; its neighbors stay unrefined (identity children
    // in the geometry builder) and the mesh must remain valid with hanging
    // constraints on the split edges.
    let parent = load_parent();
    let (fine, constraints, _) = refine_hex8_uniform(&parent, &[0]);
    assert_eq!(fine.n_elems(), parent.n_elems() + 7);
    assert!(fine.geometry.is_some());
    // The 12 edges of element 0 are split → hanging midpoints.
    assert!(!constraints.is_empty());
}

// ─── D111: the refined Q2 geometry must be the parent's Q2 field ────────────
//
// Regression pins for defect D111: `refine_uniform_3d` on any order-2 hex mesh
// used to scramble the *interior* fine geometry dofs (MFEM `Geometry::
// Constants<CUBE>` vertex order was confused with the bitwise
// `(v&1, (v>>1)&1, (v>>2)&1)` encoding, both in the reference-point table of
// `q2_eval` and in the child-octant origin of `build_refined_hex_geometry`).
// The child geometry then no longer reproduced the parent's field:
// `data/cube.mesh -o 2 -rs 1` (an *affine* Q2 mesh!) had min det(J) = −1.32
// instead of +1.953125e-3, which is what `mesh-optimizer` reported as
// "The input mesh is inverted!".  The fine vertex coordinates were always
// correct (they are exact geometry-dof picks), which is why the corner-only
// checks above passed while the interior dofs were wrong.

/// MFEM `Geometry::Constants<CUBE>` vertex order, `[0,1]³`.
const MFEM_HEX_VERTS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Local index of the parent's body-center vertex inside child `k` (see the
/// child table in `refine_nonconforming_hex`).
const CHILD_BC_LOCAL: [usize; 8] = [6, 7, 4, 5, 2, 3, 0, 1];

/// Sample points inside the reference hex (`[-1,1]³`): the 27 Q2 dof points
/// plus the 8 sub-octant centers.
fn ref_samples() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for k in -1..=1 {
        for j in -1..=1 {
            for i in -1..=1 {
                v.push([i as f64, j as f64, k as f64]);
            }
        }
    }
    for k in [-0.5, 0.5] {
        for j in [-0.5, 0.5] {
            for i in [-0.5, 0.5] {
                v.push([i, j, k]);
            }
        }
    }
    v
}

/// Every refined child's Q2 geometry must reproduce the parent element's own
/// Q2 field on the child's reference domain (a degree-2 field restricted to a
/// half-cell is still degree 2, so this is exact, not an approximation).
///
/// The child→parent correspondence is recovered from the mesh alone: a child
/// has exactly one coarse-vertex corner, whose local index *is* the child
/// index, and its body-center corner is a node shared by exactly the 8
/// children of one parent whose vertex set is the children's corner set.
fn check_child_geometry_reproduces_parent(parent: &Mesh<3>, fine: &Mesh<3>, tol: f64) {
    let n_coarse = parent.n_nodes() as u32;
    let mut groups: std::collections::HashMap<u32, Vec<(usize, u32)>> =
        std::collections::HashMap::new();
    for fe in 0..fine.n_elems() as u32 {
        let vs = fine.elem_nodes(fe);
        let own: Vec<usize> = (0..8).filter(|&k| vs[k] < n_coarse).collect();
        assert_eq!(own.len(), 1, "child {fe} must have exactly one parent-vertex corner");
        let child = own[0];
        groups.entry(vs[CHILD_BC_LOCAL[child]]).or_default().push((child, fe));
    }
    assert_eq!(groups.len(), parent.n_elems(), "one child group per parent");

    let samples = ref_samples();
    let mut max_dev = 0.0_f64;
    for (_bc, kids) in &groups {
        assert_eq!(kids.len(), 8, "8 children per parent");
        // The children's own corners are the parent's 8 vertices.
        let mut corners: Vec<u32> = kids
            .iter()
            .map(|&(child, fe)| fine.elem_nodes(fe)[child])
            .collect();
        corners.sort_unstable();
        let pe = (0..parent.n_elems() as u32)
            .find(|&p| {
                let mut pv = parent.elem_nodes(p).to_vec();
                pv.sort_unstable();
                pv == corners
            })
            .expect("parent element with this vertex set");
        for &(child, fe) in kids {
            let c = MFEM_HEX_VERTS[child];
            for xi in &samples {
                let parent_xi = [
                    c[0] - 0.5 + 0.5 * xi[0],
                    c[1] - 0.5 + 0.5 * xi[1],
                    c[2] - 0.5 + 0.5 * xi[2],
                ];
                let (_, _, x_child) = fine.element_jacobian(fe, xi);
                let (_, _, x_parent) = parent.element_jacobian(pe, &parent_xi);
                for d in 0..3 {
                    max_dev = max_dev.max((x_child[d] - x_parent[d]).abs());
                }
            }
        }
    }
    assert!(
        max_dev <= tol,
        "refined Qk geometry deviates from the parent field by {max_dev:e} (tol {tol:e})"
    );
    eprintln!("child geometry reproduces the parent Qk field, max |Δ| = {max_dev:e}");
}

#[test]
fn d111_refined_curved_hex_geometry_reproduces_parent_field() {
    let parent = load_parent();
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&parent, &all);
    check_child_geometry_reproduces_parent(&parent, &fine, 1e-15);
    // Second level: the fine mesh is itself a valid order-2 parent now.
    let all2: Vec<u32> = (0..fine.n_elems() as u32).collect();
    let (fine2, _, _) = refine_hex8_uniform(&fine, &all2);
    check_child_geometry_reproduces_parent(&fine, &fine2, 1e-15);
}

#[test]
fn d111_affine_q2_cube_refines_to_exact_trilinear_children() {
    // `data/cube.mesh` is an affine 8-hex Q2 mesh of the unit cube: every
    // refined child's geometry must be *exactly* the trilinear (here affine)
    // map of its 8 vertices, and min det(J) must be > 0 at every level.
    let cube = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/cube.mesh");
    let f = read_mfem_file(cube).expect("read cube.mesh");
    let mut m: Mesh<3> = f.mesh3d.expect("3d");
    assert_eq!(m.geom_order(), 2);
    let samples = ref_samples();
    for (level, expected_min_det) in [(1usize, 1.953125e-3_f64), (2, 2.44140625e-4)] {
        m = refine_uniform_3d(&m);
        let mut min_det = f64::INFINITY;
        let mut max_dev = 0.0_f64;
        for e in 0..m.n_elems() as u32 {
            let vs: Vec<[f64; 3]> = m.elem_nodes(e).iter().map(|&n| m.coords_of(n)).collect();
            for xi in &samples {
                let (_, det, xp) = m.element_jacobian(e, xi);
                min_det = min_det.min(det);
                // trilinear map of the 8 vertices in MFEM order on [-1,1]³
                let mut want = [0.0_f64; 3];
                for k in 0..8 {
                    let r = [
                        2.0 * MFEM_HEX_VERTS[k][0] - 1.0,
                        2.0 * MFEM_HEX_VERTS[k][1] - 1.0,
                        2.0 * MFEM_HEX_VERTS[k][2] - 1.0,
                    ];
                    let w = 0.125
                        * (1.0 + r[0] * xi[0])
                        * (1.0 + r[1] * xi[1])
                        * (1.0 + r[2] * xi[2]);
                    for d in 0..3 {
                        want[d] += w * vs[k][d];
                    }
                }
                for d in 0..3 {
                    max_dev = max_dev.max((xp[d] - want[d]).abs());
                }
            }
        }
        assert!(min_det > 0.0, "level {level}: min det(J) = {min_det:e} (mesh inverted)");
        assert!(
            (min_det - expected_min_det).abs() < 1e-15,
            "level {level}: min det(J) = {min_det:e}, expected {expected_min_det:e}"
        );
        assert!(max_dev <= 1e-15, "level {level}: geometry is not trilinear (Δ = {max_dev:e})");
    }
}

// ─── D113: order-p (p >= 2) hex geometry transfer ───────────────────────────
//
// Before D113 `amr::curved_hex` only transported geometry when the parent was
// *exactly* order 2 with 27 dofs per hex; every higher-order mesh fell back to
// straight averaging and the refined mesh silently dropped to `geom_order 1`
// (`cube.mesh -o 3 -rs 1`).  The generalization drives the dof layout from
// `HexQk::new(p).dof_coords()` and evaluates the parent field, so the order is
// preserved for any p.
//
// Reference: serial MFEM 4.10.  The probe reads `data/cube.mesh`,
// `SetCurvature(p)`, optionally applies a smooth `Transform` (the `c*`
// fixtures), writes the resulting order-p mesh, refines it `refs` times and
// dumps the refined `nodes` grid function **per element, dof by dof, in the
// geometry FE's local dof order**, followed by the refined mesh's `vertices`:
//
// ```text
// wsl g++ -std=c++17 -O2 -I$HOME/mfem410_ser tmp/d113_ref.cpp \
//     $HOME/mfem410_ser/libmfem.a -o d113
// wsl ./d113 data/cube.mesh 3 1 cube_o3_mesh.txt cube_o3_r1_cpp.txt 0 0
// wsl ./d113 data/cube.mesh 3 2 cube_o3_mesh.txt cube_o3_r2_cpp.txt 0 64
// ```
//
// (the parent mesh is written before the refinement, and the fixtures are
// `.txt` because `.gitignore` covers `*.mesh`).  For `refs = 2` the dump
// covers only the first 64 of the `NE = 512` elements; the child index of
// element `e` is `e % 8`, so those 64 cover all eight octants.

/// MFEM reference: `(ne, order, dof, node coords per element dof, vertices)`.
struct GeometryRef {
    ne: usize,
    order: usize,
    dof: usize,
    ndofs: usize,
    /// Number of elements actually dumped (`<= ne`; the `refs = 2` fixtures
    /// carry the first 64 of 512 elements, which cover all eight octants).
    dumped: usize,
    nodes: Vec<[f64; 3]>,
    verts: Vec<[f64; 3]>,
}

fn load_geometry_ref(path: &str) -> GeometryRef {
    let text = std::fs::read_to_string(path).expect("read geometry reference");
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    assert_eq!(head[0], "NE", "reference header");
    let ne: usize = head[1].parse().unwrap();
    let order: usize = head[3].parse().unwrap();
    let dof: usize = head[5].parse().unwrap();
    let ndofs: usize = head[7].parse().unwrap();
    let dumped: usize = head[9].parse().unwrap();
    let mut nodes = Vec::with_capacity(dumped * dof);
    for _ in 0..dumped * dof {
        let v: Vec<f64> = lines
            .next()
            .expect("node row")
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        nodes.push([v[0], v[1], v[2]]);
    }
    let vhead: Vec<&str> = lines.next().expect("VERTS header").split_whitespace().collect();
    assert_eq!(vhead[0], "VERTS");
    let nv: usize = vhead[1].parse().unwrap();
    let mut verts = Vec::with_capacity(nv);
    for _ in 0..nv {
        let v: Vec<f64> = lines
            .next()
            .expect("vertex row")
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        verts.push([v[0], v[1], v[2]]);
    }
    GeometryRef { ne, order, dof, ndofs, dumped, nodes, verts }
}

/// Refine `parent` `refs` times and compare every refined element's geometry
/// dofs — and every refined vertex — against the MFEM dump.
///
/// Returns `(bit-exact node count, max |Δ|)`.
fn compare_geometry_with_ref(parent_path: &str, refs: usize, dump_path: &str) -> (usize, f64) {
    let f = read_mfem_file(parent_path).expect("read parent");
    let mut m: Mesh<3> = f.mesh3d.expect("3d parent");
    let reference = load_geometry_ref(dump_path);
    assert_eq!(m.geom_order() as usize, reference.order, "parent geometry order");

    for _ in 0..refs {
        m = refine_uniform_3d(&m);
    }

    let g = m.geometry.as_ref().expect("refined mesh must carry geometry");
    assert_eq!(g.order as usize, reference.order, "geom_order must survive refinement");
    assert_eq!(g.nodes_per_elem, reference.dof, "(p+1)^3 geometry dofs per element");
    assert_eq!(g.conn.len(), m.n_elems() as usize * reference.dof);
    // MFEM's own `nodes` space has exactly this many dofs: the edge/face
    // sharing keys reproduce its first-touch numbering one for one.
    assert_eq!(g.n_nodes, reference.ndofs, "geometry node count");
    assert_eq!(m.n_elems() as usize, reference.ne, "refined element count");

    let mut exact = 0usize;
    let mut max_diff = 0.0_f64;
    let ne = reference.dumped.min(m.n_elems() as usize);
    for e in 0..ne {
        for k in 0..reference.dof {
            let d = g.conn[e * reference.dof + k] as usize;
            let got = [g.coords[3 * d], g.coords[3 * d + 1], g.coords[3 * d + 2]];
            let want = reference.nodes[e * reference.dof + k];
            let mut dmax = 0.0_f64;
            for c in 0..3 {
                dmax = dmax.max((got[c] - want[c]).abs());
            }
            assert!(
                dmax <= 1e-14,
                "element {e} dof {k}: {got:?} != MFEM {want:?} (Δ = {dmax:e})"
            );
            if dmax == 0.0 { exact += 1; }
            max_diff = max_diff.max(dmax);
        }
    }
    assert_eq!(m.n_nodes() as usize, reference.verts.len(), "refined vertex count");
    // The fine *vertex numbering* is fem-rs's own (`refine_nonconforming_hex`
    // hands out new node ids in kernel order, MFEM in its own vertex order),
    // so match every refined vertex to some still-unmatched reference vertex
    // instead of comparing by index.
    let mut used = vec![false; reference.verts.len()];
    for v in 0..reference.verts.len() {
        let got = m.coords_of(v as u32);
        let mut best = f64::INFINITY;
        let mut hit = None;
        for (k, (&u, want)) in used.iter().zip(reference.verts.iter()).enumerate() {
            if u { continue; }
            let d = (got[0] - want[0]).abs().max((got[1] - want[1]).abs()).max((got[2] - want[2]).abs());
            if d < best { best = d; hit = Some(k); }
        }
        assert!(best <= 1e-14, "vertex {v} = {got:?}: no unmatched MFEM vertex within 1e-14");
        used[hit.expect("unused reference vertex")] = true;
        if best == 0.0 { exact += 1; }
        max_diff = max_diff.max(best);
    }
    (exact, max_diff)
}

const D113_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

#[test]
fn d113_order3_geometry_matches_mfem_dump() {
    for (refs, dump) in [(1usize, "cube_o3_r1_cpp.txt"), (2, "cube_o3_r2_cpp.txt")] {
        let parent = format!("{D113_DATA}/cube_o3_mesh.txt");
        let (exact, max_diff) =
            compare_geometry_with_ref(&parent, refs, &format!("{D113_DATA}/{dump}"));
        eprintln!("cube.mesh -o 3 -rs {refs}: max |Δ| = {max_diff:e}, exact coordinates {exact}");
    }
}

#[test]
fn d113_order4_geometry_matches_mfem_dump() {
    for (refs, dump) in [(1usize, "cube_o4_r1_cpp.txt"), (2, "cube_o4_r2_cpp.txt")] {
        let parent = format!("{D113_DATA}/cube_o4_mesh.txt");
        let (exact, max_diff) =
            compare_geometry_with_ref(&parent, refs, &format!("{D113_DATA}/{dump}"));
        eprintln!("cube.mesh -o 4 -rs {refs}: max |Δ| = {max_diff:e}, exact coordinates {exact}");
    }
}

#[test]
fn d113_order3_curved_geometry_matches_mfem_dump() {
    // A genuinely non-affine order-3 parent (the probe's smooth `Transform`),
    // so the transfer cannot be satisfied by a trilinear fallback.
    let parent = format!("{D113_DATA}/cube_o3c_mesh.txt");
    let (exact, max_diff) =
        compare_geometry_with_ref(&parent, 1, &format!("{D113_DATA}/cube_o3c_r1_cpp.txt"));
    let f = read_mfem_file(&parent).expect("read parent");
    let m: Mesh<3> = f.mesh3d.expect("3d");
    eprintln!("curved cube.mesh -o 3: max |Δ| = {max_diff:e}, exact coordinates {exact}");
    assert!(m.geometry.is_some());
}

/// `check_child_geometry_reproduces_parent` for an arbitrary order: refine the
/// order-`p` parent twice and require every child's geometry to reproduce the
/// parent field (a degree-`p` field restricted to a half-cell is still degree
/// `p`, so this is exact) with `min det J > 0` at every level.
fn check_qk_refinement_reproduces_parent(parent: &Mesh<3>, levels: usize) {
    let p = parent.geom_order();
    assert!(p >= 2, "curved parent expected");
    let mut levels = levels;
    let mut cur = parent.clone();
    while levels > 0 {
        let all: Vec<u32> = (0..cur.n_elems() as u32).collect();
        let fine = refine_uniform_3d(&cur);
        assert_eq!(fine.n_elems(), 8 * cur.n_elems(), "8 children per hex");
        let g = fine.geometry.as_ref().expect("geometry must survive refinement");
        assert_eq!(g.order, p, "geom_order {p} must survive refinement (got {})", g.order);
        assert_eq!(g.nodes_per_elem, (p as usize + 1).pow(3));
        check_child_geometry_reproduces_parent(&cur, &fine, 1e-14);
        let mut min_det = f64::INFINITY;
        for e in 0..fine.n_elems() {
            for xi in &ref_samples() {
                let (_, det, _) = fine.element_jacobian(e as u32, xi);
                min_det = min_det.min(det);
            }
        }
        assert!(min_det > 0.0, "level: min det(J) = {min_det:e} (refined mesh is inverted)");
        cur = fine;
        levels -= 1;
        let _ = all;
    }
}

#[test]
fn d113_qk3_and_qk4_children_reproduce_the_parent_field() {
    for name in ["cube_o3_mesh.txt", "cube_o3c_mesh.txt", "cube_o4_mesh.txt"] {
        let path = format!("{D113_DATA}/{name}");
        let f = read_mfem_file(&path).expect("read parent");
        let m: Mesh<3> = f.mesh3d.expect("3d");
        assert!(m.geometry.is_some(), "{name}: parent must carry curved geometry");
        check_qk_refinement_reproduces_parent(&m, 2);
        eprintln!("{name}: 2 levels of Q{}-geometry refinement reproduce the parent field", m.geom_order());
    }
}
