//! D43 acceptance tests: reading MFEM's high-order H1 `nodes` section on
//! tetrahedral meshes must reproduce MFEM's own DOF numbering.
//!
//! MFEM's `nodes` grid function is an H1 space whose dofs are laid out as
//! `[vertices | mesh-edge blocks | mesh-face blocks | interior blocks]`, with
//! the mesh edges/faces numbered by finalization (element traversal order,
//! `Geometry::TETRAHEDRON::Edges`/`FaceVert` local order, first encounter
//! wins) and each block internally oriented by the *global vertex ids*
//! (`Mesh::GetElementEdges`: `cor = v[e0] < v[e1] ? 1 : -1`) or by the first
//! element's face parameterisation re-keyed through MFEM's
//! `TriDofOrd[orientation]` (`H1_FECollection::DofOrderForOrientation`).
//! fem-rs's tet bases (`TetPk`) use a different slot order, so the file's dofs
//! cannot be handed over directly: before D43 the loader fell back to fem-rs's
//! own `DofManager` numbering, which silently gave 11 of the 42 elements of
//! `data/escher-p2.mesh` another element's edge dofs (the straight-line
//! vertices stayed correct, which is why it went unnoticed).
//!
//! Reference data: serial MFEM 4.10.  For every element the harness dumps, per
//! FES-local geometry slot, (a) the coordinate of the dof the file assigns to
//! it and (b) the physical point `ElementTransformation::Transform` returns at
//! that slot's FE node.  Harness: `tmp/gll_ref/dump_curved_tet.cpp`; its output
//! is checked in under `tests/data/` (regenerate with
//! `./dump_curved_tet <file.mesh> > <dump>`, synthetic meshes with
//! `./dump_curved_tet make <out.mesh> <p> <amp> <rev>` — a 2x1x1 cartesian
//! tet mesh, `amp` = 0 straight / 0.13 curved by a smooth sinusoid, `rev = 1`
//! reverses the vertex ids so every element-local edge runs from a larger to a
//! smaller mesh vertex id).
//!
//! Checks per element:
//!  * (bit-exact) the multiset of geometry dof coordinates addressed by the
//!    element's slots equals MFEM's — pins the global block numbering and the
//!    dof sharing between elements;
//!  * (<= 1e-12, only where the two node sets coincide or the element is
//!    straight) evaluating the fem-rs isoparametric map at each FE node
//!    reproduces MFEM's point for that slot — pins the slot permutation.
//!    For curved `p >= 3` meshes the map is *reported* but not asserted: the
//!    file's nodes sit at MFEM's Gauss-Lobatto *parametric* positions while
//!    fem-rs's tet geometry basis (`TetPk`) is equispaced, so the two
//!    interpolate the same node values with different basis functions (a
//!    separate, `crates/element`-side issue recorded in the D43 report).

use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};

const TOL: f64 = 1e-12;

/// One checked mesh: the `.mesh` file, its MFEM reference dump, a label, and
/// whether the isoparametric map can be compared (see the module docs).
struct Case {
    label: &'static str,
    mesh: &'static str,
    dump: &'static str,
    check_map: bool,
}

const CASES: &[Case] = &[
    Case {
        label: "escher-p2 (P2, 42 tets)",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/escher-p2.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_escher_p2_cpp.txt"),
        check_map: true,
    },
    Case {
        label: "cartesian 2x1x1 straight P3",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_p3_straight.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_p3_straight_cpp.txt"),
        check_map: true,
    },
    Case {
        label: "cartesian 2x1x1 straight P4",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_p4_straight.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_p4_straight_cpp.txt"),
        check_map: true,
    },
    Case {
        label: "cartesian 2x1x1 curved P3",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_p3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_p3_cpp.txt"),
        check_map: false,
    },
    Case {
        // Same as the curved P3 case, but with the vertex ids reversed
        // (`v -> NV-1-v`) so that every element-local edge runs from a larger
        // to a smaller vertex id, and the face parameterisations of the second
        // element of each face are non-trivial rotations/reflections of the
        // stored one.
        label: "cartesian 2x1x1 curved P3, reversed vertex ids",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_rev_p3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_rev_p3_cpp.txt"),
        check_map: false,
    },
];

/// Parsed `dump_curved_tet` output.
struct Dump {
    nv: usize,
    ne: usize,
    p: usize,
    ndofs: usize,
    npe: usize,
    /// FE node positions in the unit tetrahedron, per (MFEM) slot.
    node_ref: Vec<[f64; 3]>,
    /// Coordinate the file gives the dof of slot `i` of element `e`.
    dof_coord: Vec<Vec<[f64; 3]>>,
    /// `ElementTransformation` image of `node_ref[i]`.
    pt: Vec<Vec<[f64; 3]>>,
    /// MFEM's `nodes` grid function values, indexed by FES dof id.
    node_xyz: Vec<[f64; 3]>,
    /// Integer barycentric indices of each FE slot, in the labelling of the
    /// element's vertices sorted by global id (`(element, pattern)` is the
    /// physical identity of a slot).
    pat: Vec<Vec<[usize; 4]>>,
    /// MFEM's FES-local dof ids per element (its `GetElementDofs`).
    edof: Vec<Vec<usize>>,
}

/// Integer barycentric coordinates of the point `(x, y, z)` of the unit
/// tetrahedron, for the order-`p` equispaced `TetPk` slots (`p·λ` is integral
/// for every slot of that reference element).
fn tet_pattern(p: usize, x: f64, y: f64, z: f64) -> [usize; 4] {
    let lam = [1.0 - x - y - z, x, y, z];
    let mut out = [0usize; 4];
    for m in 0..4 {
        let t = p as f64 * lam[m];
        assert!((t - t.round()).abs() < 1e-9, "slot not on the integer grid: {lam:?}");
        out[m] = t.round() as usize;
    }
    out
}

fn parse_dump(path: &str) -> Dump {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut header: Option<Vec<usize>> = None;
    let mut node_ref: Vec<[f64; 3]> = Vec::new();
    let mut dof_coord: Vec<Vec<[f64; 3]>> = Vec::new();
    let mut pt: Vec<Vec<[f64; 3]>> = Vec::new();
    let mut node_xyz: Vec<[f64; 3]> = Vec::new();
    let mut pat: Vec<Vec<[usize; 4]>> = Vec::new();
    let mut edof: Vec<Vec<usize>> = Vec::new();
    let mut in_noderef = false;
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        match t[0] {
            "header" => {
                header = Some(t[1..].iter().map(|v| v.parse().expect("int")).collect());
                in_noderef = false;
            }
            "pat" => {
                let e: usize = t[1].parse().unwrap();
                let i: usize = t[2].parse().unwrap();
                assert!(!t[3].eq("NOTFOUND"), "harness could not recover a pattern");
                if pat.len() <= e {
                    pat.resize_with(e + 1, Vec::new);
                }
                assert_eq!(pat[e].len(), i, "pat rows must be in slot order");
                pat[e].push([
                    t[3].parse().unwrap(),
                    t[4].parse().unwrap(),
                    t[5].parse().unwrap(),
                    t[6].parse().unwrap(),
                ]);
                in_noderef = false;
            }
            "edof" => {
                let e: usize = t[1].parse().unwrap();
                if edof.len() <= e {
                    edof.resize_with(e + 1, Vec::new);
                }
                edof[e] = t[2..].iter().map(|v| v.parse().unwrap()).collect();
                in_noderef = false;
            }
            "node" => {
                let g: usize = t[1].parse().unwrap();
                assert_eq!(node_xyz.len(), g, "node rows must be in dof order");
                node_xyz.push([t[2].parse().unwrap(), t[3].parse().unwrap(), t[4].parse().unwrap()]);
                in_noderef = false;
            }
            "noderef" => {
                let n: usize = t[1].parse().expect("noderef count");
                node_ref.reserve(n);
                in_noderef = true;
            }
            "end" => break,
            "edge" | "face" => {}
            "dofcoord" => {
                let e: usize = t[1].parse().unwrap();
                let i: usize = t[2].parse().unwrap();
                if dof_coord.len() <= e {
                    dof_coord.resize_with(e + 1, Vec::new);
                }
                assert_eq!(dof_coord[e].len(), i, "dofcoord rows must be in slot order");
                dof_coord[e].push([t[3].parse().unwrap(), t[4].parse().unwrap(), t[5].parse().unwrap()]);
                in_noderef = false;
            }
            "pt" => {
                let e: usize = t[1].parse().unwrap();
                let i: usize = t[2].parse().unwrap();
                if pt.len() <= e {
                    pt.resize_with(e + 1, Vec::new);
                }
                assert_eq!(pt[e].len(), i, "pt rows must be in slot order");
                pt[e].push([t[3].parse().unwrap(), t[4].parse().unwrap(), t[5].parse().unwrap()]);
                in_noderef = false;
            }
            tok if in_noderef => {
                let i: usize = tok.parse().unwrap();
                assert_eq!(node_ref.len(), i, "noderef rows must be in slot order");
                node_ref.push([t[1].parse().unwrap(), t[2].parse().unwrap(), t[3].parse().unwrap()]);
            }
            other => panic!("unexpected dump token {other:?} in {path}"),
        }
    }
    let h = header.expect("dump header");
    let d = Dump {
        nv: h[0],
        ne: h[1],
        p: h[5],
        ndofs: h[6],
        npe: h[7],
        node_ref,
        dof_coord,
        pt,
        node_xyz,
        pat,
        edof,
    };
    assert_eq!(d.node_ref.len(), d.npe);
    assert_eq!(d.dof_coord.len(), d.ne);
    assert_eq!(d.pt.len(), d.ne);
    assert_eq!(d.pat.len(), d.ne);
    assert_eq!(d.edof.len(), d.ne);
    assert_eq!(d.node_xyz.len(), d.ndofs, "nodes grid function length");
    for e in 0..d.ne {
        assert_eq!(d.pat[e].len(), d.npe, "pat per element");
        assert_eq!(d.edof[e].len(), d.npe, "edof per element");
    }
    d
}

#[test]
fn curved_tet_nodes_match_mfem_slot_by_slot() {
    for case in CASES {
        let file = read_mfem_file(case.mesh).unwrap_or_else(|e| panic!("read {}: {e}", case.mesh));
        let mesh: Mesh<3> = file.mesh3d.unwrap_or_else(|| panic!("{} must be 3D", case.label));
        let d = parse_dump(case.dump);

        assert_eq!(mesh.n_nodes(), d.nv, "{}: vertex count", case.label);
        assert_eq!(mesh.n_elems(), d.ne, "{}: element count", case.label);

        let geom = mesh
            .geometry
            .as_ref()
            .unwrap_or_else(|| panic!("{}: D43 geometry must be read, not dropped", case.label));
        assert_eq!(geom.order as usize, d.p, "{}: geometry order", case.label);
        assert_eq!(geom.nodes_per_elem, d.npe, "{}: nodes per element", case.label);
        assert_eq!(geom.n_nodes, d.ndofs, "{}: geometry node count", case.label);

        let ref_elem = fem_element::lagrange::factory::TetPk::new(d.p);
        let ref_coords = ref_elem.dof_coords();
        assert_eq!(ref_coords.len(), d.npe);
        // fem-rs slot `j` <-> MFEM slot `i` whenever their physical patterns
        // (integer barycentric indices in the element's sorted-vertex
        // labelling) agree.  The two reference elements order the face block
        // differently (`TetPk` by the missing axes, MFEM by `TET_FACES`), so
        // an index-wise comparison would compare different reference points.
        let my_pat: Vec<[usize; 4]> = ref_coords
            .iter()
            .map(|c| tet_pattern(d.p, c[0], c[1], c[2]))
            .collect();

        // (0) bit-exact: the geometry node vector must equal MFEM's `nodes`
        // grid function dof by dof.  This pins the value renumbering that
        // MFEM's `refine = 1` (`DoNodeReorder`) applies to the file's dof
        // vector (the slot numbering itself is pinned by (b) below).
        let mut max_node = 0.0f64;
        for (g, want) in d.node_xyz.iter().enumerate() {
            let c = mesh.geom_coords_of(g as u32);
            for k in 0..3 {
                let diff = (c[k] - want[k]).abs();
                max_node = max_node.max(diff);
                assert!(
                    diff == 0.0,
                    "{}: geometry node {g} coord {k}: got {} want {} (diff {diff:.3e})",
                    case.label,
                    c[k],
                    want[k],
                );
            }
        }
        assert_eq!(max_node, 0.0, "{}: nodes must be read bit-exactly", case.label);

        let mut max_pick = 0.0f64;
        let mut max_pt = 0.0f64;
        for e in 0..d.ne as u32 {
            // (a) bit-exact: same multiset of geometry dof coordinates.
            let mut got: Vec<[f64; 3]> = mesh
                .geometry_nodes(e)
                .iter()
                .map(|&n| {
                    let c = mesh.geom_coords_of(n);
                    [c[0], c[1], c[2]]
                })
                .collect();
            let mut want = d.dof_coord[e as usize].clone();
            let key = |v: &[f64; 3]| (v[0], v[1], v[2]);
            got.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
            want.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
            for (g, w) in got.iter().zip(want.iter()) {
                for k in 0..3 {
                    max_pick = max_pick.max((g[k] - w[k]).abs());
                }
            }
            assert_eq!(got, want, "{}: element {e} dof multiset", case.label);

            let nodes = mesh.geometry_nodes(e);
            // Re-express our (element-local) slot patterns in the same
            // sorted-vertex labelling the harness used.
            let n4 = mesh.element_nodes(e);
            let mut loc = [0usize, 1, 2, 3];
            loc.sort_by_key(|&m| n4[m]);
            let sorted_pat: Vec<[usize; 4]> = my_pat
                .iter()
                .map(|q| [q[loc[0]], q[loc[1]], q[loc[2]], q[loc[3]]])
                .collect();
            for (i, pi) in d.pat[e as usize].iter().enumerate() {
                // (b) per physical slot: our pick must be the very dof MFEM's
                // `GetElementDofs` returns.
                let j = match sorted_pat.iter().position(|q| q == pi) {
                    Some(j) => j,
                    None => panic!("{}: element {e} has no slot with pattern {pi:?}", case.label),
                };
                assert_eq!(
                    nodes[j] as usize, d.edof[e as usize][i],
                    "{}: element {e} slot {i} (pattern {pi:?}, fem-rs slot {j}) dof",
                    case.label,
                );

                // (c) the isoparametric map at that slot's FE node reproduces
                // MFEM's point (asserted where the node sets and bases agree —
                // see the module docs).
                if !case.check_map {
                    continue;
                }
                let mut vals = vec![0.0f64; ref_elem.n_dofs()];
                ref_elem.eval_basis(&ref_coords[j], &mut vals);
                let mut x = [0.0f64; 3];
                for (k, &node) in nodes.iter().enumerate() {
                    let c = mesh.geom_coords_of(node);
                    for dim in 0..3 {
                        x[dim] += vals[k] * c[dim];
                    }
                }
                for dim in 0..3 {
                    let diff = (x[dim] - d.pt[e as usize][i][dim]).abs();
                    max_pt = max_pt.max(diff);
                    assert!(
                        diff <= TOL,
                        "{}: element {e} slot {i} coord {dim}: got {} want {} (diff {diff:.3e})",
                        case.label,
                        x[dim],
                        d.pt[e as usize][i][dim],
                    );
                }
            }
        }
        eprintln!(
            "{}: max |Δ| dof pick = {max_pick:.3e}{}",
            case.label,
            if case.check_map { format!(", max |Δ| map = {max_pt:.3e}") } else { String::new() }
        );
        assert_eq!(max_pick, 0.0, "{}: geometry dof picks must be bit-exact", case.label);
    }
}

/// Report (without asserting) how far the fem-rs isoparametric map is from
/// MFEM's on the curved P3 meshes: the file's nodes are at Gauss-Lobatto
/// parametric positions while fem-rs's `TetPk` is equispaced, so the maps
/// differ by an amount that does not vanish with refinement.
#[test]
fn curved_tet_p3_map_gap_is_reported() {
    for case in CASES.iter().filter(|c| !c.check_map) {
        let file = read_mfem_file(case.mesh).unwrap();
        let mesh: Mesh<3> = file.mesh3d.unwrap();
        let d = parse_dump(case.dump);
        let ref_elem = fem_element::lagrange::factory::TetPk::new(d.p);
        let mut vals = vec![0.0f64; ref_elem.n_dofs()];
        let mut max_pt = 0.0f64;
        for e in 0..d.ne as u32 {
            let nodes = mesh.geometry_nodes(e);
            for (i, nr) in d.node_ref.iter().enumerate() {
                ref_elem.eval_basis(nr, &mut vals);
                let mut x = [0.0f64; 3];
                for (k, &node) in nodes.iter().enumerate() {
                    let c = mesh.geom_coords_of(node);
                    for dim in 0..3 {
                        x[dim] += vals[k] * c[dim];
                    }
                }
                for dim in 0..3 {
                    max_pt = max_pt.max((x[dim] - d.pt[e as usize][i][dim]).abs());
                }
            }
        }
        eprintln!("{}: isoparametric map gap vs MFEM = {max_pt:.3e}", case.label);
    }
}

/// The reference element slots must cover every mesh entity: every geometry
/// node is referenced and the table has exactly one node per slot.
#[test]
fn curved_tet_geometry_conn_is_a_surjective_entity_map() {
    let file = read_mfem_file(CASES[1].mesh).unwrap();
    let mesh: Mesh<3> = file.mesh3d.unwrap();
    let geom = mesh.geometry.as_ref().unwrap();
    let mut used = vec![false; geom.n_nodes];
    for &n in geom.conn.iter() {
        used[n as usize] = true;
    }
    assert!(used.iter().all(|&u| u), "every geometry node must be referenced");
    assert_eq!(geom.conn.len(), mesh.n_elems() * geom.nodes_per_elem);
}
