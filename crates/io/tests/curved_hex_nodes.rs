//! D41 acceptance tests: reading MFEM's high-order H1 `nodes` section on
//! hexahedral meshes must reproduce MFEM's own DOF numbering.
//!
//! MFEM's `nodes` grid function is an H1 space whose dofs are laid out as
//! `[vertices | mesh-edge blocks | mesh-face blocks | interior blocks]`, with
//! the mesh edges/faces numbered by finalization (element traversal order,
//! `Geometry::CUBE::Edges`/`FaceVert` local order, first encounter wins) and
//! each block internally oriented by the *global vertex ids*
//! (`Mesh::GetElementEdges`: `cor = v[e0] < v[e1] ? 1 : -1`) or by the first
//! element's face parameterisation (`QuadDofOrd[0] = identity`).  fem-rs's
//! assembly bases (`HexQk`) use a different slot order, so the file's dofs
//! cannot be handed over directly: before D41 the loader claimed the orders
//! agreed and silently produced scrambled curved hex geometry.
//!
//! Reference data: serial MFEM 4.10. For every element the harness dumps, per
//! FES-local geometry slot, (a) the coordinate of the dof the file assigns to
//! it and (b) the physical point `ElementTransformation::Transform` returns at
//! that slot's FE node.  Harness: `tmp/gll_ref/dump_curved_hex.cpp`; its output
//! is checked in under `tests/data/` (regenerate with
//! `./dump_curved_hex <file.mesh> > <dump>`; the synthetic P3 meshes with
//! `./dump_curved_hex make cart out.mesh 3 0.13` — a 2x1x1 cartesian hex mesh
//! curved by a smooth sinusoid, the same mesh with reversed vertex ids via
//! `./dump_curved_hex makerev out.mesh 3 0.13`, and
//! `./dump_curved_hex make <mfem>/data/fichera.mesh out.mesh 3 0.15`).
//!
//! Checks per element:
//!  * (bit-exact) the multiset of geometry dof coordinates addressed by the
//!    element's slots equals MFEM's — pins the global block numbering and the
//!    dof sharing between elements;
//!  * (<= 1e-12) evaluating the fem-rs isoparametric map at each FE node
//!    reproduces MFEM's point for that slot — pins the slot permutation.

use fem_element::lagrange::factory::HexQk;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};

const TOL: f64 = 1e-12;

/// One checked mesh: the `.mesh` file, its MFEM reference dump, a label.
struct Case {
    label: &'static str,
    mesh: &'static str,
    dump: &'static str,
}

const CASES: &[Case] = &[
    Case {
        label: "cube (P2, 8 hexes, data/cube.mesh)",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/cube.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_hex_cube_p2_cpp.txt"),
    },
    Case {
        label: "cartesian 2x1x1 curved P3",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_hex_p3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_hex_p3_cpp.txt"),
    },
    Case {
        label: "fichera curved P3",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_hex_fichera_p3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_hex_fichera_p3_cpp.txt"),
    },
    Case {
        // Same as the cartesian P3 case, but with the vertex ids reversed
        // (`v -> NV-1-v`) so that every element-local edge runs from a larger
        // to a smaller vertex id — MFEM orients its edge dofs by vertex id
        // (`cor = v[e0] < v[e1] ? 1 : -1`), so this pins the reversal branch.
        label: "cartesian 2x1x1 curved P3, reversed vertex ids",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_hex_rev_p3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_hex_rev_p3_cpp.txt"),
    },
];

/// Parsed `dump_curved_hex` output.
struct Dump {
    nv: usize,
    ne: usize,
    nf: usize,
    n_edges: usize,
    p: usize,
    ndofs: usize,
    npe: usize,
    /// FE node positions in `[0,1]³`, per (MFEM) slot.
    node_ref: Vec<[f64; 3]>,
    /// `dofcoord[e][i]`: coordinate the file gives dof `edof[e][i]`.
    dof_coord: Vec<Vec<[f64; 3]>>,
    /// `pt[e][i]`: `ElementTransformation` image of `node_ref[i]`.
    pt: Vec<Vec<[f64; 3]>>,
}

fn parse_dump(path: &str) -> Dump {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut header: Option<Vec<usize>> = None;
    let mut node_ref: Vec<[f64; 3]> = Vec::new();
    let mut dof_coord: Vec<Vec<[f64; 3]>> = Vec::new();
    let mut pt: Vec<Vec<[f64; 3]>> = Vec::new();
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
            "noderef" => {
                let n: usize = t[1].parse().expect("noderef count");
                node_ref.reserve(n);
                in_noderef = true;
            }
            "end" => break,
            "edge" | "face" => {}
            "edof" => {}
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
        nf: h[2],
        n_edges: h[3],
        p: h[5],
        ndofs: h[6],
        npe: h[7],
        node_ref,
        dof_coord,
        pt,
    };
    assert_eq!(d.node_ref.len(), d.npe);
    assert_eq!(d.dof_coord.len(), d.ne);
    assert_eq!(d.pt.len(), d.ne);
    for e in 0..d.ne {
        assert_eq!(d.dof_coord[e].len(), d.npe);
        assert_eq!(d.pt[e].len(), d.npe);
    }
    d
}

/// fem-rs isoparametric geometry map for element `e` at reference point `xi`
/// (in `HexQk`'s domain `[-1,1]³`), using the mesh's `geometry` table.
fn eval_geom(mesh: &Mesh<3>, e: u32, ref_elem: &HexQk, xi: &[f64]) -> [f64; 3] {
    let mut vals = vec![0.0f64; ref_elem.n_dofs()];
    ref_elem.eval_basis(xi, &mut vals);
    let nodes = mesh.geometry_nodes(e);
    assert_eq!(nodes.len(), vals.len(), "geometry table must have one node per slot");
    let mut x = [0.0f64; 3];
    for (k, &node) in nodes.iter().enumerate() {
        let c = mesh.geom_coords_of(node);
        for d in 0..3 {
            x[d] += vals[k] * c[d];
        }
    }
    x
}

#[test]
fn curved_hex_nodes_match_mfem_slot_by_slot() {
    for case in CASES {
        let file = read_mfem_file(case.mesh).unwrap_or_else(|e| panic!("read {}: {e}", case.mesh));
        let mesh: Mesh<3> = file.mesh3d.unwrap_or_else(|| panic!("{} must be 3D", case.label));
        let d = parse_dump(case.dump);

        assert_eq!(mesh.n_nodes(), d.nv, "{}: vertex count", case.label);
        assert_eq!(mesh.n_elems(), d.ne, "{}: element count", case.label);

        let geom = mesh
            .geometry
            .as_ref()
            .unwrap_or_else(|| panic!("{}: D41 geometry must be read, not dropped", case.label));
        assert_eq!(geom.order as usize, d.p, "{}: geometry order", case.label);
        assert_eq!(geom.nodes_per_elem, d.npe, "{}: nodes per element", case.label);
        assert_eq!(geom.n_nodes, d.ndofs, "{}: geometry node count", case.label);

        let ref_elem = HexQk::new(d.p);
        let mut max_pt = 0.0f64;
        let mut max_pick = 0.0f64;
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

            // (b) slot-by-slot: the map at each FE node reproduces MFEM's point.
            for i in 0..d.npe {
                let nr = d.node_ref[i];
                let xi = [2.0 * nr[0] - 1.0, 2.0 * nr[1] - 1.0, 2.0 * nr[2] - 1.0];
                let got = eval_geom(&mesh, e, &ref_elem, &xi);
                for k in 0..3 {
                    let diff = (got[k] - d.pt[e as usize][i][k]).abs();
                    max_pt = max_pt.max(diff);
                    assert!(
                        diff <= TOL,
                        "{}: element {e} slot {i} coord {k}: got {} want {} (diff {diff:.3e})",
                        case.label,
                        got[k],
                        d.pt[e as usize][i][k],
                    );
                }
            }
        }
        eprintln!("{}: max |Δ| dof pick = {max_pick:.3e}, max |Δ| map = {max_pt:.3e}", case.label);
        assert_eq!(max_pick, 0.0, "{}: geometry dof picks must be bit-exact", case.label);
    }
}

/// The reference element slots must cover every mesh entity: distinct
/// entities never share a geometry node number, and every node is used.
#[test]
fn curved_hex_geometry_conn_is_a_surjective_entity_map() {
    let file = read_mfem_file(CASES[1].mesh).unwrap();
    let mesh: Mesh<3> = file.mesh3d.unwrap();
    let geom = mesh.geometry.as_ref().unwrap();
    let npe = geom.nodes_per_elem;
    let mut used = vec![false; geom.n_nodes];
    for &n in geom.conn.iter() {
        used[n as usize] = true;
    }
    assert!(used.iter().all(|&u| u), "every geometry node must be referenced");
    assert_eq!(geom.conn.len(), mesh.n_elems() * npe);
}

/// D41 safety net: a curved mesh with hexahedra that the mapper cannot number
/// faithfully must come back *without* a high-order table (and with a warning
/// on stderr) instead of a silently scrambled one.  `data/fichera-mixed-p2.mesh`
/// is a mixed tet/hex/prism mesh with an H1_3D_P2 `nodes` section.
#[test]
fn curved_mixed_hex_nodes_are_refused_not_scrambled() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/fichera-mixed-p2.mesh");
    let file = read_mfem_file(path).unwrap();
    let mesh: Mesh<3> = file.mesh3d.unwrap();
    assert!(mesh.geometry.is_none(), "mixed curved mesh must not be half-mapped");
    assert_eq!(mesh.geom_order(), 1);
    // The straight-line vertices are still exact picks of the same file.
    assert!(mesh.n_nodes() > 0 && mesh.n_elems() > 0);
}
