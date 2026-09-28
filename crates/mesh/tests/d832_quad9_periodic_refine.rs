//! D832-1 (round 88 Lane B): uniform refinement propagation of a periodic
//! Quad9 mesh — the D820-3 leftover "细化传播未对拍".
//!
//! Fixture: the round-87 3×3 straight quad9 torus [0,3]²
//! (`data/d820_quad9_periodic_square.msh`), folded by `make_periodic` into
//! the 36-node quotient mesh pinned by `crates/space/tests/d820_quad9_periodic.rs`.
//! One uniform refinement must carry that quotient to the 6×6 torus:
//!
//! ```text
//! fine torus complex: V = 36, E = 72, C = 36   (V − E + C = 0)
//! H1(1) = 36            H1(2) = 36 + 72 + 36  = 144
//! H1(3) = 36 + 2·72 + 4·36 = 324            (×4 again: 144/576/1296)
//! fine quad9 node set (geometry table) = V + E + C = 144
//! ```
//!
//! # MFEM 4.10 ground truth (`tmp/d88b/d832_refine_periodic_probe.cpp`)
//!
//! On the corrected fixture (D820-3 fixture fix: Gmsh-spec Line3 node order)
//! MFEM is the numeric oracle in **both** orders:
//!
//! * refine→fold: refined mesh NE=36, NV=49, nodes stay `H1_2D_P2` (169
//!   dofs); `MakePeriodic` converts the nodes to `L2_T1_2D_P2` with
//!   `NDofs = 36·9 = 324` and gives H1(1/2/3) = **36/144/324**.
//! * fold→refine: periodic mesh NV=9 (H1 9/36/81); after `UniformRefinement`
//!   NE=36, **NV=36**, nodes `L2_T1_2D_P2` **324** per-element own-side rows,
//!   H1(1/2/3) = **36/144/324**, and the refined nodes cover exactly **144**
//!   distinct positions on the torus (quarter lattice, off-lattice 0) —
//!   the D809-2 semantics (4-vertex SQUARE children, order-2 nodes riding
//!   through) composed with the periodic fold without loss.
//!
//! fem-rs after the fixes below matches every one of those numbers, with the
//! mesh-level fold (fem-rs merges the mesh nodes where MFEM merges space
//! dofs) carried through the refinement.
//!
//! # The defects this pins shut (red before, stash-verified)
//!
//! 1. `linear_view` (the Quad9/Hex27/… → Quad4/Hex8/… route under
//!    `refine_uniform`) carried the *whole* source vertex table over, so the
//!    coarse rows' non-corner nodes (edge mids / centers, unreferenced by the
//!    corner-only view) leaked into the refined mesh as unreferenced vertex
//!    entries that still counted as mesh vertices — and hence as H1(1) dofs:
//!    63 vertices instead of 36, H1(2) 171 instead of 144, and
//!    `build_refined_quad_geometry` cloned them into the child geometry
//!    table's vertex block.  The view now compacts its vertex table to the
//!    referenced corner set (the dropped row nodes survive in the
//!    self-contained `GeometryData`), matching MFEM's refined NV = coarse
//!    corner count.
//! 2. On a periodically merged mesh the shared-dof refinement geometry built
//!    every seam-adjacent child row in the **folded (wrapped) vertex frame**:
//!    the vertex block of the geometry table is the fine mesh's folded vertex
//!    table, whose corner at a seam holds only one side's image, so the rows
//!    stopped being coherent isoparametric cells (child row corner-shoelace
//!    areas 0.5 / 1.25 / 3.25 instead of 0.25).  Periodically merged parents
//!    now take MFEM's `MakePeriodic` semantics — fully per-element own-side
//!    rows (`L2_T1` nodes, `NDofs = 36·9 = 324` in the probe), each the
//!    prolongation of that element's own coherent row.

use fem_io::gmsh::read_msh;
use fem_mesh::{ElementType, Mesh, MeshTopology};
use fem_space::{FESpace, H1Space};

const FIXTURE: &str = include_str!("../../../data/d820_quad9_periodic_square.msh");

fn periodic_quad9_mesh() -> Mesh<2> {
    read_msh(FIXTURE.as_bytes())
        .expect("read the gmsh quad9 fixture")
        .into_2d()
        .expect("2-D mesh")
        .make_periodic(&[(2, 1, [-3.0, 0.0]), (4, 3, [0.0, -3.0])], 1e-10)
        .expect("make periodic")
}

/// Distinct undirected edges of the quad corner connectivity.
fn quad_edge_count(mesh: &Mesh<2>) -> usize {
    let mut edges = std::collections::BTreeSet::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for k in 0..4 {
            let (a, b) = (ns[k], ns[(k + 1) % 4]);
            edges.insert((a.min(b), a.max(b)));
        }
    }
    edges.len()
}

/// Number of nodes referenced by at least one element (the MFEM refined-NV
/// view of the mesh).
fn referenced_nodes(mesh: &Mesh<2>) -> usize {
    let mut used = std::collections::BTreeSet::<u32>::new();
    for e in 0..mesh.n_elems() as u32 {
        used.extend(mesh.elem_nodes(e));
    }
    used.len()
}

/// Distance of a coordinate to the `1/ppu` lattice (the refinement of the
/// half-integer coarse torus lives on multiples of 1/4 after one and 1/8
/// after two refinements — vertices first, then the geometry dofs).
fn off_lattice(v: f64, ppu: i64) -> f64 {
    (v * ppu as f64 - (v * ppu as f64).round()).abs() / ppu as f64
}

/// Distinct positions of a coordinate table modulo the [0,3)² period — the
/// torus-quotient position count on the `1/ppu` lattice.
fn distinct_torus_positions(coords: &[f64], ppu: i64) -> usize {
    let mut keys = std::collections::BTreeSet::new();
    for c in coords.chunks_exact(2) {
        let kx = (c[0] * ppu as f64).round() as i64;
        let ky = (c[1] * ppu as f64).round() as i64;
        keys.insert((kx.rem_euclid(3 * ppu), ky.rem_euclid(3 * ppu)));
    }
    keys.len()
}

/// Mesh-level invariant: the refined torus is compact, connected through the
/// fold, and its geometry table is the fine torus node set (all assertions
/// independent of any space build — the round-87 lesson).
#[test]
fn d832_refined_torus_mesh_is_compact_and_periodicity_aware() {
    let coarse = periodic_quad9_mesh();
    assert_eq!(coarse.n_nodes(), 36);
    assert_eq!(coarse.element_type(0), ElementType::Quad9);
    assert_eq!(quad_edge_count(&coarse), 18, "coarse torus E");

    let fine = fem_mesh::refine_uniform(&coarse);
    assert_eq!(fine.n_elems(), 36, "3×3 → 6×6 cells");
    assert_eq!(fine.element_type(0), ElementType::Quad4, "linear children");
    assert_eq!(
        fine.n_nodes(),
        36,
        "fine torus V — stale coarse row nodes must not survive in the vertex table"
    );
    assert_eq!(referenced_nodes(&fine), fine.n_nodes(), "no orphan vertices");
    for e in 0..fine.n_elems() as u32 {
        for &n in fine.elem_nodes(e) {
            assert!(
                (n as usize) < fine.n_nodes(),
                "element {e} references node {n} ≥ n_nodes"
            );
        }
    }
    assert_eq!(quad_edge_count(&fine), 72, "fine torus E");
    // Euler characteristic of the torus quotient.
    assert_eq!(
        fine.n_nodes() as isize - quad_edge_count(&fine) as isize + fine.n_elems() as isize,
        0,
        "V − E + C = 0 on the torus"
    );
    // Every child is non-degenerate (four distinct corner positions).
    for e in 0..fine.n_elems() as u32 {
        let mut corners = std::collections::BTreeSet::new();
        for &n in fine.elem_nodes(e) {
            let c = fine.coords_of(n);
            corners.insert((c[0].to_bits(), c[1].to_bits()));
        }
        assert_eq!(corners.len(), 4, "element {e} has coincident corners");
    }
    // Vertex coordinates sit on the quarter lattice inside [0,3].
    for i in 0..fine.n_nodes() {
        let c = fine.coords_of(i as u32);
        for &v in c.iter() {
            assert!(v >= -1e-12 && v <= 3.0 + 1e-12, "vertex coord {v} out of [0,3]");
            assert!(off_lattice(v, 4) < 1e-12, "vertex coord {v} off lattice");
        }
    }
    assert_eq!(
        distinct_torus_positions(&fine.coords, 4),
        36,
        "vertex positions on the torus"
    );

    // The transported order-2 geometry: MFEM `MakePeriodic` semantics — fully
    // per-element own-side rows (the probe's refined periodic nodes are
    // `L2_T1_2D_P2` with `NDofs = 36·9 = 324`), each row the prolongation of
    // that element's own coherent row.
    let geo = fine.geometry.as_ref().expect("refined mesh carries geometry");
    assert_eq!(geo.order, 2, "row geometry survives the refinement");
    assert_eq!(geo.nodes_per_elem, 9);
    assert_eq!(geo.conn.len(), 36 * 9, "one row per child");
    assert_eq!(
        geo.n_nodes,
        36 * 9,
        "per-element own-side rows — the folded vertex frame would wrap the seam cells"
    );
    for (i, &n) in geo.conn.iter().enumerate() {
        assert!(
            (n as usize) < geo.n_nodes,
            "geometry row entry {i} references {n} ≥ table size"
        );
    }
    for &v in &geo.coords {
        assert!(v >= -1e-12 && v <= 3.0 + 1e-12, "geometry coord {v} out of [0,3]");
        assert!(off_lattice(v, 4) < 1e-12, "geometry coord {v} off lattice");
    }
    assert_eq!(
        distinct_torus_positions(&geo.coords, 4),
        144,
        "geometry dofs cover the fine torus node set, one per position mod 3"
    );
    // The seam keeps both sides: some geometry dof sits at exactly 0 and some
    // at exactly 3 in the first coordinate (a fold that collapsed the sides
    // would leave only one of the two).
    let xs: Vec<f64> = geo.coords.chunks_exact(2).map(|c| c[0]).collect();
    assert!(xs.iter().any(|&x| x <= 1e-12), "left side present");
    assert!(xs.iter().any(|&x| x >= 3.0 - 1e-12), "right side present");

    // Functional pin: every child's isoparametric cell is coherent — the
    // geometry map integrates to exactly 1/4 of the coarse cell area (0.25),
    // so the whole torus measures 9.  The wrapped-frame rows this replaces
    // measured 0.5 / 1.25 / 3.25 on the seam children.
    let rule = fem_element::quadrature::quad_rule_01(6);
    let mut total = 0.0_f64;
    for e in 0..fine.n_elems() as u32 {
        let mut area = 0.0_f64;
        for (xi, w) in rule.points.iter().zip(rule.weights.iter()) {
            let (_jac, det, _x) = fine.element_jacobian(e, xi);
            area += w * det.abs();
        }
        assert!(
            (area - 0.25).abs() < 1e-12,
            "element {e}: refined cell area {area} ≠ 0.25"
        );
        total += area;
    }
    assert!((total - 9.0).abs() < 1e-12, "torus area {total} ≠ 9");
}

/// The refinement propagates the quotient to the dof level: H1 counts equal
/// the fine torus complex, no orphan dofs, and the doubly-wrap-around blocks
/// share a corner dof.
#[test]
fn d832_refined_torus_h1_counts_match_complex() {
    let fine = fem_mesh::refine_uniform(&periodic_quad9_mesh());
    let counts: Vec<(u8, usize)> = [1u8, 2, 3]
        .iter()
        .map(|&o| (o, H1Space::new(fine.clone(), o).n_dofs()))
        .collect();
    assert_eq!(counts[0], (1, 36), "H1(1) = V");
    assert_eq!(counts[1], (2, 144), "H1(2) = V + E + C");
    assert_eq!(counts[2], (3, 324), "H1(3) = V + 2E + 4C");

    let space = H1Space::new(fine.clone(), 2);
    let mut used = std::collections::BTreeSet::new();
    for e in 0..fine.n_elems() as u32 {
        for &d in space.element_dofs(e) {
            assert!(d < space.n_dofs() as u32, "dof {d} out of range");
            used.insert(d);
        }
    }
    assert_eq!(used.len(), space.n_dofs(), "orphan dofs in the refined quotient");
    // Seam: the folded corner (kept on the (3,3) side by make_periodic) is
    // element 0's first node.  On the 6×6 torus exactly 4 children — one from
    // each of the coarse blocks (0,0), (1,0)/(0,1) reachable only through one
    // fold and (1,1) only through both — meet there; on an unfolded mesh the
    // corresponding corner would have valence 1.
    let corner = fine.elem_nodes(0)[0];
    let holders: Vec<u32> = (0..fine.n_elems() as u32)
        .filter(|&e| fine.elem_nodes(e).contains(&corner))
        .collect();
    assert_eq!(holders.len(), 4, "folded corner valence at mesh level");
    let corner_dof = space.element_dofs(0)[0];
    let dof_holders = (0..fine.n_elems() as u32)
        .filter(|&e| space.element_dofs(e).contains(&corner_dof))
        .count();
    assert_eq!(dof_holders, 4, "folded corner dof shared by all four children");
}

/// Iterating the refinement keeps the quotient compact (12×12 torus:
/// V = 144, E = 288, C = 144).
#[test]
fn d832_second_refinement_stays_compact() {
    let fine = fem_mesh::refine_uniform(&periodic_quad9_mesh());
    let fine2 = fem_mesh::refine_uniform(&fine);
    assert_eq!(fine2.n_elems(), 144);
    assert_eq!(fine2.n_nodes(), 144, "12×12 torus V");
    assert_eq!(referenced_nodes(&fine2), fine2.n_nodes());
    assert_eq!(quad_edge_count(&fine2), 288, "12×12 torus E");
    let geo = fine2.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.conn.len(), 144 * 9);
    assert_eq!(geo.n_nodes, 144 * 9, "per-element own-side rows");
    // Vertices on the quarter lattice, geometry dofs (edge mids / centers of
    // the 0.25 cells) on the eighth lattice — all distinct mod 3.
    assert_eq!(distinct_torus_positions(&fine2.coords, 4), 144);
    assert_eq!(distinct_torus_positions(&geo.coords, 8), 576);
    // Coherence carries through the second refinement: every cell measures
    // 1/16, the whole torus still 9.
    let rule = fem_element::quadrature::quad_rule_01(6);
    let mut total = 0.0_f64;
    for e in 0..fine2.n_elems() as u32 {
        let mut area = 0.0_f64;
        for (xi, w) in rule.points.iter().zip(rule.weights.iter()) {
            let (_jac, det, _x) = fine2.element_jacobian(e, xi);
            area += w * det.abs();
        }
        assert!((area - 0.0625).abs() < 1e-12, "element {e}: area {area} ≠ 1/16");
        total += area;
    }
    assert!((total - 9.0).abs() < 1e-12, "torus area {total} ≠ 9");
    let counts: Vec<usize> = [1u8, 2, 3]
        .iter()
        .map(|&o| H1Space::new(fine2.clone(), o).n_dofs())
        .collect();
    assert_eq!(counts, vec![144, 576, 1296], "H1 on the 12×12 torus");
}
