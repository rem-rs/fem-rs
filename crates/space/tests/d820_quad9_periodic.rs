//! D820-3 (round 87): periodic identification on Quad9 (Gmsh type-10) row
//! meshes — the round-84 leftover "周期×Quad8/9 未对拍".
//!
//! Fixture: 3×3 straight quad9 torus [0,3]² (`Gmsh` type-10 rows, `Line3`
//! boundary, tags 1=left 2=right 3=bottom 4=top).
//!
//! # MFEM 4.10 ground truth (`tmp/d87main/d820_periodic_quad9_probe.cpp`)
//!
//! MFEM cannot serve as the numeric oracle here: `Mesh::MakePeriodic` folds
//! **corner vertices only** (`CreatePeriodicVertexMapping` collects
//! `GetBdrElementVertices` of the boundary elements — the row midsides never
//! merge), and its two-translation composition misses the all-corner group
//! (the (3,3) image stays unmerged: 12 of 13 boundary vertices fold).  Its
//! H1(2) therefore counts 45 on this mesh where the torus complex has 36.
//! Same limitation family as D61 ("MFEM cannot build <3-cell directions at
//! all") — **upstream-report candidate**, the pins below are the
//! topological ground truth of the quotient complex:
//!
//! ```text
//! 3×3 quad torus: V = 9, E = 18, C = 9   (V − E + F = 0)
//! H1(1) = 9            H1(2) = 9 + 18 + 9·1 = 36
//! H1(3) = 9 + 2·18 + 4·9 = 81
//! ```
//!
//! # The two fem-rs defects this pins shut (both red before, stash-verified)
//!
//! 1. `Mesh::make_periodic` (crates/mesh): a node replicated through a
//!    *chain* of pairs ((0,0) → (3,0) by the x pair, (3,0) → (3,3) by the y
//!    pair) left `remap` uncompressed — the compact-numbering pass read
//!    `new_id[target]` before the target was resolved and wrote `u32::MAX`
//!    into the element connectivity.  The connectivity must be
//!    chain-free (asserted below); the space build alone cannot see this
//!    because it reads the pre-merge geometry snapshot through the
//!    unfolded wrapper.
//! 2. `DofManager::build_periodic` (crates/space): the quotient relabelling
//!    assumed covering vertex dofs are addressed by node id — true for
//!    corner meshes, false for row meshes whose builders compact a corner
//!    view (order 1: 16 dofs over 49 nodes → index panic; order 2: edge
//!    dofs below `n_uf_nodes` escaped the singleton window as "unlabelled
//!    DOF 40").  Now order-aware: node-addressed builds keep the fold-value
    //! numbering, compacted builds group by fold target and the entity
//! classes start at the final vertex count.

use fem_io::gmsh::read_msh;
use fem_mesh::MeshTopology;
use fem_space::{FESpace, H1Space};

const FIXTURE: &str = include_str!("../../../data/d820_quad9_periodic_square.msh");

fn periodic_quad9_mesh() -> fem_mesh::Mesh<2> {
    read_msh(FIXTURE.as_bytes())
        .expect("read the gmsh quad9 fixture")
        .into_2d()
        .expect("2-D mesh")
        .make_periodic(&[(2, 1, [-3.0, 0.0]), (4, 3, [0.0, -3.0])], 1e-10)
        .expect("make periodic")
}

/// Mesh-level invariant: the folded connectivity is chain-free (defect 1).
#[test]
fn d820_quad9_periodic_connectivity_chain_free() {
    let mesh = periodic_quad9_mesh();
    assert_eq!(mesh.n_elements(), 9);
    assert_eq!(mesh.element_type(0), fem_mesh::ElementType::Quad9);
    assert_eq!(mesh.n_nodes(), 36, "49 covering nodes minus 13 replicas");
    for e in 0..mesh.n_elements() as u32 {
        for &n in mesh.element_nodes(e) {
            assert!(
                (n as usize) < mesh.n_nodes(),
                "element {e} references node {n} ≥ n_nodes — make_periodic \
                 left an unresolved replica chain in the connectivity"
            );
        }
    }
}

/// Torus-quotient dof counts through the row-mesh periodic path (defect 2).
#[test]
fn d820_quad9_periodic_h1_counts_match_torus_topology() {
    let mesh = periodic_quad9_mesh();
    let counts: Vec<(u8, usize)> = [1u8, 2, 3]
        .iter()
        .map(|&o| (o, H1Space::new(mesh.clone(), o).n_dofs()))
        .collect();
    assert_eq!(counts[0], (1, 9), "H1(1) = V");
    assert_eq!(counts[1], (2, 36), "H1(2) = V + E + C");
    assert_eq!(counts[2], (3, 81), "H1(3) = V + 2E + 4C");
}

/// The quotient is complete: every dof of the periodic space is used by at
/// least one element row, and the seam is actually identified (the wrap-around
/// pair of elements shares dofs, so the distinct-dof count equals `n_dofs`).
#[test]
fn d820_quad9_periodic_quotient_complete_and_seam_merged() {
    let mesh = periodic_quad9_mesh();
    let space = H1Space::new(mesh.clone(), 2);
    let mut used = std::collections::BTreeSet::new();
    for e in 0..mesh.n_elements() as u32 {
        for &d in space.element_dofs(e) {
            assert!(d < space.n_dofs() as u32, "dof {d} out of range");
            used.insert(d);
        }
    }
    assert_eq!(
        used.len(),
        space.n_dofs(),
        "orphan dofs — the quotient merged fewer entities than the torus has"
    );
    // Seam: element 0 (corner block (0,0)) and element 8 (corner block
    // (2,2)) are adjacent only through *both* periodic identifications; the
    // corner dof of (0,0) must be the same id as the folded corner of (2,2).
    let e0 = space.element_dofs(0);
    let e8 = space.element_dofs(8);
    let shared = e0.iter().filter(|d| e8.contains(d)).count();
    assert!(
        shared >= 1,
        "the (0,0)/(2,2) wrap-around blocks share no dof — corner group \
         not identified across both translations"
    );
}
