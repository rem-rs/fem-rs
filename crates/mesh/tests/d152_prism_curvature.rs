//! D152: `Mesh::set_curvature_prism6` must place its geometry nodes at the
//! reference points of **`PrismPk`**'s DOF lattice, because `PrismPk` is the
//! element every consumer uses to evaluate a curved prism's geometry
//! (`Mesh::element_jacobian` / `element_jacobian_at`'s `Prism` arm,
//! `crates/assembly`'s `geo_ref_elem`, `crates/mesh/src/curved.rs::CurvedMesh`).
//!
//! The reference domain of fem-rs's prism family is fixed by `PrismPk`:
//! `xi = (ξ, η, ζ)` with `ξ ∈ [0,1]` the extrusion as the *first* coordinate
//! (the ordering `crates/element/src/quadrature.rs::prism_rule` and
//! `element_jacobian` use) and `(η, ζ)` the unit triangle
//! `(0,0), (1,0), (0,1)`.  The mesh's element-local vertex order is therefore
//! bottom triangle `0-1-2` (ξ = 0) then top triangle `3-4-5` (ξ = 1), with
//! triangle vertex `1` at `η = 1` and vertex `2` at `ζ = 1` — which is also
//! MFEM's `Geometry::PRISM` vertex order (MFEM's `z` ⟷ `PrismPk`'s `ξ`).
//!
//! The table's contract (as for the tet family, `tet_geometry_family_tests` in
//! `crates/mesh/src/simplex.rs`): slot `d` holds the coordinate of the geometry
//! node sitting at `PrismPk::new(p).dof_coords()[d]`.  The checks below pin
//! that:
//!
//! 1. **Slot positions**: on a straight-sided prism the stored coordinate of
//!    slot `d` must equal the *linear* prism map at `PrismPk::dof_coords()[d]`.
//!    (Note that the tautological "`element_jacobian(q_slot)` at a slot's own
//!    reference point returns the stored value" identity — the one used for the
//!    tet family split — cannot see this defect: it is a property of the nodal
//!    basis alone, and holds for *any* table when the element is the one the
//!    table is meant for.)
//! 2. **End to end**: on a straight-sided prism the order-`p` geometry must
//!    reproduce the original linear prism map exactly at every quadrature
//!    point, and the isoparametric Jacobian must equal the analytic one.
//! 3. **Sharing**: the six vertex slots reuse the mesh vertices in local vertex
//!    order and every other slot owns a fresh node, so
//!    `n_nodes = V + (npe − 6)·NE`.

use fem_element::lagrange::PrismPk;
use fem_element::quadrature::prism_rule;
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;

/// A single straight unit prism in the local vertex order of MFEM's `Prism6`
/// (bottom triangle 0-1-2, top triangle 3-4-5).
fn unit_prism() -> Mesh<3> {
    skewed_prism(1.0)
}

/// A single *straight-sided* prism whose top triangle is shifted, rotated and
/// raised, so that the map is genuinely multilinear (not affine) and a wrong
/// node ordering cannot pass accidentally.
fn skewed_prism(scale: f64) -> Mesh<3> {
    Mesh::<3> {
        coords: vec![
            0.0, 0.0, 0.0, // v0 bottom
            1.0 * scale, 0.0, 0.0, // v1
            0.0, 1.0 * scale, 0.0, // v2
            0.2 * scale, 0.3 * scale, 1.2, // v3 top
            1.1 * scale, 0.4 * scale, 1.1, // v4
            0.1 * scale, 1.2 * scale, 1.3, // v5
        ],
        conn: vec![0, 1, 2, 3, 4, 5],
        elem_tags: vec![1],
        elem_type: ElementType::Prism6,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Tri3,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    }
}

/// Two prisms sharing a quad face (extruded `unit_square_tri(2)`): exercises
/// the element-to-element path as well.
fn two_prisms() -> Mesh<3> {
    fem_mesh::extrusion::extrude_tri3_to_prisms(&Mesh::<2>::unit_square_tri(2), 1, 1.0)
}

fn prisms() -> [(&'static str, Mesh<3>); 2] {
    [("unit_prism", unit_prism()), ("two_prisms", two_prisms())]
}

/// The straight-sided prism map in `PrismPk`'s reference domain: `xi[0]` is the
/// extrusion, `(xi[1], xi[2])` the triangle.
fn linear_prism_map(v: &[[f64; 3]; 6], xi: &[f64]) -> [f64; 3] {
    let (r, s, t) = (xi[0], xi[1], xi[2]);
    let l0 = 1.0 - s - t;
    let mut x = [0.0_f64; 3];
    for d in 0..3 {
        let bottom = l0 * v[0][d] + s * v[1][d] + t * v[2][d];
        let top = l0 * v[3][d] + s * v[4][d] + t * v[5][d];
        x[d] = (1.0 - r) * bottom + r * top;
    }
    x
}

/// Analytic `J = ∂x/∂xi` of `linear_prism_map` (`j[d][k] = ∂x_d/∂xi_k`).
fn linear_prism_jacobian(v: &[[f64; 3]; 6], xi: &[f64]) -> [[f64; 3]; 3] {
    let (r, s, t) = (xi[0], xi[1], xi[2]);
    let l0 = 1.0 - s - t;
    let mut j = [[0.0_f64; 3]; 3];
    for d in 0..3 {
        let bottom = l0 * v[0][d] + s * v[1][d] + t * v[2][d];
        let top = l0 * v[3][d] + s * v[4][d] + t * v[5][d];
        j[d][0] = top - bottom;
        j[d][1] = (1.0 - r) * (v[1][d] - v[0][d]) + r * (v[4][d] - v[3][d]);
        j[d][2] = (1.0 - r) * (v[2][d] - v[0][d]) + r * (v[5][d] - v[3][d]);
    }
    j
}

fn verts_of(m: &Mesh<3>, e: u32) -> [[f64; 3]; 6] {
    let ns = m.element_nodes(e);
    std::array::from_fn(|k| m.coords_of(ns[k]))
}

/// `PrismPk`'s basis is nodal at `dof_coords()`: documents the element-side half
/// of the contract that check 1 relies on (read-only use of `crates/element`).
#[test]
fn prism_pk_basis_is_nodal_at_its_dof_coords() {
    for &p in &[1usize, 2, 3, 4] {
        let fe = PrismPk::new(p);
        let rc = fe.dof_coords();
        let n = fe.n_dofs();
        assert_eq!(rc.len(), n);
        let mut vals = vec![0.0_f64; n];
        for (s, xi) in rc.iter().enumerate() {
            fe.eval_basis(xi, &mut vals);
            for k in 0..n {
                let want = if k == s { 1.0 } else { 0.0 };
                assert!(
                    (vals[k] - want).abs() < 1e-12,
                    "p={p}: basis {k} at dof {s}'s own point {xi:?} = {} (want {want})",
                    vals[k]
                );
            }
        }
    }
}

/// The slot order `set_curvature_prism6` produces is frozen to **`PrismPk`'s**,
/// layer-major order — pinned here (p = 2, in fem-rs's reference domain
/// `(ξ extrusion, η, ζ triangle)`) because it is the contract between the mesh
/// table and the element that evaluates it, and because it is *not* MFEM's
/// prism layout.
///
/// MFEM 4.10's `H1_WedgeElement` stores its DOFs in the **entity** order
/// (probe `tmp/d152_prism_probe.cpp`, `Mesh::SetCurvature(2)` on
/// `MakeCartesian3D(1,1,1, WEDGE)`): 6 vertices, then the 3 bottom triangle
/// edges, the 3 top triangle edges and the 3 vertical edges, then the 2
/// triangles, the 3 quadrilaterals and the interior — i.e.
/// `[v0 v1 v2 v3 v4 v5 | e01 e12 e20 e34 e45 e53 e03 e14 e25 | q01 q12 q20]`,
/// which is exactly the layout `crates/space/src/dof_manager.rs`'s prism
/// builders emit.  `PrismPk` instead orders its (identical) p = 2 lattice as
/// `[v0 v1 v2 e01 e12 e20 | v3 v4 v5 e34 e45 e53 | e03 e14 e25 | q01 q12 q20]`,
/// so slots 3..12 hold different entities on the two sides: the io layer's
/// MFEM `nodes` permutation (D151) is still missing, and so is the agreement
/// between the prism H1 space and `PrismPk`.  If `PrismPk`'s order is ever
/// changed to the entity order (D119/D151), this test fails and
/// `set_curvature_prism6` must be changed with it.
#[test]
fn prism_pk_slot_order_is_frozen() {
    let p = 2usize;
    let rc = PrismPk::new(p).dof_coords();
    // Layer-major: layer ξ = 0, 1/2, 1; inside a layer the `TriPk` order
    // (3 vertices, ζ=0 edge, hypotenuse, η=0 edge) of `equispaced_nodes_tri`.
    let expected: [[f64; 3]; 18] = [
        [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.5],
        [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
        [1.0, 0.5, 0.0], [1.0, 0.5, 0.5], [1.0, 0.0, 0.5],
    ];
    assert_eq!(rc.len(), expected.len());
    for (s, (got, want)) in rc.iter().zip(&expected).enumerate() {
        for d in 0..3 {
            assert!(
                (got[d] - want[d]).abs() < 1e-14,
                "PrismPk p=2 slot {s}: reference point {got:?} != {want:?} — the mesh table's \
                 slot order changed; update `set_curvature_prism6` (and the io/space sides) \
                 together with it"
            );
        }
    }
    // And they are *not* MFEM's slots: MFEM's slot 3 is the first top vertex
    // (`xi = 1`), `PrismPk`'s slot 3 is a bottom-edge node.
    assert!(rc[3][0] != 1.0, "PrismPk slot 3 changed to MFEM's entity order — the D151 \
        permutation (and `set_curvature_prism6`) must be revisited");
}

/// 1. Every stored node must sit at the linear map's image of *its own*
///    `PrismPk` reference point.
#[test]
fn prism_geometry_nodes_sit_at_prism_pk_reference_points() {
    // Collect every (p, mesh) failure before asserting, so one run reports the
    // whole picture (`PrismPk`'s lattice and the wrong table can agree at low
    // order and diverge higher up, as for the tet/tri families).
    let mut failures: Vec<String> = Vec::new();
    for &p in &[2usize, 3, 4] {
        for (name, mut m) in prisms() {
            m.set_curvature(p as u8);
            let rc = PrismPk::new(p).dof_coords();
            let g = m.geometry.as_ref().expect("geometry");
            assert_eq!(g.nodes_per_elem, rc.len(), "p={p} {name}: slots per element");

            let mut worst = 0.0_f64;
            let mut first: Option<(usize, usize, [f64; 3], [f64; 3], usize, f64)> = None;
            for e in 0..m.n_elems() as u32 {
                let v = verts_of(&m, e);
                let nodes = m.geometry_nodes(e);
                for s in 0..rc.len() {
                    let want = linear_prism_map(&v, &rc[s]);
                    let gc = m.geom_coords_of(nodes[s]);
                    let got = [gc[0], gc[1], gc[2]];
                    let delta = (0..3)
                        .map(|d| (got[d] - want[d]).abs())
                        .fold(0.0_f64, f64::max);
                    if delta > worst {
                        worst = delta;
                    }
                    if first.is_none() && delta > 1e-12 {
                        // Which `PrismPk` slot *does* this stored node look like?
                        let (nearest, nd) = (0..rc.len())
                            .map(|j| {
                                let q = linear_prism_map(&v, &rc[j]);
                                let dd = (0..3).map(|d| (got[d] - q[d]).abs()).fold(0.0_f64, f64::max);
                                (j, dd)
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap();
                        first = Some((e as usize, s, got, want, nearest, nd));
                    }
                }
            }
            match first {
                None => eprintln!("p={p} {name}: all slots on their PrismPk reference points"),
                Some((e, s, got, want, nearest, nd)) => eprintln!(
                    "p={p} {name}: max |Δ| = {worst:.3e}; first mismatch at elem {e} slot {s}: \
                     stored {got:?}, expected {want:?} (linear map at PrismPk ref {:?}); \
                     the stored point is {nd:.3e} from PrismPk slot {nearest}'s image {:?}",
                    rc[s],
                    linear_prism_map(&verts_of(&m, e as u32), &rc[nearest]),
                ),
            }
            if worst >= 1e-12 {
                failures.push(format!("p={p} {name}: worst |Δ| = {worst:.3e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "a geometry node is not at its PrismPk reference point — the table's slot order \
         and the element evaluating it disagree, so every curved prism assembles the wrong \
         isoparametric map:\n  {}",
        failures.join("\n  ")
    );
}

/// 2. Straight-sided prism: the order-`p` geometry must interpolate the
///    original map exactly at every quadrature point, and the isoparametric
///    Jacobian must equal the analytic one.
#[test]
fn prism_curved_geometry_reproduces_the_linear_map() {
    let mut failures: Vec<String> = Vec::new();
    for &p in &[2usize, 3, 4] {
        for (name, mut m) in prisms() {
            m.set_curvature(p as u8);
            let rule = prism_rule(p.min(6) as u8);
            let mut worst_x = 0.0_f64;
            let mut worst_j = 0.0_f64;
            let mut where_x = (0usize, [0.0_f64; 3]);
            for e in 0..m.n_elems() as u32 {
                let v = verts_of(&m, e);
                for xi in &rule.points {
                    let (j, _, xp) = m.element_jacobian(e, xi);
                    let want = linear_prism_map(&v, xi);
                    let jwant = linear_prism_jacobian(&v, xi);
                    for d in 0..3 {
                        if (xp[d] - want[d]).abs() > worst_x {
                            worst_x = (xp[d] - want[d]).abs();
                            where_x = (e as usize, [xi[0], xi[1], xi[2]]);
                        }
                        for k in 0..3 {
                            worst_j = worst_j.max((j[(d, k)] - jwant[d][k]).abs());
                        }
                    }
                }
            }
            eprintln!(
                "p={p} {name}: max |x − linear| = {worst_x:.3e} at {where_x:?}, \
                 max |J − ∂linear| = {worst_j:.3e}"
            );
            if worst_x >= 1e-11 || worst_j >= 1e-10 {
                failures.push(format!(
                    "p={p} {name}: max |Δx| = {worst_x:.3e}, max |ΔJ| = {worst_j:.3e}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "the order-p prism geometry does not reproduce the straight-sided map \
         (the correct order-p table reproduces it exactly, because the map is in \
         PrismPk's span):\n  {}",
        failures.join("\n  ")
    );
}

/// 3. Sharing: the vertex slots of the table (the `PrismPk` slots whose
///    reference point is a prism vertex — for `p ≥ 2` they are *not* the first
///    six slots, the layer-major lattice interleaves them with layer 0's edge
///    nodes) reuse the mesh vertices in local vertex order, and every other
///    slot owns a fresh node.
#[test]
fn prism_geometry_reuses_the_vertices() {
    // `PrismPk`'s reference points of the six local vertices, in local order.
    let vertex_ref = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ];
    let mut failures: Vec<String> = Vec::new();
    for &p in &[2usize, 3, 4] {
        let mut m = two_prisms();
        let nv = m.n_nodes();
        let ne = m.n_elems();
        m.set_curvature(p as u8);
        let g = m.geometry.as_ref().unwrap();
        let npe = (p + 1) * (p + 1) * (p + 2) / 2;
        assert_eq!(g.nodes_per_elem, npe);
        if g.n_nodes != nv + (npe - 6) * ne {
            failures.push(format!(
                "p={p}: n_geom_nodes = {} but V + (npe−6)·NE = {} (V={nv}, NE={ne}, npe={npe})",
                g.n_nodes,
                nv + (npe - 6) * ne
            ));
        }
        let rc = PrismPk::new(p).dof_coords();
        let vertex_slot: Vec<usize> = (0..rc.len())
            .filter(|&d| {
                vertex_ref
                    .iter()
                    .any(|q| (0..3).all(|k| (rc[d][k] - q[k]).abs() < 1e-12))
            })
            .collect();
        if vertex_slot.len() != 6 {
            failures.push(format!(
                "p={p}: the reference lattice has {} points at prism vertices, expected 6",
                vertex_slot.len()
            ));
            continue;
        }
        let mut bad_slots: Vec<String> = Vec::new();
        for e in 0..ne as u32 {
            let verts = m.element_nodes(e).to_vec();
            let nodes = m.geometry_nodes(e).to_vec();
            for (v, &slot) in vertex_slot.iter().enumerate() {
                if nodes[slot] != verts[v] {
                    bad_slots.push(format!(
                        "elem {e} vertex {v} (slot {slot}): node {} instead of the element's vertex {}",
                        nodes[slot], verts[v]
                    ));
                }
            }
        }
        eprintln!("p={p}: {} vertex slot(s) not reusing the mesh vertex", bad_slots.len());
        if !bad_slots.is_empty() {
            failures.push(format!(
                "p={p}: {} vertex slots duplicated: {}",
                bad_slots.len(),
                bad_slots.join("; ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "the prism geometry table does not reuse the mesh vertices (vertex DOFs must be \
         the mesh's own vertices, shared across elements):\n  {}",
        failures.join("\n  ")
    );
}
