//! D572 — prism RT0 mass-matrix and Project parity with MFEM 4.10's
//! `RT0_3DFECollection` (the prism side of the D572 collection alignment).
//!
//! This is the prism sibling of `d560_rt0_mass_parity`: it proves the fem-rs
//! `PrismRTk(0)` basis normalization is bit-faithful to MFEM's fixed-order
//! `RT0WdgFiniteElement` — the element `RT0_3DFECollection` serves
//! (`fe_coll.hpp:1470-1476`), i.e. the generic `RT_WedgeElement(0)` tensor
//! basis with the two *triangular-face* functions doubled
//! (`fe_fixed_order.cpp:6403`) and `nk` rows halved to `n̂|F|` on the same
//! faces (`:6439`).  Golden values were dumped from MFEM 4.10
//! (`$HOME/mfem410_ser`) by `tmp/d572/d572_probe_prism_mass.cpp` on
//! `Mesh::MakeCartesian3D(1,1,1,WEDGE)` (the D482 two-wedge unit cube) and
//! converted to (element, slot) form by `tmp/d572/gen_prism_mass_truth.rs`
//! — see `tmp/d572/adjudication.md` for the D572 adjudication that picked
//! this collection.
//!
//! Measured facts these assertions carry (probe
//! `tmp/d572/d572_prism_mass.txt`):
//! * the triangular-face mass block is 4x the generic `RT_FECollection`
//!   value (2/3 vs 1/6 diagonal on the unit faces);
//! * the quadrilateral-face blocks are identical to the generic
//!   `RT_WedgeElement(0)`'s;
//! * the triangular-face Project dofs are half the generic ones
//!   (`∓0.5` vs `∓1.0` for the unit faces) — the `n̂|F|` face-flux
//!   functional, the same convention the D560 tet and the Fuentes pyramid
//!   base already carry in fem-rs.

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_space::HDivSpace;

// Generated from tmp/d572/d572_prism_mass.txt by
// tmp/d572/gen_prism_mass_truth.py — do not edit.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d572/d572_prism_rt0_truth.rs"));

/// `Mesh::MakeCartesian3D(1, 1, 1, WEDGE)` (the D482 connectivity): the unit
/// cube split into two wedges along the diagonal plane.
fn mfem_two_wedge_mesh() -> fem_mesh::Mesh<3> {
    use fem_mesh::element_type::ElementType;
    fem_mesh::Mesh::<3> {
        coords: vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        conn: vec![0, 1, 3, 4, 5, 7, 0, 3, 2, 4, 7, 6],
        vertex_parents: vec![],
        elem_tags: vec![1, 1],
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
        nc_vertex_view: None,
        geometry: None,
    }
}

/// The same cube with only wedge 0 (`hex_to_wdg` child 0) — the single-element
/// mass golden [`MFEM_PRISM_O0_EMASS`].
fn single_wedge_mesh() -> fem_mesh::Mesh<3> {
    use fem_mesh::element_type::ElementType;
    fem_mesh::Mesh::<3> {
        coords: vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        conn: vec![0, 1, 3, 4, 5, 7],
        vertex_parents: vec![],
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
        nc_vertex_view: None,
        geometry: None,
    }
}

/// The assembled mass matrix of the two-wedge cube matches MFEM's
/// `VectorFEMassIntegrator` output entry-by-entry (25 nonzeros), with each
/// golden endpoint resolved through fem-rs's own `(element, slot)` dofs and
/// the MFEM/fem-rs canonical face signs cross-multiplied (they agree on this
/// mesh; the factor is a guard).
#[test]
fn d572_prism_rt0_mass_matrix_matches_mfem_rt03d() {
    let mesh = mfem_two_wedge_mesh();
    let space = HDivSpace::new(mesh, 0);
    let mass = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorMassIntegrator { alpha: 1.0_f64 }],
        8,
    );
    assert!(MFEM_PRISM_O0_MASS.len() >= 25);
    let mut max_err = 0.0_f64;
    for &(e, s, sm, e2, s2, sm2, v) in MFEM_PRISM_O0_MASS {
        let gi = space.element_dofs(e)[s as usize] as usize;
        let gj = space.element_dofs(e2)[s2 as usize] as usize;
        let sf = space.element_signs(e)[s as usize];
        let sf2 = space.element_signs(e2)[s2 as usize];
        // the fem-rs global entry for the same physical pair, re-expressed in
        // MFEM's sign convention:  v_mfem = (sm_i*sm_j)*(sf_i*sf_j) * v_femrs
        let want = v * (sm * sm2) * (sf * sf2);
        let got = mass.get(gi, gj);
        max_err = max_err.max((got - want).abs());
        assert!(
            (got - want).abs() <= 1e-12 * (1.0 + v.abs()),
            "mass (elem {e} slot {s}) x (elem {e2} slot {s2}): fem-rs {got} vs mfem {want} (raw {v})"
        );
    }
    eprintln!(
        "d572 prism RT0 mass: {} golden entries matched, max|delta| = {max_err:.3e}",
        MFEM_PRISM_O0_MASS.len()
    );
}

/// The single-wedge element mass matrix (the 5x5 EMASS dump) matches too —
/// the pure element-level view of the same basis normalization.
#[test]
fn d572_prism_rt0_element_mass_matches_mfem_rt03d() {
    let space = HDivSpace::new(single_wedge_mesh(), 0);
    let mass = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorMassIntegrator { alpha: 1.0_f64 }],
        8,
    );
    for &(a, b, v) in MFEM_PRISM_O0_EMASS {
        let got = mass.get(a as usize, b as usize);
        assert!(
            (got - v).abs() <= 1e-12 * (1.0 + v.abs()),
            "element mass ({a},{b}): fem-rs {got} vs mfem {v}"
        );
    }
}

/// The stored dof values (`HDivSpace::interpolate_vector` = MFEM
/// `Project_RT`) match the RT0_3D `Project` dump on every element for four
/// constant fields: triangular-face dofs at `n̂|F|` (±0.5 on the unit faces),
/// quadrilateral dofs at the generic full-nk values.  Compared through each
/// side's own slot signs (the local value is `sign * stored`).
#[test]
fn d572_prism_rt0_project_dofs_match_mfem_rt03d() {
    let mesh = mfem_two_wedge_mesh();
    let space = HDivSpace::new(mesh, 0);
    let fields: [&[f64]; 4] = [&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0], &[0.9, 0.4, -1.1]];
    for (c, f) in fields.iter().enumerate() {
        let x = space.interpolate_vector(&|_| f.to_vec());
        for e in 0..2u32 {
            let dofs = space.element_dofs(e);
            let signs = space.element_signs(e);
            for (s, want) in MFEM_PRISM_O0_PROJ[e as usize][c].iter().enumerate() {
                let want = *want;
                let local = signs[s] * x.as_slice()[dofs[s] as usize];
                assert!(
                    (local - want).abs() <= 1e-12 * (1.0 + want.abs()),
                    "project elem {e} field {c} slot {s}: fem-rs local {local} vs mfem {want}"
                );
            }
        }
    }
}
