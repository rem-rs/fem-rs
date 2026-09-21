//! D560 — tet RT0 mass matrix parity with MFEM 4.10 on the 2×2×2 tet cube.
//!
//! This is the basis-side verifier of the D560/D571 dual-normalization fix:
//! the mass matrix depends only on the RT0 basis (not on the dof
//! convention), so an entry-by-entry match against MFEM's
//! `VectorFEMassIntegrator` output (golden `d560_rt0mass_tet222_mfem.txt`
//! from `tmp/d546/d560_final_probe.cpp`) proves the fem-rs `TetRTk(0)`
//! reference basis normalization is bit-faithful to MFEM's fixed-order
//! `RT0TetFiniteElement` — the element `RT0_3DFECollection` actually serves
//! (`fe_fixed_order.cpp:6246`, basis = 2× the generic
//! `RT_TetrahedronElement(0)`) — while the D560-halved interpolation duals
//! (`n̂|F|` rows in `mfem_nodal_dofs(0)`) live in the same element's
//! gridfunction convention.

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_space::HDivSpace;

const GOLDEN: &str = include_str!("data/d560_rt0mass_tet222_mfem.txt");

#[test]
fn d560_tet_rt0_mass_matrix_matches_mfem() {
    let path = format!(
        "{}/tests/data/d555_tet222_mfem.mesh",
        env!("CARGO_MANIFEST_DIR")
    );
    let mfem = read_mfem_file(&path).unwrap().mesh3d.unwrap();
    let space = HDivSpace::new(mfem, 0);

    let mut want = Vec::new();
    let mut nnz = 0usize;
    for line in GOLDEN.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] == "RT0MASS" && t[1] == "nnz" {
            nnz = t[2].parse().unwrap();
        } else if t[0] == "RT0MASS" {
            want.push((
                t[1].parse::<usize>().unwrap(),
                t[2].parse::<usize>().unwrap(),
                t[3].parse::<f64>().unwrap(),
            ));
        }
    }
    let mass = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorMassIntegrator { alpha: 1.0_f64 }],
        8,
    );
    for (i, j, v) in &want {
        let got = mass.get(*i, *j);
        assert!(
            (got - v).abs() < 1e-12 * (1.0 + v.abs()),
            "mass ({i},{j}): {got} vs mfem {v}"
        );
    }
    assert!(nnz > 0);
}

