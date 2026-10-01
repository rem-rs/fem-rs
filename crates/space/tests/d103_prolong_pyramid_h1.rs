//! D103 / D536b — H1 **pyramid** and **prism** h-refinement prolongation vs
//! MFEM 4.10.
//!
//! Before D103 `build_h1_prolongation_matrix` panicked on any Pyramid5/Prism6
//! coarse element (`lagrange_ref_3d` knew only Tet/Hex) and no pyramid point
//! locator existed anywhere (the mesh crate's findpts is simplex-only) — the
//! D557 leftover "D536b H1 pyramid prolongation locator".
//!
//! Ground truth (WSL `$HOME/mfem410_ser`, probe `tmp/d103prol/probe_d103.cpp`,
//! dumps `tests/data/d103/*.txt`):
//!
//! * **prism** (`prismh1_p.txt`): MFEM's own
//!   `FiniteElementSpace::RefinementOperator` dumped by applying it to the
//!   identity — MFEM's embedding bookkeeping is correct for prisms
//!   (`UniformRefinement3D_base`, parent = i/8 with 8 children per parent).
//! * **pyramid** (`pyrsem_h*.txt`): *semantic* truth.  MFEM 4.10's serial
//!   uniform-refinement operator is **broken on pyramids** — the PYRAMID
//!   branch emits 10 children per parent (6 pyramids + 4 tetrahedra) but
//!   `CoarseFineTransformations` embeddings are written as `parent = i/8`
//!   (`mesh.cpp:11062`), the 4 tet children are skipped entirely, and
//!   `GetLocalRefinementMatrices` would pair tet-geometry local matrices with
//!   pyramid-geometry coarse DOF lists.  The probe therefore evaluates MFEM's
//!   documented interpolation formula directly
//!   (`NodalLocalInterpolation`, fe_base.cpp:526:
//!   `I(i,j) = φ_j^coarse(parent-ref coords of fine node i)`) with MFEM's own
//!   `H1_FuentesPyramidElement` / `H1_TetrahedronElement` basis functions and
//!   its own `ElementTransformation` inversion — per child element, parent =
//!   child/nch (the refinement emits each parent's 10 children contiguously).
//!
//! The fem-rs side reproduces that physics with the physical-space locators:
//! each fine DOF coordinate is located in the coarse mesh (the pyramid locator
//! inverts the collapsed-hex P1 map, with the collapsed apex answered exactly)
//! and the coarse element's own basis is evaluated at the recovered reference
//! point.

use fem_element::lagrange::PyramidBasisType;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform_3d, Mesh, MeshTopology};
use fem_space::constraints::prolong::{
    build_h1_prolongation_matrix, build_h1_prolongation_matrix_with_pyramid_basis,
};
use fem_space::dof_manager::DofManager;

const PYR_H1_P1: &str = include_str!("data/d103/pyrsem_h11.txt");
const PYR_H1_P2: &str = include_str!("data/d103/pyrsem_h12.txt");
const PRISM_H1_P1: &str = include_str!("data/d103/prismh1_1.txt");
const PRISM_H1_P2: &str = include_str!("data/d103/prismh1_2.txt");

fn mesh3(rel: &str) -> Mesh<3> {
    let path = format!("{}/tests/{rel}", env!("CARGO_MANIFEST_DIR"));
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

/// `pyrsem` dump: CSIZE/FSIZE, CDOF/FDOF tables and per-slot semantic rows.
struct SemDump {
    csize: usize,
    fsize: usize,
    cdof: Vec<Vec<u32>>,
    fdof: Vec<Vec<u32>>,
    /// (fine element, fine slot, parent element, [(parent slot, value)])
    rows: Vec<(usize, usize, usize, Vec<(u32, f64)>)>,
    /// (element, slot) -> physical position (CPOS / FPOS lines)
    cpos: std::collections::HashMap<(usize, usize), [f64; 3]>,
    fpos: std::collections::HashMap<(usize, usize), [f64; 3]>,
}

fn parse_sem(text: &str) -> SemDump {
    let mut csize = 0usize;
    let mut fsize = 0usize;
    let mut cdof = Vec::new();
    let mut fdof = Vec::new();
    let mut rows = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("CSIZE") => csize = t[1].parse().expect("csize"),
            Some("FSIZE") => fsize = t[1].parse().expect("fsize"),
            Some("CDOF") => cdof.push(t[2..].iter().map(|v| v.parse().unwrap()).collect()),
            Some("FDOF") => fdof.push(t[2..].iter().map(|v| v.parse().unwrap()).collect()),
            Some("ROW") => {
                let parent: usize = t[3].parse().expect("parent");
                let mut pairs = Vec::new();
                for chunk in t[4..].chunks(2) {
                    pairs.push((chunk[0].parse().unwrap(), chunk[1].parse().unwrap()));
                }
                rows.push((t[1].parse().unwrap(), t[2].parse().unwrap(), parent, pairs));
            }
            _ => {}
        }
    }
    let mut cpos = std::collections::HashMap::new();
    let mut fpos = std::collections::HashMap::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("CPOS") => {
                cpos.insert(
                    (t[1].parse().unwrap(), t[2].parse().unwrap()),
                    [t[3].parse().unwrap(), t[4].parse().unwrap(), t[5].parse().unwrap()],
                );
            }
            Some("FPOS") => {
                fpos.insert(
                    (t[1].parse().unwrap(), t[2].parse().unwrap()),
                    [t[3].parse().unwrap(), t[4].parse().unwrap(), t[5].parse().unwrap()],
                );
            }
            _ => {}
        }
    }
    SemDump { csize, fsize, cdof, fdof, rows, cpos, fpos }
}

/// `RefinementOperator` dump: sparse `P i j v ; …` column lines + CDOF/FDOF.
struct OpDump {
    csize: usize,
    fsize: usize,
    entries: Vec<(usize, usize, f64)>,
    cdof: Vec<Vec<u32>>,
    fdof: Vec<Vec<u32>>,
}

fn parse_op(text: &str) -> OpDump {
    let mut csize = 0usize;
    let mut fsize = 0usize;
    let mut entries = Vec::new();
    let mut cdof = Vec::new();
    let mut fdof = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("CSIZE") => csize = t[1].parse().expect("csize"),
            Some("FSIZE") => fsize = t[1].parse().expect("fsize"),
            Some("CDOF") => cdof.push(t[2..].iter().map(|v| v.parse().unwrap()).collect()),
            Some("FDOF") => fdof.push(t[2..].iter().map(|v| v.parse().unwrap()).collect()),
            Some("P") => {
                for chunk in t[1..].chunks(4) {
                    if chunk.len() == 4 && chunk[3] == ";" {
                        entries.push((
                            chunk[0].parse().unwrap(),
                            chunk[1].parse().unwrap(),
                            chunk[2].parse().unwrap(),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    OpDump { csize, fsize, entries, cdof, fdof }
}


fn row_entries(p: &CsrMatrix<f64>, row: usize) -> std::collections::HashMap<usize, f64> {
    let mut m = std::collections::HashMap::new();
    for k in p.row_ptr[row]..p.row_ptr[row + 1] {
        m.insert(p.col_idx[k] as usize, p.values[k]);
    }
    m
}

/// The pyramid arm: mixed refined children (6 pyramids + 4 tets per parent),
/// orders 1 and 2, against the semantic probe rows.
#[test]
fn h1_pyramid_prolongation_matches_mfem_semantics() {
    let coarse = mesh3("data/d103/inline-pyramid.mesh");
    let fine = refine_uniform_3d(&coarse);
    // MFEM's inline-pyramid generator: each grid brick becomes 6 pyramids
    // (one per face, apex at the brick centre) — 2×2×2 bricks → 48 parents.
    assert_eq!(coarse.n_elements(), 48, "2×2×2 inline pyramid block");
    assert_eq!(fine.n_elements(), 480, "10 children per pyramid parent");

    for (p, dump_text) in [(1u8, PYR_H1_P1), (2u8, PYR_H1_P2)] {
        let d = parse_sem(dump_text);
        let cdm = DofManager::new(&coarse, p);
        let fdm = DofManager::new(&fine, p);
        let pmat = build_h1_prolongation_matrix(&coarse, &cdm, &fine, &fdm);

        assert_eq!(pmat.nrows, d.fsize, "p={p}: fine vsize vs MFEM FSIZE");
        assert_eq!(pmat.ncols, d.csize, "p={p}: coarse vsize vs MFEM CSIZE");
        // Global numbering cross-check: fem-rs's element dof tables must be
        // MFEM's (slot orders pinned by d191/d347/d352 for pyramids, d157 for
        // tets, d349 for the mixed numbering) — this is what turns the
        // per-slot ROW comparison into a full matrix comparison.
        for (e, want) in d.cdof.iter().enumerate() {
            let got: Vec<u32> = cdm.element_dofs(e as u32).iter().map(|&v| v as u32).collect();
            assert_eq!(got, *want, "p={p}: CDOF element {e}");
        }
        for (e, want) in d.fdof.iter().enumerate() {
            let got: Vec<u32> = fdm.element_dofs(e as u32).iter().map(|&v| v as u32).collect();
            assert_eq!(got, *want, "p={p}: FDOF element {e}");
        }

        // Coordinate-keyed: fem-rs normalizes tet vertex order on read, so slot
        // ids can differ; the DOF *positions* are the common ground.
        let coord_key = |c: &[f64]| {
            [
                (c[0] * 1e9).round() as i64,
                (c[1] * 1e9).round() as i64,
                (c[2] * 1e9).round() as i64,
            ]
        };
        let mut fine_by_coord: std::collections::HashMap<[i64; 3], u32> =
            std::collections::HashMap::new();
        for dof in 0..fdm.n_dofs as u32 {
            fine_by_coord.insert(coord_key(fdm.dof_coord(dof)), dof);
        }
        let mut coarse_by_coord: std::collections::HashMap<[i64; 3], u32> =
            std::collections::HashMap::new();
        for dof in 0..cdm.n_dofs as u32 {
            coarse_by_coord.insert(coord_key(cdm.dof_coord(dof)), dof);
        }
        let mut worst = 0.0_f64;
        let mut checked = 0usize;
        let mut missing = 0usize;
        let mut unmatched = 0usize;
        for &(f, i, parent, ref pairs) in &d.rows {
            let Some(&fg) = fine_by_coord.get(&coord_key(&d.fpos[&(f, i)])) else {
                unmatched += 1;
                continue;
            };
            let row = row_entries(&pmat, fg as usize);
            for &(j, v) in pairs {
                match coarse_by_coord.get(&coord_key(&d.cpos[&(parent, j as usize)])) {
                    Some(&cg) => match row.get(&(cg as usize)) {
                        Some(&got) => worst = worst.max((got - v).abs()),
                        None => missing += 1,
                    },
                    None => {
                        missing += 1;
                        unmatched += 1;
                    }
                }
                checked += 1;
            }
        }
        println!("H1 pyramid p={p}: {checked} entries, worst |Δ| = {worst:.3e}, missing {missing}");
        assert_eq!(unmatched, 0, "p={p}: MFEM DOF positions fem-rs does not have");
        assert_eq!(missing, 0, "p={p}: entries MFEM has that fem-rs lacks");
        assert!(worst < 5e-11, "p={p}: worst |Δ| = {worst:.3e}");
    }
}

/// The prism arm (same-arm completion of the 3-D locator chain, D103): MFEM's
/// own `RefinementOperator` on `ref-prism.mesh`, orders 1 and 2.
#[test]
fn h1_prism_prolongation_matches_mfem_operator() {
    let coarse = mesh3("data/d103/ref-prism.mesh");
    let fine = refine_uniform_3d(&coarse);

    for (p, dump_text) in [(1u8, PRISM_H1_P1), (2u8, PRISM_H1_P2)] {
        let d = parse_op(dump_text);
        let cdm = DofManager::new(&coarse, p);
        let fdm = DofManager::new(&fine, p);
        let pmat = build_h1_prolongation_matrix(&coarse, &cdm, &fine, &fdm);

        assert_eq!(pmat.nrows, d.fsize, "p={p}: fine vsize vs MFEM FSIZE");
        assert_eq!(pmat.ncols, d.csize, "p={p}: coarse vsize vs MFEM CSIZE");
        for (e, want) in d.cdof.iter().enumerate() {
            let got: Vec<u32> = cdm.element_dofs(e as u32).iter().map(|&v| v as u32).collect();
            assert_eq!(got, *want, "p={p}: CDOF element {e}");
        }
        for (e, want) in d.fdof.iter().enumerate() {
            let got: Vec<u32> = fdm.element_dofs(e as u32).iter().map(|&v| v as u32).collect();
            assert_eq!(got, *want, "p={p}: FDOF element {e}");
        }
        // global numbering coincides, so the entries compare directly
        let mut worst = 0.0_f64;
        let mut missing = 0usize;
        for &(i, j, v) in &d.entries {
            let mut got = 0.0_f64;
            for k in pmat.row_ptr[i]..pmat.row_ptr[i + 1] {
                if pmat.col_idx[k] as usize == j {
                    got = pmat.values[k];
                }
            }
            if got == 0.0 && v != 0.0 {
                missing += 1;
            }
            worst = worst.max((got - v).abs());
        }
        println!("H1 prism p={p}: {} entries, worst |Δ| = {worst:.3e}, missing {missing}", d.entries.len());
        assert_eq!(missing, 0, "p={p}: entries MFEM has that fem-rs lacks");
        assert!(worst < 5e-11, "p={p}: worst |Δ| = {worst:.3e}");
    }
}

/// The Bergot family variant stays reachable and stays a partition of unity
/// (no MFEM oracle for `pyr_type = 0` prolongation — the family wiring itself
/// is pinned by d347/d824).
#[test]
fn h1_pyramid_prolongation_bergot_variant_partition_of_unity() {
    let coords = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1.];
    let mesh = Mesh::<3>::uniform(
        coords,
        vec![0, 1, 2, 3, 4],
        vec![1],
        fem_mesh::ElementType::Pyramid5,
        vec![],
        vec![],
        fem_mesh::ElementType::Tri3,
    );
    let cdm = DofManager::new_with_pyramid_basis(&mesh, 2, PyramidBasisType::Bergot);
    let fdm = DofManager::new_with_pyramid_basis(&mesh, 3, PyramidBasisType::Bergot);
    let pmat = build_h1_prolongation_matrix_with_pyramid_basis(
        &mesh, &cdm, &mesh, &fdm,
        PyramidBasisType::Bergot,
    );
    assert_eq!(pmat.nrows, fdm.n_dofs);
    assert_eq!(pmat.ncols, cdm.n_dofs);
    for row in 0..pmat.nrows {
        let s: f64 = pmat.values[pmat.row_ptr[row]..pmat.row_ptr[row + 1]].iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "row {row} sums to {s}, expected 1"
        );
    }
}

/// p-refinement (same mesh) through the pyramid arm reproduces linear fields
/// exactly — the H1 P1 subspace is contained in the Fuentes P2/P3 spaces.
#[test]
fn h1_pyramid_p_refinement_linear_exact() {
    let coords = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1.];
    let mesh = Mesh::<3>::uniform(
        coords,
        vec![0, 1, 2, 3, 4],
        vec![1],
        fem_mesh::ElementType::Pyramid5,
        vec![],
        vec![],
        fem_mesh::ElementType::Tri3,
    );
    for fine_order in [2u8, 3u8] {
        let cdm = DofManager::new(&mesh, 1);
        let fdm = DofManager::new(&mesh, fine_order);
        let pmat = build_h1_prolongation_matrix(&mesh, &cdm, &mesh, &fdm);
        let u_c: Vec<f64> = (0..cdm.n_dofs)
            .map(|d| {
                let c = cdm.dof_coord(d as u32);
                1.0 + 2.0 * c[0] - 3.0 * c[1] + 0.5 * c[2]
            })
            .collect();
        let mut u_f = vec![0.0; fdm.n_dofs];
        pmat.spmv(&u_c, &mut u_f);
        let mut worst = 0.0_f64;
        for d in 0..fdm.n_dofs as u32 {
            let c = fdm.dof_coord(d);
            let exact = 1.0 + 2.0 * c[0] - 3.0 * c[1] + 0.5 * c[2];
            worst = worst.max((u_f[d as usize] - exact).abs());
        }
        assert!(worst < 1e-12, "fine_order={fine_order}: worst |Δ| = {worst:.3e}");
    }
}
