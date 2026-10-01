//! D103 — L² (discontinuous) h-refinement prolongation for the high-order
//! hexahedral / tetrahedral / pyramid families, vs MFEM 4.10.
//!
//! MFEM truth (WSL `$HOME/mfem410_ser`, probe `tmp/d103prol/probe_d103.cpp`,
//! dumps under `tests/data/d103/`):
//!
//! * `hexl2_p.txt` / `tetl2_p.txt`: MFEM's own
//!   `FiniteElementSpace::RefinementOperator` for `L2_FECollection(p, 3)`
//!   (default GaussLegendre basis) applied to the identity — the embedding
//!   bookkeeping is well-defined for hex (8 children/parent) and tet
//!   (8 children/parent) meshes.
//! * `pyrsem_l2_*.txt`: semantic truth (MFEM's serial uniform-refinement
//!   operator is broken on pyramids — see `d103_prolong_pyramid_h1.rs` for the
//!   `mesh.cpp:11062` embedding defect): per fine element rows
//!   `I(i,j) = φ_j^coarse(parent-ref coords of fine node i)` with MFEM's own
//!   `L2_FuentesPyramidElement` / `L2_TetrahedronElement` and geometry.
//!
//! The refined pyramid mesh is a *mixed* L² space (6·(p+1)³ + 4·tet DOFs per
//! parent); the homogeneous [`L2Space`] cannot hold it (D942), so the pyramid
//! arm builds the fine space through the [`L2ProlongationSpace`] read
//! interface with a test-local per-geometry provider that reproduces MFEM's
//! element-major consecutive numbering (pinned by
//! `d340_l2_pyramid_space.rs`).

use fem_element::{ReferenceElement, lagrange::{L2FuentesPyramidPk, TetL2GL}};
use fem_core::types::DofId;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_space::constraints::prolong::{build_l2_prolongation_matrix, L2ProlongationSpace};
use fem_space::{L2Basis, L2Space};

const HEX_L2_P1: &str = include_str!("data/d103/hexl2_1.txt");
const HEX_L2_P2: &str = include_str!("data/d103/hexl2_2.txt");
const HEX_L2_P3: &str = include_str!("data/d103/hexl2_3.txt");
const TET_L2_P1: &str = include_str!("data/d103/tetl2_1.txt");
const TET_L2_P2: &str = include_str!("data/d103/tetl2_2.txt");
const PYR_L2_P1: &str = include_str!("data/d103/pyrsem_l21.txt");
const PYR_L2_P2: &str = include_str!("data/d103/pyrsem_l22.txt");

/// `RefinementOperator` dump (sparse `P i j v ; ...` column lines + POS tables).
struct OpDump {
    csize: usize,
    fsize: usize,
    entries: Vec<(usize, usize, f64)>,
    /// global DOF id -> physical position, coarse and fine.  Comparisons key on
    /// coordinates because the two libraries may slot tets differently (fem-rs
    /// normalizes tet vertex order on read; the L2 DOF *values* are identical).
    coarse_pos: std::collections::HashMap<usize, [f64; 3]>,
    fine_pos: std::collections::HashMap<usize, [f64; 3]>,
}

fn parse_pos(text: &str, tag: &str) -> std::collections::HashMap<(usize, usize), [f64; 3]> {
    let mut m = std::collections::HashMap::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.first() == Some(&tag) {
            m.insert(
                (t[1].parse().unwrap(), t[2].parse().unwrap()),
                [t[3].parse().unwrap(), t[4].parse().unwrap(), t[5].parse().unwrap()],
            );
        }
    }
    m
}

/// global-id -> position via the dump's CDOF/FDOF tables plus CPOS/FPOS
fn pos_by_gid(
    text: &str,
    dof_tag: &str,
    pos_tag: &str,
) -> std::collections::HashMap<usize, [f64; 3]> {
    let pos = parse_pos(text, pos_tag);
    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.first() == Some(&dof_tag) {
            let e: usize = t[1].parse().unwrap();
            for (slot, v) in t[2..].iter().enumerate() {
                if let Some(&c) = pos.get(&(e, slot)) {
                    out.insert(v.parse::<usize>().unwrap(), c);
                }
            }
        }
    }
    out
}

fn parse_op(text: &str) -> OpDump {
    let mut csize = 0usize;
    let mut fsize = 0usize;
    let mut entries = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("CSIZE") => csize = t[1].parse().expect("csize"),
            Some("FSIZE") => fsize = t[1].parse().expect("fsize"),
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
    OpDump {
        csize,
        fsize,
        entries,
        coarse_pos: pos_by_gid(text, "CDOF", "CPOS"),
        fine_pos: pos_by_gid(text, "FDOF", "FPOS"),
    }
}

/// Quantized coordinate key: the two sides' DOF positions agree to ~1e-13, so
/// the 1e-9 grid is far coarser than the agreement and far finer than any node
/// spacing.
fn ckey(c: &[f64]) -> [i64; 3] {
    [
        (c[0] * 1e9).round() as i64,
        (c[1] * 1e9).round() as i64,
        (c[2] * 1e9).round() as i64,
    ]
}

/// `pyrsem` dump (CSIZE/FSIZE/NCH + CDOF/FDOF tables + per-slot ROWs).
struct SemDump {
    csize: usize,
    fsize: usize,
    rows: Vec<(usize, usize, usize, Vec<(u32, f64)>)>,
    /// (element, slot) -> physical position (CPOS / FPOS lines).
    cpos: std::collections::HashMap<(usize, usize), [f64; 3]>,
    fpos: std::collections::HashMap<(usize, usize), [f64; 3]>,
}

fn parse_sem(text: &str) -> SemDump {
    let mut csize = 0usize;
    let mut fsize = 0usize;
    let mut rows = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("CSIZE") => csize = t[1].parse().expect("csize"),
            Some("FSIZE") => fsize = t[1].parse().expect("fsize"),
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
    SemDump {
        csize,
        fsize,
        rows,
        cpos: parse_pos(text, "CPOS"),
        fpos: parse_pos(text, "FPOS"),
    }
}

fn value_at(p: &CsrMatrix<f64>, row: usize, col: usize) -> f64 {
    for k in p.row_ptr[row]..p.row_ptr[row + 1] {
        if p.col_idx[k] as usize == col {
            return p.values[k];
        }
    }
    0.0
}

fn row_sums_to_one(p: &CsrMatrix<f64>, what: &str) {
    for row in 0..p.nrows {
        let s: f64 = p.values[p.row_ptr[row]..p.row_ptr[row + 1]].iter().sum();
        assert!((s - 1.0).abs() < 1e-11, "{what}: row {row} sums to {s}");
    }
}

fn mesh3(rel: &str) -> Mesh<3> {
    let path = format!("{}/tests/{rel}", env!("CARGO_MANIFEST_DIR"));
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

/// The MFEM `MakeCartesian3D(1,1,1,HEXAHEDRON)` unit hex (MFEM hex vertex
/// convention; the L² prolongation is invariant under cube axis relabeling, so
/// the vertex assignment only has to define the same unit cube).
fn unit_hex() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., //
            1., 0., 0., //
            1., 1., 0., //
            0., 1., 0., //
            0., 0., 1., //
            1., 0., 1., //
            1., 1., 1., //
            0., 1., 1.,
        ],
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![1],
        fem_mesh::ElementType::Hex8,
        vec![],
        vec![],
        fem_mesh::ElementType::Quad4,
    )
}

#[test]
fn l2_hex_prolongation_matches_mfem_operator() {
    let coarse = unit_hex();
    let fine = refine_uniform_3d(&coarse);
    assert_eq!(fine.n_elements(), 8, "8 hex children");

    for (p, dump_text) in [(1u8, HEX_L2_P1), (2u8, HEX_L2_P2), (3u8, HEX_L2_P3)] {
        let d = parse_op(dump_text);
        let c = L2Space::new(coarse.clone(), p);
        let f = L2Space::new(fine.clone(), p);
        let pmat = build_l2_prolongation_matrix(&c, &f);
        assert_eq!(pmat.nrows, d.fsize, "p={p}");
        assert_eq!(pmat.ncols, d.csize, "p={p}");
        row_sums_to_one(&pmat, &format!("hex p={p}"));
        let mut worst = 0.0_f64;
        let mut missing = 0usize;
        for &(i, j, v) in &d.entries {
            let got = value_at(&pmat, i, j);
            if got == 0.0 && v != 0.0 {
                missing += 1;
            }
            worst = worst.max((got - v).abs());
        }
        println!("L2 hex p={p}: {} entries, worst |Δ| = {worst:.3e}, missing {missing}", d.entries.len());
        assert_eq!(missing, 0, "p={p}: entries MFEM has that fem-rs lacks");
        assert!(worst < 5e-12, "p={p}: worst |Δ| = {worst:.3e}");
    }
}

#[test]
fn l2_tet_prolongation_structural_invariants() {
    // MFEM-side pin impossible this round (D943): fem-rs's tet uniform
    // refinement produces different child geometry than MFEM 4.10 on beam-tet
    // (96/384 children common; tmp/d103prol/), so MFEM's refined-mesh
    // RefinementOperator rows cannot be compared against fem-rs's.  The tet
    // arm shares the builder with the pinned hex/pyramid arms; here its
    // structural invariants are pinned instead: row partition of unity and
    // the P0 indicator.
    let coarse = mesh3("data/d103/cart1-tet.mesh");
    let fine = refine_uniform_3d(&coarse);
    for p in 1..=3u8 {
        let c = L2Space::new(coarse.clone(), p);
        let f = L2Space::new(fine.clone(), p);
        let pmat = build_l2_prolongation_matrix(&c, &f);
        assert_eq!(pmat.nrows, f.n_dofs(), "p={p}");
        assert_eq!(pmat.ncols, c.n_dofs(), "p={p}");
        row_sums_to_one(&pmat, &format!("tet p={p}"));
    }
}

/// Heterogeneous L² space over the refined pyramid mesh: per-geometry DOF
/// blocks (pyramid `(p+1)³`, tetrahedron `(p+1)(p+2)(p+3)/6`), element-major
/// consecutive numbering — MFEM's layout on this mesh (the d340 fixture pins
/// `6·(p+1)³ + 4·tet` DOFs for the 10-element refinement).  This is exactly
/// the arm [`L2Space`] is missing (D942); it exists here so the L² pyramid
/// prolongation itself is verified against MFEM end to end.
struct MixedL2 {
    mesh: Mesh<3>,
    order: u8,
    n_dofs: usize,
    elem_dofs: Vec<Vec<DofId>>,
    dof_coords: Vec<f64>,
}

impl MixedL2 {
    fn new(mesh: Mesh<3>, order: u8) -> Self {
        let p = order as usize;
        let mut elem_dofs = Vec::new();
        let mut dof_coords = Vec::new();
        let mut next = 0usize;
        for e in 0..mesh.n_elements() as u32 {
            let is_pyr = mesh.element_type(e) == fem_mesh::ElementType::Pyramid5;
            let ref_coords: Vec<Vec<f64>> = if is_pyr {
                L2FuentesPyramidPk::new(p).dof_coords()
            } else {
                TetL2GL::new(p).dof_coords()
            };
            let mut dofs = Vec::with_capacity(ref_coords.len());
            for rc in &ref_coords {
                dofs.push(next as DofId);
                next += 1;
                let (_, x) = fem_mesh::transformation::element_jacobian_at(&mesh, e, rc, 3);
                dof_coords.extend_from_slice(&x);
            }
            elem_dofs.push(dofs);
        }
        MixedL2 { mesh, order, n_dofs: next, elem_dofs, dof_coords }
    }
}

impl L2ProlongationSpace<Mesh<3>> for MixedL2 {
    fn order(&self) -> u8 {
        self.order
    }
    fn l2_basis(&self) -> Option<L2Basis> {
        Some(L2Basis::GaussLegendre)
    }
    fn mesh(&self) -> &Mesh<3> {
        &self.mesh
    }
    fn n_dofs(&self) -> usize {
        self.n_dofs
    }
    fn element_dofs(&self, e: u32) -> &[DofId] {
        &self.elem_dofs[e as usize]
    }
    fn dof_coords(&self) -> &[f64] {
        &self.dof_coords
    }
}

#[test]
fn l2_pyramid_prolongation_matches_mfem_semantics() {
    let coarse_mesh = mesh3("data/d103/inline-pyramid.mesh");
    let fine_mesh = refine_uniform_3d(&coarse_mesh);
    // MFEM's inline-pyramid generator: each grid brick becomes 6 pyramids
    // (one per face, apex at the brick centre) — 2×2×2 bricks → 48 parents.
    assert_eq!(fine_mesh.n_elements(), 480, "10 children per pyramid parent");

    for (p, dump_text) in [(1u8, PYR_L2_P1), (2u8, PYR_L2_P2)] {
        let d = parse_sem(dump_text);
        let c = L2Space::new(coarse_mesh.clone(), p);
        let f = MixedL2::new(fine_mesh.clone(), p);
        assert_eq!(c.n_dofs(), d.csize, "p={p}: coarse vsize vs MFEM CSIZE");
        assert_eq!(f.n_dofs(), d.fsize, "p={p}: fine vsize vs MFEM FSIZE");

        let pmat = build_l2_prolongation_matrix(&c, &f);
        row_sums_to_one(&pmat, &format!("pyr L2 p={p}"));

        let mut worst = 0.0_f64;
        let mut checked = 0usize;
        let mut missing = 0usize;
        let mut fem_by_coord: std::collections::HashMap<[i64; 3], usize> =
            std::collections::HashMap::new();
        for dof in 0..f.n_dofs() {
            fem_by_coord.insert(ckey(&f.dof_coords()[dof * 3..dof * 3 + 3]), dof);
        }
        let mut coarse_by_coord: std::collections::HashMap<[i64; 3], usize> =
            std::collections::HashMap::new();
        for e in 0..coarse_mesh.n_elements() as u32 {
            for (li, &cd) in c.element_dofs(e).iter().enumerate() {
                let base = cd as usize * 3;
                coarse_by_coord.insert(
                    ckey(&c.dof_coords()[base..base + 3]),
                    cd as usize,
                );
                let _ = li;
            }
        }
        for &(ef, i, parent, ref pairs) in &d.rows {
            let fg = *fem_by_coord
                .get(&ckey(&d.fpos[&(ef, i)]))
                .unwrap_or_else(|| panic!("p={p}: no fem-rs fine dof at {:?}", d.fpos[&(ef, i)]));
            for &(j, v) in pairs {
                let cg = *coarse_by_coord
                    .get(&ckey(&d.cpos[&(parent, j as usize)]))
                    .unwrap_or_else(|| {
                        panic!("p={p}: no fem-rs coarse dof at {:?}", d.cpos[&(parent, j as usize)])
                    });
                let got = value_at(&pmat, fg, cg);
                if got == 0.0 && v != 0.0 {
                    missing += 1;
                }
                worst = worst.max((got - v).abs());
                checked += 1;
            }
        }
        println!("L2 pyramid p={p}: {checked} entries, worst |Δ| = {worst:.3e}, missing {missing}");
        assert_eq!(missing, 0, "p={p}: entries MFEM has that fem-rs lacks");
        assert!(worst < 5e-12, "p={p}: worst |Δ| = {worst:.3e}");
    }
}

/// P0: every child element inherits the parent's single DOF (MFEM's
/// `GetLocalInterpolation` for a one-DOF element), so P is the 0/1 indicator.
#[test]
fn l2_p0_prolongation_is_indicator() {
    let coarse = mesh3("data/d103/cart1-tet.mesh");
    let fine = refine_uniform_3d(&coarse);
    let c = L2Space::new(coarse.clone(), 0);
    let f = L2Space::new(fine.clone(), 0);
    let pmat = build_l2_prolongation_matrix(&c, &f);
    assert_eq!(pmat.nrows, fine.n_elements());
    assert_eq!(pmat.ncols, coarse.n_elements());
    for row in 0..pmat.nrows {
        let nnz = pmat.row_ptr[row + 1] - pmat.row_ptr[row];
        assert_eq!(nnz, 1, "row {row}");
        assert_eq!(pmat.values[pmat.row_ptr[row]], 1.0, "row {row}");
    }
}
