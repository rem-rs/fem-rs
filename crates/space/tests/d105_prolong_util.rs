//! Shared dump parsers / comparators for the d105 L²-prolongation pin tests
//! (D940/D941/D942).  Dump formats are the d103 probe's (tag-prefixed lines),
//! extended with physical DOF position tables (`CPOS`/`FPOS`) so comparisons
//! key on `(element index, quantized position)` — the common ground when slot
//! conventions may differ between the libraries.

use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_space::constraints::prolong::L2ProlongationSpace;

type HashMap<K, V> = std::collections::HashMap<K, V>;

/// Quantized coordinate key: the two sides' DOF positions agree to ~1e-13, so
/// the 1e-9 grid is far coarser than the agreement and far finer than any node
/// spacing.  `dim` leading components (2-D dumps carry z = 0).
fn ckey(pos: &[f64; 3], dim: usize) -> Vec<i64> {
    pos[..dim].iter().map(|v| (v * 1e9).round() as i64).collect()
}

/// Per-element quantized position -> fem-rs dof table.
///
/// L² DOFs are element-local, so the key is `(element index, position)`: a
/// **closed** (GaussLobatto) node placement puts corner / edge / face DOFs of
/// *different* elements at the same physical position, and a global position
/// map would collapse them onto one (wrong) element's dof.  The element index
/// is common ground because the fine meshes are the same element sequences
/// (the pyramid refinement plan even asserts MFEM's emission order, D472).
fn element_coord_index<M: MeshTopology, S: L2ProlongationSpace<M>>(
    space: &S,
) -> HashMap<(usize, Vec<i64>), usize> {
    let dim = space.mesh().dim() as usize;
    let mut m = HashMap::new();
    for e in 0..space.mesh().n_elements() as u32 {
        for &dof in space.element_dofs(e) {
            let base = dof as usize * dim;
            let pos = [
                space.dof_coords()[base],
                space.dof_coords()[base + 1],
                if dim == 3 { space.dof_coords()[base + 2] } else { 0.0 },
            ];
            m.insert((e as usize, ckey(&pos, dim)), dof as usize);
        }
    }
    m
}

/// global id -> (element, slot) from a dump's CDOF/FDOF table.
fn gid_slot(tables: &[Vec<u32>]) -> HashMap<u32, (usize, usize)> {
    let mut m = HashMap::new();
    for (e, dofs) in tables.iter().enumerate() {
        for (s, &g) in dofs.iter().enumerate() {
            m.insert(g, (e, s));
        }
    }
    m
}

/// `RefinementOperator` dump: sparse `P i j v ; …` column lines + CDOF/FDOF
/// tables + CPOS/FPOS physical DOF positions.
pub struct OpDump {
    pub csize: usize,
    pub fsize: usize,
    pub entries: Vec<(usize, usize, f64)>,
    pub cdof: Vec<Vec<u32>>,
    pub fdof: Vec<Vec<u32>>,
    pub cpos: HashMap<(usize, usize), [f64; 3]>,
    pub fpos: HashMap<(usize, usize), [f64; 3]>,
}

fn parse_pos(text: &str, tag: &str) -> HashMap<(usize, usize), [f64; 3]> {
    let mut m = HashMap::new();
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

fn parse_dof_table(text: &str, tag: &str) -> Vec<Vec<u32>> {
    let mut v = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.first() == Some(&tag) {
            v.push(t[2..].iter().map(|s| s.parse().unwrap()).collect());
        }
    }
    v
}

pub fn parse_op(text: &str) -> OpDump {
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
        cdof: parse_dof_table(text, "CDOF"),
        fdof: parse_dof_table(text, "FDOF"),
        cpos: parse_pos(text, "CPOS"),
        fpos: parse_pos(text, "FPOS"),
    }
}

/// Semantic dump (CSIZE/FSIZE + CDOF + CPOS/FPOS + per-slot `ROW`s): the
/// d103 `pyrsem` format, also used by the d105 `tetsem` / `pyrl2sem` modes.
pub struct SemDump {
    pub csize: usize,
    pub fsize: usize,
    pub cdof: Vec<Vec<u32>>,
    /// (fine element, fine slot, parent element, [(parent slot, value)])
    pub rows: Vec<(usize, usize, usize, Vec<(u32, f64)>)>,
    pub cpos: HashMap<(usize, usize), [f64; 3]>,
    pub fpos: HashMap<(usize, usize), [f64; 3]>,
}

pub fn parse_sem(text: &str) -> SemDump {
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
        cdof: parse_dof_table(text, "CDOF"),
        rows,
        cpos: parse_pos(text, "CPOS"),
        fpos: parse_pos(text, "FPOS"),
    }
}

pub fn value_at(p: &CsrMatrix<f64>, row: usize, col: usize) -> f64 {
    for k in p.row_ptr[row]..p.row_ptr[row + 1] {
        if p.col_idx[k] as usize == col {
            return p.values[k];
        }
    }
    0.0
}

pub fn row_sums_to_one(p: &CsrMatrix<f64>, what: &str) {
    for row in 0..p.nrows {
        let s: f64 = p.values[p.row_ptr[row]..p.row_ptr[row + 1]].iter().sum();
        assert!((s - 1.0).abs() < 1e-11, "{what}: row {row} sums to {s}");
    }
}

/// Compare a `RefinementOperator` dump against the fem-rs matrix, keying rows
/// and columns on `(element index, DOF position)` via the dump's FDOF/CDOF +
/// FPOS/CPOS tables.  Returns `(checked, worst |Δ|, missing)`; asserts no DOF
/// position is unmatched and no entry is missing before returning.
pub fn compare_op_coord_keyed<
    M: MeshTopology,
    SF: L2ProlongationSpace<M>,
    SC: L2ProlongationSpace<M>,
>(
    d: &OpDump,
    pmat: &CsrMatrix<f64>,
    f: &SF,
    c: &SC,
    label: &str,
) -> (usize, f64, usize) {
    let dim = f.mesh().dim() as usize;
    let f_by_coord = element_coord_index(f);
    let c_by_coord = element_coord_index(c);
    let fidx = gid_slot(&d.fdof);
    let cidx = gid_slot(&d.cdof);

    let mut worst = 0.0_f64;
    let mut checked = 0usize;
    let mut missing = 0usize;
    let mut unmatched = 0usize;
    for &(i, j, v) in &d.entries {
        let f_hit = fidx.get(&(i as u32)).and_then(|&(fe, s)| {
            d.fpos.get(&(fe, s)).and_then(|pos| f_by_coord.get(&(fe, ckey(pos, dim))))
        });
        let c_hit = cidx.get(&(j as u32)).and_then(|&(ce, s)| {
            d.cpos.get(&(ce, s)).and_then(|pos| c_by_coord.get(&(ce, ckey(pos, dim))))
        });
        let (Some(&fg), Some(&cg)) = (f_hit, c_hit) else {
            unmatched += 1;
            continue;
        };
        let got = value_at(pmat, fg, cg);
        if got == 0.0 && v != 0.0 {
            missing += 1;
        }
        worst = worst.max((got - v).abs());
        checked += 1;
    }
    println!(
        "{label}: {checked} entries, worst |Δ| = {worst:.3e}, missing {missing}, unmatched {unmatched}"
    );
    assert_eq!(unmatched, 0, "{label}: DOF positions fem-rs does not have");
    assert_eq!(missing, 0, "{label}: entries MFEM has that fem-rs lacks");
    (checked, worst, missing)
}

/// Compare a semantic dump (`ROW` lines) against the fem-rs matrix,
/// `(element, position)`-keyed like [`compare_op_coord_keyed`].
pub fn compare_sem_coord_keyed<
    M: MeshTopology,
    SF: L2ProlongationSpace<M>,
    SC: L2ProlongationSpace<M>,
>(
    d: &SemDump,
    pmat: &CsrMatrix<f64>,
    f: &SF,
    c: &SC,
    label: &str,
) -> (usize, f64, usize) {
    let dim = f.mesh().dim() as usize;
    let f_by_coord = element_coord_index(f);
    let c_by_coord = element_coord_index(c);

    let mut worst = 0.0_f64;
    let mut checked = 0usize;
    let mut missing = 0usize;
    let mut unmatched = 0usize;
    for &(ef, i, parent, ref pairs) in &d.rows {
        let Some(&fg) = d
            .fpos
            .get(&(ef, i))
            .and_then(|pos| f_by_coord.get(&(ef, ckey(pos, dim))))
        else {
            unmatched += 1;
            continue;
        };
        for &(j, v) in pairs {
            let Some(&cg) = d
                .cpos
                .get(&(parent, j as usize))
                .and_then(|pos| c_by_coord.get(&(parent, ckey(pos, dim))))
            else {
                missing += 1;
                unmatched += 1;
                continue;
            };
            let got = value_at(pmat, fg, cg);
            if got == 0.0 && v != 0.0 {
                missing += 1;
            }
            worst = worst.max((got - v).abs());
            checked += 1;
        }
    }
    println!(
        "{label}: {checked} entries, worst |Δ| = {worst:.3e}, missing {missing}, unmatched {unmatched}"
    );
    assert_eq!(unmatched, 0, "{label}: DOF positions fem-rs does not have");
    assert_eq!(missing, 0, "{label}: entries MFEM has that fem-rs lacks");
    (checked, worst, missing)
}

pub fn mesh2(rel: &str) -> fem_mesh::Mesh<2> {
    let path = format!("{}/tests/{rel}", env!("CARGO_MANIFEST_DIR"));
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    mfem.mesh2d.unwrap_or_else(|| panic!("{rel} must be a 2-D mesh"))
}

pub fn mesh3(rel: &str) -> fem_mesh::Mesh<3> {
    let path = format!("{}/tests/{rel}", env!("CARGO_MANIFEST_DIR"));
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}
