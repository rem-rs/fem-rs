//! D122-3 (round 73) — the parallel NDk (k ≥ 2) assembled mass matrix must be
//! np-invariant.
//!
//! `Σ_ij M_ij` (`= 1ᵀM1 = ∫|Σ_i φ_i|²`) is a functional of the *operator*, not
//! of the DOF numbering or the ownership split, so it must be identical at every
//! rank count and equal to the serial assembly.  Before this round the
//! H(curl) families failed it at np ≥ 2 (ND2: 6.0e-3, ND3: 1.6e-4) while
//! RT0/RT1/RT2/L2/ND1 were exact.
//!
//! ## Root cause (measured, see `tmp/d122r73/README.md`)
//!
//! The local sub-mesh is ordered **owned elements first, ghosts after**
//! (`par_partition`), so the *first* element of the local traversal that touches
//! a shared face differs between ranks.  `HCurlSpace` fixes a shared face's
//! canonical DOF functions ("anchor") from that face-**creating** element, and
//! the DP's global convention for a face DOF is the *minimum-global-element*
//! one (D412).  When a rank's creator is not the minimum-global element its
//! canonical face block is a **signed permutation** of the global one, which
//! the parallel permutation has to absorb — but `from_edge_space_ordered` only
//! writes `sign_corrections` for **edge** DOFs and leaves the face DOFs at
//! `+1.0`.  Rows/columns of the wrongly-signed face DOFs therefore land in the
//! matrix with the wrong sign, and `Σ_ij M_ij` moves.
//!
//! The diagnostic below demonstrates it directly: it maps every rank's local
//! assembly into the global DOF numbering (through the same permutation the
//! production path uses) and compares the two ranks entry by entry; before the
//! fix the disagreeing entries are exactly the face-DOF rows/columns of the
//! faces whose creating element differs across the partition boundary.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fem_assembly::standard::{MassIntegrator, VectorMassIntegrator};
use fem_assembly::{Assembler, VectorAssembler};
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::par_vector_assembler::ParVectorAssembler;
use fem_parallel::WorkerConfig;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

fn cyl_hex() -> Mesh<3> {
    let path = format!("{}/../../data/cylinder-hex.mesh", env!("CARGO_MANIFEST_DIR"));
    let m = fem_io::mfem::read_mfem_file(&path).expect("cylinder-hex.mesh");
    m.mesh3d.expect("cylinder-hex.mesh is 3-D")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fam {
    H1o2,
    L2o1,
    ND1,
    ND2,
    ND3,
    RT0,
    RT1,
    RT2,
}

const ALL_FAMS: [Fam; 8] = [
    Fam::H1o2,
    Fam::L2o1,
    Fam::ND1,
    Fam::ND2,
    Fam::ND3,
    Fam::RT0,
    Fam::RT1,
    Fam::RT2,
];

impl Fam {
    fn name(self) -> &'static str {
        match self {
            Fam::H1o2 => "H1o2",
            Fam::L2o1 => "L2o1",
            Fam::ND1 => "ND1",
            Fam::ND2 => "ND2",
            Fam::ND3 => "ND3",
            Fam::RT0 => "RT0",
            Fam::RT1 => "RT1",
            Fam::RT2 => "RT2",
        }
    }
}

/// `Σ_ij M_ij` of the family's mass matrix at one rank count (`None` =
/// serial, unpartitioned assembly).
fn entry_sum(fam: Fam, n_ranks: Option<usize>) -> f64 {
    let mesh = cyl_hex();
    let integ = VectorMassIntegrator { alpha: 1.0 };
    match n_ranks {
        None => match fam {
            Fam::H1o2 => {
                // Scalar families use the scalar assembler; the mass matrix is
                // the same operator either way.
                let s = H1Space::new(mesh, 2);
                Assembler::assemble_bilinear(&s, &[&MassIntegrator { rho: 1.0 }], 6)
                    .values
                    .iter()
                    .sum()
            }
            Fam::L2o1 => {
                let s = L2Space::new(mesh, 1);
                Assembler::assemble_bilinear(&s, &[&MassIntegrator { rho: 1.0 }], 6)
                    .values
                    .iter()
                    .sum()
            }
            Fam::RT0 | Fam::RT1 | Fam::RT2 => {
                let k = match fam {
                    Fam::RT0 => 0u8,
                    Fam::RT1 => 1,
                    _ => 2,
                };
                VectorAssembler::assemble_bilinear(&HDivSpace::new(mesh, k), &[&integ], 6)
                    .values
                    .iter()
                    .sum()
            }
            _ => {
                let k = match fam {
                    Fam::ND1 => 1u8,
                    Fam::ND2 => 2,
                    _ => 3,
                };
                VectorAssembler::assemble_bilinear(&HCurlSpace::new(mesh, k), &[&integ], 6)
                    .values
                    .iter()
                    .sum()
            }
        },
        Some(n_ranks) => {
            let out: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
            let out_rank = Arc::clone(&out);
            ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let lm = pmesh.local_mesh().clone();
                let local: f64 = match fam {
                    Fam::H1o2 => {
                        let s = ParallelFESpace::new(H1Space::new(lm, 2), &pmesh, comm.clone());
                        let m = fem_parallel::par_assembler::ParAssembler::assemble_bilinear(
                            &s,
                            &[&MassIntegrator { rho: 1.0 }],
                            6,
                        );
                        m.diag_block().values.iter().sum::<f64>()
                            + m.offd_block().values.iter().sum::<f64>()
                    }
                    Fam::L2o1 => {
                        let s = ParallelFESpace::new(L2Space::new(lm, 1), &pmesh, comm.clone());
                        let m = fem_parallel::par_assembler::ParAssembler::assemble_bilinear(
                            &s,
                            &[&MassIntegrator { rho: 1.0 }],
                            6,
                        );
                        m.diag_block().values.iter().sum::<f64>()
                            + m.offd_block().values.iter().sum::<f64>()
                    }
                    Fam::RT0 | Fam::RT1 | Fam::RT2 => {
                        let k = match fam {
                            Fam::RT0 => 0u8,
                            Fam::RT1 => 1,
                            _ => 2,
                        };
                        let s =
                            ParallelFESpace::new(HDivSpace::new(lm, k), &pmesh, comm.clone());
                        let m = ParVectorAssembler::assemble_bilinear(&s, &[&integ], 6);
                        m.diag_block().values.iter().sum::<f64>()
                            + m.offd_block().values.iter().sum::<f64>()
                    }
                    _ => {
                        let k = match fam {
                            Fam::ND1 => 1u8,
                            Fam::ND2 => 2,
                            _ => 3,
                        };
                        let s =
                            ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
                        let m = ParVectorAssembler::assemble_bilinear(&s, &[&integ], 6);
                        m.diag_block().values.iter().sum::<f64>()
                            + m.offd_block().values.iter().sum::<f64>()
                    }
                };
                *out_rank.lock().unwrap() += local;
            });
            let r = *out.lock().unwrap();
            r
        }
    }
}

/// The teeth: `Σ_ij M_ij` of the mass matrix is `∫|Σ_i φ_i|²` — a fixed number —
/// so the serial assembly, `np = 1`, `np = 2` and `np = 4` must all agree to
/// round-off.  Red before this round for ND2/ND3 (and any k ≥ 2 H(curl) space).
#[test]
fn d122r73_mass_matrix_entry_sum_is_np_invariant() {
    let mut bad = Vec::new();
    let mut table = Vec::new();
    for fam in ALL_FAMS {
        let ser = entry_sum(fam, None);
        let np1 = entry_sum(fam, Some(1));
        table.push(format!("{:<5} serial={ser:.17e} np1={np1:.17e}", fam.name()));
        if (ser - np1).abs() > 1e-12 * ser.abs() {
            bad.push(format!(
                "{} np=1: serial {ser} vs {np1} (rel {:.3e})",
                fam.name(),
                (ser - np1).abs() / ser.abs()
            ));
        }
        for n_ranks in [2usize, 4] {
            let npn = entry_sum(fam, Some(n_ranks));
            table.push(format!(
                "{:<5} np{n_ranks}={npn:.17e} (rel {:.3e})",
                fam.name(),
                (ser - npn).abs() / ser.abs()
            ));
            if (ser - npn).abs() > 1e-12 * ser.abs() {
                bad.push(format!(
                    "{} np={n_ranks}: serial {ser} vs {npn} (rel {:.3e})",
                    fam.name(),
                    (ser - npn).abs() / ser.abs()
                ));
            }
        }
    }
    for line in &table {
        println!("D122R73SUM {line}");
    }
    assert!(
        bad.is_empty(),
        "the assembled mass matrix is not np-invariant:\n  {}",
        bad.join("\n  ")
    );
}

/// D122-3 teeth, per entry: every rank's local assembly mapped into the global
/// DOF numbering (through the same permutation + sign corrections the production
/// path applies) must agree **entry by entry** — the parallel operator is one
/// matrix, not one matrix per rank.  Red before the face-DOF sign corrections
/// (24012 sign-flipped ordered pairs at np = 2, all of them face-DOF rows or
/// columns of a face whose creating element differs across the partition
/// boundary), green after.
#[test]
fn d122r73_cross_rank_matrix_entries_agree() {
    /// ND2's serial numbering split (MFEM `FiniteElementSpace::Construct`:
    /// vertices, edges, faces, interior) — used only for the failure report.
    const N_EDGES: u32 = 1938;
    const N_FACES: u32 = 3432;

    fn classify(gid: u32) -> &'static str {
        if gid < N_EDGES {
            "edge"
        } else if gid < N_EDGES + N_FACES {
            "face"
        } else {
            "interior"
        }
    }

    let mesh = cyl_hex();
    for n_ranks in [2usize, 4] {
        let mesh = mesh.clone();
        // D807-1: a rank's *ghost rows* are no longer a complete assembly of the
        // global row — the ghost layer is the anchor closure, not the whole mesh,
        // so an element that couples two ghost dofs one hop out may be missing on
        // one rank (`tmp/d807r76/`).  The entries whose assembly IS complete on
        // every rank are those whose two dofs both sit in *owned* elements: all
        // holders of an entity carried by an owned element are node neighbours of
        // an owned element, hence local (the D790-2 lemma) — so each rank sums
        // exactly the same element contributions.  The comparison is therefore
        // restricted to the intersection of the ranks' owned-element dof sets,
        // which is what makes the values comparable (and is still the whole
        // operator: 263576 of 553418 common entries at np = 2 are significant).
        type RankMap = (usize, HashMap<(u32, u32), f64>, std::collections::HashSet<u32>);
        let maps: Arc<Mutex<Vec<RankMap>>> = Arc::new(Mutex::new(Vec::new()));
        let maps_rank = Arc::clone(&maps);
        ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
            let rank = comm.rank() as usize;
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let s = ParallelFESpace::new(HCurlSpace::new(lm, 2), &pmesh, comm.clone());
            let dp = s.dof_partition();
            let local = VectorAssembler::assemble_bilinear(
                s.local_space(),
                &[&VectorMassIntegrator { alpha: 1.0 }],
                6,
            );
            let n_owned_elems = pmesh.partition().n_owned_elems;
            let mut in_owned_elem = vec![false; s.local_space().n_dofs()];
            for e in 0..n_owned_elems as u32 {
                for &d in s.local_space().element_dofs(e) {
                    in_owned_elem[d as usize] = true;
                }
            }
            let mut complete_gids: std::collections::HashSet<u32> =
                std::collections::HashSet::new();
            let mut m: HashMap<(u32, u32), f64> = HashMap::new();
            for row in 0..local.nrows {
                let pr = dp.permute_dof(row as u32);
                let dr = dp.sign_correction(row as u32);
                let row_owned = in_owned_elem[row];
                if row_owned {
                    complete_gids.insert(dp.global_dof(pr));
                }
                for k in local.row_ptr[row]..local.row_ptr[row + 1] {
                    let col = local.col_idx[k] as usize;
                    if !(row_owned && in_owned_elem[col]) {
                        continue;
                    }
                    let pc = dp.permute_dof(col as u32);
                    let dc = dp.sign_correction(col as u32);
                    let v = local.values[k] * dr * dc;
                    if v != 0.0 {
                        *m.entry((dp.global_dof(pr), dp.global_dof(pc))).or_insert(0.0) += v;
                    }
                }
            }
            maps_rank.lock().unwrap().push((rank, m, complete_gids));
        });
        let maps = maps.lock().unwrap().clone();
        let (mut diff, mut common, mut significant) = (Vec::new(), 0usize, 0usize);
        for (i, (_, mi, gi)) in maps.iter().enumerate() {
            for (_, mj, gj) in maps.iter().skip(i + 1) {
                for (k, v0) in mi {
                    // Both dofs must be complete on *both* ranks (see above).
                    if !(gi.contains(&k.0) && gi.contains(&k.1)
                        && gj.contains(&k.0) && gj.contains(&k.1))
                    {
                        continue;
                    }
                    // D807-1 tolerance: a rank's *ghost rows* are no longer a
                    // complete assembly of the global row — the ghost layer is
                    // the anchor closure, not the whole mesh, so an element
                    // coupling two ghost dofs one hop out may be missing on one
                    // rank.  The affected entries are the ones that cancel to
                    // roundoff (|v| <= 1e-16, i.e. 14+ orders below the mass
                    // matrix's O(1e-2) entries); the *physical* operator is
                    // pinned by `d122_assembled_entry_sums_match_the_serial_assembly`
                    // and by the per-entity operator comparisons, which compare
                    // against the serial assembly rather than rank-to-rank.
                    if let Some(v1) = mj.get(k) {
                        let tol = 1e-12 * v0.abs().max(v1.abs()) + 1e-16;
                        common += 1;
                        if v0.abs() > 1e-12 {
                            significant += 1;
                        }
                        if (v0 - v1).abs() > tol {
                            diff.push(format!(
                                "np={n_ranks} [{}, {}] ({:?} x {:?}): {v0:.17e} vs {v1:.17e}",
                                k.0,
                                k.1,
                                classify(k.0),
                                classify(k.1)
                            ));
                        }
                    } else if v0.abs() > 1e-16 {
                        diff.push(format!("np={n_ranks}: ({}, {}) missing on a rank", k.0, k.1));
                    }
                }
            }
        }
        assert!(
            !diff.is_empty() || common > 0,
            "np={n_ranks}: the ranks share no entries at all"
        );
        // Non-vacuity: the comparison must actually cover the operator, not just
        // a halo of roundoff-zero entries.
        assert!(
            significant > 1000,
            "np={n_ranks}: only {significant} significant entries compared"
        );
        println!("np={n_ranks}: {common} common entries, {significant} significant, {} diffs", diff.len());
        assert!(
            diff.is_empty(),
            "np={n_ranks}: the ranks disagree on {} of {common} common entries:\n  {}",
            diff.len(),
            diff.iter().take(5).cloned().collect::<Vec<_>>().join("\n  ")
        );
    }
}
