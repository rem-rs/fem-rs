//! D963: the parallel ultraweak DPG system assembled at `--ranks 2` must be
//! **row-for-row identical** (in global-DOF space) to the `--ranks 1` system.
//!
//! Regression context: after the D807-1 ghost-layer thinning the pdiffusion
//! miniapp silently produced a wrong solution at two ranks.  This probe dumps
//! every owned row of `(A, b)` — the raw uncondensed system formed with an
//! empty essential list, entries keyed by absolute global DOF ids — for the
//! pdiffusion `-prob 0` block table on `unit_square_quad(4)` and diffs the two
//! decompositions.  Per-`(row, col)` values are single-element contributions
//! (each test DOF's row is that element's equation), so the row spaces and
//! the values must agree exactly; any divergence localizes the defect to the
//! rank-local assembly or the numbering rather than the solver.

use std::fs;
use std::sync::{Arc, Mutex};

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgDomainLFIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTraceIntegrator, DpgVectorFEMassIntegrator,
};
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_dpg_weakform::ParDpgWeakForm;
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::WorkerConfig;

const PI: f64 = std::f64::consts::PI;

fn f_exact(x: &[f64]) -> f64 {
    let a = PI * (x[0] + x[1]);
    PI * PI * a.sin() * x.len() as f64
}

/// One dumped entry: `(global row, global col, value)`; RHS entries use the
/// sentinel column `u32::MAX`.
type Entry = (u32, u32, f64);

fn dump_system(n_ranks: usize, tag: &str) {
    let mesh = Mesh::<2>::unit_square_quad(4);
    let dump_dir = format!("{}/d963dump-{tag}", env!("CARGO_TARGET_TMPDIR"));
    let _ = fs::remove_dir_all(&dump_dir);
    fs::create_dir_all(&dump_dir).unwrap();
    let dump_dir = Arc::new(dump_dir);
    let mesh = Arc::new(mesh);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh_identity(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();

        let mut a = ParDpgWeakForm::new(local_mesh, partition, comm.clone());
        let test_order: u8 = 2;
        let p: u8 = 1;
        a.set_quad_order(2 * test_order);
        a.set_face_quad_order(test_order + p - 1);

        let u = a.add_trial_scalar_space(p - 1);
        let sig = a.add_trial_vector_space(p - 1, 2);
        let hatu = a.add_trial_trace_space_h1(p);
        let hatsig = a.add_trial_trace_space(p - 1);
        let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
        let v = a.add_test_space(VolKind::Scalar, test_order);

        a.add_trial_integrator(
            Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
            u,
            tau,
        );
        a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
        a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
        a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
        a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);
        a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
        a.add_test_integrator(Box::new(DpgVectorFEMassIntegrator { q: 1.0 }), tau, tau);
        a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
        a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
        a.add_domain_lf_integrator(
            Box::new(DpgDomainLFIntegrator { f: |x: &[f64]| f_exact(x) }),
            v,
        );

        a.store_matrices(true);
        a.assemble();

        // Same essential pipeline as the miniapp: the first D963 defect is
        // exercised only when the essential list is applied (it over-marked
        // partition interfaces as essential).
        let hatu = 2; // add order: u, σ, û, σ̂ — the H1-trace block index
        let pairs = a.trace_boundary_dofs(hatu);
        let merged = a.merge_dof_points(&pairs);
        let ess_ids: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();
        let mut x_local = vec![0.0_f64; a.local().size()];
        a.fill_essential_values(&mut x_local, &merged, &|pt| (PI * (pt[0] + pt[1])).sin());
        let (sys, _, _) = a.form_linear_system(&ess_ids, &x_local);

        let owned_global: Vec<u32> = a.owned_global_ids().to_vec();
        let ghost_global: Vec<u32> = a.ghost_global_ids().to_vec();
        let local = sys.a.to_local_matrix();
        let b = sys.b.as_slice();

        let mut entries: Vec<Entry> = Vec::new();
        for r in 0..local.nrows {
            let grow = owned_global[r];
            for k in local.row_ptr[r]..local.row_ptr[r + 1] {
                let c = local.col_idx[k] as usize;
                let gcol = if c < owned_global.len() {
                    owned_global[c]
                } else {
                    ghost_global[c - owned_global.len()]
                };
                entries.push((grow, gcol, local.values[k]));
            }
            entries.push((grow, u32::MAX, b[r]));
        }
        entries.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        let path = format!("{dump_dir}/rank{rank}.txt");
        let mut text = String::new();
        for (g, c, v) in &entries {
            text.push_str(&format!("{g} {c} {v:.17e}\n"));
        }
        fs::write(path, text).unwrap();
    });
}

fn load_entries(tag: &str) -> Vec<Entry> {
    let dump_dir = format!("{}/d963dump-{tag}", env!("CARGO_TARGET_TMPDIR"));
    let mut all: Vec<Entry> = Vec::new();
    for entry in fs::read_dir(&dump_dir).unwrap() {
        let path = entry.unwrap().path();
        for line in fs::read_to_string(&path).unwrap().lines() {
            let mut it = line.split_whitespace();
            let g: u32 = it.next().unwrap().parse().unwrap();
            let c: u32 = it.next().unwrap().parse().unwrap();
            let v: f64 = it.next().unwrap().parse().unwrap();
            all.push((g, c, v));
        }
    }
    all.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    all
}

/// Merge ranks' dumps into one global system: rows are disjoint across ranks
/// (each global row is owned by exactly one rank); duplicate `(row, col)`
/// pairs within a rank must not exist for this block table.
fn assert_systems_match() {
    let one = load_entries("np1");
    let two = load_entries("np2");

    let rows_of = |e: &[Entry]| -> std::collections::BTreeSet<u32> { e.iter().map(|e| e.0).collect() };
    let r1 = rows_of(&one);
    let r2 = rows_of(&two);
    assert_eq!(r1, r2, "row spaces differ between np1 and np2");

    let map: std::collections::HashMap<(u32, u32), f64> =
        one.iter().map(|&(g, c, v)| ((g, c), v)).collect();
    assert_eq!(map.len(), one.len(), "np1 has duplicate (row, col) pairs");
    let mut worst_rel = 0.0_f64;
    let mut worst_key = None;
    let mut missing = 0usize;
    for &(g, c, v) in &two {
        match map.get(&(g, c)) {
            Some(&v1) => {
                let rel = if v1 == v {
                    0.0
                } else {
                    ((v - v1) / v1.abs().max(1e-300)).abs()
                };
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_key = Some((g, c));
                }
            }
            None => missing += 1,
        }
    }
    assert_eq!(missing, 0, "np2 has {missing} entries absent from np1");
    assert!(
        worst_rel == 0.0,
        "np2 entry value diverges from np1: worst rel {worst_rel:e} at {worst_key:?}"
    );
}

/// `--ranks 1` baseline dump + compare against a fresh `--ranks 2` dump.
/// (Single `#[test]`: the dumps live in per-tag temp dirs and the comparison
/// needs both.)
#[test]
fn d963_ranks2_system_matches_ranks1() {
    dump_system(1, "np1");
    dump_system(2, "np2");
    assert_systems_match();
}

/// `-(σ, τ)` adapter — mirrors the miniapp's `NegVectorMass`
/// (`TransposeIntegrator(VectorFEMassIntegrator(-1))`).
struct NegVectorMass;
impl fem_assembly::dpg::dpg_integrators::DpgBilinear2 for NegVectorMass {
    fn assemble2(
        &self,
        ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
        trial: &fem_assembly::dpg::dpg_basis::VolVals,
        test: &fem_assembly::dpg::dpg_basis::VolVals,
        m: &mut [f64],
    ) {
        let d = ctx.dim;
        let nt = test.n_scalar;
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        for k in 0..nt {
            for c in 0..d {
                for j in 0..nsc {
                    m[k * nc + c * nsc + j] -= ctx.w * test.phi[k * d + c] * trial.phi[j];
                }
            }
        }
    }
}
