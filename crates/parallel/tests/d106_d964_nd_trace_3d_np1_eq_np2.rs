//! D964: the **3-D** parallel DPG channel (pmaxwell's ND-trace Maxwell block
//! table) must assemble a system at `--ranks 2` that is **row-for-row
//! identical** (in global-DOF space) to the `--ranks 1` system — the 3-D
//! counterpart of `d105_d963_trace_global_boundary.rs` (which pins the 2-D
//! RT/H1-trace pdiffusion table).
//!
//! Probe: dump every owned row of the formed complex system `(A, b)` for the
//! `pmaxwell -m data/inline-hex.mesh -prob 0` block table on the unit cube
//! (4×4×4 hexes, order 1, δ-order 1, ω = 2π), with the miniapp's essential
//! pipeline applied (the D963 global-boundary criterion selects the Ê ND-trace
//! dofs; prescribed values are the uniform `(1, 0)` — the *set* is what this
//! probe pins, the projection *values* are pinned at the miniapp level against
//! the C++ MPI oracle).  Entries are keyed by absolute global DOF ids; complex
//! values are dumped as `(re, im)`.
//!
//! Two decompositions are compared, per-`(row, col)` key sets exactly and
//! values to `1e-12` **absolute** (the D964 defect halved 64 skeleton rows:
//! `Δ = 5.14`; after the fix the only deltas are last-bit summation-order
//! noise `≤ 2.3e-15` — the Maxwell whitened rows sum up to four elements per
//! edge dof and the `--ranks 2` local mesh visits owned elements before
//! ghosts, unlike the 2-D pdiffusion table whose entries are single-element
//! sums and therefore pin bit-exact in `d105`).  Both the ess-applied and the
//! empty-ess system are pinned: the empty-ess run isolates the row-owner
//! completeness (D963's second defect, on edges), the ess run also pins the
//! global-boundary selection.

use std::fs;
use std::sync::Arc;

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgCurl3dPairingIntegrator, DpgCurlCurlIntegrator, DpgMixedVectorCurlIntegrator,
    DpgMixedVectorWeakCurlIntegrator, DpgTangentTraceIntegrator3D, DpgTVectorFEMassIntegrator,
    DpgVectorFEDomainLFIntegrator, DpgVectorFEMassIntegrator,
};
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_complex_dpg_weakform::ParComplexDPGWeakForm;
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::WorkerConfig;

const PI: f64 = std::f64::consts::PI;

/// One dumped entry: `(global row, global col, re, im)`; RHS entries use the
/// sentinel column `u32::MAX`.
type Entry = (u32, u32, f64, f64);

/// Manufactured plane-wave current `J` (C++ `pmaxwell.cpp`, 3-D branch):
/// `J_re = ω(−s, s, s)`, `J_im = ω(c, −c, −c)` with `c + is = e^{iωσ}` — sign
/// convention of `miniapps/dpg/pmaxwell.rs::Exact3D::j`.
fn j_source(x: &[f64], out: &mut [f64], imag: bool) {
    let a = PI * 2.0 * x.iter().sum::<f64>();
    let (c, s) = (a.cos(), a.sin());
    let w = PI * 2.0;
    if imag {
        out[0] = w * c;
        out[1] = -w * c;
        out[2] = -w * c;
    } else {
        out[0] = -w * s;
        out[1] = w * s;
        out[2] = w * s;
    }
}

fn dump_system(n_ranks: usize, tag: &str, apply_ess: bool) {
    let mesh = Mesh::<3>::unit_cube_hex(4);
    let dump_dir = format!("{}/d964dump-{tag}", env!("CARGO_TARGET_TMPDIR"));
    let _ = fs::remove_dir_all(&dump_dir);
    fs::create_dir_all(&dump_dir).unwrap();
    let dump_dir = Arc::new(dump_dir);
    let mesh = Arc::new(mesh);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh_identity(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();

        let mut a = ParComplexDPGWeakForm::new(local_mesh, partition, comm.clone());
        // pmaxwell 3-D: order 1, delta_order 1 (verified C++ config).
        let p: u8 = 1;
        let test_order = 2_u8;
        let omega = 2.0 * PI;
        let (mu, eps) = (1.0_f64, 1.0_f64);
        a.set_quad_order(2 * test_order);
        a.set_face_quad_order(test_order + p);
        a.store_matrices(true);

        let es = a.add_trial_vector_space(p - 1, 3);
        let hs = a.add_trial_vector_space(p - 1, 3);
        let hate = a.add_trial_trace_space_nd(p);
        let hath = a.add_trial_trace_space_nd(p);
        let f = a.add_test_space(VolKind::HCurl, test_order);
        let g = a.add_test_space(VolKind::HCurl, test_order);

        a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, es, f);
        a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, hs, g);
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hate, f);

        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, f, f);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, f, f);
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: -eps * omega })),
            es,
            g,
        );
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: mu * omega })),
            hs,
            f,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: mu * mu * omega * omega })),
            None,
            f,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: -mu * omega })),
            g,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorCurlIntegrator { q: -eps * omega })),
            g,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorCurlIntegrator { q: mu * omega })),
            f,
            g,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: eps * omega })),
            f,
            g,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: eps * eps * omega * omega })),
            None,
            g,
            g,
        );
        a.add_domain_lf_integrator(
            Some(Box::new(DpgVectorFEDomainLFIntegrator {
                f: |x: &[f64], out: &mut [f64]| j_source(x, out, false),
            })),
            Some(Box::new(DpgVectorFEDomainLFIntegrator {
                f: |x: &[f64], out: &mut [f64]| j_source(x, out, true),
            })),
            g,
        );

        a.assemble();

        // The miniapp's essential pipeline (global-boundary criterion, D963):
        // prescribed values uniform (1, 0) — set correctness is the pin.  The
        // empty-ess variant isolates the row-owner completeness.
        let pairs = a.trace_boundary_dofs_ix(hate);
        let ess_ids: Vec<u32> = if apply_ess {
            pairs.iter().map(|(gid, _)| *gid).collect()
        } else {
            Vec::new()
        };
        let n_local = a.local().size();
        let mut x_r = vec![0.0_f64; n_local];
        let x_i = vec![0.0_f64; n_local];
        if apply_ess {
            for &(_, sidx) in &pairs {
                x_r[sidx] = 1.0;
            }
        }
        let (sys, _x0, _) = a.form_linear_system(&ess_ids, &x_r, &x_i);

        let owned_global: Vec<u32> = a.owned_global_ids().to_vec();
        let ghost_global: Vec<u32> = a.ghost_global_ids().to_vec();
        let diag = sys.a.diag_block();
        let offd = sys.a.offd_block();
        let b_re = sys.b.re.as_slice();
        let b_im = sys.b.im.as_slice();

        let mut entries: Vec<Entry> = Vec::new();
        for r in 0..sys.n_owned {
            let grow = owned_global[r];
            for k in diag.row_ptr[r]..diag.row_ptr[r + 1] {
                let gc = owned_global[diag.col_idx[k] as usize];
                entries.push((grow, gc, diag.re_vals[k], diag.im_vals[k]));
            }
            for k in offd.row_ptr[r]..offd.row_ptr[r + 1] {
                let gc = ghost_global[offd.col_idx[k] as usize];
                entries.push((grow, gc, offd.re_vals[k], offd.im_vals[k]));
            }
            entries.push((grow, u32::MAX, b_re[r], b_im[r]));
        }
        entries.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        let path = format!("{dump_dir}/rank{rank}.txt");
        let mut text = String::new();
        for (g, c, vr, vi) in &entries {
            text.push_str(&format!("{g} {c} {vr:.17e} {vi:.17e}\n"));
        }
        fs::write(path, text).unwrap();
    });
}

fn load_entries(tag: &str) -> Vec<Entry> {
    let dump_dir = format!("{}/d964dump-{tag}", env!("CARGO_TARGET_TMPDIR"));
    let mut all: Vec<Entry> = Vec::new();
    for entry in fs::read_dir(&dump_dir).unwrap() {
        let path = entry.unwrap().path();
        for line in fs::read_to_string(&path).unwrap().lines() {
            let mut it = line.split_whitespace();
            let g: u32 = it.next().unwrap().parse().unwrap();
            let c: u32 = it.next().unwrap().parse().unwrap();
            let vr: f64 = it.next().unwrap().parse().unwrap();
            let vi: f64 = it.next().unwrap().parse().unwrap();
            all.push((g, c, vr, vi));
        }
    }
    all.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    all
}

/// Merge ranks' dumps into one global system and compare against `--ranks 1`:
/// rows are disjoint across ranks (each global row is owned by exactly one
/// rank); duplicate `(row, col)` pairs within a rank must not exist for this
/// block table.
fn assert_systems_match(tag: &str) {
    let one = load_entries(&format!("np1-{tag}"));
    let two = load_entries(&format!("np2-{tag}"));

    let rows_of = |e: &[Entry]| -> std::collections::BTreeSet<u32> {
        e.iter().map(|e| e.0).collect()
    };
    let r1 = rows_of(&one);
    let r2 = rows_of(&two);
    assert_eq!(r1, r2, "{tag}: row spaces differ between np1 and np2");

    let map: std::collections::HashMap<(u32, u32), (f64, f64)> =
        one.iter().map(|&(g, c, vr, vi)| ((g, c), (vr, vi))).collect();
    assert_eq!(map.len(), one.len(), "{tag}: np1 has duplicate (row, col) pairs");
    let mut worst_abs = 0.0_f64;
    let mut worst_key = None;
    let mut missing = 0usize;
    for &(g, c, vr, vi) in &two {
        match map.get(&(g, c)) {
            Some(&(v1r, v1i)) => {
                let dr = vr - v1r;
                let di = vi - v1i;
                let abs = (dr * dr + di * di).sqrt();
                if abs > worst_abs {
                    worst_abs = abs;
                    worst_key = Some((g, c));
                }
            }
            None => missing += 1,
        }
    }
    assert_eq!(missing, 0, "{tag}: np2 has {missing} entries absent from np1");
    // D964 defect: 64 skeleton rows halved (|Δ| = 5.14); healthy noise after
    // the edge-owner fix is last-bit summation-order rounding (≤ 2.3e-15).
    assert!(
        worst_abs <= 1e-12,
        "{tag}: np2 entry diverges from np1: worst |Δ| {worst_abs:e} at {worst_key:?}"
    );
}

/// `--ranks 1` baseline dump + compare against a fresh `--ranks 2` dump, with
/// and without the essential pipeline.  (Single `#[test]`: the dumps live in
/// per-tag temp dirs and the comparison needs both.)
#[test]
fn d964_ranks2_3d_system_matches_ranks1() {
    for apply_ess in [true, false] {
        let tag = if apply_ess { "ess" } else { "noess" };
        dump_system(1, &format!("np1-{tag}"), apply_ess);
        dump_system(2, &format!("np2-{tag}"), apply_ess);
        assert_systems_match(tag);
    }
}
