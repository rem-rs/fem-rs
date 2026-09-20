//! D404 diagnostic probe (temporary, `#[ignore]`d): why does L²-projection ZZ
//! + Dörfler(0.5) stall the NC AMR loop (ne 8→20→32→65→98→125, final L2
//! 3.2e-2) while MFEM 4.10's ZZ + `ThresholdRefiner(0.5)` reaches 4.27e-4 at
//! ne=6032?
//!
//! Per AMR level this prints: the η distribution (max/median/min, Ση²), the
//! top-η elements (with area and hanging-node flag), the per-element TRUE L2
//! error distribution, the global ZZ estimate vs true error, and how many
//! elements each marking rule (Dörfler 0.5, 0.5·‖η‖∞, 0.5·RMS(η)) would mark.
//!
//! Run with:
//! `cargo test -p fem-solver --test d404_probe -- --ignored --nocapture`

use std::collections::HashSet;
use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{topology::MeshTopology, Mesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, apply_hanging_constraints, boundary_dofs, recover_hanging_values},
};

fn u_exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }
fn forcing(x: &[f64]) -> f64 { 2.0 * PI * PI * u_exact(x) }

fn cfg() -> SolverConfig {
    SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() }
}

/// True per-element L2 error of the P1 interpolant `u_h` on triangle `e`.
fn elem_true_l2_err(uh: &[f64], space: &H1Space<Mesh<2>>, e: u32) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};
    let mesh = space.mesh();
    let quad = TriP1.quadrature(5);
    let nodes = mesh.element_nodes(e);
    let dofs = space.element_dofs(e);
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();
    let mut phi = vec![0.0_f64; 3];
    let mut err_sq = 0.0_f64;
    for (q, xi) in quad.points.iter().enumerate() {
        let w = quad.weights[q] * det_j;
        TriP1.eval_basis(xi, &mut phi);
        let uh_q: f64 = dofs.iter().zip(phi.iter()).map(|(&d, &p)| uh[d as usize] * p).sum();
        let xp = [x0[0]+(x1[0]-x0[0])*xi[0]+(x2[0]-x0[0])*xi[1],
                  x0[1]+(x1[1]-x0[1])*xi[0]+(x2[1]-x0[1])*xi[1]];
        let diff = uh_q - u_exact(&xp);
        err_sq += w * diff * diff;
    }
    err_sq.sqrt()
}

fn elem_area(mesh: &Mesh<2>, e: u32) -> f64 {
    let n = mesh.element_nodes(e);
    let (x0, x1, x2) = (mesh.node_coords(n[0]), mesh.node_coords(n[1]), mesh.node_coords(n[2]));
    0.5 * ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs()
}

#[test]
#[ignore] // diagnostic probe; run explicitly with --ignored
fn d404_l2zz_amr_diagnostic() {
    use fem_assembly::postproc::error_estimate::zz_estimator_l2_nc;
    use fem_assembly::postproc::grid_function::GridFunction;
    use fem_mesh::amr::NCState;

    let mut mesh = Mesh::<2>::unit_square_tri(2);
    let mut nc_state = NCState::new();
    let mut hanging_constraints = Vec::new();

    for level in 0..6 {
        let space = H1Space::new(mesh.clone(), 1);
        let n = space.n_dofs();

        let diffusion = DiffusionIntegrator { kappa: 1.0 };
        let source = DomainSourceIntegrator::new(forcing);
        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);
        apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);
        let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
        apply_dirichlet(&mut mat, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);

        let mut u = vec![0.0_f64; n];
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg()).unwrap();
        assert!(res.converged);
        recover_hanging_values(&mut u, &hanging_constraints);

        let ne = mesh.n_elements();
        // True per-element errors + global L2.
        let true_err: Vec<f64> = (0..ne as u32)
            .map(|e| elem_true_l2_err(&u, &space, e))
            .collect();
        let global_true: f64 = true_err.iter().map(|e| e * e).sum::<f64>().sqrt();

        // L2-ZZ indicators.
        let gf = GridFunction::new(&space, u.clone());
        let ind = zz_estimator_l2_nc(&gf, &hanging_constraints);
        let eta = &ind.eta;

        let hanging_set: HashSet<usize> =
            hanging_constraints.iter().map(|c| c.constrained).collect();
        let elem_hanging: Vec<bool> = (0..ne as u32)
            .map(|e| space.element_dofs(e).iter().any(|&d| hanging_set.contains(&(d as usize))))
            .collect();

        println!("═══ level {level}: ne={ne} ndof={n} global_true_l2={global_true:.6e} zz_total={:.6e}",
            ind.total_error);

        let mut sv: Vec<f64> = eta.clone();
        sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let med = sv[sv.len() / 2];
        println!("    η: max={:.4e} median={med:.4e} min={:.4e}  Ση²={:.4e}",
            sv[0], sv[sv.len()-1], ind.total_error * ind.total_error);

        let mut te: Vec<f64> = true_err.clone();
        te.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let tmed = te[te.len() / 2];
        println!("    e_true: max={:.4e} median={tmed:.4e} min={:.4e}  Σe²={:.4e}",
            te[0], te[te.len()-1], global_true * global_true);

        // Top-8 by η and by true error (index, value, area, hanging?).
        let top = |vals: &[f64]| -> Vec<(usize, f64, f64, bool)> {
            let mut v: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v.into_iter().take(8)
                .map(|(i, val)| (i, val, elem_area(&mesh, i as u32), elem_hanging[i]))
                .collect()
        };
        for (label, list) in [("η   ", top(eta)), ("e_tr", top(&true_err))] {
            let rows: Vec<String> = list.iter()
                .map(|&(i, val, a, h)| format!("#{}:{val:.3e}(A={a:.2e}{})", i, if h {",H"} else {""}))
                .collect();
            println!("    top-{label} {}", rows.join("  "));
        }

        // Marking-rule comparison.
        let dorfler = ind.dorfler_mark(0.5).len();
        let linf = eta.iter().filter(|&&e| e > 0.5 * sv[0]).count();
        let rms = (ind.total_error * ind.total_error / ne as f64).sqrt();
        let nrms = eta.iter().filter(|&&e| e > 0.5 * rms).count();
        println!("    marks: Dörfler(0.5)={dorfler}  0.5·max={linf}  0.5·RMS={nrms}  (of {ne})");

        if level < 5 {
            let marked = ind.dorfler_mark(0.5);
            let (new_mesh, new_c, _) = nc_state.refine(&mesh, &marked, 0);
            mesh = new_mesh;
            hanging_constraints = new_c;
        }
    }
}

/// Estimator-variant attribution: at each level compare the global ZZ norm and
/// max η for (a) L²-ZZ with constraints, (b) L²-ZZ ignoring constraints,
/// (c) nodal-average ZZ, (d) MFEM SumFluxAndCount ZZ — plus the marks each
/// rule would pick.
#[test]
#[ignore] // diagnostic probe; run explicitly with --ignored
fn d404_estimator_variants() {
    use fem_assembly::postproc::error_estimate::{zz_estimator_l2_nc, zz_estimator_nodal};
    use fem_assembly::postproc::flux_recovery::zz_estimator_mfem_nc;
    use fem_assembly::postproc::grid_function::GridFunction;
    use fem_mesh::amr::NCState;

    let mut mesh = Mesh::<2>::unit_square_tri(2);
    let mut nc_state = NCState::new();
    let mut hanging_constraints = Vec::new();

    for level in 0..5 {
        let space = H1Space::new(mesh.clone(), 1);
        let n = space.n_dofs();

        let diffusion = DiffusionIntegrator { kappa: 1.0 };
        let source = DomainSourceIntegrator::new(forcing);
        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);
        apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);
        let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
        apply_dirichlet(&mut mat, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);

        let mut u = vec![0.0_f64; n];
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg()).unwrap();
        assert!(res.converged);
        recover_hanging_values(&mut u, &hanging_constraints);

        let ne = mesh.n_elements();
        let true_err: Vec<f64> = (0..ne as u32)
            .map(|e| elem_true_l2_err(&u, &space, e))
            .collect();
        let global_true: f64 = true_err.iter().map(|e| e * e).sum::<f64>().sqrt();

        let gf = GridFunction::new(&space, u.clone());
        let a = zz_estimator_l2_nc(&gf, &hanging_constraints);
        let b = zz_estimator_l2_nc(&gf, &[]);
        let c = zz_estimator_nodal(&gf, &hanging_constraints);
        let d = zz_estimator_mfem_nc(&gf, &diffusion, &hanging_constraints);

        let stat = |ind: &fem_assembly::postproc::error_estimate::ElementIndicators| {
            let mx = ind.eta.iter().cloned().fold(0.0_f64, f64::max);
            let mut v = ind.eta.clone();
            v.sort_by(|x, y| y.partial_cmp(x).unwrap());
            (ind.total_error, mx, v[v.len()/2])
        };
        let (ta, ma, _) = stat(&a);
        let (tb, mb, _) = stat(&b);
        let (tc, mc, _) = stat(&c);
        let (td, md, _) = stat(&d);
        println!("═══ level {level}: ne={ne} ndof={n} nconstr={} global_true={global_true:.4e}",
            hanging_constraints.len());
        println!("    (a) l2zz+c:  ‖η‖₂={ta:.4e} max={ma:.4e}  eff={:.1}", ta / global_true);
        println!("    (b) l2zz-c:  ‖η‖₂={tb:.4e} max={mb:.4e}  eff={:.1}", tb / global_true);
        println!("    (c) nodal:   ‖η‖₂={tc:.4e} max={mc:.4e}  eff={:.1}", tc / global_true);
        println!("    (d) mfem_nc: ‖η‖₂={td:.4e} max={md:.4e}  eff={:.1}", td / global_true);
        // Marking rules on variant (a):
        let rms_a = (ta * ta / ne as f64).sqrt();
        println!("    marks(a): Dörfler={}  0.5·max={}  0.5·RMS={}  (of {ne})",
            a.dorfler_mark(0.5).len(),
            a.eta.iter().filter(|&&e| e > 0.5 * ma).count(),
            a.eta.iter().filter(|&&e| e > 0.5 * rms_a).count());
        // Marking rules on variant (d):
        let rms_d = (td * td / ne as f64).sqrt();
        println!("    marks(d): Dörfler={}  0.5·max={}  0.5·RMS={}  (of {ne})",
            d.dorfler_mark(0.5).len(),
            d.eta.iter().filter(|&&e| e > 0.5 * md).count(),
            d.eta.iter().filter(|&&e| e > 0.5 * rms_d).count());

        let marked = a.dorfler_mark(0.5);
        let (new_mesh, new_c, _) = nc_state.refine(&mesh, &marked, 0);
        mesh = new_mesh;
        hanging_constraints = new_c;
    }
}

/// Full 5-round AMR trajectories for {estimator × marking rule} combinations,
/// each printed as `ne→…` and `l2→…` for direct comparison with the MFEM 4.10
/// reference (linf: ne 8→32→128→428→1624→6032, l2 2.46e-1→…→4.27e-4).
#[test]
#[ignore] // diagnostic probe; run explicitly with --ignored
fn d404_marking_trajectories() {
    use fem_assembly::postproc::error_estimate::zz_estimator_l2_nc;
    use fem_assembly::postproc::flux_recovery::zz_estimator_mfem_nc;
    use fem_assembly::postproc::grid_function::GridFunction;
    use fem_mesh::amr::NCState;

    #[derive(Clone, Copy)]
    enum Mark { Dorfler, Linf, Rms }

    fn mark_with(eta: &[f64], total: f64, m: Mark) -> Vec<u32> {
        match m {
            Mark::Dorfler => {
                // Dörfler bulk criterion (same as ElementIndicators::dorfler_mark).
                let target = 0.5 * total * total;
                let mut idx: Vec<u32> = (0..eta.len() as u32).collect();
                idx.sort_unstable_by(|&a, &b| eta[b as usize].partial_cmp(&eta[a as usize]).unwrap());
                let (mut acc, mut out) = (0.0, Vec::new());
                for e in idx { acc += eta[e as usize] * eta[e as usize]; out.push(e); if acc >= target { break; } }
                out
            }
            Mark::Linf => {
                let mx = eta.iter().cloned().fold(0.0_f64, f64::max);
                eta.iter().enumerate().filter(|(_, &e)| e > 0.5 * mx).map(|(i, _)| i as u32).collect()
            }
            Mark::Rms => {
                let rms2 = total * total / eta.len() as f64;
                eta.iter().enumerate().filter(|(_, &e)| e * e > 0.25 * rms2).map(|(i, _)| i as u32).collect()
            }
        }
    }

    let combos: &[(&str, u8 /*0=a l2zz+c, 1=b l2zz-c, 2=d mfem_nc*/, Mark)] = &[
        ("l2zz+c  Dörfler(0.5) [current]", 0, Mark::Dorfler),
        ("l2zz+c  0.5·max  [MFEM linf]", 0, Mark::Linf),
        ("l2zz-c  0.5·max", 1, Mark::Linf),
        ("l2zz-c  0.5·RMS", 1, Mark::Rms),
        ("l2zz+c  0.5·RMS", 0, Mark::Rms),
        ("mfem_nc 0.5·max  [MFEM replica]", 2, Mark::Linf),
        ("mfem_nc Dörfler(0.5)", 2, Mark::Dorfler),
        ("nodal   0.5·RMS", 2, Mark::Rms),
    ];

    for &(label, variant, mark) in combos {
        let mut mesh = Mesh::<2>::unit_square_tri(2);
        let mut nc_state = NCState::new();
        let mut hanging_constraints = Vec::new();
        let mut nes = Vec::new();
        let mut l2s = Vec::new();

        for level in 0..6 {
            let space = H1Space::new(mesh.clone(), 1);
            let n = space.n_dofs();
            let diffusion = DiffusionIntegrator { kappa: 1.0 };
            let source = DomainSourceIntegrator::new(forcing);
            let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
            let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);
            apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);
            let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
            apply_dirichlet(&mut mat, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);
            let mut u = vec![0.0_f64; n];
            let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg()).unwrap();
            assert!(res.converged);
            recover_hanging_values(&mut u, &hanging_constraints);

            // Global L2 error.
            let l2: f64 = {
                let e2: f64 = (0..mesh.n_elements() as u32)
                    .map(|e| { let x = elem_true_l2_err(&u, &space, e); x * x })
                    .sum();
                e2.sqrt()
            };
            nes.push(mesh.n_elements());
            l2s.push(l2);

            if level < 5 {
                let gf = GridFunction::new(&space, u.clone());
                let ind = match variant {
                    0 => zz_estimator_l2_nc(&gf, &hanging_constraints),
                    1 => zz_estimator_l2_nc(&gf, &[]),
                    _ => zz_estimator_mfem_nc(&gf, &diffusion, &hanging_constraints),
                };
                let marked = mark_with(&ind.eta, ind.total_error, mark);
                let (new_mesh, new_c, _) = nc_state.refine(&mesh, &marked, 0);
                mesh = new_mesh;
                hanging_constraints = new_c;
            }
        }
        let ne_s: Vec<String> = nes.iter().map(|n| n.to_string()).collect();
        let l2_fmt: Vec<String> = l2s.iter().map(|e| format!("{e:.3e}")).collect();
        println!("── {label}");
        println!("   ne: {}", ne_s.join("→"));
        println!("   l2: {}", l2_fmt.join("→"));
    }
}
