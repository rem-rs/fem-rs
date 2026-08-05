//! Debug: run the full ex15 flow to Time 0.01/0.02 it1 and dump A_true +
//! solution + constraints, for 1:1 comparison with tools_ex15_ref
//! (ex15_dump_T0deref.cpp / ex15_dump_T002.cpp).
//! Usage: cargo run --release -p fem-examples --example mfem_ex15_dump_T002

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_assembly::postproc::amr_refiner::{ThresholdRefiner, ThresholdDerefiner};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_mesh::amr::{NCStateQuad, NcState2D};
use fem_space::constraints::{apply_dirichlet, boundary_dofs, conforming_assemble};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::dof_manager::{DofManager, EdgeKey};
use std::time::Instant;

const ALPHA: f64 = 0.02;

fn front(x: f64, y: f64, z: f64, t: f64) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    (-0.5 * ((r - t) / ALPHA).powi(2)).exp()
}
fn front_laplace(x: f64, y: f64, z: f64, t: f64, dim: i32) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    let a2 = ALPHA * ALPHA; let a4 = a2 * a2;
    let r_term = -2.0 * t * (x * x + y * y + z * z - (dim - 1) as f64 * a2 / 2.0) / r.max(1e-300);
    -(-0.5 * ((r - t) / ALPHA).powi(2)).exp() / a4 * (r_term + x * x + y * y + z * z + t * t - dim as f64 * a2)
}
fn ball(x: f64, y: f64, z: f64, t: f64) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    -(2.0 * (r - t) / ALPHA).atan()
}
fn ball_laplace(x: f64, y: f64, z: f64, t: f64, dim: i32) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    let a2 = ALPHA * ALPHA; let t2 = 4.0 * t * t;
    let denom = (-a2 - 4.0 * (x * x + y * y + z * z - 2.0 * r * t) - t2).powi(2);
    if dim == 2 { 2.0 * ALPHA * (a2 + t2 - 4.0 * x * x - 4.0 * y * y) / r.max(1e-300) / denom }
    else { 4.0 * ALPHA * (a2 + t2 - 4.0 * r * t) / r.max(1e-300) / denom }
}
fn composite<F0, F1>(pt: &[f64], t: f64, f0: F0, _f1: F1) -> f64
where F0: Fn(f64, f64, f64, f64) -> f64, F1: Fn(f64, f64, f64, f64) -> f64 {
    let x = pt[0]; let y = pt[1]; let z = if pt.len() == 3 { pt[2] } else { 0.0 };
    f0(x, y, z, t)
}
fn bdr_func(pt: &[f64], t: f64) -> f64 { composite(pt, t, front, ball) }
fn rhs_func(pt: &[f64], t: f64) -> f64 {
    composite(pt, t, |x, y, z, t| front_laplace(x, y, z, t, pt.len() as i32),
              |x, y, z, t| ball_laplace(x, y, z, t, pt.len() as i32))
}
fn main() {
    let _t0 = Instant::now();
    let mesh0: Mesh<2> = read_mfem_file("data/star-hilbert.mesh").unwrap().mesh2d.unwrap();
    let mut mesh = mesh0;
    let mut nc_state: NCStateQuad = NCStateQuad::new();
    let mut refiner = ThresholdRefiner::new(false);
    refiner.set_local_error_goal(0.005);
    refiner.set_nc_limit(3);
    let mut derefiner = ThresholdDerefiner::new();
    derefiner.set_threshold(0.15 * 0.005);
    derefiner.set_nc_limit(3);

    let dt = 0.01;
    let mut time = 0.0;
    let order = 2u8;
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut dump_t001 = false;
    let mut dump_t002 = false;
    let mut marked_printed = false;

    while time < 0.931 {
        refiner.reset();
        let mut ref_it = 1usize;
        loop {
            let space = H1Space::new(mesh.clone(), order);
            let cdofs = space.n_dofs();
            let quad_rule = (order as u8) * 2 + 1;
            let mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_rule);
            let source = DomainSourceIntegrator::new(|pt: &[f64]| rhs_func(pt, time));
            let rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_rule);
            let dm0 = space.dof_manager();
            let hc = fem_space::constraints::p2_hanging_constraints(
                nc_state.constraints(), dm0, nc_state.midpoints(),
            );
            let (mat_true, rhs_true, true_dofs) = conforming_assemble(&mat, &rhs_vec, &hc);
            let dm = space.dof_manager();
            let bnd_tags = space.mesh().unique_boundary_tags();
            let bnd_all = boundary_dofs(space.mesh(), dm, &bnd_tags);
            let true_set: std::collections::HashSet<usize> = true_dofs.iter().copied().collect();
            let true_idx: std::collections::HashMap<usize, usize> = true_dofs.iter().enumerate().map(|(i, &d)| (d, i)).collect();
            let mut mat_true = mat_true;
            let mut rhs_true = rhs_true;
            let bnd_vals: Vec<f64> = bnd_all.iter().filter(|d| true_set.contains(&(**d as usize)))
                .map(|&dof| bdr_func(&dm.dof_coord(dof), time)).collect();
            let bnd: Vec<u32> = bnd_all.iter().filter(|d| true_set.contains(&(**d as usize)))
                .map(|&d| true_idx[&(d as usize)] as u32).collect();
            apply_dirichlet(&mut mat_true, &mut rhs_true, &bnd, &bnd_vals);
            if (time - 0.01).abs() < 1e-9 && ref_it == 1  && !dump_t001 {
                println!("BND {} nbf={} tags={:?}", bnd.len(), mesh.n_boundary_faces(), bnd_tags);
                for (i, &d) in bnd.iter().enumerate() { println!("{i} {d}"); }
                for f in 0..mesh.n_boundary_faces() as u32 {
                    let fn_: &[u32] = mesh.face_nodes(f);
                    println!("BFACE {f} tag={} nodes={:?}", mesh.face_tag(f), fn_);
                }
            }
            let mut u = vec![0.0_f64; cdofs];
            let mut x_true = vec![0.0_f64; true_dofs.len()];
            let res = solve_pcg_gssmoother(&mat_true, &rhs_true, &mut x_true,
                &SolverConfig { rtol: 1e-6, max_iter: 500, verbose: false, ..Default::default() }).unwrap();
            if (time - 0.01).abs() < 1e-9 && ref_it == 1 {
                println!("PCGITERS {}", res.iterations);
            }
            for (&td, &v) in true_dofs.iter().zip(x_true.iter()) { u[td] = v; }
            if !hc.is_empty() {
                fem_space::constraints::recover_hanging_values(&mut u, &hc);
            }

            // ── dump (before refinement, on the solved mesh) ──
            if (time - 0.02).abs() < 1e-9 && ref_it == 1 && !dump_t002 {
                dump_t002 = true;
                println!("T002IT1 dofs={cdofs} elems={} nodes={}", mesh.n_elems(), mesh.n_nodes());
                if let Some(view) = mesh.nc_vertex_view() {
                    println!("T002NODE {}", view.len());
                    for (d, &n) in view.iter().enumerate() {
                        let p = mesh.node_coords(n);
                        println!("{d} {:.17} {:.17}", p[0], p[1]);
                    }
                } else {
                    println!("T002NODE {}", mesh.n_nodes());
                    for n in 0..mesh.n_nodes() as u32 {
                        let p = mesh.node_coords(n);
                        println!("{n} {:.17} {:.17}", p[0], p[1]);
                    }
                }
                println!("T002MESH {}", mesh.n_elems());
                for e in 0..mesh.n_elems() as u32 {
                    let ns = mesh.elem_nodes(e);
                    println!("{e} {} {} {} {}", ns[0], ns[1], ns[2], ns[3]);
                }
                println!("T002CP {}", true_dofs.len());
                for (i, &td) in true_dofs.iter().enumerate() { println!("CPR {i} {td}:1"); }
                println!("ATRUE {} {} {}", mat_true.nrows, mat_true.ncols, mat_true.nnz());
                for i in 0..mat_true.nrows {
                    for k in mat_true.row_ptr[i]..mat_true.row_ptr[i + 1] {
                        println!("{} {} {:.17e}", i, mat_true.col_idx[k], mat_true.values[k]);
                    }
                }
                println!("BTRUE {}", rhs_true.len());
                for (i, &v) in rhs_true.iter().enumerate() { println!("{i} {v:.17e}"); }
                println!("SOLU {cdofs}");
                for (d, &v) in u.iter().enumerate() { println!("{d} {v:.17e}"); }
            }
            let t_in_range = [0.03f64, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.93]
                .iter()
                .any(|&t| (time - t).abs() < 1e-9);
            if t_in_range && ref_it == 1 {
                println!("T006CP t={time} dofs={} true={} nconstr={}", cdofs, true_dofs.len(), hc.len());
                if (time - 0.06).abs() < 1e-9 {
                for c in &hc {
                    print!("TDEP {} <- {}:{}", c.constrained, c.parent_a, c.coeff_a);
                    if c.parent_b != c.parent_a || c.coeff_b != c.coeff_a {
                        print!(" {}:{}", c.parent_b, c.coeff_b);
                    }
                    for &(m, w) in &c.extra { print!(" {}:{:.17}", m, w); }
                    println!();
                }
                }
                println!("T006ATRUE {} {} {}", mat_true.nrows, mat_true.ncols, mat_true.nnz());
                for i in 0..mat_true.nrows {
                    for k in mat_true.row_ptr[i]..mat_true.row_ptr[i + 1] {
                        println!("{} {} {:.17e}", i, mat_true.col_idx[k], mat_true.values[k]);
                    }
                }
                println!("T006BTRUE {}", rhs_true.len());
                for (i, &v) in rhs_true.iter().enumerate() { println!("{i} {v:.17e}"); }
                println!("T006SOLU {cdofs}");
                for (d, &v) in u.iter().enumerate() { println!("{d} {v:.17e}"); }
            }
            if (time - 0.01).abs() < 1e-9 && ref_it == 1 && !dump_t001 {
                dump_t001 = true;
                println!("T001IT1 dofs={cdofs} elems={} nodes={}", mesh.n_elems(), mesh.n_nodes());
                println!("ATRUE {} {} {}", mat_true.nrows, mat_true.ncols, mat_true.nnz());
                for i in 0..mat_true.nrows {
                    for k in mat_true.row_ptr[i]..mat_true.row_ptr[i + 1] {
                        println!("{} {} {:.17e}", i, mat_true.col_idx[k], mat_true.values[k]);
                    }
                }
                println!("BTRUE {}", rhs_true.len());
                for (i, &v) in rhs_true.iter().enumerate() { println!("{i} {v:.17e}"); }
                println!("XTRUE {}", x_true.len());
                for (i, &v) in x_true.iter().enumerate() { println!("{i} {v:.17e}"); }
                println!("SOLU {cdofs}");
                for (d, &v) in u.iter().enumerate() { println!("{d} {v:.17e}"); }
                // ARAW dump: raw assembled matrix (before conforming_assemble)
                println!("ARAW {} {} {}", mat.nrows, mat.ncols, mat.nnz());
                for i in 0..mat.nrows {
                    for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                        println!("{} {} {:.17e}", i, mat.col_idx[k], mat.values[k]);
                    }
                }
                // PROW dump: P = build_conforming_prolongation, columns mapped
                // to view-DOF ids via true_dofs (matches C++ cP DEPROW/IDROW)
                {
                    let p = fem_space::constraints::build_conforming_prolongation(cdofs, &hc);
                    println!("PROW {} {}", p.nrows, p.ncols);
                    for i in 0..p.nrows {
                        print!("PROW {i}");
                        for k in p.row_ptr[i]..p.row_ptr[i + 1] {
                            let col = p.col_idx[k];
                            let vdof = true_dofs[col as usize];
                            print!(" {vdof}:{:.9}", p.values[k]);
                        }
                        println!();
                    }
                }
                println!("P2CONST {}", hc.len());
                for c in &hc {
                    print!("  {} <- {}:{}", c.constrained, c.parent_a, c.coeff_a);
                    if c.parent_b != c.parent_a || c.coeff_b != c.coeff_a {
                        print!(" {}:{}", c.parent_b, c.coeff_b);
                    }
                    for &(m, w) in &c.extra { print!(" {}:{:.4}", m, w); }
                    println!();
                }
                println!("P1CONST {}", nc_state.constraints().len());
                for c in nc_state.constraints() {
                    println!("  {} <- {} + {}", c.constrained, c.parent_a, c.parent_b);
                }
                let mp = nc_state.midpoints();
                let mut mism = 0;
                for c in nc_state.constraints() {
                    let key = if c.parent_a < c.parent_b { (c.parent_a as u32, c.parent_b as u32) } else { (c.parent_b as u32, c.parent_a as u32) };
                    let m = mp.get(&key).copied();
                    if m != Some(c.constrained as u32) {
                        mism += 1;
                        if mism <= 10 {
                            println!("  P1-MISMATCH {} <- {} + {} (midpoints={m:?})", c.constrained, c.parent_a, c.parent_b);
                        }
                    }
                }
                println!("P1-MISMATCH-count {mism}");
                // dump mesh element connectivity (first 5)
                println!("RUSTMESH {}", mesh.n_elems());
                for e in 0..mesh.n_elems() as u32 {
                    let ns = mesh.elem_nodes(e);
                    let c = |n: u32| { let p = mesh.node_coords(n); format!("({:.17},{:.17})", p[0], p[1]) };
                    println!("  {e}: {} {} {} {}", c(ns[0]), c(ns[1]), c(ns[2]), c(ns[3]));
                }
                // vertex view (MFEM UpdateVertices order) coordinates
                if let Some(view) = mesh.nc_vertex_view() {
                    println!("RNODE {}", view.len());
                    for (d, &n) in view.iter().enumerate() {
                        let p = mesh.node_coords(n);
                        println!("  {d}: ({:.17},{:.17})", p[0], p[1]);
                    }
                }
                // edge dof table (dof id order) midpoint coordinates
                println!("REDGE {}", dm.edge_dof_map.len());
                let mut edges: Vec<(&EdgeKey, u32)> = dm.edge_dof_map.iter().map(|(k, &v)| (k, v)).collect();
                edges.sort_by_key(|(_, v)| *v);
                for (key, dof) in edges {
                    let pa = mesh.node_coords(key.0);
                    let pb = mesh.node_coords(key.1);
                    println!("  {dof}: ({:.17},{:.17})", 0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]));
                }
                // count slaves per master via the same chain-walk as p2
                let mut total = 0usize;
                let mut masters_with = 0usize;
                for c in nc_state.constraints() {
                    let mut nslave = 0usize;
                    fn walk(x: u32, y: u32, mp: &std::collections::HashMap<(u32, u32), u32>, dm: &DofManager, n: &mut usize) {
                        let key = if x < y { (x, y) } else { (y, x) };
                        if let Some(&m) = mp.get(&key) {
                            if dm.edge_dof_map.contains_key(&EdgeKey::new(x, m)) { *n += 1; }
                            walk(x, m, mp, dm, n);
                            if dm.edge_dof_map.contains_key(&EdgeKey::new(m, y)) { *n += 1; }
                            walk(m, y, mp, dm, n);
                        }
                    }
                    walk(c.parent_a as u32, c.parent_b as u32, mp, dm, &mut nslave);
                    total += nslave;
                    if nslave > 0 { masters_with += 1; }
                }
                println!("TOTAL-SLAVES {total} masters-with-slaves {masters_with}");
                // trace the chain of the master whose edge-dof is 1171
                for c in nc_state.constraints() {
                    let e = dm.edge_dof_map.get(&EdgeKey::new(c.parent_a as u32, c.parent_b as u32)).copied();
                    if e == Some(1171) {
                        println!("  P1-M1171 {} <- {} + {}", c.constrained, c.parent_a, c.parent_b);
                        let mut x = c.parent_a as u32; let mut y = c.parent_b as u32;
                        for depth in 0..5 {
                            let key = if x < y { (x, y) } else { (y, x) };
                            let Some(&m) = mp.get(&key) else { println!("    depth{depth}: ({x},{y}) NO-MIDPOINT"); break; };
                            let e1 = dm.edge_dof_map.get(&EdgeKey::new(x, m)).copied();
                            let e2 = dm.edge_dof_map.get(&EdgeKey::new(m, y)).copied();
                            println!("    depth{depth}: ({x},{y}) mid={m} half1-dof={e1:?} half2-dof={e2:?}");
                            // diagnostic: physical node m coords + all edges/midpoints touching it
                            let pc = mesh.node_coords(m);
                            println!("      node{m}=({:.17},{:.17}) v2d={:?}", pc[0], pc[1], dm.phys_to_vertex_dof.get(&m));
                            let mut touch: Vec<String> = Vec::new();
                            for ((aa, bb), mm) in mp {
                                if *aa == m || *bb == m {
                                    let ca = mesh.node_coords(*aa); let cb = mesh.node_coords(*bb);
                                    let cm = mesh.node_coords(*mm);
                                    touch.push(format!("({aa},{bb})->{mm} geoA=({:.9},{:.9}) geoB=({:.9},{:.9}) geoM=({:.9},{:.9})",
                                        ca[0], ca[1], cb[0], cb[1], cm[0], cm[1]));
                                }
                            }
                            for t in touch.iter().take(12) { println!("      mp-touch: {t}"); }
                            let mut touch_edges: Vec<String> = Vec::new();
                            for (k, &d) in &dm.edge_dof_map {
                                if k.0 == m || k.1 == m {
                                    let ca = mesh.node_coords(k.0); let cb = mesh.node_coords(k.1);
                                    touch_edges.push(format!("({},{})->dof{} geo=({:.9},{:.9})-({:.9},{:.9})", k.0, k.1, d,
                                        ca[0], ca[1], cb[0], cb[1]));
                                }
                            }
                            for t in touch_edges.iter().take(12) { println!("      edge-touch: {t}"); }
                            // find any midpoint record whose mid is geometrically the
                            // t=1/8 point on (x, m) — i.e. mid of (x, m): the missing
                            // deep split record (C++ has (86,85)->85')
                            if depth >= 1 {
                                let ca = mesh.node_coords(x);
                                let cm = mesh.node_coords(m);
                                let target = [0.5 * (ca[0] + cm[0]), 0.5 * (ca[1] + cm[1])];
                                let mut found = 0;
                                for ((aa, bb), mm) in mp {
                                    let c1 = mesh.node_coords(*aa);
                                    let c2 = mesh.node_coords(*bb);
                                    let c3 = mesh.node_coords(*mm);
                                    if (c1[0] - target[0]).abs() < 1e-9 && (c1[1] - target[1]).abs() < 1e-9 {
                                        println!("      t1/8-as-A: node{aa} is the t=1/8 point, its midpoints: ...");
                                    }
                                    if (c3[0] - target[0]).abs() < 1e-9 && (c3[1] - target[1]).abs() < 1e-9 {
                                        found += 1;
                                        println!("      t1/8-mid: ({aa},{bb})->{mm} geo=({:.9},{:.9})-({:.9},{:.9})",
                                            c1[0], c1[1], c2[0], c2[1]);
                                    }
                                }
                                if found == 0 {
                                    println!("      t1/8-mid: NONE (target=({:.9},{:.9}))", target[0], target[1]);
                                }
                            }
                            x = x; y = m;
                        }
                    }
                }
            }
            let gf = GridFunction::new(&space, u.clone());
            // per-iteration P2 constraint dump (it3+) for 1:1 vs C++ ITERDEP
            if ref_it >= 3 {
                println!("ITERP2 it{ref_it} dofs={cdofs} elems={} rows={}", mesh.n_elems(), hc.len());
                // search element corners for the t=1/8 point (0.1298283828125,-0.01486028125)
                let mut t8 = 0;
                for e in 0..mesh.n_elems() as u32 {
                    let ns = mesh.elem_nodes(e);
                    for k in 0..4 {
                        let p = mesh.node_coords(ns[k]);
                        if (p[0] - 0.1298283828125_f64).abs() < 1e-9 && (p[1] + 0.01486028125_f64).abs() < 1e-9 { t8 += 1; }
                    }
                }
                println!("  T8CORNER {t8}");
                // global search: how many midpoint records have mid at the
                // t=1/8 geometry (0.129828383,-0.014860281) on master (69,369)
                let mp_g = nc_state.midpoints();
                let tx = 0.1298283828125_f64; let ty = -0.01486028125_f64;
                let mut cnt = 0; let mut keys: Vec<String> = Vec::new();
                for ((aa, bb), mm) in mp_g {
                    let c3 = mesh.node_coords(*mm);
                    if (c3[0] - tx).abs() < 1e-9 && (c3[1] - ty).abs() < 1e-9 {
                        cnt += 1;
                        keys.push(format!("({aa},{bb})->{mm}"));
                    }
                }
                println!("  T1/8-MID-COUNT {cnt} {keys:?}");
                for c in &hc {
                    print!("  {} <- {}:{}", c.constrained, c.parent_a, c.coeff_a);
                    if c.parent_b != c.parent_a || c.coeff_b != c.coeff_a {
                        print!(" {}:{}", c.parent_b, c.coeff_b);
                    }
                    for &(m, w) in &c.extra { print!(" {}:{:.4}", m, w); }
                    println!();
                }
            }
            if ((time - 0.01).abs() < 1e-9 || (time - 0.06).abs() < 1e-9) && ref_it == 1 {
                // manual estimator run (same inputs as refiner.apply below) so the
                // per-dof flux dump runs BEFORE the mesh is refined
                use fem_assembly::postproc::flux_recovery::zz_estimator_mfem_nc;
                let indicators = zz_estimator_mfem_nc(&gf, &diffusion, &hc);
                println!("ETA {}", indicators.eta.len());
                for (i, &e) in indicators.eta.iter().enumerate() { println!("{i} {e:.17e}"); }
                // ── per-dof flux composition dump (mirrors zz_estimator_mfem_nc
                //    internals: SumFluxAndCount + compute_element_flux + diff +
                //    compute_flux_energy), for elements 0/39/127 ──
                {
                    use fem_assembly::postproc::flux_recovery::FluxRecovery;
                    use fem_element::ReferenceElement;
                    use fem_element::lagrange::QuadQ2;
                    let re: Box<dyn ReferenceElement> = Box::new(QuadQ2);
                    let dof_coords = re.dof_coords();
                    let nd2 = space.n_dofs();
                    let dim2 = 2usize;
                    let n_ldofs = dof_coords.len();
                    let mut fsum = vec![vec![0.0f64; dim2]; nd2];
                    let mut fcnt = vec![0usize; nd2];
                    for e in 0..mesh.n_elems() as u32 {
                        let raw = diffusion.compute_element_flux(&mesh, &space, e, &u, &dof_coords);
                        let ed = space.element_dofs(e);
                        for (i, &gd) in ed.iter().enumerate() {
                            let idx = gd as usize;
                            for d in 0..dim2 { fsum[idx][d] += raw[i * dim2 + d]; }
                            fcnt[idx] += 1;
                        }
                    }
                    let mut favg = vec![vec![0.0f64; dim2]; nd2];
                    for i in 0..nd2 {
                        let c = fcnt[i] as f64;
                        if c > 0.0 { for d in 0..dim2 { favg[i][d] = fsum[i][d] / c; } }
                    }
                    let dump_elems: &[u32] = if (time - 0.06).abs() < 1e-9 { &[362, 407, 471] } else { &[0, 39, 118] };
                    for &e in dump_elems {
                        // ── diagnostics: quadrature points, mass matrix, and
                        //    pointwise interpolation at Q2_NODES ──
                        {
                            let quad = re.quadrature(4);
                            print!("E{e}QUAD");
                            for p in &quad.points { print!(" ({:.6},{:.6})", p[0], p[1]); }
                            println!();
                            let nodes = mesh.element_nodes(e);
                            let c = |i: usize| mesh.node_coords(nodes[i]);
                            let mut mass = vec![0.0f64; 81];
                            for (q, xi) in quad.points.iter().enumerate() {
                                let (e1, n1) = (xi[0], xi[1]);
                                let det_j = 0.25 * (-(1.0 - n1) * c(0)[0] + (1.0 - n1) * c(1)[0] + (1.0 + n1) * c(2)[0] - (1.0 + n1) * c(3)[0])
                                    * 0.25 * (-(1.0 - e1) * c(0)[1] - (1.0 + e1) * c(1)[1] + (1.0 + e1) * c(2)[1] + (1.0 - e1) * c(3)[1])
                                    - 0.25 * (-(1.0 - n1) * c(0)[1] + (1.0 - n1) * c(1)[1] + (1.0 + n1) * c(2)[1] - (1.0 + n1) * c(3)[1])
                                    * 0.25 * (-(1.0 - e1) * c(0)[0] - (1.0 + e1) * c(1)[0] + (1.0 + e1) * c(2)[0] + (1.0 - e1) * c(3)[0]);
                                let w_det = quad.weights[q] * det_j.abs();
                                let mut phi = vec![0.0f64; 9];
                                re.eval_basis(xi, &mut phi);
                                for i in 0..9 { for j in 0..9 { mass[i * 9 + j] += w_det * phi[i] * phi[j]; } }
                            }
                            let mut offmax = 0.0f64;
                            for i in 0..9 { for j in 0..9 { if i != j { offmax = offmax.max(mass[i * 9 + j].abs()); } } }
                            println!("E{e}MASS diag={:.6e}..{:.6e} offmax={:.6e}",
                                (0..9).map(|i| mass[i * 9 + i]).fold(f64::INFINITY, f64::min),
                                (0..9).map(|i| mass[i * 9 + i]).fold(0.0f64, f64::max), offmax);
                            // pointwise κ∇u_h at Q2_NODES (interpolation)
                            let q2n = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
                                       (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0)];
                            let ed = space.element_dofs(e);
                            let mut gr = vec![0.0f64; 18];
                            for (k, &(qx, qy)) in q2n.iter().enumerate() {
                                let xi = [qx, qy];
                                let (e1, n1) = (qx, qy);
                                let j00 = 0.25 * (-(1.0 - n1) * c(0)[0] + (1.0 - n1) * c(1)[0] + (1.0 + n1) * c(2)[0] - (1.0 + n1) * c(3)[0]);
                                let j01 = 0.25 * (-(1.0 - e1) * c(0)[0] - (1.0 + e1) * c(1)[0] + (1.0 + e1) * c(2)[0] + (1.0 - e1) * c(3)[0]);
                                let j10 = 0.25 * (-(1.0 - n1) * c(0)[1] + (1.0 - n1) * c(1)[1] + (1.0 + n1) * c(2)[1] - (1.0 + n1) * c(3)[1]);
                                let j11 = 0.25 * (-(1.0 - e1) * c(0)[1] - (1.0 + e1) * c(1)[1] + (1.0 + e1) * c(2)[1] + (1.0 - e1) * c(3)[1]);
                                let det = j00 * j11 - j01 * j10;
                                let jit = [[j11 / det, -j01 / det], [-j10 / det, j00 / det]];
                                re.eval_grad_basis(&xi, &mut gr);
                                let mut g = [0.0f64; 2];
                                for i in 0..9 {
                                    let gx = jit[0][0] * gr[i * 2] + jit[0][1] * gr[i * 2 + 1];
                                    let gy = jit[1][0] * gr[i * 2] + jit[1][1] * gr[i * 2 + 1];
                                    let v = u[ed[i] as usize];
                                    g[0] += v * gx; g[1] += v * gy;
                                }
                                println!("E{e}GRAD {k} {:.17e} {:.17e}", diffusion.kappa * g[0], diffusion.kappa * g[1]);
                            }
                        }
                        let raw = diffusion.compute_element_flux(&mesh, &space, e, &u, &dof_coords);
                        let ed = space.element_dofs(e);
                        println!("E{e}EDOF");
                        for (i, &gd) in ed.iter().enumerate() {
                            let mut cs = String::new();
                            for c in &hc {
                                if c.constrained as u32 == gd {
                                    cs.push_str(&format!(" [{} <- {}:{:.6} {}:{:.6} extra={:?}]",
                                        gd, c.parent_a, c.coeff_a, c.parent_b, c.coeff_b, c.extra));
                                }
                            }
                            println!("E{e}EDOF {i} gd={gd} u={:.17e}{cs}", u[gd as usize]);
                        }
                        println!("E{e}FLD {n_ldofs} {dim2}");
                        for (i, &gd) in ed.iter().enumerate() {
                            for d in 0..dim2 {
                                println!("E{e}FL {i} {d} {:.17e}", raw[i * dim2 + d]);
                            }
                        }
                        for (i, &gd) in ed.iter().enumerate() {
                            for d in 0..dim2 {
                                println!("E{e}FLA {i} {d} {:.17e}", favg[gd as usize][d]);
                            }
                        }
                        let mut diff = vec![0.0f64; n_ldofs * dim2];
                        for (i, &gd) in ed.iter().enumerate() {
                            for d in 0..dim2 {
                                diff[i * dim2 + d] = raw[i * dim2 + d] - favg[gd as usize][d];
                                println!("E{e}DIFF {i} {d} {:.17e}", diff[i * dim2 + d]);
                            }
                        }
                        let eng = diffusion.compute_flux_energy(&mesh, e, &diff);
                        println!("E{e}ENG {eng:.17e} {}", eng.sqrt());
                        print!("E{e}UL");
                        for &gd in ed.iter() { print!(" {:.17e}", u[gd as usize]); }
                        println!();
                    }
                }
            }
            let ne_before = mesh.n_elems();
            refiner.apply(&mut mesh, &mut nc_state, &gf, &diffusion, Some(&hc));
            let ne_after = mesh.n_elems();
            let n_marked = if ne_after > ne_before { refiner.last_marked.len() } else { 0 };
            if (dump_t001 || dump_t002) && !marked_printed {
                marked_printed = true;
                println!("MARKED {:?}", refiner.last_marked);
            }

            if refiner.stop() { break; }
            ref_it += 1;
            let _ = n_marked;
        }
        if derefiner.apply(&mut mesh, &mut nc_state, &mut refiner) {
            // derefined
        }
        time += dt;
    }
}
