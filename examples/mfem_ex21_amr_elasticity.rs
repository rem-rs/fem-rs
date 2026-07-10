//! Example 21 鈥?AMR for linear elasticity (2D). 1:1 translation of MFEM ex21.

#![allow(non_snake_case, dead_code)]
use fem_assembly::postproc::coefficient::PWConstCoeff;
use fem_assembly::postproc::error_estimate::{threshold_mark, zz_estimator};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;
use fem_linalg::SolverConfig;
use fem_mesh::{topology::MeshTopology, Mesh};
use fem_space::constraints::boundary_dofs;
use fem_space::{FESpace, VectorH1Space};

use fem_element::lagrange::SegP1;
use fem_element::ReferenceElement;

fn bdr_load<M: MeshTopology + Clone>(sp: &VectorH1Space<M>, attr: u32, fx: f64, fy: f64) -> Vec<f64> {
    let mesh = sp.mesh(); let dim = mesh.dim() as usize; let nd = sp.n_dofs(); let _ns = sp.n_scalar_dofs();
    let mut r = vec![0.0; nd];
    let nf = mesh.n_boundary_faces();
    for face in 0..nf as u32 {
        if mesh.face_tag(face) != attr as i32 { continue; }
        let (elem,_) = mesh.face_elements(face);
        let fnodes = mesh.face_nodes(face);
        let enodes = mesh.element_nodes(elem);
        let edofs: Vec<usize> = sp.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let _n_ldofs = edofs.len() / dim;
        let mut fdmap: Vec<usize> = Vec::new();
        for &fn_ in fnodes {
            if let Some(p) = enodes.iter().position(|&en| en == fn_) {
                for d in 0..dim { fdmap.push(p * dim + d); }
            }
        }
        let x0 = mesh.node_coords(fnodes[0]); let x1 = mesh.node_coords(fnodes[1]);
        let el = (0..dim).map(|i| (x1[i]-x0[i]).powi(2)).sum::<f64>().sqrt();
        let mut t = vec![0.0; dim]; let mut n = vec![0.0; dim];
        if el > 1e-30 { for i in 0..dim { t[i] = (x1[i]-x0[i])/el; } }
        n[0] = t[1]; n[1] = -t[0];
        let seg = SegP1; let q = seg.quadrature(3);
        for (qi,xi) in q.points.iter().enumerate() {
            let mut ph = vec![0.0; fnodes.len()];
            ph[0] = 0.5*(1.0-xi[0]); ph[1] = 0.5*(1.0+xi[0]);
            let w = q.weights[qi] * (el/2.0);
            let f = [fx, fy];
            for j in 0..fnodes.len() {
                for d in 0..dim {
                    let idx = fdmap[j*dim+d];
                    if idx < nd { r[idx] += w * ph[j] * f[d]; }
                }
            }
        }
    }
    r
}

fn main() {
    let mut mf = "data/beam-quad.mesh".to_string();
    let mut o = 1u8; let max_dofs = 50000usize; let max_amr = 20usize;
    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "-h" => { eprintln!("Usage: ex21 [-m mesh] [-o order]"); return; }
            "-m" | "--mesh" => { mf = a.next().unwrap_or_default(); }
            "-o" | "--order" => { o = a.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            _ => {}
        }
    }
    println!("Options:\n  --mesh {mf}\n  --order {o}");

    let data = fem_io::mfem::read_mfem_file(&mf).expect("read mesh");
    let mut mesh: Mesh<2> = data.mesh2d.expect("2D mesh");
    let dim = 2usize; let cfg = SolverConfig{rtol:1e-12,atol:0.,max_iter:2000,..SolverConfig::default()};

    let sp0 = VectorH1Space::new(mesh.clone(), o, dim as u8);
    let dm = sp0.scalar_dof_manager(); let ns = sp0.n_scalar_dofs();
    let bdry = boundary_dofs(sp0.mesh(), dm, &[1]);
    let ess: Vec<usize> = bdry.iter().flat_map(|&d| vec![d as usize, d as usize+ns]).collect();

    for it in 0..=max_amr {
        let lp = PWConstCoeff::new([(1,50.0),(2,1.0)]);
        let mu = PWConstCoeff::new([(1,50.0),(2,1.0)]);
        let sp = VectorH1Space::new(mesh.clone(), o, dim as u8);
        let n = sp.n_dofs();
        println!("\nAMR iteration {it}"); println!("Unknowns: {n}");

        let stiff = Assembler::assemble_bilinear(&sp, &[&ElasticityIntegrator::new(lp, mu)], 3);
        let mut rhs = vec![0.0; n]; for i in (n/2)..n { rhs[i] = -0.01; }
        let mut A = stiff.clone(); let mut X = vec![0.0; n];
        for &d in &ess { A.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }

        match fem_solver::solve_pcg_gssmoother(&A, &rhs, &mut X, &cfg) {
            Ok(r) => println!("  PCG: {} its res={:.3e}", r.iterations, r.final_residual),
            Err(e) => eprintln!("  PCG: {e}"),
        }
        let gf = GridFunction::new(&sp, X);
        let ind = zz_estimator(&gf);
        let me = ind.eta.iter().cloned().fold(0.0_f64, f64::max);
        println!("  Max err: {me:.6e}");
        let marked = threshold_mark(&ind.eta, 0.7*me);
        println!("  Marked {}", marked.len());
        if n > max_dofs || marked.is_empty() { println!("Stop."); break; }
        mesh = fem_mesh::amr::refine_nonconforming_quad(&mesh, &marked).0;
    }
}





