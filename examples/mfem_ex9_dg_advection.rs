#![allow(dead_code, unused_variables, unused_imports, non_snake_case)]
//! 1:1 translation of MFEM ex9 — DG Advection
//! du/dt + v·∇u = 0 using DG with upwind flux.
//! Uses the existing DGAdvectionIntegrator from fem-assembly.
//!
//! ## Usage
//! ```bash
//! cargo run --release --example mfem_ex9_dg_advection -- -m data/periodic-square.mesh -p 0 -r 2 -o 1 -dt 0.005 -tf 10
//! ```

use fem_assembly::{
    Assembler,
    dg::dg_advection::{DGAdvectionIntegrator, assemble_dg_interior_faces},
    interior_faces::InteriorFaceList,
    postproc::coefficient::ConstantVectorCoeff,
    standard::MassIntegrator,
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{refine_uniform, topology::MeshTopology, Mesh};
use fem_solver::{SolverConfig, solve_cg, ode::{Rk4, TimeStepper}};
use fem_space::{L2Space, fe_space::FESpace};
use std::fs::File;
use std::io::Write;

fn main() {
    let args = Args::parse();
    let mesh: Mesh<2> = if !args.mesh.is_empty() {
        fem_io::mfem::read_mfem_file(&args.mesh).expect("read mesh").mesh2d.expect("2D mesh")
    } else { Mesh::<2>::unit_square_tri(4) };
    let mesh = if args.refine > 0 { let mut m = mesh; for _ in 0..args.refine { m = refine_uniform(&m); } m } else { mesh };
    eprintln!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // DG space + mass
    let space = L2Space::new(mesh.clone(), args.order as u8);
    let n = space.n_dofs();
    println!("Number of unknowns: {n}");
    let qo = args.order as u8 * 2 + 1;
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator{rho:1.0}], qo);

    // DG advection integrator (volume + face)
    let vel = match args.problem {
        0 => [(2.0_f64/3.0).sqrt(), (1.0_f64/3.0).sqrt()],
        _ => [0.0, 0.0],
    };
    let dg = DGAdvectionIntegrator{velocity: ConstantVectorCoeff(vel.to_vec())};

    // K = volume + interior faces
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg], qo);
    let ifl = InteriorFaceList::build(space.mesh());
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n { for p in k_vol.row_ptr[i]..k_vol.row_ptr[i+1] { coo.add(i, k_vol.col_idx[p] as usize, k_vol.values[p]); } }
    assemble_dg_interior_faces(&mut coo, space.mesh(), &space, &ifl, args.order as u8, qo, &dg);

    // Outflow boundary: add +∫ vn·u·w ds so -K_bdry * u cancels IBP artifact (-∫ vn·u·w)
    for bf in 0..mesh.n_boundary_faces() as u32 {
        let fnodes = mesh.face_nodes(bf);
        let (el, _) = mesh.face_elements(bf);
        let pa = mesh.node_coords(fnodes[0]); let pb = mesh.node_coords(fnodes[1]);
        let h = ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt();
        let vn = vel[0]*(pb[1]-pa[1])/h + vel[1]*(-(pb[0]-pa[0]))/h;
        if vn <= 0.0 { continue; }
        let dofs: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
        let nd = dofs.len();
        let en = mesh.element_nodes(el);
        let mut p = [nd, nd];
        for k in 0..nd {
            if en[k] == fnodes[0] { p[0] = k; }
            if en[k] == fnodes[1] { p[1] = k; }
        }
        if p[0] < nd && p[1] < nd {
            coo.add(dofs[p[0]], dofs[p[0]], vn * h / 3.0);
            coo.add(dofs[p[0]], dofs[p[1]], vn * h / 6.0);
            coo.add(dofs[p[1]], dofs[p[0]], vn * h / 6.0);
            coo.add(dofs[p[1]], dofs[p[1]], vn * h / 3.0);
        }
    }
    let k_adv: CsrMatrix<f64> = coo.into_csr();

    // Initial condition (Gaussian bump for all problems)
    let mut u: Vec<f64> = space.interpolate(&|x| (-40.0*((x[0]-0.5).powi(2)+(x[1]-0.5).powi(2))).exp()).as_slice().to_vec();
    let mut mu = vec![0.0; n]; mass.spmv(&u, &mut mu);
    let l2_0 = u.iter().zip(mu.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt();
    eprintln!("  Initial L2={l2_0:.6e}");

    // Time integration: dudt = -M^{-1} * K * u
    let cg_cfg = SolverConfig{rtol:1e-9, max_iter:200, verbose:false, ..Default::default()};
    let dt = args.dt.min(args.t_final);
    let n_steps = (args.t_final/dt).ceil() as usize;
    let mut t = 0.0;
    for step in 0..n_steps {
        let dta = dt.min(args.t_final - t);
        Rk4.step(t, dta, &mut u, |_t, u, dudt| {
            let tmp = &k_adv;
            let mut f = vec![0.0; n];
            tmp.spmv(u, &mut f);
            for v in &mut f { *v = -*v; }
            let _ = solve_cg(&mass, &f, dudt, &cg_cfg);
        });
        t += dta;
        if (step+1)%200==0||step+1==n_steps {
            let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            eprintln!("  t={t:.4e}  max={mx:.6e}");
        }
        if t>=args.t_final-1e-14 { break; }
    }

    let mut mu2=vec![0.0;n]; mass.spmv(&u,&mut mu2);
    let l2 = u.iter().zip(mu2.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt();
    let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("\nL2 norm of u: {l2:.10e} (initial: {l2_0:.6e})");
    println!("max norm of u: {mx:.10e}");
}

struct Args{mesh:String,problem:usize,refine:usize,order:u8,dt:f64,t_final:f64,no_vis:bool}
impl Args {
    fn parse()->Self {
        let(mut mesh,mut p,mut r,mut o,mut dt,mut tf)=(String::new(),0usize,2usize,1u8,0.005_f64,10.0_f64);
        let mut nv=false; let mut it=std::env::args().skip(1);
        while let Some(a)=it.next(){match a.as_str(){
            "-m"|"--mesh"=>{mesh=it.next().unwrap_or(mesh);}
            "-p"|"--problem"=>{p=it.next().and_then(|s|s.parse().ok()).unwrap_or(0);}
            "-r"|"--refine"=>{r=it.next().and_then(|s|s.parse().ok()).unwrap_or(2);}
            "-o"|"--order"=>{o=it.next().and_then(|s|s.parse().ok()).unwrap_or(1);}
            "-dt"|"--time-step"=>{dt=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.005);}
            "-tf"|"--final-time"=>{tf=it.next().and_then(|s|s.parse().ok()).unwrap_or(10.0);}
            "-no-vis"|"--no-visualization"=>{nv=true;}
            _=>{}
        }}
        Args{mesh,problem:p,refine:r,order:o as u8,dt,t_final:tf,no_vis:nv}
    }
}
