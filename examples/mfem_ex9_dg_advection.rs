#![allow(dead_code, unused_variables, non_snake_case)]
//! MFEM Example 9 — DG Advection (best-effort 1:1)
//!
//! du/dt + v·∇u = 0 using DG with upwind flux.
//!
//! Library DGAdvectionIntegrator uses formula -∫ (v·∇φ_i) φ_j (gradient on test),
//! which after negation in the RHS gives the correct advection operator.
//! This sign convention matches the existing fem-rs DG infrastructure.
//!
//! ## Usage
//! cargo run --release --example mfem_ex9_dg_advection -- -m data/periodic-square.mesh -p 0 -r 2 -o 1 -dt 0.005 -tf 10

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

fn main() {
    let args = Args::parse();
    let mesh: Mesh<2> = if !args.mesh.is_empty() {
        fem_io::mfem::read_mfem_file(&args.mesh).expect("read mesh").mesh2d.expect("2D mesh")
    } else { Mesh::<2>::unit_square_tri(4) };
    let mesh = if args.refine > 0 { let mut m = mesh; for _ in 0..args.refine { m = refine_uniform(&m); } m } else { mesh };

    let space = L2Space::new(mesh.clone(), args.order as u8);
    let n = space.n_dofs(); println!("Number of unknowns: {n}");
    let qo = args.order as u8 * 2 + 1;
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator{rho:1.0}], qo);

    let vel = [(2.0_f64/3.0).sqrt(), (1.0_f64/3.0).sqrt()];
    let dg = DGAdvectionIntegrator{velocity: ConstantVectorCoeff(vel.to_vec())};

    // Assemble K = volume + interior face upwind
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg], qo);
    let ifl = InteriorFaceList::build(space.mesh());
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n { for p in k_vol.row_ptr[i]..k_vol.row_ptr[i+1] { coo.add(i, k_vol.col_idx[p] as usize, k_vol.values[p]); }}
    assemble_dg_interior_faces(&mut coo, space.mesh(), &space, &ifl, args.order as u8, qo, &dg);

    // Outflow boundary (∮ v·n·u·w from IBP of volume term)
    for bf in 0..mesh.n_boundary_faces() as u32 {
        let f = mesh.face_nodes(bf);
        let (el, _) = mesh.face_elements(bf);
        let pa = mesh.node_coords(f[0]); let pb = mesh.node_coords(f[1]);
        let h = ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt();
        let vn = vel[0]*(pb[1]-pa[1])/h + vel[1]*(-(pb[0]-pa[0]))/h;
        if vn <= 0.0 { continue; }
        let dofs: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
        let en = mesh.element_nodes(el);
        // Exact edge mass: ∫ φi·φj ds = h * [1/3, 1/6; 1/6, 1/3] for P1 edge
        // Cancels the IBP boundary artifact from the volume term
        let find = |n| en.iter().position(|&e| e==n);
        if let (Some(p0), Some(p1)) = (find(f[0]), find(f[1])) {
            coo.add(dofs[p0], dofs[p0], h * vn / 3.0);
            coo.add(dofs[p0], dofs[p1], h * vn / 6.0);
            coo.add(dofs[p1], dofs[p0], h * vn / 6.0);
            coo.add(dofs[p1], dofs[p1], h * vn / 3.0);
        }
    }
    let k_adv: CsrMatrix<f64> = coo.into_csr();

    // Initial condition
    let mut u: Vec<f64> = space.interpolate(&|x| (-40.0*((x[0]-0.5).powi(2)+(x[1]-0.5).powi(2))).exp()).as_slice().to_vec();

    // Time integration: dudt = -M^{-1} * K * u
    let cg_cfg = SolverConfig{rtol:1e-9, max_iter:200, verbose:false, ..Default::default()};
    let dt = args.dt.min(args.t_final);
    let n_steps = (args.t_final/dt).ceil() as usize;
    let mut t = 0.0;
    for step in 0..n_steps {
        let dta = dt.min(args.t_final - t);
        Rk4.step(t, dta, &mut u, |_, u, dudt| {
            let mut f = vec![0.0; n];
            k_adv.spmv(u, &mut f);
            for v in &mut f { *v = -*v; }
            let _ = solve_cg(&mass, &f, dudt, &cg_cfg);
        });
        t += dta;
        if (step+1)%200==0||step+1==n_steps { eprintln!("  t={t:.4e}"); }
        if t>=args.t_final-1e-14 { break; }
    }
    let mut mu=vec![0.0;n]; mass.spmv(&u,&mut mu);
    println!("L2 norm of u: {:.10e}", u.iter().zip(mu.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt());
}

struct Args{mesh:String,refine:usize,order:u8,dt:f64,t_final:f64}
impl Args {
    fn parse()->Self {
        let(mut m,mut r,mut o,mut dt,mut tf)=(String::new(),2usize,1u8,0.005_f64,10.0_f64);
        let mut it=std::env::args().skip(1);
        while let Some(a)=it.next(){match a.as_str(){
            "-m"|"--mesh"=>{m=it.next().unwrap_or(m);}
            "-r"|"--refine"=>{r=it.next().and_then(|s|s.parse().ok()).unwrap_or(2);}
            "-o"|"--order"=>{o=it.next().and_then(|s|s.parse().ok()).unwrap_or(1);}
            "-dt"|"--time-step"=>{dt=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.005);}
            "-tf"|"--final-time"=>{tf=it.next().and_then(|s|s.parse().ok()).unwrap_or(10.0);}
            _=>{}
        }}
        Args{mesh:m,refine:r,order:o as u8,dt,t_final:tf}
    }
}
