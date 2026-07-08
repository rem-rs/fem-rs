//! # MFEM Example 9 — DG Advection
//! Reference: `mfem/ex9.cpp`
//! `du/dt + v·∇u = 0` using DG with upwind flux.
//!
//! Uses library DGAdvectionIntegrator + assemble_dg_interior_faces.

use std::time::Instant;
use fem_assembly::dg::dg_advection::{
    DGAdvectionIntegrator, assemble_dg_interior_faces, assemble_advection_boundary,
};
use fem_assembly::interior_faces::InteriorFaceList;
use fem_assembly::Assembler;
use fem_assembly::postproc::coefficient::ConstantVectorCoeff;
use fem_assembly::standard::MassIntegrator;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{ode::Rk4, TimeStepper};
use fem_space::{L2Space, fe_space::FESpace};

fn lump(m: &CsrMatrix<f64>) -> Vec<f64> {
    (0..m.nrows).map(|i| { let mut s=0.0;
        for k in m.row_ptr[i]..m.row_ptr[i+1] { s+=m.values[k]; } s }).collect()
}

fn main() {
    let args=Args::parse(); let wall=Instant::now();

    // Create unit-square mesh
    let mesh = Mesh::<2>::unit_square_tri(8);
    let mesh = if args.refine>0 { let mut m=mesh;
        for _ in 0..args.refine { m=refine_uniform(&m); } m
    } else { mesh };
    eprintln!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    let space = L2Space::new(mesh, args.order);
    let n=space.n_dofs(); println!("Number of unknowns: {}", n);
    let q=args.order*2+1;

    // Mass matrix and lumped diagonal
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator{rho:1.0}], q);
    let md = lump(&mass);

    // DG advection operator (volume + upwind face flux)
    let dg_adv = DGAdvectionIntegrator{velocity:ConstantVectorCoeff(vec![1.0,0.0])};
    let vel_bc = ConstantVectorCoeff(vec![1.0,0.0]);
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg_adv], q);
    let ifl = InteriorFaceList::build(space.mesh());
    let mut coo = CooMatrix::new(n,n);
    for i in 0..n { for k in k_vol.row_ptr[i]..k_vol.row_ptr[i+1] {
        coo.add(i, k_vol.col_idx[k] as usize, k_vol.values[k]); }}
    assemble_dg_interior_faces(&mut coo, space.mesh(), &space, &ifl, args.order, q, &dg_adv);
    // Add outflow boundary contribution: ∫_{∂Ω_out} v (b·n) u
    let b_vec = [1.0, 0.0];
    let nbf = space.mesh().n_boundary_faces();
    eprintln!("  Boundary faces: {}", nbf);
    for f in 0..nbf as u32 {
        let fnodes = space.mesh().face_nodes(f);
        let xa = space.mesh().node_coords(fnodes[0]);
        let xb = space.mesh().node_coords(fnodes[1]);
        let dx = xb[0]-xa[0]; let dy = xb[1]-xa[1];
        let h = (dx*dx+dy*dy).sqrt();
        let nx = dy/h; let ny = -dx/h;
        let vn = b_vec[0]*nx + b_vec[1]*ny;
        if vn <= 0.0 { continue; } // outflow only (b·n > 0)

        // Find the element containing this face (face_to_elem may be None)
        let msh = space.mesh();
        let el = (0..msh.n_elems() as u32).find(|&e| {
            let en = msh.element_nodes(e);
            en.contains(&fnodes[0]) && en.contains(&fnodes[1])
        }).unwrap_or(0);
        let dofs: Vec<usize> = space.element_dofs(el).iter().map(|&d|d as usize).collect();
        let nd = dofs.len();
        // One-point quadrature at face midpoint: φ = 0.5 for the two face vertices
        let enodes = space.mesh().element_nodes(el);
        let mut phi = vec![0.0; nd];
        for i in 0..nd {
            if enodes[i] == fnodes[0] || enodes[i] == fnodes[1] { phi[i] = 0.5; }
        }
        for i in 0..nd { for j in 0..nd {
            coo.add(dofs[i], dofs[j], h * phi[i] * vn * phi[j]);
        }}
    }

    let k_adv:CsrMatrix<f64> = coo.into_csr();

    // Boundary RHS (inflow BC: g_D = 0)
    let rhs_bc = assemble_advection_boundary(&space, &vel_bc, &[1,2,3,4],
        &|_| 0.0, args.order, q);
    eprintln!("  Boundary RHS: max={:.4e}", rhs_bc.iter().cloned().fold(0.0_f64,f64::max));
    eprintln!("  K_adv: {} nnz", k_adv.nnz());

    // Dominant eigenvalue via power iteration
    let mut r = vec![1.0; n]; let mut k1 = vec![0.0; n];
    for _ in 0..100 {
        k_adv.spmv(&r, &mut k1);
        let norm = k1.iter().map(|v|v*v).sum::<f64>().sqrt().max(1e-30);
        for i in 0..n { r[i] = k1[i]/norm; }
    }
    k_adv.spmv(&r, &mut k1);
    let r2: f64 = r.iter().map(|v|v*v).sum();
    let lam = r.iter().zip(k1.iter()).map(|(a,b)|a*b).sum::<f64>() / r2.max(1e-30);
    eprintln!("  Max eigenvalue(λ_max) of K: {:.6e}", lam);

    // Initial condition: smooth Gaussian
    let iv = space.interpolate(&|x| (-40.0*((x[0]-0.5).powi(2)+(x[1]-0.5).powi(2))).exp());
    let mut u:Vec<f64> = iv.as_slice().to_vec();
    let mut mu=vec![0.0;n]; mass.spmv(&u,&mut mu);
    let l2_0 = u.iter().zip(mu.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt();
    let mx_0 = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("  Initial: L2={:.4e}, max={:.4e}", l2_0, mx_0);

    // Time integration: du/dt = -M_lump^(-1) * (K * u + rhs_bc)
    let (k_ref, rhs_ref, md_ref) = (&k_adv, &rhs_bc, &md);
    let rk4 = Rk4; let dt = args.dt.min(args.t_final);
    let n_steps = (args.t_final/dt).ceil() as usize;
    let mut t=0.0;

    for step in 0..n_steps {
        let dta = dt.min(args.t_final - t);
        rk4.step(t, dta, &mut u, |_t, u, dudt| {
            let mut tmp = rhs_ref.clone();
            k_ref.spmv(u, &mut tmp);
            for i in 0..n { dudt[i] = -tmp[i] / md_ref[i]; }
        });
        t += dta;
        if (step+1)%200==0||step+1==n_steps {
            let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            eprintln!("  t={:.4}  max={:.4e}", t, mx);
        }
        if t>=args.t_final-1e-14 { break; }
    }

    let mut mu2=vec![0.0;n]; mass.spmv(&u,&mut mu2);
    let l2 = u.iter().zip(mu2.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt();
    let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("\nL2 norm of u: {:.10e} (initial: {:.4e})", l2, l2_0);
    println!("max norm of u: {:.10e} (initial: {:.4e})", mx, mx_0);
    eprintln!("  Wall: {:.3}s", wall.elapsed().as_secs_f64());
}

#[allow(dead_code)]
struct Args{mesh:String,problem:usize,refine:usize,order:u8,dt:f64,t_final:f64,no_vis:bool}
impl Args {
    fn parse()->Self {
        let(mut p,mut r,mut o,mut dt,mut tf)=(0usize,2usize,1u8,0.001_f64,0.5_f64);
        let mut nv=false; let mut it=std::env::args().skip(1);
        while let Some(a)=it.next(){match a.as_str(){
            "-p"|"--problem"=>{p=it.next().and_then(|s|s.parse().ok()).unwrap_or(0);}
            "-r"|"--refine"=>{r=it.next().and_then(|s|s.parse().ok()).unwrap_or(2);}
            "-o"|"--order"=>{o=it.next().and_then(|s|s.parse().ok()).unwrap_or(1);}
            "-dt"|"--time-step"=>{dt=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.001);}
            "-tf"|"--final-time"=>{tf=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.5);}
            "-no-vis"|"--no-visualization"=>{nv=true;}
            _=>{}
        }}
        Args{mesh:String::new(),problem:p,refine:r,order:o as u8,dt,t_final:tf,no_vis:nv}
    }
}
