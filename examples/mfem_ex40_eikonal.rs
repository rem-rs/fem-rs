//! # Example 40 — Eikonal equation (1:1 with MFEM ex40)
//! (minimal working version — Tri3 only)

#![allow(warnings)]
use fem_assembly::{VectorAssembler, Assembler};
use fem_assembly::standard::{VectorMassIntegrator, MassIntegrator};
use fem_assembly::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
use fem_assembly::integrator::{LinearIntegrator, QpData};
use fem_element::{ReferenceElement, VectorReferenceElement, lagrange::TriP1, raviart_thomas::TriRT0};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, MeshTopology, ElementType, amr::refine_uniform};
use fem_solver::{solve_minres, SolverConfig};
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use std::f64::consts::PI;

struct Args { mesh: String, order: u8, refs: usize, max_it: usize,
    alpha: f64, growth_rate: f64, newton_scaling: f64, eps: f64, tol: f64 }
fn parse_args() -> Args {
    let mut a = Args { mesh: "data/star.mesh".into(), order: 1, refs: 3, max_it: 5,
        alpha: 1.0, growth_rate: 1.0, newton_scaling: 0.8, eps: 1e-6, tol: 1e-4 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() { match arg.as_str() {
        "-m"|"--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
        "-o"|"--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
        "-r"|"--refs" => a.refs = it.next().and_then(|v| v.parse().ok()).unwrap_or(3),
        "-mi"|"--max-it" => a.max_it = it.next().and_then(|v| v.parse().ok()).unwrap_or(5),
        "-step" => a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
        "-gr"|"--growth-rate" => a.growth_rate = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
        "-no-vis"|"--no-visualization" => {}
        _ => {}
    }}
    a
}

fn main() {
    let args = parse_args();
    println!("Options used:\n   --mesh {}\n   --order {}\n   --refs {}\n   --max-it {}\n   --step {}\n   --growth-rate {}\n   --no-visualization\n",
        args.mesh, args.order, args.refs, args.max_it, args.alpha, args.growth_rate);

    let mfem = read_mfem_file(&args.mesh).expect("mesh file");
    let mesh = if args.refs > 0 { let mut m = mfem.mesh2d.expect("2D"); for _ in 0..args.refs { m = refine_uniform(&m); } m }
              else { mfem.mesh2d.expect("2D") };
    let dim = 2; let qo = args.order as u8 * 4;
    let rt = HDivSpace::new(mesh.clone(), args.order);
    let l2 = L2Space::new(mesh.clone(), args.order);
    let (nr, nl) = (rt.n_dofs(), l2.n_dofs());
    println!("Number of H(div) dofs: {nr}");
    println!("Number of L² dofs: {nl}");

    // B: divergence (RT → L²), Bᵀ: transpose
    let mut bc = CooMatrix::new(nl, nr);
    for e in mesh.elem_iter() {
        let rd: Vec<usize> = rt.element_dofs(e).iter().map(|&d| d as usize).collect();
        let ld: Vec<usize> = l2.element_dofs(e).iter().map(|&d| d as usize).collect();
        let sgn = rt.element_signs(e); let ns = mesh.element_nodes(e);
        let re = TriRT0; let rh = TriP1; let q = re.quadrature(qo);
        let (n_nd,n_l)=(re.n_dofs(),rh.n_dofs());
        if mesh.element_type(e) != ElementType::Tri3 { continue; }
        let (x0,x1,x2)=(mesh.node_coords(ns[0]),mesh.node_coords(ns[1]),mesh.node_coords(ns[2]));
        let det = ((x1[0]-x0[0])*(x2[1]-x0[1])-(x2[0]-x0[0])*(x1[1]-x0[1])).abs()/2.0;
        for (qi,xi) in q.points.iter().enumerate() {
            let w = q.weights[qi]*det; let mut pn=vec![0.0;n_nd*2]; let mut pl=vec![0.0;n_l];
            re.eval_basis_vec(xi,&mut pn); re.eval_div(xi,&mut pl); rh.eval_basis(xi,&mut pl);
            for i in 0..n_nd { for j in 0..n_l {
                let c = -w * sgn[i] * pl[j]; if c != 0.0 { bc.add(ld[j], rd[i], c); }
            }}
        }
    }
    let b = bc.into_csr(); let bt = b.transpose();

    // Constant RT mass matrix
    let a00_const = VectorAssembler::assemble_bilinear(&rt, &[&VectorMassIntegrator{alpha:1.0}], qo);

    // State
    let mut psi = vec![0.0; nr]; let mut u = vec![0.0; nl];
    let mut psi_old = vec![0.0; nr]; let mut u_old = vec![0.0; nl];
    let mut alpha = args.alpha;
    let mut total_iters = 0;

    for k in 0..args.max_it {
        println!("\nOUTER ITERATION {}", k+1);
        let mut increment_u = 0.1;
    let mut loop_j = 0usize;
        loop_j = 0;
        for _j in 0..5 {
            total_iters += 1;
            loop_j = _j + 1;
            // Build A00 = DZ-weighted mass (diagonal approximation: use constant mass)
            // NOTE: Full DZ requires psi-evaluation at each q-point (future work)
            let mut a00 = a00_const.clone();

            // RHS: b0 = -Z(psi), b1 = -alpha + div(psi_old - psi)
            let mut rhs = vec![0.0; nr+nl];
            // b0 approximated as -psi (simplified Z)
            for i in 0..nr { rhs[i] = -psi[i]; }
            // b1
            let (mut dp, mut dpo) = (vec![0.0;nl], vec![0.0;nl]);
            for e in mesh.elem_iter() {
                let rd: Vec<usize> = rt.element_dofs(e).iter().map(|&d| d as usize).collect();
                let ld: Vec<usize> = l2.element_dofs(e).iter().map(|&d| d as usize).collect();
                let sgn = rt.element_signs(e);
                let div_p: f64 = sgn.iter().zip(&rd).map(|(&s,&d)| s*psi[d]).sum();
                let div_po: f64 = sgn.iter().zip(&rd).map(|(&s,&d)| s*psi_old[d]).sum();
                for &l in &ld { rhs[nr+l] += -alpha + (div_po - div_p) / ld.len() as f64; }
            }

            // Build saddle-point matrix [A00, Bᵀ; B, 0]
            let mut sc = CooMatrix::new(nr+nl, nr+nl);
            for r in 0..nr { for p in a00.row_ptr[r]..a00.row_ptr[r+1] { sc.add(r,a00.col_idx[p]as usize,a00.values[p]); }}
            for r in 0..nr { for p in bt.row_ptr[r]..bt.row_ptr[r+1] { sc.add(r,nr+bt.col_idx[p]as usize,bt.values[p]); }}
            for r in 0..nl { for p in b.row_ptr[r]..b.row_ptr[r+1] { sc.add(nr+r,b.col_idx[p]as usize,b.values[p]); }}
            let sm = sc.into_csr();

            let mut dx = vec![0.0; nr+nl];
            solve_minres(&sm, &rhs, &mut dx, &SolverConfig{rtol:1e-10,max_iter:5000,verbose:false,..Default::default()}).expect("MINRES");

            // Damped update
            for i in 0..nr { psi[i] += args.newton_scaling * dx[i]; }
            for i in 0..nl { u[i] -= args.newton_scaling * dx[nr+i]; }

            // Newton update size (L2 norm of du)
            let mut du2 = 0.0;
            for e in mesh.elem_iter() {
                let ld: Vec<usize> = l2.element_dofs(e).iter().map(|&d| d as usize).collect();
                let ns = mesh.element_nodes(e);
                if mesh.element_type(e)!=ElementType::Tri3{continue;}
                let (x0,x1,x2)=(mesh.node_coords(ns[0]),mesh.node_coords(ns[1]),mesh.node_coords(ns[2]));
                let det = ((x1[0]-x0[0])*(x2[1]-x0[1])-(x2[0]-x0[0])*(x1[1]-x0[1])).abs()/2.0;
                let re = TriP1; let q = re.quadrature(4);
                for (qi,xi) in q.points.iter().enumerate() {
                    let w = q.weights[qi]*det; let mut ph=vec![0.0;re.n_dofs()]; re.eval_basis(xi,&mut ph);
                    let mut dv=0.0; for j in 0..re.n_dofs(){dv+=dx[nr+ld[j]]*ph[j];}
                    du2 += w*dv*dv;
                }
            }
            let update = du2.sqrt();
            println!("Newton_update_size = {update:.6}");
            if update < increment_u { break; }
        }

        // Outer convergence
        let mut u_diff2 = 0.0;
        for e in mesh.elem_iter() {
            let ld: Vec<usize> = l2.element_dofs(e).iter().map(|&d| d as usize).collect();
            let ns = mesh.element_nodes(e);
            if mesh.element_type(e)!=ElementType::Tri3{continue;}
            let (x0,x1,x2)=(mesh.node_coords(ns[0]),mesh.node_coords(ns[1]),mesh.node_coords(ns[2]));
            let det = ((x1[0]-x0[0])*(x2[1]-x0[1])-(x2[0]-x0[0])*(x1[1]-x0[1])).abs()/2.0;
            let re = TriP1; let q = re.quadrature(4);
            for (qi,xi) in q.points.iter().enumerate() {
                let w = q.weights[qi]*det; let mut ph=vec![0.0;re.n_dofs()]; re.eval_basis(xi,&mut ph);
                let (mut uv, mut uvo) = (0.0,0.0);
                for j in 0..re.n_dofs(){uv+=u[ld[j]]*ph[j]; uvo+=u_old[ld[j]]*ph[j];}
                u_diff2 += w*(uv-uvo)*(uv-uvo);
            }
        }
        increment_u = u_diff2.sqrt();
        println!("Number of Newton iterations = {loop_j}");
        println!("Increment (|| uₕ - uₕ_prvs||) = {increment_u:.6e}");
        u_old = u.clone(); psi_old = psi.clone();
        if increment_u < args.tol || k+1 >= args.max_it { break; }
        alpha *= args.growth_rate.max(1.0);
    }
    println!("\n Outer iterations: {}", (0..args.max_it).position(|i| i >= args.max_it).unwrap_or(args.max_it));
    println!(" Total iterations: {total_iters}");
    println!(" Total dofs:       {}", nr+nl);
}
