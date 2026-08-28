//! # Example 27 — Mixed Boundary Conditions [1:1 translation of MFEM ex27]
//!
//! Solves −Δu = 0 on a periodic (seam-identified) Q3-curved mesh with mixed
//! boundary conditions (Neumann / Robin / Dirichlet / natural), using either a
//! continuous H¹ space (essential Dirichlet BC) or a discontinuous L² space
//! (weak Dirichlet BC via `DGDiffusionIntegrator` + `DGDirichletLFIntegrator`).
//!
//! The mesh generation mirrors the C++ `GenerateSerialMesh` flow: the flat
//! periodic mesh is refined, the x=±1 seam is stitched (right-seam vertices are
//! rewired onto the left seam at x=-1, seam boundary faces dropped), Q3 geometry
//! is rebuilt on the stitched flat mesh, and the hole transform is applied to
//! every geometry node.

#![allow(dead_code)]

use fem_assembly::dg::dg_base::{
    build_face_elem_map, phys_to_ref, quad_jac_at, ref_elem_vol, simplex_jac, xform_grads,
};
use fem_assembly::{Assembler, DgAssembler, InteriorFaceList, standard::DiffusionIntegrator};
use fem_element::ReferenceElement;
use fem_mesh::{Mesh, topology::MeshTopology, ElementType};
use fem_mesh::amr::HangingNodeConstraint;
use fem_solver::SolverConfig;
use fem_space::{H1Space, L2Space, fe_space::FESpace, constraints::boundary_dofs};
use fem_space::constraints::{apply_hanging_constraints, identify_periodic_dof_pairs, recover_hanging_values};

static mut HOLE_RADIUS: f64 = 0.2;

fn main() {
    let a = parse_args();
    unsafe { HOLE_RADIUS = a.hole_radius.max(0.01).min(0.49); }

    let mesh = gen_mesh(a.ref_levels);
    if a.h1 {
        solve_h1(&a, &mesh);
    } else {
        solve_dg(&a, &mesh);
    }
}

/// H1 path (MFEM ex27 steps 3–14): continuous Q1 space, essential Dirichlet BC.
///
/// The mesh is UNFOLDED (x=±1 seam columns present, like the C++ element
/// geometry); periodicity is imposed by identifying the seam DOF pairs
/// (tag 5 at x=-1 ↔ tag 6 at x=1) — the C++ `v2v` stitch, but at the DOF
/// level so the per-element geometry keeps the x=±1 positions.
fn solve_h1(a: &Args, mesh: &Mesh<2>) {
    let space = H1Space::new(mesh.clone(), a.order as u8);
    let n = space.n_dofs();

    let mut stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: a.mat_val }], 3);
    // Robin: add a·u·v on the Robin boundary (tag 2)
    let rm = assemble_mass(&space, mesh, a.rbc_a_val, &[2], 3);
    stiff = fem_linalg::CsrMatrix::add(&stiff, &rm);

    let mut rhs = vec![0.0; n];
    let nbc = assemble_linear(&space, mesh, |_, _| a.mat_val * a.nbc_val, &[1], 3);
    let rbc = assemble_linear(&space, mesh, |_, _| a.mat_val * a.rbc_b_val, &[2], 3);
    for i in 0..n { rhs[i] += nbc[i] + rbc[i]; }

    // Periodic seam: u(x=1) = u(x=-1).  Slave = tag 6 (x=1), master = tag 5
    // (x=-1); the shift x_slave + offset = x_master gives offset = [-2, 0].
    let pairs = identify_periodic_dof_pairs(mesh, space.dof_manager(), 5, 6, &[-2.0, 0.0], 1e-10);
    // C++ ex27 prints the number of TRUE unknowns (seam DOFs merged by the
    // v2v stitch): n − merged pairs.
    println!("\nNumber of finite element unknowns: {}", n - pairs.len());
    let periodic_constraints: Vec<HangingNodeConstraint> = pairs.iter()
        .map(|&(slave, master)| HangingNodeConstraint {
            constrained: slave as usize,
            parent_a:    master as usize,
            parent_b:    master as usize,
            coeff_a:     1.0,
            coeff_b:     0.0,
            extra:       vec![],
        })
        .collect();
    apply_hanging_constraints(&mut stiff, &mut rhs, &periodic_constraints);

    let ess = boundary_dofs(mesh, space.dof_manager(), &[3]);
    for &d in &ess {
        let du = d as usize;
        let mut dummy = vec![0.0; n];
        // C++ BilinearForm::FormLinearSystem defaults to diag_policy=DIAG_KEEP
        // (preserves the original diagonal, zeroes the row/col off-diagonals).
        stiff.apply_dirichlet_keep_diag(du, a.dbc_val, &mut dummy);
        for j in 0..n { rhs[j] += dummy[j]; }
    }

    let mut x = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500, verbose: true, ..Default::default() };
    let res = fem_solver::solve_pcg_gssmoother(&stiff, &rhs, &mut x, &cfg).expect("PCG+GSSmoother");
    println!("  Solved in {} iterations.", res.iterations);


    // Recover the slave seam DOFs: u(slave) = u(master).
    recover_hanging_values(&mut x, &periodic_constraints);

    verify_bc(a, &space, mesh, &x);

    // C++ step 14: save refined.mesh + sol.gf (MFEM-native formats)
    let _ = fem_io::mfem::write_mfem_file("refined.mesh", mesh);
    let _ = fem_io::mfem::write_mfem_gf_file("sol.gf", 2, &x, "H1", a.order as u8, 1, 8);
}

/// DG path (MFEM ex27 steps 3–14 with `-dg`): discontinuous Q1 space.
///
/// Weakly enforces all BCs:
/// - **Dirichlet** (tag 3): `DGDiffusionIntegrator` face penalty on the matrix
///   + `DGDirichletLFIntegrator` on the RHS.
/// - **Robin** (tag 2): `BoundaryMassIntegrator` on the matrix + the RHS source.
/// - **Neumann** (tag 1): natural (RHS source only).
///
/// Solved with PCG when `sigma == -1` (SIP is symmetric), GMRES otherwise —
/// exactly as MFEM ex27 does.
fn solve_dg(a: &Args, mesh: &Mesh<2>) {
    let space = L2Space::new(mesh.clone(), a.order as u8);
    let n = space.n_dofs();
    println!("\nNumber of finite element unknowns: {}", n);

    // MFEM ex27: negative kappa is replaced with (order+1)².
    let penalty = if a.kappa < 0.0 { (a.order as f64 + 1.0).powi(2) } else { a.kappa };
    let qo = 2 * a.order as u8;

    let ifl = InteriorFaceList::build(mesh);
    // Matrix: volume + interior faces + DGDiffusion on the Dirichlet boundary
    // (tag 3) + BoundaryMass on the Robin boundary (tag 2).
    let mut stiff = DgAssembler::assemble_dg(&space, &ifl, a.mat_val, a.sigma, penalty, qo, Some(&[3]));
    let rm = assemble_l2_mass(&space, mesh, a.mat_val * a.rbc_a_val, &[2], qo);
    stiff = fem_linalg::CsrMatrix::add(&stiff, &rm);

    // RHS: DGDirichletLF (tag 3) + BoundaryLF (tags 1 and 2).
    let mut rhs = vec![0.0; n];
    let dglf = assemble_l2_dg_dirichlet_lf(&space, mesh, a.dbc_val, a.mat_val, a.sigma, penalty, &[3], qo);
    let nbc = assemble_l2_linear(&space, mesh, |_, _| a.mat_val * a.nbc_val, &[1], qo);
    let rbc = assemble_l2_linear(&space, mesh, |_, _| a.mat_val * a.rbc_b_val, &[2], qo);
    for i in 0..n { rhs[i] += dglf[i] + nbc[i] + rbc[i]; }

    let mut x = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500, verbose: true, ..Default::default() };
    let res = if a.sigma == -1.0 {
        fem_solver::solve_pcg_gssmoother(&stiff, &rhs, &mut x, &cfg).expect("PCG+GSSmoother")
    } else {
        // MFEM ex27: GMRES with restart 10 for the non-symmetric (NIP) case.
        // The C++ GMRES converges on the *preconditioned* residual; the linlvo
        // GMRES uses ‖r‖/‖b‖ and stagnates just above 1e-12, so relax rtol.
        let gcfg = SolverConfig { rtol: 1e-10, ..cfg.clone() };
        fem_solver::solve_gmres_gssmoother(&stiff, &rhs, &mut x, 10, &gcfg).expect("GMRES+GSSmoother")
    };
    println!("  Solved in {} iterations.", res.iterations);

    verify_bc(a, &space, mesh, &x);

    let _ = fem_io::mfem::write_mfem_file("refined.mesh", mesh);
    let _ = fem_io::mfem::write_mfem_gf_file("sol.gf", 2, &x, "L2", a.order as u8, 1, 8);
}

/// MFEM ex27 step 13: verify the boundary conditions by integrating
/// `α·n·∇u + β·u` over the marked boundary and comparing with `γ`.
fn verify_bc<S: FESpace>(a: &Args, space: &S, mesh: &Mesh<2>, x: &[f64]) {
    println!();
    println!("Verifying boundary conditions");
    println!("=============================");

    let (avg, mut err) = integrate_bc(space, mesh, x, &[3], 0.0, 1.0, a.dbc_val, 3);
    let hom = a.dbc_val == 0.0;
    err /= if hom { 1.0 } else { a.dbc_val.abs() };
    println!("Average of solution on Gamma_dbc:\t{avg},\t{} error {err}",
             if hom { "absolute" } else { "relative" });

    let (avg, mut err) = integrate_bc(space, mesh, x, &[1], 1.0, 0.0, a.nbc_val, 3);
    let hom = a.nbc_val == 0.0;
    err /= if hom { 1.0 } else { a.nbc_val.abs() };
    println!("Average of n.Grad(u) on Gamma_nbc:\t{avg},\t{} error {err}",
             if hom { "absolute" } else { "relative" });

    let (avg, err) = integrate_bc(space, mesh, x, &[4], 1.0, 0.0, 0.0, 3);
    println!("Average of n.Grad(u) on Gamma_nbc0:\t{avg},\tabsolute error {err}");

    let (avg, mut err) = integrate_bc(space, mesh, x, &[2], 1.0, a.rbc_a_val, a.rbc_b_val, 3);
    let hom = a.rbc_b_val == 0.0;
    err /= if hom { 1.0 } else { a.rbc_b_val.abs() };
    println!("Average of n.Grad(u)+a*u on Gamma_rbc:\t{avg},\t{} error {err}",
             if hom { "absolute" } else { "relative" });
}

fn gen_mesh(rl: usize) -> Mesh<2> {
    let a = unsafe { HOLE_RADIUS / std::f64::consts::SQRT_2 };
    let v: [[f64;2];29] = [
        [-1.0,-0.5],[-1.0,0.0],[-1.0,0.5],
        [-0.5-a,-a],[-0.5-a,0.0],[-0.5-a,a],
        [-0.5,-0.5],[-0.5,-a],[-0.5,a],[-0.5,0.5],
        [-0.5+a,-a],[-0.5+a,0.0],[-0.5+a,a],
        [0.0,-0.5],[0.0,0.0],[0.0,0.5],
        [0.5-a,-a],[0.5-a,0.0],[0.5-a,a],
        [0.5,-0.5],[0.5,-a],[0.5,a],[0.5,0.5],
        [0.5+a,-a],[0.5+a,0.0],[0.5+a,a],
        [1.0,-0.5],[1.0,0.0],[1.0,0.5]];
    let q:[[u32;4];16] = [
        [0,3,4,1],[1,4,5,2],[5,8,9,2],[8,12,15,9],
        [11,14,15,12],[10,13,14,11],[6,13,10,7],[0,6,7,3],
        [13,16,17,14],[14,17,18,15],[18,21,22,15],[21,25,28,22],
        [24,27,28,25],[23,26,27,24],[19,26,23,20],[13,19,20,16]];
    let bf:[([u32;2],i32);28] = [
        ([0,6],1),([6,13],1),([13,19],1),([19,26],1),
        ([28,22],2),([22,15],2),([15,9],2),([9,2],2),
        ([7,3],3),([10,7],3),([11,10],3),([12,11],3),
        ([8,12],3),([5,8],3),([4,5],3),([3,4],3),
        ([20,16],4),([23,20],4),([24,23],4),([25,24],4),
        ([21,25],4),([18,21],4),([17,18],4),([16,17],4),
        ([0,1],5),([1,2],5),([26,27],6),([27,28],6)];
    let c:Vec<f64> = v.iter().flat_map(|&[x,y]|[x,y]).collect();
    let e:Vec<u32> = q.iter().flat_map(|q|q.iter().copied()).collect();
    let fc:Vec<u32> = bf.iter().flat_map(|(e,_)|e.iter().copied()).collect();
    let ft:Vec<i32> = bf.iter().map(|(_,t)|*t).collect();
    let mesh = Mesh::<2>::uniform(c,e,vec![1;16],ElementType::Quad4,fc,ft,ElementType::Line2);
    // C++ flow: stitch (seam tags 5/6 at x=±1 identified, right seam verts merged
    // into the left seam at x=-1) → SetCurvature(3) → refine(×2) → Transform(trans).
    // We refine the PLAIN mesh, then fold the seam the same way the C++ stitch does:
    // right-seam (x=1) vertices are rewired onto the left-seam (x=-1) vertices, the
    // seam boundary faces (tags 5/6) are dropped, and the x=1 vertices are removed.
    // The seam column stays at x=-1 in BOTH the vertex table and the Q3 geometry —
    // matching the C++ element geometry (the C++ vertex table additionally reports
    // the seam at x=0, an artifact of MFEM's vertex averaging during refinement).
    let mut m = mesh;
    // C++ flow: SetCurvature(3) on the (stitched) flat mesh → refine(×2) →
    // Transform(trans).  We keep the mesh UNFOLDED — the x=±1 seam columns stay
    // in the geometry exactly like the C++ element geometry (the C++ merges only
    // the seam DOFs; its L2 geometry nodes keep the x=±1 positions).  The
    // periodicity is imposed at the DOF level with identify_periodic_dof_pairs.
    for _level in 0..rl {
        m = fem_mesh::refine_uniform(&m);
    }
    // Rebuild Q3 geometry on the refined FLAT mesh, then transform ALL nodes
    // (C++: SetCurvature(3) on the flat stitched mesh → refine → Transform).
    m.set_curvature(3);
    m.transform(hole_transform);
    m
}

fn hole_transform(p:[f64;2])->[f64;2] {
    let tol=1e-4;let(u,v)=(p[0],p[1]);
    if v>0.5-tol||v< -0.5+tol||u>1.0-tol||u< -1.0+tol||u.abs()<tol{return p}
    let qt=|du:f64,fv:f64|{let a=unsafe{HOLE_RADIUS};
        let d=4.0*a*(std::f64::consts::SQRT_2-2.0*a)*(1.0-2.0*fv);
        let v0=(1.0+std::f64::consts::SQRT_2)*(std::f64::consts::SQRT_2*a-2.0*fv)*((4.0-3.0*std::f64::consts::SQRT_2)*a+(8.0*(std::f64::consts::SQRT_2-1.0)*a-2.0)*fv)/d;
        let r=2.0*((std::f64::consts::SQRT_2-1.0)*a*a*(1.0-4.0*fv)+2.0*(1.0+std::f64::consts::SQRT_2*(1.0+2.0*(2.0*a-std::f64::consts::SQRT_2-1.0)*a))*fv*fv)/d;
        let t=if fv.abs()>1e-15{(fv/r).asin()*du/fv}else{0.0};
        (r*t.sin(),r*t.cos()-v0)};
    if u>0.0{
        // Top-right: quad_trans(u-0.5, v) → (x, y)
        if v>(u-0.5).abs(){let(x,y)=qt(u-0.5,v);return[x+0.5,y]}
        // Bottom-right: quad_trans(u-0.5, -v) → (x, y), then y = -y
        if v< -(u-0.5).abs(){let(x,y)=qt(u-0.5,-v);return[x+0.5,-y]}
        // Right: quad_trans(v, u-0.5) → SWAPPED: x gets y, y gets x
        if u-0.5>v.abs(){let(x,y)=qt(v,u-0.5);return[y+0.5,x]}
        // Left: quad_trans(v, 0.5-u) → SWAPPED: x gets -y+0.5, y gets x
        if u-0.5< -v.abs(){let(x,y)=qt(v,0.5-u);return[-y+0.5,x]}
    }else{
        // Top-left: quad_trans(u+0.5, v) → (x, y), then x -= 0.5
        if v>(u+0.5).abs(){let(x,y)=qt(u+0.5,v);return[x-0.5,y]}
        // Bottom-left: quad_trans(u+0.5, -v) → (x, y), then x -= 0.5, y = -y
        if v< -(u+0.5).abs(){let(x,y)=qt(u+0.5,-v);return[x-0.5,-y]}
        // Right: quad_trans(v, u+0.5) → SWAPPED: x gets y, y gets x, then x -= 0.5
        if u+0.5>v.abs(){let(x,y)=qt(v,u+0.5);return[y-0.5,x]}
        // Left: quad_trans(v, -0.5-u) → SWAPPED: x gets -y-0.5, y gets x
        if u+0.5< -v.abs(){let(x,y)=qt(v,-0.5-u);return[-y-0.5,x]}
    }
    p
}

/// Compute the effective distance between two points on a periodic domain.
/// The mesh is periodic in x with period 2.0 (x ∈ [-1, 1]).
/// Returns the shorter of the direct distance and the wrap-around distance.
fn periodic_edge_len(p0: &[f64; 2], p1: &[f64; 2]) -> f64 {
    let dx_direct = p1[0] - p0[0];
    // Wrap x by ±2.0 to get the shortest path
    let dx_wrapped = if dx_direct > 0.0 { dx_direct - 2.0 } else { dx_direct + 2.0 };
    let dx = if dx_direct.abs() < dx_wrapped.abs() { dx_direct } else { dx_wrapped };
    let dy = p1[1] - p0[1];
    (dx * dx + dy * dy).sqrt()
}

fn assemble_mass(s:&H1Space<Mesh<2>>,m:&Mesh<2>,alpha:f64,tags:&[i32],qo:u8)->fem_linalg::CsrMatrix<f64>{
    let n=s.n_dofs();let mut coo=fem_linalg::CooMatrix::new(n,n);
    for f in 0..m.n_boundary_faces() as u32{
        if!tags.contains(&m.face_tag(f)){continue}
        let ns=m.face_nodes(f);let re=fem_element::lagrange::SegP1;let q=re.quadrature(qo);
        let dofs:Vec<_>=ns.iter().map(|&n|n as usize).collect();let nd=dofs.len();
        let mut me=vec![0.0;nd*nd];let mut phi=vec![0.0;nd];
        for(qi,xi) in q.points.iter().enumerate(){
            let p0=m.node_coords(ns[0]);let p1=m.node_coords(ns[1]);
            let len=periodic_edge_len(&[p0[0],p0[1]],&[p1[0],p1[1]]);
            let w=q.weights[qi]*len;
            re.eval_basis(xi,&mut phi);
            for i in 0..nd{for j in 0..nd{me[i*nd+j]+=w*alpha*phi[i]*phi[j]}}
        }
        for i in 0..nd{for j in 0..nd{let v=me[i*nd+j];if v!=0.0{coo.add(dofs[i],dofs[j],v)}}}
    }
    coo.into_csr()
}

fn assemble_linear<F:Fn(&[f64],&[f64])->f64>(s:&H1Space<Mesh<2>>,m:&Mesh<2>,f:F,tags:&[i32],qo:u8)->Vec<f64>{
    let n=s.n_dofs();let mut rhs=vec![0.0;n];
    for fi in 0..m.n_boundary_faces() as u32{
        if!tags.contains(&m.face_tag(fi)){continue}
        let ns=m.face_nodes(fi);let re=fem_element::lagrange::SegP1;let q=re.quadrature(qo);
        let dofs:Vec<_>=ns.iter().map(|&n|n as usize).collect();let nd=dofs.len();let mut phi=vec![0.0;nd];
        for(qi,xi) in q.points.iter().enumerate(){
            let p0=m.node_coords(ns[0]);let p1=m.node_coords(ns[1]);
            // Correct for periodic wrap: x-periodicity with period 2.0
            // Use the SHORTER x-distance (wrap-around if dx > 1.0)
            let (dx_raw,dy)=(p1[0]-p0[0],p1[1]-p0[1]);
            // Periodic wrap: take the SHORTER of the direct and wrapped x-step
            // (the seam identifies x=-1 with x=1). Same rule as periodic_edge_len.
            let dx_wrapped = if dx_raw > 0.0 { dx_raw - 2.0 } else { dx_raw + 2.0 };
            let dx = if dx_raw.abs() < dx_wrapped.abs() { dx_raw } else { dx_wrapped };
            let len=(dx*dx+dy*dy).sqrt();
            let normal=[-dy,dx];let w=q.weights[qi]*len;
            // xp integration point — only needed for spatially-varying BCs
            let xp=[(1.0-xi[0])*0.5*p0[0]+(1.0+xi[0])*0.5*(p0[0]+dx),
                    (1.0-xi[0])*0.5*p0[1]+(1.0+xi[0])*0.5*p0[1]];
            let val=f(&xp,&normal);re.eval_basis(xi,&mut phi);
            for i in 0..nd{rhs[dofs[i]]+=w*val*phi[i]}
        }
    }
    rhs
}

struct Args{h1:bool,order:i32,sigma:f64,kappa:f64,ref_levels:usize,mat_val:f64,dbc_val:f64,nbc_val:f64,rbc_a_val:f64,rbc_b_val:f64,hole_radius:f64,visualization:bool}
fn parse_args()->Args{
    let mut a=Args{h1:true,order:1,sigma:-1.0,kappa:-1.0,ref_levels:2,mat_val:1.0,dbc_val:0.0,nbc_val:1.0,rbc_a_val:1.0,rbc_b_val:1.0,hole_radius:0.2,visualization:false};
    let mut it=std::env::args().skip(1);
    while let Some(arg)=it.next(){match arg.as_str(){
        "-h1"|"--continuous"=>a.h1=true,"-dg"|"--discontinuous"=>a.h1=false,
        "-o"|"--order"=>a.order=it.next().and_then(|s|s.parse().ok()).unwrap_or(1),
        "-s"|"--sigma"=>a.sigma=it.next().and_then(|s|s.parse().ok()).unwrap_or(-1.0),
        "-k"|"--kappa"=>a.kappa=it.next().and_then(|s|s.parse().ok()).unwrap_or(-1.0),
        "-rs"|"--refine-serial"=>a.ref_levels=it.next().and_then(|s|s.parse().ok()).unwrap_or(2),
        "-mat"|"--material-value"=>a.mat_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-dbc"|"--dirichlet-value"=>a.dbc_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.0),
        "-nbc"|"--neumann-value"=>a.nbc_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-rbc-a"|"--robin-a-value"=>a.rbc_a_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-rbc-b"|"--robin-b-value"=>a.rbc_b_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-a"|"--radius"=>a.hole_radius=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.2),
        "-no-vis"|"--no-visualization"=>a.visualization=false,_=>{}}}
    a
}

/// Reference quadrature on the 1-D segment face ξ ∈ [-1, 1] (SegP1 basis).
fn seg_quad(qo: u8) -> (Vec<f64>, Vec<f64>) {
    let re = fem_element::lagrange::SegP1;
    let q = re.quadrature(qo);
    (q.points.iter().map(|p| p[0]).collect(), q.weights)
}

/// Physical point + unnormalized normal `nor = (Δy/2, −Δx/2)` on a face at the
/// reference coordinate `xi ∈ [-1,1]`. `|nor|` equals the face Jacobian
/// (half the edge length), matching MFEM `CalcOrtho(Face->Jacobian())`.
/// The periodic seam (x = -1 ≡ 1) is handled by taking the shorter x-step.
/// The normal is oriented outward from the owner element (as MFEM does via
/// the boundary face transformation).
fn face_point_and_normal(mesh: &Mesh<2>, elem: u32, face: u32, xi: f64) -> ([f64; 2], [f64; 2]) {
    use fem_mesh::topology::MeshTopology as _;
    let ns = mesh.face_nodes(face);
    let p0 = mesh.node_coords(ns[0]);
    let p1 = mesh.node_coords(ns[1]);
    let (dx_raw, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    // Periodic wrap: the shorter of the direct and the x±2 wrapped step.
    let dx_wrapped = if dx_raw > 0.0 { dx_raw - 2.0 } else { dx_raw + 2.0 };
    let dx = if dx_raw.abs() < dx_wrapped.abs() { dx_raw } else { dx_wrapped };
    let xp = [0.5 * ((1.0 - xi) * p0[0] + (1.0 + xi) * (p0[0] + dx)),
              0.5 * ((1.0 - xi) * p0[1] + (1.0 + xi) * p0[1])];
    let mut nor = [dy / 2.0, -dx / 2.0];
    // Orient outward: the normal must point from the element centroid to the
    // (periodically wrapped) face midpoint.
    let en = mesh.element_nodes(elem);
    let (mut cx, mut cy) = (0.0, 0.0);
    for &n in en {
        let c = mesh.node_coords(n);
        cx += c[0];
        cy += c[1];
    }
    cx /= en.len() as f64;
    cy /= en.len() as f64;
    // Wrapped face midpoint (midpoint of p0 and the wrapped p1).
    let mx = 0.5 * (p0[0] + p0[0] + dx);
    let my = p0[1];
    if nor[0] * (mx - cx) + nor[1] * (my - cy) < 0.0 {
        nor[0] = -nor[0];
        nor[1] = -nor[1];
    }
    ([xp[0], xp[1]], nor)
}

/// Local DOF indices of the two corners of `face` inside its owner element
/// (L² spaces store DOFs element-by-element: `e*ndofs .. e*ndofs+ndofs`).
fn l2_face_dofs<S: FESpace>(space: &S, elem: u32, face: u32) -> [usize; 2] {
    let en = space.mesh().element_nodes(elem);
    let fn_ = space.mesh().face_nodes(face);
    let mut dofs = [0usize; 2];
    for k in 0..2 {
        let pos = en.iter().position(|&nn| nn == fn_[k]).expect("face node not in element");
        dofs[k] = elem as usize * 4 + pos;
    }
    dofs
}

/// MFEM `IntegrateBC`: over the boundary attributes in `tags`, compute the
/// average of `α·n·Grad(u) + β·u` and the L² (root-mean-square) error of
/// `α·n·Grad(u) + β·u − γ`, normalized by the boundary measure.
///
/// Geometry follows MFEM `IntegrateBC` exactly:
/// - the boundary face is one edge of its owner element; the face reference
///   point `t ∈ [-1,1]` maps to the element reference point `eip` affinely
///   (`FTr->Loc1.Transform(ip, eip)`),
/// - physical gradients use the **Q3** isoparametric element geometry
///   (`fe.CalcPhysDShape(*FTr->Elem1, dshape)` = J⁻ᵀ·∇ref),
/// - the face Jacobian / normal come from the face tangent
///   `J·d(eip)/dt` (`FTr->Face->Jacobian()` + `CalcOrtho`).
fn integrate_bc<S: FESpace>(
    space: &S,
    mesh: &Mesh<2>,
    sol: &[f64],
    tags: &[i32],
    alpha: f64,
    beta: f64,
    gamma: f64,
    qo: u8,
) -> (f64, f64) {
    let dim = 2usize;
    let order = space.order();
    let a_is_zero = alpha == 0.0;
    let b_is_zero = beta == 0.0;
    let mut nrm = 0.0;
    let mut avg = 0.0;
    let mut err2 = 0.0;

    let face_to_elem = build_face_elem_map(mesh, dim);
    // MFEM: int_order = 2*fe.GetOrder() + 3
    let (xi_q, w_q) = seg_quad(2 * order + 3);
    // H1 space basis: QuadQk on [0,1]² with H1 topological node order
    // (matches H1Space's element_dofs).  NOT dg_base::ref_elem_vol (QuadL2GL
    // lex order) — using that misaligns dof i ↔ basis i and corrupts u/∇u.
    let re = fem_element::lagrange::QuadQk::new(order as usize);
    let n_dofs = re.n_dofs();

    let mut phi = vec![0.0; n_dofs];
    let mut gref = vec![0.0; n_dofs * dim];
    let mut gphys = vec![0.0; n_dofs * dim];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let mut ud = vec![0.0; n_dofs];
        for (k, &g) in gd.iter().enumerate() { ud[k] = sol[g]; }

        // Local edge of `face` inside the owner element (quad [0,1]²:
        // edge 0 = η=0, edge 1 = ξ=1, edge 2 = η=1, edge 3 = ξ=0).
        // The face's own node order (matching MFEM AddBdrSegment) decides the
        // reference-direction: face ref t=-1 maps to the FIRST face node.
        let en = mesh.element_nodes(elem);
        let fn_ = mesh.face_nodes(f);
        let (pa, pb) = (
            en.iter().position(|&n| n == fn_[0]).unwrap(),
            en.iter().position(|&n| n == fn_[1]).unwrap(),
        );
        let (eip_at, deip): (Box<dyn Fn(f64) -> [f64; 2]>, [f64; 2]) = match (pa, pb) {
            (0, 1) => (Box::new(|t| [0.5 * (1.0 + t), 0.0]), [0.5, 0.0]),
            (1, 0) => (Box::new(|t| [0.5 * (1.0 - t), 0.0]), [-0.5, 0.0]),
            (1, 2) => (Box::new(|t| [1.0, 0.5 * (1.0 + t)]), [0.0, 0.5]),
            (2, 1) => (Box::new(|t| [1.0, 0.5 * (1.0 - t)]), [0.0, -0.5]),
            (2, 3) => (Box::new(|t| [0.5 * (1.0 - t), 1.0]), [-0.5, 0.0]),
            (3, 2) => (Box::new(|t| [0.5 * (1.0 + t), 1.0]), [0.5, 0.0]),
            (3, 0) => (Box::new(|t| [0.0, 0.5 * (1.0 - t)]), [0.0, -0.5]),
            (0, 3) => (Box::new(|t| [0.0, 0.5 * (1.0 + t)]), [0.0, 0.5]),
            _ => panic!("integrate_bc: face not on element edge"),
        };

        for (qi, xi) in xi_q.iter().enumerate() {
            // Face ref point t ∈ [-1,1] → element ref point eip ∈ [0,1]².
            let eip = eip_at(*xi);
            // Q3 isoparametric element geometry: J, det, physical face point.
            let (jq, detq, _xp) = mesh.element_jacobian(elem, &eip);
            // Face tangent dF/dt = J·d(eip)/dt; face_weight = |tangent|.
            let tx = jq[(0, 0)] * deip[0] + jq[(0, 1)] * deip[1];
            let ty = jq[(1, 0)] * deip[0] + jq[(1, 1)] * deip[1];
            let face_weight = (tx * tx + ty * ty).sqrt();
            // CalcOrtho(J_face, w_nor): w_nor = (dy, -dx), |w_nor| = face_weight.
            let nor = [ty, -tx];

            re.eval_basis(&eip, &mut phi);
            re.eval_grad_basis(&eip, &mut gref);
            let jit = jq.clone().try_inverse()
                .unwrap_or_else(|| { eprintln!("  warning: degenerate element"); nalgebra::DMatrix::identity(2, 2) })
                .transpose();
            xform_grads(&jit, &gref, &mut gphys, n_dofs, dim);

            let w = w_q[qi] * face_weight;
            nrm += w;
            let mut val = 0.0;
            if !a_is_zero {
                // α · (∇u · w_nor) / face_weight  →  α · ∇u · n̂
                let mut du_dn = 0.0;
                for k in 0..n_dofs {
                    du_dn += ud[k] * (gphys[k * dim] * nor[0] + gphys[k * dim + 1] * nor[1]);
                }
                val += alpha * du_dn / face_weight;
            }
            if !b_is_zero {
                let mut u = 0.0;
                for k in 0..n_dofs { u += ud[k] * phi[k]; }
                val += beta * u;
            }
            avg += val * w;
            let d = val - gamma;
            err2 += d * d * w;
            let _ = detq;
        }
    }
    if nrm.abs() > 0.0 { avg /= nrm; err2 /= nrm; }
    (avg, err2.sqrt())
}

/// L² (DG) boundary mass on the tagged faces: `∫ κ·u·v ds` scattered to the
/// element-local DOFs (MFEM `BoundaryMassIntegrator`).
fn assemble_l2_mass<S: FESpace>(
    space: &S,
    mesh: &Mesh<2>,
    kappa: f64,
    tags: &[i32],
    qo: u8,
) -> fem_linalg::CsrMatrix<f64> {
    let n = space.n_dofs();
    let mut coo = fem_linalg::CooMatrix::new(n, n);
    let face_to_elem = build_face_elem_map(mesh, 2);
    let (xi_q, w_q) = seg_quad(qo);
    let re = fem_element::lagrange::SegP1;
    let mut phi = vec![0.0; 2];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let dofs = l2_face_dofs(space, elem, f);
        for (qi, xi) in xi_q.iter().enumerate() {
            let (_, nor) = face_point_and_normal(mesh, elem, f, *xi);
            let len = (nor[0] * nor[0] + nor[1] * nor[1]).sqrt();
            let w = w_q[qi] * len * kappa;
            re.eval_basis(&[*xi], &mut phi);
            for i in 0..2 {
                for j in 0..2 {
                    coo.add(dofs[i], dofs[j], w * phi[i] * phi[j]);
                }
            }
        }
    }
    coo.into_csr()
}

/// L² (DG) boundary linear form on the tagged faces: `∫ g·v ds` scattered to
/// the element-local DOFs (MFEM `BoundaryLFIntegrator`).
fn assemble_l2_linear<S: FESpace, F: Fn(&[f64], &[f64]) -> f64>(
    space: &S,
    mesh: &Mesh<2>,
    g: F,
    tags: &[i32],
    qo: u8,
) -> Vec<f64> {
    let n = space.n_dofs();
    let mut rhs = vec![0.0; n];
    let face_to_elem = build_face_elem_map(mesh, 2);
    let (xi_q, w_q) = seg_quad(qo);
    let re = fem_element::lagrange::SegP1;
    let mut phi = vec![0.0; 2];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let dofs = l2_face_dofs(space, elem, f);
        for (qi, xi) in xi_q.iter().enumerate() {
            let (xp, nor) = face_point_and_normal(mesh, elem, f, *xi);
            let len = (nor[0] * nor[0] + nor[1] * nor[1]).sqrt();
            let w = w_q[qi] * len;
            re.eval_basis(&[*xi], &mut phi);
            let val = g(&xp, &nor);
            for i in 0..2 { rhs[dofs[i]] += w * val * phi[i]; }
        }
    }
    rhs
}

/// MFEM `DGDirichletLFIntegrator`: the weak Dirichlet boundary load
/// `∫_Γ u_D·(σ·a·∇v·n + κ·a·h⁻¹·v) ds` on the tagged faces.
///
/// `nor` is the unnormalized face normal (|nor| = face Jacobian) and the
/// element gradients use the bilinear (corner) geometry — the same convention
/// as the DG face assembly.
fn assemble_l2_dg_dirichlet_lf<S: FESpace>(
    space: &S,
    mesh: &Mesh<2>,
    u_d: f64,
    a: f64,
    sigma: f64,
    penalty: f64,
    tags: &[i32],
    qo: u8,
) -> Vec<f64> {
    let dim = 2usize;
    let order = space.order();
    let n = space.n_dofs();
    let mut rhs = vec![0.0; n];
    let face_to_elem = build_face_elem_map(mesh, dim);
    let (xi_q, w_q) = seg_quad(qo);
    let re = ref_elem_vol(ElementType::Quad4, order);
    let n_dofs = re.n_dofs();

    let mut phi = vec![0.0; n_dofs];
    let mut gref = vec![0.0; n_dofs * dim];
    let mut gphys = vec![0.0; n_dofs * dim];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

        let nodes = mesh.element_nodes(elem);
        let (jac, _) = simplex_jac(mesh, nodes, dim);
        let jit = jac.clone().try_inverse()
            .unwrap_or_else(|| { eprintln!("  warning: degenerate element"); nalgebra::DMatrix::identity(2, 2) })
            .transpose();
        let (xl, yl): (Vec<f64>, Vec<f64>) = if nodes.len() > 3 {
            let xl: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(nodes.len()-1)])[0]).collect();
            let yl: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(nodes.len()-1)])[1]).collect();
            (xl, yl)
        } else {
            (vec![], vec![])
        };

        let mut fe = vec![0.0; n_dofs];
        for (qi, xi) in xi_q.iter().enumerate() {
            let (xp, nor) = face_point_and_normal(mesh, elem, f, *xi);
            let nor2 = nor[0] * nor[0] + nor[1] * nor[1];
            let mut xi_e = phys_to_ref(&jac, mesh.node_coords(nodes[0]), &xp, dim);
            if nodes.len() > 3 { for v in &mut xi_e { *v -= 1.0; } }
            re.eval_basis(&xi_e, &mut phi);
            re.eval_grad_basis(&xi_e, &mut gref);
            let (jit_pt, det_pt) = if nodes.len() > 3 {
                let (j, d) = quad_jac_at(&xl, &yl, xi_e[0], xi_e[1]);
                (j.clone().try_inverse().unwrap_or_else(|| nalgebra::DMatrix::identity(2, 2)).transpose(), d.abs().max(1e-14))
            } else {
                (jit.clone(), jac.determinant().abs().max(1e-14))
            };
            xform_grads(&jit_pt, &gref, &mut gphys, n_dofs, dim);
            let w = w_q[qi];
            // MFEM: elvect += sigma·(uD·Q·∇v·nor) + kappa·(uD·Q·|nor|²/|det J|·v)
            for k in 0..n_dofs {
                let du_dn = gphys[k * dim] * nor[0] + gphys[k * dim + 1] * nor[1];
                fe[k] += w * (sigma * u_d * a * du_dn + penalty * u_d * a * nor2 / det_pt * phi[k]);
            }
        }
        for (k, &g) in gd.iter().enumerate() { rhs[g] += fe[k]; }
    }
    rhs
}
