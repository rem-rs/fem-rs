//! MFEM ex27 — Mixed BCs. Periodic enforced via penalty on CooMatrix.

use fem_assembly::{Assembler, standard::DiffusionIntegrator};
use fem_element::ReferenceElement;
use fem_linalg::CooMatrix;
use fem_mesh::{Mesh, topology::MeshTopology, ElementType};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

static mut HOLE_RADIUS: f64 = 0.2;

fn main() {
    let a = parse_args();
    unsafe { HOLE_RADIUS = a.hole_radius.max(0.01).min(0.49); }

    let mesh = gen_mesh(a.ref_levels);
    let space = H1Space::new(mesh.clone(), a.order as u8);
    let n = space.n_dofs();
    println!("\nNumber of finite element unknowns: {}", n);

    let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: a.mat_val }], 3);
    let mut coo = CooMatrix::from(&stiff);

    if a.h1 {
        let robin = assemble_mass(&space, &mesh, a.rbc_a_val, &[2], 3);
        coo.add(&robin);
    }

    let mut rhs = vec![0.0; n];
    let nbc = assemble_linear(&space, &mesh, |_, _| a.mat_val * a.nbc_val, &[1], 3);
    let rbc = assemble_linear(&space, &mesh, |_, _| a.mat_val * a.rbc_b_val, &[2], 3);
    for i in 0..n { rhs[i] += nbc[i] + rbc[i]; }

    // Periodic BC: penalty method (u_left = u_right)
    let ld = boundary_dofs(&mesh, space.dof_manager(), &[5]);
    let rd = boundary_dofs(&mesh, space.dof_manager(), &[6]);
    let penalty = 1e10;
    for (&l, &r) in ld.iter().zip(rd.iter()) {
        coo.add(l as usize, l as usize, penalty);
        coo.add(r as usize, r as usize, penalty);
        coo.add(l as usize, r as usize, -penalty);
        coo.add(r as usize, l as usize, -penalty);
    }

    // Dirichlet BC on tag 3
    let ess = boundary_dofs(&mesh, space.dof_manager(), &[3]);
    let mut stiff = coo.into_csr();
    for &d in &ess {
        let du = d as usize;
        for j in stiff.row_ptr[du]..stiff.row_ptr[du+1] {
            if stiff.col_idx[j] == du { stiff.values[j] = 1.0; }
            else { stiff.values[j] = 0.0; }
        }
        for row in 0..n {
            for j in stiff.row_ptr[row]..stiff.row_ptr[row+1] {
                if stiff.col_idx[j] == du && row != du {
                    stiff.values[j] = 0.0;
                }
            }
        }
        rhs[du] = a.dbc_val;
    }

    let mut x = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
    fem_solver::solve_cg(&stiff, &rhs, &mut x, &cfg).expect("CG");
    println!("  Solved.");

    // Copy periodic DOF values
    for (&l, &r) in ld.iter().zip(rd.iter()) {
        let avg = 0.5 * (x[l as usize] + x[r as usize]);
        x[l as usize] = avg;
        x[r as usize] = avg;
    }

    let _ = fem_io::mfem::write_gf_file("sol.gf", 2, &x, "H1", a.order as u8, 1);
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
    let mut m = Mesh::<2>::uniform(c,e,vec![1;16],ElementType::Quad4,fc,ft,ElementType::Line2);
    for _ in 0..rl { m = fem_mesh::refine_uniform(&m); }
    m.set_curvature(3);
    m.transform(hole_transform);
    m
}

fn hole_transform(p:[f64;2])->[f64;2] {
    let tol=1e-4;let(u,v)=(p[0],p[1]);
    if v>0.5-tol || v< -0.5+tol || u>1.0-tol || u< -1.0+tol || u.abs()<tol {return p}
    let qt=|du:f64,fv:f64|{let a=unsafe{HOLE_RADIUS/std::f64::consts::SQRT_2};
        let d=4.0*a*(std::f64::consts::SQRT_2-2.0*a)*(1.0-2.0*fv);
        let v0=(1.0+std::f64::consts::SQRT_2)*(std::f64::consts::SQRT_2*a-2.0*fv)*((4.0-3.0*std::f64::consts::SQRT_2)*a+(8.0*(std::f64::consts::SQRT_2-1.0)*a-2.0)*fv)/d;
        let r=2.0*((std::f64::consts::SQRT_2-1.0)*a*a*(1.0-4.0*fv)+2.0*(1.0+std::f64::consts::SQRT_2*(1.0+2.0*(2.0*a-std::f64::consts::SQRT_2-1.0)*a))*fv*fv)/d;
        let t=if fv.abs()>1e-15{(fv/r).asin()*du/fv}else{0.0};
        (r*t.sin(),r*t.cos()-v0)};
    if u>0.0{
        if v>(u-0.5).abs(){let(x,y)=qt(u-0.5,v);return[x+0.5,y]}
        if v< -(u-0.5).abs(){let(x,y)=qt(u-0.5,-v);return[x+0.5,-y]}
        if u-0.5>v.abs(){let(x,y)=qt(v,u-0.5);return[x+0.5,y]}
        if u-0.5< -v.abs(){let(x,y)=qt(v,0.5-u);return[-x+0.5,y]}
    }else{
        if v>(u+0.5).abs(){let(x,y)=qt(u+0.5,v);return[x-0.5,y]}
        if v< -(u+0.5).abs(){let(x,y)=qt(u+0.5,-v);return[x-0.5,-y]}
        if u+0.5>v.abs(){let(x,y)=qt(v,u+0.5);return[x-0.5,y]}
        if u+0.5< -v.abs(){let(x,y)=qt(v,-0.5-u);return[-x-0.5,y]}
    }
    p
}

fn assemble_mass(s:&H1Space<Mesh<2>>,m:&Mesh<2>,alpha:f64,tags:&[i32],qo:u8)->CooMatrix<f64>{
    let n=s.n_dofs();let mut coo=CooMatrix::new(n,n);
    for f in 0..m.n_boundary_faces() as u32{
        if!tags.contains(&m.face_tag(f)){continue}
        let ns=m.face_nodes(f);let re=fem_element::lagrange::SegP1;let q=re.quadrature(qo);
        let dofs:Vec<_>=ns.iter().map(|&n|n as usize).collect();let nd=dofs.len();
        let mut me=vec![0.0;nd*nd];let mut phi=vec![0.0;nd];
        for(qi,xi) in q.points.iter().enumerate(){
            let p0=m.node_coords(ns[0]);let p1=m.node_coords(ns[1]);
            let(dx,dy)=(p1[0]-p0[0],p1[1]-p0[1]);let w=q.weights[qi]*(dx*dx+dy*dy).sqrt();
            re.eval_basis(xi,&mut phi);
            for i in 0..nd{for j in 0..nd{me[i*nd+j]+=w*alpha*phi[i]*phi[j]}}
        }
        for i in 0..nd{for j in 0..nd{let v=me[i*nd+j];if v!=0.0{coo.add(dofs[i],dofs[j],v)}}}
    }
    coo
}

fn assemble_linear<F:Fn(&[f64],&[f64])->f64>(s:&H1Space<Mesh<2>>,m:&Mesh<2>,f:F,tags:&[i32],qo:u8)->Vec<f64>{
    let n=s.n_dofs();let mut rhs=vec![0.0;n];
    for fi in 0..m.n_boundary_faces() as u32{
        if!tags.contains(&m.face_tag(fi)){continue}
        let ns=m.face_nodes(fi);let re=fem_element::lagrange::SegP1;let q=re.quadrature(qo);
        let dofs:Vec<_>=ns.iter().map(|&n|n as usize).collect();let nd=dofs.len();let mut phi=vec![0.0;nd];
        for(qi,xi) in q.points.iter().enumerate(){
            let p0=m.node_coords(ns[0]);let p1=m.node_coords(ns[1]);
            let(dx,dy)=(p1[0]-p0[0],p1[1]-p0[1]);let w=q.weights[qi]*(dx*dx+dy*dy).sqrt();
            let normal=[-dy,dx];let xp=[(1.0-xi[0])*p0[0]+xi[0]*p1[0],(1.0-xi[0])*p0[1]+xi[0]*p1[1]];
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
