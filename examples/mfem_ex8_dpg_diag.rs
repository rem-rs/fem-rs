//! Diagnostic: full 2×2 DPG block system on tiny mesh.
#![allow(dead_code, unused_imports, unused_variables, unused_mut, non_snake_case)]

use std::collections::HashMap;
use fem_assembly::{
    Assembler, MixedAssembler, MixedBilinearIntegrator,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    integrator::QpData,
};
use fem_element::{
    ReferenceElement,
    lagrange::{TriP1, TriP2, TriP3},
    quadrature::{seg_rule_arbitrary, tri_rule},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, Mesh};
use fem_space::{
    H1Space, L2Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};
use fem_solver::SolverConfig;

struct MixedDiffusion;
impl MixedBilinearIntegrator for MixedDiffusion {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m: &mut [f64]) {
        let nr = qp_row.n_dofs; let nc = qp_col.n_dofs; let d = qp_col.dim; let w = qp_col.weight;
        for k in 0..d { for i in 0..nr { let gik = qp_row.grad_phys[i*d+k];
            for j in 0..nc { m[i*nc+j] += w * gik * qp_col.grad_phys[j*d+k]; }
        }}
    }
}

fn n_dofs_l2(o: u8) -> usize { match o {1=>3,2=>6,3=>10,_=>panic!()}}
fn tri_ref(o: u8) -> Box<dyn ReferenceElement> { match o {
    1=>Box::new(TriP1),2=>Box::new(TriP2),3=>Box::new(TriP3),_=>panic!()
}}

struct SinvData { blocks: Vec<Vec<f64>>, nt: usize, dofs: Vec<Vec<usize>> }
fn build_sinv<M: MeshTopology>(sp: &impl FESpace<Mesh=M>, qo: u8) -> SinvData {
    let mesh = sp.mesh(); let ne = mesh.n_elements(); let o = sp.order();
    let nt = n_dofs_l2(o); let tri = tri_ref(o); let qr = tri_rule(qo);
    let mut phi = vec![0.0; nt]; let mut dphi = vec![0.0; nt*2];
    let mut blk = Vec::new(); let mut dfs = Vec::new();
    for e in mesh.elem_iter() {
        let n = mesh.element_nodes(e);
        let d: Vec<usize> = sp.element_dofs(e).iter().map(|&d| d as usize).collect();
        let x: Vec<f64> = (0..3).map(|k| mesh.node_coords(n[k])[0]).collect();
        let y: Vec<f64> = (0..3).map(|k| mesh.node_coords(n[k])[1]).collect();
        let j00=x[1]-x[0]; let j01=x[2]-x[0]; let j10=y[1]-y[0]; let j11=y[2]-y[0];
        let det=j00*j11-j01*j10; let ad=det.abs(); let id=1.0/det;
        let mut mass = vec![0.0; nt*nt]; let mut stiff = vec![0.0; nt*nt];
        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr*ad; tri.eval_basis(xi, &mut phi); tri.eval_grad_basis(xi, &mut dphi);
            let mut dpx=vec![0.0;nt]; let mut dpy=vec![0.0;nt];
            for i in 0..nt { dpx[i]=(j11*dphi[i*2]-j10*dphi[i*2+1])*id; dpy[i]=(-j01*dphi[i*2]+j00*dphi[i*2+1])*id; }
            for i in 0..nt { for j in 0..nt {
                mass[i*nt+j] += w*phi[i]*phi[j]; stiff[i*nt+j] += w*(dpx[i]*dpx[j]+dpy[i]*dpy[j]);
            }}
        }
        for i in 0..nt { for j in 0..nt { mass[i*nt+j] += stiff[i*nt+j]; }}
        solve_inv(nt, &mut mass); blk.push(mass); dfs.push(d);
    }
    SinvData { blocks: blk, nt, dofs: dfs }
}
fn apply_sinv(s: &SinvData, x: &[f64], y: &mut [f64]) {
    y.fill(0.0); let nt = s.nt;
    for (b, d) in s.blocks.iter().zip(s.dofs.iter()) {
        for i in 0..nt { let mut v = 0.0; for j in 0..nt { v += b[i*nt+j]*x[d[j]]; } y[d[i]] = v; }
    }
}
fn solve_inv(n: usize, a: &mut [f64]) {
    // Compute inverse by solving A * inv_col = e_col for each column.
    // Save a copy for repeated use.
    let a0 = a.to_vec();
    let mut inv = vec![0.0; n*n];
    for col in 0..n {
        let mut ac = a0.clone();
        let mut b = vec![0.0; n]; b[col] = 1.0;
        // Forward elimination (Gaussian elimination with partial pivot)
        for c in 0..n {
            let mut best = c; let mut bv = ac[c*n+c].abs();
            for r in (c+1)..n { let v = ac[r*n+c].abs(); if v > bv { bv=v; best=r; } }
            if bv < 1e-30 { continue; }
            if best != c { for k in c..n { ac.swap(c*n+k, best*n+k); } b.swap(c, best); }
            let piv = ac[c*n+c];
            for r in (c+1)..n { let f = ac[r*n+c]/piv;
                for k in c..n { ac[r*n+k] -= f*ac[c*n+k]; } b[r] -= f*b[c]; }
        }
        // Back substitution
        for r in (0..n).rev() {
            let mut s = b[r];
            for k in (r+1)..n { s -= ac[r*n+k]*inv[k*n+col]; }
            inv[r*n+col] = if ac[r*n+r].abs() > 1e-30 { s / ac[r*n+r] } else { 0.0 };
        }
    }
    a.copy_from_slice(&inv);
}

fn elem_edges(nodes: &[u32]) -> Vec<((u32,u32), usize)> {
    let pairs: Vec<(usize,usize)> = match nodes.len() {
        3 => vec![(0,1),(1,2),(0,2)], 4 => vec![(0,1),(1,2),(2,3),(3,0)], _ => panic!()
    };
    pairs.iter().enumerate().map(|(i,&(a,b))| {
        let key = if nodes[a]<nodes[b] {(nodes[a],nodes[b])} else {(nodes[b],nodes[a])}; (key,i)
    }).collect()
}
fn edge_xi_tri(lf: usize, xi: f64) -> (f64,f64) {
    match lf {0=>(xi,0.0),1=>(1.0-xi,xi),2=>(0.0,xi),_=>(0.0,0.0)}
}

struct Trace { n: usize, dpf: usize, off: Vec<usize>, bf: Vec<(u32,usize,[u32;2])>, interior: Vec<(u32,u32,usize,usize,[u32;2])>, ifirst: usize }
fn trace_info(mesh: &impl MeshTopology, order: u8) -> Trace {
    let dpf = (order as usize+1).max(1);
    let nbf = mesh.n_boundary_faces() as usize;
    let mut bf = Vec::new();
    let mut emap: HashMap<(u32,u32),(u32,usize)> = HashMap::new();
    for e in mesh.elem_iter() { let en = mesh.element_nodes(e); for (k,li) in elem_edges(en) { emap.entry(k).or_insert((e,li)); }}
    for b in 0..nbf {
        let fn_ = mesh.face_nodes(b as u32);
        let key = if fn_[0]<fn_[1] {(fn_[0],fn_[1])} else {(fn_[1],fn_[0])};
        if let Some(&(el,li)) = emap.get(&key) { bf.push((el,li,[key.0,key.1])); }
        else { panic!("bf {b} not found"); }
    }
    let mut interior = Vec::new();
    let mut emap2: HashMap<(u32,u32),(u32,usize)> = HashMap::new();
    for e in mesh.elem_iter() { let en = mesh.element_nodes(e); for (k,li) in elem_edges(en) {
        if let Some(&(fe,fl)) = emap2.get(&k) { interior.push((fe,e,fl,li,[k.0,k.1])); }
        else { emap2.insert(k,(e,li)); }
    }}
    let nf = nbf + interior.len();
    let off: Vec<usize> = (0..nf).map(|f| f*dpf).collect();
    Trace { n: nf*dpf, dpf, off, bf: bf, interior, ifirst: nbf }
}

fn assemble_bhat(mesh: &impl MeshTopology, l2: &impl FESpace<Mesh=impl MeshTopology>, trace: &Trace, to: u8, qo: u8) -> CsrMatrix<f64> {
    let nt = n_dofs_l2(to); let mut coo = CooMatrix::new(l2.n_dofs(), trace.n);
    let tri = tri_ref(to); let eq = seg_rule_arbitrary(qo); let mut phi = vec![0.0; nt];
    for (b, &(el,li,ref nodes)) in trace.bf.iter().enumerate() {
        let td = trace.off[b]; let d: Vec<usize> = l2.element_dofs(el).iter().map(|&d| d as usize).collect();
        let pa = mesh.node_coords(nodes[0]); let pb = mesh.node_coords(nodes[1]);
        let elen = ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt();
        let npe = mesh.element_nodes(el).len();
        for (xr,&wr) in eq.points.iter().zip(eq.weights.iter()) {
            let xi = xr[0]; let w = wr*elen;
            let (rx,ry) = if npe==3 {edge_xi_tri(li,xi)} else {(0.0,0.0)};
            tri.eval_basis(&[rx,ry], &mut phi); for i in 0..nt { coo.add(d[i], td, w*phi[i]); }
        }
    }
    for (fi, &(el,er,ll,lr,ref nodes)) in trace.interior.iter().enumerate() {
        let td = trace.off[trace.ifirst+fi];
        let pa = mesh.node_coords(nodes[0]); let pb = mesh.node_coords(nodes[1]);
        let elen = ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt();
        let dl: Vec<usize> = l2.element_dofs(el).iter().map(|&d| d as usize).collect();
        let dr: Vec<usize> = l2.element_dofs(er).iter().map(|&d| d as usize).collect();
        let npl = mesh.element_nodes(el).len(); let npr = mesh.element_nodes(er).len();
        for (xr,&wr) in eq.points.iter().zip(eq.weights.iter()) {
            let xi = xr[0]; let w = wr*elen;
            let (rxl,ryl) = if npl==3 {edge_xi_tri(ll,xi)} else {(0.0,0.0)};
            tri.eval_basis(&[rxl,ryl], &mut phi); for i in 0..nt { coo.add(dl[i], td, w*phi[i]); }
            let (rxr,ryr) = if npr==3 {edge_xi_tri(lr,1.0-xi)} else {(0.0,0.0)};
            tri.eval_basis(&[rxr,ryr], &mut phi); for i in 0..nt { coo.add(dr[i], td, -w*phi[i]); }
        }
    }
    coo.into_csr()
}

fn build_shat(b: &CsrMatrix<f64>, s: &SinvData, ntrace: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(ntrace, ntrace); let nt = s.nt;
    for (blk, dfs) in s.blocks.iter().zip(s.dofs.iter()) {
        let cols: Vec<Vec<(usize,f64)>> = (0..nt).map(|i| {
            let mut v = Vec::new();
            for p in b.row_ptr[dfs[i]]..b.row_ptr[dfs[i]+1] { if b.values[p].abs() > 1e-30 { v.push((b.col_idx[p] as usize, b.values[p])); }}
            v
        }).collect();
        for i in 0..nt { for j in 0..nt { let s_ij = blk[i*nt+j]; if s_ij.abs()<1e-30{continue;}
            for &(ci,vi) in &cols[i] { for &(cj,vj) in &cols[j] { coo.add(ci,cj,s_ij*vi*vj); }}
        }}
    }
    coo.into_csr()
}

fn main() {
    let mesh = Mesh::<2>::unit_square_tri(1);
    let dim = 2; let to = 1; let tro = 0; let teo = 1;

    let x0 = H1Space::new(mesh.clone(), to);
    let test = L2Space::new(mesh.clone(), teo);
    let trace = trace_info(&mesh, tro);

    let s0 = x0.n_dofs(); let s1 = trace.n; let st = test.n_dofs();
    println!("Trial={s0}, Trace={s1}, Test={st}");
    let qo = 3; let qf = 2;
    let f_test = Assembler::assemble_linear(&test, &[&DomainSourceIntegrator::new(|_|1.0)], qo);

    // B0
    let ess_tags: Vec<i32> = mesh.unique_boundary_tags();
    let dm = x0.dof_manager();
    let ess_dofs: Vec<u32> = boundary_dofs(&mesh as &dyn MeshTopology, dm, &ess_tags);
    println!("Essential DOFs: {:?}", ess_dofs);

    let mut b0 = MixedAssembler::assemble_bilinear(&test, &x0, &[&MixedDiffusion], qo);
    // Zero BC columns
    for &d in &ess_dofs { let c = d as usize;
        for r in 0..b0.nrows { for p in b0.row_ptr[r]..b0.row_ptr[r+1] {
            if b0.col_idx[p] as usize == c { b0.values[p] = 0.0; }
        }}
        // Set one L2 DOF entry to 1.0
        for e in mesh.elem_iter() {
            let hd = x0.element_dofs(e);
            if let Some(k) = hd.iter().position(|&hd| hd == d) {
                let ld = test.element_dofs(e);
                if k < ld.len() {
                    let l2_dof = ld[k] as usize;
                    // Set B0[l2_dof, c] = 1.0
                    for p in b0.row_ptr[l2_dof]..b0.row_ptr[l2_dof+1] {
                        if b0.col_idx[p] as usize == c { b0.values[p] = 1.0; break; }
                    }
                }
                break;
            }
        }
    }

    // Bhat
    let bhat = assemble_bhat(&mesh, &test, &trace, teo, qf);
    println!("Bhat nnz={}", bhat.nnz());

    // Sinv
    let sinv = build_sinv(&test, qo);

    // S0
    let mut s0_mat = Assembler::assemble_bilinear(&x0, &[&DiffusionIntegrator{kappa:1.0}], qo);
    apply_dirichlet(&mut s0_mat, &mut vec![0.0; s0], &ess_dofs, &vec![0.0; ess_dofs.len()]);

    // RHS
    let mut sf = vec![0.0; st]; apply_sinv(&sinv, &f_test, &mut sf);
    let ntot = s0 + s1; let mut rhs = vec![0.0; ntot];
    for r in 0..st { let v = sf[r]; if v.abs()<1e-30{continue;}
        for p in b0.row_ptr[r]..b0.row_ptr[r+1] { rhs[b0.col_idx[p] as usize] += b0.values[p]*v; }
        for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { rhs[s0+bhat.col_idx[p] as usize] += bhat.values[p]*v; }
    }
    for &d in &ess_dofs { rhs[d as usize] = 0.0; }
    println!("||RHS|| = {:e}", rhs.iter().map(|v| v*v).sum::<f64>().sqrt());

    // Shat
    let shat = build_shat(&bhat, &sinv, s1);
    println!("Shat nnz={}, S0 nnz={}", shat.nnz(), s0_mat.nnz());

    // Test A = B^T * S^{-1} * B operator: check x^T * A * x > 0
    let mut x_rand = vec![0.0; ntot];
    for i in 0..ntot { x_rand[i] = ((i*7+13) as f64).sin(); }
    for &d in &ess_dofs { x_rand[d as usize] = 0.0; }

    // Compute A*x
    let mut ax = vec![0.0; ntot];
    let mut t0 = vec![0.0; st];
    for r in 0..st {
        let mut s = 0.0;
        for p in b0.row_ptr[r]..b0.row_ptr[r+1] { s += b0.values[p]*x_rand[b0.col_idx[p] as usize]; }
        for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { s += bhat.values[p]*x_rand[s0+bhat.col_idx[p] as usize]; }
        t0[r] = s;
    }
    let mut t1 = vec![0.0; st]; apply_sinv(&sinv, &t0, &mut t1);
    for r in 0..st { let v = t1[r]; if v.abs()<1e-30{continue;}
        for p in b0.row_ptr[r]..b0.row_ptr[r+1] { ax[b0.col_idx[p] as usize] += b0.values[p]*v; }
        for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { ax[s0+bhat.col_idx[p] as usize] += bhat.values[p]*v; }
    }
    for &d in &ess_dofs { ax[d as usize] = 0.0; }

    let pAp = x_rand.iter().zip(ax.iter()).map(|(x,a)| x*a).sum::<f64>();
    println!("x^T * A * x = {:.10e} (should be >0)", pAp);

    // Check symmetry: x^T * A * y vs y^T * A * x
    let mut y_rand = vec![0.0; ntot];
    for i in 0..ntot { y_rand[i] = ((i*3+7) as f64).cos(); }
    for &d in &ess_dofs { y_rand[d as usize] = 0.0; }

    t0.fill(0.0);
    for r in 0..st {
        let mut s = 0.0;
        for p in b0.row_ptr[r]..b0.row_ptr[r+1] { s += b0.values[p]*y_rand[b0.col_idx[p] as usize]; }
        for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { s += bhat.values[p]*y_rand[s0+bhat.col_idx[p] as usize]; }
        t0[r] = s;
    }
    t1.fill(0.0); apply_sinv(&sinv, &t0, &mut t1);
    let mut ay = vec![0.0; ntot];
    for r in 0..st { let v = t1[r]; if v.abs()<1e-30{continue;}
        for p in b0.row_ptr[r]..b0.row_ptr[r+1] { ay[b0.col_idx[p] as usize] += b0.values[p]*v; }
        for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { ay[s0+bhat.col_idx[p] as usize] += bhat.values[p]*v; }
    }
    for &d in &ess_dofs { ay[d as usize] = 0.0; }

    let xAy = x_rand.iter().zip(ay.iter()).map(|(x,a)| x*a).sum::<f64>();
    let yAx = y_rand.iter().zip(ax.iter()).map(|(y,a)| y*a).sum::<f64>();
    println!("x^T A y = {:.10e}, y^T A x = {:.10e}, diff = {:.10e}", xAy, yAx, (xAy-yAx).abs());

    // Try CG
    let ntot2 = ntot as f64;
    let mut x = vec![0.0; ntot];
    let ess_set: std::collections::HashSet<usize> = ess_dofs.iter().map(|&d| d as usize).collect();
    let mut iter = 0usize;
    let res = pcg(ntot, |v,w| {
        w.fill(0.0); let mut t0v = vec![0.0; st];
        for r in 0..st { let mut s = 0.0;
            for p in b0.row_ptr[r]..b0.row_ptr[r+1] { s += b0.values[p]*v[b0.col_idx[p] as usize]; }
            for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { s += bhat.values[p]*v[s0+bhat.col_idx[p] as usize]; }
            t0v[r] = s;
        }
        let mut t1v = vec![0.0; st]; apply_sinv(&sinv, &t0v, &mut t1v);
        for r in 0..st { let vt = t1v[r]; if vt.abs()<1e-30{continue;}
            for p in b0.row_ptr[r]..b0.row_ptr[r+1] { w[b0.col_idx[p] as usize] += b0.values[p]*vt; }
            for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { w[s0+bhat.col_idx[p] as usize] += bhat.values[p]*vt; }
        }
        for &d in &ess_dofs { w[d as usize] = 0.0; }
    }, &rhs, &mut x, 200, 1e-12, 0.0,
    |r,z| { z.copy_from_slice(r); for &d in &ess_dofs {z[d as usize]=0.0;} },
    &ess_set, &mut iter);
    println!("CG: {iter} iters, residual={res:.6e}");
    println!("||x0|| = {:e}", x.iter().map(|v| v*v).sum::<f64>().sqrt());

    // Compare with standard H1 solve
    let mut x_h1 = vec![0.0; s0];
    let f_h1 = vec![0.0; s0]; // no RHS for standard Poisson with f=0 on RHS
    // Actually, f=1 gives RHS = ∫ φ_i * 1 dx
    let f_h1_rhs = Assembler::assemble_linear(&x0, &[&DomainSourceIntegrator::new(|_|1.0)], qo);
    let mut s0_copy = s0_mat.clone();
    let mut rhs_h1 = f_h1_rhs.clone();
    apply_dirichlet(&mut s0_copy, &mut rhs_h1, &ess_dofs, &vec![0.0; ess_dofs.len()]);
    let cfg = SolverConfig{rtol:1e-12,max_iter:1000,verbose:false,..Default::default()};
    let _ = fem_solver::solve_cg(&s0_copy, &rhs_h1, &mut x_h1, &cfg);
    println!("H1 solution norm = {:e}", x_h1.iter().map(|v| v*v).sum::<f64>().sqrt());
}

fn pcg(n: usize, a: impl Fn(&[f64],&mut [f64]), b: &[f64], x: &mut [f64],
       mi: usize, rtol: f64, atol: f64,
       p: impl Fn(&[f64],&mut [f64]), ess: &std::collections::HashSet<usize>,
       iter: &mut usize) -> f64 {
    let bn = b.iter().map(|v| v*v).sum::<f64>().sqrt().max(1e-300);
    let tol = (rtol*bn).max(atol);
    for &d in ess { x[d] = 0.0; }
    let mut r = vec![0.0; n]; a(x, &mut r); for i in 0..n { r[i] = b[i]-r[i]; }
    let mut z = vec![0.0; n]; p(&r, &mut z);
    let mut pk = z.clone(); let mut rz = r.iter().zip(z.iter()).map(|(x,y)| x*y).sum::<f64>();
    for it in 1..=mi {
        *iter = it;
        let mut ap = vec![0.0; n]; a(&pk, &mut ap);
        let pap = pk.iter().zip(ap.iter()).map(|(x,a)| x*a).sum::<f64>().max(1e-300);
        let al = rz / pap;
        for i in 0..n { x[i] += al*pk[i]; } for &d in ess { x[d] = 0.0; }
        for i in 0..n { r[i] -= al*ap[i]; }
        let res = r.iter().map(|v| v*v).sum::<f64>().sqrt();
        if res < tol { return res; }
        p(&r, &mut z); let rzn = r.iter().zip(z.iter()).map(|(x,y)| x*y).sum::<f64>();
        let be = rzn / rz.max(1e-30); rz = rzn;
        for i in 0..n { pk[i] = z[i] + be*pk[i]; }
    }
    let mut ax = vec![0.0; n]; a(x, &mut ax);
    (0..n).map(|i| { let d = b[i]-ax[i]; d*d }).sum::<f64>().sqrt()
}
