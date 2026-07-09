//! # Example 8 — DPG Poisson (2×2 block formulation)
//!
//! One-to-one translation of MFEM C++ ex8.
//!
//! Solves `-Δu = 1` with homogeneous Dirichlet BC using the Discontinuous
//! Petrov-Galerkin (DPG) method in its primal 2×2 block form.
//!
//! Three spaces:
//! - **Trial (X0):** H¹ continuous (`order`)
//! - **Interface (Xhat):** trace on mesh skeleton (`order - 1`)
//! - **Test (Y):** L² discontinuous (enriched)
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex8_dpg_2x2 -- -m data/star.mesh
//! cargo run --example mfem_ex8_dpg_2x2 -- -m data/square-disc.mesh
//! ```

#![allow(dead_code, unused_variables)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use fem_assembly::{
    Assembler, MixedAssembler, MixedBilinearIntegrator,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    integrator::QpData,
};
use fem_element::{
    ReferenceElement, reference::QuadratureRule,
    lagrange::{TriP1, TriP2, TriP3, QuadQ1, QuadQ2},
    quadrature::{seg_rule_arbitrary, tri_rule, quad_rule_arbitrary},
};
use fem_mesh::ElementTransformation;
use fem_mesh::element_type::ElementType;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{refine_uniform, topology::MeshTopology, Mesh};
use fem_solver::SolverConfig;
use fem_space::{
    H1Space, L2Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

// ─── Mixed Diffusion Integrator ────────────────────────────────────────────────

struct MixedDiffusion;
impl MixedBilinearIntegrator for MixedDiffusion {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m: &mut [f64]) {
        let nr = qp_row.n_dofs; let nc = qp_col.n_dofs; let d = qp_col.dim; let w = qp_col.weight;
        for k in 0..d {
            for i in 0..nr { let gik = qp_row.grad_phys[i*d+k];
                for j in 0..nc { m[i*nc+j] += w * gik * qp_col.grad_phys[j*d+k]; }
            }
        }
    }
}

// ─── Sinv: block-diagonal (M+K)^{-1} ──────────────────────────────────────────

struct SinvData { elem_blocks: Vec<Vec<f64>>, n_test: usize, elem_dofs: Vec<Vec<usize>> }

fn ref_elem_2d(et: ElementType, o: u8) -> (Box<dyn ReferenceElement>, usize) {
    match (et, o) {
        (ElementType::Tri3|ElementType::Tri6, 1) => (Box::new(TriP1), 3),
        (ElementType::Tri3|ElementType::Tri6, 2) => (Box::new(TriP2), 6),
        (ElementType::Tri3|ElementType::Tri6, 3) => (Box::new(TriP3), 10),
        (ElementType::Quad4, 1) => (Box::new(QuadQ1), 4),
        (ElementType::Quad4, 2) => (Box::new(QuadQ2), 9),
        _ => panic!("ref_elem: ({et:?}, o={o})"),
    }
}
fn get_qr(et: ElementType, qo: u8) -> QuadratureRule {
    match et {
        ElementType::Tri3|ElementType::Tri6 => tri_rule(qo),
        ElementType::Quad4 => quad_rule_arbitrary(qo),
        _ => panic!("qr: {et:?}"),
    }
}
fn transform_grads(jit: &nalgebra::DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, d: usize) {
    for i in 0..n { for j in 0..d {
        let mut s = 0.0; for k in 0..d { s += jit[(j,k)] * gr[i*d+k]; }
        gp[i*d+j] = s;
    }}
}

fn build_sinv<M: MeshTopology>(space: &impl FESpace<Mesh=M>, qo: u8) -> SinvData {
    let mesh = space.mesh(); let ne = mesh.n_elements(); let order = space.order();
    let et = mesh.element_type(0); let dim = 2;
    let (ref_elem, nt) = ref_elem_2d(et, order);
    let qr = get_qr(et, qo);
    let mut phi = vec![0.0; nt]; let mut dphi = vec![0.0; nt*dim];
    let mut eb = Vec::with_capacity(ne); let mut ed = Vec::with_capacity(ne);
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
        let jit = tr.jacobian_inv_t().clone();
        let mut mass = vec![0.0; nt*nt]; let mut stiff = vec![0.0; nt*nt];
        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * tr.det_j().abs();
            ref_elem.eval_basis(xi, &mut phi); ref_elem.eval_grad_basis(xi, &mut dphi);
            let mut gp = vec![0.0; nt*dim]; transform_grads(&jit, &dphi, &mut gp, nt, dim);
            for i in 0..nt { for j in 0..nt {
                mass[i*nt+j] += w*phi[i]*phi[j];
                let mut gdot = 0.0; for d in 0..dim { gdot += gp[i*dim+d] * gp[j*dim+d]; }
                stiff[i*nt+j] += w * gdot;
            }}
        }
        for i in 0..nt { for j in 0..nt { mass[i*nt+j] += stiff[i*nt+j]; }}
        solve_dense_inv(nt, &mut mass);
        eb.push(mass); ed.push(dofs);
    }
    SinvData { elem_blocks: eb, n_test: nt, elem_dofs: ed }
}

fn apply_sinv(s: &SinvData, x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    let nt = s.n_test;
    for (b, d) in s.elem_blocks.iter().zip(s.elem_dofs.iter()) {
        for i in 0..nt { let mut v = 0.0; for j in 0..nt { v += b[i*nt+j]*x[d[j]]; } y[d[i]] = v; }
    }
}

// ─── Edge enumeration (tri + quad) ────────────────────────────────────────────

fn elem_edges(nodes: &[u32]) -> Vec<((u32,u32), usize)> {
    let pairs: Vec<(usize,usize)> = match nodes.len() {
        3 => vec![(0,1),(1,2),(0,2)],
        4 => vec![(0,1),(1,2),(2,3),(3,0)],
        _ => panic!("npe={}", nodes.len()),
    };
    pairs.iter().enumerate().map(|(i,&(a,b))| {
        let key = if nodes[a] < nodes[b] {(nodes[a],nodes[b])} else {(nodes[b],nodes[a])};
        (key, i)
    }).collect()
}

fn edge_xi_tri(lf: usize, xi: f64) -> (f64,f64) {
    match lf {0=>(xi,0.0),1=>(1.0-xi,xi),2=>(0.0,xi),_=>(0.0,0.0)}
}
fn edge_xi_quad(lf: usize, xi: f64) -> (f64,f64) {
    match lf {0=>(xi,0.0),1=>(1.0,xi),2=>(1.0-xi,1.0),3=>(0.0,1.0-xi),_=>(0.0,0.0)}
}

// ─── Build all-face layout (boundary + interior) ──────────────────────────────

struct BfaceInfo {
    elem: u32, local_edge: usize, nodes: [u32; 2],
}

struct TraceLayout {
    n_dofs: usize, dpf: usize, n_bfaces: usize,
    face_offset: Vec<usize>,
    /// Per-boundary-face info
    bfaces: Vec<BfaceInfo>,
    /// Interior face data: (el, er, ll, lr, [node0, node1])
    interior_faces: Vec<(u32,u32,usize,usize,[u32;2])>,
    interior_first: usize,
}

fn build_all_faces(mesh: &impl MeshTopology, order: u8) -> TraceLayout {
    let dpf = (order as usize + 1).max(1);
    let nbf = mesh.n_boundary_faces() as usize;

    let mut bfaces: Vec<BfaceInfo> = Vec::with_capacity(nbf);
    // Pre-build element edge map sorted_key → (elem, local_edge)
    let mut edge_to_elem: HashMap<(u32,u32), (u32,usize)> = HashMap::new();
    for e in mesh.elem_iter() {
        let en = mesh.element_nodes(e);
        for (key, li) in elem_edges(en) {
            edge_to_elem.entry(key).or_insert((e, li));
        }
    }
    for bf in 0..nbf {
        let fnodes = mesh.face_nodes(bf as u32);
        let key = if fnodes[0] < fnodes[1] {(fnodes[0], fnodes[1])} else {(fnodes[1], fnodes[0])};
        if let Some(&(el, li)) = edge_to_elem.get(&key) {
            bfaces.push(BfaceInfo { elem: el, local_edge: li, nodes: [key.0, key.1] });
        } else {
            panic!("boundary face {bf} (nodes {},{}) not in edge map", fnodes[0], fnodes[1]);
        }
    }

    let mut edge_map: HashMap<(u32,u32), (u32,usize)> = HashMap::new();
    let mut interior: Vec<(u32,u32,usize,usize,[u32;2])> = Vec::new();
    for e in mesh.elem_iter() {
        let en = mesh.element_nodes(e);
        for (key, li) in elem_edges(en) {
            if let Some(&(fe, fl)) = edge_map.get(&key) {
                interior.push((fe, e, fl, li, [key.0, key.1]));
            } else {
                edge_map.insert(key, (e, li));
            }
        }
    }

    let n_ifaces = interior.len();
    let n_faces = nbf + n_ifaces;
    let mut off = Vec::with_capacity(n_faces);
    for f in 0..n_faces { off.push(f * dpf); }

    TraceLayout {
        n_dofs: n_faces * dpf, dpf, n_bfaces: nbf,
        face_offset: off, bfaces, interior_faces: interior,
        interior_first: nbf,
    }
}

// ─── Bhat assembly ────────────────────────────────────────────────────────────

fn assemble_bhat<M: MeshTopology>(
    mesh: &M, l2: &impl FESpace<Mesh=M>, trace: &TraceLayout,
    test_order: u8, qo: u8,
) -> CsrMatrix<f64> {
    let nt = ref_elem_2d(mesh.element_type(0), test_order).1;
    let mut coo = CooMatrix::new(l2.n_dofs(), trace.n_dofs);
    let tri: Box<dyn ReferenceElement> = match test_order {
        1 => Box::new(TriP1), 2 => Box::new(TriP2), 3 => Box::new(TriP3), _ => panic!(),
    };
    let eq = seg_rule_arbitrary(qo);
    let mut phi = vec![0.0; nt];

    // Boundary faces (use pre-computed trace.bfaces)
    for (bf, info) in trace.bfaces.iter().enumerate() {
        let td = trace.face_offset[bf];
        let dofs: Vec<usize> = l2.element_dofs(info.elem).iter().map(|&d| d as usize).collect();
        let elen = {
            let pa = mesh.node_coords(info.nodes[0]); let pb = mesh.node_coords(info.nodes[1]);
            ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt()
        };
        let npe = mesh.element_nodes(info.elem).len();
        for (xr, &wr) in eq.points.iter().zip(eq.weights.iter()) {
            let xi = xr[0]; let w = wr*elen;
            let (rx,ry) = if npe==3 {edge_xi_tri(info.local_edge,xi)} else {edge_xi_quad(info.local_edge,xi)};
            tri.eval_basis(&[rx,ry], &mut phi);
            for i in 0..nt { coo.add(dofs[i], td, w*phi[i]); }
        }
    }

    // Interior faces
    for (fi, (el, er, ll, lr, nodes)) in trace.interior_faces.iter().enumerate() {
        let fidx = trace.interior_first + fi;
        let td = trace.face_offset[fidx];
        let elen = {
            let pa = mesh.node_coords(nodes[0]); let pb = mesh.node_coords(nodes[1]);
            ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt()
        };
        let dl: Vec<usize> = l2.element_dofs(*el).iter().map(|&d| d as usize).collect();
        let dr: Vec<usize> = l2.element_dofs(*er).iter().map(|&d| d as usize).collect();
        let npe_l = mesh.element_nodes(*el).len();
        let npe_r = mesh.element_nodes(*er).len();
        for (xr, &wr) in eq.points.iter().zip(eq.weights.iter()) {
            let xi = xr[0]; let w = wr*elen;
            let (rxl,ryl) = if npe_l==3 {edge_xi_tri(*ll,xi)} else {edge_xi_quad(*ll,xi)};
            tri.eval_basis(&[rxl,ryl], &mut phi);
            for i in 0..nt { coo.add(dl[i], td, w*phi[i]); }
            let (rxr,ryr) = if npe_r==3 {edge_xi_tri(*lr,1.0-xi)} else {edge_xi_quad(*lr,1.0-xi)};
            tri.eval_basis(&[rxr,ryr], &mut phi);
            for i in 0..nt { coo.add(dr[i], td, -w*phi[i]); }
        }
    }
    coo.into_csr()
}

// ─── Shat = Bhatᵀ · S⁻¹ · Bhat ───────────────────────────────────────────────

fn build_shat(bhat: &CsrMatrix<f64>, sinv: &SinvData, ntrace: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(ntrace, ntrace);
    let nt = sinv.n_test;
    for (block, dofs) in sinv.elem_blocks.iter().zip(sinv.elem_dofs.iter()) {
        let cols: Vec<Vec<(usize,f64)>> = (0..nt).map(|i| {
            let mut v = Vec::new();
            for p in bhat.row_ptr[dofs[i]]..bhat.row_ptr[dofs[i]+1] {
                if bhat.values[p].abs() > 1e-30 { v.push((bhat.col_idx[p] as usize, bhat.values[p])); }
            }
            v
        }).collect();
        for i in 0..nt { for j in 0..nt {
            let sij = block[i*nt+j];
            if sij.abs() < 1e-30 { continue; }
            for &(ci,vi) in &cols[i] { for &(cj,vj) in &cols[j] { coo.add(ci, cj, sij*vi*vj); } }
        }}
    }
    coo.into_csr()
}

// ─── Dense inverse ────────────────────────────────────────────────────────────

fn solve_dense_inv(n: usize, a: &mut [f64]) {
    let a0 = a.to_vec();
    let mut inv = vec![0.0; n*n];
    for col in 0..n {
        let mut ac = a0.clone();
        let mut b = vec![0.0; n]; b[col] = 1.0;
        for c in 0..n {
            let mut best = c; let mut bv = ac[c*n+c].abs();
            for r in (c+1)..n { let v = ac[r*n+c].abs(); if v > bv { bv=v; best=r; } }
            if bv < 1e-30 { continue; }
            if best != c { for k in c..n { ac.swap(c*n+k, best*n+k); } b.swap(c, best); }
            let piv = ac[c*n+c];
            for r in (c+1)..n { let f = ac[r*n+c]/piv;
                for k in c..n { ac[r*n+k] -= f*ac[c*n+k]; } b[r] -= f*b[c]; }
        }
        for r in (0..n).rev() {
            let mut s = b[r];
            for k in (r+1)..n { s -= ac[r*n+k]*inv[k*n+col]; }
            inv[r*n+col] = if ac[r*n+r].abs() > 1e-30 { s / ac[r*n+r] } else { 0.0 };
        }
    }
    a.copy_from_slice(&inv);
}

// ─── Block CG preconditioner ─────────────────────────────────────────────────

fn block_cg_prec(r: &[f64], z: &mut [f64], sizes: &[usize], mats: &[&CsrMatrix<f64>], cfg: &SolverConfig) {
    let mut off = 0;
    for (k, &sz) in sizes.iter().enumerate() {
        let _ = fem_solver::solve_cg(mats[k], &r[off..off+sz], &mut z[off..off+sz], cfg);
        off += sz;
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let mfem = read_mfem_file(&args.mesh).expect("read mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("2D mesh");
    let dim = 2;

    // 3. Refine
    let rl = { let ne = mesh.n_elems() as f64;
        (10000.0/ne).ln().max(0.0)/(2.0_f64).ln()/dim as f64 } as usize;
    let mesh = if rl > 0 { let mut m = mesh; for _ in 0..rl { m = refine_uniform(&m); }
        eprintln!("  Refined: {} nodes, {} elements ({} lvl)", m.n_nodes(), m.n_elems(), rl); m
    } else { mesh };

    // 4. Spaces
    let t_order = args.order;
    let tr_order = if args.order > 0 { args.order-1 } else { 0 };
    let mut te_order = args.order;
    if dim==2 && (args.order%2==0 || args.order>1) { te_order += 1; }
    if te_order < t_order { eprintln!("  Warning: test not enriched"); }

    use fem_space::fe_space::FESpace;
    let x0 = H1Space::new(mesh.clone(), t_order);
    let test = L2Space::new(mesh.clone(), te_order);
    let trace = build_all_faces(&mesh, tr_order);

    let s0 = x0.n_dofs(); let s1 = trace.n_dofs; let st = test.n_dofs();
    println!("\nNumber of Unknowns:");
    println!("  Trial space,     X0   : {s0} (order {t_order})");
    println!("  Interface space, Xhat : {s1} (order {tr_order})");
    println!("  Test space,      Y    : {st} (order {te_order})\n");

    // 5. F on test space
    let qo = (te_order as u8*2+2).max(3);
    let f_test = Assembler::assemble_linear(&test, &[&DomainSourceIntegrator::new(|_|1.0)], qo);

    // 6. B0 (trial × test diffusion)
    let ess_tags: Vec<i32> = mesh.unique_boundary_tags();
    let dm = x0.dof_manager();
    let ess_dofs: Vec<u32> = boundary_dofs(&mesh as &dyn MeshTopology, dm, &ess_tags);
    let mut b0 = MixedAssembler::assemble_bilinear(&test, &x0, &[&MixedDiffusion], qo);

    // EliminateTrialEssentialBC for homogeneous BC: zero BC columns of B0
    // For the normal equation A = B^T * S^{-1} * B, zeroing column j of B0
    // removes the contribution of x0[j] from B*x, so x0[j] stays at its initial 0.
    for &d in &ess_dofs {
        let c = d as usize;
        for row in 0..b0.nrows { for p in b0.row_ptr[row]..b0.row_ptr[row+1] {
            if b0.col_idx[p] as usize == c { b0.values[p] = 0.0; }
        }}
    }

    // 7. Bhat (trace × test face coupling)
    let qf = (te_order as u8*2).max(2);
    let bhat = assemble_bhat(&mesh, &test, &trace, te_order, qf);

    // 8. Sinv = (M+K)^{-1}
    let sinv = build_sinv(&test, qo);

    // 9. S0 (trial stiffness with BC)
    let mut s0_mat = Assembler::assemble_bilinear(&x0, &[&DiffusionIntegrator{kappa:1.0}], qo);
    let mut zr = vec![0.0; s0];
    apply_dirichlet(&mut s0_mat, &mut zr, &ess_dofs, &vec![0.0; ess_dofs.len()]);

    // 10. RHS: b = B^T * S^{-1} * F
    let mut sf = vec![0.0; st]; apply_sinv(&sinv, &f_test, &mut sf);
    let ntot = s0 + s1;
    let mut rhs = vec![0.0; ntot];
    for i in 0..s0 { rhs[i] = b0_t(&b0, i, &sf); }
    for i in 0..st { let v = sf[i]; if v.abs()<1e-30 {continue;}
        for p in bhat.row_ptr[i]..bhat.row_ptr[i+1] { rhs[s0+bhat.col_idx[p] as usize] += bhat.values[p]*v; }
    }
    for &d in &ess_dofs { rhs[d as usize] = 0.0; }

    // 11. Shat = Bhat^T * S^{-1} * Bhat
    let shat = build_shat(&bhat, &sinv, s1);

    // 12. Preconditioner
    let pcfg = SolverConfig{rtol:1e-3,max_iter:200,verbose:false,..Default::default()};
    let bsizes = vec![s0, s1];
    let pmats = vec![&s0_mat as &CsrMatrix<f64>, &shat as &CsrMatrix<f64>];

    // 13. PCG
    let mut x = vec![0.0; ntot];
    let a_op = |v: &[f64], w: &mut [f64]| {
        w.fill(0.0);
        // 1. tmp0 = B * v = B0*v0 + Bhat*v1
        let mut t0 = vec![0.0; st];
        for r in 0..st {
            let mut s = 0.0;
            for p in b0.row_ptr[r]..b0.row_ptr[r+1] { s += b0.values[p]*v[b0.col_idx[p] as usize]; }
            for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { s += bhat.values[p]*v[s0+bhat.col_idx[p] as usize]; }
            t0[r] = s;
        }
        // 2. t1 = S^{-1} * t0
        let mut t1 = vec![0.0; st];
        apply_sinv(&sinv, &t0, &mut t1);
        // 3. w0 = B0^T * t1 (single pass)
        for r in 0..st {
            let tv = t1[r];
            if tv.abs() < 1e-30 { continue; }
            for p in b0.row_ptr[r]..b0.row_ptr[r+1] { w[b0.col_idx[p] as usize] += b0.values[p]*tv; }
        }
        // 4. w1 = Bhat^T * t1 (single pass)
        for r in 0..st {
            let tv = t1[r];
            if tv.abs() < 1e-30 { continue; }
            for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { w[s0+bhat.col_idx[p] as usize] += bhat.values[p]*tv; }
        }
        // BC
        for &d in &ess_dofs { w[d as usize] = 0.0; }
    };

    let mut iter = 0usize;
    let ess_set: std::collections::HashSet<usize> = ess_dofs.iter().map(|&d| d as usize).collect();
    // Build block Jacobi preconditioner (diagonal of each block)
    let diag_s0: Vec<f64> = (0..s0).map(|i| { let d = s0_mat.get(i,i); if d.abs() > 1e-30 { 1.0/d } else { 1.0 } }).collect();
    let diag_shat: Vec<f64> = (0..s1).map(|i| { let d = shat.get(i,i); if d.abs() > 1e-30 { 1.0/d } else { 1.0 } }).collect();
    let res = pcg(ntot, &a_op, &rhs, &mut x, 10000, 1e-12, 0.0,
        |r,z| {
            for i in 0..s0 { z[i] = diag_s0[i] * r[i]; }
            for i in 0..s1 { z[s0+i] = diag_shat[i] * r[s0+i]; }
            for &d in &ess_dofs { z[d as usize] = 0.0; }
        },
        &ess_set, &mut iter,
    );
    println!("PCG: iterations={iter}, final residual={res:.3e}");

    // 14. DPG residual ||Bx-F||_{S^{-1}}
    let mut ls = vec![0.0; st];
    for r in 0..st {
        let mut s = 0.0;
        for p in b0.row_ptr[r]..b0.row_ptr[r+1] { s += b0.values[p]*x[b0.col_idx[p] as usize]; }
        for p in bhat.row_ptr[r]..bhat.row_ptr[r+1] { s += bhat.values[p]*x[s0+bhat.col_idx[p] as usize]; }
        ls[r] = s - f_test[r];
    }
    let mut sls = vec![0.0; st]; apply_sinv(&sinv, &ls, &mut sls);
    let dres: f64 = ls.iter().zip(sls.iter()).map(|(a,b)| a*b).sum::<f64>().sqrt();
    println!("\n|| B0*x0 + Bhat*xhat - F ||_{{S^{{-1}}}} = {dres:.7}");

    // 15. Save
    let mut mf = File::create("refined.mesh").unwrap();
    fem_io::mfem::write_mfem(&mut mf, &mesh, None).unwrap();
    let mut sf = File::create("sol.gf").unwrap();
    for i in 0..s0 { writeln!(sf, "{:.14e}", x[i]).unwrap(); }
    eprintln!("  Wrote refined.mesh, sol.gf");
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn b0_t(b0: &CsrMatrix<f64>, col: usize, y: &[f64]) -> f64 {
    let mut s = 0.0;
    for row in 0..b0.nrows { for p in b0.row_ptr[row]..b0.row_ptr[row+1] {
        if b0.col_idx[p] as usize == col { s += b0.values[p]*y[row]; break; }
    }}
    s
}

// ─── PCG ──────────────────────────────────────────────────────────────────────

fn pcg(
    n: usize, a: &dyn Fn(&[f64],&mut [f64]), b: &[f64], x: &mut [f64],
    mi: usize, rtol: f64, atol: f64,
    p: impl Fn(&[f64],&mut [f64]), ess: &std::collections::HashSet<usize>,
    iter: &mut usize,
) -> f64 {
    let bn = dot(b,b).sqrt().max(1e-300); let tol = (rtol*bn).max(atol);
    // Ensure x has BC enforced
    for &d in ess { x[d] = 0.0; }
    let mut r = vec![0.0; n]; a(x, &mut r); for i in 0..n { r[i] = b[i]-r[i]; }
    let mut z = vec![0.0; n]; p(&r, &mut z);
    let mut pk = z.clone(); let mut rz = dot(&r,&z);
    for it in 1..=mi {
        *iter = it;
        let mut ap = vec![0.0; n]; a(&pk, &mut ap);
        let mut pap = dot(&pk,&ap); if pap <= 0.0 && pap.abs() < 1e-30 { pap = 1e-30; }
        let al = rz / pap;
        for i in 0..n { x[i] += al*pk[i]; }
        for &d in ess { x[d] = 0.0; } // enforce BC
        for i in 0..n { r[i] -= al*ap[i]; }
        let res = dot(&r,&r).sqrt();
        if res < tol { return res; }
        p(&r, &mut z); let rzn = dot(&r,&z); let be = rzn / rz.max(1e-30);
        rz = rzn; for i in 0..n { pk[i] = z[i] + be*pk[i]; }
    }
    let mut ax = vec![0.0; n]; a(x, &mut ax);
    let mut res = 0.0; for i in 0..n { let d = b[i]-ax[i]; res += d*d; }
    res.sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| x*y).sum() }

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args { mesh: String, order: u8 }
impl Args {
    fn parse() -> Self {
        let mut mesh = "../data/star.mesh".to_string(); let mut order = 1u8;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() { match arg.as_str() {
            "-m"|"--mesh" => { if let Some(v)=it.next() { mesh=v; } }
            "-o"|"--order" => { order=it.next().and_then(|s|s.parse().ok()).unwrap_or(1); }
            _ => {}
        }}
        Args { mesh, order }
    }
}
