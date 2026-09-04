//! ex31 Rust-side dump helper (temporary, cf. ex29 dump flow).
//! Run:
//!   cargo run --example mfem_ex31_dump -- -m data/inline-quad.mesh -r 2 -o 1
//! Writes rust_dofpos.txt / rust_A.txt / rust_b.txt / rust_elim_A.txt /
//! rust_elim_B.txt / rust_elim_X0.txt / rust_x.txt / rust_soldofs.txt /
//! rust_elmat_0.txt in the current directory.

use std::f64::consts::{PI, SQRT_2};
use std::fs::File;
use std::io::{BufWriter, Write};

use fem_assembly::standard::{CurlCurlIntegrator, DiffusionIntegrator, MassIntegrator,
    VectorMassTensorIntegrator};
use fem_assembly::coefficient::ConstantMatrixCoeff;
use fem_assembly::{VectorAssembler, Assembler, FixedOrder};
use fem_assembly::postproc::grid_function::project_bdr_coefficient_tangent_2d;
use fem_element::{VectorReferenceElement, ReferenceElement,
    nedelec::{TriNDk, QuadNDk}, lagrange::{TriP1, QuadQk}};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::{ElementType, Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{solve_pcg, GSSmoother, SolverConfig};
use fem_space::{HCurlSpace, H1Space,
    fe_space::FESpace, constraints::{boundary_dofs_hcurl, boundary_dofs}};

const A0: f64 = 1.1; const A1: f64 = 1.2; const A2: f64 = 1.3;
const PHI1: f64 = 0.4 * PI; const PHI2: f64 = 0.9 * PI;
const SXX: f64 = 2.0; const SXY: f64 = 1.0 / SQRT_2;
const SYY: f64 = 2.0; const SYZ: f64 = 1.0 / SQRT_2; const SZZ: f64 = 2.0;

fn exact_e(x: &[f64], kappa: f64) -> [f64; 3] {
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    [A0 * u.sin(), A1 * (u + PHI1).sin(), A2 * (u + PHI2).sin()]
}

fn exact_curl(x: &[f64], kappa: f64) -> [f64; 3] {
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    let (c0, c4, c9) = (u.cos(), (u + PHI1).cos(), (u + PHI2).cos());
    let a = kappa / SQRT_2;
    [A2 * c9 * a, -A2 * c9 * a, A1 * c4 * a - A0 * c0 * a]
}

fn source_3d(x: &[f64], kappa: f64) -> [f64; 3] {
    let k2 = kappa * kappa;
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    let (s0, s4, s9) = (u.sin(), (u + PHI1).sin(), (u + PHI2).sin());
    let f0 = 0.55 * (4.0 + k2) * s0 + 0.6 * (SQRT_2 - k2) * s4;
    let f1 = 0.55 * (SQRT_2 - k2) * s0 + 0.6 * (4.0 + k2) * s4 + 0.65 * SQRT_2 * s9;
    let f2 = 0.6 * SQRT_2 * s4 + 1.3 * (2.0 + k2) * s9;
    [f0, f1, f2]
}

type JacobianFn = fn(
    &Mesh<2>, u32, &[u32], &[f64],
) -> (f64, f64, f64, f64, f64, f64);

fn affine_jac(mesh: &Mesh<2>, _e: u32, nodes: &[u32], _xi: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let (j00, j01) = (x1[0] - x0[0], x2[0] - x0[0]);
    let (j10, j11) = (x1[1] - x0[1], x2[1] - x0[1]);
    let det = j00 * j11 - j01 * j10;
    let inv = 1.0 / det;
    (inv, j11 * inv, -j10 * inv, -j01 * inv, j00 * inv, det.abs())
}

fn isoparametric_jac(mesh: &Mesh<2>, _e: u32, nodes: &[u32], xi: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    let geo = QuadQk::new(1);
    let n_geo = geo.n_dofs();
    let mut grad = vec![0.0_f64; n_geo * 2];
    geo.eval_grad_basis(xi, &mut grad);
    let mut j = nalgebra::DMatrix::<f64>::zeros(2, 2);
    for k in 0..n_geo {
        let xk = mesh.node_coords(nodes[k]);
        for i in 0..2 { for d in 0..2 { j[(i, d)] += xk[i] * grad[k * 2 + d]; } }
    }
    let det = j.determinant();
    let inv = 1.0 / det;
    (inv, j[(1,1)] * inv, -j[(1,0)] * inv, -j[(0,1)] * inv, j[(0,0)] * inv, det.abs())
}

fn setup_element_ref(et: ElementType, _order: u8) -> (usize, &'static dyn VectorReferenceElement, Box<dyn ReferenceElement>, usize, JacobianFn) {
    match et {
        ElementType::Tri3 => { eprintln!("TriNDk cast not supported - skipping"); std::process::exit(0); },
        ElementType::Quad4 => { eprintln!("QuadNDk cast not supported - skipping"); std::process::exit(0); },
        _ => panic!("unsupported element type {et:?}"),
    }
}

fn dump_vec(path: &str, v: &[f64]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    for &x in v { writeln!(w, "{:.16e}", x).unwrap(); }
}

fn dump_csr(path: &str, n: usize, row_ptr: &[usize], col_idx: &[u32], values: &[f64]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    for i in 0..n {
        write!(w, "[row {i}]").unwrap();
        for k in row_ptr[i]..row_ptr[i+1] {
            write!(w, " ({} ,{:.16e})", col_idx[k], values[k]).unwrap();
        }
        writeln!(w).unwrap();
    }
}

fn main() {
    let mut mesh_arg: Option<String> = None;
    let mut ref_levels = 2usize;
    let mut order = 1u8;
    let mut freq = 1.0_f64;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => mesh_arg = it.next(),
            "-r" | "--refine" => ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            "-o" | "--order" => order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-f" | "--frequency" => freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            _ => {}
        }
    }
    let kappa = freq * PI;
    let path = mesh_arg.expect("-m mesh required");
    let mfem = read_mfem_file(&path).expect("read MFEM mesh");
    let base_mesh = mfem.mesh2d.expect("2D mesh");
    let mesh = if ref_levels > 0 {
        let mut m = base_mesh;
        for _ in 0..ref_levels { m = refine_uniform(&m); }
        m
    } else { base_mesh };

    let quad_order = order * 2 + 2;
    let nd_space = HCurlSpace::new(mesh.clone(), order);
    let z_space = H1Space::new(mesh.clone(), order);
    let n_nd = nd_space.n_dofs();
    let n_h1 = z_space.n_dofs();
    let n_total = n_nd + n_h1;
    println!("DOFs: H(Curl)={n_nd}  H1(z)={n_h1}  total={n_total}");
    println!("nelems={} nverts={} nedges={}", mesh.n_elements(), mesh.n_nodes(), nd_space.n_edges());

    // ---- dump per-DOF positions: ND dof -> edge midpoint, H1 dof -> vertex coord ----
    {
        let mut pos = vec![0.0_f64; n_total * 3];
        // collect dof -> edge midpoint by walking all elements' edges
        let edge_pairs: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
        let mut seen = vec![false; n_nd];
        for e in 0..mesh.n_elements() as u32 {
            let nodes = mesh.element_nodes(e);
            let pairs: &[(usize, usize)] = match mesh.element_type(e) {
                ElementType::Tri3 | ElementType::Tri6 => &[(0, 1), (1, 2), (0, 2)],
                ElementType::Quad4 | ElementType::Quad8 => &edge_pairs,
                _ => &[],
            };
            for &(li, lj) in pairs {
                let (gi, gj) = (nodes[li], nodes[lj]);
                let key = fem_space::EdgeKey::new(gi, gj);
                if let Some(d) = nd_space.edge_dof(key) {
                    let d = d as usize;
                    if !seen[d] {
                        seen[d] = true;
                        let ca = mesh.node_coords(gi);
                        let cb = mesh.node_coords(gj);
                        pos[3*d] = 0.5 * (ca[0] + cb[0]);
                        pos[3*d+1] = 0.5 * (ca[1] + cb[1]);
                    }
                }
            }
        }
        for n in 0..mesh.n_nodes() {
            let c = mesh.node_coords(n as u32);
            let d = n_nd + n;
            pos[3*d] = c[0]; pos[3*d+1] = c[1];
        }
        let f = File::create("rust_dofpos.txt").unwrap();
        let mut w = BufWriter::new(f);
        for x in &pos { writeln!(w, "{:.16e}", x).unwrap(); }
    }

    // ---- element matrix of element 0 (Araw element check) ----
    {
        let e: u32 = 0;
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let q = rnd.quadrature(quad_order);
        let mut em = vec![0.0_f64; (n_ld + n_lh1) * (n_ld + n_lh1)];
        // curl-curl block (in-plane)
        let mut cc = vec![0.0_f64; n_ld * n_ld];
        let mut curl = vec![0.0_f64; n_ld];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det;
            rnd.eval_curl(xi, &mut curl);
            for i in 0..n_ld { for j in 0..n_ld {
                cc[i * n_ld + j] += w * signs[i] * signs[j] * curl[i] * curl[j];
            }}
        }
        // vector mass tensor block (in-plane 2x2)
        let mut vm = vec![0.0_f64; n_ld * n_ld];
        let mut np = vec![0.0; n_ld * 2];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det;
            rnd.eval_basis_vec(xi, &mut np);
            for i in 0..n_ld {
                let pxi = signs[i] * (jit00 * np[i*2] + jit01 * np[i*2+1]);
                let pyi = signs[i] * (jit10 * np[i*2] + jit11 * np[i*2+1]);
                for j in 0..n_ld {
                    let pxj = signs[j] * (jit00 * np[j*2] + jit01 * np[j*2+1]);
                    let pyj = signs[j] * (jit10 * np[j*2] + jit11 * np[j*2+1]);
                    vm[i * n_ld + j] += w * (SXX * pxi * pxj + SXY * pxi * pyj + SXY * pyi * pxj + SYY * pyi * pyj);
                }
            }
        }
        // z block: -laplace + sigma_zz mass
        let mut zm = vec![0.0_f64; n_lh1 * n_lh1];
        let mut hp = vec![0.0_f64; n_lh1];
        let mut gr = vec![0.0_f64; n_lh1 * 2];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det;
            rh1.eval_basis(xi, &mut hp);
            rh1.eval_grad_basis(xi, &mut gr);
            for i in 0..n_lh1 {
                let (dxi, dyi) = (jit00*gr[i*2]+jit01*gr[i*2+1], jit10*gr[i*2]+jit11*gr[i*2+1]);
                for j in 0..n_lh1 {
                    let (dxj, dyj) = (jit00*gr[j*2]+jit01*gr[j*2+1], jit10*gr[j*2]+jit11*gr[j*2+1]);
                    zm[i * n_lh1 + j] += w * (dxi * dxj + dyi * dyj + SZZ * hp[i] * hp[j]);
                }
            }
        }
        // coupling: SYZ * Ey * Ez
        let mut cp = vec![0.0_f64; n_ld * n_lh1];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det * SYZ;
            rnd.eval_basis_vec(xi, &mut np);
            rh1.eval_basis(xi, &mut hp);
            for i in 0..n_ld {
                let py = signs[i] * (jit10 * np[i*2] + jit11 * np[i*2+1]);
                for j in 0..n_lh1 { cp[i * n_lh1 + j] += w * py * hp[j]; }
            }
        }
        let n = n_ld + n_lh1;
        for i in 0..n_ld { for j in 0..n_ld { em[i*n + j] += cc[i*n_ld+j] + vm[i*n_ld+j]; } }
        for i in 0..n_lh1 { for j in 0..n_lh1 { em[(n_ld+i)*n + n_ld+j] += zm[i*n_lh1+j]; } }
        for i in 0..n_ld { for j in 0..n_lh1 {
            em[i*n + n_ld+j] += cp[i*n_lh1+j];
            em[(n_ld+j)*n + i] += cp[i*n_lh1+j];
        }}
        let f = File::create("rust_elmat_0.txt").unwrap();
        let mut w = BufWriter::new(f);
        write!(w, "dofs").unwrap();
        for &d in nd_dofs.iter().chain(h1_dofs.iter()) { write!(w, " {d}").unwrap(); }
        writeln!(w).unwrap();
        write!(w, "verts").unwrap();
        for &nv in nodes { let c = mesh.node_coords(nv); write!(w, " {},{},{}", c[0], c[1], 0.0).unwrap(); }
        writeln!(w).unwrap();
        for i in 0..n {
            for j in 0..n {
                write!(w, "{:.16e}{}", em[i*n+j], if j+1 < n { " " } else { "\n" }).unwrap();
            }
        }
    }

    // ---- assemble system (mirrors the example) ----
    let curl_curl = CurlCurlIntegrator { mu: 1.0 };
    let sigma_2d = ConstantMatrixCoeff(vec![SXX, SXY, SXY, SYY]);
    let vec_mass = VectorMassTensorIntegrator { alpha: sigma_2d };
    let a_nd = VectorAssembler::assemble_bilinear(&nd_space, &[&curl_curl, &vec_mass], quad_order);

    let laplace = FixedOrder::new(DiffusionIntegrator { kappa: 1.0 }, 0);
    let z_mass = MassIntegrator { rho: SZZ };
    let a_z = Assembler::assemble_bilinear(&z_space, &[&laplace, &z_mass], quad_order);

    let mut coupling_coo = CooMatrix::<f64>::new(n_nd, n_h1);
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let q = rnd.quadrature(quad_order);
        let mut np = vec![0.0; n_ld * 2];
        let mut hp = vec![0.0; n_lh1];
        let mut em = vec![0.0_f64; n_ld * n_lh1];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det * SYZ;
            rnd.eval_basis_vec(xi, &mut np);
            rh1.eval_basis(xi, &mut hp);
            for i in 0..n_ld {
                let py = signs[i] * (jit10 * np[i * 2] + jit11 * np[i * 2 + 1]);
                if py.abs() < 1e-15 { continue; }
                for j in 0..n_lh1 { em[i * n_lh1 + j] += w * py * hp[j]; }
            }
        }
        for (li, &ri) in nd_dofs.iter().enumerate() {
            for (lj, &cj) in h1_dofs.iter().enumerate() {
                let v = em[li * n_lh1 + lj];
                if v != 0.0 { coupling_coo.add(ri, cj, v); }
            }
        }
    }
    let coupling = coupling_coo.into_csr();

    let mut sys_coo = CooMatrix::<f64>::new(n_total, n_total);
    // Layout: z (vertex) DOFs first (0..n_h1), then in-plane ND DOFs (n_h1..),
    // matching MFEM's ND_R2D GetElementVDofs.
    for r in 0..n_nd { let rr = n_h1 + r; for k in a_nd.row_ptr[r]..a_nd.row_ptr[r+1] { sys_coo.add(rr, n_h1 + a_nd.col_idx[k] as usize, a_nd.values[k]); } }
    for r in 0..n_h1 { for k in a_z.row_ptr[r]..a_z.row_ptr[r+1] { sys_coo.add(r, a_z.col_idx[k] as usize, a_z.values[k]); } }
    for r in 0..coupling.nrows {
        for k in coupling.row_ptr[r]..coupling.row_ptr[r+1] {
            let c = coupling.col_idx[k] as usize; let v = coupling.values[k];
            if v != 0.0 { sys_coo.add(n_h1 + r, c, v); sys_coo.add(c, n_h1 + r, v); }
        }
    }
    let sys_mat = sys_coo.into_csr();

    // rhs
    let src_nd = FixedOrder::new(FnVectorSource(Box::new(move |x| { let f = source_3d(x, kappa); [f[0], f[1]] })), 2);
    let rhs_nd = VectorAssembler::assemble_linear(&nd_space, &[&src_nd], quad_order);
    let src_z = FixedOrder::new(FnScalarSource(Box::new(move |x| source_3d(x, kappa)[2])), 2);
    let rhs_z = Assembler::assemble_linear(&z_space, &[&src_z], quad_order);
    let mut rhs = vec![0.0_f64; n_total];
    for i in 0..n_h1 { rhs[i] = rhs_z[i]; }
    for i in 0..n_nd { rhs[n_h1 + i] = rhs_nd[i]; }

    // ---- dump raw A and b ----
    dump_csr("rust_A.txt", n_total, &sys_mat.row_ptr, &sys_mat.col_idx, &sys_mat.values);
    dump_vec("rust_b.txt", &rhs);

    // ---- BC ----
    let nd_bdr = boundary_dofs_hcurl(&mesh, &nd_space, &mesh.unique_boundary_tags());
    let h1_bdr = boundary_dofs(&mesh, z_space.dof_manager(), &mesh.unique_boundary_tags());
    eprintln!("  BC DOFs: H(Curl)={}  H1(z)={}", nd_bdr.len(), h1_bdr.len());
    let mut x = vec![0.0_f64; n_total];
    project_bdr_coefficient_tangent_2d(&mut x[n_h1..], &nd_space,
        &|x: &[f64], out: &mut [f64]| { let e = exact_e(x, kappa); out[0] = e[0]; out[1] = e[1]; },
        &mesh.unique_boundary_tags());
    for &d in &h1_bdr { let c = z_space.dof_manager().dof_coord(d); x[d as usize] = exact_e(c, kappa)[2]; }
    dump_vec("rust_soldofs.txt", &x);

    // eliminate (DIAG_KEEP, MFEM EliminateVDofs style) and dump eliminated system + X0
    let mut elim_mat = sys_mat.clone();
    let mut elim_b = rhs.clone();
    for &d in &nd_bdr { elim_mat.apply_dirichlet_keep_diag(n_h1 + d as usize, x[n_h1 + d as usize], &mut elim_b); }
    for &d in &h1_bdr { elim_mat.apply_dirichlet_keep_diag(d as usize, x[d as usize], &mut elim_b); }
    dump_csr("rust_elim_A.txt", n_total, &elim_mat.row_ptr, &elim_mat.col_idx, &elim_mat.values);
    dump_vec("rust_elim_B.txt", &elim_b);
    dump_vec("rust_elim_X0.txt", &x);

    // solve
    let cfg = SolverConfig { rtol: 1e-12, max_iter: 500, verbose: true, ..Default::default() };
    let linlvo_mat = fem_linalg::fem_to_linlvo_csr(&elim_mat);
    let precond = GSSmoother::from_csr(&linlvo_mat).expect("GSSmoother");
    solve_pcg(&elim_mat, &elim_b, &mut x, &precond, cfg.rtol, cfg.max_iter, true).expect("PCG");
    dump_vec("rust_x.txt", &x);

    // H(Curl) error (reuse the example's computation)
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let qord = (order as u8 * 6).max(3);
        let q = rnd.quadrature(qord);
        let mut pn = vec![0.0; n_ld * 2];
        let mut ph = vec![0.0; n_lh1];
        let mut cn = vec![0.0; n_ld];
        for (qi, xi) in q.points.iter().enumerate() {
            let (inv_det, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det;
            let xp = if mesh.element_type(e) == ElementType::Quad4 {
                let geo = QuadQk::new(1); let ng = geo.n_dofs(); let mut phi = vec![0.0; ng];
                geo.eval_basis(xi, &mut phi);
                let mut p = [0.0_f64; 2];
                for k in 0..ng { let c = mesh.node_coords(nodes[k]); p[0] += phi[k] * c[0]; p[1] += phi[k] * c[1]; }
                p
            } else {
                let x0 = mesh.node_coords(nodes[0]);
                let x1 = mesh.node_coords(nodes[1]);
                let x2 = mesh.node_coords(nodes[2]);
                [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                 x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]]
            };
            rnd.eval_basis_vec(xi, &mut pn);
            rh1.eval_basis(xi, &mut ph);
            rnd.eval_curl(xi, &mut cn);
            let mut eh = [0.0_f64; 3];
            for i in 0..n_ld {
                let s = signs[i];
                eh[0] += s * x[n_h1 + nd_dofs[i]] * (jit00 * pn[i*2] + jit01 * pn[i*2+1]);
                eh[1] += s * x[n_h1 + nd_dofs[i]] * (jit10 * pn[i*2] + jit11 * pn[i*2+1]);
            }
            for j in 0..n_lh1 { eh[2] += x[h1_dofs[j]] * ph[j]; }
            let mut ce = [0.0_f64; 3];
            for i in 0..n_ld { ce[2] += signs[i] * x[n_h1 + nd_dofs[i]] * cn[i]; }
            ce[2] *= inv_det;
            let mut gr = vec![0.0_f64; n_lh1 * 2];
            rh1.eval_grad_basis(xi, &mut gr);
            for j in 0..n_lh1 {
                let (dx, dy) = (jit00*gr[j*2]+jit01*gr[j*2+1], jit10*gr[j*2]+jit11*gr[j*2+1]);
                ce[0] += x[h1_dofs[j]] * dy;
                ce[1] -= x[h1_dofs[j]] * dx;
            }
            let (ee, ec) = (exact_e(&xp, kappa), exact_curl(&xp, kappa));
            for c in 0..3 { let d = eh[c] - ee[c]; err2 += w * d * d; let dc = ce[c] - ec[c]; err2 += w * dc * dc; }
        }
    }
    let hcurl_err = err2.sqrt();
    println!("\n|| E_h - E ||_{{H(Curl)}} = {hcurl_err:.10e}");
}

// ─── helpers ────────────────────────────────────────────────────────────────

use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
struct FnVectorSource(Box<dyn Fn(&[f64]) -> [f64; 2] + Send + Sync>);
impl VectorLinearIntegrator for FnVectorSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
        let f = (self.0)(qp.x_phys);
        for i in 0..qp.n_dofs { fe[i] += qp.weight * (qp.phi_vec[i*2]*f[0] + qp.phi_vec[i*2+1]*f[1]); }
    }
}
use fem_assembly::integrator::{LinearIntegrator, QpData};
struct FnScalarSource(Box<dyn Fn(&[f64]) -> f64 + Send + Sync>);
impl LinearIntegrator for FnScalarSource {
    fn add_to_element_vector(&self, qp: &QpData<'_>, fe: &mut [f64]) {
        let f = (self.0)(qp.x_phys);
        for i in 0..qp.n_dofs { fe[i] += qp.weight * qp.phi[i] * f; }
    }
}
