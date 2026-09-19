//! ex31 Rust-side dump helper (D375; feeds `tools/ex31_cpp_helper/`).
//!
//! Assembles the **same** serial ex31 system as
//! `examples/mfem_ex31_anisotropic_maxwell.rs` (2-D restricted H(curl), order 1)
//! and dumps every intermediate the comparison harness reads:
//!
//! ```bash
//! cargo run --release --example mfem_ex31_dump -- -m data/inline-quad.mesh -r 2 -o 1
//! ```
//!
//! Writes into the current directory: `rust_dofpos.txt`, `rust_A.txt`,
//! `rust_b.txt`, `rust_elmat_0.txt`, `rust_soldofs.txt`, `rust_elim_A.txt`,
//! `rust_elim_B.txt`, `rust_elim_X0.txt`, `rust_x.txt`.
//!
//! ## Index convention (single, uniform)
//!
//! **Every** dumped vector/matrix uses the MFEM `ND_R2D` global DOF layout
//! `[z vertex DOFs 0..n_verts | in-plane ND edge DOFs n_verts..]` (H¹ dof id =
//! vertex id, ND dof id = n_verts + edge id), which is exactly the C++
//! `FiniteElementSpace` ordering.  `rust_dofpos.txt` (the permutation anchor:
//! vertex DOFs → vertex coord, edge DOFs → edge midpoint) follows the same
//! layout — the earlier draft wrote it in `[ND | vertex]` order while A/b used
//! `[z | nd]`, so one permutation could not compare both (fixed in D375).
//! `rust_soldofs.txt` is the full projection of `E_exact` (MFEM
//! `sol.ProjectCoefficient`: ND1 dof = `E·t` at the edge midpoint, H1 vertex
//! dof = `E_z`), unlike the C++ `cpp_soldofs.txt` dump it is NOT compared by
//! `compare_ex31_systems.py` (which compares A, b, elim_A/B/X0 and x only).
//! `rust_elmat_0.txt` header lines (`dofs`/`verts`) print local DOF ids and
//! coordinates for eyeballing; the comparer prints them verbatim.
//!
//! The solve replicates the C++ dump (`PCG(*A, M, B, X, 1, 500, 1e-12, 0.0)`
//! → `SetRelTol(sqrt(1e-12)) = 1e-6`) with the bit-for-bit
//! [`fem_solver::solve_pcg_gssmoother`] port, so `rust_x.txt` is comparable to
//! `cpp_x.txt` at the same stopping point.

use std::f64::consts::{PI, SQRT_2};
use std::fs::File;
use std::io::{BufWriter, Write};

use fem_assembly::standard::{CurlCurlIntegrator, DiffusionIntegrator, MassIntegrator,
    VectorMassTensorIntegrator};
use fem_assembly::coefficient::ConstantMatrixCoeff;
use fem_assembly::{VectorAssembler, Assembler, FixedOrder};
use fem_element::{VectorReferenceElement, ReferenceElement,
    nedelec::{TriNDk, QuadNDk}, lagrange::{TriP1, QuadQk}};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::{ElementType, Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{SolverConfig, fmt_g, solve_pcg_gssmoother};
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

/// Local reference elements of the `[z | nd]` combined space (order 1 only —
/// see the `-o != 1` refusal in `main`; same table as the main example).
fn setup_element_ref(et: ElementType, order: u8) -> (usize, &'static dyn VectorReferenceElement, Box<dyn ReferenceElement>, usize, JacobianFn) {
    assert_eq!(order, 1, "setup_element_ref only implements order 1");
    match et {
        ElementType::Tri3 => {
            // Leak to get 'static lifetime (acceptable for singleton reference elements)
            let nd: &'static TriNDk = Box::leak(Box::new(TriNDk::new(1)));
            (nd.n_dofs(), nd as &dyn VectorReferenceElement, Box::new(TriP1), 3, affine_jac as JacobianFn)
        },
        ElementType::Quad4 => {
            let nd: &'static QuadNDk = Box::leak(Box::new(QuadNDk::new(1)));
            (nd.n_dofs(), nd as &dyn VectorReferenceElement, Box::new(QuadQk::new(1)), 4, isoparametric_jac as JacobianFn)
        },
        _ => {
            eprintln!(
                "mfem_ex31_dump: unsupported element type {et:?} (only straight-sided Tri3 and \
                 Quad4 are implemented) — exiting with status 3"
            );
            std::process::exit(3)
        }
    }
}

fn phys_point(mesh: &Mesh<2>, et: ElementType, nodes: &[u32], xi: &[f64]) -> [f64; 2] {
    if et == ElementType::Quad4 {
        let geo = QuadQk::new(1);
        let ng = geo.n_dofs();
        let mut phi = vec![0.0; ng];
        geo.eval_basis(xi, &mut phi);
        let mut p = [0.0_f64; 2];
        for k in 0..ng {
            let c = mesh.node_coords(nodes[k]);
            p[0] += phi[k] * c[0];
            p[1] += phi[k] * c[1];
        }
        p
    } else {
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
         x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]]
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
    // The order-1 restricted space is all this harness supports (same gap as
    // the main example: no order-p local basis).
    if order != 1 {
        eprintln!(
            "mfem_ex31_dump: -o {order} is not ported (order-p restricted H(curl) space); \
             re-run with -o 1."
        );
        std::process::exit(3);
    }
    let kappa = freq * PI;
    let path = mesh_arg.expect("-m mesh required");
    let mfem = read_mfem_file(&path).expect("read MFEM mesh");
    let base_mesh = match mfem.mesh2d {
        Some(m) => m,
        None => {
            eprintln!(
                "mfem_ex31_dump: mesh '{path}' is not 2-D (only the 2-D restricted H(curl) path \
                 is ported) — exiting with status 3"
            );
            std::process::exit(3)
        }
    };
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

    // ---- per-DOF positions + full projection of E_exact --------------------
    // ONE layout everywhere: [z vertex dofs 0..n_h1 | ND edge dofs n_h1..].
    // Vertex dof -> vertex coord, edge dof -> edge midpoint (permutation
    // anchor).  The projection replicates MFEM sol.ProjectCoefficient (ND1 dof
    // functional: E(mid)·(b-a) on the canonical edge; H1 vertex dof: E_z).
    let mut pos = vec![0.0_f64; n_total * 3];
    let mut soldofs = vec![0.0_f64; n_total];
    {
        let mut seen = vec![false; n_nd];
        for e in 0..mesh.n_elements() as u32 {
            let nodes = mesh.element_nodes(e);
            let pairs: &[(usize, usize)] = match mesh.element_type(e) {
                ElementType::Tri3 | ElementType::Tri6 => &[(0, 1), (1, 2), (0, 2)],
                ElementType::Quad4 | ElementType::Quad8 => &[(0, 1), (1, 2), (2, 3), (3, 0)],
                _ => &[],
            };
            for &(li, lj) in pairs {
                let key = fem_space::EdgeKey::new(nodes[li], nodes[lj]);
                if let Some(d) = nd_space.edge_dof(key) {
                    let d = d as usize;
                    if !seen[d] {
                        seen[d] = true;
                        let ca = mesh.node_coords(key.0);
                        let cb = mesh.node_coords(key.1);
                        let mid = [(ca[0] + cb[0]) * 0.5, (ca[1] + cb[1]) * 0.5];
                        pos[3*(n_h1 + d)] = mid[0];
                        pos[3*(n_h1 + d) + 1] = mid[1];
                        let e3 = exact_e(&mid, kappa);
                        soldofs[n_h1 + d] = e3[0] * (cb[0] - ca[0]) + e3[1] * (cb[1] - ca[1]);
                    }
                }
            }
        }
        for n in 0..mesh.n_nodes() {
            let c = mesh.node_coords(n as u32);
            pos[3*n] = c[0]; pos[3*n + 1] = c[1];
            soldofs[n] = exact_e(c, kappa)[2];
        }
        dump_vec("rust_dofpos.txt", &pos);
        dump_vec("rust_soldofs.txt", &soldofs);
    }

    // ---- element matrix of element 0 (Araw element check) ----
    {
        let e: u32 = 0;
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        // MFEM per-integrator rules: CurlCurlIntegrator 2p-2 = 0 (1 point),
        // VectorFEMassIntegrator OrderW + 2p = 2 (affine elements).
        let q0 = rnd.quadrature(0);
        let q2 = rnd.quadrature(2);
        let mut em = vec![0.0_f64; (n_ld + n_lh1) * (n_ld + n_lh1)];
        // curl-curl block (in-plane, 1-point rule)
        let mut cc = vec![0.0_f64; n_ld * n_ld];
        let mut curl = vec![0.0_f64; n_ld];
        for (qi, xi) in q0.points.iter().enumerate() {
            let (_, _jit00, _jit01, _jit10, _jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q0.weights[qi] * det;
            rnd.eval_curl(xi, &mut curl);
            for i in 0..n_ld { for j in 0..n_ld {
                cc[i * n_ld + j] += w * signs[i] * signs[j] * curl[i] * curl[j];
            }}
        }
        // vector mass tensor block (in-plane 2x2, order-2 rule)
        let mut vm = vec![0.0_f64; n_ld * n_ld];
        let mut np = vec![0.0; n_ld * 2];
        for (qi, xi) in q2.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q2.weights[qi] * det;
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
        // z block: -laplace (1-point rule) + Σzz mass (order-2 rule)
        let mut zm = vec![0.0_f64; n_lh1 * n_lh1];
        let mut hp = vec![0.0_f64; n_lh1];
        let mut gr = vec![0.0_f64; n_lh1 * 2];
        for (qi, xi) in q0.points.iter().enumerate() {
            let (_, jit00, jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q0.weights[qi] * det;
            rh1.eval_grad_basis(xi, &mut gr);
            for i in 0..n_lh1 {
                let (dxi, dyi) = (jit00*gr[i*2]+jit01*gr[i*2+1], jit10*gr[i*2]+jit11*gr[i*2+1]);
                for j in 0..n_lh1 {
                    let (dxj, dyj) = (jit00*gr[j*2]+jit01*gr[j*2+1], jit10*gr[j*2]+jit11*gr[j*2+1]);
                    zm[i * n_lh1 + j] += w * (dxi * dxj + dyi * dyj);
                }
            }
        }
        for (qi, xi) in q2.points.iter().enumerate() {
            let (_, _jit00, _jit01, _jit10, _jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q2.weights[qi] * det;
            rh1.eval_basis(xi, &mut hp);
            for i in 0..n_lh1 {
                for j in 0..n_lh1 {
                    zm[i * n_lh1 + j] += w * SZZ * hp[i] * hp[j];
                }
            }
        }
        // coupling: SYZ * Ey * Ez (order-2 rule)
        let mut cp = vec![0.0_f64; n_ld * n_lh1];
        for (qi, xi) in q2.points.iter().enumerate() {
            let (_, _jit00, _jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q2.weights[qi] * det * SYZ;
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

    // ---- assemble the combined [z | nd] system (mirrors the example) ----
    // MFEM per-integrator rules: curl-curl 2p-2 = 0, masses OrderW + 2p = 2.
    let cc0 = FixedOrder::new(CurlCurlIntegrator { mu: 1.0 }, 0);
    let vm2 = FixedOrder::new(VectorMassTensorIntegrator { alpha: ConstantMatrixCoeff(vec![SXX, SXY, SXY, SYY]) }, 2);
    let a_nd = VectorAssembler::assemble_bilinear(&nd_space, &[&cc0, &vm2], quad_order);

    let laplace = FixedOrder::new(DiffusionIntegrator { kappa: 1.0 }, 0);
    let z_mass = FixedOrder::new(MassIntegrator { rho: SZZ }, 2);
    let a_z = Assembler::assemble_bilinear(&z_space, &[&laplace, &z_mass], quad_order);

    let mut coupling_coo = CooMatrix::<f64>::new(n_nd, n_h1);
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let q = rnd.quadrature(2); // MFEM VectorFEMassIntegrator rule: OrderW + 2p = 2
        let mut np = vec![0.0; n_ld * 2];
        let mut hp = vec![0.0; n_lh1];
        let mut em = vec![0.0_f64; n_ld * n_lh1];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, _jit00, _jit01, jit10, jit11, det) = jac_fn(&mesh, e, nodes, xi);
            let w = q.weights[qi] * det * SYZ;
            rnd.eval_basis_vec(xi, &mut np);
            rh1.eval_basis(xi, &mut hp);
            for i in 0..n_ld {
                let py = signs[i] * (jit10 * np[i * 2] + jit11 * np[i * 2 + 1]);
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
    for r in 0..n_nd {
        let rr = n_h1 + r;
        for k in a_nd.row_ptr[r]..a_nd.row_ptr[r + 1] {
            sys_coo.add(rr, n_h1 + a_nd.col_idx[k] as usize, a_nd.values[k]);
        }
    }
    for r in 0..n_h1 {
        for k in a_z.row_ptr[r]..a_z.row_ptr[r + 1] {
            sys_coo.add(r, a_z.col_idx[k] as usize, a_z.values[k]);
        }
    }
    for r in 0..coupling.nrows {
        for k in coupling.row_ptr[r]..coupling.row_ptr[r + 1] {
            let c = coupling.col_idx[k] as usize;
            let v = coupling.values[k];
            if v != 0.0 { sys_coo.add(n_h1 + r, c, v); sys_coo.add(c, n_h1 + r, v); }
        }
    }
    let sys_mat = sys_coo.into_csr();

    // rhs (VectorFEDomainLFIntegrator order 2·GetOrder() = 2 for the source)
    let src_nd = FixedOrder::new(FnVectorSource(Box::new(move |x| { let f = source_3d(x, kappa); [f[0], f[1]] })), 2);
    let rhs_nd = VectorAssembler::assemble_linear(&nd_space, &[&src_nd], quad_order);
    let src_z = FixedOrder::new(FnScalarSource(Box::new(move |x| source_3d(x, kappa)[2])), 2);
    let rhs_z = Assembler::assemble_linear(&z_space, &[&src_z], quad_order);
    let mut rhs = vec![0.0_f64; n_total];
    for i in 0..n_h1 { rhs[i] = rhs_z[i]; }
    for i in 0..n_nd { rhs[n_h1 + i] = rhs_nd[i]; }

    // ---- dump raw A and b (both in the [z | nd] layout) ----
    dump_csr("rust_A.txt", n_total, &sys_mat.row_ptr, &sys_mat.col_idx, &sys_mat.values);
    dump_vec("rust_b.txt", &rhs);

    // ---- BC: projected exact solution on all boundary DOFs ----
    let bdr_tags = mesh.unique_boundary_tags();
    let nd_bdr = boundary_dofs_hcurl(&mesh, &nd_space, &bdr_tags);
    let h1_bdr = boundary_dofs(&mesh, z_space.dof_manager(), &bdr_tags);
    eprintln!("  BC DOFs: H(Curl)={}  H1(z)={}", nd_bdr.len(), h1_bdr.len());
    let mut x = vec![0.0_f64; n_total];
    // Boundary values via the verified library projection (ND tangent data) —
    // the interior values in `soldofs` above carry the full-projection view.
    for &d in nd_bdr.iter() {
        x[n_h1 + d as usize] = soldofs[n_h1 + d as usize];
    }
    for &d in &h1_bdr { x[d as usize] = soldofs[d as usize]; }

    // eliminate (DIAG_KEEP, MFEM EliminateVDofs style) and dump eliminated system + X0
    let mut elim_mat = sys_mat.clone();
    let mut elim_b = rhs.clone();
    for &d in &nd_bdr { elim_mat.apply_dirichlet_keep_diag(n_h1 + d as usize, x[n_h1 + d as usize], &mut elim_b); }
    for &d in &h1_bdr { elim_mat.apply_dirichlet_keep_diag(d as usize, x[d as usize], &mut elim_b); }
    dump_csr("rust_elim_A.txt", n_total, &elim_mat.row_ptr, &elim_mat.col_idx, &elim_mat.values);
    dump_vec("rust_elim_B.txt", &elim_b);
    dump_vec("rust_elim_X0.txt", &x);

    // solve: C++ dump calls PCG(*A, M, B, X, 1, 500, 1e-12, 0.0), i.e. the
    // free-function wrapper with SetRelTol(sqrt(1e-12)) = 1e-6.
    let cfg = SolverConfig { rtol: 1e-6, max_iter: 500, verbose: true, ..Default::default() };
    solve_pcg_gssmoother(&elim_mat, &elim_b, &mut x, &cfg).expect("PCG");
    dump_vec("rust_x.txt", &x);

    // H(Curl) error (same functional as the main example, on the solved x)
    let err2 = hcurl_error_sq(&mesh, &nd_space, &z_space, &x, order, kappa);
    println!("\n|| E_h - E ||_{{H(Curl)}} = {}\n", fmt_g(err2.sqrt()));
}

fn hcurl_error_sq(
    mesh: &Mesh<2>,
    nd_space: &HCurlSpace<Mesh<2>>,
    z_space: &H1Space<Mesh<2>>,
    x: &[f64],
    order: u8,
    kappa: f64,
) -> f64 {
    let n_h1 = z_space.n_dofs();
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let qord = 2 * order + 3;
        let q = rnd.quadrature(qord);
        let mut pn = vec![0.0; n_ld * 2];
        let mut ph = vec![0.0; n_lh1];
        let mut cn = vec![0.0; n_ld];
        for (qi, xi) in q.points.iter().enumerate() {
            let (inv_det, jit00, jit01, jit10, jit11, det) = jac_fn(mesh, e, nodes, xi);
            let w = q.weights[qi] * det;
            let xp = phys_point(mesh, mesh.element_type(e), nodes, xi);
            rnd.eval_basis_vec(xi, &mut pn);
            rh1.eval_basis(xi, &mut ph);
            rnd.eval_curl(xi, &mut cn);
            let mut eh = [0.0_f64; 3];
            for i in 0..n_ld {
                let s = signs[i];
                // MFEM CalcVShape_ND: shape = vshape_ref · J⁻¹ (row vector
                // right-multiplied), so φx = jit00·φx + jit01·φy etc.
                eh[0] += s * x[n_h1 + nd_dofs[i]] * (jit00 * pn[i * 2] + jit01 * pn[i * 2 + 1]);
                eh[1] += s * x[n_h1 + nd_dofs[i]] * (jit10 * pn[i * 2] + jit11 * pn[i * 2 + 1]);
            }
            for j in 0..n_lh1 { eh[2] += x[h1_dofs[j]] * ph[j]; }
            let mut ce = [0.0_f64; 3];
            for i in 0..n_ld { ce[2] += signs[i] * x[n_h1 + nd_dofs[i]] * cn[i]; }
            ce[2] *= inv_det;
            let mut gr = vec![0.0_f64; n_lh1 * 2];
            rh1.eval_grad_basis(xi, &mut gr);
            for j in 0..n_lh1 {
                // ∇z_phys = J⁻¹·∇z_ref (column convention; see the main
                // example's compute_hcurl_error for the jit mapping).
                let dx = jit00 * gr[j * 2] + jit01 * gr[j * 2 + 1];
                let dy = jit10 * gr[j * 2] + jit11 * gr[j * 2 + 1];
                ce[0] += x[h1_dofs[j]] * dy;
                ce[1] -= x[h1_dofs[j]] * dx;
            }
            let (ee, ec) = (exact_e(&xp, kappa), exact_curl(&xp, kappa));
            for c in 0..3 {
                let d = eh[c] - ee[c]; err2 += w * d * d;
                let dc = ce[c] - ec[c]; err2 += w * dc * dc;
            }
        }
    }
    err2
}

// ─── helper source integrators (same as the main example) ───────────────────

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
