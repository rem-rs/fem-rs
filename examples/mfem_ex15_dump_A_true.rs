//! Dump the Rust it2 true-DOF system matrix (conforming assemble + Dirichlet
//! elimination, DIAG_KEEP) for 1:1 comparison with tools_ex15_ref/dump_A_true.cpp.
//! Usage: cargo run --release -p fem-examples --example mfem_ex15_dump_A_true

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::read_mfem_file;
use fem_linalg::csr_spmm;
use fem_mesh::{Mesh, MeshTopology};
use fem_mesh::amr::{NCStateQuad, NcState2D};
use fem_space::constraints::{apply_dirichlet, boundary_dofs, conforming_assemble};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;
use fem_solver::{SolverConfig, solve_pcg_gssmoother};

fn main() {
    let mesh0: Mesh<2> = read_mfem_file("data/star-hilbert.mesh")
        .expect("mesh")
        .mesh2d
        .expect("2d");
    let mut nc = NCStateQuad::new();
    let (mesh, _cons, _mm) = nc.refine(&mesh0, &[0, 7, 8, 15, 16], 3);

    let space = H1Space::new(mesh.clone(), 2);
    let quad_rule = 5u8;
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_rule);
    // RHS at t=0 (same as dump_A_true.cpp: bdr/rhs SetTime(0.0))
    let rhs_fn = |pt: &[f64]| {
        let x = pt[0]; let y = pt[1];
        let r = (x * x + y * y).sqrt();
        let a2 = 0.02f64 * 0.02;
        let a4 = a2 * a2;
        let t = 0.0f64;
        -(-0.5 * ((r - t) / 0.02).powi(2)).exp() / a4
            * (-2.0 * t * (x * x + y * y - (2 - 1) as f64 * a2 / 2.0) / r.max(1e-30)
               + x * x + y * y + t * t - 2.0 * a2)
    };
    let source = DomainSourceIntegrator::new(rhs_fn);
    let rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_rule);

    // RAW matrix (before constraints) — for A_raw comparison
    {
        let n = mat.nrows;
        println!("ARAW {} {} {}", n, n, mat.nnz());
        for i in 0..n {
            for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                println!("{} {} {:.17e}", i, mat.col_idx[k], mat.values[k]);
            }
        }
    }

    // P2 constraints (same as ex15 main)
    let dm0 = space.dof_manager();
    let hc = p2_constraints(nc.constraints(), dm0);
    // dump cP (conforming prolongation) for comparison with dump_P.cpp
    {
        let p = fem_space::constraints::build_conforming_prolongation(space.n_dofs(), &hc);
        println!("P {} {}", p.nrows, p.ncols);
        for i in 0..p.nrows {
            print!("PROW {}", i);
            for k in p.row_ptr[i]..p.row_ptr[i + 1] {
                print!(" {}:{:.6}", p.col_idx[k], p.values[k]);
            }
            println!();
        }
    }
    let (mat_true, rhs_true, true_dofs) = conforming_assemble(&mat, &rhs_vec, &hc);
    // dump A_true BEFORE Dirichlet (ConformingAssemble output)
    {
        println!("AcP {} {} {}", mat_true.nrows, mat_true.ncols, mat_true.nnz());
        for i in 0..mat_true.nrows {
            print!("AcPROW {}", i);
            for k in mat_true.row_ptr[i]..mat_true.row_ptr[i + 1] {
                print!(" {}:{:.17e}", mat_true.col_idx[k], mat_true.values[k]);
            }
            println!();
        }
    }
    // dump RA = R·A intermediate for comparison with dump_RA.cpp
    {
        let p = fem_space::constraints::build_conforming_prolongation(space.n_dofs(), &hc);
        let r = p.transpose();
        let ra = fem_linalg::csr_spmm(&r, &mat);
        println!("RA {} {} {}", ra.nrows, ra.ncols, ra.nnz());
        for i in 0..ra.nrows {
            print!("RAROW {}", i);
            for k in ra.row_ptr[i]..ra.row_ptr[i + 1] {
                print!(" {}:{:.17e}", ra.col_idx[k], ra.values[k]);
            }
            println!();
        }
    }

    // Dirichlet on all boundaries, DIAG_KEEP
    let dm = space.dof_manager();
    let bnd_tags = space.mesh().unique_boundary_tags();
    let bnd_all = boundary_dofs(space.mesh(), dm, &bnd_tags);
    let true_set: std::collections::HashSet<usize> = true_dofs.iter().copied().collect();
    let true_idx: std::collections::HashMap<usize, usize> = true_dofs
        .iter().enumerate().map(|(i, &d)| (d, i)).collect();
    let bnd_vals: Vec<f64> = bnd_all.iter()
        .filter(|d| true_set.contains(&(**d as usize)))
        .map(|&dof| {
            let c = dm.dof_coord(dof);
            let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
            (-0.5 * (r / 0.02).powi(2)).exp()
        })
        .collect();
    let bnd: Vec<u32> = bnd_all.iter()
        .filter(|d| true_set.contains(&(**d as usize)))
        .map(|&d| true_idx[&(d as usize)] as u32)
        .collect();
    let (mut mat_elim, mut rhs_elim) = (mat_true, rhs_true);
    apply_dirichlet(&mut mat_elim, &mut rhs_elim, &bnd, &bnd_vals);

    // Solve (C++ dump_u2.cpp: PCG + GSSmoother, rtol 1e-12, max_iter 500)
    let mut x_true = vec![0.0_f64; true_dofs.len()];
    let res = solve_pcg_gssmoother(
        &mat_elim, &rhs_elim, &mut x_true,
        &SolverConfig {
            rtol: 1e-6, max_iter: 500, verbose: false, ..Default::default() // MFEM PCG(RTOL=1e-12) → rel_tol=sqrt
        },
    );
    assert!(res.is_ok(), "PCG failed: {:?}", res.err());

    // Recover full vector + hanging values (C++ RecoverFEMSolution)
    let n_full = space.n_dofs();
    let mut u = vec![0.0_f64; n_full];
    for (&td, &v) in true_dofs.iter().zip(x_true.iter()) {
        u[td] = v;
    }
    if !hc.is_empty() {
        fem_space::constraints::recover_hanging_values(&mut u, &hc);
    }
    // C++ dump_u2 prints the *true-DOF* solution at each true DOF index.
    println!("UTD {}", x_true.len());
    for (i, &v) in x_true.iter().enumerate() {
        println!("{} {:.17e}", i, v);
    }
    println!("UFULL {}", u.len());
    for (i, &v) in u.iter().enumerate() {
        println!("{} {:.17e}", i, v);
    }

    // Output in the same format as dump_A_true.cpp (CSR rows).
    let n = mat_elim.nrows;
    println!("ATRUE {} {} {}", n, n, mat_elim.nnz());
    for i in 0..n {
        for k in mat_elim.row_ptr[i]..mat_elim.row_ptr[i + 1] {
            println!("{} {} {:.17e}", i, mat_elim.col_idx[k], mat_elim.values[k]);
        }
    }
    println!("BTRUE {}", rhs_elim.len());
    for i in 0..rhs_elim.len() {
        println!("{} {:.17e}", i, rhs_elim[i]);
    }
}

fn p2_constraints(
    p1: &[fem_mesh::amr::HangingNodeConstraint],
    dm: &fem_space::dof_manager::DofManager,
) -> Vec<fem_mesh::amr::HangingNodeConstraint> {
    use fem_space::dof_manager::EdgeKey;
    let mut out: Vec<fem_mesh::amr::HangingNodeConstraint> = Vec::new();
    for c in p1 {
        let (mid, a, b) = (c.constrained, c.parent_a, c.parent_b);
        let (mid, a, b) = (c.constrained, c.parent_a, c.parent_b);
        let e = dm.edge_dof_map.get(&EdgeKey::new(a as u32, b as u32)).copied();
        let Some(e) = e else { continue };
        let e = e as usize;
        if mid != e {
            out.push(fem_mesh::amr::HangingNodeConstraint::new_weighted(mid, e, e, 0.5, 0.5, vec![]));
        }
        if let Some(&s1) = dm.edge_dof_map.get(&EdgeKey::new(a as u32, mid as u32)) {
            let s1 = s1 as usize;
            if s1 != mid && s1 != e {
                out.push(fem_mesh::amr::HangingNodeConstraint::new_weighted(
                    s1, a, b, 0.375, -0.125, vec![(e, 0.75)],
                ));
            }
        }
        if let Some(&s2) = dm.edge_dof_map.get(&EdgeKey::new(mid as u32, b as u32)) {
            let s2 = s2 as usize;
            if s2 != mid && s2 != e {
                out.push(fem_mesh::amr::HangingNodeConstraint::new_weighted(
                    s2, a, b, -0.125, 0.375, vec![(e, 0.75)],
                ));
            }
        }
    }
    out
}
