//! Dump Rust it3 (140-elem mesh) A_true + solution for 1:1 comparison with
//! tools_ex15_ref/dump_A_true_it3.cpp and dump_u3_coords.cpp.
//! Usage: cargo run --release -p fem-examples --example mfem_ex15_dump_p1_it3

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_mesh::amr::{NCStateQuad, NcState2D, HangingNodeConstraint};
use fem_space::constraints::{apply_dirichlet, boundary_dofs, conforming_assemble};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::dof_manager::{DofManager, EdgeKey};

fn p2_constraints(
    p1: &[HangingNodeConstraint],
    dm: &DofManager,
) -> Vec<HangingNodeConstraint> {
    let mut out: Vec<HangingNodeConstraint> = Vec::new();
    let v2d = &dm.phys_to_vertex_dof;
    for c in p1 {
        let (mid_p, a_p, b_p) = (c.constrained as u32, c.parent_a as u32, c.parent_b as u32);
        let (mid, a, b) = (v2d[&mid_p] as usize, v2d[&a_p] as usize, v2d[&b_p] as usize);
        let e = dm.edge_dof_map.get(&EdgeKey::new(a_p, b_p)).copied();
        let Some(e) = e else { continue };
        let e = e as usize;
        if mid != e {
            out.push(HangingNodeConstraint::new_weighted(mid, e, e, 0.5, 0.5, vec![]));
        }
        if let Some(&s1) = dm.edge_dof_map.get(&EdgeKey::new(a_p, mid_p)) {
            let s1 = s1 as usize;
            if s1 != mid && s1 != e {
                out.push(HangingNodeConstraint::new_weighted(
                    s1, a, b, 0.375, -0.125, vec![(e, 0.75)],
                ));
            }
        }
        if let Some(&s2) = dm.edge_dof_map.get(&EdgeKey::new(mid_p, b_p)) {
            let s2 = s2 as usize;
            if s2 != mid && s2 != e {
                out.push(HangingNodeConstraint::new_weighted(
                    s2, a, b, -0.125, 0.375, vec![(e, 0.75)],
                ));
            }
        }
    }
    out
}

fn bdr_func(pt: &[f64], _t: f64) -> f64 {
    let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
    (-0.5 * (r / 0.02).powi(2)).exp()
}
fn rhs_func(pt: &[f64], t: f64) -> f64 {
    let x = pt[0]; let y = pt[1];
    let r = (x * x + y * y).sqrt();
    let a2 = 0.02f64 * 0.02;
    let a4 = a2 * a2;
    -(-0.5 * ((r - t) / 0.02).powi(2)).exp() / a4
        * (-2.0 * t * (x * x + y * y - (2 - 1) as f64 * a2 / 2.0) / r.max(1e-30)
           + x * x + y * y + t * t - 2.0 * a2)
}

fn main() {
    let mesh0: Mesh<2> = read_mfem_file("data/star-hilbert.mesh")
        .expect("mesh").mesh2d.expect("2d");
    let mut nc = NCStateQuad::new();
    let (m1, _, _) = nc.refine(&mesh0, &[0, 7, 8, 15, 16], 3);
    let all: Vec<u32> = (0..m1.n_elems() as u32).collect();
    let (mesh, _, _) = nc.refine(&m1, &all, 3);
    println!("it3 mesh: {} nodes {} elems", mesh.n_nodes(), mesh.n_elems());

    let space = H1Space::new(mesh.clone(), 2);
    let cdofs = space.n_dofs();
    let quad_rule = 5u8;
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_rule);
    let source = DomainSourceIntegrator::new(|pt: &[f64]| rhs_func(pt, 0.0));
    let rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_rule);
    let dm0 = space.dof_manager();
    let hc = p2_constraints(nc.constraints(), dm0);
    let (mat_true, rhs_true, true_dofs) = conforming_assemble(&mat, &rhs_vec, &hc);

    // Dirichlet (same as ex15)
    let dm = space.dof_manager();
    let bnd_tags = space.mesh().unique_boundary_tags();
    let bnd_all = boundary_dofs(space.mesh(), dm, &bnd_tags);
    let true_set: std::collections::HashSet<usize> = true_dofs.iter().copied().collect();
    let true_idx: std::collections::HashMap<usize, usize> = true_dofs
        .iter().enumerate().map(|(i, &d)| (d, i)).collect();
    let mut mat_true = mat_true;
    let mut rhs_true = rhs_true;
    let bnd_vals: Vec<f64> = bnd_all
        .iter().filter(|d| true_set.contains(&(**d as usize)))
        .map(|&dof| bdr_func(&dm.dof_coord(dof), 0.0)).collect();
    let bnd: Vec<u32> = bnd_all
        .iter().filter(|d| true_set.contains(&(**d as usize)))
        .map(|&d| true_idx[&(d as usize)] as u32).collect();
    apply_dirichlet(&mut mat_true, &mut rhs_true, &bnd, &bnd_vals);

    // A_true (after FormLinearSystem equivalent: Dirichlet applied)
    println!("ATRUE {} {} {}", mat_true.nrows, mat_true.ncols, mat_true.nnz());
    for i in 0..mat_true.nrows {
        for k in mat_true.row_ptr[i]..mat_true.row_ptr[i + 1] {
            println!("{} {} {:.17e}", i, mat_true.col_idx[k], mat_true.values[k]);
        }
    }

    // Solve
    let mut u = vec![0.0_f64; cdofs];
    let mut x_true = vec![0.0_f64; true_dofs.len()];
    let res = solve_pcg_gssmoother(
        &mat_true, &rhs_true, &mut x_true,
        &SolverConfig { rtol: 1e-6, max_iter: 500, verbose: false, ..Default::default() },
    );
    let _ = res;
    for (&td, &v) in true_dofs.iter().zip(x_true.iter()) {
        u[td] = v;
    }
    if !hc.is_empty() {
        // recover hanging values (same as ex15 recover_hanging_values)
        for c in &hc {
            let v = c.coeff_a * u[c.parent_a] + c.coeff_b * u[c.parent_b]
                + c.extra.iter().map(|&(m, w)| w * u[m]).sum::<f64>();
            u[c.constrained] = v;
        }
    }
    println!("SOLU");
    for d in 0..cdofs {
        let c = dm.dof_coord(d as u32);
        println!("{} {:.10} {:.10} {:.17e}", d, c[0], c[1], u[d]);
    }
}
