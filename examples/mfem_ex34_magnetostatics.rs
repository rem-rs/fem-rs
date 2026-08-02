//! # Example 34 — Magnetostatics with SubMesh current density  (1:1 with MFEM ex34)
//!
//! Solves `curl curl A = J` where the current density `J = -σ∇φ` is computed on a
//! SubMesh representing the conducting region.  Nédélec (H(curl)) elements for
//! the vector potential, Lagrange (H¹) for the scalar potential, Raviart-Thomas
//! (H(div)) for the current density.
//!
//! Demonstrates SubMesh extraction, multi-physics solve, and field transfer
//! between subdomain and full mesh.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex34_magnetostatics
//! cargo run --example mfem_ex34_magnetostatics -- -m ../data/fichera-mixed.mesh
//! cargo run --example mfem_ex34_magnetostatics -- -o 2 -no-vis
//! cargo run --example mfem_ex34_magnetostatics -- -hex -no-vis
//! ```

use std::fs::File;
use std::io::Write;

use fem_assembly::{
    Assembler, GridFunction, VectorAssembler,
    discrete_op::DiscreteLinearOperator,
    mixed::{
        MixedVectorGradientIntegrator, assemble_h1_hdiv_mixed,
        ref_elem_vec,
    },
    standard::{CurlCurlIntegrator, DiffusionIntegrator, VectorMassIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_linalg::{CsrMatrix, PrintLevel, SolverConfig};
use fem_assembly::geo_ref_elem_from_mesh;
use fem_mesh::{
    Mesh, extract_submesh_3d, SubMesh3D,
    ElementTransformation, ElementType, refine_uniform_3d,
    topology::MeshTopology,
};
use fem_solver::{solve_cg, solve_pcg_gssmoother};
use fem_space::{
    H1Space, HCurlSpace, HDivSpace,
    constraints::{boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv, form_linear_system},
    fe_space::FESpace, SpaceType,
};

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    order: u8,
    ref_levels: i32,
    delta: f64,
    static_cond: bool,
    mixed: bool,
    visualization: bool,
}

fn default_args() -> Args {
    Args {
        mesh_file: "data/fichera-mixed.mesh".into(),
        order: 1,
        ref_levels: 1,
        delta: 1e-6,
        static_cond: false,
        mixed: true,
        visualization: false,
    }
}

fn parse_args() -> Args {
    let mut a = default_args();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or("data/fichera-mixed.mesh".into()),
            "-o" | "--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-r" | "--refine" => a.ref_levels = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-mc" | "--magnetic-cond" => a.delta = it.next().unwrap_or("1e-6".into()).parse().unwrap_or(1e-6),
            "-sc" | "--static-condensation" => a.static_cond = true,
            "-no-sc" | "--no-static-condensation" => a.static_cond = false,
            "-mixed" | "--mixed-mesh" => a.mixed = true,
            "-hex" | "--hex-mesh" => a.mixed = false,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-vis" | "--visualization" => a.visualization = true,
            _ => {}
        }
    }
    a
}

// ─── Hard-coded element selections (matching MFEM ex34.cpp) ────────────────

const SUBELEMS_MIXED: &[u32] = &[0, 2, 3, 4, 9];
const SUBELEMS_HEX: &[u32] = &[10, 14, 34, 36, 37, 38, 39];
const SYM_PLANE_ATTRS: &[i32] = &[9, 10, 11, 12, 13, 14, 15, 16];
const PHI0_ATTR: i32 = 2;
const PHI1_ATTR: i32 = 23;

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    eprintln!("Options used:");
    eprintln!("   --mesh {}", args.mesh_file);
    eprintln!("   --order {}", args.order);
    eprintln!("   --refine {}", args.ref_levels);
    eprintln!("   --magnetic-cond {}", args.delta);
    if args.static_cond { eprintln!("   --static-condensation"); }
    if args.visualization { eprintln!("   --visualization"); }

    // 3. Read the (serial) mesh from the given mesh file.
    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("MFEM mesh must be 3D");

    // 5. Set up SubMesh for the conducting region.
    let max_attr = *mesh.elem_tags.iter().max().unwrap_or(&0);
    let submesh_attr = max_attr + 1;
    let submesh_indices = if args.mixed {
        SUBELEMS_MIXED  // full selection including prism
    } else {
        // For hex mesh, reload fichera.mesh, refine once, then use hex indices.
        let mfem2 = read_mfem_file("data/fichera.mesh").expect("failed to read fichera.mesh");
        mesh = mfem2.mesh3d.expect("fichera.mesh must be 3D");
        mesh = refine_uniform_3d(&mesh);
        SUBELEMS_HEX
    };
    for &ei in submesh_indices {
        mesh.elem_tags[ei as usize] = submesh_attr;
    }

    // Further refinements.
    let ref_levels = args.ref_levels;
    for _ in 0..ref_levels {
        mesh = refine_uniform_3d(&mesh);
    }

    let cond_attr = [submesh_attr];
    let submesh: SubMesh3D = extract_submesh_3d(&mesh, &cond_attr);
    let mesh_cond = submesh.mesh.clone();
    eprintln!("  SubMesh: {} elements, {} nodes, {} boundary faces",
        mesh_cond.n_elems(), mesh_cond.n_nodes(), mesh_cond.n_faces());

    // 6. Define finite element spaces on the SubMesh.
    let order = args.order;
    let rt_order = if order > 0 { order - 1 } else { 0 };

    let fec_h1 = H1Space::new(mesh_cond.clone(), order);
    let fec_rt = HDivSpace::new(mesh_cond.clone(), rt_order);
    eprintln!("  SubMesh H1 DOFs: {}, RT DOFs: {}",
        fec_h1.n_dofs(), fec_rt.n_dofs());
    let n_h1 = fec_h1.n_dofs();

    // ── 6a. Solve for φ: -∇·(σ∇φ) = 0 on SubMesh ──────────────────────────
    let sigma_coef = 1.0;
    let quad_order = 2 * order + 1;

    let h1_stiffness = Assembler::assemble_bilinear(
        &fec_h1, &[&DiffusionIntegrator { kappa: sigma_coef }], quad_order,
    );

    // ProjectBdrCoefficient: φ = 0 on phi0_attr (2), φ = 1 on phi1_attr (23),
    // matching MFEM ex34.cpp (phi_h1.ProjectBdrCoefficient(zero/one, bdr)).
    let dm_h1 = fec_h1.dof_manager();
    let mut phi = vec![0.0_f64; n_h1];
    {
        let mut gf_phi = GridFunction::new(&fec_h1, phi.clone());
        gf_phi.project_bdr_coefficient(&|_| 0.0, &[PHI0_ATTR], dm_h1);
        gf_phi.project_bdr_coefficient(&|_| 1.0, &[PHI1_ATTR], dm_h1);
        phi = gf_phi.dofs().to_vec();
    }

    // Essential BCs: phi0 + phi1 boundary attributes only — MFEM ex34.cpp
    // `ComputeCurrentDensityOnSubMesh` sets `ess_bdr_phi` from phi0_attr (2)
    // and phi1_attr (23) only (submesh-internal faces attr 25 are NOT
    // essential, so φ has an interior solution there, driven by the
    // Dirichlet values on attrs 2/23).
    let ess_dofs_phi = boundary_dofs(&mesh_cond, dm_h1, &[PHI0_ATTR, PHI1_ATTR]);
    let ess_vals_phi: Vec<f64> = ess_dofs_phi.iter().map(|&d| phi[d as usize]).collect();

    // FormLinearSystem (MFEM DIAG_KEEP) + PCG-GSSmoother (print 1, 200, 1e-12).
    let mut mat_h1 = h1_stiffness.clone();
    let mut B_h1 = vec![0.0_f64; n_h1];
    let mut x_h1 = phi.clone();
    form_linear_system(&mut mat_h1, &mut B_h1, &mut x_h1, &ess_dofs_phi, &ess_vals_phi);

    eprintln!("\nSolving for electric potential using PCG with a Gauss-Seidel preconditioner");
    let h1_cfg = SolverConfig {
        rtol: 1e-12, atol: 0.0, max_iter: 200,
        verbose: true, print_level: PrintLevel::Iterations,
    };
    solve_pcg_gssmoother(&mat_h1, &B_h1, &mut x_h1, &h1_cfg).expect("H1 PCG solve failed");
    phi = x_h1;

    // ── 6b. Solve for J = -σ∇φ in H(div) on SubMesh ───────────────────────
    let rt_mass = VectorAssembler::assemble_bilinear(
        &fec_rt, &[&VectorMassIntegrator { alpha: 1.0 }], quad_order,
    );

    let grad_matrix = assemble_h1_hdiv_mixed(
        &fec_h1, &fec_rt,
        &[&MixedVectorGradientIntegrator { sigma: sigma_coef }],
        quad_order,
    );

    let n_rt = fec_rt.n_dofs();
    let mut b_rt = vec![0.0_f64; n_rt];
    // G = assemble_h1_hdiv_mixed: rows=H¹, cols=HDiv
    // b_rt[i] += -Σ_j G[j,i] * phi[j]   (i = HDiv DOF, j = H¹ DOF)
    for h1_row in 0..n_h1 {
        let start = grad_matrix.row_ptr[h1_row];
        let end = grad_matrix.row_ptr[h1_row + 1];
        for j in start..end {
            let hdiv_col = grad_matrix.col_idx[j] as usize;
            b_rt[hdiv_col] -= grad_matrix.values[j] * phi[h1_row];
        }
    }

    // J·n = 0 on walls (symmetry planes + attr 25).
    let mut jn_attrs: Vec<i32> = vec![25];
    jn_attrs.extend_from_slice(SYM_PLANE_ATTRS);
    let ess_dofs_rt = boundary_dofs_hdiv(fec_rt.mesh(), &fec_rt, &jn_attrs);

    eprintln!("\nSolving for current density in H(Div) using diagonally scaled CG");
    eprintln!("Size of linear system: {}", n_rt);

    // FormLinearSystem (DIAG_KEEP) + un-preconditioned CGSolver (rel_tol
    // 1e-12, 2000 iters — matches MFEM ex34.cpp's `CGSolver cg`).
    let mut mat_rt = rt_mass.clone();
    let mut B_rt = b_rt.clone();
    let mut j_cond = vec![0.0_f64; n_rt];
    let mut x_rt = j_cond.clone();
    form_linear_system(&mut mat_rt, &mut B_rt, &mut x_rt, &ess_dofs_rt, &vec![0.0; ess_dofs_rt.len()]);
    let rt_cfg = SolverConfig {
        rtol: 1e-12, atol: 0.0, max_iter: 2000,
        verbose: true, print_level: PrintLevel::Iterations,
    };
    solve_cg(&mat_rt, &B_rt, &mut x_rt, &rt_cfg).expect("RT CG solve failed");
    j_cond = x_rt;

    // ── 6c. Save SubMesh and current density ───────────────────────────────
    {
        let mut f = File::create("cond.mesh").expect("cond.mesh");
        let dummy2d = Mesh::<2>::unit_square_tri(1);
        write_mfem(&mut f, &dummy2d, Some(&mesh_cond)).expect("write cond.mesh");
    }
    {
        let mut f = File::create("cond_j.gf").expect("cond_j.gf");
        fem_io::mfem::write_gf(&mut f, 2, &j_cond, "H1", 1, 1).expect("write cond_j.gf");
    }

    // 7. Transfer J from SubMesh RT to full mesh RT.
    let fec_rt_full = HDivSpace::new(mesh.clone(), rt_order);
    let fec_nd_full = HCurlSpace::new(mesh.clone(), order);

    let sub_elem_dofs = |e: u32| fec_rt.element_dofs(e).to_vec();
    let par_elem_dofs = |e: u32| fec_rt_full.element_dofs(e).to_vec();
    let sub_elem_signs = |e: u32| fec_rt.element_signs(e).to_vec();
    let par_elem_signs = |e: u32| fec_rt_full.element_signs(e).to_vec();
    let j_full = submesh.transfer_dofs_to_parent(&j_cond, fec_rt_full.n_dofs(), &sub_elem_dofs, &par_elem_dofs, &sub_elem_signs, &par_elem_signs);

    // 9–12. HCurl system: curl curl + delta·I.
    let n_nd = fec_nd_full.n_dofs();
    eprintln!("\nFull mesh ND DOFs: {}", n_nd);

    let nd_stiffness = VectorAssembler::assemble_bilinear(
        &fec_nd_full,
        &[
            &CurlCurlIntegrator { mu: 1.0 },
            &VectorMassIntegrator { alpha: args.delta },
        ],
        quad_order,
    );

    // RHS from J (VectorFEDomainLFIntegrator(jCoef) in MFEM).
    let nd_rhs = assemble_hcurl_rhs(&fec_nd_full, &fec_rt_full, &j_full, quad_order);

    // 8. Essential BC: PEC on all boundaries except symmetry planes.
    let nd_mesh = fec_nd_full.mesh();
    let all_tags: Vec<i32> = nd_mesh.unique_boundary_tags();
    let pec_tags: Vec<i32> = all_tags.into_iter()
        .filter(|t| !SYM_PLANE_ATTRS.contains(t))
        .collect();
    let ess_dofs_nd = if pec_tags.is_empty() { vec![] }
        else { boundary_dofs_hcurl(nd_mesh, &fec_nd_full, &pec_tags) };

    // 13. FormLinearSystem (DIAG_KEEP) + PCG-GSSmoother (print 1, 500, 1e-12)
    //     matching MFEM ex34.cpp (non-PA path).
    let mut mat_nd = nd_stiffness.clone();
    let mut B_nd = nd_rhs.clone();
    let mut a_sol = vec![0.0_f64; n_nd];
    let mut x_nd = a_sol.clone();
    form_linear_system(&mut mat_nd, &mut B_nd, &mut x_nd, &ess_dofs_nd, &vec![0.0; ess_dofs_nd.len()]);

    eprintln!("\nSolving for magnetic vector potential using CG with a Gauss-Seidel preconditioner");
    eprintln!("Size of linear system: {}", n_nd);

    let nd_cfg = SolverConfig {
        rtol: 1e-12, atol: 0.0, max_iter: 500,
        verbose: true, print_level: PrintLevel::Iterations,
    };
    solve_pcg_gssmoother(&mat_nd, &B_nd, &mut x_nd, &nd_cfg).expect("ND PCG solve failed");
    a_sol = x_nd;

    // 17. Compute B = curl A.
    let fec_rt_curl = HDivSpace::new(mesh.clone(), rt_order);
    let curl_mat = DiscreteLinearOperator::curl_3d(&fec_nd_full, &fec_rt_curl)
        .expect("CurlInterpolator assembly failed");
    let mut b_field = vec![0.0_f64; fec_rt_curl.n_dofs()];
    curl_mat.spmv(&a_sol, &mut b_field);

    // 15. Save output.
    {
        let mut f = File::create("refined.mesh").expect("refined.mesh");
        let d2 = Mesh::<2>::unit_square_tri(1);
        write_mfem(&mut f, &d2, Some(&mesh)).expect("write refined.mesh");
    }
    {
        let mut f = File::create("sol.gf").expect("sol.gf");
        fem_io::mfem::write_gf(&mut f, 2, &a_sol, "ND", 1, 1).expect("write sol.gf");
    }
    {
        let mut f = File::create("dsol.gf").expect("dsol.gf");
        fem_io::mfem::write_gf(&mut f, 2, &b_field, "H1", 1, 1).expect("write dsol.gf");
    }

    eprintln!("\nFinished.");
}

// ─── Helper: eliminate Dirichlet BC (like MFEM FormLinearSystem) ───────────

fn eliminate_bc(
    mat: &CsrMatrix<f64>,
    rhs: &mut [f64],
    constrained_dofs: &[u32],
    values: &[f64],
    sol: &mut [f64],
) -> (CsrMatrix<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
    for (&d, &v) in constrained_dofs.iter().zip(values.iter()) {
        sol[d as usize] = v;
    }
    fem_space::constraints::eliminate_dirichlet(mat, rhs, constrained_dofs, values)
}

fn expand_solution(
    x: &[f64],
    free_map: &[usize],
    constrained_map: &[usize],
    values: &[f64],
    sol: &mut [f64],
) {
    let n = sol.len();
    let expanded = fem_space::constraints::expand_from_reduced(x, free_map, constrained_map, values, n);
    sol.copy_from_slice(&expanded);
}

// ─── Helper: assemble HCurl RHS from J field DOFs ─────────────────────────

fn assemble_hcurl_rhs(
    nd_space: &HCurlSpace<Mesh<3>>,
    rt_space: &HDivSpace<Mesh<3>>,
    j_dofs: &[f64],
    quad_order: u8,
) -> Vec<f64> {
    use fem_assembly::isoparametric_jacobian;
    let mesh = nd_space.mesh();
    let dim = 3;
    let n_dofs = nd_space.n_dofs();
    let mut rhs = vec![0.0_f64; n_dofs];

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let nd_ref = ref_elem_vec(elem_type, nd_space.order(), SpaceType::HCurl).unwrap();
        let rt_ref = ref_elem_vec(elem_type, rt_space.order(), SpaceType::HDiv).unwrap();
        let quad = nd_ref.quadrature(quad_order);

        let nd_dofs: Vec<u32> = nd_space.element_dofs(e).iter().copied().collect();
        let rt_dofs: Vec<u32> = rt_space.element_dofs(e).iter().copied().collect();
        let nd_s = nd_space.element_signs(e);
        let rt_s = rt_space.element_signs(e);
        let n_nd = nd_dofs.len();
        let n_rt = rt_dofs.len();

        let nodes = mesh.element_nodes(e);
        let use_iso = !matches!(elem_type, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
        let geo_elem = if use_iso { geo_ref_elem_from_mesh(mesh, e) } else { None };

        let mut nd_phi = vec![0.0; n_nd * dim];
        let mut rt_phi = vec![0.0; n_rt * dim];
        let mut nd_phys = vec![0.0; n_nd * dim];
        let mut rt_phys = vec![0.0; n_rt * dim];
        let mut f_elem = vec![0.0; n_nd];

        for (q, xi) in quad.points.iter().enumerate() {
            let (jit, jac, det_j) = if use_iso {
                let ge = geo_elem.as_ref().expect("geo_ref_elem");
                let (jac, dj, _x) = isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, dim);
                let jit = jac.clone().try_inverse().expect("invertible Jacobian").transpose();
                (jit, jac, dj)
            } else {
                let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                (tr.jacobian_inv_t().clone(), tr.jacobian().clone(), tr.det_j())
            };
            let w = quad.weights[q] * det_j.abs();

            nd_ref.eval_basis_vec(xi, &mut nd_phi);
            rt_ref.eval_basis_vec(xi, &mut rt_phi);

            // Piola transforms + orientation signs (MFEM VectorFEDomainLFIntegrator):
            //   ND:  w_i = sign_i · J^{-T} ŵ_i
            //   RT:  φ_j = sign_j · (1/det J) J φ̂_j
            for i in 0..n_nd {
                let s = nd_s.get(i).copied().unwrap_or(1.0);
                for r in 0..dim {
                    let mut acc = 0.0;
                    for c in 0..dim { acc += jit[(r, c)] * nd_phi[i * dim + c]; }
                    nd_phys[i * dim + r] = s * acc;
                }
            }
            for j in 0..n_rt {
                let s = rt_s.get(j).copied().unwrap_or(1.0);
                for r in 0..dim {
                    let mut acc = 0.0;
                    for c in 0..dim { acc += jac[(r, c)] * rt_phi[j * dim + c]; }
                    rt_phys[j * dim + r] = s * acc / det_j;
                }
            }

            let mut j_at_q = [0.0_f64; 3];
            for j in 0..n_rt {
                let jv = j_dofs[rt_dofs[j] as usize];
                for c in 0..3 { j_at_q[c] += jv * rt_phys[j * 3 + c]; }
            }
            for i in 0..n_nd {
                let dot = (0..3).map(|c| nd_phys[i * 3 + c] * j_at_q[c]).sum::<f64>();
                f_elem[i] += w * dot;
            }
        }
        for (li, &gi) in nd_dofs.iter().enumerate() {
            rhs[gi as usize] += f_elem[li];
        }
    }
    rhs
}
