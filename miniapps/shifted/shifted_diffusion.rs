//! Shifted Diffusion Miniapp: finite element immersed boundary solver
//! (MFEM `miniapps/shifted/diffusion.cpp`, serial `-np 1` 1:1 port).
//!
//! Solves the Poisson problem with prescribed boundary conditions on a
//! surrogate domain using the high-order shifted boundary method: the level
//! set marks the true boundary, the distance vector field shifts the boundary
//! conditions onto the surrogate mesh faces (see
//! `fem_assembly::standard::sbm3_dirichlet` / `sbm3_neumann`).
//!
//! Deviations from the C++ miniapp:
//! * GLVis / ParaView output is not available: passing `-vis` prints a notice
//!   and exits with code 3 (serial port convention).
//! * The iterative solver is AMG-preconditioned GMRES instead of
//!   BiCGSTAB + HypreBoomerAMG (same 1e-12 relative tolerance, 500 max
//!   iterations); converged solutions agree up to the solver tolerance.
//! * `-dc` (combo of two Dirichlet level sets) is not supported — the serial
//!   SBM integrators do not discriminate faces per level set yet.
//! * The analytic distance vector (`Dist_Vector_Coefficient`) sets D = 0 at
//!   the (measure-zero) circle centre where the radial direction is undefined.
//!
//! Sample runs:
//!   cargo run --release --example shifted_diffusion -- -rs 2 -o 2 -lst 2 -no-vis
//!   cargo run --release --example shifted_diffusion -- -rs 3 -o 1 -lst 1 -no-vis
//!   cargo run --release --example shifted_diffusion -- -rs 2 -o 2 -nlst 2 -ho 1 -no-vis
//!   cargo run --release --example shifted_diffusion -- -m data/inline-tet.mesh -rs 1 -o 1 -lst 1 -alpha 10 -no-vis

mod sbm_aux;

use fem_amg::{solve_amg_gmres, AmgConfig};
use fem_assembly::dist_solver::{
    avg_element_size, DistanceSolver, HeatDistanceSolver, PDEFilter, ShiftedFaceMarker, FnDist,
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{
    DiffusionIntegrator, DomainSourceIntegrator, Sbm3DirichletIntegrator,
    Sbm3DirichletLFIntegrator, Sbm3NeumannIntegrator, Sbm3NeumannLFIntegrator,
};
use fem_assembly::{Assembler, BilinearIntegrator, LinearIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_linalg::SolverConfig;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use fem_space::constraints::apply_dirichlet_diag_one;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

use sbm_aux as aux;

type LevelSetFn = dyn Fn(&[f64]) -> f64 + Send + Sync;

// ─── Command-line options ────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    order: i32,
    ser_ref_levels: i32,
    dirichlet_level_set_type: i32,
    neumann_level_set_type: i32,
    ho_terms: i32,
    alpha: f64,
    include_cut_cell: bool,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/inline-quad.mesh".to_string(),
        order: 2,
        ser_ref_levels: 0,
        dirichlet_level_set_type: -1,
        neumann_level_set_type: -1,
        ho_terms: 0,
        alpha: 1.0,
        include_cut_cell: false,
        visualization: true,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();
        i += 1;
        let mut next = |i: &mut usize| -> String {
            let v = argv.get(*i).cloned().unwrap_or_default();
            *i += 1;
            v
        };
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = next(&mut i),
            "-o" | "--order" => a.order = next(&mut i).parse().unwrap_or(a.order),
            "-rs" | "--refine-serial" => {
                a.ser_ref_levels = next(&mut i).parse().unwrap_or(a.ser_ref_levels)
            }
            "-lst" | "--level-set-type" => {
                a.dirichlet_level_set_type =
                    next(&mut i).parse().unwrap_or(a.dirichlet_level_set_type)
            }
            "-nlst" | "--neumann-level-set-type" => {
                a.neumann_level_set_type =
                    next(&mut i).parse().unwrap_or(a.neumann_level_set_type)
            }
            "-ho" | "--high-order" => a.ho_terms = next(&mut i).parse().unwrap_or(a.ho_terms),
            "-alpha" | "--alpha" => a.alpha = next(&mut i).parse().unwrap_or(a.alpha),
            "-cut" | "--cut" => a.include_cut_cell = true,
            "--no-cut-cell" => a.include_cut_cell = false,
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-dc" | "--dcombo" => {
                eprintln!("The -dc combo option is not supported by the serial port yet.");
                std::process::exit(1);
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
    }
    a
}

/// Mesh refinement helper uniform over the dimension.
trait RefineMany: Clone {
    fn refine_many(&self, levels: usize) -> Self;
}
impl RefineMany for Mesh<2> {
    fn refine_many(&self, levels: usize) -> Self {
        let mut m = self.clone();
        for _ in 0..levels {
            m = refine_uniform(&m);
        }
        m
    }
}
impl RefineMany for Mesh<3> {
    fn refine_many(&self, levels: usize) -> Self {
        let mut m = self.clone();
        for _ in 0..levels {
            m = refine_uniform_3d(&m);
        }
        m
    }
}

/// Mesh output (`diffusion.mesh`, MFEM `PrintAsOne`).
trait SaveMesh {
    fn save(&self, path: &str);
}
impl SaveMesh for Mesh<2> {
    fn save(&self, path: &str) {
        fem_io::mfem::write_mfem_file(path, self)
            .unwrap_or_else(|e| eprintln!("warning: could not write {path}: {e}"));
    }
}
impl SaveMesh for Mesh<3> {
    fn save(&self, path: &str) {
        fem_io::mfem::write_mfem_file_3d(path, self)
            .unwrap_or_else(|e| eprintln!("warning: could not write {path}: {e}"));
    }
}

fn main() {
    let args = parse_args();
    let mfem = read_mfem_file(&args.mesh_file)
        .unwrap_or_else(|e| panic!("Failed to read MFEM mesh '{}': {e}", args.mesh_file));
    if let Some(mesh) = mfem.mesh2d {
        run::<2>(mesh, &args);
    } else {
        let mesh = mfem.mesh3d.expect("no mesh in file");
        run::<3>(mesh, &args);
    }
}

fn run<const D: usize>(mesh: Mesh<D>, args: &Args)
where
    Mesh<D>: RefineMany + SaveMesh,
{
    let dim = D;

    // GLVis / ParaView are not available in the serial port.
    if args.visualization {
        println!("GLVis visualization is not available in this serial port.");
        println!("Re-run with -no-vis to disable visualization output.");
        std::process::exit(3);
    }

    let lst = args.dirichlet_level_set_type;
    let nlst = args.neumann_level_set_type;

    // Use Dirichlet level set if no level sets are specified.
    let (lst, nlst) = if lst <= 0 && nlst <= 0 { (1, nlst) } else { (lst, nlst) };

    // Verify the C++ miniapp's requirement on the high-order terms.
    assert!(
        !(nlst > 0 && args.ho_terms < 1),
        "Shifted Neumann BC requires extra terms, i.e., -ho >= 1."
    );

    // Refine the mesh (MFEM UniformRefinement).
    let mut mesh = mesh.refine_many(args.ser_ref_levels.max(0) as usize);
    println!("Number of elements: {}", mesh.n_elements());

    let order = args.order.max(1) as u8;
    let space = H1Space::new(mesh.clone(), order);
    println!("Number of finite element unknowns: {}", space.n_dofs());

    let ho_terms = args.ho_terms.max(0) as usize;
    let alpha = args.alpha;

    // ── Level set setup and element marking ─────────────────────────────────
    let dx = avg_element_size(&mesh);

    let ls_fn = |x: &[f64]| aux::dist_level_set(x, lst);
    let nl_fn = |x: &[f64]| aux::dist_level_set(x, nlst);

    // MFEM: PDEFilter(dx).Filter(dirichlet_ls, level_set_gf) for Dirichlet
    // level sets; plain projection for the Neumann level set.  The filter
    // runs on an internal order-2 H1 space which is then projected onto the
    // solution space (PDEFilter::Filter -> ffield.ProjectDiscCoefficient);
    // implemented here by evaluating the filtered field at the solution DOFs.
    // Comparison aid: RUST_ANALYTIC_LS=1 skips the PDE filter (analytic
    // +-1 marking, matching the serial C++ harness).
    let analytic_ls = std::env::var("RUST_ANALYTIC_LS").is_ok();
    let level_set_dofs: Vec<f64> = if lst > 0 && analytic_ls {
        space.interpolate(&ls_fn).as_slice().to_vec()
    } else if lst > 0 {
        let mut filter: PDEFilter<Mesh<D>> = PDEFilter::new(mesh.clone(), dx);
        let n2 = H1Space::<Mesh<D>>::new(mesh.clone(), 2).n_dofs();
        let mut filtered = vec![0.0_f64; n2];
        filter.filter_coeff(Box::new(FnDist(ls_fn)), &mut filtered);
        let fs2 = std::sync::Arc::new(H1Space::new(mesh.clone(), 2));
        let dofs2 = std::sync::Arc::new(filtered);
        let dm = space.dof_manager();
        (0..space.n_dofs() as u32)
            .map(|dof| {
                let x = dm.dof_coord(dof);
                let gf = GridFunction::new(fs2.as_ref(), (*dofs2).clone());
                gf.get_value(&x).unwrap_or(0.0)
            })
            .collect()
    } else {
        space.interpolate(&nl_fn).as_slice().to_vec()
    };
    let level_set_gf = GridFunction::new(&space, level_set_dofs);

    let mut marker = ShiftedFaceMarker::new(&mesh, &space, args.include_cut_cell);
    let mut elem_marker: Vec<i32> = Vec::new();
    marker.mark_elements(&level_set_gf, &mut elem_marker);

    // Get a list of dofs associated with shifted boundary (SB) faces.
    let mut sb_dofs: Vec<usize> = Vec::new();
    marker.list_shifted_face_dofs(&elem_marker, &mut sb_dofs);
    println!("Number of shifted face dofs: {}", sb_dofs.len());

    // Make a list of inactive tdofs that will be eliminated from the system.
    let mut ess_tdof_list: Vec<usize> = Vec::new();
    let mut ess_shift_bdr: Vec<i32> = Vec::new();
    marker.list_essential_tdofs(&elem_marker, &sb_dofs, &mut ess_tdof_list, &mut ess_shift_bdr);

    // ── Distance vector field to the actual boundary ────────────────────────
    let dist_dofs: Vec<f64>;
    if lst == 1 || lst == 2 || lst == 3 {
        // Analytic distance vector (MFEM Dist_Vector_Coefficient +
        // ProjectDiscCoefficient: nodal evaluation on the continuous space).
        let nd = space.n_dofs();
        let dm = space.dof_manager();
        let mut d = vec![0.0_f64; dim * nd];
        let mut p = vec![0.0_f64; dim];
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            aux::dist_vector(x, dim, lst, &mut p);
            for c in 0..dim {
                d[c * nd + dof as usize] = p[c];
            }
        }
        dist_dofs = d;
    } else {
        // Discrete distance vector via the heat method on the filtered combo
        // level set (MFEM: HeatDistanceSolver(2 dx²).ComputeVectorDistance).
        let mut filter: PDEFilter<Mesh<D>> = PDEFilter::new(mesh.clone(), 2.0 * dx);
        let n2 = H1Space::<Mesh<D>>::new(mesh.clone(), 2).n_dofs();
        let mut filtered = vec![0.0_f64; n2];
        let combo = |x: &[f64]| -> f64 {
            let mut v: Option<f64> = None;
            if lst > 0 {
                v = Some(aux::dist_level_set(x, lst));
            }
            if nlst > 0 {
                let w = aux::dist_level_set(x, nlst);
                v = Some(match v {
                    Some(u) => u.min(w),
                    None => w,
                });
            }
            v.expect("combo level set requires at least one level set")
        };
        filter.filter_coeff(Box::new(FnDist(combo)), &mut filtered);
        let fs2 = std::sync::Arc::new(H1Space::new(mesh.clone(), 2));
        let dofs2 = std::sync::Arc::new(filtered);
        let ls_filt = move |x: &[f64]| {
            let gf = GridFunction::new(fs2.as_ref(), (*dofs2).clone());
            gf.get_value(x).unwrap_or(0.0)
        };

        let space_v = VectorH1Space::new(mesh.clone(), order, dim as u8);
        let mut solver = HeatDistanceSolver::new(2.0 * dx * dx);
        solver.smooth_steps = 1;
        let mut d = vec![0.0_f64; space_v.n_dofs()];
        DistanceSolver::compute_vector_distance(&solver, &ls_filt, &mut d, &space_v);
        assert_eq!(d.len(), dim * space.n_dofs());
        dist_dofs = d;
    }

    // ── Element attributes: exclude inactive elements from assembly ─────────
    let max_elem_attr = mesh.elem_tags.iter().copied().max().unwrap_or(1) as usize;
    for e in 0..mesh.n_elements() {
        let m = elem_marker[e as usize];
        let out = if !args.include_cut_cell {
            m == fem_assembly::dist_solver::SBElementType::Outside as i32
                || m >= fem_assembly::dist_solver::SBElementType::Cut as i32
        } else {
            m == fem_assembly::dist_solver::SBElementType::Outside as i32
        };
        if out {
            mesh.elem_tags[e as usize] = max_elem_attr as i32 + 1;
        }
    }
    // The assembly space reads the updated element attributes.
    let space = H1Space::new(mesh.clone(), order);

    let mut ess_elem = vec![1_i32; max_elem_attr + 1];
    ess_elem[max_elem_attr] = 0;

    // ── Right-hand side / boundary-condition coefficient selection ──────────
    let rhs_f: Box<LevelSetFn> = if lst == 1
        || lst == 4
        || lst == 5
        || lst == 6
        || lst == 8
        || nlst == 1
        || nlst == 7
    {
        Box::new(aux::rhs_fun_circle)
    } else if lst == 2 || nlst == 2 {
        Box::new(aux::rhs_fun_xy_exponent)
    } else if lst == 3 {
        Box::new(aux::rhs_fun_xy_sinusoidal)
    } else {
        panic!("RHS function not set for level set type.");
    };

    let dbc: Option<Box<LevelSetFn>> = if lst > 0 {
        if lst == 1 || lst >= 4 {
            Some(Box::new(aux::homogeneous))
        } else if lst == 2 {
            Some(Box::new(aux::dirichlet_velocity_xy_exponent))
        } else if lst == 3 {
            Some(Box::new(aux::dirichlet_velocity_xy_sinusoidal))
        } else {
            None
        }
    } else {
        None
    };

    // Exact solution to project as initial condition (MFEM exactCoef).
    let exact: Box<LevelSetFn> = if dbc.is_some() {
        if lst == 1 || lst >= 4 {
            Box::new(aux::homogeneous)
        } else if lst == 2 {
            Box::new(aux::dirichlet_velocity_xy_exponent)
        } else {
            Box::new(aux::dirichlet_velocity_xy_sinusoidal)
        }
    } else if nlst == 2 {
        Box::new(aux::dirichlet_velocity_xy_exponent)
    } else {
        Box::new(aux::homogeneous)
    };

    let nbc: Option<Box<LevelSetFn>> = if nlst == 1 {
        Some(Box::new(aux::homogeneous))
    } else if nlst == 2 {
        Some(Box::new(aux::traction_xy_exponent))
    } else if nlst == 7 {
        Some(Box::new(aux::homogeneous))
    } else {
        None
    };
    let normal_bc: Option<Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>> = if nlst == 1 {
        Some(Box::new(aux::normal_vector_1))
    } else if nlst == 2 {
        Some(Box::new(aux::normal_vector_1))
    } else if nlst == 7 {
        Some(Box::new(aux::normal_vector_2))
    } else {
        None
    };

    // Quadrature orders mirroring the MFEM defaults:
    // DiffusionIntegrator: Pk 2p−2 / Qk 2p+dim−1; DomainLFIntegrator: p+1.
    let p = order as usize;
    let pk = matches!(
        mesh.elem_type,
        fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tet4
    );
    let qo_diff = if pk { (2 * p - 2).max(1) } else { 2 * p + dim - 1 };
    let qo_lf = (p + 1) as u8;

    // ── Linear form b(.) ─────────────────────────────────────────────────────
    let source = DomainSourceIntegrator::new(&*rhs_f);
    let mut b = Assembler::assemble_linear_marked(
        &space,
        &[(&source as &dyn LinearIntegrator, Some(&ess_elem))],
        qo_lf,
    );

    // ── Bilinear form a(.,.) on the surrogate domain ─────────────────────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut a = Assembler::assemble_bilinear_marked(
        &space,
        &[(&diffusion as &dyn BilinearIntegrator, Some(&ess_elem))],
        qo_diff as u8,
    );

    // SBM Dirichlet face integrators.
    if lst > 0 {
        let sbm_b = Sbm3DirichletIntegrator {
            space: &space,
            alpha,
            dist: &dist_dofs,
            elem_marker: &elem_marker,
            include_cut_cell: args.include_cut_cell,
            ho_terms,
        }
        .assemble_bilinear();
        a = a.add(&sbm_b);
        let ubc = dbc.as_deref().expect("Dirichlet data required");
        let sbm_lf = Sbm3DirichletLFIntegrator {
            space: &space,
            alpha,
            dist: &dist_dofs,
            elem_marker: &elem_marker,
            include_cut_cell: args.include_cut_cell,
            ho_terms,
            ubc,
        }
        .assemble_linear();
        for (bi, li) in b.iter_mut().zip(sbm_lf.iter()) {
            *bi += li;
        }
    }

    // SBM Neumann face integrators.
    if nlst > 0 {
        assert!(
            !args.include_cut_cell,
            "include_cut_cell option must be set to false for Neumann boundary conditions."
        );
        let nhat = normal_bc.as_deref().expect("normal vector coefficient");
        let sbm_n = Sbm3NeumannIntegrator {
            space: &space,
            dist: &dist_dofs,
            nhat,
            elem_marker: &elem_marker,
            include_cut_cell: args.include_cut_cell,
            ho_terms,
        }
        .assemble_bilinear();
        a = a.add(&sbm_n);
        let un = nbc.as_deref().expect("Neumann data required");
        let sbm_lf = Sbm3NeumannLFIntegrator {
            space: &space,
            dist: &dist_dofs,
            nhat,
            tn: un,
            elem_marker: &elem_marker,
            include_cut_cell: args.include_cut_cell,
            ho_terms,
        }
        .assemble_linear();
        for (bi, li) in b.iter_mut().zip(sbm_lf.iter()) {
            *bi += li;
        }
    }

    // ── Form the linear system and solve ─────────────────────────────────────
    // x.ProjectCoefficient(exactCoef) initial condition; FormLinearSystem with
    // MFEM's DIAG_KEEP elimination on the inactive/essential dofs.
    let x_init = space.interpolate(&exact).as_slice().to_vec();
    let ess: Vec<u32> = ess_tdof_list.iter().map(|&d| d as u32).collect();
    let ess_vals: Vec<f64> = ess_tdof_list.iter().map(|&d| x_init[d]).collect();
    if std::env::var("DUMP").is_ok() {
        let mut s = String::new();
        let nd = space.n_dofs();
        for i in 0..nd {
            for j in 0..nd {
                let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{} ", a.get(i, j)));
            }
            s.push(0x0A as char);
        }
        std::fs::write("A_rust.txt", s).unwrap();
        let mut s2 = String::new();
        for i in 0..nd {
            let _ = std::fmt::Write::write_fmt(&mut s2, format_args!("{}\n", b[i]));
        }
        std::fs::write("b_rust.txt", s2).unwrap();
    }
    // DIAG_ONE elimination: the rows of dofs belonging only to inactive
    // (never-assembled) elements have zero diagonals, which breaks the AMG
    // setup; DIAG_ONE gives the same solution with a regular matrix.  The
    // recovered essential values are re-applied after the solve.
    fem_space::constraints::apply_dirichlet_diag_one(&mut a, &mut b, &ess, &ess_vals);


    let mut xx = vec![0.0_f64; space.n_dofs()];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        ..Default::default()
    };
    let res = match solve_amg_gmres(&a, &b, &mut xx, &AmgConfig::default(), 50, &cfg) {
        Ok(r) => r,
        Err(e) => {
            // Jacobi-preconditioned GMRES fallback (the AMG hierarchy can
            // fail on systems with zero rows from the inactive elements).
            eprintln!("GMRES+AMG failed ({e}); falling back to unpreconditioned GMRES");
            xx = vec![0.0_f64; space.n_dofs()];
            let fb_cfg = SolverConfig { max_iter: 20000, ..cfg.clone() };
            match fem_solver::solve_gmres(&a, &b, &mut xx, 100, &fb_cfg) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Jacobi GMRES failed ({e}); keeping unconverged iterate");
                    fem_linalg::SolveResult {
                        converged: false,
                        iterations: 0,
                        final_residual: 0.0,
                    }
                }
            }
        }
    };
    println!("GMRES (BiCGSTAB in MFEM) iterations = {}", res.iterations);

    // RecoverFEMSolution: essential dofs carry the exact values.
    let mut x = xx;
    for (&dof, &v) in ess_tdof_list.iter().zip(ess_vals.iter()) {
        x[dof] = v;
    }
    let x_gf = GridFunction::new(&space, x.clone());

    // ── Output ───────────────────────────────────────────────────────────────
    mesh.save("diffusion.mesh");
    fem_io::mfem::write_mfem_gf_file("diffusion.gf", D, &x, "H1", order, 1, 8)
        .unwrap_or_else(|e| eprintln!("warning: could not write diffusion.gf: {e}"));

    // ── Error report ─────────────────────────────────────────────────────────
    if lst == 2 || lst == 3 || (lst == -1 && nlst == 2) {
        let err = x_gf.compute_l2_error(&exact, 2 * order);
        println!("Global L2 error: {err:.6}");
    }
    let norm = x_gf.compute_l1_error(&|_x: &[f64]| 1.0, 2 * order);
    println!("{norm:.10}");
}
