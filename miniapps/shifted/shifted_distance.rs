//! Distance Miniapp: finite element distance solver
//! (MFEM `miniapps/shifted/distance.cpp`, serial `-np 1` 1:1 port).
//!
//! Computes the "distance" (through the mesh) to the zero level set of a
//! given function with the Heat, p-Laplacian or Rvachev-normalization solvers
//! (`fem_assembly::dist_solver`), for the level-set problems of the C++
//! miniapp (1: ball/sphere, 2: perturbed sine, 3: gyroid, 4: doughnut +
//! swiss cheese).
//!
//! Deviations from the C++ miniapp:
//! * Problems 0 and 5 (point sources via `DeltaCoefficient`) are not
//!   supported — the serial stack has no Delta projection yet.
//! * GLVis / ParaView output is not available: `-vis` (the C++ default)
//!   prints a notice and exits with code 3.
//! * The `ExactDistSphereLoc` local-error coefficient uses the average mesh
//!   size instead of element 0's size for the local band width.
//!
//! Sample runs:
//!   cargo run --release --example shifted_distance -- -m data/inline-quad.mesh -rs 3 -o 2 -t 1.0 -p 1 -no-vis
//!   cargo run --release --example shifted_distance -- -rs 3 -o 2 -t 1.0 -p 2 -no-vis
//!   cargo run --release --example shifted_distance -- -m data/inline-hex.mesh -rs 2 -o 2 -p 1 -no-vis

mod sbm_aux;

use fem_assembly::dist_solver::{
    avg_element_size, scalar_dist_to_vector, DistanceSolver, HeatDistanceSolver,
    NormalizationDistanceSolver, PLapDistanceSolver, PDEFilter, FnDist,
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

use sbm_aux as aux;

type LevelSetFn = dyn Fn(&[f64]) -> f64 + Send + Sync;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut mesh_file = "data/inline-quad.mesh".to_string();
    let mut solver_type = 0_i32;
    let mut problem = 1_i32;
    let mut rs_levels = 2_i32;
    let mut order = 2_i32;
    let mut t_param = 1.0_f64;
    let mut vis = true;

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        i += 1;
        let mut next = || {
            let v = argv.get(i).cloned().unwrap_or_default();
            i += 1;
            v
        };
        match arg {
            "-m" | "--mesh" => mesh_file = next(),
            "-s" | "--solver" => solver_type = next().parse().unwrap_or(solver_type),
            "-p" | "--problem" => problem = next().parse().unwrap_or(problem),
            "-rs" | "--refine-serial" => rs_levels = next().parse().unwrap_or(rs_levels),
            "-o" | "--order" => order = next().parse().unwrap_or(order),
            "-t" | "--t-param" => t_param = next().parse().unwrap_or(t_param),
            "-vis" | "--visualization" => vis = true,
            "-no-vis" | "--no-visualization" => vis = false,
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
    }
    if vis {
        println!("GLVis/ParaView output is not available in this serial port.");
        println!("Re-run with -no-vis to disable visualization output.");
        std::process::exit(3);
    }
    if problem == 0 || problem == 5 {
        eprintln!("Point-source problems (DeltaCoefficient) are not supported by the serial port.");
        std::process::exit(1);
    }

    let mfem = read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("Failed to read MFEM mesh '{}': {e}", mesh_file));
    if let Some(mesh) = mfem.mesh2d {
        run::<2>(mesh, solver_type, problem, rs_levels.max(0) as usize, order.max(1) as u8, t_param);
    } else {
        let mesh = mfem.mesh3d.expect("no mesh in file");
        run::<3>(mesh, solver_type, problem, rs_levels.max(0) as usize, order.max(1) as u8, t_param);
    }
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

/// L2 norm of a scalar GridFunction (MFEM ComputeL2Error(zero)).
fn l2_norm_scalar<M: MeshTopology, S: FESpace<Mesh = M>>(gf: &GridFunction<'_, S>, qo: u8) -> f64 {
    gf.compute_l2_error(&|_x: &[f64]| 0.0, qo)
}

/// L2 norm of a vector grid function (component-major DOFs), MFEM
/// `distance_v.ComputeL2Error(zero)`.
fn l2_norm_vector<const D: usize>(mesh: &Mesh<D>, order: u8, dofs: &[f64], qo: u8) -> f64 {
    let dim = mesh.dim() as usize;
    let h1 = H1Space::new(mesh.clone(), order);
    let n = h1.n_dofs();
    let mut err2 = 0.0_f64;
    for e in mesh.elem_iter() {
        let re = mesh.element_type(e).ref_elem(order);
        let nd = re.n_dofs();
        let rule = re.quadrature(qo);
        let mut phi = vec![0.0_f64; nd];
        let dofs_e = h1.element_dofs(e);
        for (q, xi) in rule.points.iter().enumerate() {
            let (_jit, det, _x) = jacobian_at_dyn(mesh, e, xi);
            let w = rule.weights[q] * det.abs();
            re.eval_basis(xi, &mut phi);
            let mut v2 = 0.0_f64;
            for c in 0..dim {
                let mut vc = 0.0_f64;
                for (i, &dof) in dofs_e.iter().enumerate() {
                    vc += dofs[c * n + dof as usize] * phi[i];
                }
                v2 += vc * vc;
            }
            err2 += w * v2;
        }
    }
    err2.sqrt()
}

fn run<const D: usize>(
    mesh: Mesh<D>,
    solver_type: i32,
    problem: i32,
    rs_levels: usize,
    order: u8,
    t_param: f64,
) where
    Mesh<D>: RefineMany,
{
    // Refine the mesh (MFEM UniformRefinement; serial only).
    let mesh = mesh.refine_many(rs_levels);
    let _dim = D;

    // Level set of the requested problem (distance.cpp coefficient table).
    let ls: Box<LevelSetFn> = match problem {
        1 => Box::new(aux::sphere_ls),
        2 => Box::new(aux::sine_ls),
        3 => Box::new(aux::gyroid),
        4 => Box::new(aux::doughnut_cheese),
        _ => panic!("Unrecognized -problem option."),
    };

    let dx = avg_element_size(&mesh);

    // Solve (Heat / p-Laplacian / Rvachev normalization).
    enum SolverKind {
        Heat(Box<HeatDistanceSolver>),
        PLap(Box<PLapDistanceSolver>),
        Normalization,
    }
    let solver = match solver_type {
        0 => {
            let mut ds = HeatDistanceSolver::new(t_param * dx * dx);
            ds.smooth_steps = 0;
            SolverKind::Heat(Box::new(ds))
        }
        1 => SolverKind::PLap(Box::new(PLapDistanceSolver::new(10, 50, 1e-7, 1e-7))),
        2 => SolverKind::Normalization,
        _ => panic!("Wrong solver option."),
    };

    let pfes_s = H1Space::new(mesh.clone(), order);

    // Smooth-out Gibbs oscillations from the input level set (MFEM PDEFilter
    // with radius dx; the normalization solver needs a more diffused input).
    let fw = if solver_type == 2 { 4.0 * dx } else { dx };
    let mut filter: PDEFilter<Mesh<D>> = PDEFilter::new(mesh.clone(), fw);
    let n2 = H1Space::<Mesh<D>>::new(mesh.clone(), 2).n_dofs();
    let mut filtered = vec![0.0_f64; n2];
    filter.filter_coeff(Box::new(FnDist(ls.as_ref())), &mut filtered);
    let fs2 = std::sync::Arc::new(H1Space::new(mesh.clone(), 2));
    let dofs2 = std::sync::Arc::new(filtered);
    let ls_filt = move |x: &[f64]| {
        let gf = GridFunction::new(fs2.as_ref(), (*dofs2).clone());
        gf.get_value(x).unwrap_or(0.0)
    };

    // ComputeScalarDistance.
    let mut distance_s = vec![0.0_f64; pfes_s.n_dofs()];
    match &solver {
        SolverKind::Heat(s) => s.compute_scalar_distance(&ls_filt, &mut distance_s, &pfes_s),
        SolverKind::PLap(s) => s.compute_scalar_distance(&ls_filt, &mut distance_s, &pfes_s),
        SolverKind::Normalization => {
            NormalizationDistanceSolver.compute_scalar_distance(&ls_filt, &mut distance_s, &pfes_s)
        }
    }

    // ComputeVectorDistance (MFEM: dist_solver->ComputeVectorDistance is
    // ScalarDistToVector on the scalar result).
    let distance_v = scalar_dist_to_vector(&pfes_s, &distance_s);

    // Send the solution by socket to a GLVis server — not available.

    // Norms: distance_s.ComputeL2Error(zero), distance_v.ComputeL2Error(zero).
    let qo = 2 * order as u8;
    let s_gf = GridFunction::new(&pfes_s, distance_s.clone());
    let s_norm = l2_norm_scalar(&s_gf, qo);
    let v_norm = l2_norm_vector(&mesh, order, &distance_v, qo);
    println!("Norms: {s_norm:.10} {v_norm:.10}");

    if problem == 1 {
        // Global errors against the exact sphere distance.
        let error_l1 = s_gf.compute_l1_error(&aux::exact_dist_sphere, qo);
        let error_li = max_error(&s_gf, &aux::exact_dist_sphere, qo);
        println!("Global L1 error:   {error_l1:.10}");
        println!("Global Linf error: {error_li:.10}");

        // Local errors (ExactDistSphereLoc: one zone length around the level
        // set is replaced by the exact distance).
        let h0 = avg_element_size(&mesh);
        let exact_loc = |x: &[f64]| -> f64 {
            let dim = x.len();
            let xc = x[0] - 0.5;
            let yc = if dim > 1 { x[1] - 0.5 } else { 0.0 };
            let zc = if dim > 2 { x[2] - 0.5 } else { 0.0 };
            let r = (xc * xc + yc * yc + zc * zc).sqrt();
            if (r - aux::RADIUS).abs() < h0 {
                (r - aux::RADIUS).abs()
            } else {
                s_gf.get_value(x).unwrap_or(0.0)
            }
        };
        let error_l1_loc = s_gf.compute_l1_error(&exact_loc, qo);
        let error_li_loc = max_error(&s_gf, &exact_loc, qo);
        println!("Local  L1 error:   {error_l1_loc:.10}");
        println!("Local  Linf error: {error_li_loc:.10}");
    }
}

/// MFEM GridFunction::ComputeMaxError: max |u_h − c| over quadrature points.
fn max_error<M: MeshTopology, S: FESpace<Mesh = M>>(
    gf: &GridFunction<'_, S>,
    exact: &dyn Fn(&[f64]) -> f64,
    qo: u8,
) -> f64 {
    let mesh = gf.space().mesh();
    let order = gf.space().order();
    let mut maxe = 0.0_f64;
    for e in mesh.elem_iter() {
        let re = mesh.element_type(e).ref_elem(order);
        let nd = re.n_dofs();
        let rule = re.quadrature(qo);
        let mut phi = vec![0.0_f64; nd];
        let dofs = gf.space().element_dofs(e);
        for xi in rule.points.iter() {
            re.eval_basis(xi, &mut phi);
            let mut u = 0.0_f64;
            for (i, &d) in dofs.iter().enumerate() {
                u += gf.dofs()[d as usize] * phi[i];
            }
            let (_j, _det, xp) = jacobian_at_dyn(mesh, e, xi);
            let diff = (u - exact(xp.as_slice())).abs();
            maxe = maxe.max(diff);
        }
    }
    maxe
}

/// Dimension-flexible affine Jacobian for error quadrature.
fn jacobian_at_dyn(mesh: &dyn MeshTopology, e: u32, xi: &[f64]) -> (nalgebra::DMatrix<f64>, f64, Vec<f64>) {
    let dim = mesh.topological_dim() as usize;
    let re = mesh.element_type(e).ref_elem(1);
    let nd = re.n_dofs();
    let mut phi = vec![0.0_f64; nd];
    let mut grad = vec![0.0_f64; nd * dim];
    re.eval_basis(xi, &mut phi);
    re.eval_grad_basis(xi, &mut grad);
    let nodes = mesh.element_nodes(e);
    let mut j = nalgebra::DMatrix::<f64>::zeros(dim, dim);
    let mut x = vec![0.0_f64; dim];
    for (k, &node) in nodes.iter().enumerate() {
        let xk = mesh.node_coords(node);
        for i in 0..dim {
            x[i] += phi[k] * xk[i];
            for c in 0..dim {
                j[(i, c)] += xk[i] * grad[k * dim + c];
            }
        }
    }
    let det = match dim {
        1 => j[(0, 0)],
        2 => j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)],
        _ => j.determinant(),
    };
    let jit = j.clone().try_inverse().unwrap_or(j).transpose();
    (jit, det, x)
}
