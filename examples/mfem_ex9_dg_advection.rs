//! # MFEM Example 9 — DG Advection (1:1 translation)
//!
//! Solves `du/dt + v·∇u = 0` using DG with upwind flux.
//! 1:1 translation: periodic meshes, problem types 0-3,
//! explicit (RK4) time integration, output files.
//!
//! Notes:
//! - Uses standard L2 basis (vs MFEM's GaussLobatto) → different DOF layout.
//! - Periodic meshes with degenerate quads after refinement may produce INF
//!   entries in the advection matrix (assembler needs det>0 threshold).
//! - Works best on well-shaped quad meshes.
//!
//! Reference: `mfem/ex9.cpp`

use std::fs::File;
use std::io::Write;

use fem_assembly::{
    Assembler,
    dg::dg_advection::{
        DGAdvectionIntegrator, assemble_dg_interior_faces, assemble_advection_boundary_full,
        DgAdvectionProblem, dg_velocity, dg_initial_condition, dg_inflow_bc,
    },
    interior_faces::InteriorFaceList,
    postproc::coefficient::{FnVectorCoeff, VectorCoeff},
    standard::MassIntegrator,
};
use fem_linalg::CooMatrix;
use fem_mesh::{refine_uniform, topology::MeshTopology, Mesh};
use fem_solver::{
    SolverConfig, solve_cg,
    ode::{Rk4, TimeStepper},
};
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // Display options (matching C++ args.PrintOptions(cout))
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --problem {}", args.problem);
    println!("   --refine {}", args.refine);
    println!("   --order {}", args.order);
    println!("   --no-partial-assembly");
    println!("   --no-element-assembly");
    println!("   --no-full-assembly");
    println!("   --device cpu");
    println!("   --ode-solver {}", args.ode_solver);
    println!("   --t-final {}", args.t_final);
    println!("   --time-step {}", args.dt);
    println!("   --no-visualization");
    println!("   --no-visit-datafiles");
    println!("   --no-paraview-datafiles");
    println!("   --ascii-datafiles");
    println!("   --visualization-steps 5");
    println!();
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");
    let mfem = fem_io::mfem::read_mfem_file(&args.mesh).expect("read mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("2D mesh");
    let dim = 2;

    // Bounding box for velocity/IC mapping
    let mut bb_min = vec![f64::MAX; dim];
    let mut bb_max = vec![f64::MIN; dim];
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        for d in 0..dim { bb_min[d] = bb_min[d].min(c[d]); bb_max[d] = bb_max[d].max(c[d]); }
    }

    let mesh = if args.refine > 0 {
        let mut m = mesh; for _ in 0..args.refine { m = refine_uniform(&m); } m
    } else { mesh };

    let problem = match args.problem {
        0 => DgAdvectionProblem::Translation,
        1 => DgAdvectionProblem::Rotation,
        2 => DgAdvectionProblem::RotationP2,
        3 => DgAdvectionProblem::Twist,
        _ => DgAdvectionProblem::Translation,
    };

    // Velocity coefficient
    let vel_fn = {
        let bb_min_c = bb_min.clone(); let bb_max_c = bb_max.clone();
        move |x: &[f64], out: &mut [f64]| {
            let v = dg_velocity(problem, x, &bb_min_c, &bb_max_c);
            for (i, &vi) in v.iter().enumerate() { out[i] = vi; }
        }
    };
    let vel_coeff = FnVectorCoeff(vel_fn);

    // DG space and mass matrix
    let space = L2Space::new(mesh.clone(), args.order);
    let n = space.n_dofs();
    println!("Number of unknowns: {n}");

    // Quadrature: mass uses 2*order+1 (exact for degree 2p),
    // advection uses 2*order (matching MFEM ConvectionIntegrator).
    // Avoid 3+ point/axis rules that hit degenerate element quad points.
    let qo_mass = (args.order as u8 * 2 + 2).max(3);
    let qo_adv = (args.order as u8 * 2).max(2);
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], qo_mass);

    // Advection operator: volume + interior faces
    let dg_adv = DGAdvectionIntegrator { velocity: vel_coeff };
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg_adv], qo_adv);
    let ifl = InteriorFaceList::build(space.mesh());
    let qface = (args.order as u8 * 2).max(2);

    let mut coo = CooMatrix::new(n, n);
    for i in 0..n { for p in k_vol.row_ptr[i]..k_vol.row_ptr[i+1] {
        coo.add(i, k_vol.col_idx[p] as usize, k_vol.values[p]);
    }}
    assemble_dg_interior_faces(&mut coo, space.mesh(), &space, &ifl, args.order, qface, &dg_adv);

    // Periodic face pairs (for 'boundary 0' meshes)
    if mesh.n_boundary_faces() == 0 && mesh.dim() == 2 {
        detect_periodic_pairs(space.mesh(), &mut coo, &space, args.order, qface, &dg_adv.velocity);
    }

    // Boundary contribution
    let bc_tags: Vec<i32> = mesh.unique_boundary_tags();
    let inflow_g = |x: &[f64]| dg_inflow_bc(problem, x);
    let vel_bdr = {
        let bb_min_c = bb_min.clone(); let bb_max_c = bb_max.clone();
        FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
            let v = dg_velocity(problem, x, &bb_min_c, &bb_max_c);
            for (i, &vi) in v.iter().enumerate() { out[i] = vi; }
        })
    };
    let (k_bdr, rhs_bc) = assemble_advection_boundary_full(
        &space, &vel_bdr, &bc_tags, &inflow_g, args.order, qface);
    for i in 0..n { for p in k_bdr.row_ptr[i]..k_bdr.row_ptr[i+1] {
        coo.add(i, k_bdr.col_idx[p] as usize, k_bdr.values[p]);
    }}

    let k_adv = coo.into_csr();

    // ── Initial condition ──────────────────────────────────────────────────
    let mut u = space.interpolate(&|x| dg_initial_condition(problem, x, &bb_min, &bb_max))
        .as_slice().to_vec();

    // ── Initial output files (matching C++: ex9.mesh, ex9-init.gf) ──────────
    {
        let mut mf = File::create("ex9.mesh").unwrap();
        fem_io::mfem::write_mfem(&mut mf, &mesh, None).unwrap();
        let mut sf = File::create("ex9-init.gf").unwrap();
        // MFEM GridFunction header
        writeln!(sf, "FiniteElementSpace").unwrap();
        writeln!(sf, "FiniteElementCollection: L2_{}D_P{}", dim, args.order).unwrap();
        writeln!(sf, "VDim: 1").unwrap();
        writeln!(sf, "Ordering: 0").unwrap();
        writeln!(sf).unwrap();
        for i in 0..n { writeln!(sf, "{:.7e}", u[i]).unwrap(); }
    }

    // ── Time integration (explicit RK4, matching C++ default ode_solver=4) ──
    let solver_cfg = SolverConfig { rtol: 1e-9, max_iter: 100, verbose: false, ..Default::default() };
    let dt = args.dt.min(args.t_final); let mut t = 0.0;
    let vis_steps = 5;
    let mut ti = 0;


    let steps = (args.t_final / dt).ceil() as usize;
    for _ in 0..steps {
        let dta = dt.min(args.t_final - t);
        Rk4.step(t, dta, &mut u, |_t, u, dudt| {
            let mut f = vec![0.0; n]; k_adv.spmv(u, &mut f);
            for i in 0..n { f[i] += rhs_bc[i]; }
            let _ = solve_cg(&mass, &f, dudt, &solver_cfg);
        });
        t += dta; ti += 1;
        if ti % vis_steps == 0 || t >= args.t_final - 1e-14 {
            println!("time step: {ti}, time: {t:.3}");
        }
    }

    // ── Final output file (matching C++: ex9-final.gf) ──────────────────────
    {
        let mut sf = File::create("ex9-final.gf").unwrap();
        writeln!(sf, "FiniteElementSpace").unwrap();
        writeln!(sf, "FiniteElementCollection: L2_{}D_P{}", dim, args.order).unwrap();
        writeln!(sf, "VDim: 1").unwrap();
        writeln!(sf, "Ordering: 0").unwrap();
        writeln!(sf).unwrap();
        for i in 0..n { writeln!(sf, "{:.7e}", u[i]).unwrap(); }
    }
    eprintln!("  Done. Total time: {:.3}s", t0.elapsed().as_secs_f64());
}

/// Detect periodic face pairs for 'boundary 0' meshes and assemble their
/// flux contributions using each element's OWN face nodes.
fn detect_periodic_pairs<M: MeshTopology, V: VectorCoeff>(
    mesh: &M, coo: &mut CooMatrix<f64>, space: &impl FESpace<Mesh=M>,
    order: u8, qface: u8, velocity: &V,
) {
    use std::collections::HashMap;
    // Map: sorted face key → (elem, local_face_idx, unsorted_nodes)
    let mut edge_map: HashMap<Vec<u32>, (u32, Vec<u32>)> = HashMap::new();
    for e in mesh.elem_iter() {
        let en = mesh.element_nodes(e);
        let faces: Vec<Vec<usize>> = match en.len() {
            3 => vec![vec![0,1],vec![1,2],vec![2,0]],
            4 => vec![vec![0,1],vec![1,2],vec![2,3],vec![3,0]],
            _ => vec![],
        };
        for lf in &faces {
            let mut key: Vec<u32> = lf.iter().map(|&k| en[k]).collect();
            key.sort_unstable();
            edge_map.entry(key).or_insert((e, lf.iter().map(|&k| en[k]).collect()));
        }
    }
    // Edges appearing once are "virtual boundary edges"
    let boundary: Vec<(u32, Vec<u32>)> = edge_map.into_values().collect();
    struct BEdge { elem: u32, nodes: Vec<u32>, mid: [f64;2], normal: [f64;2] }
    let mut edges: Vec<BEdge> = boundary.iter().map(|&(e, ref n)| {
        let p0 = mesh.node_coords(n[0]); let p1 = mesh.node_coords(n[1]);
        let dx = p1[0]-p0[0]; let dy = p1[1]-p0[1];
        let len = (dx*dx+dy*dy).sqrt();
        let nx = -dy/len; let ny = dx/len;
        let en = mesh.element_nodes(e);
        let cx = en.iter().map(|&n|mesh.node_coords(n)[0]).sum::<f64>()/en.len() as f64;
        let cy = en.iter().map(|&n|mesh.node_coords(n)[1]).sum::<f64>()/en.len() as f64;
        let mx = (p0[0]+p1[0])/2.0; let my = (p0[1]+p1[1])/2.0;
        let (nx,ny) = if nx*(mx-cx)+ny*(my-cy) >= 0.0 {(nx,ny)} else {(-nx,-ny)};
        BEdge{elem:e, nodes:n.clone(), mid:[mx,my], normal:[nx,ny]}
    }).collect();

    // Group by normal direction and pair opposites
    let mut groups: Vec<(Vec<usize>, Vec<usize>, usize)> = Vec::new();
    // x-direction: left vs right
    let left: Vec<usize> = (0..edges.len()).filter(|&i|edges[i].normal[0] < -0.5).collect();
    let right: Vec<usize> = (0..edges.len()).filter(|&i|edges[i].normal[0] > 0.5).collect();
    if !left.is_empty() && left.len() == right.len() { groups.push((left, right, 1)); } // dir=1 for y-sorting
    // y-direction: bottom vs top
    let bottom: Vec<usize> = (0..edges.len()).filter(|&i|edges[i].normal[1] < -0.5).collect();
    let top: Vec<usize> = (0..edges.len()).filter(|&i|edges[i].normal[1] > 0.5).collect();
    if !bottom.is_empty() && bottom.len() == top.len() { groups.push((bottom, top, 0)); } // dir=0 for x-sorting

    for (neg, pos, sort_dir) in &groups {
        let mut neg_sorted: Vec<usize> = neg.clone();
        let mut pos_sorted: Vec<usize> = pos.clone();
        neg_sorted.sort_by_key(|&i| (edges[i].mid[1-sort_dir] * 1e6) as i64);
        pos_sorted.sort_by_key(|&i| (edges[i].mid[1-sort_dir] * 1e6) as i64);
        let mut pairs: Vec<(u32, u32, Vec<u32>, Vec<u32>)> = Vec::new();
        for i in 0..neg_sorted.len() {
            // left/bottom element = neg, right/top element = pos
            // Use neg element's face nodes as left_face (for normal + left basis eval)
            // Use pos element's face nodes as right_face (for right basis eval)
            pairs.push((
                edges[neg_sorted[i]].elem,
                edges[pos_sorted[i]].elem,
                edges[neg_sorted[i]].nodes.clone(),
                edges[pos_sorted[i]].nodes.clone(),
            ));
        }
        fem_assembly::dg::dg_advection::assemble_periodic_flux(coo, mesh, space, &pairs, order, qface, velocity);
    }
}

struct Args { mesh: String, problem: usize, refine: usize, order: u8, dt: f64, t_final: f64, ode_solver: usize }
impl Args {
    fn parse() -> Self {
        let mut mesh = "../data/periodic-hexagon.mesh".to_string();
        let mut problem = 0usize; let mut refine = 2usize; let mut order = 3u8;
        let mut dt = 0.01_f64; let mut t_final = 10.0_f64; let mut ode_solver = 4usize;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() { match arg.as_str() {
            "-m"|"--mesh" => { mesh = it.next().unwrap_or(mesh); }
            "-p"|"--problem" => { problem = it.next().and_then(|s|s.parse().ok()).unwrap_or(problem); }
            "-r"|"--refine" => { refine = it.next().and_then(|s|s.parse().ok()).unwrap_or(refine); }
            "-o"|"--order" => { order = it.next().and_then(|s|s.parse().ok()).unwrap_or(order); }
            "-dt"|"--time-step" => { dt = it.next().and_then(|s|s.parse().ok()).unwrap_or(dt); }
            "-tf"|"--t-final" => { t_final = it.next().and_then(|s|s.parse().ok()).unwrap_or(t_final); }
            "-s"|"--ode-solver" => { ode_solver = it.next().and_then(|s|s.parse().ok()).unwrap_or(ode_solver); }
            "-no-vis"|"--no-visualization" => {}
            _ => {}
        }}
        Args { mesh, problem, refine, order, dt, t_final, ode_solver }
    }
}
