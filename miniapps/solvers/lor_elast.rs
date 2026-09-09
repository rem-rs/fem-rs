//! # Elasticity LOR Block Preconditioning Miniapp
//! (1:1 serial port of MFEM `miniapps/solvers/lor_elast.cpp`)
//!
//! Solves the same cantilever beam problem as MFEM Example 2
//!
//! ```text
//!   -div(sigma(u)) = 0        in Omega
//!              u   = 0        on boundary attribute 1 (fixed wall)
//!        sigma(u)*n = (0,...,-1e-2) on boundary attribute 2 (pull down)
//! ```
//!
//! with `sigma = lambda tr(eps) I + 2 mu eps` and piecewise-constant Lamé
//! parameters (material 1 = 50x stiffer than material 2), preconditioned with
//! the block-diagonal **low-order-refined AMG** solver
//!
//! ```text
//!   P^-1 = diag(AMG(A_00), ..., AMG(A_{d-1,d-1}))
//! ```
//!
//! where `A_jj = Int [ mu grad(u).grad(v) + (lambda+mu) (d_j u)(d_j v) ]` is
//! the (j,j) scalar component of the elasticity operator assembled on the P1
//! LOR space (`ElasticityComponentIntegrator` on `ParLORDiscretization` in
//! MFEM).  The outer solve uses the same tolerances as the C++ CGSolver
//! (rtol 1e-8, max 2500 iterations); the default Krylov method is flexible
//! GMRES (restart 100) because the fem-rs AMG V-cycle is not guaranteed to be
//! SPD on the anisotropic beam cells (`-cg` selects PCG where the V-cycle
//! permits it).
//!
//! ## Porting notes (serial, -np 1 semantics)
//! - MFEM runs the LEGACY path (full matrix + HypreBoomerAMG systems solve)
//!   by default and the LOR block-diagonal path under `-pa`.  This port
//!   always runs the LOR block-diagonal path with a fully assembled operator
//!   (fem-rs has no partial assembly for the high-order elasticity operator
//!   and no Hypre; the linear system is identical, only the operator storage
//!   differs).
//! - The per-block AMG mirrors the hypre setup of the miniapp (classical
//!   Ruge-Stueben at strength 0.25, `SetRelaxType`-style smoothing); the
//!   linlvo relaxation is symmetric Gauss-Seidel with 3+3 sweeps, the most
//!   robust combination measured on the beam meshes across refinement and
//!   order.
//! - Options that select unported solver variants exit with status 3 and a
//!   message (`-elast/-sys`, `-ss`, `-ca`, `-vdim`, non-CPU `-d`).
//! - Only `byNODES` vector ordering is provided (fem-rs `VectorH1Space`
//!   convention; this is the C++ default `-nodes`).
//! - Printing mirrors the C++ miniapp: the device configuration, `Number of
//!   finite element unknowns`, `Assembling: r.h.s. ... matrix ... done.`,
//!   `Size of linear system`, the Krylov iteration log and the elapsed-time
//!   summary.
//!
//! ## Sample runs
//! ```text
//! cargo run --example lor_elast -- -m data/beam-tri.mesh
//! cargo run --example lor_elast -- -m data/beam-quad.mesh -o 2
//! cargo run --example lor_elast -- -m data/beam-hex.mesh -l 1 -o 2
//! ```

use std::io::Write as _;
use std::time::Instant;

use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_solver::lor::{assemble_lor_elasticity_blocks, AmgConfig, LorElasticityPrecond};
use fem_solver::{solve_fgmres_precond, solve_pcg_precond, SolverConfig};
use fem_space::constraints::{boundary_dofs, form_linear_system};
use fem_space::dof_manager::{DofManager, EdgeKey, QuadFaceKey};
use fem_space::fe_space::FESpace;
use fem_space::lor::LorH1;
use fem_space::vector_h1::VectorH1Space;

pub fn main() {
    // 2. Parse command-line options (MFEM OptionsParser names).
    let mut mesh_file = "data/beam-tri.mesh".to_string();
    let mut order = 1u8;
    let mut ref_levels = 0usize;
    let mut visualization = false;
    let mut device = "cpu".to_string();
    let mut force_cg = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() {
                    mesh_file = v;
                }
            }
            "-o" | "--order" => {
                if let Some(v) = it.next() {
                    order = v.parse().unwrap_or(1);
                }
            }
            "-l" | "--reflevels" => {
                if let Some(v) = it.next() {
                    ref_levels = v.parse().unwrap_or(0);
                }
            }
            "-pa" | "--partial-assembly" | "-no-pa" | "--no-partial-assembly" => {
                // Accepted: this port always uses the LOR block solver with a
                // fully assembled operator (see porting notes).
            }
            "-nodes" | "--by-nodes" => {
                // Default (and only) ordering of fem-rs VectorH1Space.
            }
            "-vdim" | "--by-vdim" => {
                eprintln!(
                    "lor_elast: -vdim (byVDIM vector ordering) is not ported; \
                     fem-rs VectorH1Space is byNODES only"
                );
                std::process::exit(3);
            }
            "-elast" | "--amg-for-elasticity" | "-sys" | "--amg-for-systems" => {
                eprintln!(
                    "lor_elast: the LEGACY path (global HypreBoomerAMG on the \
                     assembled system, -elast/-sys) is not portable to fem-rs; \
                     only the block-diagonal LOR-AMG solver is provided"
                );
                std::process::exit(3);
            }
            "-ss" | "--sub-solve" | "-no-ss" | "--no-sub-solve" => {
                if arg.starts_with("-ss") || arg.starts_with("--sub-solve") {
                    eprintln!(
                        "lor_elast: -ss (CG sub-solves on the diagonal \
                         components) is not ported"
                    );
                    std::process::exit(3);
                }
            }
            "-ca" | "--component-action" | "-no-ca" | "--no-component-action" => {
                if arg.starts_with("-ca") || arg.starts_with("--component") {
                    eprintln!(
                        "lor_elast: -ca (componentwise BlockFESpaceOperator \
                         action) is not ported"
                    );
                    std::process::exit(3);
                }
            }
            "-d" | "--device" => {
                if let Some(v) = it.next() {
                    device = v;
                }
            }
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-cg" | "--conjugate-gradient" => force_cg = true,
            other => {
                eprintln!("lor_elast: ignoring unknown option '{other}'");
            }
        }
    }
    if device != "cpu" {
        eprintln!(
            "lor_elast: device '{device}' requested but only 'cpu' is \
             available in fem-rs"
        );
        std::process::exit(3);
    }
    // MFEM step 3 (Mpi::Root()): device.Print() on a vanilla cpu build.
    println!("Device configuration: cpu");
    println!("Memory configuration: std-internal");

    // 4. Read the (serial) mesh from the given mesh file.
    let mfem = read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("lor_elast: failed to read mesh {mesh_file}: {e}"));
    if let Some(mut m) = mfem.mesh2d {
        for _ in 0..ref_levels {
            m = fem_mesh::refine_uniform(&m);
        }
        let lor = LorH1::<2>::new(&m, order).expect("LOR H1 discretization");
        run::<2>(m, lor.lor_mesh().clone(), order, lor.perm().to_vec(), lor.n_ho(), force_cg, visualization);
    } else if let Some(mut m) = mfem.mesh3d {
        for _ in 0..ref_levels {
            m = fem_mesh::refine_uniform_3d(&m);
        }
        let lor = LorH1::<3>::new(&m, order).expect("LOR H1 discretization");
        run::<3>(m, lor.lor_mesh().clone(), order, lor.perm().to_vec(), lor.n_ho(), force_cg, visualization);
    } else {
        eprintln!("lor_elast: mesh file {mesh_file} has no 2-D/3-D mesh");
        std::process::exit(3);
    }
}

fn run<const D: usize>(
    mesh: Mesh<D>,
    lor_mesh: Mesh<D>,
    order: u8,
    lor_perm: Vec<u32>,
    lor_n_ho: usize,
    force_cg: bool,
    visualization: bool,
) {
    let total_timer = Instant::now();
    let dim = D;

    if *mesh.elem_tags.iter().max().unwrap_or(&0) < 2
        || *mesh.unique_boundary_tags().iter().max().unwrap_or(&0) < 2
    {
        eprintln!(
            "\nInput mesh should have at least two materials and \
             two boundary attributes! (See schematic in ex2.cpp)\n"
        );
        std::process::exit(3);
    }

    // 7. Vector finite element space: dim copies of a scalar H1 space
    //    (byNODES); the LOR space is built per component.
    let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
    let n_scalar = space.n_scalar_dofs();
    let n_dofs = space.n_dofs();
    println!("Number of finite element unknowns: {n_dofs}");
    print!("Assembling: ");
    std::io::stdout().flush().ok();

    // 8. Essential true dofs: boundary attribute 1 (the clamped wall).
    let scalar_dm = space.scalar_dof_manager().clone();
    let ess_scalar = boundary_dofs(&mesh, &scalar_dm, &[1]);
    let mut ess_tdof_list = Vec::with_capacity(ess_scalar.len() * dim);
    for c in 0..dim {
        for &d in &ess_scalar {
            ess_tdof_list.push((c * n_scalar + d as usize) as u32);
        }
    }
    let ess_vals = vec![0.0_f64; ess_tdof_list.len()];

    // 9. Linear form b: "pull down" boundary force on attribute 2, applied to
    //    the last component (VectorArrayCoefficient + PWConstCoefficient in
    //    MFEM; here assembled directly on the boundary trace).
    let assembly_timer = Instant::now();
    let mut b = assemble_boundary_traction(&mesh, &scalar_dm, order, n_scalar);
    print!("r.h.s. ... ");
    std::io::stdout().flush().ok();

    // 11. Bilinear form a: elasticity with piecewise-constant lambda/mu
    //     (attribute 1 = 50x stiffer, exactly lambda(0) = lambda(1)*50).
    let lambda_pw = fem_assembly::postproc::coefficient::PWConstCoeff::new([(1, 50.0), (2, 1.0)])
        .with_default(1.0);
    let mu_pw = fem_assembly::postproc::coefficient::PWConstCoeff::new([(1, 50.0), (2, 1.0)])
        .with_default(1.0);
    let integrator = ElasticityIntegrator::new(lambda_pw, mu_pw);
    let quad_order = 2 * order + 1;
    print!("matrix ... ");
    std::io::stdout().flush().ok();
    let mut a = Assembler::assemble_bilinear(&space, &[&integrator], quad_order);
    println!("done.");

    // 13. Block-diagonal LOR-AMG preconditioner: scalar component matrices on
    //     the P1 LOR space + one AMG hierarchy per component.
    let lam_of = |tag: i32| if tag == 1 { 50.0 } else { 1.0 };
    let mu_of = |tag: i32| if tag == 1 { 50.0 } else { 1.0 };
    let lor_blocks = assemble_lor_elasticity_blocks(&lor_mesh, &lam_of, &mu_of);
    // Mirror the MFEM/hypre BoomerAMG setup of lor_elast.cpp step 13(a):
    // classical Ruge-Stueben coarsening at the same strength threshold
    // (`SetStrengthThresh(0.25)`) with a symmetric relaxer in the spirit of
    // `SetRelaxType(16)`.  Symmetric Gauss-Seidel smoothing (3 pre + 3 post
    // sweeps) measured the most robust combination on the beam aspect-ratio
    // cells with the 50:1 material jump across refinement levels and orders.
    let amg_cfg = AmgConfig {
        strategy: fem_solver::lor::CoarsenStrategy::RugeStüben,
        smoother: fem_solver::lor::SmootherType::SymmetricGaussSeidel,
        pre_sweeps: 3,
        post_sweeps: 3,
        ..AmgConfig::default()
    };
    let prec = LorElasticityPrecond::build(
        &lor_blocks,
        &lor_perm,
        lor_n_ho,
        &ess_scalar,
        &amg_cfg,
    );

    // 12. Form the linear system (MFEM FormLinearSystem semantics).
    let mut x = vec![0.0_f64; n_dofs];
    form_linear_system(&mut a, &mut b, &mut x, &ess_tdof_list, &ess_vals);
    println!("Size of linear system: {}", space.n_dofs());
    let assembly_elapsed = assembly_timer.elapsed();

    // 14. Krylov solve.  MFEM lor_elast uses PCG + hypre BoomerAMG.  The
    //     linlvo AMG available to fem-rs is not reliably SPD on anisotropic
    //     beam meshes (negative curvature breaks the CG convergence test), so
    //     the port defaults to flexible GMRES, which tolerates the V-cycle
    //     preconditioner; `-cg` forces PCG for isotropic meshes.
    let linear_solve_timer = Instant::now();
    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 0.0,
        max_iter: 2500,
        verbose: true,
        ..SolverConfig::default()
    };
    let res = if force_cg {
        solve_pcg_precond(&a, &b, &mut x, &prec, &cfg)
    } else {
        solve_fgmres_precond(&a, &b, &mut x, 100, &prec, &cfg)
    }
    .unwrap_or_else(|e| panic!("lor_elast: Krylov solve failed: {e}"));
    let linear_solve_elapsed = linear_solve_timer.elapsed();
    if !res.converged {
        eprintln!(
            "lor_elast: Krylov solver did not converge in {} iterations (residual {:.3e})",
            res.iterations, res.final_residual
        );
    }
    let total_elapsed = total_timer.elapsed();

    // Elapsed times (MFEM lor_elast step 14).
    println!("Elapsed Times");
    println!("Assembly (s) = {}", assembly_elapsed.as_secs_f64());
    println!("Linear Solve (s) = {}", linear_solve_elapsed.as_secs_f64());
    println!("Total Solve (s) {}", total_elapsed.as_secs_f64());

    if visualization {
        eprintln!(
            "lor_elast: GLVis / ParaView output is not ported \
             (-vis accepted for CLI compatibility)"
        );
    }
}

/// Boundary traction linear form: `b_i = Int_{Gamma_2} f . phi_i dS` with
/// `f = (0, ..., 0, -1e-2)` (MFEM `VectorBoundaryLFIntegrator` +
/// `VectorArrayCoefficient`/`PWConstCoefficient`).
///
/// The trace of the scalar H1(`order`) space on a boundary face is the 1-D /
/// tensor Qp Lagrange basis at the GLL lattice points lying on that face, so
/// each face dof's basis is evaluated from its own lattice parameters, found
/// by matching the dof's physical coordinate (same technique as
/// `make_refined` / `LorH1`).
pub fn assemble_boundary_traction<const D: usize>(
    mesh: &Mesh<D>,
    dm: &DofManager,
    order: u8,
    n_scalar: usize,
) -> Vec<f64> {
    let dim = D;
    let comp = dim - 1; // pull force acts on the last component
    let force = -1.0e-2;
    let p = order as usize;

    // GLL parameters on [0, 1] (lattice positions) + Gauss-Legendre rule.
    let gll: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&t| 0.5 * (t + 1.0))
        .collect();
    let (gq, gw) = fem_element::quadrature::gauss_legendre_01(p + 1);

    let mut scale = 0.0_f64;
    for n in 0..mesh.n_nodes() as u32 {
        for d in 0..dim {
            scale = scale.max(mesh.node_coords(n)[d].abs());
        }
    }
    let tol = 1e-9 * scale.max(1.0);

    let mut b = vec![0.0_f64; dim * n_scalar];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) != 2 {
            continue;
        }
        let nodes = mesh.face_nodes(f);
        // Candidate trace dofs: vertices + edge dofs + face dofs on this face.
        let mut cand: Vec<u32> = nodes.to_vec();
        let m = nodes.len();
        for i in 0..m {
            let key = EdgeKey::new(nodes[i], nodes[(i + 1) % m]);
            if p == 2 {
                if let Some(&d) = dm.edge_dof_map.get(&key) {
                    cand.push(d);
                }
            } else if let Some(dofs) = dm.edge_pk_map.get(&key) {
                cand.extend_from_slice(dofs);
            }
        }
        if dim == 3 {
            let key = QuadFaceKey::new(nodes[0], nodes[1], nodes[2], nodes[3]);
            if let Some(dofs) = dm.quad_face_pk_map.get(&key) {
                cand.extend_from_slice(dofs);
            }
        }
        cand.sort_unstable();
        cand.dedup();

        // Physical position of every lattice point of this face, matched to a
        // candidate dof: lat[k] = (flat lattice index, dof).
        let mut lat: Vec<(usize, u32)> = Vec::new();
        if dim == 2 {
            let c0 = mesh.node_coords(nodes[0]);
            let c1 = mesh.node_coords(nodes[1]);
            for (k, &t) in gll.iter().enumerate() {
                let pt = [(1.0 - t) * c0[0] + t * c1[0], (1.0 - t) * c0[1] + t * c1[1]];
                if let Some(&d) = cand
                    .iter()
                    .find(|&&d| (0..2).all(|q| (dm.dof_coord(d)[q] - pt[q]).abs() <= tol))
                {
                    lat.push((k, d));
                }
            }
            // Integrate: weight = |dP/dt| * gw, basis = 1-D Lagrange.
            let len = ((c1[0] - c0[0]).powi(2) + (c1[1] - c0[1]).powi(2)).sqrt();
            for (jq, &t) in gq.iter().enumerate() {
                let w = gw[jq] * len;
                for &(k, d) in &lat {
                    b[comp * n_scalar + d as usize] +=
                        w * force * lagrange_1d(&gll, k, t);
                }
            }
        } else {
            debug_assert_eq!(m, 4, "3-D boundary faces must be quads");
            let c: Vec<[f64; 3]> = nodes.iter().map(|&n| {
                let cc = mesh.node_coords(n);
                [cc[0], cc[1], cc[2]]
            }).collect();
            // Bilinear face parametrisation: (0,0),(1,0),(1,1),(0,1).
            let p_at = |u: f64, v: f64| -> [f64; 3] {
                let mut x = [0.0_f64; 3];
                for d in 0..3 {
                    x[d] = (1.0 - u) * (1.0 - v) * c[0][d]
                        + u * (1.0 - v) * c[1][d]
                        + u * v * c[2][d]
                        + (1.0 - u) * v * c[3][d];
                }
                x
            };
            let p1 = p + 1;
            for j in 0..p1 {
                for i in 0..p1 {
                    let pt = p_at(gll[i], gll[j]);
                    if let Some(&d) = cand.iter().find(|&&d| {
                        (0..3).all(|q| (dm.dof_coord(d)[q] - pt[q]).abs() <= tol)
                    }) {
                        lat.push((j * p1 + i, d));
                    }
                }
            }
            for (jv, &v) in gq.iter().enumerate() {
                for (iu, &u) in gq.iter().enumerate() {
                    // |dP/du x dP/dv| via central differences of the bilinear
                    // map is unnecessary: compute analytically.
                    let du = [
                        -(1.0 - v) * c[0][0] + (1.0 - v) * c[1][0] + v * c[2][0] - v * c[3][0],
                        -(1.0 - v) * c[0][1] + (1.0 - v) * c[1][1] + v * c[2][1] - v * c[3][1],
                        -(1.0 - v) * c[0][2] + (1.0 - v) * c[1][2] + v * c[2][2] - v * c[3][2],
                    ];
                    let dv = [
                        -(1.0 - u) * c[0][0] - u * c[1][0] + u * c[2][0] + (1.0 - u) * c[3][0],
                        -(1.0 - u) * c[0][1] - u * c[1][1] + u * c[2][1] + (1.0 - u) * c[3][1],
                        -(1.0 - u) * c[0][2] - u * c[1][2] + u * c[2][2] + (1.0 - u) * c[3][2],
                    ];
                    let cx = du[1] * dv[2] - du[2] * dv[1];
                    let cy = du[2] * dv[0] - du[0] * dv[2];
                    let cz = du[0] * dv[1] - du[1] * dv[0];
                    let w = gw[iu] * gw[jv] * (cx * cx + cy * cy + cz * cz).sqrt();
                    for &(k, d) in &lat {
                        let (i, j) = (k % p1, k / p1);
                        let phi = lagrange_1d(&gll, i, u) * lagrange_1d(&gll, j, v);
                        b[comp * n_scalar + d as usize] += w * force * phi;
                    }
                }
            }
        }
    }
    b
}

/// `i`-th Lagrange basis value at `t` for the nodal set `xs`.
fn lagrange_1d(xs: &[f64], i: usize, t: f64) -> f64 {
    let mut v = 1.0_f64;
    for (k, &xk) in xs.iter().enumerate() {
        if k != i {
            v *= (t - xk) / (xs[i] - xk);
        }
    }
    v
}
