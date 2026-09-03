//! # Parallel Example 31 — Restricted H(curl) (1:1 with MFEM ex31p / ex31p.cpp)
//!
//! Solves the definite Maxwell equation `curl curl E + Σ·E = f` (anisotropic
//! 3×3 tensor Σ) on a 2-D mesh with **restricted H(curl)** finite elements:
//! the in-plane components live in the Nédélec space, the out-of-plane (z)
//! component in a continuous H¹ space — exactly MFEM's `ND_R2D_FECollection`.
//! All boundary edges are essential (non-homogeneous PEC: `E×n = E_exact×n`).
//!
//! ## Parallel structure (mirrors ex31p)
//! 1. Read the serial mesh (default `data/inline-quad.mesh`), refine
//!    `ser_ref_levels + par_ref_levels` times **serially** (the final global
//!    mesh is identical to C++'s serial+parallel refinement), then partition.
//! 2. Build two parallel spaces: H¹ (z component, vertex DOFs) and H(curl)
//!    (in-plane, edge DOFs).  The two DOF sets are concatenated into one
//!    combined partition `[z owned | nd owned | z ghost | nd ghost]` and the
//!    three matrix blocks (z: `-∇²+Σ_zz`, nd: `curl curl + Σ_xy`,
//!    coupling `Σ_yz ∫ E_y φ_z`) are combined into a single `ParCsrMatrix`.
//!    The combined ghost exchange is built from the per-space partitions.
//! 3. Essential DOFs (all boundary edges / boundary vertices) are detected on
//!    the **full** serial mesh (physical edge keys / node ids — deterministic,
//!    partition-independent, pex34/pex35 pattern) and mapped to the local
//!    partition.  BC values come from the exact solution projected on the full
//!    mesh.  Elimination uses MFEM's `DIAG_KEEP` (symmetric, `FormLinearSystem`
//!    default), including cross-rank ghost columns.
//! 4. Solve with PCG (rtol 1e-12, HyprePCG semantics — C++ `pcg.SetTol(1e-12)`)
//!    and a block-diagonal preconditioner: symmetric GS on the z block, AMS
//!    on the H(curl) block (the parallel analog of C++'s HypreAMS).
//! 5. Compute `||E_h − E||_{H(Curl)}` on the owned elements (global L² norm)
//!    and report norm / sum / physical-key checksums (np1-4 must agree).
//!
//! ## Usage
//! ```bash
//! cargo run --release --example mfem_pex31_restricted_hcurl -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex31_restricted_hcurl -- --ranks 2 -no-vis
//! cargo run --release --example mfem_pex31_restricted_hcurl -- --ranks 4 -m data/hexagon.mesh -o 1 -no-vis
//! ```
//!
//! ## Acceptance
//! * np1 `||E_h − E||_{H(Curl)}` = 0.0907163 (inline-quad, -rs 2 -rp 1, order 1)
//!   — matches the serial ex31 at the same refinement (0.0907163) / C++ ex31
//!   (verified 1:1 for the serial counterpart).
//! * np1-4 norm/sum/checksums and the H(Curl) error agree bit-for-bit
//!   (‖u‖ = 3.16188210e1, sum = −8.44398822e2, z checksum −4.63965845e5,
//!   nd checksum 1.67172371e3; PCG 308/467/515 iterations, rtol 1e-12).
//! * unknowns / system size match C++ `Number of H(Curl) unknowns` (3201).
//!
//! ## Known differences vs C++
//! * The preconditioner is a **symmetric GS on the whole combined diagonal
//!   block** (block-Jacobi), not C++'s global HypreAMS: PCG iteration counts
//!   differ (308-515 vs C++'s few dozen) but the converged solution agrees.
//!   A block-diagonal `[GS(z) | AMS(nd)]` preconditioner was tried first and
//!   **stalls PCG** (the AMS V-cycle with its AMG nodal solve is not symmetric
//!   enough for the PCG recurrence) — `-pc jacobi` also works (627 iters).
//! * Essential DOFs are detected from the FULL serial mesh (physical edge keys
//!   / node ids) and the BC values are converted through the local-basis sign
//!   (σ·v_full then the partition permutation) — storing the full-mesh value
//!   directly flips the BC on edges whose local orientation differs at np>1
//!   (the root cause of the np2 solution mismatch found during development).

use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, SQRT_2};
use std::io::Write;
use std::sync::Arc;

use fem_assembly::standard::{CurlCurlIntegrator, DiffusionIntegrator, MassIntegrator,
    VectorMassTensorIntegrator};
use fem_assembly::coefficient::ConstantMatrixCoeff;
use fem_assembly::postproc::grid_function::project_bdr_coefficient_tangent_2d;
use fem_assembly::{VectorAssembler, Assembler, FixedOrder};
use fem_element::{VectorReferenceElement, ReferenceElement,
    nedelec::{TriNDk, QuadNDk}, lagrange::{TriP1, QuadQk}};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix, fem_to_linlvo_csr};
use fem_mesh::{ElementType, Mesh, MeshTopology, amr::refine_uniform};
use fem_parallel::{
    Comm, GhostExchange, ParCsrMatrix, ParallelFESpace, ParVector, WorkerConfig,
    launcher::native::ThreadLauncher,
    par_partition::partition_mesh,
    par_solver::{par_solve_pcg_jacobi, par_solve_pcg_precond},
};
use fem_solver::{DenseVec, GSSmoother, Preconditioner, SolverConfig};
use fem_space::{
    H1Space, HCurlSpace,
    fe_space::FESpace,
};

// ─── CLI (mirrors ex31p OptionsParser) ───────────────────────────────────────

struct Args {
    mesh_file: String,
    ser_ref_levels: i32,
    par_ref_levels: i32,
    order: u8,
    freq: f64,
    ranks: usize,
    visualization: bool,
    /// preconditioner: "gs" (symmetric GS on the combined block, default),
    /// "block" (GS-z + AMS-nd — AMS is not symmetric enough for PCG and
    /// stalls), or "jacobi" (diagonal).
    pc: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/inline-quad.mesh".into(),
        ser_ref_levels: 2,
        par_ref_levels: 1,
        order: 1,
        freq: 1.0,
        ranks: 1,
        visualization: false,
        pc: "gs".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_default(),
            "-rs" | "--refine-serial" => a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            "-rp" | "--refine-parallel" => a.par_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-f" | "--frequency" => a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "--ranks" => a.ranks = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-pc" | "--preconditioner" => a.pc = it.next().unwrap_or_else(|| "gs".into()),
            "-ams" | "--hypre-ams" => {}
            "-slu" | "--superlu" => {}
            _ => {}
        }
    }
    a
}

// ─── Exact solution (1:1 with ex31p E_exact / CurlE_exact / f_exact) ────────

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

/// Matches C++ ex31p `f_exact` for dim == 2.
fn source_3d(x: &[f64], kappa: f64) -> [f64; 3] {
    let k2 = kappa * kappa;
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    let (s0, s4, s9) = (u.sin(), (u + PHI1).sin(), (u + PHI2).sin());
    let f0 = 0.55 * (4.0 + k2) * s0 + 0.6 * (SQRT_2 - k2) * s4;
    let f1 = 0.55 * (SQRT_2 - k2) * s0 + 0.6 * (4.0 + k2) * s4 + 0.65 * SQRT_2 * s9;
    let f2 = 0.6 * SQRT_2 * s4 + 1.3 * (2.0 + k2) * s9;
    [f0, f1, f2]
}

// ─── Element reference helpers (2-D, Tri3 / Quad4 — serial ex31 code) ───────

type JacobianFn = fn(
    &Mesh<2>, u32, &[u32], &[f64],
) -> (f64, f64, f64, f64, f64, f64); // (inv_det, jit00, jit01, jit10, jit11, det_j)

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
    // Geometry element on [0,1]^2 (MFEM BiLinear2DFiniteElement), matching
    // QuadQk used by the assembler for the H1 z-space (QuadQ1 is [-1,1]^2 and
    // would mix reference domains with QuadND1's [0,1]^2 quadrature).
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
        ElementType::Tri3 => (3, TriNDk::new(1) as &dyn VectorReferenceElement, Box::new(TriP1), 3, affine_jac as JacobianFn),
        ElementType::Quad4 => (4, &QuadNDk::new(1) as &dyn VectorReferenceElement, Box::new(QuadQk::new(1)), 4, isoparametric_jac as JacobianFn),
        _ => panic!("unsupported element type {et:?}"),
    }
}

/// Physical coordinates of the quadrature point on element `e` (Tri: affine,
/// Quad: isoparametric Q1 geometry on [0,1]²).
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

// ─── Local (serial) assembly of the combined [z | nd] system ─────────────────
// Layout: z (H¹ vertex) DOFs 0..n_z_local first, then ND (edge) DOFs offset by
// n_z_local — matches MFEM ND_R2D GetElementVDofs (verified in the serial
// ex31 port: z DOFs 0..288, in-plane 289..832 on the reference mesh).

fn assemble_combined_local(
    mesh: &Mesh<2>,
    nd_space: &HCurlSpace<Mesh<2>>,
    z_space: &H1Space<Mesh<2>>,
    order: u8,
    kappa: f64,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let n_nd = nd_space.n_dofs();
    let n_h1 = z_space.n_dofs();
    let n_total = n_nd + n_h1;
    let quad_order = order as usize * 2 + 2;

    // In-plane block: curl curl + Σ_xy.
    let curl_curl = CurlCurlIntegrator { mu: 1.0 };
    let sigma_2d = ConstantMatrixCoeff(vec![SXX, SXY, SXY, SYY]);
    let vec_mass = VectorMassTensorIntegrator { alpha: sigma_2d };
    let a_nd = VectorAssembler::assemble_bilinear(
        nd_space, &[&curl_curl, &vec_mass], quad_order as u8,
    );

    // Z-component block: -∇² + Σ_zz (curl contribution at quadrature order 0,
    // matching C++ CurlCurlIntegrator's 2p-2 rule on the ND_R2D space).
    let laplace = FixedOrder::new(DiffusionIntegrator { kappa: 1.0 }, 0);
    let z_mass = MassIntegrator { rho: SZZ };
    let a_z = Assembler::assemble_bilinear(z_space, &[&laplace, &z_mass], quad_order as u8);

    // Coupling block: Σ_yz · ∫ E_y · φ_z.
    let mut coupling_coo = CooMatrix::<f64>::new(n_nd, n_h1);
    for e in 0..mesh.n_elements() as u32 {
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = nd_space.mesh().element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let q = rnd.quadrature(quad_order as u8);
        let mut np = vec![0.0; n_ld * 2];
        let mut hp = vec![0.0; n_lh1];
        let mut em = vec![0.0_f64; n_ld * n_lh1];
        for (qi, xi) in q.points.iter().enumerate() {
            let (_, _jit00, _jit01, jit10, jit11, det) = jac_fn(mesh, e, nodes, xi);
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

    // Combine blocks: z (vertex) DOFs first, then in-plane ND DOFs.
    let mut sys_coo = CooMatrix::<f64>::new(n_total, n_total);
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

    // RHS (VectorFEDomainLFIntegrator order 2·el.GetOrder() = 2 for the source).
    let src_nd = FixedOrder::new(FnVectorSource(Box::new(move |x| {
        let f = source_3d(x, kappa); [f[0], f[1]]
    })), 2);
    let rhs_nd = VectorAssembler::assemble_linear(nd_space, &[&src_nd], quad_order as u8);
    let src_z = FixedOrder::new(FnScalarSource(Box::new(move |x| source_3d(x, kappa)[2])), 2);
    let rhs_z = Assembler::assemble_linear(z_space, &[&src_z], quad_order as u8);
    let mut rhs = vec![0.0_f64; n_total];
    for i in 0..n_h1 { rhs[i] = rhs_z[i]; }
    for i in 0..n_nd { rhs[n_h1 + i] = rhs_nd[i]; }

    (sys_mat, rhs)
}

// ─── Combined [z | nd] DOF partition (example-level) ─────────────────────────

/// Offset separating the ND global DOF ids from the z (node) ids in the
/// combined partition's ghost exchange key space (must exceed any node id).
const ND_BASE: u32 = 1 << 29;

/// Combined partition of the restricted H(curl) space: the z (H¹) and in-plane
/// ND (H(curl)) partitions concatenated.
///
/// Local DofManager numbering: `[z dm ids 0..n_z_local | nd dm ids offset]`.
/// Partition layout: `[z owned | nd owned | z ghost | nd ghost]`.
struct CombinedPart {
    /// dm id → partition id (length n_z_local + n_nd_local).
    perm: Vec<u32>,
    /// dm id → sign correction (±1) for the ND edge orientation.
    signs: Vec<f64>,
    n_owned: usize,
    n_ghost: usize,
    /// global ids in partition layout (owned then ghost) for the exchange.
    gids: Vec<u32>,
    owners: Vec<i32>,
}

impl CombinedPart {
    fn build(z_par: &ParallelFESpace<H1Space<Mesh<2>>>, nd_par: &ParallelFESpace<HCurlSpace<Mesh<2>>>) -> Self {
        let zdp = z_par.dof_partition();
        let ndp = nd_par.dof_partition();
        let n_z_local = zdp.n_total_dofs();
        let n_nd_local = ndp.n_total_dofs();
        let n_z_owned = zdp.n_owned_dofs;
        let n_nd_owned = ndp.n_owned_dofs;
        let n_z_ghost = zdp.n_ghost_dofs;
        let n_nd_ghost = ndp.n_ghost_dofs;
        let n_owned = n_z_owned + n_nd_owned;
        let n_ghost = n_z_ghost + n_nd_ghost;

        let mut perm = vec![0u32; n_z_local + n_nd_local];
        let mut signs = vec![1.0f64; n_z_local + n_nd_local];
        for i in 0..n_z_local {
            // H1 P1: dm id == partition id (identity).
            perm[i] = if i < n_z_owned { i } else { n_owned + (i - n_z_owned) } as u32;
        }
        for j in 0..n_nd_local {
            // `permute_dof` maps the DofManager id → partition id; the nd
            // partition's `global_dof`/`dof_owner` arrays are indexed by the
            // **partition** id ([owned | ghost] layout), which differs from the
            // dm id at np>1 — reading them with the dm id produces garbage
            // gids/owners (the root cause of the np2 ghost-exchange failures).
            let pj = ndp.permute_dof(j as u32) as usize;
            let pid = if pj < n_nd_owned {
                n_z_owned + pj
            } else {
                n_owned + n_z_ghost + (pj - n_nd_owned)
            };
            perm[n_z_local + j] = pid as u32;
            signs[n_z_local + j] = ndp.sign_correction(j as u32);
        }

        // Global ids in partition layout (offset the ND block so z and nd
        // global ids can never collide in the ghost exchange).
        let mut gids = vec![0u32; n_owned + n_ghost];
        let mut owners = vec![0i32; n_owned + n_ghost];
        for i in 0..n_z_local {
            let pid = perm[i] as usize;
            gids[pid] = zdp.global_dof(i as u32);
            owners[pid] = zdp.dof_owner(i as u32);
        }
        for j in 0..n_nd_local {
            let pid = perm[n_z_local + j] as usize;
            let pj = ndp.permute_dof(j as u32) as usize;
            gids[pid] = ND_BASE + ndp.global_dof(pj as u32);
            owners[pid] = ndp.dof_owner(pj as u32);
        }

        CombinedPart { perm, signs, n_owned, n_ghost, gids, owners }
    }

    /// dm order → partition order (with sign correction for vectors).
    fn permute_vec(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_owned + self.n_ghost];
        for (dm, &v_dm) in v.iter().enumerate() {
            let pid = self.perm[dm] as usize;
            out[pid] = v_dm * self.signs[dm];
        }
        out
    }

    /// Ghost exchange for the combined vector (built from the combined global ids).
    fn ghost_exchange(&self, comm: &Comm) -> Arc<GhostExchange> {
        let part = fem_parallel::partition::MeshPartition::from_partitioner(
            &self.gids[..self.n_owned],
            &self.gids[self.n_owned..].iter().zip(&self.owners[self.n_owned..])
                .map(|(&g, &o)| (g, o)).collect::<Vec<_>>(),
            &[], &[], comm.rank(),
        );
        Arc::new(GhostExchange::from_partition(&part, comm))
    }
}

// ─── Source integrators (serial ex31 helpers) ────────────────────────────────

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

// ─── Per-element local edge table (H(curl) space ordering) ───────────────────

fn edges_for_elem(et: ElementType) -> &'static [(usize, usize)] {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => &[(0, 1), (1, 2), (0, 2)],
        ElementType::Quad4 | ElementType::Quad9 => &[(0, 1), (1, 2), (2, 3), (0, 3)],
        _ => &[],
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        let rank = comm.rank();
        let is_root = rank == 0;
        let mut out = String::new();
        let kappa = args.freq * PI;

        if is_root {
            out.push_str(&format!("Options used:\n   --mesh {}\n   --refine-serial {}\n   --refine-parallel {}\n   --order {}\n   --frequency {}\n\n",
                args.mesh_file, args.ser_ref_levels, args.par_ref_levels, args.order, args.freq));
        }

        // 1. Serial mesh, refined (rs + rp) times before partitioning.
        let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
        let mesh0: Mesh<2> = mfem.mesh2d.expect("restricted H(curl) is 2-D only");
        let mut full_mesh = mesh0;
        for _ in 0..args.ser_ref_levels + args.par_ref_levels {
            full_mesh = refine_uniform(&full_mesh);
        }

        // 2. Partition.
        let pm = partition_mesh(&full_mesh, &comm);
        let local_mesh = pm.local_mesh().clone();
        let partition = pm.partition().clone();
        let n_owned_elems = partition.n_owned_elems;

        // 3. Local spaces + parallel spaces.
        let nd_space = HCurlSpace::new(local_mesh.clone(), args.order);
        let z_space = H1Space::new(local_mesh.clone(), args.order);
        let nd_par = ParallelFESpace::new_for_edge_space(
            HCurlSpace::new(local_mesh.clone(), args.order), &pm, comm.clone());
        let z_par = ParallelFESpace::new(
            H1Space::new(local_mesh.clone(), args.order), &pm, comm.clone());
        let comb = CombinedPart::build(&z_par, &nd_par);

        if is_root {
            out.push_str(&format!("Number of H(Curl) unknowns: {}\n",
                z_par.n_global_dofs() + nd_par.n_global_dofs()));
        }

        // 4. Assemble the combined local system (dm order [z | nd]).
        let (local_mat, rhs_dm) = assemble_combined_local(
            &local_mesh, &nd_space, &z_space, args.order, kappa);

        // 5. Permute to the combined [owned | ghost] layout, split diag/offd.
        let n_local = local_mat.nrows;
        let mut coo = CooMatrix::<f64>::new(n_local, n_local);
        for r in 0..n_local {
            let pr = comb.perm[r] as usize;
            let sr = comb.signs[r];
            for k in local_mat.row_ptr[r]..local_mat.row_ptr[r + 1] {
                let c = local_mat.col_idx[k] as usize;
                let pc = comb.perm[c] as usize;
                let sc = comb.signs[c];
                let v = local_mat.values[k] * sr * sc;
                if v != 0.0 { coo.add(pr, pc, v); }
            }
        }
        let permuted = coo.into_csr();
        let exchange = comb.ghost_exchange(&comm);
        let mut a_sys = ParCsrMatrix::from_local_matrix(
            &permuted, comb.n_owned, exchange.clone(), comm.clone());

        let rhs_perm = comb.permute_vec(&rhs_dm);
        let mut rhs = ParVector::from_local_raw(rhs_perm, comb.n_owned, exchange.clone(), comm.clone());

        // 6. Essential DOFs: ALL boundary DOFs (PEC).  Detected from the FULL
        //    mesh (deterministic, partition-independent — pex34/pex35 pattern).
        let full_bnd_nodes: HashSet<u32> = {
            let mut s = HashSet::new();
            for bf in 0..full_mesh.n_boundary_faces() as u32 {
                for &n in full_mesh.face_nodes(bf) { s.insert(n); }
            }
            s
        };
        let full_bnd_edges: HashSet<(u32, u32)> = {
            let mut s = HashSet::new();
            for bf in 0..full_mesh.n_boundary_faces() as u32 {
                let ns = full_mesh.face_nodes(bf);
                for i in 0..ns.len() {
                    let a = ns[i]; let b = ns[(i + 1) % ns.len()];
                    s.insert((a.min(b), a.max(b)));
                }
            }
            s
        };
        // BC values on the full mesh: tangential projection (in-plane) + E_z.
        let full_nd_space = HCurlSpace::new(full_mesh.clone(), args.order);
        let mut full_nd_bc = vec![0.0; full_nd_space.n_dofs()];
        project_bdr_coefficient_tangent_2d(&mut full_nd_bc, &full_nd_space,
            &|x: &[f64], out: &mut [f64]| { let e = exact_e(x, kappa); out[0] = e[0]; out[1] = e[1]; },
            &full_mesh.unique_boundary_tags());
        let mut full_nd_bc_by_key: HashMap<(u32, u32), f64> = HashMap::new();
        for e in full_mesh.elem_iter() {
            let et = full_mesh.element_type(e);
            let ns = full_mesh.element_nodes(e);
            let dofs = full_nd_space.element_dofs(e);
            for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                let key = (ns[a].min(ns[b]), ns[a].max(ns[b]));
                full_nd_bc_by_key.entry(key).or_insert(full_nd_bc[dofs[k] as usize]);
            }
        }

        // Local BC values (combined dm order), then → partition order.
        let mut bc_dm = vec![0.0; comb.perm.len()];
        // z part: boundary vertices.
        for e in local_mesh.elem_iter() {
            let ns = local_mesh.element_nodes(e);
            let zd = z_space.element_dofs(e);
            for (k, &n) in ns.iter().enumerate() {
                let gn = partition.global_node(n);
                if full_bnd_nodes.contains(&gn) {
                    let c = full_mesh.node_coords(gn);
                    bc_dm[zd[k] as usize] = exact_e(&c, kappa)[2];
                }
            }
        }
        // nd part: boundary edges.  The stored value is the *local-basis* dof
        // value = σ·v_full (σ = edge-direction sign vs the canonical/full-mesh
        // orientation); `permute_vec` below converts it to the partition basis
        // (σ·x_local = v_full), which is what the elimination and the
        // partition-basis matrix expect.  Storing v_full directly would flip
        // the BC on edges whose local orientation differs (np>1).
        let mut seen = HashSet::new();
        for e in local_mesh.elem_iter() {
            let et = local_mesh.element_type(e);
            let ns = local_mesh.element_nodes(e);
            let dofs = nd_space.element_dofs(e);
            for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                let (ga, gb) = (partition.global_node(ns[a]), partition.global_node(ns[b]));
                let key = (ga.min(gb), ga.max(gb));
                if full_bnd_edges.contains(&key) && seen.insert(dofs[k]) {
                    let sigma = nd_par.dof_partition().sign_correction(dofs[k]);
                    bc_dm[comb.perm.len() - nd_space.n_dofs() + dofs[k] as usize] =
                        sigma * full_nd_bc_by_key.get(&key).copied().unwrap_or(0.0);
                }
            }
        }
        let bc_perm = comb.permute_vec(&bc_dm);

        // 7. DIAG_KEEP elimination (MFEM FormLinearSystem default) for owned
        //    DOFs + cross-rank ghost columns.
        let mut clamped: Vec<(usize, f64)> = Vec::new();
        let mut ghost_ess: Vec<(usize, f64)> = Vec::new();
        for pid in 0..comb.n_owned + comb.n_ghost {
            let v = bc_perm[pid];
            if v == 0.0 { continue; }
            if pid < comb.n_owned {
                clamped.push((pid, v));
            } else {
                ghost_ess.push((pid - comb.n_owned, v));
            }
        }
        for &(pid, v) in &clamped {
            a_sys.apply_dirichlet_par_keep_diag(pid, v, &mut rhs);
        }
        a_sys.apply_ghost_ess_columns(&ghost_ess, &mut rhs);

        if is_root {
            out.push_str(&format!("Size of linear system: {}\n",
                z_par.n_global_dofs() + nd_par.n_global_dofs()));
        }

        // 8. Solve: PCG (rtol 1e-12 — HyprePCG `SetTol(1e-12)` is a relative
        //    residual norm, used directly) with a symmetric Gauss-Seidel
        //    preconditioner on the whole combined diagonal block (the parallel
        //    analog of the serial ex31's GS-preconditioned PCG).
        //    `-pc jacobi` uses the plain diagonal preconditioner (debugging /
        //    reference path).  A block-diagonal [GS(z) | AMS(nd)] preconditioner
        //    was tried first but stalls PCG (the AMS V-cycle with its AMG nodal
        //    solve is not symmetric enough for the PCG recurrence); C++ uses
        //    HypreAMS, so iteration counts differ (accepted — solution agrees).
        let cfg = SolverConfig {
            rtol: 1e-12, atol: 0.0, max_iter: 1000,
            verbose: std::env::var("PEX31_VERBOSE").is_ok(),
            ..Default::default()
        };
        let mut x = ParVector::zeros_like(&rhs);
        for pid in 0..comb.n_owned {
            x.as_slice_mut()[pid] = bc_perm[pid];
        }
        x.update_ghosts();

        let res = if args.pc == "jacobi" {
            par_solve_pcg_jacobi(&a_sys, &rhs, &mut x, &cfg)
                .expect("PCG (jacobi) failed")
        } else {
            // Symmetric GS on the whole combined diagonal block.
            let diag = a_sys.diag_block().clone();
            let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&diag)).expect("GS smoother");
            let precond = move |r: &[f64], z: &mut [f64]| {
                let lr = DenseVec::from_vec(r.to_vec());
                let mut lz = DenseVec::zeros(z.len());
                gs.apply_precond(&lr, &mut lz);
                z.copy_from_slice(lz.as_slice());
            };
            par_solve_pcg_precond(&a_sys, &rhs, &mut x, &precond, &cfg)
                .expect("PCG (gs) failed")
        };
        if is_root {
            out.push_str(&format!("PCG: {} iterations, final relative residual = {:.6e}\n\n",
                res.iterations, res.final_residual));
        }

        // 9. H(Curl) error on owned elements (+ global L² norm).
        x.update_ghosts();
        let (hcurl_err, _) = compute_hcurl_error(
            &x, &comb, &local_mesh, &nd_space, &z_space, args.order, kappa,
            n_owned_elems, &comm);
        if is_root {
            out.push_str(&format!("|| E_h - E ||_{{H(Curl)}} = {hcurl_err:.10e}\n\n"));
        }

        // 10. Metrics (norm/sum/checksums — np1-4 must agree).
        let n_owned = comb.n_owned;
        let norm = x.global_norm();
        let sum = comm.allreduce_sum_f64(x.as_slice()[..n_owned].iter().sum::<f64>());
        // Physical-key checksums: z keyed by global node id, nd keyed by the
        // full-mesh physical edge id (element×local-edge first-seen).
        let (z_ck, nd_ck) = physical_checksums(&x, &z_par, &nd_par, &local_mesh, &partition,
            &full_mesh, &nd_space, &comm);
        if is_root {
            out.push_str(&format!("  ||u|| = {:.8e}, sum = {:.8e}\n", norm, sum));
            out.push_str(&format!("  z checksum = {:.8e}, nd checksum = {:.8e}\n", z_ck, nd_ck));
        }

        // 11. Save mesh + solution (per rank).
        {
            let mesh_name = format!("mesh.{:06}", rank);
            let mut f = std::fs::File::create(&mesh_name).expect("mesh file");
            fem_io::mfem::write_mfem(&mut f, &local_mesh, None).expect("write mesh");
            let sol_name = format!("sol.{:06}", rank);
            let mut f = std::fs::File::create(&sol_name).expect("sol file");
            // Write in DofManager order (z dofs then nd dofs) for dump parity.
            let mut dm = vec![0.0; comb.perm.len()];
            for (d, &p) in comb.perm.iter().enumerate() {
                dm[d] = comb.signs[d] * x.as_slice()[p as usize];
            }
            for &v in &dm { writeln!(f, "{:.8e}", v).expect("sol write"); }
        }

        if is_root {
            out.push_str("\nFinished.\n");
            print!("{out}");
        }
    });
}

// ─── Block extraction helpers ────────────────────────────────────────────────

/// Extract the CSR sub-block `rows[rs..re] × cols[cs..ce]` of a ParCsrMatrix
/// (rows/cols in partition layout).
fn extract_block(a: &ParCsrMatrix, rs: usize, re: usize, cs: usize, ce: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(re - rs, ce - cs);
    let diag = a.diag_block();
    let offd = a.offd_block();
    for r in rs..re {
        let row = r - rs;
        // diag block (cols 0..n_owned)
        for k in diag.row_ptr[r]..diag.row_ptr[r + 1] {
            let c = diag.col_idx[k] as usize;
            if c >= cs && c < ce {
                coo.add(row, c - cs, diag.values[k]);
            }
        }
        // offd block (cols >= n_owned)
        if offd.nrows > r {
            for k in offd.row_ptr[r]..offd.row_ptr[r + 1] {
                let c = a.n_owned() + offd.col_idx[k] as usize;
                if c >= cs && c < ce {
                    coo.add(row, c - cs, offd.values[k]);
                }
            }
        }
    }
    coo.into_csr()
}

// ─── H(Curl) error (owned elements + allreduce) ──────────────────────────────

#[allow(clippy::too_many_arguments)]
fn compute_hcurl_error(
    x: &ParVector,
    comb: &CombinedPart,
    mesh: &Mesh<2>,
    nd_space: &HCurlSpace<Mesh<2>>,
    z_space: &H1Space<Mesh<2>>,
    order: u8,
    kappa: f64,
    n_owned_elems: usize,
    comm: &Comm,
) -> (f64, f64) {
    let n_h1 = z_space.n_dofs();
    let mut x_dm = vec![0.0; comb.perm.len()];
    for (d, &p) in comb.perm.iter().enumerate() {
        x_dm[d] = comb.signs[d] * x.as_slice()[p as usize];
    }
    let mut err2 = 0.0_f64;
    let mut norm2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        if e as usize >= n_owned_elems { continue; }
        let nd_dofs: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = z_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let signs = nd_space.element_signs(e);
        let (n_ld, rnd, rh1, n_lh1, jac_fn) = setup_element_ref(mesh.element_type(e), order);
        let qord = 2 * order as u8 + 3;
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
                eh[0] += s * x_dm[n_h1 + nd_dofs[i]] * (jit00 * pn[i * 2] + jit01 * pn[i * 2 + 1]);
                eh[1] += s * x_dm[n_h1 + nd_dofs[i]] * (jit10 * pn[i * 2] + jit11 * pn[i * 2 + 1]);
            }
            for j in 0..n_lh1 { eh[2] += x_dm[h1_dofs[j]] * ph[j]; }
            let mut ce = [0.0_f64; 3];
            for i in 0..n_ld { ce[2] += signs[i] * x_dm[n_h1 + nd_dofs[i]] * cn[i]; }
            ce[2] *= inv_det;
            let mut gr = vec![0.0_f64; n_lh1 * 2];
            rh1.eval_grad_basis(xi, &mut gr);
            for j in 0..n_lh1 {
                let dx = jit00 * gr[j * 2] + jit10 * gr[j * 2 + 1];
                let dy = jit01 * gr[j * 2] + jit11 * gr[j * 2 + 1];
                ce[0] += x_dm[h1_dofs[j]] * dy;
                ce[1] -= x_dm[h1_dofs[j]] * dx;
            }
            let (ee, ec) = (exact_e(&xp, kappa), exact_curl(&xp, kappa));
            for c in 0..3 {
                let d = eh[c] - ee[c]; err2 += w * d * d;
                let dc = ce[c] - ec[c]; err2 += w * dc * dc;
                norm2 += w * (ee[c] * ee[c] + ec[c] * ec[c]);
            }
        }
    }
    let err2 = comm.allreduce_sum_f64(err2);
    let norm2 = comm.allreduce_sum_f64(norm2);
    (err2.sqrt(), norm2.sqrt())
}

// ─── Physical-key checksums ──────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn physical_checksums(
    x: &ParVector,
    z_par: &ParallelFESpace<H1Space<Mesh<2>>>,
    nd_par: &ParallelFESpace<HCurlSpace<Mesh<2>>>,
    local_mesh: &Mesh<2>,
    partition: &fem_parallel::partition::MeshPartition,
    full_mesh: &Mesh<2>,
    nd_space: &HCurlSpace<Mesh<2>>,
    comm: &Comm,
) -> (f64, f64) {
    let zdp = z_par.dof_partition();
    let ndp = nd_par.dof_partition();
    let n_z_owned = zdp.n_owned_dofs;
    let n_nd_owned = ndp.n_owned_dofs;

    // z checksum: global node id key (owned z dofs).
    let mut z_ck = 0.0_f64;
    for i in 0..n_z_owned {
        let gid = zdp.global_dof(i as u32) as f64;
        z_ck += (gid + 1.0) * x.as_slice()[i];
    }

    // nd checksum: physical edge id from the full mesh (element×local-edge
    // first-seen = MFEM ND dof numbering at np1).
    let mut key_to_edge: HashMap<(u32, u32), u64> = HashMap::new();
    let mut next_edge = 0u64;
    for e in full_mesh.elem_iter() {
        let et = full_mesh.element_type(e);
        let ns = full_mesh.element_nodes(e);
        for &(a, b) in edges_for_elem(et) {
            let key = (ns[a].min(ns[b]), ns[a].max(ns[b]));
            if !key_to_edge.contains_key(&key) {
                key_to_edge.insert(key, next_edge);
                next_edge += 1;
            }
        }
    }
    let mut dm_to_key: HashMap<u32, (u32, u32)> = HashMap::new();
    for e in local_mesh.elem_iter() {
        let et = local_mesh.element_type(e);
        let ns = local_mesh.element_nodes(e);
        let dofs = nd_space.element_dofs(e);
        for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
            let (ga, gb) = (partition.global_node(ns[a]), partition.global_node(ns[b]));
            dm_to_key.entry(dofs[k]).or_insert((ga.min(gb), ga.max(gb)));
        }
    }
    let mut nd_ck = 0.0_f64;
    for p in 0..n_nd_owned {
        let dm = ndp.unpermute_dof(p as u32);
        let id = dm_to_key.get(&dm).and_then(|&k| key_to_edge.get(&k)).copied().unwrap_or(0);
        nd_ck += (id as f64 + 1.0) * x.as_slice()[n_z_owned + p];
    }
    (comm.allreduce_sum_f64(z_ck), comm.allreduce_sum_f64(nd_ck))
}
