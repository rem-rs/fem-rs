//! # Parallel Example 13 — Maxwell Cavity Eigenvalue (1:1 with MFEM ex13p)
//!
//! Solves the Maxwell (electromagnetic) eigenvalue problem
//! `curl curl E = λ E` with homogeneous Dirichlet BC `E×n = 0` on all
//! boundary attributes, using a Nédélec (H(curl)) FE space of the given
//! order, in 2D or 3D.
//!
//! MFEM ex13p uses the HYPRE AME solver (LOBPCG + AMS internally) with
//! `SetMaxIter(100)`, `SetTol(1e-8)`, `SetNumModes(nev)`.  This port uses
//! [`fem_parallel::par_lobpcg`] (serial `lobpcg_projected` structure) with a
//! block-diagonal Jacobi / AMG preconditioner, essential-dof zeroing and the
//! `nullspace_skip` mechanism for the curl-curl gradient nullspace — the
//! parallel analog of HYPRE AME's div-free handling.
//!
//! **Status**: on small meshes (e.g. `-rs 0 -rp 0`) the eigenvalues match the
//! serial reference to 10+ digits (11.678…/12.404…/13.539…/14.977…/16.581… on
//! beam-tet).  The refined default (`-rs 2 -rp 1`) is still limited by the
//! parallel LOBPCG's convergence on the clustered low spectrum with the large
//! gradient nullspace (modes 2+ stall at a few % residual) — see HANDOVER.
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex13_parallel_eigenvalue -- --ranks {1,2,4} -rs 0 -rp 0 -no-vis
//! cargo run --release --example mfem_pex13_parallel_eigenvalue -- -m data/star.mesh -rs 0 -rp 0 --ranks 2
//! ```

use std::collections::HashSet;

use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::{Mesh, MeshTopology, refine_uniform, refine_uniform_3d};
use fem_parallel::{
    ParCsrMatrix, ParallelFESpace, ParallelMesh, ParVector, par_lobpcg,
    par_partition::partition_mesh, par_vector_assembler::ParVectorAssembler,
    launcher::{native::ThreadLauncher, WorkerConfig},
};
use fem_space::{H1Space, HCurlSpace, fe_space::FESpace};

fn main() {
    let args = parse_args();
    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        run_pex13(comm, &args);
    });
}

struct Args {
    mesh_file: String,
    ser_ref_levels: usize,
    par_ref_levels: usize,
    order: u8,
    nev: usize,
    ranks: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/beam-tet.mesh".into(), // ex13p default
        ser_ref_levels: 2,
        par_ref_levels: 1,
        order: 1,
        nev: 5,
        ranks: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_else(|| a.mesh_file.clone()),
            "-rs" | "--refine-serial" => {
                a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
            }
            "-rp" | "--refine-parallel" => {
                a.par_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
            }
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-n" | "--num-eigs" => a.nev = it.next().and_then(|v| v.parse().ok()).unwrap_or(5),
            "--ranks" => a.ranks = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-nc" | "-c" | "-d" | "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    a
}

// ─── Mesh-type-independent core ───────────────────────────────────────────────

/// Edge table for each element type (2D + 3D), indexed into `element_nodes`.
fn edges_for_elem(et: fem_mesh::element_type::ElementType) -> &'static [(usize, usize)] {
    use fem_mesh::element_type::ElementType;
    match et {
        ElementType::Tri3 | ElementType::Tri6 => &[(0, 1), (1, 2), (2, 0)],
        ElementType::Quad4 | ElementType::Quad9 => &[(0, 1), (1, 2), (2, 3), (3, 0)],
        ElementType::Tet4 | ElementType::Tet10 => {
            &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        }
        ElementType::Hex8 | ElementType::Hex20 => &[
            (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ],
        ElementType::Prism6 => &[
            (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5),
        ],
        _ => &[],
    }
}

/// Full-mesh boundary edge keys (sorted global vertex pairs) — deterministic,
/// partition-independent (pex34/pex35 pattern).
fn full_boundary_edge_keys(mesh: &impl MeshTopology) -> HashSet<(u32, u32)> {
    let mut s = HashSet::new();
    for bf in 0..mesh.n_boundary_faces() as u32 {
        let ns = mesh.face_nodes(bf);
        for i in 0..ns.len() {
            let a = ns[i];
            let b = ns[(i + 1) % ns.len()];
            s.insert((a.min(b), a.max(b)));
        }
    }
    s
}

/// Local ND dm dofs whose edge key is a full-mesh boundary edge (partition ids).
fn owned_ess_partition_ids<M: MeshTopology>(
    par_nd: &ParallelFESpace<HCurlSpace<M>>,
    full_bnd_edges: &HashSet<(u32, u32)>,
    par_mesh: &ParallelMesh<M>,
) -> Vec<usize> {
    let nd = par_nd.local_space();
    let dp = par_nd.dof_partition();
    let part = par_mesh.partition();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for e in nd.mesh().elem_iter() {
        let et = nd.mesh().element_type(e);
        let ns = nd.mesh().element_nodes(e);
        let dofs = nd.element_dofs(e);
        for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
            let (ga, gb) = (part.global_node(ns[a]), part.global_node(ns[b]));
            let key = (ga.min(gb), ga.max(gb));
            if full_bnd_edges.contains(&key) && seen.insert(dofs[k]) {
                let pid = dp.permute_dof(dofs[k]) as usize;
                if pid < dp.n_owned_dofs {
                    out.push(pid);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// MFEM `EliminateEssentialBCDiag`: zero rows/cols of the essential dofs
/// symmetrically and set the diagonal to `diag_val` (1.0 for A,
/// `f64::MIN_POSITIVE` for M — ex13p shifts the Dirichlet eigenvalues out of
/// the computational range).
fn eliminate_ess_diag(a: &ParCsrMatrix, ess: &[usize], diag_val: f64) -> ParCsrMatrix {
    let no = a.n_owned();
    let nt = no + a.n_ghost();
    let mut coo = CooMatrix::new(nt, nt);
    for r in 0..a.n_owned() {
        let d = a.diag_block();
        for k in d.row_ptr[r]..d.row_ptr[r + 1] {
            coo.add(r, d.col_idx[k] as usize, d.values[k]);
        }
        let o = a.offd_block();
        for k in o.row_ptr[r]..o.row_ptr[r + 1] {
            coo.add(r, (o.col_idx[k] as usize) + no, o.values[k]);
        }
    }
    let mut loc = coo.into_csr();
    for &p in ess {
        loc.eliminate_essential_bc_diag_symmetric(p, diag_val);
    }
    ParCsrMatrix::from_local_matrix(&loc, no, a.ghost_exchange_arc(), a.comm().clone())
}

/// Run the eigenvalue solve.  `precond` is the LOBPCG preconditioner closure
/// (owned slices); `projector` zeroes the essential dofs of the trial blocks
/// (and, when enabled, applies the div-free projection).
fn run_lobpcg<M: MeshTopology + Clone>(
    par_nd: &ParallelFESpace<HCurlSpace<M>>,
    a: &ParCsrMatrix,
    m: &ParCsrMatrix,
    args: &Args,
    precond: &dyn Fn(&[f64], &mut [f64]),
    projector: Option<&dyn Fn(&mut [ParVector])>,
    rank: i32,
) {
    if rank == 0 {
        eprintln!("Number of unknowns: {}", par_nd.n_global_dofs());
        eprintln!("Solving for eigenvalues using ParLOBPCG");
    }
    // nullspace_skip = 1e-6: the curl-curl gradient nullspace (|λ| ≈ 0) is
    // excluded from the Ritz selection (serial lobpcg_projected mechanism).
    let res = par_lobpcg::par_lobpcg(a, Some(m), args.nev, precond, projector, 1e-6, 500, 1e-8);

    // Dense cross-check of the assembled pencil (small systems only): the
    // true eigenvalues of the matrices, independent of LOBPCG.
    if rank == 0 && a.n_owned() <= 4000 {
        dense_check(a, m, args.nev);
    }

    if rank == 0 {
        for (i, &l) in res.eigenvalues.iter().enumerate() {
            eprintln!("  Eigenmode {}: Lambda = {:.14e}", i + 1, l);
        }
        eprintln!(
            "  Converged: {} ({} iters, res={:.3e})",
            res.converged, res.iterations, res.final_residual
        );
        // Verify the returned Ritz pairs: ||Av − λBv|| / |λ|.
        for (i, &lam) in res.eigenvalues.iter().enumerate() {
            let mut v = ParVector::zeros(par_nd);
            for (pid, val) in res.eigenvectors[i].iter().enumerate() {
                v.owned_slice_mut()[pid] = *val;
            }
            v.update_ghosts();
            let mut av = ParVector::zeros(par_nd);
            let mut bv = ParVector::zeros(par_nd);
            a.spmv(&mut v, &mut av);
            m.spmv(&mut v, &mut bv);
            let mut r = 0.0f64;
            for pid in 0..a.n_owned() {
                let d = av.owned_slice()[pid] - lam * bv.owned_slice()[pid];
                r += d * d;
            }
            r = r.sqrt() / lam.abs().max(1e-14);
            eprintln!("  [resid] mode {}: rel_res={r:.3e}", i + 1);
        }
        eprintln!("\nFinished.");
    }
}

/// Dense generalized eigenvalues of a serial CSR pencil (small matrices only).
fn dense_csr_eig(
    a: &fem_linalg::CsrMatrix<f64>,
    m: &fem_linalg::CsrMatrix<f64>,
    nev: usize,
    label: &str,
) {
    let n = a.nrows;
    if n > 4000 {
        return;
    }
    let mut am = nalgebra::DMatrix::<f64>::zeros(n, n);
    let mut mm = nalgebra::DMatrix::<f64>::zeros(n, n);
    for r in 0..n {
        for k in a.row_ptr[r]..a.row_ptr[r + 1] {
            let c = a.col_idx[k] as usize;
            if c < n {
                am[(r, c)] = a.values[k];
            }
        }
        for k in m.row_ptr[r]..m.row_ptr[r + 1] {
            let c = m.col_idx[k] as usize;
            if c < n {
                mm[(r, c)] = m.values[k];
            }
        }
    }
    // A x = λ M x via Cholesky symmetrisation: M = L Lᵀ, A_s = L⁻¹ A L⁻ᵀ.
    let mi = match mm.cholesky() {
        Some(c) => match c.l().try_inverse() {
            Some(li) => li,
            None => return,
        },
        None => return,
    };
    let op = &mi * &am * mi.transpose();
    let se = nalgebra::SymmetricEigen::new(op);
    let mut ev: Vec<f64> = se.eigenvalues.iter().copied().collect();
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut printed = 0;
    for &l in &ev {
        if l > 1e-8 && printed < nev {
            eprintln!("  [{label}] eigenvalue {printed}: {l:.10e}");
            printed += 1;
        }
    }
}

/// Dense generalized eigendecomposition of the assembled pencil (np1 only).
fn dense_check(a: &ParCsrMatrix, m: &ParCsrMatrix, nev: usize) {
    let n = a.n_owned();
    let d = a.diag_block();
    let md = m.diag_block();
    let mut am = nalgebra::DMatrix::<f64>::zeros(n, n);
    let mut mm = nalgebra::DMatrix::<f64>::zeros(n, n);
    for r in 0..n {
        for k in d.row_ptr[r]..d.row_ptr[r + 1] {
            let c = d.col_idx[k] as usize;
            if c < n {
                am[(r, c)] = d.values[k];
            }
        }
        for k in md.row_ptr[r]..md.row_ptr[r + 1] {
            let c = md.col_idx[k] as usize;
            if c < n {
                mm[(r, c)] = md.values[k];
            }
        }
    }
    // A x = λ M x via Cholesky symmetrisation (M SPD after elimination; the
    // BC dofs have diag = f64::MIN_POSITIVE — the free block dominates the
    // smallest eigenvalues).
    let mi = match mm.clone().cholesky() {
        Some(c) => match c.l().try_inverse() {
            Some(li) => li,
            None => return,
        },
        None => return,
    };
    let op = &mi * &am * mi.transpose();
    let se = nalgebra::SymmetricEigen::new(op);
    let mut ev: Vec<f64> = se.eigenvalues.iter().copied().collect();
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut printed = 0;
    for &l in &ev {
        if l > 1e-8 && printed < nev {
            eprintln!("  [dense] eigenvalue {printed}: {l:.10e}");
            printed += 1;
        }
    }
}

// ─── 3D path (default: beam-tet.mesh) ────────────────────────────────────────

fn run_3d(comm: fem_parallel::comm::Comm, args: &Args) {
    let rank = comm.rank();
    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut serial = mfem.mesh3d.expect("3D mesh required");
    let total_ref = args.ser_ref_levels + args.par_ref_levels; // rs + rp merged serial
    for _ in 0..total_ref {
        serial = refine_uniform_3d(&serial);
    }
    let full_bnd_edges = full_boundary_edge_keys(&serial);
    let par_mesh: ParallelMesh<Mesh<3>> = partition_mesh(&serial, &comm);
    let local_mesh = par_mesh.local_mesh().clone();

    let _par_h1 = ParallelFESpace::new(H1Space::new(local_mesh.clone(), 1), &par_mesh, comm.clone());
    let par_nd = ParallelFESpace::new(
        HCurlSpace::new(local_mesh.clone(), args.order),
        &par_mesh,
        comm.clone(),
    );
    let quad = 2 * args.order as u8 + 1;

    let mut a = ParVectorAssembler::assemble_bilinear(
        &par_nd,
        &[&CurlCurlIntegrator { mu: 1.0 }],
        quad,
    );
    let mut m = ParVectorAssembler::assemble_bilinear(
        &par_nd,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        quad,
    );

    // PEC: all boundary attributes essential (ex13p marks every bdr attr).
    let ess = owned_ess_partition_ids(&par_nd, &full_bnd_edges, &par_mesh);
    if rank == 0 {
        eprintln!("  ess dofs (owned): {}", ess.len());
    }
    a = eliminate_ess_diag(&a, &ess, 1.0);
    m = eliminate_ess_diag(&m, &ess, f64::MIN_POSITIVE);

    // Preconditioner: block-diagonal Jacobi (the AMG V-cycle on the pure
    // curl-curl gives the same eigenvalues without better convergence).
    let diag3 = a.diagonal();
    let n3 = a.n_owned();
    let precond = move |r: &[f64], z: &mut [f64]| {
        for i in 0..n3.min(r.len()) {
            let d = diag3[i];
            z[i] = if d.abs() > 1e-300 { r[i] / d } else { 0.0 };
        }
    };

    // Essential-dof zeroing (the eigenvectors of the eliminated pencil have
    // zero BC components; leaving them in the trial space pollutes the
    // Rayleigh quotient: A_dd = 1.0 but B_dd = f64::MIN_POSITIVE).
    let ess_c = ess.clone();
    let proj = move |block: &mut [ParVector]| {
        for v in block.iter_mut() {
            let vs = v.owned_slice_mut();
            for &p in &ess_c {
                vs[p] = 0.0;
            }
        }
    };

    run_lobpcg(&par_nd, &a, &m, args, &precond, Some(&proj), rank);
}

// ─── 2D path (variant: star.mesh / square-disc.mesh) ─────────────────────────

fn run_2d(comm: fem_parallel::comm::Comm, args: &Args) {
    let rank = comm.rank();
    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut serial = mfem.mesh2d.expect("2D mesh required");
    let total_ref = args.ser_ref_levels + args.par_ref_levels;
    for _ in 0..total_ref {
        serial = refine_uniform(&serial);
    }
    let full_bnd_edges = full_boundary_edge_keys(&serial);
    let par_mesh: ParallelMesh<Mesh<2>> = partition_mesh(&serial, &comm);
    let local_mesh = par_mesh.local_mesh().clone();

    let par_nd = ParallelFESpace::new(
        HCurlSpace::new(local_mesh.clone(), args.order),
        &par_mesh,
        comm.clone(),
    );
    let _par_h1 = ParallelFESpace::new(H1Space::new(local_mesh.clone(), 1), &par_mesh, comm.clone());
    let quad = 2 * args.order as u8 + 1;

    let mut a = ParVectorAssembler::assemble_bilinear(
        &par_nd,
        &[&CurlCurlIntegrator { mu: 1.0 }],
        quad,
    );
    let mut m = ParVectorAssembler::assemble_bilinear(
        &par_nd,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        quad,
    );

    let ess = owned_ess_partition_ids(&par_nd, &full_bnd_edges, &par_mesh);
    if rank == 0 {
        eprintln!("  ess dofs (owned): {}", ess.len());
    }
    a = eliminate_ess_diag(&a, &ess, 1.0);
    m = eliminate_ess_diag(&m, &ess, f64::MIN_POSITIVE);

    // 2D: block-diagonal Jacobi preconditioner.
    let diag = a.diagonal();
    let n_owned = a.n_owned();
    let precond = move |r: &[f64], z: &mut [f64]| {
        for i in 0..n_owned.min(r.len()) {
            let d = diag[i];
            z[i] = if d.abs() > 1e-300 { r[i] / d } else { 0.0 };
        }
    };

    let ess_c = ess.clone();
    let proj = move |block: &mut [ParVector]| {
        for v in block.iter_mut() {
            let vs = v.owned_slice_mut();
            for &p in &ess_c {
                vs[p] = 0.0;
            }
        }
    };

    run_lobpcg(&par_nd, &a, &m, args, &precond, Some(&proj), rank);
}

fn run_pex13(comm: fem_parallel::comm::Comm, args: &Args) {
    let rank = comm.rank();
    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    if mfem.mesh3d.is_some() {
        run_3d(comm, args);
    } else if mfem.mesh2d.is_some() {
        run_2d(comm, args);
    } else {
        if rank == 0 {
            eprintln!("Mesh file has neither 2D nor 3D mesh");
        }
    }
}
