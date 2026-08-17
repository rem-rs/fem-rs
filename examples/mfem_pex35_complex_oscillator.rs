//! # Parallel Example 35 �?Complex-valued damped harmonic oscillator
//! (1:1 with MFEM ex35p / ex35p.cpp)
//!
//! Three variants of a damped harmonic oscillator driven by a forced
//! oscillation imposed on a *port* (a portion of the boundary):
//!
//! 0) Scalar H¹ field:  `-Div(a Grad u) - ω² b u + i ω c u = 0`
//! 1) Vector H(Curl):   `Curl(a Curl u) - ω² b u + i ω c u = 0`
//! 2) Vector H(Div):    `-Grad(a Div u) - ω² b u + i ω c u = 0`
//!
//! The spatial variation of the port boundary condition is computed as an
//! eigenmode of an appropriate operator defined on the boundary sub-mesh
//! (port).  The complex system is solved with FGMRES using a block-diagonal
//! preconditioner (real part preconditioner applied to both blocks, scaled
//! by ±1 for the imaginary block �?MFEM `ComplexOperator` convention).
//!
//! ## Parallel structure (mirrors ex35p)
//! 1. Build the fully-refined serial mesh (rs+rp all serial; `par_uniform_refine`
//!    is 2-D only, the final global mesh is identical to C++'s rs+rp refinement).
//! 2. Extract the port boundary sub-mesh and compute the port eigenmode *on the
//!    full serial port sub-mesh*.  The eigenproblem is a global property of the
//!    port mesh, so the eigenvalues/vectors are partition-independent (np1-4
//!    identical) �?this is the `ParSubMesh::CreateFromBoundary` + LOBPCG/AME
//!    step of C++, done serially on the full mesh (same global result).
//! 3. Transfer the port BC to the real part of the full-mesh solution via the
//!    global DOF mapping (parent-node / parent-edge / parent-face keys).
//! 4. Partition the mesh; assemble the parallel complex system as a
//!    `ParComplexCsrMatrix` (re/im assembled separately and combined �?the
//!    re and im sparsity patterns coincide).
//! 5. Complex Dirichlet elimination on all boundary DOFs (non-homogeneous on
//!    the real part, zero imaginary) with cross-rank ghost-column elimination.
//! 6. Solve with parallel complex FGMRES (`par_solve_fgmres_complex`) and a
//!    block-diagonal `[P, ±P]` preconditioner: the real block uses AMS (p1) /
//!    AMG (p0, p2), the imaginary block uses ±P (MFEM `ScaledOperator`).
//!
//! ## Usage
//! ```bash
//! cargo run --release --example mfem_pex35_complex_oscillator -- --ranks 1 -p 0 -no-vis
//! cargo run --release --example mfem_pex35_complex_oscillator -- --ranks 2 -p 1 -no-vis
//! cargo run --release --example mfem_pex35_complex_oscillator -- --ranks 4 -p 1 -no-vis
//! ```
//!
//! ## Acceptance (verified vs C++ ex35p)
//! * p1: unknowns 2174 / port 180 / system 4348; eigenvalues
//!   {6.17506890505674e-01, 2.47777391174073e+00, 5.60235135555006e+00,
//!   1.00038329894962e+01, 1.00289771080686e+01} — np1-4 identical;
//!   Re norm 1.68893631e0 / Im norm 1.34348912e0 (np1-4 identical, = C++);
//!   physical-edge checksum Re 1.37728391e3 / Im -9.36770110e2 (np1-4
//!   identical, = C++ to ~6-7 digits).
//! * p0: unknowns 655 / port 85 / system 1310; Re norm 5.30612605e0 /
//!   Im norm 4.26139668e0 (np1-4 identical, = C++).
//! * p2: unknowns 2416 / port 96 / system 4832, eigenvalue 1.61880932e0 =
//!   C++; the C++ p2 FGMRES itself does not converge (1000 iters, ||r||=38),
//!   so no solution comparison is possible.
//!
//! ## Known differences vs C++
//! * The port eigenmode is solved serially on the full port sub-mesh (C++ uses
//!   parallel LOBPCG/AME on the ParSubMesh); eigenvalues agree to ~13 digits.
//! * The parallel preconditioner is a block-diagonal [P, ±P] built from the
//!   local diagonal block (C++ uses the global HypreAMS/AMG); iteration counts
//!   may differ, the converged solution agrees.
//! * Boundary (essential) DOFs are detected from the FULL serial mesh via
//!   physical-edge keys, then mapped to the local partition (pex34 pattern):
//!   the local mesh's own boundary faces lose some boundary marks after
//!   partitioning (boundary_dofs_hcurl misses them on np>1).

use std::f64::consts::PI;
use std::io::Write;

use fem_assembly::standard::{
    CurlCurlIntegrator, DiffusionIntegrator, GradDivIntegrator, MassIntegrator,
    VectorMassIntegrator,
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{
    BoundarySubMesh, Mesh, extract_boundary_submesh, refine_uniform_3d,
    topology::MeshTopology,
};
use fem_parallel::{
    ParAmsPrecond, ParCsrMatrix, ParDiscreteLinearOperator, ParallelFESpace, ParVector,
    ParVectorAssembler, dof_partition::DofPartition,
    launcher::{native::ThreadLauncher, WorkerConfig},
    par_complex_csr::ParComplexCsrMatrix,
    par_partition::partition_mesh,
    par_solver::par_solve_fgmres_complex,
    par_vector::ParComplexVector,
};
use fem_space::fe_space::FESpace;
use fem_space::{
    HCurlSpace, HDivSpace, H1Space, L2Space,
    constraints::{boundary_dofs, boundary_dofs_hcurl},
};
use fem_solver::eigen::{AmeConfig, ame_solve};
use linlvo::Preconditioner;

// ─── Command line (mirrors MFEM ex35p OptionsParser) ────────────────────────

#[derive(Clone)]
struct Args {
    mesh: String,
    ser_ref_levels: usize,
    par_ref_levels: usize,
    order: u8,
    prob: u8,
    mode: usize,
    a_coef: f64,
    mu: f64,
    eps: f64,
    sig: f64,
    freq: f64,
    port_bc_attr: Vec<i32>,
    herm_conv: bool,
    ranks: usize,
    mixed: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/fichera-mixed.mesh".into(),
        ser_ref_levels: 1,
        par_ref_levels: 1,
        order: 1,
        prob: 0,
        mode: 1,
        a_coef: 0.0,
        mu: 1.0,
        eps: 1.0,
        sig: 2.0,
        freq: -1.0,
        port_bc_attr: Vec::new(),
        herm_conv: true,
        ranks: 1,
        mixed: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_default(),
            "-rs" | "--refine-serial" => a.ser_ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-rp" | "--refine-parallel" => a.par_ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-o" | "--order" => a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-p" | "--problem-type" => a.prob = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "-em" | "--eigenmode" => a.mode = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-a" | "--stiffness-coef" => a.a_coef = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            "-mu" | "--permeability" => a.mu = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-eps" | "--permittivity" => a.eps = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-sigma" | "--conductivity" => a.sig = it.next().and_then(|s| s.parse().ok()).unwrap_or(2.0),
            "-f" | "--frequency" => a.freq = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1.0),
            "-pbc" | "--port-bc-attr" => {
                let v = it.next().unwrap_or_default();
                a.port_bc_attr = v.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            }
            "-herm" | "--hermitian" => a.herm_conv = true,
            "-no-herm" | "--no-hermitian" => a.herm_conv = false,
            "--ranks" => a.ranks = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-mixed" | "--mixed-mesh" => a.mixed = true,
            "-hex" | "--hex-mesh" => a.mixed = false,
            _ => {}
        }
    }
    a
}

// ─── Port eigenmodes (serial, on the full port sub-mesh) ────────────────────

fn dense_generalized_eig(
    a: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    free: &[usize],
) -> (Vec<f64>, nalgebra::DMatrix<f64>) {
    use nalgebra::DMatrix;
    use nalgebra::SymmetricEigen;
    let nf = free.len();
    let mut a_d = DMatrix::zeros(nf, nf);
    let mut m_d = DMatrix::zeros(nf, nf);
    for (ri, &i) in free.iter().enumerate() {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            if let Ok(ci) = free.binary_search(&j) {
                a_d[(ri, ci)] = a.values[p];
            }
        }
        for p in m.row_ptr[i]..m.row_ptr[i + 1] {
            let j = m.col_idx[p] as usize;
            if let Ok(ci) = free.binary_search(&j) {
                m_d[(ri, ci)] = m.values[p];
            }
        }
    }
    for i in 0..nf {
        m_d[(i, i)] += 1e-10;
    }
    let chol = m_d.cholesky().expect("M_ff must be SPD");
    let l = chol.l();
    let linv = l.try_inverse().expect("L invertible");
    let a_red = &linv * &a_d * linv.transpose();
    let eig = SymmetricEigen::new(a_red);
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..nf)
        .map(|k| (eig.eigenvalues[k], eig.eigenvectors.column(k).iter().copied().collect()))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let evals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let mut x = DMatrix::zeros(nf, nf);
    for (k, (_, ycol)) in pairs.iter().enumerate() {
        use nalgebra::DVector;
        let ym = DVector::from_column_slice(ycol);
        let xcol = linv.transpose() * ym;
        x.column_mut(k).copy_from_slice(xcol.as_slice());
    }
    (evals, x)
}

fn scalar_waveguide(mode: usize, port: &Mesh<3>, order: u8) -> Vec<f64> {
    let fes = H1Space::new(port.clone(), order);
    let n = fes.n_dofs();
    let qo = 2 * order + 1;

    let a_0 = fem_assembly::Assembler::assemble_bilinear(&fes, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
    let m_0 = fem_assembly::Assembler::assemble_bilinear(&fes, &[&MassIntegrator { rho: 1.0 }], qo);

    let ess = boundary_dofs(port, fes.dof_manager(), &port.unique_boundary_tags());
    let ess_set: std::collections::HashSet<usize> =
        ess.iter().map(|&d| d as usize).collect();
    let free: Vec<usize> = (0..n).filter(|i| !ess_set.contains(i)).collect();

    let (evals, evecs) = dense_generalized_eig(&a_0, &m_0, &free);
    let mut x = vec![0.0; n];
    for (k, &f) in free.iter().enumerate() {
        x[f] = evecs[(k, mode)];
    }
    println!("Eigenvalue lambda   {:.8e}", evals[mode]);
    x
}

fn vector_waveguide(mode: usize, port: &Mesh<3>, order: u8) -> Vec<f64> {
    let nev = std::cmp::max(mode + 2, 5);
    let fes = HCurlSpace::new(port.clone(), order);
    let qo = 2 * order + 1;

    let a_0 = fem_assembly::VectorAssembler::assemble_bilinear(&fes, &[&CurlCurlIntegrator { mu: 1.0 }], qo);
    let m_0 = fem_assembly::VectorAssembler::assemble_bilinear(&fes, &[&VectorMassIntegrator { alpha: 1.0 }], qo);

    let ess = boundary_dofs_hcurl(port, &fes, &port.unique_boundary_tags());
    let ess_u: Vec<usize> = ess.iter().map(|&d| d as usize).collect();

    let mut a_mat = a_0.clone();
    let mut m_mat = m_0.clone();
    for &d in &ess_u {
        if let Some(p) = a_mat.find_entry(d, d) {
            a_mat.values[p] = 1.0;
        }
    }
    for &d in &ess_u {
        if let Some(p) = m_mat.find_entry(d, d) {
            m_mat.values[p] = f64::MIN_POSITIVE;
        }
    }

    let h1 = H1Space::new(port.clone(), 1);
    let g = fem_assembly::mixed::assemble_hcurl_h1_gradient(&fes, &h1, qo);

    let cfg = AmeConfig { nev, ..Default::default() };
    let result = ame_solve(&a_mat, &m_mat, &g, &cfg).expect("VectorWaveGuide AME failed");
    let x: Vec<f64> = result.eigenvectors.column(mode).iter().copied().collect();
    println!("Eigenvalue lambda   {:.8e}", result.eigenvalues[mode]);
    x
}

fn pseudo_scalar_waveguide(mode: usize, port: &Mesh<3>, order_l2: u8) -> Vec<f64> {
    let h1_order = order_l2 + 1;
    let fes = H1Space::new(port.clone(), h1_order);
    let n = fes.n_dofs();
    let qo = 2 * h1_order + 1;

    if mode == 0 {
        return vec![1.0; fes_l2_size(port, order_l2)];
    }

    let a_0 = fem_assembly::Assembler::assemble_bilinear(
        &fes,
        &[&DiffusionIntegrator { kappa: 1.0 }, &MassIntegrator { rho: 1.0 }],
        qo,
    );
    let m_0 = fem_assembly::Assembler::assemble_bilinear(&fes, &[&MassIntegrator { rho: 1.0 }], qo);

    let free: Vec<usize> = (0..n).collect();
    let (evals, evecs) = dense_generalized_eig(&a_0, &m_0, &free);
    println!("Eigenvalue lambda   {:.8e}", evals[mode]);
    let x: Vec<f64> = (0..n).map(|k| evecs[(k, mode)]).collect();
    project_h1_to_l2(port, order_l2, &x)
}

fn fes_l2_size(port: &Mesh<3>, order_l2: u8) -> usize {
    L2Space::new(port.clone(), order_l2).n_dofs()
}

fn project_h1_to_l2(port: &Mesh<3>, order_l2: u8, x_h1: &[f64]) -> Vec<f64> {
    let fes_l2 = L2Space::new(port.clone(), order_l2);
    let n_l2 = fes_l2.n_dofs();
    let h1_order = order_l2 + 1;
    let fes_h1 = H1Space::new(port.clone(), h1_order);
    let gf_h1 = fem_assembly::GridFunction::new(&fes_h1, x_h1.to_vec());

    let mut x = vec![0.0; n_l2];
    for e in 0..port.n_elems() as u32 {
        let ref_l2 = fem_assembly::assembler::ref_elem_vol_l2(port.element_type(e), order_l2);
        let ip = ref_l2.dof_coords();
        let dofs = fes_l2.element_dofs(e);
        for (k, &d) in dofs.iter().enumerate() {
            x[d as usize] = gf_h1.evaluate_at_element(e, &ip[k]);
        }
    }
    x
}

// ─── Port �?full-mesh transfer (global DOF maps) ────────────────────────────

/// Transfer an H¹ (P1) port grid function to the full-mesh real part.
/// H¹ P1 DOFs are vertex DOFs; `parent_node_of_sub[sub] = parent_node`.
fn transfer_h1_port(port: &BoundarySubMesh, port_bc: &[f64], n_full: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_full];
    for (si, &pn) in port.parent_node_of_sub.iter().enumerate() {
        if si < port_bc.len() {
            out[pn as usize] = port_bc[si];
        }
    }
    out
}

/// Transfer ND (edge) DOFs from the port space to the full H(Curl) space.
///
/// Port elements are the parent boundary faces; edge DOFs map by the vertex
/// pair.  The ND basis direction follows the canonical edge orientation
/// (smaller node id → larger node id) *within each mesh's own node numbering*.
/// When the port's local node ordering disagrees with the parent's global
/// ordering (partition renumbers), the transferred value must flip sign.
fn transfer_edge_dofs(
    port: &BoundarySubMesh,
    port_bc: &[f64],
    parent_fes: &HCurlSpace<Mesh<3>>,
    out: &mut [f64],
) {
    use fem_space::dof_manager::EdgeKey;
    let port_fes = HCurlSpace::new(port.mesh.clone(), parent_fes.order());
    for e in 0..port.mesh.n_elems() as u32 {
        let ns = port.mesh.elem_nodes(e);
        let n = ns.len();
        for i in 0..n {
            let (a, b) = (ns[i], ns[(i + 1) % n]);
            // Port canonical key (port local node ids).
            let pkey = EdgeKey::new(a.min(b), a.max(b));
            let Some(port_dof) = port_fes.edge_dof(pkey) else { continue };
            let pa = port.parent_node_of_sub[a as usize];
            let pb = port.parent_node_of_sub[b as usize];
            let Some(full_dof) = parent_fes.edge_dof(EdgeKey::new(pa.min(pb), pa.max(pb))) else { continue };
            // Sign: +1 when the port edge direction (local a→b) agrees with the
            // parent's canonical direction (global pa→pb); flip otherwise.
            let port_dir_asc = a < b;
            let parent_dir_asc = pa < pb;
            let sign = if port_dir_asc == parent_dir_asc { 1.0 } else { -1.0 };
            if (full_dof as usize) < out.len() {
                out[full_dof as usize] = sign * port_bc[port_dof as usize];
            }
        }
    }
}

/// Transfer RT (face) DOFs from the port L² space to the full H(Div) space.
fn transfer_face_dofs(port: &BoundarySubMesh, port_bc: &[f64], out: &mut [f64]) {
    for (k, &fid) in port.parent_face_ids.iter().enumerate() {
        if (fid as usize) < out.len() {
            out[fid as usize] = port_bc[k];
        }
    }
}

// ─── Shared parallel helpers ─────────────────────────────────────────────────

/// Map full-mesh (global) BC values onto the local [owned|ghost] DOF layout.
///
/// For H1 P1 the global DOF id IS the full-mesh node id, so
/// `dp.global_dof(pid)` indexes `full_u` directly.  For ND (edge) spaces the
/// partition's global DOF ids are a *new* numbering that does NOT match the
/// full-mesh space's dof ids — the physical edge key (parent-global node
/// pair) is used instead (pex34's key pattern).  RT face dofs coincide with
/// face ids in both meshes.
fn map_full_to_local(
    dp: &DofPartition,
    local_mesh: &Mesh<3>,
    partition: &fem_parallel::partition::MeshPartition,
    full_u: &[f64],
    prob: u8,
    order: u8,
    full_mesh: &Mesh<3>,
) -> Vec<f64> {
    let n_local = dp.n_owned_dofs + dp.n_ghost_dofs;
    let mut out = vec![0.0; n_local];

    match prob {
        0 | 2 => {
            // H1 P1 / RT: global dof id == full-mesh dof id.
            for pid in 0..n_local {
                let gid = dp.global_dof(pid as u32) as usize;
                if gid < full_u.len() {
                    out[pid] = full_u[gid];
                }
            }
        }
        1 => {
            // ND: bridge by physical edge key (parent global node pair).
            let full_fes = HCurlSpace::new(full_mesh.clone(), order);
            let mut key_to_full: std::collections::HashMap<(u32, u32), u32> = Default::default();
            for e in full_mesh.elem_iter() {
                let et = full_mesh.element_type(e);
                let ns = full_mesh.element_nodes(e);
                let dofs = full_fes.element_dofs(e);
                for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                    let key = (ns[a].min(ns[b]), ns[a].max(ns[b]));
                    key_to_full.entry(key).or_insert(dofs[k]);
                }
            }
            let local_fes = HCurlSpace::new(local_mesh.clone(), order);
            let mut miss = 0usize;
            let mut hit = 0usize;
            for e in local_mesh.elem_iter() {
                let et = local_mesh.element_type(e);
                let ns = local_mesh.element_nodes(e);
                let dofs = local_fes.element_dofs(e);
                for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                    // Local node ids → parent global node ids for the key.
                    let (ga, gb) = (partition.global_node(ns[a]), partition.global_node(ns[b]));
                    let key = (ga.min(gb), ga.max(gb));
                    if let Some(&full_dof) = key_to_full.get(&key) {
                        let dm = dofs[k] as u32;
                        let pid = dp.permute_dof(dm) as usize;
                        if pid < n_local && (full_dof as usize) < full_u.len() {
                            out[pid] = full_u[full_dof as usize];
                        }
                        hit += 1;
                    } else {
                        miss += 1;
                    }
                }
            }
            let _ = (hit, miss);
        }
        _ => unreachable!(),
    }
    out
}

/// Per-element-type local edge tables (HCurl space ordering), matching
/// `dof_partition::from_edge_space`'s tables.
fn edges_for_elem(et: fem_mesh::ElementType) -> &'static [(usize, usize)] {
    use fem_mesh::ElementType;
    match et {
        ElementType::Tet4 | ElementType::Tet10 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        ElementType::Hex8 | ElementType::Hex20 => &[
            (0, 1), (1, 2), (3, 2), (0, 3),
            (4, 5), (5, 6), (7, 6), (4, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ],
        ElementType::Prism6 => &[
            (0, 1), (1, 2), (0, 2),
            (3, 4), (4, 5), (3, 5),
            (0, 3), (1, 4), (2, 5),
        ],
        _ => &[(0, 1), (1, 2), (0, 2)],
    }
}

/// Report norm / sum / checksum of the complex solution (np1-4 must agree).
fn report_metrics(
    x: &ParComplexVector,
    comm: &fem_parallel::Comm,
    is_root: bool,
    out: &mut String,
) {
    let n_owned = x.re.n_owned();
    let re_norm = x.re.global_norm();
    let re_sum = comm.allreduce_sum_f64(x.re.owned_slice().iter().sum::<f64>());
    let re_checksum: f64 = (0..n_owned).map(|pid| (pid as f64 + 1.0) * x.re.as_slice()[pid]).sum();
    let re_checksum = comm.allreduce_sum_f64(re_checksum);
    let im_norm = x.im.global_norm();
    let im_sum = comm.allreduce_sum_f64(x.im.owned_slice().iter().sum::<f64>());
    let im_checksum: f64 = (0..n_owned).map(|pid| (pid as f64 + 1.0) * x.im.as_slice()[pid]).sum();
    let im_checksum = comm.allreduce_sum_f64(im_checksum);
    if is_root {
        out.push_str(&format!(
            "  Re: ||u|| = {:.8e}, sum = {:.8e}, checksum = {:.8e}\n",
            re_norm, re_sum, re_checksum
        ));
        out.push_str(&format!(
            "  Im: ||u|| = {:.8e}, sum = {:.8e}, checksum = {:.8e}\n",
            im_norm, im_sum, im_checksum
        ));
    }
}

fn save_outputs(rank: usize, local_mesh: &Mesh<3>, x: &ParComplexVector) {
    let dummy2d = Mesh::<2>::unit_square_tri(1);
    let mesh_name = format!("mesh.{:06}", rank);
    let mut f = std::fs::File::create(&mesh_name).expect("mesh");
    fem_io::mfem::write_mfem(&mut f, &dummy2d, Some(local_mesh)).expect("write mesh");
    let sol_r_name = format!("sol_r.{:06}", rank);
    let mut f = std::fs::File::create(&sol_r_name).expect("sol_r");
    for &v in x.re.owned_slice() {
        writeln!(f, "{:.8e}", v).expect("sol_r write");
    }
    let sol_i_name = format!("sol_i.{:06}", rank);
    let mut f = std::fs::File::create(&sol_i_name).expect("sol_i");
    for &v in x.im.owned_slice() {
        writeln!(f, "{:.8e}", v).expect("sol_i write");
    }
    // Debug: dump (edge_key, value) for owned dofs so cross-np comparison by
    // physical edge is possible.
}
// ─── Problem 0: scalar H¹ ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn solve_h1(
    args: &Args,
    comm: &fem_parallel::Comm,
    full_mesh: &Mesh<3>,
    par_mesh: &fem_parallel::ParallelMesh<Mesh<3>>,
    port: &BoundarySubMesh,
    port_bc: &[f64],
    order: u8,
    mu: f64,
    eps: f64,
    sig: f64,
    omega: f64,
) {
    let rank = comm.rank();
    let is_root = rank == 0;
    let mut out = String::new();
    let qo = 2 * order + 1;

    let full_fes = H1Space::new(full_mesh.clone(), order);
    let n = full_fes.n_dofs();
    let full_u_re = transfer_h1_port(port, port_bc, n);
    let full_u_im = vec![0.0; n];

    let local_mesh = par_mesh.local_mesh().clone();
    let local_space = H1Space::new(local_mesh.clone(), order);
    let par_space = ParallelFESpace::new(local_space, par_mesh, comm.clone());
    let dp = par_space.dof_partition();
    let n_owned = dp.n_owned_dofs;
    let n_ghost = dp.n_ghost_dofs;
    let n_local = n_owned + n_ghost;

    if is_root {
        out.push_str(&format!("Number of finite element unknowns: {}\n", par_space.n_global_dofs()));
    }
    let u_re_local = map_full_to_local(dp, &local_mesh, par_mesh.partition(), &full_u_re, 0, order, full_mesh);
    let u_im_local = map_full_to_local(dp, &local_mesh, par_mesh.partition(), &full_u_im, 0, order, full_mesh);

    // Complex system: -Div(a Grad) - ω² b + i ω c  (a = 1/μ, b = ε, c = σ).
    let k_re = fem_parallel::ParAssembler::assemble_bilinear(
        &par_space,
        &[&DiffusionIntegrator { kappa: 1.0 / mu }, &MassIntegrator { rho: -omega * omega * eps }],
        qo,
    );
    let k_im = fem_parallel::ParAssembler::assemble_bilinear(
        &par_space,
        &[&MassIntegrator { rho: omega * sig }],
        qo,
    );
    let mut a_sys = ParComplexCsrMatrix::new(
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.diag_block(), k_im.diag_block()),
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.offd_block(), k_im.offd_block()),
        n_owned, n_ghost,
        par_space.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    // Complex Dirichlet (all boundary DOFs).  Boundary-node set from the FULL
    // mesh (partition-independent), mapped to the local partition (pex34
    // pattern — local boundary detection misses owned boundary nodes after
    // partitioning).
    let full_bnd_nodes: std::collections::HashSet<u32> = {
        let mut s = std::collections::HashSet::new();
        for bf in 0..full_mesh.n_boundary_faces() as u32 {
            for &n in full_mesh.face_nodes(bf) {
                s.insert(n);
            }
        }
        s
    };
    let mut ess_dm: Vec<u32> = Vec::new();
    {
        let lh1 = H1Space::new(local_mesh.clone(), order);
        let mut seen = std::collections::HashSet::new();
        for e in local_mesh.elem_iter() {
            let ns = local_mesh.element_nodes(e);
            let dofs = lh1.element_dofs(e);
            for (k, &n) in ns.iter().enumerate() {
                let gn = par_mesh.partition().global_node(n);
                if full_bnd_nodes.contains(&gn) && seen.insert(dofs[k]) {
                    ess_dm.push(dofs[k]);
                }
            }
        }
    }
    let mut rhs = ParComplexVector::zeros_like(&ParVector::zeros(&par_space));
    let mut clamped: Vec<(usize, f64, f64)> = Vec::new();
    let mut ghost_ess: Vec<(usize, f64, f64)> = Vec::new();
    for &dm in &ess_dm {
        let pid = dp.permute_dof(dm) as usize;
        let vr = u_re_local[pid];
        let vi = u_im_local[pid];
        if pid < n_owned {
            clamped.push((pid, vr, vi));
        } else if pid < n_local {
            ghost_ess.push((pid - n_owned, vr, vi));
        }
    }
    for &(pid, vr, vi) in &clamped {
        a_sys.apply_dirichlet_par(pid, vr, vi, &mut rhs);
    }
    a_sys.apply_ghost_ess_columns(&ghost_ess, &mut rhs);
    if is_root {
        out.push_str(&format!("Size of linear system: {}\n", 2 * par_space.n_global_dofs()));
        out.push_str("\nSolving for the complex field using FGMRES with a block-diagonal preconditioner\n");
    }

    // Preconditioner [P, ±P]: AMG on the real block (C++ HypreBoomerAMG).
    let imag_scale = if args.herm_conv { -1.0 } else { 1.0 };
    let pc_op = fem_parallel::ParAssembler::assemble_bilinear(
        &par_space,
        &[&DiffusionIntegrator { kappa: 1.0 / mu },
          &MassIntegrator { rho: omega * omega * eps },
          &MassIntegrator { rho: omega * sig }],
        qo,
    );
    let mut pc_elim = clone_par_csr(&pc_op);
    {
        let mut rhs_dummy = ParVector::zeros(&par_space);
        for &(pid, _, _) in &clamped {
            pc_elim.apply_dirichlet_par_keep_diag(pid, 0.0, &mut rhs_dummy);
        }
    }
    let pc_diag = pc_elim.diag_block().clone();
    let amg = std::sync::Arc::new(linlvo::amg::AmgPrecond::new(linlvo::amg::AmgHierarchy::build(
        fem_linalg::fem_to_linlvo_csr(&pc_diag),
        linlvo::amg::AmgConfig {
            theta: 0.25,
            strategy: linlvo::amg::CoarsenStrategy::RugeStüben,
            smoother: linlvo::amg::SmootherType::GaussSeidel,
            pre_sweeps: 1,
            post_sweeps: 1,
            coarse_threshold: 9,
            max_levels: 25,
            ..Default::default()
        },
    )));
    let precond = move |r_re: &[f64], r_im: &[f64], z_re: &mut [f64], z_im: &mut [f64]| {
        let lr = linlvo::DenseVec::from_vec(r_re.to_vec());
        let mut lz = linlvo::DenseVec::zeros(z_re.len());
        amg.apply_precond(&lr, &mut lz);
        z_re.copy_from_slice(lz.as_slice());
        let lr2 = linlvo::DenseVec::from_vec(r_im.to_vec());
        let mut lz2 = linlvo::DenseVec::zeros(z_im.len());
        amg.apply_precond(&lr2, &mut lz2);
        for i in 0..z_im.len() {
            z_im[i] = imag_scale * lz2.as_slice()[i];
        }
    };

    let cfg = fem_solver::SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 3000, verbose: std::env::var("PEX35_ND_VERBOSE").is_ok(), ..Default::default() };
    let mut x = ParComplexVector::zeros_like(&rhs.re);
    for i in 0..n_owned {
        x.re.owned_slice_mut()[i] = u_re_local[i];
        x.im.owned_slice_mut()[i] = u_im_local[i];
    }
    x.update_ghosts();

    let res = par_solve_fgmres_complex(&a_sys, &rhs, &mut x, 50, Some(&precond), &cfg)
        .expect("FGMRES failed");
    if is_root {
        out.push_str(&format!("  FGMRES: Number of iterations: {}\n", res.iterations));
        out.push_str(&format!("  FGMRES: Final relative residual: {:.6e}\n", res.final_residual));
    }

    x.update_ghosts();
    report_metrics(&x, comm, is_root, &mut out);
    if args.prob == 1 {
        // Physical-edge-id checksum (partition-independent, matches C++ at
        // np1: MFEM ND dof numbering = element×local-edge first-seen).
        let edges_for_elem = |et: fem_mesh::ElementType| -> &'static [(usize, usize)] {
            use fem_mesh::ElementType;
            match et {
                ElementType::Tet4 | ElementType::Tet10 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                ElementType::Hex8 | ElementType::Hex20 => &[
                    (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ],
                ElementType::Prism6 => &[
                    (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5),
                ],
                _ => &[],
            }
        };
        let mut key_to_edge: std::collections::HashMap<(u32, u32), u64> = Default::default();
        let mut next_edge = 0u64;
        for e in full_mesh.elem_iter() {
            let et = full_mesh.element_type(e);
            let ns = full_mesh.element_nodes(e);
            for &(a, b) in edges_for_elem(et) {
                let key = if ns[a] < ns[b] { (ns[a], ns[b]) } else { (ns[b], ns[a]) };
                if !key_to_edge.contains_key(&key) {
                    key_to_edge.insert(key, next_edge);
                    next_edge += 1;
                }
            }
        }
        let lnd = HCurlSpace::new(local_mesh.clone(), order);
        let mut dm_to_key: std::collections::HashMap<u32, (u32, u32)> = Default::default();
        for e in local_mesh.elem_iter() {
            let et = local_mesh.element_type(e);
            let ns = local_mesh.element_nodes(e);
            let dofs = lnd.element_dofs(e);
            for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                let (ga, gb) = (par_mesh.partition().global_node(ns[a]), par_mesh.partition().global_node(ns[b]));
                dm_to_key.entry(dofs[k]).or_insert((ga.min(gb), ga.max(gb)));
            }
        }
        let mut re_ck = 0.0_f64;
        let mut im_ck = 0.0_f64;
        for pid in 0..n_owned {
            let dm = dp.unpermute_dof(pid as u32);
            let id = dm_to_key.get(&dm).and_then(|&k| key_to_edge.get(&k)).copied().unwrap_or(0);
            re_ck += (id as f64 + 1.0) * x.re.as_slice()[pid];
            im_ck += (id as f64 + 1.0) * x.im.as_slice()[pid];
        }
        let re_ck = comm.allreduce_sum_f64(re_ck);
        let im_ck = comm.allreduce_sum_f64(im_ck);
        if is_root {
            out.push_str(&format!("  Re: physical-edge checksum = {:.8e}\n", re_ck));
            out.push_str(&format!("  Im: physical-edge checksum = {:.8e}\n", im_ck));
        }
    }
    save_outputs(rank as usize, &local_mesh, &x);

    if is_root {
        out.push_str("\nFinished.\n");
        print!("{out}");
    }
}

// ─── Problem 1: vector H(Curl) ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn solve_hcurl(
    args: &Args,
    comm: &fem_parallel::Comm,
    full_mesh: &Mesh<3>,
    par_mesh: &fem_parallel::ParallelMesh<Mesh<3>>,
    port: &BoundarySubMesh,
    port_bc: &[f64],
    order: u8,
    mu: f64,
    eps: f64,
    sig: f64,
    omega: f64,
) {
    let rank = comm.rank();
    let is_root = rank == 0;
    let mut out = String::new();
    let qo = 2 * order + 1;

    let full_fes = HCurlSpace::new(full_mesh.clone(), order);
    let n = full_fes.n_dofs();
    let mut full_u_re = vec![0.0; n];
    transfer_edge_dofs(port, port_bc, &full_fes, &mut full_u_re);
    let full_u_im = vec![0.0; n];

    let local_mesh = par_mesh.local_mesh().clone();
    let local_space = HCurlSpace::new(local_mesh.clone(), order);
    let par_space = ParallelFESpace::new_for_edge_space(local_space, par_mesh, comm.clone());
    let dp = par_space.dof_partition();
    let n_owned = dp.n_owned_dofs;
    let n_ghost = dp.n_ghost_dofs;
    let n_local = n_owned + n_ghost;

    if is_root {
        out.push_str(&format!("Number of finite element unknowns: {}\n", par_space.n_global_dofs()));
    }
    let u_re_local = map_full_to_local(dp, &local_mesh, par_mesh.partition(), &full_u_re, 1, order, full_mesh);
    let u_im_local = map_full_to_local(dp, &local_mesh, par_mesh.partition(), &full_u_im, 1, order, full_mesh);
    // Complex system: Curl(a Curl) - ω² b + i ω c.
    let k_re = ParVectorAssembler::assemble_bilinear(
        &par_space,
        &[&CurlCurlIntegrator { mu: 1.0 / mu }, &VectorMassIntegrator { alpha: -omega * omega * eps }],
        qo,
    );
    let k_im = ParVectorAssembler::assemble_bilinear(
        &par_space,
        &[&VectorMassIntegrator { alpha: omega * sig }],
        qo,
    );
    let mut a_sys = ParComplexCsrMatrix::new(
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.diag_block(), k_im.diag_block()),
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.offd_block(), k_im.offd_block()),
        n_owned, n_ghost,
        par_space.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    // Essential DOFs: all boundary edges.  The boundary-edge set is computed
    // from the FULL serial mesh (deterministic, partition-independent — pex34
    // pattern), then mapped to the local partition.  Using the local mesh's
    // own boundary faces is unreliable after partitioning (some owned boundary
    // edges lose their boundary mark, observed on np2).
    let full_ess_keys: std::collections::HashSet<(u32, u32)> = {
        let mut s = std::collections::HashSet::new();
        let full_fes_b = HCurlSpace::new(full_mesh.clone(), order);
        for bf in 0..full_mesh.n_boundary_faces() as u32 {
            let nodes = full_mesh.face_nodes(bf);
            if nodes.len() >= 2 {
                for i in 0..nodes.len() {
                    let a = nodes[i];
                    let b = nodes[(i + 1) % nodes.len()];
                    s.insert((a.min(b), a.max(b)));
                }
            }
        }
        let _ = &full_fes_b;
        s
    };
    let lfes_ess = HCurlSpace::new(local_mesh.clone(), order);
    let mut ess_dm: Vec<u32> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for e in local_mesh.elem_iter() {
            let et = local_mesh.element_type(e);
            let ns = local_mesh.element_nodes(e);
            let dofs = lfes_ess.element_dofs(e);
            for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                let (ga, gb) = (par_mesh.partition().global_node(ns[a]), par_mesh.partition().global_node(ns[b]));
                let key = (ga.min(gb), ga.max(gb));
                if full_ess_keys.contains(&key) && seen.insert(dofs[k]) {
                    ess_dm.push(dofs[k]);
                }
            }
        }
    }
    let mut rhs = ParComplexVector::zeros_like(&ParVector::zeros(&par_space));
    let mut clamped: Vec<(usize, f64, f64)> = Vec::new();
    let mut ghost_ess: Vec<(usize, f64, f64)> = Vec::new();
    for &dm in &ess_dm {
        let pid = dp.permute_dof(dm) as usize;
        let vr = u_re_local[pid];
        let vi = u_im_local[pid];
        if pid < n_owned {
            clamped.push((pid, vr, vi));
        } else if pid < n_local {
            ghost_ess.push((pid - n_owned, vr, vi));
        }
    }
    for &(pid, vr, vi) in &clamped {
        a_sys.apply_dirichlet_par(pid, vr, vi, &mut rhs);
    }
    a_sys.apply_ghost_ess_columns(&ghost_ess, &mut rhs);
    if is_root {
        out.push_str(&format!("Size of linear system: {}\n", 2 * par_space.n_global_dofs()));
        out.push_str("\nSolving for the complex field using FGMRES with a block-diagonal preconditioner\n");
    }

    // Preconditioner [P, ±P]: AMS on the real block (C++ HypreAMS).
    let imag_scale = if args.herm_conv { -1.0 } else { 1.0 };
    let pc_op = ParVectorAssembler::assemble_bilinear(
        &par_space,
        &[&CurlCurlIntegrator { mu: 1.0 / mu },
          &VectorMassIntegrator { alpha: omega * omega * eps },
          &VectorMassIntegrator { alpha: omega * sig }],
        qo,
    );
    let mut pc_elim = clone_par_csr(&pc_op);
    {
        let mut rhs_dummy = ParVector::zeros(&par_space);
        for &(pid, _, _) in &clamped {
            pc_elim.apply_dirichlet_par_keep_diag(pid, 0.0, &mut rhs_dummy);
        }
    }

    let h1_par = {
        let h1 = H1Space::new(local_mesh.clone(), 1);
        ParallelFESpace::new(h1, par_mesh, comm.clone())
    };
    let grad = ParDiscreteLinearOperator::gradient(&h1_par, &par_space);
    let ams = std::sync::Arc::new(ParAmsPrecond::new(
        &pc_elim, &grad,
        fem_solver::AmsConfig {
            smoother_omega: 1.0,
            smoother_sweeps: 1,
            edge_smoother: linlvo::precond::AmsEdgeSmoother::SymmetricGaussSeidel,
            cycle: linlvo::precond::AmsCycle::MultiplicativeV11,
            face_space: false,
            node_solver: linlvo::precond::AuxSpaceSolver::Amg(linlvo::amg::AmgConfig {
                coarse_threshold: 9,
                max_levels: 25,
                ..Default::default()
            }),
            singularity_regularization: 1e-10,
        },
    ));
    let precond = move |r_re: &[f64], r_im: &[f64], z_re: &mut [f64], z_im: &mut [f64]| {
        ams.apply(r_re, z_re);
        let mut zz = vec![0.0; z_im.len()];
        ams.apply(r_im, &mut zz);
        for i in 0..z_im.len() {
            z_im[i] = imag_scale * zz[i];
        }
    };

    let cfg = fem_solver::SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 3000, verbose: std::env::var("PEX35_ND_VERBOSE").is_ok(), ..Default::default() };
    let mut x = ParComplexVector::zeros_like(&rhs.re);
    for i in 0..n_owned {
        x.re.owned_slice_mut()[i] = u_re_local[i];
        x.im.owned_slice_mut()[i] = u_im_local[i];
    }
    x.update_ghosts();

    let res = par_solve_fgmres_complex(&a_sys, &rhs, &mut x, 50, Some(&precond), &cfg)
        .expect("FGMRES failed");
    if is_root {
        out.push_str(&format!("  FGMRES: Number of iterations: {}\n", res.iterations));
        out.push_str(&format!("  FGMRES: Final relative residual: {:.6e}\n", res.final_residual));
    }

    x.update_ghosts();
    report_metrics(&x, comm, is_root, &mut out);
    if args.prob == 1 {
        // Physical-edge-id checksum (partition-independent, matches C++ at
        // np1: MFEM ND dof numbering = element×local-edge first-seen).
        let edges_for_elem = |et: fem_mesh::ElementType| -> &'static [(usize, usize)] {
            use fem_mesh::ElementType;
            match et {
                ElementType::Tet4 | ElementType::Tet10 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                ElementType::Hex8 | ElementType::Hex20 => &[
                    (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ],
                ElementType::Prism6 => &[
                    (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5),
                ],
                _ => &[],
            }
        };
        let mut key_to_edge: std::collections::HashMap<(u32, u32), u64> = Default::default();
        let mut next_edge = 0u64;
        for e in full_mesh.elem_iter() {
            let et = full_mesh.element_type(e);
            let ns = full_mesh.element_nodes(e);
            for &(a, b) in edges_for_elem(et) {
                let key = if ns[a] < ns[b] { (ns[a], ns[b]) } else { (ns[b], ns[a]) };
                if !key_to_edge.contains_key(&key) {
                    key_to_edge.insert(key, next_edge);
                    next_edge += 1;
                }
            }
        }
        let lnd = HCurlSpace::new(local_mesh.clone(), order);
        let mut dm_to_key: std::collections::HashMap<u32, (u32, u32)> = Default::default();
        for e in local_mesh.elem_iter() {
            let et = local_mesh.element_type(e);
            let ns = local_mesh.element_nodes(e);
            let dofs = lnd.element_dofs(e);
            for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                let (ga, gb) = (par_mesh.partition().global_node(ns[a]), par_mesh.partition().global_node(ns[b]));
                dm_to_key.entry(dofs[k]).or_insert((ga.min(gb), ga.max(gb)));
            }
        }
        let mut re_ck = 0.0_f64;
        let mut im_ck = 0.0_f64;
        for pid in 0..n_owned {
            let dm = dp.unpermute_dof(pid as u32);
            let id = dm_to_key.get(&dm).and_then(|&k| key_to_edge.get(&k)).copied().unwrap_or(0);
            re_ck += (id as f64 + 1.0) * x.re.as_slice()[pid];
            im_ck += (id as f64 + 1.0) * x.im.as_slice()[pid];
        }
        let re_ck = comm.allreduce_sum_f64(re_ck);
        let im_ck = comm.allreduce_sum_f64(im_ck);
        if is_root {
            out.push_str(&format!("  Re: physical-edge checksum = {:.8e}\n", re_ck));
            out.push_str(&format!("  Im: physical-edge checksum = {:.8e}\n", im_ck));
        }
    }
    save_outputs(rank as usize, &local_mesh, &x);

    if is_root {
        out.push_str("\nFinished.\n");
        print!("{out}");
    }
}

// ─── Problem 2: vector H(Div) ────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn solve_hdiv(
    args: &Args,
    comm: &fem_parallel::Comm,
    full_mesh: &Mesh<3>,
    par_mesh: &fem_parallel::ParallelMesh<Mesh<3>>,
    port: &BoundarySubMesh,
    port_bc: &[f64],
    order: u8,
    rt_order: u8,
    mu: f64,
    eps: f64,
    sig: f64,
    omega: f64,
) {
    let rank = comm.rank();
    let is_root = rank == 0;
    let mut out = String::new();
    let qo = 2 * order + 1;

    let full_fes = HDivSpace::new(full_mesh.clone(), rt_order);
    let n = full_fes.n_dofs();
    let mut full_u_re = vec![0.0; n];
    transfer_face_dofs(port, port_bc, &mut full_u_re);
    let full_u_im = vec![0.0; n];

    let local_mesh = par_mesh.local_mesh().clone();
    let local_space = HDivSpace::new(local_mesh.clone(), rt_order);
    let rt_part = DofPartition::from_face_space(
        &HDivSpace::new(local_mesh.clone(), rt_order),
        par_mesh.partition(),
        comm,
    );
    let par_space = ParallelFESpace::new_with_dof_partition(
        local_space,
        rt_part,
        comm.clone(),
    );
    let dp = par_space.dof_partition();
    let n_owned = dp.n_owned_dofs;
    let n_ghost = dp.n_ghost_dofs;
    let n_local = n_owned + n_ghost;

    if is_root {
        out.push_str(&format!("Number of finite element unknowns: {}\n", par_space.n_global_dofs()));
    }
    let u_re_local = map_full_to_local(dp, &local_mesh, par_mesh.partition(), &full_u_re, 2, order, full_mesh);
    let u_im_local = map_full_to_local(dp, &local_mesh, par_mesh.partition(), &full_u_im, 2, order, full_mesh);

    // Complex system: -Grad(a Div) - ω² b + i ω c.
    let k_re = ParVectorAssembler::assemble_bilinear(
        &par_space,
        &[&GradDivIntegrator { kappa: 1.0 / mu }, &VectorMassIntegrator { alpha: -omega * omega * eps }],
        qo,
    );
    let k_im = ParVectorAssembler::assemble_bilinear(
        &par_space,
        &[&VectorMassIntegrator { alpha: omega * sig }],
        qo,
    );
    let mut a_sys = ParComplexCsrMatrix::new(
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.diag_block(), k_im.diag_block()),
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.offd_block(), k_im.offd_block()),
        n_owned, n_ghost,
        par_space.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    // Essential DOFs: all boundary faces (RT0 face dof).  Boundary-face set
    // from the FULL mesh (partition-independent), matched locally (pex34
    // pattern).
    let full_bnd_faces: std::collections::HashSet<Vec<u32>> = {
        let mut s = std::collections::HashSet::new();
        for bf in 0..full_mesh.n_boundary_faces() as u32 {
            let mut nodes: Vec<u32> = full_mesh.face_nodes(bf).to_vec();
            nodes.sort_unstable();
            s.insert(nodes);
        }
        s
    };
    let lrt = HDivSpace::new(local_mesh.clone(), rt_order);
    let mut ess_dm: Vec<u32> = Vec::new();
    {
        use fem_space::dof_manager::FaceKey;
        let mut seen = std::collections::HashSet::new();
        for bf in 0..local_mesh.n_boundary_faces() as u32 {
            let nodes = local_mesh.face_nodes(bf);
            let mut g: Vec<u32> = nodes.iter()
                .map(|&n| par_mesh.partition().global_node(n)).collect();
            g.sort_unstable();
            if !full_bnd_faces.contains(&g) || nodes.len() < 3 {
                continue;
            }
            let dof = if nodes.len() == 3 {
                lrt.tri_face_dof(FaceKey::new(nodes[0], nodes[1], nodes[2]))
            } else {
                let mut found = None;
                for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
                    if let Some(d) = lrt.tri_face_dof(FaceKey::new(nodes[i], nodes[j], nodes[k])) {
                        found = Some(d);
                        break;
                    }
                }
                found
            };
            if let Some(d) = dof {
                if seen.insert(d) {
                    ess_dm.push(d);
                }
            }
        }
    }
    let mut rhs = ParComplexVector::zeros_like(&ParVector::zeros(&par_space));
    let mut clamped: Vec<(usize, f64, f64)> = Vec::new();
    let mut ghost_ess: Vec<(usize, f64, f64)> = Vec::new();
    for &dm in &ess_dm {
        let pid = dp.permute_dof(dm) as usize;
        let vr = u_re_local[pid];
        let vi = u_im_local[pid];
        if pid < n_owned {
            clamped.push((pid, vr, vi));
        } else if pid < n_local {
            ghost_ess.push((pid - n_owned, vr, vi));
        }
    }
    for &(pid, vr, vi) in &clamped {
        a_sys.apply_dirichlet_par(pid, vr, vi, &mut rhs);
    }
    a_sys.apply_ghost_ess_columns(&ghost_ess, &mut rhs);
    if is_root {
        out.push_str(&format!("Size of linear system: {}\n", 2 * par_space.n_global_dofs()));
        out.push_str("\nSolving for the complex field using FGMRES with a block-diagonal preconditioner\n");
    }

    // Preconditioner [P, ±P]: AMG on the real block (C++ HypreBoomerAMG).
    let imag_scale = if args.herm_conv { -1.0 } else { 1.0 };
    let pc_op = ParVectorAssembler::assemble_bilinear(
        &par_space,
        &[&GradDivIntegrator { kappa: 1.0 / mu },
          &VectorMassIntegrator { alpha: omega * omega * eps },
          &VectorMassIntegrator { alpha: omega * sig }],
        qo,
    );
    let mut pc_elim = clone_par_csr(&pc_op);
    {
        let mut rhs_dummy = ParVector::zeros(&par_space);
        for &(pid, _, _) in &clamped {
            pc_elim.apply_dirichlet_par_keep_diag(pid, 0.0, &mut rhs_dummy);
        }
    }
    let pc_diag = pc_elim.diag_block().clone();
    let amg = std::sync::Arc::new(linlvo::amg::AmgPrecond::new(linlvo::amg::AmgHierarchy::build(
        fem_linalg::fem_to_linlvo_csr(&pc_diag),
        linlvo::amg::AmgConfig {
            theta: 0.25,
            strategy: linlvo::amg::CoarsenStrategy::RugeStüben,
            smoother: linlvo::amg::SmootherType::GaussSeidel,
            pre_sweeps: 1,
            post_sweeps: 1,
            coarse_threshold: 9,
            max_levels: 25,
            ..Default::default()
        },
    )));
    let precond = move |r_re: &[f64], r_im: &[f64], z_re: &mut [f64], z_im: &mut [f64]| {
        let lr = linlvo::DenseVec::from_vec(r_re.to_vec());
        let mut lz = linlvo::DenseVec::zeros(z_re.len());
        amg.apply_precond(&lr, &mut lz);
        z_re.copy_from_slice(lz.as_slice());
        let lr2 = linlvo::DenseVec::from_vec(r_im.to_vec());
        let mut lz2 = linlvo::DenseVec::zeros(z_im.len());
        amg.apply_precond(&lr2, &mut lz2);
        for i in 0..z_im.len() {
            z_im[i] = imag_scale * lz2.as_slice()[i];
        }
    };

    let cfg = fem_solver::SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 3000, verbose: std::env::var("PEX35_ND_VERBOSE").is_ok(), ..Default::default() };
    let mut x = ParComplexVector::zeros_like(&rhs.re);
    for i in 0..n_owned {
        x.re.owned_slice_mut()[i] = u_re_local[i];
        x.im.owned_slice_mut()[i] = u_im_local[i];
    }
    x.update_ghosts();

    let res = par_solve_fgmres_complex(&a_sys, &rhs, &mut x, 50, Some(&precond), &cfg)
        .expect("FGMRES failed");
    if is_root {
        out.push_str(&format!("  FGMRES: Number of iterations: {}\n", res.iterations));
        out.push_str(&format!("  FGMRES: Final relative residual: {:.6e}\n", res.final_residual));
    }

    x.update_ghosts();
    report_metrics(&x, comm, is_root, &mut out);
    if args.prob == 1 {
        // Physical-edge-id checksum (partition-independent, matches C++ at
        // np1: MFEM ND dof numbering = element×local-edge first-seen).
        let edges_for_elem = |et: fem_mesh::ElementType| -> &'static [(usize, usize)] {
            use fem_mesh::ElementType;
            match et {
                ElementType::Tet4 | ElementType::Tet10 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                ElementType::Hex8 | ElementType::Hex20 => &[
                    (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ],
                ElementType::Prism6 => &[
                    (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5),
                ],
                _ => &[],
            }
        };
        let mut key_to_edge: std::collections::HashMap<(u32, u32), u64> = Default::default();
        let mut next_edge = 0u64;
        for e in full_mesh.elem_iter() {
            let et = full_mesh.element_type(e);
            let ns = full_mesh.element_nodes(e);
            for &(a, b) in edges_for_elem(et) {
                let key = if ns[a] < ns[b] { (ns[a], ns[b]) } else { (ns[b], ns[a]) };
                if !key_to_edge.contains_key(&key) {
                    key_to_edge.insert(key, next_edge);
                    next_edge += 1;
                }
            }
        }
        let lnd = HCurlSpace::new(local_mesh.clone(), order);
        let mut dm_to_key: std::collections::HashMap<u32, (u32, u32)> = Default::default();
        for e in local_mesh.elem_iter() {
            let et = local_mesh.element_type(e);
            let ns = local_mesh.element_nodes(e);
            let dofs = lnd.element_dofs(e);
            for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                let (ga, gb) = (par_mesh.partition().global_node(ns[a]), par_mesh.partition().global_node(ns[b]));
                dm_to_key.entry(dofs[k]).or_insert((ga.min(gb), ga.max(gb)));
            }
        }
        let mut re_ck = 0.0_f64;
        let mut im_ck = 0.0_f64;
        for pid in 0..n_owned {
            let dm = dp.unpermute_dof(pid as u32);
            let id = dm_to_key.get(&dm).and_then(|&k| key_to_edge.get(&k)).copied().unwrap_or(0);
            re_ck += (id as f64 + 1.0) * x.re.as_slice()[pid];
            im_ck += (id as f64 + 1.0) * x.im.as_slice()[pid];
        }
        let re_ck = comm.allreduce_sum_f64(re_ck);
        let im_ck = comm.allreduce_sum_f64(im_ck);
        if is_root {
            out.push_str(&format!("  Re: physical-edge checksum = {:.8e}\n", re_ck));
            out.push_str(&format!("  Im: physical-edge checksum = {:.8e}\n", im_ck));
        }
    }
    save_outputs(rank as usize, &local_mesh, &x);

    if is_root {
        out.push_str("\nFinished.\n");
        print!("{out}");
    }
}

/// Clone a `ParCsrMatrix` (it does not derive Clone �?rebuild from blocks).
fn clone_par_csr(a: &ParCsrMatrix) -> ParCsrMatrix {
    ParCsrMatrix::from_blocks(
        a.diag_block().clone(),
        a.offd_block().clone(),
        a.n_owned(),
        a.n_ghost(),
        a.ghost_exchange_handle(),
        a.comm().clone(),
    )
}

fn main() {
    let args = parse_args();

    let (mut mu, eps, sig) = (args.mu, args.eps, args.sig);
    if args.a_coef != 0.0 {
        mu = 1.0 / args.a_coef;
    }
    let omega = if args.freq > 0.0 { 2.0 * PI * args.freq } else { 2.0 * PI };

    let mut port_bc_attr = args.port_bc_attr.clone();
    if port_bc_attr.is_empty()
        && (args.mesh.ends_with("fichera-mixed.mesh") || args.mesh.ends_with("fichera.mesh"))
    {
        port_bc_attr = vec![7, 8, 11, 12];
    }

    let mesh_file = if args.mixed { args.mesh.as_str() } else { "data/fichera.mesh" };
    let mfem = read_mfem_file(mesh_file).expect("failed to read mesh");
    let mut full_mesh: Mesh<3> = mfem.mesh3d.expect("3D mesh required");
    for _ in 0..(args.ser_ref_levels + args.par_ref_levels) {
        full_mesh = refine_uniform_3d(&full_mesh);
    }

    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --refine-serial {}", args.ser_ref_levels);
    println!("   --refine-parallel {}", args.par_ref_levels);
    println!("   --order {}", args.order);
    println!("   --problem-type {}", args.prob);
    println!("   --eigenmode {}", args.mode);
    println!("   --stiffness-coef {}", args.a_coef);
    println!("   --permeability {mu}");
    println!("   --permittivity {eps}");
    println!("   --conductivity {sig}");
    println!("   --frequency {}", args.freq);
    println!(
        "   --port-bc-attr {}",
        port_bc_attr.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
    );
    println!("   --hermitian {}", args.herm_conv);
    println!("   --no-visualization true");
    println!("   --mixed-mesh {}", args.mixed);
    println!();

    let full_mesh_arc = std::sync::Arc::new(full_mesh);
    let port_bc_attr_arc = std::sync::Arc::new(port_bc_attr);

    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        let full_mesh = &*full_mesh_arc;
        let port_bc_attr = &*port_bc_attr_arc;
        let is_root = comm.rank() == 0;

        let port = extract_boundary_submesh(full_mesh, port_bc_attr);
        let order = args.order;
        let rt_order = order.saturating_sub(1);

        let port_bc: Vec<f64> = match args.prob {
            0 => scalar_waveguide(args.mode, &port.mesh, order),
            1 => vector_waveguide(args.mode, &port.mesh, order),
            2 => pseudo_scalar_waveguide(args.mode, &port.mesh, rt_order),
            _ => panic!("Unrecognized problem type: {}", args.prob),
        };

        let port_n: usize = match args.prob {
            0 => H1Space::new(port.mesh.clone(), order).n_dofs(),
            1 => HCurlSpace::new(port.mesh.clone(), order).n_dofs(),
            2 => L2Space::new(port.mesh.clone(), rt_order).n_dofs(),
            _ => 0,
        };
        if is_root {
            println!("Number of finite element port BC unknowns: {port_n}");
        }

        let par_mesh = partition_mesh(full_mesh, &comm);

        match args.prob {
            0 => solve_h1(&args, &comm, full_mesh, &par_mesh, &port, &port_bc, order, mu, eps, sig, omega),
            1 => solve_hcurl(&args, &comm, full_mesh, &par_mesh, &port, &port_bc, order, mu, eps, sig, omega),
            2 => solve_hdiv(&args, &comm, full_mesh, &par_mesh, &port, &port_bc, order, rt_order, mu, eps, sig, omega),
            _ => panic!("Unrecognized problem type: {}", args.prob),
        }
    });
}
