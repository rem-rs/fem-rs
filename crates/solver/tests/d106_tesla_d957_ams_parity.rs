//! d106 regression pin: the round-106 D957 closeout — the tesla curl-curl
//! solve through the full MFEM `HypreAMS` analog (face-space `id_ND` Pi
//! blocks + singular block-Pi cycle `0345430`) under the **hypre PCG
//! convergence semantics** (`two_norm = 0`: the printed/stopped norm is the
//! preconditioned one, `gamma = <C*r,r>/<C*b,b> < tol²`, hypre
//! `krylov/pcg.c:278`), pinned against the d116a acceptance values.
//!
//! Round-116 (D1230 close, `tmp/d116a/REPORT.md`): **exact C++ parity** —
//! ball-quad `-bm` o2 **8** and inline-hex `-bm` o2 **7** = the C++ probe's
//! 8/7, with `<C*b,b>` = 1.2476451863821637e-7 / 1.0190254659113007e-8
//! matching the C++ probe's printed `1.247645e-07` / `1.019025e-08` in every
//! digit (`tmp/d105ams/cpp_bm_o2_*.log`).  Three mechanisms closed together:
//! * the linlvo singular cycle ran "03455430" — one EXTRA `B_Piz` arm —
//!   where hypre 2.28 `ams.c:3710` spells case-13 (singular) as `"0345430"`
//!   = GS Px Py Pz **Py Px** GS (the reverse sweep drops the second Pz;
//!   `hypre_ParCSRSubspacePrec` digit table ams.c:3625-3632/3965-3968);
//! * `rap_hypre_order` emitted the KT coarse operator through
//!   `from_coo`, sorting each row — hypre stores the RAP rows in their
//!   first-touch creation order (par_rap.c build loop), and the next AMG
//!   level's l1 norms / HMIS measures sum over that stored order;
//! * the linlvo handoff now carries the D1230 in-row storage order (A =
//!   [diagonal] + reverse first-touch insertion, `hypre.cpp:943`
//!   `hypre_CSRMatrixReorder` swap over MFEM's prepended linked list; Pi =
//!   reverse of the first-inserting host's trial-dof list, `SetSubMatrix`
//!   prepend `bilinearform.cpp:2502`, no rectangular reorder) plus the
//!   D1095 canonical renumbering — level-0 `A_Pix` is bitwise 29521/29521
//!   against the real MFEM 4.10 + hypre 2.28 run dump and its row order
//!   matches 438/517 (the residual 79 rows carry the fem-rs-vs-MFEM H1 hex
//!   local dof order, registered as D1246: measured effect ≤ 1.5e-7
//!   relative on `<C*b,b>`, no hierarchy tie flips).
//!
//! Historical pins (rounds 106-115, superseded by the exact parity):
//! 9/7→7/6 (r109), `<C*b,b>` 1.2417e-7→1.2406e-7→1.2629e-7 (d106/d107/r109),
//! 1.0190330289083565e-8→1.0190329983168683e-8 (d110a).
//!
//! C++ truth: `tmp/d105ams/cpp_bm_o2_ballquad.log` /
//! `cpp_bm_o2_inlinehex.log` (MFEM 4.10 tesla probe, `mpirun -np 1`).

use fem_assembly::postproc::coefficient::ScalarCoeff;
use fem_assembly::standard::CurlCurlIntegrator;
use fem_io::mfem::read_mfem_file;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_assembler::permute_vec;
use fem_parallel::par_mixed_assembler::{permute_rect_csr, ParMixedAssembler};
use fem_parallel::{
    Comm, ParDiscreteLinearOperator, ParVector, ParVectorAssembler, ParallelFESpace, ParallelMesh,
    WorkerConfig, partition_mesh,
};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};
use linlvo::amg::{AmgConfig, SmootherType as AuxSmoother};
use linlvo::precond::{AmsConfig, AmsCycle, AmsEdgeSmoother, AmsPrecond, AuxSpaceSolver};
use linlvo::{DenseVec, Preconditioner};

/// Vacuum permeability `mu0_` (electromagnetics.hpp).
const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

/// `bar_magnet` (tesla.cpp:429-462), the d103 acceptance parameters.
fn bar_magnet(x: &[f64]) -> [f64; 3] {
    let bm = [0.0_f64, -0.5, 0.0, 0.0, 0.5, 0.0, 0.2, 1.0];
    let a = [bm[3] - bm[0], bm[4] - bm[1], bm[5] - bm[2]];
    let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if h == 0.0 {
        return [0.0; 3];
    }
    let r = bm[6];
    let xu = [x[0] - bm[0], x[1] - bm[1], x[2] - bm[2]];
    let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
    let xu_perp = [
        xu[0] - xa / (h * h) * a[0],
        xu[1] - xa / (h * h) * a[1],
        xu[2] - xa / (h * h) * a[2],
    ];
    let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2]).sqrt();
    if xa >= 0.0 && xa <= h * h && xp <= r {
        let s = bm[7] / h;
        [s * a[0], s * a[1], s * a[2]]
    } else {
        [0.0; 3]
    }
}

/// `ConstantCoefficient(1/mu0_)`.
struct VacMuInv;
impl ScalarCoeff for VacMuInv {
    fn eval(&self, _ctx: &fem_assembly::postproc::coefficient::CoeffCtx<'_>) -> f64 {
        1.0 / MU0
    }
}

/// The three `id_ND` interpolation blocks `Pi_x/Pi_y/Pi_z` (the MFEM
/// `HypreAMS::MakeGradientAndInterpolation` payload — `Project_ND` point
/// functionals `pi_d(k,j) = phi_j(x_k)·(J(x_k)·t_k)_d`, hex meshes only).
/// Port of the tesla miniapp's `assemble_pi_blocks`.
fn assemble_pi_blocks(
    h1: &H1Space<fem_mesh::Mesh<3>>,
    nd: &HCurlSpace<fem_mesh::Mesh<3>>,
) -> [fem_linalg::CsrMatrix<f64>; 3] {
    use fem_element::nedelec::HexNDk;
    use fem_mesh::element_type::ElementType;

    let mesh = h1.mesh_topology();
    let hnd = HexNDk::new(nd.order() as usize);
    let anchors = hnd.dof_anchors();
    let n_nd = nd.n_dofs();
    let n_h1 = h1.n_dofs();
    let p_h1 = h1.get_order();

    let mut host_of: Vec<(u32, usize)> = vec![(u32::MAX, usize::MAX); n_nd];
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        assert!(matches!(et, ElementType::Hex8 | ElementType::Hex20), "{et:?}");
        for (m, &gdof) in nd.element_dofs(e).iter().enumerate() {
            host_of[gdof as usize] = (e, m);
        }
    }

    let mut coo = [
        fem_linalg::CooMatrix::<f64>::new(n_nd, n_h1),
        fem_linalg::CooMatrix::<f64>::new(n_nd, n_h1),
        fem_linalg::CooMatrix::<f64>::new(n_nd, n_h1),
    ];
    for (dof, &(e, m)) in host_of.iter().enumerate() {
        assert!(m != usize::MAX, "ND dof {dof} without host element");
        let et = mesh.element_type(e);
        let h1_ref = fem_space::ref_elem::h1_field_element(et, p_h1, h1.pyramid_basis());
        let h1_dofs = h1.element_dofs_u32(e);
        let signs = nd.element_signs(e);
        let (xi, tk) = &anchors[m];
        let (jac, _x) = fem_mesh::element_jacobian_at(mesh, e, xi, 3);
        let j = [
            [jac[(0, 0)], jac[(0, 1)], jac[(0, 2)]],
            [jac[(1, 0)], jac[(1, 1)], jac[(1, 2)]],
            [jac[(2, 0)], jac[(2, 1)], jac[(2, 2)]],
        ];
        let mut shape = vec![0.0_f64; h1_ref.n_dofs()];
        h1_ref.eval_basis(xi, &mut shape);
        let mut t = [0.0_f64; 3];
        for (r, t_r) in t.iter_mut().enumerate() {
            *t_r = j[r][0] * tk[0] + j[r][1] * tk[1] + j[r][2] * tk[2];
        }
        let s = signs[m];
        for (&gdof, &phij) in h1_dofs.iter().zip(shape.iter()) {
            let v = s * phij;
            coo[0].add(dof, gdof as usize, v * t[0]);
            coo[1].add(dof, gdof as usize, v * t[1]);
            coo[2].add(dof, gdof as usize, v * t[2]);
        }
    }
    [
        std::mem::replace(&mut coo[0], fem_linalg::CooMatrix::new(0, 0)).into_csr(),
        std::mem::replace(&mut coo[1], fem_linalg::CooMatrix::new(0, 0)).into_csr(),
        std::mem::replace(&mut coo[2], fem_linalg::CooMatrix::new(0, 0)).into_csr(),
    ]
}

/// Owned-row extraction of a partition-permuted rectangular local matrix.
fn keep_owned_rows(mat: &fem_linalg::CsrMatrix<f64>, n_owned: usize) -> fem_linalg::CsrMatrix<f64> {
    let mut coo = fem_linalg::CooMatrix::<f64>::new(n_owned, mat.ncols);
    for row in 0..n_owned.min(mat.nrows) {
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            coo.add(row, mat.col_idx[k] as usize, mat.values[k]);
        }
    }
    coo.into_csr()
}

/// Row/column renumbering of a linlvo CSR (the D1095 canonical handoff; the
/// tesla miniapp's twin).  Pattern-preserving — entries move, values stay.
/// NOTE: emitted through COO + `from_coo`, so rows come out sorted; used only
/// for G (order-insensitive in the singular AMS cycle, ams.c).
fn permute_linlvo_csr(
    m: &linlvo::CsrMatrix<f64>,
    row_perm: Option<&[usize]>,
    col_perm: Option<&[usize]>,
) -> linlvo::CsrMatrix<f64> {
    let mut coo = linlvo::sparse::CooMatrix::<f64>::new(m.nrows(), m.ncols());
    for r in 0..m.nrows() {
        let nr = match row_perm {
            Some(p) => p[r],
            None => r,
        };
        for k in m.row_ptr()[r]..m.row_ptr()[r + 1] {
            let c = m.col_idx()[k];
            let nc = match col_perm {
                Some(p) => p[c],
                None => c,
            };
            coo.push(nr, nc, m.values()[k]);
        }
    }
    linlvo::sparse::CsrMatrix::from_coo(&coo)
}

/// The hypre `hypre_PCGSolve` port (`two_norm = 0`, MFEM sets no
/// `SetUseTwoNorm`): preconditioned-norm table, stop at
/// `gamma = <C*r,r>/<C*b,b> < tol²` (pcg.c:406/677).  `perm` is the D1095
/// partition→canonical map the AMS was built under (vectors swapped in/out
/// at the preconditioner boundary, the tesla miniapp's `TeslaAms::apply`
/// twin).  Returns `(iterations, <C*b,b>)`.
fn ams_pcg_iterations(
    a: &fem_parallel::ParCsrMatrix,
    b: &ParVector,
    ams: &AmsPrecond<f64>,
    perm: &[usize],
    rtol: f64,
    max_iter: usize,
) -> (usize, f64) {
    let apply = |r: &[f64], z: &mut [f64]| {
        // partition → canonical → AMS → canonical → partition.
        let mut rc = vec![0.0_f64; perm.len()];
        for (p, &c) in perm.iter().enumerate() {
            rc[c] = r[p];
        }
        let lr = DenseVec::from_vec(rc);
        let mut lz = DenseVec::zeros(perm.len());
        ams.apply_precond(&lr, &mut lz);
        let lz = lz.as_slice();
        for (p, &c) in perm.iter().enumerate() {
            z[p] = lz[c];
        }
    };
    // Pre-loop: p = C·b; bi_prod = <C*b,b> (pcg.c:357-378).
    let n_owned = b.as_slice().len();
    let mut cb = ParVector::zeros_like(b);
    apply(b.as_slice(), cb.as_slice_mut());
    let bi_prod = b.global_dot(&cb);
    assert!(bi_prod > 0.0, "zero rhs");
    let eps = rtol * rtol;

    // r = b - A·x with x₀ = 0; p = C·r; gamma = <r, p>.
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    let mut x = ParVector::zeros_like(b);
    a.spmv(&mut x, &mut ax);
    for i in 0..n_owned {
        r.as_slice_mut()[i] = b.as_slice()[i] - ax.as_slice()[i];
    }
    let mut z = ParVector::zeros_like(b);
    apply(r.as_slice(), z.as_slice_mut());
    let mut rz = r.global_dot(&z);
    let mut p = z.clone_vec();

    let mut iterations = 0usize;
    while iterations < max_iter {
        iterations += 1;
        let mut ap = ParVector::zeros_like(b);
        let mut pm = p.clone_vec();
        a.spmv(&mut pm, &mut ap);
        let sdotp = p.global_dot(&ap);
        if sdotp == 0.0 {
            break; // pcg.c:528 "Zero sdotp value in PCG"
        }
        let alpha = rz / sdotp;
        for i in 0..n_owned {
            x.as_slice_mut()[i] += alpha * p.as_slice()[i];
            r.as_slice_mut()[i] -= alpha * ap.as_slice()[i];
        }
        // s = C·r; gamma = <r, s> (pcg.c:589-591).
        apply(r.as_slice(), z.as_slice_mut());
        let rz_new = r.global_dot(&z);
        // The basic convergence test (pcg.c:677): i_prod/bi_prod < eps.
        if rz_new / bi_prod < eps {
            break;
        }
        // Subnormal gamma guard (pcg.c:703-707).
        if !(rz_new > f64::MIN_POSITIVE) {
            break;
        }
        let beta = rz_new / rz;
        for i in 0..n_owned {
            p.as_slice_mut()[i] = z.as_slice()[i] + beta * p.as_slice()[i];
        }
        rz = rz_new;
    }
    (iterations, bi_prod)
}

/// One D957 acceptance solve: `-bm` o2 curl-curl + the face-space AMS +
/// hypre-semantics PCG.  Returns `(iterations, <C*b,b>)`.
fn run_case(mesh_path: &str, n_h1_expect: usize, n_nd_expect: usize, it_want: usize, cb_want: f64) {
    let mfem = read_mfem_file(mesh_path).expect("mesh reads");
    let mesh0 = std::sync::Arc::new(mfem.mesh3d.expect("3-D mesh"));
    let order = 2_u32;
    let launcher = ThreadLauncher::new(WorkerConfig::new(1));
    launcher.launch(move |comm: Comm| {
        let par_mesh: ParallelMesh<fem_mesh::Mesh<3>> = partition_mesh(&mesh0, &comm);
        let local = par_mesh.local_mesh().clone();
        let h1 =
            ParallelFESpace::new(H1Space::new(local.clone(), order as u8), &par_mesh, comm.clone());
        let nd =
            ParallelFESpace::new(HCurlSpace::new(local.clone(), order as u8), &par_mesh, comm.clone());
        let rt = ParallelFESpace::new(
            HDivSpace::new(local.clone(), order.saturating_sub(1) as u8),
            &par_mesh,
            comm.clone(),
        );
        assert_eq!(h1.n_global_dofs(), n_h1_expect);
        assert_eq!(nd.n_global_dofs(), n_nd_expect);

        // curlMuInvCurl: CurlCurlIntegrator(muInv), MFEM default rule
        // (2·order on hexes; bilininteg.cpp CurlCurlIntegrator).
        let curl_mu_inv_curl = ParVectorAssembler::assemble_bilinear(
            &nd,
            &[&CurlCurlIntegrator { mu: VacMuInv }],
            2 * order as u8,
        );

        // jd = mu0 · weakCurlMuInv · M (the d103-pinned pipeline; the
        // VectorFECurlIntegrator default rule 2·order − 2).
        let m_local = rt
            .local_space()
            .interpolate_vector(&|x: &[f64]| {
                let v = bar_magnet(x);
                vec![v[0], v[1], v[2]]
            })
            .as_slice()
            .to_vec();
        let weak_curl = ParMixedAssembler::assemble_hdiv_hcurl_curl_with_coeff(
            &nd,
            &rt,
            (2 * order - 2) as u8,
            VacMuInv,
        );
        let mut m_vec = ParVector::from_local_raw(
            permute_vec(&m_local, rt.dof_partition()),
            rt.dof_partition().n_owned_dofs,
            rt.dof_ghost_exchange_arc(),
            comm.clone(),
        );
        m_vec.update_ghosts();
        let mut jd = ParVector::zeros(&nd);
        weak_curl.spmv(m_vec.as_slice(), jd.as_slice_mut());
        for v in jd.as_slice_mut() {
            *v *= MU0;
        }

        // The D957 face-space AMS (tesla_solver.cpp:355-365 semantics) under
        // the exact C++→hypre handoff: the D1095 canonical dof renumbering
        // plus the D1230 in-row storage order — A = [diagonal] + reverse
        // first-touch insertion with the diagonal swapped into slot 0
        // (`hypre.cpp:943` + MFEM's prepended linked list), Pi = reverse of
        // the first-inserting host's trial-dof list (`SetSubMatrix` prepend,
        // `bilinearform.cpp:2502`, no rectangular reorder).  Without it the
        // HMIS measures of the level-1+ operators divert at the ulp level and
        // the run tie-breaks away from the C++ probe (d116a).
        let nd_dp = nd.dof_partition();
        let h1_dp = h1.dof_partition();
        let nd_canon: Vec<usize> = (0..nd_dp.n_total_dofs())
            .map(|p| nd_dp.unpermute_dof(p as u32) as usize)
            .collect();
        let h1_canon: Vec<usize> = (0..h1_dp.n_total_dofs())
            .map(|p| h1_dp.unpermute_dof(p as u32) as usize)
            .collect();
        let nd_canon_u32: Vec<u32> = nd_canon.iter().map(|&c| c as u32).collect();
        let h1_canon_u32: Vec<u32> = h1_canon.iter().map(|&c| c as u32).collect();
        let nd_local = nd.local_space();
        let element_dofs: Vec<Vec<u32>> = (0..nd_local.mesh_topology().n_elements() as u32)
            .map(|e| nd_local.element_dofs(e).to_vec())
            .collect();
        // First-inserting host per canonical ND dof: later `SetSubMatrix`
        // hosts overwrite values but insert nothing new, so the first host
        // alone fixes the handoff row order.
        let mut first_host = vec![u32::MAX; nd_dp.n_total_dofs()];
        for e in 0..nd_local.mesh_topology().n_elements() as u32 {
            for &d in nd_local.element_dofs(e) {
                if first_host[d as usize] == u32::MAX {
                    first_host[d as usize] = e;
                }
            }
        }
        let host_part: Vec<u32> = (0..nd_dp.n_total_dofs())
            .map(|p| first_host[nd_canon[p]])
            .collect();
        let h1_local = h1.local_space();
        let h1_element_dofs: Vec<Vec<u32>> = (0..h1_local.mesh_topology().n_elements() as u32)
            .map(|e| h1_local.element_dofs_u32(e).to_vec())
            .collect();
        let pi_canonical = assemble_pi_blocks(h1.local_space(), nd.local_space());
        let pi_local: Vec<fem_linalg::CsrMatrix<f64>> = pi_canonical
            .iter()
            .map(|m| keep_owned_rows(&permute_rect_csr(m, nd_dp, h1_dp), nd_dp.n_owned_dofs))
            .collect();
        let a_canon = fem_parallel::par_ptap_handoff::mfem_ptap_handoff_matrix(
            curl_mu_inv_curl.diag_block(),
            &nd_canon_u32,
            &element_dofs,
        );
        let la = fem_linalg::fem_to_linlvo_csr(&a_canon);
        let grad = ParDiscreteLinearOperator::gradient(&h1, &nd);
        let lg = permute_linlvo_csr(
            &fem_linalg::fem_to_linlvo_csr(&grad),
            Some(&nd_canon),
            Some(&h1_canon),
        );
        let lpi: Vec<linlvo::CsrMatrix<f64>> = pi_local
            .iter()
            .map(|m| {
                fem_linalg::fem_to_linlvo_csr(&fem_parallel::par_ptap_handoff::mfem_discrete_op_handoff_matrix(
                    m,
                    &nd_canon_u32,
                    &h1_canon_u32,
                    &host_part,
                    &h1_element_dofs,
                ))
            })
            .collect();
        let ams = AmsPrecond::<f64>::with_pi(
            &la,
            &lg,
            &lpi,
            AmsConfig {
                edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
                cycle: AmsCycle::MultiplicativeV11,
                singular_problem: true,
                node_solver: AuxSpaceSolver::Amg(AmgConfig {
                    smoother: AuxSmoother::L1SymmetricGaussSeidel,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .expect("AMS setup");

        let (iters, cb_b) =
            ams_pcg_iterations(&curl_mu_inv_curl, &jd, &ams, &nd_canon, 1e-12, 50);
        assert_eq!(iters, it_want, "D957 AMS PCG iterations vs the d106 pin");
        assert!(
            (cb_b - cb_want).abs() / cb_want < 1e-6,
            "<C*b,b> {cb_b:.16e} vs d106 pin {cb_want:.16e}"
        );
    });
}

#[test]
fn d106_tesla_d957_ams_ballquad_o2() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d103tesla/ball-quad.mesh");
    // Round-116 (D1230 close): **exact C++ parity** — 8 iterations and
    // `<C*b,b>` = 1.2476451863821637e-7 against the C++ probe's printed
    // 1.247645e-07 (`tmp/d105ams/cpp_bm_o2_ballquad.log`; the module doc
    // carries the three mechanisms: the 03455430→0345430 cycle fix, the
    // rap first-touch emission, the D1095+D1230 handoff order).
    run_case(path, 517, 1460, 8, 1.247_645_186_382_163_7e-7);
}

#[test]
fn d106_tesla_d957_ams_inlinehex_o2() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d103tesla/inline-hex.mesh");
    // Round-116: 7 iterations = C++ 7; `<C*b,b>` matches the C++ probe's
    // printed 1.019025e-08 in every digit.
    run_case(path, 729, 1944, 7, 1.019_025_465_911_300_7e-8);
}
