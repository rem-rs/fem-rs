//! d106 regression pin: the round-106 D957 closeout — the tesla curl-curl
//! solve through the full MFEM `HypreAMS` analog (face-space `id_ND` Pi
//! blocks + singular block-Pi cycle `0345430`) under the **hypre PCG
//! convergence semantics** (`two_norm = 0`: the printed/stopped norm is the
//! preconditioned one, `gamma = <C*r,r>/<C*b,b> < tol²`, hypre
//! `krylov/pcg.c:278`), pinned against the d106 acceptance runs
//! (`tmp/d106ams_final_*.log`).
//!
//! Round-106 findings pinned here:
//! * the red→green iteration counts: ball-quad `-bm` o2 **9** (C++ 8; the
//!   residual 1-iteration gap was attributed in round 106 to the linlvo
//!   SA-AMG vs BoomerAMG quality inside `B_Pi` — fem-rs's iteration-8 C-norm
//!   ratio 1.97e-12 sits just over the 1e-12 stop line,
//!   `tmp/d106ams_final_bm_o2_ballquad.log`),
//!   inline-hex `-bm` o2 **7** = C++ 7;
//! * the JD anchors (`||M||`, `||JD||`) stay bitwise against the C++ probe;
//! * `<C*b,b>` (the AMS quadratic form on the rhs) matches the C++ probe to
//!   0.5% (1.2417e-7 vs 1.2476e-7 ball-quad, 1.0189e-8 vs 1.0190e-8
//!   inline-hex).
//!
//! Round-107 (D1042 dissection, `tmp/d107amg/REPORT.md`): the round-106
//! attribution is **refuted** — with the B_Pi blocks replaced by exact LU
//! solves (single-level hierarchy) ball-quad gets *worse* (14 iterations),
//! and every B_Pi strength knob moves both tiers one step together (V(2,1):
//! 8/6; RS: 11/6; plain GS: 7/6), so the ball-quad 9 vs 8 gap is NOT B_Pi
//! accuracy.  The per-tier alignment (D1063, open) must come from the arm
//! spectral character elsewhere.  After the D1000-LOR singleton-absorption
//! fix the ball-quad `<C*b,b>` pin moves to 1.2406164773014794e-7; the
//! `‖M‖/‖JD‖`/`‖B‖` anchors stay bitwise/13-digit
//! (`tmp/d107amg/rs_bm_o2_*.log`).
//!
//! Round-109 (D1076 close, `tmp/d109a/REPORT.md`): the B_Pi arm is now the
//! **hypre-faithful HMIS(10)+aggressive(`Create2ndS`)+interp-6 stack**
//! (`CoarsenStrategy::HmisAms`), verified stage-by-stage against hypre 2.28
//! itself (level-0 multipass P bitwise equal, V-cycle arm equal to ~1e-13)
//! after correcting the round-108 ledger: BoomerAMG relax-8's l1 comes from
//! `ComputeL1Norms` option 4, which degenerates to the signed diagonal on
//! one rank ⇒ plain symmetric Gauss-Seidel (`par_amg_setup.c:3280`,
//! `ams.c:678-692`).  Iterations drop to **7 / 6** (C++ 8 / 7) — one BETTER
//! than the C++ probe on both tiers, because linlvo's assembled `A_Pi`
//! operators are measurably different from hypre's `PixᵀAPix`
//! (`tmp/d109a/`: hypre's own arm gives ‖y‖² = 3.433e-8 on ours vs
//! 2.794e-8 on C++'s; first PCG residual 7.6e-6 vs 1.36e-5).  The
//! `<C*b,b>` pins move to 1.262911894545921e-7 / 1.0190330289083565e-8.
//! The residual assembly-parity delta is registered as D1095 (assembly
//! lane territory, not the AMG).
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

/// The hypre `hypre_PCGSolve` port (`two_norm = 0`, MFEM sets no
/// `SetUseTwoNorm`): preconditioned-norm table, stop at
/// `gamma = <C*r,r>/<C*b,b> < tol²` (pcg.c:406/677).  Returns
/// `(iterations, <C*b,b>)`.
fn ams_pcg_iterations(
    a: &fem_parallel::ParCsrMatrix,
    b: &ParVector,
    ams: &AmsPrecond<f64>,
    n_owned: usize,
    rtol: f64,
    max_iter: usize,
) -> (usize, f64) {
    let apply = |r: &[f64], z: &mut [f64]| {
        let lr = DenseVec::from_vec(r.to_vec());
        let mut lz = DenseVec::zeros(n_owned);
        ams.apply_precond(&lr, &mut lz);
        z.copy_from_slice(lz.as_slice());
    };
    // Pre-loop: p = C·b; bi_prod = <C*b,b> (pcg.c:357-378).
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

        // The D957 face-space AMS (tesla_solver.cpp:355-365 semantics).
        let nd_dp = nd.dof_partition();
        let h1_dp = h1.dof_partition();
        let pi_canonical = assemble_pi_blocks(h1.local_space(), nd.local_space());
        let pi_local: Vec<fem_linalg::CsrMatrix<f64>> = pi_canonical
            .iter()
            .map(|m| keep_owned_rows(&permute_rect_csr(m, nd_dp, h1_dp), nd_dp.n_owned_dofs))
            .collect();
        let la = fem_linalg::fem_to_linlvo_csr(curl_mu_inv_curl.diag_block());
        let grad = ParDiscreteLinearOperator::gradient(&h1, &nd);
        let lg = fem_linalg::fem_to_linlvo_csr(&grad);
        let lpi: Vec<linlvo::CsrMatrix<f64>> =
            pi_local.iter().map(fem_linalg::fem_to_linlvo_csr).collect();
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
            ams_pcg_iterations(&curl_mu_inv_curl, &jd, &ams, nd_dp.n_owned_dofs, 1e-12, 50);
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
    // C++ 8; since the round-109 hypre-faithful B_Pi arm (HMIS+aggressive +
    // interp-6, plain-SGS levels) the PCG lands one iteration EARLIER
    // (module doc + D1095): 9 → 7.  The `<C*b,b>` pin moved
    // 1.2406164773014794e-7 → 1.262911894545921e-7 (+1.8%, now 1.2% above
    // the C++ probe's 1.2476e-7).
    run_case(path, 517, 1460, 7, 1.262_911_894_545_921e-7);
}

#[test]
fn d106_tesla_d957_ams_inlinehex_o2() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d103tesla/inline-hex.mesh");
    run_case(path, 729, 1944, 6, 1.019_033_028_908_356_5e-8);
}
