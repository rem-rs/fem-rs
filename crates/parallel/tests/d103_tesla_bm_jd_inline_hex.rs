//! d103 regression pin: the tesla `-bm` RHS pipeline on the **straight**
//! `inline-hex.mesh`, where fem-rs is bit-exact against the C++ reference.
//!
//! C++ truth (`tmp/d103tesla/cpp_bm_o2_inlinehex.log`, MFEM 4.10 tesla probe,
//! `mpirun -np 1`, `-o 2 -m inline-hex.mesh -bm '0.5 0.5 0.3 0.5 0.5 0.7 0.1 1'
//! -maxit 1 -no-vis -no-visit`):
//!
//! ```text
//! Number of H1      unknowns: 729
//! Number of H(Curl) unknowns: 1944
//! Number of H(Div)  unknowns: 1728
//! PROBE ||M||_2  = 2.1650635094610959e-01
//! PROBE ||JD||_2 = 3.7073613048202986e-01
//! ```
//!
//! `M` is MFEM's `ParGridFunction::ProjectCoefficient` on the RT1 space (nodal
//! face/interior dof functionals) and `jd = mu0 · WeakCurlMuInv · M` with the
//! **default** `VectorFECurlIntegrator` rule `trial order + test order − 1 =
//! 2·order − 2` (tesla_solver.cpp:335-338 sets no custom IntRule for this
//! form).  The JD pin is red with any deviation of that quadrature order
//! (measured: using the custom `irOrder = OrderW + 2·order` rule instead gives
//! `||JD|| = 3.84e-1`).

use fem_assembly::postproc::coefficient::ScalarCoeff;
use fem_io::mfem::read_mfem_file;
use fem_parallel::par_assembler::permute_vec;
use fem_parallel::par_mixed_assembler::ParMixedAssembler;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::{Comm, ParDiscreteLinearOperator, ParVector, WorkerConfig, partition_mesh};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

/// Vacuum permeability `mu0_` (electromagnetics.hpp).
const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

/// `bar_magnet` (tesla.cpp:429-462) with the sample-run parameters.
fn bar_magnet(x: &[f64]) -> [f64; 3] {
    let bm = [0.5_f64, 0.5, 0.3, 0.5, 0.5, 0.7, 0.1, 1.0];
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

#[test]
fn d103_tesla_bm_jd_pipeline_inline_hex_o2() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d103tesla/inline-hex.mesh");
    let mfem = read_mfem_file(path).expect("inline-hex.mesh reads");
    let mesh0 = mfem.mesh3d.expect("3-D mesh");

    // dof counts == MFEM GlobalTrueVSize (cpp_bm_o2_inlinehex.log).
    assert_eq!(H1Space::new(mesh0.clone(), 2).n_dofs(), 729);
    assert_eq!(HCurlSpace::new(mesh0.clone(), 2).n_dofs(), 1944);
    assert_eq!(HDivSpace::new(mesh0.clone(), 1).n_dofs(), 1728);

    // The miniapp's parallel pipeline at --ranks 1.
    let mesh0 = std::sync::Arc::new(mesh0);
    let launcher = ThreadLauncher::new(WorkerConfig::new(1));
    launcher.launch(move |comm: Comm| {
        let par_mesh = partition_mesh(&mesh0, &comm);
        let local = par_mesh.local_mesh().clone();
        let order = 2_u32;

        let h1 = ParallelFESpace::new(
            H1Space::new(local.clone(), order as u8),
            &par_mesh,
            comm.clone(),
        );
        let nd = ParallelFESpace::new(
            HCurlSpace::new(local.clone(), order as u8),
            &par_mesh,
            comm.clone(),
        );
        let rt = ParallelFESpace::new(
            HDivSpace::new(local.clone(), order.saturating_sub(1) as u8),
            &par_mesh,
            comm.clone(),
        );
        assert_eq!(nd.n_global_dofs(), 1944);
        assert_eq!(rt.n_global_dofs(), 1728);

        // M: the RT nodal interpolation (engine — bit-exact on straight hexes).
        let m_local = rt
            .local_space()
            .interpolate_vector(&|x: &[f64]| {
                let v = bar_magnet(x);
                vec![v[0], v[1], v[2]]
            })
            .as_slice()
            .to_vec();
        let m2: f64 = m_local.iter().map(|v| v * v).sum();
        let m_norm = m2.sqrt();
        assert!(
            (m_norm - 2.165_063_509_461_095_9e-1).abs() < 5e-16,
            "||M|| {m_norm:.16e} vs C++ 2.1650635094610959e-01"
        );

        // jd = mu0 · WeakCurlMuInv · M with the VectorFECurlIntegrator default
        // rule (RT order + ND order − 1 = 2·order − 2).
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
        let jd2: f64 = (0..nd.dof_partition().n_owned_dofs)
            .map(|i| {
                let v = jd.as_slice()[i];
                v * v
            })
            .sum();
        let jd_norm = comm.allreduce_sum_f64(jd2).sqrt();
        assert!(
            (jd_norm - 3.707_361_304_820_298_6e-1).abs() < 5e-16,
            "||JD|| {jd_norm:.16e} vs C++ 3.7073613048202986e-01"
        );

        // The gradient exists for the AMS (shape smoke check).
        let grad = ParDiscreteLinearOperator::gradient(&h1, &nd);
        assert_eq!(grad.nrows, nd.dof_partition().n_owned_dofs);
    });
}

/// μ⁻¹ = 1/mu0 (vacuum — no `-ms`/`-pwm`).
struct VacMuInv;
impl ScalarCoeff for VacMuInv {
    fn eval(&self, _ctx: &fem_assembly::postproc::coefficient::CoeffCtx<'_>) -> f64 {
        1.0 / MU0
    }
}
