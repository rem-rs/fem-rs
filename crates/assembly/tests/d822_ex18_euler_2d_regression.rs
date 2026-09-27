//! D822-3 (round 84) — the **2-D Euler face-flux stride** regression pin.
//!
//! # Root cause (bisected to `24e00a8d`, round-81 D815-2)
//!
//! D815-2 shared the face-data layout between the 2-D and 3-D arms of
//! [`DgHyperbolicConservationLaws`]: `nor_qp` went `[f64; 2]` → `[f64; 3]`
//! (2-D z padded with 0).  The residual loops passed the **padded 3-slice**
//! into `numerical_flux`, where `rusanov_combine` takes `dim = normal.len()`
//! as the flux-tensor stride — but the 2-D `EulerFlux::compute_flux` writes
//! stride-2 rows (`flux_out[eq·2 + d]`).  The read-back `fl[eq·3 + d]` was
//! therefore scrambled for eq = 1, 2 (momenta read neighbours' fluxes) and
//! identically zero for eq = 3 (energy face flux vanished); only eq = 0
//! happened to line up.  Free-stream preservation broke and ex18's default
//! (periodic-square, order 1) diverged to `Solution error: NaN` from step 27
//! (t ≈ 0.07) — silently, rc = 0.  The fix truncates the face normal to the
//! operator's `self.dim` at the two flux-call sites, which restores the
//! pre-D815-2 slice bit for bit (2-D) and is the identity slice in 3-D.
//!
//! # What is pinned here
//!
//! Both tests run the operator itself (`residual_into` through the public
//! `mult` / `mult_residual`) on a tiny Quad4 mesh — the layer where the bug
//! lived.  Either test goes red if the padded normal ever reaches
//! `rusanov_combine` again:
//!
//! * `free_stream_preservation_2d_quad` — MFEM-anchored **exact** property:
//!   a consistent numerical flux has `f̂(q, q, n̂) = F(q)·n̂`, so a uniform
//!   state's pre-mass residual `z` is pure cancellation noise (the d816c 3-D
//!   oracle used the same property as its truth anchor, ~1e-16 on straight
//!   fixtures).  Under the bad stride the momentum/energy face terms stop
//!   cancelling the volume term ⇒ residuals O(0.1).
//! * `rk3_evolution_end_state_pin_2d_quad` — ex18's SSP-RK3 over a smooth,
//!   always-physical nonuniform state; pins the deterministic end-state
//!   signature.  The bad arithmetic drops the entire energy face flux and
//!   scrambles the momentum fluxes ⇒ the trajectory leaves the pinned basin
//!   by orders of magnitude within the 40 steps.
//!
//! Empirical teeth (round 84, `tmp/d84fix/`): both red on `24e00a8d`'s
//! arithmetic (free stream max|z| = 4.330e-1; evolution deviation 2.5e-2
//! relative), green with the fix.

use fem_assembly::dg::{DgHyperbolicConservationLaws, EulerFlux, RusanovFlux};
use fem_io::mfem::read_mfem;
use fem_mesh::simplex::Mesh;

const GAMMA: f64 = 1.4;

/// QuadL2GL(1) dofs per element on the 2×2 quad mesh.
const DP: usize = 4;
/// 2-D Euler conserved variables: (ρ, ρu, ρv, E).
const NQ: usize = 4;
/// Elements in the 2×2 quad mesh below.
const N_ELEM: usize = 4;

/// 2×2 Quad4 unit square, straight, no boundary elements — every outer edge
/// is an unpaired edge, i.e. the operator's reflecting wall.
const QUAD2X2: &str = "MFEM mesh v1.0

dimension
2

elements
4
1 3 0 1 4 3
1 3 1 2 5 4
1 3 3 4 7 6
1 3 4 5 8 7

boundary
0

vertices
9
2
0 0
0.5 0
1 0
0 0.5
0.5 0.5
1 0.5
0 1
0.5 1
1 1
";

fn mesh_2x2() -> Mesh<2> {
    read_mfem(std::io::Cursor::new(QUAD2X2.as_bytes().to_vec()))
        .expect("read the inline quad mesh")
        .mesh2d
        .expect("2-D mesh")
}

fn build_op(mesh: &Mesh<2>) -> DgHyperbolicConservationLaws {
    DgHyperbolicConservationLaws::new(
        mesh,
        1,
        Box::new(RusanovFlux { inner: EulerFlux { gamma: GAMMA } }),
        true, // ex18's volume term (MFEM HyperbolicFormIntegrator ∫F·∇v)
    )
}

/// The operator's vector layout: `(e·dp + j)·nq + eq`.
fn state(u: &mut [f64], e: usize, j: usize, eq: usize) -> &mut f64 {
    &mut u[(e * DP + j) * NQ + eq]
}

/// Free-stream preservation (MFEM-anchored exact property — see module doc).
#[test]
fn free_stream_preservation_2d_quad() {
    let mesh = mesh_2x2();
    let op = build_op(&mesh);
    assert_eq!(op.n_dofs(), N_ELEM * DP * NQ);

    // Uniform gas at rest: [ρ, ρu, ρv, E] = [1, 0, 0, 2.5] (γ = 1.4 ⇒ p = 1).
    let u = vec![1.0, 0.0, 0.0, 2.5]
        .into_iter()
        .cycle()
        .take(N_ELEM * DP * NQ)
        .collect::<Vec<f64>>();

    let mut z = vec![0.0; op.n_dofs()];
    op.mult_residual(&u, &mut z);
    let max_abs = z.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(
        max_abs < 1e-12,
        "free-stream preservation broken: max|z| = {max_abs:.3e} \
         (padded face normal reaching rusanov_combine scrambles the 2-D \
         stride-2 Euler flux read-back — the D822-3 regression)"
    );
}

/// ex18's SSP-RK3 (examples/mfem_ex18_euler.rs `step`) over the operator.
fn ssprk3(op: &DgHyperbolicConservationLaws, dt: f64, steps: usize, u: &mut [f64]) {
    let n = u.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut u1 = vec![0.0; n];
    let mut u2 = vec![0.0; n];
    for _ in 0..steps {
        op.mult(u, &mut k1);
        for i in 0..n {
            u1[i] = u[i] + dt * k1[i];
        }
        op.mult(&u1, &mut k2);
        for i in 0..n {
            u2[i] = 0.75 * u[i] + 0.25 * (u1[i] + dt * k2[i]);
        }
        op.mult(&u2, &mut k3);
        for i in 0..n {
            u[i] = u[i] / 3.0 + 2.0 / 3.0 * (u2[i] + dt * k3[i]);
        }
    }
}

/// Smooth, always-physical nonuniform state from an integer dof formula
/// (d816c's u2 recipe, 2-D): no projection needed — the dof values ARE the
/// state, so the pin exercises the operator in isolation.
fn nonuniform_state() -> Vec<f64> {
    let mut u = vec![0.0; N_ELEM * DP * NQ];
    for e in 0..N_ELEM {
        for j in 0..DP {
            let g = e * DP + j;
            let rho = 1.0 + 0.025 * (g % 5) as f64; // 1.000 .. 1.100
            let mx = 0.1 * (((g / 3) % 3) as f64 - 1.0); // -0.1, 0, 0.1
            let my = 0.05 * ((2 * ((g / 7) % 2)) as f64 - 1.0); // -0.05, 0.05
            let pr = 1.0 + 0.05 * (g % 7) as f64; // 1.00 .. 1.30
            let energy = pr / (GAMMA - 1.0) + 0.5 * rho * (mx * mx + my * my);
            *state(&mut u, e, j, 0) = rho;
            *state(&mut u, e, j, 1) = rho * mx;
            *state(&mut u, e, j, 2) = rho * my;
            *state(&mut u, e, j, 3) = energy;
        }
    }
    u
}

/// Deterministic end-state signature of a short SSP-RK3 evolution (module
/// doc: why this bites on the bad stride arithmetic).
#[test]
fn rk3_evolution_end_state_pin_2d_quad() {
    let mesh = mesh_2x2();
    let op = build_op(&mesh);

    let mut u = nonuniform_state();
    // CFL = dt·c/h ≈ 0.005·1.5/0.5 = 0.015 — deeply inside the stable region,
    // so the only way this test moves is a real change of the flux arithmetic.
    ssprk3(&op, 5.0e-3, 40, &mut u);

    let norm: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
    // Observed with the D822-3 fix in place (round 84, tmp/d84fix/).  The
    // bad-arithmetic value is 1.24310876736033418e1 — a 2.5e-2 relative
    // deviation, nine orders above the tolerance.
    const PIN: f64 = 12.1215111491817;
    const RTOL: f64 = 1e-9;
    assert!(
        (norm - PIN).abs() <= RTOL * PIN.abs(),
        "RK3 end-state signature drifted: {norm:.17e} vs pinned {PIN:.17e} \
         (2-D Euler face-flux arithmetic changed — if intentional, re-pin \
         against an MFEM oracle run)"
    );
}
