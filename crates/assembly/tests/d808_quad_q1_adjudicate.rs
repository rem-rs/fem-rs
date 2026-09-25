//! D797-2 / D808-1 — **adjudication probe** for the `[-1,1]²` quad Q1 PA frame.
//!
//! Registered debt D797-2: "`quad_q1.wgsl` 与 CPU builder 同为 `[-1,1]` 自洽但无帧
//! pin、与全树 `[0,1]` 口径不一致".  The question this file answers numerically:
//! is the quad Q1 PA path (builder + apply, and therefore the WGSL shader that
//! consumes the same `pd` buffer with the same `[-1,1]` tables) *self-consistent
//! with the assembled operator* on a mesh where the frame and the metric
//! orientation are both observable?
//!
//! The shipped unit test (`pa::quad_q1::tests::quad_q1_pa_matches_assembled`)
//! uses `Mesh::<2>::unit_square_quad(4)`: an **axis-aligned affine** mesh, where
//! `J` is diagonal, `J⁻¹ = J⁻ᵀ`, and the `[-1,1]²` frame differs from the
//! `[0,1]²` frame only by an overall constant that cancels between the
//! reference gradients and `det J`.  Neither a wrong frame nor a transposed
//! metric can show up there.
//!
//! This file warps the mesh with a smooth (non-affine) map, so
//! (a) the quadrature-point *locations* matter and (b) `J` is not symmetric.
//!
//! **Verdict (D808-1): (b) self-consistent.**  Measured on this machine
//! (`tmp/d808/quad_q1_adjudication.txt`):
//!
//! ```text
//! affine unit square (the shipped fixture): n=9 |A·x|max=6.458e-1  rel dev = 3.868e-16
//! warped (bilinear, non-symmetric J):      n=9 |A·x|max=6.349e-1  rel dev = 8.743e-16
//! ```
//!
//! The `[-1,1]²` pair is *not* a wrong quadrature: on that frame the reference
//! gradients (`l0/l1 = (1∓t)/2`), the stored `J⁻ᵀ`, the `|det J|` and the unit
//! Gauss weights are mutually consistent, and the `nq = 2` Gauss points of the
//! `[-1,1]` and `[0,1]` frames are exact images of one another (`ξ = 2t-1`,
//! `w_ξ = 2·w_t`), so the two frames differ by an overall factor that cancels
//! between the reference gradients and the Jacobian.  A transposed reference
//! metric would show up here as `O(1e-1)`; it does not, which also disposes of
//! the "cartesian-only fixture hides a `J⁻¹` vs `J⁻ᵀ` mix-up" worry for this
//! kernel (`pa::quad_q1` stores `jit = J⁻ᵀ`, `J[d][a] = ∂x_d/∂ξ_a`).
//!
//! The debt that remains open is therefore only the *pin* (及口径说明), which
//! lives on the shader side — `crates/linalg-gpu/tests/d808_quad_q1_wgsl_frame.rs`
//! pins the shipped `GP`/`GW` against the element layer's `[0,1]` rule, the
//! shipped kernel against the CPU kernel on the warped fixture above, and a
//! `[0,1]`-swap negative control.  This file is the CPU-side regression pin
//! that those shader pins rest on.

use fem_assembly::pa::{build_quad_q1_pa_data, pa_apply_quad_q1};
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// A 2×2 quad mesh of the unit square, pushed through a smooth warp so that
/// every element is a genuinely bilinear (non-affine) quad with a non-diagonal
/// Jacobian.
fn warped_quad_mesh() -> Mesh<2> {
    let mut mesh = Mesh::<2>::unit_square_quad(2);
    let mut coords = Vec::with_capacity(mesh.coords.len());
    for c in mesh.coords.chunks(2) {
        let (x, y) = (c[0], c[1]);
        coords.push(x + 0.18 * y * (1.0 - x) * x);
        coords.push(y + 0.13 * x * (1.0 - y) * y);
    }
    mesh.coords = coords;
    mesh
}

/// `max |PA·x − A·x| / max |A·x|` on one mesh.
fn pa_vs_assembled(mesh: &Mesh<2>) -> (f64, f64, usize) {
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();
    let a = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        2,
    );
    let pd = build_quad_q1_pa_data(mesh, &|_| 1.0);
    let elem_dofs: Vec<Vec<u32>> = (0..mesh.n_elems() as u32)
        .map(|e| space.element_dofs(e).to_vec())
        .collect();
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
    let mut y_pa = vec![0.0; n];
    pa_apply_quad_q1(&pd, &elem_dofs, &x, &mut y_pa);
    let mut y_asm = vec![0.0; n];
    a.spmv(&x, &mut y_asm);
    let num = y_pa
        .iter()
        .zip(y_asm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let den = y_asm.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    (num / den.max(1e-300), den, n)
}

#[test]
fn d808_quad_q1_pa_frame_adjudication() {
    for (name, mesh) in [
        ("affine unit square (the shipped fixture)", Mesh::<2>::unit_square_quad(2)),
        ("warped (bilinear, non-symmetric J)", warped_quad_mesh()),
    ] {
        let (rel, den, n) = pa_vs_assembled(&mesh);
        println!("{name}: n={n} |A·x|max={den:.3e}  rel dev = {rel:.3e}");
    }
    let (affine_rel, _, _) = pa_vs_assembled(&Mesh::<2>::unit_square_quad(2));
    let (warped_rel, _, _) = pa_vs_assembled(&warped_quad_mesh());
    assert!(affine_rel < 1e-12, "affine fixture: rel dev {affine_rel:.3e}");
    assert!(
        warped_rel < 1e-12,
        "warped fixture: quad Q1 PA vs assembled rel dev = {warped_rel:.3e} \
         (the `[-1,1]²` frame and/or the metric orientation is wrong off the \
         affine axis-aligned case)"
    );
}
