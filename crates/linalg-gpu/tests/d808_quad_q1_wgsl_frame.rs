//! D797-2 (round 75) — the **quadrature frame** of `wgsl/quad_q1.wgsl`.
//!
//! Registered debt: "`quad_q1.wgsl` 与 CPU builder 同为 `[-1,1]` 自洽但无帧 pin、
//! 与全树 `[0,1]` 口径不一致".  Round 73's D777 moved the hex shaders'
//! `GP`/`GW` to the element layer's `[0,1]` rule
//! ([`fem_element::quadrature::gauss_legendre_01`]); the 2-D quad Q1 pair
//! (shipped `quad_q1.wgsl` + `fem_assembly::pa::quad_q1`, i.e.
//! `build_quad_q1_pa_data` / `pa_apply_quad_q1`) stayed on the legacy `[-1,1]`
//! bilinear frame (`l0(t) = (1-t)/2`, `GL_PTS = ±1/√3`, `GL_WTS = 1`).
//!
//! **Adjudication (D808-1, numeric, no GPU needed).**  The pair is
//! *self-consistent*, and it is not a wrong quadrature: on the bilinear
//! `[-1,1]²` frame the diffusion integrand's physical gradient `J⁻ᵀ∇_ξφ`,
//! `|det J|` and the weights transform exactly, and the `nq = 2` Gauss points
//! of the two frames are images of one another (`ξ = 2t-1`).  The
//! adjudication fixture is
//! `crates/assembly/tests/d808_quad_q1_adjudicate.rs`: on a **warped**
//! (bilinear, non-symmetric-`J`) 2×2 quad mesh — where the shipped
//! unit-square test cannot see either a frame or a metric-orientation error —
//! PA vs `Assembler::assemble_bilinear` is `8.7e-16` relative, the same order
//! as on the axis-aligned affine fixture (`3.9e-16`).  What is *missing* is
//! the pin, and that is what this file adds:
//!
//! 1. [`shipped_quad_q1_frame_matches_cpu_pa`] — **no GPU needed**: the
//!    shipped shader text is parsed (`GP`, `GW`) and its `cs_main` is emulated
//!    in Rust from those numbers on the same `PaData` + element DOF list the
//!    CPU kernel takes, then compared with `pa_apply_quad_q1` on a **warped**
//!    quad (so the quadrature-point *locations* matter, not just the weights).
//!    The shader is element-major (`er.vals[e*4+i]`, no scatter), so the mesh
//!    is one element and the CPU side is gathered into the element's own DOF
//!    order.  This is a *cross-implementation* judgement: the shipped text is
//!    checked against an independent kernel, which text equality alone (a
//!    "file == generator output" pin) could never do.
//! 2. [`quad_q1_gp_gw_are_the_tree_frame_images`] — the frame judgement the
//!    shader cannot satisfy by itself: the shipped `[-1,1]` rule must be the
//!    image of the **element layer's** `[0,1]` rule under `ξ = 2t-1`
//!    (`gauss_legendre_01(2)`), node by node and weight by weight.  This is
//!    the machine-checked form of "the pair is a self-consistent `[-1,1]`
//!    path, equivalent to the tree's `[0,1]`口径, not a second convention".
//! 3. [`quad_q1_frame_swap_is_detected`] — the **negative control**: the same
//!    emulation with `GP`/`GW` replaced by the `[0,1]` rule (the mistake the
//!    hex shaders made before D777, and the one a "regenerate the shader from
//!    the element layer" change would introduce if the CPU builder were left
//!    alone) must *not* reproduce the CPU kernel.  Without this the numeric
//!    pin could be vacuous.
//! 4. [`shipped_quad_q1_wgsl_parses`] — no GPU needed: the shipped text must
//!    parse as WGSL (`naga`, the front end `wgpu` itself uses).  **This pin
//!    found a real defect on its first run (D808-4)**: the file used
//!    `let pa=if(a==0u){l0x}else{l1x};`, and WGSL has no `if`-*expressions*, so
//!    the shipped `quad_q1.wgsl` (and therefore `gpu_pa_apply_quad_q1`, and
//!    `build.rs`'s `f64` variant of it) had **never compiled** — the same
//!    structural-error class as D777's dropped `cs_main` brace.  The file's
//!    four `if`-expressions are now `select(l1x,l0x,a==0u)` (identical
//!    semantics); `tri3.wgsl` is checked alongside it.
//! 5. [`shipped_quad_q1_gpu_shader_matches_cpu_pa`] — the same comparison on
//!    the real device (`f32` path; self-skips with a visible `SKIP:` line when
//!    the machine has no wgpu adapter, exactly like `tests/gpu_mms.rs`).
//!
//! Keeping the pair on `[-1,1]` is deliberate: `pa::quad_qk` documents quad as
//! the crate's odd-one-out (`QuadQk` on `[0,1]²`), and the two frames are
//! numerically equivalent for this kernel, so migrating the shipped 2-D
//! shader without migrating its CPU builder (or vice versa) is the *only*
//! way to break it — which is exactly what pins 1 and 2 forbid.

use fem_assembly::pa::{build_quad_q1_pa_data, pa_apply_quad_q1};
use fem_element::lagrange::quad::QuadQ1;
use fem_element::ReferenceElement;
use fem_mesh::boundary::BoundaryTag;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// The shipped `f32` source in the working tree (read as a *file*, like
/// `tests/d777_pa_frame.rs`, not through the crate's private `include_str!`).
const SHIPPED_QUAD_Q1: &str = include_str!("../wgsl/quad_q1.wgsl");

/// `GP`/`GW` of the shipped shader, parsed from its own text.
struct ShippedFrame {
    gp: Vec<f64>,
    gw: Vec<f64>,
}

impl ShippedFrame {
    fn parse(src: &str) -> Self {
        let src = src.replace("\r\n", "\n");
        Self {
            gp: f64_array(&src, "const GP:array<f32,2>=array("),
            gw: f64_array(&src, "const GW:array<f32,2>=array("),
        }
    }

    /// Emulate `cs_main` of the shipped shader from the parsed constants: the
    /// **element-local** residual `ye` for element `e` of the `pd` buffer the
    /// CPU builder produced (`pd.data[(e*4+qi)*6 ..]`, `qi = qy*2+qx`), exactly
    /// as `run_pa_shader`'s `er` buffer holds it.
    ///
    /// `gp` is a parameter so the negative control can run the same kernel on a
    /// different rule.
    fn apply_element(&self, pd: &[f64], dofs: &[u32], x: &[f64], gp: &[f64], ye: &mut [f64; 4]) {
        let l0 = |t: f64| 0.5 * (1.0 - t);
        let l1 = |t: f64| 0.5 * (1.0 + t);
        let d0 = |_: f64| -0.5;
        let d1 = |_: f64| 0.5;
        let qa = |n: usize| (n & 1) ^ ((n >> 1) & 1);
        let qb = |n: usize| n >> 1;

        let mut xe = [0.0f64; 4];
        for i in 0..4 {
            xe[i] = x[dofs[i] as usize];
        }
        *ye = [0.0; 4];
        for qy in 0..2 {
            for qx in 0..2 {
                let qi = qy * 2 + qx;
                let off = qi * 6;
                let (jit00, jit01, jit10, jit11) =
                    (pd[off], pd[off + 1], pd[off + 2], pd[off + 3]);
                let sc = self.gw[qx] * self.gw[qy] * pd[off + 4] * pd[off + 5];
                let (l0x, l1x, d0x, d1x) = (l0(gp[qx]), l1(gp[qx]), d0(gp[qx]), d1(gp[qx]));
                let (l0y, l1y, d0y, d1y) = (l0(gp[qy]), l1(gp[qy]), d0(gp[qy]), d1(gp[qy]));
                let mut fl = [0.0f64; 2];
                for j in 0..4 {
                    let (a, b) = (qa(j), qb(j));
                    let (pa, pb) = (if a == 0 { l0x } else { l1x }, if b == 0 { l0y } else { l1y });
                    let (da, db) = (if a == 0 { d0x } else { d1x }, if b == 0 { d0y } else { d1y });
                    let pg = [jit00 * da * pb + jit01 * pa * db, jit10 * da * pb + jit11 * pa * db];
                    fl[0] += pg[0] * xe[j];
                    fl[1] += pg[1] * xe[j];
                }
                for i in 0..4 {
                    let (a, b) = (qa(i), qb(i));
                    let (pa, pb) = (if a == 0 { l0x } else { l1x }, if b == 0 { l0y } else { l1y });
                    let (da, db) = (if a == 0 { d0x } else { d1x }, if b == 0 { d0y } else { d1y });
                    let pg = [jit00 * da * pb + jit01 * pa * db, jit10 * da * pb + jit11 * pa * db];
                    ye[i] += sc * (pg[0] * fl[0] + pg[1] * fl[1]);
                }
            }
        }
    }
}

/// Read the numbers of the first `marker ... )` occurrence of a shader source.
fn f64_array(src: &str, marker: &str) -> Vec<f64> {
    let start = src.find(marker).unwrap_or_else(|| panic!("marker {marker:?} not found"));
    let after = &src[start + marker.len()..];
    let close = after.find(')').expect("array literal close");
    after[..close]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("f64 literal"))
        .collect()
}

/// One **warped** quadrilateral: the `QuadQ1` corners pushed through a smooth
/// map, so the bilinear Jacobian varies with the quadrature point and is not
/// diagonal.
fn warped_quad_mesh() -> Mesh<2> {
    let warp = |c: [f64; 2]| -> [f64; 2] {
        let (x, y) = (c[0], c[1]);
        [x + 0.25 * y * (1.0 - x) * x, y + 0.18 * x * (1.0 - y) * y]
    };
    let mut coords = Vec::new();
    for c in QuadQ1.dof_coords() {
        let p = warp([c[0], c[1]]);
        coords.extend_from_slice(&p);
    }
    let conn: Vec<u32> = (0..4).collect();
    let faces: [&[u32]; 4] = [&[0, 1], &[1, 2], &[2, 3], &[3, 0]];
    let face_conn: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
    let face_tags: Vec<BoundaryTag> = (1..=4).collect();
    Mesh::<2>::uniform(
        coords,
        conn,
        vec![1],
        fem_mesh::ElementType::Quad4,
        face_conn,
        face_tags,
        fem_mesh::ElementType::Line2,
    )
}

/// `(pd.data, element dof list, x, CPU element-local residual)` on the warped
/// single-element fixture.
fn cpu_pa() -> (Vec<f64>, Vec<u32>, Vec<f64>, [f64; 4]) {
    let mesh = warped_quad_mesh();
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();
    assert_eq!(n, 4, "one quad element");
    let pd = build_quad_q1_pa_data(space.mesh(), &|_| 1.0);
    let dofs: Vec<u32> = space.element_dofs(0).to_vec();
    let elem_dofs: Vec<Vec<u32>> = vec![dofs.clone()];
    let mut rng: u64 = 42;
    let x: Vec<f64> = (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 11) as f64) / ((1u64 << 53) as f64)
        })
        .collect();
    let mut y = vec![0.0; n];
    pa_apply_quad_q1(&pd, &elem_dofs, &x, &mut y);
    // One element ⇒ every global DOF is written exactly once, so the global
    // vector *is* the element-local residual in `dofs` order.
    let local: [f64; 4] = std::array::from_fn(|i| y[dofs[i] as usize]);
    (pd.data, dofs, x, local)
}

fn max_relative_error(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| {
            let d = (a - b).abs();
            if d.is_nan() || !a.is_finite() {
                f64::INFINITY
            } else {
                d
            }
        })
        .fold(0.0f64, f64::max);
    worst / scale
}

/// Pin 1 (no GPU): the shipped shader's kernel, evaluated from its own parsed
/// `GP`/`GW` on the CPU caller's `pd` buffer, reproduces `pa_apply_quad_q1` on
/// a warped quad — where the quadrature-point *locations* are observable.
#[test]
fn shipped_quad_q1_frame_matches_cpu_pa() {
    let sh = ShippedFrame::parse(SHIPPED_QUAD_Q1);
    let (pd, dofs, x, y_cpu) = cpu_pa();
    let mut y_shader = [0.0f64; 4];
    sh.apply_element(&pd, &dofs, &x, &sh.gp, &mut y_shader);
    let rel = max_relative_error(&y_shader, &y_cpu);
    assert!(
        rel < 1e-12,
        "shipped quad_q1.wgsl kernel vs CPU `pa_apply_quad_q1`: relative error {rel:.3e}\n\
         shader GP = {:?}\nshader GW = {:?}",
        sh.gp,
        sh.gw
    );
    eprintln!("D808-1 shipped quad_q1 wgsl vs CPU PA on a warped quad: rel = {rel:.3e}");
    // Non-degenerate fixture: the residual is not uniformly zero/flat.
    assert!(y_cpu.iter().any(|v| v.abs() > 1e-3), "flat fixture: {y_cpu:?}");
}

/// Pin 2: the frame judgement.  The shipped `[-1,1]` rule must be the image of
/// the **element layer's** `[0,1]` rule under `ξ = 2t-1` — the rule
/// `hex_rule`/`quad_rule_01` build (`gauss_legendre_01`), which is what the
/// `pd` buffer the shader consumes is framed on.  A shader-only pin cannot
/// make this judgement (any `[-1,1]` table would satisfy "the file equals
/// itself"); this ties the shipped numbers to the element crate.
#[test]
fn quad_q1_gp_gw_are_the_tree_frame_images() {
    let sh = ShippedFrame::parse(SHIPPED_QUAD_Q1);
    let (x01, w01) = fem_element::quadrature::gauss_legendre_01(2);
    assert_eq!(sh.gp.len(), 2);
    assert_eq!(sh.gw.len(), 2);
    for i in 0..2 {
        // `[0,1] → [-1,1]`: ξ = 2t-1, w_ξ = 2·w_t (Σw = 2 instead of 1).
        let want_gp = 2.0 * x01[i] - 1.0;
        let want_gw = 2.0 * w01[i];
        assert!(
            (sh.gp[i] - want_gp).abs() < 1e-15,
            "GP[{i}] = {:.17e} is not the [0,1] Gauss node {:.17e} mapped to [-1,1] \
             (want {:.17e})",
            sh.gp[i],
            x01[i],
            want_gp
        );
        assert!(
            (sh.gw[i] - want_gw).abs() < 1e-15,
            "GW[{i}] = {:.17e} is not 2× the [0,1] weight {:.17e}",
            sh.gw[i],
            w01[i]
        );
    }
    // Σw = 2 is the `[-1,1]` signature (the `[0,1]` rule sums to 1): the frame
    // the shipped file is *in* is `[-1,1]`, and the test says so out loud.
    let wsum: f64 = sh.gw.iter().sum();
    assert!((wsum - 2.0).abs() < 1e-15, "GW sums to {wsum}, not 2");
}

/// Pin 3: the negative control.  Swapping the shipped `GP`/`GW` for the `[0,1]`
/// rule (the pre-D777 hex mistake, i.e. what a "make the shader match the
/// element layer" change without the matching CPU change would do) must fail to
/// reproduce the CPU kernel by a *macroscopic* margin.  This is what makes
/// pin 1 a real numeric pin and not a tautology.
#[test]
fn quad_q1_frame_swap_is_detected() {
    let sh = ShippedFrame::parse(SHIPPED_QUAD_Q1);
    let (pd, dofs, x, y_cpu) = cpu_pa();
    let (x01, _w01) = fem_element::quadrature::gauss_legendre_01(2);
    assert_ne!(x01, sh.gp, "the control rule must differ from the shipped one");
    let mut y_wrong = [0.0f64; 4];
    sh.apply_element(&pd, &dofs, &x, &x01, &mut y_wrong);
    let rel = max_relative_error(&y_wrong, &y_cpu);
    // The `pd` buffer carries the `[-1,1]`-framed reference metric and
    // `|det J|`; evaluating the reference basis *and* the rule on `[0,1]`
    // against it is a different quadrature, not round-off.
    assert!(
        rel > 1e-3,
        "the frame pin is vacuous: a `[0,1]` GP/GW swap changes the result by only {rel:.3e}"
    );
    eprintln!("D808-1 frame-swap control ([0,1] rule against the [-1,1] pd): rel = {rel:.3e}");
}

/// Pin 3 (no GPU, no device): the shipped source must **parse as WGSL**.
///
/// This is the round-73 lesson made executable: a pin whose only judgement is
/// "the file equals what a generator/text edit expects" cannot see a
/// *structural* shader error.  It has now bitten twice —
/// * D777: the hex template dropped the closing brace of `cs_main`, so neither
///   the shipped `q3`/`q4` nor any dynamically generated degree compiled;
/// * D808-4 (found by this round's device pin): `quad_q1.wgsl` used
///   `let pa=if(a==0u){l0x}else{l1x};` — WGSL has no `if`-*expressions*
///   (naga: "expected `;`, found `{`"), so `gpu_pa_apply_quad_q1` had never
///   compiled a pipeline, and `wgpu`'s validation error surfaced only as a
///   panic on the device path.
///
/// `naga` is the same front end `wgpu` uses (dev-dependency, `wgsl-in`), so
/// this pin runs everywhere — including machines where the device pin
/// self-skips.
#[test]
fn shipped_quad_q1_wgsl_parses() {
    let src = SHIPPED_QUAD_Q1.replace("\r\n", "\n");
    match naga::front::wgsl::parse_str(&src) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.emit_to_string(&src);
            panic!("wgsl/quad_q1.wgsl does not parse as WGSL:\n{msg}");
        }
    }
    // The other shipped 2-D shader, for the same reason (`pa_apply::TRI3_WGSL`).
    let tri = include_str!("../wgsl/tri3.wgsl").replace("\r\n", "\n");
    naga::front::wgsl::parse_str(&tri)
        .unwrap_or_else(|e| panic!("wgsl/tri3.wgsl does not parse as WGSL:\n{}", e.emit_to_string(&tri)));
}

/// Pin 4 (GPU, self-skipping): the shipped shader compiled and run by wgpu on
/// the `f32` path reproduces the CPU PA apply to `f32` accuracy.
#[test]
fn shipped_quad_q1_gpu_shader_matches_cpu_pa() {
    let Some(gpu) = gpu_ctx() else { return };
    let (pd, dofs, x, y_cpu) = cpu_pa();
    let pd32: Vec<f32> = pd.iter().map(|v| *v as f32).collect();
    let x32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    let mut y32 = vec![0.0f32; x.len()];
    fem_linalg_gpu::pa_apply::gpu_pa_apply_quad_q1(&gpu, &pd32, &dofs, &x32, &mut y32);
    let y_gpu: Vec<f64> = y32.iter().map(|v| *v as f64).collect();
    // `run_pa_shader` copies `er` verbatim: the GPU result is **element
    // local** (`res[e*ldof + i]`), which is the order `cpu_pa` already
    // gathered the CPU side into (one element ⇒ every DOF appears once).
    let rel = max_relative_error(&y_gpu, &y_cpu);
    assert!(
        rel < 1e-4,
        "GPU f32 quad_q1 PA vs CPU PA: relative error {rel:.3e}"
    );
    eprintln!("D808-1 GPU f32 quad_q1: relative error vs CPU PA = {rel:.3e}");
}

/// GPU context or `None` (after a visible `SKIP` line) when this machine has no
/// wgpu adapter — the same contract as `tests/gpu_mms.rs`.
fn gpu_ctx() -> Option<fem_linalg_gpu::GpuContext> {
    match pollster::block_on(fem_linalg_gpu::GpuContext::new()) {
        Ok(gpu) => Some(gpu),
        Err(fem_linalg_gpu::GpuError::NoAdapter) => {
            eprintln!("SKIP: no GPU adapter");
            None
        }
        Err(e) => panic!("GPU context error: {e}"),
    }
}
