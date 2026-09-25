//! D777 (round 73): the GPU hex PA WGSL shaders' **quadrature frame**.
//!
//! Round 72's D721 residue (`tmp/d721x/README.md`) regenerated the
//! `bary/dary` node arrays of `wgsl/hex_q{3,4}.wgsl` to the element layer's
//! `[0,1]` Gauss–Lobatto nodes, but the generator kept emitting its own
//! `gauss_legendre_f64(nq)` table for `GP`/`GW` — a `[-1,1]` rule — while
//! every other implementation in the tree is `[0,1]`: the element layer's
//! `hex_rule`/`quad_rule_01` (via `gauss_legendre_01`), the CPU PA kernels
//! (`pa_apply_hex_qk` / `hex_q1` / `q2` / `q3` / `q4`) that build the very
//! `pd` buffer the shaders consume, and — before D777 — the `p = 1`/`p = 2`
//! shaders' own bespoke bases, which were a *self-consistent* `[-1,1]` frame
//! and therefore differed from the CPU path by an overall factor (`2` in 3-D),
//! while the round-72 `q3`/`q4` pair was frame-inconsistent outright.
//!
//! What this file pins:
//!
//! 1. [`shipped_hex_qk_shader_matches_cpu_pa`] — **no GPU needed**: the shipped
//!    shader text is parsed (`GP`, `GW`, the `bary`/`dary` node array, the
//!    `QA/QB/QC` slot tables) and its compute kernel is emulated in Rust from
//!    those parsed numbers on the same `PaData` + element DOF list the CPU
//!    kernel takes, then compared against `fem_assembly::pa::pa_apply_hex_qk`.
//!    Red before the D777 fix (a frame error is `O(1)`, not round-off);
//!    green after (only the tensor-contraction order differs).
//! 2. [`shipped_hex_qk_gpu_shader_matches_cpu_pa`] — the same comparison on the
//!    real device (`f32` path; self-skips with a visible `SKIP:` line when the
//!    machine has no wgpu adapter, exactly like `tests/gpu_mms.rs`).
//!
//! Both use a **single distorted (non-affine) hexahedron**, so the shader's
//! quadrature-point *locations* matter, not just its weights, and the GPU's
//! element-major result layout (`run_pa_shader` copies `er` verbatim) is
//! identical to the CPU's global scatter (one element ⇒ every DOF written once).

use fem_assembly::pa::{build_hex_qk_pa_data, pa_apply_hex_qk};
use fem_element::lagrange::hex::HexQ1;
use fem_element::ReferenceElement;
use fem_mesh::boundary::BoundaryTag;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// `f32` sources shipped in `wgsl/` (read from the working tree, not through
/// the crate's private `include_str!`s, so this test sees the *files*).
const SHIPPED: [(&str, usize); 4] = [
    (include_str!("../wgsl/hex_q1.wgsl"), 1),
    (include_str!("../wgsl/hex_q2.wgsl"), 2),
    (include_str!("../wgsl/hex_q3.wgsl"), 3),
    (include_str!("../wgsl/hex_q4.wgsl"), 4),
];

/// The shipped shader's constants and slot tables, parsed from its own text.
struct ShippedShader {
    nq: usize,
    nloc: usize,
    gp: Vec<f64>,
    gw: Vec<f64>,
    nodes: Vec<f64>,
    qa: Vec<u32>,
    qb: Vec<u32>,
    qc: Vec<u32>,
}

impl ShippedShader {
    fn parse(src: &str, p: usize) -> Self {
        let src = src.replace("\r\n", "\n");
        let nq = p + 1;
        let nloc = nq * nq * nq;
        let gp = f64_array(&src, &format!("const GP:array<f32,{nq}>=array("));
        let gw = f64_array(&src, &format!("const GW:array<f32,{nq}>=array("));
        // The node array is the first `let n=` of `bary` (the derivative
        // re-declares the same numbers).
        let nodes = f64_array(&src, &format!("let n=array<f32,{nq}>("));
        let axis = |name: &str| -> Vec<u32> {
            u32_array(&src, &format!("const {name}:array<u32,{nloc}>=array("))
        };
        Self {
            nq,
            nloc,
            gp,
            gw,
            nodes,
            qa: axis("QA"),
            qb: axis("QB"),
            qc: axis("QC"),
        }
    }

    /// The shader's `bary` — the product form over the parsed node array.
    fn bary(&self, t: f64, i: usize) -> f64 {
        let mut r = 1.0;
        for j in 0..self.nq {
            if j != i {
                r *= (t - self.nodes[j]) / (self.nodes[i] - self.nodes[j]);
            }
        }
        r
    }

    /// The shader's `dary` — the exact derivative of the product form, which
    /// stays valid when `t` lands on a node (the `p+1`-point Gauss rule
    /// contains `ξ = 0` for even `p`, and `0` is a GLL node there).
    fn dary(&self, t: f64, i: usize) -> f64 {
        let mut r = 0.0;
        for m in 0..self.nq {
            if m == i {
                continue;
            }
            let mut term = 1.0 / (self.nodes[i] - self.nodes[m]);
            for j in 0..self.nq {
                if j != i && j != m {
                    term *= (t - self.nodes[j]) / (self.nodes[i] - self.nodes[j]);
                }
            }
            r += term;
        }
        r
    }

    /// Emulate `cs_main` of the shipped shader element by element, from the
    /// parsed constants and slot tables.  `pd.data`, `dofs` and `x` are exactly
    /// the buffers `run_pa_shader` uploads (11 numbers per quadrature point).
    fn apply(&self, pd: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64]) {
        let (nq, nloc) = (self.nq, self.nloc);
        let nqp = nq * nq;
        let ne = dofs.len() / nloc;
        for e in 0..ne {
            let mut xe = vec![0.0f64; nloc];
            for i in 0..nloc {
                xe[i] = x[dofs[e * nloc + i] as usize];
            }
            let mut ye = vec![0.0f64; nloc];
            for qz in 0..nq {
                for qy in 0..nq {
                    for qx in 0..nq {
                        let qi = qz * nqp + qy * nq + qx;
                        let off = (e * nloc + qi) * 11;
                        let (j00, j01, j02) = (pd[off], pd[off + 1], pd[off + 2]);
                        let (j10, j11, j12) = (pd[off + 3], pd[off + 4], pd[off + 5]);
                        let (j20, j21, j22) = (pd[off + 6], pd[off + 7], pd[off + 8]);
                        let sc = self.gw[qx] * self.gw[qy] * self.gw[qz] * pd[off + 9] * pd[off + 10];
                        let (mut bx, mut dx) = (vec![0.0; nq], vec![0.0; nq]);
                        let (mut by, mut dy) = (vec![0.0; nq], vec![0.0; nq]);
                        let (mut bz, mut dz) = (vec![0.0; nq], vec![0.0; nq]);
                        for i in 0..nq {
                            bx[i] = self.bary(self.gp[qx], i);
                            dx[i] = self.dary(self.gp[qx], i);
                            by[i] = self.bary(self.gp[qy], i);
                            dy[i] = self.dary(self.gp[qy], i);
                            bz[i] = self.bary(self.gp[qz], i);
                            dz[i] = self.dary(self.gp[qz], i);
                        }
                        let mut fl = [0.0f64; 3];
                        for j in 0..nloc {
                            let (a, b, c) = (
                                self.qa[j] as usize,
                                self.qb[j] as usize,
                                self.qc[j] as usize,
                            );
                            let g0 = dx[a] * by[b] * bz[c];
                            let g1 = bx[a] * dy[b] * bz[c];
                            let g2 = bx[a] * by[b] * dz[c];
                            let pg0 = j00 * g0 + j01 * g1 + j02 * g2;
                            let pg1 = j10 * g0 + j11 * g1 + j12 * g2;
                            let pg2 = j20 * g0 + j21 * g1 + j22 * g2;
                            fl[0] += pg0 * xe[j];
                            fl[1] += pg1 * xe[j];
                            fl[2] += pg2 * xe[j];
                        }
                        for i in 0..nloc {
                            let (a, b, c) = (
                                self.qa[i] as usize,
                                self.qb[i] as usize,
                                self.qc[i] as usize,
                            );
                            let g0 = dx[a] * by[b] * bz[c];
                            let g1 = bx[a] * dy[b] * bz[c];
                            let g2 = bx[a] * by[b] * dz[c];
                            let pg0 = j00 * g0 + j01 * g1 + j02 * g2;
                            let pg1 = j10 * g0 + j11 * g1 + j12 * g2;
                            let pg2 = j20 * g0 + j21 * g1 + j22 * g2;
                            ye[i] += sc * (pg0 * fl[0] + pg1 * fl[1] + pg2 * fl[2]);
                        }
                    }
                }
            }
            for i in 0..nloc {
                y[dofs[e * nloc + i] as usize] += ye[i];
            }
        }
    }
}

/// Read the numbers of the first `marker ... )` occurrence of a shader source.
fn f64_array(src: &str, marker: &str) -> Vec<f64> {
    array_text(src, marker)
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("f64 literal"))
        .collect()
}

fn u32_array(src: &str, marker: &str) -> Vec<u32> {
    array_text(src, marker)
        .split(',')
        .map(|s| s.trim().parse::<u32>().expect("u32 literal"))
        .collect()
}

fn array_text<'a>(src: &'a str, marker: &str) -> &'a str {
    let start = src
        .find(marker)
        .unwrap_or_else(|| panic!("marker {marker:?} not found"));
    let after = &src[start + marker.len()..];
    let close = after.find(')').expect("array literal close");
    &after[..close]
}

/// One distorted hexahedron: the 8 `HexQ1` corners pushed through a smooth map,
/// so the element's Jacobian varies with the quadrature point.
fn distorted_hex_mesh() -> Mesh<3> {
    let warp = |c: [f64; 3]| -> [f64; 3] {
        let (x, y, z) = (c[0], c[1], c[2]);
        [
            x + 0.15 * y * z,
            y + 0.10 * x * z,
            z + 0.05 * x * y,
        ]
    };
    let mut coords = Vec::new();
    for c in HexQ1.dof_coords() {
        let p = warp([c[0], c[1], c[2]]);
        coords.extend_from_slice(&p);
    }
    let conn: Vec<u32> = (0..8).collect();
    // Boundary quads (only needed for a well-formed mesh; tags are arbitrary).
    let faces: [&[u32]; 6] = [
        &[0, 3, 2, 1],
        &[4, 5, 6, 7],
        &[0, 1, 5, 4],
        &[1, 2, 6, 5],
        &[2, 3, 7, 6],
        &[3, 0, 4, 7],
    ];
    let face_conn: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
    let face_tags: Vec<BoundaryTag> = (1..=6).collect();
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1],
        fem_mesh::ElementType::Hex8,
        face_conn,
        face_tags,
        fem_mesh::ElementType::Quad4,
    )
}

/// CPU PA apply of the distorted hex at degree `p`: `(pd.data, flat dofs, x, y)`.
fn cpu_pa(p: usize) -> (Vec<f64>, Vec<u32>, Vec<f64>, Vec<f64>) {
    let mesh = distorted_hex_mesh();
    let space = H1Space::new(mesh, p as u8);
    let n = space.n_dofs();
    let dofs: Vec<u32> = (0..space.mesh().n_elements() as u32)
        .flat_map(|e| space.element_dofs(e).to_vec())
        .collect();
    let pd = build_hex_qk_pa_data(space.mesh(), &|_| 1.0, p);
    let mut rng: u64 = 42;
    let x: Vec<f64> = (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 11) as f64) / ((1u64 << 53) as f64)
        })
        .collect();
    let mut y = vec![0.0; n];
    let elem_dofs: Vec<Vec<u32>> = (0..space.mesh().n_elements() as u32)
        .map(|e| space.element_dofs(e).to_vec())
        .collect();
    pa_apply_hex_qk(&pd, &elem_dofs, p, &x, &mut y);
    (pd.data, dofs, x, y)
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
/// constants, reproduces the CPU PA apply.  Red before D777 — the shader's
/// `GP`/`GW` are a `[-1,1]` rule while the `pd` buffer it consumes is `[0,1]`.
#[test]
fn shipped_hex_qk_shader_matches_cpu_pa() {
    for (src, p) in SHIPPED {
        let sh = ShippedShader::parse(src, p);
        let (pd, dofs, x, y_cpu) = cpu_pa(p);
        let mut y_shader = vec![0.0; y_cpu.len()];
        sh.apply(&pd, &dofs, &x, &mut y_shader);
        let rel = max_relative_error(&y_shader, &y_cpu);
        assert!(
            rel < 1e-12,
            "p={p}: shipped wgsl kernel vs CPU PA apply: relative error {rel:.3e}\n\
             shader GP = {:?}\nshader GW = {:?}\nshader nodes = {:?}",
            sh.gp,
            sh.gw,
            sh.nodes
        );
    }
}

/// Pin 2 (GPU, self-skipping): the same shaders compiled and run by wgpu on the
/// `f32` path reproduce the CPU PA apply to `f32` accuracy.
#[test]
fn shipped_hex_qk_gpu_shader_matches_cpu_pa() {
    let Some(gpu) = gpu_ctx() else { return };
    for (_, p) in SHIPPED {
        let (pd, dofs, x, y_cpu) = cpu_pa(p);
        let pd32: Vec<f32> = pd.iter().map(|v| *v as f32).collect();
        let x32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
        let mut y32 = vec![0.0f32; x.len()];
        fem_linalg_gpu::pa_apply::gpu_pa_apply_hex_qk(&gpu, p, &pd32, &dofs, &x32, &mut y32);
        let y_gpu: Vec<f64> = y32.iter().map(|v| *v as f64).collect();
        // `run_pa_shader` copies `er` verbatim: the GPU result is **element
        // local** (`res[e*ldof + i]`), while the CPU kernel scatters into the
        // global vector.  Gather the CPU side into the element's own local
        // order (one element ⇒ every global DOF appears exactly once).
        let y_cpu_local: Vec<f64> = dofs.iter().map(|&d| y_cpu[d as usize]).collect();
        let y_cpu = y_cpu_local;
        let rel = max_relative_error(&y_gpu, &y_cpu);
        if rel >= 1e-4 {
            let scale = y_cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let mut worst: Vec<(usize, f64, f64, f64)> = (0..y_cpu.len())
                .map(|i| (i, y_cpu[i], y_gpu[i], y_gpu[i] - y_cpu[i]))
                .collect();
            worst.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
            for (i, c, g, d) in worst.iter().take(8) {
                eprintln!("  dof {i}: cpu {c:.12e} gpu {g:.12e} diff {d:.3e}");
            }
            eprintln!("  scale {scale:.6e}  n={}", y_cpu.len());
        }
        assert!(
            rel < 1e-4,
            "p={p}: GPU f32 PA vs CPU PA: relative error {rel:.3e}"
        );
        eprintln!("D777 GPU f32 p={p}: relative error vs CPU PA = {rel:.3e}");
    }
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
