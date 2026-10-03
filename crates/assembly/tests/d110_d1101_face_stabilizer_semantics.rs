//! d110 / D1101 pins for the WG Maxwell **face-stabilizer semantics**
//! (`fem_assembly::wg::assemble_wg_maxwell`'s `add_face_penalty_hcurl`).
//!
//! # The debt and the verdict
//!
//! The doc comment claimed the stabilizer "penalizes tangential jumps" — the
//! discontinuous-gallery form with cross-element blocks `K_lr = −α∫φ_i·ψ_j`.
//! The implementation assembles **two-sided local face mass**: each element
//! sharing a face contributes its own signed block `α∫_F φ_i·φ_j dS` and
//! there are **no cross blocks**.
//!
//! Verdict (D1101): **the documentation was wrong; the code is the right
//! single-space semantics.**  The referee is the WG-Maxwell literature
//! itself: the stabilizer of the WG Maxwell method is *local* — it penalizes
//! the mismatch between the interior trace `v_0` and the independent
//! boundary variable `v_b` on each element's **own** boundary
//! (`s_1(v,w) = Σ_T h_T⁻¹⟨(v_0−v_b)×n,(w_0−w_b)×n⟩ + h_T⁻¹⟨(εv_0−v_b)·n,…⟩`;
//! Chunmei Wang, arXiv:1610.04310; Mu–Wang–Ye–Zhang 2013) — *not* a
//! DG-style inter-element jump.  This kernel is the **single-space
//! reduction** of that form: one (conforming Nédélec) trace per side, no
//! `v_b`, so the local penalty degenerates to the face-trace mass.  And the
//! documented jump form is not merely wrong here — it is **vacuous**: on a
//! conforming H(curl) space the tangential trace is continuous, so the
//! jump penalty matrix is identically zero and the volume curl–curl form
//! (kernel = discrete gradients) would stay singular.  The module's own
//! SPD/no-zero-row tests and every consumer rely on the stabilizer for
//! invertibility.
//!
//! # Pins
//!
//! 1. **tangential-jump nullity** (the evidence): for random global
//!    coefficient vectors the tangential trace of the global Nédélec field
//!    at every interior-face quadrature point agrees between the two
//!    neighboring elements to round-off — the jump the doc described is
//!    *identically zero on this space* (while the full-vector trace is
//!    genuinely discontinuous: the "tangential" qualifier is load-bearing).
//! 2. **the stabilizer removes exactly the volume kernel** (red under the
//!    documented jump form, green today): for a gradient interpolant `e`
//!    the volume-only matrix annihilates it (`‖A(0)·e‖ ≈ 0`, the d1080
//!    annihilation pin) while the stabilized matrix does not
//!    (`‖A(10)·e‖ = O(1)` — the face-mass energy `eᵀ(A(10)−A(0))e > 0`).
//!    Under the jump-form "fix" the stabilizer would vanish, `A(10) ≈ A(0)`,
//!    and this pin would go red.  (The exact two-sided block structure is
//!    already locked entry-wise by `d109_d1080_wg_maxwell_signs.rs` pin 3
//!    and `d810_wg_face_geometry.rs`.)

use fem_assembly::{assemble_wg_maxwell, InteriorFaceList};
use fem_element::nedelec::TriNDk;
use fem_element::quadrature::seg_rule;
use fem_element::VectorReferenceElement;
use fem_mesh::topology::MeshTopology as _;
use fem_mesh::Mesh;
use fem_space::HCurlSpace;

/// A deterministic pseudo-random coefficient vector (xorshift), reproducible
/// across platforms.
fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 2000) as f64 - 1000.0) / 1000.0
        })
        .collect()
}

fn spmv_norm(k: &fem_linalg::CsrMatrix<f64>, x: &[f64]) -> f64 {
    let mut y = vec![0.0_f64; k.nrows];
    k.spmv(x, &mut y);
    y.iter().map(|v| v * v).sum::<f64>().sqrt()
}

// ─── Pin 1: the tangential jump is identically zero on the space ─────────────

#[test]
fn d1101_tangential_jump_of_conforming_nd_is_identically_zero() {
    let name = "unit_square_tri(2)";
    let mesh = Mesh::<2>::unit_square_tri(2);
    let nd = HCurlSpace::new(mesh.clone(), 1);
    assert!(
        (0..mesh.n_elements() as u32)
            .map(|e| nd.element_signs(e))
            .flatten()
            .any(|&s| s < 0.0),
        "{name}: expected reversed H(curl) edges (pin key)"
    );

    let phi = TriNDk::new(1);
    let n_loc = phi.n_dofs();
    let qf = seg_rule(4);
    let dofs: Vec<Vec<usize>> = (0..mesh.n_elements() as u32)
        .map(|e| nd.element_dofs(e).iter().map(|&d| d as usize).collect())
        .collect();
    let signs: Vec<Vec<f64>> = (0..mesh.n_elements() as u32)
        .map(|e| nd.element_signs(e).to_vec())
        .collect();

    // Element corner geometry (straight mesh — exact affine maps).
    let verts: Vec<[[f64; 2]; 3]> = (0..mesh.n_elements() as u32)
        .map(|e| {
            let mut v = [[0.0; 2]; 3];
            for (k, &p) in mesh.element_nodes(e).iter().enumerate() {
                let c = mesh.geom_coords_of(p);
                v[k] = [c[0], c[1]];
            }
            v
        })
        .collect();

    // Coefficient sample of the global field at (dof, side) — the physical
    // Nédélec vector via the covariant Piola J^{-T}·φ̂, through the signed
    // dof table.
    let samples = 4;
    let cs: Vec<Vec<f64>> = (0..samples).map(|k| pseudo_random(nd.n_dofs(), 0x9E3779B97F4A7C15 ^ k)).collect();

    let mut max_jump_t = 0.0_f64;
    let mut max_jump_full = 0.0_f64;
    for f in &InteriorFaceList::build(&mesh).faces {
        // Physical unit tangent of the face chord.
        let a = {
            let c = mesh.geom_coords_of(f.face_nodes[0]);
            [c[0], c[1]]
        };
        let b = {
            let c = mesh.geom_coords_of(f.face_nodes[1]);
            [c[0], c[1]]
        };
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let t = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
        // Left-side field values, read back in matching (QP, sample) order on
        // the right side.
        let mut left_vals: Vec<(f64, f64)> = Vec::new();
        let mut right_ctr = 0_usize;

        for (side, el) in [f.elem_left, f.elem_right].into_iter().enumerate() {
            // The element's local directed edge (k → k+1) through the face
            // (MFEM triangle edges {0,1},{1,2},{2,0}) and its Jacobian.
            let en = mesh.element_nodes(el);
            let (mut k, mut fwd) = (usize::MAX, true);
            for t2 in 0..3 {
                if en[t2] == f.face_nodes[0] && en[(t2 + 1) % 3] == f.face_nodes[1] {
                    k = t2;
                    fwd = true;
                    break;
                }
                if en[t2] == f.face_nodes[1] && en[(t2 + 1) % 3] == f.face_nodes[0] {
                    k = t2;
                    fwd = false;
                    break;
                }
            }
            assert!(k < 3, "face nodes not on element {el}");
            let v = &verts[el as usize];
            let det = (v[1][0] - v[0][0]) * (v[2][1] - v[0][1])
                - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]);
            // J columns: x1−x0, x2−x0.
            let j11 = v[1][0] - v[0][0];
            let j21 = v[1][1] - v[0][1];
            let j12 = v[2][0] - v[0][0];
            let j22 = v[2][1] - v[0][1];
            // Covariant Piola map J^{-T} = adj(J)ᵀ/det (J⁻¹ transposed):
            // J⁻¹ = adj(J)/det, so J^{-T} = [[j22, −j21], [−j12, j11]]/det.
            let inv_det = 1.0 / det;
            let jt = [
                [j22 * inv_det, -j21 * inv_det],
                [-j12 * inv_det, j11 * inv_det],
            ];

            for xi in &qf.points {
                let frac = if fwd { xi[0] } else { 1.0 - xi[0] };
                let ref_v = [[0.0_f64, 0.0_f64], [1.0, 0.0], [0.0, 1.0]];
                let eip = [
                    ref_v[k][0] + frac * (ref_v[(k + 1) % 3][0] - ref_v[k][0]),
                    ref_v[k][1] + frac * (ref_v[(k + 1) % 3][1] - ref_v[k][1]),
                ];
                let mut pb = vec![0.0_f64; n_loc * 2];
                phi.eval_basis_vec(&eip, &mut pb);
                for c in &cs {
                    let mut ux = 0.0_f64;
                    let mut uy = 0.0_f64;
                    for i in 0..n_loc {
                        let s = signs[el as usize].get(i).copied().unwrap_or(1.0);
                        let ci = s * c[dofs[el as usize][i]];
                        // Covariant Piola: (ux, uy) += ci · J^{-T}·φ̂_i.
                        ux += ci * (jt[0][0] * pb[i * 2] + jt[0][1] * pb[i * 2 + 1]);
                        uy += ci * (jt[1][0] * pb[i * 2] + jt[1][1] * pb[i * 2 + 1]);
                    }
                    if side == 0 {
                        left_vals.push((ux, uy));
                    } else {
                        let (lx, ly) = left_vals[right_ctr];
                        right_ctr += 1;
                        max_jump_t = max_jump_t.max(((ux - lx) * t[0] + (uy - ly) * t[1]).abs());
                        max_jump_full = max_jump_full.max(((ux - lx).powi(2) + (uy - ly).powi(2)).sqrt());
                    }
                }
            }
        }
    }
    println!(
        "{name}: max |tangential jump| = {max_jump_t:.3e}, max |full-vector jump| = {max_jump_full:.3e}"
    );
    assert!(
        max_jump_t < 1e-13,
        "{name}: tangential jump = {max_jump_t:.3e} — H(curl) conformity broken \
         (D1101: the documented jump penalty is identically zero on this space)"
    );
    assert!(
        max_jump_full > 1e-2,
        "{name}: full-vector jump = {max_jump_full:.3e} — expected the normal \
         trace to be discontinuous (the 'tangential' qualifier is load-bearing)"
    );
}

// ─── Pin 2: the stabilizer removes exactly the volume kernel ─────────────────

#[test]
fn d1101_stabilizer_removes_the_volume_kernel() {
    let name = "unit_square_tri(3)";
    let mesh = Mesh::<2>::unit_square_tri(3);
    let nd = HCurlSpace::new(mesh.clone(), 1);
    assert!(
        (0..mesh.n_elements() as u32).map(|e| nd.element_signs(e)).flatten().any(|&s| s < 0.0),
        "{name}: expected reversed H(curl) edges (pin key)"
    );

    // The gradient interpolant: discretely curl-free, so A(0)·e ≈ 0 (the
    // d1080 annihilation identity through the signed dof table).
    let e_h = nd
        .interpolate_vector(&|x: &[f64]| vec![2.0 + x[1], -3.0 + x[0]])
        .as_slice()
        .to_vec();
    let e_norm: f64 = e_h.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(e_norm > 0.1, "{name}: vanishing interpolant, pin is vacuous");

    let (a0, _) = assemble_wg_maxwell(&nd, 3, 0.0, &[]);
    let (a10, _) = assemble_wg_maxwell(&nd, 3, 10.0, &[]);
    let n0 = spmv_norm(&a0, &e_h);
    let n10 = spmv_norm(&a10, &e_h);

    // Stabilizer energy of the kernel vector: eᵀ(A(10)−A(0))e.
    let mut se = vec![0.0_f64; a10.nrows];
    a0.spmv(&e_h, &mut se);
    let mut a10e = vec![0.0_f64; a10.nrows];
    a10.spmv(&e_h, &mut a10e);
    let energy: f64 = e_h.iter().zip(a10e.iter().zip(se.iter())).map(|(ei, (a, s))| ei * (a - s)).sum();

    println!("{name}: ‖A(0)e‖ = {n0:.3e}, ‖A(10)e‖ = {n10:.3e}, eᵀSe = {energy:.6e}");
    assert!(n0 < 1e-12, "{name}: volume form lost its annihilation ({n0:.3e})");
    assert!(
        n10 > 1.0,
        "{name}: stabilized matrix does not remove the gradient kernel, \
         ‖A(10)e‖ = {n10:.3e} (D1101: under the documented jump form the \
         stabilizer would be identically zero and this pin red)"
    );
    assert!(energy > 0.0, "{name}: stabilizer energy on the kernel vector must be positive");
}
