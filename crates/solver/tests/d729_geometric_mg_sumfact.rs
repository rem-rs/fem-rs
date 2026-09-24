//! D729 — `SumFactDiffusionOp` must reproduce the assembled (CSR) diffusion
//! operator to round-off.
//!
//! The geometric-MG level's `mat_vec` used to prefer `pa_op` over `sf_op`
//! because the sum-factorization operator deviated "~8.6e-6 from the CSR (a
//! precision bug under investigation)".  The real defect was the geometry
//! Jacobian: `SumFactDiffusionOp::build` paired the `[-1,1]²` Gauss rule with
//! MFEM `BiLinear2DFiniteElement`'s `[0,1]²` derivative formulas, i.e. it
//! evaluated the bilinear map at extrapolated points (`η = -0.577` read as a
//! `[0,1]` coordinate instead of the corresponding `0.211`).  Affine cells have
//! a constant Jacobian and hid the bug; **every non-parallelogram cell** was
//! wrong.  Measured with this file's `sf_vs_csr` on a 4×4 grid of mildly skewed
//! (convex, unperturbed-orientation) quads, before the fix:
//!
//! | order | deviation (relative to max |A·x|) |
//! |---|---|
//! | 1 | 5.15e-2 |
//! | 2 | 1.16e-1 |
//! | 3 | 1.04e-1 |
//! | 4 | 1.13e-1 |
//!
//! and after it 5.4e-16, 1.1e-15, 1.3e-15, 2.2e-15.  (The registered 8.6e-6
//! was the same defect on a much milder mesh.)  The operator is now built
//! entirely on `[0,1]²` (nodes/rule/Jacobian) — the frame MFEM's
//! `GeometricFactors`/`PADiffusionSetup2D` use for `Geometry::SQUARE`
//! (`mesh/mesh.cpp:15445`, `fem/integ/bilininteg_diffusion_kernels.cpp:147`).
//!
//! Acceptance: `max |(sf·x) − (A·x)| / max |A·x| < 1e-12` for orders 1..4 on a
//! mesh with skewed, non-parallelogram cells, with `qo = 2p+1` (ex26's own
//! quadrature order) and a non-constant vector (a constant vector gives
//! `A·1 ≈ 0` and hides sign/index errors).

use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_solver::SumFactDiffusionOp;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// `nx × nx` grid of quads on `[0,1]²`.  When `distort` is set the interior
/// vertices are perturbed, so the cells are not parallelograms and the bilinear
/// Jacobian varies inside each element.
fn quad_mesh(nx: usize, distort: bool) -> Mesh<2> {
    let n = nx + 1;
    let h = 1.0 / nx as f64;
    let mut vertices = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            let mut x = i as f64 * h;
            let mut y = j as f64 * h;
            if distort && i > 0 && i < nx && j > 0 && j < nx {
                // Small deterministic skew, in a bounded range so the cells
                // stay convex and correctly oriented.
                let sx = ((3 * i + 5 * j) % 7) as f64 - 3.0; // -3..=3
                let sy = ((7 * i + 2 * j) % 5) as f64 - 2.0; // -2..=2
                x += 0.03 * h * sx;
                y += 0.03 * h * sy;
            }
            vertices.push(x);
            vertices.push(y);
        }
    }
    let mut conn = Vec::with_capacity(4 * nx * nx);
    for j in 0..nx {
        for i in 0..nx {
            let v = |ii: usize, jj: usize| (jj * n + ii) as u32;
            conn.extend_from_slice(&[v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)]);
        }
    }
    let attrs = vec![1i32; nx * nx];
    let mut bdr_attrs = Vec::new();
    let mut bdr = Vec::new();
    for i in 0..nx {
        bdr.extend_from_slice(&[i as u32, (i + 1) as u32]);
        bdr_attrs.push(1);
    }
    Mesh::<2>::uniform(
        vertices,
        conn,
        attrs,
        ElementType::Quad4,
        bdr,
        bdr_attrs,
        ElementType::Line2,
    )
}

/// `max |(sf·x) − (A·x)| / max |A·x|` for the given order.
fn sf_vs_csr(order: u8, nx: usize, distort: bool) -> f64 {
    let mesh = quad_mesh(nx, distort);
    let space = H1Space::new(mesh.clone(), order);
    let p = order as usize;
    let ldofs = (p + 1) * (p + 1);
    let qo = (2 * order + 1).max(3);
    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
    let n = space.n_dofs();
    let mut elem_dofs: Vec<u32> = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        elem_dofs.extend_from_slice(space.element_dofs(e));
    }
    let sf = SumFactDiffusionOp::build(&mesh, n, order, qo, 1.0, |e| {
        let start = e as usize * ldofs;
        elem_dofs[start..start + ldofs].to_vec()
    });
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
    let mut y_sf = vec![0.0; n];
    let mut y_csr = vec![0.0; n];
    sf.mult_raw(&x, &mut y_sf);
    mat.spmv(&x, &mut y_csr);
    let num = y_sf
        .iter()
        .zip(y_csr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let den = y_csr.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    num / den.max(1e-300)
}

/// Skewed (non-parallelogram) cells — the configuration that exposed D729.
#[test]
fn d729_sumfact_diffusion_matches_csr_on_skewed_quads() {
    for order in [1u8, 2, 3, 4] {
        let dev = sf_vs_csr(order, 4, true);
        assert!(
            dev < 1e-12,
            "SumFactDiffusionOp order {order} (4x4 skewed): relative deviation from the CSR \
             operator = {dev:.3e}"
        );
    }
}

/// Affine cells (a straight grid, constant Jacobian) and a single element —
/// the cases the old code got right, kept as a no-regression pin.
#[test]
fn d729_sumfact_diffusion_matches_csr_on_affine_quads() {
    for (nx, order) in [(1usize, 1u8), (1, 2), (4, 1), (4, 4)] {
        let dev = sf_vs_csr(order, nx, false);
        assert!(
            dev < 1e-12,
            "SumFactDiffusionOp order {order} ({nx}x{nx} undistorted): relative deviation = {dev:.3e}"
        );
    }
}
