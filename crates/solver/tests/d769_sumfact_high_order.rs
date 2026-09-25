//! D769 — `SumFactDiffusionOp` (and `PADiffusionOp`) hard-coded their mat-vec
//! scratch buffers to the p <= 4 sizes, so every order above 4 panicked with an
//! index-out-of-bounds on the first `mult_raw` call.
//!
//! The old sizes and the reason they broke:
//!
//! | scratch | declared | needed |
//! |---|---|---|
//! | `tp_x`/`tp_y` | `[f64; 25]` (p1² at p1 = 5) | `(p+1)²` — 36 at p = 5, 49 at p = 6 |
//! | `s_b`/`s_g` | `[[f64; 8]; 8]` | `q1d x (p+1)`; `q1d = (2p+1+2)/2 = p+1`, so it caps at p = 7 |
//!
//! so the very first order past the documented `max p=4` comment (`p >= 5`,
//! `(p+1)² = 36 > 25`) hit `tp_x[tp_idx]` and panicked.  The scratch buffers are
//! now sized from `p`/`q1d` at call time.
//!
//! Acceptance (the same one D729 pinned for p <= 4, extended upwards): the
//! sum-factorization operator reproduces the assembled CSR diffusion operator
//! to `< 1e-12` relative on skewed, non-parallelogram cells with `qo = 2p+1`,
//! and `mult_constrained` applies the `DIAG_ONE` essential-DOF convention.

use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_solver::SumFactDiffusionOp;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// `nx × nx` grid of quads on `[0,1]²`; when `distort` is set the interior
/// vertices are perturbed so the cells are not parallelograms (the D729
/// geometry case that exposes frame/point mismatches).
fn quad_mesh(nx: usize, distort: bool) -> Mesh<2> {
    let n = nx + 1;
    let h = 1.0 / nx as f64;
    let mut vertices = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            let mut x = i as f64 * h;
            let mut y = j as f64 * h;
            if distort && i > 0 && i < nx && j > 0 && j < nx {
                let sx = ((3 * i + 5 * j) % 7) as f64 - 3.0;
                let sy = ((7 * i + 2 * j) % 5) as f64 - 2.0;
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

/// Builds the CSR matrix, the `SumFactDiffusionOp` and the two scratch-free
/// checks for one order on a `nx × nx` distorted grid.
fn check_order(order: u8, nx: usize) {
    let mesh = quad_mesh(nx, true);
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

    // `p = 4` built the old buffers exactly; anything above overflowed them.
    assert_eq!(sf.q1d, (2 * order as usize + 3) / 2, "rule factor");

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
    let dev = num / den.max(1e-300);
    // Captured by default; `--nocapture` reports the measured round-off.
    println!("SumFact order {order} ({nx}x{nx} skewed): rel dev vs CSR = {dev:.3e}");
    assert!(
        dev < 1e-12,
        "SumFactDiffusionOp order {order} ({nx}x{nx} skewed): relative deviation \
         from the CSR operator = {dev:.3e}"
    );

    // `mult_constrained` must zero the essential column (input side) and impose
    // MFEM's `DIAG_ONE` row convention (y[d] = x[d]).
    let bc: Vec<u32> = (0..=order as u32).collect();
    let mut y_con = vec![0.0; n];
    sf.mult_constrained(&x, &mut y_con, &bc);
    // Row convention on the constrained rows.
    for &d in &bc {
        assert_eq!(y_con[d as usize], x[d as usize], "DIAG_ONE row at dof {d}");
    }
    // Column convention: the unconstrained entries must equal the raw mat-vec
    // applied to the BC-zeroed input.
    let mut x0 = x.clone();
    for &d in &bc {
        x0[d as usize] = 0.0;
    }
    let mut y_raw = vec![0.0; n];
    sf.mult_raw(&x0, &mut y_raw);
    for i in 0..n {
        if bc.contains(&(i as u32)) {
            continue;
        }
        assert!(
            (y_con[i] - y_raw[i]).abs() <= 1e-14 * y_raw[i].abs().max(1.0),
            "constrained mat-vec differs from the BC-zeroed raw one at dof {i}"
        );
    }
}

/// p = 5 and p = 6: the two orders the round-71 registration measured as
/// panics (scratch `tp_x`/`tp_y` were `[f64; 25]`).
#[test]
fn d769_sumfact_p5_p6_matches_csr() {
    for order in [5u8, 6] {
        check_order(order, 3);
    }
}

/// p = 4 (the last order that used to fit) must not regress.
#[test]
fn d769_sumfact_p4_no_regression() {
    check_order(4, 3);
}

/// `PADiffusionOp`'s gather buffer was `[f64; 64]` — enough for `(p+1)²` only
/// up to p = 7.  Order 8 must not panic either (its scratch is now sized from
/// the local DOF count).
#[test]
fn d769_pa_diffusion_p8_does_not_panic() {
    use fem_solver::PADiffusionOp;
    let order = 8u8;
    let nx = 2usize;
    let mesh = quad_mesh(nx, true);
    let space = H1Space::new(mesh.clone(), order);
    let p = order as usize;
    let ldofs = (p + 1) * (p + 1);
    let qo = 2 * order + 1;
    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
    let n = space.n_dofs();
    let mut elem_dofs: Vec<u32> = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        elem_dofs.extend_from_slice(space.element_dofs(e));
    }
    let pa = PADiffusionOp::build(&mesh, n, order, qo, 1.0, |e| {
        let start = e as usize * ldofs;
        elem_dofs[start..start + ldofs].to_vec()
    });
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
    let mut y_pa = vec![0.0; n];
    let mut y_csr = vec![0.0; n];
    pa.mult_raw(&x, &mut y_pa);
    mat.spmv(&x, &mut y_csr);
    let num = y_pa
        .iter()
        .zip(y_csr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let den = y_csr.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let dev = num / den.max(1e-300);
    println!("PADiffusionOp order {order} ({nx}x{nx} skewed): rel dev vs CSR = {dev:.3e}");
    assert!(
        dev < 1e-12,
        "PADiffusionOp order {order}: relative deviation from the CSR operator = {dev:.3e}"
    );
}
