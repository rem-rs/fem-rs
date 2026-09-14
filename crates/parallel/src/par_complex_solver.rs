//! Parallel complex iterative solvers for the distributed DPG systems.
//!
//! The C++ DPG miniapps solve the assembled complex normal system
//! `A = Bᴴ G⁻¹ B` (Hermitian positive definite) with MFEM's `CGSolver` on the
//! `ComplexOperator` and a `BlockDiagonalPreconditioner` of per-block
//! `HypreBoomerAMG` / `HypreAMS` solvers wrapped in `ComplexPreconditioner`
//! (`miniapps/dpg/pacoustics.cpp`, `util/preconditioners.hpp`).  fem-rs has no
//! Hypre, so — like the real [`crate::par_solver`] path — the diagonal blocks
//! are solved with our own symmetric Gauss–Seidel sweeps, here in complex
//! arithmetic on the split re/im CSR blocks (MFEM applies the *real* block
//! solvers to the real and imaginary parts separately, ignoring the
//! imaginary parts of the diagonal blocks; applying the exact complex blocks
//! is strictly stronger but changes the PCG path, so the iteration counts are
//! not expected to match the C++ run — the round-33 `pdiffusion` precedent).

use fem_linalg::complex_csr::{ComplexCoo, ComplexCsr};
use fem_solver::{SolveResult, SolverConfig, SolverError};

use crate::par_complex_csr::ParComplexCsrMatrix;
use crate::par_vector::ParComplexVector;

/// Block-diagonal preconditioner with one complex symmetric Gauss–Seidel
/// sweep pair per diagonal block (`M = (D+L) D⁻¹ (D+U)` per block, symmetric
/// hence PCG-compatible), built from the owned diagonal block of a
/// [`ParComplexCsrMatrix`] plus the owned block offsets.
///
/// This is the complex analogue of
/// [`fem_assembly::dpg_weakform::DpgBlockGs`] used by `pdiffusion`.
pub struct ComplexBlockDiagGs {
    /// Diagonal blocks (owned×owned, complex).
    pub blocks: Vec<ComplexCsr>,
    /// Block offsets inside the owned segment (`len = nblocks + 1`).
    pub offsets: Vec<usize>,
}

impl ComplexBlockDiagGs {
    /// Extract the per-block matrices from the owned diagonal block.
    pub fn from_diag_block(diag: &ComplexCsr, offsets: &[usize]) -> Self {
        let nb = offsets.len() - 1;
        let blocks = (0..nb)
            .map(|b| {
                let (r0, r1) = (offsets[b], offsets[b + 1]);
                let mut coo = ComplexCoo::new(r1 - r0, r1 - r0);
                for i in r0..r1 {
                    for p in diag.row_ptr[i]..diag.row_ptr[i + 1] {
                        let c = diag.col_idx[p] as usize;
                        if (r0..r1).contains(&c) {
                            coo.add(
                                i - r0,
                                c - r0,
                                diag.re_vals[p],
                                diag.im_vals[p],
                            );
                        }
                    }
                }
                coo.into_complex_csr()
            })
            .collect();
        ComplexBlockDiagGs { blocks, offsets: offsets.to_vec() }
    }

    /// Apply: `z ← M⁻¹ r` (complex, split re/im; owned segment only).
    pub fn apply(&self, r_re: &[f64], r_im: &[f64], z_re: &mut [f64], z_im: &mut [f64]) {
        for b in 0..self.blocks.len() {
            let (r0, r1) = (self.offsets[b], self.offsets[b + 1]);
            let n = r1 - r0;
            let blk = &self.blocks[b];
            let mut zbr = vec![0.0_f64; n];
            let mut zbi = vec![0.0_f64; n];
            // forward: (D + L) t = r   (complex arithmetic)
            for i in 0..n {
                let mut sr = r_re[r0 + i];
                let mut si = r_im[r0 + i];
                let mut dr = 1.0;
                let mut di = 0.0;
                for p in blk.row_ptr[i]..blk.row_ptr[i + 1] {
                    let c = blk.col_idx[p] as usize;
                    let ar = blk.re_vals[p];
                    let ai = blk.im_vals[p];
                    if c < i {
                        // s -= a * z_c
                        sr -= ar * zbr[c] - ai * zbi[c];
                        si -= ar * zbi[c] + ai * zbr[c];
                    } else if c == i {
                        dr = ar;
                        di = ai;
                    }
                }
                // z_i = s / d
                let den = dr * dr + di * di;
                if den > 1e-300 {
                    zbr[i] = (sr * dr + si * di) / den;
                    zbi[i] = (si * dr - sr * di) / den;
                }
            }
            // t ← D t
            for i in 0..n {
                for p in blk.row_ptr[i]..blk.row_ptr[i + 1] {
                    if blk.col_idx[p] as usize == i {
                        let dr = blk.re_vals[p];
                        let di = blk.im_vals[p];
                        let (zr, zi) = (zbr[i], zbi[i]);
                        zbr[i] = dr * zr - di * zi;
                        zbi[i] = dr * zi + di * zr;
                        break;
                    }
                }
            }
            // backward: (D + U) z = t
            for i in (0..n).rev() {
                let mut sr = zbr[i];
                let mut si = zbi[i];
                let mut dr = 1.0;
                let mut di = 0.0;
                for p in blk.row_ptr[i]..blk.row_ptr[i + 1] {
                    let c = blk.col_idx[p] as usize;
                    let ar = blk.re_vals[p];
                    let ai = blk.im_vals[p];
                    if c > i {
                        sr -= ar * zbr[c] - ai * zbi[c];
                        si -= ar * zbi[c] + ai * zbr[c];
                    } else if c == i {
                        dr = ar;
                        di = ai;
                    }
                }
                let den = dr * dr + di * di;
                if den > 1e-300 {
                    zbr[i] = (sr * dr + si * di) / den;
                    zbi[i] = (si * dr - sr * di) / den;
                } else {
                    zbr[i] = 0.0;
                    zbi[i] = 0.0;
                }
            }
            z_re[r0..r1].copy_from_slice(&zbr);
            z_im[r0..r1].copy_from_slice(&zbi);
        }
    }
}

/// Parallel complex Hermitian preconditioned Conjugate Gradient
/// (MFEM `CGSolver` on the `ComplexOperator` of `pacoustics.cpp`).
///
/// `A` must be Hermitian positive definite (the DPG normal system
/// `A = Bᴴ G⁻¹ B` after essential elimination).  The preconditioner closure
/// maps the *owned* segments of the residual (`r_re`, `r_im`) to the owned
/// segments of `z` (ghost values are left untouched), exactly like
/// [`crate::par_solver::par_solve_pcg_precond`].
pub fn par_solve_complex_pcg<F>(
    a: &ParComplexCsrMatrix,
    b: &ParComplexVector,
    x: &mut ParComplexVector,
    precond: &F,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &[f64], &mut [f64], &mut [f64]),
{
    let n = a.n_owned();

    // r = b − A x
    let mut r = b.clone_complex();
    let mut ax = ParComplexVector::zeros_like(&b.re);
    a.spmv(x, &mut ax);
    for i in 0..n {
        r.re.as_slice_mut()[i] -= ax.re.as_slice()[i];
        r.im.as_slice_mut()[i] -= ax.im.as_slice()[i];
    }

    // z = M⁻¹ r
    let mut z = ParComplexVector::zeros_like(&b.re);
    precond(
        &r.re.as_slice()[..n],
        &r.im.as_slice()[..n],
        &mut z.re.as_slice_mut()[..n],
        &mut z.im.as_slice_mut()[..n],
    );

    let mut p = z.clone_complex();
    let mut rz = r.global_dot_complex(&z);
    let rz_abs = (rz.0 * rz.0 + rz.1 * rz.1).sqrt();
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParComplexVector::zeros_like(&b.re);

    for iter in 0..cfg.max_iter {
        // ap = A p
        a.spmv(&mut p, &mut ap);
        let pap = p.global_dot_complex(&ap);
        let pap_abs = (pap.0 * pap.0 + pap.1 * pap.1).sqrt();
        if pap_abs < 1e-300 || rz_abs < 1e-300 {
            break;
        }
        // α = (r, z) / (p, Ap)  — complex division
        let den = pap.0 * pap.0 + pap.1 * pap.1;
        let al_re = (rz.0 * pap.0 + rz.1 * pap.1) / den;
        let al_im = (rz.1 * pap.0 - rz.0 * pap.1) / den;

        // x += α p,  r -= α ap
        x.zaxpy(al_re, al_im, &p);
        r.zaxpy(-al_re, -al_im, &ap);

        let rr = r.global_norm_squared();
        let res_norm = rr.sqrt() / b_norm;

        if cfg.verbose && x.re.comm().is_root() {
            log::info!("par_complex_pcg iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < cfg.rtol || rr.sqrt() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        // z = M⁻¹ r
        precond(
            &r.re.as_slice()[..n],
            &r.im.as_slice()[..n],
            &mut z.re.as_slice_mut()[..n],
            &mut z.im.as_slice_mut()[..n],
        );

        let rz_new = r.global_dot_complex(&z);
        let den = rz.0 * rz.0 + rz.1 * rz.1;
        // β = (r_new, z_new) / (r, z)
        let be_re = (rz_new.0 * rz.0 + rz_new.1 * rz.1) / den;
        let be_im = (rz_new.1 * rz.0 - rz_new.0 * rz.1) / den;
        // p = z + β p  (elementwise, no aliasing)
        let len = p.re.data.len();
        for i in 0..len {
            let (pr, pi) = (p.re.data[i], p.im.data[i]);
            p.re.data[i] = be_re * pr - be_im * pi + z.re.data[i];
            p.im.data[i] = be_im * pr + be_re * pi + z.im.data[i];
        }
        rz = rz_new;
    }

    let final_res = r.global_norm_squared().sqrt() / b_norm;
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: final_res })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::Launcher;
    use crate::par_partition::partition_mesh_identity;
    use crate::par_vector::ParVector;
    use crate::WorkerConfig;
    use fem_linalg::complex_csr::ComplexCoo;
    use fem_mesh::Mesh;
    use std::sync::{Arc, Mutex};

    /// A 2×2 complex Hermitian system `[[2, 1+i], [1-i, 3]]` distributed over
    /// the ranks of a trivial 1-element partition: PCG must recover the exact
    /// solution `[1+i, −1+2i]` (x = A⁻¹ b with b = A x).
    #[test]
    fn complex_pcg_hermitian_tiny_system() {
        let mesh = Arc::new(Mesh::<2>::unit_square_quad(1));
        let out = Arc::new(Mutex::new(None::<String>));
        let out2 = Arc::clone(&out);
        let ma = Arc::clone(&mesh);
        ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm| {
            let _pm = partition_mesh_identity(&ma, &comm);
            let mut coo = ComplexCoo::new(2, 2);
            coo.add(0, 0, 2.0, 0.0);
            coo.add(0, 1, 1.0, 1.0);
            coo.add(1, 0, 1.0, -1.0);
            coo.add(1, 1, 3.0, 0.0);
            let diag = coo.into_complex_csr();
            let offd = ComplexCoo::new(2, 0).into_complex_csr();
            let exchange = Arc::new(crate::ghost::GhostExchange::from_trivial());
            let a = ParComplexCsrMatrix::new(diag, offd, 2, 0, exchange, comm.clone());
            // b = A x with x = [1+i, -1+2i]:
            //   b0 = 2(1+i) + (1+i)(-1+2i) = 2+2i + (-3+i) = -1+3i
            //   b1 = (1-i)(1+i) + 3(-1+2i) = 2 + (-3+6i) = -1+6i
            let vr = ParVector::from_local_raw(
                vec![-1.0, -1.0],
                2,
                a.ghost_exchange_handle(),
                comm.clone(),
            );
            let vi = ParVector::from_local_raw(
                vec![3.0, 6.0],
                2,
                a.ghost_exchange_handle(),
                comm.clone(),
            );
            let x = ParComplexVector { re: vr, im: vi };
            let x0 = ParComplexVector { re: x.re.clone_vec(), im: x.im.clone_vec() };
            let mut xs = ParComplexVector::zeros_like(&x0.re);
            let id = |r: &[f64], ri: &[f64], z: &mut [f64], zi: &mut [f64]| {
                z.copy_from_slice(r);
                zi.copy_from_slice(ri);
            };
            let cfg = SolverConfig { rtol: 1e-13, max_iter: 50, ..SolverConfig::default() };
            let res = par_solve_complex_pcg(&a, &x0, &mut xs, &id, &cfg).expect("pcg");
            assert!(res.converged, "PCG must converge");
            assert!(
                (xs.re.as_slice()[0] - 1.0).abs() < 1e-10
                    && (xs.im.as_slice()[0] - 1.0).abs() < 1e-10
                    && (xs.re.as_slice()[1] + 1.0).abs() < 1e-10
                    && (xs.im.as_slice()[1] - 2.0).abs() < 1e-10,
                "wrong solution re={:?} im={:?}",
                xs.re.as_slice(),
                xs.im.as_slice()
            );
            *out2.lock().unwrap() = Some(format!("iters={}", res.iterations));
        });
        assert!(out.lock().unwrap().is_some(), "rank 0 must report");
    }

    /// The block symmetric Gauss–Seidel preconditioner must solve a
    /// block-diagonal system exactly in the absence of off-block couplings.
    #[test]
    fn block_diag_gs_solves_block_diagonal() {
        let mut coo = ComplexCoo::new(2, 2);
        coo.add(0, 0, 2.0, 1.0);
        coo.add(1, 1, 4.0, 0.0);
        let blk = coo.into_complex_csr();
        let gs = ComplexBlockDiagGs { blocks: vec![blk], offsets: vec![0, 2] };
        let zr = &mut [0.0; 2];
        let zi = &mut [0.0; 2];
        gs.apply(&[4.0, -8.0], &[2.0, 0.0], zr, zi);
        // x0 = (4+2i)/(2+i) = 2, x1 = -2
        assert!((zr[0] - 2.0).abs() < 1e-13 && zi[0].abs() < 1e-13);
        assert!((zr[1] + 2.0).abs() < 1e-13 && zi[1].abs() < 1e-13);
    }
}
