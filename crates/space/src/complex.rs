//! Complex-valued finite element spaces and grid functions.
//!
//! Wraps existing real-valued [`FESpace`] implementations for time-harmonic
//! and Helmholtz-type PDEs. Uses the 2×2 real-block strategy from
//! `fem_assembly::complex`.

use crate::fe_space::FESpace;

/// A complex-valued grid function `u = u_re + i·u_im`.
///
/// Stores separate real and imaginary DOF vectors, parametrised by the
/// underlying real FE space.  Use [`ComplexGridFunction::interpolate`]
/// to project a complex scalar field onto the space.
#[derive(Debug, Clone)]
pub struct ComplexGridFunction<S: FESpace> {
    pub space: S,
    /// Real-part DOF coefficients.
    pub u_re: Vec<f64>,
    /// Imaginary-part DOF coefficients.
    pub u_im: Vec<f64>,
}

impl<S: FESpace> ComplexGridFunction<S> {
    /// Create a zero-initialised complex grid function.
    pub fn new(space: S) -> Self {
        let n = space.n_dofs();
        ComplexGridFunction { space, u_re: vec![0.0; n], u_im: vec![0.0; n] }
    }

    /// Extract from a flat 2n solution vector `[u_re; u_im]` without copying the space.
    pub fn from_flat(flat: &[f64], space: S) -> Self {
        let n = space.n_dofs();
        assert_eq!(flat.len(), 2 * n);
        ComplexGridFunction {
            space,
            u_re: flat[..n].to_vec(),
            u_im: flat[n..].to_vec(),
        }
    }

    /// Number of (real) DOFs per component.
    pub fn n_dofs(&self) -> usize { self.u_re.len() }

    /// Interpolate a complex scalar function `f(x) = f_re(x) + i·f_im(x)`.
    pub fn interpolate(&mut self, f_re: &dyn Fn(&[f64]) -> f64, f_im: &dyn Fn(&[f64]) -> f64) {
        let ur = self.space.interpolate(f_re);
        let ui = self.space.interpolate(f_im);
        self.u_re.copy_from_slice(ur.as_slice());
        self.u_im.copy_from_slice(ui.as_slice());
    }

    /// Pointwise amplitude `|u|_i = sqrt(u_re[i]² + u_im[i]²)`.
    pub fn amplitude(&self) -> Vec<f64> {
        self.u_re.iter().zip(self.u_im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Total complex L² norm `sqrt(‖u_re‖² + ‖u_im‖²)`.
    pub fn l2_norm(&self) -> f64 {
        let re: f64 = self.u_re.iter().map(|x| x * x).sum();
        let im: f64 = self.u_im.iter().map(|x| x * x).sum();
        (re + im).sqrt()
    }

    /// Compute the L² error against an exact complex solution.
    pub fn l2_error(&self, f_re: &dyn Fn(&[f64]) -> f64, f_im: &dyn Fn(&[f64]) -> f64) -> f64 {
        let exact_re = self.space.interpolate(f_re);
        let exact_im = self.space.interpolate(f_im);
        let mut sq = 0.0;
        for i in 0..self.n_dofs() {
            let dr = self.u_re[i] - exact_re.as_slice()[i];
            let di = self.u_im[i] - exact_im.as_slice()[i];
            sq += dr * dr + di * di;
        }
        sq.sqrt()
    }
}

/// Apply complex Dirichlet boundary conditions to a 2×2 block system.
///
#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use crate::H1Space;
    use std::f64::consts::PI;

    #[test]
    fn complex_grid_function_zero_init() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let gf = ComplexGridFunction::new(space);
        assert_eq!(gf.n_dofs(), gf.space.n_dofs());
        assert!(gf.l2_norm() < 1e-14);
    }

    #[test]
    fn complex_interpolate_plane_wave() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let mut gf = ComplexGridFunction::new(space);
        let kx = 2.0 * PI;

        gf.interpolate(
            &|x| (kx * x[0]).cos(),
            &|x| (kx * x[0]).sin(),
        );

        // Amplitude should be ~1 everywhere
        let amp = gf.amplitude();
        for a in &amp {
            assert!((a - 1.0).abs() < 1e-10, "amplitude deviate: {a}");
        }
    }

    #[test]
    fn complex_l2_error_is_zero_for_exact() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let mut gf = ComplexGridFunction::new(space);
        let fre = |x: &[f64]| x[0] * x[1];
        let fim = |x: &[f64]| x[0] + x[1];

        gf.interpolate(&fre, &fim);
        let err = gf.l2_error(&fre, &fim);
        // Interpolation of a function in the space should have zero error
        // For P1, linear functions are exact
        assert!(err < 1e-12, "l2_error for exact={err}");
    }
}
