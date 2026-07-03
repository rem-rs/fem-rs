//! Complex-valued finite element spaces and grid functions.
//!
//! Wraps existing real-valued [`FESpace`] implementations for time-harmonic
//! and Helmholtz-type PDEs. Uses the 2×2 real-block strategy from
//! `fem_assembly::complex`.

use fem_core::types::DofId;
use fem_linalg::{CsrMatrix, Vector};
use fem_mesh::topology::MeshTopology;
use crate::fe_space::{FESpace, SpaceType};

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
/// For each DOF in `dofs` with prescribed complex value `(val_re[i], val_im[i])`:
/// - Zero the row/column in both real and imaginary blocks
/// - Set diagonal to 1
/// - Set RHS accordingly
pub fn apply_complex_dirichlet(
    k_re: &mut CsrMatrix<f64>,
    k_im: &mut CsrMatrix<f64>,
    rhs_re: &mut [f64],
    rhs_im: &mut [f64],
    dofs: &[u32],
    val_re: &[f64],
    val_im: &[f64],
) {
    assert_eq!(dofs.len(), val_re.len());
    assert_eq!(dofs.len(), val_im.len());

    for (i, &dof) in dofs.iter().enumerate() {
        let d = dof as usize;
        // Real part: zero row, set diagonal=1, set rhs_re
        k_re.apply_dirichlet_row_zeroing(d, val_re[i], rhs_re);
        // Imaginary part: zero row, set diagonal=1, set rhs_im
        k_im.apply_dirichlet_row_zeroing(d, val_im[i], rhs_im);
    }
}

// ─── ComplexSpace ────────────────────────────────────────────────────────────

/// A complex-valued FE space wrapping any real [`FESpace`].
///
/// Delegates all [`FESpace`] methods to the inner real space and provides
/// complex-specific operations: complex interpolation, grid function
/// creation, and boundary DOF queries.
///
/// Works seamlessly with [`ComplexAssembler`](fem_assembly::complex::ComplexAssembler)
/// from the assembly crate.
///
/// # Examples
/// ```rust,ignore
/// use fem_space::complex::ComplexSpace;
/// use fem_space::H1Space;
///
/// let h1 = H1Space::new(mesh, 1);
/// let csp = ComplexSpace::new(h1);
/// let gf = csp.create_grid_function(); // zero-initialised
/// ```
#[derive(Debug, Clone)]
pub struct ComplexSpace<S: FESpace> {
    pub inner: S,
}

impl<S: FESpace> ComplexSpace<S> {
    /// Wrap an existing real FE space.
    pub fn new(inner: S) -> Self {
        ComplexSpace { inner }
    }

    /// Create a zero-initialised complex grid function for this space.
    pub fn create_grid_function(&self) -> ComplexGridFunction<S>
    where S: Clone
    {
        ComplexGridFunction::new(self.inner.clone())
    }

    /// Interpolate a complex scalar field onto this space.
    pub fn interpolate(
        &self,
        f_re: &dyn Fn(&[f64]) -> f64,
        f_im: &dyn Fn(&[f64]) -> f64,
    ) -> ComplexGridFunction<S>
    where S: Clone
    {
        let mut gf = self.create_grid_function();
        gf.interpolate(f_re, f_im);
        gf
    }

    /// Get boundary DOFs for complex Dirichlet conditions.
    ///
    /// Collects all DOFs whose nodes lie on boundary faces.
    /// For P1 elements this is exact; for higher-order or vector elements,
    /// use `constraints::boundary_dofs` / `boundary_dofs_hcurl` / `boundary_dofs_hdiv`.
    pub fn boundary_dofs(&self) -> Vec<u32> {
        use std::collections::HashSet;
        let mesh = self.inner.mesh();
        let mut bdy_nodes: HashSet<u32> = HashSet::new();
        for f in 0..mesh.n_boundary_faces() {
            for &n in mesh.face_nodes(f as fem_core::FaceId) {
                bdy_nodes.insert(n);
            }
        }
        let mut dofs: Vec<u32> = Vec::new();
        for e in 0..mesh.n_elements() as u32 {
            for (local, &global) in self.inner.element_dofs(e).iter().enumerate() {
                let en = mesh.element_nodes(e);
                if local < en.len() && bdy_nodes.contains(&en[local]) {
                    if !dofs.contains(&global) { dofs.push(global); }
                }
            }
        }
        dofs.sort_unstable();
        dofs
    }
}

impl<S: FESpace> FESpace for ComplexSpace<S> {
    type Mesh = S::Mesh;

    fn mesh(&self) -> &Self::Mesh { self.inner.mesh() }
    fn n_dofs(&self) -> usize { self.inner.n_dofs() }
    fn element_dofs(&self, elem: u32) -> &[DofId] { self.inner.element_dofs(elem) }
    fn space_type(&self) -> SpaceType { self.inner.space_type() }
    fn order(&self) -> u8 { self.inner.order() }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        self.inner.interpolate(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use crate::H1Space;
    use std::f64::consts::PI;

    #[test]
    fn complex_grid_function_zero_init() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let gf = ComplexGridFunction::new(space);
        assert_eq!(gf.n_dofs(), gf.space.n_dofs());
        assert!(gf.l2_norm() < 1e-14);
    }

    #[test]
    fn complex_interpolate_plane_wave() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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

    #[test]
    fn complex_space_delegates_to_inner() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let csp = ComplexSpace::new(h1.clone());
        assert_eq!(csp.n_dofs(), h1.n_dofs());
        assert_eq!(csp.order(), h1.order());
        assert_eq!(csp.element_dofs(0), h1.element_dofs(0));
    }

    #[test]
    fn complex_space_create_grid_function() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let csp = ComplexSpace::new(h1);
        let gf = csp.create_grid_function();
        assert_eq!(gf.n_dofs(), csp.n_dofs());
        assert!(gf.l2_norm() < 1e-14);
    }

    #[test]
    fn complex_space_interpolate() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let csp = ComplexSpace::new(h1);
        let gf = csp.interpolate(&|x| x[0], &|x| x[1]);
        let amp = gf.amplitude();
        for (i, a) in amp.iter().enumerate() {
            let coord = csp.mesh().node_coords(i as u32);
            let expected = (coord[0].powi(2) + coord[1].powi(2)).sqrt();
            assert!((a - expected).abs() < 1e-10, "amplitude at node {i}: {a}, expected {expected}");
        }
    }

    #[test]
    fn complex_space_boundary_dofs_non_empty() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let csp = ComplexSpace::new(h1);
        let bdy = csp.boundary_dofs();
        assert!(!bdy.is_empty(), "should have boundary DOFs on unit square");
        // All boundary DOFs should be valid
        for &d in &bdy {
            assert!((d as usize) < csp.n_dofs(), "invalid boundary DOF {d}");
        }
    }
}
