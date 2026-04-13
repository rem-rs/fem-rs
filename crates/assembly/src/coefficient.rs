//! Coefficient types for spatially-varying, piecewise, and composed material
//! properties.
//!
//! The coefficient system provides a unified abstraction for material parameters
//! used by bilinear and linear form integrators.  Every integrator that
//! previously accepted a plain `f64` coefficient is now generic over
//! [`ScalarCoeff`], with `f64` as the default type parameter for full backwards
//! compatibility.
//!
//! # Hierarchy
//!
//! | Trait | Returns | Typical use |
//! |---|---|---|
//! | [`ScalarCoeff`] | `f64` | Conductivity κ, density ρ, reaction α |
//! | [`VectorCoeff`] | `[f64; dim]` | Body force, convection velocity |
//! | [`MatrixCoeff`] | `[f64; dim×dim]` | Anisotropic diffusion tensor |
//!
//! # Built-in scalar coefficients
//!
//! | Type | Description |
//! |---|---|
//! | `f64` | Constant coefficient (zero-cost, `is_constant = true`) |
//! | [`FnCoeff`] | Closure `Fn(&[f64]) -> f64` for spatially-varying coefficients |
//! | [`CtxFnCoeff`] | Closure `Fn(&CoeffCtx) -> f64` for tag-aware coefficients |
//! | [`PWConstCoeff`] | Piecewise constant per element tag (material region) |
//! | [`PWCoeff`] | Piecewise sub-coefficients per element tag |
//! | [`GridFunctionCoeff`] | Evaluates a DOF vector u_h(x) = Σ uᵢ φᵢ(x) |
//!
//! # Composition
//!
//! | Type | Description |
//! |---|---|
//! | [`ProductCoeff`] | `a(x) · b(x)` |
//! | [`SumCoeff`] | `a(x) + b(x)` |
//! | [`TransformedCoeff`] | `f(c(x))` |
//! | [`InnerProductCoeff`] | `a(x) · b(x)` (vector dot product → scalar) |
//!
//! # Examples
//!
//! ```rust,ignore
//! use fem_assembly::{standard::DiffusionIntegrator, coefficient::*};
//!
//! // Constant (unchanged from before):
//! let integ = DiffusionIntegrator { kappa: 1.0 };
//!
//! // Spatially varying:
//! let integ = DiffusionIntegrator { kappa: FnCoeff(|x: &[f64]| 1.0 + x[0].powi(2)) };
//!
//! // Piecewise constant (two-material):
//! let integ = DiffusionIntegrator { kappa: PWConstCoeff::new([(1, 1.0), (2, 100.0)]) };
//! ```

use std::collections::HashMap;

use fem_core::types::ElemId;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Evaluation context
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Evaluation context passed to coefficients at each quadrature point.
///
/// Contains the physical location and element identity — everything a
/// material property needs to determine its value.  Intentionally lean:
/// coefficients never see basis functions or integration weights.
#[derive(Debug, Clone)]
pub struct CoeffCtx<'a> {
    /// Physical coordinates of the quadrature point (length `dim`).
    pub x: &'a [f64],
    /// Element index (for element-level lookups).
    pub elem_id: ElemId,
    /// Element material / region tag (from mesh physical groups).
    pub elem_tag: i32,
    /// Spatial dimension (2 or 3).
    pub dim: usize,
    /// Basis function values at this QP (length `n_dofs`), if available.
    ///
    /// Populated by the assembler.  Required by [`GridFunctionCoeff`].
    pub phi: Option<&'a [f64]>,
    /// Global DOF indices for this element, if available.
    ///
    /// Populated by the assembler.  Required by [`GridFunctionCoeff`].
    pub elem_dofs: Option<&'a [u32]>,
}

impl<'a> CoeffCtx<'a> {
    /// Build a `CoeffCtx` from assembler-level quadrature-point data.
    ///
    /// This is the canonical way integrators construct a context.
    #[inline]
    pub fn from_qp(
        x_phys: &'a [f64],
        dim: usize,
        elem_id: ElemId,
        elem_tag: i32,
        phi: Option<&'a [f64]>,
        elem_dofs: Option<&'a [u32]>,
    ) -> Self {
        CoeffCtx { x: x_phys, elem_id, elem_tag, dim, phi, elem_dofs }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Scalar coefficient trait
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A scalar-valued coefficient evaluated at quadrature points.
///
/// This is the fundamental building block.  All bilinear integrators that
/// previously took `f64` are now generic over `C: ScalarCoeff` with `f64`
/// as the default, preserving full backwards compatibility.
///
/// # Performance
///
/// For `f64`, the compiler monomorphizes and inlines `eval()` to a simple
/// load — identical to the previous code with no trait overhead.
pub trait ScalarCoeff: Send + Sync {
    /// Evaluate the coefficient at the given context.
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64;

    /// Whether this coefficient is spatially constant.
    ///
    /// When `true`, the assembler *may* hoist the evaluation outside the
    /// quadrature-point loop.  Default: `false`.
    fn is_constant(&self) -> bool { false }
}

// ── f64 is a constant scalar coefficient ────────────────────────────────────

impl ScalarCoeff for f64 {
    #[inline(always)]
    fn eval(&self, _ctx: &CoeffCtx<'_>) -> f64 { *self }
    #[inline(always)]
    fn is_constant(&self) -> bool { true }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Vector coefficient trait
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A vector-valued coefficient: returns `dim` components.
pub trait VectorCoeff: Send + Sync {
    /// Evaluate into `out` (length ≥ `ctx.dim`).  Overwrites contents.
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Matrix coefficient trait
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A matrix-valued coefficient: returns a `dim × dim` matrix in row-major order.
pub trait MatrixCoeff: Send + Sync {
    /// Evaluate into `out` (length ≥ `ctx.dim * ctx.dim`, row-major).
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FnCoeff — closure-based scalar coefficient
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Wraps a closure `Fn(&[f64]) -> f64` as a spatially-varying scalar coefficient.
///
/// # Example
/// ```rust,ignore
/// let kappa = FnCoeff(|x: &[f64]| 1.0 + x[0].powi(2));
/// ```
pub struct FnCoeff<F>(pub F);

impl<F: Fn(&[f64]) -> f64 + Send + Sync> ScalarCoeff for FnCoeff<F> {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 { (self.0)(ctx.x) }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CtxFnCoeff — context-aware closure (access to element tag, etc.)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Wraps a closure `Fn(&CoeffCtx) -> f64` as a scalar coefficient.
///
/// Use this when you need access to element tags, element ID, or basis
/// function values — not just the physical position.
///
/// # Example
/// ```rust,ignore
/// let kappa = CtxFnCoeff(|ctx: &CoeffCtx| {
///     if ctx.elem_tag == 1 { 1.0 } else { 100.0 }
/// });
/// ```
pub struct CtxFnCoeff<F>(pub F);

impl<F: Fn(&CoeffCtx<'_>) -> f64 + Send + Sync> ScalarCoeff for CtxFnCoeff<F> {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 { (self.0)(ctx) }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PmlCoeff — baseline PML damping profile
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Baseline scalar PML damping profile on an axis-aligned box.
///
/// Returns `sigma(x) >= 0` with polynomial ramp in the outer layer of width
/// `thickness` near each box boundary.
#[derive(Debug, Clone)]
pub struct PmlCoeff {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub thickness: f64,
    pub sigma_max: f64,
    pub power: f64,
}

impl PmlCoeff {
    pub fn new(min: Vec<f64>, max: Vec<f64>, thickness: f64, sigma_max: f64) -> Self {
        PmlCoeff {
            min,
            max,
            thickness,
            sigma_max,
            power: 2.0,
        }
    }

    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }
}

impl ScalarCoeff for PmlCoeff {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let d = ctx.dim;
        let mut sigma = 0.0;
        for k in 0..d {
            let x = ctx.x[k];
            let lo = self.min[k];
            let hi = self.max[k];
            let t = self.thickness.max(1e-14);

            let s = if x < lo + t {
                ((lo + t - x) / t).clamp(0.0, 1.0)
            } else if x > hi - t {
                ((x - (hi - t)) / t).clamp(0.0, 1.0)
            } else {
                0.0
            };

            sigma += self.sigma_max * s.powf(self.power);
        }
        sigma
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PWConstCoeff — piecewise constant per element tag
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Piecewise constant coefficient: one value per element tag (material region).
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::coefficient::PWConstCoeff;
/// // Steel (tag 1) has kappa=50, copper (tag 2) has kappa=400
/// let kappa = PWConstCoeff::new([(1, 50.0), (2, 400.0)]);
/// ```
pub struct PWConstCoeff {
    values: HashMap<i32, f64>,
    default: f64,
}

impl PWConstCoeff {
    /// Create from an iterator of `(tag, value)` pairs.
    pub fn new(entries: impl IntoIterator<Item = (i32, f64)>) -> Self {
        PWConstCoeff {
            values: entries.into_iter().collect(),
            default: 0.0,
        }
    }

    /// Set the default value for unmatched tags (default: 0.0).
    pub fn with_default(mut self, default: f64) -> Self {
        self.default = default;
        self
    }
}

impl ScalarCoeff for PWConstCoeff {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        *self.values.get(&ctx.elem_tag).unwrap_or(&self.default)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PWCoeff — piecewise sub-coefficients per element tag
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Piecewise coefficient: different sub-coefficients per element tag.
///
/// Each tag maps to a `Box<dyn ScalarCoeff>`, allowing different spatial
/// functions in different material regions.
///
/// # Example
/// ```rust,ignore
/// let kappa = PWCoeff::new(1.0)
///     .add_region(1, FnCoeff(|x: &[f64]| 1.0 + x[0]))
///     .add_region(2, 100.0_f64);
/// ```
pub struct PWCoeff {
    regions: HashMap<i32, Box<dyn ScalarCoeff>>,
    default: Box<dyn ScalarCoeff>,
}

impl PWCoeff {
    /// Create with a default coefficient for unmatched tags.
    pub fn new(default: impl ScalarCoeff + 'static) -> Self {
        PWCoeff {
            regions: HashMap::new(),
            default: Box::new(default),
        }
    }

    /// Add a sub-coefficient for a specific element tag.
    pub fn add_region(mut self, tag: i32, coeff: impl ScalarCoeff + 'static) -> Self {
        self.regions.insert(tag, Box::new(coeff));
        self
    }
}

impl ScalarCoeff for PWCoeff {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.regions
            .get(&ctx.elem_tag)
            .unwrap_or(&self.default)
            .eval(ctx)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GridFunctionCoeff — evaluate a FE solution at quadrature points
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Evaluates a grid function `u_h(x) = Σ uᵢ φᵢ(x)` at quadrature points.
///
/// **Requires** that the grid function lives on the *same* FE space being
/// assembled over, since it reuses the assembler's pre-computed basis values
/// (`phi`) and element DOF indices (`elem_dofs`) from [`CoeffCtx`].
///
/// # Panics
///
/// Panics at evaluation time if `CoeffCtx::phi` or `CoeffCtx::elem_dofs`
/// is `None`.
///
/// # Example
/// ```rust,ignore
/// let u_prev = solve(&K, &rhs);
/// let rho = GridFunctionCoeff::new(u_prev.to_vec());
/// let integ = MassIntegrator { rho };
/// ```
pub struct GridFunctionCoeff {
    dof_values: Vec<f64>,
}

impl GridFunctionCoeff {
    /// Create from a DOF coefficient vector (typically a solved solution).
    pub fn new(dof_values: Vec<f64>) -> Self {
        GridFunctionCoeff { dof_values }
    }
}

impl ScalarCoeff for GridFunctionCoeff {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let phi = ctx.phi.expect(
            "GridFunctionCoeff requires basis values in CoeffCtx; \
             use it only with the standard assembler"
        );
        let dofs = ctx.elem_dofs.expect(
            "GridFunctionCoeff requires element DOFs in CoeffCtx"
        );
        dofs.iter()
            .zip(phi.iter())
            .map(|(&d, &phi_i)| self.dof_values[d as usize] * phi_i)
            .sum()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Composition: ProductCoeff, SumCoeff, TransformedCoeff
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Product of two scalar coefficients: `(a · b)(x)`.
pub struct ProductCoeff<A: ScalarCoeff, B: ScalarCoeff> {
    pub a: A,
    pub b: B,
}

impl<A: ScalarCoeff, B: ScalarCoeff> ScalarCoeff for ProductCoeff<A, B> {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.a.eval(ctx) * self.b.eval(ctx)
    }
    fn is_constant(&self) -> bool {
        self.a.is_constant() && self.b.is_constant()
    }
}

/// Sum of two scalar coefficients: `(a + b)(x)`.
pub struct SumCoeff<A: ScalarCoeff, B: ScalarCoeff> {
    pub a: A,
    pub b: B,
}

impl<A: ScalarCoeff, B: ScalarCoeff> ScalarCoeff for SumCoeff<A, B> {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.a.eval(ctx) + self.b.eval(ctx)
    }
    fn is_constant(&self) -> bool {
        self.a.is_constant() && self.b.is_constant()
    }
}

/// Transformed coefficient: applies `f(c(x))`.
///
/// # Example
/// ```rust,ignore
/// let temp = GridFunctionCoeff::new(temperature.to_vec());
/// let kappa = TransformedCoeff { inner: temp, transform: |t| 50.0 + 0.1 * t };
/// ```
pub struct TransformedCoeff<C: ScalarCoeff, F: Fn(f64) -> f64 + Send + Sync> {
    pub inner: C,
    pub transform: F,
}

impl<C: ScalarCoeff, F: Fn(f64) -> f64 + Send + Sync> ScalarCoeff for TransformedCoeff<C, F> {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        (self.transform)(self.inner.eval(ctx))
    }
}

/// Inner product of two vector coefficients: `a(x) · b(x)` → scalar.
pub struct InnerProductCoeff<A: VectorCoeff, B: VectorCoeff> {
    pub a: A,
    pub b: B,
}

impl<A: VectorCoeff, B: VectorCoeff> ScalarCoeff for InnerProductCoeff<A, B> {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let dim = ctx.dim;
        // Stack-allocate for dim ≤ 3 (common case).
        let mut va = [0.0_f64; 3];
        let mut vb = [0.0_f64; 3];
        self.a.eval(ctx, &mut va[..dim]);
        self.b.eval(ctx, &mut vb[..dim]);
        va[..dim].iter().zip(vb[..dim].iter()).map(|(a, b)| a * b).sum()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Vector coefficient implementations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Constant vector coefficient.
///
/// # Example
/// ```rust,ignore
/// let body_force = ConstantVectorCoeff(vec![0.0, -9.81]);
/// ```
pub struct ConstantVectorCoeff(pub Vec<f64>);

impl VectorCoeff for ConstantVectorCoeff {
    #[inline]
    fn eval(&self, _ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        out[..self.0.len()].copy_from_slice(&self.0);
    }
}

/// Vector coefficient from a closure `Fn(&[f64], &mut [f64])`.
///
/// The closure receives `(x_phys, out)` and must fill `out[0..dim]`.
///
/// # Example
/// ```rust,ignore
/// let wind = FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
///     out[0] = x[1];  // u = y
///     out[1] = -x[0]; // v = -x
/// });
/// ```
pub struct FnVectorCoeff<F>(pub F);

impl<F: Fn(&[f64], &mut [f64]) + Send + Sync> VectorCoeff for FnVectorCoeff<F> {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        (self.0)(ctx.x, out);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Matrix coefficient implementations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Constant matrix coefficient (row-major, `dim × dim`).
///
/// # Example
/// ```rust,ignore
/// // 2×2 anisotropic diffusion tensor
/// let D = ConstantMatrixCoeff(vec![10.0, 0.0, 0.0, 1.0]);
/// ```
pub struct ConstantMatrixCoeff(pub Vec<f64>);

impl MatrixCoeff for ConstantMatrixCoeff {
    #[inline]
    fn eval(&self, _ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        out[..self.0.len()].copy_from_slice(&self.0);
    }
}

/// Matrix coefficient from a closure `Fn(&[f64], &mut [f64])`.
///
/// The closure receives `(x_phys, out)` and must fill `out[0..dim*dim]`
/// in row-major order.
pub struct FnMatrixCoeff<F>(pub F);

impl<F: Fn(&[f64], &mut [f64]) + Send + Sync> MatrixCoeff for FnMatrixCoeff<F> {
    #[inline]
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        (self.0)(ctx.x, out);
    }
}

/// Scalar × identity matrix: `c(x) · I`.
///
/// Converts any scalar coefficient to a diagonal matrix coefficient.
/// Common for isotropic diffusion with spatially-varying conductivity.
pub struct ScalarMatrixCoeff<C: ScalarCoeff>(pub C);

impl<C: ScalarCoeff> MatrixCoeff for ScalarMatrixCoeff<C> {
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        let d = ctx.dim;
        for v in out[..d * d].iter_mut() { *v = 0.0; }
        let val = self.0.eval(ctx);
        for i in 0..d { out[i * d + i] = val; }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Convenience free functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Multiply two coefficients: `c = a * b`.
pub fn product<A: ScalarCoeff, B: ScalarCoeff>(a: A, b: B) -> ProductCoeff<A, B> {
    ProductCoeff { a, b }
}

/// Sum two coefficients: `c = a + b`.
pub fn sum<A: ScalarCoeff, B: ScalarCoeff>(a: A, b: B) -> SumCoeff<A, B> {
    SumCoeff { a, b }
}

/// Transform a coefficient: `c = f(inner)`.
pub fn transform<C: ScalarCoeff, F: Fn(f64) -> f64 + Send + Sync>(
    inner: C,
    f: F,
) -> TransformedCoeff<C, F> {
    TransformedCoeff { inner, transform: f }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(x: &[f64], tag: i32) -> CoeffCtx<'_> {
        CoeffCtx {
            x,
            elem_id: 0,
            elem_tag: tag,
            dim: x.len(),
            phi: None,
            elem_dofs: None,
        }
    }

    #[test]
    fn f64_is_constant_scalar_coeff() {
        let c: f64 = std::f64::consts::PI;
        let ctx = make_ctx(&[1.0, 2.0], 1);
        assert_eq!(c.eval(&ctx), std::f64::consts::PI);
        assert!(c.is_constant());
    }

    #[test]
    fn fn_coeff_evaluates_at_position() {
        let c = FnCoeff(|x: &[f64]| x[0] * x[1]);
        let ctx = make_ctx(&[3.0, 4.0], 1);
        assert_eq!(c.eval(&ctx), 12.0);
        assert!(!c.is_constant());
    }

    #[test]
    fn ctx_fn_coeff_uses_element_tag() {
        let c = CtxFnCoeff(|ctx: &CoeffCtx| {
            if ctx.elem_tag == 1 { 10.0 } else { 20.0 }
        });
        assert_eq!(c.eval(&make_ctx(&[0.0], 1)), 10.0);
        assert_eq!(c.eval(&make_ctx(&[0.0], 2)), 20.0);
    }

    #[test]
    fn pw_const_coeff_by_tag() {
        let c = PWConstCoeff::new([(1, 10.0), (2, 20.0)]);
        assert_eq!(c.eval(&make_ctx(&[0.0], 1)), 10.0);
        assert_eq!(c.eval(&make_ctx(&[0.0], 2)), 20.0);
        // Unmatched tag → default (0.0)
        assert_eq!(c.eval(&make_ctx(&[0.0], 99)), 0.0);
    }

    #[test]
    fn pw_const_coeff_custom_default() {
        let c = PWConstCoeff::new([(1, 10.0)]).with_default(42.0);
        assert_eq!(c.eval(&make_ctx(&[0.0], 99)), 42.0);
    }

    #[test]
    fn pw_coeff_dispatches_subcoefficients() {
        let c = PWCoeff::new(1.0_f64)
            .add_region(1, FnCoeff(|x: &[f64]| x[0] + 100.0))
            .add_region(2, 200.0_f64);

        let ctx1 = make_ctx(&[5.0], 1);
        assert_eq!(c.eval(&ctx1), 105.0); // FnCoeff: 5 + 100

        let ctx2 = make_ctx(&[5.0], 2);
        assert_eq!(c.eval(&ctx2), 200.0); // constant 200

        let ctx3 = make_ctx(&[5.0], 99);
        assert_eq!(c.eval(&ctx3), 1.0);   // default
    }

    #[test]
    fn grid_function_coeff_evaluates() {
        // 3 DOFs, solution u = [1.0, 2.0, 3.0]
        // basis values at this QP: phi = [0.5, 0.3, 0.2]
        // u_h = 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 0.5 + 0.6 + 0.6 = 1.7
        let gf = GridFunctionCoeff::new(vec![1.0, 2.0, 3.0]);
        let phi = [0.5, 0.3, 0.2];
        let dofs: [u32; 3] = [0, 1, 2];
        let ctx = CoeffCtx {
            x: &[0.0, 0.0],
            elem_id: 0,
            elem_tag: 1,
            dim: 2,
            phi: Some(&phi),
            elem_dofs: Some(&dofs),
        };
        assert!((gf.eval(&ctx) - 1.7).abs() < 1e-14);
    }

    #[test]
    fn product_coeff() {
        let c = product(2.0_f64, FnCoeff(|x: &[f64]| x[0]));
        let ctx = make_ctx(&[5.0], 1);
        assert_eq!(c.eval(&ctx), 10.0);
        assert!(!c.is_constant()); // one factor is non-constant
    }

    #[test]
    fn product_of_constants_is_constant() {
        let c = product(2.0_f64, 3.0_f64);
        assert!(c.is_constant());
        assert_eq!(c.eval(&make_ctx(&[0.0], 1)), 6.0);
    }

    #[test]
    fn sum_coeff() {
        let c = sum(1.0_f64, FnCoeff(|x: &[f64]| x[0]));
        let ctx = make_ctx(&[5.0], 1);
        assert_eq!(c.eval(&ctx), 6.0);
    }

    #[test]
    fn transformed_coeff() {
        let c = transform(FnCoeff(|x: &[f64]| x[0]), |v| v * v);
        let ctx = make_ctx(&[3.0], 1);
        assert_eq!(c.eval(&ctx), 9.0);
    }

    #[test]
    fn pml_coeff_zero_inside_domain_core() {
        let c = PmlCoeff::new(vec![0.0, 0.0], vec![1.0, 1.0], 0.2, 5.0);
        let ctx = make_ctx(&[0.5, 0.5], 1);
        assert_eq!(c.eval(&ctx), 0.0);
    }

    #[test]
    fn pml_coeff_positive_in_layer() {
        let c = PmlCoeff::new(vec![0.0, 0.0], vec![1.0, 1.0], 0.2, 5.0);
        let ctx = make_ctx(&[0.95, 0.5], 1);
        assert!(c.eval(&ctx) > 0.0);
    }

    #[test]
    fn inner_product_coeff() {
        let a = ConstantVectorCoeff(vec![1.0, 2.0]);
        let b = ConstantVectorCoeff(vec![3.0, 4.0]);
        let c = InnerProductCoeff { a, b };
        let ctx = make_ctx(&[0.0, 0.0], 1);
        assert_eq!(c.eval(&ctx), 11.0); // 1*3 + 2*4
    }

    #[test]
    fn constant_vector_coeff() {
        let v = ConstantVectorCoeff(vec![1.0, -9.81]);
        let ctx = make_ctx(&[0.0, 0.0], 1);
        let mut out = [0.0; 2];
        v.eval(&ctx, &mut out);
        assert_eq!(out, [1.0, -9.81]);
    }

    #[test]
    fn fn_vector_coeff() {
        let v = FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
            out[0] = x[1];
            out[1] = -x[0];
        });
        let ctx = make_ctx(&[3.0, 4.0], 1);
        let mut out = [0.0; 2];
        v.eval(&ctx, &mut out);
        assert_eq!(out, [4.0, -3.0]);
    }

    #[test]
    fn scalar_matrix_coeff() {
        let m = ScalarMatrixCoeff(2.0_f64);
        let ctx = make_ctx(&[0.0, 0.0], 1);
        let mut out = [0.0; 4];
        m.eval(&ctx, &mut out);
        assert_eq!(out, [2.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn constant_matrix_coeff() {
        let m = ConstantMatrixCoeff(vec![1.0, 2.0, 3.0, 4.0]);
        let ctx = make_ctx(&[0.0, 0.0], 1);
        let mut out = [0.0; 4];
        m.eval(&ctx, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }
}
