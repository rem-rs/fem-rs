//! Automatic differentiation (forward dual numbers) — 1:1 port of MFEM's
//! native AD infrastructure.
//!
//! Correspondence to MFEM (`miniapps/autodiff/admfem.hpp` native branch +
//! `linalg/dual.hpp`):
//!
//! | MFEM                                        | fem-rs                                   |
//! |---------------------------------------------|------------------------------------------|
//! | `future::dual<real_t, real_t>`              | [`Dual`] (alias [`AdFloat`] = `AD1Type`) |
//! | `AD2Type = future::dual<AD1Type, AD1Type>`  | [`Ad2`]                                  |
//! | `ad::ADFloatType` / `ad::ADVectorType`      | [`AdFloat`] / `Vec<AdFloat>`             |
//! | `QFunctionAutoDiff<F, n, m>`                | [`QFunctionAutoDiff`]                    |
//! | `QVectorFuncAutoDiff<F, m, n, k>`           | [`QVectorFuncAutoDiff`]                  |
//! | `VectorFuncAutoDiff<m, n, k>` (closure)     | [`vector_func_jacobian`]                 |
//!
//! The forward (tangent) mode is the exact analogue of MFEM's native
//! `future::dual` path (`MFEM_USE_CODIPACK` off): one dual seed per state
//! component, first derivatives in `gradient`, second derivatives through
//! *nested* duals (`forward-over-forward`).  Just like in MFEM, user code is
//! written once as a functor generic over the scalar type
//! ([`AdScalar`]) and is then evaluated with `f64` (plain value),
//! [`AdFloat`] (value + gradient) or [`Ad2`] (Hessian).
//!
//! The p-Laplacian application from MFEM's `example.hpp` lives in
//! [`plaplacian`]; the miniapp port is `miniapps/autodiff/`.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub mod plaplacian;

pub use plaplacian::{PLaplacianForm, PLaplacianIntegrator};

// ─── Dual numbers (mfem::future::dual) ───────────────────────────────────────

/// Dual number: a `value` plus its `gradient` (first derivative w.r.t. the
/// active direction).  Mirrors `mfem::future::dual<value_type, gradient_type>`.
///
/// Nesting (`Dual<Dual, Dual>`) yields second derivatives, exactly like
/// MFEM's `AD2Type`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dual<V = f64, G = f64> {
    /// the actual numerical value
    pub value: V,
    /// the partial derivative of `value` w.r.t. the active direction
    pub gradient: G,
}

impl Dual {
    /// Create a dual number from a value and a first-order seed.
    #[inline]
    pub const fn new(value: f64, gradient: f64) -> Self {
        Dual { value, gradient }
    }
}

impl<V, G> Dual<V, G> {
    /// Nested constructor (`Dual<Dual, Dual>`).
    #[inline]
    pub const fn nested(value: V, gradient: G) -> Self {
        Dual { value, gradient }
    }
}

impl From<f64> for Dual {
    /// Promote a plain `f64` to a dual with zero gradient (MFEM
    /// `operator=(real_t a)`).
    #[inline]
    fn from(a: f64) -> Self {
        Dual { value: a, gradient: 0.0 }
    }
}

impl From<Dual> for Dual<Dual, Dual> {
    /// Promote a first-order dual to a second-order dual with zero
    /// second-order parts (`{u, 0}` in MFEM's Hessian seeding).
    #[inline]
    fn from(a: Dual) -> Self {
        Dual {
            value: Dual::new(a.value, 0.0),
            gradient: Dual::new(a.gradient, 0.0),
        }
    }
}

// Arithmetic for Dual (first order).  All the formulas are the standard dual
// arithmetic; they are written out concretely (not generically over V, G) so
// every operation is transparent and warning-free.

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Dual { value: self.value + o.value, gradient: self.gradient + o.gradient }
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Dual { value: self.value - o.value, gradient: self.gradient - o.gradient }
    }
}

impl Mul for Dual {
    type Output = Self;
    /// `(u·v)' = u'·v + u·v'`
    #[inline]
    fn mul(self, o: Self) -> Self {
        Dual {
            value: self.value * o.value,
            gradient: self.gradient * o.value + self.value * o.gradient,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    /// `(u/v)' = (u'·v − u·v') / v²`
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / (o.value * o.value);
        Dual {
            value: self.value / o.value,
            gradient: (self.gradient * o.value - self.value * o.gradient) * inv,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Dual { value: -self.value, gradient: -self.gradient }
    }
}

impl Add<f64> for Dual {
    type Output = Self;
    #[inline]
    fn add(self, a: f64) -> Self {
        Dual { value: self.value + a, gradient: self.gradient }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, a: f64) -> Self {
        Dual { value: self.value - a, gradient: self.gradient }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, a: f64) -> Self {
        Dual { value: self.value * a, gradient: self.gradient * a }
    }
}

impl Div<f64> for Dual {
    type Output = Self;
    #[inline]
    fn div(self, a: f64) -> Self {
        Dual { value: self.value / a, gradient: self.gradient / a }
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn add(self, o: Dual) -> Dual {
        Dual { value: self + o.value, gradient: o.gradient }
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn sub(self, o: Dual) -> Dual {
        Dual { value: self - o.value, gradient: -o.gradient }
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn mul(self, o: Dual) -> Dual {
        Dual { value: self * o.value, gradient: self * o.gradient }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    /// `(c/v)' = −c·v'/v²`
    #[inline]
    fn div(self, o: Dual) -> Dual {
        let inv = 1.0 / (o.value * o.value);
        Dual { value: self / o.value, gradient: -self * o.gradient * inv }
    }
}

impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl MulAssign for Dual {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl AddAssign<f64> for Dual {
    #[inline]
    fn add_assign(&mut self, a: f64) {
        self.value += a;
    }
}

/// `sqrt(x)` for a dual: `{√v, g / (2√v)}` (MFEM dual.hpp `operator sqrt`).
#[must_use]
#[inline]
pub fn sqrt(x: Dual) -> Dual {
    let r = x.value.sqrt();
    Dual { value: r, gradient: x.gradient / (2.0 * r) }
}

/// `ln(x)` for a dual: `{ln v, g / v}`.
#[must_use]
#[inline]
pub fn ln(x: Dual) -> Dual {
    Dual { value: x.value.ln(), gradient: x.gradient / x.value }
}

/// `exp(x)` for a dual: `{eᵛ, eᵛ·g}`.
#[must_use]
#[inline]
pub fn exp(x: Dual) -> Dual {
    let e = x.value.exp();
    Dual { value: e, gradient: e * x.gradient }
}

/// `pow(x, n)` for a dual with a constant exponent: `{vⁿ, n·vⁿ⁻¹·g}`.
///
/// Note MFEM's `pow(ee*ee + norm2, p/2)` in `example.hpp` uses exactly this
/// form (constant exponent, dual base).
#[must_use]
#[inline]
pub fn powf(x: Dual, n: f64) -> Dual {
    Dual { value: x.value.powf(n), gradient: n * x.value.powf(n - 1.0) * x.gradient }
}

// ─── Second-order (nested) dual arithmetic: Dual<Dual, Dual> ─────────────────

/// Second-order arithmetic is ordinary dual arithmetic with `AdFloat`
/// components: the formulas above hold componentwise.  We spell them out so
/// each step stays visible and warning-free.

impl Add for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Dual::nested(self.value + o.value, self.gradient + o.gradient)
    }
}

impl Sub for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Dual::nested(self.value - o.value, self.gradient - o.gradient)
    }
}

impl Mul for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Dual::nested(
            self.value * o.value,
            self.gradient * o.value + self.value * o.gradient,
        )
    }
}

impl Div for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = o.value * o.value; // AdFloat: 1/v² (with its own ε₁ info)
        let inv = Dual::new(1.0, 0.0) / inv;
        Dual::nested(
            self.value / o.value,
            (self.gradient * o.value - self.value * o.gradient) * inv,
        )
    }
}

impl Neg for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Dual::nested(-self.value, -self.gradient)
    }
}

impl Add<f64> for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn add(self, a: f64) -> Self {
        Dual::nested(self.value + a, self.gradient)
    }
}

impl Sub<f64> for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn sub(self, a: f64) -> Self {
        Dual::nested(self.value - a, self.gradient)
    }
}

impl Mul<f64> for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn mul(self, a: f64) -> Self {
        Dual::nested(self.value * a, self.gradient * a)
    }
}

impl Div<f64> for Dual<Dual, Dual> {
    type Output = Self;
    #[inline]
    fn div(self, a: f64) -> Self {
        Dual::nested(self.value / a, self.gradient / a)
    }
}

impl Add<Dual<Dual, Dual>> for f64 {
    type Output = Dual<Dual, Dual>;
    #[inline]
    fn add(self, o: Dual<Dual, Dual>) -> Self::Output {
        Dual::nested(self + o.value, o.gradient)
    }
}

impl Sub<Dual<Dual, Dual>> for f64 {
    type Output = Dual<Dual, Dual>;
    #[inline]
    fn sub(self, o: Dual<Dual, Dual>) -> Self::Output {
        Dual::nested(self - o.value, -o.gradient)
    }
}

impl Mul<Dual<Dual, Dual>> for f64 {
    type Output = Dual<Dual, Dual>;
    #[inline]
    fn mul(self, o: Dual<Dual, Dual>) -> Self::Output {
        Dual::nested(self * o.value, self * o.gradient)
    }
}

impl Div<Dual<Dual, Dual>> for f64 {
    type Output = Dual<Dual, Dual>;
    #[inline]
    fn div(self, o: Dual<Dual, Dual>) -> Self::Output {
        let inv = Dual::new(1.0, 0.0) / (o.value * o.value);
        Dual::nested(self / o.value, -self * o.gradient * inv)
    }
}

impl AddAssign for Dual<Dual, Dual> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl SubAssign for Dual<Dual, Dual> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl MulAssign for Dual<Dual, Dual> {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl AddAssign<f64> for Dual<Dual, Dual> {
    #[inline]
    fn add_assign(&mut self, a: f64) {
        self.value += a;
    }
}

/// `sqrt` of a second-order dual: `c = sqrt(x)` satisfies `c² = x`, so
/// `c.value = sqrt(x.value)` and `c.gradient = x.gradient / (2·c.value)`
/// (all products in `AdFloat`, carrying the ε₁ sensitivity).
#[must_use]
#[inline]
pub fn sqrt2(x: Dual<Dual, Dual>) -> Dual<Dual, Dual> {
    let r = sqrt(x.value);
    Dual::nested(r, x.gradient * (Dual::new(1.0, 0.0) / (r + r)))
}

/// `ln` of a second-order dual: `{ln v, g / v}`.
#[must_use]
#[inline]
pub fn ln2(x: Dual<Dual, Dual>) -> Dual<Dual, Dual> {
    Dual::nested(
        ln(x.value),
        x.gradient * (Dual::new(1.0, 0.0) / x.value),
    )
}

/// `pow(x, n)` with constant exponent for a second-order dual:
/// value part `vⁿ`, gradient part `n·vⁿ⁻¹·g` (both in `AdFloat`).
#[must_use]
#[inline]
pub fn powf2(x: Dual<Dual, Dual>, n: f64) -> Dual<Dual, Dual> {
    Dual::nested(
        powf(x.value, n),
        x.gradient * powf(x.value, n - 1.0) * n,
    )
}

// ─── Accessors (mfem::future::get_value / get_gradient) ──────────────────────

/// Extract the plain value (recursive for nested duals).
pub trait GetValue {
    /// The plain `f64` value.
    fn get_value(&self) -> f64;
}

impl GetValue for f64 {
    #[inline]
    fn get_value(&self) -> f64 {
        *self
    }
}

impl<V: GetValue, G> GetValue for Dual<V, G> {
    #[inline]
    fn get_value(&self) -> f64 {
        self.value.get_value()
    }
}

// ─── AdScalar: the generic functor scalar ────────────────────────────────────

/// Scalar type that a [`QFunction`] / [`QVectorFunc`] functor can be
/// evaluated with: `f64` (plain value), [`AdFloat`] (value + first
/// derivatives) or [`Ad2`] (value + second derivatives).
///
/// This is the Rust analogue of MFEM's C++ template parameter
/// `TDataType` in the `MyEnergyFunctor` / `MyResidualFunctor` functors.
pub trait AdScalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + AddAssign<f64>
{
    /// Additive identity (MFEM: `uu = 0.0` / `rez = 0.`).
    fn zero() -> Self;
    /// Promote a parameter (`real_t pp = vparam[0]`) to the scalar type.
    fn of_f64(a: f64) -> Self;
    /// Constant-exponent power (`std::pow(x, n)` with plain `n`).
    fn powf_s(self, n: f64) -> Self;
    /// Square root.
    fn sqrt_s(self) -> Self;
    /// Natural logarithm.
    fn ln_s(self) -> Self;
}

impl AdScalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn of_f64(a: f64) -> Self {
        a
    }
    #[inline]
    fn powf_s(self, n: f64) -> Self {
        self.powf(n)
    }
    #[inline]
    fn sqrt_s(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn ln_s(self) -> Self {
        self.ln()
    }
}

impl AdScalar for Dual {
    #[inline]
    fn zero() -> Self {
        Dual::new(0.0, 0.0)
    }
    #[inline]
    fn of_f64(a: f64) -> Self {
        Dual::new(a, 0.0)
    }
    #[inline]
    fn powf_s(self, n: f64) -> Self {
        powf(self, n)
    }
    #[inline]
    fn sqrt_s(self) -> Self {
        sqrt(self)
    }
    #[inline]
    fn ln_s(self) -> Self {
        ln(self)
    }
}

impl AdScalar for Dual<Dual, Dual> {
    #[inline]
    fn zero() -> Self {
        Dual::nested(Dual::new(0.0, 0.0), Dual::new(0.0, 0.0))
    }
    #[inline]
    fn of_f64(a: f64) -> Self {
        Dual::nested(Dual::new(a, 0.0), Dual::new(0.0, 0.0))
    }
    #[inline]
    fn powf_s(self, n: f64) -> Self {
        powf2(self, n)
    }
    #[inline]
    fn sqrt_s(self) -> Self {
        sqrt2(self)
    }
    #[inline]
    fn ln_s(self) -> Self {
        ln2(self)
    }
}

/// MFEM `ad::ADFloatType` (`future::dual<real_t, real_t>`): one first-order
/// seed per evaluation.
pub type AdFloat = Dual;
/// MFEM `AD2Type` (`future::dual<AD1Type, AD1Type>`): forward-over-forward
/// nested dual carrying second derivatives.
pub type Ad2 = Dual<AdFloat, AdFloat>;

// ─── QFunctionAutoDiff (energy functors) ─────────────────────────────────────

/// Scalar quadrature-function functor, generic over the AD scalar type.
///
/// Rust analogue of MFEM's `template<typename TDataType, ...> class
/// MyEnergyFunctor { TDataType operator()(vparam, uu); }`: implement
/// [`QFunction::eval`] once with `T: AdScalar` and the driver evaluates it
/// with plain `f64`, [`AdFloat`] (Grad) or [`Ad2`] (Hessian).
pub trait QFunction {
    /// Evaluate the energy for parameters `vparam` and state `uu`.
    fn eval<T: AdScalar>(&self, vparam: &[f64], uu: &[T]) -> T;
}

/// Evaluates a templated scalar function together with its first derivatives
/// (Grad) and Hessian, all by forward-mode AD.  1:1 port of MFEM's
/// `QFunctionAutoDiff` (native branch).
#[derive(Debug, Clone)]
pub struct QFunctionAutoDiff<F> {
    /// The user energy functor.
    pub func: F,
}

impl<F: QFunction> QFunctionAutoDiff<F> {
    /// Wrap a [`QFunction`] functor.
    pub fn new(func: F) -> Self {
        QFunctionAutoDiff { func }
    }

    /// Plain value evaluation (MFEM `Eval`).
    pub fn eval(&self, vparam: &[f64], uu: &[f64]) -> f64 {
        self.func.eval(vparam, uu).get_value()
    }

    /// First derivative of the energy w.r.t. every state component
    /// (MFEM `Grad`): one forward seed per component.
    pub fn grad(&self, vparam: &[f64], uu: &[f64], rr: &mut [f64]) {
        let n = uu.len();
        debug_assert_eq!(rr.len(), n, "grad: rr must have the state size");
        let mut aduu: Vec<AdFloat> = uu.iter().map(|&u| Dual::new(u, 0.0)).collect();
        let mut rez;
        for (ii, r) in rr.iter_mut().enumerate() {
            aduu[ii].gradient = 1.0;
            rez = self.func.eval(vparam, &aduu);
            *r = rez.gradient;
            aduu[ii].gradient = 0.0;
        }
    }

    /// Same as [`Self::grad`] (MFEM `VectorFunc` mirrors `Grad`).
    pub fn vector_func(&self, vparam: &[f64], uu: &[f64], rr: &mut [f64]) {
        self.grad(vparam, uu, rr);
    }

    /// Hessian of the energy (MFEM `Hessian`): forward-over-forward seeding
    /// on nested duals, filling the lower triangle and mirroring it to the
    /// upper triangle.  `jac` is row-major `n × n`.
    pub fn hessian(&self, vparam: &[f64], uu: &[f64], jac: &mut [f64]) {
        let n = uu.len();
        debug_assert_eq!(jac.len(), n * n, "hessian: jac must be n×n row-major");
        for v in jac.iter_mut() {
            *v = 0.0;
        }
        let mut aduu: Vec<Ad2> = (0..n)
            .map(|k| Dual::nested(Dual::new(uu[k], 0.0), Dual::new(0.0, 0.0)))
            .collect();
        for ii in 0..n {
            aduu[ii].value = Dual::new(uu[ii], 1.0);
            for jj in 0..=ii {
                aduu[jj].gradient = Dual::new(1.0, 0.0);
                let rez = self.func.eval::<Ad2>(vparam, &aduu);
                jac[ii * n + jj] = rez.gradient.gradient;
                jac[jj * n + ii] = jac[ii * n + jj];
                aduu[jj].gradient = Dual::new(0.0, 0.0);
            }
            aduu[ii].value = Dual::new(uu[ii], 0.0);
        }
    }

    /// Same as [`Self::hessian`] (MFEM `Jacobian` mirrors `Hessian`).
    pub fn jacobian(&self, vparam: &[f64], uu: &[f64], jac: &mut [f64]) {
        self.hessian(vparam, uu, jac);
    }
}

// ─── QVectorFuncAutoDiff (residual functors) ─────────────────────────────────

/// Vector quadrature-function functor (a residual at an integration point),
/// generic over the AD scalar type.  Rust analogue of MFEM's
/// `MyResidualFunctor` template functor.
pub trait QVectorFunc {
    /// Evaluate the residual for parameters `vparam`, state `uu`, writing
    /// into `rr` (same length as `uu`).
    fn eval<T: AdScalar>(&self, vparam: &[f64], uu: &[T], rr: &mut [T]);
}

/// Evaluates a templated vector function together with its Jacobian by
/// forward-mode AD.  1:1 port of MFEM's `QVectorFuncAutoDiff` (native
/// branch).
#[derive(Debug, Clone)]
pub struct QVectorFuncAutoDiff<F> {
    /// The user residual functor.
    pub func: F,
}

impl<F: QVectorFunc> QVectorFuncAutoDiff<F> {
    /// Wrap a [`QVectorFunc`] functor.
    pub fn new(func: F) -> Self {
        QVectorFuncAutoDiff { func }
    }

    /// Plain evaluation of the residual (MFEM `VectorFunc`).
    pub fn vector_func(&self, vparam: &[f64], uu: &[f64], rr: &mut [f64]) {
        self.func.eval(vparam, uu, rr);
    }

    /// Jacobian of the residual, `jac[jj * state_size + ii] = ∂rr_jj/∂uu_ii`
    /// (MFEM `Jacobian`): one forward seed per state component.  `jac` is
    /// row-major `vector_size × state_size` — here the residual length is
    /// equal to the state length, as in MFEM's p-Laplacian usage.
    pub fn jacobian(&self, vparam: &[f64], uu: &[f64], jac: &mut [f64]) {
        let n = uu.len();
        debug_assert_eq!(jac.len(), n * n, "jacobian: jac must be n×n row-major");
        for v in jac.iter_mut() {
            *v = 0.0;
        }
        let mut aduu: Vec<AdFloat> = uu.iter().map(|&u| Dual::new(u, 0.0)).collect();
        let mut rr: Vec<AdFloat> = vec![Dual::new(0.0, 0.0); n];
        for ii in 0..n {
            aduu[ii].gradient = 1.0;
            self.func.eval(vparam, &aduu, &mut rr);
            for (jj, r) in rr.iter().enumerate() {
                jac[jj * n + ii] = r.gradient;
            }
            aduu[ii].gradient = 0.0;
        }
    }
}

// ─── VectorFuncAutoDiff (closure flavour) ────────────────────────────────────

/// Jacobian of a closure `F(&[f64] params, &[AdFloat] state, &mut [AdFloat] result)`
/// by forward seeding — the closure-based MFEM `VectorFuncAutoDiff` used with
/// lambda expressions in `seq_test.cpp`.  `jac` is row-major
/// `result_size × state_size`.
pub fn vector_func_jacobian<F>(
    vparam: &[f64],
    vstate: &[f64],
    result_size: usize,
    f: F,
    jac: &mut [f64],
) where
    F: Fn(&[f64], &[AdFloat], &mut [AdFloat]),
{
    let n = vstate.len();
    debug_assert_eq!(jac.len(), result_size * n, "jacobian: bad jac size");
    for v in jac.iter_mut() {
        *v = 0.0;
    }
    let mut aduu: Vec<AdFloat> = vstate.iter().map(|&u| Dual::new(u, 0.0)).collect();
    let mut rr: Vec<AdFloat> = vec![Dual::new(0.0, 0.0); result_size];
    for ii in 0..n {
        aduu[ii].gradient = 1.0;
        f(vparam, &aduu, &mut rr);
        for (jj, r) in rr.iter().enumerate() {
            jac[jj * n + ii] = r.gradient;
        }
        aduu[ii].gradient = 0.0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative difference helper.
    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
    }

    // ── 1. Dual arithmetic vs analytic derivatives ───────────────────────────

    #[test]
    fn dual_arithmetic_matches_analytic() {
        // f(x) = sqrt(1+x^2) * ln(1+x^3) + powf(2+x^2, 3.5), x = 1.7
        let x = 1.7_f64;
        let xd = Dual::new(x, 1.0);
        let f = sqrt(Dual::new(1.0, 0.0) + xd * xd) * ln(Dual::new(1.0, 0.0) + xd * xd * xd)
            + powf(Dual::new(2.0, 0.0) + xd * xd, 3.5);

        let s = (1.0 + x * x).sqrt();
        // analytic derivative
        let d_s = x / s;
        let t = 1.0 + x * x * x;
        let d_t = 3.0 * x * x;
        let u = (2.0 + x * x).powf(3.5);
        let d_u = 3.5 * (2.0 + x * x).powf(2.5) * 2.0 * x;
        let expected = d_s * t.ln() + s * (d_t / t) + d_u;

        assert!(rel(f.value, expected_value(x)) < 1e-14, "value {}", f.value);
        assert!(rel(f.gradient, expected) < 1e-13, "grad {} vs {}", f.gradient, expected);
    }

    fn expected_value(x: f64) -> f64 {
        (1.0 + x * x).sqrt() * (1.0 + x * x * x).ln() + (2.0 + x * x).powf(3.5)
    }

    // ── 2. p-Laplacian energy functor: Grad/Hessian vs analytic & FD ─────────

    /// The exact MFEM `MyEnergyFunctor` from `miniapps/autodiff/example.hpp`
    /// (state `uu = [u_x, u_y, u_z, u]`, params `[pp, ee, ff]`).
    #[derive(Debug, Clone, Copy)]
    struct MyEnergyFunctor;

    impl QFunction for MyEnergyFunctor {
        fn eval<T: AdScalar>(&self, vparam: &[f64], uu: &[T]) -> T {
            let pp = vparam[0];
            let ee = vparam[1];
            let ff = vparam[2];
            let norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];
            T::powf_s(T::of_f64(ee * ee) + norm2, pp / 2.0) / T::of_f64(pp)
                - uu[3] * T::of_f64(ff)
        }
    }

    /// Analytic grad of the p-Laplacian energy density.
    fn analytic_grad(pp: f64, ee: f64, ff: f64, uu: &[f64; 4]) -> [f64; 4] {
        let norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];
        let t = (ee * ee + norm2).powf((pp - 2.0) / 2.0);
        [t * uu[0], t * uu[1], t * uu[2], -ff]
    }

    /// Analytic Hessian of the p-Laplacian energy density:
    /// `H = b^((p−2)/2)·I + (p−2)·b^((p−4)/2)·∇u ∇uᵀ` with `b = ee² + |∇u|²`
    /// (identical to MFEM `pLaplace::AssembleElementGrad`'s `aa1`/`aa0`).
    fn analytic_hessian(pp: f64, ee: f64, uu: &[f64; 4]) -> [[f64; 4]; 4] {
        let norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];
        let b = ee * ee + norm2;
        let c1 = (pp - 2.0) * b.powf((pp - 4.0) / 2.0);
        let c2 = b.powf((pp - 2.0) / 2.0);
        let mut h = [[0.0_f64; 4]; 4];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = c1 * uu[i] * uu[j] + c2 * (i == j) as u8 as f64;
            }
        }
        h
    }

    #[test]
    fn plaplacian_functor_grad_and_hessian_exact() {
        let vparam = [4.6_f64, 1e-8, 1.0];
        let uu = [0.8_f64, -1.3, 2.1, 0.4];

        let adf = QFunctionAutoDiff::new(MyEnergyFunctor);

        // Grad vs analytic
        let mut rr = [0.0_f64; 4];
        adf.grad(&vparam, &uu, &mut rr);
        let ga = analytic_grad(vparam[0], vparam[1], vparam[2], &uu);
        for k in 0..4 {
            assert!(
                rel(rr[k], ga[k]) < 1e-12,
                "grad[{k}] = {} vs analytic {}",
                rr[k],
                ga[k]
            );
        }

        // Hessian vs analytic
        let mut hh = [0.0_f64; 16];
        adf.hessian(&vparam, &uu, &mut hh);
        let ha = analytic_hessian(vparam[0], vparam[1], &uu);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    rel(hh[i * 4 + j], ha[i][j]) < 1e-12,
                    "hessian[{i}][{j}] = {} vs analytic {}",
                    hh[i * 4 + j],
                    ha[i][j]
                );
            }
        }
    }

    #[test]
    fn plaplacian_functor_grad_fd_check() {
        // Central finite differences of the energy must match AD Grad to
        // FD accuracy.
        let vparam = [3.5_f64, 0.25, 2.0];
        let uu = [0.6_f64, -0.4, 1.1, 2.0];
        let adf = QFunctionAutoDiff::new(MyEnergyFunctor);
        let mut rr = [0.0_f64; 4];
        adf.grad(&vparam, &uu, &mut rr);
        let h = 1e-6;
        for k in 0..4 {
            let mut up = uu;
            up[k] += h;
            let mut um = uu;
            um[k] -= h;
            let fd = (adf.eval(&vparam, &up) - adf.eval(&vparam, &um)) / (2.0 * h);
            let scale = rr[k].abs().max(1.0);
            assert!(
                (rr[k] - fd).abs() / scale < 5e-7,
                "grad[{k}] ad {} vs fd {}",
                rr[k],
                fd
            );
        }
    }

    // ── 3. Residual-functor Jacobian (seq_test.cpp core check) ───────────────

    /// MFEM `DiffusionResidual` from `miniapps/autodiff/seq_test.cpp`
    /// (params `[kappa, load]`).
    #[derive(Debug, Clone, Copy)]
    struct DiffusionResidual;

    impl QVectorFunc for DiffusionResidual {
        fn eval<T: AdScalar>(&self, vparam: &[f64], uu: &[T], rr: &mut [T]) {
            let kappa = vparam[0];
            let load = vparam[1];
            rr[0] = T::of_f64(kappa) * uu[0];
            rr[1] = T::of_f64(kappa) * uu[1];
            rr[2] = T::of_f64(kappa) * uu[2];
            rr[3] = T::of_f64(-load);
        }
    }

    #[test]
    fn seq_test_diffusion_jacobian_matches_cpp() {
        // Exact state/params of `seq_test.cpp` — the Jacobian must equal the
        // C++ output diag(3,3,3,0) (see the captured reference run).
        let param = [3.0_f64, 2.0];
        let state = [1.0_f64, 2.0, 3.0, 4.0];

        let rdf = QVectorFuncAutoDiff::new(DiffusionResidual);

        let mut rr = [0.0_f64; 4];
        rdf.vector_func(&param, &state, &mut rr);
        for (r, e) in rr.iter().zip([3.0_f64, 6.0, 9.0, -2.0]) {
            assert_eq!(*r, e, "residual must match C++ rr0");
        }

        let mut hh = [0.0_f64; 16];
        rdf.jacobian(&param, &state, &mut hh);
        let expect = [
            3.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 3.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        for (h, e) in hh.iter().zip(expect) {
            assert_eq!(*h, e, "Jacobian must match C++ hh1 bit-for-bit");
        }
    }

    #[test]
    fn seq_test_diffusion_energy_matches_cpp() {
        // `adf.Eval(param, state)` = 13 in the C++ run.
        #[derive(Debug, Clone, Copy)]
        struct DiffusionFunctional;
        impl QFunction for DiffusionFunctional {
            fn eval<T: AdScalar>(&self, vparam: &[f64], uu: &[T]) -> T {
                let kappa = vparam[0];
                let load = vparam[1];
                T::of_f64(kappa) * (uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2])
                    / T::of_f64(2.0)
                    - T::of_f64(load) * uu[3]
            }
        }
        let param = [3.0_f64, 2.0];
        let state = [1.0_f64, 2.0, 3.0, 4.0];
        let adf = QFunctionAutoDiff::new(DiffusionFunctional);
        assert_eq!(adf.eval(&param, &state), 13.0, "energy must match C++");

        // And its Grad/Hessian = kappa*I on the gradient block:
        let mut rr = [0.0_f64; 4];
        adf.grad(&param, &state, &mut rr);
        assert_eq!(rr, [3.0, 6.0, 9.0, -2.0]);
        let mut hh = [0.0_f64; 16];
        adf.hessian(&param, &state, &mut hh);
        assert_eq!(hh, [
            3.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 3.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ]);
    }

    #[test]
    fn closure_jacobian_matches_functor() {
        // The lambda-expression flavour from seq_test.cpp.
        let param = [3.0_f64, 2.0];
        let state = [1.0_f64, 2.0, 3.0, 4.0];
        let func = |vp: &[f64], uu: &[AdFloat], rr: &mut [AdFloat]| {
            let k = Dual::new(vp[0], 0.0);
            let l = Dual::new(vp[1], 0.0);
            rr[0] = k * uu[0];
            rr[1] = k * uu[1];
            rr[2] = k * uu[2];
            rr[3] = -l;
        };
        let mut hh = [0.0_f64; 16];
        vector_func_jacobian(&param, &state, 4, func, &mut hh);
        let expect = [
            3.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 3.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        for (h, e) in hh.iter().zip(expect) {
            assert_eq!(*h, e);
        }
    }

    // ── 4. TMOP cross-check: AD vs analytic eval_p vs finite differences ────

    /// Generic (AD-capable) mirror of fem-mesh TMOP metric energies:
    /// `W001 = |J|²`, `W002 = 0.5·|J|²/det − 1`, `W007 = I1·(1 + 1/I2) − 4`
    /// (invariants exactly as in `crates/mesh/src/tmop/{metrics,invariants}.rs`).
    fn tmop_w<T: AdScalar>(id: i32, j: &[[T; 2]; 2]) -> T {
        let fnorm2 = j[0][0] * j[0][0] + j[1][0] * j[1][0] + j[0][1] * j[0][1] + j[1][1] * j[1][1];
        let det = j[0][0] * j[1][1] - j[1][0] * j[0][1];
        match id {
            1 => fnorm2,
            2 => T::of_f64(0.5) * fnorm2 / det - T::of_f64(1.0),
            7 => {
                let i2 = det * det;
                fnorm2 * (T::of_f64(1.0) + T::of_f64(1.0) / i2) - T::of_f64(4.0)
            }
            _ => panic!("tmop_w: metric {id} not covered by the AD cross-check"),
        }
    }

    #[test]
    fn tmop_metric_dw_dj_ad_vs_analytic_vs_fd() {
        use fem_mesh::tmop::{TmopMetric001, TmopMetric002, TmopMetric007, TmopQualityMetric};

        // A representative set of non-degenerate Jacobians (target -> physical).
        let jacobi = [
            [[1.3_f64, 0.2], [-0.1, 0.9]],
            [[0.7, 0.4], [0.3, 1.2]],
            [[1.0, 0.5], [0.25, 1.0]],
        ];
        let metrics: [(i32, Box<dyn TmopQualityMetric>); 3] = [
            (1, Box::new(TmopMetric001)),
            (2, Box::new(TmopMetric002)),
            (7, Box::new(TmopMetric007)),
        ];

        for (id, metric) in &metrics {
            for jpt in &jacobi {
                // analytic derivative from fem-mesh (P = dW/dJ)
                let mut p = [[0.0_f64; 2]; 2];
                metric.eval_p(jpt, &mut p);

                // AD derivative through the mirrored energy
                let mut ad_grad = [[0.0_f64; 2]; 2];
                for a in 0..2 {
                    for b in 0..2 {
                        let mut jd = [[Dual::new(0.0, 0.0); 2]; 2];
                        for (i, row) in jd.iter_mut().enumerate() {
                            for (k, cell) in row.iter_mut().enumerate() {
                                *cell = Dual::new(jpt[i][k], (i == a && k == b) as u8 as f64);
                            }
                        }
                        ad_grad[a][b] = tmop_w::<Dual>(*id, &jd).gradient;
                    }
                }

                // central finite differences
                let h = 1e-6;
                let mut fd = [[0.0_f64; 2]; 2];
                for a in 0..2 {
                    for b in 0..2 {
                        let mut jp = *jpt;
                        jp[a][b] += h;
                        let mut jm = *jpt;
                        jm[a][b] -= h;
                        fd[a][b] = (metric.eval_w(&jp) - metric.eval_w(&jm)) / (2.0 * h);
                    }
                }

                for a in 0..2 {
                    for b in 0..2 {
                        let e_ad = (ad_grad[a][b] - p[a][b]).abs();
                        let e_fd = (fd[a][b] - p[a][b]).abs();
                        let scale = p[a][b].abs().max(1.0);
                        assert!(
                            e_ad / scale < 1e-12,
                            "metric {id} dW/dJ[{a}][{b}]: AD {} vs analytic {} (err {e_ad})",
                            ad_grad[a][b],
                            p[a][b]
                        );
                        assert!(
                            e_fd / scale < 1e-5,
                            "metric {id} dW/dJ[{a}][{b}]: FD {} vs analytic {} (err {e_fd})",
                            fd[a][b],
                            p[a][b]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tmop_nodal_energy_gradient_ad_vs_analytic_vs_fd() {
        use fem_mesh::tmop::{TmopMetric001, TmopMetric002, TmopQualityMetric};

        // One straight-sided quad element: energy E(x) = Σ_q w_q W(J_q(x))
        // over 2×2 Gauss points on the unit square; the Jacobian at a point is
        // J = Σ_k x_k ⊗ ∇φ_k(ξ_q) with the bilinear basis on [0,1]².
        // Node order (H1 topological, matches fem-mesh Quad4): v0..v3 CCW.
        const CORNERS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        // 2×2 Gauss rule on [0,1]²: points ±1/√3 mapped by (t+1)/2, w = 1/4.
        let a = 0.5 * (1.0 - 1.0 / 3.0_f64.sqrt());
        let b = 0.5 * (1.0 + 1.0 / 3.0_f64.sqrt());
        let gauss: [(f64, [f64; 2]); 4] =
            [(0.25, [a, a]), (0.25, [b, a]), (0.25, [b, b]), (0.25, [a, b])];

        // Deformed (but non-degenerate) node positions.
        let nodes = [[0.1_f64, 0.0], [1.15, 0.12], [0.9, 1.25], [-0.05, 0.95]];

        // ∇φ_k at a reference point for the bilinear quad.
        let grad_phi = |xi: &[f64; 2]| -> [[f64; 2]; 4] {
            let mut g = [[0.0_f64; 2]; 4];
            for (k, c) in CORNERS.iter().enumerate() {
                // h_a(ξ)·h_b(η), h_0 = 1−t, h_1 = t
                let ha = if c[0] < 0.5 { 1.0 - xi[0] } else { xi[0] };
                let hb = if c[1] < 0.5 { 1.0 - xi[1] } else { xi[1] };
                let dha = if c[0] < 0.5 { -1.0 } else { 1.0 };
                let dhb = if c[1] < 0.5 { -1.0 } else { 1.0 };
                g[k] = [dha * hb, ha * dhb];
            }
            g
        };

        let energy_at = |id: i32, x: &[[f64; 2]; 4]| -> f64 {
            let mut e = 0.0;
            for (w, xi) in &gauss {
                let gp = grad_phi(xi);
                let mut j = [[0.0_f64; 2]; 2];
                for k in 0..4 {
                    j[0][0] += x[k][0] * gp[k][0];
                    j[0][1] += x[k][0] * gp[k][1];
                    j[1][0] += x[k][1] * gp[k][0];
                    j[1][1] += x[k][1] * gp[k][1];
                }
                e += w
                    * match id {
                        1 => tmop_w::<f64>(1, &j),
                        2 => tmop_w::<f64>(2, &j),
                        _ => unreachable!(),
                    };
            }
            e
        };

        for id in [1_i32, 2] {
            // analytic derivative via fem-mesh eval_p: dE/dx_k = Σ_q w_q P_q : dJ_q/dx_k
            let metric: Box<dyn TmopQualityMetric> = if id == 1 {
                Box::new(TmopMetric001)
            } else {
                Box::new(TmopMetric002)
            };
            let mut analytic = [0.0_f64; 8];
            for (w, xi) in &gauss {
                let gp = grad_phi(xi);
                let mut j = [[0.0_f64; 2]; 2];
                for k in 0..4 {
                    j[0][0] += nodes[k][0] * gp[k][0];
                    j[0][1] += nodes[k][0] * gp[k][1];
                    j[1][0] += nodes[k][1] * gp[k][0];
                    j[1][1] += nodes[k][1] * gp[k][1];
                }
                let mut p = [[0.0_f64; 2]; 2];
                metric.eval_p(&j, &mut p);
                for k in 0..4 {
                    // dJ[i][b]/dx_k[a] = gp[k][b] * (a == i)
                    for a in 0..2 {
                        for b in 0..2 {
                            analytic[k * 2 + a] += w * p[a][b] * gp[k][b];
                        }
                    }
                }
            }

            // AD derivative: seed the 8 nodal coordinates one at a time.
            let mut ad = [0.0_f64; 8];
            for d in 0..8 {
                let mut xd = [[Dual::new(0.0, 0.0); 2]; 4];
                for k in 0..4 {
                    xd[k][0] = Dual::new(nodes[k][0], (d == k * 2) as u8 as f64);
                    xd[k][1] = Dual::new(nodes[k][1], (d == k * 2 + 1) as u8 as f64);
                }
                let mut e = Dual::new(0.0, 0.0);
                for (w, xi) in &gauss {
                    let gp = grad_phi(xi);
                    let mut j = [[Dual::new(0.0, 0.0); 2]; 2];
                    for k in 0..4 {
                        j[0][0] = j[0][0] + xd[k][0] * Dual::new(gp[k][0], 0.0);
                        j[0][1] = j[0][1] + xd[k][0] * Dual::new(gp[k][1], 0.0);
                        j[1][0] = j[1][0] + xd[k][1] * Dual::new(gp[k][0], 0.0);
                        j[1][1] = j[1][1] + xd[k][1] * Dual::new(gp[k][1], 0.0);
                    }
                    e += *w * tmop_w::<Dual>(id, &j);
                }
                ad[d] = e.gradient;
            }

            // finite differences of the same energy
            let h = 1e-6;
            let mut fd = [0.0_f64; 8];
            for d in 0..8 {
                let mut xp = nodes;
                xp[d / 2][d % 2] += h;
                let mut xm = nodes;
                xm[d / 2][d % 2] -= h;
                fd[d] = (energy_at(id, &xp) - energy_at(id, &xm)) / (2.0 * h);
            }

            for d in 0..8 {
                let scale = analytic[d].abs().max(1.0);
                assert!(
                    (ad[d] - analytic[d]).abs() / scale < 1e-12,
                    "metric {id} dE/dx[{d}]: AD {} vs analytic {}",
                    ad[d],
                    analytic[d]
                );
                assert!(
                    (fd[d] - analytic[d]).abs() / scale < 1e-5,
                    "metric {id} dE/dx[{d}]: FD {} vs analytic {}",
                    fd[d],
                    analytic[d]
                );
            }
        }
    }

    // ── 5. Nested-dual smoke: Ad2 path reaches the same Hessian as FD ───────

    #[test]
    fn nested_dual_hessian_fd_check() {
        // f(u) = pow(1 + |u|^2, 1.75), 3D-style state.
        struct F;
        impl QFunction for F {
            fn eval<T: AdScalar>(&self, _vp: &[f64], uu: &[T]) -> T {
                let n2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];
                T::powf_s(T::of_f64(1.0) + n2, 1.75)
            }
        }
        let adf = QFunctionAutoDiff::new(F);
        let uu = [0.4_f64, -0.9, 1.2];
        let mut hh = [0.0_f64; 9];
        adf.hessian(&[], &uu, &mut hh);

        let h = 1e-4;
        let hf = |k: usize, l: usize| -> f64 {
            let mut f = |d1: usize, d2: usize| -> f64 {
                let mut u2 = uu;
                u2[d1] += h;
                u2[d2] += h;
                let mut m1 = uu;
                m1[d1] += h;
                m1[d2] -= h;
                let mut m2 = uu;
                m2[d1] -= h;
                m2[d2] += h;
                let mut m3 = uu;
                m3[d1] -= h;
                m3[d2] -= h;
                (F.eval::<f64>(&[], &u2) - F.eval::<f64>(&[], &m1)
                    - F.eval::<f64>(&[], &m2)
                    + F.eval::<f64>(&[], &m3))
                    / (4.0 * h * h)
            };
            f(k, l)
        };
        for k in 0..3 {
            for l in 0..3 {
                let scale = hf(k, l).abs().max(1.0);
                assert!(
                    (hh[k * 3 + l] - hf(k, l)).abs() / scale < 1e-6,
                    "H[{k}][{l}] = {} vs FD {}",
                    hh[k * 3 + l],
                    hf(k, l)
                );
            }
        }
    }
}
