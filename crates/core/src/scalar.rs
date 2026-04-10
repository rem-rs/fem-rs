use num_traits::{Float, NumAssign};

/// Floating-point scalar type used throughout fem-rs.
///
/// Implemented for `f32` and `f64`.  The default throughout the library is
/// `f64`; `f32` is reserved for memory-critical WASM paths.
///
/// `zero()` and `one()` come from the `Float` supertrait (via `num_traits`);
/// use `num_traits::Zero::zero()` or `<T as num_traits::Zero>::zero()` when
/// you need them unambiguously.
pub trait Scalar:
    Copy
    + Clone
    + Send
    + Sync
    + 'static
    + std::fmt::Debug
    + std::fmt::Display
    + Float
    + NumAssign
    + bytemuck::Pod
{
    /// Conversion from `f64` (lossless for `f64`, lossy for `f32`).
    fn from_f64(v: f64) -> Self;
}

impl Scalar for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self { v }
}

impl Scalar for f32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self { v as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn f64_zero_one() {
        assert_eq!(f64::zero(), 0.0_f64);
        assert_eq!(f64::one(),  1.0_f64);
    }

    #[test]
    fn f32_zero_one() {
        assert_eq!(f32::zero(), 0.0_f32);
        assert_eq!(f32::one(),  1.0_f32);
    }

    #[test]
    fn from_f64_roundtrip() {
        let v = std::f64::consts::PI;
        assert_eq!(f64::from_f64(v), v);
        let _ = f32::from_f64(v);
    }

    #[test]
    fn scalar_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<f64>();
        assert_send_sync::<f32>();
    }
}
