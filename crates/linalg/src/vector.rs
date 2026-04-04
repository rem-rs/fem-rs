use fem_core::Scalar;

/// A heap-allocated column vector with BLAS-like operations.
#[derive(Debug, Clone)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T: Scalar> Vector<T> {
    /// Create a vector of `n` zeros.
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![T::zero(); n] }
    }

    /// Create from an existing `Vec<T>`.
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Length.
    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    /// True if length is zero.
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Borrow as slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] { &self.data }

    /// Mutably borrow as slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] { &mut self.data }

    /// Consume into inner `Vec`.
    pub fn into_vec(self) -> Vec<T> { self.data }

    /// `y = α x + y`  (BLAS daxpy)
    pub fn axpy(&mut self, alpha: T, x: &Self) {
        assert_eq!(self.len(), x.len(), "axpy: length mismatch");
        for (yi, &xi) in self.data.iter_mut().zip(x.data.iter()) {
            *yi += alpha * xi;
        }
    }

    /// `y = α x`  (in-place scale + assign from x)
    pub fn assign_scaled(&mut self, alpha: T, x: &Self) {
        assert_eq!(self.len(), x.len());
        for (yi, &xi) in self.data.iter_mut().zip(x.data.iter()) {
            *yi = alpha * xi;
        }
    }

    /// Scale in place: `x = α x`.
    pub fn scale(&mut self, alpha: T) {
        for v in self.data.iter_mut() { *v *= alpha; }
    }

    /// Euclidean dot product `x · y`.
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(self.len(), other.len(), "dot: length mismatch");
        self.data.iter().zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    /// Euclidean norm `‖x‖₂`.
    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
    }

    /// Fill with constant value.
    pub fn fill(&mut self, v: T) {
        for x in self.data.iter_mut() { *x = v; }
    }

    /// Copy `src` into `self[offset .. offset + src.len()]`.
    ///
    /// # Panics
    /// Panics if `offset + src.len() > self.len()`.
    pub fn set_sub_vector(&mut self, offset: usize, src: &[T]) {
        self.data[offset..offset + src.len()].copy_from_slice(src);
    }

    /// Return a slice `self[offset .. offset + len]`.
    ///
    /// # Panics
    /// Panics if `offset + len > self.len()`.
    pub fn get_sub_vector(&self, offset: usize, len: usize) -> &[T] {
        &self.data[offset..offset + len]
    }
}

impl<T: Scalar> std::ops::Index<usize> for Vector<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T { &self.data[i] }
}

impl<T: Scalar> std::ops::IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, i: usize) -> &mut T { &mut self.data[i] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy() {
        let mut y = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let x = Vector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        y.axpy(2.0, &x);
        assert_eq!(y.as_slice(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn dot_norm() {
        let v = Vector::<f64>::from_vec(vec![3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-14);
    }

    #[test]
    fn sub_vector_roundtrip() {
        let mut v = Vector::<f64>::zeros(10);
        v.set_sub_vector(3, &[1.0, 2.0, 3.0]);
        assert_eq!(v.get_sub_vector(3, 3), &[1.0, 2.0, 3.0]);
        assert_eq!(v[2], 0.0);
        assert_eq!(v[6], 0.0);
    }
}
