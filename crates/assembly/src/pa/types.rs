//! Shared PA data types and traits.

/// Per-element quadrature-point data for PA apply.
/// Layout: flat array of `[n_elems × nqp × (dim*dim + 2)]` f64 values.
/// Per qp: [J⁻ᵀ_00..J⁻ᵀ_22 (row-major), |detJ|, κ].
#[derive(Clone, Debug)]
pub struct PaData {
    pub n_elems: usize,
    pub nqp:     usize,
    pub dim:     usize,
    pub data:    Vec<f64>,
}

impl PaData {
    pub fn new(n_elems: usize, nqp: usize, dim: usize) -> Self {
        let nf = dim * dim + 2; // J⁻ᵀ + |detJ| + κ
        PaData { n_elems, nqp, dim, data: vec![0.0; n_elems * nqp * nf] }
    }

    /// Access the J⁻ᵀ row-major values, |detJ|, and κ at element `e`, qp `q`.
    pub fn elem_qp(&self, e: usize, q: usize) -> &[f64] {
        let nf = self.dim * self.dim + 2;
        let start = (e * self.nqp + q) * nf;
        &self.data[start..start + nf]
    }

    pub fn elem_qp_mut(&mut self, e: usize, q: usize) -> &mut [f64] {
        let nf = self.dim * self.dim + 2;
        let start = (e * self.nqp + q) * nf;
        &mut self.data[start..start + nf]
    }
}
