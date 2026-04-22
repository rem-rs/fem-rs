//! Runtime tuning via environment variables.
//!
//! All variables are read once per process on first use (lazy static).
//! To test overrides, use a dedicated integration test binary (see
//! `tests/local_rayon_min_env.rs`) so nothing else initializes the cache first.
//!
//! | Variable | Default | Meaning |
//! |----------|---------|---------|
//! | [`FEM_PARALLEL_LOCAL_RAYON_MIN`] | `4096` | Minimum length (owned DOFs for reductions, or full local length for `axpy`/`scale`) before Rayon is used for local CPU work on native targets. Set to `1` to force Rayon everywhere (not recommended for tiny vectors). |
//! | [`FEM_LINALG_SPMV_PARALLEL_MIN_ROWS`](crate::FEM_LINALG_SPMV_PARALLEL_MIN_ROWS) | `128` | Minimum CSR row count before Rayon parallelizes local `CsrMatrix::spmv` / `spmv_add` inside [`ParCsrMatrix`](crate::par_csr::ParCsrMatrix) blocks. Defined in `fem-linalg`, re-exported at the crate root. |
//! | [`FEM_ASSEMBLY_PARALLEL_MIN_ELEMS`](crate::FEM_ASSEMBLY_PARALLEL_MIN_ELEMS) | `64` | Minimum volume element count on a rank before Rayon parallelizes [`Assembler`](fem_assembly::Assembler) volume bilinear/linear assembly. Defined in `fem-assembly` (enabled by this crate), re-exported at the crate root. |

use std::sync::OnceLock;

/// Environment variable name for [`local_rayon_min`].
pub const FEM_PARALLEL_LOCAL_RAYON_MIN: &str = "FEM_PARALLEL_LOCAL_RAYON_MIN";

const DEFAULT_LOCAL_RAYON_MIN: usize = 4096;

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

static LOCAL_RAYON_MIN: OnceLock<usize> = OnceLock::new();

/// Minimum local size for Rayon-accelerated loops in [`crate::par_vector::ParVector`] and Krylov drivers.
///
/// Ignored on `wasm32` (no Rayon). Overridden by [`FEM_PARALLEL_LOCAL_RAYON_MIN`].
#[inline]
pub fn local_rayon_min() -> usize {
    *LOCAL_RAYON_MIN.get_or_init(|| parse_usize_env(FEM_PARALLEL_LOCAL_RAYON_MIN, DEFAULT_LOCAL_RAYON_MIN))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_rayon_min_default_positive() {
        assert!(local_rayon_min() >= 1);
    }
}
