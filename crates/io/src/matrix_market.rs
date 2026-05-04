//! Matrix Market (.mtx) file I/O — thin wrapper around linger's mmio module.
//!
//! Supports reading/writing sparse matrices in the standard Matrix Market
//! coordinate format used by SuiteSparse and NIST.
//!
//! # Example
//! ```rust,ignore
//! use fem_io::matrix_market::{read_matrix_market, write_matrix_market};
//!
//! // Read a matrix
//! let a = read_matrix_market("stiffness.mtx").unwrap();
//!
//! // Write a matrix
//! write_matrix_market("output.mtx", &a).unwrap();
//!
//! // Read / write complex matrix
//! use fem_io::matrix_market::{read_matrix_market_complex, write_matrix_market_complex};
//! let z = read_matrix_market_complex("helmholtz.mtx").unwrap();
//! write_matrix_market_complex("helmholtz_out.mtx", &z).unwrap();
//! ```

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use fem_linalg::complex_csr::{ComplexCoo, ComplexCsr};
use fem_linalg::{CooMatrix, CsrMatrix};
use linger::sparse::{
    CooMatrix as LingerCoo, CsrMatrix as LingerCsr,
    read_matrix_market as linger_read, write_matrix_market as linger_write,
};

pub use linger::MmioError;

// ─── Conversion helpers ───────────────────────────────────────────────────────

fn linger_csr_to_fem(lc: LingerCsr<f64>) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows:   lc.nrows(),
        ncols:   lc.ncols(),
        row_ptr: lc.row_ptr().to_vec(),
        col_idx: lc.col_idx().iter().map(|&c| c as u32).collect(),
        values:  lc.values().to_vec(),
    }
}

fn fem_csr_to_linger(a: &CsrMatrix<f64>) -> LingerCsr<f64> {
    LingerCsr::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Read a Matrix Market `.mtx` file into a `CsrMatrix<f64>`.
///
/// Supports `real general`, `real symmetric`, `integer general`, and `pattern` variants.
pub fn read_matrix_market<P: AsRef<Path>>(path: P) -> Result<CsrMatrix<f64>, MmioError> {
    let lc: LingerCsr<f64> = linger_read(path)?;
    Ok(linger_csr_to_fem(lc))
}

/// Read a Matrix Market `.mtx` file into a `CooMatrix<f64>`.
///
/// Preserves duplicate entries as separate (row, col, val) triplets.
pub fn read_matrix_market_coo<P: AsRef<Path>>(path: P) -> Result<CooMatrix<f64>, MmioError> {
    use linger::sparse::read_matrix_market_coo as linger_read_coo;
    let lc: LingerCoo<f64> = linger_read_coo(path)?;
    let mut coo = CooMatrix::<f64>::new(lc.nrows(), lc.ncols());
    for ((r, c), v) in lc.row_indices().iter().zip(lc.col_indices()).zip(lc.values()) {
        coo.add(*r, *c, *v);
    }
    Ok(coo)
}

/// Write a `CsrMatrix<f64>` to a Matrix Market `.mtx` file.
///
/// Writes in `%%MatrixMarket matrix coordinate real general` format.
pub fn write_matrix_market<P: AsRef<Path>>(path: P, a: &CsrMatrix<f64>) -> Result<(), MmioError> {
    let lc = fem_csr_to_linger(a);
    linger_write(path, &lc)
}

// ─── Complex Matrix Market I/O ────────────────────────────────────────────────

/// Error type for complex Matrix Market I/O.
#[derive(Debug)]
pub enum ComplexMmioError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for ComplexMmioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Format(s) => write!(f, "Matrix Market format error: {s}"),
        }
    }
}

impl std::error::Error for ComplexMmioError {}

impl From<std::io::Error> for ComplexMmioError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

/// Read a complex Matrix Market `.mtx` file into a [`ComplexCsr`].
///
/// Accepts `%%MatrixMarket matrix coordinate complex general` and the
/// symmetric variant.  Data lines must be `row col real imag` (1-based).
pub fn read_matrix_market_complex<P: AsRef<Path>>(path: P) -> Result<ComplexCsr, ComplexMmioError> {
    let f = std::fs::File::open(path)?;
    let reader = BufReader::new(f);
    read_matrix_market_complex_from_reader(reader)
}

/// Write a [`ComplexCsr`] to a Matrix Market `.mtx` file in
/// `%%MatrixMarket matrix coordinate complex general` format.
pub fn write_matrix_market_complex<P: AsRef<Path>>(
    path: P,
    a: &ComplexCsr,
) -> Result<(), ComplexMmioError> {
    let mut f = std::fs::File::create(path)?;
    write_matrix_market_complex_to_writer(a, &mut f)
}

// Internal: parse from any BufRead (also used in tests).
fn read_matrix_market_complex_from_reader<R: BufRead>(
    mut reader: R,
) -> Result<ComplexCsr, ComplexMmioError> {
    let mut line = String::new();
    let mut nnz = 0usize;
    let mut is_symmetric = false;
    let mut header_done = false;
    let mut size_done = false;
    let mut coo: Option<ComplexCoo> = None;
    let mut count = 0usize;

    while reader.read_line(&mut line)? > 0 {
        let s = line.trim();
        if s.is_empty() {
            line.clear();
            continue;
        }

        if s.starts_with('%') {
            if !header_done {
                let lower = s.to_ascii_lowercase();
                if !lower.contains("complex") {
                    return Err(ComplexMmioError::Format(
                        "expected 'complex' field in Matrix Market header".into(),
                    ));
                }
                is_symmetric = lower.contains("symmetric");
                header_done = true;
            }
            line.clear();
            continue;
        }

        if !size_done {
            // Size line: nrows ncols nnz
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.len() < 3 {
                return Err(ComplexMmioError::Format("malformed size line".into()));
            }
            let nrows = toks[0].parse::<usize>().map_err(|e| ComplexMmioError::Format(e.to_string()))?;
            let ncols = toks[1].parse::<usize>().map_err(|e| ComplexMmioError::Format(e.to_string()))?;
            nnz   = toks[2].parse::<usize>().map_err(|e| ComplexMmioError::Format(e.to_string()))?;
            coo = Some(ComplexCoo::new(nrows, ncols));
            size_done = true;
            line.clear();
            continue;
        }

        // Data line: row col re im  (1-based indices)
        let toks: Vec<&str> = s.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(ComplexMmioError::Format(format!("data line too short: {s}")));
        }
        let r  = toks[0].parse::<usize>().map_err(|e| ComplexMmioError::Format(e.to_string()))? - 1;
        let c  = toks[1].parse::<usize>().map_err(|e| ComplexMmioError::Format(e.to_string()))? - 1;
        let re = toks[2].parse::<f64>().map_err(|e| ComplexMmioError::Format(e.to_string()))?;
        let im = toks[3].parse::<f64>().map_err(|e| ComplexMmioError::Format(e.to_string()))?;

        let coo_ref = coo.as_mut().ok_or_else(|| ComplexMmioError::Format("missing size line".into()))?;
        coo_ref.add(r, c, re, im);
        if is_symmetric && r != c {
            // Lower-triangular storage: also add (c, r) with conjugate imaginary.
            coo_ref.add(c, r, re, -im);
        }
        count += 1;
        if count > nnz {
            return Err(ComplexMmioError::Format("more data lines than declared nnz".into()));
        }

        line.clear();
    }

    let coo = coo.ok_or_else(|| ComplexMmioError::Format("empty file".into()))?;
    Ok(coo.into_complex_csr())
}

// Internal: write to any Write.
fn write_matrix_market_complex_to_writer<W: Write>(
    a: &ComplexCsr,
    mut w: W,
) -> Result<(), ComplexMmioError> {
    let nnz: usize = a.row_ptr.last().copied().unwrap_or(0);
    writeln!(w, "%%MatrixMarket matrix coordinate complex general")?;
    writeln!(w, "% Generated by fem-io")?;
    writeln!(w, "{} {} {}", a.nrows, a.ncols, nnz)?;
    for i in 0..a.nrows {
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr] as usize;
            let re = a.re_vals[ptr];
            let im = a.im_vals[ptr];
            writeln!(w, "{} {} {:.17e} {:.17e}", i + 1, j + 1, re, im)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn roundtrip_matrix_market() {
        let a = laplacian_1d(5);
        let tmp = NamedTempFile::new().unwrap();
        write_matrix_market(tmp.path(), &a).unwrap();
        let b = read_matrix_market(tmp.path()).unwrap();

        assert_eq!(b.nrows, a.nrows);
        assert_eq!(b.ncols, a.ncols);
        // Check all values match
        for i in 0..a.nrows {
            for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[p] as usize;
                let val_a = a.values[p];
                // Find same (i, j) in b
                let val_b = (b.row_ptr[i]..b.row_ptr[i + 1])
                    .find(|&q| b.col_idx[q] as usize == j)
                    .map(|q| b.values[q])
                    .expect("entry missing in read-back matrix");
                assert!((val_a - val_b).abs() < 1e-14, "value mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn read_mtx_from_string() {
        let mtx = b"%%MatrixMarket matrix coordinate real general\n3 3 4\n1 1 4.0\n1 2 1.0\n2 1 1.0\n3 3 9.0\n";
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(mtx).unwrap();
        tmp.flush().unwrap();
        let a = read_matrix_market(tmp.path()).unwrap();
        assert_eq!(a.nrows, 3);
        assert_eq!(a.ncols, 3);
    }

    // ─── Complex Matrix Market tests ─────────────────────────────────────────

    fn make_complex_mtx_str(nrows: usize, ncols: usize, entries: &[(usize, usize, f64, f64)]) -> String {
        let mut s = format!(
            "%%MatrixMarket matrix coordinate complex general\n% test\n{nrows} {ncols} {}\n",
            entries.len()
        );
        for (r, c, re, im) in entries {
            s.push_str(&format!("{} {} {re} {im}\n", r + 1, c + 1));
        }
        s
    }

    #[test]
    fn read_complex_general_correct_counts() {
        let s = make_complex_mtx_str(3, 3, &[(0, 0, 1.0, 2.0), (1, 1, 3.0, -1.0), (2, 2, 0.5, 0.5)]);
        let csr = read_matrix_market_complex_from_reader(std::io::Cursor::new(s)).unwrap();
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 3);
        let nnz = *csr.row_ptr.last().unwrap();
        assert_eq!(nnz, 3);
    }

    #[test]
    fn read_complex_values_are_correct() {
        let s = make_complex_mtx_str(2, 2, &[(0, 0, 3.0, -4.0), (1, 0, 1.0, 2.0)]);
        let csr = read_matrix_market_complex_from_reader(std::io::Cursor::new(s)).unwrap();
        // Row 0: one entry at col 0 with re=3, im=-4
        let p0 = csr.row_ptr[0];
        assert_eq!(csr.col_idx[p0], 0);
        assert!((csr.re_vals[p0] - 3.0).abs() < 1e-14);
        assert!((csr.im_vals[p0] - (-4.0)).abs() < 1e-14);
    }

    #[test]
    fn roundtrip_complex_write_then_read() {
        let s = make_complex_mtx_str(3, 3, &[
            (0, 0, 1.0, 2.0),
            (0, 2, -0.5, 0.25),
            (1, 1, 3.0, -1.5),
            (2, 0, 0.0, 1.0),
        ]);
        let csr = read_matrix_market_complex_from_reader(std::io::Cursor::new(s)).unwrap();
        let tmp = NamedTempFile::new().unwrap();
        write_matrix_market_complex(tmp.path(), &csr).unwrap();
        let csr2 = read_matrix_market_complex(tmp.path()).unwrap();
        assert_eq!(csr2.nrows, csr.nrows);
        assert_eq!(csr2.ncols, csr.ncols);
        let nnz  = *csr.row_ptr.last().unwrap();
        let nnz2 = *csr2.row_ptr.last().unwrap();
        assert_eq!(nnz, nnz2);
        // All re/im values should round-trip.
        for (a, b) in csr.re_vals.iter().zip(&csr2.re_vals) {
            assert!((a - b).abs() < 1e-13, "re mismatch: {a} vs {b}");
        }
        for (a, b) in csr.im_vals.iter().zip(&csr2.im_vals) {
            assert!((a - b).abs() < 1e-13, "im mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn read_complex_symmetric_expands_lower_triangle() {
        // Symmetric: only lower triangle stored. Off-diagonal entries should be mirrored.
        let s = "\
%%MatrixMarket matrix coordinate complex symmetric
% symmetric test
3 3 4
1 1 2.0 0.0
2 1 1.0 0.5
3 2 0.5 -0.5
3 3 4.0 0.0
";
        let csr = read_matrix_market_complex_from_reader(std::io::Cursor::new(s)).unwrap();
        // Total nnz should be 4 + 2 off-diagonal mirrors = 6.
        let nnz = *csr.row_ptr.last().unwrap();
        assert_eq!(nnz, 6, "symmetric expansion should yield 6 entries");
    }

    #[test]
    fn read_complex_rejects_real_header() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 1.0 0.0\n";
        let res = read_matrix_market_complex_from_reader(std::io::Cursor::new(s));
        assert!(res.is_err(), "should reject 'real' field as non-complex");
    }
}
