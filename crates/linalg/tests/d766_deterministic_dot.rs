//! D766/D754 — `fem_linalg::Vector::dot` must be **bitwise reproducible**.
//!
//! ## The defect (red evidence, reproducible on the pre-fix code)
//!
//! With the `parallel` feature and `n >= 4096`, `Vector::dot` used to be
//! `a.par_iter().zip(b.par_iter()).map(..).sum::<f64>()`.  Rayon's `sum` splits
//! the range by the number of threads and by whatever the work-stealing
//! splitter decides at run time, then combines the partial sums in that
//! (thread-count- and schedule-dependent) tree shape.  Floating-point addition
//! is not associative, so the same input produced different last bits:
//!
//! * `mfem_ex26_geom_mg -m data/inline-hex.mesh -gr 0 -or 2 -no-vis` wrote
//!   `sol.gf` with **5 distinct sha256 values in 5 runs** (6653 of 274627
//!   entries differing, max relative 2.1e-14) while the same binary with
//!   `RAYON_NUM_THREADS=1` was stable.  (The jitter reaches the solution
//!   through the parallel assembly path; `Vector::dot` itself has no
//!   production consumer today — this suite pins the API so that it cannot
//!   become a second, silent source later.)
//! * On the pre-fix code the thread-count sweep below returns different bits
//!   for different pool sizes (`to_bits()` mismatch), and the `n == 4096`
//!   block-window pin fails because the rayon tree is not the 8-unrolled
//!   serial kernel.
//!
//! ## The fix
//!
//! `dot_f64_parallel` cuts the range into fixed `DOT_REDUCTION_BLOCK = 4096`
//! chunks (boundaries are a pure function of `len`), reduces each chunk with
//! the serial 8-unrolled `dot_f64` on whichever worker picks it up, and
//! combines the chunk partials serially in chunk-index order — the same pattern
//! as `fem_parallel::ParVector`'s `deterministic_dot` (D746).
//!
//! Run with the feature on (the gate must enable it, otherwise every test below
//! is compiled out and the suite is vacuous):
//!
//! ```text
//! cargo test --release -p fem-linalg --features parallel --test d766_deterministic_dot
//! ```

#![cfg(feature = "parallel")]

use fem_linalg::Vector;
use rayon::ThreadPoolBuilder;

/// Deterministic pseudo-random fill: xorshift64 → `[-1, 1]`, scaled by a
/// rotating power of two so that the products have mixed magnitudes and the
/// summation is sensitive to the association order.
fn mixed_magnitude_vec(n: usize, seed: u64) -> Vector<f64> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let scale = (1u64 << (i % 5)) as f64; // 1, 2, 4, 8, 16
            next() * scale
        })
        .collect();
    Vector::from_vec(data)
}

/// The documented **serial** association of `dot_f64`: eight independent
/// accumulators over the 8-unrolled body, combined `s0 + s1 + ... + s7`, then
/// the `n % 8` tail in index order.  Replicated here so the test pins the
/// parallel path against the serial kernel rather than against itself.
fn serial_8unrolled_dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut s = [0.0_f64; 8];
    let end8 = n / 8 * 8;
    let mut i = 0;
    while i < end8 {
        for k in 0..8 {
            s[k] += a[i + k] * b[i + k];
        }
        i += 8;
    }
    let mut sum = ((((((s[0] + s[1]) + s[2]) + s[3]) + s[4]) + s[5]) + s[6]) + s[7];
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// The defect: `dot` returned a different value for the same input when the
/// Rayon pool size changed (the partial-sum tree follows the thread count).
#[test]
fn dot_is_bitwise_stable_across_thread_counts() {
    for &n in &[4096usize, 5000, 12288, 100_000] {
        let a = mixed_magnitude_vec(n, 0xD766_0001 ^ n as u64);
        let b = mixed_magnitude_vec(n, 0xD766_0002 ^ n as u64);
        let mut reference: Option<u64> = None;
        for threads in [1usize, 2, 3, 4, 8] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool");
            let bits = pool.install(|| a.dot(&b)).to_bits();
            match reference {
                None => reference = Some(bits),
                Some(r) => assert_eq!(
                    bits, r,
                    "n = {n}: dot differs between Rayon pool sizes (threads = {threads}): \
                     {} vs {}",
                    f64::from_bits(r),
                    f64::from_bits(bits)
                ),
            }
        }
    }
}

/// The same pool, twenty consecutive calls: no run-to-run drift either.
#[test]
fn dot_is_bitwise_stable_over_repeated_calls() {
    let n = 65_536;
    let a = mixed_magnitude_vec(n, 0xD766_0010);
    let b = mixed_magnitude_vec(n, 0xD766_0011);
    let first = a.dot(&b).to_bits();
    for k in 1..20 {
        assert_eq!(
            a.dot(&b).to_bits(),
            first,
            "call {k} of 20 drifted (n = {n})"
        );
    }
}

/// Block-window pin: below `PAR_VEC_MIN` the serial kernel is used, and **at**
/// `PAR_VEC_MIN` the parallel path is exactly one block, i.e. the same
/// `dot_f64` kernel — so both lengths agree with the serial association
/// computed here independently.  (Above `PAR_VEC_MIN` the association is the
/// blockwise one pinned by `dot_is_the_serial_combination_of_its_fixed_blocks`.)
#[test]
fn dot_matches_the_serial_kernel_on_the_threshold_window() {
    for &n in &[4095usize, 4096] {
        let a = mixed_magnitude_vec(n, 0xD766_0020 ^ n as u64);
        let b = mixed_magnitude_vec(n, 0xD766_0021 ^ n as u64);
        let expected = serial_8unrolled_dot(a.as_slice(), b.as_slice());
        let got = a.dot(&b);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "n = {n}: dot = {got} but the serial 8-unrolled kernel gives {expected}"
        );
    }
}

/// Association pin: the reduction is exactly "fixed 4096-blocks from index 0,
/// each reduced by the serial 8-unrolled kernel, partials added left to right".
/// That is what makes the value a function of the length instead of a function
/// of the Rayon schedule.
#[test]
fn dot_is_the_serial_combination_of_its_fixed_blocks() {
    const BLOCK: usize = 4096;
    for &n in &[4096usize, 5000, 16_384, 20_000, 100_000] {
        let a = mixed_magnitude_vec(n, 0xD766_0030 ^ n as u64);
        let b = mixed_magnitude_vec(n, 0xD766_0031 ^ n as u64);
        let mut expected = 0.0_f64;
        let mut off = 0;
        while off < n {
            let hi = (off + BLOCK).min(n);
            expected += serial_8unrolled_dot(&a.as_slice()[off..hi], &b.as_slice()[off..hi]);
            off = hi;
        }
        assert_eq!(
            a.dot(&b).to_bits(),
            expected.to_bits(),
            "n = {n}: dot = {} but the 4096-block serial association gives {expected}",
            a.dot(&b)
        );
    }
}
