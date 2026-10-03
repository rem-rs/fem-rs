//! d110 / D1102 bit-identical pin for the WG Maxwell volume kernel
//! (`fem_assembly::wg` — `weak_curl_matrix` via `assemble_wg_maxwell`).
//!
//! # The debt
//!
//! The 3-D arm of `weak_curl_matrix` recomputed the physical curl component
//! `(J·curl̂_i)_sc/detJ` **inside the (i, j) double loop** — once per flux
//! column `j` although it depends only on the ND dof `i` and the component
//! `sc = j / n_ss` (recomputed `n_ss` times per (i, sc) pair: 4× at ND2, 10×
//! at ND3).  D1102 is a pure micro-refactor: hoist the three components per
//! dof out of the column loop.  Zero-dead-code discipline + the family's
//! "don't move arithmetic silently" rule require the refactor to be
//! **bit-identical**, and that is what this pin locks.
//!
//! # Pin
//!
//! FNV-1a 64-bit checksums over the full CSR content (row index, column
//! index, and the exact `f64::to_bits` of every value) of `assemble_wg_
//! maxwell` on four mesh/order shapes that cover both arms of the hoisted
//! loop (2-D scalar control + 3-D at ND1/P₀, ND2/[P₁]³, ND3/[P₂]³ — the
//! P₀-flux shape never enters the hoisted inner loop, the P1/P2 flux shapes
//! do, 4× and 10× respectively) and both penalty settings (volume only and
//! volume+faces).  The constants were captured from the **pre-change**
//! kernel; any floating-point change anywhere in the WG volume or face path
//! breaks them.

use fem_assembly::assemble_wg_maxwell;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::HCurlSpace;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// FNV-1a over the exact CSR bit pattern (structure + `to_bits` values).
fn csr_checksum(k: &CsrMatrix<f64>) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    h = fnv1a(h, &(k.nrows as u64).to_le_bytes());
    h = fnv1a(h, &(k.ncols as u64).to_le_bytes());
    for i in 0..k.nrows {
        for p in k.row_ptr[i]..k.row_ptr[i + 1] {
            h = fnv1a(h, &(i as u32).to_le_bytes());
            h = fnv1a(h, &k.col_idx[p].to_le_bytes());
            h = fnv1a(h, &k.values[p].to_bits().to_le_bytes());
        }
    }
    h
}

// Captured from the pre-change kernel (HEAD efab4acf, D1080 state) and
// verified identical after the D1102 hoist (the same run printed both).
const CHECKSUMS: &[(&str, u64)] = &[
    ("tri3 o1 q3 p0", 0xff45c0a0973d9e16),
    ("tri3 o1 q3 p10", 0x20c913a4d3fdacdb),
    ("tet2 o1 q3 p0", 0xd4abd7dd66d0eeb1),
    ("tet2 o1 q3 p10", 0xe087ebdbb0003d1d),
    ("tet1 o2 q4 p0", 0x5f00cefb1a2a9cf0),
    ("tet1 o2 q4 p10", 0xd3336ce8640171b7),
    ("tet1 o3 q5 p0", 0xcf5d0b1722b440a0),
    ("tet1 o3 q5 p10", 0x25168a9294d49214),
];

#[test]
fn d1102_wg_maxwell_bit_identical_checksums() {
    let mesh_tri = Mesh::<2>::unit_square_tri(3);
    let mesh_tet = Mesh::<3>::unit_cube_tet(2);
    let mesh_tet1 = Mesh::<3>::unit_cube_tet(1);

    let shapes: Vec<(&str, CsrMatrix<f64>)> = vec![
        (
            "tri3 o1 q3 p0",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tri.clone(), 1), 3, 0.0, &[]).0,
        ),
        (
            "tri3 o1 q3 p10",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tri.clone(), 1), 3, 10.0, &[]).0,
        ),
        (
            "tet2 o1 q3 p0",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tet.clone(), 1), 3, 0.0, &[]).0,
        ),
        (
            "tet2 o1 q3 p10",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tet.clone(), 1), 3, 10.0, &[]).0,
        ),
        (
            "tet1 o2 q4 p0",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tet1.clone(), 2), 4, 0.0, &[]).0,
        ),
        (
            "tet1 o2 q4 p10",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tet1.clone(), 2), 4, 10.0, &[]).0,
        ),
        (
            "tet1 o3 q5 p0",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tet1.clone(), 3), 5, 0.0, &[]).0,
        ),
        (
            "tet1 o3 q5 p10",
            assemble_wg_maxwell(&HCurlSpace::new(mesh_tet1, 3), 5, 10.0, &[]).0,
        ),
    ];

    for (label, k) in &shapes {
        println!("{label}: {:016x}", csr_checksum(k));
    }
    for (label, k) in &shapes {
        let c = csr_checksum(k);
        match CHECKSUMS.iter().find(|(l, _)| *l == *label) {
            Some((_, expected)) => assert_eq!(
                c, *expected,
                "{label}: WG Maxwell bit pattern changed (D1102: the curl \
                 hoist must be bit-identical)"
            ),
            None => panic!("CHECKSUMS not filled in yet — captured {label}: {c:016x}"),
        }
    }
}
