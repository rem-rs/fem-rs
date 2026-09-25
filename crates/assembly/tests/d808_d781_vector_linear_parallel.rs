//! D781 — **audit pin** for the parallel reduction of the *vector* linear form
//! (`VectorAssembler::assemble_linear`, the H(curl)/H(div) `VectorLinearIntegrator`
//! path).
//!
//! Round 72's D754 found that the *scalar* volume/boundary linear assembly was
//! non-deterministic: `elem_iter().into_par_iter().fold(dense local
//! rhs).reduce(+=)` combines the per-worker partial vectors in a Rayon
//! reduction tree whose shape follows the thread count and the work-stealing
//! schedule, so the same mesh produced different RHS bits from run to run
//! (measured: 5 distinct `sol.gf` hashes in 5 runs of `mfem_ex26_geom_mg`).  The
//! fix is `elem_block_len`-sized blocks collected in element order + one stable
//! sort by DOF, which restores the serial loop's association exactly.
//!
//! D781 registers the **same audit for the vector path**, which still uses the
//! pre-D754 shape (`vector_assembler.rs`,
//! `assemble_linear_single_many_with_basis`).  The audit's constructive half is
//! in `tmp/d808/d781_audit.md`; this file is its executable half:
//!
//! * the reference is the **in-crate serial loop** for the same space and
//!   integrators — `VectorAssembler::assemble_linear_nd_canonical`, which
//!   accumulates element by element with the same per-element kernel
//!   (`accumulate_vector_linear_element_blocks`) and the same (empty, for 2-D
//!   and hex/ND1) face-block transform, so for these spaces it *is* the serial
//!   association the parallel path must reproduce bit for bit;
//! * the parallel result is taken under pools of 1/2/4/8 threads, repeatedly,
//!   with a mesh (2048 triangles ⇒ 4096? see below) far above every adaptive
//!   `assembly_parallel_min_elems()` threshold, so the parallel branch is
//!   guaranteed to be the one under test.
//!
//! The test is `#[cfg(feature = "parallel")]`: without the feature the branch
//! under test is not compiled at all (`fem-assembly`'s default features), so
//! run it with `cargo test --release -p fem-assembly --features parallel`.

#![cfg(feature = "parallel")]

use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_assembly::VectorAssembler;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::HCurlSpace;

/// `∫ u·v dx`-style source functional with a fixed vector field, so the
/// element vectors are non-trivial and differ per element.
struct Source(fn(&[f64]) -> Vec<f64>);

impl VectorLinearIntegrator for Source {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let ue = (self.0)(qp.x_phys);
        for i in 0..qp.n_dofs {
            let mut dot = 0.0;
            for c in 0..qp.dim {
                dot += qp.phi_vec[i * qp.dim + c] * ue[c];
            }
            f_elem[i] += qp.weight * dot;
        }
    }
}

fn field(x: &[f64]) -> Vec<f64> {
    // Smooth, non-symmetric, non-polynomial in both components: the element
    // vectors are full-precision reals, so a re-associated sum cannot be
    // expected to agree by accident.
    vec![
        (0.7 + x[0]).sin() * (1.3 + x[1]).cos(),
        (0.4 + x[0] * x[1]).exp() + x[1].sin(),
    ]
}

/// FNV-1a over the bit patterns: a fixed-width fingerprint of a whole RHS.
fn digest(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// D781: the parallel vector linear assembly must be bitwise equal to the
/// serial loop for every thread count and every repeat.
#[test]
fn d808_vector_linear_parallel_is_bitwise_serial() {
    // 16×16 cells, two triangles each = 512 elements: well above every
    // adaptive threshold (max 64, and 8 on machines with ≥ 8 threads).
    let mesh = Mesh::<2>::make_cartesian_2d_tri(16, 16, 1.0, 1.0);
    let n_elems = mesh.n_elements();
    assert!(n_elems >= 512, "fixture too small for the parallel branch: {n_elems}");
    let space = HCurlSpace::new(mesh, 1);
    let src = Source(field);
    let integrators: [&dyn VectorLinearIntegrator; 1] = [&src];
    let qo = 4u8;

    // Serial reference: the in-crate element-ordered loop (public canonical
    // entry; 2-D ND1 has no face blocks, so it is the same association as the
    // default path's serial branch).
    let serial = VectorAssembler::assemble_linear_nd_canonical(&space, &integrators, qo);
    assert!(
        serial.iter().any(|&v| v.abs() > 1e-6),
        "degenerate fixture: the load vector is (numerically) zero"
    );
    let serial_digest = digest(&serial);

    let mut digests = Vec::new();
    for threads in [1usize, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        for rep in 0..4 {
            let par = pool.install(|| VectorAssembler::assemble_linear(&space, &integrators, qo));
            assert_eq!(par.len(), serial.len());
            let d = digest(&par);
            digests.push((threads, rep, d));
            let mut worst = (0usize, 0.0f64);
            for i in 0..serial.len() {
                let diff = (par[i] - serial[i]).abs();
                if diff > worst.1 {
                    worst = (i, diff);
                }
                assert_eq!(
                    par[i].to_bits(),
                    serial[i].to_bits(),
                    "dof {i} (threads = {threads}, rep = {rep}): parallel {:.17e} vs serial \
                     {:.17e} — the parallel vector linear assembly is not the serial \
                     association (D781/D754 family)",
                    par[i],
                    serial[i]
                );
            }
            let _ = worst;
        }
    }
    let distinct: std::collections::BTreeSet<u64> = digests.iter().map(|d| d.2).collect();
    println!(
        "D808-2 vector linear parallel audit: n_dofs={}, elems={}, serial digest {serial_digest:#018x}, \
         {} parallel runs, {} distinct digests",
        serial.len(),
        n_elems,
        digests.len(),
        distinct.len()
    );
    assert_eq!(distinct.len(), 1, "parallel runs produced different RHS values");
    assert_eq!(*distinct.iter().next().unwrap(), serial_digest);
}
