//! d107 / D1041 red-green pin: the parallel weak-divergence kernel
//! ([`fem_parallel::ParMixedAssembler::assemble_hcurl_h1_weak_div`]) must
//! carry the **H(curl) element orientation signs** on its trial columns.
//!
//! Round 106 (round-106 section of `tmp/d105ams/REPORT.md`) measured on the
//! ball-quad `-cr` ring current, with `jr` bitwise-identical on both sides:
//!
//! ```text
//! C++   VectorFEWeakDivergenceIntegrator:  ‖W·jr‖ = 7.6e-17
//! fem   frozen assembly kernel (no signs): ‖W·jr‖ = 7.401e-1
//! fem   round-106 signed local bypass:     ‖W·jr‖ = 1.016e-16
//! fem   d107 signed kernel (this pin):     ‖W·jr‖ = 8.439e-17
//! ```
//!
//! The frozen `fem_assembly::mixed::assemble_hcurl_h1_weak_div` kernel scatters
//! the RAW unsigned element-local trial shapes: its D58 face-block
//! canonicalization is a no-op on hexes, so on any mesh whose element-local
//! edge directions disagree with the global convention (cylinder-hex,
//! ball-quad) the columns of the shared edges are assembled with mismatched
//! directions and `W` stops annihilating discretely divergence-free fields.
//! D107 fixes the parallel entry through the signed kernel
//! `fem_parallel::par_mixed_assembler::assemble_hcurl_h1_weak_div_signed`
//! (the frozen serial twin lives in the D1041-forbidden `crates/assembly` and
//! stays red — which these pins use as their live red reference).
//!
//! Two geometry-keyed meshes with genuinely reversed edges:
//! - `data/cylinder-hex.mesh` (252 straight hexes, mixed node orderings),
//! - `tests/data/ball-quad.mesh` (56 curved hexes, the d103/MFEM tesla mesh,
//!   whose `-cr` ring interpolant is discretely divergence-free by design).

use fem_assembly::mixed::{
    assemble_hcurl_h1_gradient, assemble_hcurl_h1_weak_div as assemble_weak_div_unsigned_frozen,
    HCurlH1WeakDiv,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology as _;
use fem_parallel::par_assembler::permute_vec;
use fem_parallel::par_mixed_assembler::ParMixedAssembler;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::{Comm, WorkerConfig, partition_mesh};
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, H1Space};

/// `current_ring` parameters of the round-106 `-cr` run
/// (`-cr '0 0 -0.2 0 0 0.2 0.2 0.4 1'`): an annular ring of unit current
/// around the z axis segment (0,0,-0.2)→(0,0,0.2) with radii 0.2/0.4.
const CR: [f64; 9] = [0.0, 0.0, -0.2, 0.0, 0.0, 0.2, 0.2, 0.4, 1.0];

/// `current_ring` (tesla.cpp:384-425), verbatim math of the miniapp helper.
fn current_ring(cr: &[f64; 9], x: &[f64]) -> [f64; 3] {
    let a = [cr[3] - cr[0], cr[4] - cr[1], cr[5] - cr[2]];
    let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if h == 0.0 {
        return [0.0; 3];
    }
    let (mut ra, mut rb) = (cr[6], cr[7]);
    if ra > rb {
        std::mem::swap(&mut ra, &mut rb);
    }
    let xu = [x[0] - cr[0], x[1] - cr[1], x[2] - cr[2]];
    let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
    let xu_perp = [
        xu[0] - xa / (h * h) * a[0],
        xu[1] - xa / (h * h) * a[1],
        xu[2] - xa / (h * h) * a[2],
    ];
    let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2]).sqrt();
    if xa >= 0.0 && xa <= h * h && xp >= ra && xp <= rb {
        let ju = [
            (a[1] * xu_perp[2] - a[2] * xu_perp[1]) / h,
            (a[2] * xu_perp[0] - a[0] * xu_perp[2]) / h,
            (a[0] * xu_perp[1] - a[1] * xu_perp[0]) / h,
        ];
        let s = cr[8] / (h * (rb - ra));
        [s * ju[0], s * ju[1], s * ju[2]]
    } else {
        [0.0; 3]
    }
}

/// Geometry-keyed pin meshes with the weak-div quadrature order tesla uses at
/// o1 (`ir_order = typical_order_w(geom_order, hex=true) + 2·order`) and the
/// field order.
const CASES: [(&str, &str, u8, u32); 2] = [
    ("ball-quad", "tests/data/ball-quad.mesh", 7, 1),
    ("cylinder-hex", "../../data/cylinder-hex.mesh", 4, 1),
];

/// True when at least one element of the mesh carries a negative H(curl)
/// orientation sign — the guard that keeps the pin keyed to meshes that
/// actually exercise reversed edges (if a fem-rs change ever canonicalizes
/// them to all-+1, this fires and the pin must be re-keyed).
fn has_reversed_edges(mesh: &fem_mesh::Mesh<3>, order: u8) -> bool {
    let nd = HCurlSpace::new(mesh.clone(), order);
    (0..mesh.n_elements() as u32)
        .map(|e| nd.element_signs(e))
        .flatten()
        .any(|&s| s < 0.0)
}

/// Algebraic sign pin (cylinder-hex, the sign-heavy mesh): with both kernels
/// assembling the same `b(v, u) = -∫ ∇v·u` integrand, the signed weak-div
/// matrix must equal the **negated transpose of the sign-correct gradient
/// kernel** ([`assemble_hcurl_h1_gradient`] applies the H(curl) signs to its
/// rows, the signed kernel to the columns; the hex-only pin mesh has no face
/// blocks, so `W = -Gᵀ` entry-for-entry — measured bitwise at 1 rank).  The
/// unsigned frozen kernel deviates by 9.7e-1 on this mesh (live red
/// reference), so the identity doubles as the sign red/green.
///
/// Runs at 1 and 2 ranks (the partition/sign-correction layering of
/// [`fem_parallel::par_mixed_assembler::permute_rect_csr`] is part of the
/// identity: `want = -G(c,d)·s_nd(c)·s_h1(d)` through both partitions).
#[test]
fn weak_div_kernel_equals_negated_gradient_transpose_on_reversed_edges() {
    let (name, rel, qo, order) = ("cylinder-hex", "../../data/cylinder-hex.mesh", 4u8, 1u32);
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{name}: 3-D mesh"));
    assert!(
        has_reversed_edges(&mesh, order as u8),
        "{name}: expected at least one reversed edge (pin geometry key)"
    );
    let mesh = std::sync::Arc::new(mesh);
    for n_ranks in [1usize, 2] {
        let mesh = mesh.clone();
        ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm: Comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local = pmesh.local_mesh().clone();
            let h1 = ParallelFESpace::new(H1Space::new(local.clone(), order as u8), &pmesh, comm.clone());
            let nd = ParallelFESpace::new(HCurlSpace::new(local.clone(), order as u8), &pmesh, comm.clone());

            // Green: the fixed parallel kernel (owned H1 rows × local ND
            // cols, partition order, partition sign corrections applied).
            let integ = HCurlH1WeakDiv::new(1.0_f64);
            let w = ParMixedAssembler::assemble_hcurl_h1_weak_div(&h1, &nd, qo);
            // Reference: the serial sign-correct gradient, canonical order.
            let g = assemble_hcurl_h1_gradient(nd.local_space(), h1.local_space(), qo);
            // Red: the frozen unsigned kernel, canonical order.
            let w_red = assemble_weak_div_unsigned_frozen(
                h1.local_space(),
                nd.local_space(),
                &[&integ],
                qo,
            );

            let h1p = h1.dof_partition();
            let ndp = nd.dof_partition();
            let n_h1 = h1.local_space().n_dofs();
            let n_nd = nd.local_space().n_dofs();
            assert_eq!(w.nrows, h1p.n_owned_dofs);
            assert_eq!(w.ncols, ndp.n_total_dofs());
            assert_eq!(g.nrows, n_nd);
            assert_eq!(g.ncols, n_h1);

            let mut max_dev = 0.0_f64;      // signed kernel vs -Gᵀ
            let mut max_dev_red = 0.0_f64;  // frozen kernel vs -Gᵀ (canonical)
            for d in 0..n_h1 as u32 {
                let p = h1p.permute_dof(d) as usize;
                if p >= h1p.n_owned_dofs {
                    continue; // ghost H1 row — deliberately dropped
                }
                let sr = h1p.sign_correction(d);
                for c in 0..n_nd as u32 {
                    let q = ndp.permute_dof(c) as usize;
                    let want = -g.get(c as usize, d as usize) * ndp.sign_correction(c) * sr;
                    max_dev = max_dev.max((w.get(p, q) - want).abs());
                    max_dev_red =
                        max_dev_red.max((w_red.get(d as usize, c as usize) - want).abs());
                }
            }
            assert!(
                max_dev < 1e-12,
                "{name} ({} ranks): signed weak-div vs -Gᵀ max dev {max_dev:.3e}",
                comm.size()
            );
            assert!(
                max_dev_red > 1e-6,
                "{name} ({} ranks): frozen kernel dev only {max_dev_red:.3e} — the \
                 mesh no longer exercises reversed H(curl) edges, re-key the pin",
                comm.size()
            );
        });
    }
}

/// Physics half of the pin (ball-quad, the MFEM tesla sample mesh): the ND
/// interpolant `jr` of the round-106 ring current is discretely
/// divergence-free in the correct pairing — `‖W·jr‖ ≈ 0` (C++ 7.6e-17,
/// round-106 bypass 1.016e-16, fixed kernel 8.439e-17) — while the unsigned
/// frozen kernel gives `O(‖jr‖)` (7.401e-1).  Single rank, tesla's own
/// quadrature order.
///
/// (The same `W·jr` on cylinder-hex is NOT an annihilation property — that
/// mesh does not make the ring interpolant discretely divergence-free; its
/// sign coverage lives in the algebraic pin above.)
#[test]
fn weak_div_kernel_annihilates_ring_current_interpolant() {
    let (name, rel, qo, order) = ("ball-quad", "tests/data/ball-quad.mesh", 7u8, 1u32);
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{name}: 3-D mesh"));
    let mesh = std::sync::Arc::new(mesh);
    ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm: Comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let local = pmesh.local_mesh().clone();
        let h1 = ParallelFESpace::new(H1Space::new(local.clone(), order as u8), &pmesh, comm.clone());
        let nd = ParallelFESpace::new(HCurlSpace::new(local.clone(), order as u8), &pmesh, comm.clone());

        // jr = ProjectCoeff(current_ring): the ND nodal-functionals
        // interpolation, canonical (DofManager) dof order.
        let jr_local = nd
            .local_space()
            .interpolate_vector(&|x: &[f64]| {
                let v = current_ring(&CR, x);
                vec![v[0], v[1], v[2]]
            })
            .as_slice()
            .to_vec();
        let jr2: f64 = jr_local.iter().map(|v| v * v).sum();
        let jr_norm = jr2.sqrt();
        assert!(
            (jr_norm - 2.946275507539475e0).abs() < 5e-15,
            "{name}: ring interpolant norm {jr_norm:.16e} vs the round-106 probe \
             2.946275507539475e0"
        );

        // Green: the fixed kernel, action on the partition-permuted jr.
        let w = ParMixedAssembler::assemble_hcurl_h1_weak_div(&h1, &nd, qo);
        let jr_par = permute_vec(&jr_local, nd.dof_partition());
        let mut x_div = vec![0.0_f64; w.nrows];
        w.spmv(&jr_par, &mut x_div);
        let g2: f64 = x_div.iter().map(|v| v * v).sum();
        let green = comm.allreduce_sum_f64(g2).sqrt();

        // Red: the frozen unsigned kernel on the canonical jr.
        let integ = HCurlH1WeakDiv::new(1.0_f64);
        let w_red = assemble_weak_div_unsigned_frozen(
            h1.local_space(),
            nd.local_space(),
            &[&integ],
            qo,
        );
        let mut x_red = vec![0.0_f64; w_red.nrows];
        w_red.spmv(&jr_local, &mut x_red);
        let r2: f64 = x_red.iter().map(|v| v * v).sum();
        let red = r2.sqrt();

        println!("{name}: ‖jr‖ = {jr_norm:.6e}  ‖W_signed·jr‖ = {green:.3e}  \
                  ‖W_unsigned·jr‖ = {red:.3e}");
        assert!(
            green < 1e-12,
            "{name}: signed weak-div must annihilate the ring interpolant \
             (got {green:.3e}; C++ 7.6e-17, round-106 bypass 1.016e-16, \
             d107 kernel 8.439e-17)"
        );
        assert!(
            red > 1e-3,
            "{name}: the unsigned frozen kernel no longer pollutes the RHS \
             ({red:.3e}) — the mesh lost its reversed edges, re-key the pin"
        );
    });
}
