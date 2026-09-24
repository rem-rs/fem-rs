//! D615: the hex IGLL **assembly** leg — the mass matrix of the
//! `RT_FECollection(p, 3, GaussLobatto, IntegratedGLL)` space must be
//! assembled from [`HexRTk::new`] (the IntegratedGLL element — the same
//! element the LOR stack pins), not silently fall through to the
//! GaussLegendre default.
//!
//! 框架因子口径 (frame-factor convention — **D757 re-anchor after D721**):
//! the `[-1,1]³` pull-back the old 1/4 field factor came from is GONE — D721
//! moved the whole hex family (bases, slot rows, quadrature, geometry) onto
//! MFEM's `[0,1]³` unit cube, so a fem-rs IGLL basis function's physical field
//! is now MFEM's physical field **verbatim** (factor 1, formerly 1/4 per D591)
//! and the mass matrix is quadratic in the basis, so **fem-rs's unsigned
//! element matrix = MFEM's × 1** — `FRAME = 1` below, formerly 16.  The
//! collapse is pinned bit-exactly against the same MFEM probe dump (the old
//! `16·M_femrs` equals `M_mfem` to the printed 17 digits), not tolerated.
//!
//! What is compared: MFEM's RAW `VectorFEMassIntegrator::AssembleElementMatrix`
//! output (unsigned, reference slot order — exactly what `BilinearForm` sums
//! with the dof_map signs) against fem-rs's per-element accumulation
//! unmangled per slot: `M_nosign[i][j] = signs[i]·signs[j]·M_local[dofs[i]][dofs[j]]`.
//! The per-element route is required because a raw global read of a SHARED
//! face slot carries the neighbour's contribution too (the interior x-face
//! of the two-element fixture: global (2,2) = el0[2][2] + el1[4][4] — pinned
//! as a bonus check below), which would masquerade as a factor-2 defect.
//!
//! Truth provenance: `tmp/d614/d615_probe.cpp` (compiled in WSL against the
//! MFEM 4.10 tree, `g++ -std=c++17 -O2 -I$HOME/mfem410_ser d615_probe.cpp
//! $HOME/mfem410_ser/libmfem.a -o d615_probe`), dumping the per-element
//! matrices of the IGLL collection + `VectorFEMassIntegrator` on the D591
//! probe meshes for p = 0..2 → `data/d615_hex_igll_mass_mfem.txt`.
//! Methodology (the D235 discipline, same rule both sides): every block
//! integrates with the EXPLICIT `IntRules.Get(CUBE, 20)` rule
//! (`SetIntRule`, 11³ Gauss points); the IGLL mass integrand on a warped
//! trilinear map is rational (`1/det J`), so no polynomial exactness exists
//! there and a same-rule comparison is the only rule-tight option.  The
//! warp is the MILD vertex-5 displacement (1.7, 0.75, 0.1), det J ∈
//! [0.39, 1.0] (`tmp/d614/probe_rule_check2.log`): the D591 interpolate-probe
//! displacement (1.3, 0.4, 1.2) FOLDS element 1 — its mass entries oscillate
//! in sign across rule orders 4..20 (`tmp/d614/probe_rule_check.log`) — and
//! is unusable as mass truth (interpolate parity, D591, is pointwise and
//! unaffected).
//!
//! RED-first note (round 61): without the D615 arm,
//! `vec_ref_elem_with_basis(.., quad_igll = true)` on a hex cell fell through
//! to `vec_ref_elem_choice`'s **GaussLegendre** element — a silent wrong
//! basis (different open modes ⇒ different element matrices; e.g. the p=0
//! variant-0 element-0 (0,0) entry assembled to 16·(GL local) with GL's
//! `V_mfem/4` frame where MFEM's IGLL matrix wants `V_mfem/16` — every
//! diagonal off by exactly 4×).

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::vector_assembler::accumulate_vector_bilinear_element_blocks_with_basis;
use fem_assembly::VectorAssembler;
use fem_linalg::CooMatrix;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::HDivSpace;

/// One probe block: (p, variant, per-element row-major matrices).
fn parse_fixture_matrices(text: &str) -> Vec<(usize, usize, Vec<Vec<f64>>)> {
    let mut blocks = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.first() == Some(&"HEXIGLLMASS") {
            let p: usize = tokens[1].trim_start_matches("p=").parse().unwrap();
            let variant: usize = tokens[2].trim_start_matches("variant=").parse().unwrap();
            let mut elems = Vec::new();
            loop {
                let line = match lines.peek() {
                    Some(l) => *l,
                    None => break,
                };
                let tk: Vec<&str> = line.split_whitespace().collect();
                if tk.first() != Some(&"elem") {
                    break;
                }
                let n: usize = tk[2].trim_start_matches("n=").parse().unwrap();
                lines.next(); // consume the "elem e n=" header
                let mut mat = Vec::with_capacity(n * n);
                for _ in 0..n {
                    let row: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                    assert_eq!(row.len(), n, "row width");
                    for r in row {
                        mat.push(r.parse::<f64>().unwrap());
                    }
                }
                elems.push(mat);
            }
            blocks.push((p, variant, elems));
        }
    }
    blocks
}

/// The probe's two meshes (variant 0 = two unit hexes in [0,2]×[0,1]×[0,1];
/// variant 1 = node 5 = VTX (2,1,0) displaced to (1.7, 0.75, 0.1) — the mild
/// warp, det J ∈ [0.39, 1.0]).
fn probe_mesh(variant: usize) -> Mesh<3> {
    let mut mesh =
        Mesh::<3>::make_cartesian_3d(2, 1, 1, fem_mesh::element_type::ElementType::Hex8, 2.0, 1.0, 1.0, true);
    if variant == 1 {
        let off = 5 * 3;
        mesh.coords[off] = 1.7;
        mesh.coords[off + 1] = 0.75;
        mesh.coords[off + 2] = 0.1;
    }
    mesh
}

/// Core acceptance: for every (p, variant, element) block, the IGLL space's
/// per-element mass accumulation reproduces MFEM's raw element matrix × 1
/// (the frame factor asserted explicitly; D757 collapsed it from 16 to 1 with
/// the D721 `[0,1]³` flip).  Pin: relative deviation ≤ 4e-13 (the entries span
/// ~16 orders of magnitude down to 1e-18 off-diagonal dust; measured max
/// printed).
#[test]
fn d615_hex_igll_mass_matrix_matches_mfem() {
    let blocks = parse_fixture_matrices(include_str!("data/d615_hex_igll_mass_mfem.txt"));
    assert_eq!(blocks.len(), 2 * 3, "2 mesh variants x p = 0..2");

    const FRAME: f64 = 1.0; // D757/D721: fem-rs IGLL basis = MFEM physical, verbatim
    let mut checked = 0usize;
    let mut max_rel = 0.0f64;

    for (p, variant, expect) in &blocks {
        let mesh = probe_mesh(*variant);
        let space = HDivSpace::new_gauss_lobatto_integrated_gll(mesh, *p as u8);
        assert!(space.quad_integrated_gll(), "IGLL variant flag must be live (D591)");

        // The per-element IGLL accumulation (the same pipeline
        // `assemble_bilinear_quad_igll` runs per element) at the SAME rule
        // the probe used — `IntRules.Get(CUBE, 20)` on both sides.
        for (e, want) in expect.iter().enumerate() {
            let mut coo = CooMatrix::<f64>::new(space.n_dofs(), space.n_dofs());
            accumulate_vector_bilinear_element_blocks_with_basis(
                &space,
                e as u32,
                &[&VectorMassIntegrator { alpha: 1.0 }],
                20,
                &mut coo,
                &[],
                true,
            );
            let m = coo.into_csr();

            let dofs = space.element_dofs(e as u32);
            let signs = space.element_signs(e as u32);
            assert_eq!(dofs.len() * dofs.len(), want.len(), "p={p} variant={variant} elem={e} size");
            for (idx, &w) in want.iter().enumerate() {
                let i = idx / dofs.len();
                let j = idx % dofs.len();
                // Unmangle the signed local matrix back to the raw local one
                // (the accumulation carries sign_i·sign_j).
                let got = FRAME * signs[i] * signs[j] * m.get(dofs[i] as usize, dofs[j] as usize);
                let rel = (got - w).abs() / w.abs().max(1e-2);
                max_rel = max_rel.max(rel);
                assert!(
                    rel <= 4e-13,
                    "p={p} variant={variant} elem={e} [{i}][{j}]: got {got}, want {w} (rel {rel:.2e})"
                );
                checked += 1;
            }
        }
    }
    println!(
        "d615: checked {checked} mass entries against the MFEM probe, max rel dev {max_rel:.2e}"
    );
    assert!(checked > 3000, "fixture coverage regressed: {checked}");
}

/// The frame factor is a property of the IGLL basis, not of the tolerance:
/// `FRAME = 1` (D757, after D721's `[0,1]³` flip — formerly 16), i.e. on the
/// unit cube at p = 0 the fem-rs unsigned (0,0) entry must equal MFEM's 1/3
/// verbatim, and the GLOBAL assembly must agree with the local entries away
/// from the shared face while the shared-face slot accumulates BOTH elements
/// (the honest global-matrix semantics).
#[test]
fn d615_hex_igll_frame_factor_and_global_sanity() {
    let mesh = probe_mesh(0);
    let space = HDivSpace::new_gauss_lobatto_integrated_gll(mesh, 0);
    let m = VectorAssembler::assemble_bilinear_quad_igll(
        &space,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        6,
    );
    let dofs = space.element_dofs(0);
    let signs = space.element_signs(0);

    // MFEM probe (p=0 variant=0 elem=0): elmat[0][0] = 0.3333333333333332.
    let mfem_00 = 0.3333333333333332_f64;
    let nosign_00 = signs[0] * signs[0] * m.get(dofs[0] as usize, dofs[0] as usize);
    println!(
        "d615: fem-rs global (0,0) = {nosign_00:.17e}, MFEM = {mfem_00:.17e}"
    );
    // dof 0 = the z- boundary face — not shared, so the global entry IS the
    // element's own contribution.
    assert!(
        (nosign_00 - mfem_00).abs() <= 4e-17,
        "frame factor 1 not honoured: {nosign_00} vs {mfem_00}"
    );

    // The shared interior x-face slot (2): the global entry accumulates both
    // elements' contributions (el0[2][2] + el1[4][4] = 2 × 1/48) — pinning
    // the honest global semantics that a naive local read would misread as
    // a factor-2 defect.
    let shared = m.get(dofs[2] as usize, dofs[2] as usize);
    println!("d615: shared-face global (2,2) = {shared:.17e} (= 2 × 1/48)");
    // D757: the absolute pin re-anchors with the frame factor.  Pre-flip the
    // entry lived at 2·(1/3)/16 = 1/24 and the pin was 8e-17 ≈ 1.9e-15
    // relative; after D721 the same entry is 16× larger (2/3), so the
    // ulp-equivalent budget is 1.9e-15 · 2/3 = 1.3e-15.  The check still
    // separates the factor-2 global semantics from a single-element (factor-1)
    // read by ~15 orders of magnitude.
    assert!(
        (shared - 2.0 * mfem_00).abs() <= 1.3e-15,
        "shared-face accumulation broken: {shared} vs 2×{mfem_00}"
    );
}
