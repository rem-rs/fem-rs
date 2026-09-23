//! D638: the HCurl hex IGLL **assembly** leg — the mass matrix of the
//! `ND_FECollection(p, 3, GaussLobatto, IntegratedGLL)` space must be
//! assembled from [`fem_element::nedelec::HexNDk::new_integrated_gll`] (the
//! IntegratedGLL element — the LOR basis pair of MFEM `fem/lor/lor.hpp`), not
//! silently fall through to the GaussLegendre default.
//!
//! 框架因子口径 (frame-factor convention, pinned — not tolerated away):
//! unlike the HDiv RTk IGLL basis (fem-rs physical field = MFEM/4, mass/16 —
//! D615), the ND hex IGLL element carries the SAME frame as the GaussLegendre
//! nodal element: the element-level dump test
//! `hex_ndk::ndk_integrated_gll_matches_mfem_dump` pins
//! `V_ref_femrs = V_ref_mfem/2`, exactly the factor the `[-1,1]` covariant
//! pull-back cancels — the PHYSICAL fem-rs field equals MFEM's, and with the
//! measure identity `dξ·det J_femrs = dξ̂·det J_mfem` the element mass matrix
//! matches with **FRAME = 1** (asserted per entry below).
//!
//! What is compared: MFEM's RAW `VectorFEMassIntegrator::AssembleElementMatrix`
//! output (unsigned, reference slot order — exactly what `BilinearForm` sums
//! with the dof_map signs) against fem-rs's per-element accumulation
//! unmangled per slot: `M_nosign[i][j] = signs[i]·signs[j]·M_local[dofs[i]][dofs[j]]`.
//!
//! Truth provenance: `tmp/d635b/d638_probe.cpp` (compiled in WSL against the
//! MFEM 4.10 tree, `g++ -std=c++17 -O2 -I$HOME/mfem410_ser d638_probe.cpp
//! $HOME/mfem410_ser/libmfem.a -o d638_probe`), dumping the per-element
//! matrices of the IGLL collection + `VectorFEMassIntegrator` on the D615
//! probe meshes for p = 1..3 → `data/d638_hcurl_hex_igll_mass_mfem.txt`.
//! Methodology (the D235 discipline, same rule both sides): every block
//! integrates with the EXPLICIT `IntRules.Get(CUBE, 20)` rule
//! (`SetIntRule`, 11³ Gauss points); the warp is the MILD vertex-5
//! displacement (1.7, 0.75, 0.1), det J ∈ [0.39, 1.0].
//!
//! RED-first note (round 62): without the D638 arm,
//! `vec_ref_elem_with_basis(.., quad_igll = true)` on a hex cell fell through
//! to `vec_ref_elem_choice`'s **GaussLegendre** `HexNDk::new` — a silent
//! wrong basis for p ≥ 2 (different open modes ⇒ different element
//! matrices); for p = 1 the integrated and Gauss-Legendre open bases coincide
//! (both the constant `1` in MFEM's normalisation), so p = 1 pins the frame
//! while p ≥ 2 carries the RED→GREEN.

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::vector_assembler::accumulate_vector_bilinear_element_blocks_with_basis;
use fem_linalg::CooMatrix;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::HCurlSpace;

/// One probe block: (p, variant, per-element row-major matrices).
fn parse_fixture_matrices(text: &str) -> Vec<(usize, usize, Vec<Vec<f64>>)> {
    let mut blocks = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.first() == Some(&"NDHEXIGLLMASS") {
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
    let mut mesh = Mesh::<3>::make_cartesian_3d(
        2,
        1,
        1,
        fem_mesh::element_type::ElementType::Hex8,
        2.0,
        1.0,
        1.0,
        true,
    );
    if variant == 1 {
        let off = 5 * 3;
        mesh.coords[off] = 1.7;
        mesh.coords[off + 1] = 0.75;
        mesh.coords[off + 2] = 0.1;
    }
    mesh
}

/// Core acceptance: for every (p, variant, element) block, the HCurl hex IGLL
/// space's per-element mass accumulation reproduces MFEM's raw element matrix
/// at frame 1 (asserted explicitly).  Pin: relative deviation ≤ 4e-13 (the
/// entries span ~16+ orders of magnitude down to 1e-36 off-diagonal dust;
/// measured max printed).
#[test]
fn d638_hcurl_hex_igll_mass_matrix_matches_mfem() {
    let blocks = parse_fixture_matrices(include_str!(
        "data/d638_hcurl_hex_igll_mass_mfem.txt"
    ));
    assert_eq!(blocks.len(), 2 * 3, "2 mesh variants x p = 1..3");

    const FRAME: f64 = 1.0; // ND hex IGLL: same frame as the GL nodal element
    let mut checked = 0usize;
    let mut max_rel = 0.0f64;

    for (p, variant, expect) in &blocks {
        let mesh = probe_mesh(*variant);
        let space = HCurlSpace::new_gauss_lobatto_integrated_gll(mesh, *p as u8);
        assert!(
            space.quad_integrated_gll(),
            "IGLL variant flag must be live (D368/D591)"
        );

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
            assert_eq!(
                dofs.len() * dofs.len(),
                want.len(),
                "p={p} variant={variant} elem={e} size"
            );
            for (idx, &w) in want.iter().enumerate() {
                let i = idx / dofs.len();
                let j = idx % dofs.len();
                // Unmangle the signed local matrix back to the raw local one
                // (the accumulation carries sign_i·sign_j).
                let got =
                    FRAME * signs[i] * signs[j] * m.get(dofs[i] as usize, dofs[j] as usize);
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
        "d638: checked {checked} ND IGLL mass entries against the MFEM probe, max rel dev {max_rel:.2e}"
    );
    assert!(checked > 50000, "fixture coverage regressed: {checked}");
}
