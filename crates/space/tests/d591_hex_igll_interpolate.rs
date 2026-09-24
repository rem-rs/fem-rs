//! D591: `HDivSpace::interpolate_vector` consumes the hex IGLL integrated
//! functionals — the 3-D analogue of the D368 quad leg.
//!
//! `HDivSpace::new_gauss_lobatto_integrated_gll` on a hex mesh now builds
//! MFEM's `RT_FECollection(p, 3, GaussLobatto, IntegratedGLL)` variant
//! (identical dof numbering/signs — the maps are variant-independent, round
//! 40) and `interpolate_vector` serves MFEM `RT_HexahedronElement::Project`'s
//! dispatch to `ProjectIntegrated` (`fe_rt.hpp:126-131`): every DOF is the
//! sub-cell face-flux integral `Σ w·f(x(ξ))·(adj(J)·t)` over its GLL
//! sub-cell face, via `HexRTk::new(p).integrated_functionals()` (the round-58
//! table pinned by `d577_hex_igll_functionals_mfem_truth`).
//!
//! 框架因子口径 (frame-factor convention), pinned by `d591_hex_igll_frame_factor_pinned`
//! below: the DOF **values** need NO factor.  Through the `[-1,1]` pull-back
//! `adj(J_ξ) = adj(J_mfem)/4` pointwise (D577 header) while the `[-1,1]`
//! sub-cell quadrature weights are 4× the `[0,1]` ones, so the consumption
//! sum reproduces MFEM's DOF value exactly.  The standing `V_mfem/16`
//! reference-basis frame debt (hex_rtk module docs) lives in field
//! **evaluation**: reconstructing the field from these DOFs with the fem-rs
//! IGLL basis + contravariant Piola yields MFEM's physical field times 1/4
//! (asserted at full precision below — NOT absorbed into a tolerance).
//!
//! Truth provenance: `tmp/d603/hex_igll_project_probe.cpp` (archived),
//! compiled in WSL against the MFEM 4.10 tree
//! (`g++ -std=c++17 -O2 -I$HOME/mfem410_ser hex_igll_project_probe.cpp
//!   $HOME/mfem410_ser/libmfem.a`), dumping `GridFunction::ProjectCoefficient`
//! per-element SIGNED local dof vectors for the IGLL collection on
//!   variant 0: MakeCartesian3D(2,1,1) over [0,2]x[0,1]x[0,1] (sfc_ordering
//!              defaulted to true, matched here),
//!   variant 1: the same mesh with vertex 5 (VTX(2,1,0) = (2,1,0)) displaced
//!              to (1.3, 0.4, 1.2) — trilinear map, J varies inside the
//!              sub-faces,
//! for p = 0..=2 and fields f0 = (1,0,0) [constant], f1 = (0.5,-1,2),
//! f2 = (x², y(x+0.5), z³): `data/d591_hex_igll_project_mfem.txt`.
//! The p=1 variant-0 f0 block reproduces the D577 `HEX_VALS_K1_F0` pattern
//! (0.25 sub-cell integrals), cross-linking the two truth sources.

use fem_element::raviart_thomas::HexRTk;
use fem_element::reference::VectorReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::HDivSpace;

/// One probe block: (p, variant, field, per-element signed local dof vectors).
fn parse_fixture(text: &str) -> Vec<(usize, usize, usize, Vec<Vec<f64>>)> {
    let mut blocks = Vec::new();
    let mut cur: Option<(usize, usize, usize)> = None;
    let mut elems: Vec<Vec<f64>> = Vec::new();
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens[0] == "HEXIGLL" {
            if let Some(h) = cur.take() {
                blocks.push((h.0, h.1, h.2, std::mem::take(&mut elems)));
            }
            // "HEXIGLL p=%d variant=%d field=%d"
            let p: usize = tokens[1].trim_start_matches("p=").parse().unwrap();
            let variant: usize = tokens[2].trim_start_matches("variant=").parse().unwrap();
            let field: usize = tokens[3].trim_start_matches("field=").parse().unwrap();
            cur = Some((p, variant, field));
        } else if tokens[0] == "elem" {
            // "elem %d n=%d : v1 v2 ..."
            assert_eq!(tokens[3], ":");
            elems.push(tokens[4..].iter().map(|t| t.parse().unwrap()).collect());
        }
    }
    if let Some(h) = cur.take() {
        blocks.push((h.0, h.1, h.2, elems));
    }
    blocks
}

/// The probe's fields.
fn probe_field(f: usize, x: &[f64]) -> Vec<f64> {
    match f {
        0 => vec![1.0, 0.0, 0.0],
        1 => vec![0.5, -1.0, 2.0],
        _ => vec![x[0] * x[0], x[1] * (x[0] + 0.5), x[2] * x[2] * x[2]],
    }
}

/// The probe's two meshes: variant 0 = two unit hexes in [0,2]x[0,1]x[0,1];
/// variant 1 = the same with node 5 (= VTX(2,1,0) = (2,1,0)) displaced to
/// (1.3, 0.4, 1.2).  `make_cartesian_3d` copies MFEM's VTX numbering, so node
/// 5 is the same physical vertex on both sides.  `sfc_ordering = true`
/// matches the probe's `MakeCartesian3D` default, so element order matches.
fn probe_mesh(variant: usize) -> Mesh<3> {
    let mut mesh =
        Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 2.0, 1.0, 1.0, true);
    if variant == 1 {
        let off = 5 * 3;
        mesh.coords[off] = 1.3;
        mesh.coords[off + 1] = 0.4;
        mesh.coords[off + 2] = 1.2;
    }
    mesh
}

/// Core acceptance: for every (p, mesh variant, field) block, the hex IGLL
/// space's `interpolate_vector` reproduces MFEM's per-element signed local
/// ProjectIntegrated dof vectors.  Pin: relative deviation ≤ 4e-15 (measured
/// max printed; ~2 ulp on the unit cube — D577 measured ≤ 1.2e-16 there —
/// growing to ~1e-15 on the distorted trilinear mesh where MFEM's `[0,1]`
/// sub-cell sampling and fem-rs' `[-1,1]` one differ by coordinate ulps
/// amplified by the pointwise adj(J) variation).
#[test]
fn d591_hex_igll_interpolate_matches_mfem_project_integrated() {
    let blocks = parse_fixture(include_str!("data/d591_hex_igll_project_mfem.txt"));
    assert_eq!(blocks.len(), 2 * 3 * 3, "2 mesh variants x p=0..2 x 3 fields");

    let mut checked = 0usize;
    let mut max_rel = 0.0f64;
    for (p, variant, field, expect) in &blocks {
        let mesh = probe_mesh(*variant);
        let space = HDivSpace::new_gauss_lobatto_integrated_gll(mesh, *p as u8);
        // the variant flag must be live for hexes (D591)
        assert!(space.quad_integrated_gll(), "p={p} variant={variant}");
        // ...and the space is still the plain RT hex dof layout
        let m = p + 1;
        assert_eq!(
            space.element_dofs(0).len(),
            6 * m * m + 3 * p * m * m,
            "p={p}: dof count"
        );

        let g = space.interpolate_vector(&|x| probe_field(*field, x));
        let gs = g.as_slice();
        for (e, want) in expect.iter().enumerate() {
            let dofs = space.element_dofs(e as u32);
            let signs = space.element_signs(e as u32);
            assert_eq!(
                dofs.len(),
                want.len(),
                "p={p} variant={variant} field={field} elem={e}"
            );
            for (i, &w) in want.iter().enumerate() {
                let got = signs[i] * gs[dofs[i] as usize];
                let rel = (got - w).abs() / w.abs().max(1e-2);
                max_rel = max_rel.max(rel);
                assert!(
                    rel <= 4e-15,
                    "p={p} variant={variant} field={field} elem={e} dof {i}: got {got}, want {w} (rel {rel:.2e})"
                );
                checked += 1;
            }
        }
    }
    println!(
        "d591: checked {checked} dof values against the MFEM probe, max rel dev {max_rel:.2e}"
    );
    assert!(checked > 1000);
}

/// 框架因子口径 (pinned, not tolerated away): D721 removed the last frame
/// factor — DOF values carry none and the field RECONSTRUCTION through the
/// fem-rs IGLL basis now **is** MFEM's physical field exactly (the historical
/// `V_mfem/16` debt made it MFEM/4).  On the unit cube the IGLL RT0 dofs are
/// the whole-face fluxes `0 0 1 0 -1 0` (= MFEM's probe line) and the
/// reconstruction ratio is exactly 1.
#[test]
fn d591_hex_igll_frame_factor_pinned() {
    let mesh = probe_mesh(0);
    let space = HDivSpace::new_gauss_lobatto_integrated_gll(mesh, 0);
    let g = space.interpolate_vector(&|_x| vec![1.0, 0.0, 0.0]);
    let gs = g.as_slice();
    let dofs = space.element_dofs(0);
    let signs = space.element_signs(0);

    // x+ face dof = +1, x- face dof = -1 (whole-face flux, MFEM probe line
    // `elem 0 n=6 : 0 0 1 0 -1 0`) — bit-exact.
    let local: Vec<f64> = (0..6).map(|i| signs[i] * gs[dofs[i] as usize]).collect();
    let expect = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    for (got, want) in local.iter().zip(expect) {
        assert_eq!(*got, want, "IGLL RT0 dof values must equal MFEM's exactly");
    }

    // Reconstruct at the element centre with the fem-rs IGLL basis:
    // u_phys = (1/det J) · J · Σ dof_i · sign_i · phi_ref_i(centre).
    let rt = HexRTk::new(0);
    let centre = [0.5, 0.5, 0.5];
    let mut phi = vec![0.0; rt.n_dofs() * 3];
    rt.eval_basis_vec(&centre, &mut phi);
    // D721: the reference cube IS the physical unit cube at the centre:
    // J = diag(1, 1, 1), det = 1.
    let jac = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let det = 1.0;
    let mut u = [0.0f64; 3];
    for i in 0..6 {
        for r in 0..3 {
            u[r] += gs[dofs[i] as usize]
                * signs[i]
                * (jac[r][0] * phi[i * 3]
                    + jac[r][1] * phi[i * 3 + 1]
                    + jac[r][2] * phi[i * 3 + 2])
                / det;
        }
    }
    // The reconstruction must be (1, 0, 0) = MFEM's field — the frame factor
    // is asserted at full precision, never absorbed by a tolerance.
    println!("d591: IGLL-frame reconstruction of the unit x-field = {}", u[0]);
    assert_eq!(u[0], 1.0, "fem-rs IGLL field frame must equal MFEM's (D721)");
    assert_eq!(u[1], 0.0);
    assert_eq!(u[2], 0.0);
}

/// The default constructor is untouched: on the same hex mesh it keeps the
/// nodal GaussLegendre engine semantics (its own MFEM `Project_RT` parity is
/// pinned by d342) and the variant flag stays off.
#[test]
fn d591_default_hex_space_unchanged() {
    let mesh = probe_mesh(0);
    let space = HDivSpace::new(mesh, 1);
    assert!(!space.quad_integrated_gll());
    let g = space.interpolate_vector(&|_x| vec![1.0, 0.0, 0.0]);
    // GL RT1 nodal dofs of f=(1,0,0) on the unit cube are ±1 point samples
    // (scaled), NOT the IGLL 0.25 sub-cell integrals — the two engines must
    // disagree; only their shared dof layout (36 slots) is identical.
    let dofs = space.element_dofs(0);
    let signs = space.element_signs(0);
    let local: Vec<f64> = (0..dofs.len())
        .map(|i| signs[i] * g.as_slice()[dofs[i] as usize])
        .collect();
    assert_eq!(local.len(), 36);
    assert!(
        local.iter().any(|&v| (v - 0.25).abs() > 1e-12),
        "default (GL) hex interpolation must not produce IGLL integral values"
    );
}

/// The quad IGLL routing is untouched by the hex extension: on a quad mesh
/// the two constructors still differ exactly in the variant flag.
#[test]
fn d591_quad_igll_flag_routing_unchanged() {
    let quad_mesh = Mesh::<2>::unit_square_quad(2);
    let def = HDivSpace::new(quad_mesh.clone(), 1);
    let igll = HDivSpace::new_gauss_lobatto_integrated_gll(quad_mesh, 1);
    assert!(!def.quad_integrated_gll());
    assert!(igll.quad_integrated_gll());
}
