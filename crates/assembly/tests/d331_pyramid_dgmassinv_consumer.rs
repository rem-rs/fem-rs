//! D331 (consumer pin) — `DGMassInverse` on a straight pyramid mesh.
//!
//! `crates/assembly/src/dgmassinv.rs:271` is the first *reachable* consumer of
//! `fem_mesh::transformation::element_jacobian_at` on pyramids: it integrates
//! the local mass matrix with `|det J|` from that helper.  Before D331 the
//! helper paired the layer-ordered `PyramidPk(1)` geometry basis with the raw
//! **vertex-ordered** node table, which swaps the two base corners of every
//! straight pyramid (`v2 ↔ v3`): measured `∫|det J| = 0.173755809543588` on the
//! unit pyramid instead of `1/3` (−47.9 %).
//!
//! The checks below are basis-independent: every nodal basis is a partition of
//! unity, so the *full sum* of the element mass matrix is exactly
//! `∫_e Σᵢⱼ φᵢφⱼ = ∫_e 1 = |e|` — the pyramid volume.  The second test
//! cross-checks the helper against the assembly-side dispatcher
//! `geo_ref_elem_from_mesh` (the `GeoPyrP1` path fixed in D304, which MFEM's
//! pyramid mass matrices are pinned against by `d304_pyramid_h1_mass`).

use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::element_jacobian_at;
use fem_mesh::Mesh;

const REF_VERTICES_FLAT: [f64; 15] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
];

fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        REF_VERTICES_FLAT.to_vec(),
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Sheared/scaled/translated`image of the reference pyramid with `det A = 9`.
fn affine_pyramid() -> Mesh<3> {
    let a = [[2.0, 0.3, 0.0], [0.0, 1.5, 0.2], [0.0, 0.0, 3.0]];
    let b = [1.0, -2.0, 0.5];
    let mut coords = Vec::with_capacity(15);
    for v in REF_VERTICES_FLAT.chunks_exact(3) {
        for i in 0..3 {
            coords.push((0..3).map(|j| a[i][j] * v[j]).sum::<f64>() + b[i]);
        }
    }
    Mesh::<3>::uniform(
        coords,
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// The consumer kernel of `dgmassinv.rs:271` — the mass accumulation
/// `mᵢⱼ += (quad.weights[qi] * |det J|) φᵢφⱼ` with the Jacobian coming from
/// `element_jacobian_at` — driven on a pyramid mesh.
///
/// `DGMassInverse` itself cannot run on a pyramid mesh today: `L2Space`
/// accepts Tri3/Quad4 (2-D) and Tet4/Hex8 (3-D) only and panics on a 5-node
/// element (`crates/space/src/l2.rs:136`), so that call site is *latent* for
/// pyramids.  This test exercises the reachable part of the same code path
/// (the `ref_elem_vol` basis and quadrature `dgmassinv` uses), whose sum is
/// the element volume for any partition-of-unity basis.
#[test]
fn dgmassinv_kernel_mass_sum_equals_the_pyramid_volume() {
    use fem_assembly::mixed::ref_elem_vol;
    use fem_element::ReferenceElement;

    for (name, mesh, volume) in [
        ("unit", unit_pyramid(), 1.0 / 3.0),
        ("affine (det A = 9)", affine_pyramid(), 9.0 / 3.0),
    ] {
        for order in [1u8, 2, 3] {
            let re = ref_elem_vol(ElementType::Pyramid5, order).unwrap();
            let quad = re.quadrature(2 * order + 1);
            let ne = re.n_dofs();
            let mut phi = vec![0.0_f64; ne];
            let mut m = vec![0.0_f64; ne * ne];
            for (q, xi) in quad.points.iter().enumerate() {
                let (jac, _xp) = element_jacobian_at(&mesh, 0, xi, 3);
                let w = quad.weights[q] * jac.determinant().abs();
                re.eval_basis(xi, &mut phi);
                for i in 0..ne {
                    for j in 0..ne {
                        m[i * ne + j] += w * phi[i] * phi[j];
                    }
                }
            }
            let sum: f64 = m.iter().sum();
            assert!(
                (sum - volume).abs() <= 1e-13 * volume.max(1.0),
                "{name} pyramid, order {order}: Σᵢⱼ Mᵢⱼ = {sum:.15} instead of the element volume \
                 {volume:.15} — `element_jacobian_at` is twisting the straight-pyramid geometry \
                 (D331)"
            );
        }
    }
}

/// `element_jacobian_at` must reproduce the assembly-side straight-pyramid
/// geometry (`geo_ref_elem_from_mesh` → `GeoPyrP1`, D304) pointwise.
#[test]
fn element_jacobian_at_agrees_with_geo_ref_elem_from_mesh() {
    for (name, mesh) in [("unit", unit_pyramid()), ("affine", affine_pyramid())] {
        let ge = geo_ref_elem_from_mesh(&mesh, 0)
            .expect("straight pyramids must take the isoparametric path");
        let nodes = mesh.geometry_nodes(0);
        let npe = ge.n_dofs();
        assert_eq!(npe, 5, "GeoPyrP1 dof count");
        for xi in [
            vec![0.25, 0.25, 0.25],
            vec![0.5, 0.1, 0.2],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ] {
            let (jac, xp) = element_jacobian_at(&mesh, 0, &xi, 3);
            let (jref, det_ref, xp_ref) = isoparametric_jacobian(&mesh, nodes, ge.as_ref(), &xi, 3);
            let mut worst = (jac.determinant() - det_ref).abs();
            for i in 0..3 {
                worst = worst.max((xp[i] - xp_ref[i]).abs());
                for j in 0..3 {
                    worst = worst.max((jac[(i, j)] - jref[(i, j)]).abs());
                }
            }
            assert!(
                worst < 1e-13,
                "{name} pyramid at {xi:?}: `element_jacobian_at` differs from the GeoPyrP1 \
                 path by {worst:.3e}"
            );
        }
    }
}
