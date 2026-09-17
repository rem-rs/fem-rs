//! D289 assembly-layer regression: externally supplied hex RT dofs (the
//! MFEM VisIt/DC-file scenario) evaluate through the GL-framed postprocessing
//! exactly like MFEM `GridFunction::GetDivergence`.
//!
//! The C++ side (`tmp/d309/probe_rt_dof_div.cpp`, MFEM 4.10) projects
//! u = (x+2y+3z+1, 0, 0) (div = 1) onto `RT_FECollection(1, 3)` on a single
//! affine hex `1.0 x 1.2 x 0.8` and dumps the `Project_RT` dofs — exactly the
//! coefficient vector an MFEM data collection would carry.  Feeding those
//! dofs into [`compute_element_divergence`] (D245: GaussLegendre `vec_ref_elem`
//! arm, D265: isoparametric hex geometry) must return div = 1 at the hex
//! centroid.  The same dofs produced by `HDivSpace::interpolate_vector`
//! (D289: GaussLegendre dual in `fill_dual_matrix`) must agree.

use fem_assembly::postproc::postprocess::compute_element_divergence;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::{fe_space::FESpace, HDivSpace};

/// MFEM 4.10 `Project_RT` dofs of u = (x+2y+3z+1, 0, 0) on the single hex
/// (`RT_FECollection(1, 3)`, 36 dofs, %.17e transcription).
const MFEM_RT1_DOFS: &str = "
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 2.89378497978710225e+00
4.22400000000000020e+00 4.22400000000000020e+00 5.55421502021289726e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 -3.26399999999999979e+00 -1.93378497978710207e+00
-4.59421502021289729e+00 -3.26399999999999979e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
2.41378497978710183e+00 3.74399999999999933e+00 3.74399999999999977e+00
5.07421502021289772e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00
";

/// Single affine hex `1.0 x 1.2 x 0.8`.
fn hex_mesh() -> Mesh<3> {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    mesh.transform(|p| [p[0], 1.2 * p[1], 0.8 * p[2]]);
    mesh
}

/// Externally supplied (DC-file) dofs give the MFEM divergence (div = 1 for
/// this field) through the GL-framed postprocessing.
#[test]
fn external_dc_file_dofs_diverge_like_mfem() {
    let mesh = hex_mesh();
    let space = HDivSpace::new(mesh, 1);
    let external: Vec<f64> = MFEM_RT1_DOFS
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("token parses"))
        .collect();
    assert_eq!(external.len(), space.element_dofs(0).len());

    let divs = compute_element_divergence(&space, &external);
    assert_eq!(divs.len(), 1);
    assert!(
        (divs[0] - 1.0).abs() <= 1e-13,
        "external dofs: div = {:.17e}, expected 1",
        divs[0]
    );
}

/// The dofs `interpolate_vector` produces for the same field agree with the
/// MFEM file values, so interpolated and external fields are
/// indistinguishable downstream (pre-D289 the RT0 arm stored 4x MFEM values).
#[test]
fn interpolated_dofs_agree_with_external_dc_file_dofs() {
    let mesh = hex_mesh();
    let space = HDivSpace::new(mesh, 1);
    let interp = space
        .interpolate_vector(&|x| {
            vec![x[0] + 2.0 * x[1] + 3.0 * x[2] + 1.0, 0.0, 0.0]
        })
        .as_slice()
        .to_vec();
    let external: Vec<f64> = MFEM_RT1_DOFS
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("token parses"))
        .collect();

    for (i, (&g, &w)) in interp.iter().zip(external.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-14 * (1.0 + w.abs()),
            "dof {i}: interp {g:.17e} vs mfem {w:.17e}"
        );
    }

    // and both give the same divergence
    let d_ext = compute_element_divergence(&space, &external);
    let d_int = compute_element_divergence(&space, &interp);
    assert!((d_ext[0] - d_int[0]).abs() <= 1e-14);
    assert!((d_ext[0] - 1.0).abs() <= 1e-13);
}

/// RT0 companion: u = (1, 2, 3) has div = 0; MFEM `Project_RT` dofs
/// [-3.6, -1.6, 0.96, 1.6, -0.96, 3.6] must give 0 (pre-D289 interpolation
/// produced exactly 4x these values, so any GL-framed divergence computed
/// from interpolated dofs was corrupted — here pinned to 0).
#[test]
fn external_rt0_dofs_diverge_like_mfem() {
    let mesh = hex_mesh();
    let space = HDivSpace::new(mesh, 0);
    let external: Vec<f64> = [-3.5999999999999996, -1.6, 0.96, 1.6, -0.96, 3.6].to_vec();

    let divs = compute_element_divergence(&space, &external);
    assert!(
        divs[0].abs() <= 1e-13,
        "external RT0 dofs: div = {:.17e}, expected 0",
        divs[0]
    );

    // interpolated dofs must equal the file values (pre-D289 they were 4x)
    let interp = space
        .interpolate_vector(&|_| vec![1.0, 2.0, 3.0])
        .as_slice()
        .to_vec();
    for (i, (&g, &w)) in interp.iter().zip(external.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-15 * (1.0 + w.abs()),
            "rt0 dof {i}: interp {g:.17e} vs mfem {w:.17e}"
        );
    }
}
