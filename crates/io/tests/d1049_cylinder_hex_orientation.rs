//! D1049: the hex orientation check must be MFEM's center trilinear Jacobian.
//!
//! At the pre-fix HEAD, `check_element_orientation`'s linear branch applied
//! the tet-style corner determinant `det[v1-v0, v2-v0, v3-v0]` to HEXAHEDRA —
//! on `data/cylinder-hex.mesh` that measures the warpedness of the first
//! four corners (the bottom *face*), flagged 70/252 elements (exactly the
//! rod region; some on corner det = −3.5e-19, pure rounding noise) and
//! printed the stray `Elements with wrong orientation: 70 / 252 (not fixed)`
//! line that polluted joule's stdout byte comparison.  MFEM 4.10 checks
//! WEDGE/PYRAMID/HEXAHEDRON through `Mesh::GetElementJacobian` at
//! `Geometries.GetCenter` in the linear **and** the curved case
//! (`mesh/mesh.cpp:7346` `CheckElementOrientation`): on this mesh that is the
//! P1 trilinear Jacobian at (0.5, 0.5, 0.5), and it reports 0.
//!
//! Probe truth (`tmp/d1049/`, WSL `$HOME/mfem410_ser` C++ probe
//! `d1049_dump.cpp` vs the same Rust calls, 2026-10-02): all 252 center dets
//! bit-identical, `center_neg = 0` both sides; the corner formula is 70
//! negatives on both sides (it was never MFEM's hex check).

use fem_io::mfem::read_mfem_file;

const MESH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/cylinder-hex.mesh");

/// The reader must load cylinder-hex without flagging any element (red at
/// the pre-fix HEAD: 70 / 252 "not fixed").
#[test]
fn cylinder_hex_is_clean_under_the_mfem_orientation_check()
{
    let mut mesh = read_mfem_file(MESH).expect("read").mesh3d.expect("3-D");
    assert_eq!(mesh.n_elems(), 252);
    assert_eq!(mesh.check_element_orientation(false), 0, "detect pass");
    assert_eq!(mesh.check_element_orientation(true), 0, "fix pass is a no-op");
}

/// The center trilinear Jacobian determinant fem-rs evaluates is the exact
/// double MFEM 4.10 computes (`GetElementJacobian` at the geometry center):
/// bit-identical on a sample spanning both regions and the two det scales,
/// and non-negative on all 252 elements.
#[test]
fn center_dets_are_bit_identical_to_the_mfem_probe()
{
    let mesh = read_mfem_file(MESH).expect("read").mesh3d.expect("3-D");
    // (element, det) from the C++ probe's %.17e print (round-trip exact).
    // 11/16/17/251 are rod elements the old corner formula flagged.
    let truth: [(u32, f64); 12] = [
        (0, 4.37099210698607427e-03),
        (1, 4.86111111111111206e-03),
        (6, 6.24999999999999861e-03),
        (11, 4.86111111111110859e-03),
        (16, 3.90759238962054296e-03),
        (17, 5.33787047185639400e-03),
        (90, 6.24999999999999341e-03),
        (100, 3.90759238962053602e-03),
        (144, 1.62037037037037271e-02),
        (198, 2.54629629629629442e-02),
        (251, 2.54629629629629546e-02),
        (145, 2.08333333333332489e-02), // annulus region
    ];
    for (e, det) in truth {
        let (_j, d, _x) = mesh.element_jacobian(e, &[0.5, 0.5, 0.5]);
        assert_eq!(
            d.to_bits(),
            det.to_bits(),
            "element {e}: center det must match the C++ probe bit-for-bit"
        );
    }
    let negatives = (0..252u32)
        .filter(|&e| mesh.element_jacobian(e, &[0.5, 0.5, 0.5]).1 < 0.0)
        .count();
    assert_eq!(negatives, 0, "MFEM reports 0 wrong orientations here");
}
