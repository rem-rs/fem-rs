//! Repro / regression: H1 hex P>=3 dof coordinates must match the assembly
//! basis (GLL-noded `HexQk`, MFEM `H1_FECollection` GaussLobatto).
//!
//! Debt from round 2: "H1Space hex dof coords are equispaced while assembly
//! evaluates GLL-noded HexQk → hex P3 nodal interpolation is inexact."
//! Same family as the tri Pk>=3 fix (f931f92).

use fem_element::lagrange::factory::HexQk;
use fem_element::ReferenceElement;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::h1::H1Space;
use fem_space::fe_space::FESpace;

/// Map HexQk reference coords on [-1,1]^3 to the unit cube [0,1]^3.
fn to_unit(c: &[f64]) -> [f64; 3] {
    [0.5 * (c[0] + 1.0), 0.5 * (c[1] + 1.0), 0.5 * (c[2] + 1.0)]
}

/// Diagnostic: for every element-local slot i of the single-hex mesh, the
/// dof attached to slot i must carry the GLL nodal coordinates of HexQk
/// slot i (mapped from [-1,1]^3 to the physical unit cube).
#[test]
fn hex_p3_dof_coords_match_hexqk_gll_nodes() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let space = H1Space::<Mesh<3>>::new(mesh, 3);
    let dm = space.dof_manager();
    let dofs = space.element_dofs(0);

    let hex = HexQk::new(3);
    let ref_coords = hex.dof_coords();
    assert_eq!(dofs.len(), ref_coords.len(), "single hex P3: dof count");

    let mut n_bad = 0usize;
    let mut max_err = 0.0_f64;
    for (i, &d) in dofs.iter().enumerate() {
        let want = to_unit(&ref_coords[i]);
        let got = dm.dof_coord(d);
        let err = (got[0] - want[0]).abs().max((got[1] - want[1]).abs()).max((got[2] - want[2]).abs());
        if err > 1e-13 {
            n_bad += 1;
            if n_bad <= 12 {
                eprintln!("slot {i}: dof {d} coord {:?}  want {:?}  err {err:.3e}", got, want);
            }
        }
        max_err = max_err.max(err);
    }
    eprintln!("hex P3: {n_bad}/{} slots have dof_coord != HexQk GLL node (max err {max_err:.3e})", dofs.len());
    assert_eq!(n_bad, 0, "{n_bad} hex P3 dof coords disagree with HexQk GLL nodes (max err {max_err:.3e})");
}

/// Nodal interpolation of a smooth polynomial on the unit cube must be exact
/// (the Q3 tensor space contains every per-variable degree <= 3 polynomial).
#[test]
fn hex_p3_interpolate_exact_for_cubic_polynomial() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let space = H1Space::<Mesh<3>>::new(mesh, 3);
    let f = |x: &[f64]| x[0] * x[0] * x[1] + x[1] * x[2] + 1.0;
    let u = space.interpolate(&f);

    // Evaluate u_h = sum_i u[dof_i] * phi_i(x) with the assembly basis HexQk
    // at a set of interior/sample points; must reproduce f exactly.
    let hex = HexQk::new(3);
    let dofs = space.element_dofs(0);
    let n_ldofs = hex.n_dofs();
    let mut phi = vec![0.0_f64; n_ldofs];
    let sample: Vec<[f64; 3]> = vec![
        [0.1, 0.2, 0.3], [0.7, 0.4, 0.9], [0.5, 0.5, 0.5],
        [0.25, 0.75, 0.125], [0.9, 0.05, 0.6], [0.32, 0.61, 0.77],
    ];
    let mut max_err = 0.0_f64;
    for x in &sample {
        let xi = [2.0 * x[0] - 1.0, 2.0 * x[1] - 1.0, 2.0 * x[2] - 1.0];
        hex.eval_basis(&xi, &mut phi);
        let mut uh = 0.0_f64;
        for (i, &d) in dofs.iter().enumerate() {
            uh += u.as_slice()[d as usize] * phi[i];
        }
        let err = (uh - f(x)).abs();
        max_err = max_err.max(err);
        assert!(err < 1e-12, "hex P3 nodal interpolation inexact at {x:?}: uh={uh:.15}, f={:.15}, err={err:.3e}", f(x));
    }
    assert!(max_err < 1e-12);
}

/// Multi-element control (2x2x2 hex mesh): nodal interpolation must be exact
/// on every element.  This validates, against the assembly basis, both the
/// element-local dof layout and the dof coordinates, including the
/// shared-edge orientation flips and the shared-face position matching in
/// build_pk_hex.
#[test]
fn hex_p3_multielement_interpolate_exact() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = H1Space::<Mesh<3>>::new(mesh, 3);
    let f = |x: &[f64]| x[0] * x[0] * x[1] + x[1] * x[2] + 1.0;
    let u = space.interpolate(&f);

    let hex = HexQk::new(3);
    let n_ldofs = hex.n_dofs();
    let mut phi = vec![0.0_f64; n_ldofs];

    let mesh = space.mesh();
    for e in 0..mesh.n_elements() as u32 {
        let dofs = space.element_dofs(e);
        assert_eq!(dofs.len(), n_ldofs);
        // Element bounding box (structured mesh): sample inside element e.
        let ns = mesh.element_nodes(e);
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for &n in ns.iter() {
            let c = mesh.node_coords(n);
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        // Sample the element interior at its center and quarter points.
        let ctr = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
        let q1 = [0.25 * (3.0 * lo[0] + hi[0]), 0.25 * (3.0 * lo[1] + hi[1]), 0.25 * (3.0 * lo[2] + hi[2])];
        let q2 = [0.25 * (lo[0] + 3.0 * hi[0]), 0.25 * (lo[1] + 3.0 * hi[1]), 0.25 * (lo[2] + 3.0 * hi[2])];
        for x in [ctr, q1, q2] {
            // Map the physical sample point to the reference [-1,1]^3 of
            // this element (structured mesh: axis-aligned boxes).
            let xi = [
                2.0 * (x[0] - lo[0]) / (hi[0] - lo[0]) - 1.0,
                2.0 * (x[1] - lo[1]) / (hi[1] - lo[1]) - 1.0,
                2.0 * (x[2] - lo[2]) / (hi[2] - lo[2]) - 1.0,
            ];
            hex.eval_basis(&xi, &mut phi);
            let mut uh = 0.0_f64;
            for (i, &d) in dofs.iter().enumerate() {
                uh += u.as_slice()[d as usize] * phi[i];
            }
            let err = (uh - f(&x)).abs();
            assert!(err < 1e-12, "elem {e} at {x:?}: uh={uh:.15}, f={:.15}, err={err:.3e}", f(&x));
        }
    }
}

/// P2 control: GLL coincides with equispaced at p=2; the Q2 layout must
/// already agree with HexQk::new(2) (guarded so the P>=3 fix cannot regress it).
#[test]
fn hex_p2_dof_coords_match_hexqk_gll_nodes() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let space = H1Space::<Mesh<3>>::new(mesh, 2);
    let dm = space.dof_manager();
    let dofs = space.element_dofs(0);

    let hex = HexQk::new(2);
    let ref_coords = hex.dof_coords();
    assert_eq!(dofs.len(), ref_coords.len());

    for (i, &d) in dofs.iter().enumerate() {
        let want = to_unit(&ref_coords[i]);
        let got = dm.dof_coord(d);
        let err = (got[0] - want[0]).abs().max((got[1] - want[1]).abs()).max((got[2] - want[2]).abs());
        assert!(err < 1e-13, "hex P2 slot {i} (dof {d}): coord {got:?} != GLL {want:?} (err {err:.3e})");
    }
}

/// Same layout/coords consistency check at P4 (general build_pk_hex path).
#[test]
fn hex_p4_dof_coords_match_hexqk_gll_nodes() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let space = H1Space::<Mesh<3>>::new(mesh, 4);
    let dm = space.dof_manager();
    let dofs = space.element_dofs(0);

    let hex = HexQk::new(4);
    let ref_coords = hex.dof_coords();
    assert_eq!(dofs.len(), ref_coords.len());

    for (i, &d) in dofs.iter().enumerate() {
        let want = to_unit(&ref_coords[i]);
        let got = dm.dof_coord(d);
        let err = (got[0] - want[0]).abs().max((got[1] - want[1]).abs()).max((got[2] - want[2]).abs());
        assert!(err < 1e-13, "hex P4 slot {i} (dof {d}): coord {got:?} != GLL {want:?} (err {err:.3e})");
    }
}
