//! D715 — `element_jacobian_at` must interpolate a **curved** hex
//! (`geom_order > 1`) isoparametrically instead of "straightening" it through
//! the P1 vertex table.
//!
//! Round 68 (D709) proved the defect end to end: the dyn-`MeshTopology`
//! geometry path `fem_mesh::element_jacobian_at` evaluates the **P1**
//! (`HexQk(1)`) basis over the element's 8 corner vertices whenever the
//! per-element geometry table is not P1-sized — i.e. every curved hex.  The
//! in-crate isoparametric path `Mesh::element_jacobian` (D708-aligned with
//! MFEM 4.10) evaluates the order-`geom_order` `HexQk` basis over the full
//! high-order table, so the two paths disagree at O(curvature) on curved
//! hexes and coincide only on straight ones.  Consumers of the dyn path —
//! `HDivSpace::dof_nodal_coords` (hdiv.rs), the hdiv-error / complex-DPG
//! assemblers — therefore sampled a straightened ghost of the real element.
//!
//! Pins (fixture: `d708_curved_hex.mesh`, the single curved P2 cylinder-slab
//! hex of the multidomain mesh, `H1_3D_P2` nodes):
//!
//! 1. straight hexes: `element_jacobian_at` == `Mesh::element_jacobian`
//!    **bitwise** (J entries and mapped points, f64 bits) — the D715 fix is
//!    gated on `geom_order >= 2`, so the straight branch must be inert; the
//!    bit dump (`--nocapture`) is the before/after bitwise evidence;
//! 2. curved hex: the same comparison (red before the fix, bitwise-equal
//!    after);
//! 3. `HDivSpace::dof_nodal_coords` (RT0) on the curved hex returns exactly
//!    the six isoparametric face-centre points — the hdiv-side defect
//!    D715 closes (the d709 conjugacy sweep runs on straight geometry only,
//!    so it cannot see this);
//! 4. two curved hexes sharing a face: the shared RT0 dof resolves to the
//!    same **curved** face centre from both sides (no straightened
//!    disagreement, no drift off the isoparametric surface).

use fem_io::mfem::read_mfem_file;
use fem_mesh::element_jacobian_at;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::hdiv::HDivSpace;

fn data(rel: &str) -> String {
    format!("{}/tests/data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// Sample lattice on `[-1,1]^3`: corners, face centres, mid-edges and
/// interior points (the RT0 nodal points are a subset).
const LATTICE: [[f64; 3]; 16] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.25, -0.5, 0.75],
];

/// The six RT0 hex nodal reference points: face centres on `[-1,1]^3`.
const RT0_FACE_CENTRES: [[f64; 3]; 6] = [
    [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
];

/// Worst mismatch between the dyn path and the in-crate isoparametric path
/// over the sample lattice: `(bit-distance J, bit-distance x, |ΔJ|, |Δx|)`.
fn worst_deviation(mesh: &Mesh<3>, e: u32) -> (u64, u64, f64, f64) {
    let mut worst_jb = 0_u64;
    let mut worst_xb = 0_u64;
    let mut worst_jv = 0.0_f64;
    let mut worst_xv = 0.0_f64;
    for xi in &LATTICE {
        let (j_at, x_at) = element_jacobian_at(mesh, e, xi, 3);
        let (j_in, _det, x_in) = mesh.element_jacobian(e, xi);
        for i in 0..3 {
            for k in 0..3 {
                worst_jb = worst_jb.max(j_at[(i, k)].to_bits().abs_diff(j_in[(i, k)].to_bits()));
                worst_jv = worst_jv.max((j_at[(i, k)] - j_in[(i, k)]).abs());
            }
            worst_xb = worst_xb.max(x_at[i].to_bits().abs_diff(x_in[i].to_bits()));
            worst_xv = worst_xv.max((x_at[i] - x_in[i]).abs());
        }
    }
    (worst_jb, worst_xb, worst_jv, worst_xv)
}

/// Pin 1 + the D715 bitwise dump: straight hexes must be inert under the fix.
#[test]
fn straight_hex_dyn_path_bitwise_inert() {
    let meshes: Vec<(&str, Mesh<3>)> = vec![
        ("unit_cube_hex", Mesh::<3>::unit_cube_hex(2)),
        ("cartesian_2x1x1", {
            let mut m = Mesh::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 2.0, 1.0, 1.0, false);
            m.set_curvature(1); // reset to linear (no geometry table)
            m
        }),
    ];
    for (name, mesh) in &meshes {
        assert_eq!(mesh.geom_order(), 1, "{name}: straight geometry");
        for e in 0..mesh.n_elems() as u32 {
            let (jb, xb, _jv, _xv) = worst_deviation(mesh, e);
            // Bit dump for the before/after diff (`cargo test ... -- --nocapture
            // > tmp/d715/...`); identical arithmetic on both paths must give
            // identical bits on straight hexes.
            for xi in &LATTICE {
                let (j_at, x_at) = element_jacobian_at(mesh, e, xi, 3);
                let mut line = String::new();
                for i in 0..3 {
                    for k in 0..3 {
                        line.push_str(&format!("{:016x} ", j_at[(i, k)].to_bits()));
                    }
                }
                for i in 0..3 {
                    line.push_str(&format!("{:016x} ", x_at[i].to_bits()));
                }
                eprintln!("BIT {name} e{e} {} {} {} {line}", xi[0], xi[1], xi[2]);
            }
            assert_eq!(jb, 0, "{name}: J bits differ between the two paths");
            assert_eq!(xb, 0, "{name}: x bits differ between the two paths");
        }
    }
}

/// Pin 2: on the curved P2 hex the dyn path must equal the in-crate
/// isoparametric path (red before D715: the P1 straightening deviated at
/// O(curvature)).
#[test]
fn d708_curved_hex_dyn_path_is_isoparametric() {
    let mfem = read_mfem_file(data("d708_curved_hex.mesh")).expect("read curved hex");
    let mesh: Mesh<3> = mfem.mesh3d.expect("3-D");
    assert_eq!(mesh.n_elems(), 1);
    assert_eq!(mesh.geom_order(), 2, "H1_3D_P2 nodes");
    assert_eq!(mesh.geometry_nodes(0).len(), 27, "order-2 hex table");

    let (jb, xb, jv, xv) = worst_deviation(&mesh, 0);
    eprintln!("D715 curved hex deviation: bits J {jb} x {xb}, values |ΔJ| {jv:.3e} |Δx| {xv:.3e}");
    assert_eq!(jb, 0, "curved hex: J bits differ (max |ΔJ| = {jv:.3e})");
    assert_eq!(xb, 0, "curved hex: x bits differ (max |Δx| = {xv:.3e})");
}

/// Face-centre physical points through the in-crate isoparametric path.
fn isoparametric_face_centres(mesh: &Mesh<3>, e: u32) -> [[f64; 3]; 6] {
    let mut out = [[0.0; 3]; 6];
    for (k, xi) in RT0_FACE_CENTRES.iter().enumerate() {
        let (_j, _det, x) = mesh.element_jacobian(e, xi);
        out[k] = [x[0], x[1], x[2]];
    }
    out
}

/// Every RT0 dof point must coincide with an isoparametric face centre
/// (tolerance 1e-12; the sets are compared because dof->slot order is an
/// hdiv-internal convention).
fn assert_rt0_points_are_face_centres(mesh: &Mesh<3>, label: &str) {
    let space = HDivSpace::new(mesh.clone(), 0);
    let pts = space.dof_nodal_coords();
    assert_eq!(pts.len(), space.n_dofs(), "{label}: dof count");
    let want = isoparametric_face_centres(mesh, 0);
    let mut used = [false; 6];
    let mut worst = 0.0_f64;
    for p in &pts {
        let (best_k, best_d) = want
            .iter()
            .enumerate()
            .map(|(k, w)| {
                let d = (0..3).map(|c| (p[c] - w[c]).abs()).fold(0.0_f64, f64::max);
                (k, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        worst = worst.max(best_d);
        assert!(
            best_d < 1e-12,
            "{label}: RT0 point {p:?} matches no isoparametric face centre \
             (worst {best_d:.3e}) — straightened geometry"
        );
        used[best_k] = true;
    }
    assert!(used.iter().all(|&u| u), "{label}: some face centre unmatched");
    eprintln!("D715 {label}: RT0 face-centre match worst {worst:.3e}");
}

/// Pin 3: RT0 nodal points on the curved hex are the isoparametric face
/// centres (red before D715: the straightened trilinear face centres sat off
/// the curved surface).
#[test]
fn d708_curved_hex_rt0_nodal_points_isoparametric() {
    let mfem = read_mfem_file(data("d708_curved_hex.mesh")).expect("read curved hex");
    let mesh: Mesh<3> = mfem.mesh3d.expect("3-D");
    assert_rt0_points_are_face_centres(&mesh, "d708 curved hex");
}

/// Pin 4: two curved hexes sharing a face — the shared RT0 dofs resolve to
/// the same **curved** face centre (the d709 consistency scan, on curved
/// geometry).
///
/// Sensitivity note: at a face centre the order-2 hex basis puts weight 1 on
/// the face-centre geometry node, so RT0 nodal points move exactly with the
/// face-centre nodes (edge/interior bends are invisible there — probed in
/// `tmp/d715`).  `set_curvature` places face nodes on the trilinear map and
/// does NOT deduplicate them across elements (one copy per element row), so a
/// shared-face bend must move every copy at once to keep the two maps
/// identical — which is exactly the post-fix expectation for `dof_nodal_coords`.
#[test]
fn curved_two_hex_shared_face_rt0_points_agree() {
    let mut mesh = Mesh::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 2.0, 1.0, 1.0, false);
    mesh.set_curvature(2);

    // Bend every geometry node sitting at `target` (any element's copy):
    // elem 0's bottom face centre and the shared face x=1 centre — off the
    // trilinear map, i.e. genuinely curved faces.
    let bend_all = |mesh: &mut Mesh<3>, target: [f64; 3], d: [f64; 3]| {
        let g = mesh.geometry.as_mut().unwrap();
        let mut hits = 0;
        for i in 0..g.n_nodes {
            if (0..3).all(|c| (g.coords[i * 3 + c] - target[c]).abs() < 1e-12) {
                g.coords[i * 3] += d[0];
                g.coords[i * 3 + 1] += d[1];
                g.coords[i * 3 + 2] += d[2];
                hits += 1;
            }
        }
        assert!(hits >= 1, "no geometry node at {target:?}");
        hits
    };
    assert_eq!(
        bend_all(&mut mesh, [0.5, 0.5, 0.0], [0.03, -0.02, 0.04]),
        1,
        "elem 0's private bottom face centre"
    );
    assert_eq!(
        bend_all(&mut mesh, [1.0, 0.5, 0.5], [-0.04, 0.02, 0.03]),
        2,
        "the shared face centre: one geometry-node copy per element"
    );

    let space = HDivSpace::new(mesh.clone(), 0);
    let pts = space.dof_nodal_coords();
    // Every dof point must lie on an isoparametric face centre of EITHER
    // element (element 1's centres through its own curved map).
    let mut want: Vec<[f64; 3]> = Vec::new();
    for e in 0..2u32 {
        want.extend(isoparametric_face_centres(&mesh, e));
    }
    for (d, p) in pts.iter().enumerate() {
        let best = want
            .iter()
            .map(|w| (0..3).map(|c| (p[c] - w[c]).abs()).fold(0.0_f64, f64::max))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best < 1e-12,
            "dof {d} at {p:?} off every curved face centre (worst {best:.3e})"
        );
    }
}
