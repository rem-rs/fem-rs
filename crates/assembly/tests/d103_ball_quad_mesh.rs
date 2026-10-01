//! d103: pin the fem-rs reading of the C++-converted ball mesh
//! (`tmp/d103tesla/ball-quad.mesh` — the exact `UniformRefinement +
//! SetCurvature(2)` conversion of MFEM's default `ball-nurbs.mesh`, exported
//! by the d103 C++ probe): total volume == MFEM's own measure of the same
//! mesh at matching quadrature (`tmp/d103tesla/volume_probe.cpp` prints
//! 4.1776825820582077 at order-6 tensor Gauss) and the H1/ND/RT true-dof
//! counts == MFEM's `GlobalTrueVSize` (79 / 202 / 180,
//! `tmp/d103tesla/cpp_bm_o1.log`).

use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

/// Total measure of the mesh via 4-point-per-dimension Gauss on every
/// element, through the mesh's own (curved) isoparametric map — the same
/// geometry source the assemblers use.
fn total_volume(mesh: &fem_mesh::Mesh<3>) -> f64 {
    let (xs, ws) = gauss4();
    let mut vol = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let (pts, wts): (Vec<[f64; 3]>, Vec<f64>) = match et {
            fem_mesh::ElementType::Hex8
            | fem_mesh::ElementType::Hex20
            | fem_mesh::ElementType::Hex27 => {
                let mut p = Vec::with_capacity(64);
                let mut w = Vec::with_capacity(64);
                for (zk, wk) in xs.iter().zip(ws.iter()) {
                    for (yj, wj) in xs.iter().zip(ws.iter()) {
                        for (xi, wi) in xs.iter().zip(ws.iter()) {
                            p.push([*xi, *yj, *zk]);
                            w.push(wi * wj * wk);
                        }
                    }
                }
                (p, w)
            }
            fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10 => (
                vec![
                    [0.585_410_196_624_968_5, 0.138_196_601_125_010_5, 0.138_196_601_125_010_5],
                    [0.138_196_601_125_010_5, 0.585_410_196_624_968_5, 0.138_196_601_125_010_5],
                    [0.138_196_601_125_010_5, 0.138_196_601_125_010_5, 0.585_410_196_624_968_5],
                    [0.138_196_601_125_010_5, 0.138_196_601_125_010_5, 0.138_196_601_125_010_5],
                ],
                vec![1.0 / 24.0; 4],
            ),
            other => panic!("d103: unsupported element {other:?}"),
        };
        for (xi, wq) in pts.iter().zip(wts.iter()) {
            let (j, _xp) = fem_mesh::element_jacobian_at(mesh, e, xi, 3);
            vol += wq * j.determinant().abs();
        }
    }
    vol
}

/// 4-point Gauss-Legendre on [0,1] (MFEM-compatible values).
fn gauss4() -> ([f64; 4], [f64; 4]) {
    (
        [
            0.069_431_844_202_973_71,
            0.330_009_478_207_571_87,
            0.669_990_521_792_428_1,
            0.930_568_155_797_026_2,
        ],
        [
            0.173_927_422_568_726_92,
            0.326_072_577_431_273_05,
            0.326_072_577_431_273_05,
            0.173_927_422_568_726_92,
        ],
    )
}

#[test]
fn d103_ball_quad_mesh_geometry_and_sizes() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d103tesla/ball-quad.mesh");
    let mfem = read_mfem_file(path).expect("ball-quad.mesh reads");
    let mesh = mfem.mesh3d.expect("3-D mesh");

    // Geometry order: the `nodes` table is quadratic (MFEM SetCurvature(2)).
    let geo_order = mesh.geometry.as_ref().map(|g| g.order).unwrap_or(1);
    assert_eq!(geo_order, 2, "ball-quad carries a quadratic nodes table");

    // Volume pin: C++ probe at matching quadrature (order-6 tensor Gauss per
    // hex) prints 4.1776825820582077.
    let vol = total_volume(&mesh);
    let cpp_volume = 4.177_682_582_058_207_7e0;
    assert!(
        (vol - cpp_volume).abs() < 1e-12,
        "volume {vol:.16e} vs C++ {cpp_volume:.16e}"
    );

    // MFEM GlobalTrueVSize on this mesh (order 1).
    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);
    let rt = HDivSpace::new(mesh, 0);
    assert_eq!(h1.n_dofs(), 79);
    assert_eq!(nd.n_dofs(), 202);
    assert_eq!(rt.n_dofs(), 180);
}
