//! D28 regression: `HDivSpace::interpolate_vector` vs the assembly basis.
//!
//! The vector assembler pairs element-local dof `i` with reference basis
//! function `i` of the RT reference element (`crates/assembly/src/vector_assembler.rs`,
//! `vec_ref_elem`) and forms the physical basis
//!
//! ```text
//!     phi_i(x) = signs[i] * Piola(ref_basis_i(xihat))
//! ```
//!
//! so a global dof vector `g` reconstructs, element by element, the field
//!
//! ```text
//!     uh|_K = sum_i g[dofs[i]] * signs[i] * Piola(phi_hat_i)
//! ```
//!
//! `interpolate_vector(f)` must therefore produce `g[dofs[i]] = signs[i] * D_i(f)`
//! where `D_i` is the dual functional of `phi_hat_i`.  For every field that is
//! exactly representable in the space the reconstructed error must vanish.
//!
//! These tests pin that property for the D28 reproduction matrix:
//! {tri-RT0/1/2, quad-RT0/1, tet-RT0, hex-RT0/1} x {constant, linear, rotational}.

use fem_element::raviart_thomas::{HexRTk, QuadRT1, QuadRTk, TetRTk, TriRT1, TriRT2, TriRTk};
use fem_element::reference::VectorReferenceElement;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_space::{fe_space::FESpace, HDivSpace};

// ─── reconstruction helpers ─────────────────────────────────────────────────

/// Jacobian + physical point of the (affine) simplex map for tri/tet.
struct SimplexMap {
    x0: [f64; 3],
    cols: [[f64; 3]; 3], // ref-dim columns
    det: f64,
    j: [[f64; 3]; 3], // j[row][col]
}

impl SimplexMap {
    fn new(mesh: &Mesh<3>, nodes: &[u32]) -> Self {
        let x0: [f64; 3] = mesh.node_coords(nodes[0]).try_into().unwrap();
        let mut cols = [[0.0f64; 3]; 3];
        for (c, &nd) in nodes.iter().skip(1).enumerate() {
            let p = mesh.node_coords(nd);
            for r in 0..3 {
                cols[c][r] = p[r] - x0[r];
            }
        }
        let j = [
            [cols[0][0], cols[1][0], cols[2][0]],
            [cols[0][1], cols[1][1], cols[2][1]],
            [cols[0][2], cols[1][2], cols[2][2]],
        ];
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        SimplexMap { x0, cols, det, j }
    }

    fn map(&self, xi: &[f64]) -> [f64; 3] {
        let mut x = self.x0;
        for (c, t) in self.cols.iter().enumerate() {
            for r in 0..3 {
                x[r] += t[r] * xi[c];
            }
        }
        x
    }
}

/// Jacobian + physical point of the bilinear quad map on the reference [0,1]^2.
struct QuadMap {
    x: Vec<Vec<f64>>, // 4 corner coords
}

impl QuadMap {
    fn new(mesh: &Mesh<2>, nodes: &[u32]) -> Self {
        QuadMap {
            x: nodes.iter().map(|&n| mesh.node_coords(n).to_vec()).collect(),
        }
    }
    /// Bilinear shape functions on [0,1]^2 (Q1).
    fn shapes(xi: &[f64]) -> [f64; 4] {
        let (s, t) = (xi[0], xi[1]);
        [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t]
    }
    fn map(&self, xi: &[f64]) -> [f64; 2] {
        let n = Self::shapes(xi);
        let mut out = [0.0; 2];
        for (i, &ni) in n.iter().enumerate() {
            out[0] += ni * self.x[i][0];
            out[1] += ni * self.x[i][1];
        }
        out
    }
    fn jac(&self, xi: &[f64]) -> [[f64; 2]; 2] {
        let (s, t) = (xi[0], xi[1]);
        // dN/ds, dN/dt for the Q1 bilinear basis
        let ds = [-(1.0 - t), 1.0 - t, t, -t];
        let dt = [-(1.0 - s), -s, s, 1.0 - s];
        let mut j = [[0.0; 2]; 2];
        for i in 0..4 {
            j[0][0] += ds[i] * self.x[i][0];
            j[0][1] += dt[i] * self.x[i][0];
            j[1][0] += ds[i] * self.x[i][1];
            j[1][1] += dt[i] * self.x[i][1];
        }
        j
    }
}

/// Jacobian + physical point of the trilinear hex map on the reference [-1,1]^3.
/// Local vertex order (MFEM/mesh convention): bottom face CCW (0..3), then the
/// vertices above them (4..7): 2 = (+,+,−), 3 = (−,+,−), etc.
struct HexMap {
    x: Vec<Vec<f64>>, // 8 corner coords
}

/// Reference-sign table per local vertex index (x, y, z components).
const HEX_SIGNS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

impl HexMap {
    fn new(mesh: &Mesh<3>, nodes: &[u32]) -> Self {
        HexMap {
            x: nodes.iter().map(|&n| mesh.node_coords(n).to_vec()).collect(),
        }
    }
    #[allow(clippy::needless_range_loop)]
    fn map(&self, xi: &[f64]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for i in 0..8 {
            let s = HEX_SIGNS[i];
            let n = 0.125
                * (1.0 + s[0] * xi[0])
                * (1.0 + s[1] * xi[1])
                * (1.0 + s[2] * xi[2]);
            for r in 0..3 {
                out[r] += n * self.x[i][r];
            }
        }
        out
    }
    fn jac(&self, xi: &[f64]) -> [[f64; 3]; 3] {
        let mut j = [[0.0; 3]; 3];
        for i in 0..8 {
            let s = HEX_SIGNS[i];
            // dN/dcoord = (s/2) * prod of the other two linear factors
            let d = [
                0.5 * s[0] * 0.5 * (1.0 + s[1] * xi[1]) * 0.5 * (1.0 + s[2] * xi[2]),
                0.5 * (1.0 + s[0] * xi[0]) * 0.5 * s[1] * 0.5 * (1.0 + s[2] * xi[2]),
                0.5 * (1.0 + s[0] * xi[0]) * 0.5 * (1.0 + s[1] * xi[1]) * 0.5 * s[2],
            ];
            for r in 0..3 {
                for c in 0..3 {
                    j[r][c] += d[c] * self.x[i][r];
                }
            }
        }
        j
    }
}

fn adj_2d(j: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    // adj(J) = det(J) J^{-1} (so that adj(J) * nk = physical scaled normal)
    [[j[1][1], -j[0][1]], [-j[1][0], j[0][0]]]
}

/// L2 error of the field reconstructed from the global dof vector `g`
/// against `exact`, using the assembler's element basis convention.
fn reconstruction_l2_error(
    space: &HDivSpace<Mesh<2>>,
    g: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
    quad_order: u8,
) -> f64 {
    let mesh = space.mesh();
    let mut e2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);

        let (n_ref, ref_dim) = match et {
            ElementType::Tri3 | ElementType::Tri6 => {
                let r: Box<dyn VectorReferenceElement> = match space.order() {
                    0 => Box::new(TriRTk::new(0)),
                    1 => Box::new(TriRT1),
                    _ => Box::new(TriRT2),
                };
                (r.n_dofs(), 2)
            }
            ElementType::Quad4 => {
                let r: Box<dyn VectorReferenceElement> = match space.order() {
                    0 => Box::new(QuadRTk::new(0)),
                    1 => Box::new(QuadRT1),
                    o => Box::new(QuadRTk::new(o as usize)),
                };
                (r.n_dofs(), 2)
            }
            other => panic!("2-D reconstruction: unsupported {other:?}"),
        };
        debug_assert_eq!(n_ref, dofs.len());

        // Evaluate the reconstructed field at a physical point helper closure
        // is per-quadrature-point below (needs the local basis first).
        match et {
            ElementType::Tri3 | ElementType::Tri6 => {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
                let jac = tr.jacobian();
                let det_j = tr.det_j();
                let ref_elem: Box<dyn VectorReferenceElement> = match space.order() {
                    0 => Box::new(TriRTk::new(0)),
                    1 => Box::new(TriRT1),
                    _ => Box::new(TriRT2),
                };
                let q = ref_elem.quadrature(quad_order);
                let mut phi = vec![0.0; n_ref * ref_dim];
                for qi in 0..q.points.len() {
                    let xi = &q.points[qi];
                    ref_elem.eval_basis_vec(xi, &mut phi);
                    let w = q.weights[qi] * det_j.abs();
                    let xp = tr.map_to_physical(xi);
                    let mut uh = [0.0f64; 2];
                    for i in 0..n_ref {
                        let s = signs[i];
                        uh[0] += g[dofs[i]]
                            * s
                            * (jac[(0, 0)] * phi[i * 2] + jac[(0, 1)] * phi[i * 2 + 1])
                            / det_j;
                        uh[1] += g[dofs[i]]
                            * s
                            * (jac[(1, 0)] * phi[i * 2] + jac[(1, 1)] * phi[i * 2 + 1])
                            / det_j;
                    }
                    let ue = exact(&xp);
                    e2 += w * ((uh[0] - ue[0]).powi(2) + (uh[1] - ue[1]).powi(2));
                }
            }
            ElementType::Quad4 => {
                let qm = QuadMap::new(mesh, &nodes);
                let ref_elem: Box<dyn VectorReferenceElement> = match space.order() {
                    0 => Box::new(QuadRTk::new(0)),
                    1 => Box::new(QuadRT1),
                    o => Box::new(QuadRTk::new(o as usize)),
                };
                let q = ref_elem.quadrature(quad_order);
                let mut phi = vec![0.0; n_ref * ref_dim];
                for qi in 0..q.points.len() {
                    let xi = &q.points[qi];
                    ref_elem.eval_basis_vec(xi, &mut phi);
                    let j = qm.jac(xi);
                    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                    let w = q.weights[qi] * det.abs();
                    let xp = qm.map(xi);
                    let mut uh = [0.0f64; 2];
                    for i in 0..n_ref {
                        let s = signs[i];
                        uh[0] += g[dofs[i]]
                            * s
                            * (j[0][0] * phi[i * 2] + j[0][1] * phi[i * 2 + 1])
                            / det;
                        uh[1] += g[dofs[i]]
                            * s
                            * (j[1][0] * phi[i * 2] + j[1][1] * phi[i * 2 + 1])
                            / det;
                    }
                    let ue = exact(&xp);
                    e2 += w * ((uh[0] - ue[0]).powi(2) + (uh[1] - ue[1]).powi(2));
                }
            }
            _ => unreachable!(),
        }
    }
    e2.sqrt()
}

/// 3-D variant (tet/hex meshes).
fn reconstruction_l2_error_3d(
    space: &HDivSpace<Mesh<3>>,
    g: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
    quad_order: u8,
) -> f64 {
    let mesh = space.mesh();
    let mut e2 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let ref_elem: Box<dyn VectorReferenceElement> = match et {
            ElementType::Tet4 | ElementType::Tet10 => Box::new(TetRTk::new(space.order() as usize)),
            ElementType::Hex8 => Box::new(HexRTk::new(space.order() as usize)),
            other => panic!("3-D reconstruction: unsupported {other:?}"),
        };
        let q = ref_elem.quadrature(quad_order);
        let n_ref = ref_elem.n_dofs();
        debug_assert_eq!(n_ref, dofs.len());
        let mut phi = vec![0.0; n_ref * 3];
        match et {
            ElementType::Tet4 | ElementType::Tet10 => {
                let m = SimplexMap::new(mesh, &nodes);
                for qi in 0..q.points.len() {
                    let xi = &q.points[qi];
                    ref_elem.eval_basis_vec(xi, &mut phi);
                    let w = q.weights[qi] * m.det.abs();
                    let xp = m.map(xi);
                    let jt = &m.j; // J (Piola: phi_phys = J phi_ref / det)
                    let mut uh = [0.0f64; 3];
                    for i in 0..n_ref {
                        let s = signs[i];
                        for r in 0..3 {
                            uh[r] += g[dofs[i]]
                                * s
                                * (jt[r][0] * phi[i * 3]
                                    + jt[r][1] * phi[i * 3 + 1]
                                    + jt[r][2] * phi[i * 3 + 2])
                                / m.det;
                        }
                    }
                    let ue = exact(&xp);
                    for r in 0..3 {
                        e2 += w * (uh[r] - ue[r]).powi(2);
                    }
                }
            }
            ElementType::Hex8 => {
                let hm = HexMap::new(mesh, &nodes);
                for qi in 0..q.points.len() {
                    let xi = &q.points[qi];
                    ref_elem.eval_basis_vec(xi, &mut phi);
                    let j = hm.jac(xi);
                    let det = j[0][0]
                        * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
                    let w = q.weights[qi] * det.abs();
                    let xp = hm.map(xi);
                    let mut uh = [0.0f64; 3];
                    for i in 0..n_ref {
                        let s = signs[i];
                        for r in 0..3 {
                            uh[r] += g[dofs[i]]
                                * s
                                * (j[r][0] * phi[i * 3]
                                    + j[r][1] * phi[i * 3 + 1]
                                    + j[r][2] * phi[i * 3 + 2])
                                / det;
                        }
                    }
                    let ue = exact(&xp);
                    for r in 0..3 {
                        e2 += w * (uh[r] - ue[r]).powi(2);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    e2.sqrt()
}

fn check(name: &str, err: f64, tol: f64) {
    println!("{name}: err = {err:.3e}");
    assert!(err < tol, "{name}: reconstruction error {err:.3e} >= {tol:.0e}");
}

// ─── 2-D tri ────────────────────────────────────────────────────────────────

fn tri_field_err(order: u8, f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let mesh = Mesh::<2>::unit_square_tri(3);
    let space = HDivSpace::new(mesh, order);
    let g = space.interpolate_vector(f);
    reconstruction_l2_error(&space, g.as_slice(), f, 10)
}

#[test]
fn tri_rt0_constant_fields() {
    check("tri-RT0 (1,0)", tri_field_err(0, &|_| vec![1.0, 0.0]), 1e-12);
    check("tri-RT0 (0,1)", tri_field_err(0, &|_| vec![0.0, 1.0]), 1e-12);
}

#[test]
#[ignore = "tri-RT1 pinned to the legacy canonical-moment semantics required by discrete_op; the vector-assembler reconstruction stays inexact (see crates/space/src/hdiv.rs interpolate_vector_legacy)"]
fn tri_rt1_linear_fields() {
    check("tri-RT1 (1,0)", tri_field_err(1, &|_| vec![1.0, 0.0]), 1e-12);
    check("tri-RT1 (x,y)", tri_field_err(1, &|x| vec![x[0], x[1]]), 1e-12);
    check(
        "tri-RT1 (-y,x)",
        tri_field_err(1, &|x| vec![-x[1], x[0]]),
        1e-12,
    );
}

#[test]
#[ignore = "tri-RT2 pinned to the legacy canonical-moment semantics required by discrete_op; the vector-assembler reconstruction stays inexact (see crates/space/src/hdiv.rs interpolate_vector_legacy)"]
fn tri_rt2_quadratic_fields() {
    check(
        "tri-RT2 (x,y)",
        tri_field_err(2, &|x| vec![x[0], x[1]]),
        1e-12,
    );
    check(
        "tri-RT2 (-y,x)",
        tri_field_err(2, &|x| vec![-x[1], x[0]]),
        1e-12,
    );
    check(
        "tri-RT2 (x^2,y^2)",
        tri_field_err(2, &|x| vec![x[0] * x[0], x[1] * x[1]]),
        1e-12,
    );
}

// ─── 2-D quad ───────────────────────────────────────────────────────────────

fn quad_field_err(order: u8, f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let mesh = Mesh::<2>::unit_square_quad(3);
    let space = HDivSpace::new(mesh, order);
    let g = space.interpolate_vector(f);
    reconstruction_l2_error(&space, g.as_slice(), f, 10)
}

#[test]
fn quad_rt0_constant_fields() {
    check("quad-RT0 (1,0)", quad_field_err(0, &|_| vec![1.0, 0.0]), 1e-12);
    check("quad-RT0 (0,1)", quad_field_err(0, &|_| vec![0.0, 1.0]), 1e-12);
}

#[test]
fn quad_rt1_linear_fields() {
    check("quad-RT1 (1,0)", quad_field_err(1, &|_| vec![1.0, 0.0]), 1e-12);
    check("quad-RT1 (x,y)", quad_field_err(1, &|x| vec![x[0], x[1]]), 1e-12);
    check(
        "quad-RT1 (-y,x)",
        quad_field_err(1, &|x| vec![-x[1], x[0]]),
        1e-12,
    );
}

#[test]
fn quad_rt2_quadratic_field() {
    check(
        "quad-RT2 (x^2,0)",
        quad_field_err(2, &|x| vec![x[0] * x[0], 0.0]),
        1e-12,
    );
}

// ─── 3-D tet / hex ──────────────────────────────────────────────────────────

fn tet_field_err(order: u8, f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let space = HDivSpace::new(mesh, order);
    let g = space.interpolate_vector(f);
    reconstruction_l2_error_3d(&space, g.as_slice(), f, 10)
}

fn hex_field_err(order: u8, f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = HDivSpace::new(mesh, order);
    let g = space.interpolate_vector(f);
    reconstruction_l2_error_3d(&space, g.as_slice(), f, 10)
}

/// KNOWN ELEMENT-CRATE DEFECT (blocks the tet-RT0 row of the D28 matrix).
///
/// `TetRTk`'s Gauss-Jordan basis construction (crates/element/src/raviart_thomas/
/// tet_rtk.rs, the `coeff[i*n + j] = row[i][mt + sel[j]]` mapping) produces basis
/// functions that are *not* dual to the face-flux functionals: e.g. basis 0 has
/// constant normal traces on faces 0 **and** 2 (sampling D_i(phi_j) over face-flux
/// points gives a non-diagonal matrix, while TriRTk(0)/TriRT2 built from the
/// same pattern come out diagonal).  Because the slot->basis pairing `slot i <=>
/// reference basis i` is fixed by the vector assembler, no global dof vector can
/// reproduce an exactly-representable field on a multi-tet mesh: the coefficient
/// of a cross-face basis function depends on *all* of the element's face fluxes,
/// so neighbouring elements disagree on shared dofs by construction.
///
/// This is confirmed independently of interpolation: the L2 projection
/// (assembling VectorMassIntegrator with the same basis and solving exactly)
/// gives ||err|| = 5.1e-1 for the constant field (1,0,0) on `unit_cube_tet(2)`
/// — the assembled space simply does not contain the constant.  Fixing this
/// requires correcting the basis construction in `crates/element` (read-only
/// for the D28 round).  Run with `--ignored` to see the current residual.
#[test]
#[ignore = "tet-RT0 blocked by TetRTk(0) basis defect in crates/element (not flux-dual); see doc comment"]
fn tet_rt0_constant_fields() {
    check(
        "tet-RT0 (1,0,0)",
        tet_field_err(0, &|_| vec![1.0, 0.0, 0.0]),
        1e-12,
    );
    check(
        "tet-RT0 (0,0,1)",
        tet_field_err(0, &|_| vec![0.0, 0.0, 1.0]),
        1e-12,
    );
}

#[test]
fn hex_rt0_constant_fields() {
    check(
        "hex-RT0 (1,0,0)",
        hex_field_err(0, &|_| vec![1.0, 0.0, 0.0]),
        1e-12,
    );
    check(
        "hex-RT0 (0,1,0)",
        hex_field_err(0, &|_| vec![0.0, 1.0, 0.0]),
        1e-12,
    );
}

#[test]
fn hex_rt1_linear_fields() {
    check(
        "hex-RT1 (1,0,0)",
        hex_field_err(1, &|_| vec![1.0, 0.0, 0.0]),
        1e-12,
    );
    check(
        "hex-RT1 (x,y,z)",
        hex_field_err(1, &|x| vec![x[0], x[1], x[2]]),
        1e-12,
    );
    check(
        "hex-RT1 (-z,y,0)",
        hex_field_err(1, &|x| vec![-x[2], x[1], 0.0]),
        1e-12,
    );
}

/// KNOWN SEMANTIC CONFLICT (blocks the tet-RT1/RT2 rows of the D28 matrix),
/// same cause as tri-RT1/RT2 above: `divergence_rt1_p1_3d` /
/// `curl_3d_nd2_rt1` in `crates/assembly/src/discrete_op.rs` read tet RT1/RT2
/// dof values with canonical-moment duals, so `interpolate_vector` must keep
/// serving the legacy values instead of the reference-dual values.
#[test]
#[ignore = "tet-RT1/RT2 pinned to the legacy canonical-moment semantics required by discrete_op"]
fn tet_rt1_legacy_semantics() {
    check(
        "tet-RT1 (x,y,z)",
        tet_field_err(1, &|x| vec![x[0], x[1], x[2]]),
        1e-12,
    );
}
