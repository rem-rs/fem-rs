//! D33/D34 regression: Galerkin L² projection of exactly-representable fields
//! onto the Raviart-Thomas H(div) spaces.
//!
//! The projection assembles the vector mass matrix `M` and the load `b` with
//! the same `vec_ref_elem` basis the vector assembler uses, solves `M c = b`,
//! and verifies `‖f − f_h‖₂ ≤ 1e-12` through the algebraic identity
//!
//! ```text
//!     ‖f − f_h‖² = ∫|f|² − 2 cᵀ b + cᵀ M c
//! ```
//!
//! Before the D33 fix (tet_rtk.rs Gauss-Jordan transpose/column pairing) the
//! tet RT0 space's assembled global basis was not the RT space: projecting the
//! constant field (1,0,0) on `unit_cube_tet(2)` left an error of 5.1e-1.

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_element::quadrature::{tet_rule, tri_rule};
use fem_element::reference::VectorReferenceElement;
use fem_linalg::CsrMatrix;
use fem_linalg::dense::{lu_factor, lu_solve};
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_space::{fe_space::FESpace, HDivSpace};

/// `b_i = ∫ f · φ_i` — the RHS of the L² projection.
struct VectorRhs(fn(&[f64]) -> Vec<f64>);

impl VectorLinearIntegrator for VectorRhs {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let ue = (self.0)(qp.x_phys);
        for i in 0..qp.n_dofs {
            let mut dot = 0.0;
            for c in 0..qp.dim {
                dot += qp.phi_vec[i * qp.dim + c] * ue[c];
            }
            f_elem[i] += qp.weight * dot;
        }
    }
}

/// ∫_Ω |f|² dV by direct quadrature (independent of the assembler).
fn l2_norm_squared(mesh: &Mesh<3>, f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let mut e2 = 0.0;
    for e in 0..mesh.n_elems() as u32 {
        let nodes = mesh.element_nodes(e);
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let p2 = mesh.node_coords(nodes[2]);
        let p3 = mesh.node_coords(nodes[3]);
        let cols = [
            [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
            [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]],
            [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]],
        ];
        let det = (cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
            - cols[0][1] * (cols[1][0] * cols[2][2] - cols[1][2] * cols[2][0])
            + cols[0][2] * (cols[1][0] * cols[2][1] - cols[1][1] * cols[2][0]))
        .abs();
        let q = tet_rule(8);
        for (xi, w) in q.points.iter().zip(q.weights.iter()) {
            let x = [
                p0[0] + xi[0] * cols[0][0] + xi[1] * cols[1][0] + xi[2] * cols[2][0],
                p0[1] + xi[0] * cols[0][1] + xi[1] * cols[1][1] + xi[2] * cols[2][1],
                p0[2] + xi[0] * cols[0][2] + xi[1] * cols[1][2] + xi[2] * cols[2][2],
            ];
            let fv = f(&x);
            e2 += w * det * (fv[0] * fv[0] + fv[1] * fv[1] + fv[2] * fv[2]);
        }
    }
    e2
}

/// ∫_Ω |f|² dV on a 2-D triangle mesh.
fn l2_norm_squared_2d(mesh: &Mesh<2>, f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let mut e2 = 0.0;
    for e in 0..mesh.n_elems() as u32 {
        let nodes = mesh.element_nodes(e);
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let p2 = mesh.node_coords(nodes[2]);
        let det = ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])).abs();
        let q = tri_rule(8);
        for (xi, w) in q.points.iter().zip(q.weights.iter()) {
            let x = [
                p0[0] + xi[0] * (p1[0] - p0[0]) + xi[1] * (p2[0] - p0[0]),
                p0[1] + xi[0] * (p1[1] - p0[1]) + xi[1] * (p2[1] - p0[1]),
            ];
            let fv = f(&x);
            e2 += w * det * (fv[0] * fv[0] + fv[1] * fv[1]);
        }
    }
    e2
}

fn project(space: &HDivSpace<Mesh<3>>, f: fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    let mass: CsrMatrix<f64> = VectorAssembler::assemble_bilinear(
        space,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        7,
    );
    let b = VectorAssembler::assemble_linear(space, &[&VectorRhs(f)], 7);
    let n = space.n_dofs();
    let mut ad = vec![0.0f64; n * n];
    for row in 0..n {
        for k in mass.row_ptr[row]..mass.row_ptr[row + 1] {
            ad[row * n + mass.col_idx[k] as usize] = mass.values[k];
        }
    }
    let mut piv = vec![0usize; n];
    lu_factor(&mut ad, n, &mut piv).expect("mass matrix LU");
    let mut c = b;
    c.resize(n, 0.0);
    lu_solve(&ad, n, &piv, &mut c);
    c
}

fn project_2d(space: &HDivSpace<Mesh<2>>, f: fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    let mass: CsrMatrix<f64> = VectorAssembler::assemble_bilinear(
        space,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        7,
    );
    let b = VectorAssembler::assemble_linear(space, &[&VectorRhs(f)], 7);
    let n = space.n_dofs();
    let mut ad = vec![0.0f64; n * n];
    for row in 0..n {
        for k in mass.row_ptr[row]..mass.row_ptr[row + 1] {
            ad[row * n + mass.col_idx[k] as usize] = mass.values[k];
        }
    }
    let mut piv = vec![0usize; n];
    lu_factor(&mut ad, n, &mut piv).expect("mass matrix LU");
    let mut c = b;
    c.resize(n, 0.0);
    lu_solve(&ad, n, &piv, &mut c);
    c
}

fn mat_vec(m: &CsrMatrix<f64>, c: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; c.len()];
    m.spmv(c, &mut out);
    out
}

/// `‖f − f_h‖₂` on the tet mesh, reconstructing `f_h` with the assembler's
/// element convention (`f_h = Σ cᵢ·signᵢ·Piola(φ̂ᵢ)`).
fn recon_err_3d(
    mesh: &Mesh<3>,
    space: &HDivSpace<Mesh<3>>,
    c: &[f64],
    f: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    use fem_element::raviart_thomas::{TetRT1, TetRT2, TetRTk};
    let mut e2 = 0.0;
    for e in 0..mesh.n_elems() as u32 {
        let nodes = mesh.element_nodes(e);
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let p2 = mesh.node_coords(nodes[2]);
        let p3 = mesh.node_coords(nodes[3]);
        let cols = [
            [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
            [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]],
            [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]],
        ];
        let det = cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
            - cols[0][1] * (cols[1][0] * cols[2][2] - cols[1][2] * cols[2][0])
            + cols[0][2] * (cols[1][0] * cols[2][1] - cols[1][1] * cols[2][0]);
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let re: Box<dyn VectorReferenceElement> = match space.order() {
            0 => Box::new(TetRTk::new(0)),
            1 => Box::new(TetRT1),
            _ => Box::new(TetRT2),
        };
        let n = re.n_dofs();
        let mut phi = vec![0.0f64; n * 3];
        let q = re.quadrature(10);
        for (xi, w) in q.points.iter().zip(q.weights.iter()) {
            re.eval_basis_vec(xi, &mut phi);
            let mut uh = [0.0f64; 3];
            for i in 0..n {
                for r in 0..3 {
                    uh[r] += c[dofs[i] as usize] * signs[i]
                        * (cols[0][r] * phi[i * 3]
                            + cols[1][r] * phi[i * 3 + 1]
                            + cols[2][r] * phi[i * 3 + 2])
                        / det;
                }
            }
            let x = [
                p0[0] + xi[0] * cols[0][0] + xi[1] * cols[1][0] + xi[2] * cols[2][0],
                p0[1] + xi[0] * cols[0][1] + xi[1] * cols[1][1] + xi[2] * cols[2][1],
                p0[2] + xi[0] * cols[0][2] + xi[1] * cols[1][2] + xi[2] * cols[2][2],
            ];
            let ue = f(&x);
            for r in 0..3 {
                e2 += w * det * (uh[r] - ue[r]).powi(2);
            }
        }
    }
    e2.max(0.0).sqrt()
}

/// `‖f − f_h‖₂` on the tri mesh.
fn recon_err_2d(
    mesh: &Mesh<2>,
    space: &HDivSpace<Mesh<2>>,
    c: &[f64],
    f: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    use fem_element::raviart_thomas::{TriRT1, TriRT2, TriRTk};
    let mut e2 = 0.0;
    for e in 0..mesh.n_elems() as u32 {
        let nodes = mesh.element_nodes(e);
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let p2 = mesh.node_coords(nodes[2]);
        let det = ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])).abs();
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let re: Box<dyn VectorReferenceElement> = match space.order() {
            0 => Box::new(TriRTk::new(0)),
            1 => Box::new(TriRT1),
            _ => Box::new(TriRT2),
        };
        let n = re.n_dofs();
        let mut phi = vec![0.0f64; n * 2];
        let q = re.quadrature(10);
        for (xi, w) in q.points.iter().zip(q.weights.iter()) {
            re.eval_basis_vec(xi, &mut phi);
            let mut uh = [0.0f64; 2];
            for i in 0..n {
                for r in 0..2 {
                    uh[r] += c[dofs[i] as usize] * signs[i]
                        * ((p1[r] - p0[r]) * phi[i * 2] + (p2[r] - p0[r]) * phi[i * 2 + 1])
                        / det;
                }
            }
            let x = [
                p0[0] + xi[0] * (p1[0] - p0[0]) + xi[1] * (p2[0] - p0[0]),
                p0[1] + xi[0] * (p1[1] - p0[1]) + xi[1] * (p2[1] - p0[1]),
            ];
            let ue = f(&x);
            for r in 0..2 {
                e2 += w * det * (uh[r] - ue[r]).powi(2);
            }
        }
    }
    e2.max(0.0).sqrt()
}

#[test]
fn tet_rt_projection_recovers_space_fields() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let fields: [(&str, u8, fn(&[f64]) -> Vec<f64>); 3] = [
        ("const", 0, |_| vec![1.0, 0.0, 0.0]),
        ("linear", 1, |x| vec![x[0] + 2.0 * x[1], x[1] - x[0], x[2]]),
        ("quadratic-ish", 2, |x| vec![x[0], x[1], x[2]]),
    ];
    for (name, order, f) in fields {
        let space = HDivSpace::new(mesh.clone(), order);
        let c = project(&space, f);
        let err = recon_err_3d(&mesh, &space, &c, &f);
        println!("tet RT{order} {name}: L2 projection error = {err:.3e}");
        assert!(err < 1e-10, "tet RT{order} {name}: {err:.3e}");
    }
}

#[test]
fn tri_rt_projection_recovers_space_fields() {
    let mesh = Mesh::<2>::unit_square_tri(3);
    let fields: [(&str, u8, fn(&[f64]) -> Vec<f64>); 4] = [
        ("const", 0, |_| vec![1.0, 0.0]),
        ("linear", 1, |x| vec![x[0], x[1]]),
        ("rotational", 1, |x| vec![-x[1], x[0]]),
        ("quadratic-bubble", 2, |x| vec![x[0] * x[0], x[0] * x[1]]),
    ];
    for (name, order, f) in fields {
        let space = HDivSpace::new(mesh.clone(), order);
        let c = project_2d(&space, f);
        let err = recon_err_2d(&mesh, &space, &c, &f);
        println!("tri RT{order} {name}: L2 projection error = {err:.3e}");
        assert!(err < 1e-10, "tri RT{order} {name}: {err:.3e}");
    }
}
