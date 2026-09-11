//! D57 regression: the ex3 2-D (`beam-tri`-class) pipeline must match MFEM.
//!
//! Two defects were found and fixed in round 18:
//!
//! 1. **L2 evaluator transpose** (`l2_err_2d` in the ex3 example): the 2-D
//!    covariant Piola map applied `J^{-1}` where it needs `J^{-T}` — the
//!    off-diagonal entries of the adjugate were not transposed.  Invisible on
//!    right triangles with diagonal Jacobians, wrong on any sheared triangle.
//! 2. **The `solve_report_2d` re-assembly**: it built a *fresh, un-eliminated*
//!    matrix and solved it against the *eliminated* right-hand side — a
//!    different (wrong-BC) linear system whose solution carries an O(10) L2
//!    error on the refined beam mesh.
//!
//! This test pins the whole 2-D pipeline (assembly -> elimination -> PCG ->
//! reconstruction -> L2) on the 2-triangle unit square and its uniform
//! refinements against the C++ `probe2d` harness (MFEM 4.10, `ND1`,
//! `E = (sin(ky), sin(kx))`, `f = (1+k^2) E`):
//!
//! ```text
//! rf0 (n=5):    interp L2 0.5281841004551823   solve L2 0.4916776270584529
//! rf3 (n=208):  RAW  A_fro^2 24682563.55555561 tr 49258.66666666665 b_l2 6.745099151146084
//!               ELIM A_fro^2 22653505.88888892 tr 49258.66666666665 B_l2 6.710379638837318
//!               interp L2 0.1129127648371693   solve L2 0.1128343119531691
//! ```

use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::{
    accumulate_vector_bilinear_element_blocks, nd_element_local_dofs, VectorAssembler,
};
use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_core::NodeId;
use fem_element::VectorReferenceElement;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_space::constraints::{boundary_dofs_hcurl, form_linear_system};
use fem_space::{FESpace, HCurlSpace};

const KAPPA: f64 = std::f64::consts::PI;

fn e_exact(x: &[f64]) -> [f64; 2] {
    [(KAPPA * x[1]).sin(), (KAPPA * x[0]).sin()]
}

struct Src2D {
    kappa: f64,
}
impl VectorLinearIntegrator for Src2D {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let c = 1.0 + self.kappa * self.kappa;
        let fx = c * (self.kappa * x[1]).sin();
        let fy = c * (self.kappa * x[0]).sin();
        for i in 0..qp.n_dofs {
            f[i] += qp.weight * (qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy);
        }
    }
}

/// The unit square `[0,1]^2` split into two triangles ((0,1,2) and (0,2,3)).
fn square2() -> Mesh<2> {
    let coords = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let nid = |i: usize| -> NodeId { i as NodeId };
    let conn: Vec<NodeId> = vec![nid(0), nid(1), nid(2), nid(0), nid(2), nid(3)];
    let elem_tags = vec![1i32, 1];
    let face_conn: Vec<NodeId> =
        vec![nid(0), nid(1), nid(1), nid(2), nid(2), nid(3), nid(3), nid(0)];
    let face_tags = vec![1i32, 1, 1, 1];
    Mesh::<2>::uniform(
        coords,
        conn,
        elem_tags,
        ElementType::Tri3,
        face_conn,
        face_tags,
        ElementType::Line2,
    )
}

fn fro_tr(mat: &fem_linalg::CsrMatrix<f64>) -> (f64, f64) {
    let mut fro = 0.0_f64;
    for v in mat.values.iter() {
        fro += v * v;
    }
    let mut tr = 0.0_f64;
    for d in 0..mat.nrows {
        for k in mat.row_ptr[d]..mat.row_ptr[d + 1] {
            if mat.col_idx[k] as usize == d {
                tr += mat.values[k];
            }
        }
    }
    (fro, tr)
}

fn l2_err_2d(mesh: &Mesh<2>, sp: &HCurlSpace<Mesh<2>>, u: &[f64]) -> f64 {
    use fem_element::nedelec::TriNDk;
    let k = sp.order() as usize;
    let mut e2 = 0.0;
    for e in mesh.elem_iter() {
        let r = TriNDk::new(k);
        let n = r.n_dofs();
        let q = r.quadrature((2 * k + 3) as u8);
        let mut p = vec![0.0; n * 2];
        let uloc = nd_element_local_dofs(sp, e, u);
        let nd = mesh.elem_nodes(e);
        let x0 = mesh.node_coords(nd[0]);
        let x1 = mesh.node_coords(nd[1]);
        let x2 = mesh.node_coords(nd[2]);
        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let det_j = j00 * j11 - j01 * j10;
        let inv_det = 1.0 / det_j;
        // Covariant Piola: J^{-T} = [[j11, -j10], [-j01, j00]] / det — the
        // transpose of the adjugate (D57: J^{-1} was applied here before).
        let jt00 = j11 * inv_det;
        let jt01 = -j10 * inv_det;
        let jt10 = -j01 * inv_det;
        let jt11 = j00 * inv_det;
        for (qi, xi) in q.points.iter().enumerate() {
            r.eval_basis_vec(xi, &mut p);
            let w = q.weights[qi] * det_j.abs();
            let mut uh = [0.0; 2];
            for a in 0..n {
                uh[0] += uloc[a] * (jt00 * p[a * 2] + jt01 * p[a * 2 + 1]);
                uh[1] += uloc[a] * (jt10 * p[a * 2] + jt11 * p[a * 2 + 1]);
            }
            let xp = [
                (1.0 - xi[0] - xi[1]) * x0[0] + xi[0] * x1[0] + xi[1] * x2[0],
                (1.0 - xi[0] - xi[1]) * x0[1] + xi[0] * x1[1] + xi[1] * x2[1],
            ];
            let ex = e_exact(&xp);
            e2 += w * ((uh[0] - ex[0]).powi(2) + (uh[1] - ex[1]).powi(2));
        }
    }
    e2.sqrt()
}

#[test]
fn square2d_nd1_pipeline_matches_mfem() {
    let mut mesh = square2();
    for _ in 0..3 {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }
    let space = HCurlSpace::new(mesh, 1);
    assert_eq!(space.n_dofs(), 208);

    let tags = space.mesh().unique_boundary_tags();
    let ess_bdr = boundary_dofs_hcurl(space.mesh(), &space, &tags);
    let qo = 2u8;
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&Src2D { kappa: KAPPA }], qo);
    let u_proj = space.interpolate_vector(&|x| e_exact(x).to_vec()).into_vec();
    let bc_vals: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d as usize]).collect();

    let mut mat = {
        let n = space.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        let cc = CurlCurlIntegrator { mu: 1.0 };
        let mass = VectorMassIntegrator { alpha: 1.0 };
        for e in 0..space.mesh().n_elements() as u32 {
            accumulate_vector_bilinear_element_blocks(&space, e, &[&cc], qo, &mut coo, &[]);
            accumulate_vector_bilinear_element_blocks(&space, e, &[&mass], 2 * qo + 3, &mut coo, &[]);
        }
        coo.into_csr()
    };
    let (fro, tr) = fro_tr(&mat);
    assert!((fro - 24682563.55555561).abs() < 1e-7 * fro, "RAW fro {fro}");
    assert!((tr - 49258.66666666665).abs() < 1e-9 * tr, "RAW tr {tr}");

    let mut x = u_proj.clone();
    form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bc_vals);
    let (fro, tr) = fro_tr(&mat);
    let bl2: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((fro - 22653505.88888892).abs() < 1e-7 * fro, "ELIM fro {fro}");
    assert!((bl2 - 6.710379638837318).abs() < 1e-9 * bl2, "B_l2 {bl2}");

    let precond = fem_solver::GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&mat)).unwrap();
    match fem_solver::solve_pcg(&mat, &rhs, &mut x, &precond, 1e-12, 500, false) {
        Ok(_) => {}
        Err(_) => panic!("PCG failed to converge"),
    }
    let l2 = l2_err_2d(space.mesh(), &space, &x);
    assert!(
        (l2 - 0.1128343119531691).abs() < 1e-9,
        "solve L2 {l2} vs C++ 0.1128343119531691"
    );
}
