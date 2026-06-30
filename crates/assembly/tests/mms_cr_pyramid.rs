//! MMS convergence tests for Crouzeix-Raviart CR1 and PyraND1 elements.
//!
//! CR1 on triangles: projection convergence O(h^2) in L2.
//! PyraND1 on pyramids: element matrix symmetry + positive diagonal.

use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_element::ReferenceElement;

fn convergence_rate(errors: &[f64], ns: &[usize]) -> Vec<f64> {
    (0..errors.len() - 1)
        .map(|i| (errors[i] / errors[i + 1]).ln()
              / (ns[i + 1] as f64 / ns[i] as f64).ln())
        .collect()
}

// ─── CR1 element: interpolation at edge midpoints ──────────────────────────

fn cr1_interpolation_error(n: usize) -> f64 {
    use fem_element::crouzeix_raviart::cr1_basis;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let tri_ref: &dyn ReferenceElement = &fem_element::lagrange::TriP1;
    let mut err_sq = 0.0_f64;
    let quad = tri_ref.quadrature(4);
    let mut phi = [0.0_f64; 3];
    let mut grad = [0.0_f64; 6];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let n0 = mesh.node_coords(nodes[0]);
        let n1 = mesh.node_coords(nodes[1]);
        let n2 = mesh.node_coords(nodes[2]);
        let det_j = ((n1[0]-n0[0])*(n2[1]-n0[1]) - (n1[1]-n0[1])*(n2[0]-n0[0])).abs();

        // Edge midpoints in reference coords
        let ref_mid = [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]];
        // Interpolate f at edge midpoints
        let mut f_mid = [0.0_f64; 3];
        for i in 0..3 {
            let xi = ref_mid[i][0]; let eta = ref_mid[i][1];
            let x = n0[0] + (n1[0]-n0[0])*xi + (n2[0]-n0[0])*eta;
            let y = n0[1] + (n1[1]-n0[1])*xi + (n2[1]-n0[1])*eta;
            f_mid[i] = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
        }

        for q in 0..quad.n_points() {
            let xi = &quad.points[q];
            let w = quad.weights[q] * det_j;
            cr1_basis(xi, &mut phi);
            tri_ref.eval_grad_basis(xi, &mut grad);

            let x = n0[0] + (n1[0]-n0[0])*xi[0] + (n2[0]-n0[0])*xi[1];
            let y = n0[1] + (n1[1]-n0[1])*xi[0] + (n2[1]-n0[1])*xi[1];
            let f_exact = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
            let f_h = f_mid[0]*phi[0] + f_mid[1]*phi[1] + f_mid[2]*phi[2];
            err_sq += w * (f_h - f_exact).powi(2);
        }
    }
    err_sq.sqrt()
}

#[test]
fn cr1_projection_convergence() {
    let ns = [4usize, 8, 16];
    let errors: Vec<f64> = ns.iter().map(|&n| cr1_interpolation_error(n)).collect();
    let rates = convergence_rate(&errors, &ns);
    eprintln!("CR1 projection: errors={:?}, rates={:?}", errors, rates);
    assert!(errors[1] < errors[0], "CR1 error should decrease 4->8");
    assert!(errors[2] < errors[1], "CR1 error should decrease 8->16");
    assert!(rates[0] > 1.5, "CR1 rate[0]={:.3} < 1.5 (expected ~2)", rates[0]);
}

// ─── PyraND1 element matrix: symmetry and positivity ──────────────────────

#[test]
fn pyrand1_element_matrix_symmetry_and_positivity() {
    use fem_element::lagrange::factory::{vec_ref_elem, VecFamily, ElemType};
    let ref_elem = vec_ref_elem(VecFamily::Nedelec, ElemType::Pyramid, 1u8);
    assert_eq!(ref_elem.n_dofs(), 8);
    let n = 8;
    let mut vals = vec![0.0; n * 3];
    let mut curls = vec![0.0; n * 3];
    let mut ke = vec![0.0_f64; n * n];
    let quad = ref_elem.quadrature(3);
    for q in 0..quad.n_points() {
        let w = quad.weights[q];
        ref_elem.eval_basis_vec(&quad.points[q], &mut vals);
        ref_elem.eval_curl(&quad.points[q], &mut curls);
        for i in 0..n { for j in 0..n {
            let cc = (0..3).map(|d| curls[i*3+d] * curls[j*3+d]).sum::<f64>();
            let mm = (0..3).map(|d| vals[i*3+d] * vals[j*3+d]).sum::<f64>();
            ke[i * n + j] += w * (cc + mm);
        }}
    }
    for i in 0..n { for j in 0..n {
        assert!((ke[i*n+j] - ke[j*n+i]).abs() < 1e-12,
            "PyraND1 not symmetric at ({i},{j}): diff={}", (ke[i*n+j] - ke[j*n+i]).abs());
    }}
    for i in 0..n {
    assert!(ke[i*n+i] > 0.0, "PyraND1 diag {i} = {:.6e} <= 0", ke[i*n+i]);
    }
}
