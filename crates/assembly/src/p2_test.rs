//! Quick test: verify P2 Jacobian = P1 for a flat triangle
#[cfg(test)]
mod tests {
    use fem_element::lagrange::factory::TriPk;
    use fem_element::ReferenceElement;
    use fem_element::lagrange::TriP1;

    #[test]
    fn p2_jacobian_matches_p1_for_flat_tri() {
        let p2 = TriPk::new(2);
        let p1 = TriP1;
        let dim = 2;
        let embed = 3;
        let xi = &[1.0/3.0, 1.0/3.0];
        let nodes: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ];
        
        let mut gp = vec![0.0; p2.n_dofs() * dim];
        p2.eval_grad_basis(xi, &mut gp);
        
        let mut j = vec![0.0; embed * dim];
        for k in 0..p2.n_dofs() {
            for i in 0..embed {
                for d in 0..dim {
                    j[i + d * embed] += nodes[k][i] * gp[k * dim + d];
                }
            }
        }
        
        // Reference: P1 Jacobian for triangle (0,0)-(1,0)-(0,1)
        let j_ref = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (a, b) in j.iter().zip(j_ref.iter()) {
            assert!((a-b).abs() < 1e-14, "P2 Jacobian differs: {:.6e} vs {:.6e}", a, b);
        }
    }
}
