//! spde_gaussian_field — sample and evaluate a 2D Gaussian random field.
//!
//! Generates a random field on a triangle mesh via KL expansion.
//! Analogous to MFEM miniapp `spde`.
//!
//! Usage:
//!   cargo run --example spde_gaussian_field

use fem_mesh::{Mesh, topology::MeshTopology};
use fem_stochastic::{
    SquaredExponentialCovariance2D, KarhunenLoeveExpansion2D, RandomField2D,
};

fn main() {
    let mesh = Mesh::<2>::unit_square_tri(20);
    let pts: Vec<[f64; 2]> = (0..mesh.n_nodes() as usize).map(|i| {
        let c = mesh.node_coords(i as u32);
        [c[0], c[1]]
    }).collect();

    let cov = SquaredExponentialCovariance2D { sigma2: 0.1, length: 0.3 };
    let kl = KarhunenLoeveExpansion2D::new(32, 32, 16, 0.0, &cov);

    use rand::SeedableRng;
    let field = kl.realisation(&pts, &mut rand::rngs::StdRng::from_seed([0u8; 32]));

    let n = field.len();
    let mean = field.iter().sum::<f64>() / n as f64;
    let variance = field.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let min_v = field.iter().fold(f64::MAX, |a, &v| a.min(v));
    let max_v = field.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));

    println!("=== spde_gaussian_field: 2D KL random field ===");
    println!("  Nodes: {n}, KL modes: 16");
    println!("  σ²=0.1, correlation length=0.3");
    println!("  Mean={mean:.6e}, Variance={variance:.6e}");
    println!("  Min={min_v:.6e}, Max={max_v:.6e}");
}

#[cfg(test)]
mod tests {
    use fem_mesh::{Mesh, topology::MeshTopology};
    use fem_stochastic::{SquaredExponentialCovariance2D, KarhunenLoeveExpansion2D, RandomField2D};

    #[test]
    fn spde_field_is_finite() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let pts: Vec<[f64; 2]> = (0..mesh.n_nodes() as usize).map(|i| {
            let c = mesh.node_coords(i as u32); [c[0], c[1]]
        }).collect();
        let cov = SquaredExponentialCovariance2D { sigma2: 0.1, length: 0.3 };
        let kl = KarhunenLoeveExpansion2D::new(16, 16, 8, 0.0, &cov);
        let field = kl.realisation(&pts, &mut rand::rngs::StdRng::from_seed([0u8; 32]));
        assert_eq!(field.len(), pts.len());
        assert!(field.iter().all(|x| x.is_finite()));
    }
}
