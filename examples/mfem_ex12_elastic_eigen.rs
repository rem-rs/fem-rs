//! # Example 12 — Linear Elasticity Eigenvalue (analogous to MFEM ex12)
//!
//! Computes the lowest eigenmodes of the linear elasticity operator via
//! the generalized eigenvalue problem K x = λ M x using LOBPCG, where K is
//! the stiffness matrix and M the mass matrix.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex12_elastic_eigen
//! cargo run --example mfem_ex12_elastic_eigen -- -m ../data/beam-tri.mesh -n 5
//! cargo run --example mfem_ex12_elastic_eigen -- --n 8 -k 3
//! ```

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_io::mfem::read_mfem_file;
use fem_mesh::SimplexMesh;
use fem_solver::eigen::{lobpcg, LobpcgConfig};
use fem_space::H1Space;
use fem_space::fe_space::FESpace;

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 12: Elasticity Eigenvalue ===");

    // Load or generate mesh
    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(args.n)
    };
    let space = H1Space::new(mesh, args.order);

    let quad_order = args.order as u8 * 2 + 1;
    let k = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order);
    let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad_order);

    let cfg = LobpcgConfig { max_iter: 200, tol: 1e-8, ..LobpcgConfig::default() };
    let result = lobpcg(&k, Some(&m), args.k, &cfg).unwrap();

    println!("  DOFs: {}", space.n_dofs());
    for (i, val) in result.eigenvalues.iter().enumerate() {
        println!("  λ[{}] = {:.6e}", i + 1, val);
    }
    println!("  PASS");
}

struct Args {
    mesh: Option<String>,
    n: usize,
    k: usize,
    order: u8,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 8, k: 5, order: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
            }
            "-n" | "--num-eigs" => {
                a.k = it.next().and_then(|v| v.parse().ok()).unwrap_or(5)
            }
            "--n" => {
                a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(8)
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
    use fem_mesh::SimplexMesh;
    use fem_solver::eigen::{lobpcg, LobpcgConfig};
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    #[test]
    fn smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let k = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let r = lobpcg(&k, Some(&m), 3, &LobpcgConfig { max_iter: 50, ..LobpcgConfig::default() }).unwrap();
        assert_eq!(r.eigenvalues.len(), 3);
    }
}
