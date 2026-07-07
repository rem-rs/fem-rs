//! # Example 24 — Mixed Discrete Operators (analogous to MFEM ex24)
//!
//! Projects gradient / 2D-curl / divergence operators via `DiscreteLinearOperator`
//! on mixed FE spaces.  Three problem types:
//!
//! ```text
//!   0 (grad): grad p       for p ∈ H¹        → E ∈ H(curl)
//!   1 (curl): curl v       for v ∈ H(curl)   → c ∈ L²   (2-D scalar curl)
//!   2 (div):  div v        for v ∈ H(div)    → f ∈ L²
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex24_discrete_ops
//! cargo run --example mfem_ex24_discrete_ops -- -m ../data/star.mesh -p 0
//! cargo run --example mfem_ex24_discrete_ops -- -p 2 --n 8 --order 2
//! ```

use std::f64::consts::PI;

use fem_assembly::DiscreteLinearOperator;
use fem_io::mfem::read_mfem_file;
use fem_mesh::SimplexMesh;
use fem_space::{
    H1Space, HCurlSpace, HDivSpace, L2Space,
    fe_space::FESpace,
};

fn main() {
    let args = parse_args();
    println!("=== Example 24: Mixed Discrete Operators (MFEM ex24) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Mesh: {}×{} P{}", args.n, args.n, args.order);
    }
    println!(
        "  Problem type: {} ({})",
        args.prob,
        match args.prob {
            0 => "grad: H¹→H(curl)",
            1 => "curl: H(curl)→L²  (2-D scalar curl)",
            _ => "div: H(div)→L²",
        }
    );

    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(args.n)
    };

    match args.prob {
        0 => run_grad(&mesh, args.order),
        1 => run_curl(&mesh, args.order),
        2 => run_div(&mesh, args.order),
        _ => eprintln!("problem must be 0 (grad), 1 (curl), or 2 (div)"),
    }
}

/// Grad: trial = H¹, test = H(curl).  Interpolate p = sin(πx)sin(πy),
/// then apply the gradient discrete operator.
fn run_grad(mesh: &SimplexMesh<2>, p: u8) {
    let trial = H1Space::new(mesh.clone(), p);
    let test = HCurlSpace::new(mesh.clone(), p);

    let p_vec: Vec<f64> = trial
        .interpolate(&|x| (PI * x[0]).sin() * (PI * x[1]).sin())
        .as_slice()
        .to_vec();

    let grad_mat = DiscreteLinearOperator::gradient(&trial, &test)
        .expect("gradient DLO assembly failed");
    let mut e = vec![0.0; test.n_dofs()];
    grad_mat.spmv(&p_vec, &mut e);

    let enrm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!(
        "  ∇: H¹({p})→H(curl)({p})  DOFs {}→{}, ‖∇p_h‖₂ = {:.4e}",
        trial.n_dofs(),
        test.n_dofs(),
        enrm
    );
}

/// Curl: trial = H(curl), test = L².  Scalar curl in 2D.
fn run_curl(mesh: &SimplexMesh<2>, p: u8) {
    let trial = HCurlSpace::new(mesh.clone(), p);
    let l2_order = if p <= 1 { 0 } else { p - 1 };
    let test = L2Space::new(mesh.clone(), l2_order);

    let v_vec: Vec<f64> = trial
        .interpolate_vector(&|x| vec![(PI * x[1]).sin(), (PI * x[0]).sin()])
        .as_slice()
        .to_vec();

    match DiscreteLinearOperator::curl_2d(&trial, &test) {
        Ok(curl_mat) => {
            let mut cv = vec![0.0; test.n_dofs()];
            curl_mat.spmv(&v_vec, &mut cv);
            let cnrm: f64 = cv.iter().map(|v| v * v).sum::<f64>().sqrt();
            println!(
                "  ∇×: H(curl)({p})→L²({l2_order})  DOFs {}→{}, ‖curl v_h‖₂ = {:.4e}",
                trial.n_dofs(),
                test.n_dofs(),
                cnrm
            );
        }
        Err(e) => println!("  ∇×: not available: {e}"),
    }
}

/// Div: trial = H(div), test = L².
fn run_div(mesh: &SimplexMesh<2>, p: u8) {
    let trial = HDivSpace::new(mesh.clone(), p);
    let test = L2Space::new(mesh.clone(), p);

    let v_vec: Vec<f64> = trial
        .interpolate_vector(&|x| {
            vec![(PI * x[0]).sin() * (PI * x[1]).sin(); 2]
        })
        .as_slice()
        .to_vec();

    let div_mat = DiscreteLinearOperator::divergence(&trial, &test)
        .expect("divergence DLO assembly failed");
    let mut div_v = vec![0.0; test.n_dofs()];
    div_mat.spmv(&v_vec, &mut div_v);
    let dnrm: f64 = div_v.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!(
        "  ∇·: H(div)({p})→L²({p})  DOFs {}→{}, ‖div v_h‖₂ = {:.4e}",
        trial.n_dofs(),
        test.n_dofs(),
        dnrm
    );
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
    prob: u8,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 4,
        order: 1,
        prob: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => {
                a.n = it
                    .next()
                    .unwrap_or("4".into())
                    .parse()
                    .unwrap_or(4)
            }
            "-o" | "--order" => {
                a.order = it
                    .next()
                    .unwrap_or("1".into())
                    .parse()
                    .unwrap_or(1)
            }
            "-p" | "--problem" => {
                a.prob = it
                    .next()
                    .unwrap_or("0".into())
                    .parse()
                    .unwrap_or(0)
            }
            _ => {}
        }
    }
    a
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex24_grad_operator_produces_finite_gradient() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        run_grad(&mesh, 1);
    }

    #[test]
    fn ex24_curl_operator_produces_finite_curl() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        run_curl(&mesh, 1);
    }

    #[test]
    fn ex24_div_operator_produces_finite_divergence() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        run_div(&mesh, 1);
    }
}
