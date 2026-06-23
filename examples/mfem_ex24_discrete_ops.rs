use std::f64::consts::PI;

use fem_assembly::DiscreteLinearOperator;
use fem_mesh::SimplexMesh;
use fem_space::{
    H1Space, HCurlSpace, HDivSpace, L2Space,
    fe_space::FESpace,
};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 24: Mixed Discrete Operators ===");

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("Mesh: {}×{}, {} elements", args.n, args.n, mesh.n_elems());

    let p = args.order;
    match args.problem {
        0 => run_grad(&mesh, p),
        1 => run_curl(&mesh, p),
        2 => run_div(&mesh, p),
        _ => eprintln!("problem must be 0 (grad), 1 (curl), or 2 (div)"),
    }
}

fn run_grad(mesh: &SimplexMesh<2>, p: u8) {
    let trial = H1Space::new(mesh.clone(), p);
    let test  = HCurlSpace::new(mesh.clone(), p);

    // Interpolate p = sin(πx) sin(πy)
    let p_vec: Vec<f64> = trial.interpolate(&|x| (PI * x[0]).sin() * (PI * x[1]).sin()).as_slice().to_vec();

    let grad_mat = DiscreteLinearOperator::gradient(&trial, &test).unwrap();
    let mut e = vec![0.0; test.n_dofs()];
    grad_mat.spmv(&p_vec, &mut e);

    println!("  ∇:  H¹({p})→H(curl)({p})  DOFs {}→{}, ‖E‖₂ = {:.4e}",
        trial.n_dofs(), test.n_dofs(),
        e.iter().map(|v| v * v).sum::<f64>().sqrt());
}

fn run_curl(mesh: &SimplexMesh<2>, p: u8) {
    let trial = HCurlSpace::new(mesh.clone(), p);
    // curl_2d maps H(curl) → L²  (scalar curl in 2D)
    let l2_order = if p == 1 { 0 } else { p - 1 };
    let test = L2Space::new(mesh.clone(), l2_order);

    let v_vec: Vec<f64> = trial.interpolate_vector(&|x| vec![(PI * x[1]).sin(), (PI * x[0]).sin()]).as_slice().to_vec();

    match DiscreteLinearOperator::curl_2d(&trial, &test) {
        Ok(curl_mat) => {
            let mut cv = vec![0.0; test.n_dofs()];
            curl_mat.spmv(&v_vec, &mut cv);
            println!("  ∇×: H(curl)({p})→L²({l2_order})  DOFs {}→{}, ‖curl v‖₂ = {:.4e}",
                trial.n_dofs(), test.n_dofs(),
                cv.iter().map(|v| v * v).sum::<f64>().sqrt());
        }
        Err(e) => println!("  ∇×: not available: {e}"),
    }
}

fn run_div(mesh: &SimplexMesh<2>, p: u8) {
    let trial = HDivSpace::new(mesh.clone(), p);
    let test  = L2Space::new(mesh.clone(), p);

    let v_vec: Vec<f64> = trial.interpolate_vector(
        &|x| vec![(PI * x[0]).sin() * (PI * x[1]).sin(); 2]
    ).as_slice().to_vec();

    let div_mat = DiscreteLinearOperator::divergence(&trial, &test).unwrap();
    let mut div_v = vec![0.0; test.n_dofs()];
    div_mat.spmv(&v_vec, &mut div_v);

    println!("  ∇·: H(div)({p})→L²({p})  DOFs {}→{}, ‖div v‖₂ = {:.4e}",
        trial.n_dofs(), test.n_dofs(),
        div_v.iter().map(|v| v * v).sum::<f64>().sqrt());
}

struct Args { n: usize, order: u8, problem: u8 }

fn parse_args() -> Args {
    let mut a = Args { n: 4, order: 1, problem: 0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"     => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(4); }
            "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-p" | "--problem" => { a.problem = it.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
            _ => {}
        }
    }
    a
}
