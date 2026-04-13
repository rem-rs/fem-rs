use fem_examples::{apply_dirichlet, dirichlet_nodes_fn, p1_assemble_poisson, pcg_solve};
use fem_mesh::SimplexMesh;

fn main() {
	let n = 16;
	let mesh = SimplexMesh::<2>::unit_square_tri(n);

	let (mut k, mut rhs) = p1_assemble_poisson(&mesh, |_, _| 1.0, |_, _| 0.0);
	let bcs = dirichlet_nodes_fn(&mesh, &[1, 3], |_, y| y);
	apply_dirichlet(&mut k, &mut rhs, &bcs);

	let (u, iters, res) = pcg_solve(&k, &rhs, 1e-10, 5000);
	println!("mfem_volta done: dofs={}, iters={}, res={:.3e}, u_min={:.3e}, u_max={:.3e}", u.len(), iters, res, u.iter().fold(f64::INFINITY, |a, &v| a.min(v)), u.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)));
}
