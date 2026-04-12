use fem_examples::{apply_dirichlet, dirichlet_nodes, p1_assemble_poisson, pcg_solve};
use fem_mesh::SimplexMesh;

fn main() {
	let n = 16;
	let mesh = SimplexMesh::<2>::unit_square_tri(n);

	let src = |x: f64, y: f64| {
		if (0.3..=0.7).contains(&x) && (0.3..=0.7).contains(&y) {
			1.0e3
		} else {
			0.0
		}
	};
	let (mut k, mut rhs) = p1_assemble_poisson(&mesh, |_, _| 1.0, src);
	let bcs = dirichlet_nodes(&mesh, &[1, 2, 3, 4]);
	apply_dirichlet(&mut k, &mut rhs, &bcs);

	let (u, iters, res) = pcg_solve(&k, &rhs, 1e-10, 5000);
	let l2 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
	println!("mfem_tesla done: dofs={}, iters={}, res={:.3e}, ||Az||2={:.3e}", u.len(), iters, res, l2);
}
