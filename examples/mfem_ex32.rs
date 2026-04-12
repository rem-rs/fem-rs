use fem_amg::AmgConfig;
use fem_examples::maxwell::{assemble_hcurl_eigen_system_from_marker, solve_hcurl_eigen_preconditioned_amg};
use fem_mesh::SimplexMesh;
use fem_solver::{LobpcgConfig, SolverConfig};
use fem_space::{H1Space, HCurlSpace};

fn main() {
	let n = 8;
	let h1 = H1Space::new(SimplexMesh::<2>::unit_square_tri(n), 1);
	let hcurl = HCurlSpace::new(SimplexMesh::<2>::unit_square_tri(n), 1);
	let attrs = [1, 2, 3, 4];
	let ess_bdr = [1, 1, 1, 1];
	let eig_system = assemble_hcurl_eigen_system_from_marker(&h1, &hcurl, &attrs, &ess_bdr, 1.0, 1.0, 4);

	let eig_cfg = LobpcgConfig { max_iter: 800, tol: 1e-8, verbose: false };
	let inner_cfg = SolverConfig {
		rtol: 1e-2,
		atol: 1e-12,
		max_iter: 20,
		verbose: false,
		..SolverConfig::default()
	};

	let result = solve_hcurl_eigen_preconditioned_amg(
		&eig_system,
		3,
		&eig_cfg,
		AmgConfig::default(),
		&inner_cfg,
	).expect("mfem_ex32 eigen solve failed");

	println!("mfem_ex32 done: converged={}, iters={}, eigs={:?}", result.converged, result.iterations, result.eigenvalues);
}
