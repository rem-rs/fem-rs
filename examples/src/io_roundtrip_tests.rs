//! IO roundtrip tests for fem-rs.
//!
//! Verifies that writing data to a file and reading it back produces the
//! same result (up to format limitations).
//!
//! | Format | Test | Roundtrip |
//! |--------|------|-----------|
//! | GMSH .msh | 2-D tri mesh | Write → Read → Compare |
//! | Matrix Market .mtx | CSR matrix | Write → Read → Compare |
//! | Abaqus .inp | 3-D tet mesh | Read → Write(msh) → Read(msh) → Compare |
//! | Netgen .vol | 3-D tet mesh | Read → Write(msh) → Read(msh) → Compare |

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{topology::MeshTopology, Mesh};
use fem_space::{
    fe_space::FESpace,
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
};

// ─── Helpers ────────────────────────────────────────────────────────────

/// Create a temporary file path.
fn temp_path(suffix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("fem_rs_roundtrip_{}", suffix));
    let _ = std::fs::remove_file(&p); // clean from previous run
    p
}

/// Maximum difference in node coordinates.
fn coords_max_diff(a: &Mesh<2>, b: &Mesh<2>) -> f64 {
    let mut d = 0.0_f64;
    let n = a.n_nodes().min(b.n_nodes());
    for i in 0..n {
        let ca = a.node_coords(i as u32);
        let cb = b.node_coords(i as u32);
        for k in 0..2 {
            d = d.max((ca[k] - cb[k]).abs());
        }
    }
    d
}

/// Compare connectivity of two meshes (element count, node indices per element).
fn conn_matches(a: &Mesh<2>, b: &Mesh<2>) -> bool {
    if a.n_elements() != b.n_elements() { return false; }
    if a.n_nodes() != b.n_nodes() { return false; }
    for e in 0..a.n_elements() as u32 {
        let na = a.element_nodes(e);
        let nb = b.element_nodes(e);
        if na.len() != nb.len() { return false; }
        for k in 0..na.len() {
            if na[k] != nb[k] { return false; }
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════
// Test: GMSH roundtrip (2-D)
// ═══════════════════════════════════════════════════════════════════════

/// Write a 2-D Tri3 mesh to GMSH format, read it back, and compare.
#[test]
fn io_gmsh_2d_roundtrip() {
    let orig = Mesh::<2>::unit_square_tri(6);

    let path = temp_path("gmsh_2d.msh");
    fem_io::write_msh_file(&orig, &path).expect("write_msh failed");
    let msh = fem_io::read_msh_file(&path).expect("read_msh failed");
    let round = msh.into_2d().expect("into_2d failed");
    let _ = std::fs::remove_file(&path);

    assert!(conn_matches(&orig, &round), "GMSH: connectivity mismatch");
    let cd = coords_max_diff(&orig, &round);
    assert!(cd < 1e-12, "GMSH: max coord diff = {:.3e}", cd);
    assert_eq!(orig.n_elements(), round.n_elements());
    assert_eq!(orig.n_nodes(), round.n_nodes());
    eprintln!("  [io] gmsh-2d-roundtrip: n_nodes={}, n_elem={}, coord_max_diff={:.3e}",
        round.n_nodes(), round.n_elements(), cd);
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Matrix Market roundtrip
// ═══════════════════════════════════════════════════════════════════════

/// Assemble a stiffness matrix, write to .mtx, read back, and compare
/// every entry.
#[test]
fn io_matrix_market_roundtrip() {
    use fem_linalg::CsrMatrix;

    let mesh = Mesh::<2>::unit_square_tri(6);
    let space = H1Space::new(mesh, 1);
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let orig: CsrMatrix<f64> = Assembler::assemble_bilinear(&space, &[&diff], 3);

    let path = temp_path("matrix_market.mtx");
    fem_io::write_matrix_market(&path, &orig).expect("write_matrix_market failed");
    let round = fem_io::read_matrix_market(&path).expect("read_matrix_market failed");
    let _ = std::fs::remove_file(&path);

    // Convert to dense and compare (handles symmetric storage differences)
    fn to_dense(a: &CsrMatrix<f64>) -> Vec<f64> {
        let mut d = vec![0.0; a.nrows * a.ncols];
        for i in 0..a.nrows {
            for pk in a.row_ptr[i]..a.row_ptr[i + 1] {
                d[i * a.ncols + a.col_idx[pk] as usize] = a.values[pk];
            }
        }
        d
    }
    assert_eq!(orig.nrows, round.nrows);
    assert_eq!(orig.ncols, round.ncols);

    let d_orig = to_dense(&orig);
    let d_round = to_dense(&round);
    let max_diff = d_orig.iter().zip(d_round.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-14,
        "Matrix Market roundtrip: max dense diff = {:.3e}", max_diff);
    eprintln!("  [io] mtx-roundtrip: {}×{}, nnz={}, max_diff={:.3e}",
        orig.nrows, orig.ncols, orig.nnz(), max_diff);
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Abaqus → GMSH cross-format roundtrip
// ═══════════════════════════════════════════════════════════════════════

/// Read a 3-D Tet4 mesh from Abaqus .inp, write it to GMSH .msh, read
/// it back.  Compare element/node counts.
#[test]
fn io_abaqus_to_gmsh_roundtrip() {
    // Use the existing Abaqus file in the repo
    let abaqus_path = concat!(env!("CARGO_MANIFEST_DIR"), "/meshes/named_sets_tet.inp");
    let mesh3d = fem_io::read_abaqus_inp_file(abaqus_path)
        .expect("read_abaqus_inp_file failed");

    let n_elem_orig = mesh3d.n_elements();
    let n_node_orig = mesh3d.n_nodes();

    // Write to GMSH .msh (3-D)
    let msh_path = temp_path("abaqus_roundtrip.msh");
    fem_io::write_msh_file(&mesh3d, &msh_path).expect("write_msh (3d) failed");
    let msh = fem_io::read_msh_file(&msh_path).expect("read_msh (3d) failed");
    let mesh3d_round = msh.into_3d().expect("into_3d failed");
    let _ = std::fs::remove_file(&msh_path);

    assert_eq!(mesh3d_round.n_elements(), n_elem_orig,
        "Abaqus→GMSH: element count mismatch: {} vs {}",
        mesh3d_round.n_elements(), n_elem_orig);
    assert_eq!(mesh3d_round.n_nodes(), n_node_orig,
        "Abaqus→GMSH: node count mismatch: {} vs {}",
        mesh3d_round.n_nodes(), n_node_orig);
    eprintln!("  [io] abaqus-to-gmsh: n_nodes={}, n_elem={} (roundtrip ok)",
        mesh3d_round.n_nodes(), mesh3d_round.n_elements());
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Netgen → GMSH cross-format roundtrip
// ═══════════════════════════════════════════════════════════════════════

/// Read a 3-D Tet4 mesh from Netgen .vol, write to GMSH .msh, read back.
#[test]
fn io_netgen_to_gmsh_roundtrip() {
    let netgen_path = concat!(env!("CARGO_MANIFEST_DIR"), "/meshes/surface_tags_tet.vol");
    let mesh3d = fem_io::read_netgen_vol_file(netgen_path)
        .expect("read_netgen_vol_file failed");

    let n_elem_orig = mesh3d.n_elements();
    let n_node_orig = mesh3d.n_nodes();

    let msh_path = temp_path("netgen_roundtrip.msh");
    fem_io::write_msh_file(&mesh3d, &msh_path).expect("write_msh failed");
    let msh = fem_io::read_msh_file(&msh_path).expect("read_msh failed");
    let mesh3d_round = msh.into_3d().expect("into_3d failed");
    let _ = std::fs::remove_file(&msh_path);

    assert_eq!(mesh3d_round.n_elements(), n_elem_orig,
        "Netgen→GMSH: element count mismatch");
    assert_eq!(mesh3d_round.n_nodes(), n_node_orig,
        "Netgen→GMSH: node count mismatch");
    eprintln!("  [io] netgen-to-gmsh: n_nodes={}, n_elem={} (roundtrip ok)",
        mesh3d_round.n_nodes(), mesh3d_round.n_elements());
}

// ═══════════════════════════════════════════════════════════════════════
// Test: VTK write + solution field consistency
// ═══════════════════════════════════════════════════════════════════════

/// Solve a Poisson problem, write the solution to VTK, verify the file
/// is created and non-empty (smoke check — VTK is ASCII, no reader for
/// the full roundtrip for 2-D yet).
#[test]
fn io_vtk_solution_smoke() {
    use std::f64::consts::PI;

    let mesh = Mesh::<2>::unit_square_tri(8);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let diff = DiffusionIntegrator { kappa: 1.0 };
    let src = DomainSourceIntegrator::new(|x: &[f64]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin());
    let mut mat = Assembler::assemble_bilinear(&space, &[&diff], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n];
    let cfg = fem_solver::SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..fem_solver::SolverConfig::default() };
    fem_solver::solve_cg(&mat, &rhs, &mut u, &cfg).expect("CG failed");

    // Write VTK
    let mut writer = fem_io::VtkWriter::new(&mesh);
    writer.add_point_data(fem_io::DataArray::scalars("u", u));
    let vtk_path = temp_path("solution.vtk");
    writer.write_file(&vtk_path).expect("VTK write failed");

    // Smoke check: file exists and has content
    let metadata = std::fs::metadata(&vtk_path).expect("VTK file should exist");
    assert!(metadata.len() > 100, "VTK file too small: {} bytes", metadata.len());
    let _ = std::fs::remove_file(&vtk_path);

    eprintln!("  [io] vtk-solution-smoke: file size = {} bytes", metadata.len());
}
