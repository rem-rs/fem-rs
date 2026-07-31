//! # Example 26 — Geometric Multigrid for Poisson  [1:1 translation of MFEM ex26]
//!
//! Solves the Poisson problem `−Δu = 1` with homogeneous Dirichlet BCs using
//! a geometric multigrid preconditioner.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex26_geom_mg
//! cargo run --example mfem_ex26_geom_mg -- -m data/star.mesh
//! cargo run --example mfem_ex26_geom_mg -- -m data/fichera.mesh
//! ```

use std::io::Write;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::{
    read_mfem_file, write_mfem, write_mfem_file, write_mfem_file_3d, write_mfem_gf_file,
};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{
    GeometricMgLevel, GeometricMgHierarchy, GeometricMgConfig, GeometricMgPrecond,
    GeometricMgAsPrecond, MgCycleType, MgSmootherType, solve_pcg,
    StoredElementOperator, PADiffusionOp, SumFactDiffusionOp,
};
use fem_space::{
    H1Space, fe_space::FESpace, constraints::boundary_dofs,
    build_h1_prolongation_matrix,
};
use fem_mesh::ElementType;

fn main() {
    let args = parse_args();

    if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        match (mfem.mesh2d, mfem.mesh3d) {
            (Some(m2), _) => solve(m2, &args),
            (_, Some(m3)) => solve(m3, &args),
            _ => panic!("MFEM mesh has neither a 2D nor a 3D representation"),
        }
    } else {
        // Default mesh: match C++ ex26 (`../data/star.mesh` relative to the build dir).
        let mfem = read_mfem_file("data/star.mesh").expect("failed to read data/star.mesh");
        solve(mfem.mesh2d.expect("data/star.mesh must be 2D"), &args);
    }
}

/// Dimension-dispatch for the mesh operations that are split between 2-D and
/// 3-D in the library (uniform refinement, MFEM mesh writers, GLVis serialization).
trait Ex26Mesh: fem_mesh::topology::MeshTopology + Clone {
    const DIM: usize;
    fn uniform_refine(&self) -> Self;
    fn glvis_bytes(&self) -> Vec<u8>;
    fn write_refined(&self, path: &str);
}

impl Ex26Mesh for Mesh<2> {
    const DIM: usize = 2;
    fn uniform_refine(&self) -> Self { fem_mesh::refine_uniform(self) }
    fn glvis_bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        write_mfem(&mut v, self, None).unwrap();
        v
    }
    fn write_refined(&self, path: &str) { write_mfem_file(path, self).expect("mesh write failed"); }
}

impl Ex26Mesh for Mesh<3> {
    const DIM: usize = 3;
    fn uniform_refine(&self) -> Self { fem_mesh::refine_uniform_3d(self) }
    fn glvis_bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        // write_mfem serializes the 3-D mesh via the `mesh_3d` argument.
        write_mfem(&mut v, &Mesh::<2>::unit_square_tri(1), Some(self)).unwrap();
        v
    }
    fn write_refined(&self, path: &str) { write_mfem_file_3d(path, self).expect("mesh write failed"); }
}

/// Run the ex26 solve for a `D`-dimensional mesh.
///
/// The `PADiffusionOp` / `SumFactDiffusionOp` partial-assembly operators are
/// 2-D only (Tri3 / Quad4); for 3-D meshes the V-cycle falls back to the
/// element-by-element operator (bitwise-identical to the CSR matrix).
fn solve<const D: usize>(mesh: Mesh<D>, args: &Args)
where
    Mesh<D>: Ex26Mesh,
{
    let dim = D;
    let mut mesh_data = mesh.glvis_bytes();

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("data/star.mesh"));
    println!("   --geometric-refinements {}", args.geometric_refs);
    println!("   --order-refinements {}", args.order_refs);
    println!("   --device cpu");
    if !args.visualization { println!("   --no-visualization"); }
    println!("Device configuration: cpu (host-std)");

    // 4. Uniform refinement onto coarse mesh.
    let coarse_mesh = {
        let ne = mesh.n_elements();
        let ref_levels = if ne > 0 {
            ((5000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
        } else { 0 };
        let mut m = mesh;
        for _ in 0..ref_levels { m = m.uniform_refine(); }
        m
    };

    // 5. FE space hierarchy.
    let mut meshes = vec![coarse_mesh];
    for _ in 0..args.geometric_refs {
        let fine = meshes.last().unwrap().uniform_refine();
        meshes.push(fine);
    }

    let mut spaces: Vec<H1Space<Mesh<D>>> = Vec::new();
    for m in &meshes {
        spaces.push(H1Space::new(m.clone(), 1));
    }
    let finest_mesh = meshes.last().unwrap().clone();
    for k in 1..=args.order_refs {
        spaces.push(H1Space::new(finest_mesh.clone(), 1u8 << k));
    }
    let n_spaces = spaces.len();

    println!("Number of finite element unknowns: {}", spaces.last().unwrap().n_dofs());

    // 6. RHS.  Match MFEM's LinearForm quadrature: integrate the P_order
    //    source exactly (constant coefficient → rule of order `2*order+1`).
    let fine_space = spaces.last().unwrap();
    let n_dofs = fine_space.n_dofs();
    let rhs_quad = (2 * fine_space.order() + 1).max(3) as u8;
    let mut rhs = Assembler::assemble_linear(fine_space, &[&DomainSourceIntegrator::new(|_| 1.0)], rhs_quad);

    // 7. Solution vector.
    let mut x = vec![0.0; n_dofs];

    // 8. Build MG hierarchy: per-level matrices with symmetric BC elimination.
    let boundary_tags: Vec<i32> = fine_space.mesh().unique_boundary_tags();
    // Zero RHS at BC DOFs (matching MFEM Multigrid::FormFineLinearSystem).
    let bc_fine = boundary_dofs(fine_space.mesh(), fine_space.dof_manager(), &boundary_tags);
    for &d in &bc_fine { rhs[d as usize] = 0.0; }

    // The on-the-fly PA operators are 2-D only (Tri3 / Quad4).  For 3-D meshes
    // (Tet4 / Hex8) the V-cycle uses elem_op (bitwise-identical to the CSR).
    let et0 = fine_space.mesh().element_type(0);
    let use_pa = matches!(et0, ElementType::Tri3 | ElementType::Tri6 | ElementType::Quad4);

    let mut levels: Vec<GeometricMgLevel> = Vec::new();
    let mut prolong: Vec<fem_linalg::CsrMatrix<f64>> = Vec::new();
    for i in 0..n_spaces {
        let space = &spaces[i];
        let qo = (2 * space.order() + 1).max(3) as u8;
        // Assemble CSR + element matrices in one pass (same integration)
        let (mut mat, elem_dofs, elem_mats, ldofs, n_elems) =
            Assembler::assemble_bilinear_with_elements(
                space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        // Save raw diagonal before BC modification
        let raw_diag = mat.diagonal();
        let raw_dinv: Vec<f64> = raw_diag.iter()
            .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 }).collect();
        // Apply symmetric BC elimination for the CSR matrix (PCG outer / coarse CG)
        let bc = boundary_dofs(space.mesh(), space.dof_manager(), &boundary_tags);
        let mut dummy = vec![0.0; mat.nrows];
        for &d in &bc { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy); }
        // Build element-by-element operator (raw matrices, no BC mods)
        let elem_op = StoredElementOperator {
            elem_dofs: elem_dofs.clone(), elem_mats: elem_mats.clone(),
            ldofs, n_elems, n_dofs: mat.nrows,
        };
        // Build on-the-fly PA operator (matches MFEM AddMultPA), 2-D only.
        let pa_op = if use_pa {
            let elem_dofs_clone = elem_dofs.clone();
            Some(PADiffusionOp::build(
                space.mesh(), mat.nrows, space.order(), qo, 1.0,
                |e| {
                    let start = e as usize * ldofs;
                    elem_dofs_clone[start..start + ldofs].to_vec()
                },
            ))
        } else {
            None
        };
        // Sum-factorization PA operator (Quad4 only).  Uses GLL nodes matching
        // the QuadQk basis of the CSR assembly (see SumFactDiffusionOp::build),
        // so it reproduces the CSR operator up to ~1e-7 rounding.
        let sf_op = if space.mesh().element_type(0) == ElementType::Quad4 {
            let e_dofs = elem_dofs.clone();
            Some(SumFactDiffusionOp::build(
                space.mesh(), mat.nrows, space.order(), qo, 1.0,
                |e| {
                    let start = e as usize * ldofs;
                    e_dofs[start..start + ldofs].to_vec()
                },
            ))
        } else {
            None
        };
        levels.push(GeometricMgLevel {
            mat, bc_dofs: bc,
            elem_op: Some(elem_op), raw_diag, raw_dinv,
            pa_op, sf_op,
        });
    }
    for i in 0..n_spaces - 1 {
        prolong.push(build_h1_prolongation_matrix(
            spaces[i].mesh(), spaces[i].dof_manager(),
            spaces[i + 1].mesh(), spaces[i + 1].dof_manager(),
        ));
    }

    levels.reverse();
    prolong.reverse();
    let hierarchy = GeometricMgHierarchy::new(levels, prolong);
    println!("Size of linear system: {}", hierarchy.finest_matrix().nrows);

    // 9. Solve with PCG + MG V(1,1)-cycle (matching C++: V-cycle, 1 pre + 1
    //    post sweep, Chebyshev(2) smoothers, coarse CG rtol sqrt(1e-4)).
    let mg_config = GeometricMgConfig {
        pre_sweeps: 1, post_sweeps: 1,
        smoother: MgSmootherType::Chebyshev(2),
        max_eig_override: None,
        max_eig_overrides: Vec::new(),
        jacobi_omega: 0.8,
        coarse_max_iter: 200,
        coarse_rtol: (1e-4f64).sqrt(), // C++: pcg->SetRelTol(sqrt(1e-4))
        cycle_type: MgCycleType::V,
    };
    let mg = GeometricMgPrecond::new(mg_config, &hierarchy);
    let precond = GeometricMgAsPrecond { mg: &mg, hierarchy: &hierarchy };

    // PCG tolerance: `1e-12` in the (B r, r) norm, equivalent to C++'s
    // `PCG(..., 1e-12, 0.0)` (which internally uses SetRelTol(sqrt(1e-12))).
    if let Err(e) = solve_pcg(hierarchy.finest_matrix(), &rhs, &mut x, &precond, 1e-12, 2000, true) {
        eprintln!("PCG: No convergence! ({e})");
    }

    // 10. Save.
    {
        fine_space.mesh().write_refined("refined.mesh");
        write_mfem_gf_file("sol.gf", dim, &x, "H1", fine_space.order(), 1, 14).expect("sol write failed");
    }

    // 11. GLVis visualization (1:1 with C++ ex26 section 12)
    if args.visualization {
        use std::io::Write;
        use std::net::TcpStream;
        let keys = "keys amrRljcUUuu\n";
        let glvis_send = |stream: &mut TcpStream| -> std::io::Result<()> {
            write!(stream, "solution\n")?;
            stream.write_all(&mesh_data)?;
            writeln!(stream, "FiniteElementSpace")?;
            writeln!(stream, "FiniteElementCollection: H1_{dim}D_P{}", fine_space.order())?;
            writeln!(stream, "VDim: 1")?;
            writeln!(stream, "Ordering: 1")?;
            writeln!(stream)?;
            for v in &x { writeln!(stream, "{:.7e}", v)?; }
            write!(stream, "{keys}")?;
            writeln!(stream, "window_title 'Solution'")?;
            stream.flush()
        };
        if let Ok(mut sock) = TcpStream::connect("localhost:19916") {
            let _ = glvis_send(&mut sock);
        }
    }
}

// ─── CLI (1:1 with C++ OptionsParser) ─────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    geometric_refs: usize,
    order_refs: usize,
    visualization: bool,
}

fn parse_args() -> Args {
    // Default mesh matches C++ ex26 (`../data/star.mesh`).
    let mut a = Args { mesh: None, geometric_refs: 0, order_refs: 2, visualization: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "-gr" | "--geometric-refinements" => {
                a.geometric_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(0)
            }
            "-or" | "--order-refinements" => {
                a.order_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(2)
            }
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-d" | "--device" => { let _ = it.next(); } // CPU only
            _ => {}
        }
    }
    a
}
