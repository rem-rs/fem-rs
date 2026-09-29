//! D843-3: hanging-dof true-dof compression vs MFEM 4.10 — bitwise pin.
//!
//! Truth: `tmp/d92c/probe_d92c.cpp` (WSL `$HOME/work/d92c`, MFEM 4.10 serial)
//! replicating ex6 on `data/star.mesh` through iteration 1:
//! - iter0: conforming 31×31 system, `FormLinearSystem(..., copy_interior=1)`
//!   DIAG_KEEP elimination and the PCG+GSSmoother solution (X0) — this pins
//!   the GS **backward** sweep's reverse-order row accumulation
//!   (`Gauss_Seidel_back`, sparsemat.cpp:2571 scans `j--`).
//! - iter1: the non-conforming compressed system — cP rows
//!   (`BuildConformingInterpolation`), `A_true = (R·A)·P` via
//!   `ConformingAssemble`'s two first-touch `Mult` products, X = R·x, and
//!   B = Pᵀb − A_e·X with B[ess] = (A·X)[ess].

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::read_mfem_file;
use fem_linalg::PrintLevel;
use fem_mesh::amr::{refine_nonconforming_quad_aniso, QuadRefineDir};
use fem_mesh::Mesh;
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::constraints::{
    conforming::ConformingInterpolation, eliminate_rowcol_keep_diag, eliminate_vdofs_in_rhs,
};
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space};

const FIXTURE: &str = include_str!("data/d843_star_iter1_mfem.txt");

struct Fixture {
    true_vsize: usize,
    ess: Vec<u32>,
    xf: Vec<f64>,
    cp_rows: Vec<Vec<(u32, f64)>>,
    a0: Vec<Vec<(u32, f64)>>,
    b0: Vec<f64>,
    x0: Vec<f64>,
    a1full: Vec<Vec<(u32, f64)>>,
    a1r: Vec<Vec<(u32, f64)>>,
    x1: Vec<f64>,
    b1: Vec<f64>,
}

fn parse_vec(line: &str) -> Vec<f64> {
    // lines look like "XF n=86 0.5 0.25 ..." or "B0 n=31 ..."
    let body = match line.split_once('=') {
        Some((_, rest)) => rest,
        None => line,
    };
    let body = match body.split_once(' ') {
        Some((_, rest)) => rest,
        None => body,
    };
    body.split_whitespace()
        .map(|t| t.parse::<f64>().expect("fixture float"))
        .collect()
}

fn parse_rows(line: &str) -> Vec<(u32, f64)> {
    let mut out = Vec::new();
    let mut rest = line.split_once(':').map(|(_, r)| r).unwrap_or("");
    while let Some(open) = rest.find('(') {
        let close = rest[open..].find(')').expect("fixture pair close") + open;
        let inner = &rest[open + 1..close];
        let (c, v) = inner.split_once(',').expect("fixture pair comma");
        out.push((c.parse::<u32>().expect("fixture col"), v.parse::<f64>().expect("fixture val")));
        rest = &rest[close + 1..];
    }
    out
}

fn load_fixture() -> Fixture {
    let mut f = Fixture {
        true_vsize: 0,
        ess: Vec::new(),
        xf: Vec::new(),
        cp_rows: vec![Vec::new(); 86],
        a0: Vec::new(),
        b0: Vec::new(),
        x0: Vec::new(),
        a1full: Vec::new(),
        a1r: Vec::new(),
        x1: Vec::new(),
        b1: Vec::new(),
    };
    for line in FIXTURE.lines() {
        if let Some(v) = line.strip_prefix("TRUEVSIZE ") {
            f.true_vsize = v.parse().unwrap();
        } else if line.starts_with("ESS n=") {
            f.ess = line
                .split(':')
                .nth(1)
                .unwrap()
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect();
        } else if line.starts_with("XF n=") {
            f.xf = parse_vec(line);
        } else if line.starts_with("CPR ") {
            let r: usize = line[4..]
                .split_once(':')
                .unwrap()
                .0
                .parse()
                .expect("fixture cpr row");
            f.cp_rows[r] = parse_rows(line);
        } else if line.starts_with("A0R ") {
            f.a0.push(parse_rows(line));
        } else if line.starts_with("B0 n=") {
            f.b0 = parse_vec(line);
        } else if line.starts_with("X0 n=") {
            f.x0 = parse_vec(line);
        } else if line.starts_with("A1FULL ") {
            f.a1full.push(parse_rows(line));
        } else if line.starts_with("A1R ") {
            f.a1r.push(parse_rows(line));
        } else if line.starts_with("X n=") {
            f.x1 = parse_vec(line);
        } else if line.starts_with("B n=") {
            f.b1 = parse_vec(line);
        }
    }
    f
}

/// ex6 iter0 refinement batch: 15 elements, all-isotropic (round-91 probe
/// REFS line: (0,3) (1,3) (2,3) (3,3) (4,3) (5,3) (7,3) (8,3) (10,3)
/// (11,3) (13,3) (14,3) (16,3) (17,3) (19,3)).
fn refine_star_once() -> (Mesh<2>, Vec<fem_mesh::amr::HangingNodeConstraint>) {
    let mesh = read_mfem_file("../../data/star.mesh")
        .expect("star.mesh")
        .mesh2d
        .expect("2-D");
    let marked: Vec<(u32, QuadRefineDir)> = [0u32, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19]
        .iter()
        .map(|&i| (i, QuadRefineDir::Both))
        .collect();
    refine_nonconforming_quad_aniso(&mesh, &marked, None)
}

#[test]
fn d843_cp_and_compressed_system_match_mfem() {
    let fx = load_fixture();
    let (mesh, cons) = refine_star_once();
    assert_eq!(mesh.n_nodes(), 86, "refined vertex count (round-91)");
    assert_eq!(fx.true_vsize, 76);

    let space = H1Space::new(mesh.clone(), 1);
    let cdofs = space.n_dofs();
    assert_eq!(cdofs, 86);
    let conf = ConformingInterpolation::from_hanging_constraints(cdofs, &cons);
    assert_eq!(conf.n_true(), fx.true_vsize, "GetTrueVSize");

    // cP rows, bitwise (values + stored column order)
    for (i, row) in fx.cp_rows.iter().enumerate() {
        assert_eq!(
            conf.cp_row(i),
            row.as_slice(),
            "cP row {i} vs MFEM BuildConformingInterpolation"
        );
    }

    // assemble the iter1 system (quadrature ex6.cpp: a: 2*order, b: 2*order+1)
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mat = Assembler::assemble_bilinear(&space, &[&diffusion], 2);
    let rhs = Assembler::assemble_linear(&space, &[&source], 3);

    // full-space matrix bitwise vs MFEM's assembled `mat` (A1FULL)
    for (i, row) in fx.a1full.iter().enumerate() {
        let got: Vec<(u32, f64)> = (mat.row_ptr[i]..mat.row_ptr[i + 1])
            .map(|k| (mat.col_idx[k], mat.values[k]))
            .collect();
        assert_eq!(&got, row.as_slice(), "A_full row {i}");
    }

    // compressed + eliminated system
    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &mesh.unique_boundary_tags());
    let ess_true: Vec<u32> = bnd
        .iter()
        .filter_map(|&d| conf.true_index_of(d).map(|t| t as u32))
        .collect();
    assert_eq!(ess_true, fx.ess, "GetEssentialTrueDofs");

    let mut a_true = conf.compress(&mat);
    let ae = eliminate_rowcol_keep_diag(&mut a_true, &ess_true);
    for (i, row) in fx.a1r.iter().enumerate() {
        let got: Vec<(u32, f64)> = (a_true.row_ptr[i]..a_true.row_ptr[i + 1])
            .map(|k| (a_true.col_idx[k], a_true.values[k]))
            .collect();
        assert_eq!(&got, row.as_slice(), "A_true (eliminated) row {i}");
    }

    // B = Pᵀb − A_e·X with B[ess] = (A·X)[ess], X = R·u (warm start from the
    // fixture's post-update grid function, boundary zeroed)
    let mut u = fx.xf.clone();
    for &d in &bnd {
        u[d as usize] = 0.0;
    }
    let xs = conf.restrict(&u);
    let mut bsys = conf.mult_transpose(&rhs);
    eliminate_vdofs_in_rhs(&ae, &a_true, &ess_true, &xs, &mut bsys);
    assert_eq!(bsys, fx.b1, "B after EliminateVDofsInRHS");
    assert_eq!(xs, fx.x1, "X = R·x (warm start restriction)");
}

#[test]
fn d843_conforming_iter0_solve_matches_mfem_bitwise() {
    let fx = load_fixture();
    // iter0: the unrefined conforming star, 31 dofs
    let mesh = read_mfem_file("../../data/star.mesh")
        .expect("star.mesh")
        .mesh2d
        .expect("2-D");
    let space = H1Space::new(mesh.clone(), 1);
    assert_eq!(space.n_dofs(), 31);
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 2);
    let rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &mesh.unique_boundary_tags());
    let ae = eliminate_rowcol_keep_diag(&mut mat, &bnd);
    for (i, row) in fx.a0.iter().enumerate() {
        let got: Vec<(u32, f64)> = (mat.row_ptr[i]..mat.row_ptr[i + 1])
            .map(|k| (mat.col_idx[k], mat.values[k]))
            .collect();
        assert_eq!(&got, row.as_slice(), "iter0 eliminated row {i}");
    }
    let mut xs = vec![0.0; 31]; // x = 0 at iter0
    let mut bsys = rhs;
    eliminate_vdofs_in_rhs(&ae, &mat, &bnd, &xs, &mut bsys);
    assert_eq!(bsys, fx.b0, "iter0 B");

    let cfg = SolverConfig {
        rtol: 1e-6, // legacy sqrt(1e-12)
        atol: 0.0,
        max_iter: 200,
        print_level: PrintLevel::FirstAndLast,
        ..SolverConfig::default()
    };
    let _ = solve_pcg_gssmoother(&mat, &bsys, &mut xs, &cfg);
    assert_eq!(xs, fx.x0, "iter0 solution (pins the GS backward-sweep order)");
}
