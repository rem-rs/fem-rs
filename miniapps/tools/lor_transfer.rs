//! # LOR Transfer Miniapp (port of MFEM `miniapps/tools/lor-transfer.cpp`)
//!
//! Maps functions between a high-order (HO) H1 space and a low-order
//! refined (LOR) H1 space built with `make_refined` (Gauss-Lobatto), and
//! reports the mass of each representation.
//!
//! Simplified port (same scope reduction as `lor_solvers`): H1 spaces with
//! the pointwise `InterpolationGridTransfer` operator (`-h1 -t` in C++),
//! serial 2D quad meshes. The L2-projection transfer (mass-matrix based,
//! C++ default) and the weighted/velocity variants are not ported.
//!
//! Operators (matching MFEM's `InterpolationGridTransfer`):
//! - R (HO → LOR): `RefinementOperator` — the fine dof values are the
//!   coarse function evaluated at the fine dof points (via `find_points`).
//! - P (LOR → HO): `DerefinementOperator` — per coarse element, the fine
//!   function is L2-projected onto the coarse element space:
//!   `x_c = A⁻¹ Rlocᵀ Mf u_fine` with `A = Rlocᵀ Mf Rloc`, solved densely.
//!
//! NOTE: C++ computes `int lref = order + 1` from the *default* order (3)
//! BEFORE parsing options, so `-o` does not change lref — the effective
//! default is 4 unless -lref is given.  Ported faithfully.
//!
//! Sample runs:
//!   cargo run --release --example tools_lor_transfer -- -m data/inline-quad.mesh -o 2 -no-vis
//!   cargo run --release --example tools_lor_transfer -- -m data/inline-quad.mesh -o 2 -lref 3 -no-vis

use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::DomainSourceIntegrator;
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::find_points;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::{FESpace, H1Space};

fn main() {
    // Parse command-line options.
    let mut mesh_file = String::from("../../data/star.mesh");
    let mut problem = 1usize;
    let mut order = 3usize;
    let mut lref = 0usize;
    let mut lorder = 0usize;

    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => mesh_file = it.next().unwrap().clone(),
            "-p" | "--problem" => problem = it.next().unwrap().parse().unwrap(),
            "-o" | "--order" => order = it.next().unwrap().parse().unwrap(),
            "-lref" | "--lor-ref-level" => lref = it.next().unwrap().parse().unwrap(),
            "-lo" | "--lor-order" => lorder = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            other => panic!("Unknown option: {other}"),
        }
    }
    if lref == 0 {
        lref = 4; // C++: order(default=3) + 1, computed pre-parse
    }

    let t0 = std::time::Instant::now();

    // Read the mesh (2D quad, this port).
    let mfem = read_mfem_file(&mesh_file).expect("cannot read mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("lor-transfer port: 2D quad meshes only");
    if mesh.elem_type != ElementType::Quad4 {
        panic!("lor-transfer port: Quad4 meshes only (got {:?})", mesh.elem_type);
    }

    // Create the low-order refined mesh: Mesh::MakeRefined(mesh, lref, GaussLobatto).
    let mesh_lor = fem_space::make_refined_2d(&mesh, lref);

    // Create the spaces.
    let fespace = H1Space::new(mesh.clone(), order as u8);
    if lorder == 0 {
        lorder = 1;
        eprintln!("Switching the H1 LOR space order from 0 to 1");
    }
    let fespace_lor = H1Space::new(mesh_lor.clone(), lorder as u8);

    println!("HO  space dofs: {}", fespace.n_dofs());
    println!("LOR space dofs: {}", fespace_lor.n_dofs());

    // ── HO projection (MFEM ProjectCoefficient = nodal interpolation) ─────
    let rho_dofs: Vec<f64> = project_coefficient_h1(&fespace, problem);
    let rho = GridFunction::new(&fespace, rho_dofs.clone());
    let ho_mass = compute_mass(&rho, None, "HO       ");

    // ── R: HO -> LOR (refinement operator = point evaluation) ─────────────
    let mut pts = Vec::with_capacity(mesh_lor.n_nodes() * 2);
    for n in 0..mesh_lor.n_nodes() as u32 {
        let c = mesh_lor.coords_of(n);
        pts.extend_from_slice(&c);
    }
    let (elem_ids, ref_xi) = find_points(&mesh, &pts, mesh_lor.n_nodes());
    let mut rho_lor_dofs = vec![0.0f64; fespace_lor.n_dofs()];
    for (j, &e) in elem_ids.iter().enumerate() {
        if e < 0 {
            panic!("LOR dof {j} not located in the HO mesh");
        }
        rho_lor_dofs[j] = rho.evaluate_at_element(e as u32, &ref_xi[j]);
    }
    let rho_lor = GridFunction::new(&fespace_lor, rho_lor_dofs.clone());
    let _mass_r_ho = compute_mass(&rho_lor, Some(ho_mass), "R(HO)    ");

    // ── P: LOR -> HO (local derefinement L2 projection) ───────────────────
    // Matches MFEM InterpolationGridTransfer::BackwardOperator
    // (FiniteElementSpace::DerefinementOperator): per coarse element the
    // fine function is L2-projected onto the coarse element space:
    //   x_c = A⁻¹ Rlocᵀ Mf u_fine,   A = Rlocᵀ Mf Rloc,
    // with Rloc[fi,ci] = φ_ci(t_fi) the coarse basis at the fine dof
    // positions (the lref+1 GaussLobatto lattice points of the element) and
    // Mf the fine (P1, per-subcell) mass matrix on the coarse element.
    let s: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(lref + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect();
    let n_fine = (lref + 1) * (lref + 1);
    let n_coarse = (order + 1) * (order + 1);

    // Lattice-point → LOR-vertex lookup (P1: dof == vertex), by position.
    // Quantize positions (~1e-9) so both arithmetic paths hash equally.
    let quant = |x: f64| (x * 1e9).round() as i64;
    let mut pos_to_lor: std::collections::HashMap<(i64, i64), u32> =
        std::collections::HashMap::new();
    for n in 0..mesh_lor.n_nodes() as u32 {
        let c = mesh_lor.coords_of(n);
        pos_to_lor
            .entry((quant(c[0]), quant(c[1])))
            .or_insert(n);
    }

    // Coarse tensor-product Lagrange basis on the GLL nodes.
    let coarse_basis = |u: f64, v: f64, phi: &mut [f64]| {
        let b1 = |t: f64, i: usize| -> f64 {
            let mut val = 1.0;
            for (m, &sm) in s.iter().enumerate() {
                if m != i {
                    val *= (t - sm) / (s[i] - sm);
                }
            }
            val
        };
        for ci in 0..n_coarse {
            let (i, j) = (ci % (order + 1), ci / (order + 1));
            phi[ci] = b1(u, i) * b1(v, j);
        }
    };

    // Local fine mass matrix + transfer matrix (identical on all straight
    // elements — built once).
    let mut mf = vec![0.0f64; n_fine * n_fine];
    let mut rloc = vec![0.0f64; n_fine * n_coarse];
    {
        let (gpts, gwts) = fem_element::quadrature::gauss_legendre_01(lref + 2);
        let mut phi_c = vec![0.0f64; n_coarse];
        for a in 0..lref {
            for b in 0..lref {
                let (da, db) = (s[a + 1] - s[a], s[b + 1] - s[b]);
                for gi in 0..gpts.len() {
                    for gj in 0..gpts.len() {
                        let su = s[a] + gpts[gi] * da;
                        let sv = s[b] + gpts[gj] * db;
                        let w = gwts[gi] * gwts[gj] * da * db;
                        let tu = (su - s[a]) / da;
                        let tv = (sv - s[b]) / db;
                        let bu = [1.0 - tu, tu];
                        let bv = [1.0 - tv, tv];
                        for iu in 0..2 {
                            for iv in 0..2 {
                                let fi = (a + iu) + (b + iv) * (lref + 1);
                                let wi = bu[iu] * bv[iv];
                                for ju in 0..2 {
                                    for jv in 0..2 {
                                        let fj =
                                            (a + ju) + (b + jv) * (lref + 1);
                                        mf[fi * n_fine + fj] +=
                                            w * wi * bu[ju] * bv[jv];
                                    }
                                }
                                coarse_basis(s[a + iu], s[b + iv], &mut phi_c);
                                for ci in 0..n_coarse {
                                    rloc[fi * n_coarse + ci] += w * wi * phi_c[ci];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Coarse local mass matrix Mc = ∫φ_ci φ_cj (needed for the normal
    // equations); accumulate with the same quadrature as B.
    let mut m_c = vec![0.0f64; n_coarse * n_coarse];
    {
        let (gpts, gwts) = fem_element::quadrature::gauss_legendre_01(lref + 2);
        let mut phi_c = vec![0.0f64; n_coarse];
        for a in 0..lref {
            for b in 0..lref {
                let (da, db) = (s[a + 1] - s[a], s[b + 1] - s[b]);
                for gi in 0..gpts.len() {
                    for gj in 0..gpts.len() {
                        let su = s[a] + gpts[gi] * da;
                        let sv = s[b] + gpts[gj] * db;
                        let w = gwts[gi] * gwts[gj] * da * db;
                        coarse_basis(su, sv, &mut phi_c);
                        for ci in 0..n_coarse {
                            for cj in 0..n_coarse {
                                m_c[ci * n_coarse + cj] += w * phi_c[ci] * phi_c[cj];
                            }
                        }
                    }
                }
            }
        }
    }
    let mc_inv = {
        let mut inv = vec![0.0f64; n_coarse * n_coarse];
        for col in 0..n_coarse {
            let mut rhs = vec![0.0; n_coarse];
            rhs[col] = 1.0;
            let mut mc_copy = m_c.clone();
            let sol = gauss_solve(&mut mc_copy, &mut rhs, n_coarse)
                .expect("singular local coarse mass matrix");
            for r in 0..n_coarse {
                inv[r * n_coarse + col] = sol[r];
            }
        }
        inv
    };

    // Apply P: per coarse element, gather u_fine (lattice values),
    // rhs = Rlocᵀ Mf u_fine, x_c = A⁻¹ rhs, scatter into the HO dofs.
    let mut rho_pr_dofs = vec![0.0f64; fespace.n_dofs()];
    let mut dof_count = vec![0u32; fespace.n_dofs()];
    let mut u_fine = vec![0.0f64; n_fine];
    let mut mu = vec![0.0f64; n_fine];
    let mut rhs = vec![0.0f64; n_coarse];
    let mut phi_c = vec![0.0f64; n_coarse];
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        let v0 = mesh.coords_of(ns[0]);
        let v1 = mesh.coords_of(ns[1]);
        let v3 = mesh.coords_of(ns[3]);
        for bj in 0..lref + 1 {
            for bi in 0..lref + 1 {
                let su = s[bi];
                let sv = s[bj];
                let px = v0[0] + su * (v1[0] - v0[0]) + sv * (v3[0] - v0[0]);
                let py = v0[1] + su * (v1[1] - v0[1]) + sv * (v3[1] - v0[1]);
                let fi = bi + bj * (lref + 1);
                let key = (quant(px), quant(py));
                let lor_v = pos_to_lor[&key];
                u_fine[fi] = rho_lor_dofs[lor_v as usize];
            }
        }
        let _ = &mu;
        for ci in 0..n_coarse {
            let mut acc = 0.0;
            for k in 0..n_fine {
                acc += rloc[k * n_coarse + ci] * u_fine[k];
            }
            rhs[ci] = acc;
        }
        // x_c = Mc⁻¹ rhs (precomputed inverse).
        let x_c: Vec<f64> = (0..n_coarse)
            .map(|r| {
                (0..n_coarse).map(|c| mc_inv[r * n_coarse + c] * rhs[c]).sum()
            })
            .collect();
        let elem_dofs = fespace.element_dofs(e);
        for (i, &dof) in elem_dofs.iter().enumerate().take(n_coarse) {
            rho_pr_dofs[dof as usize] += x_c[i];
            dof_count[dof as usize] += 1;
        }
    }
    let _ = phi_c;
    // Shared coarse dofs receive identical local projections from every
    // adjacent element — average them.
    for (v, &cnt) in rho_pr_dofs.iter_mut().zip(dof_count.iter()) {
        if cnt > 1 {
            *v /= cnt as f64;
        }
    }
    let rho_pr = GridFunction::new(&fespace, rho_pr_dofs.clone());
    let _mass_pr_r = compute_mass(&rho_pr, Some(ho_mass), "P(R(HO)) ");

    let mut max_diff = 0.0f64;
    for (a, b) in rho_dofs.iter().zip(rho_pr_dofs.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    println!("|HO - P(R(HO))|_inf   = {max_diff:.12e}");

    // ── LOR projection ────────────────────────────────────────────────────
    let lor_dofs: Vec<f64> = project_coefficient_h1(&fespace_lor, problem);
    let rho_lor2 = GridFunction::new(&fespace_lor, lor_dofs);
    let _lor_mass = compute_mass(&rho_lor2, None, "LOR      ");

    println!("\nElapsed: {:.2}s", t0.elapsed().as_secs_f64());
}

/// MFEM `GridFunction::ProjectCoefficient` on nodal (Gauss-Lobatto) H1
/// spaces = interpolation at the dof locations.
fn project_coefficient_h1(space: &H1Space<Mesh<2>>, problem: usize) -> Vec<f64> {
    let dm = space.dof_manager();
    (0..space.n_dofs())
        .map(|d| rho_exact(&dm.dof_coord(d as u32), problem))
        .collect()
}

/// C++ `RHO_exact`.
fn rho_exact(x: &[f64], problem: usize) -> f64 {
    let r: f64 = (x[0] * x[0] + x[1] * x[1]).sqrt();
    match problem {
        1 => x[1] + 0.25 * (2.0 * std::f64::consts::PI * r).cos(),
        2 => x[1] * x[1] * x[1] + 2.0 * x[0] * x[1] + x[0],
        3 => std::f64::consts::FRAC_PI_2 - (5.0 * (2.0 * r - 1.0)).atan(),
        4 => {
            if r < 0.1 {
                1.0
            } else {
                0.0
            }
        }
        5 => 2.0 + 2.0 * x[0] * x[0] + 3.0 * x[1] * x[1] - x[0] * x[1] + 0.1 * r.sin(),
        _ => 1.0,
    }
}

/// C++ `compute_mass`: ∫ gf dx with MFEM's integration-order convention
/// (2·elem_order + OrderW(2) + coeff_order(1)).
fn compute_mass(gf: &GridFunction<H1Space<Mesh<2>>>, old: Option<f64>, label: &str) -> f64 {
    let space = gf.space();
    let p = space.order();
    let quad_rule = 2 * p + 3;
    let one = DomainSourceIntegrator::new(|_pt| 1.0);
    let lf = Assembler::assemble_linear(space, &[&one], quad_rule);
    let newmass: f64 = lf.iter().zip(gf.dofs().iter()).map(|(a, b)| a * b).sum();
    match old {
        Some(o) if o >= 0.0 => {
            let pct = (newmass - o).abs() * 100.0 / o;
            println!("H1 {label} mass   = {newmass:.17e} ({pct:.4}%)");
        }
        _ => {
            println!("H1 {label} mass   = {newmass:.17e}");
        }
    }
    newmass
}

/// Dense Gaussian elimination with partial pivoting (row-major `n×n`).
fn gauss_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in col + 1..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let inv = 1.0 / a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] * inv;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut sm = b[r];
        for c in r + 1..n {
            sm -= a[r * n + c] * x[c];
        }
        x[r] = sm / a[r * n + r];
    }
    Some(x)
}
