//! Miniapp: MFEM `nurbs_ex5` — mixed Darcy with NURBS H(div).
//!
//! **Status: partially ported (round 26).**  This file is *not* a complete 1:1
//! port of `miniapps/nurbs/nurbs_ex5.cpp`; the stages listed below are, and
//! everything from the right-hand side onwards exits with status 3 rather than
//! printing numbers this port cannot produce.
//!
//! What MFEM's example does (`k u + grad p = f`, `-div u = g`, natural BC
//! `-p = <given pressure>`; `R_space` = `NURBS_HDivFECollection(order, dim)` on
//! `NURBSExtension(mesh->NURBSext, order)`, `W_space` = `NURBSFECollection(order)`
//! on the *stolen* extension, `BlockOperator` + block-diagonal preconditioner
//! with `DSmoother(M)` and `GSSmoother(B·diag(M)⁻¹·Bᵀ)`, MINRES
//! `rtol = atol = 1e-10`, `max_iter = 10000`).
//!
//! Ported 1:1 (verified against the C++ binary, MFEM 4.10):
//!
//! | stage | `square-nurbs.mesh -o 1` (default `-r 6`) |
//! |---|---|
//! | mesh read + `ref_levels = floor(log(10000./NE)/log(2.)/dim)` | 6 levels, 4096 elements |
//! | `NURBS_HDivFECollection` + `NURBSExtension` construction | `dim(R) = 8580` |
//! | `W_space = NURBSFECollection(order)` | `dim(W) = 4225` |
//! | `dim(R+W)` | `12805` |
//! | `R_space->GetEssentialTrueDofs(ess_bdr = 1)` | `Number boundary dofs in H(div): 260` |
//! | `W_space->GetEssentialTrueDofs(ess_bdr = 1)` | `Number boundary dofs in H1: 256` |
//! | `NURBSExtension::GetBdrElementDofTable` (D96) | the signed `bel_dof` rows of all three modes (`BdrDofMode::H1/HDiv/HCurl`), byte-identical to C++ — `NurbsHDivSpace::boundary_dof_table` |
//!
//! **Not ported** (each is a hard blocker for the solve block, so the example
//! stops with `exit(3)` *before* printing anything it cannot reproduce):
//!
//! * `LinearForm::Assemble` for `fform` — its `VectorFEDomainLFIntegrator` part
//!   *and* the `VectorFEBoundaryFluxLFIntegrator` natural-BC term
//!   `∫_Γ (v·n) g ds`.  The signed boundary DOF table that term needs now exists
//!   (D96: `Mode::H_DIV` negates every low-side boundary entity's DOFs, and
//!   `Vector::AddElementVector` reads a negative row entry as a subtraction);
//!   what is still missing is the *boundary element* assembly path — the
//!   boundary FE of `fes->GetBE(i)` (a `NURBS1D/2DFiniteElement` bound to the
//!   analysis extension's boundary knot vector and span index, quadrature
//!   `oa*order + ob = 2*order`) and the rational geometry of the *refined*
//!   boundary patch, `mesh->GetBdrElementTransformation(i)` (the boundary
//!   analogue of `NurbsFESpace::geometry`).
//! * The block MINRES solve: `Blocksolvers`/`BlockDiagonalPreconditioner` with
//!   `DSmoother(M)` and `GSSmoother(B diag(M)⁻¹ Bᵀ)`.  `fem-solver` has
//!   `MinresSolver` and `BlockDiagonalPrecond`, but not the
//!   `DSmoother`-backed Schur complement `S = B·diag(M)⁻¹·Bᵀ` this example
//!   needs, so the iteration log (462 iterations for the default grid) cannot
//!   be reproduced.
//! * `VisItDataCollection` / `ParaViewDataCollection` (no writer in fem-rs) and
//!   the `ex5.mesh` / `sol_u.gf` / `sol_p.gf` NURBS outputs (no NURBS mesh
//!   writer — see `nurbs_ex1`).
//!
//! `nurbs_ex5.cpp`'s own `pa` branch is dead code there (`pa = false;` is
//! assigned right after the collection is chosen), so partial assembly is not
//! part of the 1:1 configuration.

use fem_space::nurbs_fe_space::{NurbsFESpace, NurbsHDivSpace};

struct Args {
    mesh: String,
    order: usize,
    ref_levels: i64,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: "data/square-nurbs.mesh".to_string(), order: 1, ref_levels: -1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_else(|| a.mesh.clone()),
            "-o" | "--order" => {
                a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    let text = std::fs::read_to_string(&args.mesh).expect("failed to read the NURBS mesh file");

    // 3. `Mesh *mesh = new Mesh(mesh_file, 1, 1); int dim = mesh->Dimension();`
    let geo = fem_space::NurbsExtension::from_mesh_str(&text).expect("NURBS mesh");
    let dim = geo.dim();
    let n_elems = geo.n_elements();

    // 4. `ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim)` when
    //    `-r` is not given, then that many `mesh->UniformRefinement()`.
    let ref_levels = if args.ref_levels < 0 {
        ((10000.0_f64 / n_elems as f64).ln() / std::f64::consts::LN_2 / dim as f64).floor() as i64
    } else {
        args.ref_levels
    } as usize;

    // 5. `hdiv_coll = new NURBS_HDivFECollection(order, dim);
    //     l2_coll = new NURBSFECollection(order);
    //     NURBSext = new NURBSExtension(mesh->NURBSext, order);
    //     mfem::out << "Create NURBS fec and ext" << std::endl;`
    println!("Create NURBS fec and ext");
    // `W_space = new FiniteElementSpace(mesh, NURBSext, l2_coll)` then
    // `R_space = new FiniteElementSpace(mesh, W_space->StealNURBSext(),
    //  hdiv_coll)` — both spaces share the one analysis extension, so the
    // scalar space is `NurbsHDivSpace::scalar_space`.
    let r_space = NurbsHDivSpace::from_mesh_str(&text, ref_levels, args.order)
        .expect("H(div) NURBS space");
    let w_space: &NurbsFESpace = r_space.scalar_space();

    // 6. `block_offsets[1] = R_space->GetVSize(); block_offsets[2] =
    //     W_space->GetVSize();` and the banner.
    let n_u = r_space.n_dofs();
    let n_p = w_space.n_dofs();
    println!("***********************************************************");
    println!("dim(R) = {n_u}");
    println!("dim(W) = {n_p}");
    println!("dim(R+W) = {}", n_u + n_p);
    println!("***********************************************************");

    // `ess_bdr = 1` -> `GetEssentialTrueDofs(ess_bdr, ess_tdof_list)` for both
    // spaces (MFEM computes and prints them, then discards both lists — no
    // boundary condition is eliminated in this example).
    if geo.max_bdr_attribute() > 0 {
        println!("Number boundary dofs in H(div): {}", r_space.essential_dofs().len());
        println!("Number boundary dofs in H1: {}", w_space.boundary_dofs().len());
    } else {
        println!("Number boundary dofs in H(div): 0");
        println!("Number boundary dofs in H1: 0");
    }

    // 7.-11. `fform` (with `VectorFEBoundaryFluxLFIntegrator`), `gform`, the
    // Darcy `BlockOperator` and the MINRES solve.
    eprintln!(
        "nurbs_ex5: stages 7-11 are not ported (the boundary-element assembly path that \
         `VectorFEBoundaryFluxLFIntegrator` needs — the boundary FE and the refined boundary \
         patch geometry — and the DSmoother/GSSmoother block preconditioner; the signed \
         bel_dof table itself is available as `NurbsHDivSpace::boundary_dof_table`); see the \
         module docs. Run MFEM's nurbs_ex5 for stages 7-11."
    );
    std::process::exit(3);
}
