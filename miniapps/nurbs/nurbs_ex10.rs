//! Miniapp: NURBS Example 10 — dynamic nonlinear elasticity on a NURBS mesh.
//! Intended 1:1 port of MFEM `nurbs_ex10.cpp` (MFEM 4.10): `dv/dt = H(x) + S v`,
//! `dx/dt = v` with a Neo-Hookean hyperelastic model on
//! `FiniteElementSpace(mesh, NURBSext, fec, vdim = dim)`, implicit SDIRK
//! time integration with a Newton solve of the reduced backward-Euler system.
//!
//! # Status: NOT PORTED (deliberate gap stub, exit 3)
//!
//! The serial building blocks all exist in fem-rs — the H¹ mesh-based port is
//! `examples/mfem_ex10_hyperelastic_dyn.rs`, `HyperelasticModel::
//! pk1_and_tangent` lives in `crates/assembly/src/physics/
//! nonlinear_hyperelasticity.rs`, and the NURBS patch-aware assembly of
//! `nurbs_patch_ex1` in `crates/assembly/src/iga/nurbs_patch.rs`.  What is
//! missing is the *vector NURBS* layer, which is exactly the D-debt this stub
//! documents:
//!
//! 1. a vector NURBS H¹ space — `FiniteElementSpace(mesh, NURBSext, fec,
//!    vdim = dim)` with per-component essential DOFs
//!    (`GetEssentialTrueDofs(ess_bdr)` over the interleaved layout) —
//!    `crates/space/src/nurbs_fe_space.rs` (NurbsFESpace is scalar-only);
//! 2. `VectorMassIntegrator` / `VectorDiffusionIntegrator` on that space
//!    (per-component scalar assembly + block scatter, ordering byVDIM);
//! 3. `HyperelasticNLFIntegrator` on NURBS elements: residual,
//!    element Jacobian (`GetGradient`), and `GetEnergy` with the
//!    NURBS element transformation (weights included);
//! 4. `ProjectCoefficient(velo, ELEMENTL2)` for the vector NURBS space;
//! 5. `GetElasticEnergyDensity` (ElasticEnergyCoefficient) for GLVis/output.
//!
//! With (1)–(4) in place the driver itself (SDIRK23, Newton + MINRES/Jacobi,
//! the `ReducedSystemOperator`, the energy printouts) can be ported 1:1 from
//! `examples/mfem_ex10_hyperelastic_dyn.rs` and `nurbs_ex10.cpp`.

fn main() {
    eprintln!("nurbs_ex10: not implemented in fem-rs — refusing to print numbers that");
    eprintln!("cannot be checked against MFEM 4.10. Missing:");
    eprintln!("  * vector NURBS H1 space (FiniteElementSpace(mesh, NURBSext, fec, vdim=dim))");
    eprintln!("    with per-component essential DOFs (crates/space/src/nurbs_fe_space.rs)");
    eprintln!("  * VectorMassIntegrator / VectorDiffusionIntegrator on NURBS elements");
    eprintln!("    (crates/assembly)");
    eprintln!("  * HyperelasticNLFIntegrator on NURBS elements: Mult / GetGradient /");
    eprintln!("    GetEnergy with NeoHookeanModel (crates/assembly/src/physics)");
    eprintln!("  * vector ELEMENTL2 projection (ProjectCoefficient) for initial conditions");
    eprintln!("  * elastic energy density projection (GetElasticEnergyDensity)");
    eprintln!("The scalar/H1 reference port is examples/mfem_ex10_hyperelastic_dyn.rs; the");
    eprintln!("NURBS patch-aware assembly groundwork is crates/assembly/src/iga/nurbs_patch.rs.");
    std::process::exit(3);
}
