//! # Example 27 — Mixed Boundary Conditions [1:1 translation of MFEM ex27]
//!
//! Solves −Δu = 0 on a periodic (seam-identified) Q3-curved mesh with mixed
//! boundary conditions (Neumann / Robin / Dirichlet / natural), using either a
//! continuous H¹ space (essential Dirichlet BC) or a discontinuous L² space
//! (weak Dirichlet BC via `DGDiffusionIntegrator` + `DGDirichletLFIntegrator`).
//!
//! The mesh generation mirrors the C++ `GenerateSerialMesh` flow: the flat
//! mesh is refined, Q3 geometry is built and warped, and the x=±1 seam is
//! stitched last via `make_periodic` (the C++ `v2v` + `RemoveUnusedVertices`
//! block) — merged connectivity, seam boundary faces dropped, and each
//! element keeps its own pre-merge geometry corners exactly like MFEM's
//! discontinuous nodal GridFunction.

use fem_assembly::dg::dg_base::{
    build_face_elem_map, face_point_geom, ref_elem_vol, xform_grads,
};
use fem_assembly::{
    Assembler, DgAssembler, ElimPolicy, InteriorFaceList, eliminate_ess_tdofs,
    standard::DiffusionIntegrator,
};
use fem_element::ReferenceElement;
use fem_mesh::{Mesh, topology::MeshTopology, ElementType};
use fem_solver::{SolverConfig, fmt_g};
use fem_space::{H1Space, L2Space, fe_space::FESpace, constraints::boundary_dofs};

static mut HOLE_RADIUS: f64 = 0.2;

fn main() {
    let mut a = parse_args();
    // MFEM ex27 replaces a negative DG penalty *before* printing the options
    // (`if (kappa < 0 && !h1) { kappa = (order+1)*(order+1); }`,
    // ex27.cpp:137-140 then `args.PrintOptions(mfem::out)`), so the canonical
    // `--kappa` row shows the replaced value (the `-dg` default prints 4).
    if !a.h1 && a.kappa < 0.0 { a.kappa = (a.order as f64 + 1.0).powi(2); }
    // MFEM `OptionsParser::ParseCheck` ends with `PrintOptions(out)`
    // (optparser.cpp:270-271 → :331), so the upstream stdout opens with this
    // block, before any mesh work (D771 byte-exact surface).
    print_options_block(&a);
    // The hole-radius reset messages are printed *after* the options block and
    // use the raw `-a` value (ex27.cpp:142-152).
    if a.hole_radius < 0.01 { println!("Hole radius too small, resetting to 0.01."); }
    if a.hole_radius > 0.49 { println!("Hole radius too large, resetting to 0.49."); }
    unsafe { HOLE_RADIUS = a.hole_radius.max(0.01).min(0.49); }

    let mesh = gen_mesh(a.ref_levels);
    if a.h1 {
        solve_h1(&a, &mesh);
    } else {
        solve_dg(&a, &mesh);
    }
}

/// Canonical stdout opening block — MFEM `OptionsParser::PrintOptions`
/// (`optparser.cpp:331`): `Options used:` followed by one `   <long_name>
/// <value>` row per declared option, in `AddOption` order.  `ENABLE` pairs
/// print the long name that matches the current bool (MFEM prints
/// `options[j].long_name` for true and `options[j+1].long_name` for false);
/// numeric values go through `WriteValue` (`os << value`) i.e. the C++
/// default-ostream `%g` form — that is `fem_solver::fmt_g` (6 significant
/// digits, the D343/ex29/ex24 convention already used across the tree).
fn print_options_block(a: &Args) {
    println!("Options used:");
    println!("   {}", if a.h1 { "--continuous" } else { "--discontinuous" });
    println!("   --order {}", a.order);
    println!("   --sigma {}", fmt_g(a.sigma));
    println!("   --kappa {}", fmt_g(a.kappa));
    println!("   --refine-serial {}", a.ref_levels);
    println!("   --material-value {}", fmt_g(a.mat_val));
    println!("   --dirichlet-value {}", fmt_g(a.dbc_val));
    println!("   --neumann-value {}", fmt_g(a.nbc_val));
    println!("   --robin-a-value {}", fmt_g(a.rbc_a_val));
    println!("   --robin-b-value {}", fmt_g(a.rbc_b_val));
    println!("   --radius {}", fmt_g(a.hole_radius));
    println!("   {}", if a.visualization { "--visualization" } else { "--no-visualization" });
}

/// H1 path (MFEM ex27 steps 3–14): continuous Q1 space, essential Dirichlet BC.
///
/// The mesh is UNFOLDED (x=±1 seam columns present, like the C++ element
/// geometry); periodicity is imposed by identifying the seam DOF pairs
/// (tag 5 at x=-1 ↔ tag 6 at x=1) — the C++ `v2v` stitch, but at the DOF
/// level so the per-element geometry keeps the x=±1 positions.
fn solve_h1(a: &Args, mesh: &Mesh<2>) {
    let space = H1Space::new(mesh.clone(), a.order as u8);
    let n = space.n_dofs();

    let mut stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: a.mat_val }], 3);
    // Robin: add a·u·v on the Robin boundary (tag 2)
    let rm = assemble_mass(&space, mesh, a.rbc_a_val, &[2], 3);
    stiff = fem_linalg::CsrMatrix::add(&stiff, &rm);

    let mut rhs = vec![0.0; n];
    let nbc = assemble_linear(&space, mesh, |_, _| a.mat_val * a.nbc_val, &[1], 3);
    let rbc = assemble_linear(&space, mesh, |_, _| a.mat_val * a.rbc_b_val, &[2], 3);
    for i in 0..n { rhs[i] += nbc[i] + rbc[i]; }

    // Periodicity lives in the MESH: `gen_mesh` merged the seam via
    // `make_periodic` (the C++ `v2v` stitch), so the space is constructed
    // through the periodic-quotient dof tables and `n` already is MFEM's
    // merged true-dof count (the old DOF-level
    // `identify_periodic_dof_pairs` stitch is gone with D799-1).  No leading
    // blank line — the C++ stream goes straight from the
    // `Number of finite element unknowns:` row to the solver history.
    println!("Number of finite element unknowns: {}", n);

    let ess = boundary_dofs(mesh, space.dof_manager(), &[3]);
    // C++ BilinearForm::FormLinearSystem defaults to diag_policy=DIAG_KEEP
    // (preserves the original diagonal, zeroes the row/col off-diagonals) —
    // the D706 serial core entry (D739).  The former hand-rolled loop pushed
    // each dof's deltas through a scratch vector and *added* them to the RHS,
    // so the essential rows kept the reactions of the earlier eliminated dofs
    // (`b(r) = A(r,r)·x(r) + Σ A(r,r')·x(r')`) — MFEM's EliminateVDofsInRHS
    // ends with `mat->PartMult`, i.e. `B(r) = A(r,r)·x(r)` as an assignment
    // (bilinearform.cpp:1239).  The two agree for homogeneous data (the
    // example's default `-dbc 0`) and the core entry is used from here on.
    let mut x_bc = vec![0.0; n];
    for &d in &ess { x_bc[d as usize] = a.dbc_val; }
    // D753: `FormLinearSystem` also returns the **projected solution** as the
    // initial iterate — `X = R·x` with x's essential entries = dbc_val and the
    // interior zero (C++ ex27 does `u.ProjectBdrCoefficient(dbcCoef, dbc_bdr)`
    // before FormLinearSystem, `u = 0` elsewhere).  MFEM's legacy `PCG()`
    // wrapper leaves `iterative_mode = true`, so `CGSolver::Mult` starts from
    // `r = B − A·X` (solvers.cpp:875-879).  Starting from `X = 0` (the old
    // code) inflated the first printed `(B r, r)` to 396.994 where the C++
    // gold prints 37.0097 — the whole iteration history then diverged.
    // `eliminate_ess_tdofs` returns exactly that X (form.rs:97: a bitwise copy
    // of `x` for a conforming space, i.e. MFEM's `copy_interior = 1`).
    let mut x = eliminate_ess_tdofs(&mut stiff, &ess, &x_bc, &mut rhs, ElimPolicy::DiagKeep);

    // D753 / round-32 lesson (the two PCG API layers): C++ ex27 calls the
    // legacy `PCG(*A, M, B, X, 1, 500, 1e-12, 0.0)`, and that wrapper does
    // `SetRelTol(sqrt(RTOLERANCE))` = **rel_tol 1e-6** (solvers.cpp:1076-1077;
    // the abs tolerance is `sqrt(0.0) = 0`).  This solver's `rtol` *is* MFEM's
    // `rel_tol` — the criterion is `nom <= max(rtol²·nom0, atol²)`
    // (solvers.cpp:919, mirrored at `iterative.rs:258`) — so the 1:1 value is
    // `sqrt(1e-12) = 1e-6`, not 1e-12 (the old value gave 41 iterations /
    // ARF 0.505 where the C++ gold converges in 29 / 0.603605).
    let cfg = SolverConfig { rtol: 1e-6, atol: 0.0, max_iter: 500, verbose: true, ..Default::default() };
    // MFEM's legacy `PCG()` wrapper prints the iteration history and the ARF
    // from inside `Mult` and *nothing else* — there is no "Solved in N
    // iterations." row in the canonical ex27 stdout (D771), so the result is
    // discarded.
    let _ = fem_solver::solve_pcg_gssmoother(&stiff, &rhs, &mut x, &cfg).expect("PCG+GSSmoother");

    verify_bc(a, &space, mesh, &x);

    // C++ step 14: save refined.mesh + sol.gf (MFEM-native formats)
    let _ = fem_io::mfem::write_mfem_file("refined.mesh", mesh);
    let _ = fem_io::mfem::write_mfem_gf_file("sol.gf", 2, &x, "H1", a.order as u8, 1, 8);
}

/// DG path (MFEM ex27 steps 3–14 with `-dg`): discontinuous Q1 space.
///
/// Weakly enforces all BCs:
/// - **Dirichlet** (tag 3): `DGDiffusionIntegrator` face penalty on the matrix
///   + `DGDirichletLFIntegrator` on the RHS.
/// - **Robin** (tag 2): `BoundaryMassIntegrator` on the matrix + the RHS source.
/// - **Neumann** (tag 1): natural (RHS source only).
///
/// Solved with PCG when `sigma == -1` (SIP is symmetric), GMRES otherwise —
/// exactly as MFEM ex27 does.
fn solve_dg(a: &Args, mesh: &Mesh<2>) {
    let space = L2Space::new(mesh.clone(), a.order as u8);
    let n = space.n_dofs();
    println!("Number of finite element unknowns: {}", n);

    // MFEM ex27: a negative kappa was already replaced with (order+1)² in
    // `main` (before the options block), exactly as ex27.cpp:137-140 does.
    let penalty = a.kappa;
    let qo = 2 * a.order as u8;

    let ifl = InteriorFaceList::build(mesh);
    // Matrix: volume + interior faces + DGDiffusion on the Dirichlet boundary
    // (tag 3) + BoundaryMass on the Robin boundary (tag 2).
    let mut stiff = DgAssembler::assemble_dg(&space, &ifl, a.mat_val, a.sigma, penalty, qo, Some(&[3]));
    let rm = assemble_l2_mass(&space, mesh, a.mat_val * a.rbc_a_val, &[2], qo);
    stiff = fem_linalg::CsrMatrix::add(&stiff, &rm);

    // RHS: DGDirichletLF (tag 3) + BoundaryLF (tags 1 and 2).
    let mut rhs = vec![0.0; n];
    let dglf = assemble_l2_dg_dirichlet_lf(&space, mesh, a.dbc_val, a.mat_val, a.sigma, penalty, &[3], qo);
    let nbc = assemble_l2_linear(&space, mesh, |_, _| a.mat_val * a.nbc_val, &[1], qo);
    let rbc = assemble_l2_linear(&space, mesh, |_, _| a.mat_val * a.rbc_b_val, &[2], qo);
    for i in 0..n { rhs[i] += dglf[i] + nbc[i] + rbc[i]; }

    let mut x = vec![0.0; n];
    // D779: C++ ex27 calls the legacy wrapper `PCG(*A, M, B, X, 1, 500, 1e-12,
    // 0.0)`, and that wrapper is `SetRelTol(sqrt(RTOLERANCE))`
    // (solvers.cpp:1076) ⇒ **rel_tol 1e-6**, not 1e-12.  This solver's `rtol`
    // *is* MFEM's `rel_tol` (the criterion is `nom <= max(rtol²·nom0, atol²)`,
    // solvers.cpp:919 ≈ iterative.rs:258), so 1e-12 ran the DG system to a
    // needlessly tight residual — 128 iterations / 150 history lines where the
    // C++ gold converges in 82 / 104.  Same D753/round-32 lesson as the H1 path.
    let cfg = SolverConfig { rtol: 1e-6, atol: 0.0, max_iter: 500, verbose: true, ..Default::default() };
    // As in the H1 path: the legacy `PCG()`/`GMRES()` wrappers print only the
    // iteration history + ARF (D771 canonical surface), so the result is
    // discarded.
    let _ = if a.sigma == -1.0 {
        fem_solver::solve_pcg_gssmoother(&stiff, &rhs, &mut x, &cfg).expect("PCG+GSSmoother")
    } else {
        // MFEM ex27: GMRES with restart 10 for the non-symmetric (NIP) case.
        // The C++ GMRES converges on the *preconditioned* residual; the linlvo
        // GMRES uses ‖r‖/‖b‖ and stagnates just above 1e-12, so relax rtol.
        let gcfg = SolverConfig { rtol: 1e-10, ..cfg.clone() };
        fem_solver::solve_gmres_gssmoother(&stiff, &rhs, &mut x, 10, &gcfg).expect("GMRES+GSSmoother")
    };

    verify_bc(a, &space, mesh, &x);

    let _ = fem_io::mfem::write_mfem_file("refined.mesh", mesh);
    let _ = fem_io::mfem::write_mfem_gf_file("sol.gf", 2, &x, "L2", a.order as u8, 1, 8);
}

/// MFEM ex27 step 13: verify the boundary conditions by integrating
/// `α·n·∇u + β·u` over the marked boundary and comparing with `γ`.
fn verify_bc<S: FESpace>(a: &Args, space: &S, mesh: &Mesh<2>, x: &[f64]) {
    println!();
    println!("Verifying boundary conditions");
    println!("=============================");

    let (avg, mut err) = integrate_bc(space, mesh, x, &[3], 0.0, 1.0, a.dbc_val, 3);
    let hom = a.dbc_val == 0.0;
    err /= if hom { 1.0 } else { a.dbc_val.abs() };
    println!("Average of solution on Gamma_dbc:\t{}, \t{} error {}",
             fmt_g(avg), if hom { "absolute" } else { "relative" }, fmt_g(err));

    let (avg, mut err) = integrate_bc(space, mesh, x, &[1], 1.0, 0.0, a.nbc_val, 3);
    let hom = a.nbc_val == 0.0;
    err /= if hom { 1.0 } else { a.nbc_val.abs() };
    println!("Average of n.Grad(u) on Gamma_nbc:\t{}, \t{} error {}",
             fmt_g(avg), if hom { "absolute" } else { "relative" }, fmt_g(err));

    let (avg, err) = integrate_bc(space, mesh, x, &[4], 1.0, 0.0, 0.0, 3);
    println!("Average of n.Grad(u) on Gamma_nbc0:\t{}, \tabsolute error {}", fmt_g(avg), fmt_g(err));

    let (avg, mut err) = integrate_bc(space, mesh, x, &[2], 1.0, a.rbc_a_val, a.rbc_b_val, 3);
    let hom = a.rbc_b_val == 0.0;
    err /= if hom { 1.0 } else { a.rbc_b_val.abs() };
    println!("Average of n.Grad(u)+a*u on Gamma_rbc:\t{}, \t{} error {}",
             fmt_g(avg), if hom { "absolute" } else { "relative" }, fmt_g(err));
}

fn gen_mesh(rl: usize) -> Mesh<2> {
    let a = unsafe { HOLE_RADIUS / std::f64::consts::SQRT_2 };
    let v: [[f64;2];29] = [
        [-1.0,-0.5],[-1.0,0.0],[-1.0,0.5],
        [-0.5-a,-a],[-0.5-a,0.0],[-0.5-a,a],
        [-0.5,-0.5],[-0.5,-a],[-0.5,a],[-0.5,0.5],
        [-0.5+a,-a],[-0.5+a,0.0],[-0.5+a,a],
        [0.0,-0.5],[0.0,0.0],[0.0,0.5],
        [0.5-a,-a],[0.5-a,0.0],[0.5-a,a],
        [0.5,-0.5],[0.5,-a],[0.5,a],[0.5,0.5],
        [0.5+a,-a],[0.5+a,0.0],[0.5+a,a],
        [1.0,-0.5],[1.0,0.0],[1.0,0.5]];
    let q:[[u32;4];16] = [
        [0,3,4,1],[1,4,5,2],[5,8,9,2],[8,12,15,9],
        [11,14,15,12],[10,13,14,11],[6,13,10,7],[0,6,7,3],
        [13,16,17,14],[14,17,18,15],[18,21,22,15],[21,25,28,22],
        [24,27,28,25],[23,26,27,24],[19,26,23,20],[13,19,20,16]];
    let bf:[([u32;2],i32);28] = [
        ([0,6],1),([6,13],1),([13,19],1),([19,26],1),
        ([28,22],2),([22,15],2),([15,9],2),([9,2],2),
        ([7,3],3),([10,7],3),([11,10],3),([12,11],3),
        ([8,12],3),([5,8],3),([4,5],3),([3,4],3),
        ([20,16],4),([23,20],4),([24,23],4),([25,24],4),
        ([21,25],4),([18,21],4),([17,18],4),([16,17],4),
        ([0,1],5),([1,2],5),([26,27],6),([27,28],6)];
    let c:Vec<f64> = v.iter().flat_map(|&[x,y]|[x,y]).collect();
    let e:Vec<u32> = q.iter().flat_map(|q|q.iter().copied()).collect();
    let fc:Vec<u32> = bf.iter().flat_map(|(e,_)|e.iter().copied()).collect();
    let ft:Vec<i32> = bf.iter().map(|(_,t)|*t).collect();
    let mesh = Mesh::<2>::uniform(c,e,vec![1;16],ElementType::Quad4,fc,ft,ElementType::Line2);
    let mut m = mesh;
    // C++ flow: stitch → SetCurvature(3, true) → refine(×ref) → Transform(trans).
    // We refine the PLAIN (unfolded) mesh, build the Q3 geometry, and warp it;
    // the stitch runs last, with the same semantics as the C++ `v2v` block.
    for _level in 0..rl {
        m = fem_mesh::refine_uniform(&m);
    }
    m.set_curvature(3);
    m.transform(hole_transform);
    // C++ stitch (ex27.cpp `GenerateSerialMesh`): `v2v` identifies the x=+1
    // seam vertices with the x=-1 ones, `RemoveUnusedVertices` drops the
    // merged column, and the seam edges become ordinary interior edges.  In
    // C++ the stitch runs on the linear mesh *before* `SetCurvature(3, true)`
    // — a DISCONTINUOUS nodal GF — so every element keeps its own pre-merge
    // node values and the seam quads' geometry stays on their own side of the
    // seam.  `make_periodic` mirrors exactly that: the connectivity is merged
    // and the seam boundary faces (tags 5/6) are dropped, while the
    // already-built high-order geometry keeps each element's pre-merge
    // corners (`geometry_nodes != element_nodes` ⇒ the spaces construct
    // through `DofManager::build_periodic`, the periodic-quotient path the
    // torus/klein meshes use).  H1 then gets MFEM's merged true-dof count,
    // and DG sees the seam as an interior face — before D799-1 it saw two
    // disconnected bdr faces and assembled zero coupling across the seam.
    // Side A = tag 5 (x=-1), side B = tag 6 (x=+1): a B node at (1, y) folds
    // onto the A node at (-1, y), i.e. `x_B = x_A + [2, 0]`.
    m.make_periodic(&[(5, 6, [2.0, 0.0])], 1e-9).expect("make_periodic")
}

fn hole_transform(p:[f64;2])->[f64;2] {
    let tol=1e-4;let(u,v)=(p[0],p[1]);
    if v>0.5-tol||v< -0.5+tol||u>1.0-tol||u< -1.0+tol||u.abs()<tol{return p}
    let qt=|du:f64,fv:f64|{let a=unsafe{HOLE_RADIUS};
        let d=4.0*a*(std::f64::consts::SQRT_2-2.0*a)*(1.0-2.0*fv);
        let v0=(1.0+std::f64::consts::SQRT_2)*(std::f64::consts::SQRT_2*a-2.0*fv)*((4.0-3.0*std::f64::consts::SQRT_2)*a+(8.0*(std::f64::consts::SQRT_2-1.0)*a-2.0)*fv)/d;
        let r=2.0*((std::f64::consts::SQRT_2-1.0)*a*a*(1.0-4.0*fv)+2.0*(1.0+std::f64::consts::SQRT_2*(1.0+2.0*(2.0*a-std::f64::consts::SQRT_2-1.0)*a))*fv*fv)/d;
        let t=if fv.abs()>1e-15{(fv/r).asin()*du/fv}else{0.0};
        (r*t.sin(),r*t.cos()-v0)};
    if u>0.0{
        // Top-right: quad_trans(u-0.5, v) → (x, y)
        if v>(u-0.5).abs(){let(x,y)=qt(u-0.5,v);return[x+0.5,y]}
        // Bottom-right: quad_trans(u-0.5, -v) → (x, y), then y = -y
        if v< -(u-0.5).abs(){let(x,y)=qt(u-0.5,-v);return[x+0.5,-y]}
        // Right: quad_trans(v, u-0.5) → SWAPPED: x gets y, y gets x
        if u-0.5>v.abs(){let(x,y)=qt(v,u-0.5);return[y+0.5,x]}
        // Left: quad_trans(v, 0.5-u) → SWAPPED: x gets -y+0.5, y gets x
        if u-0.5< -v.abs(){let(x,y)=qt(v,0.5-u);return[-y+0.5,x]}
    }else{
        // Top-left: quad_trans(u+0.5, v) → (x, y), then x -= 0.5
        if v>(u+0.5).abs(){let(x,y)=qt(u+0.5,v);return[x-0.5,y]}
        // Bottom-left: quad_trans(u+0.5, -v) → (x, y), then x -= 0.5, y = -y
        if v< -(u+0.5).abs(){let(x,y)=qt(u+0.5,-v);return[x-0.5,-y]}
        // Right: quad_trans(v, u+0.5) → SWAPPED: x gets y, y gets x, then x -= 0.5
        if u+0.5>v.abs(){let(x,y)=qt(v,u+0.5);return[y-0.5,x]}
        // Left: quad_trans(v, -0.5-u) → SWAPPED: x gets -y-0.5, y gets x
        if u+0.5< -v.abs(){let(x,y)=qt(v,-0.5-u);return[-y-0.5,x]}
    }
    p
}

/// Compute the effective distance between two points on a periodic domain.
/// The mesh is periodic in x with period 2.0 (x ∈ [-1, 1]).
/// Returns the shorter of the direct distance and the wrap-around distance.
fn periodic_edge_len(p0: &[f64; 2], p1: &[f64; 2]) -> f64 {
    let dx_direct = p1[0] - p0[0];
    // Wrap x by ±2.0 to get the shortest path
    let dx_wrapped = if dx_direct > 0.0 { dx_direct - 2.0 } else { dx_direct + 2.0 };
    let dx = if dx_direct.abs() < dx_wrapped.abs() { dx_direct } else { dx_wrapped };
    let dy = p1[1] - p0[1];
    (dx * dx + dy * dy).sqrt()
}

fn assemble_mass(s:&H1Space<Mesh<2>>,m:&Mesh<2>,alpha:f64,tags:&[i32],qo:u8)->fem_linalg::CsrMatrix<f64>{
    let n=s.n_dofs();let mut coo=fem_linalg::CooMatrix::new(n,n);
    for f in 0..m.n_boundary_faces() as u32{
        if!tags.contains(&m.face_tag(f)){continue}
        let ns=m.face_nodes(f);let re=fem_element::lagrange::SegP1;let q=re.quadrature(qo);
        let dofs:Vec<_>=ns.iter().map(|&n|n as usize).collect();let nd=dofs.len();
        let mut me=vec![0.0;nd*nd];let mut phi=vec![0.0;nd];
        for(qi,xi) in q.points.iter().enumerate(){
            let p0=m.node_coords(ns[0]);let p1=m.node_coords(ns[1]);
            let len=periodic_edge_len(&[p0[0],p0[1]],&[p1[0],p1[1]]);
            let w=q.weights[qi]*len;
            re.eval_basis(xi,&mut phi);
            for i in 0..nd{for j in 0..nd{me[i*nd+j]+=w*alpha*phi[i]*phi[j]}}
        }
        for i in 0..nd{for j in 0..nd{let v=me[i*nd+j];if v!=0.0{coo.add(dofs[i],dofs[j],v)}}}
    }
    coo.into_csr()
}

fn assemble_linear<F:Fn(&[f64],&[f64])->f64>(s:&H1Space<Mesh<2>>,m:&Mesh<2>,f:F,tags:&[i32],qo:u8)->Vec<f64>{
    let n=s.n_dofs();let mut rhs=vec![0.0;n];
    for fi in 0..m.n_boundary_faces() as u32{
        if!tags.contains(&m.face_tag(fi)){continue}
        let ns=m.face_nodes(fi);let re=fem_element::lagrange::SegP1;let q=re.quadrature(qo);
        let dofs:Vec<_>=ns.iter().map(|&n|n as usize).collect();let nd=dofs.len();let mut phi=vec![0.0;nd];
        for(qi,xi) in q.points.iter().enumerate(){
            let p0=m.node_coords(ns[0]);let p1=m.node_coords(ns[1]);
            // Correct for periodic wrap: x-periodicity with period 2.0
            // Use the SHORTER x-distance (wrap-around if dx > 1.0)
            let (dx_raw,dy)=(p1[0]-p0[0],p1[1]-p0[1]);
            // Periodic wrap: take the SHORTER of the direct and wrapped x-step
            // (the seam identifies x=-1 with x=1). Same rule as periodic_edge_len.
            let dx_wrapped = if dx_raw > 0.0 { dx_raw - 2.0 } else { dx_raw + 2.0 };
            let dx = if dx_raw.abs() < dx_wrapped.abs() { dx_raw } else { dx_wrapped };
            let len=(dx*dx+dy*dy).sqrt();
            let normal=[-dy,dx];let w=q.weights[qi]*len;
            // xp integration point — only needed for spatially-varying BCs
            let xp=[(1.0-xi[0])*0.5*p0[0]+(1.0+xi[0])*0.5*(p0[0]+dx),
                    (1.0-xi[0])*0.5*p0[1]+(1.0+xi[0])*0.5*p0[1]];
            let val=f(&xp,&normal);re.eval_basis(xi,&mut phi);
            for i in 0..nd{rhs[dofs[i]]+=w*val*phi[i]}
        }
    }
    rhs
}

struct Args{h1:bool,order:i32,sigma:f64,kappa:f64,ref_levels:usize,mat_val:f64,dbc_val:f64,nbc_val:f64,rbc_a_val:f64,rbc_b_val:f64,hole_radius:f64,visualization:bool}
fn parse_args()->Args{
    // `visualization` mirrors MFEM's `-vis/--visualization -no-vis/--no-visualization`
    // ENABLE pair, whose default is `true` (ex27.cpp:90); the GLVis socket step
    // itself is not ported, so the flag only feeds the canonical options block.
    let mut a=Args{h1:true,order:1,sigma:-1.0,kappa:-1.0,ref_levels:2,mat_val:1.0,dbc_val:0.0,nbc_val:1.0,rbc_a_val:1.0,rbc_b_val:1.0,hole_radius:0.2,visualization:true};
    let mut it=std::env::args().skip(1);
    while let Some(arg)=it.next(){match arg.as_str(){
        "-h1"|"--continuous"=>a.h1=true,"-dg"|"--discontinuous"=>a.h1=false,
        "-o"|"--order"=>a.order=it.next().and_then(|s|s.parse().ok()).unwrap_or(1),
        "-s"|"--sigma"=>a.sigma=it.next().and_then(|s|s.parse().ok()).unwrap_or(-1.0),
        "-k"|"--kappa"=>a.kappa=it.next().and_then(|s|s.parse().ok()).unwrap_or(-1.0),
        "-rs"|"--refine-serial"=>a.ref_levels=it.next().and_then(|s|s.parse().ok()).unwrap_or(2),
        "-mat"|"--material-value"=>a.mat_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-dbc"|"--dirichlet-value"=>a.dbc_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.0),
        "-nbc"|"--neumann-value"=>a.nbc_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-rbc-a"|"--robin-a-value"=>a.rbc_a_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-rbc-b"|"--robin-b-value"=>a.rbc_b_val=it.next().and_then(|s|s.parse().ok()).unwrap_or(1.0),
        "-a"|"--radius"=>a.hole_radius=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.2),
        "-vis"|"--visualization"=>a.visualization=true,
        "-no-vis"|"--no-visualization"=>a.visualization=false,_=>{}}}
    a
}

/// Gauss-Legendre quadrature on the reference segment **[0, 1]** (points `ξ`,
/// weights summing to 1) for a polynomial of degree `qo`.
///
/// MFEM's face rule is `IntRules.Get(Geometry::SEGMENT, order)` — the same
/// points on the segment reference **[-1, 1]**, i.e. `τ = 2ξ − 1` with the
/// weights doubled (they sum to 2 = the reference length).  Call sites that
/// need MFEM's convention (`integrate_bc`, the L2 DG boundary helpers) perform
fn seg_quad(qo: u8) -> (Vec<f64>, Vec<f64>) {
    let re = fem_element::lagrange::SegP1;
    let q = re.quadrature(qo);
    (q.points.iter().map(|p| p[0]).collect(), q.weights)
}

/// The mesh's single element family (D799-2).
///
/// MFEM picks the element's finite element through
/// `FECollection::FiniteElementForGeometry(Geometry::Type)`, i.e. from the
/// element's *own* geometry — never from a fixed guess.  These meshes are
/// built with one geometry type (`Mesh::uniform` in `gen_mesh`), so the family
/// is the type of the first element; uniformity is asserted so that a mixed
/// mesh fails loudly instead of silently assembling with the wrong basis.
fn mesh_family(mesh: &Mesh<2>) -> ElementType {
    let et = mesh.element_type(0);
    assert!(
        (0..mesh.n_elements() as u32).all(|e| mesh.element_type(e) == et),
        "ex27: mixed element families are not supported (element 0 is {et:?})"
    );
    et
}

/// Volume reference element by **element family** (D799-2).
///
/// `ref_elem_vol` already dispatches `Quad4`/`Tri3`/`Tet4` to the right basis
/// (the L²/DG `GaussLegendre` default, MFEM `DG_FECollection` = `L2_FECollection`);
/// the H¹ branch uses the nodal basis of the same family (`QuadQk`, `TriPk` —
/// MFEM's `H1_FECollection`).  Both used to be selected against a hardcoded
/// `ElementType::Quad4`, which on a triangle mesh handed the L² path a 4-dof
/// `QuadL2GL` basis (the space has 3 dofs per element) and the H¹ path a
/// `QuadQk` basis with the wrong dof count and node layout — silent
/// mis-assembly, or an index panic from the 4×4 scatter over 3 dofs.
fn vol_ref_elem(et: ElementType, order: u8, l2: bool) -> Box<dyn ReferenceElement> {
    if l2 {
        ref_elem_vol(et, order)
    } else {
        match et {
            ElementType::Quad4 => Box::new(fem_element::lagrange::QuadQk::new(order as usize)),
            ElementType::Tri3 => Box::new(fem_element::lagrange::factory::TriPk::new(order as usize)),
            _ => panic!("ex27: no H1 volume reference element for {et:?}"),
        }
    }
}


/// MFEM `IntegrateBC`: over the boundary attributes in `tags`, compute the
/// average of `α·n·Grad(u) + β·u` and the L² (root-mean-square) error of
/// `α·n·Grad(u) + β·u − γ`, normalized by the boundary measure.
///
/// Geometry follows MFEM `IntegrateBC` exactly:
/// - the boundary face is one edge of its owner element; the face reference
///   point `t ∈ [-1,1]` maps to the element reference point `eip` affinely
///   (`FTr->Loc1.Transform(ip, eip)`),
/// - physical gradients use the **Q3** isoparametric element geometry
///   (`fe.CalcPhysDShape(*FTr->Elem1, dshape)` = J⁻ᵀ·∇ref),
/// - the face Jacobian / normal come from the face tangent
///   `J·d(eip)/dt` (`FTr->Face->Jacobian()` + `CalcOrtho`).
fn integrate_bc<S: FESpace>(
    space: &S,
    mesh: &Mesh<2>,
    sol: &[f64],
    tags: &[i32],
    alpha: f64,
    beta: f64,
    gamma: f64,
    _qo: u8,
) -> (f64, f64) {
    let dim = 2usize;
    let order = space.order();
    let a_is_zero = alpha == 0.0;
    let b_is_zero = beta == 0.0;
    let mut nrm = 0.0;
    let mut avg = 0.0;
    let mut err2 = 0.0;

    let face_to_elem = build_face_elem_map(mesh, dim);
    // MFEM: int_order = 2*fe.GetOrder() + 3
    let (xi_q, w_q) = seg_quad(2 * order + 3);
    // D779: the element basis must match the space's DOF table.  An L² space
    // numbers its element DOFs in lexicographic `L2_DOF_MAP` order (QuadL2GL),
    // while H1 uses the topological node order (QuadQk).  Pairing the L² table
    // with `QuadQk` (the old unconditional choice, correct only for H1)
    // misaligns dof i ↔ basis i, so the DG verification rows reported
    // n.Grad(u) = 0.449 instead of the imposed 1.0.
    let re = vol_ref_elem(mesh_family(mesh), order, space.l2_basis().is_some());
    let n_dofs = re.n_dofs();

    let mut phi = vec![0.0; n_dofs];
    let mut gref = vec![0.0; n_dofs * dim];
    let mut gphys = vec![0.0; n_dofs * dim];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let mut ud = vec![0.0; n_dofs];
        for (k, &g) in gd.iter().enumerate() { ud[k] = sol[g]; }

        // Local edge of `face` inside the owner element (quad [0,1]²:
        // edge 0 = η=0, edge 1 = ξ=1, edge 2 = η=1, edge 3 = ξ=0).
        // The face's own node order (matching MFEM AddBdrSegment) decides the
        // reference-direction: face ref t=-1 maps to the FIRST face node.
        let en = mesh.element_nodes(elem);
        let fn_ = mesh.face_nodes(f);
        let (pa, pb) = (
            en.iter().position(|&n| n == fn_[0]).unwrap(),
            en.iter().position(|&n| n == fn_[1]).unwrap(),
        );
        let (eip_at, deip): (Box<dyn Fn(f64) -> [f64; 2]>, [f64; 2]) = match (pa, pb) {
            (0, 1) => (Box::new(|t| [0.5 * (1.0 + t), 0.0]), [0.5, 0.0]),
            (1, 0) => (Box::new(|t| [0.5 * (1.0 - t), 0.0]), [-0.5, 0.0]),
            (1, 2) => (Box::new(|t| [1.0, 0.5 * (1.0 + t)]), [0.0, 0.5]),
            (2, 1) => (Box::new(|t| [1.0, 0.5 * (1.0 - t)]), [0.0, -0.5]),
            (2, 3) => (Box::new(|t| [0.5 * (1.0 - t), 1.0]), [-0.5, 0.0]),
            (3, 2) => (Box::new(|t| [0.5 * (1.0 + t), 1.0]), [0.5, 0.0]),
            (3, 0) => (Box::new(|t| [0.0, 0.5 * (1.0 - t)]), [0.0, -0.5]),
            (0, 3) => (Box::new(|t| [0.0, 0.5 * (1.0 + t)]), [0.0, 0.5]),
            _ => panic!("integrate_bc: face not on element edge"),
        };

        for (qi, xi) in xi_q.iter().enumerate() {
            // D778: MFEM's face rule is `IntRules.Get(Geometry::SEGMENT,
            // 2p+3)` — Gauss-Legendre on the segment reference **[-1,1]** —
            // while `seg_quad` returns the [0,1] rule (weights sum 1).  Map the
            // point by `t = 2ξ−1` and scale the weight by 2 before feeding
            // `eip_at`/`deip`, whose `0.5*(1±t)` arms are the [-1,1] face→
            // element map (`FTr->Loc1`).  Feeding ξ directly (the old code)
            // sampled only the second half of every boundary edge, and the
            // missing ×2 halved `nrm`/`avg`.  Same form as the parallel
            // `mfem_pex27_parallel_robin_bc.rs` integrator.
            let t = 2.0 * xi - 1.0;
            // Face ref point t ∈ [-1,1] → element ref point eip ∈ [0,1]².
            let eip = eip_at(t);
            // Q3 isoparametric element geometry: J, det, physical face point.
            let (jq, detq, _xp) = mesh.element_jacobian(elem, &eip);
            // Face tangent dF/dt = J·d(eip)/dt; face_weight = |tangent|.
            let tx = jq[(0, 0)] * deip[0] + jq[(0, 1)] * deip[1];
            let ty = jq[(1, 0)] * deip[0] + jq[(1, 1)] * deip[1];
            let face_weight = (tx * tx + ty * ty).sqrt();
            // CalcOrtho(J_face, w_nor): w_nor = (dy, -dx), |w_nor| = face_weight.
            let nor = [ty, -tx];

            re.eval_basis(&eip, &mut phi);
            re.eval_grad_basis(&eip, &mut gref);
            let jit = jq.clone().try_inverse()
                .unwrap_or_else(|| { eprintln!("  warning: degenerate element"); nalgebra::DMatrix::identity(2, 2) })
                .transpose();
            xform_grads(&jit, &gref, &mut gphys, n_dofs, dim);

            let w = w_q[qi] * 2.0 * face_weight;
            nrm += w;
            let mut val = 0.0;
            if !a_is_zero {
                // α · (∇u · w_nor) / face_weight  →  α · ∇u · n̂
                let mut du_dn = 0.0;
                for k in 0..n_dofs {
                    du_dn += ud[k] * (gphys[k * dim] * nor[0] + gphys[k * dim + 1] * nor[1]);
                }
                val += alpha * du_dn / face_weight;
            }
            if !b_is_zero {
                let mut u = 0.0;
                for k in 0..n_dofs { u += ud[k] * phi[k]; }
                val += beta * u;
            }
            avg += val * w;
            let d = val - gamma;
            err2 += d * d * w;
            let _ = detq;
        }
    }
    if nrm.abs() > 0.0 { avg /= nrm; err2 /= nrm; }
    (avg, err2.sqrt())
}

/// L² (DG) boundary mass on the tagged faces: `∫ κ·u·v ds` scattered to the
/// element-local DOFs (MFEM `BoundaryMassIntegrator`).
fn assemble_l2_mass<S: FESpace>(
    space: &S,
    mesh: &Mesh<2>,
    kappa: f64,
    tags: &[i32],
    qo: u8,
) -> fem_linalg::CsrMatrix<f64> {
    let n = space.n_dofs();
    let mut coo = fem_linalg::CooMatrix::new(n, n);
    let face_to_elem = build_face_elem_map(mesh, 2);
    let (xi_q, w_q) = seg_quad(qo);
    let re = ref_elem_vol(mesh_family(mesh), space.order());
    let n_dofs = re.n_dofs();
    let mut phi = vec![0.0; n_dofs];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let (fa, fb) = (mesh.face_nodes(f)[0], mesh.face_nodes(f)[1]);
        for (qi, xi) in xi_q.iter().enumerate() {
            // D804: MFEM `AddBdrFaceIntegrator(BoundaryMassIntegrator)` is a
            // **BdrFace** integrator: the FaceElementTransformations path —
            // shapes of the *element* at the Loc1-composed point
            // (`face_point_geom`, D795-1), the measure from the FACE
            // transformation, and a scatter over the element's dofs.  The
            // D779-era bdr-element route (1-D `SegP1` shapes scattered to the
            // two face-local dofs only) is MFEM's *other* —
            // `AddBoundaryIntegrator` — path, which ex27 uses for H1 but NOT
            // for DG.
            let g = face_point_geom(mesh, elem, fa, fb, *xi);
            let len = (g.nor[0] * g.nor[0] + g.nor[1] * g.nor[1]).sqrt();
            let w = w_q[qi] * len * kappa;
            re.eval_basis(&g.eip, &mut phi);
            for (i, &gi) in gd.iter().enumerate() {
                for (j, &gj) in gd.iter().enumerate() {
                    coo.add(gi, gj, w * phi[i] * phi[j]);
                }
            }
        }
    }
    coo.into_csr()
}

/// L² (DG) boundary linear form on the tagged faces: `∫ g·v ds` scattered to
/// the element-local DOFs (MFEM `BoundaryLFIntegrator`).
fn assemble_l2_linear<S: FESpace, F: Fn(&[f64], &[f64]) -> f64>(
    space: &S,
    mesh: &Mesh<2>,
    g: F,
    tags: &[i32],
    qo: u8,
) -> Vec<f64> {
    let n = space.n_dofs();
    let mut rhs = vec![0.0; n];
    let face_to_elem = build_face_elem_map(mesh, 2);
    let (xi_q, w_q) = seg_quad(qo);
    let re = ref_elem_vol(mesh_family(mesh), space.order());
    let n_dofs = re.n_dofs();
    let mut phi = vec![0.0; n_dofs];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let (fa, fb) = (mesh.face_nodes(f)[0], mesh.face_nodes(f)[1]);
        for (qi, xi) in xi_q.iter().enumerate() {
            // D804: MFEM `AddBdrFaceIntegrator(BoundaryLFIntegrator)` — the
            // BdrFace path (lininteg.cpp:169): `val = Tr.Face->Weight() ·
            // ip.weight · Q` with `el.CalcShape(eip)` — the *element's* shapes
            // at the Loc1-composed point, scattered over the element's dofs
            // (not the D779 1-D-shape/face-local-dof route, which is MFEM's
            // `AddBoundaryIntegrator` path — that is what the H1 branch uses).
            // The measure is `|nor|` of the element's facet geometry.
            let gpt = face_point_geom(mesh, elem, fa, fb, *xi);
            let len = (gpt.nor[0] * gpt.nor[0] + gpt.nor[1] * gpt.nor[1]).sqrt();
            let w = w_q[qi] * len;
            re.eval_basis(&gpt.eip, &mut phi);
            // ex27's g is constant (`mat·nbc`, `mat·rbc_b`); the physical
            // point is not needed for it, so pass a placeholder.
            let val = g(&[0.0; 2], &gpt.nor);
            for (k, &gk) in gd.iter().enumerate() { rhs[gk] += w * val * phi[k]; }
        }
    }
    rhs
}

/// MFEM `DGDirichletLFIntegrator`: the weak Dirichlet boundary load
/// `∫_Γ u_D·(σ·a·∇v·n + κ·a·h⁻¹·v) ds` on the tagged faces.
///
/// D795-1: MFEM's `AssembleRHSElementVect` (lininteg.cpp:877-924) evaluates the
/// **isoparametric** element geometry — `nor = CalcOrtho(Tr.Jacobian())` (the
/// curved face tangent, `[0,1]` reference segment, weights summing to 1) and
/// `Tr.Elem1->Weight()` / `CalcAdjugate(Tr.Elem1->Jacobian())` at the
/// reference-composed point `Tr.GetElement1IntPoint()`:
///
/// ```text
///   dshape_dn_j = ip.weight·u_D·a·(∇_xφ_j·nor)      (det(J) cancels)
///   elvect_j   += sigma·dshape_dn_j
///               + kappa·ip.weight·u_D·a·|nor|²/det(J)·φ_j
/// ```
///
/// so this is the exact adjoint of `DgAssembler`'s boundary face term.
fn assemble_l2_dg_dirichlet_lf<S: FESpace>(
    space: &S,
    mesh: &Mesh<2>,
    u_d: f64,
    a: f64,
    sigma: f64,
    penalty: f64,
    tags: &[i32],
    qo: u8,
) -> Vec<f64> {
    let dim = 2usize;
    let order = space.order();
    let n = space.n_dofs();
    let mut rhs = vec![0.0; n];
    let face_to_elem = build_face_elem_map(mesh, dim);
    let (xi_q, w_q) = seg_quad(qo);
    let re = ref_elem_vol(mesh_family(mesh), order);
    let n_dofs = re.n_dofs();

    let mut phi = vec![0.0; n_dofs];
    let mut gref = vec![0.0; n_dofs * dim];
    let mut gphys = vec![0.0; n_dofs * dim];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let (fa, fb) = (mesh.face_nodes(f)[0], mesh.face_nodes(f)[1]);

        let mut fe = vec![0.0; n_dofs];
        for (qi, xi) in xi_q.iter().enumerate() {
            // D795-1: reference composition (`FTr->Loc1.Transform(ip, eip)`) on
            // the isoparametric geometry — no physical inverse map, no corner
            // chord normal, and the `[0,1]` segment weights are used as-is
            // (`IntRules.Get(SEGMENT, 2·order)`, `Σw = 1`).
            let g = face_point_geom(mesh, elem, fa, fb, *xi);
            let nor2 = g.nor[0] * g.nor[0] + g.nor[1] * g.nor[1];
            re.eval_basis(&g.eip, &mut phi);
            re.eval_grad_basis(&g.eip, &mut gref);
            xform_grads(&g.jit, &gref, &mut gphys, n_dofs, dim);
            let w = w_q[qi];
            // MFEM: elvect += sigma·(uD·Q·∇v·nor) + kappa·(uD·Q·|nor|²/|det J|·v)
            for k in 0..n_dofs {
                let du_dn = gphys[k * dim] * g.nor[0] + gphys[k * dim + 1] * g.nor[1];
                fe[k] += w * (sigma * u_d * a * du_dn + penalty * u_d * a * nor2 / g.det_j * phi[k]);
            }
        }
        for (k, &g) in gd.iter().enumerate() { rhs[g] += fe[k]; }
    }
    rhs
}
