//! # Serial hp-refinement miniapp (1:1 port of MFEM `miniapps/meshing/hpref.cpp`)
//!
//! Demonstrates h- and p-refinement in a serial finite element
//! discretization of the Poisson problem (cf. ex1): -Delta u = 1 with
//! homogeneous Dirichlet boundary conditions.  Refinements are performed
//! iteratively, each iteration having h- or p-refinements, chosen randomly
//! (deterministically, via MFEM's `DetRand`) for each iteration.
//!
//! Port notes (vs C++):
//! - 2D quad meshes only (`-dim 2` default mesh, or a quad mesh file via
//!   `-m`).  The `-dim 3` hex path and `star-mixed.mesh` (simplices) are not
//!   ported: `PRefinementSupported`-style hp support here covers quads.
//! - MFEM's variable-order H1 semantics are reproduced through the
//!   variant-scheme space (`H1Space::p_refine_update` /
//!   `with_element_orders`): per-edge DOF sets per adjacent element order,
//!   minimum-rule constraints on conforming edges, and master/slave aliasing
//!   on hanging (non-conforming) edges.
//! - `mesh.GeneralRefinement([Refinement(elem)])` (isotropic XY at 0.5) is
//!   realized with two `general_refinement_2d` splits and MFEM's in-place
//!   child ordering [BL, BR, TR, TL]; children inherit the parent order
//!   (MFEM `fespace.cpp Update`).
//! - The true (constrained) system is built with `conforming_assemble`
//!   (MFEM `BilinearForm::ConformingAssemble`); the printed unknown count is
//!   `GetTrueVSize()`.
//! - `order.gf` (element orders as a P0 L2 field) and `refined.mesh` are
//!   written; `sol.gf` (variable-order grid function, MFEM fes format 100)
//!   is not written.
//!
//! Sample runs:
//!   cargo run --release --example mesh_hpref -- -dim 2 -n 1000 -no-vis
//!   cargo run --release --example mesh_hpref -- -m data/star-mixed.mesh -pref -n 100 -no-vis

use std::collections::HashMap;

use fem_assembly::{
    postproc::grid_function::GridFunction, Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_io::mfem::write_mfem_file;
use fem_mesh::amr::general_refinement::{general_refinement_2d, Refinement};
use fem_mesh::amr::HangingNodeConstraint;
use fem_mesh::{Mesh, element_type::ElementType};
use fem_space::{
    FESpace, H1Space,
    constraints::{apply_dirichlet, boundary_dofs, conforming_assemble, recover_hanging_values},
    p_refine::{PRefineConstraint, detect_nc_geometry_2d},
};
use fem_solver::{SolverConfig, solve_pcg_gssmoother};

// ─── Deterministic "random" integers (MFEM DetRand) ───────────────────────────

/// Deterministic function for "random" integers (MFEM `DetRand`).
fn det_rand(seed: &mut i32) -> i32 {
    *seed += 1;
    (1.0e5 * (*seed as f64 * 1.1234 * std::f64::consts::PI).sin()).abs() as i32
}

// ─── PRefineConstraint → HangingNodeConstraint conversion ─────────────────────

/// Multi-parent p/hp constraints are expressed through the general
/// two-parent + extra `HangingNodeConstraint` layout consumed by
/// `conforming_assemble` / `recover_hanging_values`.
fn to_hanging_constraint(c: &PRefineConstraint) -> HangingNodeConstraint {
    let constrained = c.constrained as usize;
    match c.parents.len() {
        0 => HangingNodeConstraint {
            constrained, parent_a: constrained, parent_b: constrained,
            coeff_a: 1.0, coeff_b: 0.0, extra: vec![],
        },
        1 => {
            let (p, w) = c.parents[0];
            let p = p as usize;
            HangingNodeConstraint {
                constrained, parent_a: p, parent_b: p,
                coeff_a: w, coeff_b: 0.0, extra: vec![],
            }
        }
        _ => {
            let (pa, ca) = c.parents[0];
            let (pb, cb) = c.parents[1];
            HangingNodeConstraint {
                constrained, parent_a: pa as usize, parent_b: pb as usize,
                coeff_a: ca, coeff_b: cb,
                extra: c.parents[2..].iter().map(|&(d, w)| (d as usize, w)).collect(),
            }
        }
    }
}

// ─── Isotropic quad refinement with MFEM child ordering ───────────────────────

/// MFEM `ncmesh_tables.hpp`: Hilbert-curve child visit order / successor state
/// for an XY-split quad, indexed by the element's Hilbert state.  NCMesh
/// orders leaf elements along a space-filling curve
/// (`NCMesh::CollectLeafElements`), so the slots the children take in the
/// element array follow these tables, not the creation order.
const QUAD_HILBERT_CHILD_ORDER: [[usize; 4]; 8] = [
    [0, 1, 2, 3], [0, 3, 2, 1], [1, 2, 3, 0], [1, 0, 3, 2],
    [2, 3, 0, 1], [2, 1, 0, 3], [3, 0, 1, 2], [3, 2, 1, 0],
];
const QUAD_HILBERT_CHILD_STATE: [[u8; 4]; 8] = [
    [1, 0, 0, 5], [0, 1, 1, 4], [3, 2, 2, 7], [2, 3, 3, 6],
    [5, 4, 4, 1], [4, 5, 5, 0], [7, 6, 6, 3], [6, 7, 7, 2],
];

/// Hilbert states of the 4 roots of `MakeCartesian2D(2, 2, QUADRILATERAL,
/// true)` (MFEM assigns them while laying out the initial Hilbert curve; the
/// values match the NCMesh `root_state` of the equivalent C++ run).
const INITIAL_ROOT_STATES: [u8; 4] = [0, 1, 1, 4];

/// `mesh.GeneralRefinement([Refinement(elem)])` for quads: isotropic (XY)
/// midpoint split creating 4 children that replace the parent in MFEM's leaf
/// (Hilbert-SFC) order, plus order inheritance (MFEM: children inherit the
/// parent's element order in `FiniteElementSpace::Update`).  `states` tracks
/// the Hilbert state per element; the split is built from two
/// `general_refinement_2d` calls (X, then Y on both halves).
fn h_refine_with_orders(
    mesh: &Mesh<2>,
    orders: &[u8],
    states: &[u8],
    e: u32,
) -> (Mesh<2>, Vec<u8>, Vec<u8>) {
    let r1 = general_refinement_2d(mesh, &[Refinement::with_midpoint(e, 1)]);
    let children = r1.transforms.find_children(e);
    assert_eq!(children.len(), 2, "X split must create 2 children");
    let r2 = general_refinement_2d(&r1.mesh, &[
        Refinement::with_midpoint(children[0], 2),
        Refinement::with_midpoint(children[1], 2),
    ]);
    let mut mesh = r2.mesh;

    // The two-step split leaves the 4 leaves at slots [c0, c0+1, c1, c1+1] in
    // corner order [v0, v3, v1, v2] (X split: left/right, then Y split of
    // each); MFEM's creation order is corners [v0, v1, v2, v3] (ncmesh.cpp
    // Refine, XY case).
    let (c0, c1) = (children[0] as usize, children[1] as usize);
    let by_corner = [c0, c1 + 1, c1 + 2, c0 + 1]; // corner -> my slot
    let p = e as usize;

    // Order inheritance through the chained embeddings: for each new element
    // walk the r2 → r1 → original parent chain and take the ancestor's order.
    // `RefinementTransforms.embeddings` holds one entry per NEW element:
    // `Some(parent)` for children, `None` for unrefined elements.  The
    // unrefined ones keep consecutive old ids, skipping the refined old
    // slots, so a sequential walk recovers the full old-id mapping (a plain
    // None-count is wrong as soon as an earlier slot was refined).
    fn parent_map(emb: &[Option<u32>]) -> Vec<usize> {
        let mut map = Vec::with_capacity(emb.len());
        let mut old = 0usize;
        for k in 0..emb.len() {
            match emb[k] {
                Some(p) => {
                    map.push(p as usize);
                    old = p as usize + 1;
                }
                None => {
                    map.push(old);
                    old += 1;
                }
            }
        }
        map
    }
    let par1 = parent_map(&r1.transforms.embeddings);
    let par2 = parent_map(&r2.transforms.embeddings);
    let n_old = orders.len();
    let mut new_orders = Vec::with_capacity(mesh.n_elems());
    let mut new_states = Vec::with_capacity(mesh.n_elems());
    for k in 0..mesh.n_elems() {
        // parent of new element k in the r1 mesh, then its ancestor
        let p1 = par2[k];
        let p0 = par1[p1];
        debug_assert!(p0 < n_old);
        new_orders.push(orders[p0]);
        new_states.push(states[p0]);
    }

    // Reorder the 4 children to MFEM's Hilbert visit order: slot p+i receives
    // creation-order child QUAD_HILBERT_CHILD_ORDER[st][i], whose Hilbert
    // successor state is QUAD_HILBERT_CHILD_STATE[st][i].
    let st = states[p] as usize;
    let mut child_conn = [[0u32; 4]; 4];
    let mut child_tag = [0i32; 4];
    let mut child_ord = [0u8; 4];
    for (corner, &slot) in by_corner.iter().enumerate() {
        for k in 0..4 {
            child_conn[corner][k] = mesh.conn[slot * 4 + k];
        }
        child_tag[corner] = mesh.elem_tags[slot];
        child_ord[corner] = new_orders[slot];
    }
    for (i, &ch) in QUAD_HILBERT_CHILD_ORDER[st].iter().enumerate() {
        for k in 0..4 {
            mesh.conn[(p + i) * 4 + k] = child_conn[ch][k];
        }
        mesh.elem_tags[p + i] = child_tag[ch];
        new_orders[p + i] = child_ord[ch];
        new_states[p + i] = QUAD_HILBERT_CHILD_STATE[st][i];
    }

    (mesh, new_orders, new_states)
}

// ─── Element evaluation on edges (CheckH1Continuity helpers) ──────────────────

/// Reference-corner coordinates of a quad in H1 order.
const QUAD_CORNER_REF: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// Evaluate the FE solution at physical point `p` assumed to lie on a local
/// edge of element `e`.  Returns None if `p` is not on any edge of `e`.
fn eval_on_elem_edge(
    gf: &GridFunction<H1Space<Mesh<2>>>,
    mesh: &Mesh<2>,
    e: u32,
    p: &[f64; 2],
) -> Option<f64> {
    let ns = mesh.elem_nodes(e);
    for k in 0..4 {
        let (i, j) = (ns[k], ns[(k + 1) % 4]);
        let ca = mesh.coords_of(i);
        let cb = mesh.coords_of(j);
        let abx = cb[0] - ca[0];
        let aby = cb[1] - ca[1];
        let len2 = abx * abx + aby * aby;
        if len2 < 1e-30 { continue; }
        let apx = p[0] - ca[0];
        let apy = p[1] - ca[1];
        let cross = apx * aby - apy * abx;
        let t = (apx * abx + apy * aby) / len2;
        if cross * cross <= 1e-22 * len2 && t >= -1e-9 && t <= 1.0 + 1e-9 {
            let tt = t.clamp(0.0, 1.0);
            // Reference coordinates: lerp between the two corner references.
            let (ri, rj) = (&QUAD_CORNER_REF[k], &QUAD_CORNER_REF[(k + 1) % 4]);
            let xi = [(1.0 - tt) * ri[0] + tt * rj[0], (1.0 - tt) * ri[1] + tt * rj[1]];
            return Some(gf.evaluate_at_element(e, &xi));
        }
    }
    // Point may be a corner of e (quadrature point at a segment end).
    for (k, &n) in ns.iter().enumerate() {
        let c = mesh.coords_of(n);
        if (c[0] - p[0]).abs() < 1e-12 && (c[1] - p[1]).abs() < 1e-12 {
            return Some(gf.evaluate_at_element(e, &QUAD_CORNER_REF[k]));
        }
    }
    None
}

/// C++ `CheckH1Continuity` for hp spaces on (possibly NC) quad meshes:
/// the maximum jump of the solution across interior edges, sampled with
/// Gauss-Legendre quadrature of order 2 * faceOrder.  Conforming edges
/// compare the two adjacent elements; master edges compare the coarse
/// element against every fine element on the slave segments.
fn check_h1_continuity(
    gf: &GridFunction<H1Space<Mesh<2>>>,
    mesh: &Mesh<2>,
    space: &H1Space<Mesh<2>>,
) -> f64 {
    // Element adjacency per element edge.
    let mut edge_elems: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for k in 0..4 {
            let (a, b) = (ns[k], ns[(k + 1) % 4]);
            let key = if a < b { (a, b) } else { (b, a) };
            edge_elems.entry(key).or_default().push(e);
        }
    }

    let mut error_max = 0.0f64;

    // ── Conforming edges (two adjacent elements) ──
    for (&(u, v), elems) in &edge_elems {
        if elems.len() != 2 { continue; }
        let (e1, e2) = (elems[0], elems[1]);
        let face_order = space.element_order(e1).max(space.element_order(e2)) as usize;
        let (pts, _wts) = fem_element::quadrature::gauss_legendre_01(face_order + 1);
        let xu = mesh.coords_of(u);
        let xv = mesh.coords_of(v);
        for &t in &pts {
            let p = [xu[0] + t * (xv[0] - xu[0]), xu[1] + t * (xv[1] - xu[1])];
            if let (Some(v1), Some(v2)) = (
                eval_on_elem_edge(gf, mesh, e1, &p),
                eval_on_elem_edge(gf, mesh, e2, &p),
            ) {
                error_max = error_max.max((v1 - v2).abs());
            }
        }
    }

    // ── Non-conforming (master/slave) edges ──
    let nc = detect_nc_geometry_2d(mesh);
    let mut elems_by_edge: HashMap<(u32, u32), Vec<u32>> = edge_elems
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    for s in &nc.slaves {
        let (a, b) = (s.key.0, s.key.1);
        let fine_elems = elems_by_edge.entry((a, b)).or_default().clone();
        // The coarse element owns the master edge.
        let coarse = elems_by_edge
            .entry((s.master.0, s.master.1))
            .or_default()
            .first()
            .copied();
        let Some(coarse) = coarse else { continue };
        let mut face_order = space.element_order(coarse) as usize;
        for &f in &fine_elems {
            face_order = face_order.max(space.element_order(f) as usize);
        }
        let (pts, _wts) = fem_element::quadrature::gauss_legendre_01(face_order + 1);
        let ca = mesh.coords_of(a);
        let cb = mesh.coords_of(b);
        let cu = mesh.coords_of(s.master.0);
        let cv = mesh.coords_of(s.master.1);
        for &t in &pts {
            let p = [ca[0] + t * (cb[0] - ca[0]), ca[1] + t * (cb[1] - ca[1])];
            // Master parameter of the sample point (from master.0).
            let mx = cv[0] - cu[0];
            let my = cv[1] - cu[1];
            let tm = ((p[0] - cu[0]) * mx + (p[1] - cu[1]) * my)
                / (mx * mx + my * my);
            let v_coarse = eval_on_elem_edge(gf, mesh, coarse, &p);
            if let Some(vc) = v_coarse {
                for &f in &fine_elems {
                    if let Some(vf) = eval_on_elem_edge(gf, mesh, f, &p) {
                        error_max = error_max.max((vc - vf).abs());
                    }
                }
            }
            let _ = tm;
        }
    }

    error_max
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    // 1. Parse command-line options (MFEM OptionsParser subset).
    let mut mesh_file = String::new();
    let mut order = 1usize;
    let mut visualization = true;
    let mut num_iter = 0usize;
    let mut dim = 2usize;
    let mut deterministic = true;
    let mut project_solution = false;
    let mut only_pref = false;

    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => mesh_file = it.next().unwrap().clone(),
            "-o" | "--order" => order = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-n" | "--num-iter" => num_iter = it.next().unwrap().parse().unwrap(),
            "-dim" | "--dim" => dim = it.next().unwrap().parse().unwrap(),
            "-det" | "--deterministic" => deterministic = true,
            "-not-det" | "--not-deterministic" => deterministic = false,
            "-proj" | "--project-solution" => project_solution = true,
            "-no-proj" | "--no-project" => project_solution = false,
            "-pref" | "--only-p-refinement" => only_pref = true,
            "-no-pref" | "--hp-refinement" => only_pref = false,
            "-h" | "--help" => {
                println!("Usage: mesh_hpref [-m mesh] [-o order] [-n num-iter] [-dim d] [-pref] [-proj] [-no-vis]");
                return;
            }
            other => panic!("Unknown option: {other}"),
        }
    }
    let _ = visualization; // GLVis socket output is not ported (as in other ports)

    if dim != 2 {
        panic!("mesh_hpref: only 2D (quad) meshes are supported; use -dim 2");
    }
    if project_solution {
        panic!("mesh_hpref: -proj (coefficient projection) is not ported");
    }

    // 3. Construct or load a coarse mesh.
    let mut mesh: Mesh<2> = if !mesh_file.is_empty() {
        match fem_io::mfem::read_mfem_file(&mesh_file) {
            Ok(m) => m.mesh2d.expect("mesh_hpref: no 2D mesh found in file"),
            Err(e) => panic!("mesh_hpref: cannot read mesh '{mesh_file}': {e}"),
        }
    } else {
        // Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true)
        // (default sfc_ordering = true → Hilbert-curve element order).
        Mesh::make_cartesian_2d_sfc(2, 2, 1.0, 1.0)
    };
    if mesh.elem_type != ElementType::Quad4 {
        panic!(
            "mesh_hpref: this port supports Quad4 meshes only (got {:?})",
            mesh.elem_type
        );
    }

    // mesh.EnsureNCMesh(): the flat quad mesh with geometric NC detection
    // plays the role of the NC mesh (no-op).

    // 4. Define a finite element space (H1_FECollection(order, dim)).
    //    The loop only needs the per-element order array (MFEM rebuilds the
    //    space every iteration, but nothing consumes it until the solve), so
    //    the space is constructed once after the loop.
    let mut orders = vec![order as u8; mesh.n_elems()];
    // Hilbert-SFC state per element (MFEM NCMesh root states).
    let mut states = INITIAL_ROOT_STATES.to_vec();

    // 5. Iteratively perform h- and p-refinements (MFEM DetRand sequence).
    let mut num_h = 0usize;
    let mut num_p = 0usize;
    let mut seed = 0i32;

    for iter in 0..num_iter {
        let r1 = if deterministic { det_rand(&mut seed) } else {
            // -not-det: use the wall clock as entropy (not MFEM's rand()).
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as i32
        };
        let r2 = if deterministic { det_rand(&mut seed) } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as i32
        };
        let elem = (r1 as usize) % mesh.n_elems();
        let hp = if only_pref { 1 } else { (r2 as usize) % 2 };

        println!(
            "hp-refinement iteration {iter}: {}-refinement",
            if hp == 1 { "p" } else { "h" }
        );

        if hp == 1 {
            // p-ref: refs.Append(pRefinement(elem, 1)); fespace.PRefineAndUpdate(refs);
            orders[elem] += 1;
            num_p += 1;
        } else {
            // h-ref: mesh.GeneralRefinement(refs); fespace.Update(false);
            let (new_mesh, new_orders, new_states) =
                h_refine_with_orders(&mesh, &orders, &states, elem as u32);
            mesh = new_mesh;
            orders = new_orders;
            states = new_states;
            num_h += 1;
        }
    }

    // Define the variable-order space and its hp constraints (MFEM's final
    // fespace state after the loop).
    let space = H1Space::new_variable(mesh.clone(), orders.clone());

    let size = {
        // GetTrueVSize(): ndofs minus the constrained (slave) dofs.
        space.n_dofs() - space.hp_constraints().len()
    };
    println!("Number of finite element unknowns: {size}");
    let max_p = space.order();
    println!(
        "Total number of h-refinements: {num_h}\nTotal number of p-refinements: {num_p}\nMaximum order {max_p}\n"
    );

    // 6-12. Assemble and solve -Delta u = 1 with homogeneous Dirichlet BCs.
    let quad_order = 2 * max_p as u8 + 1;
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_order);
    let source = DomainSourceIntegrator::new(|_pt| 1.0);
    let rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_order);

    // Conforming assembly (MFEM ConformingAssemble) over the hp constraints.
    let hp_cons = space.hp_constraints();
    let hanging: Vec<HangingNodeConstraint> =
        hp_cons.iter().map(to_hanging_constraint).collect();
    let (mut mat_true, mut rhs_true, true_dofs) =
        conforming_assemble(&mat, &rhs_vec, &hanging);

    // Essential BCs: all boundary attributes, restricted to true dofs.
    let bnd_tags = space.mesh().unique_boundary_tags();
    let bnd_all = boundary_dofs(space.mesh(), space.dof_manager(), &bnd_tags);
    let true_set: std::collections::HashSet<usize> =
        true_dofs.iter().copied().collect();
    let true_idx: HashMap<usize, usize> =
        true_dofs.iter().enumerate().map(|(i, &d)| (d, i)).collect();
    let bnd: Vec<u32> = bnd_all
        .iter()
        .filter(|d| true_set.contains(&(**d as usize)))
        .map(|&d| true_idx[&(d as usize)] as u32)
        .collect();
    let zeros = vec![0.0f64; bnd.len()];
    apply_dirichlet(&mut mat_true, &mut rhs_true, &bnd, &zeros);

    // GSSmoother M((SparseMatrix&)(*A)); PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
    let mut x_true = vec![0.0f64; true_dofs.len()];
    let _res = solve_pcg_gssmoother(
        &mat_true,
        &rhs_true,
        &mut x_true,
        &SolverConfig {
            // MFEM PCG rel_tol=1e-12 → this solver's sqrt convention (ex15).
            rtol: 1e-6,
            atol: 0.0,
            max_iter: 200,
            verbose: true,
            ..Default::default()
        },
    )
    .expect("PCG solve failed");

    // a.RecoverFEMSolution(X, b, x) — scatter + recover constrained dofs.
    let mut x = vec![0.0f64; space.n_dofs()];
    for (&td, &v) in true_dofs.iter().zip(x_true.iter()) {
        x[td] = v;
    }
    recover_hanging_values(&mut x, &hanging);

    // H1 continuity check (MFEM_VERIFY(h1error < 1.0e-12)).
    let gf = GridFunction::new(&space, x.clone());
    let h1error = check_h1_continuity(&gf, &mesh, &space);
    println!("H1 continuity error {h1error}");
    assert!(h1error < 1.0e-12, "H1 continuity is not satisfied");

    // order.gf: element orders as a P0 L2 field (MFEM writes xo over an
    // L2_FECollection(0, dim) space).
    write_mfem_file("refined.mesh", &mesh).expect("write refined.mesh");
    let _ = fem_io::mfem::write_mfem_gf_file(
        "order.gf", 2, &orders.iter().map(|&p| p as f64).collect::<Vec<f64>>(),
        "L2", 0, 1, 8,
    )
    .map_err(|e| eprintln!("warning: could not write order.gf: {e}"));
    println!("Saved refined.mesh and order.gf");
}
