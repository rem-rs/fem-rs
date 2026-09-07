//! # 3:1 Refinement Miniapp (1:1 port of MFEM `miniapps/meshing/ref321.cpp`)
//!
//! Performs random 3:1 anisotropic refinements of a quadrilateral mesh.
//! A diffusion equation is solved in an H1 finite element space defined on
//! the refined mesh, and its continuity is verified.
//!
//! Port notes (vs C++):
//! - Supports 2D quad meshes (`-mm -dim 2` sample run and any quad mesh file).
//!   The C++ default `star.mesh` (triangles) and `-dim 3` (hex) use NCMesh
//!   simplex/hex trees that are not covered by `general_refinement_2d`.
//! - MFEM's NCMesh eliminates constrained dofs; here the full space is
//!   assembled and constrained via `conforming_assemble` (same math, see
//!   ex15). The printed unknown count is the constrained (true) size and
//!   matches C++ `GetTrueVSize()` for order 1 at all tested `-r`.
//! - `std::mt19937` is re-implemented exactly and `-mm` uses MFEM's
//!   Hilbert SFC element ordering (`make_cartesian_2d_sfc`), so the random
//!   refinement sequence matches C++ for the same `-r`.
//! - Order 1 only: for order >= 2 MFEM's NCMesh uses TraverseEdge point
//!   matrices (slave-edge dof aliasing, free master-edge dofs), which is
//!   not ported yet.
//!
//! Sample runs:
//!   cargo run --release --example mesh_ref321 -- -mm -dim 2 -r 100 -no-vis
//!   cargo run --release --example mesh_ref321 -- -m data/xxx.mesh -r 100 -no-vis

use std::collections::{HashMap, HashSet};

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::mfem::{write_mfem_file, write_mfem_gf_file};
use fem_mesh::amr::general_refinement::{general_refinement_2d, Refinement};
use fem_mesh::{Mesh, element_type::ElementType};
use fem_space::{
    FESpace, H1Space,
    constraints::{apply_dirichlet, boundary_dofs, conforming_assemble},
};
use fem_solver::{SolverConfig, solve_pcg_gssmoother};

// ─── std::mt19937 (exact) ─────────────────────────────────────────────────────

/// Exact re-implementation of C++ `std::mt19937` (MT19937, 32-bit).
struct Mt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, mti: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.mti >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                let mut nxt = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    nxt ^= 0x9908_b0df;
                }
                self.mt[i] = nxt;
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

// ─── Non-conforming constraint detection (geometric, P1) ──────────────────────

/// Projection parameter of point `p` on segment `(a, b)`, or None if `p` is
/// not strictly inside (collinear within tolerance, 0 < t < 1).
fn strictly_inside(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Option<f64> {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let apx = p[0] - a[0];
    let apy = p[1] - a[1];
    let len2 = abx * abx + aby * aby;
    if len2 < 1e-30 {
        return None;
    }
    let t = (apx * abx + apy * aby) / len2;
    // collinearity: |cross|^2 small relative to segment length^2
    let cross = apx * aby - apy * abx;
    if cross * cross > 1e-22 * len2 {
        return None;
    }
    if t > 1e-9 && t < 1.0 - 1e-9 {
        Some(t)
    } else {
        None
    }
}

/// Detect hanging-node constraints on a refined quad mesh (P1).
///
/// A vertex is *hanging* iff some element edge strictly contains it; the
/// constraint is expressed on the longest containing edge (the master):
/// `u_c = (1-t)·u_u + t·u_v` for a vertex inside edge (u,v) at parameter t.
/// The constrained set is exactly the set of dofs MFEM's NCMesh eliminates
/// at order 1: `n_vertices - constraints.len() == GetTrueVSize()`.
fn detect_nc_constraints(
    mesh: &Mesh<2>,
    space: &H1Space<Mesh<2>>,
) -> Vec<fem_mesh::amr::HangingNodeConstraint> {
    use fem_mesh::amr::HangingNodeConstraint;

    let v2d = &space.dof_manager().phys_to_vertex_dof;

    // Collect unique element edges.
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in [(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0])].iter() {
            let key = if a < b { (a, b) } else { (b, a) };
            if edge_set.insert(key) {
                edges.push(key);
            }
        }
    }

    let coords = |n: u32| mesh.coords_of(n);

    // Longest element edge strictly containing point `p`.
    let longest_container = |p: [f64; 2]| -> Option<(u32, u32, f64)> {
        let mut best: Option<(u32, u32, f64, f64)> = None; // (u, v, t, len2)
        for &(u, v) in &edges {
            if let Some(t) = strictly_inside(p, coords(u), coords(v)) {
                let cu = coords(u);
                let cv = coords(v);
                let len2 = (cv[0] - cu[0]).powi(2) + (cv[1] - cu[1]).powi(2);
                if best.map(|(_, _, _, l)| len2 > l).unwrap_or(true) {
                    best = Some((u, v, t, len2));
                }
            }
        }
        best.map(|(u, v, t, _)| (u, v, t))
    };

    let mut out: Vec<HangingNodeConstraint> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();

    for n in 0..mesh.n_nodes() as u32 {
        let p = coords(n);
        if let Some((u, v, t)) = longest_container(p) {
            let dof = *v2d.get(&n).unwrap_or(&n) as usize;
            if seen.insert(dof) {
                out.push(HangingNodeConstraint {
                    constrained: dof,
                    parent_a: u as usize,
                    parent_b: v as usize,
                    coeff_a: 1.0 - t,
                    coeff_b: t,
                    extra: Vec::new(),
                });
            }
        }
    }

    out
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    // 1. Parse command-line options (MFEM OptionsParser subset).
    let mut mesh_file = String::from("../../data/star.mesh");
    let mut order = 1usize;
    let mut visualization = true;
    let mut make_mesh = false;
    let mut num_refs = 1usize;
    let mut tdim = 2usize;

    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => mesh_file = it.next().unwrap().clone(),
            "-o" | "--order" => order = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-mm" | "--make-mesh" => make_mesh = true,
            "-no-mm" | "--no-make-mesh" => make_mesh = false,
            "-dim" | "--dimension" => tdim = it.next().unwrap().parse().unwrap(),
            "-r" | "--refs" => num_refs = it.next().unwrap().parse().unwrap(),
            "-h" | "--help" => {
                println!("Usage: mesh_ref321 [-m mesh] [-o order] [-r refs] [-dim d] [-mm] [-no-vis]");
                return;
            }
            other => panic!("Unknown option: {other}"),
        }
    }
    let _ = visualization; // GLVis socket output is not ported (as in other ports)

    if tdim != 2 {
        panic!("mesh_ref321: only 2D (quad) meshes are supported; use -dim 2");
    }
    if order != 1 {
        panic!(
            "mesh_ref321: only order 1 is supported (MFEM NCMesh TraverseEdge \
             point matrices for order >= 2 are not ported yet)"
        );
    }

    // 2. Create or read the mesh from the given mesh file.
    let mut mesh: Mesh<2> = if make_mesh {
        // Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL)
        // (default sfc_ordering = true → Hilbert-curve element order)
        Mesh::make_cartesian_2d_sfc(2, 2, 1.0, 1.0)
    } else {
        match fem_io::mfem::read_mfem_file(&mesh_file) {
            Ok(m) => m.mesh2d.expect("mesh_ref321: no 2D mesh found in file"),
            Err(e) => panic!("mesh_ref321: cannot read mesh '{mesh_file}': {e}"),
        }
    };

    if mesh.elem_type != ElementType::Quad4 {
        panic!(
            "mesh_ref321: this port supports Quad4 meshes only (got {:?}); \
             use -mm for the 2x2 Cartesian quad mesh",
            mesh.elem_type
        );
    }

    // 3. Randomly perform 3:1 refinements in the mesh (std::mt19937 gen(1)).
    let mut gen = Mt19937::new(1);
    for _ in 0..num_refs {
        let elem = (gen.next() as usize) % mesh.n_elems();
        let t = (gen.next() as usize) % tdim;
        let type_ = if t == 0 { 1u8 } else { 2u8 }; // X : Y
        refine31(&mut mesh, elem as u32, type_);
    }

    // 4. H1 space on the refined mesh; print the constrained (true) size.
    let space = H1Space::new(mesh.clone(), order as u8);
    let cdofs = space.n_dofs();

    // 5. Solve the Poisson problem, as in ex1.
    let quad_rule = (order as u8) * 2 + 1;
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_rule);
    let source = DomainSourceIntegrator::new(|_pt| 1.0);
    let rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_rule);

    // Constrained-space elimination (MFEM NCMesh conforming assembly).
    let hc = detect_nc_constraints(&mesh, &space);
    let (mut mat_true, mut rhs_true, true_dofs) = conforming_assemble(&mat, &rhs_vec, &hc);
    println!("Number of finite element unknowns: {}", true_dofs.len());

    let mut x = vec![0.0f64; cdofs];

    {
        // Dirichlet BC on all boundaries (homogeneous: x was set to 0.0).
        let bnd_tags = space.mesh().unique_boundary_tags();
        let bnd_all = boundary_dofs(space.mesh(), space.dof_manager(), &bnd_tags);
        let true_set: HashSet<usize> = true_dofs.iter().copied().collect();
        let true_idx: HashMap<usize, usize> =
            true_dofs.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        let bnd: Vec<u32> = bnd_all
            .iter()
            .filter(|d| true_set.contains(&(**d as usize)))
            .map(|&d| true_idx[&(d as usize)] as u32)
            .collect();
        let zeros = vec![0.0f64; bnd.len()];
        apply_dirichlet(&mut mat_true, &mut rhs_true, &bnd, &zeros);

        // GSSmoother M(A); PCG(*A, M, B, X, 1, 2000, 1e-12, 0.0);
        let mut x_true = vec![0.0f64; true_dofs.len()];
        let _res = solve_pcg_gssmoother(
            &mat_true,
            &rhs_true,
            &mut x_true,
            &SolverConfig {
                // MFEM PCG rel_tol=1e-12 → solver's sqrt convention (ex15).
                rtol: 1e-6,
                atol: 0.0,
                max_iter: 2000,
                verbose: false,
                ..Default::default()
            },
        )
        .expect("PCG solve failed");

        // a.RecoverFEMSolution(X, b, x);
        for (&td, &v) in true_dofs.iter().zip(x_true.iter()) {
            x[td] = v;
        }
    }

    // 6. Verify the continuity of the projected function in H1.
    let gf = GridFunction::new(&space, x.clone());
    let h1err = check_h1_continuity(&gf, &mesh, order);
    println!("Error of H1 continuity: {h1err:.5e}");
    assert!(h1err < 1.0e-7, "H1 continuity error {h1err} >= 1e-7");

    // 7. Save the refined mesh and the solution.
    write_mfem_file("ref321.mesh", &mesh).expect("write ref321.mesh");
    write_mfem_gf_file("sol.gf", 2, &x, "H1", order as u8, 1, 8).expect("write sol.gf");
    println!("Saved ref321.mesh and sol.gf");
}

/// Refine 3:1 via 2 refinements with scalings 2/3 and 1/2 (C++ `Refine31`).
fn refine31(mesh: &mut Mesh<2>, elem: u32, type_: u8) {
    let r1 = general_refinement_2d(mesh, &[Refinement::new(elem, type_, 2.0 / 3.0)]);
    let children = r1.transforms.find_children(elem);
    assert_eq!(children.len(), 2, "expected exactly 2 children after split");
    let elem1 = children[0];
    let r2 = general_refinement_2d(&r1.mesh, &[Refinement::with_midpoint(elem1, type_)]);
    *mesh = r2.mesh;
}

/// C++ `CheckH1Continuity`: max jump of the FE solution across interior
/// faces, sampled with Gauss-Legendre quadrature of order 2*order.
fn check_h1_continuity(gf: &GridFunction<H1Space<Mesh<2>>>, mesh: &Mesh<2>, order: usize) -> f64 {
    // Interior edges = node-pair edges shared by exactly two elements.
    let mut edge_elems: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in [(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[0])].iter() {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_elems.entry(key).or_default().push(e);
        }
    }

    // IntRules.Get(SEGMENT, 2*faceOrder): Gauss-Legendre with order+1 points.
    let (pts, _wts) = fem_element::quadrature::gauss_legendre_01(order + 1);

    let mut error_max = 0.0f64;
    for (&(u, v), elems) in &edge_elems {
        if elems.len() != 2 {
            continue;
        }
        let (e1, e2) = (elems[0], elems[1]);
        let xu = mesh.coords_of(u);
        let xv = mesh.coords_of(v);
        for &t in &pts {
            let p = [
                xu[0] + t * (xv[0] - xu[0]),
                xu[1] + t * (xv[1] - xu[1]),
            ];
            let v1 = eval_on_elem(gf, mesh, e1, &p);
            let v2 = eval_on_elem(gf, mesh, e2, &p);
            error_max = error_max.max((v1 - v2).abs());
        }
    }
    error_max
}

    // Evaluate the FE solution at physical point `p` on `e`; `p` must lie on
    // one of the element's edges (Loc1/Loc2 transform in the C++ miniapp).
    fn eval_on_elem(
        gf: &GridFunction<H1Space<Mesh<2>>>,
        mesh: &Mesh<2>,
        e: u32,
        p: &[f64; 2],
    ) -> f64 {
        let ns = mesh.elem_nodes(e);
        // Quad4 local edges: 0 (n0,n1) bottom, 1 (n1,n2) right, 2 (n3,n2)
        // top, 3 (n0,n3) left; reference domain [0,1]^2.
        let local_edges = [(0usize, 1usize), (1, 2), (3, 2), (0, 3)];
        for &(a, b) in &local_edges {
            let ca = mesh.coords_of(ns[a]);
            let cb = mesh.coords_of(ns[b]);
            let abx = cb[0] - ca[0];
            let aby = cb[1] - ca[1];
            let len2 = abx * abx + aby * aby;
            if len2 < 1e-30 {
                continue;
            }
            let apx = p[0] - ca[0];
            let apy = p[1] - ca[1];
            let cross = apx * aby - apy * abx;
            let t = (apx * abx + apy * aby) / len2;
            if cross * cross <= 1e-22 * len2 && t >= -1e-9 && t <= 1.0 + 1e-9 {
                let tt = t.clamp(0.0, 1.0);
                // Map edge parameter to [0,1]^2 reference coordinates.
                let (rx, ry) = match (a, b) {
                    (0, 1) => (tt, 0.0),
                    (1, 2) => (1.0, tt),
                    (3, 2) => (tt, 1.0),
                    (0, 3) => (0.0, tt),
                    _ => unreachable!(),
                };
                return gf.evaluate_at_element(e, &[rx, ry]);
            }
        }
        // p is a vertex of e (quadrature point at a segment end shared by two
        // local edges) — evaluate at the matching corner.
        for (i, &n) in ns.iter().enumerate() {
            let c = mesh.coords_of(n);
            if (c[0] - p[0]).abs() < 1e-12 && (c[1] - p[1]).abs() < 1e-12 {
                let (rx, ry) = match i {
                    0 => (0.0, 0.0),
                    1 => (1.0, 0.0),
                    2 => (1.0, 1.0),
                    _ => (0.0, 1.0),
                };
                return gf.evaluate_at_element(e, &[rx, ry]);
            }
        }
        panic!("check_h1_continuity: point not on element {e}");
    }
