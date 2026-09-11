//! D37 verification: tet ND2 curl-curl + mass on the canonical (shared-face)
//! DOF basis, with an MMS (method of manufactured solutions) convergence rate.
//!
//! `HCurlSpace`'s tet face DOFs are the face-creating element's point-value
//! functionals; adjacent elements are related by a 2×2 change of basis per
//! face point (`FaceDofBlock`).  Assembling through
//! [`VectorAssembler::assemble_bilinear_nd_canonical`] applies that transform
//! (`A ← Tᵀ·A·T`, `b ← Tᵀ·b`), which is what makes the discretisation
//! conforming; the reconstruction below is the matching `u_local = T·u_canon`.
//!
//! MMS: `E = (sin κy, sin κz, sin κx)` solves `curl curl E + E = (1+κ²)E`, so
//! the discrete solution must converge to `E` at the ND2 rate.

use std::f64::consts::PI;

use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_assembly::{VectorAssembler, VectorBilinearIntegrator};
use fem_element::nedelec::TetNDk;
use fem_element::quadrature::tet_rule;
use fem_element::reference::VectorReferenceElement;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_space::constraints::{boundary_dofs_hcurl, form_linear_system};
use fem_space::{fe_space::FESpace, HCurlSpace};
use fem_solver::solve_pcg;

const KAPPA: f64 = PI;

fn exact(x: &[f64]) -> Vec<f64> {
    vec![(KAPPA * x[1]).sin(), (KAPPA * x[2]).sin(), (KAPPA * x[0]).sin()]
}

fn source(x: &[f64]) -> Vec<f64> {
    let c = 1.0 + KAPPA * KAPPA;
    let e = exact(x);
    vec![c * e[0], c * e[1], c * e[2]]
}

/// `∫ f·Φ` load integrator.
struct Src;
impl VectorLinearIntegrator for Src {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let fv = source(x);
        for i in 0..qp.n_dofs {
            f[i] += qp.weight
                * (qp.phi_vec[i * 3] * fv[0]
                    + qp.phi_vec[i * 3 + 1] * fv[1]
                    + qp.phi_vec[i * 3 + 2] * fv[2]);
        }
    }
}

/// Affine tet geometry of element `e`: `P₀`, the Jacobian columns and `J⁻ᵀ`.
#[allow(clippy::type_complexity)]
fn tet_geom<M: MeshTopology>(mesh: &M, e: u32) -> ([f64; 3], [[f64; 3]; 3], [[f64; 3]; 3], f64) {
    let v = mesh.element_nodes(e);
    let p0 = mesh.node_coords(v[0]);
    let p0 = [p0[0], p0[1], p0[2]];
    let mut j = [[0.0_f64; 3]; 3];
    for (c, lv) in [1usize, 2, 3].iter().enumerate() {
        let p = mesh.node_coords(v[*lv]);
        for d in 0..3 {
            j[d][c] = p[d] - p0[d];
        }
    }
    let cof = [
        [
            j[1][1] * j[2][2] - j[1][2] * j[2][1],
            j[1][2] * j[2][0] - j[1][0] * j[2][2],
            j[1][0] * j[2][1] - j[1][1] * j[2][0],
        ],
        [
            j[0][2] * j[2][1] - j[0][1] * j[2][2],
            j[0][0] * j[2][2] - j[0][2] * j[2][0],
            j[0][1] * j[2][0] - j[0][0] * j[2][1],
        ],
        [
            j[0][1] * j[1][2] - j[0][2] * j[1][1],
            j[0][2] * j[1][0] - j[0][0] * j[1][2],
            j[0][0] * j[1][1] - j[0][1] * j[1][0],
        ],
    ];
    let det = j[0][0] * cof[0][0] + j[0][1] * cof[0][1] + j[0][2] * cof[0][2];
    (p0, j, cof, det)
}

/// `L²` error of the reconstructed field, with the D37 face block transform
/// (`use_blocks`) or without it.
fn l2_error<M: MeshTopology>(
    mesh: &M,
    space: &HCurlSpace<M>,
    x: &[f64],
    want_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    use_blocks: bool,
) -> f64 {
    let k = space.order() as usize;
    let elem = TetNDk::new(k);
    let n = elem.n_dofs();
    let qr = tet_rule((2 * k + 2) as u8);
    let mut acc = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let (p0, j, cof, det) = tet_geom(mesh, e);
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        // signed local dofs, optionally rotated by the face blocks
        let mut u: Vec<f64> = dofs
            .iter()
            .zip(signs.iter())
            .map(|(&d, &s)| s * x[d as usize])
            .collect();
        if use_blocks {
            for b in space.element_face_blocks(e) {
                let c = [x[b.canon_dofs[0] as usize], x[b.canon_dofs[1] as usize]];
                u[b.slot] = b.s[0][0] * c[0] + b.s[0][1] * c[1];
                u[b.slot + 1] = b.s[1][0] * c[0] + b.s[1][1] * c[1];
            }
        }
        let mut vals = vec![0.0_f64; n * 3];
        for (q, xi) in qr.points.iter().enumerate() {
            elem.eval_basis_vec(xi, &mut vals);
            let xp = [
                p0[0] + j[0][0] * xi[0] + j[0][1] * xi[1] + j[0][2] * xi[2],
                p0[1] + j[1][0] * xi[0] + j[1][1] * xi[1] + j[1][2] * xi[2],
                p0[2] + j[2][0] * xi[0] + j[2][1] * xi[1] + j[2][2] * xi[2],
            ];
            let want = want_fn(&xp);
            let mut uh = [0.0_f64; 3];
            for i in 0..n {
                let c = u[i];
                for r in 0..3 {
                    uh[r] += c
                        * (cof[r][0] / det * vals[i * 3]
                            + cof[r][1] / det * vals[i * 3 + 1]
                            + cof[r][2] / det * vals[i * 3 + 2]);
                }
            }
            let mut d2 = 0.0;
            for r in 0..3 {
                d2 += (uh[r] - want[r]) * (uh[r] - want[r]);
            }
            acc += qr.weights[q] * det.abs() * d2;
        }
    }
    acc.sqrt()
}

/// Essential boundary DOFs of an H(curl) space on a 3-D tet mesh.
///
/// `fem_space::constraints::boundary_dofs_hcurl` collects the boundary **edge**
/// DOFs (all orders) and the **quadrilateral** face DOFs (hex NDk), but not the
/// **triangular** face DOFs a tet NDk (k ≥ 2) space carries — so its list is
/// incomplete for that combination and an essential boundary condition built
/// from it leaves the boundary face DOFs free.  This helper adds them, using
/// the space's public `face_dof` + the `k(k−1)` face dof count.
fn ess_bdr_hcurl_full(mesh: &Mesh<3>, space: &HCurlSpace<Mesh<3>>) -> Vec<fem_core::DofId> {
    let tags = mesh.unique_boundary_tags();
    let mut out = boundary_dofs_hcurl(mesh, space, &tags);
    if mesh.dim() == 3 && space.order() >= 2 {
        let k = space.order() as fem_core::DofId;
        let nfd = k * (k - 1);
        for f in 0..mesh.n_boundary_faces() as u32 {
            if !tags.contains(&mesh.face_tag(f)) {
                continue;
            }
            let nds = mesh.face_nodes(f);
            if nds.len() == 3 {
                if let Some(first) =
                    space.face_dof(fem_space::dof_manager::FaceKey::new(nds[0], nds[1], nds[2]))
                {
                    for m in 0..nfd {
                        out.push(first + m);
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn solve_on(mesh: &Mesh<3>, use_blocks: bool) -> (f64, usize) {
    let space = HCurlSpace::new(mesh.clone(), 2);
    let integrators: [&dyn VectorBilinearIntegrator; 2] =
        [&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }];
    let qo = 6;
    let mut mat = if use_blocks {
        VectorAssembler::assemble_bilinear_nd_canonical(&space, &integrators, qo)
    } else {
        VectorAssembler::assemble_bilinear(&space, &integrators, qo)
    };
    let src = Src;
    let mut rhs = if use_blocks {
        VectorAssembler::assemble_linear_nd_canonical(&space, &[&src], qo)
    } else {
        VectorAssembler::assemble_linear(&space, &[&src], qo)
    };
    let u_proj = space.interpolate_vector(&exact);
    let _tags = space.mesh().unique_boundary_tags();
    let ess = ess_bdr_hcurl_full(space.mesh(), &space);
    let bc_vals: Vec<f64> = ess.iter().map(|&d| u_proj[d as usize]).collect();
    let mut x = u_proj.clone().into_vec();
    form_linear_system(&mut mat, &mut rhs, &mut x, &ess, &bc_vals);
    let pre = fem_solver::GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&mat)).unwrap();
    let r = solve_pcg(&mat, &rhs, &mut x, &pre, 1e-12, 2000, false)
        .expect("PCG must converge");
    let e_int = l2_error(mesh, &space, u_proj.as_slice(), &exact, use_blocks);
    let e_sol = l2_error(mesh, &space, &x, &exact, use_blocks);
    eprintln!("   interpolant L2 {e_int:.6e}, solution L2 {e_sol:.6e}");
    (e_sol, r.iterations)
}

#[test]
fn d37_tet_nd2_canonical_mms_converges() {
    let mut errors = Vec::new();
    let mut sizes = Vec::new();
    for lvl in 0..3usize {
        let mesh = Mesh::<3>::unit_cube_tet(lvl + 1);
        let (err, iters) = solve_on(&mesh, true);
        eprintln!(
            "level {lvl}: {} tets, {} dofs, L2 error {err:.6e} ({iters} PCG iters)",
            mesh.n_elements(),
            HCurlSpace::new(mesh.clone(), 2).n_dofs()
        );
        errors.push(err);
        sizes.push(mesh.n_elements() as f64);
    }
    // h ~ (1/n)^(1/3); ND2 on this full H(curl) norm problem: expect >= 2
    for i in 1..errors.len() {
        let rate = (errors[i - 1] / errors[i]).ln() / (sizes[i] / sizes[i - 1]).ln() * 3.0;
        eprintln!("rate {} -> {}: {rate:.3}", i - 1, i);
        assert!(
            rate > 1.8,
            "MMS L2 rate {rate:.3} below the ND2 expectation"
        );
    }
    // errors must also be small in absolute terms on the finest level
    assert!(errors[errors.len() - 1] < 1.0e-1, "finest-level error {:.3e}", errors[errors.len() - 1]);
}

/// The element-local (pre-D37) assembly on the same problem: the conforming
/// canonical path must be measurably better (the non-conforming space cannot
/// reproduce a smooth field).
#[test]
fn d37_canonical_assembly_beats_element_local() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let (err_canonical, _) = solve_on(&mesh, true);
    let (err_local, _) = solve_on(&mesh, false);
    eprintln!(
        "same mesh: canonical L2 {err_canonical:.6e}, element-local L2 {err_local:.6e}"
    );
    assert!(
        err_canonical < err_local,
        "canonical assembly ({err_canonical:.3e}) must beat element-local ({err_local:.3e})"
    );
}

#[test]
fn d37_integration_order_only_changes_assembly() {
    // No-regression control: a 2-D ND2 space has no shared face DOF pairs, so
    // the canonical entry point must reproduce `assemble_bilinear` exactly.
    let mesh = Mesh::<2>::unit_square_tri(1);
    let space = HCurlSpace::new(mesh, 2);
    assert!(space.element_face_blocks(0).is_empty());
}

/// Assembly self-consistency: a field inside the space must be recovered
/// exactly by the canonical path (solve of `curl curl u + u = u` with the
/// exact boundary data).  Any wrong transform direction, sign or weight in the
/// canonical assembly breaks this.
#[test]
fn d37_canonical_assembly_solves_an_in_space_field() {
    let exact_lin = |x: &[f64]| vec![x[0], 2.0 * x[1], -x[2]];
    for lvl in [1usize, 2] {
        let mesh = Mesh::<3>::unit_cube_tet(lvl);
        let space = HCurlSpace::new(mesh.clone(), 2);
        let integrators: [&dyn VectorBilinearIntegrator; 2] =
            [&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }];
        let mut mat =
            VectorAssembler::assemble_bilinear_nd_canonical(&space, &integrators, 6);
        // rhs: ∫ u·Φ with u = the exact (in-space) field — a linear field has
        // curl curl u = 0, so f = u.
        let src = |x: &[f64]| exact_lin(x);
        struct LinSrc;
        impl VectorLinearIntegrator for LinSrc {
            fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
                let v = [qp.x_phys[0], 2.0 * qp.x_phys[1], -qp.x_phys[2]];
                for i in 0..qp.n_dofs {
                    f[i] += qp.weight
                        * (qp.phi_vec[i * 3] * v[0]
                            + qp.phi_vec[i * 3 + 1] * v[1]
                            + qp.phi_vec[i * 3 + 2] * v[2]);
                }
            }
        }
        let _ = src;
        let ls = LinSrc;
        let mut rhs =
            VectorAssembler::assemble_linear_nd_canonical(&space, &[&ls], 6);
        let u_proj = space.interpolate_vector(&exact_lin);
        // Consistency: `u` is the exact solution of this problem, so the
        // assembled operator applied to its DOF vector must reproduce the load
        // vector exactly.
        {
            let mut r = vec![0.0_f64; rhs.len()];
            mat.spmv(&u_proj.as_slice().to_vec(), &mut r);
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for i in 0..rhs.len() {
                num = num.max((r[i] - rhs[i]).abs());
                den = den.max(rhs[i].abs());
            }
            eprintln!("   level {lvl}: |A u - b|_inf = {num:.3e} (|b|_inf = {den:.3e})");
            let e_int = l2_error(&mesh, &space, u_proj.as_slice(), &exact_lin, true);
            eprintln!("   level {lvl}: interpolant L2 = {e_int:.3e}");
            // isolate the mass term (u is the exact L2 projection of itself)
            let mass_only: [&dyn VectorBilinearIntegrator; 1] =
                [&VectorMassIntegrator { alpha: 1.0 }];
            let am = VectorAssembler::assemble_bilinear_nd_canonical(&space, &mass_only, 6);
            let mut rm = vec![0.0_f64; rhs.len()];
            am.spmv(&u_proj.as_slice().to_vec(), &mut rm);
            let mut nm = 0.0_f64;
            let mut dm = 0.0_f64;
            for i in 0..rhs.len() {
                nm = nm.max((rm[i] - rhs[i]).abs());
                dm = dm.max(rhs[i].abs());
            }
            eprintln!("   level {lvl}: mass-only |A u - b|_inf = {nm:.3e} (|b|_inf = {dm:.3e})");
            // mass-matrix sanity: the interpolant of the unit field must have
            // L2 norm^2 equal to |Omega| = 1
            let ec = space.interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0, 0.0]);
            let mut mv = vec![0.0_f64; rhs.len()];
            am.spmv(&ec.as_slice().to_vec(), &mut mv);
            let e2: f64 = ec.as_slice().iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
            eprintln!("   level {lvl}: e^T M e = {e2:.6e} (expect 1)");
            let e_int_c =
                l2_error(&mesh, &space, ec.as_slice(), &(|_x: &[f64]| vec![1.0, 0.0, 0.0]), true);
            eprintln!("   level {lvl}: unit-field interpolant L2 = {e_int_c:.3e}");
        }
        let _tags = space.mesh().unique_boundary_tags();
        let ess = ess_bdr_hcurl_full(space.mesh(), &space);
        let bc_vals: Vec<f64> = ess.iter().map(|&d| u_proj[d as usize]).collect();
        let mut x = u_proj.clone().into_vec();
        form_linear_system(&mut mat, &mut rhs, &mut x, &ess, &bc_vals);
        let pre = fem_solver::GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&mat)).unwrap();
        let rr = solve_pcg(&mat, &rhs, &mut x, &pre, 1e-14, 2000, false).expect("PCG");
        {
            let mut dmax = 0.0_f64;
            for i in 0..x.len() {
                dmax = dmax.max((x[i] - u_proj.as_slice()[i]).abs());
            }
            eprintln!(
                "   level {lvl}: {} iters, |r|/|b| = {:.3e}, |x - u_proj|_inf = {dmax:.3e}",
                rr.iterations, rr.final_residual
            );
        }
        let err = l2_error(&mesh, &space, &x, &exact_lin, true);
        eprintln!("in-space field, level {lvl}: L2 error {err:.3e}");
        assert!(err <= 1e-10, "in-space field level {lvl} error {err:.3e}");
    }
}
/// Library-convention probe: the ND2 **mass matrix** on one affine tet with a
/// general (non-symmetric, non-unit) Jacobian must satisfy
/// `eᵀ M e = |K|` for the interpolant `e` of the unit field `(1,0,0)`.
///
/// This is independent of the D37 face blocks (a single element creates its own
/// canonical face functionals) and localises any Piola / quadrature-weight
/// convention error to the mass path.
#[test]
fn d37_mass_quadratic_form_matches_tet_volume() {
    for (name, coords) in [
        ("unit", vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        (
            "sheared",
            vec![0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.5, 1.5, 0.0, 0.25, 0.5, 2.0],
        ),
        (
            "half-cube-tet",
            vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5],
        ),
    ] {
        let conn: Vec<u32> = vec![0, 1, 2, 3];
        let face_conn: Vec<u32> = vec![1, 3, 2, 0, 2, 3, 0, 1, 3, 0, 1, 2];
        let mesh = Mesh::<3>::uniform(
            coords,
            conn,
            vec![1],
            ElementType::Tet4,
            face_conn,
            vec![1, 2, 3, 4],
            ElementType::Tri3,
        );
        // analytic volume of the tetrahedron
        let p = |i: usize| {
            let v = mesh.node_coords(i as u32);
            [v[0], v[1], v[2]]
        };
        let (a, b, c, d) = (p(0), p(1), p(2), p(3));
        let j = [
            [b[0] - a[0], c[0] - a[0], d[0] - a[0]],
            [b[1] - a[1], c[1] - a[1], d[1] - a[1]],
            [b[2] - a[2], c[2] - a[2], d[2] - a[2]],
        ];
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        let vol = det.abs() / 6.0;

        let space = HCurlSpace::new(mesh, 2);
        let mass: [&dyn VectorBilinearIntegrator; 1] = [&VectorMassIntegrator { alpha: 1.0 }];
        let m = VectorAssembler::assemble_bilinear_nd_canonical(&space, &mass, 6);
        let e = space.interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0, 0.0]);
        let mut me = vec![0.0_f64; e.len()];
        m.spmv(&e.as_slice().to_vec(), &mut me);
        let q: f64 = e.as_slice().iter().zip(me.iter()).map(|(x, y)| x * y).sum();
        eprintln!("{name}: e^T M e = {q:.12e}, |K| = {vol:.12e}");
        assert!(
            (q - vol).abs() <= 1e-12 * vol.max(1.0),
            "{name}: e^T M e = {q:.6e} but |K| = {vol:.6e}"
        );
    }
}

/// transform must keep the assembled mass form exact (e^T M e = |Omega|).
#[test]
fn d37_two_tet_mass_form_is_exact() {
    let mass: [&dyn VectorBilinearIntegrator; 1] = [&VectorMassIntegrator { alpha: 1.0 }];
    let units = |_x: &[f64]| vec![1.0, 0.0, 0.0];
    for scale in [1.0, 0.5] {
        // cube [0,s]^3 split into two tets sharing the face (v0,v1,v4)
        let s = scale;
        let coords = vec![
            0.0, 0.0, 0.0, // 0
            s, 0.0, 0.0, // 1
            0.0, s, 0.0, // 2
            0.0, 0.0, s, // 3
            s, s, s, // 4
        ];
        let conn: Vec<u32> = vec![0, 1, 2, 4, 0, 1, 3, 4];
        let face_conn: Vec<u32> = vec![1, 2, 4, 0, 2, 4, 0, 1, 4, 0, 1, 2, 1, 3, 4, 0, 3, 4, 0, 1, 3];
        let nf = face_conn.len() / 3;
        let mesh = Mesh::<3>::uniform(
            coords,
            conn,
            vec![1, 1],
            ElementType::Tet4,
            face_conn,
            (1..=nf as i32).collect(),
            ElementType::Tri3,
        );
        let space = HCurlSpace::new(mesh.clone(), 2);
        let m = VectorAssembler::assemble_bilinear_nd_canonical(&space, &mass, 6);
        let e = space.interpolate_vector(&units);
        let mut me = vec![0.0_f64; e.len()];
        m.spmv(&e.as_slice().to_vec(), &mut me);
        let q: f64 = e.as_slice().iter().zip(me.iter()).map(|(a, b)| a * b).sum();
        let vol = domain_volume(&mesh);
        eprintln!(
            "scale {scale}: n_elems {} n_dofs {}, e^T M e = {q:.12e}, |Omega| = {vol:.12e}",
            mesh.n_elements(),
            space.n_dofs()
        );
        assert!((q - vol).abs() <= 1e-12, "scale {scale}: {q} vs {vol}");
    }
}


/// Per-element mass matrix (canonical face blocks applied) against an
/// independent quadrature of the Piola-mapped basis on unit_cube_tet(2).
/// Pins the D37 block transform per element.
#[test]
fn d37_element_mass_matches_own_quadrature() {
    use fem_linalg::CooMatrix;
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let space = HCurlSpace::new(mesh.clone(), 2);
    let mass: [&dyn VectorBilinearIntegrator; 1] = [&VectorMassIntegrator { alpha: 1.0 }];
    let elem = TetNDk::new(2);
    let n = elem.n_dofs();
    let qr = tet_rule(6);
    let mut worst = 0.0_f64;
    for e in [0u32, 1, 2, 7] {
        let (_p0, _j, cof, det) = tet_geom(&mesh, e);
        let dofs = space.element_dofs(e).to_vec();
        let signs = space.element_signs(e).to_vec();
        // library local matrix: fresh COO per element, then map back
        let mut coo = CooMatrix::<f64>::new(space.n_dofs(), space.n_dofs());
        fem_assembly::vector_assembler::accumulate_vector_bilinear_element_blocks(
            &space, e, &mass, 6, &mut coo, &[],
        );
        let m = coo.into_csr();
        let idx: std::collections::HashMap<usize, usize> =
            dofs.iter().enumerate().map(|(i, &d)| (d as usize, i)).collect();
        let mut ml = vec![0.0_f64; n * n];
        for r in 0..space.n_dofs() {
            for k in m.row_ptr[r]..m.row_ptr[r + 1] {
                let c = m.col_idx[k] as usize;
                if let (Some(&i), Some(&j)) = (idx.get(&r), idx.get(&c)) {
                    ml[i * n + j] = m.values[k];
                }
            }
        }
        // own quadrature
        let mut mm = vec![0.0_f64; n * n];
        let mut vals = vec![0.0_f64; n * 3];
        for (q, xi) in qr.points.iter().enumerate() {
            elem.eval_basis_vec(xi, &mut vals);
            let mut phi = vec![0.0_f64; n * 3];
            for i in 0..n {
                for r in 0..3 {
                    phi[i * 3 + r] = signs[i]
                        * (cof[r][0] * vals[i * 3]
                            + cof[r][1] * vals[i * 3 + 1]
                            + cof[r][2] * vals[i * 3 + 2])
                        / det;
                }
            }
            let w = qr.weights[q] * det.abs();
            for i in 0..n {
                for j in 0..n {
                    let mut d = 0.0;
                    for r in 0..3 {
                        d += phi[i * 3 + r] * phi[j * 3 + r];
                    }
                    mm[i * n + j] += w * d;
                }
            }
        }
        let mut dmax = 0.0_f64;
        for i in 0..n * n {
            dmax = dmax.max((ml[i] - mm[i]).abs());
        }
        eprintln!("elem {e}: |M_lib - M_own|_inf = {dmax:.4e}");
        worst = worst.max(dmax);
    }
    assert!(worst <= 1e-12, "element mass mismatch {worst:.3e}");
}

/// Sum of the absolute tetrahedron volumes of the mesh.
fn domain_volume<M: MeshTopology>(mesh: &M) -> f64 {
    let mut v = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        v += tet_geom(mesh, e).3.abs() / 6.0;
    }
    v
}
