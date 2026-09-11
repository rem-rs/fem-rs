//! Regression tests: 3-D ultraweak DPG Maxwell assembly with ND (vector)
//! trace spaces.
//!
//! With a polynomial-exact tuple (E, H, Ê, Ĥ) the assembled normal system
//! `A x = b` (`A = Bᵀ G⁻¹ B`) must be satisfied to machine precision: the
//! element-wise residuals of the first-order system
//! `iωμ H + ∇×E = 0`, `−iωε E + ∇×H − J = 0` and the trace identities
//! `Ê = E_tan`, `Ĥ = H_tan` then hold as *functions*, so `B x − f = 0`
//! independent of the test Gram `G`.  This pins the ND-trace face bases,
//! canonical-face parametrisation, covariant maps, the per-element outward
//! sign, the MFEM edge-orientation (sign-encoded) dof sharing, and every
//! graph-norm cross block at once.
//!
//! C++ counterpart: MFEM 4.9 `ComplexDPGWeakForm` probe (harness `resid.cpp`,
//! `~/work/mx3` in WSL) reports residual ~5e-15 for the same tuples with the
//! `+E_tan` convention (`dof = E(x_k)·(J tk_k)`, MFEM
//! `VectorFiniteElement::Project_ND` + `ProjectBdrCoefficientTangent`).

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::{
    eval_vol_space, face_jacobian_3d, face_point_3d, local_face_table, nd_face_dof_nodes,
    nd_face_dof_tangents, scalar_ref_elem, vol_quadrature, VolKind, VolVals,
};
use fem_assembly::dpg::dpg_integrators::{
    DpgCurl3dPairingIntegrator, DpgCurlCurlIntegrator, DpgMixedVectorCurlIntegrator,
    DpgMixedVectorWeakCurlIntegrator, DpgTangentTraceIntegrator3D, DpgTVectorFEMassIntegrator,
    DpgVectorFEDomainLFIntegrator, DpgVectorFEMassIntegrator, VolCtx, DpgBilinear2,
};
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};

const OMEGA: f64 = 1.7;
const MU: f64 = 1.0;
const EPS: f64 = 1.0;

/// Exact E (real-valued): case 0 → `e_dir` (constant); case 1 → (y, 0, 0).
fn e_exact(case_no: u8, dir: u8, x: &[f64]) -> [f64; 3] {
    match case_no {
        0 => {
            let mut v = [0.0; 3];
            v[dir as usize] = 1.0;
            v
        }
        _ => [x[1], 0.0, 0.0],
    }
}

/// Exact H = ∇×E / (i ω μ), real part (zero in both cases).
fn h_exact_re(_case_no: u8, _dir: u8, _x: &[f64]) -> [f64; 3] {
    [0.0; 3]
}

/// Exact H imaginary part: case 1 → (0, 0, −1/ω) (H_z = 1/(iω)).
fn h_exact_im(case_no: u8, _dir: u8, _x: &[f64]) -> [f64; 3] {
    match case_no {
        0 => [0.0; 3],
        _ => [0.0, 0.0, -1.0 / OMEGA / MU],
    }
}

/// Exact J = −iωεE + ∇×H (real, imag parts).
fn j_exact(case_no: u8, dir: u8, x: &[f64]) -> ([f64; 3], [f64; 3]) {
    match case_no {
        0 => {
            let mut ji = [0.0; 3];
            ji[dir as usize] = -OMEGA * EPS;
            ([0.0; 3], ji)
        }
        _ => ([0.0; 3], [-OMEGA * EPS * x[1], 0.0, 0.0]),
    }
}

/// Physical coordinate of element-local reference point `xi` on element `e`.
fn elem_point(mesh: &Mesh<3>, e: u32, xi: &[f64]) -> Vec<f64> {
    let et = mesh.element_type(0);
    let is_simplex = matches!(et, ElementType::Tet4);
    if is_simplex {
        let nodes = mesh.element_nodes(e);
        let t = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
        t.map_to_physical(xi)
    } else {
        let geo = fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, e).unwrap();
        let gnodes = mesh.geometry_nodes(e).to_vec();
        let (_jac, _det, xp) = fem_assembly::vector_assembler::isoparametric_jacobian(
            mesh, &gnodes, geo.as_ref(), xi, 3,
        );
        xp
    }
}

/// Integrator subsets for localizing assembly issues.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Everything (full ultraweak form).
    All,
    /// F rows only: (E,∇×F) + iωμ(H,F) + <n×Ê,F>.
    FRows,
    /// G rows only: −iωε(E,G) + (H,∇×G) + <n×Ĥ,G> − (J,G).
    GRows,
}

fn build_form(mesh: &Mesh<3>, p: u8, mode: Mode) -> (ComplexDPGWeakForm<Mesh<3>>, [usize; 4]) {
    let test_order = p + 1;
    let mut a: ComplexDPGWeakForm<Mesh<3>> = ComplexDPGWeakForm::new(mesh.clone());
    a.set_quad_order(2 * test_order);
    let es = a.add_trial_vector_space(p - 1, 3);
    let hs = a.add_trial_vector_space(p - 1, 3);
    let hate = a.add_trial_trace_space_nd(p);
    let hath = a.add_trial_trace_space_nd(p);
    let f = a.add_test_space(VolKind::HCurl, test_order);
    let g = a.add_test_space(VolKind::HCurl, test_order);

    if mode == Mode::All || mode == Mode::FRows {
        a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, es, f);
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: MU * OMEGA })),
            hs,
            f,
        );
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hate, f);
    }
    if mode == Mode::All || mode == Mode::GRows {
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: -EPS * OMEGA })),
            es,
            g,
        );
        a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, hs, g);
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);
    }

    // Test norms (mass blocks keep G HPD in every mode; the exact-solution
    // identity is norm-independent, so the graph-norm cross blocks are not
    // needed in the restricted modes).
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
    if mode == Mode::All {
        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, f, f);
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: MU * MU * OMEGA * OMEGA })),
            None,
            f,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: -MU * OMEGA })),
            g,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorCurlIntegrator { q: -EPS * OMEGA })),
            g,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorCurlIntegrator { q: EPS * OMEGA })),
            f,
            g,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: MU * OMEGA })),
            f,
            g,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: EPS * EPS * OMEGA * OMEGA })),
            None,
            g,
            g,
        );
    }
    (a, [es, hs, hate, hath])
}

/// Fill the exact tuple `(E, H, Ê, Ĥ)` into `(x_r, x_i)`.
fn set_exact(
    a: &ComplexDPGWeakForm<Mesh<3>>,
    blocks: &[usize; 4],
    l2_order: u8,
    case_no: u8,
    dir: u8,
) -> (Vec<f64>, Vec<f64>) {
    let mesh = a.mesh();
    let et = mesh.element_type(0);
    let (es, hs, hate, hath) = (blocks[0], blocks[1], blocks[2], blocks[3]);
    let n_e = scalar_ref_elem(et, l2_order).n_dofs();
    let tr = a.nd_trace(hate);
    let trh = a.nd_trace(hath);
    let offs = a.trial_offsets();
    let mut xr = vec![0.0_f64; a.size()];
    let mut xi = vec![0.0_f64; a.size()];

    // Volume blocks: interpolate E / H at the L2 dof nodes.
    let dof_par = scalar_ref_elem(et, l2_order).dof_coords();
    for e in 0..mesh.n_elements() as u32 {
        for (i, xii) in dof_par.iter().enumerate() {
            let xp = elem_point(mesh, e, xii);
            let ev = e_exact(case_no, dir, &xp);
            let hr = h_exact_re(case_no, dir, &xp);
            let hi = h_exact_im(case_no, dir, &xp);
            for c in 0..3 {
                xr[offs[es] + e as usize * 3 * n_e + c * n_e + i] = ev[c];
                xr[offs[hs] + e as usize * 3 * n_e + c * n_e + i] = hr[c];
                xi[offs[hs] + e as usize * 3 * n_e + c * n_e + i] = hi[c];
            }
        }
    }

    // Trace blocks: MFEM Project_ND moments `v(x_k) · (J tk_k)` with the
    // sign-encoded face dof orientation.
    for (blk, tr_sp, field) in [(hate, &tr, 0_u8), (hath, &trh, 1_u8)] {
        let base = offs[blk];
        let p_us = tr_sp.order() as usize;
        for face in 0..tr_sp.n_faces() {
            let is_quad = tr_sp.is_quad_face(face);
            let nodes = nd_face_dof_nodes(p_us, is_quad);
            let tks = nd_face_dof_tangents(p_us, is_quad);
            let signed = tr_sp.face_signed_dofs(face).to_vec();
            for (j, &dof) in tr_sp.face_dof_list(face).iter().enumerate() {
                let param = &nodes[j];
                let xk = face_point_3d(tr_sp, face, param);
                let jac = face_jacobian_3d(tr_sp, face, param);
                let tk = tks[j];
                let jt: Vec<f64> = (0..3)
                    .map(|d| jac[0][d] * tk[0] + jac[1][d] * tk[1])
                    .collect();
                let (vr, vi) = if field == 0 {
                    let ev = e_exact(case_no, dir, &xk);
                    (ev[0] * jt[0] + ev[1] * jt[1] + ev[2] * jt[2], 0.0)
                } else {
                    let hr = h_exact_re(case_no, dir, &xk);
                    let hi = h_exact_im(case_no, dir, &xk);
                    (
                        hr[0] * jt[0] + hr[1] * jt[1] + hr[2] * jt[2],
                        hi[0] * jt[0] + hi[1] * jt[1] + hi[2] * jt[2],
                    )
                };
                let s = if signed[j] < 0 { -1.0 } else { 1.0 };
                xr[base + dof] = s * vr;
                xi[base + dof] = s * vi;
            }
        }
    }
    (xr, xi)
}

/// Assemble the (possibly restricted) ultraweak Maxwell form and return
/// `max |A x_exact − b|` together with the argmax dof.
fn residual_of(
    mesh: &Mesh<3>,
    p: u8,
    case_no: u8,
    dir: u8,
    mode: Mode,
    with_rhs: bool,
) -> (f64, usize) {
    let (mut a, blocks) = build_form(mesh, p, mode);
    let g = 1; // G test block index (F = 0, G = 1)
    if with_rhs {
        let case_r = case_no;
        let dir_r = dir;
        let case_i = case_no;
        let dir_i = dir;
        a.add_domain_lf_integrator(
            Some(Box::new(DpgVectorFEDomainLFIntegrator {
                f: move |x: &[f64], out: &mut [f64]| {
                    let (jr, _) = j_exact(case_r, dir_r, x);
                    out.copy_from_slice(&jr);
                },
            })),
            Some(Box::new(DpgVectorFEDomainLFIntegrator {
                f: move |x: &[f64], out: &mut [f64]| {
                    let (_, ji) = j_exact(case_i, dir_i, x);
                    out.copy_from_slice(&ji);
                },
            })),
            g,
        );
    }
    a.assemble();

    let (xr, xi) = set_exact(&a, &blocks, p - 1, case_no, dir);
    let mat_r = a.block_mat_r();
    let mat_i = a.block_mat_i();
    let n = mat_r.nrows;
    let mut t1 = vec![0.0_f64; n];
    let mut t2 = vec![0.0_f64; n];
    let mut axr = vec![0.0_f64; n];
    let mut axi = vec![0.0_f64; n];
    mat_r.spmv(&xr, &mut t1);
    axr.copy_from_slice(&t1);
    mat_i.spmv(&xi, &mut t2);
    for i in 0..n {
        axr[i] -= t2[i];
    }
    mat_i.spmv(&xr, &mut t1);
    axi.copy_from_slice(&t1);
    mat_r.spmv(&xi, &mut t2);
    for i in 0..n {
        axi[i] += t2[i];
    }
    let br = a.rhs_r();
    let bi = a.rhs_i();
    let mut worst = (0.0_f64, 0usize);
    for i in 0..n {
        let d = (axr[i] - br[i]).abs().max((axi[i] - bi[i]).abs());
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

fn run_identity(mesh: &Mesh<3>, p: u8, case_no: u8, dir: u8, tol: f64) {
    let (full, fi) = residual_of(mesh, p, case_no, dir, Mode::All, true);
    let (frows, _i) = residual_of(mesh, p, case_no, dir, Mode::FRows, false);
    let (grows, _i2) = residual_of(mesh, p, case_no, dir, Mode::GRows, true);
    eprintln!(
        "identity p={p} case={case_no} dir={dir}: full {full:.3e} (dof {fi}) | F-rows {frows:.3e} | G-rows {grows:.3e}"
    );
    // sigma*m consistency across all (face, position) pairs per global dof
    if std::env::var("DPG_SIGMA_CHECK").is_ok() {
        let a = build_form(mesh, p, Mode::FRows).0;
        let tr = a.nd_trace(2);
        let mut seen: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        let mut bad = 0usize;
        for face in 0..tr.n_faces() {
            let is_quad = tr.is_quad_face(face);
            let nodes = nd_face_dof_nodes(p as usize, is_quad);
            let tks = nd_face_dof_tangents(p as usize, is_quad);
            let signed = tr.face_signed_dofs(face).to_vec();
            for (j, &dof) in tr.face_dof_list(face).iter().enumerate() {
                let xk = face_point_3d(&tr, face, &nodes[j]);
                let jac = face_jacobian_3d(&tr, face, &nodes[j]);
                let tk = tks[j];
                let jt: Vec<f64> = (0..3)
                    .map(|d| jac[0][d] * tk[0] + jac[1][d] * tk[1])
                    .collect();
                let ev = e_exact(case_no, dir, &xk);
                let m = ev[0] * jt[0] + ev[1] * jt[1] + ev[2] * jt[2];
                let s = if signed[j] < 0 { -1.0 } else { 1.0 };
                let v = s * m;
                match seen.get(&dof) {
                    Some(prev) => {
                        if ((*prev) - v).abs() > 1e-12 {
                            bad += 1;
                            eprintln!(
                                "  sigma*m INCONSISTENT dof {dof} face {face}: {v:+.6} vs prev {prev:+.6}"
                            );
                        }
                    }
                    None => {
                        seen.insert(dof, v);
                    }
                }
            }
        }
        eprintln!("  sigma*m consistency: {bad} inconsistencies over {} dofs", seen.len());
    }
    assert!(
        full < tol,
        "exact identity fails: p={p} case {case_no}: max |Ax − b| = {full:.3e} (tol {tol:.1e})"
    );
}

#[test]
fn maxwell_3d_identity_hex_p1_const_e() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    run_identity(&mesh, 1, 0, 0, 1e-10);
}

/// Multi-element variant: exercises the interior-face trace dof sharing and
/// the Elem2 (reversed canonical-cycle) face of every interior quad.
/// Regression for the round-14 fix: the element-side evaluation point of a
/// trace face is now the MFEM Loc1/Loc2 vertex-matched interpolation of the
/// canonical face parameter (exact reference face-plane coordinates), not a
/// Newton-refined mirrored seed.
#[test]
fn maxwell_3d_identity_hex2_p1_const_e() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    for dir in 0..3u8 {
        run_identity(&mesh, 1, 0, dir, 1e-10);
    }
}

/// `nx × ny × nz` axis-aligned hex mesh of the unit cube (MFEM vertex order
/// per hex, lexicographic element order) — used to isolate a single interior
/// face orientation.
fn unit_box_hex(nx: usize, ny: usize, nz: usize) -> Mesh<3> {
    let (np_i, np_j, np_k) = (nx + 1, ny + 1, nz + 1);
    let mut coords = Vec::new();
    for k in 0..np_k {
        for j in 0..np_j {
            for i in 0..np_i {
                coords.push(i as f64 / nx as f64);
                coords.push(j as f64 / ny as f64);
                coords.push(k as f64 / nz as f64);
            }
        }
    }
    let nid = |i: usize, j: usize, k: usize| -> u32 {
        (k * np_i * np_j + j * np_i + i) as u32
    };
    let mut conn = Vec::new();
    let mut elem_tags = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                conn.extend_from_slice(&[
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                ]);
                elem_tags.push(1i32);
            }
        }
    }
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut add_quad = |a: u32, b: u32, c: u32, d: u32, t: i32| {
        face_conn.extend_from_slice(&[a, b, c, d]);
        face_tags.push(t);
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if k == 0 {
                    add_quad(nid(i, j, 0), nid(i, j + 1, 0), nid(i + 1, j + 1, 0), nid(i + 1, j, 0), 1);
                }
                if k == nz - 1 {
                    add_quad(nid(i, j, nz), nid(i + 1, j, nz), nid(i + 1, j + 1, nz), nid(i, j + 1, nz), 2);
                }
                if j == 0 {
                    add_quad(nid(i, 0, k), nid(i + 1, 0, k), nid(i + 1, 0, k + 1), nid(i, 0, k + 1), 3);
                }
                if j == ny - 1 {
                    add_quad(nid(i, ny, k), nid(i, ny, k + 1), nid(i + 1, ny, k + 1), nid(i + 1, ny, k), 4);
                }
                if i == 0 {
                    add_quad(nid(0, j, k), nid(0, j, k + 1), nid(0, j + 1, k + 1), nid(0, j + 1, k), 5);
                }
                if i == nx - 1 {
                    add_quad(nid(nx, j, k), nid(nx, j + 1, k), nid(nx, j + 1, k + 1), nid(nx, j, k + 1), 6);
                }
            }
        }
    }
    Mesh::uniform(
        coords, conn, elem_tags, ElementType::Hex8, face_conn, face_tags, ElementType::Quad4,
    )
}

/// 2×1×1: exactly one interior face (x-normal, Elem2 cycle = (B,A,D,C)
/// reflection of the canonical cycle).
#[test]
fn maxwell_3d_identity_hex210_p1_const_e() {
    let mesh = unit_box_hex(2, 1, 1);
    for dir in 0..3u8 {
        run_identity(&mesh, 1, 0, dir, 1e-10);
    }
}

/// 1×2×1: one y-normal interior face.
#[test]
fn maxwell_3d_identity_hex020_p1_const_e() {
    let mesh = unit_box_hex(1, 2, 1);
    for dir in 0..3u8 {
        run_identity(&mesh, 1, 0, dir, 1e-10);
    }
}

/// 1×1×2: one z-normal interior face.
#[test]
fn maxwell_3d_identity_hex002_p1_const_e() {
    let mesh = unit_box_hex(1, 1, 2);
    for dir in 0..3u8 {
        run_identity(&mesh, 1, 0, dir, 1e-10);
    }
}

#[test]
fn maxwell_3d_identity_tet2_p1_const_e() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    run_identity(&mesh, 1, 0, 0, 1e-10);
}

#[test]
fn maxwell_3d_identity_hex_p2_linear_e() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    run_identity(&mesh, 2, 1, 0, 1e-10);
}

#[test]
fn maxwell_3d_identity_tet_p1_const_e() {
    let mesh = Mesh::<3>::unit_cube_tet(1);
    run_identity(&mesh, 1, 0, 0, 1e-10);
}

#[test]
fn maxwell_3d_identity_tet_p2_linear_e() {
    let mesh = Mesh::<3>::unit_cube_tet(1);
    run_identity(&mesh, 2, 1, 0, 1e-10);
}

/// FD validation of the 3-D ND element curl (`eval_curl` vs central
/// differences of `eval_basis_vec`).
#[test]
fn nd_volume_curl_matches_basis_fd() {
    use fem_element::nedelec::{HexNDk, TetNDk};
    use fem_element::VectorReferenceElement;
    for (fe, name) in [
        (&HexNDk::new(1) as &dyn VectorReferenceElement, "HexNDk(1)"),
        (&HexNDk::new(2) as &dyn VectorReferenceElement, "HexNDk(2)"),
        (&TetNDk::new(1) as &dyn VectorReferenceElement, "TetNDk(1)"),
        (&TetNDk::new(2) as &dyn VectorReferenceElement, "TetNDk(2)"),
    ] {
        let n = fe.n_dofs();
        let mut vp = vec![0.0; n * 3];
        let mut vm = vec![0.0; n * 3];
        let mut c = vec![0.0; n * 3];
        let xi = [0.31, 0.42, 0.53];
        let h = 1e-6;
        fe.eval_curl(&xi, &mut c);
        let mut der: [Vec<f64>; 3] = [vec![0.0; n * 3], vec![0.0; n * 3], vec![0.0; n * 3]];
        for d in 0..3 {
            let mut a = xi;
            a[d] += h;
            let mut b = xi;
            b[d] -= h;
            fe.eval_basis_vec(&a, &mut vp);
            fe.eval_basis_vec(&b, &mut vm);
            for i in 0..n {
                for k in 0..3 {
                    der[d][i * 3 + k] = (vp[i * 3 + k] - vm[i * 3 + k]) / (2.0 * h);
                }
            }
        }
        let mut worst = 0.0_f64;
        for i in 0..n {
            let (d0, d1, d2) = (&der[0], &der[1], &der[2]);
            let curl_fd = [
                d1[i * 3 + 2] - d2[i * 3 + 1],
                d2[i * 3] - d0[i * 3 + 2],
                d0[i * 3 + 1] - d1[i * 3],
            ];
            for k in 0..3 {
                worst = worst.max((c[i * 3 + k] - curl_fd[k]).abs());
            }
        }
        assert!(
            worst < 1e-5,
            "{name}: eval_curl disagrees with FD of eval_basis_vec by {worst:.3e}"
        );
    }
}

/// Stokes check on one hex: `∫_K ∇×F_i dV = ∫_∂K n×F_i dS` for every basis
/// function of the broken ND test space.
#[test]
fn nd_hex_stokes_identity() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let et = mesh.element_type(0);
    let order = 2u8;
    let n_f = VolKind::HCurl.n_dofs_per_elem(et, order);
    let geo = fem_assembly::vector_assembler::geo_ref_elem_from_mesh(&mesh, 0).unwrap();
    let gnodes = mesh.geometry_nodes(0).to_vec();
    let (qpts, qwts) = vol_quadrature(et, 8);

    let mut vol_int = vec![0.0_f64; 3 * n_f];
    for (q, xi) in qpts.iter().enumerate() {
        let (jac, det, _xp) = fem_assembly::vector_assembler::isoparametric_jacobian(
            &mesh, &gnodes, geo.as_ref(), xi, 3,
        );
        let inv = jac.clone().try_inverse().unwrap();
        let mut jit = nalgebra::DMatrix::<f64>::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                jit[(r, c)] = inv[(c, r)];
            }
        }
        let mut tv = VolVals::default();
        eval_vol_space(VolKind::HCurl, order, et, 3, &jac, det, &jit, xi, None, &mut tv);
        for i in 0..n_f {
            for c in 0..3 {
                vol_int[i * 3 + c] += qwts[q] * det.abs() * tv.curl[i * 3 + c];
            }
        }
    }

    let tr = fem_assembly::dpg::dpg_basis::TraceSpace::new_nd(mesh.clone(), 1);
    let nodes = mesh.element_nodes(0);
    let lfs = local_face_table(&nodes, 3);
    let mut surf_int = vec![0.0_f64; 3 * n_f];
    for (li, lf) in lfs.iter().enumerate() {
        let fid = tr.elem_face_id(0, li);
        let is_qf = tr.is_quad_face(fid);
        let (fpts, fwts) = fem_assembly::dpg::dpg_basis::face_quadrature(3, is_qf, 6);
        for (q, fparam) in fpts.iter().enumerate() {
            let (_xp, normal, _ms) = fem_assembly::dpg_weakform::face_geo_nodes(
                &mesh,
                tr.face_nodes(fid),
                is_qf,
                fparam,
                3,
            );
            let xi0 = fem_assembly::dpg::dpg_basis::face_param_to_elem_ref(et, lf, is_qf, fparam);
            let (jac, det, _x2) = fem_assembly::vector_assembler::isoparametric_jacobian(
                &mesh, &gnodes, geo.as_ref(), &xi0, 3,
            );
            let inv = jac.clone().try_inverse().unwrap();
            let mut jit = nalgebra::DMatrix::<f64>::zeros(3, 3);
            for r in 0..3 {
                for c in 0..3 {
                    jit[(r, c)] = inv[(c, r)];
                }
            }
            let mut tv = VolVals::default();
            eval_vol_space(VolKind::HCurl, order, et, 3, &jac, det, &jit, &xi0, None, &mut tv);
            for i in 0..n_f {
                let (fx, fy, fz) = (tv.phi[i * 3], tv.phi[i * 3 + 1], tv.phi[i * 3 + 2]);
                let cx = normal[1] * fz - normal[2] * fy;
                let cy = normal[2] * fx - normal[0] * fz;
                let cz = normal[0] * fy - normal[1] * fx;
                surf_int[i * 3] += fwts[q] * cx;
                surf_int[i * 3 + 1] += fwts[q] * cy;
                surf_int[i * 3 + 2] += fwts[q] * cz;
            }
        }
    }
    let mut worst = 0.0_f64;
    for i in 0..3 * n_f {
        worst = worst.max((vol_int[i] - surf_int[i]).abs());
    }
    eprintln!("nd_hex_stokes_identity worst |vol - surf| = {worst:.3e}");
    assert!(worst < 1e-11, "Stokes identity fails: {worst:.3e}");
}

/// Direct check of `DpgCurl3dPairingIntegrator`: with E = e_0 constant, the
/// assembled column must equal the Stokes-validated `∫_K (curl F_i)_0 dV`.
#[test]
fn curl_pairing_integrator_direct() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let et = mesh.element_type(0);
    let order = 2u8;
    let n_f = VolKind::HCurl.n_dofs_per_elem(et, order);
    let geo = fem_assembly::vector_assembler::geo_ref_elem_from_mesh(&mesh, 0).unwrap();
    let gnodes = mesh.geometry_nodes(0).to_vec();
    let (qpts, qwts) = vol_quadrature(et, 8);

    let mut be = vec![0.0_f64; n_f * 3];
    let mut ref_cols = vec![0.0_f64; n_f];
    for (q, xi) in qpts.iter().enumerate() {
        let (jac, det, xp) = fem_assembly::vector_assembler::isoparametric_jacobian(
            &mesh, &gnodes, geo.as_ref(), xi, 3,
        );
        let inv = jac.clone().try_inverse().unwrap();
        let mut jit = nalgebra::DMatrix::<f64>::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                jit[(r, c)] = inv[(c, r)];
            }
        }
        let mut tv = VolVals::default();
        eval_vol_space(VolKind::HCurl, order, et, 3, &jac, det, &jit, xi, None, &mut tv);
        let ev = VolVals {
            phi: vec![1.0, 0.0, 0.0],
            grad: vec![0.0; 3],
            div: vec![0.0],
            curl: vec![0.0; 3],
            n_expanded: 3,
            n_scalar: 1,
            vdim: 3,
            curl_dim: 3,
        };
        let ctx = VolCtx { w: qwts[q] * det.abs(), x: xp, dim: 3, elem: 0 };
        let mut m = vec![0.0_f64; n_f * 3];
        DpgCurl3dPairingIntegrator { q: 1.0 }.assemble2(&ctx, &ev, &tv, &mut m);
        for k in 0..n_f * 3 {
            be[k] += m[k];
        }
        for i in 0..n_f {
            ref_cols[i] += qwts[q] * det.abs() * tv.curl[i * 3];
        }
    }
    let mut worst = 0.0_f64;
    for i in 0..n_f {
        worst = worst.max((be[i * 3] - ref_cols[i]).abs());
        worst = worst.max(be[i * 3 + 1].abs());
        worst = worst.max(be[i * 3 + 2].abs());
    }
    assert!(worst < 1e-12, "curl pairing direct check fails: {worst:.3e}");
}



/// Per-element row-sum sanity for hex2 (finite after assembly; the exact
/// identity itself is pinned by `maxwell_3d_identity_hex2_p1_const_e`).
#[test]
fn hex2_residual_documentation() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let p = 1u8;
    let (mut a, _blocks) = build_form(&mesh, p, Mode::FRows);
    a.store_matrices(true);
    a.assemble();
    let n_e = scalar_ref_elem(mesh.element_type(0), 0).n_dofs();
    let mut per_elem = Vec::new();
    let (mat_r, mat_i) = (a.block_mat_r().clone(), a.block_mat_i().clone());
    let offs = a.trial_offsets();
    let n_e3 = 3 * n_e;
    for e in 0..mesh.n_elements() as u32 {
        let mut worst = 0.0_f64;
        for r in 0..n_e3 {
            let g = offs[0] + e as usize * n_e3 + r;
            let mut ar = 0.0_f64;
            let mut ai = 0.0_f64;
            for p_ in mat_r.row_ptr[g]..mat_r.row_ptr[g + 1] {
                ar += mat_r.values[p_];
            }
            for p_ in mat_i.row_ptr[g]..mat_i.row_ptr[g + 1] {
                ai += mat_i.values[p_];
            }
            worst = worst.max((ar - a.rhs_r()[g]).abs().max((ai - a.rhs_i()[g]).abs()));
        }
        per_elem.push(worst);
    }
    eprintln!("hex2 per-element E-row residual sums: {per_elem:?}");
    assert!(
        per_elem.iter().all(|v| v.is_finite()),
        "residuals must be finite"
    );
}

