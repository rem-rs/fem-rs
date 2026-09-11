//! D55 regression: the hex `NDk` (k >= 2) ex3 discretisation must equal MFEM's
//! exactly, which pins two things that were wrong before the round-18 fix:
//!
//! 1. **Curl-curl quadrature.**  MFEM `CurlCurlIntegrator` assembles Qk
//!    (tensor) elements with `2*el.GetOrder()` and only Pk simplices with
//!    `2*el.GetOrder()-2` (`fem/bilininteg.cpp`,
//!    `CurlCurlIntegrator::AssembleElementMatrix`); `ND_HexahedronElement` is
//!    Qk (its `VectorTensorFiniteElement` base passes `FunctionSpace::Qk`).
//!    fem-rs's `CurlCurlIntegrator::integration_order` reports `2k-2` for
//!    every geometry, so a caller that trusts it under-integrates the hex
//!    block (at k=2: 2 Gauss points per direction where the integrand needs
//!    degree 4).  The single-hex invariants below pin the MFEM rule numbers.
//!
//! 2. **No hex DofTransformation.**  `ND_FECollection::
//!    DofTransformationForGeometry` returns NULL for tensor-product
//!    geometries, and `ND_DofTransformation::TransformPrimal` skips quad
//!    faces: MFEM relates hex quad-face DOFs across elements by a *signed
//!    permutation* only (`ND_FECollection::QuadDofOrd`), which is exactly
//!    `HCurlSpace`'s `match_face_dof` encoding — so `element_face_blocks`
//!    must stay empty for hex and the signed permutation must make the
//!    assembled system match MFEM's.  The eliminated-system invariants below
//!    (permutation-invariant Frobenius norm / trace / RHS norm, plus the
//!    dense exact solution) prove that it does.
//!
//! C++ reference harness (MFEM 4.10, `~/work/probe5`/`probe6`/`probe7`,
//! mesh = 8 unit cubes on `[0,8]x[0,1]x[0,1]` = `data/beam-hex.mesh`,
//! order 2, curl-curl + mass at MFEM's rules, elimination via
//! `BilinearForm::FormLinearSystem`):
//!
//! ```text
//! 1 hex  : RAW  CURL fro^2 142.7733333333335  tr 48; MASS fro^2 0.1734 tr 1.92
//!  beam  : RAW  n 348  A_fro^2 1271.683674074104    tr 399.3599999999997   b_l2 14.23163235937767
//!  beam  : ELIM n 348  A_fro^2 899.4145975308757    tr 399.3599999999997   B_l2 20.52605700884046
//!  beam  : dense-solve L2 = 0.1147046452560189   (PCG: 0.1147046508268888)
//!  beam x2 (refine 1): ELIM n 2120 A_fro^2 35981.50522468546 tr 6205.44 B_l2 23.39886478543981
//!  beam x2 (refine 1): dense-solve L2 = 0.07338872685037881
//! ```

use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::{accumulate_vector_bilinear_element_blocks, nd_element_local_dofs, VectorAssembler};
use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_core::NodeId;
use fem_element::nedelec::HexNDk;
use fem_element::VectorReferenceElement;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_space::constraints::{boundary_dofs_hcurl, form_linear_system};
use fem_space::{FESpace, HCurlSpace};

const KAPPA: f64 = std::f64::consts::PI;

/// MFEM ex3's exact solution E = (sin(k y), sin(k z), sin(k x)).
fn e_exact(x: &[f64]) -> [f64; 3] {
    [
        (KAPPA * x[1]).sin(),
        (KAPPA * x[2]).sin(),
        (KAPPA * x[0]).sin(),
    ]
}

/// MFEM ex3's f = (1 + k^2) E (curl curl E + E with div-free harmonic E).
struct Src;
impl VectorLinearIntegrator for Src {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let c = 1.0 + KAPPA * KAPPA;
        let fx = c * (KAPPA * x[1]).sin();
        let fy = c * (KAPPA * x[2]).sin();
        let fz = c * (KAPPA * x[0]).sin();
        for i in 0..qp.n_dofs {
            f[i] += qp.weight
                * (qp.phi_vec[i * 3] * fx + qp.phi_vec[i * 3 + 1] * fy + qp.phi_vec[i * 3 + 2] * fz);
        }
    }
}

/// A unit-cube hex grid `nx x ny x nz` (no file IO in tests).  `grid(8, 1, 1)`
/// is `data/beam-hex.mesh`.
fn grid(nx: usize, ny: usize, nz: usize) -> Mesh<3> {
    let (a, b, c) = (nx, ny, nz);
    let (npx, npy, npz) = (a + 1, b + 1, c + 1);
    let mut coords = Vec::new();
    for k in 0..npz {
        for j in 0..npy {
            for i in 0..npx {
                coords.push(i as f64);
                coords.push(j as f64);
                coords.push(k as f64);
            }
        }
    }
    let nid = |i: usize, j: usize, k: usize| -> NodeId { (k * npx * npy + j * npx + i) as NodeId };
    let mut conn = Vec::new();
    let mut elem_tags = Vec::new();
    for k in 0..c {
        for j in 0..b {
            for i in 0..a {
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
    let mut face_tags: Vec<i32> = Vec::new();
    // x-normal boundary faces: only the two outer planes.
    for i in [0usize, a] {
        for k in 0..c {
            for j in 0..b {
                face_conn.extend_from_slice(&[
                    nid(i, j, k),
                    nid(i, j + 1, k),
                    nid(i, j + 1, k + 1),
                    nid(i, j, k + 1),
                ]);
                face_tags.push(1);
            }
        }
    }
    for j in 0..=b {
        for k in 0..c {
            for i in 0..a {
                face_conn.extend_from_slice(&[
                    nid(i, j, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j, k),
                ]);
                face_tags.push(1);
            }
        }
    }
    for k in 0..=c {
        for j in 0..b {
            for i in 0..a {
                face_conn.extend_from_slice(&[
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                ]);
                face_tags.push(1);
            }
        }
    }
    Mesh::<3>::uniform(
        coords,
        conn,
        elem_tags,
        ElementType::Hex8,
        face_conn,
        face_tags,
        ElementType::Quad4,
    )
}

/// `(frobenius^2, trace)` of a CSR matrix.
fn fro_tr(mat: &fem_linalg::CsrMatrix<f64>) -> (f64, f64) {
    let mut fro = 0.0_f64;
    for v in mat.values.iter() {
        fro += v * v;
    }
    let mut tr = 0.0_f64;
    for d in 0..mat.nrows {
        for k in mat.row_ptr[d]..mat.row_ptr[d + 1] {
            if mat.col_idx[k] as usize == d {
                tr += mat.values[k];
            }
        }
    }
    (fro, tr)
}

/// One hex, integrators separated: pins the Qk curl-curl rule (`2k`) and the
/// mass rule (exact at `2k+3`).  C++ `probe7 2 1`:
/// `CURL fro^2 142.7733333333335 tr 48`; `MASS fro^2 0.1734 tr 1.92`.
#[test]
fn one_hex_per_integrator_invariants_match_mfem() {
    let space = HCurlSpace::new(grid(1, 1, 1), 2);
    assert_eq!(space.n_dofs(), 54);
    // Assemble curl-curl alone at MFEM's Qk rule `2k` (4 at k = 2).
    let cc = {
        let n = space.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        let integ = CurlCurlIntegrator { mu: 1.0 };
        for e in 0..space.mesh().n_elements() as u32 {
            accumulate_vector_bilinear_element_blocks(
                &space, e, &[&integ], 4, &mut coo, &[],
            );
        }
        coo.into_csr()
    };
    let (fro, tr) = fro_tr(&cc);
    assert!((fro - 142.7733333333335).abs() < 1e-10, "CURL fro^2 {fro}");
    assert!((tr - 48.0).abs() < 1e-10, "CURL tr {tr}");
    // Mass: the `2k+3` rule is exact, so it matches C++'s `2k` (affine) matrix.
    let mm = {
        let n = space.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        let integ = VectorMassIntegrator { alpha: 1.0 };
        for e in 0..space.mesh().n_elements() as u32 {
            accumulate_vector_bilinear_element_blocks(
                &space, e, &[&integ], 7, &mut coo, &[],
            );
        }
        coo.into_csr()
    };
    let (fro, tr) = fro_tr(&mm);
    assert!((fro - 0.1734).abs() < 1e-12, "MASS fro^2 {fro}");
    assert!((tr - 1.92).abs() < 1e-12, "MASS tr {tr}");
}

/// `HexNDk::eval_curl` must be the exact curl of `eval_basis_vec` (guards the
/// assembly against a silently wrong element curl, which would corrupt every
/// hex curl-curl matrix while leaving the mass matrix — and any
/// interpolation-only test — untouched).
#[test]
fn hex_ndk_curl_matches_finite_differences() {
    let r = HexNDk::new(2);
    let n = r.n_dofs();
    let h = 1e-6;
    let (mut p0, mut px, mut py, mut pz) = (
        vec![0.0; n * 3],
        vec![0.0; n * 3],
        vec![0.0; n * 3],
        vec![0.0; n * 3],
    );
    let mut c = vec![0.0; n * 3];
    for xi in [[0.31, -0.22, 0.17], [-0.4, 0.5, -0.6], [0.0, 0.0, 0.0]] {
        r.eval_basis_vec(&xi, &mut p0);
        r.eval_basis_vec(&[xi[0] + h, xi[1], xi[2]], &mut px);
        r.eval_basis_vec(&[xi[0], xi[1] + h, xi[2]], &mut py);
        r.eval_basis_vec(&[xi[0], xi[1], xi[2] + h], &mut pz);
        r.eval_curl(&xi, &mut c);
        let mut worst = 0.0_f64;
        for i in 0..n {
            let dz_y = (pz[i * 3 + 1] - p0[i * 3 + 1]) / h;
            let dy_z = (py[i * 3 + 2] - p0[i * 3 + 2]) / h;
            let dx_z = (px[i * 3 + 2] - p0[i * 3 + 2]) / h;
            let dz_x = (pz[i * 3] - p0[i * 3]) / h;
            let dy_x = (py[i * 3] - p0[i * 3]) / h;
            let dx_y = (px[i * 3 + 1] - p0[i * 3 + 1]) / h;
            let fd = [dy_z - dz_y, dz_x - dx_z, dx_y - dy_x];
            for d in 0..3 {
                worst = worst.max((c[i * 3 + d] - fd[d]).abs());
            }
        }
        assert!(worst < 1e-5, "HexNDk eval_curl wrong at {xi:?}: {worst:.3e}");
    }
}

/// Eliminated beam-hex ND2 system (refine 0 and 1): permutation-invariant
/// invariants must match the MFEM harness to ~1e-12, and the exact (dense)
/// Galerkin solution's L2 error must match to 14 digits.  This proves hex
/// signed-permutation face handling + boundary DOF collection + elimination
/// are MFEM-exact (no hex `DofTransformation` exists in MFEM either).
#[test]
fn beam_hex_eliminated_system_matches_mfem() {
    for (nz, want_n, want_fro, want_tr, want_bl2, want_l2) in [
        (
            1usize,
            348usize,
            899.4145975308757,
            399.3599999999997,
            20.52605700884046,
            1.14704645256019e-1,
        ),
        (
            2usize,
            2120usize,
            35981.50522468546,
            6205.440000000119,
            23.39886478543981,
            7.338872685037881e-2,
        ),
    ] {
        let mesh0 = grid(8, 1, 1);
        let mesh = if nz > 1 {
            fem_mesh::amr::refine_uniform_3d(&mesh0)
        } else {
            mesh0
        };
        let space = HCurlSpace::new(mesh, 2);
        assert_eq!(space.n_dofs(), want_n);

        let tags = space.mesh().unique_boundary_tags();
        let ess_bdr = boundary_dofs_hcurl(space.mesh(), &space, &tags);
        let qo = 4u8; // MFEM Qk curl-curl rule 2k; mass 2k+3 (exact) below.
        let mut rhs = VectorAssembler::assemble_linear(&space, &[&Src], qo);
        let u_proj = space.interpolate_vector(&|x| e_exact(x).to_vec()).into_vec();
        let bc_vals: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d as usize]).collect();

        let mut mat = {
            let n = space.n_dofs();
            let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
            let cc = CurlCurlIntegrator { mu: 1.0 };
            let mass = VectorMassIntegrator { alpha: 1.0 };
            for e in 0..space.mesh().n_elements() as u32 {
                accumulate_vector_bilinear_element_blocks(
                    &space, e, &[&cc], qo, &mut coo, &[],
                );
                accumulate_vector_bilinear_element_blocks(
                    &space, e, &[&mass], 2 * qo + 3, &mut coo, &[],
                );
            }
            coo.into_csr()
        };
        let mut x = u_proj.clone();
        form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bc_vals);

        let (fro, tr) = fro_tr(&mat);
        let bl2: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (fro - want_fro).abs() < 5e-11 * want_fro,
            "nz{nz}: A_fro^2 {fro} vs {want_fro}"
        );
        assert!((tr - want_tr).abs() < 5e-11 * want_tr, "nz{nz}: A_tr {tr}");
        assert!(
            (bl2 - want_bl2).abs() < 5e-12 * want_bl2,
            "nz{nz}: B_l2 {bl2} vs {want_bl2}"
        );

        // Exact (dense LU) Galerkin solution -> L2 error vs the C++ dense solve.
        let n = mat.nrows;
        let mut dm = nalgebra::DMatrix::<f64>::zeros(n, n);
        for row in 0..n {
            for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
                dm[(row, mat.col_idx[k] as usize)] = mat.values[k];
            }
        }
        let lu = nalgebra::LU::new(dm);
        let mut bvec = nalgebra::DVector::<f64>::zeros(n);
        bvec.copy_from_slice(&rhs);
        let xsol = lu.solve(&bvec).unwrap();
        x.copy_from_slice(xsol.as_slice());

        // L2 error with the ex3 evaluator (HexNDk basis + trilinear map).
        let mut e2 = 0.0_f64;
        let k = 2usize;
        let r = HexNDk::new(k);
        let nd = r.n_dofs();
        let q = r.quadrature((2 * k + 3) as u8);
        let mut p = vec![0.0; nd * 3];
        for e in space.mesh().elem_iter() {
            let uloc = nd_element_local_dofs(&space, e, &x);
            for (qi, xi) in q.points.iter().enumerate() {
                r.eval_basis_vec(xi, &mut p);
                let (j, xp) = crate_common_jac(space.mesh(), e, xi);
                let w = q.weights[qi] * j.determinant().abs();
                let jt = j.try_inverse().unwrap_or_default().transpose();
                let mut uh = [0.0; 3];
                for a in 0..nd {
                    for c in 0..3 {
                        let mut v = 0.0;
                        for kk in 0..3 {
                            v += jt[(c, kk)] * p[a * 3 + kk];
                        }
                        uh[c] += uloc[a] * v;
                    }
                }
                let ex = e_exact(&xp);
                e2 += w
                    * ((uh[0] - ex[0]).powi(2) + (uh[1] - ex[1]).powi(2) + (uh[2] - ex[2]).powi(2));
            }
        }
        let l2 = e2.sqrt();
        assert!(
            (l2 - want_l2).abs() < 5e-13 * want_l2,
            "nz{nz}: dense L2 {l2} vs {want_l2}"
        );
    }
}

/// Trilinear hex Jacobian at `xi` (ex3 `jac_3d`'s hex branch).
fn crate_common_jac(
    mesh: &Mesh<3>,
    e: u32,
    xi: &[f64],
) -> (nalgebra::DMatrix<f64>, [f64; 3]) {
    let n = mesh.element_nodes(e);
    const C: [[f64; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let mut j = nalgebra::DMatrix::<f64>::zeros(3, 3);
    let mut xp = [0.0_f64; 3];
    for (i, c) in C.iter().enumerate() {
        let p = mesh.node_coords(n[i]);
        let a = [1.0 + xi[0] * c[0], 1.0 + xi[1] * c[1], 1.0 + xi[2] * c[2]];
        let dn = [c[0] * a[1] * a[2] / 8.0, a[0] * c[1] * a[2] / 8.0, a[0] * a[1] * c[2] / 8.0];
        let ni = a[0] * a[1] * a[2] / 8.0;
        for d in 0..3 {
            for cc in 0..3 {
                j[(d, cc)] += p[d] * dn[cc];
            }
            xp[d] += p[d] * ni;
        }
    }
    (j, xp)
}
