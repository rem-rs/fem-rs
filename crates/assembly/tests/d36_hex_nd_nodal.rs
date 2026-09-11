//! Round-15 D36 regressions: the hexahedral Nédélec (`HexNDk`) family is
//! *nodal* — every DOF is the point-value functional
//! `σ_i(Φ) = Φ(x_i)·t̂_i` at MFEM's `FE::Nodes` (Gauss-Legendre open points
//! along the component direction) with the unnormalized tangent
//! `t̂_i = J·(2 e_a)` (MFEM `dof2tk`), and the local DOFs are the point values
//! themselves.
//!
//! These tests pin, for `k = 1, 2, 3`:
//!
//! 1. `HCurlSpace::interpolate_vector` reproduces fields that lie in the space
//!    (constant and linear) *exactly* — the end-to-end check of the nodal DOF
//!    semantics,
//! 2. the L2 mass projection of those fields is exact,
//! 3. tangential traces agree across shared quad faces of a multi-hex mesh,
//!    including faces whose adjacent elements see the face with
//!    differently-oriented local vertex cycles (MFEM
//!    `DofOrderForOrientation(QUARE, or)` signed permutation).

use fem_assembly::postproc::coefficient::{CoeffCtx, VectorCoeff};
use fem_assembly::standard::{VectorDomainLFIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::VectorAssembler;
use fem_element::nedelec::HexNDk;
use fem_element::reference::VectorReferenceElement;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::HCurlSpace;

/// Local quad faces of a hex, MFEM `Geometry::Constants<CUBE>::FaceVert`
/// order (same table as `HCurlSpace::HEX_QUAD_FACES`).
const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
];

fn random_coeffs(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            2.0 * r - 1.0
        })
        .collect()
}

/// Physical field `Σ_g c_g s_g J^-T φ_ref(ξ)` of element `e`, plus the
/// physical image of `ξ`.
fn eval_field(
    mesh: &Mesh<3>,
    space: &HCurlSpace<Mesh<3>>,
    refe: &dyn VectorReferenceElement,
    coeff: &[f64],
    e: u32,
    xi: &[f64],
) -> ([f64; 3], [f64; 3]) {
    let nodes = mesh.element_nodes(e);
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("hex geometry element");
    let (jac, _det, xp) = isoparametric_jacobian(mesh, nodes, geo.as_ref(), xi, 3);
    let jit = jac.try_inverse().expect("invertible hex Jacobian").transpose();
    let n = refe.n_dofs();
    let mut vref = vec![0.0_f64; n * 3];
    refe.eval_basis_vec(xi, &mut vref);
    let dofs = space.element_dofs(e);
    let signs = space.element_signs(e);
    let mut u = [0.0_f64; 3];
    for i in 0..n {
        let c = coeff[dofs[i] as usize] * signs[i];
        for d in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += jit[(d, k)] * vref[i * 3 + k];
            }
            u[d] += c * s;
        }
    }
    (u, [xp[0], xp[1], xp[2]])
}

/// Reference coordinates of the physical point `x` in the (box) element `e`:
/// the map of a box is affine, so `ξ = J(0)^{-1}(x − x(0))` inverts it for any
/// local frame orientation (including permuted/reflected vertex orders).
fn box_params(mesh: &Mesh<3>, e: u32, x: &[f64]) -> [f64; 3] {
    let nodes = mesh.element_nodes(e);
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("hex geometry element");
    let (jac, _det, xc) = isoparametric_jacobian(mesh, nodes, geo.as_ref(), &[0.0, 0.0, 0.0], 3);
    let d = jac.try_inverse().expect("invertible box Jacobian");
    let mut xi = [0.0_f64; 3];
    for r in 0..3 {
        xi[r] = d[(r, 0)] * (x[0] - xc[0]) + d[(r, 1)] * (x[1] - xc[1]) + d[(r, 2)] * (x[2] - xc[2]);
    }
    xi
}

/// Whether element `e` contains the point (box meshes).
fn elem_contains(mesh: &Mesh<3>, e: u32, x: &[f64]) -> bool {
    let xi = box_params(mesh, e, x);
    xi.iter().all(|&v| v > -1.0 - 1e-9 && v < 1.0 + 1e-9)
}

// ─── 1. interpolation / projection of in-space fields is exact ──────────────

/// In-space fields (per component `Q_{k-1,k,k} × Q_{k,k-1,k} × Q_{k,k,k-1}`):
/// a constant, a linear field, and (for `k >= 2`) a field using the open
/// x-direction modes.
#[derive(Clone, Copy)]
enum Field {
    ConstX,
    LinearY,
    LinearXy,
}

impl Field {
    fn value(&self, x: &[f64]) -> [f64; 3] {
        match self {
            Field::ConstX => [1.0, 0.0, 0.0],
            Field::LinearY => [0.0, 2.0 * x[0] + 1.0, 0.0],
            Field::LinearXy => [x[0], x[1], 0.0],
        }
    }
}

/// `VectorCoeff` adapter so the same field drives the assembler and the
/// pointwise checks.
struct FieldCoeff(Field);

impl VectorCoeff for FieldCoeff {
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        let v = self.0.value(ctx.x);
        out[..3].copy_from_slice(&v);
    }
}

/// L2 error of the *interpolant* of `f` (via `interpolate_vector`) — exact
/// for in-space fields.
fn interp_error(mesh: &Mesh<3>, k: u8, field: Field) -> f64 {
    let space = HCurlSpace::new(mesh.clone(), k);
    let u = space.interpolate_vector(&|x: &[f64]| field.value(x).to_vec());
    let refe = HexNDk::new(k as usize);
    let qr = refe.quadrature((2 * k + 4) as u8);
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        for (q, xi) in qr.points.iter().enumerate() {
            let (uh, xp) = eval_field(mesh, &space, &refe, u.as_slice(), e, xi);
            let ex = field.value(&xp);
            // |det J| from the geometry map
            let nodes = mesh.element_nodes(e);
            let geo = geo_ref_elem_from_mesh(mesh, e).unwrap();
            let (_j, det, _p) = isoparametric_jacobian(mesh, nodes, geo.as_ref(), xi, 3);
            for d in 0..3 {
                err2 += qr.weights[q] * det.abs() * (uh[d] - ex[d]).powi(2);
            }
        }
    }
    err2.sqrt()
}

/// L2 error of the mass projection `M x = b` of `f`.
fn projection_error(mesh: &Mesh<3>, k: u8, field: Field) -> f64 {
    let space = HCurlSpace::new(mesh.clone(), k);
    let mass = VectorMassIntegrator { alpha: 1.0 };
    let m: CsrMatrix<f64> = VectorAssembler::assemble_bilinear(
        &space,
        &[&mass as &dyn fem_assembly::vector_integrator::VectorBilinearIntegrator],
        2 * k + 3,
    );
    let src = VectorDomainLFIntegrator { f: FieldCoeff(field) };
    let b = VectorAssembler::assemble_linear(
        &space,
        &[&src as &dyn fem_assembly::vector_integrator::VectorLinearIntegrator],
        2 * k + 3,
    );
    let n = space.n_dofs();
    let d = m.to_dense();
    let a = nalgebra::DMatrix::from_row_slice(n, n, &d);
    let x = a
        .lu()
        .solve(&nalgebra::DVector::from_column_slice(&b))
        .expect("hex mass matrix singular");

    let refe = HexNDk::new(k as usize);
    let qr = refe.quadrature((2 * k + 4) as u8);
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let geo = geo_ref_elem_from_mesh(mesh, e).unwrap();
        for (q, xi) in qr.points.iter().enumerate() {
            let (uh, xp) = eval_field(mesh, &space, &refe, x.as_slice(), e, xi);
            let (_j, det, _p) = isoparametric_jacobian(mesh, nodes, geo.as_ref(), xi, 3);
            let ex = field.value(&xp);
            for dd in 0..3 {
                err2 += qr.weights[q] * det.abs() * (uh[dd] - ex[dd]).powi(2);
            }
        }
    }
    err2.sqrt()
}

#[test]
fn hex_ndk_interpolation_is_exact() {
    for k in 1..=3u8 {
        for mesh in [Mesh::<3>::unit_cube_hex(1), Mesh::<3>::unit_cube_hex(2)] {
            for field in in_space_fields(k) {
                let err = interp_error(&mesh, k, field);
                assert!(
                    err < 1e-12,
                    "hex ND{k} interpolate {}: L2 error {err:.3e}",
                    field_name(field),
                );
            }
        }
    }
}

#[test]
fn hex_ndk_projection_is_exact() {
    for k in 1..=3u8 {
        for mesh in [Mesh::<3>::unit_cube_hex(1), Mesh::<3>::unit_cube_hex(2)] {
            for field in in_space_fields(k) {
                let err = projection_error(&mesh, k, field);
                assert!(
                    err < 1e-12,
                    "hex ND{k} projection {}: L2 error {err:.3e}",
                    field_name(field),
                );
            }
        }
    }
}

/// Fields lying in `HCurl(Hex8, ND_k)`: the constant and `(0, 2x+1, 0)` are
/// in every order; `(x, y, 0)` needs the open x-direction modes (`k >= 2`).
fn in_space_fields(k: u8) -> Vec<Field> {
    let mut v = vec![Field::ConstX, Field::LinearY];
    if k >= 2 {
        v.push(Field::LinearXy);
    }
    v
}

fn field_name(f: Field) -> &'static str {
    match f {
        Field::ConstX => "const-x",
        Field::LinearY => "linear-y",
        Field::LinearXy => "linear-xy",
    }
}

// ─── 2. shared-face tangential trace identity ───────────────────────────────

/// Two hexes sharing the physical plane `x = 1`, the second built with a
/// *rotated* local frame (local `ξ -> +z`, `η -> +x`, `ζ -> +y`) so that the
/// two elements see the shared quad face with different local vertex cycles
/// — the `DofOrderForOrientation` case.  The interface vertices are shared
/// node ids (topologically conforming mesh).
fn two_hex_rotated() -> Mesh<3> {
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        1.0, 1.0, 0.0, // 2
        0.0, 1.0, 0.0, // 3
        0.0, 0.0, 1.0, // 4
        1.0, 0.0, 1.0, // 5
        1.0, 1.0, 1.0, // 6
        0.0, 1.0, 1.0, // 7
        2.0, 0.0, 1.0, // 8  = v2 of the rotated element
        2.0, 0.0, 0.0, // 9  = v3
        2.0, 1.0, 1.0, // 10 = v6
        2.0, 1.0, 0.0, // 11 = v7
    ];
    // Element 0: standard order.  Element 1: (v0..v7) =
    // ((1,0,0),(1,0,1),(2,0,1),(2,0,0),(1,1,0),(1,1,1),(2,1,1),(2,1,0)) —
    // the same box with the local axes (ξ,η,ζ) = (z,x,y).
    let conn: Vec<u32> = vec![
        0, 1, 2, 3, 4, 5, 6, 7, //
        1, 5, 8, 9, 2, 6, 10, 11,
    ];
    let elem_tags = vec![1i32, 1i32];
    let face_conn: Vec<u32> = Vec::new();
    let face_tags: Vec<i32> = Vec::new();
    Mesh::uniform(
        coords,
        conn,
        elem_tags,
        ElementType::Hex8,
        face_conn,
        face_tags,
        ElementType::Quad4,
    )
}

/// Interior faces of a hex mesh: `(face vertex cycle, elements)`.
fn interior_faces(mesh: &Mesh<3>) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut map: std::collections::HashMap<Vec<u32>, Vec<u32>> = std::collections::HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        for f in HEX_FACES.iter() {
            let mut key: Vec<u32> = f.iter().map(|&i| nodes[i]).collect();
            key.sort_unstable();
            map.entry(key).or_default().push(e);
        }
    }
    map.into_iter()
        .filter(|(_, es)| es.len() == 2)
        .map(|(k, es)| (k, es))
        .collect()
}

/// Max tangential-trace mismatch over all shared faces of `mesh`.
fn trace_mismatch(mesh: &Mesh<3>, k: u8, seed: u64) -> f64 {
    let space = HCurlSpace::new(mesh.clone(), k);
    let x = random_coeffs(space.n_dofs(), seed);
    let refe = HexNDk::new(k as usize);
    let (pts, wts) = fem_assembly::dpg::dpg_basis::face_quadrature(3, true, (2 * k + 3) as u8);
    let mut worst = 0.0_f64;
    for (face, es) in interior_faces(mesh) {
        // Physical face vertices in the stored cycle order (the key is sorted;
        // use the first element's cycle for the bilinear parametrisation).
        let c: Vec<[f64; 3]> = face
            .iter()
            .map(|&n| {
                let p = mesh.node_coords(n);
                [p[0], p[1], p[2]]
            })
            .collect();
        for (q, st) in pts.iter().enumerate() {
            let (s, t) = (st[0], st[1]);
            let mut xp = [0.0_f64; 3];
            for d in 0..3 {
                xp[d] = (1.0 - s) * (1.0 - t) * c[0][d]
                    + s * (1.0 - t) * c[1][d]
                    + s * t * c[2][d]
                    + (1.0 - s) * t * c[3][d];
            }
            let _ = wts[q];
            assert!(elem_contains(mesh, es[0], &xp), "face point inside elem 0");
            assert!(elem_contains(mesh, es[1], &xp), "face point inside elem 1");
            let mut tr = Vec::new();
            for &e in es.iter() {
                let xi = box_params(mesh, e, &xp);
                let (u, _p) = eval_field(mesh, &space, &refe, &x, e, &xi);
                // Outward normal of the element on this face (cross product of
                // the face's in-plane directions).
                let nrm = face_normal(&c);
                let d = u[0] * nrm[0] + u[1] * nrm[1] + u[2] * nrm[2];
                tr.push([u[0] - d * nrm[0], u[1] - d * nrm[1], u[2] - d * nrm[2]]);
            }
            for d in 0..3 {
                worst = worst.max((tr[0][d] - tr[1][d]).abs());
            }
        }
    }
    worst
}

fn face_normal(c: &[[f64; 3]]) -> [f64; 3] {
    let a = [c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]];
    let b = [c[3][0] - c[0][0], c[3][1] - c[0][1], c[3][2] - c[0][2]];
    let n = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let s = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / s, n[1] / s, n[2] / s]
}

#[test]
fn hex_ndk_shared_face_trace_matches() {
    let meshes = [Mesh::<3>::unit_cube_hex(2), two_hex_rotated()];
    for (mi, mesh) in meshes.iter().enumerate() {
        for k in 1..=3u8 {
            let m = trace_mismatch(mesh, k, 12345 + k as u64);
            assert!(
                m < 1e-13,
                "mesh {mi} hex ND{k}: shared-face trace mismatch {m:.3e}",
            );
        }
    }
}

#[test]
fn hex_ndk_two_hex_shared_face_count() {
    let mesh = two_hex_rotated();
    assert_eq!(interior_faces(&mesh).len(), 1, "the two hexes share one face");
    let space = HCurlSpace::new(mesh, 2);
    assert!(space.n_dofs() > 0);
}
