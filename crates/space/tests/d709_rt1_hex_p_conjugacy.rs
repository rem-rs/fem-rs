//! D709 — RT1-hex P-conjugacy pins (round 68).
//!
//! The d690 evidence chain showed that multiset/diagonal/sum invariants are
//! blind to *slot-position* defects: an entrywise-equal mass with permuted
//! columns still matches them all.  This file pins a **conjugacy invariant**
//! that any slot/sign permutation breaks at O(1):
//!
//! RT0 ⊂ RT1 on the hex, so with the L2 projection `Π = M1⁻¹B`
//! (`B = ∫φ1·φ0`, the RT1×RT0 mixed mass on the *same* mesh) the mass Grams
//! must agree:
//!
//! ```text
//!     (B u)ᵀ M1⁻¹ (B u)  ==  uᵀ M0 u
//! ```
//!
//! for any RT0 coefficient vector `u` — the L2 norm of the field equals the
//! L2 norm of its projection, because the projection reproduces it exactly.
//! `B` scatters each element's mixed block through `element_dofs` /
//! `element_signs` of *both* spaces, so a permutation of the RT1 face-grid
//! slots or a wrong orientation sign breaks the identity — the exact defect
//! class d690's invariants could not see.  `M1⁻¹` acts through Jacobi PCG.
//!
//! Run on a straight 2×2×2 hex mesh (`d525_hex222.mesh`) and on the curved
//! (P2) wall hex of `d708_curved_hex.mesh`; plus the D526 cross-element
//! nodal-point consistency sweep on the refined multidomain cylinder
//! (per-slot physical agreement of every shared dof, panicking on any
//! face-grid transform break).

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_element::raviart_thomas::HexRTk;
use fem_element::VectorReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::submesh::extract_submesh_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_hex8_uniform, Mesh};
use fem_solver::{solve_pcg_dsmoother, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::hdiv::HDivSpace;

fn data(rel: &str) -> String {
    format!("{}/tests/data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// Local quadrature matrices through the element's own (possibly curved) map:
/// `m1` (36×36), `b` (36×6), `m0` (6×6) with `φ = J·a / det` (contravariant
/// Piola, `J` from the isoparametric geometry at each quadrature point).
fn local_rt_rt0_mass<M: MeshTopology>(
    mesh: &M,
    e: u32,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let el1: Box<dyn VectorReferenceElement> = Box::new(HexRTk::new_gauss_legendre(1));
    let el0: Box<dyn VectorReferenceElement> = Box::new(HexRTk::new_gauss_legendre(0));
    let n1 = el1.n_dofs();
    let n0 = el0.n_dofs();

    // The integrand carries the map's rational 1/det, so over-integrate; the
    // residual quadrature error sits far below the pin tolerance.
    let quad = el1.quadrature(16);

    let mut m1 = vec![vec![0.0_f64; n1]; n1];
    let mut b = vec![vec![0.0_f64; n0]; n1];
    let mut m0 = vec![vec![0.0_f64; n0]; n0];
    let mut phi1 = vec![0.0_f64; n1 * 3];
    let mut phi0 = vec![0.0_f64; n0 * 3];

    // D709: the SAME isoparametric geometry path the assembler uses for M1
    // (curved P2 hex maps included) — `element_jacobian_at` would
    // straighten a curved hex and break the identity.
    let geo_elem = fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, e)
        .expect("hex geometry reference element");
    let geo_nds = mesh.geometry_nodes(e);
    for (xi, w) in quad.points.iter().zip(quad.weights.iter()) {
        let (jac, det, _x) = fem_assembly::vector_assembler::isoparametric_jacobian(
            mesh,
            geo_nds,
            geo_elem.as_ref(),
            xi,
            3,
        );
        assert!(det > 0.0, "inverted curved hex at {xi:?} (det {det})");
        el1.eval_basis_vec(xi, &mut phi1);
        el0.eval_basis_vec(xi, &mut phi0);

        // φ = J·a / det, weight includes |det| (positive here).
        let p1 = |i: usize| {
            let a = [phi1[3 * i], phi1[3 * i + 1], phi1[3 * i + 2]];
            [
                (jac[(0, 0)] * a[0] + jac[(0, 1)] * a[1] + jac[(0, 2)] * a[2]) / det,
                (jac[(1, 0)] * a[0] + jac[(1, 1)] * a[1] + jac[(1, 2)] * a[2]) / det,
                (jac[(2, 0)] * a[0] + jac[(2, 1)] * a[1] + jac[(2, 2)] * a[2]) / det,
            ]
        };
        let p0 = |j: usize| {
            let a = [phi0[3 * j], phi0[3 * j + 1], phi0[3 * j + 2]];
            [
                (jac[(0, 0)] * a[0] + jac[(0, 1)] * a[1] + jac[(0, 2)] * a[2]) / det,
                (jac[(1, 0)] * a[0] + jac[(1, 1)] * a[1] + jac[(1, 2)] * a[2]) / det,
                (jac[(2, 0)] * a[0] + jac[(2, 1)] * a[1] + jac[(2, 2)] * a[2]) / det,
            ]
        };

        let w = *w * det;
        for i in 0..n1 {
            let fi = p1(i);
            for i2 in 0..n1 {
                let f2 = p1(i2);
                m1[i][i2] += w * (fi[0] * f2[0] + fi[1] * f2[1] + fi[2] * f2[2]);
            }
            for j in 0..n0 {
                let f0 = p0(j);
                b[i][j] += w * (fi[0] * f0[0] + fi[1] * f0[1] + fi[2] * f0[2]);
            }
        }
        for j in 0..n0 {
            let f0 = p0(j);
            for j2 in 0..n0 {
                let g0 = p0(j2);
                m0[j][j2] += w * (f0[0] * g0[0] + f0[1] * g0[1] + f0[2] * g0[2]);
            }
        }
    }
    (m1, b, m0)
}

/// The RT0→RT1 mixed mass `B = ∫φ1·φ0` in global dofs, scattered through both
/// spaces' tables and orientation signs (each element's block is its own
/// integral, so shared-dof accumulation is the standard assembly semantics).
fn global_mixed_mass(sp0: &HDivSpace<Mesh<3>>, sp1: &HDivSpace<Mesh<3>>) -> CsrMatrix<f64> {
    let mesh = sp0.mesh();
    let n0 = sp0.n_dofs();
    let mut coo = CooMatrix::<f64>::new(sp1.n_dofs(), n0);
    for e in 0..mesh.n_elems() as u32 {
        let (_m1_loc, b_loc, _m0) = local_rt_rt0_mass(mesh, e);
        let d0 = sp0.element_dofs(e);
        let s0 = sp0.element_signs(e);
        let d1 = sp1.element_dofs(e);
        let s1 = sp1.element_signs(e);
        for i in 0..d1.len() {
            for j in 0..d0.len() {
                let v = s1[i] * b_loc[i][j] * s0[j];
                if v != 0.0 {
                    coo.add(d1[i] as usize, d0[j] as usize, v);
                }
            }
        }
    }
    coo.into_csr()
}

fn assert_p_conjugacy(mesh: &Mesh<3>, label: &str) {
    let sp0 = HDivSpace::new(mesh.clone(), 0);
    let sp1 = HDivSpace::new(mesh.clone(), 1);

    let mass = VectorMassIntegrator { alpha: 1.0 };
    let qp1 =
        fem_assembly::standard::mfem_vector_mass_quad_order_rt_hex(1, mesh.geom_order());
    let m1 = VectorAssembler::assemble_bilinear(&sp1, &[&mass], qp1);
    let qp0 =
        fem_assembly::standard::mfem_vector_mass_quad_order_rt_hex(0, mesh.geom_order());
    let m0 = VectorAssembler::assemble_bilinear(&sp0, &[&mass], qp0);
    let b = global_mixed_mass(&sp0, &sp1);

    // Deterministic pseudo-random RT0 coefficient vector.
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let u: Vec<f64> = (0..sp0.n_dofs()).map(|_| next()).collect();

    let spmv = |m: &CsrMatrix<f64>, v: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0_f64; m.nrows];
        m.spmv(v, &mut y);
        y
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();

    // rhs = (B u)ᵀ M1⁻¹ (B u) through the Jacobi-PCG solve.
    let bu = spmv(&b, &u);
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };
    let mut proj = vec![0.0_f64; sp1.n_dofs()];
    solve_pcg_dsmoother(&m1, &bu, &mut proj, &cfg).expect("M1 CG solve failed");

    let lhs = dot(&u, &spmv(&m0, &u));
    let rhs = dot(&bu, &proj);
    let rel = (lhs - rhs).abs() / (lhs.abs() + rhs.abs() + 1e-300);
    assert!(
        rel < 1e-6,
        "{label}: uᵀM0u={lhs:e} but (Bu)ᵀM1⁻¹(Bu)={rhs:e} (rel {rel:e})"
    );
}

#[test]
fn d709_p_conjugacy_straight_hex222() {
    let mfem = read_mfem_file(data("d525_hex222.mesh")).expect("read hex222");
    let mesh: Mesh<3> = mfem.mesh3d.expect("3-D");
    assert_eq!(mesh.n_elems(), 8);
    assert_p_conjugacy(&mesh, "straight 2x2x2 hex");
}

#[test]
fn d709_p_conjugacy_curved_wall_hex() {
    let mfem = read_mfem_file(data("d708_curved_hex.mesh")).expect("read curved hex");
    let mesh: Mesh<3> = mfem.mesh3d.expect("3-D");
    assert_eq!(mesh.geom_order(), 2);
    assert_p_conjugacy(&mesh, "curved P2 wall hex");
}

/// D526 cross-element consistency on the refined multidomain cylinder: two
/// elements sharing a face must assign the same physical nodal point to the
/// shared dof — `dof_nodal_coords` panics on any slot-transform permutation.
#[test]
fn d709_cylinder_nodal_point_consistency() {
    let mfem = read_mfem_file(data("d708_multidomain_hex.mesh")).expect("read parent mesh");
    let parent: Mesh<3> = mfem.mesh3d.expect("3-D");
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let parent = refine_hex8_uniform(&parent, &all).0;
    let sub = extract_submesh_3d(&parent, &[1]);
    let space = HDivSpace::new(sub.mesh.clone(), 1);
    let pts = space.dof_nodal_coords();
    assert_eq!(pts.len(), space.n_dofs());
}
