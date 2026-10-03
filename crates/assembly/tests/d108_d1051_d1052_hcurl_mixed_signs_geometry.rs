//! d108 / D1051 + D1052 red-green pins for the HCurl×H¹ mixed family in
//! `fem_assembly::mixed`.
//!
//! # D1051 — `assemble_hcurl_h1_mixed` (curl coupling, H¹ rows × H(curl) cols)
//!
//! The kernel scattered the **raw unsigned** element-local curl columns: no
//! `col_space.element_signs(e)` (the same disease D1041 fixed for the
//! weak-divergence sibling via the parallel signed kernel), *and* — found
//! while pinning — a 2-D slot shift (`[i·dim + dim−1]` reads over a
//! stride-1 `eval_curl` buffer) and a missing curl Piola transform (the
//! entries were `detJ`-scaled reference curls instead of physical curls;
//! 3-D additionally misses the `J` factor).  MFEM's `MixedScalarCurlIntegrator`
//! transforms the trial curl to the physical element and scatters through the
//! signed `GetElementDofs` table (`SparseMatrix::AddSubMatrix` negates the
//! columns of negative-vdir dofs).
//!
//! Pins:
//! 1. **Algebraic (Whitney oracle)** — on affine simplices the physical curl
//!    of the ND1 basis is constant per element
//!    (`curl_z(Φ_i^phys) = curl̂_i/detJ` in 2-D, `(J·curl̂_i)_z/detJ` in 3-D),
//!    so with P1 test rows the exact global matrix is
//!    `M[row(v_j), col(edge_i)] = s_i · (∫_K λ_j dx) · curl_z(Φ_i^phys)` with
//!    `s_i` the space's own orientation sign.  The pin rebuilds that oracle
//!    from vertex coordinates + the element-crate reference curls and requires
//!    entry-wise equality.
//! 2. **Physical annihilation** — the ND1 interpolant of an exact gradient is
//!    discretely curl-free (`Σ edge circulations of ∇u around any element
//!    vanish identically`, the de Rham `curl∘I = I∘curl` at Whitney level), so
//!    `‖M · E_interp‖ ≈ 0`.  The unsigned/slot-shifted kernel gives `O(‖E‖)`.
//!
//! # D1052 — `assemble_hcurl_h1_gradient` iso branch geometry row
//!
//! The isoparametric branch passed `mesh.element_nodes(e)` (the 8 hex
//! corners) to `isoparametric_jacobian`, but the curved geometry reference
//! element (ball-quad hexes: 27-node Q2 lattice) indexes the **geometry
//! table** row `mesh.geometry_nodes(e)` — pre-fix, any curved non-simplex
//! mesh panics with an index-out-of-bounds (tesla survived only by routing
//! its gradient through `DiscreteLinearOperator::gradient`).  The pin runs
//! the kernel on the curved d103/MFEM tesla ball-quad mesh and requires
//! entry-wise agreement with a hand-rolled isoparametric reference; the
//! straight cylinder-hex mesh is the no-op control.
//!
//! Geometry keys (same guard discipline as
//! `crates/parallel/tests/d107_d1041_weak_div_signs.rs`): every pin mesh must
//! actually carry reversed H(curl) edges (D1051) / high-order geometry
//! (D1052), or the pin demands a re-key.

use fem_assembly::mixed::{
    assemble_hcurl_h1_gradient, assemble_hcurl_h1_mixed, ref_elem_vec, HCurlH1CurlIntegrator,
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology as _;
use fem_mesh::Mesh;
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{HCurlSpace, H1Space};

/// True when at least one element of the mesh carries a negative H(curl)
/// orientation sign — the guard that keeps the pins keyed to meshes that
/// actually exercise reversed edges.
fn has_reversed_edges(mesh: &Mesh<2>, order: u8) -> bool {
    let nd = HCurlSpace::new(mesh.clone(), order);
    (0..mesh.n_elements() as u32)
        .map(|e| nd.element_signs(e))
        .flatten()
        .any(|&s| s < 0.0)
}

fn has_reversed_edges_3d(mesh: &Mesh<3>, order: u8) -> bool {
    let nd = HCurlSpace::new(mesh.clone(), order);
    (0..mesh.n_elements() as u32)
        .map(|e| nd.element_signs(e))
        .flatten()
        .any(|&s| s < 0.0)
}

/// Whitney-oracle reference for the D1051 kernel at order 1 on affine
/// simplices (see the module docs): `ref[r][c]` accumulates, per element,
/// `s_i · (∫_K λ_j dx) · curl_z(Φ_i^phys)` for every local vertex row `j` and
/// local ND1 column `i`.
fn whitney_curl_reference(
    mesh: &Mesh<2>,
    h1: &H1Space<Mesh<2>>,
    nd: &HCurlSpace<Mesh<2>>,
) -> Vec<f64> {
    let n_r = h1.n_dofs();
    let n_c = nd.n_dofs();
    let mut r = vec![0.0_f64; n_r * n_c];

    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        assert!(
            matches!(et, ElementType::Tri3 | ElementType::Tri6),
            "Whitney oracle: tri-only, got {et:?}"
        );
        let ref_c = ref_elem_vec(et, 1, SpaceType::HCurl).unwrap();
        let n_loc = ref_c.n_dofs();

        let v: Vec<Vec<f64>> = mesh
            .element_nodes(e)
            .iter()
            .map(|&n| mesh.geom_coords_of(n).to_vec())
            .collect();
        let (j00, j01) = (v[1][0] - v[0][0], v[2][0] - v[0][0]);
        let (j10, j11) = (v[1][1] - v[0][1], v[2][1] - v[0][1]);
        let det = j00 * j11 - j01 * j10;
        let area = det.abs() / 2.0;
        let moment = area / 3.0; // ∫_K λ_j dx, P1

        // ND1 reference curls are constant on the element; any reference
        // point works.  2-D `eval_curl` writes n_loc scalars.
        let mut curl_hat = vec![0.0_f64; n_loc];
        ref_c.eval_curl(&[1.0 / 3.0, 1.0 / 3.0], &mut curl_hat);

        let signs = nd.element_signs(e);
        let nd_dofs = nd.element_dofs(e);
        let h1_dofs = h1.element_dofs(e);
        for i in 0..n_loc {
            let s = signs.get(i).copied().unwrap_or(1.0);
            // Physical scalar curl on the affine element: curl̂/detJ.
            let curl_phys = curl_hat[i] / det;
            for j in 0..v.len() {
                let row = h1_dofs[j] as usize;
                let col = nd_dofs[i] as usize;
                r[row * n_c + col] += s * moment * curl_phys;
            }
        }
    }
    r
}

/// Tet twin of [`whitney_curl_reference`]: the 3-D curl carries the
/// `J·curl̂/detJ` transform and the integrator pairs against its z-component.
fn whitney_curl_reference_tet(
    mesh: &Mesh<3>,
    h1: &H1Space<Mesh<3>>,
    nd: &HCurlSpace<Mesh<3>>,
) -> Vec<f64> {
    let n_r = h1.n_dofs();
    let n_c = nd.n_dofs();
    let mut r = vec![0.0_f64; n_r * n_c];

    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        assert!(
            matches!(et, ElementType::Tet4 | ElementType::Tet10),
            "Whitney oracle: tet-only, got {et:?}"
        );
        let ref_c = ref_elem_vec(et, 1, SpaceType::HCurl).unwrap();
        let n_loc = ref_c.n_dofs();

        let v: Vec<Vec<f64>> = mesh
            .element_nodes(e)
            .iter()
            .map(|&n| mesh.geom_coords_of(n).to_vec())
            .collect();
        // J columns: p1−p0, p2−p0, p3−p0.
        let mut jac = nalgebra::DMatrix::<f64>::zeros(3, 3);
        for (c, vc) in v.iter().skip(1).enumerate() {
            for rr in 0..3 {
                jac[(rr, c)] = vc[rr] - v[0][rr];
            }
        }
        let det = jac.determinant();
        let vol = det.abs() / 6.0;
        let moment = vol / 4.0; // ∫_K λ_j dx, P1

        // 3-D `eval_curl` writes n_loc × 3.
        let mut curl_hat = vec![0.0_f64; n_loc * 3];
        ref_c.eval_curl(&[0.25, 0.25, 0.25], &mut curl_hat);

        let signs = nd.element_signs(e);
        let nd_dofs = nd.element_dofs(e);
        let h1_dofs = h1.element_dofs(e);
        for i in 0..n_loc {
            let s = signs.get(i).copied().unwrap_or(1.0);
            // curl_z(Φ_i^phys) = (J·curl̂_i)_z / detJ.
            let mut cz = 0.0_f64;
            for k in 0..3 {
                cz += jac[(2, k)] * curl_hat[i * 3 + k];
            }
            let curl_phys_z = cz / det;
            for j in 0..v.len() {
                let row = h1_dofs[j] as usize;
                let col = nd_dofs[i] as usize;
                r[row * n_c + col] += s * moment * curl_phys_z;
            }
        }
    }
    r
}

fn max_dev(a: &CsrMatrix<f64>, dense: &[f64], ncols: usize) -> f64 {
    let mut m = 0.0_f64;
    for row in 0..a.nrows {
        for col in 0..a.ncols {
            m = m.max((a.get(row, col) - dense[row * ncols + col]).abs());
        }
    }
    m
}

// ─── D1051 pin 1: algebraic Whitney identity ─────────────────────────────────

#[test]
fn d1051_mixed_curl_matches_whitney_oracle_tri() {
    let name = "unit_square_tri(3)";
    let mesh = Mesh::<2>::unit_square_tri(3);
    assert!(
        has_reversed_edges(&mesh, 1),
        "{name}: expected at least one reversed edge (pin geometry key)"
    );
    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);

    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], 3);
    assert_eq!(m.nrows, h1.n_dofs());
    assert_eq!(m.ncols, nd.n_dofs());

    let oracle = whitney_curl_reference(&mesh, &h1, &nd);
    let dev = max_dev(&m, &oracle, nd.n_dofs());
    println!("{name}: max |M − Whitney oracle| = {dev:.3e}");
    assert!(
        dev < 1e-13,
        "{name}: mixed curl kernel deviates from the Whitney oracle by {dev:.3e} \
         (D1051: signs must reach the trial columns, 2-D curl slot must be the \
         last of each dof chunk, curl must be Piola-transformed)"
    );
}

#[test]
fn d1051_mixed_curl_matches_whitney_oracle_tet() {
    let name = "unit_cube_tet(2)";
    let mesh = Mesh::<3>::unit_cube_tet(2);
    assert!(
        has_reversed_edges_3d(&mesh, 1),
        "{name}: expected at least one reversed edge (pin geometry key)"
    );
    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);

    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], 3);
    let oracle = whitney_curl_reference_tet(&mesh, &h1, &nd);
    let dev = max_dev(&m, &oracle, nd.n_dofs());
    println!("{name}: max |M − Whitney oracle| = {dev:.3e}");
    assert!(
        dev < 1e-13,
        "{name}: mixed curl kernel deviates from the tet Whitney oracle by {dev:.3e} \
         (3-D path: J·curl̂/detJ transform + per-dof trial signs)"
    );
}

// ─── D1051 pin 2: physical annihilation (discrete curl of a gradient) ────────

#[test]
fn d1051_mixed_curl_annihilates_gradient_interpolant() {
    let name = "unit_square_tri(3)";
    let mesh = Mesh::<2>::unit_square_tri(3);
    assert!(
        has_reversed_edges(&mesh, 1),
        "{name}: expected at least one reversed edge (pin geometry key)"
    );
    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);

    // E = ∇u with u = 1 + 2x − 3y + xy: the ND1 interpolant (edge line
    // integrals) is discretely curl-free on every affine element.
    let e_h = nd
        .interpolate_vector(&|x: &[f64]| vec![2.0 + x[1], -3.0 + x[0]])
        .as_slice()
        .to_vec();
    let e_norm: f64 = e_h.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(e_norm > 0.1, "{name}: vanishing interpolant, pin is vacuous");

    let m = assemble_hcurl_h1_mixed(&h1, &nd, &[&HCurlH1CurlIntegrator], 3);
    let mut me = vec![0.0_f64; m.nrows];
    m.spmv(&e_h, &mut me);
    let norm: f64 = me.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("{name}: ‖E‖ = {e_norm:.6e}  ‖M·E‖ = {norm:.3e}");
    assert!(
        norm < 1e-12,
        "{name}: curl-grad annihilation broken, ‖M·E‖ = {norm:.3e} \
         (D1051: with correct signs the Whitney interpolant of ∇u has \
         element-wise zero curl, so every row must vanish)"
    );
}

// ─── D1052 pin: gradient kernel on curved (27-node) hex geometry ─────────────

/// Hand-rolled isoparametric reference for the gradient form
/// `B[i,j] = ∫ ψ_i · ∇φ_j dx` (ND rows, H¹ cols) — the same quadrature, the
/// geometry-table node row, the covariant Piola transforms and the ND row
/// signs, written against public primitives only.
fn gradient_reference(mesh: &Mesh<3>, nd: &HCurlSpace<Mesh<3>>, h1: &H1Space<Mesh<3>>, qo: u8) -> Vec<f64> {
    let dim = 3usize;
    let n_r = nd.n_dofs();
    let n_c = h1.n_dofs();
    let mut r = vec![0.0_f64; n_r * n_c];

    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let h1_ref = fem_space::ref_elem::h1_field_element(et, h1.order(), h1.pyramid_basis());
        let nd_ref = ref_elem_vec(et, nd.order(), SpaceType::HCurl).unwrap();
        let n_nd = nd_ref.n_dofs();
        let n_h1 = h1_ref.n_dofs();
        let quad = h1_ref.quadrature(qo);

        let geo = fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, e)
            .expect("curved hex geometry element");
        let gnodes = mesh.geometry_nodes(e).to_vec();
        let signs = nd.element_signs(e);
        let nd_dofs: Vec<usize> = nd.element_dofs(e).iter().map(|&d| d as usize).collect();
        let h1_dofs: Vec<usize> = h1.element_dofs(e).iter().map(|&d| d as usize).collect();

        let mut gr = vec![0.0_f64; n_h1 * dim];
        let mut gp = vec![0.0_f64; n_h1 * dim];
        let mut nb = vec![0.0_f64; n_nd * dim];

        for (xi, wq) in quad.points.iter().zip(quad.weights.iter()) {
            let (jac, det, _) =
                fem_assembly::vector_assembler::isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, dim);
            let jit = jac.try_inverse().expect("invertible J").transpose();
            h1_ref.eval_grad_basis(xi, &mut gr);
            for k in 0..n_h1 {
                for c in 0..dim {
                    gp[k * dim + c] = (0..dim).map(|kk| jit[(c, kk)] * gr[k * dim + kk]).sum();
                }
            }
            nd_ref.eval_basis_vec(xi, &mut nb);
            for i in 0..n_nd {
                let s = signs.get(i).copied().unwrap_or(1.0);
                // Covariant Piola: ψ = J⁻ᵀ ψ̂.
                let psi: Vec<f64> = (0..dim)
                    .map(|c| s * (0..dim).map(|k| jit[(c, k)] * nb[i * dim + k]).sum::<f64>())
                    .collect();
                for j in 0..n_h1 {
                    let dot: f64 = (0..dim).map(|c| gp[j * dim + c] * psi[c]).sum();
                    r[nd_dofs[i] * n_c + h1_dofs[j]] += wq * det * dot;
                }
            }
        }
    }
    r
}

fn dense_max_dev(a: &CsrMatrix<f64>, dense: &[f64], ncols: usize) -> f64 {
    let mut m = 0.0_f64;
    for row in 0..a.nrows {
        for col in 0..a.ncols {
            m = m.max((a.get(row, col) - dense[row * ncols + col]).abs());
        }
    }
    m
}

/// D1052: the iso branch must build the Jacobian from `geometry_nodes(e)`.
/// Pre-fix this test panicked with index-out-of-bounds on the 27-node curved
/// hexes (the red evidence lives in `tmp/d108b/`).
#[test]
fn d1052_gradient_kernel_curved_hex_geometry() {
    let (name, rel, qo, order) = ("ball-quad", "tests/data/ball-quad.mesh", 7u8, 1u8);
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{name}: 3-D mesh"));

    // Geometry guard: the mesh must exercise the high-order geometry table
    // (27 geometry nodes per hex vs 8 corners) — otherwise the pin is not
    // keyed to the D1052 defect and needs a different mesh.
    assert!(
        mesh.geom_order() > 1,
        "{name}: geom_order = {} — the pin needs a curved mesh",
        mesh.geom_order()
    );
    let e0 = 0u32;
    assert_eq!(mesh.element_type(e0), ElementType::Hex8, "{name}: hex mesh");
    assert!(
        mesh.geometry_nodes(e0).len() > mesh.element_nodes(e0).len(),
        "{name}: geometry table row ({}) must exceed the corner row ({}) — \
         re-key the pin to a curved mesh",
        mesh.geometry_nodes(e0).len(),
        mesh.element_nodes(e0).len()
    );
    assert!(
        has_reversed_edges_3d(&mesh, order),
        "{name}: expected reversed H(curl) edges (sign coverage)"
    );

    let h1 = H1Space::new(mesh.clone(), order);
    let nd = HCurlSpace::new(mesh.clone(), order);

    // Pre-fix: index-out-of-bounds panic right here (element_nodes has 8
    // entries, the Q2 geometry lattice needs 27).
    let b = assemble_hcurl_h1_gradient(&nd, &h1, qo);
    assert_eq!(b.nrows, nd.n_dofs());
    assert_eq!(b.ncols, h1.n_dofs());

    let oracle = gradient_reference(&mesh, &nd, &h1, qo);
    let dev = dense_max_dev(&b, &oracle, h1.n_dofs());
    println!("{name}: max |B − isoparametric reference| = {dev:.3e}");
    assert!(
        dev < 1e-10,
        "{name}: gradient kernel deviates from the curved-geometry reference \
         by {dev:.3e} (D1052: iso branch must read mesh.geometry_nodes(e))"
    );
}

/// Control: on a *straight* hex mesh `geometry_nodes(e)` equals
/// `element_nodes(e)`, so the fix is a no-op there — the kernel must still
/// match the same reference on the sign-heavy cylinder-hex mesh.
#[test]
fn d1052_gradient_kernel_straight_hex_control() {
    let (name, rel, qo, order) = ("cylinder-hex", "../../data/cylinder-hex.mesh", 4u8, 1u8);
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{name}: 3-D mesh"));
    let e0 = 0u32;
    assert_eq!(mesh.element_type(e0), ElementType::Hex8, "{name}: hex mesh");
    assert!(
        mesh.geometry_nodes(e0).len() == mesh.element_nodes(e0).len(),
        "{name}: control mesh must be straight (equal node rows)"
    );
    assert!(
        has_reversed_edges_3d(&mesh, order),
        "{name}: expected reversed H(curl) edges (sign coverage)"
    );

    let h1 = H1Space::new(mesh.clone(), order);
    let nd = HCurlSpace::new(mesh.clone(), order);
    let b = assemble_hcurl_h1_gradient(&nd, &h1, qo);
    let oracle = gradient_reference(&mesh, &nd, &h1, qo);
    let dev = dense_max_dev(&b, &oracle, h1.n_dofs());
    println!("{name}: max |B − reference| = {dev:.3e}");
    assert!(
        dev < 1e-10,
        "{name}: gradient kernel deviates from the straight-hex reference by {dev:.3e}"
    );
}
