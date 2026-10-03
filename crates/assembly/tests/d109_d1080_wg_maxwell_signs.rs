//! d109 / D1080 red-green pins for the WG Maxwell path
//! (`fem_assembly::wg::assemble_wg_maxwell`).
//!
//! # The debt
//!
//! The whole WG Maxwell chain was sign-blind: the volume weak-curl stiffness
//! `C_wᵀ M_Σ⁻¹ C_w` (`weak_curl_matrix`) and the face-penalty per-element
//! scatter (`add_face_penalty_hcurl`) scattered **raw unsigned** element-local
//! dofs — zero `element_signs` in the file.  MFEM's convention (the family
//! every sibling follows since D1041/D1051): the scatter goes through the
//! **signed** `GetElementDofs`/`GetElementVDofs` tables and
//! `SparseMatrix::AddSubMatrix` multiplies each entry by the *product of the
//! row and column signs* (`linalg/sparsemat.cpp:2795-2811`: row sign `s`,
//! column sign `t = -s`, `if (t < 0) { a = -a; }`); the element matrix of
//! `BilinearForm::AssembleElementMatrix` lands at
//! `mat->AddSubMatrix(vdofs_, vdofs_, elmat)` (`fem/bilininteg…/bilinearform.cpp:420`)
//! and the interior-face matrix at the concatenated signed tables of *both*
//! elements (`fem/bilinearform.cpp:683-697`), i.e. every element block is
//! congruent with that element's own signs (`A ← S A S`).
//! `fem/fespace.hpp:127-137` documents the semantics: a Nédélec dof referenced
//! with the opposite orientation comes back negative and "the value expected
//! by this element should have the opposite sign".
//!
//! While pinning, the volume path showed a second disease of the D1051 family
//! (fixed in the same stroke): `weak_curl_matrix` fed the **raw reference
//! curl** into the pairing instead of the *physical* curl of the mapped
//! Nédélec basis — 2-D `curl̂/detJ`, 3-D `(J·curl̂)/detJ` (MFEM
//! `CalcPhysCurlShape`) — leaving every entry det(J)-scaled (2-D: the element
//! block was `detJ²` × the true one) and, in 3-D, missing the `J` factor
//! altogether.  The weight follows the D696 family verdict (**signed**
//! `ip.weight·Trans.Weight()`), so det(J) cancels exactly on affine elements.
//!
//! # Pins (Whitney-oracle / annihilation discipline, geometric guards)
//!
//! 1. **Volume Whitney oracle (tri + tet, `penalty = 0` ⇒ volume block only)**:
//!    on an affine simplex the physical curl of every ND1 basis is constant,
//!    so with the P₀ scalar/vector flux the *exact* WG element matrix is
//!    closed-form: `A[i,j] = s_i·s_j·|K|·(curl Φ_i^phys · curl Φ_j^phys)`
//!    (2-D scalar curls, 3-D vector curls).  The pin rebuilds it from vertex
//!    coordinates + the element crate's reference curls and requires
//!    entry-wise equality.
//! 2. **Annihilation**: the ND1 interpolant of an exact gradient is
//!    discretely curl-free (the Whitney de Rham identity — the per-element
//!    circulation sum telescopes to zero *through the signed dof table*), so
//!    `A·E ≈ 0`.  The unsigned kernel breaks it on every reversed-edge
//!    element.
//! 3. **Face penalty mirror** on a straight two-triangle patch: the full
//!    assembled matrix (volume + every face block, interior and boundary)
//!    against an independent analytic mirror — corner-composed face reference
//!    points (exact on straight edges), the face measure as the chord length,
//!    `α = penalty/h`, and the **per-element sign products** on both the left
//!    and the right block (the `[vdofs; vdofs2]` diagonal of MFEM's face
//!    scatter).
//!
//! Geometry guards (d108 convention): every pin mesh must actually carry
//! reversed H(curl) edges (`element_signs` with a negative entry), or the pin
//! demands a re-key.

use fem_assembly::{assemble_wg_maxwell, InteriorFaceList};
use fem_element::nedelec::{TetNDk, TriNDk};
use fem_element::quadrature::seg_rule;
use fem_element::VectorReferenceElement;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology as _;
use fem_mesh::Mesh;
use fem_space::HCurlSpace;

/// True when at least one element carries a negative H(curl) orientation sign —
/// the guard that keeps the pins keyed to meshes that exercise reversed edges.
fn has_reversed_edges<const D: usize>(mesh: &Mesh<D>, order: u8) -> bool {
    let nd = HCurlSpace::new(mesh.clone(), order);
    (0..mesh.n_elements() as u32).map(|e| nd.element_signs(e)).flatten().any(|&s| s < 0.0)
}

fn dense_of(k: &CsrMatrix<f64>, n: usize) -> Vec<f64> {
    let mut d = vec![0.0_f64; n * n];
    for i in 0..n {
        for p in k.row_ptr[i]..k.row_ptr[i + 1] {
            d[i * n + k.col_idx[p] as usize] += k.values[p];
        }
    }
    d
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

// ─── Whitney oracles (volume block, penalty = 0) ─────────────────────────────

/// 2-D ND1/P₀ oracle: `A[g_i, g_j] += s_i·s_j·|K|·cz_i·cz_j` with
/// `cz_i = curl̂_i/detJ` (constant on the affine triangle).
fn wg_volume_oracle_tri(mesh: &Mesh<2>, nd: &HCurlSpace<Mesh<2>>) -> Vec<f64> {
    let n = nd.n_dofs();
    let mut r = vec![0.0_f64; n * n];
    let phi = TriNDk::new(1);
    let n_loc = phi.n_dofs();
    for e in 0..mesh.n_elements() as u32 {
        let v: Vec<[f64; 2]> = mesh
            .element_nodes(e)
            .iter()
            .map(|&p| {
                let c = mesh.geom_coords_of(p);
                [c[0], c[1]]
            })
            .collect();
        let det = (v[1][0] - v[0][0]) * (v[2][1] - v[0][1])
            - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]);
        let area = det.abs() / 2.0;
        // Reference curls are constant; any reference point works.
        let mut curl_hat = vec![0.0_f64; n_loc];
        phi.eval_curl(&[1.0 / 3.0, 1.0 / 3.0], &mut curl_hat);
        let signs = nd.element_signs(e);
        let dofs = nd.element_dofs(e);
        for i in 0..n_loc {
            let si = signs.get(i).copied().unwrap_or(1.0);
            let czi = curl_hat[i] / det; // physical scalar curl
            for j in 0..n_loc {
                let sj = signs.get(j).copied().unwrap_or(1.0);
                let czj = curl_hat[j] / det;
                r[dofs[i] as usize * n + dofs[j] as usize] += si * sj * area * czi * czj;
            }
        }
    }
    r
}

/// 3-D ND1/[P₀]³ oracle: `A[g_i, g_j] += s_i·s_j·|K|·(c_i·c_j)` with
/// `c_i = J·curl̂_i/detJ` (constant vector on the affine tetrahedron).
fn wg_volume_oracle_tet(mesh: &Mesh<3>, nd: &HCurlSpace<Mesh<3>>) -> Vec<f64> {
    let n = nd.n_dofs();
    let mut r = vec![0.0_f64; n * n];
    let phi = TetNDk::new(1);
    let n_loc = phi.n_dofs();
    for e in 0..mesh.n_elements() as u32 {
        let v: Vec<[f64; 3]> = mesh
            .element_nodes(e)
            .iter()
            .map(|&p| {
                let c = mesh.geom_coords_of(p);
                [c[0], c[1], c[2]]
            })
            .collect();
        let mut jac = nalgebra::DMatrix::<f64>::zeros(3, 3);
        for (c, vc) in v.iter().skip(1).enumerate() {
            for rr in 0..3 {
                jac[(rr, c)] = vc[rr] - v[0][rr];
            }
        }
        let det = jac.determinant();
        let vol = det.abs() / 6.0;
        let mut curl_hat = vec![0.0_f64; n_loc * 3];
        phi.eval_curl(&[0.25, 0.25, 0.25], &mut curl_hat);
        let signs = nd.element_signs(e);
        let dofs = nd.element_dofs(e);
        for i in 0..n_loc {
            let si = signs.get(i).copied().unwrap_or(1.0);
            // Physical curl: J·curl̂/detJ.
            let ci: Vec<f64> = (0..3)
                .map(|rr| (0..3).map(|k| jac[(rr, k)] * curl_hat[i * 3 + k]).sum::<f64>() / det)
                .collect();
            for j in 0..n_loc {
                let sj = signs.get(j).copied().unwrap_or(1.0);
                let cj: Vec<f64> = (0..3)
                    .map(|rr| (0..3).map(|k| jac[(rr, k)] * curl_hat[j * 3 + k]).sum::<f64>() / det)
                    .collect();
                let dot: f64 = (0..3).map(|d| ci[d] * cj[d]).sum();
                r[dofs[i] as usize * n + dofs[j] as usize] += si * sj * vol * dot;
            }
        }
    }
    r
}

// ─── Pin 1a/1b: volume Whitney oracles ───────────────────────────────────────

#[test]
fn d1080_wg_volume_matches_whitney_oracle_tri() {
    let name = "unit_square_tri(3)";
    let mesh = Mesh::<2>::unit_square_tri(3);
    assert!(
        has_reversed_edges(&mesh, 1),
        "{name}: expected at least one reversed edge (pin geometry key)"
    );
    let nd = HCurlSpace::new(mesh.clone(), 1);

    // penalty = 0 ⇒ the face stabilizer contributes nothing; the matrix is
    // exactly the volume weak-curl block.
    let (a, _) = assemble_wg_maxwell(&nd, 3, 0.0, &[]);
    assert_eq!(a.nrows, nd.n_dofs());

    let oracle = wg_volume_oracle_tri(&mesh, &nd);
    let dev = max_dev(&a, &oracle, nd.n_dofs());
    println!("{name}: max |A − WG Whitney oracle| = {dev:.3e}");
    assert!(
        dev < 1e-13,
        "{name}: WG volume block deviates from the Whitney oracle by {dev:.3e} \
         (D1080: per-dof orientation signs must reach C_w, and the curl must be \
         the physical one curl̂/detJ — raw reference curls leave detJ²-scaled, \
         unsigned entries)"
    );
}

#[test]
fn d1080_wg_volume_matches_whitney_oracle_tet() {
    let name = "unit_cube_tet(2)";
    let mesh = Mesh::<3>::unit_cube_tet(2);
    assert!(
        has_reversed_edges(&mesh, 1),
        "{name}: expected at least one reversed edge (pin geometry key)"
    );
    let nd = HCurlSpace::new(mesh.clone(), 1);

    let (a, _) = assemble_wg_maxwell(&nd, 3, 0.0, &[]);
    let oracle = wg_volume_oracle_tet(&mesh, &nd);
    let dev = max_dev(&a, &oracle, nd.n_dofs());
    println!("{name}: max |A − WG Whitney oracle| = {dev:.3e}");
    assert!(
        dev < 1e-13,
        "{name}: WG volume block deviates from the tet Whitney oracle by {dev:.3e} \
         (3-D path: J·curl̂/detJ transform + per-dof signs — the raw reference \
         curl misses J entirely)"
    );
}

// ─── Pin 2: annihilation of a curl-free interpolant ──────────────────────────

#[test]
fn d1080_wg_volume_annihilates_gradient_interpolant() {
    let name = "unit_square_tri(3)";
    let mesh = Mesh::<2>::unit_square_tri(3);
    assert!(
        has_reversed_edges(&mesh, 1),
        "{name}: expected at least one reversed edge (pin geometry key)"
    );
    let nd = HCurlSpace::new(mesh.clone(), 1);

    // E = ∇u with u = 1 + 2x − 3y + xy: the ND1 interpolant (edge
    // circulations) is discretely curl-free element-wise — through the signed
    // dof table the circulations telescope to zero.
    let e_h = nd
        .interpolate_vector(&|x: &[f64]| vec![2.0 + x[1], -3.0 + x[0]])
        .as_slice()
        .to_vec();
    let e_norm: f64 = e_h.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(e_norm > 0.1, "{name}: vanishing interpolant, pin is vacuous");

    let (a, _) = assemble_wg_maxwell(&nd, 3, 0.0, &[]);
    let mut ae = vec![0.0_f64; a.nrows];
    a.spmv(&e_h, &mut ae);
    let norm: f64 = ae.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("{name}: ‖E‖ = {e_norm:.6e}  ‖A·E‖ = {norm:.3e}");
    assert!(
        norm < 1e-12,
        "{name}: curl-grad annihilation broken, ‖A·E‖ = {norm:.3e} \
         (D1080: the weak curl of the interpolant vanishes only when the \
         element matrix carries the same signed dof table the interpolant used)"
    );
}

// ─── Pin 3: face-penalty scatter mirror (straight two-triangle patch) ────────

/// Independent full mirror of `assemble_wg_maxwell` on a straight tri mesh:
/// volume from the Whitney oracle, faces from corner-composed analytic face
/// geometry, **both element blocks of every face signed per element**.
fn wg_full_mirror(mesh: &Mesh<2>, nd: &HCurlSpace<Mesh<2>>, qo: u8, penalty: f64) -> Vec<f64> {
    let n = nd.n_dofs();
    let mut r = wg_volume_oracle_tri(mesh, nd);
    let phi = TriNDk::new(1);
    let n_loc = phi.n_dofs();
    let ref_v = [[0.0_f64, 0.0_f64], [1.0, 0.0], [0.0, 1.0]];

    let add_face = |el: u32, er: u32, fnodes: &[u32], r: &mut Vec<f64>| {
        // Face geometry, straight-edge exact: chord measure, corner-composed
        // physical point → composed element reference point.
        let pe = |p: u32| {
            let c = mesh.geom_coords_of(p);
            [c[0], c[1]]
        };
        let (a0, b0) = (pe(fnodes[0]), pe(fnodes[1]));
        let h = ((b0[0] - a0[0]).powi(2) + (b0[1] - a0[1]).powi(2)).sqrt();
        let alpha = penalty / h.max(1e-14);
        let qf = seg_rule(qo);
        let dofs_l: Vec<usize> = nd.element_dofs(el).iter().map(|&d| d as usize).collect();
        let dofs_r: Vec<usize> = if el != er {
            nd.element_dofs(er).iter().map(|&d| d as usize).collect()
        } else {
            dofs_l.clone()
        };
        for (es, dofs) in [(el, &dofs_l), (er, &dofs_r)] {
            // Compose the face's rule point into this element's reference
            // edge: find the element's local directed edge (k → k+1) through
            // the face nodes (MFEM triangle edges: {0,1},{1,2},{2,0}).
            let en = mesh.element_nodes(es);
            let mut k = usize::MAX;
            let mut fwd = true;
            for t in 0..3 {
                if en[t] == fnodes[0] && en[(t + 1) % 3] == fnodes[1] {
                    k = t;
                    fwd = true;
                    break;
                }
                if en[t] == fnodes[1] && en[(t + 1) % 3] == fnodes[0] {
                    k = t;
                    fwd = false;
                    break;
                }
            }
            assert!(k < 3, "face nodes not on element {es}");
            let signs = nd.element_signs(es);
            for (qi, xi) in qf.points.iter().enumerate() {
                let frac = if fwd { xi[0] } else { 1.0 - xi[0] };
                let eip = [
                    ref_v[k][0] + frac * (ref_v[(k + 1) % 3][0] - ref_v[k][0]),
                    ref_v[k][1] + frac * (ref_v[(k + 1) % 3][1] - ref_v[k][1]),
                ];
                let w = qf.weights[qi] * h;
                let mut pb = vec![0.0_f64; n_loc * 2];
                phi.eval_basis_vec(&eip, &mut pb);
                for i in 0..n_loc {
                    let si = signs.get(i).copied().unwrap_or(1.0);
                    for j in 0..n_loc {
                        let sj = signs.get(j).copied().unwrap_or(1.0);
                        let v = alpha
                            * w
                            * (pb[i * 2] * pb[j * 2] + pb[i * 2 + 1] * pb[j * 2 + 1]);
                        if v.abs() > 1e-30 {
                            r[dofs[i] * n + dofs[j]] += si * sj * v;
                        }
                    }
                }
            }
            if el == er {
                break;
            }
        }
    };

    for f in &InteriorFaceList::build(mesh).faces {
        add_face(f.elem_left, f.elem_right, &f.face_nodes, &mut r);
    }
    let bmap = fem_assembly::dg::dg_base::build_face_elem_map(mesh, 2);
    for bf in mesh.face_iter() {
        if let Some(&el) = bmap.get(&bf) {
            let fnodes: Vec<u32> = mesh.face_nodes(bf).to_vec();
            add_face(el, el, &fnodes, &mut r);
        }
    }
    r
}

#[test]
fn d1080_wg_face_penalty_scatters_signed_blocks() {
    let name = "unit_square_tri(2)";
    // Straight 2×2 tri mesh — coarse enough for the analytic mirror, and it
    // carries reversed H(curl) edges (guarded below; the 2-element diagonal
    // patch does not — its global edge orientations all coincide with the
    // local ones).
    let mesh = Mesh::<2>::unit_square_tri(2);
    assert!(
        has_reversed_edges(&mesh, 1),
        "{name}: expected reversed H(curl) edges (pin geometry key)"
    );
    let nd = HCurlSpace::new(mesh.clone(), 1);

    let (a, _) = assemble_wg_maxwell(&nd, 3, 10.0, &[]);
    let mirror = wg_full_mirror(&mesh, &nd, 3, 10.0);
    let dev = max_dev(&a, &mirror, nd.n_dofs());
    println!("{name}: max |A − signed full mirror| = {dev:.3e}");
    assert!(
        dev < 1e-11,
        "{name}: WG Maxwell deviates from the signed analytic mirror by {dev:.3e} \
         (D1080: the face blocks must be scattered with each element's own \
         signed dof table — MFEM's [vdofs; vdofs2] face scatter — and the \
         volume with the physical-curl Whitney oracle)"
    );
}
