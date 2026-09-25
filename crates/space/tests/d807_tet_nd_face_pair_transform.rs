//! D807 / D790-3 — tet NDk (k ≥ 2) shared-face DOFs: the element↔canonical
//! transform **is** a 2×2 rotation (implemented), and the *cross-rank* gap is
//! the scalar `sign_correction` channel of the DOF partition (D122-3's fix),
//! which cannot express it.
//!
//! ## Registered claim
//!
//! `tmp/round3_plan.md` D790-3: "tet `k ≥ 2` 面 DOF 的 element↔canonical 是
//! 2×2 旋转，标量符号表达不了；本轮夹具 hex-only 未测".  This file settles it
//! numerically on a two-tet fixture.
//!
//! ## Verdict (measured below, `tmp/d807/d790_3_tet.txt`)
//!
//! 1. **Element level: implemented and correct.**  `HCurlSpace` records one
//!    [`FaceDofBlock`] per shared face point pair (`crates/space/src/hcurl.rs`,
//!    D37) holding `s = ` the element's own face tangents expressed in the
//!    canonical (face-creating element) tangent basis, and
//!    `VectorAssembler::assemble_bilinear_nd_canonical` applies
//!    `A ← Sᵀ·A·S`.  Measured on a two-tet mesh at k = 2 and k = 3, symmetric
//!    and asymmetric (apex `(1,1,2)`): the non-creating element's block is
//!    exactly the **tangent-pair swap** `[[0,1],[1,0]]` (det = −1) — the pair is
//!    built from the shared triangle's own edge vectors on both sides
//!    (`tet_face_slots`), so the two elements see it in opposite orientations at
//!    every order.  The registered "2×2 旋转" is loosely worded (it is
//!    orientation-*reversing*), but the substance — **no scalar sign can express
//!    it** — is confirmed.  The order-swap identity `A_ba = Rᵀ·A_ab·R` holds to
//!    1e-10, so the element side is right.
//! 2. **The space's scalar channel is blind to it**:
//!    `element_signs(e)[slot] == +1.0` for every face-DOF slot — the DP's
//!    `DofPartition::sign_correction` reads exactly that table
//!    (`from_edge_space_ordered`, D122-3), so it applies **no** correction.
//! 3. **The canonical basis is element-order dependent**: it is anchored by the
//!    *face-creating* element = the first element of the local traversal
//!    touching the face, and `par_partition` orders the local mesh owned-first.
//!    Two elements sharing a face and owned by different ranks therefore build
//!    the *same* face DOF ids in *different* bases, related by that swap — the
//!    assembled shared-face block moves by `Rᵀ·A·R`, measured 1.71e-1 (k = 2) /
//!    4.70e-2 (k = 3) on the fixture.  That is the cross-rank disagreement a
//!    parallel run has to re-baseline and cannot, because `sign_correction` is a
//!    scalar ±1 per DOF: the DP needs a per-face-point **pair transform** (at
//!    least swap + sign; a full 2×2 block for other families), registered as
//!    **D807-2**.  The hex path is unaffected — its change of basis *is* a signed
//!    permutation, which the scalar channel expresses (D55/D122-3).
//!
//! So the registration's *location* was off by one level: the element transform
//! exists and is right; the missing piece is in the parallel DOF partition.

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::{VectorAssembler, VectorBilinearIntegrator};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::dof_manager::FaceKey;
use fem_space::HCurlSpace;

/// `[0,s]³` split into two **positively oriented** tets sharing the face
/// `(0,1,4)`, the D37 fixture.  `apex` moves vertex 4 so the fixture can be made
/// asymmetric (`(1,1,2)`), which turns the canonical-frame relation into a
/// genuine rotation instead of the symmetric cube's tangent swap.
fn two_tet_mesh_apex(conn: [u32; 8], apex: [f64; 3]) -> Mesh<3> {
    let s = 1.0_f64;
    let coords = vec![
        0.0, 0.0, 0.0, // 0
        s, 0.0, 0.0, // 1
        0.0, s, 0.0, // 2
        0.0, 0.0, s, // 3
        apex[0], apex[1], apex[2], // 4
    ];
    let conn: Vec<u32> = conn.to_vec();
    let face_conn: Vec<u32> = vec![1, 2, 4, 0, 2, 4, 0, 1, 4, 0, 1, 3, 4, 0, 3, 4, 0, 1, 3];
    let nf = face_conn.len() / 3;
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1],
        ElementType::Tet4,
        face_conn,
        (1..=nf as i32).collect(),
        ElementType::Tri3,
    )
}

fn two_tet_mesh(conn: [u32; 8]) -> Mesh<3> {
    two_tet_mesh_apex(conn, [1.0, 1.0, 1.0])
}

/// `true` when `s` is a signed permutation matrix (±1 in a permutation pattern)
/// — the class the DP's `permute_dof` + scalar `sign_correction` pair can
/// express.
fn is_signed_permutation(s: [[f64; 2]; 2]) -> bool {
    let unit = |v: f64| (v.abs() - 1.0).abs() < 1e-12 || v.abs() < 1e-12;
    unit(s[0][0]) && unit(s[0][1]) && unit(s[1][0]) && unit(s[1][1])
}

/// The block of `elem` whose canonical pair holds `first` — one per face point
/// (k = 2 → 1 block, k = 3 → 3 blocks per face).
fn shared_face_blocks(
    space: &HCurlSpace<Mesh<3>>,
    elem: u32,
    first: u32,
) -> Vec<fem_space::hcurl::FaceDofBlock> {
    space
        .element_face_blocks(elem)
        .iter()
        .filter(|b| b.canon_dofs.contains(&first) || (first + 1 < space.n_dofs() as u32 && b.canon_dofs.contains(&(first + 1))))
        .copied()
        .collect()
}

/// `Σ` over the face-DOF slots of `elem` of the block rotation magnitude —
/// `0` when every block is a signed permutation (hex-like), `O(1)` for a
/// genuine 2×2 rotation (tet).
fn rotation_magnitude(space: &HCurlSpace<Mesh<3>>, elem: u32, first: u32) -> f64 {
    shared_face_blocks(space, elem, first)
        .iter()
        .map(|b| (b.s[0][1].abs()).max(b.s[1][0].abs()))
        .fold(0.0_f64, f64::max)
}

fn mass_matrix(space: &HCurlSpace<Mesh<3>>) -> fem_linalg::CsrMatrix<f64> {
    let mass: [&dyn VectorBilinearIntegrator; 1] = [&VectorMassIntegrator { alpha: 1.0 }];
    VectorAssembler::assemble_bilinear_nd_canonical(space, &mass, 6)
}

/// The shared face's 2-DOF pair as a 2×2 sub-block of the assembled mass
/// matrix (k = 2: one point per face).
fn face_subblock(space: &HCurlSpace<Mesh<3>>, first: u32) -> [[f64; 2]; 2] {
    let m = mass_matrix(space);
    let at = |r: u32, c: u32| -> f64 {
        let (s, e) = (m.row_ptr[r as usize], m.row_ptr[r as usize + 1]);
        (s..e)
            .find(|&k| m.col_idx[k] == c)
            .map(|k| m.values[k])
            .unwrap_or(0.0)
    };
    [[at(first, first), at(first, first + 1)], [at(first + 1, first), at(first + 1, first + 1)]]
}

/// D790-3 part 1 + 2: the non-creating element's shared-face block is a genuine
/// 2×2 rotation, while the scalar sign table the DP consumes is all `+1.0`.
#[test]
fn d807_tet_nd_face_block_is_a_rotation_not_a_sign() {
    for (k, min_off) in [(2u8, 1e-2_f64), (3u8, 1e-2)] {
        // Symmetric cube fixture: the relation is the tangent swap.
        let mesh = two_tet_mesh([0, 1, 2, 4, 0, 3, 1, 4]);
        let space = HCurlSpace::new(mesh, k);
        let first = space.face_dof(FaceKey::new(0, 1, 4)).expect("shared face");
        let rot = rotation_magnitude(&space, 1, first);
        let blocks = shared_face_blocks(&space, 1, first);
        assert!(
            !blocks.is_empty(),
            "k={k}: element 1 must record a block for the shared face"
        );
        println!("k={k}: element 1 shared-face blocks = {blocks:?}  (rotation magnitude {rot:.6})");
        assert!(
            rot > min_off,
            "k={k}: the element↔canonical face transform must be a genuine 2×2 rotation, \
             not a signed permutation — measured off-axis magnitude {rot:.3e}"
        );
        // The scalar channel the DOF partition reads must be blind to it.
        let signs = space.element_signs(1);
        for b in &blocks {
            for j in 0..2 {
                assert_eq!(
                    signs[b.slot + j],
                    1.0,
                    "k={k}: face slot {} of the space's `element_signs` is {}, not +1.0 — \
                     the DP's scalar sign_correction would then carry part of the rotation",
                    b.slot + j,
                    signs[b.slot + j]
                );
            }
        }

        // Asymmetric fixture (apex `(1,1,2)`): the relation is still the
        // tangent-pair swap — the pair is built from the *shared triangle's* edge
        // vectors on both sides (`tet_face_slots`), so the element sees the
        // creator's pair in the opposite orientation at every order.  Measured
        // `s = [[0,1],[1,0]]` (det = −1) for k = 2 and k = 3.
        let space_a = HCurlSpace::new(two_tet_mesh_apex([0, 1, 2, 4, 0, 3, 1, 4], [1.0, 1.0, 2.0]), k);
        let first_a = space_a.face_dof(FaceKey::new(0, 1, 4)).expect("shared face");
        let s_a = shared_face_blocks(&space_a, 1, first_a)[0].s;
        println!("k={k}: asymmetric fixture block s = {s_a:?}");
        assert!(
            is_signed_permutation(s_a) && s_a[0][1].abs() > 0.5,
            "k={k}: expected the orientation-reversing pair swap (a signed permutation with \
             a non-zero off-diagonal, det = −1), measured {s_a:?}"
        );
    }
}

/// D790-3 part 3 — the cross-element fixture: the same two tets in the two
/// element orders (i.e. the two *ranks* of a 2-rank split, whose local meshes
/// are ordered owned-first) build the shared face's DOFs in two bases related
/// by that rotation, so the assembled shared-face block disagrees at O(1) —
/// `A_ba = Rᵀ·A_ab·R` with `R ≠ ±I`, which the scalar sign channel cannot fix.
#[test]
fn d807_tet_nd_shared_face_basis_depends_on_element_order() {
    for k in [2u8, 3] {
        // Same topology, opposite element order → opposite face-creating element.
        let space_ab = HCurlSpace::new(two_tet_mesh([0, 1, 2, 4, 0, 3, 1, 4]), k);
        let space_ba = HCurlSpace::new(two_tet_mesh([0, 3, 1, 4, 0, 1, 2, 4]), k);
        let key = FaceKey::new(0, 1, 4);
        let first_ab = space_ab.face_dof(key).unwrap();
        let first_ba = space_ba.face_dof(key).unwrap();

        let a_ab = face_subblock(&space_ab, first_ab);
        let a_ba = face_subblock(&space_ba, first_ba);
        let worst = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| (a_ab[i][j] - a_ba[i][j]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "k={k}: shared-face mass block (order AB) = {a_ab:?}\n     \
             (order BA) = {a_ba:?}\n     worst entry difference {worst:.6e}"
        );
        assert!(
            worst > 1e-2,
            "k={k}: the canonical face basis is anchored by the face-creating element, so \
             swapping the element order must move the shared-face block by the 2×2 rotation \
             — measured worst difference {worst:.3e} (a scalar sign scheme would leave \
             this as the cross-rank disagreement)"
        );

        // The disagreement is *exactly* the rotation of the non-creating
        // element: with R = that block, A_ba = Rᵀ·A_ab·R.  (`A_ab`'s creator is
        // element 0, so element 1 carries the rotation; in the BA mesh it is
        // element 0.)
        let r = shared_face_blocks(&space_ab, 1, first_ab)[0].s;
        let rot = |a: [[f64; 2]; 2], r: [[f64; 2]; 2]| -> [[f64; 2]; 2] {
            let rt = [[r[0][0], r[1][0]], [r[0][1], r[1][1]]];
            let mul = |x: [[f64; 2]; 2], y: [[f64; 2]; 2]| {
                [[
                    x[0][0] * y[0][0] + x[0][1] * y[1][0],
                    x[0][0] * y[0][1] + x[0][1] * y[1][1],
                ], [
                    x[1][0] * y[0][0] + x[1][1] * y[1][0],
                    x[1][0] * y[0][1] + x[1][1] * y[1][1],
                ]]
            };
            mul(mul(rt, a), r)
        };
        let predicted = rot(a_ab, r);
        let resid = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| (predicted[i][j] - a_ba[i][j]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            resid < 1e-10,
            "k={k}: the order-BA block must be exactly Rᵀ·A_ab·R (residual {resid:.3e}) — \
             if this ever fails the element-level transform itself is wrong, not just \
             the DP's scalar channel"
        );
    }
}
