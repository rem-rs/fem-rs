//! D516: the rational weights of a NURBS mesh — where MFEM takes them from,
//! and the hard numbers that keep fem-rs from silently falling back to unit
//! weights.
//!
//! The reference numbers are MFEM **4.10** probes (`tmp/r55/d516_probe.cpp`,
//! `tmp/r55/d516_sweep.cpp`, `tmp/r55/d516_ref.cpp`; dumps in
//! `tmp/r55/d516_ref_ball.txt` and `tmp/r55/d516_sweep_mfem410.txt`), run as
//! `Mesh(mesh, 1, 1)` against `/home/quan/mfem410_ser/libmfem.a`.
//!
//! Three facts are pinned:
//!
//! 1. `NURBSExtension::Load` fills the weights with
//!    `weights.Load(input, GetNDof())`, and `Vector::Load(std::istream &, int)`
//!    reads **exactly** that many values — so the weights are the *first*
//!    `GetNDof()` numbers after the keyword and nothing else.  Every shipped
//!    `data/*nurbs*.mesh` that has a `weights` section carries exactly
//!    `GetNDof()` of them (table below), so no truncation happens in practice;
//!    the rule matters for the section's *end* (a following block is not part
//!    of it) and for short sections, which are rejected loudly.
//! 2. The `patches` flavour has **no** `weights` section: `NURBSExtension::Load`
//!    guards it with `if (patches.Size() == 0)`, and the weights come from the
//!    patch control points instead (`Mesh::ReadNURBSMesh` →
//!    `SetCoordsFromPatches` → `Set{1,2,3}DSolutionVector`,
//!    `weights(l) = patch(i,j,k,vdim)`).
//! 3. `FiniteElementSpace(mesh, fec)` with `NURBSext_ == NULL` — what
//!    `nurbs_patch_ex1` builds — takes `NURBSext = mesh->NURBSext`, so the
//!    analysis extension keeps the mesh's rational weights.  For
//!    `ball-nurbs.mesh` that is 517 values of which **360 are non-unit**, and
//!    dropping them perturbs the element-1 shape functions by up to
//!    `5.09e-3` (the reference `EL1_SHAPE_CENTER` values below).

use fem_space::nurbs_extension::NurbsExtension;
use fem_space::nurbs_fe_space::NurbsFESpace;

const BALL: &str = include_str!("../../../data/ball-nurbs.mesh");
const SQUARE: &str = include_str!("../../../data/square-nurbs.mesh");

fn ball() -> NurbsFESpace {
    // `NurbsFESpace::from_mesh_str` mirrors `nurbs_ex1`'s
    // `NURBSExtension(mesh->NURBSext, order)` (unit weights);
    // `nurbs_patch_ex1` builds `FiniteElementSpace(&mesh, fec)` instead, which
    // inherits the mesh's rational weights.
    NurbsFESpace::from_mesh_isoparametric_str(BALL, 0).expect("ball space")
}

#[test]
fn uniform_refinement_reproduces_mfems_refined_weights() {
    // MFEM 4.10: `Mesh::UniformRefinement` re-derives the control net with
    // `NURBSPatch::KnotInsert` on the *homogeneous* control points, so the
    // refined weights are A5.5 blends of the original ones.  Reference dumps
    // `tmp/r55/d516_refw2_ball_r{1,2}.txt` (`tmp/r55/d516_refw2.cpp`): every
    // element's DOF row plus the weights in that order, for all 56 / 448
    // elements.  The DOF numbering, the count of non-unit weights, the weights
    // of element 8 (the first element in the outer shell, i.e. the first one
    // that is not entirely unit-weighted) and two weighted sums over *all*
    // element weights are pinned.
    //
    // `S1 = Σ_e Σ_i w[e][i]·(1 + i mod 7)` and
    // `S2 = Σ_e Σ_i w[e][i]²·(1 + e mod 5)`, summed element-major and
    // index-ascending as in the awk that produced them; a single wrong entry
    // anywhere in the refined net moves both.
    let mfem_el8: [f64; 10] = [
        1.0,
        0.94560560180420006,
        0.87516398000246998,
        0.87516398000246998,
        0.94560560180420006,
        0.94560560180420006,
        0.88617048340899007,
        0.8078131649910375,
        0.8078131649910375,
        0.88617048340899007,
    ];
    for (ref_levels, n_dofs, nonunit, s1, s2) in [
        (1usize, 976usize, 720usize, 25998.050249320342_f64, 18178.701633952915_f64),
        (2, 2584, 2016, 208032.72783975414, 147023.8443226385),
    ] {
        let mut ext = NurbsExtension::from_mesh_str(BALL).expect("ball");
        for _ in 0..ref_levels {
            ext.uniform_refinement(2).expect("uniform refinement");
        }
        assert_eq!(ext.n_dofs(), n_dofs, "r{ref_levels}: GetNDof");
        assert_eq!(ext.n_elements(), if ref_levels == 1 { 56 } else { 448 }, "r{ref_levels}: NE");
        let w = ext.weights();
        assert_eq!(w.len(), n_dofs, "r{ref_levels}: weights cover the control net");
        assert_eq!(
            w.iter().filter(|&&v| v != 1.0).count(),
            nonunit,
            "r{ref_levels}: non-unit weights"
        );
        if ref_levels == 1 {
            let el8 = ext.element_dofs(8);
            assert_eq!(el8.len(), 125);
            let got: Vec<f64> = el8.iter().map(|&d| w[d]).collect();
            assert_eq!(&got[..10], &mfem_el8[..], "r1: element 8 weights");
        }
        let (mut a, mut b) = (0.0_f64, 0.0_f64);
        for e in 0..ext.n_elements() {
            for (i, &d) in ext.element_dofs(e).iter().enumerate() {
                let v = w[d];
                a += v * (1 + (i % 7)) as f64;
                b += v * v * (1 + e % 5) as f64;
            }
        }
        assert_eq!(a, s1, "r{ref_levels}: S1");
        assert_eq!(b, s2, "r{ref_levels}: S2");
    }
}

#[test]
fn nurbs_ex1_style_construction_resets_the_analysis_weights() {
    // The other MFEM construction: `NURBSExtension(mesh->NURBSext, order)`
    // (`mesh/nurbs.cpp:2998-3001`) sets every weight to one, so the analysis
    // basis is polynomial over the rational geometry.
    let ext = NurbsExtension::from_mesh_str(BALL).expect("ball");
    let orders: Vec<usize> = (0..ext.n_knot_vectors())
        .map(|i| ext.knot_vector(i).order())
        .collect();
    let space = NurbsFESpace::from_mesh_str(BALL, 0, &orders).expect("nurbs_ex1-style space");
    assert!(space.extension().weights().iter().all(|&w| w == 1.0));
    assert!(space.element_weights(1).iter().all(|&w| w == 1.0));
    // ... which is exactly the basis the unit-weight fallback used to produce
    // for the isoparametric space by accident.
    let mut shape = vec![0.0_f64; 125];
    space.fe_shape(1, &[0.5, 0.5, 0.5], &mut shape);
    let bern = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    for (flat, &v) in shape.iter().enumerate() {
        let proj = bern[flat % 5] * bern[(flat / 5) % 5] * bern[flat / 25];
        assert!((v - proj).abs() < 1e-16, "element 1 shape[{flat}]");
    }
}

/// `(mesh file, GetNDof, non-unit weights, first non-unit, last non-unit)` —
/// MFEM 4.10, `NurbsExtension::GetNDof()` and `GetWeights()`.
///
/// `nc-nurbs3d.mesh` and `nc3-nurbs.mesh` are the `MFEM NURBS NC-patch mesh
/// v1.0` flavour (`NCNURBSExtension`), which `NurbsExtension` does not port —
/// its banner is rejected up front (see the module's "what is not reproduced"
/// list), so they are not in the table.
const MFEM_WEIGHTS: &[(&str, usize, usize, i64, i64)] = &[
    ("ball-nurbs.mesh", 517, 360, 16, 516),
    ("beam-hex-nurbs.mesh", 36, 0, -1, -1),
    ("beam-quad-nurbs-sf.mesh", 18, 0, -1, -1),
    ("beam-quad-nurbs.mesh", 18, 0, -1, -1),
    ("cube-nurbs.mesh", 8, 0, -1, -1),
    ("disc-nurbs.mesh", 25, 8, 8, 24),
    ("pipe-nurbs-2d.mesh", 9, 3, 4, 8),
    ("pipe-nurbs-log.mesh", 120, 72, 16, 119),
    ("pipe-nurbs.mesh", 120, 72, 16, 119),
    ("segment-nurbs.mesh", 2, 0, -1, -1),
    ("square-disc-nurbs-patch.mesh", 38, 12, 19, 36),
    ("square-disc-nurbs.mesh", 24, 8, 9, 23),
    ("square-nurbs-pw.mesh", 4, 0, -1, -1),
    ("square-nurbs.mesh", 4, 0, -1, -1),
];

#[test]
fn weights_match_mfem_for_every_shipped_nurbs_mesh() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(MFEM_WEIGHTS.len(), 14, "the table must cover every mesh below");
    for &(mesh, n_dofs, nonunit, first, last) in MFEM_WEIGHTS {
        let path = root.join("data").join(mesh);
        let ext = NurbsExtension::from_mesh_file(&path)
            .unwrap_or_else(|e| panic!("{mesh}: {e}"));
        assert_eq!(ext.n_dofs(), n_dofs, "{mesh}: GetNDof");
        let w = ext.weights();
        assert_eq!(w.len(), n_dofs, "{mesh}: weights length");
        let idx: Vec<usize> = w.iter().enumerate().filter(|(_, &v)| v != 1.0).map(|(i, _)| i).collect();
        assert_eq!(idx.len(), nonunit, "{mesh}: non-unit weights");
        assert_eq!(
            (idx.first().map(|&i| i as i64).unwrap_or(-1), idx.last().map(|&i| i as i64).unwrap_or(-1)),
            (first, last),
            "{mesh}: first/last non-unit weight (MFEM 4.10)"
        );
    }
}

#[test]
fn ball_weights_are_the_first_get_ndof_values_of_the_section() {
    let ext = NurbsExtension::from_mesh_str(BALL).expect("ball");
    assert_eq!(ext.n_dofs(), 517);
    let w = ext.weights();
    assert_eq!(w.len(), 517);

    // MFEM `W 517` dump: the first 16 weights are one (the inner-cube patch's
    // controls), `w[16]` is the first non-unit value, `w[516]` the last.
    for (i, &v) in w.iter().enumerate().take(16) {
        assert_eq!(v, 1.0, "w[{i}]");
    }
    assert_eq!(w[16], 0.8912112036084);
    assert_eq!(w[17], 0.85911675639653995);
    assert_eq!(w[18], 0.8912112036084);
    assert_eq!(w[515], 0.92966629338500995);
    assert_eq!(w[516], 0.94056488160479002);
    let nonunit = w.iter().filter(|&&v| v != 1.0).count();
    assert_eq!(nonunit, 360, "non-unit weights in the mesh extension");
}

#[test]
fn weights_section_takes_the_first_get_ndof_values_and_rejects_a_short_one() {
    // A longer section is harmless: `Vector::Load(in, GetNDof())` consumes the
    // first `GetNDof()` values and leaves the rest in the stream (this is what
    // `ball-nurbs.mesh`'s trailing `FiniteElementSpace` block looked like to a
    // naive token count).
    let long = SQUARE.replace(
        "\n\nFiniteElementSpace",
        "\n9 9 9\n\nFiniteElementSpace",
    );
    assert_ne!(long, SQUARE, "the fixture must have been edited");
    let base = NurbsExtension::from_mesh_str(SQUARE).expect("square");
    let with_extra = NurbsExtension::from_mesh_str(&long).expect("extra values are ignored");
    assert_eq!(base.n_dofs(), 4);
    assert_eq!(with_extra.weights(), base.weights());

    // A section shorter than `GetNDof()` cannot be reproduced: MFEM would keep
    // reading and consume the next block's tokens.  Loud error, no fallback.
    let short = SQUARE.replace(
        "weights\n1\n1\n1\n1\n\nFiniteElementSpace",
        "weights\n1\n1\n1\n\nFiniteElementSpace",
    );
    assert_ne!(short, SQUARE, "the fixture must have been edited");
    let err = NurbsExtension::from_mesh_str(&short).expect_err("short section");
    assert!(
        err.contains("reads 4 values, the section holds 3"),
        "unexpected message: {err}"
    );
}

#[test]
fn patches_flavour_weights_come_from_the_patch_control_points() {
    // `square-disc-nurbs-patch.mesh` has no `weights` section at all; MFEM's
    // weights are the homogeneous last component of the patch control points
    // (`Set3DSolutionVector`/`Set2DSolutionVector`), 12 of them non-unit.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/square-disc-nurbs-patch.mesh");
    let text = std::fs::read_to_string(&path).expect("mesh file");
    assert!(
        !text.lines().any(|l| l.trim() == "weights"),
        "the fixture is expected to have no weights section"
    );
    let ext = NurbsExtension::from_mesh_str(&text).expect("patches flavour");
    assert_eq!(ext.n_dofs(), 38);
    let idx: Vec<usize> = ext
        .weights()
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 1.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(idx.len(), 12, "non-unit weights (MFEM 4.10)");
    assert_eq!((idx[0], idx[idx.len() - 1]), (19, 36), "first/last non-unit weight");
}

#[test]
fn ball_analysis_space_is_rational() {
    // MFEM 4.10, `FiniteElementSpace fespace(&mesh, mesh.GetNodes()->OwnFEC())`
    // on `ball-nurbs.mesh`: element 1's `NURBSFiniteElement::Weights()`
    // (`NURBSExtension::LoadFE` → `weights.GetSubVector(el_dofs)`).
    let space = ball();
    let w1 = space.element_weights(1);
    assert_eq!(w1.len(), 125);
    let mfem_el1_prefix: [f64; 15] = [
        1.0,
        0.8912112036084,
        0.85911675639653995,
        0.8912112036084,
        1.0,
        0.8912112036084,
        0.76225952641915995,
        0.71866517354005,
        0.76225952641915995,
        0.8912112036084,
        0.85911675639653995,
        0.71866517354005,
        0.67127243159192995,
        0.71866517354005,
        0.85911675639653995,
    ];
    assert_eq!(&w1[..15], &mfem_el1_prefix[..], "element 1 weights[0..15]");
    assert_eq!(
        w1.iter().filter(|&&v| v != 1.0).count(),
        84,
        "element 1 non-unit weights (MFEM 4.10 EL1_WEIGHTS)"
    );
    // The inner-cube patch is entirely unit-weighted.
    assert!(space.element_weights(0).iter().all(|&v| v == 1.0));

    // MFEM's `EL1_SHAPE_CENTER` = `NURBS3DFiniteElement::CalcShape` at
    // `(0.5, 0.5, 0.5)`, i.e. the rational values after the weights.
    let mut shape = vec![0.0_f64; 125];
    space.fe_shape(1, &[0.5, 0.5, 0.5], &mut shape);
    let mfem: [f64; 5] = [
        0.00027711280891722968,
        0.00098786415988171526,
        0.0014284335453174273,
        0.00098786415988171526,
        0.00027711280891722968,
    ];
    for (i, &want) in mfem.iter().enumerate() {
        assert_eq!(shape[i], want, "element 1 shape[{i}] at the centre");
    }
    let sum: f64 = shape.iter().sum();
    assert!((sum - 1.0).abs() < 1e-15, "rational shapes sum to one: {sum}");

    // What the unit-weight fallback used to produce: the Bernstein tensor of
    // the single-span order-4 basis at the centre, `[1,4,6,4,1]/16` per
    // direction.  MFEM's (`EL1_SHAPE_CENTER`) largest deviation from it is
    // 5.0902244597286553e-3 (local index 38) — the percent-level error the
    // earlier `unit_weights()` reset silently introduced.
    let bern = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let mut worst = 0.0_f64;
    for (flat, &v) in shape.iter().enumerate() {
        let proj = bern[flat % 5] * bern[(flat / 5) % 5] * bern[flat / 25];
        worst = worst.max((v - proj).abs());
    }
    assert!(
        (worst - 5.0902244597286553e-3).abs() < 1e-15,
        "worst |rational - polynomial| at the centre: {worst} (MFEM 5.0902244597286553e-3)"
    );
    assert!(
        worst > 1e-3,
        "the deviation must be material, not a rounding artefact: {worst}"
    );
}
