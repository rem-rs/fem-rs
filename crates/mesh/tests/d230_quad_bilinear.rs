//! D230 — `ElementTransformation::from_simplex` on Quad4 must be the true
//! bilinear isoparametric map (MFEM `IsoparametricTransformation`,
//! `Geometry::SQUARE`), not the first-3-node affine triangle map.
//!
//! Round 39 (D160) had to work around exactly this inside the `mandel` /
//! `mondrian` toys: with the affine map, the sampled points leave the element
//! and the material-based refinement marking came out **429** where MFEM's
//! bilinear sampling marks **410** elements (C++ `mandel -no-vis` iteration 1,
//! `tmp/d160/mandel_marks.txt`).  This suite reproduces that scenario at the
//! library level:
//!
//! - the library `ElementTransformation::from_simplex` path (bilinear) must
//!   reproduce the MFEM truth **410** on the iteration-1 mesh (uniform 32×32
//!   quad grid = uniformly-refined `inline-quad.mesh`, sd = 2);
//! - the pre-fix behavior (first-3-node affine map, inlined here as witness)
//!   must still produce the historical **429**;
//! - on a warped quad the bilinear map must differ from the affine witness
//!   and match the closed-form bilinear evaluation.

use fem_mesh::transformation::ElementTransformation;
use fem_mesh::Mesh;

/// The `mandel` material function (Mandelbrot escape count), replicated from
/// `miniapps/toys/mandel.rs` / C++ `mandel.cpp`.
fn mandel_material(p: &[f64], pmin: &[f64], pmax: &[f64]) -> i32 {
    let mut pn = vec![0.0_f64; p.len()];
    for i in 0..p.len() {
        pn[i] = (p[i] - pmin[i]) / (pmax[i] - pmin[i]);
    }
    pn[0] -= 0.1;

    let col = pn[0] * 1080.0;
    let row = pn[1] * 1080.0;
    let (width, height) = (1080.0_f64, 1080.0_f64);
    let c_re = (col - width / 2.0) * 4.0 / width;
    let c_im = (row - height / 2.0) * 4.0 / width;
    let (mut x, mut y) = (0.0_f64, 0.0_f64);
    let mut iteration = 0;
    let maxit = 10000;
    while x * x + y * y <= 4.0 && iteration < maxit {
        let x_new = x * x - y * y + c_re;
        y = 2.0 * x * y + c_im;
        x = x_new;
        iteration += 1;
    }
    if iteration < maxit { iteration } else { -1 }
}

/// The mandel marking rule: an element is marked when its sampled materials
/// are not all equal (C++ `mandel.cpp`, matsum prefix criterion).
fn is_marked(mats: &[i32]) -> bool {
    let mut matsum = 0_i64;
    let mut refine = false;
    for (j, &m) in mats.iter().enumerate() {
        matsum += m as i64;
        if matsum != m as i64 * (j as i64 + 1) {
            refine = true;
        }
    }
    refine
}

/// sd = 2 sample grid: the (sd+1)² geometry-refiner points (x fastest).
fn sample_grid(sd: usize) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity((sd + 1) * (sd + 1));
    for j in 0..=sd {
        for i in 0..=sd {
            pts.push([i as f64 / sd as f64, j as f64 / sd as f64]);
        }
    }
    pts
}

/// The pre-D230 library behavior on Quad4: affine map through the first
/// `dim + 1 = 3` nodes (`from_simplex_nodes` old code path), inlined here as
/// the historical witness.
fn transform_affine3(mesh: &Mesh<2>, e: u32, s: f64, t: f64) -> [f64; 2] {
    let ns = mesh.elem_nodes(e);
    let x0 = mesh.coords_of(ns[0]);
    let a = mesh.coords_of(ns[1]);
    let b = mesh.coords_of(ns[2]);
    [x0[0] + s * (a[0] - x0[0]) + t * (b[0] - x0[0]),
     x0[1] + s * (a[1] - x0[1]) + t * (b[1] - x0[1])]
}

fn count_marked(mesh: &Mesh<2>, transform: &dyn Fn(u32, f64, f64) -> [f64; 2]) -> usize {
    let sd = 2_usize;
    let (pmin, pmax) = mesh.bounding_box();
    let pmin = [pmin[0], pmin[1]];
    let pmax = [pmax[0], pmax[1]];
    let samples = sample_grid(sd);
    let mut marked = 0_usize;
    for e in 0..mesh.n_elems() as u32 {
        let mats: Vec<i32> = samples
            .iter()
            .map(|sp| {
                let [x, y] = transform(e, sp[0], sp[1]);
                mandel_material(&[x, y], &pmin, &pmax)
            })
            .collect();
        if is_marked(&mats) {
            marked += 1;
        }
    }
    marked
}

/// The iteration-1 mandel mesh: `inline-quad.mesh` (4×4 unit-square quads)
/// after 3 uniform refinements = the uniform 32×32 quad grid (marking is a
/// per-element geometric criterion, so element order is irrelevant).
fn mandel_iteration1_mesh() -> Mesh<2> {
    Mesh::<2>::make_cartesian_2d(32, 32, 1.0, 1.0)
}

#[test]
fn d230_library_bilinear_marks_410_like_mfem() {
    let mesh = mandel_iteration1_mesh();
    let marked = count_marked(&mesh, &|e, s, t| {
        let tr = ElementTransformation::from_simplex(&mesh, e);
        let p = tr.map_to_physical(&[s, t]);
        [p[0], p[1]]
    });
    assert_eq!(marked, 410, "MFEM mandel iteration-1 marking (C++ truth)");
}

#[test]
fn d230_prefix_affine_witness_marks_429() {
    // Falsification witness: the same mesh, sample grid and material, but the
    // *pre-fix* affine-through-3-nodes transform must reproduce the
    // historical (wrong) 429.
    let mesh = mandel_iteration1_mesh();
    let marked = count_marked(&mesh, &|e, s, t| transform_affine3(&mesh, e, s, t));
    assert_eq!(marked, 429, "pre-fix affine sampling (historical witness)");
}

#[test]
fn d230_warped_quad_bilinear_truth() {
    // A genuinely warped quad: the bilinear map differs from the affine
    // first-3-node map, and `from_simplex` must follow the bilinear truth.
    // Corner slots follow the mesh element vertex order
    // [v0, v1, v2, v3] = [(0,0), (1,0), (1,1), (0,1)] of the reference square.
    let corners = [[0.0_f64, 0.0], [2.0, 0.1], [1.9, 1.3], [0.2, 0.9]];
    let mut mesh = Mesh::<2>::make_cartesian_2d(1, 1, 1.0, 1.0);
    // make_cartesian_2d numbers the unit square's nodes 0,1,2,3 as
    // (0,0),(1,0),(0,1),(1,1); place the warped corners accordingly.
    mesh.coords = [corners[0], corners[1], corners[3], corners[2]]
        .iter()
        .flat_map(|c| [c[0], c[1]])
        .collect();
    // The element's vertices in its own order, read back from the mesh.
    let quad: [[f64; 2]; 4] = std::array::from_fn(|k| {
        let c = mesh.coords_of(mesh.elem_nodes(0)[k]);
        [c[0], c[1]]
    });
    assert_eq!(quad, corners, "element vertex order");
    let tr = ElementTransformation::from_simplex(&mesh, 0);

    // Closed-form bilinear map on [0, 1]^2 (MFEM SQUARE, unit-weight basis).
    let bilinear = |s: f64, t: f64| -> [f64; 2] {
        let phi = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
        let mut x = [0.0_f64; 2];
        for (k, &w) in phi.iter().enumerate() {
            x[0] += w * quad[k][0];
            x[1] += w * quad[k][1];
        }
        x
    };

    for &(s, t) in &[(0.25_f64, 0.5_f64), (0.5, 0.25), (0.75, 0.75), (0.5, 0.5)] {
        let got = tr.map_to_physical(&[s, t]);
        let want = bilinear(s, t);
        assert!(
            (got[0] - want[0]).abs() < 1e-15 && (got[1] - want[1]).abs() < 1e-15,
            "({s},{t}): got {got:?}, want {want:?}"
        );
        // On a warped quad the pre-fix affine map differs (falsification).
        let old = transform_affine3(&mesh, 0, s, t);
        assert!(
            (old[0] - want[0]).abs() > 1e-6 || (old[1] - want[1]).abs() > 1e-6,
            "affine witness unexpectedly equals the bilinear map at ({s},{t})"
        );
    }

    // The four corners are exact (bilinear interpolates its vertices).
    for (s, t, k) in [(0.0_f64, 0.0_f64, 0_usize), (1.0, 0.0, 1), (1.0, 1.0, 2), (0.0, 1.0, 3)] {
        let p = tr.map_to_physical(&[s, t]);
        assert!((p[0] - quad[k][0]).abs() < 1e-15 && (p[1] - quad[k][1]).abs() < 1e-15);
    }
}

#[test]
fn d230_quad_constant_accessors_are_origin_linearization() {
    // Documented semantics: for the quad variant the constant-J accessors
    // report the first-3-node affine linearization at the reference origin
    // (unchanged from the pre-D230 values).
    let corners = [[0.0_f64, 0.0], [2.0, 0.1], [1.9, 1.3], [0.2, 0.9]];
    let mut mesh = Mesh::<2>::make_cartesian_2d(1, 1, 1.0, 1.0);
    mesh.coords = [corners[0], corners[1], corners[3], corners[2]]
        .iter()
        .flat_map(|c| [c[0], c[1]])
        .collect();
    let tr = ElementTransformation::from_simplex(&mesh, 0);
    let j = tr.jacobian();
    // Columns: v1 - v0, v2 - v0.
    assert!((j[(0, 0)] - 2.0).abs() < 1e-15 && (j[(1, 0)] - 0.1).abs() < 1e-15);
    assert!((j[(0, 1)] - 1.9).abs() < 1e-15 && (j[(1, 1)] - 1.3).abs() < 1e-15);
}
