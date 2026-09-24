//! D760 — `error_estimate::zz_estimator`'s one-point sampling station on
//! hexes.
//!
//! # The defect
//!
//! The estimator sampled every 3-D cell at `xi = [0.25, 0.25, 0.25]` — the
//! **tet centroid** (barycentric `1/(d+1)` on the unit simplex) dispatched on
//! the dimension alone, so hexes inherited it verbatim.  Once D721 moved the
//! whole hex family (bases, `hex_rule`, `vertex_shapes`) onto MFEM's natural
//! `[0,1]³` frame, that literal became a point **1/4 of the way across the
//! cube** instead of its centre (its pre-D721 meaning was a `[-1,1]³` point at
//! 0.25 ↔ `[0,1]` 0.625 — neither is a centroid).
//!
//! # Adjudication (MFEM 4.10, `tmp/d760/zz_rule_probe.cpp`, run log
//! `tmp/d760/zz_rule_probe_out.log`)
//!
//! MFEM's `ZZErrorEstimator` (`fem/gridfunc.cpp:4657`) has no hand-picked hex
//! station at all: `ComputeElementFlux` evaluates the element flux and
//! `ComputeFluxEnergy` the flux-difference energy, each at the integrator's
//! own rule (`DiffusionIntegrator` → `IntRules.Get(geom, 2*order)`), and the
//! flux is smoothed by the plain arithmetic nodal average
//! (`GridFunction::ComputeFlux` = `SumFluxAndCount` + `/count` — exactly this
//! estimator's recovery step).  The probe prints that rule on a hex:
//!
//! ```text
//! CUBE 2*order=2 : npoints=8       (flux order 1)
//!    ip[0] = (0.21132486540518711, 0.21132486540518711, 0.21132486540518711)  w=0.125
//! CUBE 2*order=4 : npoints=27      (flux order 2)
//!    ip[13] = (0.5, 0.5, 0.5)  w=0.0877914951989026
//! centroid (0.5,0.5,0.5) among the order-2 CUBE rule points: 0
//! ```
//!
//! MFEM's hex point set is the family's Gauss rule, **[0,1]³-native in MFEM
//! 4.10** (weight sum 1, nodes `0.5 ± 1/(2√3)` — the same frame D721 gave the
//! fem-rs hex family), and it contains the centroid exactly at flux order 2
//! (the 3×3×3 rule's centre node, the point of largest weight) but not at
//! order 1.  The one-point collocation `zz_estimator` collapses that rule to a
//! single station, so the station must be the element's **reference centroid**
//! — the symmetric representative of MFEM's point set, and (pin 2) the fixed
//! point of a relabeling of the reference cube.
//!
//! MFEM anchor for the same physical setup (2×2×2 unit-cube hexes, P1 H¹,
//! `u = 0.5 + 2xyz + x² + 2y² + 3z²`): `ZZErrorEstimator(Diffusion)` gives
//! `total = 1.0801234497346435`, `eta[e] = 0.38188130791298663` on every
//! element.  fem-rs's one-point version is the same estimator with the rule
//! collapsed to one station — measured here: centroid station 0.4208610500
//! (ratio 1.102), the pre-D760 quarter point 0.3844829625 (ratio 1.007).  The
//! pre-fix station is *numerically* closer to the 8-point integral on this one
//! case (a coincidence of the collapsed rule, and it is the *quadrature* that
//! is being compared here, not a parity quantity), but it is frame-dependent —
//! pin 2 measures a 3.05e-2 drift for the same physical mesh under a reference
//! relabeling, i.e. the estimator's value would depend on how the corner list
//! was written.  That is the defect D760 removes.
//!
//! # Acceptance
//!
//! 1. **spec pin**: `zz_estimator`'s per-element values equal the documented
//!    formula re-evaluated with the station `(0.5, 0.5, 0.5)` (the test owns
//!    the station literal, the implementation must match it);
//! 2. **reference-frame relabeling invariance**: the same physical cells
//!    written with the corner list permuted by a symmetry of the reference
//!    cube (`ξ₀ ↦ 1−ξ₀`) must give the same estimate.  The centroid is that
//!    symmetry's fixed point; the pre-D760 quarter point moves with the
//!    relabeling, so the old station was frame-dependent (and thus wrong) —
//!    asserted explicitly;
//! 3. the pre-D760 station's deviation is material (documented magnitude).
//!
//! The discriminating field needs a **non-affine** P1 gradient: on a straight
//! tensor cell the station shift is a constant physical offset and the
//! difference `∇u_h(station₂) − ∇u_h(station₁)` cancels against the same shift
//! in the nodal recovery whenever `∇u_h` is affine (which quadratic-only
//! fields interpolate to), so `u` carries the trilinear term `2xyz`.
//!
//! Run:
//!   cargo test -p fem-assembly --test d760_zz_sample_point -- --nocapture

use fem_assembly::postproc::error_estimate::zz_estimator;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Non-affine P1 gradient (see the module doc): trilinear term + quadratics.
fn u_3d(x: &[f64]) -> f64 { 0.5 + 2.0 * x[0] * x[1] * x[2] + x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2] }
fn u_2d(x: &[f64]) -> f64 { 0.5 + 3.0 * x[0] * x[1] + x[0] * x[0] + 2.0 * x[1] * x[1] }

/// The estimator's documented formula with a caller-supplied station:
/// `∇u_h` sampled at `station`, arithmetically averaged at the mesh nodes,
/// re-averaged per element over the element's nodes, and the squared deviation
/// weighted by the element volume.  `vol` is passed explicitly (the unit-cube
/// fixtures below have an exactly known cell volume; `elem_vol` integrates the
/// same constant).
fn zz_with_station<M, S>(gf: &GridFunction<'_, S>, station: &[f64], vol: &dyn Fn(u32) -> f64) -> Vec<f64>
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let m = gf.space().mesh();
    let ne = m.n_elements();
    let d = m.dim() as usize;
    let eg: Vec<Vec<f64>> = (0..ne as u32)
        .map(|e| gf.evaluate_gradient_at_element(e, station))
        .collect();
    let nn = m.n_nodes();
    let mut ns = vec![vec![0.0_f64; d]; nn];
    let mut nc = vec![0u32; nn];
    for e in 0..ne as u32 {
        for &n in m.element_nodes(e) {
            for di in 0..d { ns[n as usize][di] += eg[e as usize][di]; }
            nc[n as usize] += 1;
        }
    }
    for n in 0..nn {
        if nc[n] > 0 {
            for di in 0..d { ns[n][di] /= nc[n] as f64; }
        }
    }
    (0..ne as u32)
        .map(|e| {
            let nlist = m.element_nodes(e);
            let npe = nlist.len() as f64;
            let mut rec = vec![0.0_f64; d];
            for &n in nlist {
                for di in 0..d { rec[di] += ns[n as usize][di] / npe; }
            }
            let diff = (0..d).map(|di| (eg[e as usize][di] - rec[di]).powi(2)).sum::<f64>();
            (diff * vol(e)).sqrt()
        })
        .collect()
}

fn max_dev(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

/// 2×2×2 unit-cube hexes; every cell has volume 1/8 exactly.
fn hex_patch() -> Mesh<3> { Mesh::<3>::unit_cube_hex(2) }

/// `unit_cube_hex`'s conn is the standard ring `[(0,0,0),(1,0,0),(1,1,0),
/// (0,1,0)] × {z=0, z=1}`.  Relabel every cell's corner list by the reference
/// symmetry `ξ₀ ↦ 1−ξ₀` (swap the ring slots `{0,1}`, `{2,3}`, `{4,5}`,
/// `{6,7}`): the same physical cells, a different reference parameterization
/// (`G ↦ G∘σ`).
fn relabeled_x_reflection_hex(mesh: &mut Mesh<3>) {
    const P: [usize; 8] = [1, 0, 3, 2, 5, 4, 7, 6];
    let npe = 8;
    let ne = mesh.n_elems();
    let mut conn = vec![0u32; ne * npe];
    for e in 0..ne {
        for k in 0..npe {
            conn[e * npe + k] = mesh.conn[e * npe + P[k]];
        }
    }
    mesh.conn = conn;
}

/// A **graded** 2-quad row (x breaks at 0, 0.25, 1): the same station shift
/// is a different physical offset per element, so the station is visible.
fn quad_patch() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 1.0, 0.25, 1.0, 1.0, 1.0],
        vec![0, 1, 4, 3, 1, 2, 5, 4],
        vec![1, 1],
        ElementType::Quad4,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

/// `ξ₁ ↦ 1−ξ₁` on the quad ring (swap `{0,3}`, `{1,2}`).
fn relabeled_y_reflection_quad(mesh: &mut Mesh<2>) {
    const P: [usize; 4] = [3, 2, 1, 0];
    let npe = 4;
    let ne = mesh.n_elems();
    let mut conn = vec![0u32; ne * npe];
    for e in 0..ne {
        for k in 0..npe {
            conn[e * npe + k] = mesh.conn[e * npe + P[k]];
        }
    }
    mesh.conn = conn;
}

/// MFEM's `ZZErrorEstimator(Diffusion)` on this mesh/field (probe log above),
/// uniform over the 8 elements.
const MFEM_ETA: f64 = 0.38188130791298663;

#[test]
fn d760_hex_station_is_the_reference_centroid() {
    let mesh = hex_patch();
    let space = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&space, space.interpolate(&u_3d).as_slice().to_vec());

    let eta = zz_estimator(&gf).eta;
    let spec = zz_with_station(&gf, &[0.5, 0.5, 0.5], &|_e| 0.125);
    let old = zz_with_station(&gf, &[0.25, 0.25, 0.25], &|_e| 0.125);

    let spec_dev = max_dev(&eta, &spec);
    let old_dev = max_dev(&eta, &old);
    eprintln!("D760 hex eta[0] = {:.17e}", eta[0]);
    eprintln!("   centroid-station formula {:.17e} (|Δ| {spec_dev:.1e})", spec[0]);
    eprintln!("   pre-D760 quarter point  {:.17e} (|Δ| {old_dev:.1e})", old[0]);
    eprintln!(
        "   MFEM ZZErrorEstimator on the same setup: {MFEM_ETA:.17e} (ratio {:.4})",
        eta[0] / MFEM_ETA
    );

    assert!(
        spec_dev <= 1e-12,
        "zz_estimator must sample the reference centroid [0.5;3]: |Δ| = {spec_dev:.3e}"
    );
    // D760 sensitivity: the old station gives a materially different estimate
    // (the defect was not a roundoff question).
    assert!(
        old_dev > 1e-2 * eta[0],
        "pre-D760 station must differ materially (|Δ| = {old_dev:.3e} vs eta {:.3e})",
        eta[0]
    );
    // Order cross-check against the MFEM run: the collapsed rule cannot
    // reproduce the 8-point integral exactly, but the same estimator on the
    // same mesh/field must land in the same ballpark.
    assert!(
        eta[0] > 0.25 * MFEM_ETA && eta[0] < 4.0 * MFEM_ETA,
        "fem-rs one-point eta {:.6e} vs MFEM 8-point {MFEM_ETA:.6e} (same order)",
        eta[0]
    );
}

#[test]
fn d760_hex_station_invariant_under_reference_relabeling() {
    let mesh_a = hex_patch();
    let mut mesh_b = hex_patch();
    relabeled_x_reflection_hex(&mut mesh_b);

    let space_a = H1Space::new(mesh_a, 1);
    let space_b = H1Space::new(mesh_b, 1);
    // Identical node coordinates → identical nodal interpolants.
    let ua = space_a.interpolate(&u_3d);
    let ub = space_b.interpolate(&u_3d);
    for (a, b) in ua.as_slice().iter().zip(ub.as_slice()) {
        assert_eq!(a, b, "same coords ⇒ same nodal values");
    }
    let gf_a = GridFunction::new(&space_a, ua.as_slice().to_vec());
    let gf_b = GridFunction::new(&space_b, ub.as_slice().to_vec());

    let centroid = max_dev(
        &zz_with_station(&gf_a, &[0.5, 0.5, 0.5], &|_e| 0.125),
        &zz_with_station(&gf_b, &[0.5, 0.5, 0.5], &|_e| 0.125),
    );
    let quarter_a = zz_with_station(&gf_a, &[0.25, 0.25, 0.25], &|_e| 0.125);
    let quarter = max_dev(&quarter_a, &zz_with_station(&gf_b, &[0.25, 0.25, 0.25], &|_e| 0.125));
    eprintln!("D760 relabeling: centroid |Δ| = {centroid:.3e}, quarter point |Δ| = {quarter:.3e}");

    assert!(
        centroid <= 1e-13,
        "the centroid is the relabeling fixed point: |Δ| = {centroid:.3e}"
    );
    assert!(
        quarter > 1e-2 * quarter_a[0],
        "the pre-D760 quarter point moves with the frame: |Δ| = {quarter:.3e} on eta {:.3e}",
        quarter_a[0]
    );

    // The shipped estimator (not just the formula) is frame-invariant.
    let d = max_dev(&zz_estimator(&gf_a).eta, &zz_estimator(&gf_b).eta);
    eprintln!("D760 relabeling: zz_estimator |Δ| = {d:.3e}");
    assert!(d <= 1e-13, "zz_estimator must be frame-invariant: {d:.3e}");
}

/// The same family-relative station rule on the 2-D quads (frame `[-1,1]²`,
/// centroid `(0,0)`; the graded row makes the station visible).
#[test]
fn d760_quad_station_is_the_frame_centroid() {
    let mesh = quad_patch();
    let space = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&space, space.interpolate(&u_2d).as_slice().to_vec());
    let area = |e: u32| if e == 0 { 0.25 } else { 0.75 };
    let eta = zz_estimator(&gf).eta;
    let dev = max_dev(&eta, &zz_with_station(&gf, &[0.0, 0.0], &area));
    let old = max_dev(&eta, &zz_with_station(&gf, &[1.0 / 3.0, 1.0 / 3.0], &area));
    eprintln!("D760 quad eta = {eta:?}  spec |Δ| = {dev:.1e}  pre-D760 |Δ| = {old:.1e}");
    assert!(dev <= 1e-12, "quad station is the [-1,1]² centroid (0,0): |Δ| = {dev:.3e}");
    assert!(old > 1e-2 * eta[0], "pre-D760 quad station differed: |Δ| = {old:.3e}");

    let mut mesh_b = quad_patch();
    relabeled_y_reflection_quad(&mut mesh_b);
    let space_b = H1Space::new(mesh_b, 1);
    let gf_b = GridFunction::new(&space_b, space_b.interpolate(&u_2d).as_slice().to_vec());
    let d = max_dev(&eta, &zz_estimator(&gf_b).eta);
    eprintln!("D760 quad relabeling |Δ| = {d:.3e}");
    assert!(d <= 1e-13, "quad estimator must be frame-invariant: {d:.3e}");
}
