//! D562: patch-wise partial assembly (`-pa`) regression.
//!
//! Pins the MFEM-quirk semantics discovered on this debt:
//! * `SetupPatchPA` overwrites the single member `pa_data` per patch, so
//!   `AddMultNURBSPA` applies **every** patch with the **last patch's**
//!   quadrature data (instrumented proxy of `bilininteg_diffusion_pa.cpp`,
//!   `tmp/d562/pa_probe*.cpp`).  The C++ per-patch yp for the unit vector
//!   e60 (ball-nurbs, patch 0) is dumped in `tmp/d562/cpp_yp0.txt`; this
//!   test asserts fem-rs's contraction reproduces it to the last bit.
//! * The patch VDOF maps match MFEM's `GetPatchVDofs` lists
//!   (`tmp/d562/d562_vdofs.cpp` dump).

use fem_assembly::nurbs_patch::{
    apply_to_knot_intervals, assemble_domain_lf_exact, segment_rule, setup_patch_pa_diffusion,
    NurbsMeshGeometry, NurbsPatchRules,
};
use fem_space::nurbs_extension::NurbsExtension;
use fem_space::nurbs_fe_space::NurbsFESpace;

fn ball_setup() -> (NurbsExtension, NurbsFESpace, NurbsMeshGeometry, NurbsPatchRules) {
    let text = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/ball-nurbs.mesh"),
    )
    .expect("ball-nurbs.mesh");
    let ext = NurbsExtension::from_mesh_str(&text).unwrap();
    let space = NurbsFESpace::from_mesh_isoparametric_str(&text, 0).unwrap();
    let geo = NurbsMeshGeometry::from_mesh_text(&text, &ext).unwrap();
    let dim = ext.dim();
    let fe_order = space.orders().iter().copied().max().unwrap();
    let ir_order = 2 * fe_order;
    let mut rules = NurbsPatchRules::new(ext.n_patches(), dim);
    let base = segment_rule(ir_order as u8);
    for p in 0..ext.n_patches() {
        let pkv = ext.patch_knot_vectors(p).unwrap();
        let ir1d: Vec<Vec<(f64, f64)>> =
            pkv.iter().map(|kv| apply_to_knot_intervals(&base, kv)).collect();
        rules.set_patch_rules_1d(p, ir1d);
    }
    rules.finalize(&ext);
    (ext, space, geo, rules)
}

    #[test]
    fn d562_patch_pa_patch0_yp_matches_mfem() {
        let (_ext, space, geo, rules) = ball_setup();
        let pa = setup_patch_pa_diffusion(&space, &geo, &rules, 1.0);

        // x = e60: dof 60 lives at patch-0 local index 119 (and in patches 3, 6).
        let mut xp = vec![0.0_f64; pa.patch_vdofs(0).len()];
        let mut found = false;
        for (l, &g) in pa.patch_vdofs(0).iter().enumerate() {
            if g == 60 {
                xp[l] = 1.0;
                found = true;
            }
        }
        assert!(found, "dof 60 must be a patch-0 dof");
        let mut yp0 = vec![0.0_f64; pa.patch_vdofs(0).len()];
        pa.add_mult_patch_pa_hook(0, &xp, &mut yp0);

        let cpp_text = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/assembly/tests/data/d562/cpp_yp0.txt"),
        )
        .expect("cpp_yp0.txt");
        let cpp: Vec<f64> = cpp_text
            .lines()
            .skip(1)
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect();
        assert_eq!(cpp.len(), yp0.len(), "yp length");
        for (l, &cv) in cpp.iter().enumerate() {
            assert_eq!(yp0[l].to_bits(), cv.to_bits(), "yp0[{l}] bit-exact");
        }
        // Spot-print the pinned diagonal for the record.
        println!("yp0[119] = {:.17}", yp0[119]);
    }

    #[test]
    fn d562_patch_pa_patch0_seeded_yp_matches_mfem() {
        // Same as above but with the operator probe's seeded input vector
        // (Lcg(1000), essential dofs zeroed by the ConstrainedOperator).
        let (_ext, space, geo, rules) = ball_setup();
        let pa = setup_patch_pa_diffusion(&space, &geo, &rules, 1.0);
        let ess: Vec<usize> = {
            let ess_dofs = space.boundary_dofs_marked(&[true; 6][..]);
            ess_dofs.iter().map(|&d| d as usize).collect()
        };

        struct Lcg(u64);
        impl Lcg {
            fn next_f64(&mut self) -> f64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
            }
        }
        let mut g = Lcg(1000);
        let v: Vec<f64> = (0..space.n_dofs()).map(|_| g.next_f64()).collect();
        let mut xp = vec![0.0_f64; pa.patch_vdofs(0).len()];
        for (l, &g) in pa.patch_vdofs(0).iter().enumerate() {
            xp[l] = if ess.contains(&g) { 0.0 } else { v[g] };
        }
        let mut yp0 = vec![0.0_f64; pa.patch_vdofs(0).len()];
        pa.add_mult_patch_pa_hook(0, &xp, &mut yp0);

        let text = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../crates/assembly/tests/data/d562/cpp_seeded_full.txt"),
        )
        .unwrap();
        let mut cpp_xp: Option<Vec<f64>> = None;
        let mut diffs = 0usize;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("XP0:") {
                let cx: Vec<f64> = rest.split_whitespace().map(|t| t.parse().unwrap()).collect();
                assert_eq!(cx.len(), xp.len());
                for (l, (&cv, &mv)) in cx.iter().zip(&xp).enumerate() {
                    assert_eq!(cv.to_bits(), mv.to_bits(), "xp[{l}] bit-exact");
                }
                cpp_xp = Some(cx);
            }
            if let Some(rest) = line.strip_prefix("YPALL p=") {
                let (p, vals) = rest.split_once(':').unwrap();
                let p: usize = p.parse().unwrap();
                let cv: Vec<f64> = vals.split_whitespace().map(|t| t.parse().unwrap()).collect();
                let mut x = vec![0.0_f64; cv.len()];
                for (l, &g) in pa.patch_vdofs(p).iter().enumerate() {
                    x[l] = if ess.contains(&g) { 0.0 } else { v[g] };
                }
                let mut yp = vec![0.0_f64; cv.len()];
                pa.add_mult_patch_pa_hook(p, &x, &mut yp);
                for (l, (&c, &m)) in cv.iter().zip(&yp).enumerate() {
                    if c.to_bits() != m.to_bits() {
                        diffs += 1;
                        println!("p={p} l={l} cpp={c:.17} rust={m:.17}");
                    }
                }
            }
        }
        assert!(cpp_xp.is_some(), "XP0 line missing");
        assert_eq!(diffs, 0, "patch contraction bit differences");

        // Full constrained-operator action on the same seeded vector.
        let mut y = vec![0.0_f64; space.n_dofs()];
        let mut z = v.clone();
        for &i in &ess {
            z[i] = 0.0;
        }
        pa.add_mult_nurbs_pa(&z, &mut y);
        for &i in &ess {
            y[i] = v[i];
        }
        let mut h: u64 = 0xcbf29ce484222325;
        for &val in &y {
            for b in val.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        println!("op0_hash={h:016x} (C++ f5782e57691b76cb)");
    }

#[test]
fn d562_patch_pa_diagonal_is_last_patch_quirk() {
    // The whole operator's diagonal must reflect the LAST patch's pa_data:
    // patch 6 contributes A(60,60) = 0.041955158592036829 and patches 0 and 3
    // the values printed by the instrumented C++ proxy.
    let (_ext, space, geo, rules) = ball_setup();
    let pa = setup_patch_pa_diffusion(&space, &geo, &rules, 1.0);
    let ess: Vec<usize> = Vec::new();
    let mut y = vec![0.0_f64; space.n_dofs()];
    let mut x = vec![0.0_f64; space.n_dofs()];
    x[60] = 1.0;
    pa.add_mult_nurbs_pa(&x, &mut y);
    let cpp = [0.041955158592036843, 0.041955158592036829, 0.041955158592036829];
    let total: f64 = cpp.iter().sum();
    assert_eq!(y[60].to_bits(), total.to_bits(), "A(60,60)");
}

#[test]
fn d562_pa6_diag_matches_mfem_member() {
    // The shared (last-patch) pa_data diagonal o11 must equal the C++ member
    // dump: 0.00099896077775976427 (pa_probe6.cpp instrumentation).
    let (_ext, space, geo, rules) = ball_setup();
    let pa = setup_patch_pa_diffusion(&space, &geo, &rules, 1.0);
    let pd6 = pa.compute_pa_data(&space, &geo, &rules, 1.0, 6);
    let cpp = 0.00099896077775976427_f64;
    assert_eq!(pd6[0].to_bits(), cpp.to_bits());
}
