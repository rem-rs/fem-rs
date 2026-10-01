//! d102 — embedded-collection (ND_R2D / RT_R2D / ND_R1D / RT_R1D) element
//! probe pinned against the MFEM 4.10 oracle.
//!
//! Truth source: `tmp/d102r2d/probe_r2d.cpp` built against `$HOME/mfem410_ser`
//! (output `tmp/d102r2d/probe_ref.txt`, compact extract in
//! `d102_probe_extract.txt` next to this file).  The probe dumps, for
//! MFEM's `ND_R2D_TriangleElement` / `RT_R2D_TriangleElement` /
//! `ND_R2D_QuadrilateralElement` / `RT_R2D_QuadrilateralElement` (p ranges as
//! in the collection verifies) and the `ND_R2D_SegmentElement` /
//! `ND_R1D_SegmentElement` / `RT_R1D_SegmentElement` families:
//!
//! * the dof sites (`FE::Nodes`),
//! * reference `CalcVShape` / `CalcCurlShape` / `CalcDivShape` at two fixed
//!   reference points,
//! * physical `CalcVShape` / `CalcPhysCurlShape` on a sheared single-element
//!   mesh,
//! * `FE::Project` of a smooth 3-component field `E(x, y)` on that element.
//!
//! The physical block's map is rebuilt from the probe's `corner` lines (the
//! transformed reference corners in the connectivity order MFEM's loader
//! produced — it cyclically rotates the probe mesh's triangle).  Acceptance:
//! every compared quantity agrees with MFEM to a relative 1e-10
//! (round-102 policy: 结果一致级, not bitwise); the measured worst relative
//! difference is asserted into `d102_probe_worst_rel_diff_is_pin()`.

use fem_element::embedded::{
    Jac2D, NdR1dSegment, NdR2dQuad, NdR2dSegment, NdR2dTri, RtR1dSegment, RtR2dQuad, RtR2dTri,
};

const REF: &str = include_str!("d102_probe_extract.txt");

const SEG_XS: [f64; 2] = [0.137, 0.712];

fn tri_xi(q: usize) -> [f64; 2] {
    [[0.137, 0.263], [0.412, 0.198]][q]
}
fn quad_xi(q: usize) -> [f64; 2] {
    [[0.137, 0.263], [0.712, 0.348]][q]
}

fn e_field(x: f64, y: f64) -> [f64; 3] {
    [
        (2.0 * x + 0.3).sin() + 0.1 * y,
        (3.0 * y - 0.2).cos() - 0.2 * x,
        x * y + 0.7,
    ]
}

/// Affine (triangle) / bilinear (quad) map rebuilt from the dumped corners.
#[derive(Default)]
struct PhysMap {
    corners: Vec<[f64; 2]>,
    tri: bool,
}

impl PhysMap {
    fn transform(&self, xi: [f64; 2]) -> [f64; 2] {
        let c = &self.corners;
        if self.tri {
            [
                c[0][0] + xi[0] * (c[1][0] - c[0][0]) + xi[1] * (c[2][0] - c[0][0]),
                c[0][1] + xi[0] * (c[1][1] - c[0][1]) + xi[1] * (c[2][1] - c[0][1]),
            ]
        } else {
            let (x, y) = (xi[0], xi[1]);
            let phi = [(1.0 - x) * (1.0 - y), x * (1.0 - y), x * y, (1.0 - x) * y];
            let mut p = [0.0, 0.0];
            for (k, &pk) in phi.iter().enumerate() {
                p[0] += pk * c[k][0];
                p[1] += pk * c[k][1];
            }
            p
        }
    }
    fn jac(&self, xi: [f64; 2]) -> Jac2D {
        let c = &self.corners;
        if self.tri {
            let (j00, j10) = (c[1][0] - c[0][0], c[1][1] - c[0][1]);
            let (j01, j11) = (c[2][0] - c[0][0], c[2][1] - c[0][1]);
            Jac2D { j00, j01, j10, j11, det: j00 * j11 - j01 * j10 }
        } else {
            let (x, y) = (xi[0], xi[1]);
            let g = [
                [y - 1.0, x - 1.0],
                [1.0 - y, -x],
                [y, x],
                [-y, 1.0 - x],
            ];
            let mut j = [0.0_f64; 4];
            for (k, gk) in g.iter().enumerate() {
                j[0] += gk[0] * c[k][0];
                j[1] += gk[1] * c[k][0];
                j[2] += gk[0] * c[k][1];
                j[3] += gk[1] * c[k][1];
            }
            Jac2D { j00: j[0], j01: j[1], j10: j[2], j11: j[3], det: j[0] * j[3] - j[1] * j[2] }
        }
    }
}

/// Relative difference |a−b| / max(1, |b|).
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / 1.0_f64.max(b.abs())
}

/// Tangent / normal axes per family (MFEM `tk` / `nk` tables, in-plane part).
fn tangent(family: &str, tk: u8) -> [f64; 2] {
    match (family, tk) {
        ("ND_R2D_Triangle", 0) => [1.0, 0.0],
        ("ND_R2D_Triangle", 1) => [-1.0, 1.0],
        ("ND_R2D_Triangle", 2) => [0.0, -1.0],
        ("ND_R2D_Triangle", 3) => [0.0, 1.0],
        ("ND_R2D_Quadrilateral", 0) => [1.0, 0.0],
        ("ND_R2D_Quadrilateral", 1) => [0.0, 1.0],
        ("ND_R2D_Quadrilateral", 2) => [-1.0, 0.0],
        ("ND_R2D_Quadrilateral", 3) => [0.0, -1.0],
        _ => panic!("tangent({family}, {tk})"),
    }
}

fn normal(family: &str, nk: u8) -> [f64; 2] {
    match (family, nk) {
        ("RT_R2D_Triangle", 0) => [0.0, -1.0],
        ("RT_R2D_Triangle", 1) => [1.0, 1.0],
        ("RT_R2D_Triangle", 2) => [-1.0, 0.0],
        ("RT_R2D_Quadrilateral", 0) => [0.0, -1.0],
        ("RT_R2D_Quadrilateral", 1) => [1.0, 0.0],
        ("RT_R2D_Quadrilateral", 2) => [0.0, 1.0],
        ("RT_R2D_Quadrilateral", 3) => [-1.0, 0.0],
        _ => panic!("normal({family}, {nk})"),
    }
}

#[test]
fn d102_probe_agrees_with_mfem() {
    let mut worst = 0.0_f64;
    let mut n_checked = 0usize;
    let mut family = String::new();
    let mut p = 0usize;
    let mut dof = 0usize;
    let mut map = PhysMap::default();
    let mut nodes: Vec<Vec<f64>> = vec![];
    let mut pending_corners: Vec<[f64; 2]> = vec![];

    let check = |worst: &mut f64, n: &mut usize, got: f64, want: f64, ctx: String| {
        let r = rel(got, want);
        assert!(r <= 1e-10, "d102 probe mismatch ({ctx}): rust={got:.17} mfem={want:.17} rel={r:.3e}");
        if r > *worst {
            *worst = r;
        }
        *n += 1;
    };

    for line in REF.lines() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f[0].starts_with('[') && f.len() == 3 {
            // "[<family> p=<p> dof=<dof>]" — no space after '['.
            family = f[0].trim_start_matches('[').to_string();
            p = f[1]
                .trim_start_matches("p=")
                .parse()
                .unwrap_or_else(|e| panic!("header parse failed for line {line:?}: {e}"));
            dof = f[2]
                .trim_start_matches("dof=")
                .trim_end_matches(']')
                .parse()
                .unwrap_or_else(|e| panic!("header parse failed for line {line:?}: {e}"));
            map = PhysMap { corners: std::mem::take(&mut pending_corners), tri: !family.contains("Quadrilateral") };
            nodes = vec![];
            continue;
        }
        match f[0] {
            "corner" => pending_corners.push([f[2].parse().unwrap(), f[3].parse().unwrap()]),
            "node" => {
                let coords: Vec<f64> = f[2..].iter().map(|v| v.parse().unwrap()).collect();
                let k: usize = f[1].parse().unwrap();
                nodes.push(coords.clone());
                let el_site = match family.as_str() {
                    "ND_R2D_Triangle" => {
                        let el = NdR2dTri::new(p);
                        el.nodes()[k].to_vec()
                    }
                    "ND_R2D_Quadrilateral" => {
                        let el = NdR2dQuad::new(p);
                        el.nodes()[k].to_vec()
                    }
                    "RT_R2D_Triangle" => {
                        let el = RtR2dTri::new(p);
                        el.nodes()[k].to_vec()
                    }
                    "RT_R2D_Quadrilateral" => {
                        let el = RtR2dQuad::new(p);
                        el.nodes()[k].to_vec()
                    }
                    "ND_R2D_Segment" => vec![NdR2dSegment::new(p).nodes()[k]],
                    "ND_R1D_Segment" => vec![NdR1dSegment::new(p).nodes()[k]],
                    "RT_R1D_Segment" => vec![RtR1dSegment::new(p).nodes()[k]],
                    other => panic!("unknown family {other}"),
                };
                for (c, &v) in coords.iter().enumerate() {
                    check(&mut worst, &mut n_checked, el_site[c], v, format!("{family} node {k} c{c}"));
                }
            }
            "vref" | "vphy" => {
                let q: usize = f[1].parse().unwrap();
                let k: usize = f[2].parse().unwrap();
                let vals: Vec<f64> = f[3..].iter().map(|v| v.parse().unwrap()).collect();
                let phys = f[0] == "vphy";
                let (buf, vdim) = if family.contains("Segment") {
                    let x = SEG_XS[q];
                    match family.as_str() {
                        "ND_R2D_Segment" => {
                            let el = NdR2dSegment::new(p);
                            let mut b = vec![0.0; el.n_dofs() * 2];
                            if phys { unreachable!() } else { el.eval_vshape_ref(x, &mut b) }
                            (b, 2)
                        }
                        "ND_R1D_Segment" => {
                            let el = NdR1dSegment::new(p);
                            let mut b = vec![0.0; el.n_dofs() * 3];
                            el.eval_vshape_ref(x, &mut b);
                            (b, 3)
                        }
                        "RT_R1D_Segment" => {
                            let el = RtR1dSegment::new(p);
                            let mut b = vec![0.0; el.n_dofs() * 3];
                            el.eval_vshape_ref(x, &mut b);
                            (b, 3)
                        }
                        other => panic!("unknown segment family {other}"),
                    }
                } else {
                    let xi = if map.tri { tri_xi(q) } else { quad_xi(q) };
                    let mut b = vec![0.0_f64; dof * 3];
                    match (family.as_str(), phys) {
                        ("ND_R2D_Triangle", false) => NdR2dTri::new(p).eval_vshape_ref(&xi, &mut b),
                        ("ND_R2D_Triangle", true) => {
                            NdR2dTri::new(p).eval_vshape_phys(&xi, &map.jac(xi), &mut b)
                        }
                        ("ND_R2D_Quadrilateral", false) => {
                            NdR2dQuad::new(p).eval_vshape_ref(&xi, &mut b)
                        }
                        ("ND_R2D_Quadrilateral", true) => {
                            NdR2dQuad::new(p).eval_vshape_phys(&xi, &map.jac(xi), &mut b)
                        }
                        ("RT_R2D_Triangle", false) => RtR2dTri::new(p).eval_vshape_ref(&xi, &mut b),
                        ("RT_R2D_Triangle", true) => {
                            RtR2dTri::new(p).eval_vshape_phys(&xi, &map.jac(xi), &mut b)
                        }
                        ("RT_R2D_Quadrilateral", false) => {
                            RtR2dQuad::new(p).eval_vshape_ref(&xi, &mut b)
                        }
                        ("RT_R2D_Quadrilateral", true) => {
                            RtR2dQuad::new(p).eval_vshape_phys(&xi, &map.jac(xi), &mut b)
                        }
                        (other, _) => panic!("unknown family {other}"),
                    }
                    (b, 3)
                };
                for (c, &v) in vals.iter().enumerate().take(vdim) {
                    check(
                        &mut worst,
                        &mut n_checked,
                        buf[k * vdim + c],
                        v,
                        format!("{family} {} q{q} k{k} c{c}", f[0]),
                    );
                }
            }
            "cref" | "dref" => {
                // extraction keeps q = 0 only
                let k: usize = f[2].parse().unwrap();
                let vals: Vec<f64> = f[3..].iter().map(|v| v.parse().unwrap()).collect();
                if family.contains("Segment") {
                    let x = SEG_XS[0];
                    match family.as_str() {
                        "ND_R2D_Segment" => {
                            let el = NdR2dSegment::new(p);
                            let mut b = vec![0.0; el.n_dofs()];
                            el.eval_curl_ref(x, &mut b);
                            check(&mut worst, &mut n_checked, b[k], vals[0], format!("{family} cref k{k}"));
                        }
                        "ND_R1D_Segment" => {
                            let el = NdR1dSegment::new(p);
                            let mut b = vec![0.0; el.n_dofs() * 3];
                            el.eval_curl_ref(x, &mut b);
                            for (c, &v) in vals.iter().enumerate() {
                                check(&mut worst, &mut n_checked, b[k * 3 + c], v, format!("{family} cref k{k} c{c}"));
                            }
                        }
                        "RT_R1D_Segment" => {
                            let el = RtR1dSegment::new(p);
                            let mut b = vec![0.0; el.n_dofs()];
                            el.eval_div_ref(x, &mut b);
                            check(&mut worst, &mut n_checked, b[k], vals[0], format!("{family} dref k{k}"));
                        }
                        other => panic!("unknown segment family {other}"),
                    }
                } else if f[0] == "cref" {
                    let xi = if map.tri { tri_xi(0) } else { quad_xi(0) };
                    let mut b = vec![0.0_f64; dof * 3];
                    if map.tri {
                        NdR2dTri::new(p).eval_curl_ref(&xi, &mut b);
                    } else {
                        NdR2dQuad::new(p).eval_curl_ref(&xi, &mut b);
                    }
                    for (c, &v) in vals.iter().enumerate() {
                        check(&mut worst, &mut n_checked, b[k * 3 + c], v, format!("{family} cref k{k} c{c}"));
                    }
                } else {
                    let xi = if map.tri { tri_xi(0) } else { quad_xi(0) };
                    let mut b = vec![0.0_f64; dof];
                    if map.tri {
                        RtR2dTri::new(p).eval_div_ref(&xi, &mut b);
                    } else {
                        RtR2dQuad::new(p).eval_div_ref(&xi, &mut b);
                    }
                    check(&mut worst, &mut n_checked, b[k], vals[0], format!("{family} dref k{k}"));
                }
            }
            "proj" => {
                let k: usize = f[1].parse().unwrap();
                let want: f64 = f[2].parse().unwrap();
                let site = &nodes[k];
                let (xr, yr) = (site[0], site[1]);
                let xi = [xr, yr];
                let jac = map.jac(xi);
                let phys = map.transform(xi);
                let vc = e_field(phys[0], phys[1]);
                let got = if family.starts_with("ND") {
                    let tk = match family.as_str() {
                        "ND_R2D_Triangle" => NdR2dTri::new(p).dof2tk()[k],
                        "ND_R2D_Quadrilateral" => NdR2dQuad::new(p).dof2tk()[k],
                        other => panic!("{other}"),
                    };
                    if tk == 4 {
                        vc[2]
                    } else {
                        let t = tangent(&family, tk);
                        let tx = [jac.j00 * t[0] + jac.j01 * t[1], jac.j10 * t[0] + jac.j11 * t[1]];
                        vc[0] * tx[0] + vc[1] * tx[1]
                    }
                } else {
                    let nk = match family.as_str() {
                        "RT_R2D_Triangle" => RtR2dTri::new(p).dof2nk()[k],
                        "RT_R2D_Quadrilateral" => RtR2dQuad::new(p).dof2nk()[k],
                        other => panic!("{other}"),
                    };
                    let z_nk = if family.contains("Triangle") { 3 } else { 4 };
                    if nk == z_nk {
                        jac.det * vc[2]
                    } else {
                        let n = normal(&family, nk);
                        // adjJ = [j11, -j01; -j10, j00]; dof = n^T adjJ vc
                        let ajv = [
                            jac.j11 * vc[0] - jac.j01 * vc[1],
                            -jac.j10 * vc[0] + jac.j00 * vc[1],
                        ];
                        n[0] * ajv[0] + n[1] * ajv[1]
                    }
                };
                check(&mut worst, &mut n_checked, got, want, format!("{family} proj k{k}"));
            }
            _ => {}
        }
    }
    assert!(
        n_checked > 800,
        "too few checks ran ({n_checked}) — fixture truncated?"
    );
    println!(
        "d102 probe: {n_checked} checks vs MFEM 4.10 oracle, worst rel diff = {worst:.3e}"
    );
    // Round-102 acceptance line (工程容差级): keep the honest number visible.
    assert!(worst <= 1e-10, "worst rel diff {worst:.3e} > 1e-10");
}
