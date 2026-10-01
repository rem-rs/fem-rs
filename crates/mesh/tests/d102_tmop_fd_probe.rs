//! Round-102 debugging probe (temporary): FD self-consistency of a fem-rs 3D
//! metric against one record of the C++ oracle dump. Enabled only when
//! `D102_DUMP` points at a dump file (metric selectable via `D102_MID`);
//! plain `cargo test` skips it.

use fem_mesh::tmop::TmopQualityMetric3D;

#[test]
fn d102_fd_probe_3d() {
    let path = match std::env::var("D102_DUMP") {
        Ok(p) => p,
        Err(_) => return,
    };
    let mid = std::env::var("D102_MID").unwrap_or_else(|_| "315".into());
    let dump = std::fs::read_to_string(&path).unwrap();
    let mut dim = 0usize;
    let mut recs: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, Vec<f64>, Vec<f64>)> = Vec::new();
    let mut fields: Vec<Vec<f64>> = Vec::new();
    let mut weight = 0.0;
    let mut ew = 0.0;
    {
        let flush = |recs: &mut Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, Vec<f64>, Vec<f64>)>,
                     fields: &mut Vec<Vec<f64>>,
                     weight: &mut f64,
                     ew: &mut f64| {
            if fields.len() >= 5 {
                recs.push((
                    fields[0].clone(),
                    fields[1].clone(),
                    fields[2].clone(),
                    *weight,
                    *ew,
                    fields[3].clone(),
                    fields[4].clone(),
                ));
            }
            fields.clear();
        };
        for line in dump.lines() {
            let ws: Vec<&str> = line.split_whitespace().collect();
            if ws.is_empty() {
                continue;
            }
            match ws[0] {
                "DIM" => dim = ws[1].parse().unwrap(),
                "RECORD" => flush(&mut recs, &mut fields, &mut weight, &mut ew),
                "JPT" | "W" | "DS" | "EP" | "EH" => fields.push(
                    ws[1..].iter().map(|s| s.parse::<f64>().unwrap()).collect(),
                ),
                "WEIGHT" => weight = ws[1].parse().unwrap(),
                "EW" => ew = ws[1].parse().unwrap(),
                _ => {}
            }
        }
        flush(&mut recs, &mut fields, &mut weight, &mut ew);
    }
    assert_eq!(dim, 3);
    let (jpt, _w, ds, weight, _ew, ep, eh) = &recs[0];
    let nd = ds.len() / 3;

    let metric: Box<dyn TmopQualityMetric3D> = match mid.as_str() {
        "301" => Box::new(fem_mesh::tmop::TmopMetric301),
        "302" => Box::new(fem_mesh::tmop::TmopMetric302),
        "303" => Box::new(fem_mesh::tmop::TmopMetric303),
        "304" => Box::new(fem_mesh::tmop::TmopMetric304),
        "311" => Box::new(fem_mesh::tmop::TmopMetric311 { eps: 1e-4 }),
        "313" => Box::new(fem_mesh::tmop::TmopMetric313 { min_det_t: -0.1 }),
        "315" => Box::new(fem_mesh::tmop::TmopMetric315),
        "316" => Box::new(fem_mesh::tmop::TmopMetric316),
        "318" => Box::new(fem_mesh::tmop::TmopMetric318),
        "321" => Box::new(fem_mesh::tmop::TmopMetric321),
        "322" => Box::new(fem_mesh::tmop::TmopMetric322),
        "323" => Box::new(fem_mesh::tmop::TmopMetric323),
        "342" => Box::new(fem_mesh::tmop::TmopMetric342),
        "352" => Box::new(fem_mesh::tmop::TmopMetric352 { tau0: -0.1 }),
        "360" => Box::new(fem_mesh::tmop::TmopMetric360),
        other => panic!("unsupported probe metric {other}"),
    };

    let jpt3 = [[jpt[0], jpt[3], jpt[6]], [jpt[1], jpt[4], jpt[7]], [jpt[2], jpt[5], jpt[8]]];
    let mut p = [[0.0f64; 3]; 3];
    metric.eval_p(&jpt3, &mut p);
    let pr: Vec<f64> = (0..9).map(|k| p[k % 3][k / 3]).collect();
    let mut pmax = 0.0f64;
    for s in 0..9 {
        pmax = pmax.max((pr[s] - ep[s]).abs());
    }
    println!("P vs dump EP: max diff {pmax}");

    let ds_rows: Vec<[f64; 3]> =
        (0..nd).map(|i| [ds[i], ds[i + nd], ds[i + 2 * nd]]).collect();
    let mut a = vec![0.0f64; nd * 3 * nd * 3];
    metric.assemble_h(&jpt3, &ds_rows, *weight, &mut a);
    let mut amax = 0.0f64;
    for k in 0..a.len() {
        amax = amax.max((a[k] - eh[k]).abs());
    }
    println!("A vs dump EH: max diff {amax}");

    // FD dP/dJ (central differences of the metric's own EvalP).
    let h = 1e-6;
    let mut dp = vec![0.0f64; 81];
    for k in 0..9 {
        let mut jp = jpt.clone();
        let mut jm = jpt.clone();
        jp[k] += h;
        jm[k] -= h;
        let jp3 = [[jp[0], jp[3], jp[6]], [jp[1], jp[4], jp[7]], [jp[2], jp[5], jp[8]]];
        let jm3 = [[jm[0], jm[3], jm[6]], [jm[1], jm[4], jm[7]], [jm[2], jm[5], jm[8]]];
        let mut pp = [[0.0f64; 3]; 3];
        let mut pm = [[0.0f64; 3]; 3];
        metric.eval_p(&jp3, &mut pp);
        metric.eval_p(&jm3, &mut pm);
        for s in 0..9 {
            dp[s * 9 + k] = (pp[s % 3][s / 3] - pm[s % 3][s / 3]) / (2.0 * h);
        }
    }
    let mut afd = vec![0.0f64; nd * 3 * nd * 3];
    for r in 0..3 {
        for c in 0..3 {
            for rr in 0..3 {
                for cc in 0..3 {
                    let g = dp[(r + c * 3) * 9 + rr + cc * 3];
                    for i in 0..nd {
                        for j in 0..nd {
                            afd[i + r * nd + (j + rr * nd) * nd * 3] +=
                                weight * ds_rows[i][c] * ds_rows[j][cc] * g;
                        }
                    }
                }
            }
        }
    }
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    let mut worst_k = 0usize;
    for k in 0..a.len() {
        let d = (a[k] - afd[k]).abs();
        if d > worst {
            worst = d;
            worst_k = k;
        }
        scale = scale.max(afd[k].abs());
    }
    println!("EH vs FD: worst abs={worst} at entry {worst_k} scale={scale} rel={}", worst / scale);
    let mut shown = 0;
    for k in 0..a.len() {
        let d = (a[k] - afd[k]).abs();
        if d > 1e-10 && shown < 8 {
            let row = k % (3 * nd);
            let col = k / (3 * nd);
            println!("MISMATCH row={row} (dof {},c {}) col={col} (dof {},c {}): a={:+.6e} fd={:+.6e}", row % nd, row / nd, col % nd, col / nd, a[k], afd[k]);
            shown += 1;
        }
    }
    let ah = 3 * nd;
    let row = worst_k % ah;
    let col = worst_k / ah;
    println!(
        "worst entry: row={row} (dof {}, comp {}) col={col} (dof {}, comp {})",
        row % nd,
        row / nd,
        col % nd,
        col / nd
    );
    assert!(pmax < 1e-12, "P mismatch vs C++ dump");
    assert!(amax < 1e-12, "A mismatch vs C++ dump");
    assert!(worst / scale < 1e-5, "A not FD-consistent");
}

#[test]
fn d102_fd_probe_321_terms() {
    let path = match std::env::var("D102_DUMP") {
        Ok(p) => p,
        Err(_) => return,
    };
    let dump = std::fs::read_to_string(&path).unwrap();
    let mut fields: Vec<Vec<f64>> = Vec::new();
    let mut weight = 0.0;
    let mut rec: Option<(Vec<f64>, Vec<f64>, f64)> = None;
    for line in dump.lines() {
        let ws: Vec<&str> = line.split_whitespace().collect();
        if ws.is_empty() { continue; }
        match ws[0] {
            "RECORD" => { if let Some(r) = rec.take() { rec = Some(r); break; } fields.clear(); }
            "JPT" | "DS" => fields.push(ws[1..].iter().map(|s| s.parse::<f64>().unwrap()).collect()),
            "WEIGHT" => weight = ws[1].parse().unwrap(),
            _ => {}
        }
    }
    // (rebind removed: `rec` is already the take-result; the former no-op line silenced nothing.)
    let jpt = &fields[0];
    let ds = &fields[1];
    let nd = ds.len() / 3;
    let jpt3 = [[jpt[0], jpt[3], jpt[6]], [jpt[1], jpt[4], jpt[7]], [jpt[2], jpt[5], jpt[8]]];
    let ds_rows: Vec<[f64; 3]> = (0..nd).map(|i| [ds[i], ds[i + nd], ds[i + 2 * nd]]).collect();

    use fem_mesh::tmop::InvariantsEvaluator3D;
    let h = 1e-6;

    // FD of dI1/dI2/dI3b arrays.
    let di_fd = |f: &dyn Fn(&[f64; 9]) -> [f64; 9]| -> Vec<f64> {
        let mut dd = vec![0.0f64; 81];
        for c in 0..9 {
            let mut jp = jpt3;
            let mut jm = jpt3;
            // perturb flat col-major entry
            let r = c % 3;
            let cc = c / 3;
            jp[r][cc] += h;
            jm[r][cc] -= h;
            let jpv = to_flat(&jp);
            let jmv = to_flat(&jm);
            let dp = f(&jpv);
            let dm = f(&jmv);
            for a in 0..9 {
                dd[a * 9 + c] = (dp[a] - dm[a]) / (2.0 * h);
            }
        }
        dd
    };
    fn to_flat(m: &[[f64; 3]; 3]) -> [f64; 9] {
        let mut f = [0.0; 9];
        for c in 0..3 { for r in 0..3 { f[r + 3 * c] = m[r][c]; } }
        f
    }
    let contract = |dd: &[f64], w: f64| -> Vec<f64> {
        let ah = 3 * nd;
        let mut a = vec![0.0; ah * ah];
        for j in 0..3 { for s in 0..3 { for l in 0..3 { for t in 0..3 {
            let g = dd[(j + 3 * s) * 9 + (l + 3 * t)];
            for i in 0..nd { for k in 0..nd {
                a[i + nd * j + (k + nd * l) * ah] += w * ds_rows[i][s] * ds_rows[k][t] * g;
            } }
        } } } }
        a
    };

    // term 1: ddI1
    {
        let mut ie = InvariantsEvaluator3D::new(Some(&to_flat(&jpt3)));
        ie.set_derivative_matrix(nd, &ds);
        let mut a = vec![0.0f64; nd * 3 * nd * 3];
        ie.assemble_dd_i1(weight, &mut a);
        let dd = di_fd(&|j| { let mut e = InvariantsEvaluator3D::new(Some(j)); *e.get_di1() });
        let afd = contract(&dd, weight);
        let d = (0..a.len()).fold(0.0f64, |m, k| m.max((a[k] - afd[k]).abs()));
        println!("term ddI1: max |a-fd| = {d:.3e}");
    }
    // term 2: ddI2 with c1 = w/I3b^2
    {
        let flat = to_flat(&jpt3);
        let mut ie = InvariantsEvaluator3D::new(Some(&flat));
        let i3b = ie.get_i3b();
        let c1 = weight / (i3b * i3b);
        ie.set_derivative_matrix(nd, &ds);
        let mut a = vec![0.0f64; nd * 3 * nd * 3];
        ie.assemble_dd_i2(c1, &mut a);
        let dd = di_fd(&|j| { let mut e = InvariantsEvaluator3D::new(Some(j)); *e.get_di2() });
        let afd = contract(&dd, c1);
        let d = (0..a.len()).fold(0.0f64, |m, k| m.max((a[k] - afd[k]).abs()));
        println!("term ddI2: max |a-fd| = {d:.3e}");
    }
    // term 3: ddI3b with c3
    {
        let flat = to_flat(&jpt3);
        let mut ie = InvariantsEvaluator3D::new(Some(&flat));
        let i3b = ie.get_i3b();
        let i2 = ie.get_i2();
        let c2 = -2.0 * weight / (i3b * i3b * i3b);
        let c3 = c2 * i2;
        ie.set_derivative_matrix(nd, &ds);
        let mut a = vec![0.0f64; nd * 3 * nd * 3];
        ie.assemble_dd_i3b(c3, &mut a);
        let dd = di_fd(&|j| { let mut e = InvariantsEvaluator3D::new(Some(j)); *e.get_di3b() });
        let afd = contract(&dd, c3);
        let d = (0..a.len()).fold(0.0f64, |m, k| m.max((a[k] - afd[k]).abs()));
        println!("term ddI3b: max |a-fd| = {d:.3e}");
    }
    // term 4: tprod(c2, dI2, dI3b)
    {
        let flat = to_flat(&jpt3);
        let mut ie = InvariantsEvaluator3D::new(Some(&flat));
        let i3b = ie.get_i3b();
        let c2 = -2.0 * weight / (i3b * i3b * i3b);
        ie.set_derivative_matrix(nd, &ds);
        let di2 = ie.get_di2().clone();
        let di3b = ie.get_di3b().clone();
        let mut a = vec![0.0f64; nd * 3 * nd * 3];
        ie.assemble_tprod_xy(c2, &di2, &di3b, &mut a);
        // FD: d(c2 * [dI2 x dI3b + dI3b x dI2]) contraction — the second
        // derivative of the scalar c2 * dI2·(stuff)... approximate via
        // d/dJ of (c2 * (dI2 contracted) )? Use the cross-Hessian:
        // T(x,y) = c2 * Σ D dI2 D dI3b — FD of the scalar q(J) =
        // (Σ_i dI2_i * v_i) * (Σ_i dI3b_i * v_i) with v = D^T ... skip;
        // instead compare with 2*kernel? Just print kernel-only note.
        println!("term tprod: c2 = {c2:.6e} (assembled; no independent FD here)");
    }
}
