//! TMOP metric checker — 1:1 CLI of MFEM `miniapps/tools/tmop-check-metric.cpp`
//! (MFEM 4.10), extended with a `-dump` comparison mode (round 102).
//!
//! ## Modes
//!
//! * `-mid <id>` without `-dump`: run the fem-rs metric self-check (EvalW
//!   matrix-form consistency + finite-difference convergence of EvalP /
//!   AssembleH on deterministic Jacobian samples, `fem_mesh::tmop::check`),
//!   then print the three `---` summary lines with the same wording as the
//!   C++ miniapp. (The C++ program drives its FD checks through the FE-space
//!   `TMOP_Integrator`; fem-rs checks at the metric level, which is the
//!   deliverable of the metric port.)
//! * `-mid <id> -dump <file>`: read a record dump produced by the round-102
//!   C++ oracle (`tmp/d102tmop/d102_dump_metric.cpp`, built against
//!   mfem410_ser), rebuild the same metric through the `fem_assembly` `-mid`
//!   factory and compare EvalW / EvalP / AssembleH per record. Prints the
//!   worst relative differences and exits 0 (all ≤ 1e-10) or 4 (tolerance
//!   exceeded).
//!
//! Metric ids accepted by the C++ switch in `tmop-check-metric.cpp` plus the
//! round-102 completions (mesh-optimizer ids 0/4/66, A-combo 49, untanglers
//! 211/252/311/352 — all real MFEM 4.10 `TMOP_Metric` classes); anything else
//! prints `Unknown metric_id: <id>` and exits with code 3, exactly like
//! MFEM's `default:` branch.
//!
//! Parameters follow the MFEM drivers: tauval = -0.1 (metrics 22/252/313/352),
//! gamma = 0.5 (66/80/332/333/334/347) / 0.9 (49/126), eps = 1e-4 (211/311).
//! The `-A` flag is accepted for CLI parity; metrics 14/50 map to the
//! T-metric versions used by mesh-optimizer (the A-versions are ids 11 and
//! 107's siblings `TMOP_AMetric_014/050`, exercised through 49/126).

use fem_assembly::tmop_form::{metric_from_id_2d, metric_from_id_3d, SharedMinDet, TmopMetric};
use fem_mesh::{check_metric_2d, check_metric_3d};
use std::process::exit;

/// The C++ switch list (tmop-check-metric.cpp) plus the round-102 additions.
const CPP_IDS: &[i32] = &[
    // T-metrics, 2-D
    0, 1, 2, 4, 7, 9, 14, 22, 50, 55, 56, 58, 66, 77, 80, 85, 90, 94, 98, 211, 252,
    // T-metrics, 3-D
    301, 302, 303, 304, 311, 313, 315, 316, 318, 321, 322, 323, 328, 332, 333, 334, 338, 342,
    347, 352, 360,
    // A-metrics
    11, 36, 49, 51, 107, 126,
];

/// One record of the C++ oracle dump.
struct DumpRecord {
    jpt: Vec<f64>,
    w: Vec<f64>,
    ds: Vec<f64>,
    weight: f64,
    ew: f64,
    ep: Option<Vec<f64>>,
    eh: Option<Vec<f64>>,
}

struct Dump {
    dim: usize,
    id: i32,
    deriv: bool,
    records: Vec<DumpRecord>,
}

fn parse_dump(path: &str) -> Result<Dump, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut dim = 0usize;
    let mut id = 0i32;
    let mut deriv = false;
    let mut records: Vec<DumpRecord> = Vec::new();
    let mut cur: Option<DumpRecord> = None;
    for line in text.lines() {
        let ws: Vec<&str> = line.split_whitespace().collect();
        if ws.is_empty() {
            continue;
        }
        match ws[0] {
            "D102DUMP" | "MODE" | "END" => {}
            "DIM" => dim = ws[1].parse().map_err(|e| format!("DIM: {e}"))?,
            "MID" => id = ws[1].parse().map_err(|e| format!("MID: {e}"))?,
            "DERIV" => deriv = ws[1] == "1",
            "RECORD" => {
                if let Some(c) = cur.take() {
                    records.push(c);
                }
                cur = Some(DumpRecord {
                    jpt: Vec::new(),
                    w: Vec::new(),
                    ds: Vec::new(),
                    weight: 0.0,
                    ew: 0.0,
                    ep: None,
                    eh: None,
                })
            }
            "JPT" | "W" | "DS" => {
                let c = cur.as_mut().ok_or("field before RECORD")?;
                let v: Vec<f64> = ws[1..]
                    .iter()
                    .map(|s| s.parse::<f64>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("{}: {e}", ws[0]))?;
                match ws[0] {
                    "JPT" => c.jpt = v,
                    "W" => c.w = v,
                    _ => c.ds = v,
                }
            }
            "WEIGHT" => {
                let c = cur.as_mut().ok_or("field before RECORD")?;
                c.weight = ws[1].parse().map_err(|e| format!("WEIGHT: {e}"))?;
            }
            "EW" => {
                let c = cur.as_mut().ok_or("field before RECORD")?;
                c.ew = ws[1].parse().map_err(|e| format!("EW: {e}"))?;
            }
            "EP" => {
                let c = cur.as_mut().ok_or("field before RECORD")?;
                c.ep = Some(
                    ws[1..]
                        .iter()
                        .map(|s| s.parse::<f64>())
                        .collect::<Result<_, _>>()
                        .map_err(|e| format!("EP: {e}"))?,
                );
            }
            "EH" => {
                let c = cur.as_mut().ok_or("field before RECORD")?;
                c.eh = Some(
                    ws[1..]
                        .iter()
                        .map(|s| s.parse::<f64>())
                        .collect::<Result<_, _>>()
                        .map_err(|e| format!("EH: {e}"))?,
                );
            }
            other => return Err(format!("unexpected dump field: {other}")),
        }
    }
    if let Some(c) = cur.take() {
        records.push(c);
    }
    if dim != 2 && dim != 3 {
        return Err(format!("bad DIM {dim}"));
    }
    Ok(Dump { dim, id, deriv, records })
}

/// Max relative difference between two flat arrays, scaled by the reference
/// magnitude `max(|ref|)` (0 when both are identically zero).
fn flat_rel_diff(rust: &[f64], cpp: &[f64]) -> f64 {
    let scale = cpp.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let (mut worst, mut abs_worst) = (0.0f64, 0.0f64);
    for (a, b) in rust.iter().zip(cpp.iter()) {
        let d = (a - b).abs();
        worst = worst.max(if scale > 0.0 { d / scale } else { 0.0 });
        abs_worst = abs_worst.max(d);
    }
    // Absolute tolerance for references that cancel to ~0 (1-ulp noise on an
    // exactly-zero mathematical value is not a port difference).
    if abs_worst <= 1e-12 {
        0.0
    } else {
        worst
    }
}

/// Engineering-tolerance comparison: pass when the absolute difference is at
/// or below 1e-12 (near-zero references, e.g. mu_360 on an isotropic Jacobian
/// where EvalW cancels to 0 up to 1 ulp) or the relative difference is at or
/// below 1e-10.
fn scalar_rel_diff(a: f64, b: f64) -> f64 {
    let abs = (a - b).abs();
    if abs <= 1e-12 {
        return 0.0;
    }
    if b == 0.0 {
        return f64::INFINITY;
    }
    abs / b.abs()
}

fn compare_dump(dump: &Dump) -> bool {
    let min_det = SharedMinDet::new(-0.1);
    let metric = if dump.dim == 2 {
        metric_from_id_2d(dump.id, &min_det)
    } else {
        metric_from_id_3d(dump.id, &min_det)
    };
    let metric = match metric {
        Some(m) => m,
        None => {
            println!("metric_from_id: unsupported id {}", dump.id);
            return false;
        }
    };

    let mut worst_ew = 0.0f64;
    let mut worst_ep = 0.0f64;
    let mut worst_eh = 0.0f64;

    for (r, rec) in dump.records.iter().enumerate() {
        // The nodal-FE dof count varies per dump (straight meshes use linear
        // nodes, curved meshes carry the order-q nodal space): derive it from
        // the DS width instead of assuming 2^dim.
        let nd = rec.ds.len() / dump.dim;
        let jpt: Vec<f64> = rec.jpt.clone();
        let w: Vec<f64> = rec.w.clone();
        let ew = match (&metric, dump.dim) {
            (TmopMetric::D2(m), 2) => {
                let jpt2 = [[jpt[0], jpt[2]], [jpt[1], jpt[3]]];
                let w2 = [[w[0], w[2]], [w[1], w[3]]];
                m.set_target_jacobian(&w2);
                // EvalW
                let ew = m.eval_w(&jpt2);
                let d_ew = scalar_rel_diff(ew, rec.ew);
                worst_ew = worst_ew.max(d_ew);
                if let Some(ep) = &rec.ep {
                    let mut p = [[0.0f64; 2]; 2];
                    m.eval_p(&jpt2, &mut p);
                    let pr: Vec<f64> = vec![p[0][0], p[1][0], p[0][1], p[1][1]];
                    worst_ep = worst_ep.max(flat_rel_diff(&pr, ep));
                }
                if let Some(eh) = &rec.eh {
                    let ds_rows: Vec<[f64; 2]> =
                        (0..nd).map(|i| [rec.ds[i], rec.ds[i + nd]]).collect();
                    let mut a = vec![0.0f64; nd * 2 * nd * 2];
                    m.assemble_h(&jpt2, &ds_rows, rec.weight, &mut a);
                    worst_eh = worst_eh.max(flat_rel_diff(&a, eh));
                }
                ew
            }
            (TmopMetric::D3(m), 3) => {
                let jpt3 = [
                    [jpt[0], jpt[3], jpt[6]],
                    [jpt[1], jpt[4], jpt[7]],
                    [jpt[2], jpt[5], jpt[8]],
                ];
                let ew = m.eval_w(&jpt3);
                let d_ew = scalar_rel_diff(ew, rec.ew);
                worst_ew = worst_ew.max(d_ew);
                if let Some(ep) = &rec.ep {
                    let mut p = [[0.0f64; 3]; 3];
                    m.eval_p(&jpt3, &mut p);
                    let pr: Vec<f64> = (0..9).map(|k| p[k % 3][k / 3]).collect();
                    worst_ep = worst_ep.max(flat_rel_diff(&pr, ep));
                }
                if let Some(eh) = &rec.eh {
                    let ds_rows: Vec<[f64; 3]> = (0..nd)
                        .map(|i| [rec.ds[i], rec.ds[i + nd], rec.ds[i + 2 * nd]])
                        .collect();
                    let mut a = vec![0.0f64; nd * 3 * nd * 3];
                    m.assemble_h(&jpt3, &ds_rows, rec.weight, &mut a);
                    worst_eh = worst_eh.max(flat_rel_diff(&a, eh));
                }
                ew
            }
            _ => {
                println!("metric dimension mismatch at record {r}");
                return false;
            }
        };
        if !ew.is_finite() && rec.ew.is_finite() && rec.ew != 0.0 {
            println!("non-finite EvalW at record {r}");
            return false;
        }
    }

    let ok = worst_ew <= 1e-10 && worst_ep <= 1e-10 && worst_eh <= 1e-10;
    println!(
        "--- EvalW:     worst rel diff: {:.3e} ({} records)",
        worst_ew,
        dump.records.len()
    );
    if dump.deriv {
        println!("--- EvalP:     worst rel diff: {:.3e}", worst_ep);
        println!("--- AssembleH: worst rel diff: {:.3e}", worst_eh);
    } else {
        println!(
            "--- EvalP/AssembleH: not compared (metric {} has no C++ derivative path; \
             fem-rs implements them analytically, FD-verified in the d102 tests)",
            dump.id
        );
    }
    ok
}

fn self_check(id: i32) -> bool {
    let min_det = SharedMinDet::new(-0.1);
    match metric_from_id_2d(id, &min_det) {
        Some(TmopMetric::D2(m)) => {
            let res = check_metric_2d(m.as_ref(), 100, 10);
            println!(
                "--- EvalW:     {} errors out of {} comparisons with det(T) > 0.",
                res.eval_w_errors, res.eval_w_total
            );
            println!(
                "--- EvalP:     avg rate of convergence (should be 2): {:.5}",
                res.avg_dF_rate
            );
            println!(
                "--- AssembleH: avg rate of convergence (should be 2): {:.5}",
                res.min_ddF_rate
            );
            res.eval_w_errors == 0 && res.avg_dF_rate > 1.5 && res.min_ddF_rate > 1.5
        }
        Some(TmopMetric::D3(m)) => {
            let res = check_metric_3d(m.as_ref(), 100, 10);
            println!(
                "--- EvalW:     {} errors out of {} comparisons with det(T) > 0.",
                res.eval_w_errors, res.eval_w_total
            );
            println!(
                "--- EvalP:     avg rate of convergence (should be 2): {:.5}",
                res.avg_dF_rate
            );
            println!(
                "--- AssembleH: avg rate of convergence (should be 2): {:.5}",
                res.min_ddF_rate
            );
            res.eval_w_errors == 0 && res.avg_dF_rate > 1.5 && res.min_ddF_rate > 1.5
        }
        None => {
            println!("Unknown metric_id: {id}");
            exit(3);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut metric_id: i32 = 2;
    let mut a_metric_version = false;
    let mut verbose = false;
    let mut convergence_iter: i32 = 10;
    let mut dump: Option<String> = None;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-mid" | "--metric-id" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { metric_id = val; } }
            }
            "-A" | "-Ametric" => a_metric_version = true,
            "-no-A" | "--no-Ametric" => a_metric_version = false,
            "-v" | "-verbose" => verbose = true,
            "-no-v" | "--no-verbose" => verbose = false,
            "-i" | "--iterations" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { convergence_iter = val; } }
            }
            "-dump" | "--dump" => {
                match it.next() {
                    Some(v) => dump = Some(v.clone()),
                    None => { eprintln!("-dump requires a file argument"); exit(1); }
                }
            }
            // C++ `args.ParseCheck()`: OptionsParser rejects anything it does
            // not know (`Unrecognized option: <opt>` + usage + exit 1).
            other => {
                eprintln!("Unrecognized option: {other}");
                exit(1);
            }
        }
    }

    // C++ `args.PrintOptions(cout)`.
    println!("Options used:");
    println!("   --metric-id {metric_id}");
    println!("   --{}", if a_metric_version { "Ametric" } else { "no-Ametric" });
    println!("   --{}", if verbose { "verbose" } else { "no-verbose" });
    println!("   --iterations {convergence_iter}");
    if let Some(d) = &dump {
        println!("   --dump {d}");
    }

    if !CPP_IDS.contains(&metric_id) {
        // C++ `default: cout << "Unknown metric_id: " << metric_id << endl; return 3;`
        println!("Unknown metric_id: {metric_id}");
        exit(3);
    }

    let ok = match &dump {
        Some(path) => match parse_dump(path) {
            Ok(d) => {
                if d.id != metric_id {
                    println!("dump MID {} != -mid {metric_id}", d.id);
                    exit(1);
                }
                compare_dump(&d)
            }
            Err(e) => {
                println!("dump parse error: {e}");
                exit(1);
            }
        },
        None => self_check(metric_id),
    };

    if !ok {
        exit(4);
    }
}
