//! TMOP metric magnitude — 1:1 port of MFEM miniapps/tools/tmop-metric-magnitude.cpp
//!
//! Tracks how TMOP metrics change under geometric perturbations.  The C++
//! program evaluates `metric->EvalW(J)` for the perturbed Jacobian `J` and
//! prints `Magnitude of metric <id>: <value>` followed by the three
//! perturbation factors; this port reproduces that line exactly (6 significant
//! digits, `fem_solver::fmt_g`).
//!
//! Metric-zoo gap (round 32 measurement): the C++ switch accepts
//! `{1,2,7,9,14,50,55,56,58,77,85,98}` (2-D), `{301,302,303,304,315,316,321,322,323,360}`
//! (3-D) and the A-metrics `{11,36,107}`; `fem_mesh::tmop` implements all of
//! them **except `85`, `98`, `322`, `11`, `36`, `107`** (6 of 25 ids).  Those
//! ids print `Unknown metric_id: <id>` and exit with code 3, the same code
//! MFEM's `default:` branch returns for an id it does not know.

use fem_mesh::tmop::{
    TmopQualityMetric,
    TmopMetric001, TmopMetric002, TmopMetric007, TmopMetric009,
    TmopMetric014, TmopMetric050, TmopMetric055, TmopMetric056,
    TmopMetric058, TmopMetric077,
    TmopMetric301, TmopMetric302, TmopMetric303, TmopMetric304,
    TmopMetric315, TmopMetric316, TmopMetric321, TmopMetric323, TmopMetric360,
};
use fem_solver::fmt_g;

use std::f64::consts::PI;

/// Form 2D perturbed Jacobian.
fn form_2d_jac(perturb_v: f64, perturb_ar: f64, perturb_s: f64) -> [[f64; 2]; 2] {
    let volume: f64 = 1.0 * perturb_v;
    let a_r: f64 = 1.0 * perturb_ar;
    let skew_angle: f64 = PI / 2.0 / perturb_s;

    // Aspect ratio matrix
    let m_ar = [[1.0 / a_r.sqrt(), 0.0], [0.0, a_r.sqrt()]];

    // Skew matrix
    let m_skew = [[1.0, skew_angle.cos()], [0.0, skew_angle.sin()]];

    // Rotation (identity for now)
    let m_rot = [[1.0, 0.0], [0.0, 1.0]];

    // J = M_rot * M_skew * M_ar * sqrt(volume / sin(skew_angle))
    let tmp = mat_mul_2x2(&m_rot, &m_skew);
    let j = mat_mul_2x2(&tmp, &m_ar);
    let scale = (volume / skew_angle.sin()).sqrt();

    [[j[0][0] * scale, j[0][1] * scale], [j[1][0] * scale, j[1][1] * scale]]
}

/// Form 3D perturbed Jacobian.
fn form_3d_jac(perturb_v: f64, perturb_ar: f64, perturb_s: f64) -> [[f64; 3]; 3] {
    let volume: f64 = 1.0 * perturb_v;
    let ar_1: f64 = 1.0 * perturb_ar;
    let ar_2: f64 = 1.0;
    let ar_3: f64 = 1.0;

    let skew_angle_12: f64 = PI / 2.0 / perturb_s;
    let skew_angle_13: f64 = PI / 2.0;
    let skew_angle_23: f64 = PI / 2.0;

    let j = [
        [
            ar_1.powf(1.0 / 3.0),
            ar_2.powf(1.0 / 3.0) * skew_angle_12.cos(),
            ar_3.powf(1.0 / 3.0) * skew_angle_13.cos(),
        ],
        [
            0.0,
            ar_2.powf(1.0 / 3.0) * skew_angle_12.sin(),
            ar_3.powf(1.0 / 3.0) * skew_angle_13.sin() * skew_angle_23.cos(),
        ],
        [
            0.0,
            0.0,
            ar_3.powf(1.0 / 3.0) * skew_angle_13.sin() * skew_angle_23.sin(),
        ],
    ];

    let sin3: f64 = skew_angle_12.sin() * skew_angle_13.sin() * skew_angle_23.sin();
    let ar3: f64 = ar_1.powf(1.0 / 3.0) * ar_2.powf(1.0 / 3.0) * ar_3.powf(1.0 / 3.0);
    let scale: f64 = (volume / (sin3 * ar3)).powf(1.0 / 3.0);

    [
        [j[0][0] * scale, j[0][1] * scale, j[0][2] * scale],
        [j[1][0] * scale, j[1][1] * scale, j[1][2] * scale],
        [j[2][0] * scale, j[2][1] * scale, j[2][2] * scale],
    ]
}

fn mat_mul_2x2(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut metric_id = 2;
    let mut perturb_v = 1.0;
    let mut perturb_ar = 1.0;
    let mut perturb_s = 1.0;

    // Simple argument parsing
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-mid" | "--metric-id" => {
                i += 1;
                if i < args.len() {
                    metric_id = args[i].parse().expect("Invalid metric_id");
                }
            }
            "-pv" | "--perturb-factor-volume" => {
                i += 1;
                if i < args.len() {
                    perturb_v = args[i].parse().expect("Invalid perturb_v");
                }
            }
            "-par" | "--perturb-factor-aspect-ratio" => {
                i += 1;
                if i < args.len() {
                    perturb_ar = args[i].parse().expect("Invalid perturb_ar");
                }
            }
            "-ps" | "--perturb-factor-skew" => {
                i += 1;
                if i < args.len() {
                    perturb_s = args[i].parse().expect("Invalid perturb_s");
                }
            }
            _ => {}
        }
        i += 1;
    }

    if !(perturb_v > 0.0 && perturb_ar > 0.0 && perturb_s >= 1.0) {
        // C++: MFEM_VERIFY(perturb_v > 0.0 && perturb_ar > 0.0 && perturb_s >= 1.0, "Invalid input")
        eprintln!("MFEM_VERIFY failed: Invalid input");
        std::process::exit(3);
    }

    let dim = if metric_id < 300 { 2 } else { 3 };

    if dim == 2 {
        let j = form_2d_jac(perturb_v, perturb_ar, perturb_s);
        let j_arr = [[j[0][0], j[0][1]], [j[1][0], j[1][1]]];

        let metric: Box<dyn TmopQualityMetric> = match metric_id {
            1 => Box::new(TmopMetric001),
            2 => Box::new(TmopMetric002),
            7 => Box::new(TmopMetric007),
            9 => Box::new(TmopMetric009),
            14 => Box::new(TmopMetric014),
            50 => Box::new(TmopMetric050),
            55 => Box::new(TmopMetric055),
            56 => Box::new(TmopMetric056),
            58 => Box::new(TmopMetric058),
            77 => Box::new(TmopMetric077),
            // 85 / 98 / 11 / 36 / 107 exist in C++ (`TMOP_Metric_085`,
            // `TMOP_Metric_098`, `TMOP_AMetric_011/036/107`) but have no
            // `fem_mesh::tmop` implementation — same exit code as MFEM's
            // unknown-id branch, plus an explicit gap note on stderr.
            _ => unknown_metric(metric_id, "2-D"),
        };

        let w = metric.eval_w(&j_arr);
        println!("Magnitude of metric {metric_id}: {}", fmt_g(w));
    } else {
        let j = form_3d_jac(perturb_v, perturb_ar, perturb_s);
        let j_arr = [
            [j[0][0], j[0][1], j[0][2]],
            [j[1][0], j[1][1], j[1][2]],
            [j[2][0], j[2][1], j[2][2]],
        ];

        let metric: Box<dyn fem_mesh::tmop::TmopQualityMetric3D> = match metric_id {
            301 => Box::new(TmopMetric301),
            302 => Box::new(TmopMetric302),
            303 => Box::new(TmopMetric303),
            304 => Box::new(TmopMetric304),
            315 => Box::new(TmopMetric315),
            316 => Box::new(TmopMetric316),
            321 => Box::new(TmopMetric321),
            323 => Box::new(TmopMetric323),
            360 => Box::new(TmopMetric360),
            // 322 is in the C++ switch but not implemented in `fem_mesh::tmop`.
            _ => unknown_metric(metric_id, "3-D"),
        };

        let w = metric.eval_w(&j_arr);
        println!("Magnitude of metric {metric_id}: {}", fmt_g(w));
    }

    println!("  volume perturbation factor: {}", fmt_g(perturb_v));
    println!("  aspect ratio pert factor:   {}", fmt_g(perturb_ar));
    println!("  skew perturbation factor:   {}", fmt_g(perturb_s));
}

/// C++ `default:` branch of the metric switch (`cout << "Unknown metric_id: "
/// << metric_id << endl; return 3;`).
fn unknown_metric(metric_id: i32, dim: &str) -> ! {
    println!("Unknown metric_id: {metric_id}");
    eprintln!(
        "tmop-metric-magnitude (Rust port): the C++ program accepts {dim} metric id \
         {metric_id}, but `fem_mesh::tmop` has no implementation for it. Missing ids \
         overall (present in C++ miniapps/tools/tmop-metric-magnitude.cpp, absent in \
         fem-rs): 85, 98, 322 (T-metrics) and 11, 36, 107 (A-metrics). Exit 3."
    );
    std::process::exit(3);
}
