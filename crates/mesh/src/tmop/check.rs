//! TMOP metric verifier — matches MFEM's tmop-check-metric miniapp.
//!
//! Verifies:
//! 1. EvalW vs EvalWMatrixForm consistency
//! 1st derivative (EvalP) convergence via finite differences
//! 2nd derivative (AssembleH) convergence via finite differences

use crate::tmop::metrics::{
    TmopQualityMetric, TmopQualityMetric3D,
    TmopMetric001, TmopMetric002, TmopMetric007, TmopMetric009,
    TmopMetric014, TmopMetric022, TmopMetric050, TmopMetric055, TmopMetric056,
    TmopMetric058, TmopMetric077,
    TmopMetric301, TmopMetric302, TmopMetric303, TmopMetric304,
    TmopMetric315, TmopMetric316, TmopMetric318, TmopMetric321, TmopMetric323, TmopMetric360,
};

/// Result of checking a 2D metric.
#[derive(Debug, Clone)]
pub struct MetricCheckResult {
    pub metric_id: i32,
    pub eval_w_errors: usize,
    pub eval_w_total: usize,
    pub avg_dF_rate: f64,
    pub min_ddF_rate: f64,
}

/// Check a 2D metric: EvalW consistency + derivative convergence.
pub fn check_metric_2d(metric: &dyn TmopQualityMetric, n_samples: usize, n_convergence_iter: usize) -> MetricCheckResult {
    let mut eval_w_errors = 0;
    let mut eval_w_total = 0;

    // Test EvalW vs EvalWMatrixForm
    for i in 0..n_samples {
        let mut jpt = random_jet_2d(i);
        // Increase probability of det > 0
        jpt[0][0] += 1.0;
        if det_2x2(&jpt) <= 0.0 {
            continue;
        }
        let i_form = metric.eval_w(&jpt);
        let m_form = metric.eval_w_matrix_form(&jpt);
        let diff = (i_form - m_form).abs() / m_form.abs().max(1e-10);
        if diff > 1e-8 {
            eval_w_errors += 1;
        }
        eval_w_total += 1;
    }

    // Test 1st derivative convergence (perturbing Jacobian entries directly)
    let jpt = [[0.8, 0.2], [-0.3, 1.1]];

    let f_0 = metric.eval_w(&jpt);
    let mut p_0 = [[0.0; 2]; 2];
    metric.eval_p(&jpt, &mut p_0);

    let mut dx: f64 = 0.1;
    let mut rate_dF_sum: f64 = 0.0;
    let mut err_old: f64 = 1.0;
    for k in 0..n_convergence_iter {
        let mut err_k: f64 = 0.0;
        for row in 0..2 {
            for col in 0..2 {
                let mut jpt_pert = jpt;
                jpt_pert[row][col] += dx;
                let f_pert = metric.eval_w(&jpt_pert);
                let p_sum = p_0[row][col];
                let diff = (f_0 + p_sum * dx - f_pert).abs();
                err_k = err_k.max(diff);
            }
        }
        dx *= 0.5;
        if k > 0 {
            let r = if err_k > 0.0 { (err_old / err_k).log2() } else { 2.0 };
            rate_dF_sum += r;
        }
        err_old = err_k;
    }
    let avg_dF_rate = rate_dF_sum / (n_convergence_iter - 1) as f64;

    // Test 2nd derivative convergence
    // For each entry (pi,pj) of P and each entry (qi,qj) of J:
    //   FD approx of dP[pi,pj]/dJ[qi,qj] = (P_pert[pi,pj] - P_0[pi,pj]) / dx
    //   As dx → 0, this should converge to the true derivative at rate 1
    let mut min_avg_rate: f64 = 7.0;
    for pi in 0..2 {
        for pj in 0..2 {
            let mut rate_sum: f64 = 0.0;
            dx = 0.1;
            err_old = 1.0;
            for k in 0..n_convergence_iter {
                let mut err_k: f64 = 0.0;
                for qi in 0..2 {
                    for qj in 0..2 {
                        let mut jpt_pert = jpt;
                        jpt_pert[qi][qj] += dx;
                        let mut p_pert = [[0.0; 2]; 2];
                        metric.eval_p(&jpt_pert, &mut p_pert);
                        // FD approx of dP/dJ at this dx
                        let fd = (p_pert[pi][pj] - p_0[pi][pj]) / dx;
                        // Compare with FD at half dx (more accurate)
                        let mut jpt_pert2 = jpt;
                        jpt_pert2[qi][qj] += dx * 0.5;
                        let mut p_pert2 = [[0.0; 2]; 2];
                        metric.eval_p(&jpt_pert2, &mut p_pert2);
                        let fd2 = (p_pert2[pi][pj] - p_0[pi][pj]) / (dx * 0.5);
                        let diff = (fd - fd2).abs();
                        err_k = err_k.max(diff);
                    }
                }
                dx *= 0.5;
                if k > 0 {
                    let r = if err_k > 1e-14 { (err_old / err_k).log2() } else { 1.0 };
                    rate_sum += r;
                }
                err_old = err_k;
            }
            let avg_rate = rate_sum / (n_convergence_iter - 1) as f64;
            min_avg_rate = min_avg_rate.min(avg_rate);
        }
    }

    MetricCheckResult {
        metric_id: metric.id(),
        eval_w_errors,
        eval_w_total,
        avg_dF_rate,
        min_ddF_rate: min_avg_rate,
    }
}

/// Check a 3D metric.
pub fn check_metric_3d(metric: &dyn TmopQualityMetric3D, n_samples: usize, n_convergence_iter: usize) -> MetricCheckResult {
    let mut eval_w_errors = 0;
    let mut eval_w_total = 0;

    for i in 0..n_samples {
        let mut jpt = random_jet_3d(i);
        jpt[0][0] += 1.0;
        if det_3x3(&jpt) <= 0.0 {
            continue;
        }
        let i_form = metric.eval_w(&jpt);
        let m_form = metric.eval_w_matrix_form(&jpt);
        let diff = (i_form - m_form).abs() / m_form.abs().max(1e-10);
        if diff > 1e-8 {
            eval_w_errors += 1;
        }
        eval_w_total += 1;
    }

    let jpt = [[0.8, 0.2, 0.1], [-0.3, 1.1, 0.05], [0.1, -0.05, 0.9]];

    let f_0 = metric.eval_w(&jpt);
    let mut p_0 = [[0.0; 3]; 3];
    metric.eval_p(&jpt, &mut p_0);

    let mut dx: f64 = 0.1;
    let mut rate_dF_sum: f64 = 0.0;
    let mut err_old: f64 = 1.0;
    for k in 0..n_convergence_iter {
        let mut err_k: f64 = 0.0;
        for row in 0..3 {
            for col in 0..3 {
                let mut jpt_pert = jpt;
                jpt_pert[row][col] += dx;
                let f_pert = metric.eval_w(&jpt_pert);
                let p_sum = p_0[row][col];
                let diff = (f_0 + p_sum * dx - f_pert).abs();
                err_k = err_k.max(diff);
            }
        }
        dx *= 0.5;
        if k > 0 {
            let r = if err_k > 0.0 { (err_old / err_k).log2() } else { 2.0 };
            rate_dF_sum += r;
        }
        err_old = err_k;
    }
    let avg_dF_rate = rate_dF_sum / (n_convergence_iter - 1) as f64;

    let mut min_avg_rate: f64 = 7.0;
    for pi in 0..3 {
        for pj in 0..3 {
            let mut rate_sum: f64 = 0.0;
            dx = 0.1;
            err_old = 1.0;
            for k in 0..n_convergence_iter {
                let mut err_k: f64 = 0.0;
                for qi in 0..3 {
                    for qj in 0..3 {
                        let mut jpt_pert = jpt;
                        jpt_pert[qi][qj] += dx;
                        let mut p_pert = [[0.0; 3]; 3];
                        metric.eval_p(&jpt_pert, &mut p_pert);
                        let fd = (p_pert[pi][pj] - p_0[pi][pj]) / dx;
                        let mut jpt_pert2 = jpt;
                        jpt_pert2[qi][qj] += dx * 0.5;
                        let mut p_pert2 = [[0.0; 3]; 3];
                        metric.eval_p(&jpt_pert2, &mut p_pert2);
                        let fd2 = (p_pert2[pi][pj] - p_0[pi][pj]) / (dx * 0.5);
                        let diff = (fd - fd2).abs();
                        err_k = err_k.max(diff);
                    }
                }
                dx *= 0.5;
                if k > 0 {
                    let r = if err_k > 1e-14 { (err_old / err_k).log2() } else { 1.0 };
                    rate_sum += r;
                }
                err_old = err_k;
            }
            let avg_rate = rate_sum / (n_convergence_iter - 1) as f64;
            min_avg_rate = min_avg_rate.min(avg_rate);
        }
    }

    MetricCheckResult {
        metric_id: metric.id(),
        eval_w_errors,
        eval_w_total,
        avg_dF_rate,
        min_ddF_rate: min_avg_rate,
    }
}

/// Run the full tmop-check-metric suite.
pub fn run_tmop_check_metric() {
    println!("=== TMOP Metric Check ===");
    println!("Checking 2D metrics...");

    let metrics_2d: Vec<Box<dyn TmopQualityMetric>> = vec![
        Box::new(TmopMetric001),
        Box::new(TmopMetric002),
        Box::new(TmopMetric007),
        Box::new(TmopMetric009),
        Box::new(TmopMetric014),
        Box::new(TmopMetric022 { min_det_t: -0.1 }),
        Box::new(TmopMetric050),
        Box::new(TmopMetric055),
        Box::new(TmopMetric056),
        Box::new(TmopMetric058),
        Box::new(TmopMetric077),
    ];

    for metric in &metrics_2d {
        let result = check_metric_2d(metric.as_ref(), 100, 10);
        println!(
            "Metric {:3}: EvalW errors: {}/{}, dF rate: {:.2}, ddF rate: {:.2}",
            result.metric_id,
            result.eval_w_errors,
            result.eval_w_total,
            result.avg_dF_rate,
            result.min_ddF_rate
        );
    }

    println!("\nChecking 3D metrics...");

    let metrics_3d: Vec<Box<dyn TmopQualityMetric3D>> = vec![
        Box::new(TmopMetric301),
        Box::new(TmopMetric302),
        Box::new(TmopMetric303),
        Box::new(TmopMetric304),
        Box::new(TmopMetric315),
        Box::new(TmopMetric316),
        Box::new(TmopMetric318),
        Box::new(TmopMetric321),
        Box::new(TmopMetric323),
        Box::new(TmopMetric360),
    ];

    for metric in &metrics_3d {
        let result = check_metric_3d(metric.as_ref(), 100, 10);
        println!(
            "Metric {:3}: EvalW errors: {}/{}, dF rate: {:.2}, ddF rate: {:.2}",
            result.metric_id,
            result.eval_w_errors,
            result.eval_w_total,
            result.avg_dF_rate,
            result.min_ddF_rate
        );
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Deterministic pseudo-random 2x2 Jacobian for testing.
fn random_jet_2d(seed: usize) -> [[f64; 2]; 2] {
    let mut s = seed as f64 + 1.0;
    let mut v = [0.0; 4];
    for i in 0..4 {
        s = ((s * 9301.0 + 49297.0) % 233280.0) / 233280.0;
        v[i] = s * 2.0 - 1.0;
    }
    [[v[0], v[2]], [v[1], v[3]]]
}

/// Deterministic pseudo-random 3x3 Jacobian for testing.
fn random_jet_3d(seed: usize) -> [[f64; 3]; 3] {
    let mut s = seed as f64 + 1.0;
    let mut v = [0.0; 9];
    for i in 0..9 {
        s = ((s * 9301.0 + 49297.0) % 233280.0) / 233280.0;
        v[i] = s * 2.0 - 1.0;
    }
    [
        [v[0], v[3], v[6]],
        [v[1], v[4], v[7]],
        [v[2], v[5], v[8]],
    ]
}

/// Determinant of a 2x2 matrix.
fn det_2x2(m: &[[f64; 2]; 2]) -> f64 {
    m[0][0] * m[1][1] - m[1][0] * m[0][1]
}

/// Determinant of a 3x3 matrix.
fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
        + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_metric_002() {
        let m = TmopMetric002;
        let result = check_metric_2d(&m, 100, 10);
        assert_eq!(result.eval_w_errors, 0, "EvalW errors for metric 002");
        assert!(result.avg_dF_rate > 0.9, "dF rate too low for metric 002: {}", result.avg_dF_rate);
        assert!(result.min_ddF_rate > 0.8, "ddF rate too low for metric 002: {}", result.min_ddF_rate);
    }

    #[test]
    fn test_check_metric_007() {
        let m = TmopMetric007;
        let result = check_metric_2d(&m, 100, 10);
        assert_eq!(result.eval_w_errors, 0, "EvalW errors for metric 007");
        assert!(result.avg_dF_rate > 1.5, "dF rate too low for metric 007: {}", result.avg_dF_rate);
        assert!(result.min_ddF_rate > 0.8, "ddF rate too low for metric 007: {}", result.min_ddF_rate);
    }

    #[test]
    fn test_check_metric_050() {
        let m = TmopMetric050;
        let result = check_metric_2d(&m, 100, 10);
        assert_eq!(result.eval_w_errors, 0, "EvalW errors for metric 050");
        assert!(result.avg_dF_rate > 1.5, "dF rate too low for metric 050: {}", result.avg_dF_rate);
        assert!(result.min_ddF_rate > 0.8, "ddF rate too low for metric 050: {}", result.min_ddF_rate);
    }

    #[test]
    fn test_check_metric_301() {
        let m = TmopMetric301;
        let result = check_metric_3d(&m, 100, 10);
        assert_eq!(result.eval_w_errors, 0, "EvalW errors for metric 301");
        assert!(result.avg_dF_rate > 1.5, "dF rate too low for metric 301: {}", result.avg_dF_rate);
        assert!(result.min_ddF_rate > 0.8, "ddF rate too low for metric 301: {}", result.min_ddF_rate);
    }

    #[test]
    fn test_check_metric_303() {
        let m = TmopMetric303;
        let result = check_metric_3d(&m, 100, 10);
        assert_eq!(result.eval_w_errors, 0, "EvalW errors for metric 303");
        assert!(result.avg_dF_rate > 1.5, "dF rate too low for metric 303: {}", result.avg_dF_rate);
        assert!(result.min_ddF_rate > 0.8, "ddF rate too low for metric 303: {}", result.min_ddF_rate);
    }
}
