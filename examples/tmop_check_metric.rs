//! TMOP metric checker — 1:1 port of MFEM miniapps/tools/tmop-check-metric.cpp
//!
//! Verifies:
//! 1. EvalW vs EvalWMatrixForm consistency
//! 2. 1st derivative (EvalP) convergence rate
//! 3. 2nd derivative (AssembleH) convergence rate

use fem_mesh::tmop::{
    TmopQualityMetric, TmopQualityMetric3D,
    TmopMetric001, TmopMetric002, TmopMetric007, TmopMetric009,
    TmopMetric014, TmopMetric022, TmopMetric050, TmopMetric055, TmopMetric056,
    TmopMetric058, TmopMetric077,
    TmopMetric301, TmopMetric302, TmopMetric303, TmopMetric304,
    TmopMetric315, TmopMetric316, TmopMetric318, TmopMetric321, TmopMetric323, TmopMetric360,
};
use fem_mesh::check_metric_2d;
use fem_mesh::check_metric_3d;
use fem_mesh::MetricCheckResult;

fn main() {
    println!("=== TMOP Metric Check (Rust) ===\n");

    // Check 2D metrics
    println!("Checking 2D metrics...");
    let metrics_2d: Vec<(&str, Box<dyn TmopQualityMetric>)> = vec![
        ("001", Box::new(TmopMetric001)),
        ("002", Box::new(TmopMetric002)),
        ("007", Box::new(TmopMetric007)),
        ("009", Box::new(TmopMetric009)),
        ("014", Box::new(TmopMetric014)),
        ("022", Box::new(TmopMetric022 { min_det_t: -0.1 })),
        ("050", Box::new(TmopMetric050)),
        ("055", Box::new(TmopMetric055)),
        ("056", Box::new(TmopMetric056)),
        ("058", Box::new(TmopMetric058)),
        ("077", Box::new(TmopMetric077)),
    ];

    for (name, metric) in &metrics_2d {
        let result = check_metric_2d(metric.as_ref(), 100, 10);
        println!(
            "  Metric {:3}: EvalW errors: {}/{}, dF rate: {:.2}, ddF rate: {:.2}",
            name,
            result.eval_w_errors,
            result.eval_w_total,
            result.avg_dF_rate,
            result.min_ddF_rate
        );
    }

    // Check 3D metrics
    println!("\nChecking 3D metrics...");
    let metrics_3d: Vec<(&str, Box<dyn fem_mesh::tmop::TmopQualityMetric3D>)> = vec![
        ("301", Box::new(TmopMetric301)),
        ("302", Box::new(TmopMetric302)),
        ("303", Box::new(TmopMetric303)),
        ("304", Box::new(TmopMetric304)),
        ("315", Box::new(TmopMetric315)),
        ("316", Box::new(TmopMetric316)),
        ("318", Box::new(TmopMetric318)),
        ("321", Box::new(TmopMetric321)),
        ("323", Box::new(TmopMetric323)),
        ("360", Box::new(TmopMetric360)),
    ];

    for (name, metric) in &metrics_3d {
        let result = check_metric_3d(metric.as_ref(), 100, 10);
        println!(
            "  Metric {:3}: EvalW errors: {}/{}, dF rate: {:.2}, ddF rate: {:.2}",
            name,
            result.eval_w_errors,
            result.eval_w_total,
            result.avg_dF_rate,
            result.min_ddF_rate
        );
    }

    println!("\n=== Done ===");
}
