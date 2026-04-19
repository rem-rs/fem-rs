//! Shared utilities for built-in multiphysics template driver examples.

use std::{
    env,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

use fem_solver::MultiphysicsTemplateSpec;

/// Standardized coupling summary reported by template drivers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateCouplingSummary {
    pub steps: usize,
    pub converged_steps: usize,
    pub max_coupling_iters_used: usize,
}

/// Standardized adaptive scheduler summary for subcycling diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateAdaptiveSummary {
    pub sync_retries: usize,
    pub rejected_sync_steps: usize,
    pub rollback_count: usize,
}

/// Optional trait for template-style drivers.
///
/// This offers a common shape for future refactoring of examples into reusable
/// runners while keeping example-specific state/result types.
pub trait TemplateRunner {
    type Args;
    type Result;

    fn spec() -> &'static MultiphysicsTemplateSpec;
    fn run(args: &Self::Args) -> Self::Result;
    fn summary(result: &Self::Result) -> TemplateCouplingSummary;
}

/// Print a consistent header for built-in template examples.
pub fn print_template_header(
    example_name: &str,
    spec: &MultiphysicsTemplateSpec,
    config_line: &str,
) {
    println!("=== fem-rs {} ===", example_name);
    println!("  template: {} [{}]", spec.template.title(), spec.template.id());
    println!("  fields: {}", spec.field_nodes.join(", "));
    println!("  coupling style(default): {:?}", spec.default_coupling_style);
    println!("  config: {config_line}");
}

/// Print a consistent coupling-iteration summary for template examples.
pub fn print_template_coupling_summary(summary: TemplateCouplingSummary) {
    println!("  converged steps: {}/{}", summary.converged_steps, summary.steps);
    println!(
        "  max coupling iterations used: {}",
        summary.max_coupling_iters_used
    );
}

/// Print a consistent adaptive-scheduler summary for template examples.
pub fn print_template_adaptive_summary(summary: TemplateAdaptiveSummary) {
    println!("  sync retries used: {}", summary.sync_retries);
    println!("  rejected sync steps: {}", summary.rejected_sync_steps);
    println!("  rollback count: {}", summary.rollback_count);
}

/// Print a consistent CLI help block for template examples.
pub fn print_template_cli_help(bin: &str, options: &[(&str, &str)]) {
    println!("Usage:");
    println!("  {bin} [options]");
    println!();
    println!("Options:");
    for (flag, desc) in options {
        println!("  {:<24} {}", flag, desc);
    }
    println!("  {:<24} {}", "-h, --help", "Show this help message and exit");
}

/// Optional KPI row written by built-in multiphysics template examples.
///
/// Set `FEM_TEMPLATE_KPI_CSV` to a file path to enable CSV export.
/// Additional optional tags:
/// - `FEM_TEMPLATE_KPI_RUN_ID`
/// - `FEM_TEMPLATE_KPI_TAG`
pub fn maybe_write_template_kpi_csv(
    template_id: &str,
    coupling: TemplateCouplingSummary,
    adaptive: TemplateAdaptiveSummary,
    metrics: &[(&str, f64)],
) -> std::io::Result<()> {
    let Some(path_value) = env::var_os("FEM_TEMPLATE_KPI_CSV") else {
        return Ok(());
    };
    let path = Path::new(&path_value);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file_exists = path.exists();
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);

    if !file_exists {
        write_kpi_header(&mut writer)?;
    }

    let run_id = env::var("FEM_TEMPLATE_KPI_RUN_ID").unwrap_or_else(|_| "default".to_string());
    let tag = env::var("FEM_TEMPLATE_KPI_TAG").unwrap_or_else(|_| "default".to_string());
    write!(
        writer,
        "{},{},{},{},{},{},{},{},{}",
        sanitize_csv_field(template_id),
        sanitize_csv_field(&run_id),
        sanitize_csv_field(&tag),
        coupling.steps,
        coupling.converged_steps,
        coupling.max_coupling_iters_used,
        adaptive.sync_retries,
        adaptive.rejected_sync_steps,
        adaptive.rollback_count,
    )?;

    let metrics_serialized = metrics
        .iter()
        .map(|(name, value)| format!("{}={:.12e}", sanitize_csv_field(name), value))
        .collect::<Vec<_>>()
        .join(";");
    write!(writer, ",{}", sanitize_csv_field(&metrics_serialized))?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn write_kpi_header(writer: &mut BufWriter<File>) -> std::io::Result<()> {
    writeln!(
        writer,
        "template_id,run_id,tag,steps,converged_steps,max_coupling_iters_used,sync_retries,rejected_sync_steps,rollback_count,metrics"
    )
}

fn sanitize_csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
