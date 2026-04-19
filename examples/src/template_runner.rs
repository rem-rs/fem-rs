//! Shared utilities for built-in multiphysics template driver examples.

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
