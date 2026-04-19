//! Example 47: built-in multiphysics template catalog preview.
//!
//! Prints COMSOL-like built-in template nodes and their coupling metadata.

use fem_solver::{builtin_template_catalog, TemplateCouplingStyle};

fn main() {
    println!("=== fem-rs Example 47: multiphysics template catalog ===");
    for (i, spec) in builtin_template_catalog().iter().enumerate() {
        println!("{}: {} [{}]", i + 1, spec.template.title(), spec.template.id());
        println!("   fields: {}", spec.field_nodes.join(", "));
        println!("   coupling style: {}", coupling_style_name(spec.default_coupling_style));
        println!("   time integrator: {}", spec.default_time_integrator);
        println!("   nonlinear solver: {}", spec.default_nonlinear_solver);
        println!("   notes: {}", spec.notes);
        for edge in spec.coupling_edges {
            println!("   - {}", edge);
        }
        println!();
    }
}

fn coupling_style_name(style: TemplateCouplingStyle) -> &'static str {
    match style {
        TemplateCouplingStyle::Monolithic => "monolithic",
        TemplateCouplingStyle::Partitioned => "partitioned",
        TemplateCouplingStyle::Hybrid => "hybrid",
    }
}

#[cfg(test)]
mod tests {
    use fem_solver::{BuiltinMultiphysicsTemplate, builtin_template_catalog};

    #[test]
    fn ex47_template_catalog_is_stable() {
        let cat = builtin_template_catalog();
        assert_eq!(cat.len(), 5);
        assert_eq!(cat[0].template, BuiltinMultiphysicsTemplate::JouleHeating);
        assert_eq!(cat[1].template, BuiltinMultiphysicsTemplate::FluidStructureInteraction);
        assert_eq!(cat[2].template, BuiltinMultiphysicsTemplate::AcousticsStructure);
        assert_eq!(cat[3].template, BuiltinMultiphysicsTemplate::ElectromagneticThermalStress);
        assert_eq!(cat[4].template, BuiltinMultiphysicsTemplate::ReactionFlowThermal);
    }
}
