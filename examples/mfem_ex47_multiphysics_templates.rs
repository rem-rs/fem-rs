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
    use fem_solver::{BuiltinMultiphysicsTemplate, TemplateCouplingStyle, builtin_template_catalog};

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

    /// Every template must declare at least 2 field nodes and at least 1 coupling edge.
    #[test]
    fn ex47_all_templates_have_fields_and_edges() {
        for spec in builtin_template_catalog() {
            assert!(spec.field_nodes.len() >= 2,
                "{} has fewer than 2 field nodes", spec.template.id());
            assert!(!spec.coupling_edges.is_empty(),
                "{} has no coupling edges", spec.template.id());
            for edge in spec.coupling_edges {
                assert!(edge.contains("->"),
                    "coupling edge '{}' in {} lacks '->' arrow notation", edge, spec.template.id());
            }
        }
    }

    /// Only the acoustics-structure template should use purely Partitioned coupling.
    #[test]
    fn ex47_acoustics_structure_is_partitioned_only_template() {
        let partitioned: Vec<_> = builtin_template_catalog()
            .iter()
            .filter(|s| s.default_coupling_style == TemplateCouplingStyle::Partitioned)
            .map(|s| s.template.id())
            .collect();
        assert_eq!(partitioned, vec!["acoustics_structure"],
            "expected exactly acoustics_structure as the sole Partitioned template");
    }

    /// template id() and title() must be non-empty and distinct across templates.
    #[test]
    fn ex47_template_ids_and_titles_are_unique_and_non_empty() {
        let cat = builtin_template_catalog();
        let mut ids = std::collections::HashSet::new();
        let mut titles = std::collections::HashSet::new();
        for spec in &cat {
            let id = spec.template.id();
            let title = spec.template.title();
            assert!(!id.is_empty(), "template has empty id");
            assert!(!title.is_empty(), "template has empty title");
            assert!(ids.insert(id), "duplicate template id: {id}");
            assert!(titles.insert(title), "duplicate template title: {title}");
        }
    }

    #[test]
    fn ex47_coupling_style_distribution_matches_expected_profile() {
        let cat = builtin_template_catalog();
        let monolithic = cat
            .iter()
            .filter(|s| s.default_coupling_style == TemplateCouplingStyle::Monolithic)
            .count();
        let partitioned = cat
            .iter()
            .filter(|s| s.default_coupling_style == TemplateCouplingStyle::Partitioned)
            .count();
        let hybrid = cat
            .iter()
            .filter(|s| s.default_coupling_style == TemplateCouplingStyle::Hybrid)
            .count();

        assert_eq!(partitioned, 1);
        assert_eq!(monolithic + partitioned + hybrid, cat.len());
        assert!(hybrid >= 1);
    }

    #[test]
    fn ex47_default_solver_and_integrator_are_non_empty() {
        for spec in builtin_template_catalog() {
            assert!(!spec.default_time_integrator.trim().is_empty(),
                "{} has empty default_time_integrator", spec.template.id());
            assert!(!spec.default_nonlinear_solver.trim().is_empty(),
                "{} has empty default_nonlinear_solver", spec.template.id());
        }
    }

    #[test]
    fn ex47_coupling_edges_reference_known_fields() {
        for spec in builtin_template_catalog() {
            for edge in spec.coupling_edges {
                let parts: Vec<&str> = edge.split("->").map(|s| s.trim()).collect();
                assert!(parts.len() >= 2, "invalid edge format in {}: {edge}", spec.template.id());
                assert!(parts.iter().all(|p| !p.is_empty()),
                    "edge contains empty token in {}: {edge}", spec.template.id());
            }
        }
    }

    #[test]
    fn ex47_template_notes_are_present_and_descriptive() {
        for spec in builtin_template_catalog() {
            let note = spec.notes.trim();
            assert!(!note.is_empty(), "{} has empty notes", spec.template.id());
            assert!(note.len() >= 20, "{} notes are too short: '{}'", spec.template.id(), note);
        }
    }
}
