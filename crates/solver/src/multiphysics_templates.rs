//! Built-in multiphysics template catalog and node-style metadata.
//!
//! This module defines a stable interface for COMSOL-like multiphysics
//! template nodes. The first stage focuses on discoverability and consistent
//! configuration; each template can later be connected to concrete coupled
//! assemblers/solvers without changing the public API.

use std::fmt;

/// Built-in multiphysics templates planned for first-class support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinMultiphysicsTemplate {
    /// Electric + thermal coupling (Joule heating).
    JouleHeating,
    /// Fluid-structure interaction.
    FluidStructureInteraction,
    /// Acoustics-structure interaction.
    AcousticsStructure,
    /// Electromagnetic + thermal + mechanics coupling.
    ElectromagneticThermalStress,
    /// Reaction engineering (chemistry + flow + thermal).
    ReactionFlowThermal,
}

impl BuiltinMultiphysicsTemplate {
    pub const ALL: [Self; 5] = [
        Self::JouleHeating,
        Self::FluidStructureInteraction,
        Self::AcousticsStructure,
        Self::ElectromagneticThermalStress,
        Self::ReactionFlowThermal,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::JouleHeating => "joule_heating",
            Self::FluidStructureInteraction => "fsi",
            Self::AcousticsStructure => "acoustics_structure",
            Self::ElectromagneticThermalStress => "electromagnetic_thermal_stress",
            Self::ReactionFlowThermal => "reaction_flow_thermal",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::JouleHeating => "Electric - Thermal (Joule Heating)",
            Self::FluidStructureInteraction => "Fluid - Structure (FSI)",
            Self::AcousticsStructure => "Acoustics - Structure",
            Self::ElectromagneticThermalStress => "Magnetic - Thermal - Structural Stress",
            Self::ReactionFlowThermal => "Chemistry - Flow - Thermal (Reaction Engineering)",
        }
    }
}

impl fmt::Display for BuiltinMultiphysicsTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.title())
    }
}

/// Coupling topology for a built-in template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateCouplingStyle {
    /// All fields solved in one monolithic nonlinear system.
    Monolithic,
    /// Fields solved in staggered blocks (Picard/fixed-point style).
    Partitioned,
    /// Problem can switch between monolithic and partitioned modes.
    Hybrid,
}

/// Metadata for one built-in template node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiphysicsTemplateSpec {
    pub template: BuiltinMultiphysicsTemplate,
    pub field_nodes: &'static [&'static str],
    pub coupling_edges: &'static [&'static str],
    pub default_coupling_style: TemplateCouplingStyle,
    pub default_time_integrator: &'static str,
    pub default_nonlinear_solver: &'static str,
    pub notes: &'static str,
}

/// Runtime options shared by all template drivers.
#[derive(Debug, Clone)]
pub struct TemplateRuntimeConfig {
    pub dt: f64,
    pub t_end: f64,
    pub max_coupling_iterations: usize,
    pub conservative_transfer: bool,
    pub use_line_search_newton: bool,
}

impl Default for TemplateRuntimeConfig {
    fn default() -> Self {
        Self {
            dt: 1.0e-2,
            t_end: 1.0,
            max_coupling_iterations: 20,
            conservative_transfer: true,
            use_line_search_newton: true,
        }
    }
}

/// Lightweight trait for node-style template registration.
///
/// Concrete templates can implement this trait and later expose full
/// coupled-problem builders while preserving a stable metadata API.
pub trait MultiphysicsTemplateNode: Send + Sync {
    fn template(&self) -> BuiltinMultiphysicsTemplate;
    fn spec(&self) -> &'static MultiphysicsTemplateSpec;

    /// Validate template-generic runtime options.
    fn validate_runtime_config(&self, cfg: &TemplateRuntimeConfig) -> Result<(), String> {
        if !(cfg.dt.is_finite() && cfg.dt > 0.0) {
            return Err("dt must be finite and > 0".to_string());
        }
        if !(cfg.t_end.is_finite() && cfg.t_end > 0.0) {
            return Err("t_end must be finite and > 0".to_string());
        }
        if cfg.max_coupling_iterations == 0 {
            return Err("max_coupling_iterations must be >= 1".to_string());
        }
        Ok(())
    }
}

const JOULE_HEATING_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::JouleHeating,
    field_nodes: &["electric_potential", "temperature"],
    coupling_edges: &[
        "electric_potential -> joule_source -> temperature",
        "temperature -> conductivity_update -> electric_potential",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "implicit_euler_or_sdirk2",
    default_nonlinear_solver: "coupled_newton_line_search",
    notes: "Suitable for DC/low-frequency electro-thermal coupling.",
};

const FSI_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::FluidStructureInteraction,
    field_nodes: &["fluid_velocity_pressure", "solid_displacement", "mesh_motion"],
    coupling_edges: &[
        "fluid_traction -> solid_boundary_load",
        "solid_displacement -> fluid_moving_boundary",
        "mesh_motion -> ale_convection_velocity",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "generalized_alpha_or_bdf2",
    default_nonlinear_solver: "partitioned_picard_or_coupled_newton",
    notes: "Supports moving-boundary ALE workflows with optional monolithic upgrades.",
};

const ACOUSTICS_STRUCTURE_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::AcousticsStructure,
    field_nodes: &["acoustic_pressure", "solid_displacement"],
    coupling_edges: &[
        "acoustic_pressure -> structural_normal_load",
        "structure_normal_acceleration -> acoustic_boundary_condition",
    ],
    default_coupling_style: TemplateCouplingStyle::Partitioned,
    default_time_integrator: "newmark_or_generalized_alpha",
    default_nonlinear_solver: "linear_or_quasi_newton",
    notes: "Typical vibro-acoustic coupling with interface continuity constraints.",
};

const EM_THERMAL_STRESS_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::ElectromagneticThermalStress,
    field_nodes: &[
        "magneto_quasistatic_field",
        "temperature",
        "structural_displacement",
    ],
    coupling_edges: &[
        "em_losses -> thermal_source",
        "temperature -> thermal_expansion -> structural_load",
        "temperature -> material_update -> em_field",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "imex_or_sdirk2",
    default_nonlinear_solver: "staggered_plus_newton_corrector",
    notes: "For electromagnetic heating and thermo-mechanical stress prediction.",
};

const REACTION_FLOW_THERMAL_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::ReactionFlowThermal,
    field_nodes: &["species", "fluid_velocity_pressure", "temperature"],
    coupling_edges: &[
        "species_and_temperature -> reaction_rate",
        "reaction_heat_release -> temperature",
        "temperature_and_species -> density_viscosity_update -> flow",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "imex_ark3_or_bdf2",
    default_nonlinear_solver: "newton_krylov_or_partitioned_picard",
    notes: "Captures reactive transport with thermal and flow feedback.",
};

/// Return the built-in template specification by template key.
pub fn builtin_template_spec(t: BuiltinMultiphysicsTemplate) -> &'static MultiphysicsTemplateSpec {
    match t {
        BuiltinMultiphysicsTemplate::JouleHeating => &JOULE_HEATING_SPEC,
        BuiltinMultiphysicsTemplate::FluidStructureInteraction => &FSI_SPEC,
        BuiltinMultiphysicsTemplate::AcousticsStructure => &ACOUSTICS_STRUCTURE_SPEC,
        BuiltinMultiphysicsTemplate::ElectromagneticThermalStress => &EM_THERMAL_STRESS_SPEC,
        BuiltinMultiphysicsTemplate::ReactionFlowThermal => &REACTION_FLOW_THERMAL_SPEC,
    }
}

/// List all built-in template specs in stable order.
pub fn builtin_template_catalog() -> Vec<&'static MultiphysicsTemplateSpec> {
    BuiltinMultiphysicsTemplate::ALL
        .iter()
        .map(|t| builtin_template_spec(*t))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_template_catalog_contains_expected_templates() {
        let cat = builtin_template_catalog();
        assert_eq!(cat.len(), 5);
        let ids: Vec<&str> = cat.iter().map(|s| s.template.id()).collect();
        assert!(ids.contains(&"joule_heating"));
        assert!(ids.contains(&"fsi"));
        assert!(ids.contains(&"acoustics_structure"));
        assert!(ids.contains(&"electromagnetic_thermal_stress"));
        assert!(ids.contains(&"reaction_flow_thermal"));
    }

    #[test]
    fn runtime_config_validation_rejects_invalid_values() {
        struct Dummy;
        impl MultiphysicsTemplateNode for Dummy {
            fn template(&self) -> BuiltinMultiphysicsTemplate {
                BuiltinMultiphysicsTemplate::JouleHeating
            }
            fn spec(&self) -> &'static MultiphysicsTemplateSpec {
                builtin_template_spec(BuiltinMultiphysicsTemplate::JouleHeating)
            }
        }

        let n = Dummy;
        let mut cfg = TemplateRuntimeConfig::default();
        assert!(n.validate_runtime_config(&cfg).is_ok());

        cfg.dt = 0.0;
        assert!(n.validate_runtime_config(&cfg).is_err());
        cfg.dt = 1e-2;
        cfg.t_end = -1.0;
        assert!(n.validate_runtime_config(&cfg).is_err());
        cfg.t_end = 1.0;
        cfg.max_coupling_iterations = 0;
        assert!(n.validate_runtime_config(&cfg).is_err());
    }
}
