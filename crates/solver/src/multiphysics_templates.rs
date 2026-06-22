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
    /// Thermoelastic coupling.
    ThermoelasticCoupled,
    /// 3-D electrothermal coupling.
    Electrothermal3D,
    /// Electric + thermal coupling (Joule heating).
    JouleHeating,
    /// Fluid-structure interaction.
    FluidStructureInteraction,
    /// Quasi-ALE moving mesh with conservative transfer.
    MovingMeshAle,
    /// Moving-mesh transient heat (quasi-ALE).
    MovingMeshHeat,
    /// Acoustics-structure interaction.
    AcousticsStructure,
    /// Electromagnetic + thermal + mechanics coupling.
    ElectromagneticThermalStress,
    /// Reaction engineering (chemistry + flow + thermal).
    ReactionFlowThermal,
    /// Cut-cell immersed boundary with embedded geometry coupling.
    ImmersedBoundary,
}

impl BuiltinMultiphysicsTemplate {
    pub const ALL: [Self; 10] = [
        Self::ThermoelasticCoupled,
        Self::Electrothermal3D,
        Self::JouleHeating,
        Self::FluidStructureInteraction,
        Self::MovingMeshAle,
        Self::MovingMeshHeat,
        Self::AcousticsStructure,
        Self::ElectromagneticThermalStress,
        Self::ReactionFlowThermal,
        Self::ImmersedBoundary,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::ThermoelasticCoupled => "thermoelastic_coupled",
            Self::Electrothermal3D => "electrothermal_3d",
            Self::JouleHeating => "joule_heating",
            Self::FluidStructureInteraction => "fsi",
            Self::MovingMeshAle => "moving_mesh_ale",
            Self::MovingMeshHeat => "moving_mesh_heat",
            Self::AcousticsStructure => "acoustics_structure",
            Self::ElectromagneticThermalStress => "electromagnetic_thermal_stress",
            Self::ReactionFlowThermal => "reaction_flow_thermal",
            Self::ImmersedBoundary => "immersed_boundary",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::ThermoelasticCoupled => "Thermoelastic Coupling",
            Self::Electrothermal3D => "3-D Electrothermal Coupling",
            Self::JouleHeating => "Electric - Thermal (Joule Heating)",
            Self::FluidStructureInteraction => "Fluid - Structure (FSI)",
            Self::MovingMeshAle => "Moving-Mesh ALE Transfer",
            Self::MovingMeshHeat => "Moving-Mesh Heat (Quasi-ALE)",
            Self::AcousticsStructure => "Acoustics - Structure",
            Self::ElectromagneticThermalStress => "Magnetic - Thermal - Structural Stress",
            Self::ReactionFlowThermal => "Chemistry - Flow - Thermal (Reaction Engineering)",
            Self::ImmersedBoundary => "Immersed Boundary",
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

const THERMOELASTIC_COUPLED_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::ThermoelasticCoupled,
    field_nodes: &["structural_displacement", "temperature"],
    coupling_edges: &[
        "temperature -> thermal_expansion -> structural_load",
        "structural_dissipation -> thermal_source -> temperature",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "steady_newton_or_imex_split",
    default_nonlinear_solver: "coupled_newton_or_partitioned_split",
    notes: "Supports steady monolithic thermoelastic solves plus transient split and IMEX workflow variants.",
};

const ELECTROTHERMAL_3D_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::Electrothermal3D,
    field_nodes: &["electric_potential", "temperature"],
    coupling_edges: &[
        "electric_potential -> joule_source -> temperature",
        "temperature -> conductivity_update -> electric_potential",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "fixed_point_steady_or_pseudo_transient",
    default_nonlinear_solver: "relaxed_picard",
    notes: "Provides a practical tetrahedral electrothermal workflow with temperature-dependent conductivity and Joule heating feedback.",
};

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

const MOVING_MESH_ALE_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::MovingMeshAle,
    field_nodes: &["mesh_motion", "transported_scalar"],
    coupling_edges: &[
        "mesh_motion -> conservative_transfer -> transported_scalar",
        "mesh_motion -> smoothing_update -> mesh_quality",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "prescribed_motion_transfer_loop",
    default_nonlinear_solver: "explicit_mesh_update_plus_projection",
    notes: "Covers quasi-ALE moving-mesh transfer workflows that emphasize mesh validity and conservative scalar transport.",
};

const MOVING_MESH_HEAT_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::MovingMeshHeat,
    field_nodes: &["mesh_motion", "temperature"],
    coupling_edges: &[
        "mesh_motion -> conservative_transfer -> temperature",
        "mesh_motion -> ale_convection_velocity -> temperature",
    ],
    default_coupling_style: TemplateCouplingStyle::Hybrid,
    default_time_integrator: "implicit_euler_on_deforming_mesh",
    default_nonlinear_solver: "partitioned_ale_update_plus_linear_heat_solve",
    notes: "Tracks quasi-ALE moving-mesh heat workflows with conservative field transfer across mesh updates.",
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

const IMMERSED_BOUNDARY_SPEC: MultiphysicsTemplateSpec = MultiphysicsTemplateSpec {
    template: BuiltinMultiphysicsTemplate::ImmersedBoundary,
    field_nodes: &["embedded_geometry", "embedded_solution"],
    coupling_edges: &[
        "embedded_geometry -> cut_cell_quadrature -> embedded_solution",
        "embedded_geometry -> nitsche_boundary_terms -> embedded_solution",
    ],
    default_coupling_style: TemplateCouplingStyle::Monolithic,
    default_time_integrator: "steady_cut_cell_solve",
    default_nonlinear_solver: "direct_or_cg_with_nitsche_bc",
    notes: "Represents immersed-boundary cut-cell workflows where level-set geometry drives quadrature, active-set assembly, and weak boundary enforcement.",
};

/// Return the built-in template specification by template key.
pub fn builtin_template_spec(t: BuiltinMultiphysicsTemplate) -> &'static MultiphysicsTemplateSpec {
    match t {
        BuiltinMultiphysicsTemplate::ThermoelasticCoupled => &THERMOELASTIC_COUPLED_SPEC,
        BuiltinMultiphysicsTemplate::Electrothermal3D => &ELECTROTHERMAL_3D_SPEC,
        BuiltinMultiphysicsTemplate::JouleHeating => &JOULE_HEATING_SPEC,
        BuiltinMultiphysicsTemplate::FluidStructureInteraction => &FSI_SPEC,
        BuiltinMultiphysicsTemplate::MovingMeshAle => &MOVING_MESH_ALE_SPEC,
        BuiltinMultiphysicsTemplate::MovingMeshHeat => &MOVING_MESH_HEAT_SPEC,
        BuiltinMultiphysicsTemplate::AcousticsStructure => &ACOUSTICS_STRUCTURE_SPEC,
        BuiltinMultiphysicsTemplate::ElectromagneticThermalStress => &EM_THERMAL_STRESS_SPEC,
        BuiltinMultiphysicsTemplate::ReactionFlowThermal => &REACTION_FLOW_THERMAL_SPEC,
        BuiltinMultiphysicsTemplate::ImmersedBoundary => &IMMERSED_BOUNDARY_SPEC,
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
        assert_eq!(cat.len(), 10);
        let ids: Vec<&str> = cat.iter().map(|s| s.template.id()).collect();
        assert!(ids.contains(&"thermoelastic_coupled"));
        assert!(ids.contains(&"electrothermal_3d"));
        assert!(ids.contains(&"joule_heating"));
        assert!(ids.contains(&"fsi"));
        assert!(ids.contains(&"moving_mesh_ale"));
        assert!(ids.contains(&"moving_mesh_heat"));
        assert!(ids.contains(&"acoustics_structure"));
        assert!(ids.contains(&"electromagnetic_thermal_stress"));
        assert!(ids.contains(&"reaction_flow_thermal"));
        assert!(ids.contains(&"immersed_boundary"));
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
