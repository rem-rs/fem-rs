//! Plugin API traits and registries for fem-pro assembly/IGA extensions.
//!
//! Professional edition crates can register custom integrators, mesh
//! modifiers, and physics models through these registries.  When no pro
//! plugins are loaded the registries return `None` and built-in behavior
//! is unchanged — zero overhead, no pro code in the OSS binary.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use nalgebra::DMatrix;
use fem_core::FemResult;
use fem_element::reference::ReferenceElement;
use fem_mesh::Mesh;

/// Per-element information passed to [`ProIntegrator`].
pub struct ElementInfo {
    /// Global element ID in the mesh.
    pub element_id: usize,
    /// Number of trial/test DOFs for this element.
    pub n_dofs: usize,
    /// Number of quadrature points used in the assembly loop.
    pub quad_points: usize,
}

/// Trait for externally-registered element integrators (pro edition).
///
/// Implement this to provide custom bilinear-form element matrices that
/// the open-source assembler does not define (e.g. GPU-accelerated or
/// physics-specific integrators).
pub trait ProIntegrator: Send + Sync {
    /// Human-readable integrator name, used as the registry key.
    fn name(&self) -> &str;

    /// Compute the element-level matrix for a pair of trial/test
    /// reference elements at the given quadrature configuration.
    fn assemble_element_matrix(
        &self,
        element: &ElementInfo,
        trial: &dyn ReferenceElement,
        test: &dyn ReferenceElement,
    ) -> FemResult<DMatrix<f64>>;
}

/// Trait for externally-registered mesh modifiers (pro edition).
///
/// Used for CAD defeaturing, mesh morphing, and other geometry
/// transformations that the open-source edition does not provide.
pub trait ProMeshModifier: Send + Sync {
    /// Human-readable modifier name, used as the registry key.
    fn name(&self) -> &str;

    /// Modify the mesh in-place.
    fn modify(&self, mesh: &mut Mesh<3>) -> FemResult<()>;
}

/// Trait for externally-registered nonlinear physics models (pro edition).
///
/// Implement this to provide custom constitutive models for nonlinear
/// solvers (e.g. advanced plasticity, damage, fatigue).
pub trait ProPhysicsModel: Send + Sync {
    /// Human-readable model name, used as the registry key.
    fn name(&self) -> &str;

    /// Compute the residual vector `R(state)` for the current state.
    fn compute_residual(&self, state: &[f64]) -> FemResult<Vec<f64>>;

    /// Compute the tangent stiffness matrix (as a flattened CSR vector)
    /// for the current state.
    fn compute_tangent(&self, state: &[f64]) -> FemResult<Vec<f64>>;
}

/// Global registry for pro integrators.
pub struct IntegratorRegistry {
    integrators: HashMap<String, Box<dyn ProIntegrator>>,
}

impl IntegratorRegistry {
    /// Access the global singleton registry.
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: OnceLock<Mutex<IntegratorRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            Mutex::new(IntegratorRegistry {
                integrators: HashMap::new(),
            })
        })
    }

    /// Register a pro integrator under its `name()` key.
    pub fn register(&mut self, integrator: Box<dyn ProIntegrator>) {
        let name = integrator.name().to_string();
        self.integrators.insert(name, integrator);
    }

    /// Look up a registered integrator by name.
    ///
    /// Returns `None` if the integrator is not registered (the normal case
    /// in the open-source edition).
    pub fn get(&self, name: &str) -> Option<&dyn ProIntegrator> {
        self.integrators.get(name).map(|s| s.as_ref())
    }
}

/// Global registry for pro mesh modifiers.
pub struct MeshModifierRegistry {
    modifiers: HashMap<String, Box<dyn ProMeshModifier>>,
}

impl MeshModifierRegistry {
    /// Access the global singleton registry.
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: OnceLock<Mutex<MeshModifierRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            Mutex::new(MeshModifierRegistry {
                modifiers: HashMap::new(),
            })
        })
    }

    /// Register a pro mesh modifier under its `name()` key.
    pub fn register(&mut self, modifier: Box<dyn ProMeshModifier>) {
        let name = modifier.name().to_string();
        self.modifiers.insert(name, modifier);
    }

    /// Look up a registered mesh modifier by name.
    pub fn get(&self, name: &str) -> Option<&dyn ProMeshModifier> {
        self.modifiers.get(name).map(|s| s.as_ref())
    }
}

/// Global registry for pro physics models.
pub struct PhysicsModelRegistry {
    models: HashMap<String, Box<dyn ProPhysicsModel>>,
}

impl PhysicsModelRegistry {
    /// Access the global singleton registry.
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: OnceLock<Mutex<PhysicsModelRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            Mutex::new(PhysicsModelRegistry {
                models: HashMap::new(),
            })
        })
    }

    /// Register a pro physics model under its `name()` key.
    pub fn register(&mut self, model: Box<dyn ProPhysicsModel>) {
        let name = model.name().to_string();
        self.models.insert(name, model);
    }

    /// Look up a registered physics model by name.
    pub fn get(&self, name: &str) -> Option<&dyn ProPhysicsModel> {
        self.models.get(name).map(|s| s.as_ref())
    }
}
