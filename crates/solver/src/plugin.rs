//! Plugin API traits and registries for fem-pro solver extensions.
//!
//! These traits define extension points that professional-edition crates
//! (e.g. `pro-solver`) can register against at runtime.  When no pro plugins
//! are loaded the registries return `None` and built-in solvers are used —
//! zero overhead, no pro code in the OSS binary.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use fem_core::FemResult;
use fem_linalg::{CsrMatrix, Vector};

/// Trait for externally-registered solvers (pro edition).
///
/// Implement this trait to plug in a professional solver (FETI/BDDC,
/// JDQZ, GPU-native solvers, etc.) that the open-source crates do not
/// provide.
pub trait ProSolver: Send + Sync {
    /// Human-readable solver name, used as the registry key.
    fn name(&self) -> &str;

    /// Solve the linear system `matrix · x = rhs` and return the solution
    /// vector.
    fn solve(&self, matrix: &CsrMatrix<f64>, rhs: &Vector<f64>)
        -> FemResult<Vector<f64>>;
}

/// Global registry for pro solvers.
///
/// When no pro plugins are loaded, `get()` returns `None` — fall back to
/// built-in solvers (CG, GMRES, BiCGSTAB, etc.).
pub struct SolverRegistry {
    solvers: HashMap<String, Box<dyn ProSolver>>,
}

impl SolverRegistry {
    /// Access the global singleton registry.
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: OnceLock<Mutex<SolverRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            Mutex::new(SolverRegistry {
                solvers: HashMap::new(),
            })
        })
    }

    /// Register a pro solver under its `name()` key.
    pub fn register(&mut self, solver: Box<dyn ProSolver>) {
        let name = solver.name().to_string();
        self.solvers.insert(name, solver);
    }

    /// Look up a registered solver by name.
    ///
    /// Returns `None` if the solver is not registered (the normal case in
    /// the open-source edition).
    pub fn get(&self, name: &str) -> Option<&dyn ProSolver> {
        self.solvers.get(name).map(|s| s.as_ref())
    }
}
