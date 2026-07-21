//! User Element (UEL) framework.
//!
//! Allows users to define custom element types by implementing the [`UserElement`]
//! trait, then registering them with the assembly system.
//!
//! # Usage
//!
//! ```rust,ignore
//! use fem_assembly::uel::*;
//!
//! #[derive(Debug)]
//! struct MySpring {
//!     k: f64,
//!     node_i: usize,
//!     node_j: usize,
//! }
//!
//! impl UserElement for MySpring {
//!     fn n_dofs(&self) -> usize { 2 }
//!     fn element_type(&self) -> &str { "Spring" }
//!     fn assemble_stiffness(&self, _u: &[f64]) -> Vec<f64> {
//!         vec![self.k, -self.k, -self.k, self.k]  // 2×2 row-major
//!     }
//! }
//! ```

use std::collections::HashMap;

/// Trait for user-defined finite elements.
///
/// Implement this trait to define custom element types that can be
/// assembled into the global system.
pub trait UserElement: std::fmt::Debug + Send + Sync {
    /// Number of DOFs for this element.
    fn n_dofs(&self) -> usize;

    /// Element type identifier string.
    fn element_type(&self) -> &str;

    /// Global DOF indices for this element.
    fn dof_indices(&self) -> &[usize];

    /// Assemble the element stiffness matrix (row-major, n_dofs × n_dofs).
    ///
    /// `u` — current displacement vector (global, for nonlinear elements).
    fn assemble_stiffness(&self, u: &[f64]) -> Vec<f64>;

    /// Assemble the element internal force vector (length n_dofs).
    fn assemble_internal_force(&self, u: &[f64]) -> Vec<f64> {
        let k = self.assemble_stiffness(u);
        let n = self.n_dofs();
        let mut f = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                f[i] += k[i * n + j] * u.get(self.dof_indices()[j]).copied().unwrap_or(0.0);
            }
        }
        f
    }

    /// Assemble the element mass matrix (row-major, n_dofs × n_dofs).
    fn assemble_mass(&self) -> Vec<f64> {
        vec![0.0; self.n_dofs() * self.n_dofs()]
    }

    /// Number of integration points (for output).
    fn n_integration_points(&self) -> usize { 1 }

    /// Compute stresses at integration points (for output).
    fn output_stress(&self, _u: &[f64]) -> Vec<f64> { Vec::new() }
}

/// A registered user element with metadata.
#[derive(Debug)]
pub struct UserElementInfo {
    pub type_name: String,
    pub n_dofs: usize,
}

/// Global registry for user element types.
///
/// Singleton accessed via [`uel_registry()`].
#[derive(Debug, Default)]
pub struct UserElementRegistry {
    types: HashMap<String, UserElementInfo>,
}

impl UserElementRegistry {
    pub fn new() -> Self { Self { types: HashMap::new() } }

    /// Register a user element type.
    pub fn register(&mut self, name: &str, n_dofs: usize) {
        self.types.insert(name.to_string(), UserElementInfo {
            type_name: name.to_string(),
            n_dofs,
        });
    }

    /// Check if a type is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// List all registered types.
    pub fn registered_types(&self) -> Vec<&UserElementInfo> {
        self.types.values().collect()
    }
}

// Global singleton (using a mutex for thread safety)
use std::sync::Mutex;
static UEL_REGISTRY: Mutex<Option<UserElementRegistry>> = Mutex::new(None);

/// Get or initialize the global UEL registry.
pub fn uel_registry() -> std::sync::MutexGuard<'static, Option<UserElementRegistry>> {
    let mut guard = UEL_REGISTRY.lock().unwrap();
    if guard.is_none() {
        *guard = Some(UserElementRegistry::new());
    }
    guard
}

/// Assemble a set of user elements into the global system.
///
/// `elements` — user element instances
/// `u` — current displacement vector
/// `n_global_dofs` — total number of DOFs
/// Returns (stiffness_matrix_coo, rhs_vector) as COO entries.
pub fn assemble_user_elements<E: UserElement>(
    elements: &[E],
    u: &[f64],
    n_global_dofs: usize,
) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    let mut coo_entries = Vec::new();
    let mut rhs = vec![0.0; n_global_dofs];

    for elem in elements {
        let n = elem.n_dofs();
        let dofs = elem.dof_indices();

        // Stiffness
        let k = elem.assemble_stiffness(u);
        for i in 0..n {
            let gi = dofs[i];
            if gi >= n_global_dofs { continue; }
            for j in 0..n {
                let gj = dofs[j];
                if gj >= n_global_dofs { continue; }
                let val = k[i * n + j];
                if val.abs() > 1e-30 {
                    coo_entries.push((gi, gj, val));
                }
            }
        }

        // Internal force
        let f_int = elem.assemble_internal_force(u);
        for i in 0..n {
            let gi = dofs[i];
            if gi < n_global_dofs {
                rhs[gi] -= f_int[i];
            }
        }
    }

    (coo_entries, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple spring element for testing.
    #[derive(Debug)]
    struct SpringElement {
        k: f64,
        dofs: [usize; 2],
    }

    impl UserElement for SpringElement {
        fn n_dofs(&self) -> usize { 2 }
        fn element_type(&self) -> &str { "Spring" }
        fn dof_indices(&self) -> &[usize] { &self.dofs }

        fn assemble_stiffness(&self, _u: &[f64]) -> Vec<f64> {
            vec![self.k, -self.k, -self.k, self.k]
        }
    }

    #[test]
    fn spring_element_stiffness() {
        let spring = SpringElement { k: 100.0, dofs: [0, 1] };
        let k = spring.assemble_stiffness(&[]);
        assert_eq!(k.len(), 4);
        assert!((k[0] - 100.0).abs() < 1e-10);
        assert!((k[1] - (-100.0)).abs() < 1e-10);
    }

    #[test]
    fn spring_internal_force() {
        let spring = SpringElement { k: 100.0, dofs: [0, 1] };
        let u = vec![0.01, 0.0];
        let f = spring.assemble_internal_force(&u);
        assert!((f[0] - 1.0).abs() < 1e-10, "f[0] = {}", f[0]); // k*(u0-u1) = 1
        assert!((f[1] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn assemble_multiple_elements() {
        let elements = vec![
            SpringElement { k: 100.0, dofs: [0, 1] },
            SpringElement { k: 200.0, dofs: [1, 2] },
        ];
        let u = vec![0.0; 3];
        let (coo, _rhs) = assemble_user_elements(&elements, &u, 3);
        assert!(coo.len() >= 4, "should have at least 4 COO entries");
    }

    #[test]
    fn registry_works() {
        let mut registry = UserElementRegistry::new();
        registry.register("Spring", 2);
        assert!(registry.is_registered("Spring"));
        assert!(!registry.is_registered("Unknown"));
    }
}
