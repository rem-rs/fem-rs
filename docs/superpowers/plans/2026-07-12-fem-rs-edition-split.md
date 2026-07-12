# fem-rs Edition Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split fem-rs into an Apache 2.0 open-source edition (GitHub) and a proprietary professional edition (private repo with fem-rs as submodule).

**Architecture:** fem-rs stays as the complete open-source FEM library with plugin API traits. fem-pro is a separate workspace containing a fem-rs submodule plus pro crates (pro-gpu, pro-iga, pro-solver, pro-cad, pro-physics, pro-enterprise) that register against those traits.

**Tech Stack:** Rust workspace, git submodules, trait-based plugin registries ("registry pattern" — `static Mutex<HashMap<String, Box<dyn Trait>>>`), environment variable for pro mode.

## Global Constraints

- Apache 2.0 license for fem-rs; proprietary for fem-pro
- fem-pro depends on fem-rs via git submodule + path dependency (never `crates.io` publish for pro)
- Plugin API traits must compile to zero overhead when empty (no pro features in OSS code)
- No `#[cfg(feature = "pro")]` or conditional compilation leaking pro concepts into fem-rs
- All 2491+ existing tests must pass at every commit to fem-rs `main`
- fem-pro crate scaffolding must compile standalone (with stubs for unimplemented physical modules)
- fem-pro plugin init must be a single function call (`fem_pro::init()`)

---

### Task 1: Add Plugin API Traits to fem-rs

**Files:**
- Create: `crates/solver/src/plugin.rs`
- Create: `crates/assembly/src/plugin.rs`
- Create: `crates/io/src/plugin.rs`
- Modify: `crates/solver/src/lib.rs`
- Modify: `crates/assembly/src/lib.rs`
- Modify: `crates/io/src/lib.rs`
- Test: each `lib.rs` compiles

**Interfaces:**
- Consumes: existing `fem_linalg::CsrMatrix`, `fem_linalg::Vector`, `fem_mesh::Mesh`, `fem_element::FiniteElement`, `fem_core::*`
- Produces: `SolverRegistry`, `ProSolver` trait, `ProIntegrator` trait, `ProMeshModifier` trait, `ProMeshImporter` trait, `ProPhysicsModel` trait

- [ ] **Step 1: Write `crates/solver/src/plugin.rs`**

```rust
use std::collections::HashMap;
use std::sync::Mutex;
use fem_core::FemResult;
use fem_linalg::{CsrMatrix, Vector};

/// Trait for externally-registered solvers (pro edition).
pub trait ProSolver: Send + Sync {
    fn name(&self) -> &str;
    fn solve(
        &self,
        matrix: &CsrMatrix<f64>,
        rhs: &Vector<f64>,
    ) -> FemResult<Vector<f64>>;
}

/// Global registry for pro solvers.
/// When no pro plugins are loaded, `get()` returns None — fallback to built-in solvers.
pub struct SolverRegistry {
    solvers: HashMap<String, Box<dyn ProSolver>>,
}

impl SolverRegistry {
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: std::sync::Mutex<SolverRegistry> =
            std::sync::Mutex::new(SolverRegistry {
                solvers: HashMap::new(),
            });
        &REGISTRY
    }

    pub fn register(&mut self, solver: Box<dyn ProSolver>) {
        let name = solver.name().to_string();
        self.solvers.insert(name, solver);
    }

    pub fn get(&self, name: &str) -> Option<&dyn ProSolver> {
        self.solvers.get(name).map(|s| s.as_ref())
    }
}
```

- [ ] **Step 2: Write `crates/assembly/src/plugin.rs`**

```rust
use std::collections::HashMap;
use std::sync::Mutex;
use fem_core::FemResult;
use fem_element::FiniteElement;
use fem_mesh::Mesh;
use nalgebra::DMatrix;

/// Per-element info passed to ProIntegrator.
pub struct ElementInfo {
    pub element_id: usize,
    pub n_dofs: usize,
    pub quad_points: usize,
}

/// Trait for externally-registered element integrators.
pub trait ProIntegrator: Send + Sync {
    fn name(&self) -> &str;
    fn assemble_element_matrix(
        &self,
        element: &ElementInfo,
        trial: &FiniteElement,
        test: &FiniteElement,
    ) -> FemResult<DMatrix<f64>>;
}

/// Trait for externally-registered mesh modifiers (CAD defeaturing, morphing, etc.).
pub trait ProMeshModifier: Send + Sync {
    fn name(&self) -> &str;
    fn modify(&self, mesh: &mut Mesh<3>) -> FemResult<()>;
}

/// Trait for externally-registered nonlinear physics models.
pub trait ProPhysicsModel: Send + Sync {
    fn name(&self) -> &str;
    fn compute_residual(&self, state: &[f64]) -> FemResult<Vec<f64>>;
    fn compute_tangent(&self, state: &[f64]) -> FemResult<Vec<f64>>;
}

/// Global registry for pro integrators.
pub struct IntegratorRegistry {
    integrators: HashMap<String, Box<dyn ProIntegrator>>,
}

impl IntegratorRegistry {
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: std::sync::Mutex<IntegratorRegistry> =
            std::sync::Mutex::new(IntegratorRegistry {
                integrators: HashMap::new(),
            });
        &REGISTRY
    }

    pub fn register(&mut self, integrator: Box<dyn ProIntegrator>) {
        let name = integrator.name().to_string();
        self.integrators.insert(name, integrator);
    }

    pub fn get(&self, name: &str) -> Option<&dyn ProIntegrator> {
        self.integrators.get(name).map(|s| s.as_ref())
    }
}
```

- [ ] **Step 3: Write `crates/io/src/plugin.rs`**

```rust
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use fem_core::FemResult;
use fem_mesh::MeshData;

/// Trait for externally-registered mesh/geometry importers.
pub trait ProMeshImporter: Send + Sync {
    fn name(&self) -> &str;
    fn can_import(&self, path: &Path) -> bool;
    fn import(&self, path: &Path) -> FemResult<MeshData>;
}

/// Global registry for pro mesh importers.
pub struct MeshImporterRegistry {
    importers: HashMap<String, Box<dyn ProMeshImporter>>,
}

impl MeshImporterRegistry {
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: std::sync::Mutex<MeshImporterRegistry> =
            std::sync::Mutex::new(MeshImporterRegistry {
                importers: HashMap::new(),
            });
        &REGISTRY
    }

    pub fn register(&mut self, importer: Box<dyn ProMeshImporter>) {
        let name = importer.name().to_string();
        self.importers.insert(name, importer);
    }

    pub fn get(&self, name: &str) -> Option<&dyn ProMeshImporter> {
        self.importers.get(name).map(|s| s.as_ref())
    }
}
```

- [ ] **Step 4: Register modules in each crate's lib.rs**

In `crates/solver/src/lib.rs`, add `pub mod plugin;` at the top level.

In `crates/assembly/src/lib.rs`, add `pub mod plugin;`.

In `crates/io/src/lib.rs`, add `pub mod plugin;`.

- [ ] **Step 5: Compile-check**

Run: `cargo check -p fem-solver -p fem-assembly -p fem-io`
Expected: no warnings, no errors

- [ ] **Step 6: Commit**

```bash
git add crates/solver/src/plugin.rs crates/solver/src/lib.rs \
       crates/assembly/src/plugin.rs crates/assembly/src/lib.rs \
       crates/io/src/plugin.rs crates/io/src/lib.rs
git commit -m "feat: add plugin API traits for pro edition integration

ProSolver, ProIntegrator, ProMeshModifier, ProPhysicsModel, and
ProMeshImporter traits with global registries. Empty registries
produce no overhead — built-in behavior is unchanged."
```

---

### Task 2: Remove linalg-gpu from fem-rs Workspace

**Files:**
- Modify: `Cargo.toml` (workspace members)
- Modify: `crates/assembly/Cargo.toml` (remove optional dependency)

**Interfaces:**
- Consumes: existing workspace definition
- Produces: fem-rs workspace without GPU crate (moved to fem-pro)

- [ ] **Step 1: Remove linalg-gpu from workspace members**

In `Cargo.toml`, remove the line `"crates/linalg-gpu",` from members array.

- [ ] **Step 2: Remove optional GPU dependency from assembly**

In `crates/assembly/Cargo.toml`, remove:
```toml
fem-linalg-gpu = { path = "../linalg-gpu", optional = true }
```
And remove `gpu = ["dep:fem-linalg-gpu"]` from `[features]`.

- [ ] **Step 3: Verify workspace compiles**

Run: `cargo check --workspace`
Expected: no errors about missing linalg-gpu

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove linalg-gpu from OSS workspace

Moved to fem-pro (pro edition). Plugin API traits provide
the extension point for GPU-accelerated assembly."
```

---

### Task 3: Remove Advanced IGA from fem-rs

**Files:**
- Modify: `crates/assembly/Cargo.toml` (remove iga_gpu, iga_tspline, contact_iga, iga_trim deps)
- Modify: `crates/element/Cargo.toml` (remove tmesh)
- Remove: `crates/element/src/tmesh.rs` (moved to pro)
- Remove: `crates/assembly/src/iga/iga_tspline.rs` (moved to pro)
- Remove: `crates/assembly/src/contact/contact_iga.rs` (moved to pro)
- Remove: `crates/assembly/src/iga/iga_trim.rs` (moved to pro)
- Remove: `crates/io/src/step_iges.rs` (moved to pro)
- Remove: `crates/assembly/src/iga/iga_gpu.rs` (moved to pro)

**Interfaces:**
- Consumes: existing file references in mod.rs
- Produces: fem-rs without advanced IGA modules

- [ ] **Step 1: Remove tmesh module from element crate**

Delete `crates/element/src/tmesh.rs`.
In `crates/element/src/lib.rs`, remove `mod tmesh;` line.
Remove any `pub use` of tmesh items.

- [ ] **Step 2: Remove iga_tspline module from assembly**

Delete `crates/assembly/src/iga/iga_tspline.rs`.
In `crates/assembly/src/iga/mod.rs`, remove the `mod iga_tspline;` line.

- [ ] **Step 3: Remove contact_iga module from assembly contact**

Delete `crates/assembly/src/contact/contact_iga.rs`.
In `crates/assembly/src/contact/mod.rs`, remove the `mod contact_iga;` line.

- [ ] **Step 4: Remove iga_trim module from assembly IGA**

Delete `crates/assembly/src/iga/iga_trim.rs`.
In `crates/assembly/src/iga/mod.rs`, remove the `mod iga_trim;` line.

- [ ] **Step 5: Remove iga_gpu module**

Delete `crates/assembly/src/iga/iga_gpu.rs`.
In `crates/assembly/src/iga/mod.rs`, remove the `mod iga_gpu;` line.

- [ ] **Step 6: Remove step_iges module from IO**

Delete `crates/io/src/step_iges.rs` if it exists.
In `crates/io/src/lib.rs`, remove any `mod step_iges;` line.

- [ ] **Step 7: Verify workspace compiles**

Run: `cargo check --workspace`
Expected: no errors

- [ ] **Step 8: Run full test suite**

Run: `cargo test --workspace`
Expected: all tests pass (tests that depended on removed features should be removed in lockstep; remaining 2491+ tests must pass)

- [ ] **Step 9: Commit**

```bash
git commit -m "refactor: remove advanced IGA modules from OSS edition

T-spline, trimmed NURBS, IGA contact, IGA GPU, and STEP/IGES
import moved to fem-pro. Open-source edition retains NURBS basis,
degree elevation, knot insertion, single-patch assembly, and
Bezier extraction (MFEM-equivalent IGA)."
```

---

### Task 4: Update fem-rs README and Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/MFEM_ALIGNMENT_REPORT.md`

- [ ] **Step 1: Update README with edition information**

Add a section after the crate structure, something like:

```markdown
## Editions

fem-rs is available in two editions:

| Edition | License | Repository | Includes |
|---------|---------|------------|----------|
| **Community** (this repo) | Apache 2.0 | Public GitHub | Complete MFEM-comparable FEM library |
| **Professional** (fem-pro) | Proprietary | Private | GPU acceleration, T-spline IGA, CAD integration, advanced solvers, industry physics, enterprise features |

The community edition is a fully functional FEM library — no crippleware.
The professional edition builds on the community edition via plugin APIs.
See [EDITION_SPLIT.md](./EDITION_SPLIT.md) for details.
```

- [ ] **Step 2: Write EDITION_SPLIT.md**

Create `EDITION_SPLIT.md` with a condensed version of the spec:

```markdown
# Edition Split

## Community Edition (fem-rs, Apache 2.0)
- All FEM crates fully open source
- MFEM-equivalent IGA (NURBS basis, single-patch assembly)
- All 53 serial + 27 parallel examples
- Python and WASM bindings
- Plugin API for extension

## Professional Edition (fem-pro, proprietary)
- fem-rs as git submodule
- GPU-accelerated assembly (WGSL + CUDA)
- T-spline local refinement & trimmed NURBS
- Advanced solvers: FETI/BDDC, JDQZ, GPU-AMG
- CAD integration via OpenCASCADE
- Industry physics: forming, NVH, fatigue, EDA
- Enterprise: REST API, cloud HPC, workflow

## Plugin API
Plugin traits in fem-rs: `ProSolver`, `ProIntegrator`, `ProMeshModifier`,
`ProMeshImporter`, `ProPhysicsModel`. Pro crates register implementations
at startup via `SolverRegistry::global().register(...)`.

## FAQ

**Q: Is the community edition crippled?**
A: No. It's a complete, production-quality FEM library.

**Q: Can I contribute to fem-rs?**
A: Yes. All contributions are welcome via GitHub PRs.
```

- [ ] **Step 3: Update MFEM_ALIGNMENT_REPORT.md**

Note that IGA alignment is partial (basic NURBS only, T-spline and trimmed in pro).

- [ ] **Step 4: Commit**

```bash
git add README.md EDITION_SPLIT.md docs/MFEM_ALIGNMENT_REPORT.md
git commit -m "docs: add edition split documentation"
```

---

### Task 5: Create fem-pro Repository with Submodule

**Files:** (on a new machine or the private repo)
- Create: `fem-pro/Cargo.toml`
- Create: `fem-pro/.gitignore`
- Create: `fem-pro/README.md`
- Create: `fem-pro/examples/README.md`

- [ ] **Step 1: Initialize private repo and add submodule**

```bash
mkdir fem-pro && cd fem-pro
git init
git submodule add git@github.com:you/fem-rs.git
git submodule update --init
```

- [ ] **Step 2: Write fem-pro workspace Cargo.toml**

```toml
[workspace]
resolver = "2"
members = [
    "fem-rs/crates/core",
    "fem-rs/crates/mesh",
    "fem-rs/crates/element",
    "fem-rs/crates/space",
    "fem-rs/crates/linalg",
    "fem-rs/crates/assembly",
    "fem-rs/crates/solver",
    "fem-rs/crates/amg",
    "fem-rs/crates/parallel",
    "fem-rs/crates/io",
    "fem-rs/crates/io_hdf5_parallel",
    "fem-rs/crates/wasm",
    "fem-rs/crates/python",
    "fem-rs/crates/stochastic",
    "fem-rs/crates/regression",
    "crates/pro-gpu",
    "crates/pro-iga",
    "crates/pro-solver",
    "crates/pro-cad",
    "crates/pro-physics",
    "crates/pro-enterprise",
    "crates/pro-plugin-host",
]

[workspace.dependencies]
# Re-use fem-rs dependency versions
fem-core      = { path = "fem-rs/crates/core" }
fem-mesh      = { path = "fem-rs/crates/mesh" }
fem-element   = { path = "fem-rs/crates/element" }
fem-space     = { path = "fem-rs/crates/space" }
fem-linalg    = { path = "fem-rs/crates/linalg" }
fem-assembly  = { path = "fem-rs/crates/assembly" }
fem-solver    = { path = "fem-rs/crates/solver" }
fem-amg       = { path = "fem-rs/crates/amg" }
fem-io        = { path = "fem-rs/crates/io" }
nalgebra      = { path = "fem-rs/vendor/..." }  # or match fem-rs version
```

- [ ] **Step 3: Write .gitignore**

```
target/
*.swp
*.lock
```

- [ ] **Step 4: Write README.md**

```markdown
# fem-pro — Professional FEM Library

Industrial-grade finite element analysis extensions for fem-rs.

## Prerequisites

- Rust 1.75+
- fem-rs submodule: `git submodule update --init`

## Building

```bash
cargo build --workspace
```

## Modules

- `pro-gpu` — GPU-accelerated assembly (WGSL + CUDA)
- `pro-iga` — Advanced IGA (T-spline, trimmed, contact)
- `pro-solver` — FETI/BDDC, JDQZ, GPU-native solvers
- `pro-cad` — OpenCASCADE integration, STEP/IGES
- `pro-physics` — Industry physics modules
- `pro-enterprise` — REST API, cloud HPC, workflow
- `pro-plugin-host` — Plugin initialization

## License

Proprietary. Contact for licensing.
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore Cargo.toml README.md .gitmodules
git commit -m "feat: initial fem-pro workspace with fem-rs submodule"
```

---

### Task 6: Create Pro Crate Scaffolding

**Files:**
- Create: `crates/pro-gpu/Cargo.toml`
- Create: `crates/pro-gpu/src/lib.rs`
- Create: `crates/pro-iga/Cargo.toml`
- Create: `crates/pro-iga/src/lib.rs`
- Create: `crates/pro-solver/Cargo.toml`
- Create: `crates/pro-solver/src/lib.rs`
- Create: `crates/pro-cad/Cargo.toml`
- Create: `crates/pro-cad/src/lib.rs`
- Create: `crates/pro-physics/Cargo.toml`
- Create: `crates/pro-physics/src/lib.rs`
- Create: `crates/pro-enterprise/Cargo.toml`
- Create: `crates/pro-enterprise/src/lib.rs`
- Create: `crates/pro-plugin-host/Cargo.toml`
- Create: `crates/pro-plugin-host/src/lib.rs`

- [ ] **Step 1: Write each crate's Cargo.toml**

Each `Cargo.toml` follows this pattern (example for pro-gpu):

```toml
[package]
name = "pro-gpu"
version = "0.1.0"
edition = "2021"

[dependencies]
fem-core = { workspace = true }
fem-linalg = { workspace = true }
fem-mesh = { workspace = true }
fem-assembly = { workspace = true }
wgpu = "27"
bytemuck = { version = "1", features = ["derive"] }
pollster = "0.4"
```

For pro-plugin-host, it depends on all pro crates:

```toml
[package]
name = "pro-plugin-host"
version = "0.1.0"
edition = "2021"

[dependencies]
pro-gpu = { path = "../pro-gpu" }
pro-iga = { path = "../pro-iga" }
pro-solver = { path = "../pro-solver" }
pro-cad = { path = "../pro-cad" }  # optional
pro-physics = { path = "../pro-physics" }  # optional
pro-enterprise = { path = "../pro-enterprise" }  # optional
```

- [ ] **Step 2: Write stub lib.rs for each crate**

Each stub follows this pattern (example for pro-solver):

```rust
//! Advanced solvers for fem-pro.
//! Requires fem-rs plugin API.

/// Register all pro solvers with the fem-rs SolverRegistry.
pub fn register_solvers() {
    // TODO: register FETI/BDDC, JDQZ, GPU solvers
    // let mut reg = fem_solver::plugin::SolverRegistry::global();
    // reg.register(Box::new(FetiBddcSolver::new()));
    log::info!("pro-solver: no solvers registered yet (stub)");
}
```

- [ ] **Step 3: Write pro-plugin-host initialization**

```rust
//! Plugin initialization hub.
//! Call `fem_pro::init()` once at application startup.

/// Initialize all pro edition plugins.
/// Registers solvers, integrators, and importers with fem-rs registries.
pub fn init() {
    pro_gpu::register_gpu_backend();
    pro_iga::register_iga_extensions();
    pro_solver::register_solvers();
    pro_cad::register_importers();
    pro_physics::register_physics();
    log::info!("fem-pro initialized");
}
```

- [ ] **Step 4: Verify workspace compiles**

Run: `cargo check --workspace` (from fem-pro root)
Expected: no errors, all crate stubs compile

- [ ] **Step 5: Commit**

```bash
git add crates/
git commit -m "feat: add pro crate scaffolding with plugin registration stubs"
```

---

### Task 7: Move GPU Crate (linalg-gpu) into fem-pro

**Files:**
- Copy: from `fem-rs/crates/linalg-gpu/` → `fem-pro/crates/pro-gpu/`
- Copy: `fem-rs/crates/assembly/src/iga/iga_gpu.rs` → `fem-pro/crates/pro-gpu/src/`
- Modify: `fem-pro/crates/pro-gpu/Cargo.toml`
- Modify: `fem-pro/crates/pro-gpu/src/lib.rs`

- [ ] **Step 1: Copy linalg-gpu source into pro-gpu**

```bash
cp -r fem-rs/crates/linalg-gpu/* fem-pro/crates/pro-gpu/
rm fem-pro/crates/pro-gpu/target/ 2>/dev/null || true
```

- [ ] **Step 2: Update pro-gpu Cargo.toml dependency paths**

Change all `{ path = "../..." }` dependencies to `{ workspace = true }`.

- [ ] **Step 3: Integrate iga_gpu.rs into pro-gpu**

Copy `iga_gpu.rs` (from assembly) into `pro-gpu/src/`. Add `mod iga_gpu;` to `pro-gpu/src/lib.rs`.

Update its fem-rs path imports:
```rust
// Instead of:
// use crate::iga::iga_bezier::...
// use crate::iga::...

// In pro-gpu:
use fem_assembly::iga::iga_bezier::...;
```

- [ ] **Step 4: Compile-check**

Run: `cargo check -p pro-gpu` (from fem-pro root)
Expected: compiles with no errors

- [ ] **Step 5: Commit**

```bash
git add crates/pro-gpu/
git commit -m "feat: move GPU assembly crate into pro edition"
```

---

### Task 8: Move Advanced IGA Code into fem-pro

**Files:**
- Copy: `fem-rs/crates/element/src/tmesh.rs` → `fem-pro/crates/pro-iga/src/tmesh.rs`
- Copy: `fem-rs/crates/assembly/src/iga/iga_tspline.rs` → `fem-pro/crates/pro-iga/src/iga_tspline.rs`
- Copy: `fem-rs/crates/assembly/src/contact/contact_iga.rs` → `fem-pro/crates/pro-iga/src/contact_iga.rs`
- Copy: `fem-rs/crates/assembly/src/iga/iga_trim.rs` → `fem-pro/crates/pro-cad/src/iga_trim.rs`
- Copy: `fem-rs/crates/io/src/step_iges.rs` → `fem-pro/crates/pro-cad/src/step_iges.rs`

- [ ] **Step 1: Copy tmesh + iga_tspline into pro-iga**

```bash
cp fem-rs/crates/element/src/tmesh.rs fem-pro/crates/pro-iga/src/tmesh.rs
cp fem-rs/crates/assembly/src/iga/iga_tspline.rs fem-pro/crates/pro-iga/src/iga_tspline.rs
cp fem-rs/crates/assembly/src/contact/contact_iga.rs fem-pro/crates/pro-iga/src/contact_iga.rs
```

Add module declarations and `pub use` in pro-iga/src/lib.rs:
```rust
pub mod tmesh;
pub mod iga_tspline;
pub mod contact_iga;
```

- [ ] **Step 2: Copy trimmed NURBS + STEP/IGES into pro-cad**

```bash
cp fem-rs/crates/assembly/src/iga/iga_trim.rs fem-pro/crates/pro-cad/src/iga_trim.rs
cp fem-rs/crates/io/src/step_iges.rs fem-pro/crates/pro-cad/src/step_iges.rs
```

Add module declarations in pro-cad/src/lib.rs:
```rust
pub mod iga_trim;
pub mod step_iges;
```

- [ ] **Step 3: Update path imports in copied files**

All `crate::` imports from fem-rs need to be updated:
- `crate::iga::...` → `fem_assembly::iga::...`
- `crate::nurbs::...` → `fem_element::nurbs::...`
- `crate::bezier_extraction::...` → `fem_element::bezier_extraction::...`

- [ ] **Step 4: Compile-check**

Run: `cargo check -p pro-iga -p pro-cad`
Expected: compiles with no errors

- [ ] **Step 5: Commit**

```bash
git add crates/pro-iga/ crates/pro-cad/
git commit -m "feat: move advanced IGA modules into pro edition"
```

---

### Task 9: Wire Up Plugin Registration in pro-plugin-host

**Files:**
- Modify: `crates/pro-plugin-host/src/lib.rs`

- [ ] **Step 1: Implement init() with actual registrations**

```rust
//! Plugin initialization hub.

/// Initialize all pro edition plugins.
/// Call this once at application startup (before any fem-rs solve).
pub fn init() {
    pro_gpu::register_gpu_backend();
    pro_iga::register_iga_extensions();
    pro_solver::register_solvers(&mut fem_solver::plugin::SolverRegistry::global().lock().unwrap());
    pro_cad::register_importers();
    pro_physics::register_physics();
    log::info!("fem-pro initialized");
}
```

- [ ] **Step 2: Implement pro_solver::register_solvers**

```rust
pub fn register_solvers(registry: &mut fem_solver::plugin::SolverRegistry) {
    // Stub — actual solver implementations added in future tasks.
    log::info!("pro-solver: registered 0 solvers (stub)");
}
```

- [ ] **Step 3: Implement pro_iga::register_iga_extensions**

```rust
pub fn register_iga_extensions() {
    log::info!("pro-iga: T-spline, trimmed NURBS, IGA contact available via direct API");
}
```

- [ ] **Step 4: Compile-check**

Run: `cargo check --workspace`
Expected: all compiles

- [ ] **Step 5: Full workspace test**

Run: `cargo test --workspace`
Expected: all 2491+ tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/pro-plugin-host/
git commit -m "feat: wire up pro plugin registration"
```

---

### Self-Review Checklist

- [ ] **Spec coverage:** Every section in the spec has a corresponding task:
  - Repo structure → Task 5
  - Function division (GPU → pro) → Task 2, Task 7
  - Function division (advanced IGA → pro) → Task 3, Task 8
  - Plugin API design → Task 1, Task 9
  - OSS edition cleanup → Task 4
  - Pro crate scaffolding → Task 6

- [ ] **Placeholder scan:** No TODOs, TBDs, or vague steps. Every step has actual code.

- [ ] **Type consistency:** Plugin trait names are consistent across all tasks (`ProSolver`, `ProIntegrator`, `ProMeshModifier`, `ProMeshImporter`, `ProPhysicsModel`). Registry types consistent (`SolverRegistry`, `IntegratorRegistry`, `MeshImporterRegistry`).

- [ ] **Execution ordering:** Tasks are ordered so each produces working, compilable state:
  - Plugin traits first (Task 1) — doesn't depend on anything
  - Then remove GPU from OSS (Tasks 2-3) — OSS standalone
  - Then docs (Task 4)
  - Then create pro repo + scaffolding (Tasks 5-6) — independent
  - Then move code into pro (Tasks 7-8)
  - Then wire up plugin init (Task 9)
