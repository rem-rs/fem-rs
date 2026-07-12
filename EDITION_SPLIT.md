# Edition Split

## Community Edition (fem-rs, Apache 2.0)

All FEM crates fully open source:
- MFEM-equivalent IGA (NURBS basis, single-patch assembly, Bezier extraction)
- WGSL GPU acceleration (optional `--features gpu`)
- All 53 serial + 27 parallel examples
- Python and WASM bindings
- Plugin API for extension (no-op when empty)

## Professional Edition (fem-pro, proprietary)

fem-rs as git submodule with additional crates:
- **pro-gpu** — CUDA-native solvers and batched assembly
- **pro-iga** — T-spline local refinement, trimmed NURBS, multi-patch C¹/C², IGA contact
- **pro-solver** — FETI/BDDC, JDQZ, GPU-AMG, parallel-in-time
- **pro-cad** — OpenCASCADE bindings, STEP/IGES import, defeaturing, mesh morphing
- **pro-physics** — Advanced plasticity, NVH, fatigue, forming, EDA multi-physics, XFEM
- **pro-enterprise** — REST API, cloud HPC, workflow, license management

## Plugin API

Plugin traits in fem-rs: `ProSolver`, `ProIntegrator`, `ProMeshModifier`,
`ProMeshImporter`, `ProPhysicsModel`. Pro crates register implementations
at startup via `fem_pro::init()`.

```rust
// Open-source edition: built-in solvers only.
use fem_rs::prelude::*;
let solver = CGSolver::new(pc);
solver.solve(&matrix, &rhs)?;

// Professional edition: fem-pro prelude re-exports fem-rs + pro.
use fem_pro::prelude::*;
fem_pro::init();
let solver = SolverRegistry::global().get("feti-bddc")?;
solver.solve(&matrix, &rhs)?;
```

## FAQ

**Q: Is the community edition crippled?**
A: No. It's a complete, production-quality FEM library comparable to MFEM.

**Q: Can I contribute to fem-rs?**
A: Yes. All contributions are welcome via GitHub PRs.

**Q: Can I use fem-rs in commercial products?**
A: Yes. fem-rs is Apache 2.0 licensed. The professional edition adds
industrial-grade features under a separate proprietary license.

**Q: How do I build with GPU acceleration?**
A: `cargo build --features gpu` enables WGSL compute shaders for
assembly. CUDA support is a professional edition feature.
