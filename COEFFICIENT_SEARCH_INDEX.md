# Coefficient Types & Patterns Search — Documentation Index

**Search Date:** April 4, 2026  
**Codebase:** fem-rs (all 13 crates)  
**Status:** ✅ Complete  

---

## 📖 Documentation Files

### 1. **COEFFICIENT_TYPES_SEARCH.md** — Complete Technical Reference
   **Size:** 17 KB | **Sections:** 13 | **Code Examples:** 25+
   
   **Contents:**
   - Complete core crate structure (error types, scalar trait, index aliases)
   - All coefficient patterns with code examples
   - Function types found in codebase (interpolation, ODE, nonlinear)
   - Mesh tags and material labels structure
   - Piecewise constant coefficient patterns
   - Block matrix architecture
   - Complete integrator trait hierarchy
   - Assembly loop structure with detailed walkthrough
   - Finite element spaces overview
   - 11 files with coefficient mentions (with citations)
   - Key observations and recommendations
   
   **Best For:** Deep understanding, implementation details, exact code references

   **Quick Navigation:**
   - §1: Core types → `/crates/core/src/`
   - §2: Assembly patterns → `/crates/assembly/src/standard/`
   - §3: Function types → Various (e.g., `partial.rs`, `ode.rs`)
   - §8: Assembly pattern → `assembler.rs` (main entry point)

---

### 2. **COEFFICIENT_QUICK_REFERENCE.md** — Quick Lookup Tables
   **Size:** 8.5 KB | **Tables:** 12 | **Status Matrix:** Yes
   
   **Contents:**
   - 6 coefficient patterns summary table
   - Type aliases (existing vs proposed)
   - Missing type aliases (opportunities)
   - Integrator architecture diagram
   - Quadrature point data structure
   - Assembly loop pseudocode
   - Element tag usage map
   - Crate dependency graph
   - Examples: piecewise constant implementation
   - Files to modify for coefficient support
   - Key code locations by grep term
   - Coefficient options summary
   - Design recommendations
   
   **Best For:** Quick answers, developer reference, implementation planning

   **When to use:**
   - "What patterns exist?" → Table 1
   - "What type aliases do we have?" → Table 2
   - "Where are the gaps?" → Table 3
   - "How do I find code?" → Table 10

---

### 3. **COEFFICIENT_ARCHITECTURE.md** — Visual Hierarchies & Data Flow
   **Size:** 11 KB | **Diagrams:** ASCII trees & flow charts | **Paths:** 4
   
   **Contents:**
   - Type hierarchy diagram (ASCII)
   - Storage location tree structure (crates → files)
   - 5 detailed crate organization trees:
     - Core types layout
     - Mesh tags structure
     - Integrators organization
     - FE spaces hierarchy
     - Linear algebra components
   - 4 data flow paths with examples:
     1. Uniform coefficient
     2. Spatially-varying coefficient
     3. Per-element (by tag) coefficient
     4. Per-DOF (GridFunction) coefficient
   - Type alias opportunities (current + proposed)
   - Assembly signature enhancements
   - Status matrix (what exists vs missing)
   
   **Best For:** Understanding architecture, data flow, implementation planning

   **Visual Elements:**
   - Tree diagrams for file organization
   - Data flow charts with `→` arrows
   - Status matrix with ✅/❌/⚠️ indicators
   - Type signature comparisons

---

## 🔍 Quick Question → Document Mapping

| Question | Best Document | Section |
|----------|---------------|---------|
| What coefficient patterns exist? | Quick Reference | Table 1 |
| Where is MassIntegrator defined? | Main Reference | §2 |
| What are element tags? | Main Reference | §4 |
| How does assembly work? | Architecture | "Assembly Loop Pattern" |
| What's missing from the library? | Architecture | "Summary" table |
| Which files mention coefficients? | Main Reference | §11 |
| How do I add ScalarFn? | Quick Reference | Design Recommendations |
| Where is QpData? | Architecture | "Storage Locations §3" |
| What's the error type? | Main Reference | §1 |
| How do I create GridFunction? | Architecture | "Data Flow §4" |

---

## 📂 Files Examined (Organized by Crate)

### Core (`crates/core/src/`)
- ✓ `lib.rs` — module structure
- ✓ `error.rs` — error types, FemError, FemResult
- ✓ `types.rs` — index aliases (NodeId, ElemId, DofId, FaceId)
- ✓ `scalar.rs` — Scalar trait (f32/f64)
- ✓ `point.rs` — coordinate/matrix aliases

### Mesh (`crates/mesh/src/`)
- ✓ `lib.rs` — module exports
- ✓ `boundary.rs` — BoundaryTag, PhysicalGroup
- ✓ `topology.rs` — MeshTopology trait
- ✓ `simplex.rs` — SimplexMesh with elem_tags

### Assembly (`crates/assembly/src/`)
- ✓ `lib.rs` — module structure
- ✓ `integrator.rs` — BilinearIntegrator, LinearIntegrator, QpData
- ✓ `assembler.rs` — main assembly loop
- ✓ `vector_integrator.rs` — vector FE integrators
- ✓ `partial.rs` — PADiffusionOperator, PAMassOperator, LumpedMassOperator
- ✓ `dg.rs` — DG assembly
- ✓ `mixed.rs` — mixed bilinear forms
- ✓ `nonlinear.rs` — nonlinear forms
- ✓ `vector_assembler.rs` — vector assembly
- ✓ `standard/mass.rs` — MassIntegrator
- ✓ `standard/diffusion.rs` — DiffusionIntegrator
- ✓ `standard/curl_curl.rs` — CurlCurlIntegrator
- ✓ `standard/vector_mass.rs` — VectorMassIntegrator

### Space (`crates/space/src/`)
- ✓ `lib.rs` — module structure
- ✓ `fe_space.rs` — FESpace trait, SpaceType
- ✓ `h1.rs` — H1Space
- ✓ `l2.rs` — L2Space
- ✓ `vector_h1.rs` — VectorH1Space
- ✓ `hcurl.rs` — HCurlSpace
- ✓ `hdiv.rs` — HDivSpace

### Linear Algebra (`crates/linalg/src/`)
- ✓ `block.rs` — BlockVector, BlockMatrix

### Solver (`crates/solver/src/`)
- ✓ `ode.rs` — TimeStepper, RHS function patterns

---

## 🎯 Key Findings Summary

### ✅ Exists
1. **6 coefficient patterns** in current codebase
2. **7 type aliases** in core (NodeId, ElemId, DofId, FaceId, Coord2/3, Vec2/3, Mat2x2/3x3, BoundaryTag)
3. **Element tags** (stored and accessible via MeshTopology trait)
4. **BilinearIntegrator trait** with extensible design
5. **Spatially-varying coefficients** (PADiffusionOperator pattern)
6. **Block structures** (BlockVector, BlockMatrix)

### ❌ Missing
1. **ScalarFn type alias** (raw `dyn Fn(&[f64])->f64` used instead)
2. **VectorFn type alias** (raw `dyn Fn(&[f64])->Vec<f64>` used instead)
3. **MatrixFn type alias** (not found)
4. **ElementTag type alias** (parallel to BoundaryTag)
5. **GridFunction struct** (only separate interpolation + vector)
6. **Per-element material lookup in assembly** (tags exist but unused)
7. **Context-aware integrators** (QpData only, no element reference)

### ⚠️ Partial
1. **Per-DOF coefficients** (LumpedMass exists; general case missing)

---

## 🚀 For Implementation Planning

### Phase 1: Type Aliases (Low effort, high value)
```rust
// Add to core/types.rs
pub type ScalarFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
pub type VectorFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;
pub type MaterialResolver = Box<dyn Fn(i32, &[f64]) -> f64 + Send + Sync>;
pub type ElementTag = i32;
```
**Files to modify:** `core/lib.rs`, `core/types.rs`

### Phase 2: GridFunction (Medium effort)
```rust
// Create space/grid_function.rs
pub struct GridFunction<S: FESpace> {
    pub data: Vector<f64>,
    space: S,
}
```
**Files to modify:** `space/lib.rs`, `space/grid_function.rs`

### Phase 3: Enhanced Assembly (Higher effort)
- Extend `QpData` with element context
- Pass element tags to integrators
- Support material maps in assembly

**Files to modify:** `assembly/integrator.rs`, `assembly/assembler.rs`

---

## 📋 Search Methods Used

### Grep Patterns
| Pattern | Matches | Files |
|---------|---------|-------|
| `(?i)(coeff\|coefficient)` | 9 matches | assembly, linalg, space, solver |
| `type\s+(ScalarFn\|VectorFn\|MatrixFn)` | 0 matches | (none found) |
| `GridFunction` | 0 matches | (none found) |
| `(?i)material\|tag\|region\|zone` | 15 files | mesh, assembly |
| `(?i)piecewise.*constant\|per.element` | 15 files | mesh |
| `element_tag\|elem_tags` | multiple | mesh, assembly |
| `dyn Fn` | 10+ locations | Various |

### File Reads
- 100+ .rs files examined
- 25+ files read completely
- Full focus on: integrators, core types, assembly, spaces

---

## 🔗 Related Documents

- `DESIGN_PLAN.md` — Overall fem-rs design
- `ASSEMBLY_CRATE_ANALYSIS.md` — Assembly-specific analysis
- `TECHNICAL_SPEC.md` — Technical specifications
- `MFEM_MAPPING.md` — Mapping to MFEM concepts

---

## 📞 Using These Documents

**For developers:**
1. Start with COEFFICIENT_QUICK_REFERENCE.md for overview
2. Use COEFFICIENT_TYPES_SEARCH.md for specific file locations
3. Refer to COEFFICIENT_ARCHITECTURE.md for data flow understanding

**For planning:**
1. Check "Design Recommendations" in Quick Reference
2. Review "What Exists vs What's Missing" matrix in Architecture
3. Reference exact file paths from Main Reference

**For implementation:**
1. Copy exact code locations from Main Reference
2. Use type signatures from Architecture
3. Follow patterns from existing integrators

---

**Generated:** 2026-04-04  
**Status:** Ready for implementation phase  
**Quality:** Comprehensive with exact citations
