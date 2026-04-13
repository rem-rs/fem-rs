# C1 Backend Contract Freeze (v0.1)

Date: 2026-04-13
Scope: cross-subproject interface baseline for linger + reed + jsmpi.
Status: frozen for C1; additive-only changes allowed until C2 starts.

## 1. Goals

- Freeze backend-selection API surface before external backend implementation work.
- Keep default behavior unchanged for existing fem-rs applications.
- Define deterministic fallback behavior for unsupported targets (especially wasm).

## 2. Canonical Backend IDs

- native-linger
- hypre-rs
- petsc-rs
- mumps
- mkl

Notes:
- hypre-rs is the pure-Rust HYPRE-equivalent capability track.
- petsc-rs is the pure-Rust PETSc-equivalent capability track.
- mumps/mkl remain optional backend tracks (current C1 runtime falls back to native-linger).

## 3. linger Contract (source of truth)

C1 public API baseline in linger:

- ExternalBackend enum:
  - HypreRs
  - PetscRs (canonical)
  - PetscFfi (legacy compatibility alias)
  - Mumps
  - Mkl
- BackendCapabilities struct:
  - hypre_rs
  - petsc_ffi (legacy field retained for compatibility; canonical capability alias = petsc-rs)
  - mumps
  - mkl
  - wasm_target
- EffectiveBackend enum:
  - NativeLinger
  - External(ExternalBackend)
- BackendSelectionReport struct:
  - requested
  - effective
  - capabilities
  - note

Builder contract:

- SolverBuilder::external_backend(ExternalBackend)
- SolverBuilder::backend_capabilities()
- SolverBuilder::backend_selection_report()

Policy:

- C1 always executes through native-linger path.
- Requested external backends are resolved to deterministic fallback notes until wiring is implemented.
- On wasm targets, external FFI requests must resolve to native-linger with explicit reason.

## 4. reed Integration Contract

reed must consume linger contract without app-level API churn:

- Preserve existing resource strings and object factory behavior.
- Add a backend-selection handoff path that can pass canonical backend IDs to linger.
- Own integration landing paths for GPU and MKL-backed execution while honoring linger capability reports.
- If no backend is requested, reed defaults to native-linger behavior.

Minimum C1 acceptance for reed:

- One integration path can request a backend ID and obtain a deterministic selection report.
- Existing examples continue to run unchanged.
- Ownership note: C2+ implementation of GPU and MKL tracks is reed-led; linger remains contract source-of-truth.

## 5. jsmpi Integration Contract

jsmpi must expose runtime constraints for backend selection:

- wasm/browser execution must advertise external FFI backends as unavailable.
- Capability/fallback messaging must be deterministic and user-visible.
- No hidden silent downgrade for explicit backend requests.

Minimum C1 acceptance for jsmpi:

- Browser path can report fallback from petsc-rs/mumps/mkl to native-linger.
- Contract note is surfaced in logs or diagnostics output.

## 6. Compatibility Rules (C1)

- Additive changes only: no rename/removal of contract fields.
- Canonical backend IDs are stable through C2.
- Default build and default runtime behavior must remain unchanged.

## 7. Exit Criteria Mapping

C1 is considered complete when:

- linger exposes the contract APIs above.
- reed and jsmpi document and consume the same backend IDs.
- fallback behavior is deterministic on wasm and non-wasm targets.
