# C1 Backend Contract Freeze (v0.2, Pure Rust)

Date: 2026-04-19
Scope: cross-subproject interface baseline for linger + reed + jsmpi.
Status: frozen for C1; additive-only changes allowed until C2 starts.

## 1. Goals

- Freeze backend-selection API surface for pure-Rust execution paths.
- Keep default behavior unchanged for existing fem-rs applications.
- Define deterministic fallback behavior for unsupported targets (especially wasm).

## 2. Canonical Backend IDs

- native-linger
- mumps
- mkl

Notes:
- `native-linger` is the default and primary solver path.
- `mumps` and `mkl` are compatibility contract names resolved to linger native direct solves.
- No additional external-solver track is part of the active contract.

## 3. linger Contract (source of truth)

C1 public API baseline in linger:

- ExternalBackend enum:
  - Mumps
  - Mkl
- BackendCapabilities struct:
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
- Requested compatibility backends are resolved to deterministic notes.
- On wasm targets, direct-solver compatibility requests resolve to native-linger with explicit reason.

## 4. reed Integration Contract

reed must consume linger contract without app-level API churn:

- Preserve existing resource strings and object factory behavior.
- Add a backend-selection handoff path that can pass canonical backend IDs to linger.
- Own integration landing paths for GPU execution while honoring linger capability reports.
- If no backend is requested, reed defaults to native-linger behavior.

Minimum C1 acceptance for reed:

- One integration path can request a backend ID and obtain a deterministic selection report.
- Existing examples continue to run unchanged.

## 5. jsmpi Integration Contract

jsmpi must expose runtime constraints for backend selection:

- wasm/browser execution must advertise native-direct compatibility requests as unavailable.
- Capability/fallback messaging must be deterministic and user-visible.
- No hidden silent downgrade for explicit backend requests.

Minimum C1 acceptance for jsmpi:

- Browser path can report fallback from mumps/mkl compatibility requests to native-linger.
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
