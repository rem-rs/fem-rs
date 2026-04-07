# fem-rs Parallel Infrastructure Architecture

## Overview
The fem-rs project implements a **modular, pluggable MPI abstraction layer** that allows the same solver code to compile and run against three different backends without any `#[cfg]` guards:

1. **Native MPI** (`mpi 0.8` via `rsmpi`) — for HPC environments
2. **Serial** — single-rank stub for testing without MPI
3. **WASM Web Workers** — planned for browser-based parallelism

Total codebase: **2,715 lines** across 14 files in `crates/parallel/src/`.

---

## 1. TRAIT-BASED ABSTRACTION LAYER

### Core Trait: `CommBackend` (object-safe)
**File:** `crates/parallel/src/backend/mod.rs`

All backends must implement this single trait:

```rust
pub trait CommBackend: Send + Sync {
    // Topology queries
    fn rank(&self) -> Rank;
    fn size(&self) -> usize;

    // Synchronisation
    fn barrier(&self);

    // Collectives (typed but operate on raw bytes internally)
    fn allreduce_sum_f64(&self, local: f64) -> f64;
    fn allreduce_sum_i64(&self, local: i64) -> i64;
    fn broadcast_bytes(&self, root: Rank, buf: &mut Vec<u8>);

    // Point-to-point
    fn send_bytes(&self, dest: Rank, tag: i32, data: &[u8]);
    fn recv_bytes(&self, src: Rank, tag: i32) -> Vec<u8>;

    // Sparse variable-length all-to-all
    fn alltoallv_bytes(&self, sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)>;
}
```

**Design principles:**
- **Object-safe**: `Comm` holds `Box<dyn CommBackend>` in an `Arc`, enabling runtime polymorphism
- **Send + Sync**: Required so `Comm` can be `Send + Sync` (necessary for `MeshTopology` trait bound)
- **Byte-level API**: All ops move raw `[u8]` to decouple backends from specific numeric types
- **No generics**: Avoids monomorphization explosion across backends

---

## 2. PUBLIC COMMUNICATOR API: `Comm`
**File:** `crates/parallel/src/comm.rs`

The **only** user-facing handle into the parallel environment. It wraps a `BackendBox` in an `Arc`:

```rust
pub struct Comm {
    inner: std::sync::Arc<Box<dyn CommBackend>>,
}

impl Comm {
    pub fn from_backend(backend: BackendBox) -> Self;

    // Topology
    pub fn rank(&self) -> Rank;
    pub fn size(&self) -> usize;
    pub fn is_root(&self) -> bool;

    // Synchronisation
    pub fn barrier(&self);

    // Typed collectives (wrappers around raw byte ops)
    pub fn allreduce_sum_f64(&self, local: f64) -> f64;
    pub fn allreduce_sum_i64(&self, local: i64) -> i64;
    pub fn broadcast_usize(&self, root: Rank, val: usize) -> usize;

    // Typed point-to-point (uses bytemuck::Pod for safe serialization)
    pub fn send<T: Pod>(&self, dest: Rank, tag: i32, data: &[T]);
    pub fn recv<T: Pod>(&self, src: Rank, tag: i32) -> Vec<T>;

    // Raw byte ops (used by GhostExchange)
    pub fn send_bytes(&self, dest: Rank, tag: i32, data: &[u8]);
    pub fn recv_bytes(&self, src: Rank, tag: i32) -> Vec<u8>;
    pub fn broadcast_bytes(&self, root: Rank, buf: &mut Vec<u8>);
    pub fn alltoallv_bytes(&self, sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)>;
}
```

**Key properties:**
- Cheaply cloneable (Arc-wrapped backend)
- Stateless: all communication is explicit via method calls
- No global MPI_COMM_WORLD coupling: can be composed and passed around freely

---

## 3. BACKEND IMPLEMENTATIONS

### 3.1 `SerialBackend` (native, no-MPI)
**File:** `crates/parallel/src/backend/native.rs`

Single-rank stub always available:

```rust
pub struct SerialBackend;

impl CommBackend for SerialBackend {
    fn rank(&self) -> Rank { 0 }
    fn size(&self) -> usize { 1 }
    fn barrier(&self) {}  // no-op
    fn allreduce_sum_f64(&self, local: f64) -> f64 { local }  // identity
    fn allreduce_sum_i64(&self, local: i64) -> i64 { local }   // identity
    // send/recv panic (no other rank exists)
}
```

**Use case:** Testing without MPI install; serial solver verification.

---

### 3.2 `NativeMpiBackend` (feature-gated)
**File:** `crates/parallel/src/backend/native.rs`

Wraps `rsmpi` (`mpi` crate v0.8) and delegates directly to MPI functions:

```rust
#[cfg(feature = "mpi")]
pub struct NativeMpiBackend {
    rank: Rank,
    size: i32,
}

impl CommBackend for NativeMpiBackend {
    fn barrier(&self) {
        SystemCommunicator::world().barrier();  // MPI_Barrier
    }

    fn allreduce_sum_f64(&self, local: f64) -> f64 {
        SystemCommunicator::world()
            .all_reduce_into(&local, &mut result, &SystemOperation::sum());
        result
    }

    fn alltoallv_bytes(&self, sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)> {
        // Step 1: AllToAll on message counts
        let mut send_counts = vec![0i64; n];
        for (dest, data) in sends {
            send_counts[*dest as usize] = data.len() as i64;
        }
        let mut recv_counts = vec![0i64; n];
        world.all_to_all_into(&send_counts, &mut recv_counts);

        // Step 2: post all sends first (MPI buffering)
        for (dest, data) in sends {
            let tag = TAG_A2AV + self.rank;
            world.process_at_rank(*dest).send_with_tag(data.as_slice(), tag);
        }

        // Step 3: blocking receives in sender-rank order
        let mut results = Vec::new();
        for src in 0..n {
            if recv_counts[src] == 0 { continue; }
            let tag = TAG_A2AV + src as i32;
            let (msg, _status) = world
                .process_at_rank(src as i32)
                .receive_vec_with_tag::<u8>(tag);
            results.push((src as Rank, msg));
        }
        results
    }
}
```

**Activation:** Feature `mpi` + `#[cfg(not(target_arch = "wasm32"))]`

---

### 3.3 `ChannelBackend` (in-process test backend)
**File:** `crates/parallel/src/backend/channel.rs`

Implements `CommBackend` using `Mutex` + `Condvar` primitives so N OS threads can communicate:

```rust
pub struct ChannelBackend {
    rank: Rank,
    shared: Arc<ChannelShared>,
}

pub(crate) struct ChannelShared {
    n: usize,
    p2p: Vec<Vec<MsgQueue>>,  // [from][to] message queues
    a2av_slots: Vec<Vec<Mutex<Option<Vec<u8>>>>>,  // staging grid
    barrier: Mutex<(usize, usize)>,  // (arrived, gen)
    barrier_cv: Condvar,
    reduce_f64: Mutex<AllReduceF64>,  // { sum, result, arrived, gen }
    reduce_f64_cv: Condvar,
    // ... similar for i64, broadcast, alltoallv
}
```

**Algorithm:**
- **Barrier**: generation-counted rendezvous pattern on `Condvar`
- **AllReduce**: accumulators with generation counter; last arrival computes and caches; waiters read cache
- **Point-to-point**: per-rank message FIFO queues; send pushes, recv blocks on `Condvar`
- **AllToAllv**: 4-phase protocol with staging slots and generation-counted barriers

**Use case:** Unit tests; CI environments; threaded simulation of multi-rank behavior.

---

### 3.4 `WasmWorkerBackend` (planned Web Worker backend)
**File:** `crates/parallel/src/backend/wasm.rs`

Currently a **stub** for `wasm32-unknown-unknown` target:

```rust
pub struct WasmWorkerBackend {
    rank: u32,
    size: u32,
    arena: Option<SharedArena>,  // SharedArrayBuffer handle (future)
}

impl WasmWorkerBackend {
    pub fn single() -> Self {  // Rank-0 / size-1 stub
        WasmWorkerBackend { rank: 0, size: 1, arena: None }
    }

    pub fn from_init_msg(rank: u32, size: u32, arena: SharedArena) -> Self {
        // Full implementation: TODO!
    }
}

unsafe impl Send for WasmWorkerBackend {}
unsafe impl Sync for WasmWorkerBackend {}
```

**Planned architecture:**
- Each MPI "process" = Web Worker loading same WASM module
- Shared `SharedArrayBuffer` arena with fixed per-rank slots
- Blocking via `Atomics.wait()` / `Atomics.notify()`
- Equivalent semantics to MPI without round-trips through JS main thread

---

## 4. LAUNCHER ABSTRACTIONS

### Trait: `Launcher`
**File:** `crates/parallel/src/launcher/mod.rs`

```rust
pub trait Launcher: Sized {
    fn init() -> Option<Self>;          // Return None if already init
    fn world_comm(&self) -> Comm;       // Get MPI_COMM_WORLD equivalent
}

pub struct WorkerConfig {
    pub n_workers: usize,
    pub stack_bytes: Option<usize>,
}
```

---

### 4.1 `MpiLauncher` (native + feature-gated)
**File:** `crates/parallel/src/launcher/native.rs`

Thin wrapper around `mpi::initialize()`:

```rust
pub struct MpiLauncher {
    #[cfg(feature = "mpi")]
    universe: ::mpi::environment::Universe,
}

impl Launcher for MpiLauncher {
    fn init() -> Option<Self> {
        #[cfg(feature = "mpi")]
        { ::mpi::initialize().map(|u| MpiLauncher { universe: u }) }
        #[cfg(not(feature = "mpi"))]
        { Some(MpiLauncher {}) }  // Falls back to serial
    }

    fn world_comm(&self) -> Comm {
        #[cfg(feature = "mpi")]
        { Comm::from_backend(Box::new(NativeMpiBackend::from_world(&self.universe))) }
        #[cfg(not(feature = "mpi"))]
        { Comm::from_backend(Box::new(SerialBackend)) }
    }
}
```

**Note:** Multi-process spawning via `mpirun` / `mpiexec` (external to this code).

---

### 4.2 `ThreadLauncher` (native in-process)
**File:** `crates/parallel/src/launcher/native.rs`

Spawns N OS threads with `ChannelBackend` shared state:

```rust
pub struct ThreadLauncher {
    config: WorkerConfig,
}

impl ThreadLauncher {
    pub fn new(config: WorkerConfig) -> Self { /* ... */ }

    pub fn launch<F>(&self, f: F)
    where
        F: Fn(Comm) + Send + Sync + 'static,
    {
        if self.config.n_workers <= 1 {
            // Fast path: run directly
            f(Comm::from_backend(Box::new(SerialBackend)));
            return;
        }

        // Multi-worker: build shared state + spawn threads
        let shared = ChannelShared::new(self.config.n_workers);
        let f_arc = Arc::new(f);

        let handles: Vec<_> = (0..n_workers as i32)
            .map(|rank| {
                let shared_clone = Arc::clone(&shared);
                let f_clone = Arc::clone(&f_arc);
                std::thread::spawn(move || {
                    let backend = ChannelBackend::new(rank, shared_clone);
                    let comm = Comm::from_backend(Box::new(backend));
                    f_clone(comm);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }
}
```

**Usage example:**
```rust
let launcher = ThreadLauncher::new(WorkerConfig::new(4));
launcher.launch(|comm| {
    println!("rank {} / {}", comm.rank(), comm.size());
});
```

---

### 4.3 `WorkerLauncher` (wasm32 stub)
**File:** `crates/parallel/src/launcher/wasm.rs`

Currently a stub; full impl deferred to `fem-wasm` Phase 11:

```rust
pub struct WorkerLauncher {
    config: WorkerConfig,
}

impl WorkerLauncher {
    pub fn spawn<F>(&self, _f: F) where F: Fn(Comm) + 'static {
        if self.config.n_workers <= 1 {
            let comm = self.world_comm();
            _f(comm);
            return;
        }
        todo!("WorkerLauncher::spawn — SharedArrayBuffer + wasm-bindgen integration pending")
    }
}
```

---

## 5. GHOST (HALO) EXCHANGE INFRASTRUCTURE

### `GhostExchange` — Pre-computed halo pattern
**File:** `crates/parallel/src/ghost.rs` (~247 lines)

Encapsulates ownership metadata and communication pattern:

```rust
pub struct GhostExchange {
    channels: Vec<NeighbourChannel>,
}

struct NeighbourChannel {
    rank: Rank,
    send_local_ids: Vec<u32>,    // Owned nodes to send to neighbor
    recv_local_ids: Vec<u32>,    // Ghost slots to receive from neighbor
}

impl GhostExchange {
    pub fn from_partition(partition: &MeshPartition, comm: &Comm) -> Self {
        // Fast path: serial (return empty)
        if comm.size() == 1 {
            return GhostExchange { channels: Vec::new() };
        }

        // Multi-rank:
        // 1. Group ghost nodes by owner rank → requests (serialized as u32 bytes)
        // 2. alltoallv_bytes to distribute requests to owners
        // 3. Owners map global IDs back to local → send_local_ids
        // 4. Assemble channels from union of send_map + recv_slots
    }

    pub fn forward(&self, comm: &Comm, data: &mut [f64]) {
        // Owner → ghost update
        // Uses TAG_FWD + sender_rank for independent tag derivation
        for ch in &self.channels {
            let send_tag = TAG_FWD + my_rank;
            let recv_tag = TAG_FWD + ch.rank;
            // Pack owned values, send, receive ghost values
        }
    }

    pub fn reverse(&self, comm: &Comm, data: &mut [f64]) {
        // Ghost → owner accumulation (zeroes ghosts)
        // Uses TAG_REV + sender_rank
        for ch in &self.channels {
            let send_tag = TAG_REV + my_rank;
            let recv_tag = TAG_REV + ch.rank;
            // Pack ghost values (zero), send, receive accumulated
        }
    }

    pub fn n_neighbours(&self) -> usize;
    pub fn is_trivial(&self) -> bool;
}
```

**Key design:**
- Built once at `ParallelMesh` construction
- Reusable across solve steps
- Uses tag prefixes (`0x1000`, `0x2000`) to avoid collision with user messages
- Fully backend-agnostic: identical code works with MPI / Serial / ChannelBackend / WASM

---

## 6. DISTRIBUTED MESH: `ParallelMesh<M>`

### `ParallelMesh` — Local submesh + ownership metadata
**File:** `crates/parallel/src/par_mesh.rs`

```rust
pub struct ParallelMesh<M: MeshTopology> {
    local_mesh: M,                          // Local sub-mesh (local indices)
    comm: Comm,                             // Communicator
    partition: MeshPartition,               // Node/element ownership
    ghost_exchange: GhostExchange,          // Pre-computed halo pattern
    global_n_nodes: usize,                  // Sum of owned nodes (via allreduce)
    global_n_elems: usize,                  // Sum of local elems (via allreduce)
}

impl<M: MeshTopology> ParallelMesh<M> {
    pub fn new(local_mesh: M, comm: Comm, partition: MeshPartition) -> Self {
        // Performs allreduce to compute global statistics
        // Builds GhostExchange from partition
    }

    pub fn local_mesh(&self) -> &M;
    pub fn comm(&self) -> &Comm;
    pub fn partition(&self) -> &MeshPartition;
    pub fn ghost_exchange(&self) -> &GhostExchange;

    pub fn global_n_nodes(&self) -> usize;   // Mesh-wide node count
    pub fn global_n_elems(&self) -> usize;   // Mesh-wide element count
    pub fn n_owned_nodes(&self) -> usize;
    pub fn n_ghost_nodes(&self) -> usize;
    pub fn n_total_nodes(&self) -> usize;

    pub fn forward_exchange(&self, data: &mut [f64]);
    pub fn reverse_exchange(&self, data: &mut [f64]);
    pub fn global_sum_owned(&self, data: &[f64]) -> f64;  // sum over owned nodes only
}
```

---

### `MeshPartition` — Ownership descriptor
**File:** `crates/parallel/src/partition.rs`

```rust
pub struct MeshPartition {
    // Node ownership
    pub n_owned_nodes: usize,
    pub n_ghost_nodes: usize,
    pub global_node_ids: Vec<NodeId>,       // [local_id] = global_id
    pub node_owner: Vec<Rank>,              // [local_id] = owner_rank

    // Element ownership
    pub n_local_elems: usize,
    pub global_elem_ids: Vec<ElemId>,       // [local_id] = global_id

    // Reverse lookup (built lazily)
    node_global_to_local: HashMap<NodeId, u32>,
    elem_global_to_local: HashMap<ElemId, u32>,
}

impl MeshPartition {
    pub fn new_serial(n_nodes: usize, n_elems: usize) -> Self;
    pub fn from_partitioner(owned_nodes, ghost_nodes, local_elems) -> Self;

    pub fn global_node(&self, local_id: u32) -> NodeId;
    pub fn local_node(&self, global_id: NodeId) -> Option<u32>;
    pub fn ghost_nodes(&self) -> impl Iterator<Item = (u32, Rank)>;
    pub fn build_lookup(&mut self);
}
```

---

### `partition_simplex` — Contiguous block partitioner
**File:** `crates/parallel/src/par_simplex.rs`

Distributes a serial `SimplexMesh<D>` across ranks:

```rust
pub fn partition_simplex<const D: usize>(
    mesh: &SimplexMesh<D>,
    comm: &Comm,
) -> ParallelMesh<SimplexMesh<D>> {
    if comm.size() == 1 {
        // Fast path: wrap as-is
        let partition = MeshPartition::new_serial(n_nodes, n_elems);
        return ParallelMesh::new(mesh.clone(), comm.clone(), partition);
    }

    // Multi-rank:
    // 1. Contiguous element partition: rank r owns [r·chunk, (r+1)·chunk)
    // 2. Compute node owners: owner = lowest rank whose element chunk contains it
    // 3. Classify nodes touched by local elements as owned vs ghost
    // 4. Build partition descriptor
    // 5. Create ParallelMesh
}
```

**Strategy:** Each rank receives the **full** serial mesh and extracts its portion. Memory-inefficient for very large meshes but correct and straightforward (streaming partitioner can be added later).

---

## 7. FEATURE FLAGS AND BUILD MATRIX

### Workspace Cargo.toml
**File:** `Cargo.toml` (root)

```toml
[workspace.dependencies]
mpi = { version = "0.8", features = ["user-operations"] }

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

### Parallel Crate Cargo.toml
**File:** `crates/parallel/Cargo.toml`

```toml
[dependencies]
fem-core = { path = "../core" }
fem-mesh = { path = "../mesh" }
fem-linalg = { path = "../linalg" }
log = { workspace = true }
bytemuck = { workspace = true }

[dependencies.mpi]
workspace = true
optional = true

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
rayon = { workspace = true }

[features]
default = []
mpi = ["dep:mpi"]
```

### Target Matrix

| Target | Feature | Backend | Launcher | Status |
|--------|---------|---------|----------|--------|
| `x86_64` / ARM + `mpi` | `mpi` enabled | `NativeMpiBackend` | `MpiLauncher` | ✅ complete |
| `x86_64` / ARM, no `mpi` | (none) | `SerialBackend` | `ThreadLauncher` | ✅ complete |
| `wasm32` | (none) | `WasmWorkerBackend` | `WorkerLauncher` | 🔲 stub |

---

## 8. EXAMPLE: `par_mesh_verify.rs`

**File:** `examples/par_mesh_verify.rs`

Comprehensive verification suite:

```rust
let launcher = MpiLauncher::init().expect("MPI already initialised");
let comm = launcher.world_comm();

let serial_mesh = SimplexMesh::<2>::unit_square_tri(8);
let pmesh: ParallelMesh<SimplexMesh<2>> = partition_simplex(&serial_mesh, &comm);

// 10 invariant checks:
// 1. Global element count = serial total
// 2. Sum of owned nodes = serial total
// 3. Owned-node coordinates match serial
// 4. Element connectivity consistent (single-rank)
// 5. Boundary faces completeness
// 6. Global sum via allreduce
// 7. Ghost exchange is no-op on single rank
// 8. Local mesh passes internal checks
// 9. (per-rank reporting)
// 10. (summary)

let xs: Vec<f64> = (0..pmesh.n_total_nodes())
    .map(|lid| pmesh.node_coords(lid as u32)[0])
    .collect();
let global_sum_x = pmesh.global_sum_owned(&xs);  // Uses allreduce
```

**Run:**
```sh
# Serial (no MPI):
cargo run --example par_mesh_verify

# MPI (4 ranks):
mpirun -n 4 ./target/debug/examples/par_mesh_verify
```

---

## 9. HOW THE ABSTRACTION ENABLES MPI INTEGRATION

### Adding Real MPI Support

The infrastructure is **already designed** for plug-and-play MPI backends:

1. **`NativeMpiBackend` exists** and implements `CommBackend` — just requires `mpi` feature to compile.

2. **All collectives are routed through `CommBackend`**:
   ```
   Comm::allreduce_sum_f64()
     → backend.allreduce_sum_f64()
       → NativeMpiBackend::allreduce_sum_f64()
         → SystemCommunicator::world().all_reduce_into()
   ```

3. **Tag namespace isolation** prevents collisions:
   - `GhostExchange::forward`: tags `0x1000 + rank`
   - `GhostExchange::reverse`: tags `0x2000 + rank`
   - User code: can use different tag ranges (e.g., `0x3000+`)
   - `NativeMpiBackend::alltoallv_bytes`: uses `0x3000 + rank`

4. **Backend-agnostic algorithms**:
   - `GhostExchange` calls only `comm.send_bytes/recv_bytes/alltoallv_bytes`
   - Works identically on `NativeMpiBackend`, `ChannelBackend`, `WasmWorkerBackend`
   - No backend-specific code in higher layers

5. **Feature-gated compilation**:
   ```rust
   #[cfg(feature = "mpi")]
   impl CommBackend for NativeMpiBackend { /* ... */ }
   ```
   Without the feature, builds use `SerialBackend` instead.

### Next Steps to Enable MPI

1. ✅ **`NativeMpiBackend` is complete** — just needs the `mpi` feature enabled
2. ✅ **`Comm` and `CommBackend` are stable abstractions**
3. ✅ **`GhostExchange` is fully backend-agnostic**
4. 🔲 **Sub-communicators** (`Comm::split()`) — planned Phase 10
5. 🔲 **Custom collectives** (min, max, etc.) — can add methods to `CommBackend`
6. 🔲 **Non-blocking sends** (`Isend`, `Irecv`) — planned Phase 10
7. 🔲 **Streaming partitioner** — replace "replicate + extract" with rank-0 reads and MPI distribute

---

## 10. KEY DESIGN PATTERNS

### Pattern 1: Object-Safety via Byte-Level API
- `CommBackend` trait is object-safe (no generic methods)
- All ops move `[u8]` to decouple from numeric types
- Higher layers use `bytemuck::Pod` for safe serialization
- **Benefit**: No monomorphization explosion; single `Box<dyn CommBackend>` works for all T

### Pattern 2: Reference Counting for Communicators
```rust
type BackendBox = Box<dyn CommBackend>;

pub struct Comm {
    inner: std::sync::Arc<BackendBox>,
}

impl Clone for Comm {
    fn clone(&self) -> Self {
        Comm { inner: std::sync::Arc::clone(&self.inner) }
    }
}
```
- **Benefit**: `Comm` is cheap to clone and pass around; Arc handles lifetime automatically

### Pattern 3: Generation-Counted Barriers (ChannelBackend)
```rust
fn rendezvous(&self, state: &Mutex<(usize, usize)>, cv: &Condvar) {
    let mut g = state.lock().unwrap();
    let my_gen = g.1;
    g.0 += 1;
    if g.0 == n {
        g.0 = 0;
        g.1 = my_gen.wrapping_add(1);
        cv.notify_all();
    } else {
        g = cv.wait_while(g, |s| s.1 == my_gen).unwrap();
    }
}
```
- Reusable synchronization primitive for all collectives
- Avoids lost-wakeup races and thundering-herd on `notify_all()`

### Pattern 4: Tag-Space Partitioning
- `GhostExchange::forward`: `0x1000 + rank`
- `GhostExchange::reverse`: `0x2000 + rank`
- `NativeMpiBackend::alltoallv_bytes`: `0x3000 + rank`
- User code: should avoid `0x1000..0x4000`
- **Benefit**: Eliminates tag collisions without central registry

### Pattern 5: Contiguous Block Partitioning
- Rank `r` owns elements `[r·chunk, (r+1)·chunk)`
- Simple, deterministic, no metadata overhead
- Nodes owned by first rank to "see" them
- **Limitation**: Load imbalance if element size varies; can upgrade to METIS later

---

## 11. TESTING INFRASTRUCTURE

### Unit Test Backends
- **`ChannelBackend`**: OS-thread-based in-process fake MPI
- **`SerialBackend`**: Single-rank identity ops
- **`ThreadLauncher`**: Spawn N threads with shared communicator state

### Example Test Pattern
```rust
#[test]
fn test_ghost_exchange() {
    ThreadLauncher::new(WorkerConfig::new(4)).launch(|comm| {
        let rank = comm.rank();
        // Test code runs on all 4 ranks simultaneously
        // Real allreduce / send / recv / alltoallv backed by ChannelBackend
    });
}
```

### Running Examples
```sh
# Serial (no MPI install):
cargo run --example par_mesh_verify

# MPI 4-rank:
mpirun -n 4 cargo run --example par_mesh_verify --features fem-parallel/mpi

# Threading (in-process):
ThreadLauncher::new(WorkerConfig::new(4)).launch(|comm| { /* ... */ });
```

---

## 12. SUMMARY TABLE: Files & Responsibilities

| File | LOC | Responsibility |
|------|-----|-----------------|
| `comm.rs` | 202 | `Comm` public API; typed wrappers around raw byte ops |
| `backend/mod.rs` | 89 | `CommBackend` trait definition |
| `backend/native.rs` | 181 | `SerialBackend` + `NativeMpiBackend` (rsmpi) |
| `backend/channel.rs` | 297 | `ChannelBackend` + `ChannelShared` (thread-based fake MPI) |
| `backend/wasm.rs` | 169 | `WasmWorkerBackend` (stub; TODO SAB impl) |
| `launcher/mod.rs` | 69 | `Launcher` trait; `WorkerConfig` |
| `launcher/native.rs` | 131 | `MpiLauncher` + `ThreadLauncher` |
| `launcher/wasm.rs` | 120 | `WorkerLauncher` (stub; TODO wasm-bindgen) |
| `ghost.rs` | 247 | `GhostExchange` halo exchange pattern |
| `partition.rs` | 232 | `MeshPartition` ownership descriptor |
| `par_simplex.rs` | 197 | Contiguous block partitioner |
| `par_mesh.rs` | 178 | `ParallelMesh<M>` distributed mesh |
| `metis.rs` | ~70 | METIS partitioner bindings (optional) |
| `lib.rs` | 56 | Public re-exports; feature docs |
| **Total** | **2,715** | — |

---

## 13. INTEGRATION WITH SOLVER STACK

The `ParallelMesh` integrates seamlessly with the rest of fem-rs:

```
fem_solver
  └─ assembly (on local sub-mesh via ParallelMesh::local_mesh())
       └─ fem_space, fem_element
  └─ allreduce over owned nodes (ParallelMesh::global_sum_owned)
  └─ ghost exchange (ParallelMesh::forward_exchange / reverse_exchange)
       └─ GhostExchange
            └─ Comm (object-safe CommBackend)
                └─ NativeMpiBackend / SerialBackend / ChannelBackend / WasmWorkerBackend
```

No changes needed to solver code when switching backends—just recompile with a different feature flag.

