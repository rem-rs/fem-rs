# fem-rs Parallel Architecture — Visual Summary

## Trait Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  CommBackend (object-safe trait)                            │
│  • rank() / size()                                          │
│  • barrier()                                                │
│  • allreduce_sum_f64/i64, broadcast_bytes                  │
│  • send_bytes, recv_bytes                                   │
│  • alltoallv_bytes                                          │
└────────────────────────────────────────────────────────────┬┘
                          ▲
        ┌─────────────────┼─────────────────┬─────────────┐
        │                 │                 │             │
┌───────────────────┬──────────────────┬───────────────┬──────────────┐
│ SerialBackend     │ NativeMpiBackend │ ChannelBackend│ WasmWorkerBE │
│ (identity ops)    │ (rsmpi wrapper)  │ (Mutex+Cond) │ (SAB stub)   │
│ always enabled    │ feat: mpi        │ for testing  │ wasm32 only  │
└────────┬──────────┴────────┬─────────┴────────┬──────┴──────┬───────┘
         │                   │                  │            │
         └───────────────────┼──────────────────┼────────────┘
                             │                  │
                             ▼                  │
                   Arc<Box<dyn CommBackend>>    │
                             │                  │
                             └──────────┬───────┘
                                        │
                                        ▼
                            ┌─────────────────────┐
                            │  Comm (Arc-wrapped) │
                            │ • typed collectives │
                            │ • point-to-point    │
                            │ • cloneable         │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ GhostExchange       │
                            │ • forward()         │
                            │ • reverse()         │
                            │ • fully agnostic    │
                            └─────────────────────┘
```

## Data Flow: Partition → ParallelMesh → Solver

```
┌──────────────────────────────────────────────────────────────────────┐
│  Serial Mesh (all ranks receive full copy)                           │
└─────────���──────────────────┬─────────────────────────────────────────┘
                             │
                    partition_simplex()
                             │
        ┌────────────────────┴────────────────────┐
        │ For each rank in parallel:              │
        │  • Extract elements [r·chunk, (r+1)·ch) │
        │  • Classify nodes: owned vs ghost       │
        │  • Build MeshPartition descriptor       │
        └────────────────────┬────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MeshPartition                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ n_owned_nodes, n_ghost_nodes                                │   │
│  │ global_node_ids[]: Vec<u32>  → [local_id] = global_id      │   │
│  │ node_owner[]: Vec<Rank>       → [local_id] = owner_rank    │   │
│  │ global_elem_ids[]: Vec<u32>   → [local_id] = global_id     │   │
│  │ node_global_to_local: HashMap (reverse lookup)             │   │
│  │ elem_global_to_local: HashMap (reverse lookup)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
         ParallelMesh::new()
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
   (allreduce)         GhostExchange::from_partition()
   for global stats           │
        │            ┌────────┴────────────────┐
        │            │ Distributed request     │
        │            │ via alltoallv_bytes     │
        │            │ (backend-agnostic)      │
        │            └────────┬────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ParallelMesh<M>                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ local_mesh: M                                               │   │
│  │ comm: Comm (Arc<Box<dyn CommBackend>>)                      │   │
│  │ partition: MeshPartition                                    │   │
│  │ ghost_exchange: GhostExchange                               │   │
│  │ global_n_nodes: usize (allreduced)                          │   │
│  │ global_n_elems: usize (allreduced)                          │   │
│  │                                                              │   │
│  │ Methods:                                                    │   │
│  │ • forward_exchange(&mut [f64]) → owner→ghost halo update   │   │
│  │ • reverse_exchange(&mut [f64]) → ghost→owner accumulation  │   │
│  │ • global_sum_owned(&[f64]) → f64 with allreduce            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
      ┌────────────┴────────────┐
      │                         │
      ▼                         ▼
┌─────────────┐         ┌──────────────────────────────┐
│  Assembly   │         │  Iterative Solve             │
│  (on local  │         │  ┌──────────────────────────┐│
│   submesh)  │         │  │ 1. assemble()            ││
│             │         │  │ 2. forward_exchange()    ││
└─────────────┘         │  │ 3. LinSolver::matvec()   ││
                        │  │ 4. reverse_exchange()    ││
                        │  │ 5. global_sum_owned()    ││
                        │  │ 6. barrier() sync        ││
                        │  └──────────────────────────┘│
                        └──────────────────────────────┘
```

## Message Tag Space Partitioning

```
Tag range 0x0000 — 0x0FFF: [reserved for user]

Tag range 0x1000 — 0x1FFF: GhostExchange::forward()
  • sender encodes rank in tag
  • TAG_FWD (0x1000) + comm.rank()
  • enables independent tag derivation on both sides

Tag range 0x2000 — 0x2FFF: GhostExchange::reverse()
  • same structure: TAG_REV (0x2000) + comm.rank()

Tag range 0x3000 — 0x3FFF: NativeMpiBackend::alltoallv_bytes()
  • used internally for sparse all-to-all exchange
  • TAG_A2AV (0x3000) + sender_rank

Tag range 0x4000 — 0xFFFF: [reserved for user / future]
```

## Launcher Pipeline

```
┌────────────────────────────────────────────────────────┐
│  main() code (same for all backends)                   │
│  let launcher = ???Launcher::init();                   │
│  let comm = launcher.world_comm();                     │
│  partition_simplex(..., &comm);  // backend-agnostic  │
│  solver.run(&pmesh, &comm);      // no cfg guards      │
└──────────┬─────────────────────────────────┬───────────┘
           │                                 │
           │ feature gate at compile time   │ runtime selection
           │                                 │
    ┌──────▼─────────┐            ┌─────────▼────────────┐
    │ Build config   │            │ ThreadLauncher::    │
    │ selects        │            │ launch(closure)     │
    │ which Launcher │            │                     │
    │ compiles in    │            │ Spawn N threads → │
    └──────┬─────────┘            │ each gets          │
           │                       │ ChannelBackend     │
    ┌──────┴──────────┐           │                    │
    │                 │           └────────┬───────────┘
    │                 │                    │
 ┌──▼────────┐   ┌────▼────────────┐  ┌───▼────────┐
 │ MpiLauncher   │ (no feature)     │  │ in-process │
 │              │ SerialBackend    │  │ testing    │
 │ (with `mpi`)  │                  │  │            │
 │              └────┬─────────────┘  └────────────┘
 │              1 rank only
 │
 ├─ feature "mpi" enabled
 ├─ calls mpi::initialize()
 ├─ NativeMpiBackend wraps world()
 ├─ returns Comm with real MPI
 ├─ mpirun -n 4 ./solver
 └─ multi-rank HPC execution

```

## Feature-Gated Compilation

```
Cargo.toml feature = "mpi"
        │
        ├─ [dependencies.mpi] = optional
        │
        ├─ #[cfg(feature = "mpi")]
        │  impl CommBackend for NativeMpiBackend { ... }
        │
        └─ #[cfg(feature = "mpi")]
           impl Launcher for MpiLauncher { ... real init ... }

Without "mpi" feature:
        └─ SerialBackend always compiles
        └─ MpiLauncher::init() returns serial launcher
        └─ No MPI install needed
        └─ Identical public API
        └─ Tests pass unchanged
```

## Backend Implementation Comparison

| Operation | SerialBackend | ChannelBackend | NativeMpiBackend |
|-----------|---------------|----------------|------------------|
| `barrier()` | no-op | Condvar rendezvous | `MPI_Barrier` |
| `allreduce_sum_f64` | identity (return input) | accumulate + cache result | `MPI_Allreduce` |
| `send_bytes` | panic | push to queue | `MPI_Send` |
| `recv_bytes` | panic | block on Condvar | `MPI_Recv` |
| `alltoallv_bytes` | empty result | 4-phase protocol | AllToAll counts + sparse sends |

## ChannelBackend: 4-Phase AllToAllv Protocol

```
Phase 1: Send deposit
  Each rank writes its sends into a2av_slots[my_rank][dest]
  (shared grid indexed [src][dst])

     rendezvous()  ← generation-counted barrier

Phase 2: Send complete
  All ranks have now deposited their data

Phase 3: Receive read
  Each rank reads from a2av_slots[src][my_rank]
  across all src ranks
  Collects into result Vec<(Rank, Vec<u8>)>

     rendezvous()  ← generation-counted barrier

Phase 4: Receive complete
  All ranks have read; slots can be reused
  Implicit in next a2av call

Key: Mutex<(usize, usize)> = (arrived_count, generation)
     Condvar wakes all when arrived_count == n
     Returns when generation advances
```

## Tag derivation pattern (both directions work independently)

```
GhostExchange::forward() — owner → ghost
┌─────────────────────────────────────┐
│ For each neighbor:                  │
│                                     │
│ send_tag = TAG_FWD + my_rank        │
│ recv_tag = TAG_FWD + neighbor_rank  │
│                                     │
│ I send with send_tag                │
│ I receive with recv_tag             │
│                                     │
│ (neighbor will send with their tag) │
│ (neighbor will receive with my tag) │
└─────────────────────────────────────┘

Both sides independently derive the same tag:
sender uses TAG_FWD + its own rank
receiver uses TAG_FWD + sender's rank = TAG_FWD + its_recv_source
→ tags match even though neither pre-coordinates!
```

## Memory layout: ParallelMesh node indexing

```
Local node array (contiguous):
┌─────────────────────┬──────────────────────┐
│  Owned nodes        │  Ghost nodes         │
│  [0 .. n_owned)     │  [n_owned .. n_total)│
└─────────────────────┴──────────────────────┘

Owned nodes:
  • rank is authoritative for values
  • send in forward halo exchange
  • receive contributions in reverse halo exchange

Ghost nodes:
  • read-only copies from neighbor
  • receive in forward halo exchange
  • send contributions back in reverse halo exchange
  • zeroed after reverse exchange
```

## Trait Bounds Summary

```
CommBackend: Send + Sync (required)
  └─ Allows Box<dyn CommBackend> to be Send + Sync
  └─ Necessary for ParallelMesh: MeshTopology to work
     (MeshTopology has Send + Sync supertraits)

Comm: Arc<Box<dyn CommBackend>>
  └─ Automatically Send + Sync if CommBackend is
  └─ Cloneable: cheap to pass around
  └─ Can be stored in ParallelMesh<M>

ParallelMesh<M>: requires M: MeshTopology
  └─ Forwards all mesh queries to local_mesh
  └─ Counts (n_nodes, n_elements) are LOCAL
  └─ Use global_n_nodes() / global_n_elems() for mesh-wide

GhostExchange: backend-agnostic
  └─ Uses only Comm methods
  └─ Works identically on all CommBackend implementations
  └─ No #[cfg] conditionals
```

