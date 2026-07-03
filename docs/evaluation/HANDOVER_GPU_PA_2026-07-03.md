# 交接文档 — GPU Partial Assembly (Tensor-Product PA)

- 日期：2026-07-03
- 编写者：MiMoCode agent（基于工作树代码盘点）
- 前置文档：`docs/evaluation/MFEM_GAP_ANALYSIS_2026-07-02.md` §4.4 M8 里程碑
- 相关代码：`crates/linalg-gpu/`、`crates/assembly/src/pa/`

---

## 当前状态

### CPU PA 已有（`crates/assembly/src/pa/`）

| 文件 | 内容 | 状态 |
|---|---|---|
| `types.rs` | `PaData` 结构体：逐元素逐QP的 `[J⁻ᵀ, detJ, κ]` 数据 | ✅ 完成 |
| `mod.rs` | 导出 `pa_apply_*` 函数 | ✅ |
| `hex_q1.rs` | Hex Q1 的 PA build + apply | ✅ 完成 |
| `quad_q1.rs` | Quad Q1 的 PA build + apply | ✅ 完成 |
| `q2.rs` | Hex Q2 / Quad Q2 的 PA build + apply（含 sum-factorization） | ⚠️ 有 dead_code warnings |
| `q3.rs` | Hex Q3 PA（含 `pa_apply_hex_q3_sf` sum-factorization 版本） | ⚠️ 有 dead_code warnings |
| `q4.rs` | Hex Q4 PA | ⚠️ 有 dead_code + snake_case warnings |

### CPU PA 已知问题

```
warning: function `build_hex_q1_pa_data` is never used          → hex_q1.rs
warning: function `build_quad_q1_pa_data` is never used          → quad_q1.rs
warning: function `build_hex_q2_pa_data` is never used          → q2.rs
warning: function `build_quad_q2_pa_data` is never used          → q2.rs
warning: function `build_hex_q3_pa_data` is never used          → q3.rs
warning: function `hex_q4_ixyz` is never used                    → q4.rs
warning: function `build_hex_q4_pa_data` is never used          → q4.rs
warning: variables `B`, `D`, `G`, `Ms`, `X`, `Kl`, `Cw`, etc.  → 多处 snake_case
```

所有 `build_*_pa_data` 函数都定义了但 **从未被调用** — 没有端到端的 PA 装配-应用链路。

### GPU 已有（`crates/linalg-gpu/`）

| 文件 | 内容 | 状态 |
|---|---|---|
| `context.rs` | `GpuContext` wgpu 上下文管理 | ✅ |
| `backend.rs` / `backend_wgpu.rs` | wgpu 后端抽象 | ✅ |
| `backend_cuda.rs` | CUDA 后端骨架（feature-gated） | ⚠️ 占位 |
| `buffer.rs` | `DeviceBuffer<T>` GPU 内存管理 | ✅ |
| `csr.rs` | `GpuCsrMatrix` GPU CSR SpMV | ✅ |
| `vector.rs` | `GpuVector` GPU 向量 | ✅ |
| `cg.rs` | `solve_cg_gpu` / `solve_pcg_jacobi_gpu` | ✅ 已验证 |
| `spmv_pipeline.rs` | 异步 SpMV multi-stream pipeline | ✅ |
| `jacobi.rs` | `GpuJacobiPrecond` GPU Jacobi 预条件 | ✅ |
| `amg_precond.rs` | GPU AMG 预条件（feature-gated） | ⚠️ |
| `pa_apply.rs` | **GPU PA 核心**：Hex Q1 WGSL shader + host scatter | ⚠️ 见下 |
| `direct.rs` | GPU 直接求解器 | ⚠️ 基础 |
| `assembly.rs` | GPU 装配函数 | ✅ |
| `vector_pipeline.rs` | 向量操作 pipeline | ✅ |

### GPU PA (`pa_apply.rs`) 当前状态

- 实现了 Hex Q1 的 WGSL compute shader
- Shader 在 8 个 QP（2×2×2 Gauss）上循环，计算梯度变换，累加元素残差
- 有 `build_pa_data_gpu` 将 CPU `PaData` 拷贝到 GPU buffer
- 有 `apply_pa_gpu` 启动 compute shader + host-side scatter
- **已知 warning**：
  - `nqp` 参数未使用（line 253）
  - `dev.poll()` 的 `Result` 未处理（line 284）

---

## 待完成的工作

### P2.1 连通 CPU PA 装配到 PA Apply（P0）

**当前缺口**：CPU 端 `build_*_pa_data()` 存在但未导出、未测试。建议：

1. 在 `crates/assembly/src/pa/mod.rs` 中添加 `pub use` 导出所有 `build_*_pa_data` 函数
2. 编写一个端到端测试：`build_hex_q1_pa_data` + `pa_apply_hex_q1` 结果与 `CsrMatrix::spmv` 对比
3. 对 Q2/Q3/Q4 同理

```rust
// 期望的端到端模式
let pa_data = build_hex_q1_pa_data(&mesh, &kappa);
let y_pa = pa_apply_hex_q1(&pa_data, &x);
let y_spmv = csr_matrix.spmv(&x);
assert!((y_pa - y_spmv).norm() < 1e-12);
```

### P2.2 GPU PA：Hex Q2/Q3/Q4 WGSL Shader

**当前缺口**：只有 Hex Q1 有 GPU shader。

1. 为 Hex Q2（27 节点，3×3×3 QP）编写 WGSL compute shader
2. 为 Hex Q3（64 节点，4×4×4 QP）编写 WGSL shader（含 sum-factorization 优化）
3. 为 Hex Q4（125 节点，5×5×5 QP）编写 WGSL shader
4. 注册到 `GpuContext` 的 ShaderModule 缓存

### P2.3 GPU PA：Tet 元素支持

**当前缺口**：目前只支持 Hex/Quad 张量积网格。Tet 需要：

1. Tet 的参考单元形状函数在 GPU 上没有张量积结构 → 每个 QP 需要完整的基函数值
2. 参考 `element/src/lagrange/tet.rs` 中的 TetPk 实现在 GPU 上实现等价计算
3. Simplex 元素的 PA 通常用"预计算基函数值于 QP → GPU 纹理读取"的方式

### P2.4 清理 warning + snake_case

**位置**：
- `crates/assembly/src/pa/q3.rs`：变量 B/D → b/d
- `crates/assembly/src/pa/q4.rs`：变量 B/D → b/d
- `crates/linalg-gpu/src/pa_apply.rs`：`nqp` unused + `dev.poll` Result 未处理

### P2.5 性能基准

参考 `docs/baselines/mfem_parity.md`，增加 GPU PA vs CPU PA vs CPU SpMV 的对比：

| 算例 | DOF | CPU SpMV | CPU PA | GPU PA | 加速比 |
|---|---|---|---|---|---|
| Hex Q1 Poisson | 1M | ? | ? | ? | ? |
| Hex Q2 Poisson | 1M | ? | ? | ? | ? |

验收标准（来自评估文档 M8）：
> `cargo bench -p fem-benches --bench pa_gpu` 与 CPU PA 对比 ≥ 5x 加速（对 ≥ 1M DOF）

---

## 关键文件索引

| 路径 | 作用 |
|---|---|
| `crates/assembly/src/pa/hex_q1.rs:30` | `build_hex_q1_pa_data` — CPU PA 数据构建 |
| `crates/assembly/src/pa/hex_q1.rs` (末尾) | `pa_apply_hex_q1` — CPU PA apply |
| `crates/assembly/src/pa/q2.rs:44` | `build_hex_q2_pa_data` — 存在但未导出 |
| `crates/assembly/src/pa/q3.rs:44` | `build_hex_q3_pa_data` + sum-factorization |
| `crates/assembly/src/pa/q4.rs:57` | `build_hex_q4_pa_data` |
| `crates/linalg-gpu/src/pa_apply.rs` | GPU PA: WGSL shader + `apply_pa_gpu` |
| `crates/linalg-gpu/src/context.rs` | `GpuContext` wgpu 初始化 |
| `crates/linalg-gpu/src/cg.rs` | `solve_cg_gpu` — 已验证的 GPU CG |
| `docs/baselines/mfem_parity.md` | 已有的 CPU 性能基线 |

---

## 下个 Session 的提示词

复制以下内容到新 session 开始：

```
继续 GPU PA 工作。参考 `docs/evaluation/HANDOVER_GPU_PA_2026-07-03.md`。

第一步：连通 CPU PA 链路
1. 在 `crates/assembly/src/pa/mod.rs` 中导出所有 `build_*_pa_data` 函数
2. 写端到端测试：`build_hex_q1_pa_data` + `pa_apply_hex_q1` 结果与 SpMV 对比
3. 对 Q2/Q3/Q4 重复

第二步：修 warning
- `pa/q3.rs` 和 `pa/q4.rs` 中 snake_case 变量
- `linalg-gpu/src/pa_apply.rs` 中 `nqp` unused + `dev.poll` Result

第三步：验证 GPU PA
- 运行现有的 `cargo test -p fem-linalg-gpu` 确认通过
- 扩展 GPU PA 到 Hex Q2 shader

第四步：性能基准
- `cargo bench -p fem-benches --bench pa_gpu` 与 CPU PA 对比
- 预期 ≥ 5x 加速（≥ 1M DOF）
```

---

## 现有测试状态

```bash
cargo test -p fem-linalg-gpu   # GPU crate tests（需要 wgpu 设备）
cargo test -p fem-assembly     # 含 CPU PA 的 assembly crate
cargo check --examples         # 全示例编译检查
```

所有 crate 编译通过，`cargo check --examples` 无错误。GPU 测试需要 wgpu 兼容设备。

---

## 关键依赖

- `wgpu = "27"`（已 workspace 管理）
- `fem-ceed` 有 `reed` crate 的 workspace-pinned 依赖（feature gate `reed`），目前 CPU-only
- 无 CUDA/HIP/OCCA 后端（只有 `backend_cuda.rs` 骨架，feature gate `cuda`）
