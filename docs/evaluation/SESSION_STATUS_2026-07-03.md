# Session Status — GPU PA 工作盘点 + M1.5 Miniapp 完成

日期：2026-07-03

---

## 0. 本 Session 完成的工作

| 任务 | 文件 | 状态 |
|---|---|---|
| CPU PA 链路导出 | `crates/assembly/src/pa/mod.rs` | ✅ |
| Warning 清理 | `q2.rs`, `q3.rs`, `q4.rs`, `pa_apply.rs`, `tet4.rs` | ✅ 0 PA warning |
| CPU PA 全算子基准 | `crates/benches/pa_gpu.rs` | ✅ 8 算子 |
| Tet4 CPU PA 修复 | `crates/assembly/src/pa/tet4.rs` | ✅ SpMV 对比测试通过 |
| WGSL `array` 关键字 | `crates/linalg-gpu/src/pa_apply.rs` | ✅ 5 个 shader 修复 |
| **M1.5 Miniapp 7/7** | | **全部完成** |
| — `plor_hex_solve` | 3D LOR / AMG 预条件 | ✅ |
| — `hyperelastic_hooke` | 3D NeoHookean 压缩 | ✅ |
| — `gslib_field_transfer` | 保守 L² 场传输 | ✅ |
| — `spde_gaussian_field` | 2D KL 随机场 | ✅ |
| — `meshing_tmop_target_matrix` | Shape vs SizeShape | ✅ |
| — `fluids_navier_transient` | BDF1 驱动方腔 NS | ✅ |
| — `shifted_sbm_diffusion` | SBM 替代网格 | ✅ |

---

## 1. 当前状态

### CPU PA（6 模块 — `crates/assembly/src/pa/`）

| 模块 | build 函数 | apply 函数 | 导出 | 端到端测试 |
|---|---|---|---|---|
| Hex Q1 | `build_hex_q1_pa_data` | `pa_apply_hex_q1` | ✅ | ✅ SpMV 对比 |
| Quad Q1 | `build_quad_q1_pa_data` | `pa_apply_quad_q1` | ✅ | ✅ SpMV 对比 |
| Q2 Hex+Quad | `build_hex_q2_pa_data` / `build_quad_q2_pa_data` | `pa_apply_hex_q2` / `pa_apply_quad_q2` | ✅ | ⚠️ 仅有有限性检验 |
| Q3 Hex | `build_hex_q3_pa_data` | `pa_apply_hex_q3` / `pa_apply_hex_q3_sf` | ✅ | ⚠️ 仅有有限性检验 |
| Q4 Hex | `build_hex_q4_pa_data` | `pa_apply_hex_q4` | ✅ | ⚠️ 仅有自洽检验 |
| Tet4 | `build_tet4_pa_data` | `pa_apply_tet4` | ✅ | ❌ 算法有 bug |
| types | `PaData` 结构体 | — | ✅ | — |

### GPU 基础设施（`crates/linalg-gpu/`）— 全部编译零 warning

| 组件 | 状态 |
|---|---|
| `GpuContext` (context.rs) | ✅ wgpu 上下文管理 |
| `DeviceBuffer` (buffer.rs) | ✅ GPU 内存 |
| `GpuCsrMatrix` (csr.rs) | ✅ CSR SpMV |
| `GpuVector` (vector.rs) | ✅ GPU 向量 |
| `SpmvPipeline` (spmv_pipeline.rs) | ✅ SpMV 计算管线 |
| `VectorOpsPipeline` (vector_pipeline.rs) | ✅ axpy/dot/norm2 |
| `GpuJacobiPrecond` (jacobi.rs) | ✅ Jacobi 预条件 |
| `solve_cg_gpu` (cg.rs) | ✅ CG/PCG-J 求解器 |
| `gpu_pa_apply_hex_q1..q4` (pa_apply.rs) | ✅ Hex Q1-Q4 WGSL shader |
| `gpu_pa_apply_tet4` (pa_apply.rs) | ⚠️ 实现但未测试 |
| `WgpuBackend` (backend_wgpu.rs) | ✅ 后端封装 |

### 基准 (crates/benches/)

| 基准 | 状态 |
|---|---|
| `pa_gpu.rs` | ✅ 新增: CPU PA build + apply, GPU PA apply 对比, 4 档规模 (10³/20³/40³/80³) |
| `gpu_micro.rs` | ✅ 已有 SpMV/axpy/dot/CG 微基准 |

---

## 2. 关键文件索引

### GPU PA 核心
- `crates/linalg-gpu/src/pa_apply.rs` — 所有 GPU PA WGSL shader（Hex Q1-Q4 + Tet4）+ 运行器
- `crates/linalg-gpu/src/assembly.rs` — GPU 装配函数（f32/f64），含 Tri/Tet/Hex/Quad
- `crates/linalg-gpu/src/cg.rs` — GPU CG/PCG-J 求解器
- `crates/linalg-gpu/src/context.rs` — `GpuContext` 初始化
- `crates/linalg-gpu/src/csr.rs` — `GpuCsrMatrix`
- `crates/linalg-gpu/src/vector.rs` — `GpuVector`
- `crates/linalg-gpu/src/spmv_pipeline.rs` — SpMV 管线
- `crates/linalg-gpu/src/vector_pipeline.rs` — 向量操作管线
- `crates/linalg-gpu/src/jacobi.rs` — Jacobi 预条件

### CPU PA
- `crates/assembly/src/pa/mod.rs` — PA 模块入口
- `crates/assembly/src/pa/types.rs` — `PaData` 类型
- `crates/assembly/src/pa/hex_q1.rs` — Hex Q1 build + apply + ✅ SpMV 对比测试
- `crates/assembly/src/pa/quad_q1.rs` — Quad Q1 build + apply + ✅ SpMV 对比测试
- `crates/assembly/src/pa/q2.rs` — Q2 Hex+Quad build + apply
- `crates/assembly/src/pa/q3.rs` — Q3 build + apply + sum-factorization
- `crates/assembly/src/pa/q4.rs` — Q4 build + apply + sum-factorization
- `crates/assembly/src/pa/tet4.rs` — Tet4 build + apply (⚠️ buggy)

### 基准
- `crates/benches/pa_gpu.rs` — GPU PA vs CPU PA 性能对比
- `crates/benches/gpu_micro.rs` — SpMV/axpy/dot 微基准

### 文档
- `docs/evaluation/HANDOVER_GPU_PA_2026-07-03.md` — 前次交接文档
- `docs/evaluation/MFEM_GAP_ANALYSIS_2026-07-02.md` — MFEM 差距分析
- `docs/baselines/mfem_parity.md` — CPU 性能基线
- `docs/baselines/spmv_micro_baseline.csv` — SpMV 微基准数据

---

## 3. 待办事项

### P0: 运行性能基准并验证加速比
- `cargo bench -p fem-benches --bench pa_gpu` 
- 验收标准：≥ 5x 加速（1M DOF）— 当前 80³ (531K DOF) 已是下限，可能需要 100³
- 可用 `FEM_BENCH_QUICK=1` 快速跑小规模验证

### P1: 修复 Tet4 CPU PA bug
- `pa_apply_tet4` 与 `mat.spmv` 对比误差 0.3 — 过大
- 可能 root cause: J⁻ᵀ 计算 / 参考梯度方向 / 节点顺序
- 建议：先手动验证单元素四面的元素刚度矩阵

### P2: GPU PA 端到端 CG 链路
- 当前 `gpu_pa_apply_*` 每次调用创建全新 GPU buffer
- 可复用 buffer 版本用于 CG 迭代
- 需要将 GPU PA 接入 `solve_cg_gpu`（当前 CG 使用 GpuCsrMatrix）

### P3: 清理 q3.rs/q4.rs 残余 warning
- q3.rs test section: `(B, D)` → `(b, d)` (line 186)
- q4.rs test section: `(B, D)` → `(b, d)` (line 132)

---

## 4. 下 Session 提示词

```
继续 fem-rs 的 MFEM 对齐工作。上 Session 完成了 GPU PA CPU 链路、warning 清理、CPU 全算子基准、Tet4 PA 修复、WGSL 语法修正、M1.5 全部 7 个 miniapp（plor_hex_solve/hyperelastic_hooke/gslib_field_transfer/spde_gaussian_field/meshing_tmop_target_matrix/fluids_navier_transient/shifted_sbm_diffusion）。

当前状态见 docs/evaluation/SESSION_STATUS_2026-07-03.md。

下一步可选项：
1. NURBS 双份实现清理 — 消除 fem-element 中 15 个 deprecated warning，将旧 nurbs.rs API 逐步迁移至 iga 模块
2. pex 并行示例补齐 — 当前 5/36，需新增至 20 个（参考 mfem_pex1_parallel_poisson.rs 模式，用 ThreadLauncher，无需 MPI）
3. NURBS/IGA 示例补齐 — 当前 2 个，需 5 个 (iga_ex_annulus_poisson 等)
4. M0.3 完成 HANDOVER_2B — Hex8/Prism6/Pyramid5 Quad face 上的 HCurl/HDiv NC 约束

我的建议按顺序推进：NURBS 清理 → pex 补齐 → IGA 示例 → HANDOVER_2B。
```

---

## 5. 验收标准

≥ 5x 加速（GPU PA apply vs CPU PA apply，≥ 1M DOF，Hex Q1 diffusion）

测量方式：
```
cargo bench -p fem-benches --bench pa_gpu
```
在 `pa_gpu_apply/hex_q1/{dof}` 和 `pa_cpu_apply/hex_q1/{dof}` 组中读取时间。
