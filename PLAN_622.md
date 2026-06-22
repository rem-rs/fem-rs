# PLAN 622 — fem-rs 功能差距补全计划

> 2026-06-22 · 基于源码实际状态，不依赖过时文档

## 背景

截至 2026-06-22，fem-rs 在标准连续 FEM 管线上已高度完整：
Poisson → 弹性 → Maxwell → Darcy → Stokes → Navier-Stokes → 多物理场。
但通过代码审查发现以下实质性差距需要补全。

## 差距总览

| 编号 | 功能 | 状态 | 优先级 |
|------|------|------|--------|
| A1 | Quad/Hex H(div) 元素 | 缺失 | P0 |
| A2 | ProjectCoefficient (L2 投影) | 缺失 | P0 |
| A3 | Face neighbor 查询 | 缺失 | P1 |
| B1 | DPG 方法框架 | 缺失 | P1 |
| C1 | GPU 加速装配 | 缺失 | P1 |
| C2 | 混合精度 (f32) | 缺失 | P2 |
| C3 | p-multigrid / FMG | 缺失 | P2 |
| D1 | TMOP 完整 target-matrix | 部分 | P1 |
| D2 | NURBS multi-patch / trimming | 部分 | P2 |
| D3 | ROM / POD 降阶模型 | 缺失 | P2 |
| E1 | 接触力学 | 缺失 | P2 |
| E2 | SDC 时间积分 | 缺失 | P3 |
| E3 | L1 / W1 误差范数 | 缺失 | P3 |

---

## Phase A: 基础元素与空间补全 (P0)

### A1 — Quad/Hex Raviart-Thomas 元素

**现状**: `fem-element::raviart_thomas` 仅有 TriRT0/TriRT1/TriRT2/TetRT0/TetRT1，
Quad 和 Hex 上完全不存在 H(div) 元素。

**影响**: 无法在 Quad 网格上用混合方法求解 Darcy/Stokes，这是混合 FEM 在
四边形网格上的基础需求。

**工作**:
1. `crates/element/src/raviart_thomas/quad_rt0.rs` — QuadRT0 (4 DOFs, 4 faces)
2. `crates/element/src/raviart_thomas/quad_rt1.rs` — QuadRT1 (12 DOFs)
3. `crates/element/src/raviart_thomas/hex_rt0.rs` — HexRT0 (6 DOFs, 6 faces)
4. 注册 DOF 符号约定（normal-flux moment per face）并与 Tri/Tet 保持一致
5. 更新 `VectorAssembler` 和 `HDivSpace` 以支持 Quad/Hex RT 元素
6. 新增测试：Quad/Hex RT0 精度、DOF 方向一致性
7. 更新 `mfem_ex4_darcy` 支持 Quad mesh 路径

**验收**: `cargo test -p fem-element --lib raviart_thomas` 通过，
Quad4 网格上 Darcy 问题收敛

### A2 — ProjectCoefficient (L2 投影到 GridFunction)

**现状**: `FESpace::interpolate(f)` 是节点插值（f 在 DOF 坐标处求值），
而 MFEM 的 `GridFunction::ProjectCoefficient` 是求解质量矩阵线性系统
`M x = b` 做 L2 投影，两者数学上不等价。

**影响**: 无法将连续系数函数精确投影到有限元空间（混合方法、后处理等场景必需）。

**工作**:
1. `crates/assembly/src/grid_function.rs` 新增 `GridFunction::project_coefficient()`
2. 实现 `assemble_l2_projection_rhs(space, coeff)` — ∫ φ_i f dx
3. 复用已有 `MassIntegrator` 组装质量矩阵
4. 用 PCG 求解 `M x = b`
5. 新增测试：常数/线性函数的 L2 投影精度验证
6. 与 `interpolate` 做对比测试，验证 L2 投影的优越性

**验收**: 常数投影误差为 0，线性投影误差随 h 以 O(h²) 收敛

### A3 — Face Neighbor 查询

**现状**: `MeshTopology` 缺少面邻居查询接口，仅有 `face_nodes()` 和 `face_tag()`，
无法按面查询相邻元素。

**工作**:
1. `crates/mesh/src/topology.rs` 新增 `face_neighbors(face_id) -> (ElemId, Option<ElemId>)`
2. 分内部面（返回两个相邻元素）和边界面（返回一个元素）
3. 预计算 `face_elements` 映射（在网格构造时建立）

**验收**: `cargo test -p fem-mesh --lib topology` 通过

---

## Phase B: 方法论补全 (P1)

### B1 — DPG 方法框架

**现状**: 代码中零存在。DPG 是近年来重要的 FEM 方法论分支，用于构建
天生稳定的离散格式（特别适合对流占优问题）。

**工作**:
1. `crates/assembly/src/dpg.rs` — DPG 基础框架
2. Trial space / Test space 分离机制（DPG 中 test space 为间断空间）
3. Enriched test space 构造 (p+Δp)
4. Optimal test functions 计算（需局部 Riesz 表示）
5. DPG 装配流程：element-wise 求解局部问题 → 组装 global stiffness
6. 新增 `mfem_ex23_dpg` 示例（对流-扩散 DPG）

**验收**: 1D/2D 对流扩散 DPG 解稳定收敛，无数值震荡

---

## Phase C: 工程基础设施 (P1-P2)

### C1 — GPU 加速装配

**现状**: GPU 仅用于 wgpu SpMV（`fem-linalg-gpu`），元素装配全部走 CPU。

**工作**:
1. 将装配流程重组为 GPU-friendly 的数据布局（SoA, coalesced access）
2. 实现 GPU 上的 Jacobian/积分求值（根据元素类型）
3. 实现 GPU 上的 COO 累加
4. 集成到 `Assembler`，提供 `assemble_bilinear_gpu()` 入口
5. 新增 GPU 装配 benchmark

### C2 — 混合精度支持

**现状**: 所有 FEM 路径硬编码 `f64`，`Scalar` trait 定义了 `f32` 但从未使用。

**工作**:
1. 将 `Assembler`、`H1Space`、solver 等泛型化支持 `Scalar`
2. 为 f32 路径添加数值稳定性保障
3. 新增 f32 vs f64 对比 benchmark
4. Python bindings 支持 f32/f64 选择

### C3 — p-Multigrid / FMG

**现状**: 仅有 h-multigrid 和 LOR 预条件器。无 p-multigrid 和 FMG。

**工作**:
1. 利用已有 `p_refine` 模块构建层级间 prolongation/restriction
2. 实现 V-cycle p-multigrid (层间: high-p → low-p → high-p)
3. 实现 FMG (nested iteration, coarse → fine)
4. 新增 `mfem_ex26` 的 p-multigrid 变体

---

## Phase D: 高级功能 (P1-P2)

### D1 — TMOP 完整 target-matrix 核

**现状**: `mfem_tmop_mesh_quality.rs` 明确标注 "full target-matrix TMOP kernels are still pending"

**工作**:
1. 实现 TMOP target-matrix 构造（ideal shape/size/alignment）
2. 实现 shape/size/barrier 等标准 TMOP 度量
3. 牛顿法优化 interior nodes
4. 新增 2D/3D 完整 TMOP 示例

### D2 — NURBS multi-patch / trimming

**现状**: 单 patch 1D/2D IGA 已实现

**工作**:
1. Multi-patch 界面 C0 连续性约束
2. Trimming curve 集成
3. 全局 IGA 装配支持 multi-patch
4. 新增 multi-patch 示例

### D3 — ROM / POD 降阶模型

**现状**: 零存在

**工作**:
1. POD (Proper Orthogonal Decomposition) 快照方法
2. Galerkin ROM 投影
3. 离线/在线分离
4. 新增 ROM 示例

---

## Phase E: 专项功能 (P2-P3)

### E1 — 接触力学
- Signorini 边界条件
- 罚函数法 / Lagrange 乘子法
- Hertz 接触验证

### E2 — SDC 时间积分
- Spectral Deferred Correction 框架
- Gauss-Lobatto 节点配置

### E3 — L1 / W1 误差范数
- `GridFunction::compute_l1_error`
- `GridFunction::compute_w1_error`

---

## 执行顺序

```
Phase A (1-2周) → Phase B (2-3周) → Phase C (并行推进)
  └─ C2 (混合精度) 可随时启动（低依赖）
  └─ D1 (TMOP) 可与 B1 并行
Phase D/E 在 A/B 完成后按需推进
```

## 建议第一项: A1 — Quad/Hex RT 元素

这是最基础的元素类型缺口。补全后：
- 解除 Quad/Hex 网格上混合方法的限制
- 为后续 DPG(Variational Multiscale 等需要不同 trial/test 组合的场景)打基础
- 工作量适中，可独立完成和验证
