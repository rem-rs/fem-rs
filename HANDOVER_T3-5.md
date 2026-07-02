# 交接文档 — fem-rs 当前状态与 T3-5 攻坚计划

## 会话概览

本次会话完成了全部分阶段改进（Stage 0–11 + Tier 1 + Tier 2 + 部分 Tier 3）。剩余关键未完成任务为 **T3-5: 3D NCMesh ND2/RT2 悬挂面约束**。

## 已完成工作

### Stage 0–11（全部完成）
- 工程卫生、高阶单元修复、MixedAssembler、IO 对齐、MMS 护栏、DG 框架、AMS/ADS、非线性工具箱、分布式 ParMesh
- 详见 git log (`git log --oneline | head -20`)

### Tier-1（7/7，全部完成）
TetRT2 dof_coords → DG Euler3D HLLC → Slope limiter 接入 → CR1/PyraND1 MMS → 分布式 TFQMR → GetProlongation → 3D Maxwell ND2 BC

### Tier-2（8/8，全部完成）
GPU f64 assembly → CUDA backend → GPU AMG → ParMesh streaming → AMS h-independence → 元素级 ghost exchange → Exodus/CGNS reader → 2D ADS ND1→RT0 curl

### Tier-3 部分完成
- T3-3: `par_direct_solve` baseline（local diag LU，完整 gather-to-rank-0 尚需 MPI_Gatherv）
- T3-4: rmetis-first 文档确认（METIS FFI 已移除）+ rmetis 优化（precomputed deg, O(deg²)→O(deg log deg) coarsening, balance pass）

## 准备攻坚 T3-5

### T3-5 目标

为 3D NCMesh（非协调网格）添加 HDIV（RT2）和 HCurl（ND2）元素的悬挂面约束传播。当前 `constraints.rs` 只处理标量 H1 P1/P2 的悬挂约束，缺少向量元（Nédélec、Raviart-Thomas）的切向/法向连续约束。

### 前置了解

核心文件：
- `crates/mesh/src/amr.rs` — NCMesh 细化、`HangingFaceConstraint` 结构体（仅几何，无 HCurl/HDiv 逻辑）
- `crates/space/src/constraints.rs` — `apply_hanging_constraints`/`apply_hanging_face_constraints`（仅标量 H1 P1/P2）
- `crates/space/src/hcurl.rs` — HCurlSpace DOF 编号（含 `quad_face_to_dof` 用于 Hex NDk 面 DOF）
- `crates/space/src/hdiv.rs` — HDivSpace DOF 编号（含 `FaceDofMap`）

### 设计要点

1. **HCurl（ND2）悬挂边约束**：
   - 在 3D Tet NCMesh 中，悬挂面将粗边暴露在细元素内部
   - ND2 有边 DOF + 面 DOF + 内部 DOF。边 DOF 需从粗边线性插值
   - 约束形式：`u_fine_edge = Σ c_i · u_coarse_edge_i`

2. **HDiv（RT2）悬挂面约束**：
   - HDiv 的法向通量约束：细面的法向通量必须等于对应粗面的通量
   - 需结合 `HangingFaceConstraint` 的面节点信息计算约束系数

3. **实现路径**：
   - 在 `constraints.rs` 中添加 `apply_hanging_constraints_hcurl` 和 `apply_hanging_constraints_hdiv`
   - 从 `HangingFaceConstraint`/`HangingNodeConstraint` 推导出细 DOF → 粗 DOF 的线性约束
   - 约束矩阵应用到系统矩阵：`A' = C^T · A · C`，其中 C 是约束矩阵

### 测试验收

- Tet4 NCMesh 上 ND1/ND2 的 curl-curl 特征值准确性（与均匀细化对比）
- Tet4 NCMesh 上 RT0/RT1 的 Darcy MMS 收敛阶（NP ≈ 1，flux ≈ 1）
- 悬挂面处无伪振荡（通过 patch test）

### 参考代码

- `crates/space/src/constraints.rs:apply_hanging_constraints` — H1 标量约束模板
- `crates/element/src/nedelec/hex_ndk.rs` — Hex NDk 的面 DOF 结构
- `crates/mesh/src/amr.rs:308-452` — 3D 非协调细化的 `refine_nonconforming_3d_internal`（生成 HangingFaceConstraint）
