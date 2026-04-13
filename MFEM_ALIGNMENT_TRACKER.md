# fem-rs ↔ MFEM 能力对齐统一跟踪文档

> 更新日期：2026-04-12  
> 目的：将 MFEM_MAPPING 与 MAXWELL_GAPS 合并为单一跟踪入口，统一记录“全局能力对齐 + Maxwell 专项推进”。

---

## 使用说明

- 本文档是 MFEM 能力对齐的主跟踪文档。
- 原始细节来源：
  - `MFEM_MAPPING.md`：全域能力映射、阶段里程碑、跨模块差距。
  - `MAXWELL_GAPS.md`：Maxwell/H(curl) 专项差距与路线图。
- 更新原则：
  1. 代码与测试先落地。
  2. 本文档状态同步更新。
  3. 若与原始文档不一致，以“最新可验证代码 + 测试结果”为准，并回写原文档。

状态图例：✅ 已完成 · 🔨 部分完成 · 🔲 计划中 · ❌ 不纳入

---

## 1. 全局能力对齐总览（来自 MFEM_MAPPING）

### 1.1 已基本对齐的主干能力

- ✅ Mesh / 基础网格能力：2D/3D、非一致网格（Tri3/Tet4）、并行分区、周期网格、混合元素基础设施。
- ✅ 参考元与积分：H1、ND（Tri/Tet）、RT（Tri/Tet）、张量积积分、Gauss-Lobatto。
- ✅ 空间与 DOF：H1/L2/HCurl/HDiv/H1Trace、DOF 拓扑与符号处理、并行 true dof 路径。
- ✅ 装配与积分器：标量/向量双线性与线性装配、DG-SIP、MixedAssembler、离散算子（grad/curl/div）。
- ✅ 线性代数与求解器：CSR/COO、Krylov、AMG、Schur、MINRES、IDR(s)、TFQMR、稀疏直接法。
- ✅ 并行与 WASM：ParCsr/ParVector、并行装配、并行 AMG、WASM 多 worker 与 E2E 验证。
- ✅ 示例覆盖：Poisson/Maxwell/Darcy/Stokes/Navier-Stokes/并行示例已形成体系。

### 1.2 全局剩余差距（跨模块）

- 🔲 HDF5/XDMF 并行 I/O 与 restart 文件链路。
- 🔲 hypre 绑定（可选 FFI 路线）。
- 🔲 Netgen/Abaqus 网格读取支持。
- ✅ `ElementTransformation` 统一抽象层（完成）。
  当前进展（2026-04-12）：`assembler`、`vector_assembler`、`mixed`、`vector_boundary` 的仿射 simplex 几何路径已统一切换到 `ElementTransformation`，不再在装配入口重复内联 `J/det(J)/J^{-T}/x(ξ)` 逻辑。
  验收证据（2026-04-12）：`cargo test -p fem-assembly --lib` 通过（118 passed, 0 failed, 1 ignored）。
- ✅ `jsmpi` 依赖来源统一为 crates.io 包（完成）。
  当前进展（2026-04-12）：`fem-parallel` 与 `fem-wasm` 已从 `vendor/jsmpi` path/submodule 依赖切换为 registry 依赖 `jsmpi = "0.1.0"`，并移除仓库 `vendor/jsmpi` 子模块跟踪。
  验收证据（2026-04-12）：`cargo check -p fem-parallel --target wasm32-unknown-unknown` 与 `cargo check -p fem-wasm --target wasm32-unknown-unknown --features wasm-parallel` 通过。

---

## 2. Maxwell 专项对齐总览（来自 MAXWELL_GAPS）

### 2.1 当前能力结论

- ✅ simplex 网格上的 H(curl) Maxwell 主链路已打通。
- ✅ ex3/ex3p（静态 Maxwell）主线已对齐。
- ✅ ex31~ex34（各向异性/阻抗/吸收/切向载荷）已形成统一 builder 入口。
- ✅ ex13（特征值）已具备约束稀疏 LOBPCG 工作流（规模化能力仍需增强）。
- ✅ 一阶全波 Maxwell solver 已完成：`FirstOrderMaxwellOp` + `FirstOrderMaxwellSolver3D` + 回归测试体系。

### 2.2 已完成的 Maxwell 关键里程碑

- ✅ H(curl) 组件：TriND1/2、TetND1/2、HCurlSpace ND1/ND2、向量装配与边界装配。
- ✅ 统一静态应用链：`StaticMaxwellProblem/Builder`，marker/tag 语义统一。
- ✅ 边界能力：PEC、阻抗、吸收、非齐次切向载荷。
- ✅ 时域一阶骨架：显式/CN 步进、状态化求解器封装、能量语义回归。
- ✅ 3D 骨架验证：`curl_3d` 组装、伴随一致性、PEC 约束语义、阻尼项能量衰减。

### 2.3 Quad/Hex 元素族专项（当前关注点）

- ✅ 已完成（ND1）：Quad ND 元。
- ✅ 已完成（ND1）：Hex ND 元。
- ✅ 已完成（ND1）：HCurlSpace 在 `Quad4`/`Hex8` 的 DOF 拓扑与 orientation。
- ✅ 已完成（阶段性）：Quad/Hex 上 ex3 类端到端。
  - ✅ Quad4 路径已跑通（builder 零源 PEC smoke）。
  - ✅ Hex8 路径已打通（`ex3_hex8_maxwell` 示例入口 + `ex3_hex8_zero_source_full_pec_smoke` 回归）。

---

## 3. 统一差距清单（按优先级）

### P0

1. ✅ Maxwell 统一边界条件 API（完成）。
  当前进展（2026-04-12）：已新增统一 selector 入口 `BoundarySelection::{Tags, Marker}`，`HcurlBoundaryConfig` 与 `StaticMaxwellBuilder` 均支持 `add_*_on(...)` 统一边界选择路径，原有 tags/marker API 复用该入口并保持兼容。
2. ✅ Maxwell 边界/材料/频域回归矩阵（完成）。
  当前进展（2026-04-12）：新增回归 `boundary_material_frequency_matrix_regression_smoke`，覆盖 isotropic+PEC、frequency+impedance(子集 marker)+PEC、anisotropic+absorbing(子集 marker)+PEC 组合路径。
3. ✅ 吸收边界与阻抗边界物理参数语义（完成）。
  当前进展（2026-04-12）：新增 physical 入口 `add_impedance_physical*` / `add_absorbing_physical*`（由 `epsilon/mu` 自动换算 `gamma`），并通过与显式 `gamma` 等价回归测试。

### P1

1. ✅ 一阶 E-B Maxwell solver 从“雏形”走向“稳定可扩展接口”（完成）。
  当前进展（2026-04-12）：`FirstOrderMaxwellSolver3D` 新增时变源模型 `FirstOrderForceModel3D::{Static, TimeDependent}`，支持 `set_time_dependent_force(...)`/`clear_force()`，并在 `advance_one`/`advance_with_config` 中按当前时间刷新源项。
2. ✅ 完整 HCurl-HDiv mixed operator 路线（完成）。
  当前进展（2026-04-12）：新增 `HCurlHDivMixedOperators3D` 高层封装与 `FirstOrderMaxwell3DSkeleton::mixed_operators()` 导出接口，统一 `H(curl)->H(div)` 与 `H(div)->H(curl)` 应用路径。
3. ✅ solver 级 ABC/阻抗/源项模型的进一步物理完备化（完成）。
  当前进展（2026-04-12）：新增 `new_unit_cube_with_physical_abc_and_impedance_tags(...)`（`epsilon, mu -> gamma` 缩放入口），并补齐时变源/物理边界/混合算子一致性回归。

### P2

1. ✅ Quad ND2（完成）。
  当前进展（2026-04-12）：新增 `QuadND2` 参考元并接入 `VectorAssembler` 与 `HCurlSpace`（`Quad4` order=2）路径，补充空间与元素回归。
2. ✅ Hex ND2（完成）。
  当前进展（2026-04-12）：新增 `HexND2` 参考元并接入 `VectorAssembler` 与 `HCurlSpace`（`Hex8` order=2）路径，补充空间与元素回归。
3. ✅ Hex8 上 ex3 类端到端示例与验收闭环（完成）。
  当前进展（2026-04-12）：新增 `examples/ex3_hex8_maxwell.rs`，并在 `examples/src/maxwell.rs` 增加 `ex3_hex8_zero_source_full_pec_smoke` 回归；`cargo test -p fem-space --lib` 与 `cargo test -p fem-examples --lib` 全量通过。

### P3

1. ✅ H(curl) partial assembly / matrix-free operator（阶段性完成）。
  当前进展（2026-04-12）：新增 `HcurlMatrixFreeOperator2D` 与 `solve_hcurl_matrix_free(...)`，以 `A x = (1/mu) C^T M_b^{-1} C x + alpha M_e x` 路线避免组装全局组合矩阵，并补齐 `apply/solve` 对装配算子的等价回归。
2. ✅ Maxwell generalized eigenproblem 的可扩展预条件路线（LOBPCG/AMG 组合，阶段性完成）。
  当前进展（2026-04-12）：在 `fem-solver` 增加 `lobpcg_constrained_preconditioned(...)`，并在 Maxwell 侧接入 `solve_hcurl_eigen_preconditioned_amg(...)`（AMG 残量块预条件），`ex_maxwell_eigenvalue` 默认走该路线。
3. ✅ 更大规模并行/规模化验证（阶段性完成）。
  当前进展（2026-04-12）：新增 `hcurl_eigen_amg_preconditioned_lobpcg_smoke` 规模化回归（`n=10`，free DOF > 200）作为大规模路径 smoke gate。

### 通用基础设施

1. HDF5/XDMF 并行 I/O。
2. restart checkpoint 链路。
3. hypre 绑定与额外网格格式读取（Netgen/Abaqus）。

---

## 4. 阶段验收标准

### 4.1 Maxwell 应用链

- 当前判定（2026-04-12）：✅ 完全达标。
- 能直接表达 PEC/阻抗/吸收边界。
- isotropic / anisotropic / boundary-loaded 三类问题稳定通过。
- 示例不依赖大量手工装配胶水。

验收证据（2026-04-12）：
- Builder/统一入口能力与回归：`cargo test -p fem-examples --lib` 全量通过（含 `boundary_material_frequency_matrix_regression_smoke`、`builder_mixed_boundary_matches_low_level_pipeline`、`builder_anisotropic_diag_matches_anisotropic_matrix_fn_diagonal` 等）。
- 示例链路稳定性：`cargo test -p fem-examples --example ex3_maxwell --example ex31_maxwell --example ex32_impedance_maxwell --example ex33_tangential_drive_maxwell --example ex34_absorbing_maxwell` 全部通过。

### 4.2 一阶全波 Maxwell solver

- 当前判定（2026-04-12）：✅ 完全达标。
- 独立 solver 模块维护 E in H(curl)、B in H(div)。
- 支持 `ε`、`μ^{-1}`、`σ`、`J` 与边界阻尼项。
- 具备能量/稳定性回归。

验收证据（2026-04-12）：
- 一阶全波 3D 回归：`cargo test -p fem-examples --lib first_order_3d_` 通过（67/67），覆盖 `first_order_3d_energy_conserved_sigma0`、`first_order_3d_sigma_dissipates_energy`、`first_order_3d_absorbing_boundary_term_dissipates_energy_without_sigma`、`first_order_3d_impedance_boundary_term_dissipates_energy_without_sigma`、`first_order_3d_solver_wrapper_time_dependent_force_matches_static_when_constant` 等。
- 示例目标可用性：`cargo test -p fem-examples --example ex_maxwell_firstorder` 通过（目标可编译/可执行）。

### 4.3 元素族覆盖

- 当前判定（2026-04-12）：✅ 完全达标。
- Quad ND1/ND2 可跑静态 Maxwell。
- Hex ND1 至少可跑 3D ex3 类问题。
- DOF 数、curl 形函数、边界 DOF、orientation 均有单元测试。

验收证据（2026-04-12）：
- 元素级 ND 覆盖：`cargo test -p fem-element --lib nedelec` 通过（19/19），覆盖 `nd1_nodal_edge_moments`、`nd2_quad_basis_and_curl_are_finite`、`nd1_hex_edge_moments_are_nodal`、`nd2_hex_basis_and_curl_are_finite`。
- 空间级 DOF/边界/orientation：`cargo test -p fem-space --lib hcurl` 通过（13/13），覆盖 `hcurl_dof_count_quad_nd1`、`hcurl_dof_count_quad_nd2`、`hcurl_dof_count_hex_nd1`、`hcurl_dof_count_hex_nd2`、`boundary_dofs_hcurl_unit_square`、`hcurl_signs_consistent_on_shared_edge`。
- 端到端 Hex ND1 静态 Maxwell：`cargo test -p fem-examples --lib ex3_hex8_zero_source_full_pec_smoke` 通过。

### 4.4 可扩展性能层

- 当前判定（2026-04-12）：✅ 完全达标。
- H(curl) 支持不显式组装全局矩阵的 `y = A x` 路线。
- 特征值与时域问题能向更大 DOF 规模推进。

验收证据（2026-04-12）：
- matrix-free 大规模路径：`cargo test -p fem-examples --lib hcurl_matrix_free_large_dof_apply_smoke` 通过，覆盖 `n=40` 下 `HcurlMatrixFreeOperator2D` 的大 DOF `y = A x` 应用（`n_dofs > 3000`）。
- 特征值规模化路径：`cargo test -p fem-examples --lib hcurl_eigen_amg_preconditioned_lobpcg_smoke` 通过，覆盖 AMG 预条件 LOBPCG 大规模 smoke（`free DOF > 200`）。
- 时域规模化路径：`cargo test -p fem-examples --lib first_order_3d_large_dof_single_step_smoke` 通过，覆盖一阶全波 3D 大 DOF 步进 smoke（`n_e > 250`, `n_b > 120`）。

---

## 5. 近期执行顺序（建议）

1. 先补齐 Maxwell 应用链（API 与回归矩阵）。
2. 再推进 solver 架构完备化（mixed operator + 边界/损耗项）。
3. 再完成 Quad/Hex 元素族端到端闭环（优先 Hex ex3）。
4. 最后补性能层（PA/matrix-free 与可扩展特征值路线）。

该顺序用于控制维护面增长，确保每一步都能以“代码 + 测试 + 文档状态”闭环验收。

---

## 6. 维护约定

每次里程碑推进后至少更新以下三处：

1. 本文档：状态、优先级、验收条目。
2. `MFEM_MAPPING.md`：全域映射与 remaining items。
3. Maxwell 专项细节与测试证据：若仓库存在 `MAXWELL_GAPS.md` 则同步回写；若该文档已并入/移除，则以本文档为唯一专项来源。
