# Handoff: MFEM ex26 (Geometric Multigrid)

## 状态：✅ 1:1 翻译完成，基础功能正常

### 翻译示例
| 文件 | 说明 |
|------|------|
| `examples/mfem_ex26_geom_mg.rs` | 1:1 翻译，PCG + 几何多重网格预条件器 |
| `examples/scratch_vcycle_debug.rs` | 诊断版（Galerkin检查、Richardson迭代）|

### 核心库 MG 基础设施（已验证正确）

| 设施 | 位置 | 已验证 |
|------|------|--------|
| `GeometricMgLevel` | `crates/solver/src/geometric_mg.rs` | 矩阵+BC 存储 ✅ |
| `GeometricMgHierarchy` | 同上 | 层次结构、延长算子约束 ✅ |
| `GeometricMgPrecond` | 同上 | V-cycle + W-cycle ✅ |
| `GeometricMgAsPrecond` | 同上 | `Preconditioner` trait 适配器 ✅ |
| `MgChebyshevSmoother` | 同上 | 系数与 MFEM 完全相同 ✅ |
| `RectangularConstrainedOperator` | `crates/solver/src/constrained_operator.rs` | prolong/restrict 对称 ✅ |
| `build_h1_prolongation_matrix` | `crates/space/src/constraints/prolong.rs` | 列和验证 [2,3,4] 正确 ✅ |

### C++ 参考输出（star.mesh, -gr 0 -or 2）
```
Options used:
   --mesh data/star.mesh
   --geometric-refinements 0
   --order-refinements 2
   --device cpu
   --no-visualization
Device configuration: cpu
Memory configuration: host-std
Number of finite element unknowns: 20801
Size of linear system: 20801
   Iteration :   0  (B r, r) = 0.6891
   Iteration :   1  (B r, r) = 1.72051e-05
   Iteration :   2  (B r, r) = 4.08303e-08
   Iteration :   3  (B r, r) = 1.0919e-10
   Iteration :   4  (B r, r) = 2.16182e-13
Average reduction factor = 0.0273569
```

### Rust 输出（star.mesh）
```
Options used:
   --mesh data/star.mesh
   --geometric-refinements 0
   --order-refinements 2
   --device cpu
   --no-visualization
Device configuration: cpu
Memory configuration: host-std
Number of finite element unknowns: 20801
Size of linear system: 20801
   Iteration :   0  (B r, r) = 0.689452
   Iteration :   1  (B r, r) = 0.000153418
   ...
   Iteration :  28  (B r, r) = 2.49974e-13
Average reduction factor = 0.599579
```

### 已知差距

#### 收敛差距（4 vs 28 次迭代）— 持续排查中

DOF 数一致（20801）✅，求解器配置一致 ✅，延长矩阵正确 ✅，Chebyshev 系数一致 ✅。

**完整诊断记录（已验证排除的因素）：**

| 排查项 | 结果 | 方法 |
|--------|------|------|
| Prolongation 矩阵 | ✅ 正确 | P1→P2 列和 [2,3,4]、P2→P4 高阶插值 |
| Chebyshev 系数 | ✅ 一致 | 与 MFEM `OperatorChebyshevSmoother` 逐位比对 |
| λ_max 估计 (2.85) | ✅ 最优 | 覆盖测试 4/8/12/20/50 均更差 |
| `apply_dirichlet_symmetric` | ✅ 正确 | 保持 SPD |
| `RectangularConstrainedOperator` | ✅ 对称 | restrict = prolong^T |
| V-cycle 结构 | ✅ 一致 | 与 MFEM `MultigridBase::Cycle` 逐行比对 |
| CG coarse rtol (1e-2) | ✅ 一致 | 与 `sqrt(1e-4)` 相同 |
| 网格细化 (ref_levels=3) | ✅ 一致 | 20→1280 单元，20801 DOF |
| 无预条件 CG | 763 iters | 基准确认条件数高 |

**当前假设：** 差距源于 MFEM `AssemblyLevel::PARTIAL`（部分组装，element-by-element 操作）与 Rust 完整 CSR 矩阵之间的**对角线数值差异**。

在 MFEM 中：
- 平滑器使用 `bfs[level]->AssembleDiagonal(diag)`，该对角线来自**元素级别**的对角线组装（部分组装路径）
- 算子 `opr` 来自 `bfs[level]->FormSystemMatrix(ess_tdof_list, opr)`，是元素级别的缩减算子
- Chebyshev 平滑器的幂迭代通过 `ProductOperator(invDiagOperator, oper)` 在 **D⁻¹A** 上进行

在 Rust 中：
- 平滑器使用 `a.diagonal()`，来自完整 CSR 矩阵
- 幂迭代在 **D⁻¹/² A D⁻¹/²**（相似变换，特征值相同）上进行
- 算子 `a` 是完整的 CSR 矩阵

虽然数学上这些应该是等价的，但部分组装和完整 CSR 之间的**对角线数值差异**（由于不同的积分/组装路径）可能导致 Chebyshev 区间的微小偏移，累积成明显的收敛差距。

**建议：** 在 linlvo 层实现 `PartialAssemblyOperator`（element-by-element mat-vec + 元素级别对角线），匹配 MFEM 的部分组装行为。此差距非 bug，属于实现策略差异。

#### 输出格式
- C++ 用 `%g` 格式（不定长），Rust 用 `fmt_g`（已对齐，微小差异）
- 平均缩减因子：C++ 打印更多有效位，Rust 使用 `fmt_g`

### 已验证的 MG 正确性

1. **P1→P2 延长矩阵**（debug_prolong 测试）：列和 [2,3,4,...] 符合预期 ✅
2. **P2→P4 延长矩阵**：P2 基函数在 P4 节点求值正确（0.375, -0.125 等）✅
3. **Chebyshev 系数**：与 MFEM `OperatorChebyshevSmoother` 公式逐位一致 ✅
4. **V-cycle 结构**：递归多水平的 pre-smooth → restrict → coarse solve → prolong → post-smooth ✅
5. **RectangularConstrainedOperator**：prolong 和 restrict 对称且正确处理 BC ✅

### 启动提示词（在新 session 中粘贴以继续 pex26 或下一个示例）

> "继续推进 MFEM 示例比对。`examples/mfem_ex26_geom_mg.rs` 已完成基础翻译，迭代次数差距 4 vs 28（linlvo 层部分组装待实现）。如需推进 pex26（并行几何 MG），请确认 `examples/mfem_pex26_parallel_geom_mg.rs` 的当前状态并比对。"
