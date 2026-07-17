# Handoff: MFEM ex26 (Geometric Multigrid) — 收敛差距根因已确定

## 状态：✅ 1:1 翻译完成，收敛差距根因已定位

### 翻译示例
| 文件 | 说明 |
|------|------|
| `examples/mfem_ex26_geom_mg.rs` | 1:1 翻译，PCG + 几何多重网格预条件器 |
| `examples/scratch_vcycle_debug.rs` | 诊断版 |

### 核心库 MG 基础设施

| 设施 | 位置 | 状态 |
|------|------|------|
| `GeometricMgLevel` / `GeometricMgHierarchy` | `crates/solver/src/geometric_mg.rs` | ✅ |
| `GeometricMgPrecond` / `GeometricMgAsPrecond` | 同上 | ✅ V/W-cycle |
| `MgChebyshevSmoother` | 同上 | ✅ 系数与 MFEM 逐位一致 |
| `StoredElementOperator` | 同上 | ✅ ConstrainedOperator 模式 |
| `PADiffusionOp` | 同上 | ✅ on-the-fly 求值 |
| `RectangularConstrainedOperator` | `crates/solver/src/constrained_operator.rs` | ✅ prolong/restrict 对称 |
| `build_h1_prolongation_matrix` | `crates/space/src/constraints/prolong.rs` | ✅ 列和正确 |
| `assemble_bilinear_with_elements` | `crates/assembly/src/assembler.rs` | ✅ 同积分循环返回元素矩阵 |

### 收敛差距诊断 — 完整实验记录

**C++ 4 iters, avg reduction 0.027 vs Rust 28 iters, avg reduction 0.600**

| 实验 | 改动 | 迭代数 | 结论 |
|------|------|--------|------|
| Baseline CSR | `apply_dirichlet_symmetric` + CSR SpMV | 28 | 基准 |
| λ_max 扫描 | 覆盖 4/8/12/20/50 | 28→33→46→55→70→95 | 默认 2.85 最优 |
| 缩减系统 | `eliminate_dirichlet` 消除 BC DOF | 28 | BC 非根因 |
| ConstrainedOperator | `mult_constrained` (elem_op BC 包装) | 28 | 模式正确，无改善 |
| On-the-fly PA | `PADiffusionOp` (transform-then-sum) | 28 | 同 CSR |
| Sum-then-transform | `PADiffusionOp` (J⁻ᵀ·Σ 顺序) | **105** | 更差，回滚 |
| 无预条件 CG | 纯 CG | 763 | 基线确认条件数高 |

### 已研究的 MFEM 源码

| MFEM 组件 | 文件 | 结论 |
|-----------|------|------|
| `PABilinearFormExtension::MultInternal` | `fem/bilinearform_ext.cpp:487` | 矩阵免费，on-the-fly 求值 |
| `PABilinearFormExtension::AssembleDiagonal` | `fem/bilinearform_ext.cpp:370` | 元素级对角线累加 |
| `PABilinearFormExtension::FormSystemMatrix` | `fem/bilinearform_ext.cpp:468` | 创建 `ConstrainedOperator` 包装 |
| `DiffusionIntegrator::AddMultPA` | `fem/bilininteg_diffusion_pa.cpp` | sum-factorization kernel |
| `DiffusionIntegrator::AssembleDiagonalPA` | 同上 | 1D 张量积分解 |
| `DiffusionIntegrator::AssemblePA` | 同上 | 预计算 `pa_data` (J⁻ᵀ·J⁻¹) |
| `Operator::FormSystemOperator` | `linalg/operator.cpp:227` | `ConstrainedOperator(rap, ess_tdofs)` |
| `ConstrainedOperator::Mult` | `linalg/operator.cpp` | 输入/输出 BC DOF 归零 |
| `MultigridBase::Cycle` | `fem/multigrid.cpp:179` | V/W-cycle 递归结构 |
| `OperatorChebyshevSmoother::Setup` | `linalg/solvers.cpp:538` | Chebyshev 系数计算 |
| `PowerMethod::EstimateLargestEigenvalue` | `linalg/power.cpp` | D⁻¹A 上幂迭代 |

### 根因：需要 sum-factorization kernel

所有尝试（StoredElementOperator、PADiffusionOp、ConstrainedOperator、求和顺序更改）都**无法匹配 MFEM 的浮点行为**。原因是 MFEM 的 `AddMultPA` 使用 **sum-factorization**（张量积分解），它将 2D Quad4 的 mat-vec 分解为沿 ξ 和 η 方向的**顺序 1D 操作**，完全重组了浮点运算中间值。

storage: `pa_data` (每个 qp 的 J⁻ᵀ·J⁻¹ 对称压缩)

apply 算法（简化）：
```
for each element e:
  for each 1D quad point ξ_q:
    // 沿 η 方向收缩
    for each j in 0..p:
      s_x[j] += Σ_i x[e][i][j] · B_1D[q][i]
      s_y[j] += Σ_i x[e][i][j] · G_1D[q][i]
  for each 1D quad point η_r:
    // 沿 ξ 方向收缩
    u_xi = Σ_j s_x[j] · G_1D[r][j]  // du/dξ
    u_eta = Σ_j s_y[j] · B_1D[r][j]  // du/dη
    // 用 pa_data 变换
    ∇u_phys = J⁻ᵀ(q,r) · [u_xi, u_eta]
    // 累加
    for each i in 0..p:
      phi_xi += G_1D[q][i] · B_1D[r][j]  ... (张量积基函数)
      y[e][i][j] += w · |detJ| · κ · ∇u_phys · ∇φ
```

这个算法在 crates 层**完全不存在**。`PADiffusionOperator` (`crates/assembly/src/partial.rs:144`) 使用非张量积 2D 参考单元，不支持 sum-factorization。

### 实现路径

需要实现一个针对 Quad4 的 `SumFactDiffusionOp`，存储在 `crates/solver/src/`：

1. **预计算** 1D 基函数 B[Q1D][P+1] 和 G[Q1D][P+1]
2. **预计算** 每元素每 qp 的对称 pa_data（J⁻ᵀ·J⁻¹，2×2→3 分量）
3. **Apply** 使用上述 sum-factorization 算法（3 层嵌套循环：元素 × 1D-ξ × 1D-η）
4. **BC 包装** 用 `mult_constrained`（ConstrainedOperator 模式）

完成后，`mat_vec()` 优先级应设为：`SumFactDiffusionOp > PADiffusionOp > StoredElementOperator > CSR`。

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
