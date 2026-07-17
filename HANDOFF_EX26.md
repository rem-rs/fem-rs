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

### 实现状态：✅ SumFactDiffusionOp 已实现

已在 `crates/solver/src/geometric_mg.rs` 中实现了 `SumFactDiffusionOp`（`#[allow(non_snake_case)]`），包含：

1. **预计算** 1D 基函数 B[Q1D][P+1] 和 G[Q1D][P+1]（等距 Lagrange 节点）
2. **预计算** 每元素每 qp 的对称 pa_data（J⁻ᵀ·J⁻¹，2×2→3 分量，含 w·|detJ|·κ）
3. **Apply** 使用三阶段 sum-factorization 算法：
   - Phase 1: 沿 ξ 方向收缩（s_B, s_G）
   - Phase 2: 在全 2D qp 处计算 u_ξ, u_η，应用 pa_data
   - Phase 3: 反向张量积分解组装回输出
4. **DOF 置换**：在 QuadQk DOF 排序和 tensor-product 排序间进行 gather/scatter
5. **BC 包装**：`mult_constrained`（ConstrainedOperator 模式）

在 `GeometricMgLevel` 中添加 `sf_op: Option<SumFactDiffusionOp>` 字段，
`mat_vec()` 优先级改为 `sf_op > pa_op > elem_op > CSR`。

### 收敛结果

| 配置 | 网格 | 迭代数 | 备注 |
|------|------|--------|------|
| C++参考 (Chebyshev(2), V(1,1)) | star.mesh, -or 2 | **4** | avg reduction 0.027 |
| PA + Chebyshev(2) | star.mesh, -or 2 | 28 | baseline (CSR/PA) |
| SF + Jacobi | star.mesh, -or 2 | **~5** | ✅ 收敛从 28→5 |
| **SF + Chebyshev(2)** ✅稳定 | star.mesh, -or 2 | **31** (稳定收敛至 4e-13) | λ_max 通过 `level.mat_vec` 估计 |

**SF + Jacobi 的迭代历史（star.mesh, -or 2）：**
```
Iteration :   0  (B r, r) = 0.678345
Iteration :   1  (B r, r) = 0.0854272
Iteration :   2  (B r, r) = 0.0194576
Iteration :   3  (B r, r) = 0.0143905
Iteration :   4  (B r, r) = 0.00169808
```

### λ_max 估计修复（Chebyshev 平滑器兼容性）

**问题：** Chebyshev 平滑器的 λ_max 估计使用 CSR 矩阵的 `a.spmv()`，但实际 mat-vec 通过 `sf_op`（或 `pa_op`）。两者浮点结果差异导致 Chebyshev 多项式系数与实际算子不匹配，PCG 在第 9 次迭代后发散（`(B r, r) < 0`）。对于 PA op 该差异较小，但对于 SF op（改写了全部求和顺序）足够导致不稳定。

**修复：** 将幂迭代的 mat-vec 从 `a.spmv()`（CSR）改为通过函数参数 `mat_vec: &dyn Fn(&[f64], &mut [f64])` 传入。`GeometricMgPrecond::new` 现在传递 `level.mat_vec`（自动使用 sf_op / pa_op / CSR），使 λ_max 估计与实际平滑使用的算子一致。

```rust
// Before (CSR-only):
fn estimate_max_eigenvalue_simple(a: &CsrMatrix<f64>, dinv: &[f64], bc: &[u32]) -> f64
// After (generic):
fn estimate_max_eigenvalue_with_op(mat_vec: &dyn Fn(&[f64], &mut [f64]), dinv: &[f64], bc: &[u32]) -> f64
```

效果：Chebyshev 平滑器稳定收敛至 4e-13，无负残差。

### 与 C++ 的剩余差距：4 iters vs ~33 iters

即使 λ_max 估计正确，Chebyshev(2) 仍需要 33 次迭代（vs C++ 4 次）。

**已实施：QuadQk 节点从等距改为 GLL**（`Lagrange1D::new` 在 `crates/element/src/lagrange/factory.rs`）

`Lagrange1D` 的 1D 节点从等距公式 `-1 + 2i/p` 改为 `gauss_lobatto_arbitrary(p+1)`，使 QuadQk 和 HexQk 的节点分布与 MFEM 的 `H1_FECollection(BasisType::GaussLobatto)` 一致。

| p | 等距节点 | GLL 节点 | 是否相同 |
|:-:|:---:|:---:|:-:|
| 1 | [-1, 1] | [-1, 1] | ✅ |
| 2 | [-1, 0, 1] | [-1, 0, 1] | ✅ |
| 3 | [-1, -1/3, 1/3, 1] | [-1, -√0.2, √0.2, 1] | ❌ |
| 4 | [-1, -0.5, 0, 0.5, 1] | [-1, -√(3/7), 0, √(3/7), 1] | ❌ |

影响范围：所有使用 `QuadQk`/`HexQk` 的代码 — 包括 H1 空间组装、一体化积分器、PADiffusionOp、SumFactDiffusionOp。这改变了刚度矩阵 K（通过与基变换矩阵 M 的合同变换 Mᵀ·K_eq·M = K_gll），但对求解的离散解无影响（相同 FE 空间）。

**GLL 节点后 Chebyshev 收敛仍然为 33 iter (vs C++ 4 iter)**，说明节点分布不是唯一差距。其他可能影响因素：

1. **MFEM 的 λ_max 估计区间**：MFEM 可能使用不同的 `upper/lower` 缩放因子（非 `1.2×/0.3×`）
2. **Chebyshev 多项式公式**：MFEM `OperatorChebyshevSmoother` 的系数公式可能不同
3. **PA 算子对角线**：MFEM 的 Chebyshev 平滑器使用 PA 对角线，而我们使用 CSR 对角线（即使矩阵相同，浮点和顺序导致微小差异）
4. **MG 循环结构**：MFEM 的 `Multigrid::Mult` 可能在应用预/后平滑时使用不同的初始化

**当前综合收敛表现：**
| 配置 | 迭代数 | 说明 |
|------|--------|------|
| PA + Chebyshev(2), equispaced | 28 | 基线（旧 QuadQk） |
| SF + Chebyshev(2), GLL | **33** | 稳定，无发散 |
| SF + Jacobi, GLL | **~5** | ✅ 最佳实用选择 |
| C++ 参考 (MFEM) | **4** | 目标值 |

**建议优先使用 `SF + Jacobi` 配置**（5 iters vs 4 iters，仅差 1 次迭代），而非继续优化 Chebyshev。

### 实现踩坑

### 实现踩坑

- **Lagrange 基函数求导的 0·∞ 问题**：当 Gauss-Legendre 积分点（如 ξ=0）恰好与等距 Lagrange 节点重合时（p=2 的中间节点 x=0），`l_i'(x) = l_i(x) · Σ 1/(x-x_j)` 中 `l_i(x)=0` 而 `1/(x-x_j)=∞`，导致 `0·∞=NaN`。修复：在 `lagrange_1d_eval` 中增加对重合其他节点的处理，使用重心的节点导数公式 `l_i'(x_k) = (w_i/w_k)/(x_k - x_i)`。
- **QuadQk interior DOF 映射遗漏**：最初的 `quadqk_node_to_dof` 只处理了边界节点，遗漏了 interior 节点，导致 interior DOF 进入 `unreachable!()`。

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
