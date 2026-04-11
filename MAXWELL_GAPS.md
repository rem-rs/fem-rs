# Nedelec H(curl) 矢量 FEM — Maxwell 完整实现差距分析

> 生成日期：2026-04-11  
> 对比基准：MFEM master（fe_nd.cpp / ex3p / maxwell_solver miniapp）

---

## 已实现（全部真实验证）

| Phase | 组件 | 说明 |
|-------|------|------|
| 12 | **TriND1 / TetND1** | Whitney 1-forms，3/6 edge DOFs |
| 49 | **TriND2 / TetND2** | Vandermonde 逆，8/20 DOFs，linear curl 验证 |
| 23/49/51 | **HCurlSpace ND1/ND2** | edge + face DOF sharing + sign convention；3D ND2 = `2*n_edges + 2*n_faces`（Phase 51 修复） |
| 24 | **CurlCurlIntegrator** | `∫ μ curl u · curl v`，2D scalar / 3D vector curl，对称 + PSD 验证 |
| 24 | **VectorMassIntegrator** | `∫ α u·v`，H(curl) 和 H(div) 通用 |
| 24 | **VectorAssembler** | 协变 Piola `J^{-T}` / 逆变 `J/det`，sign 自动应用 |
| 29/51/52 | **de Rham 离散算子** | grad (P1→ND1, P2→ND2)，curl_2d，curl_3d (ND2→RT1)，div，`div(curl)=0` 验证 |
| 50 | **AMS 预条件器** | via linger，`solve_pcg_ams` / `solve_gmres_ams` |
| ex3 | **ex3_maxwell** | `∇×∇×E + E = f`，制造解 `E=(sin πy, sin πx)`，O(h) 收敛 |
| pex3 | **pex3_maxwell** | 并行版，ThreadLauncher，ParVectorAssembler |
| 30 | **Newmark-β** | `M ü + K u = f(t)`，用于波动方程 |

### MFEM ex3p 对齐情况

```cpp
// MFEM ex3p 核心 ——————————————————————————————————————
ND_FECollection fec(order, dim);
ParBilinearForm a(&fespace);
a.AddDomainIntegrator(new CurlCurlIntegrator(one));       // ↔ CurlCurlIntegrator{mu:1.0}
a.AddDomainIntegrator(new VectorFEMassIntegrator(one));   // ↔ VectorMassIntegrator{alpha:1.0}
HypreAMS ams(A, &fespace);                                // ↔ solve_pcg_ams (Phase 50)
```
**结论：ex3 这条路完全对齐 MFEM。**

---

## 差距（优先级排序）

### 🔴 高优先级

#### 差距 1：张量系数 μ(x)、ε(x)（各向异性材料）

`CurlCurlIntegrator` 的 `mu` 字段是 `ScalarCoeff`，不支持张量：
```rust
pub struct CurlCurlIntegrator<C: ScalarCoeff = f64> { pub mu: C }
```
MFEM 的 `CurlCurlIntegrator` 接受 `MatrixCoefficient`，用于：
- 铁芯变压器（各向异性 μ 张量）
- 各向异性超导体（各向异性 σ）
- 双折射介质（各向异性 ε 张量）

`MatrixCoeff` trait 已存在（`ConstantMatrixCoeff`、`FnMatrixCoeff`、`ScalarMatrixCoeff`），
但 `CurlCurlIntegrator` 和 `VectorMassIntegrator` 没有矩阵系数版本。

**需要做：**
```rust
// 新增：
pub struct CurlCurlTensorIntegrator<C: MatrixCoeff = ConstantMatrixCoeff> { pub mu: C }
// 实现：∫ (μ curl u) · curl v，其中 μ 是 dim×dim 矩阵

pub struct VectorMassTensorIntegrator<C: MatrixCoeff = ConstantMatrixCoeff> { pub alpha: C }
// 实现：∫ (α u) · v，其中 α 是 dim×dim 矩阵（各向异性介电常数 ε）
```

**工作量：小**（复用现有 integrator 框架，只改内积计算）

---

#### 差距 2：H(curl) 切向边界积分（吸收/阻抗 BC）

MFEM ex3 中 Silver-Müller 吸收边界条件：
```
n × (μ⁻¹ curl E) + γ (n × E) × n = g   on Γ_abs
```
需要边界双线性积分算子：
```
∫_Γ γ (n×u)·(n×v) dS
```

现有 `BoundaryMassIntegrator` 是 H1 的标量版（只处理节点 DOF 的 `u·v`），
没有 H(curl) 切向分量版。H(curl) 的边界积分需要：
1. 知道边界面法向量 `n`
2. 对每个 H(curl) 基函数计算 `n × φᵢ`（切向分量）
3. 集成到矩阵 `K_ij = ∫_Γ γ (n×φᵢ)·(n×φⱼ) dS`

**需要做：**
- 新增 `VectorBoundaryQpData`（含 `phi_vec`、`curl`、`normal`）
- 新增 `VectorBoundaryBilinearIntegrator` trait
- 新增 `VectorBoundaryAssembler`（迭代边界面，Piola 变换基函数）
- 新增 `TangentialMassIntegrator`：`∫_Γ γ (n×u)·(n×v) dS`

**工作量：中**（需要边界面上的 Piola 变换和新的 assembler 分支）

---

### 🟡 中优先级

#### 差距 3：时域 Maxwell（`ε∂²E/∂t² + σ∂E/∂t + curl(μ⁻¹ curl E) = J`）

MFEM `maxwell_solver` miniapp 用 E-B 一阶系统 + Symplectic 蛙跳：
```
ε ∂E/∂t = curl(μ⁻¹ B) - σE - J
∂B/∂t   = -curl E
```

fem-rs 有 Newmark-β（Phase 30），但没有接到 H(curl) 空间上做时域 Maxwell。

**需要做：**
- 新建 `ex_maxwell_time.rs`
- 组合 `VectorAssembler`（组装 εM、σM、K）
- 用 Newmark-β 求解 `ε M Ë + σ M Ė + K E = J(t)`
- 制造解验证（如 `E(t) = sin(πt) * (sin πy, sin πx)`）

**工作量：中**（组合现有组件，主要是 example 级别的胶水代码）

---

#### 差距 4：Maxwell 特征值（腔体谐振频率）

MFEM ex11/ex13：
```
curl curl E = ω² ε E    →    K x = λ M x
```
求最小非零特征值 ω²（腔体谐振频率）。

fem-rs 已有 LOBPCG（Phase 20），但没有接到 H(curl) 广义特征值问题上。

**需要做：**
- 新建 `ex_maxwell_eigenvalue.rs`
- 用 `VectorAssembler` 组装 `(K = curl-curl, M = vector-mass)`
- 施加 PEC 边界条件（n×E=0 → 消去边界 edge DOF）
- 调用 `LobpcgSolver` 求解广义特征值 `K x = λ M x`
- 验证：单位正方形腔体最低谐振频率 `ω² = π²(m² + n²)`（m=n=1 → ω² = 2π²）

**工作量：小**（主要是 example 级别）

---

### 🟢 低优先级

#### 差距 5：Quad/Hex 网格上的 Nedelec 元

MFEM 有 `ND_QuadrilateralElement`、`ND_HexahedronElement`（Legendre 多项式张量积）。
fem-rs 只有单纯形（三角形/四面体）。

**工作量：大**（需要新 element 族、新 space 拓扑处理）

---

## 实现路线图（Phase 54+）

| Phase | 内容 | 工作量 |
|-------|------|--------|
| **54a** | `CurlCurlTensorIntegrator<MatrixCoeff>` + `VectorMassTensorIntegrator<MatrixCoeff>` | 小 |
| **54b** | `VectorBoundaryBilinearIntegrator` trait + `TangentialMassIntegrator` + `VectorBoundaryAssembler` | 中 |
| **54c** | `ex_maxwell_time.rs`（时域，Newmark-β）+ 制造解收敛验证 | 中 |
| **54d** | `ex_maxwell_eigenvalue.rs`（LOBPCG 腔体谐振）| 小 |
| 55 | Quad/Hex Nedelec（`ND_QuadrilateralElement`、`ND_HexahedronElement`） | 大 |
