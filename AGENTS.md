# 🎯 MFEM 示例比对

> 当我说"**mfem示例比对**"时，请按以下流程执行。

**前提：C++ 示例和 Rust 示例必须是 1:1 翻译，物理设定、边界条件、材料参数、求解器配置完全一致，比对结果才有意义。** 如果 Rust 示例和 C++ 示例已偏离，必须先修正为 1:1 翻译，再做比对。

**核心原则：比对的目的不是「让示例跑起来」，而是「发现 fem-rs 核心库的短板并修复」。**
当比对发现 C++ 有的功能 fem-rs 缺少时（例如某类积分器、预处理器、网格操作），这恰恰是最有价值的工作——改进核心库 `crates/` 的代码，而不是在示例里绕过去。示例翻译是手段，核心库完善才是目的。

将 Rust fem-rs 示例与 C++ MFEM 对应示例进行逐行比对，确保物理设定、求解器配置、输出格式完全一致。

## 工作流程

### 1. 获取 C++ 参考源

C++ 参考源在 WSL 的 `/tmp/mfem/` 下。需要先确认该路径下的源码是最新的：

```bash
# 从 GitHub 获取最新 MFEM exN.cpp
curl -sL "https://raw.githubusercontent.com/mfem/mfem/refs/heads/master/examples/exN.cpp"
```

**注意：** `/tmp/mfem/examples/exN.cpp` 可能被本地修改过（如 ex2 的 `ref_levels` 被注释为 0），务必与上游对比 diff。

### 2. 确认 C++ 编译运行

```bash
cd /tmp/mfem && g++ -std=c++17 -O3 -I. examples/exN.cpp -L. -lmfem -o /tmp/exN_cpp
/tmp/exN_cpp -m data/xxx.mesh -no-vis 2>&1
```

关键输出包括但不限于：
- `Number of finite element unknowns: N`
- `Size of linear system: N`
- 每步迭代 `(B r, r)` 值
- `Average reduction factor = ...`
- 生成的文件（`refined.mesh`, `displaced.mesh`, `sol.gf` 等）

### 3. 定位 Rust 对应示例

```bash
# 搜索对应示例
grep -r "exN\|example N" examples/
```

一般在 `examples/mfem_exN_xxx.rs`。阅读文件顶部注释确认对应关系。

### 4. 逐项比对清单

| 检查项 | 说明 |
|--------|------|
| **默认网格** | C++ 默认 `../data/xxx.mesh`，Rust 是否一致？ |
| **网格加密** | `ref_levels` 公式是否一致？`floor(log(5000\|50000./NE)/log(2.)/dim)` |
| **边界条件** | Dirichlet 标签、Neumann 标签是否对应？ |
| **材料参数** | 单材料/多材料？`PWConstCoeff` 分段系数是否正确？ |
| **载荷** | 体积力 vs 边界牵引力？方向/大小一致？ |
| **求解器** | PCG + GS/SSOR/Jacobi？rtol/max_iter 一致？ |
| **输出文件** | `refined.mesh`, `displaced.mesh`, `sol.gf` 文件名和格式一致？ |
| **打印格式** | `Number of finite element unknowns:`, `Size of linear system:`, 迭代日志格式一致？ |

### 5. 编译运行 Rust 示例

```bash
# 复制 C++ 网格文件到 Rust 项目
wsl cp /tmp/mfem/data/xxx.mesh /mnt/c/Users/lilu/works/fem-rs/data/

cargo run --example mfem_exN_xxx -- -m data/xxx.mesh -no-vis 2>&1
```

### 6. 迭代数据对比

如果两边都是 PCG + 相同预处理器 + 相同收敛判据 `(M⁻¹rₖ, rₖ)`，前几步 `(B r, r)` 应该逐位一致。
弹性力学等复杂问题可能因预处理器实现差异导致路径分叉，但**平均缩减因子**应接近（<1%偏差）。

### 7. 修正步骤

发现问题后：
1. 不要修改 Rust 示例代码，理论上不需要修改！
2. 涉及求解器/网格核心逻辑，修改对应 crate 代码
3. Rust 缺少 C++ 的某个特性（如边界牵引力积分器、分段常数系数），检查现有 API 是否已支持，如果不支持，在对应 crate 代码补全
4. 编译通过后重新运行对比

### 8. 提交

```bash
git add examples/mfem_exN_xxx.rs   # 以及其他修改的源文件
git commit -m 'feat: 1:1 port of MFEM exN (xxx)'
git push
```

不提交生成的 `refined.mesh`、`displaced.mesh`、`sol.gf` 等输出文件。

---

## 踩坑记录（持续更新）

### ex7 曲面网格基础设施短板

| 短板 | C++ MFEM | fem-rs | 修复方案 |
|------|----------|--------|----------|
| **Quad4 曲面** | `-e 1` 立方体→球面 | `SurfaceAssembler` 只支持 Tri3 | ✅ `SurfaceQuad4*` 系列积分器 + `SurfaceQuad4Assembler` |
| **`dim()` vs `space_dim()` 混淆** | `mesh->Dimension() = 2`，`SpaceDimension() = 3` | `MeshTopology::dim()` 返回嵌入维 | ✅ `topological_dim()` 方法 + `DofManager` 改用 `topo_dim` |
| **曲面 AMR** | `RefineAtVertex` + 局部加密 | 无曲面网格细化 | ✅ `refine_uniform_surface_tri3/quad4` + `refine_at_vertex_surface` 在 `crates/mesh/src/amr/` |
| **高阶曲面 (order≥2)** | `SetNodalFESpace` + 等参映射 | `CurvedMesh` 对 `Mesh<3>` Tri3 用 Tet 边枚举 | ✅ `CurvedMesh::elevate_to_order` 改用 `element_type_at(0).dim()` |

### ex4 H(div) 四边形/曲边界网格 L² 误差差距

| 网格 | 类型 | MFEM L² | fem-rs L² | 状态 |
|------|------|---------|-----------|------|
| beam-tri.mesh | 直线三角 | 0.0567 | 0.0567 | ✅ 完美 |
| square-disc.mesh | 曲边界三角 | 0.0147 | 0.141 | ❌ 10x |
| star.mesh | 四边形 | 0.016 | 2.162 | ❌ 大差距 |

### ex25 完成状态

**状态：** ✅ 1:1 翻译完成，核心库已补充 7 项基础设施（SesquilinearForm, RestrictedCoefficient 等），8 个 Bug 已修复。收敛差距 1.57x (C++ 251 vs Rust 394 iters)，原因在 linlvo 层需后续排查。详见 `HANDOFF_EX25.md`。

三角形直线网格（beam-tri）完美匹配，说明组装路径（GradDivIntegrator + VectorMassIntegrator + Piola 变换）对仿射 Tri3 正确。四边形和曲边界网格的差异在**组装层**，目前原因未明。

### ex26 完成状态

**状态：** ✅ 1:1 翻译完成，基础功能正常。

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 默认网格 | ✅ | star.mesh 默认，unit_square_tri(n) 内置后备 |
| 网格加密 | ✅ | `floor(log(5000/NE)/log(2)/dim)` 公式一致 |
| 边界条件 | ✅ | 全边界 Dirichlet，`apply_dirichlet_symmetric` |
| 求解器 | ✅ | PCG + MG(V(1,1), Chebyshev(2), CG coarse rtol=1e-2) |
| 输出文件 | ✅ | `refined.mesh` + `sol.gf` |
| 输出格式 | ✅ | "Iteration : N (B r, r) = ...", "Average reduction factor = ..." |
| **收敛性** | ❌ | C++ 4 iters vs Rust 28 iters (avg red. 0.027 vs 0.600) |

**收敛差距根因：** MFEM 使用 `AssemblyLevel::PARTIAL`（部分组装，element-by-element），Rust 使用完整 CSR 矩阵。虽然数学等价，但特征值估计、对角线数值、浮点运算顺序差异导致 Chebyshev 平滑器效率不同。

**已验证正确的 MG 基础设施：**
- 延长矩阵（P1→P2 列和 [2,3,4] 正确，P2→P4 高阶插值正确）
- Chebyshev 系数与 MFEM `OperatorChebyshevSmoother` 逐位一致
- V-cycle/W-cycle 递归结构，prolong/restrict 对称（`RectangularConstrainedOperator`）
- 详见 `HANDOFF_EX26.md`

**后续：** 如需改善收敛性，需在 linlvo 层实现 `PartialAssemblyOperator`（element-by-element mat-vec）。px26（并行几何 MG）示例 `examples/mfem_pex26_parallel_geom_mg.rs` 已存在。

**已排除的原因：**
- 插值符号修正（`interpolate_vector` 的 `element_sign`）：已验证正确
- 等参 Jacobian（`isoparametric_jacobian` + `QuadQ1`）：与 MFEM 的 bilinear mapping 逻辑一致
- L² 误差计算（`compute_hdiv_l2_error`）：对新模块，三/四边形分别用仿射/等参 Jacobian
- `eliminate_dirichlet` BC 消除：与 beam-tri 一致
- `HDivSpace::build_2d_quad` DOF 分配和 element_signs：DOF 计数正确，signs 符合预期
- QuadRT0 参考单元基函数验证：满足 ∫Φ_i·n̂_j = δ_ij

**关键线索：四边形 RT0 插值误差不随网格加密收敛**

对比试验（统一单位正方形网格，RT0，MMS 解）：

| h | 三角形 L² | 四边形 L² |
|---|----------|----------|
| 0.5 | 0.317 | 1.25 |
| 0.25 | 0.160 | 1.37 |
| 0.125 | 0.080 | 1.40 |
| 0.0625 | 0.040 | 1.41 |

三角形 O(h) 收敛 ✅。四边形 ~1.4 **恒定不收敛** ❌。说明 `interpolate_vector` 或 `compute_hdiv_l2_error` 的四边形路径存在未被发现的 bug。

**后续排查方向：**
- 对比 MFEM `RT_FECollection` 对 Quad4 的局部 DOF 排序和面法向约定
- 检查 `QuadRT0` 的 `eval_basis_vec` 与 `compute_hdiv_l2_error` 中 Piola 变换的符号一致性
- 在 `compute_hdiv_l2_error` 中对单个四边形单元输出中间值（jac, det_j, phys_phi, xp）与解析解对比

### H(curl) 求解器选择（ex3 经验）

| 预条件器 | 收敛性 | 适用场景 |
|----------|--------|----------|
| Jacobi | 慢但稳定（~1358 iters for 25K DOFs） | 简单测试、小问题 |
| SSOR(ω=1) | 比 Jacobi 稍快（~677 iters）但大问题可能不收敛 | 对应 MFEM GSSmoother |
| **AMS** (Hiptmair-Xu) | 快（~411 iters for 25K DOFs），对问题规模不敏感 | **生产环境默认选择** |

### AMS 预条件器使用要点

AMS 需要离散梯度算子 `G: H¹ → H(curl)`，构建方式：

```rust
let h1_space = H1Space::new(mesh.clone(), 1);
let g = DiscreteLinearOperator::gradient(&h1_space, &hcurl_space)?;
```

**关键约束：**
- `apply_dirichlet`（行归零）会破坏矩阵 SPD 性质，AMS-PCG 会崩溃
- 必须用 `eliminate_dirichlet` → `solve_pcg_ams` → `expand_from_reduced` 流程
- 离散梯度 G 也需对应缩减（删除受约束 DOF 的行）

### 代码分层规则

```
crates/          ← fem-rs 核心库，所有算法改进放这里
  assembly/src/ams_solver.rs  ← AMS 求解器便利函数
  solver/src/precond.rs       ← solve_pcg_ams 底层实现
examples/        ← 1:1 翻译示例（与 MFEM 对照的基准）
  mfem_exN_xxx.rs             ← 禁止修改！只做对照
  src/maxwell.rs              ← 示例共享工具库，可微调但优先改核心库
```

**原则：** 翻译好的示例代码不要改。发现性能或功能差距时，改进核心 `crates/` 的代码，示例只做验证调用。

### 依赖链约束

```
fem-core → fem-mesh → fem-space → fem-assembly → fem-solver
```

- `fem-solver` 只有 `fem-linalg` 依赖，不能引用 `fem-space` 或 `fem-assembly`
- AMS 便利函数（`solve_hcurl_ams`）放在 `fem-assembly` 中（有 `fem-solver` + `fem-space` 依赖）
- `DiscreteLinearOperator` 在 `fem-assembly::discrete_op` 中

### 回归基线更新

改变求解器等核心逻辑后，需要更新回归基线：

```bash
FEM_UPDATE_BASELINES=1 cargo test -p fem-examples --lib 2>&1
git add tests/baselines/*.json
```

---

# Python Environment

This project uses **uv** (https://github.com/astral-sh/uv) to manage Python environments and package management.

## Setup

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or on Windows:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   uv pip install -r pyproject.toml  # if pyproject.toml exists
   # or
   uv pip install <package-name>
   ```

## Key Commands

- `uv venv` — Create a virtual environment in `.venv/`
- `uv pip install <pkg>` — Install a package
- `uv pip install -r requirements.txt` — Install from requirements file
- `uv pip list` — List installed packages
- `uv run <script.py>` — Run a script with the environment's Python
- `uv build` — Build Python packages
- `uv publish` — Publish packages to PyPI

## Notes

- The `.venv/` directory is git-ignored.
- Python bindings for this project are built using **maturin** and installed via `uv pip install`.

## Building Python Bindings

The Python bindings use PyO3 + maturin, located at `crates/python/`.

1. Ensure a Python 3.11+ interpreter is available:
   ```bash
   uv python install 3.11
   ```

2. Build the development version:
   ```bash
   uv venv
   uv pip install maturin numpy scipy
   maturin develop
   ```

3. Or set `PYO3_PYTHON` to point to the uv-managed Python before `cargo build`:
   ```powershell
   $env:PYO3_PYTHON = "$(uv python find 3.11)"
   cargo build -p fem-py
   Copy-Item target/debug/fem_py.dll python/fem/_core.pyd -Force
   ```

4. Run the test suite:
   ```powershell
   $env:PYTHONPATH = "python"
   uv run python -m pytest tests/test_full_pipeline.py -x -q
   ```

The Python module exposes mesh generation, FE spaces (H¹/L²/VectorH¹/HCurl/ComplexH¹),
assembly (stiffness/mass/source), linear solvers (CG/GMRES/PCG/BiCGSTAB/LU/Cholesky),
extrusion, supermesh construction, and complex grid function operations.

---

# MFEM 示例翻译比对状态 (2026-07)

## ✅ ex22 — 复数 Helmholtz (1:1 完成)

| 维度 | 功能 | 状态 |
|------|------|------|
| p=0 (H1) | DiffusionIntegrator + MassIntegrator | ✅ |
| p=1 (HCurl) | CurlCurlIntegrator + VectorMassIntegrator | ✅ |
| p=2 (HDiv) | GradDivIntegrator + VectorMassIntegrator | ✅ |
| 2D (Tri3/Quad4) | 全部三种变体 | ✅ |
| 3D (Tet4/Hex8) | 全部三种变体 | ✅ |
| CLI | -m -r -o -p -mu -eps -sigma --omega -f -a -herm | ✅ |
| 求解器 | GMRES + 块对角 GS/DSmoother | ✅ |
| 精确解 L² 误差 | inline-* 网格自动检测 | ✅ |
| 输出 | refined.mesh + sol_r.gf + sol_i.gf + sol_z.gf | ✅ |

**未对齐：** GLVis 可视化、Partial Assembly、输出文件精度 (8位 vs 14位)

## ✅ ex23 — 波动方程 (1:1 完成)

| 维度 | 功能 | 状态 |
|------|------|------|
| 空间类型 | H1 (标量) | ✅ |
| 时间积分 | GeneralizedAlpha2 (s=10/11/12) | ✅ |
| 2D/3D | Mesh<2> + Mesh<3> | ✅ |
| 预条件器 | DSmoother (Jacobi) | ✅ |
| 输出 | ex23.mesh + ex23-init.gf + ex23-final.gf | ✅ |
| CLI | -m -r -o -s -tf -dt -c -dir/-neu -vis | ✅ |

**未对齐：** GLVis 可视化、VisIt 输出、ODE 类型 11/12 未完整验证

## ✅ ex24 — 离散算子 (1:1 完成，有降级)

| 维度 | 功能 | 状态 |
|------|------|------|
| p=0 (grad) H1→HCurl | DiscreteLinearOperator::gradient + 混合矩阵 | ✅ |
| p=1 (curl 3D) HCurl→HDiv | DiscreteLinearOperator::curl_3d + 弱 curl | ✅ |
| p=2 (div) HDiv→L2 | assemble_hdiv_l2_mixed + DLO divergence | ✅ (o=1), ⚠️ (o=2) |
| 2D | inline-tri.mesh + star.mesh | ✅ |
| 求解器 | CG + DSmoother (Jacobi) | ✅ |
| L² 投影 | project_coefficient / project_hcurl_coefficient / project_hdiv_coefficient_2d | ✅ |

**注意：** p=2 o=2 (RT1→P1) 有已知 bug，误差 ~1.42。TriRT1 的 nodal basis test 已通过 (DOF_k(ψ_i)=δ_{ki})，但实际散度投影仍然不精确。降级到 RT0→P0 (o=1)。根源可能在 project_hdiv_coefficient_2d 或 sign 应用路径。

**核心库修复 (`crates/`):**
| 修复 | 文件 |
|------|------|
| 二阶 GeneralizedAlpha2 时间积分器 | `solver/src/ode/structural.rs` |
| project_hdiv_coefficient_2d/3d (HDiv L² 投影) | `assembly/src/postproc/grid_function.rs` |
| assemble_hcurl_hdiv_weak_curl: HCurl element_signs + 移除多余 1/detJ | `assembly/src/mixed.rs` |
| TriRT1: Vandermonde 边顺序→TRI_FACES | `element/src/raviart_thomas/tri_rt1.rs` |
| TriRT1: 矩 1 定义 t→2t-1 (对齐 MFEM) | `element/src/raviart_thomas/tri_rt1.rs` |
| interpolate_vector RT1: 矩 1 定义 t→2t-1 | `space/src/hdiv.rs` |
| INLINE 3D hex/tet 网格解析 | `io/src/mfem.rs` |

---

# MFEM 示例翻译比对状态 (2026-07)

## ✅ ex22 — 复数 Helmholtz (1:1 完成)

## ✅ ex23 — 波动方程 (1:1 完成)

## ✅ ex24 — 离散算子 (1:1 完成，有降级)

---
