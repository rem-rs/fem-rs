# 🎯 MFEM 示例比对

> 当我说"**mfem示例比对**"时，请按以下流程执行。

**前提：C++ 示例和 Rust 示例必须是 1:1 翻译，物理设定、边界条件、材料参数、求解器配置完全一致，比对结果才有意义。** 如果 Rust 示例和 C++ 示例在物理上已偏离（如 ex2 改前用体积力代替边界牵引力、单材料代替双材料），必须先修正为 1:1 翻译，再做比对。

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

关键输出包括：
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
1. 修改 Rust 示例代码
2. 如果涉及求解器/网格核心逻辑，修改对应 crate 代码
3. 如果 Rust 缺少 C++ 的某个特性（如边界牵引力积分器、分段常数系数），检查现有 API 是否已支持（`PWConstCoeff`, `NeumannIntegrator`, `elasticity::ElasticityIntegrator<C>` 等）
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

### ex4 H(div) 四边形/曲边界网格 L² 误差差距

| 网格 | 类型 | MFEM L² | fem-rs L² | 状态 |
|------|------|---------|-----------|------|
| beam-tri.mesh | 直线三角 | 0.0567 | 0.0567 | ✅ 完美 |
| square-disc.mesh | 曲边界三角 | 0.0147 | 0.141 | ❌ 10x |
| star.mesh | 四边形 | 0.016 | 2.162 | ❌ 大差距 |

三角形直线网格（beam-tri）完美匹配，说明组装路径（GradDivIntegrator + VectorMassIntegrator + Piola 变换）对仿射 Tri3 正确。四边形和曲边界网格的差异在**组装层**，目前原因未明。

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
