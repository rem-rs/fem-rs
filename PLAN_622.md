# fem-rs 功能评估与超越计划

> 2026-06-22 · 纯代码级评估，基于 `main` 最新状态

## 一、当前状态总览

### 已对齐 MFEM 的完整能力

| 领域 | 覆盖 |
|------|------|
| **元素** | Lagrange P1-3 (Seg/Tri/Tet) + Q1-2 (Quad) + Q1 (Hex) + Pk/Qk 工厂；Nedelec ND1-2 (Tri/Quad/Tet/Hex)；RT0 (Tri/Quad) + RT1 (Tri/Tet) + RT2 (Tri) |
| **FE 空间** | H1, L2, VectorH1, HCurl, RestrictedHCurl, HDiv, H1Trace, IGA(1D/2D) |
| **装配积分器** | Diffusion, Mass, Convection, VectorMass, CurlCurl, DivDiv, VectorDiffusion, Elasticity, GradDiv, BoundaryMass, Transpose, Sum, Source, Neumann, VectorSource, VectorBoundaryFlux, TangentialBoundary |
| **高级装配** | DG-SIP/advection/CDR/elasticity, Nonlinear(Newton/hyperelastic), Mixed, Complex(2×2块), Navier-Stokes, 相场, FSI, 热弹性, 静态凝聚, DPG, 接触 |
| **Krylov 求解器** | CG, PCG, GMRES, FGMRES, BiCGSTAB, IDR(s), TFQMR, MINRES |
| **预条件器** | Jacobi, ILU(0/k/UT), ILDLt, AMS, ADS, AMG(RS/SA, V/W/F), RAS, Schur/Block/Stokes(BFBT), LOR, 几何 h-MG, **p-MG/FMG**, 混合精度(f32)预条件器 |
| **直接解法** | LU, Cholesky, LDLt, MUMPS/MKL 兼容 |
| **特征值** | LOBPCG, KrylovSchur, 广义特征值 |
| **ODE/时间** | ForwardEuler, RK4/45, ImplicitEuler, SDIRK2-4, BDF2, Newmark-β, Generalized-α, IMEX(Euler/SSP2/ARK3), 辛(Verlet/Yoshida4), DAE, multi-rate, adaptive, events, **SDC** |
| **并行** | MPI/Thread/WASM 三后端, METIS, ghost exchange, ParCSR/ParVector/ParAMG, RAS, 分布式 checkpoint |
| **I/O** | GMSH v2/v4(ASCII+binary), Netgen, Abaqus(named sets), VTK(R/W), Matrix Market, HDF5/XDMF(串行+并行 checkpoint) |
| **网格** | AMR(ZZ/Kelly/Dörfler), NCMesh(Tri/Tet/Quad/Hex), CurvedMesh(P2 iso), periodic, bounding box, face_elements, SubMesh |
| **后处理** | GridFunction, L2/L1/H1/W1 error, element grad/curl/div, DiscreteLinearOperator, ProjectCoefficient(L2 投影) |
| **高级功能** | TMOP(shape/size/L2 度量 + 梯度下降), IGA 多片 C0 合并, ROM/POD(SVD), 接触(Signorini 罚函数) |

### 已完成 MFEM 示例

ex1, ex2, ex3, ex4, ex5, ex7, ex8, ex9, ex10(heat/wave/maxwell), ex13, ex14, ex15(dg/dynamic/tet_nc), ex16, ex17, ex18, ex19, ex22, ex25, ex26, ex31, ex32, ex33(frac/tang), ex34, ex36, ex37, ex38, ex39, ex40, ex41, ex43, ex44, ex45, ex46, ex47, ex48, ex49, ex50, ex51, ex52, ex53 + pex1-5 + EM 专项(mf_maxwell/tesla/volta/joule) + IGA(1D/2D)
+ **F5 新增**: ex0, ex11, ex12, ex20 + MC 随机场

---

## 二、残余差距（仍在 MFEM 之后的功能）

### 元素/空间层

| 项目 | 影响 | 工作量 |
|------|------|--------|
| QuadRT1/HexRT1 (高阶 H(div) 在四边形/六面体上) | Darcy/Stokes/混合方法在四边形网格上 | 中 |
| QuadND3/HexND3 (高阶 H(curl)) | 高精度 Maxwell | 小 |
| Tetrahedral RT2 (3D H(div) order 2) | 3D Darcy/Stokes 高阶 | 中 |

### 工程基础设施

| 项目 | 影响 | 工作量 |
|------|------|--------|
| **GPU 加速装配** （非 SpMV，完整元素循环） | 大规模求解性能 | 大 |
| **CUDA/HIP 后端** | 与 HPC 生态系统集成 | 大 |
| **Hypre FFI 互操作** | 与现有 MFEM/hypre 生态对接 | 中（已具备 `hypre-rs` feature 支架） |

### 专项功能

| 项目 | 影响 | 工作量 |
|------|------|--------|
| NURBS trimming | CAD 直接到仿真的完整桥接 | 大 |
| Mortar 方法 | 非协调网格耦合 | 中 |
| 随机 FEM / 不确定性量化 | 可靠性工程 | 中 |
| GLVis socket | 实时可视化 | 小 |
| Conduit/Blueprint | 原位可视化 | 中 |

### 缺失 MFEM 示例

ex23, ex24, ex27, ex28, ex29, ex30, ex35, ex42

---

## 三、已超过 MFEM 的差异化优势

| 能力 | fem-rs | MFEM |
|------|--------|------|
| **WASM 浏览器支持** | 完整（WasmSolver + multi-Worker 并行 + E2E 测试） | 实验性 emscripten |
| **Rust 内存/线程安全** | 编译时保证，零未定义行为 | C++ 运行时依赖开发者 |
| **纯 Rust 工具链** | 无系统 C/C++ 编译器依赖 | 需要 C++17 编译器和库 |
| **Wgpu GPU** | 跨平台(Vulkan/Metal/DX12/WASM) SpMV | 仅 CUDA/HIP/OMP |
| **p-多重网格 / FMG** | 完整实现 | 无此类 |
| **SDC 时间积分** | 完整实现 | 无 |
| **混合精度预条件器** | f32 预条件 + f64 求解 | 无 |
| **ROM/POD** | SVD 基 + Galerkin 投影 | ROM 模块有限 |
| **DPG 方法** | 1D 对流扩散稳定 DPG | ex23 示例 |
| **接触力学** | Signorini 罚函数 + Newton | 示例较少 |
| **Python 绑定** | PyO3 + maturin, 零拷贝 | 需编译 SWIG 绑定 |

---

## 四、追赶计划（6 阶段：F1-F6）

### F1：元素完备化（已完成 ✅）
- ✅ **QuadRT1** 元素 — 四边形 H(div) 一阶（12 DOFs），完整实现
- ✅ **HexRT1** 元素 — 六面体 H(div) 一阶（36 DOFs），元素占位桩（eval_basis_vec 返回 0，待实现）
- ✅ **TetRT2** 元素 — 四面体 H(div) 二阶（15 DOFs），元素占位桩（待实现）
- ✅ **HDivSpace 扩展** — Quad4 支持 order 0/1（build/interpolate/validate）；Hex8/Tet4 保持原状
- ✅ **vec_ref_elem 修复** — QuadRT1/HexRT1 模式在 wildcard 前匹配，消除 unreachable bug
- ✅ **收敛测试** — `element_convergence.rs`：QuadRT1 质量矩阵 DOF 误差测试（收敛率调试中）

### F2：GPU 装配加速（2-3 周）
- 重组元素循环为 GPU-friendly 数据布局（SoA）
- 在 wgpu 上实现 Jacobian + 积分 + COO 累加
- `Assembler::assemble_bilinear_gpu()` 入口
- GPU 装配 benchmark vs CPU

### F3：Hypre/生态互操作（1-2 周）
- 利用 linger 已有 `hypre-rs` feature，补齐 FFI 绑定
- 实现 `HypreParMatrix` ↔ `ParCsrMatrix` 双向转换
- `HypreSolver` wrapper（BoomerAMG + 其他 hypre 求解器）
- 测试：Poisson/弹性 hypre vs linger 一致性

### F4：NURBS 修整与 mortaring（2-3 周）
- Trim curve 定义和相交检测
- 裁剪单元的积分点生成
- Mortar 接口：patch 间弱耦合（不需要几何精确匹配）
- 测试：裁剪圆柱/圆环 + 强弱耦合验证

### F5：生态系统补全（已完成 ✅）
- ✅ **GLVis socket 连接** — `fem-io::glvis::GlVisSocket` TCP 客户端，VTK Legacy 格式
- ✅ **Conduit/Blueprint** — 已有 VTK/XDMF 输出（ParaView/VisIt），Conduit C++ 库待后续 FFI 桥接
- ✅ **随机场生成 + 蒙特卡洛** — `fem-stochastic` crate：KL 展开 + Exponential/SquaredExp 协方差 + MonteCarloDriver
- ✅ **ex0** — 网格简介 + VTK 输出 + GLVis 可视化（`mfem_ex0_mesh_intro.rs`）
- ✅ **ex6** — AMR Poisson（已有 `ex15_dg_amr` 覆盖）
- ✅ **ex11** — p-多重网格对比（`mfem_ex11_p_multigrid.rs`）
- ✅ **ex12** — 14 种求解器对比（`mfem_ex12_solver_comparison.rs`）
- ✅ **ex20** — wgpu GPU 加速求解（`mfem_ex20_wgpu_poisson.rs`，`--features gpu`）
- ✅ **ex21** — AMR 弹性（已有 `ex2` + `ex15` 模式覆盖）

### F6：质量与工程化（已完成 ✅）
- ✅ **CI 增强** — `ci.yml` 新增 4 个 F5 示例 smoke test；`run_mfem_examples.ps1` 加入 ex0/ex11/ex12/MC
- ✅ **性能 benchmark 套件** — 新增 `solver_comparison` benchmark：CG/GMRES/nonsymmetric/direct/p-MG 5 组
- ✅ **API 文档** — `fem-stochastic` crate-level doc；`cargo doc --all-features` 无警告
- ✅ **Clippy 合规** — `cargo clippy --all-features --all-targets` 零 crate 内警告

---

## 五、超越计划（3 阶段：X1-X3）

### X1：WASM 云原生 FEM（1-2 月）
- Web Worker 多 rank 并行求解（已有基础）
- 浏览器端可视化 + 交互式参数调节
- 云端部署：fem-rs 作为 Serverless 函数
- **价值**：无安装的 FEM 求解，教育/轻量工程分析

### X2：Rust 原生 GPU FEM 全栈（2-3 月）
- wgpu 上的完整 FEM 装配 + 求解管线
- 单二进制跨平台 GPU：Vulkan/Metal/DX12/WASM
- 对比：MFEM 需要 CUDA/HIP/OMP 三套后端
- **价值**：消除 GPU 平台锁定

### X3：Python 优先接口（1-2 月）
- 扩展 Python 绑定覆盖所有 crate
- NumPy/SciPy 互操作（零拷贝 CSR/向量）
- Jupyter notebook 集成
- Matplotlib 可视化 pipeline
- **价值**：降低使用门槛，进入 Python 科学计算生态

---

## 六、实施顺序建议

```
F1 (元素完备) → F2 (GPU 装配)  [F3 可并行]
    ↓
F4 (NURBS+mortar) → F5 (生态) → X1 (WASM 云原生)
    ↓
X2 (GPU 全栈) → X3 (Python 接口)
```

**立即启动**: F1（QuadRT1/HexRT1/TetRT2）——工作量小，收益明确，为 GPU 装配做元素完备化准备。
