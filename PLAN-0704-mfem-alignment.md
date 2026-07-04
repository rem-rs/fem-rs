# fem-rs × MFEM 100% 对齐改进计划 (0704)

> 目标：将前期评估中列出的全部缺口逐项对齐到 MFEM 等价能力，分三阶段推进。
> **推进顺序：Phase 1（算法深度）→ Phase 2（结构/可扩展性）→ Phase 3（I/O/生态）**。
> 每个 work item 给出：基线（含文件位置）、对齐目标、任务清单、验收标准、规模（S/M/L）。

---

## 总览与里程碑

| 阶段 | 内容 | 里程碑 | 状态 |
|------|------|--------|------|
| P1 | 算法深度对齐（13 项） | M1.1–M1.13 | 🚧 本计划优先推进 |
| P2 | 结构/可扩展性对齐（5 项） | M2.1–M2.5 | 待启动 |
| P3 | I/O/生态对齐（5 项） | M3.1–M3.5 | 待启动 |

**完成定义（DoD）：** 某项达到验收标准、有对应回归测试、`cargo test --workspace` + `cargo clippy --workspace -- -D warnings` 全绿、`README.md` 状态表更新。

---

# Phase 1 — 算法深度对齐（优先推进）

按依赖与收益排序：先做低风险解锁项（M1.1–M1.5），再做核心求解深度（M1.6–M1.9），最后做元/空间深度（M1.10–M1.13）。

## M1.1 Krylov 补齐：MINRES / GCR　〔S〕　✅ 完成

| 字段 | 内容 |
|------|------|
| **基线** | `crates/solver/src/lib.rs` 有 CG/PCG/GMRES/FGMRES/BiCGSTAB/IDR(s)/TFQMR；MINRES 仅在 `block.rs` 鞍点内部，无独立入口；无 GCR |
| **MFEM 对齐** | `MINRESSolver` / `GCRSolver` 等价的独立求解器 |
| **实现** | `solve_minres` / `solve_minres_operator`（Lanczos + 显式 Givens QR + 回代后端求解，`lib.rs:1327-1545`）；`solve_gcr` / `solve_gcr_operator`（restarted GCR，非对称系统可靠，`lib.rs:1547-1680`）|
| **验收** | 15 个测试全通过：MINRES 通过 SPD/不定/Helmholtz shift/residual 递减/vs block 交叉验证；GCR 通过非对称/全空间 SPD/operator/gmres 对比 |
| **任务** | ① `lib.rs` 新增 `solve_minres`（对称不定，三项 Lanczos + Curry 条件稳定化） ② 新增 `solve_gcr`（广义共机残差，允许非对称） ③ 新增矩阵无关 `solve_minres_operator` / `solve_gcr_operator` |
| **验收** | 不定系统（Stokes、Helmholtz 负 shift）MINRES 残差单调下降；GCR 对非对称对流扩散收敛阶与 GMRES 一致；单元测试用构造矩阵比对已知解 |
| **规模** | S · `lib.rs` + block_minres 可抽取合并 |

## M1.2 非线性求解器：LBFGS / 信赖域　〔M〕

| 字段 | 内容 |
|------|------|
| **基线** | `crates/assembly/src/nonlinear.rs` 有 `NonlinearForm` trait + `NewtonSolver`（Armijo 线搜索）；`solver/src/multiphysics.rs` 有 `CoupledNewtonSolver`；无 LBFGS、无信赖域 |
| **MFEM 对齐** | `LBFGSSolver` + 信赖域（Steihaug-CG / dogleg）等价 |
| **任务** | ① `nonlinear.rs` 新增 `LbfgsSolver`（历史 m=5..20，双循环递归，对接 `NonlinearForm`） ② 新增 `TrustRegionSolver`（Steihaug-CG 截断共轭梯度 + Δ 自适应） ③ 在 `examples/` 增加 LBFGS/TR 求解路径对比 |
| **验收** | 超弹性（Neo-Hookean）LBFGS 收敛步数 ≤ Newton 的 1.5×、无 Jacobian 装配；信赖域对坏初值鲁棒性优于 Newton；与 `NewtonSolver` 在 `hyperelastic_hooke.rs` 交叉验证位移一致 |
| **规模** | M · `nonlinear.rs` ~500 行新增 |

## M1.3 时间积分：ABM / Crank-Nicolson　〔S〕

| 字段 | 内容 |
|------|------|
| **基线** | `solver/src/ode.rs`（1957 行）+ `butcher.rs`（587 行）。有 FE/RK4/RK45/SDIRK/BDF/IMEX/Verlet，无 Adams-Bashforth-Moulton、无显式 Crank-Nicolson |
| **MFEM 对齐** | `AdamsBashforthSolver` / `AdamsMoultonSolver`（预测-校正） + `CrankNicolsonSolver`(θ=0.5) |
| **任务** | ① `ode.rs` 新增 `AdamsBashforthMoulton`（变阶 1–5，RK4 启动，PECE 模式） ② 新增 `CrankNicolson`（θ 方法，θ=0.5；与已有 `ImplicitEuler`(θ=1) 共用线性求解路径） |
| **验收** | 对 `mfem_ex10_heat_equation`，ABM5 误差 O(Δt⁵)，CrankNicolson O(Δt²) 且保能量近似守恒；与 SDIRK-2 同问题交叉验证 |
| **规模** | S · `ode.rs` ~250 行 |

## M1.4 LOR 预条件真实化　〔M〕

| 字段 | 内容 |
|------|------|
| **基线** | `solver/src/lor.rs` 委托 Jacobi PCG，非真实低阶精化 |
| **MFEM 对齐** | `LORSolver` / `LowOrderRefinedOperator` — 高阶 H1 投影到 P1/Q1 LOR 空间，用 LOR 矩阵做 AMG 预条件 |
| **任务** | ① 实现 `LowOrderRefinedOperator`：高阶节点 → LOR 节点嵌入矩阵 `P`，`A_LOR = Pᵀ A_HO P` ② 接入 `fem-amg` 的 `AmgPrecond` 做 LOR-AMG ③ 提供 `solve_pcg_lor_amg` 顶层 API |
| **验收** | P3/P4 Poisson 用 LOR-AMG 的 PCG 迭代数与 P1 直接近（≤ 1.5×）；替换现有 Jacobi 委托后所有调用方测试仍绿 |
| **规模** | M · `lor.rs` 重写 ~400 行 |

## M1.5 装配期 BC 消元 + Robin 线性积分器　〔M〕

| 字段 | 内容 |
|------|------|
| **基线** | `assembly/src/form.rs` 的 `BilinearForm` 无就地本质边界消元；Dirichlet 在 `bc.rs` 单独处理；`integrator.rs` 无专用 `RobinLFIntegrator` |
| **MFEM 对齐** | `BilinearForm::EliminateEssentialBC` / `EliminateEssentialBCFromDiag` 风格就地消元；`RobinLFIntegrator` 一等公民 |
| **任务** | ① `form.rs` 增加 `eliminate_essential_bc(&self, ess_dofs, rhs)` / `eliminate_essential_bc_from_diag`（行列归零 + RHS 右端项修正） ② `integrator.rs` 新增 `RobinLFIntegrator`（∫(κ u v + ∂u/∂n g)），封装 coef + 边界标记 ③ 弃用旧混合路径，迁移 `mfem_ex27_robin_bc.rs` 到新 API |
| **验收** | BC 消元后解与旧路径 `bc.rs` 逐节点一致（相对差 < 1e-14）；Robin LF 与现有组合法（boundary mass + Neumann）结果一致 |
| **规模** | M · `form.rs` + `integrator.rs` ~350 行新增 |

## M1.6 分片矩阵系数（按 tag）　〔S〕

| 字段 | 内容 |
|------|------|
| **基线** | `assembly/src/coefficient.rs` 有 `PWCoeff`（标量按 tag 分片）；`ConstantMatrixCoeff` / `FnMatrixCoeff` 整体矩阵系数，无按 tag 分片能力 |
| **MFEM 对齐** | `PWConstMatrixCoefficient` 等价 |
| **任务** | ① 新增 `PwMatrixCoeff`：`PwMatrixCoeff { data: HashMap<i32, Box<dyn MatrixCoeff>>, default }` ② 与其他 `MatrixCoeff` 同接口对接 `CurlCurlTensorIntegrator` / `VectorDiffusionIntegrator` |
| **验收** | 各向异性 Maxwell 多材料接口 `mfem_ex31_anisotropic_maxwell.rs` 用 `PwMatrixCoeff` 分区域赋值 |
| **规模** | S · `coefficient.rs` ~120 行 |

## M1.7 DG BR1　〔S〕

| 字段 | 内容 |
|------|------|
| **基线** | `assembly/src/dg_br2.rs`（BR2 局部 lifting）；有 SIPG/LDG/upwind 但无 BR1 |
| **MFEM 对齐** | BR1（Bassi–Rebay 1）全局 lifting operator |
| **任务** | ① 新增 `assembly/src/dg_br1.rs`：全局 lifting `{L̄, r̄}` via 元素级积分 + 全局求解 ② 分支进 `dg_reduced.rs` / `form.rs` 的 `DgScheme` 枚举；测试与 SIPG/BR2 对照 |
| **验收** | DG 扩散 BR1 L2 误差阶 O(hᵖ⁺¹) 与 SIPG 一致；对流-扩散稳定性测试通过 |
| **规模** | S · `dg_br1.rs` ~200 行 |

## M1.8 特征值求解器：ARPACK / FEAST　〔L〕

| 字段 | 内容 |
|------|------|
| **基线** | `solver/src/eigen.rs`（700 行）有 LOBPCG（含约束/预条件）+ Krylov-Schur；无 ARPACK、无 FEAST |
| **MFEM 对齐** | `ARPACKSolver`（IRAM/复 Arnoldi）与 FEAST（等值面积分） |
| **任务** | ① **ARPACK**：纯 Rust IRAM（Implicitly Restarted Arnoldi，shift-invert 模式），放 `eigen.rs`；评估 `arpack-ng` FFI 作 fallback ② **FEAST**：contour quadrature + 并行（Rayon）矩阵 `(zI−A)⁻¹` 求解（复用直接求解），子空间 Rayleight Ritz ③ 统一 `Eigensolver` trait，四种方法同接口 |
| **验收** | Laplacian/Maxwell 空腔前 20 根与 LOBPCG 相对误差 < 1e-6；FEAST 指定区间无漏根；新增 `mfem_ex13_*` 的 ARPACK/FEAST 路径 |
| **规模** | L · `eigen.rs` ~800 行 |

## M1.9 VEM 高阶（P2/Pk）　〔L〕

| 字段 | 内容 |
|------|------|
| **基线** | `space/src/vem.rs` 仅 P1，注释 "高阶 VEM 留待将来"；`examples/vem_poisson_polygonal.rs` 存在 |
| **MFEM 对齐** | VEM 任意阶（至少到 P3，多边形） |
| **任务** | ① `vem.rs` 实现 P2/P3：投影算子 `Π∇`（Vandermonde 多项式基）、稳定项 `S = h·(I−Π)`、虚拟 DOF 自由度管理 ② `assembly/src/vem_poisson.rs` 扩展装配到高阶 ③ `examples/vem_poisson_polygonal.rs` 增加 P2/P3 收敛阶 |
| **验收** | P2 多边形 Poisson L2 误差 O(h³)，P3 O(h⁴)；凸/非凸多边形无 spurious mode |
| **规模** | L · `vem.rs` + `vem_poisson.rs` ~600 行 |

## M1.10 IGA FESpace 高阶桥接　〔M〕

| 字段 | 内容 |
|------|------|
| **基线** | `space/src/iga_fe_space.rs` 桥接仅 p+1∈{2,3}（Line2/3、Quad4/9）；高阶 B 样条装配在 `fem-assembly/iga*` 但 FESpace 桥受限 |
| **MFEM 对齐** | IGA 任意阶（Line/Quad/Hex）+ k-refinement |
| **任务** | ① `iga_fe_space.rs` 重写 DOF 映射为 Greville 节点 → 任意阶节点表 ② 打通 `IgaFESpace1D/2D/3D` 与标准 `BilinearForm` 装配 ③ 测试 `mfem_ex_iga_poisson_*` P3/P4 收敛 |
| **验收** | IGA Poisson P3 误差 O(h³)、P4 O(h⁴)；k-refinement（p↑ h 固定）误差指数下降 |
| **规模** | M · `iga_fe_space.rs` ~300 行 |

## M1.11 高阶 PA sum-factorization　〔L〕

| 字段 | 内容 |
|------|------|
| **基线** | `assembly/src/pa/` 仅 Hex Q1–Q4、Quad Q1–Q2、Tet4；高阶 PA 不完整 |
| **MFEM 对齐** | 张量积 sum-factorization 任意阶（Hex/Tet/Quad），矩阵自由求解 |
| **任务** | ① `pa/` 新增通用 `hex_qk.rs` / `tet_qk.rs` / `quad_qk.rs`：1D 基函数张量积 + Apply kernel ② 接入 `partial.rs` 的 `MatFreeOperator` trait ③ 与 dense 装配逐元素核验 Apply 一致性 |
| **验收** | P5 Hex Poisson PA Apply 与 CSRA·x 相对误差 < 1e-12；PCG-PA 求解与装配求解残差一致 |
| **规模** | L · `pa/` 约 500 行 |

## M1.12 显式层次基　〔M〕

| 字段 | 内容 |
|------|------|
| **基线** | `element/src/lagrange/` 多项式单形体基（Vandermonde 预计算）；无独立层次基模块 |
| **MFEM 对齐** | 层次基（`H1_Hierarchical` / `H1_Reader`），p-MG 多网格更友好 |
| **任务** | ① 新增 `element/src/hierarchical.rs`：Seg/Tri/Tet 层次基（Lobatto node 核多形式、Koornwinder-Dubiner 三角形） ② 注册到 `element/src/factory.rs` 的 Pk 创建路径 ③ 空间测试 p-MG 迭代数 |
| **验收** | 层次基 p-MG（P1→P2→P3）迭代数 < 等阶基；单元测试基函数正则性 |
| **规模** | M · `hierarchical.rs` ~350 行 |

## M1.13 保协调 AMR 闭包　〔M〕

| 字段 | 内容 |
|------|------|
| **基线** | `mesh/src/simplex.rs` 的 `refine_marked` 单遍过细化（注释自承），无闭包迭代 |
| **MFEM 对齐** | 多遍闭包算法（Rebalance + 闭包 loop），确保无悬挂边残留 |
| **任务** | ① 实现 `closure_refine`：按照从 1‑ring → 2‑ring 传播的闭包迭代（类似 MFEM NCMesh closure 算法） ② 集成到 `refine_marked` 的统一入口 ③ 随机标记集压力测试 |
| **验收** | 任意四面体/三角形标记集闭包后无悬挂边残留；无过度细化（与最小闭包解 1:1）；单元测试覆盖 100 轮随机标记 |
| **规模** | M · `simplex.rs` ~200 行 |

---

# Phase 2 — 结构/可扩展性对齐

Phase 1 完成后启动。

## M2.1 并行 AMG 跨进程粗化　〔XL〕

| **基线** | `crates/parallel/src/par_amg.rs`（1591 行）本地聚合不跨进程边界；`fem-amg RS`/`SA` 串行版在 `linlvo` |
| **MFEM 对齐** | BoomerAMG 风格的并行 RS 粗化 + 跨进程插值 |
| **任务** | ① 在 `linlvo` 或 `fem-parallel` 实现并行强连接矩阵（跨 MPI C/F 分划传播） ② 并行 RS 粗化：跨进程 strong coupling 判定、全局 C/F 标记 ③ 并行插值（Dirichlet / extended+I） ④ 全局 Galerkin triple product `R·A·P`（alltoallv 稀疏矩阵乘积） |
| **验收** | 4 进程 2D anisotropic (ratio 1000) PCG 迭代数 ≈ 串行 AMG 的 1.1× 以内；弱扩展测试 100K DOF/rank 线性迭代数 |
| **规模** | XL · 需修改 `linlvo` 并行结构，~2000 行 |

## M2.2 NURBS 网格 + IGA 生产线　〔XL〕

| **基线** | `mesh/src/step_iges.rs` NURBS 抽取 stub（回退 facet）；`space/src/iga_fe_space.rs` p≤2；`assembly/src/iga*` 装配存在但 FESpace 桥受限 |
| **MFEM 对齐** | 完整 NURBS 网格（STEP → NurbsPatch → NurbsMesh）+ 任意阶 IGA 装配 |
| **任务** | ① 补 `step_iges.rs` B 样条曲面/实体抽取（引用 `knot vector`/`control point`） ② `NurbsPatch2D/3D` → `NurbsMesh` 桥接（单元分割 + DOF 分配） ③ IGA FESpace p≤2 解锁（M1.10 依赖） ④ 测试真实 STEP 文件 → IGA Poisson |
| **验收** | 读取标准 STEP 文件导入齿轮模型做 IGA Laplace；k-refinement 收敛阶正确 |
| **规模** | XL · 约 2500 行 |

## M2.3 并行 I/O collective MPI-IO　〔L〕

| **基线** | `parallel/src/par_hdf5.rs` per-rank 文件 + XDMF 聚合；无 collective write |
| **MFEM 对齐** | HDF5 MPIO file driver 做 collective parallel I/O |
| **任务** | ① 用 `H5Pset_fapl_mpio` 创建 collective write data transfer ② 并行网格序列化 `.pmesh` 格式 ③ 统一 checkpoint 路径：collective write 快于 per-rank 时优先使用 |
| **验收** | 100 进程 checkpoint 写耗时 < per-rank 方案；.pmesh 单文件可被串行读回 |
| **规模** | L · `par_hdf5.rs` ~500 行 |

## M2.4 GPU 装配完整化　〔XL〕

| **基线** | `linalg-gpu` CUDA（PTX）+ wgpu；GPU 装配仅 Poisson/质量/弹性 P1；PA sum-factorization 低阶 |
| **MFEM 对齐** | 完整设备端 partial assembly（任意阶）+ GPU AMS/ADS |
| **任务** | ① 设备端 PA：hex_qk/tet_qk 任意阶 Apply（借用 `linalg-gpu` vector pipeline） ② 增加 GPU AMS（H(curl) 辅助空间预条件） ③ GPU 直接求解器（`linalg-gpu/src/direct.rs` 现有基础扩展） |
| **验收** | GPU P3 Poisson PA Apply 与 CPU PA 逐分量一致；GPU AMS 预条件 Maxwell 腔体迭代数等于 CPU AMS |
| **规模** | XL · ~2000 行 |

## M2.5 linlvo 关键算法在树化　〔L〕（可选）

| **基线** | `vendor/linger`（原名 linlvo）提供 Krylov/AMG/直接求解器核心 |
| **MFEM 对齐** | 关键路径（AMG 粗化、插值、Galerkin 装配）从 `linlvo` 迁入 fem-amg 或 fem-parallel 树内 |
| **验收** | 迁入后 `cargo test` 不变，`git log` 可见 |
| **规模** | L · ~1000 行抽取 |

---

# Phase 3 — I/O / 生态对齐

Phase 1–2 完成后启动，优先级最低。

## M3.1 Exodus / CGNS　〔M〕

| **基线** | `io/src/cgns_exodus.rs` 返回描述性错误，主体未实现 |
| **任务** | ① 实现 `read_exodus_hdf5_impl`（exodus `/nodes/nodal_variables` + 单元连接表） ② 实现 `read_cgns`（base/solution/section 读取） |
| **验收** | 读入标准 exodus `.e` 文件（如 box\_mesh\_4x4x4），节点坐标/单元类型正确 |

## M3.2 高阶 VTK 输出（Bezier 抽取）　〔L〕

| **基线** | `io/src/vtk.rs` 写线性 `PointData`/`CellData`；无 HO |
| **MFEM 对齐** | 高阶插值 → Bezier 细分 + `VTM` 输出 |
| **任务** | ① 实现 `BezierExtractor`：将 P/Q 元均匀细分（tessellation），投影节点值 ② `vtk.rs` 新增 `write_vtu_higher_order` ③ `.vtm` 多块集合支持 |
| **验收** | P3 L2 field VTK 可视化与 ParaView Bezier 抽取值一致 |

## M3.3 GLVis 原生 HO 协议　〔S〕

| **基线** | `io/src/glvis.rs` 发送 legacy VTK over TCP |
| **MFEM 对齐** | MFEM native GLVis socket protocol（可区分元素类型 + 阶 + 解） |
| **任务** | ① 解析 GLVis 二进制协议格式 ② `glvis.rs` 新增 `send_native_solution` |
| **验收** | 流式 P3 解到 GLVis 显示正确 |

## M3.4 Sidre / Conduit　〔L〕（可选）

| **基线** | 无 |
| **任务** | 评估 `conduit-rs` 或 `sidre-rs` FFI；增加 `blueprint::mesh` 输出 |

## M3.5 PUMI 集成　〔L〕（可选）

| **基线** | 无 |
| **任务** | 评估 `pumi-rs` FFI；适配 APF mesh → fem-rs mesh 桥接 |

---

# 推进路线图

```
Phase 1  (算法深度)     ◄──── 本阶段优先
│  M1.1(MINRES/GCR)
│  M1.3(ABM/CN)
│  M1.5(BC消元+Robin)
│  M1.6(分片矩阵系数)
│  M1.7(BR1)
│  │
│  ├ M1.2(LBFGS/信赖域)
│  ├ M1.4(LOR真实)
│  │
│  ├ M1.8(ARPACK/FEAST) ─── 可与 M1.2/1.4 并行
│  ├ M1.9(VEM高阶)      ─── 可与 M1.8 并行
│  ├ M1.10(IGA高阶桥接) ─── 可与 M1.8/1.9 并行
│  ├ M1.11(HO sum-fact) ─── 依赖 M1.1（需要 Krylov）
│  ├ M1.12(层次基)      ─── 独立
│  └ M1.13(AMR闭包)     ─── 独立
│
Phase 2  (结构/可扩展性)
│  M2.1(并行AMG跨进程粗化)  ◄──── 最大收益
│  M2.2(NURBS网格)          ◄──── M1.10 向上依赖
│  M2.3(并行I/O)
│  M2.4(GPU装配)
│  M2.5(linlvo在树化)
│
Phase 3  (I/O/生态)
   M3.1–M3.5
```

**推荐并行调度策略：**
1. 抢占 M1.1/3/5/6/7（S 级共 5 项）——低技术风险，快速建立信心，2–3 个工作日内可并行交付。
2. 资源分叉：一组做 M1.2/1.4/1.8/1.10/1.12（M 级，5 项并行独立），另一组做 M1.9/1.11/1.13（L 级，3 项并行独立）。
3. Phase 1 全部交付后开始 Phase 2 优先项 M2.1（XL 级、最大收益项）。

---

*文档历史：0704 初版，基于代码基线（所有 crate `crates/*/src/`）与 MFEM 4.7 功能对照。*
