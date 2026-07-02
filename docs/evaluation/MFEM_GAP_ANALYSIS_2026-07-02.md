# fem-rs vs MFEM 差距评估与改进计划

- 日期：2026-07-02
- 参考 MFEM 版本：v4.7 (https://mfem.org, 编号示例 ex0–ex36 + pex1–pex28 + miniapps)
- 评估者：MiMoCode agent，基于当前工作树代码 + MFEM 官方文档比对
- 当前项目提交：`b4e98d9 Phase 3E-3N + Phase 4`
- 前置基线：`docs/baselines/mfem_parity.md` (2026-06-30 性能基线)
- 本文件位置：`docs/evaluation/MFEM_GAP_ANALYSIS_2026-07-02.md`

> 本文件是长期性文档，不要删除。后续对齐进度以在此文件追加"进度记录"段的方式更新，或新增日期后缀的姐妹文件 `MFEM_GAP_ANALYSIS_YYYY-MM-DD.md`。

---

## 0. 方法论

评估基于三个来源：

1. 本项目实际代码：遍历 `crates/*/src/**/*.rs`，统计模块、公开 API、示例入口
2. MFEM 官方公开能力：
   - 编号示例 `examples/ex0.cpp` – `examples/ex36.cpp`
   - 并行示例 `examples/ex1p.cpp` – `examples/ex36p.cpp`
   - Miniapps：electromagnetics / fluids / performance / solvers / shifted / meshing / tools / nurbs / gslib / parelag / mtop / adjoint / tmop / autodiff / dpg / common / hdiv-linear-solver / hooke / mhd / multidomain / navier / plasma / spde
3. README + baselines：`README.md`、`docs/baselines/mfem_parity.md`、`HANDOVER_*.md`

评估口径：功能覆盖 + 示例对齐 + 公开 API 兼容，不做性能基准或生态成熟度判断。

---

## 1. 项目当前能力概览（代码盘点）

### 1.1 fem-mesh (18 模块)

- `simplex`：`SimplexMesh<D>` (D=1/2/3)，支持 Line2/3、Tri3/6、Quad4/8/9、Tet4/10、Hex8/20/27、Prism6/15、Pyramid5/13
- `amr`：Tet/Hex/Prism/Pyramid uniform + 非一致 (NC) + 各向异性细化，213 tests
- `curved`：P2 几何曲面网格；`moving_mesh` ALE
- `tmop`：目标矩阵优化 baseline（README 明确"仅基线"，完整 target-matrix 待完成）
- `cad`、`step_iges`：CAD 边界读取（占位）
- `cut_cell`：切割单元
- `hp_amr`：hp 决策（smoothness indicator、HpAction enum）
- `poly_mesh`：多边形网格（供 VEM）
- `dec`：离散外微分（存在，深度未验证）
- `submesh`、`boundary`、`point_locator`、`transformation`、`lor`

### 1.2 fem-element (15 模块，58 个 .rs 文件)

**Lagrange 家族**（阶数覆盖远超 P1–P3）：
- Seg：**P1–P6**
- Tri：**P1–P10**（十阶）
- Tet：**P1–P6**
- Quad：`QuadQ1/Q2/Q3/Q4` 张量积 + `QuadP1–P4` serendipity 派生
- Hex：`HexQ1/Q2/Q3` 张量积
- Prism：`PrismPk` 泛型阶
- Pyramid：`PyramidPk` 泛型阶
- 工厂 `ref_elem(ElemType, k)` + `vec_ref_elem(VecFamily, k)`

**Nédélec H(curl)**：
- Tri / Quad / Tet / Hex：**ND1、ND2、NDk（任意阶）** 全套
- Prism：ND1 + NDk 骨架（未验证）
- Pyramid：PyraND1 + PyraNDk **骨架但底层 `EDGES`/`TRI_FACES`/`tri_face_dof` 未激活**（dead_code warning）

**Raviart-Thomas H(div)**：
- Tri：RT0 / RT1 / RT2 / RTk 全套
- Tet：RT0 / RT1 / RT2 / RTk 全套
- Quad：RT0 / RT1 / RTk
- Hex：RT0 / RT1 / RTk
- Prism / Pyramid：RT0 + RTk 骨架（未验证）

**Brezzi-Douglas-Marini H(div)**：
- Tri / Tet / Quad：BDMk
- Hex：BDMk 存在，**内部 `gauss_3d` 3D 装配路径未启用**（unused warning）

**其它单元**：
- Bernstein Pk（Seg/Tri/Tet/Quad/Hex）
- Serendipity Pk（Quad/Hex）
- Crouzeix-Raviart：CrTri1/2、CrTet1/2、`CrouzeixRaviartVec1`（`cr1_tet_grad` 参数 warning，逻辑可能残缺）
- 非协调元：`QuadQ1Rot` / `QuadQ1RotVec`
- IGA：`BsplineBasis` + `NurbsBasis` + Böhm 节点插入（`iga/mod.rs`）
- 旧 NURBS：`BSplineBasis1D` / `NurbsPatch2D/3D` / `NurbsMesh2D/3D` / `greville_abscissae`（**与新 `iga/mod.rs` 潜在重复**）
- `basis_cache` 基函数缓存
- `tri6_geom` 二阶三角形几何

### 1.3 fem-space (19 模块，20 个 .rs 文件)

**已导出的空间**：
- 标量：`H1Space`、`L2Space`、`CRSpace`
- 向量：`VectorH1Space`、`VectorCRSpace`
- 矢量场：`HCurlSpace`、`RestrictedHCurlSpace`、`HDivSpace`
- Trace：`H1TraceSpace`、`HCurlTraceSpace`、`HDivTraceSpace`
- IGA：`IgaFESpace1D`/`2D`、`IgaSinglePatchMesh1D/2D`、`IgaMultiPatchMesh2D`
- 组合：`BlockFESpace`（多场耦合）、`skeleton`（DG face）、`vem`（VEM）
- 变阶：`p_refine`（Tet4→Tet10、Tri6→Tri10、Quad4→Quad9、Hex8→Hex20、Hex20→Hex27 等）

**约束系统 `constraints.rs`**：
- H¹ hanging：`apply_hanging_constraints`（PᵀKP 静力凝聚）
- 面 hanging：`apply_hanging_face_constraints`（Tet4 三角面）
- P2 prolongation：`prolongate_p2_hanging`
- Vector-FE 边界：`boundary_dofs_hcurl` / `boundary_dofs_hdiv`
- 周期性：`identify_periodic_dof_pairs` / `apply_periodic`

**已知空间层缺口**：
1. **Quad face 上的 HCurl/HDiv NC 约束未完成**（`HANDOVER_2B_VECTOR_FE_NC.md`）—— Tet4 三角面已通，Hex/Prism/Pyramid 四边形悬挂面走不通
2. `HCurlTraceSpace` / `HDivTraceSpace` 结构体 `n_bfaces` 字段声明但从未读取（dead_code warning）→ trace space API 建立但无实际调用点
3. `RestrictedHCurlSpace` 存在但公开示例侧未见使用

### 1.4 fem-assembly (80 模块，项目最大 crate)

- 标准积分器：diffusion、mass、source、elasticity、convection、Neumann、Robin
- DG：SIP、BR2、LDG、advection、Euler 2D/3D、CDR、curved、limiters
- 混合：MixedAssembler、DPG (2D/3D framework)、DPG-Stokes/Maxwell/Elasticity
- HDG：framework、Stokes、Elasticity、Maxwell
- WG：Poisson、Stokes、Maxwell
- VEM：Poisson（仅一阶）
- CutFEM + ghost-penalty
- XFEM：level-set、integrators、crack propagation
- 非线性：nonlinear、hyperelasticity、plasticity (J2/DP)、crystal plasticity、damage、phasefield + pf_solver、cahn_allen
- 流体：navier_stokes（Oseen/Picard/ALE）
- 耦合：FSI、thermoelastic
- 接触：contact、mortar、nitsche、self-contact
- Complex（复数系统）、Partial assembly（matrix-free）
- IGA：iga_assembler、iga_trim
- 后处理：gradient recovery、Kelly、error estimator、DWR
- Adjoint PDE、transfer（含守恒 L² 投影）、静力凝聚

### 1.5 fem-solver (30+ 模块)

- 迭代法：CG/PCG/GMRES/FGMRES/BiCGSTAB/IDR(s)/TFQMR
- 预条件：Jacobi、ILU0/ILUk/ILUT、AMS (H(curl))、ADS (H(div))
- 直接法：SparseLU、Cholesky、LDLᵀ、MUMPS/MKL 兼容 API
- 特征值：LOBPCG（含约束/预条件）、KrylovSchur
- ODE：RK 系列、IMEX、BDF、辛积分、DAE
- 多物理：`CoupledNewtonSolver`、multiphysics_sync、multiphysics_templates
- 多率：multirate；混合精度：mixed_precision
- 减阶模型：rom；SDC；p-multigrid；LOR
- Adjoint、event detection
- HypreBoomerAMG 兼容 API
- Active-set 求解器（3D 无摩擦接触）
- GPU：CG/GMRES（wgpu 后端）

### 1.6 fem-parallel

Comm/Backend 抽象 (Thread/MPI/WASM)，METIS 分区、SFC、RCB、ParCSR、ParAMG、ParAssembler、GhostExchange、分布式 AMR、并行 HDF5 checkpoint、RAS 预条件。

### 1.7 fem-io

GMSH v2/v4、MFEM `.mesh` v1.0/v1.2（含二阶单元）、Netgen、Abaqus、STL、OBJ、VTK/VTU、PVTU、PVD、Matrix Market、HDF5、XDMF、GLVis、CGNS/Exodus（骨架）。

### 1.8 其他 crate

- `fem-amg`：SA-AMG、RS-AMG、Chebyshev、V/W/F cycle
- `fem-wasm`：浏览器 Poisson、多 Worker 并行
- `fem-ceed`：libCEED-style QFunctions（feature-gated）
- `fem-python`：PyO3 绑定
- `fem-stochastic`：随机场（`mfem_mc_random_field`）

### 1.9 示例：79 个可执行 example

编号 `mfem_ex0` – `mfem_ex53`（非连续）+ `mfem_pex1`–`pex5` + `tmop`/`joule`/`tesla`/`volta`/`maxwell`/`phasefield`/`mc_random_field`/`vec_ref_elem` 等。

---

## 2. MFEM 官方能力 vs fem-rs 对齐结果

### 2.1 核心编号示例 ex0-ex36

| MFEM | 主题 | fem-rs 对应 | 状态 |
|---|---|---|---|
| ex0 | Poisson 最小示例 | `mfem_ex0_mesh_intro` + `ex1_poisson` | 有 |
| ex1 | Poisson H1 | `mfem_ex1_poisson` | 有 |
| ex2 | 线性弹性 | `mfem_ex2_elasticity` | 有 |
| ex3 | Maxwell H(curl) | `mfem_ex3_maxwell_cavity` | 有 |
| ex4 | Darcy H(div) | `mfem_ex4_darcy` | 有 |
| ex5 | 混合 Darcy 鞍点 | `mfem_ex5_mixed_darcy` | 有 |
| ex6 | AMR Poisson | `mfem_ex15_dg_amr` 覆盖 | 编号错位但覆盖 |
| ex7 | 曲面 Poisson（球面上） | `mfem_ex29_curved_poisson` 是曲域非曲面 | 主题错位 |
| ex8 | Hybridization | `mfem_ex8_hybridization` | 有 |
| ex9 | DG advection | `mfem_ex9_dg_advection` | 有 |
| ex10 | 非线性弹性动力（hyperelastic ODE） | 缺失（`ex10_heat/wave` 主题不同） | 缺失 |
| ex11 | Laplacian 特征值 | `mfem_ex13_laplacian_eigen` | 编号错位但覆盖 |
| ex12 | 线性弹性特征值 | 缺失 | 缺失 |
| ex13 | Maxwell 特征值 | `mfem_ex13_eigenvalue` | 有 |
| ex14 | DG Poisson | `ex14_dc_current` 主题错位 | 主题错位 |
| ex15 | 动态 AMR | `mfem_ex15_*` 三个 | 有 |
| ex16 | 时间相关非线性热 | `mfem_ex16_nonlinear_heat` | 有 |
| ex17 | DG 弹性 | `mfem_ex17_dg_elasticity` | 有 |
| ex18 | Euler DG | `mfem_ex18_euler` | 有 |
| ex19 | 不可压超弹性动力学 | `ex19_navier_stokes` 主题错位 | 主题错位 |
| ex20 | 辛积分器 | `ex20_wgpu_poisson` 是 GPU 演示 | 主题错位 |
| ex21 | AMR 弹性（近不可压） | 缺失 | 缺失 |
| ex22 | 复数 Helmholtz | `mfem_ex22_complex_helmholtz` | 有 |
| ex23 | 波方程 | `mfem_ex23_wave_equation` | 有 |
| ex24 | 离散算子 grad/curl/div | `mfem_ex24_discrete_ops` | 有 |
| ex25 | PML | `mfem_ex25_pml_helmholtz` | 有 |
| ex26 | 几何多重网格 | `mfem_ex26_geom_mg` | 有 |
| ex27 | Robin BC | `mfem_ex27_robin_bc` | 有 |
| ex28 | 滑动接触弹性 | `mfem_ex28_sliding_elasticity` | 有 |
| ex29 | 曲域 Poisson | `mfem_ex29_curved_poisson` | 有 |
| ex30 | AMR 残差估计器 | 估计器已实现无独立示例 | 部分 |
| ex31 | 各向异性 Maxwell | `mfem_ex31_anisotropic_maxwell` | 有 |
| ex32 | 阻抗 Maxwell | `mfem_ex32_impedance_maxwell` | 有 |
| ex33 | 分数阶 Laplacian | `mfem_ex33_fractional_laplacian` | 有 |
| ex34 | 吸收 Maxwell | `mfem_ex34_absorbing_maxwell` | 有 |
| ex35 | 多领域 Poisson | 缺失 | 缺失 |
| ex36 | 障碍问题 | `mfem_ex36_obstacle` | 有 |

**核心示例覆盖率：约 29/37 (78%)。** 编号错位主题实际缺失的 7 例：ex7、ex10、ex12、ex14、ex19、ex20、ex21、ex35。项目额外覆盖了 ex37–ex53 若干高级主题（拓扑优化、浸没边界、TMOP、模板 FSI 等，MFEM 本身也有 ex37–ex40 但主题不同）。

---

### 2.2 并行 pex 示例

| MFEM pex | 覆盖 |
|---|---|
| ex1p 并行 Poisson | `mfem_pex1_parallel_poisson` |
| ex2p 并行弹性 | 缺失 |
| ex3p 并行 Maxwell | `mfem_pex3_maxwell_cavity` |
| ex4p 并行 Darcy | `mfem_pex2_mixed_darcy` + `mfem_pex5_hdiv_darcy` |
| ex5p 并行混合 Darcy | 已覆盖 |
| ex6p 并行 AMR | 缺失 |
| ex9p–ex36p 各类并行 | 大部分缺失，仅热方程 `pex4_parallel_heat` |

**并行示例覆盖率：约 5/36 (14%)。** 主要缺口在弹性并行、DG 并行、AMR 并行、非线性并行、时间相关并行。

---

### 2.3 Miniapps 对齐

| MFEM miniapp | 主题 | fem-rs | 状态 |
|---|---|---|---|
| electromagnetics/volta | 静电 | `mfem_volta` | 有 |
| electromagnetics/tesla | 磁静力 | `mfem_tesla` | 有 |
| electromagnetics/joule | 焦耳热 | `mfem_joule` | 有 |
| electromagnetics/maxwell | 时域 Maxwell | `mfem_maxwell` | 有 |
| fluids/navier | 时间相关 NS | 稳态 NS 已有，时间相关 NS miniapp 未对齐 | 部分 |
| meshing/mesh-optimizer (TMOP) | 网格质量优化 | `mfem_tmop_mesh_quality` + `tmop_hex8_optimise`，README 承认仅 baseline | 部分 |
| meshing/mobius-strip 等 | 拓扑网格构造 | 缺失 | 缺失 |
| performance/ex1 | 高阶性能测试 | `docs/baselines/mfem_parity.md` 有基础表 | 部分 |
| nurbs/nurbs_ex1..11 | NURBS 各类问题 | 仅 `mfem_ex_iga_poisson_1d/2d_patch` 2 个 | 严重缺口 |
| solvers/plor-solvers | Low-Order Refined | `solver/src/lor.rs` 有实现无示例 | 部分 |
| shifted/diffusion, distance | Shifted BC / 距离函数 | 缺失 | 缺失 |
| dpg/* | DPG 求解器 | framework 存在，无 example 展示 | 部分 |
| adjoint/* | 伴随/优化 | `adjoint_pde` + `topology_optimization` | 有 |
| autodiff/* | 自动微分 | 缺失 | 缺失 |
| plasma/* | 等离子体六方程 | 缺失 | 缺失 |
| mhd/* | 磁流体 | 缺失 | 缺失 |
| hooke | 超弹性 | `HyperelasticityForm` 存在无独立示例 | 部分 |
| gslib/* | 网格间数据传输 | `transfer.rs` 存在，无 GSLIB 风格示例 | 部分 |
| multidomain | 多域耦合 | `multiphysics_templates` 无独立示例 | 部分 |
| spde | 随机 PDE | `fem-stochastic` + `mfem_mc_random_field` | 有 |
| parelag | 元素凝聚 AMG | 通用 AMG 有，element-agglomeration 未实现 | 部分 |
| hdiv-linear-solver | H(div) 专用 | ADS 存在 | 有 |
| tools/{convert-dc,load-dc,display-basis} | 数据转换 | 缺失 | 部分工具 |

**Miniapp 覆盖率**：电磁 miniapp 完整 (4/4)；fluids/meshing/nurbs/shifted/dpg/autodiff/plasma/mhd/tools 大部分缺失。

---

### 2.4 MFEM 有但 fem-rs 缺失的能力

| 能力 | MFEM 位置 | fem-rs 现状 |
|---|---|---|
| 完整 hp-adaptive 求解演示 | ex15 hp | `hp_amr` 模块有，示例仅 h-adaptivity |
| NURBS AMR / knot insertion 集成到求解流水线 | miniapps/nurbs | `insert_knot` 存在，未接入 AMR 流程 |
| GSLIB 风格网格点采样传输 | miniapps/gslib | `transfer.rs` 仅 L² 投影，无点采样 API |
| Shifted Boundary Method | miniapps/shifted | 缺失（CutFEM 有，SBM 无） |
| Algoim-quality cut quadrature | miniapps/shifted | `xfem_level_set` 存在但非高精度 algoim |
| 完整 tensor product PA GPU | libCEED 集成 | `partial.rs` + `fem-ceed`（gated），未验证完整 tensor product GPU 优化 |
| Kernel Fusion Assembly GPU | RAJA/UMPIRE 后端 | 仅 CG/GMRES GPU（wgpu），装配 GPU 缺失 |
| Hyperelastic 动力学 ODE (ex10) | ex10 | `HyperelasticityForm` 有静态，动力学 ODE 无示例 |
| Symplectic 积分（Hamiltonian ex20） | ex20 | `ode.rs` 有 symplectic 集成，无 ex20 主题演示 |
| Autodiff 材料切线 | miniapps/autodiff | 手写切线（例如 J2/DP），无 AD |
| MHD / 等离子体 | miniapps/mhd, plasma | 缺失 |
| BlockNonlinearForm 通用抽象 | mfem::BlockNonlinearForm | 有 `NonlinearForm` + 手动 block 组合，无统一 BlockNonlinearForm |
| MFEM `.gf` GridFunction 读写 | 内建 | 缺失（`grid_function.rs` 是内部结构） |
| MFEM `.mesh` 写出 | 内建 | 仅读取，`gmsh_writer` 只有 GMSH |
| GLVis 双向 socket 交互 | 内建 | `glvis.rs` 有基础，未验证双向 |
| DataCollection / ParaView time series | mfem::DataCollection | `pvd.rs` 有 PVD，无完整 DataCollection 抽象 |
| MFEM-compatible RestrictedFiniteElementSpace | mfem::PetscBCHandler | `restricted_hcurl.rs` 有基础 |
| HYPRE 完整绑定（ParCSR/BoomerAMG/AMS/ADS 通过 FFI） | libmfem-hypre.a | 有兼容 API `HypreBoomerAMG`，未 FFI |
| MUMPS/PETSc 完整绑定 | libmfem-mumps.a, libmfem-petsc.a | 只有兼容 API 存根 |
| SUNDIALS ODE/DAE 完整绑定 | libmfem-sundials.a | 无 |
| GinkgoWrappers | mfem::ginkgo | 无 |
| Occa/RAJA/CUDA 计算后端 | Backend switching | 只有 wgpu，无 CUDA/HIP/OCCA |
| **Nédélec pyramid 完整实现** | mfem::fem 内建 | `nedelec/pyramid.rs` PyraND1/PyraNDk 骨架存在但底层 `EDGES`/`TRI_FACES`/`tri_face_dof` 未激活（dead_code warning），实际 DOF 装配未通 |
| **3D Hex BDMk 完整装配** | mfem::HexBrezziDouglasMariniFECollection | `brezzi_douglas_marini/hex_bdmk.rs` 结构存在但 `gauss_3d` 3D 装配路径未启用（unused warning），Hex BDM 组装无法走通 |
| **Prism/Pyramid RT 高阶验证** | 内建 | `raviart_thomas/{prism,pyramid}.rs` 有 RT0 + RTk 骨架但 `n_tri_moments`/`n_quad_moments` 未使用，高阶自由度定义未完成 |
| **Crouzeix-Raviart Tet 3D 梯度** | 内建 | `crouzeix_raviart.rs::cr1_tet_grad` 参数 `xi` 未使用（warning），3D CR 梯度实现可能残缺 |
| **NURBS 双份实现清理** | 单一 NURBS 路径 | 存在 `element/src/nurbs.rs`（老）与 `element/src/iga/mod.rs`（新 Böhm knot insertion）两套 API，未合并 |

---

### 2.5 fem-rs 有但 MFEM 无/弱的能力（超越点）

| 能力 | fem-rs 位置 | MFEM 现状 |
|---|---|---|
| Rust 内存安全 + `Send`/`Sync` 保证 | 全项目 | C++ 手动管理，MPI/OpenMP 混合易出错 |
| WASM 浏览器求解 | `fem-wasm` (Poisson + 多 Worker) | 无 |
| Native NURBS 基（B-spline + NURBS 完整算子） | `element/iga/mod.rs` | Bézier extraction，非纯 NURBS |
| Weak Galerkin (WG) 完整族 (Poisson/Stokes/Maxwell) | `wg_*.rs` | 无 |
| Virtual Element Method (VEM) | `vem_poisson.rs` + `poly_mesh.rs` + `space/vem.rs` | 无 |
| DPG framework (Poisson/Stokes/Maxwell/Elasticity 2D+3D) | `dpg_*.rs` 6 模块 | MFEM 有 dpg miniapp 但不作为标准示例 |
| CutFEM ghost-penalty MMS 收敛测试 | `cutfem.rs` | 无（有 shifted 但不是 CutFEM） |
| 相场断裂 Miehe split + AT1/AT2 | `phasefield.rs` + `pf_solver.rs` | miniapp 有基础，无完整 Miehe |
| 晶体塑性 12 FCC 滑移系 | `crystal_plasticity.rs` | 无 |
| Cahn-Hilliard / Allen-Cahn IMEX | `cahn_allen.rs` | 无 |
| XFEM 裂纹扩展（最大环向应力） | `xfem_crack.rs` | 有 shifted 但非 XFEM 裂纹 |
| Coupled multi-physics Newton (通用 3 场+) | `solver/multiphysics.rs` | mfem 的 BlockNonlinearForm 需要手动 assemble Jacobian |
| Active-set 3D 无摩擦接触 | `solver/active_set.rs` | 有基础 |
| 混合精度 FGMRES | `solver/mixed_precision.rs` | 无标准示例 |
| ROM (POD/DEIM) | `solver/rom.rs` | 无 |
| RAS Schwarz + Schur 完整并行 | `parallel/par_ras.rs` | 通过 HYPRE 提供 |
| Rust-native METIS (fem-rmetis) | `crates/rmetis` | 依赖外部 METIS |
| Comm 抽象 (Thread/MPI/WASM 三后端) | `parallel/comm.rs` + `launcher/*` | 仅 MPI |
| SDC (谱推迟修正) | `solver/sdc.rs` | 无 |
| SDIRK/IMEX/BDF/DAE/multirate 完整 ODE 栈 | `solver/{ode,bdf,dae,multirate}.rs` | SDIRK + IMEX 主要 |
| BlockGMRES + BlockDiag/BlockTri/Schur 预条件族 | `solver/{block,block_gmres,block_operator}.rs` | 有类似能力 |
| **高阶 Lagrange 元素覆盖** | Tri P1–**P10**、Tet P1–P6、Seg P1–P6 (`element/src/lagrange/`) | MFEM 元素库支持任意阶，但常见示例集中于 P1–P3；fem-rs 直接把 P10 三角形作为标准类型导出 |
| **NDk/RTk 任意阶泛型元素** | Nedelec `TriNDk/QuadNDk/TetNDk/HexNDk`、Raviart-Thomas `TriRTk/TetRTk/QuadRTk/HexRTk` | MFEM 通过 `FiniteElementCollection` 提供，fem-rs 通过类型泛型直接编译期展开 |
| **Bernstein + Serendipity 双基** | `bernstein.rs` + `serendipity.rs`（Seg/Tri/Tet/Quad/Hex 全套） | MFEM 有 Bernstein，但 Serendipity 系统性覆盖较弱 |

---

## 3. 差距总结（分优先级）

### P0 - 阻塞性（影响 MFEM 兼容性口径）

1. **VTK 编译错误**：`crates/io/src/vtk.rs:37` 中 `ElementType` 匹配未覆盖 `Hex27` 和 `Polygon`，导致 `cargo check --examples` 直接失败。**必须先修**。
2. **HANDOVER_2B 阻塞**：`build_hcurl_hanging_constraints` 只处理 Tet4 三角面，Hex/Prism/Pyramid 四边形面 HCurl 悬挂约束缺失。这是 3D H(curl) AMR 的完整性障碍。
3. **`ElementType::Polygon` 与 `Hex27` 分支**：多个 match 表达式未覆盖，编译警告 → 未来可能崩。
4. **MFEM 编号错位的 7 例**：ex10、ex12、ex14、ex19、ex20、ex21、ex35 主题错位或缺失，README 中标注为 "MFEM 一对一对齐" 但实际不是。

### P1 - 主题严重缺口（承诺"MFEM parity"但欠账）

5. **并行示例只有 5 个**：pex 覆盖率 14%，与 README 声明的 "parallel examples" 存在显著缺口。
6. **NURBS/IGA 只有 2 个示例**：MFEM nurbs miniapp 有 11 个变种，我们只覆盖 Poisson 1D/2D。
7. **DPG/HDG/WG/VEM 均无 example**：framework 存在，但用户无法通过 `cargo run --example` 演示，导致 README "已完成" 存在解释性偏差。
8. **本轮新增模块（塑性、晶体塑性、Cahn-Hilliard、Allen-Cahn、XFEM 裂纹、多物理耦合 Newton）无 example**：与前节讨论一致。
9. **TMOP baseline only**：README 已声明是 baseline，但 miniapp 完整 target-matrix 未实现。
10. **元素库高阶单元未闭环**（新增，来自 §1.2 代码盘点）：
    - Nédélec pyramid：`nedelec/pyramid.rs` 的 `EDGES`/`TRI_FACES`/`tri_face_dof` 是 dead_code，PyraND1/PyraNDk 骨架未打通实际 DOF 装配 → 3D H(curl) 混合网格上锥体单元不可用
    - BDM Hex 3D：`brezzi_douglas_marini/hex_bdmk.rs::gauss_3d` 定义但从未调用，HexBDMk 3D 装配路径断链 → 高阶 H(div) Hex 网格 BDM 组装无法走通
    - Prism/Pyramid RT 高阶：`n_tri_moments` / `n_quad_moments` unused → RTk 在这两类单元上高阶自由度定义未完成
    - Crouzeix-Raviart Tet：`cr1_tet_grad` 的 `xi` 参数未使用 → 3D CR1 梯度可能返回错误值，需数值验证

### P2 - 生态/基础设施

11. **libCEED 完整 tensor product PA GPU 未验证**：`fem-ceed` 存在但深度依赖 workspace-pinned reed crate，QFunction path 覆盖率不明。
12. **完整 GPU 装配缺失**：只有 CG/GMRES GPU (wgpu)，assembly kernel GPU 缺失。
13. **`.gf` GridFunction 读写 / `.mesh` 写出**：MFEM 生态兼容性欠账，用户无法在 MFEM ↔ fem-rs 间简单传输结果。
14. **GLVis 双向 socket 交互未验证**。
15. **HYPRE/PETSc/MUMPS/SUNDIALS 全是 stub API**，不做真正 FFI。README 已声明 "compatibility contract 非 FFI"，但可视化选项/文档需要更明确。
16. **NURBS 双份实现**：`element/src/nurbs.rs` 与 `element/src/iga/mod.rs` 并存，需清理为单一 API。

### P3 - 高级研究主题

17. Shifted Boundary Method
18. Autodiff 材料模型
19. MHD / Plasma
20. Algoim 高精度切割积分
21. 完整多域耦合的 `submesh` API
22. HYPRE-parallel BoomerAMG 真 FFI（针对亿级 DOF）

---

## 4. 改进与提升计划

### 4.1 计划总览（4 阶段 12 里程碑）

| 阶段 | 时长（估算） | 主目标 |
|---|---|---|
| **P0 阻塞修复** | 1 周 | VTK 编译修复 + 4 个错位 example 归位 + HANDOVER_2B 收尾 |
| **P1 示例补齐** | 4-6 周 | pex 并行示例补齐 + NURBS/DPG/HDG/WG/VEM/新模块示例 |
| **P2 生态兼容** | 6-10 周 | MFEM `.gf`/`.mesh` 双向 I/O + GLVis 完整 + 完整 tensor product PA GPU |
| **P3 研究前沿** | 长期 | SBM/AD/MHD/Algoim/亿级 HYPRE FFI |

---

### 4.2 P0 阻塞修复（M0 里程碑）

#### M0.1 修复 VTK 匹配

- **文件**：`crates/io/src/vtk.rs:37`
- **动作**：给 `ElementType::Hex27` 和 `ElementType::Polygon` 增加真正的 VTK cell type 映射（Hex27 → VTK_TRIQUADRATIC_HEXAHEDRON=29；Polygon → VTK_POLYGON=7）
- **验收**：`cargo check --examples` 完成，无 E0004 错误

#### M0.2 修复 MFEM 编号错位

- **动作**：
  - `mfem_ex10_heat_equation.rs`、`mfem_ex10_wave_equation.rs`、`mfem_ex10_maxwell_time.rs` 三个 `ex10_*` 都不对齐 MFEM ex10，应重命名去掉 `mfem_ex10_` 前缀或改名 `mfem_ex10_hyperelastic_time` 并实现真的 hyperelastic 动力学
  - 同理修 `ex14_dc_current` → `dc_current.rs`；`ex19_navier_stokes` → 拆为 `mfem_ex19_hyperelastic_dyn`（新增）+ `navier_stokes_kovasznay.rs`
  - `ex20_wgpu_poisson` → `wgpu_poisson.rs` 或 `gpu_poisson.rs`；`mfem_ex20_symplectic` 新增（对齐 MFEM ex20）
- **验收**：所有以 `mfem_ex<N>_` 前缀命名的示例，其 PDE + method + BC 与 MFEM `examples/exN.cpp` 一致；重命名后 `cargo build --examples` 通过

#### M0.3 完成 HANDOVER_2B

- 参见 `HANDOVER_2B_VECTOR_FE_NC.md`
- **验收**：Hex8/Prism6/Pyramid5 Quad face 上的 HCurl/HDiv NC 约束正常传播，`cargo test -p fem-space` 增加 quad-face NC 测试并通过

#### M0.4 补齐 `ElementType` 分支

- **文件**：`crates/mesh/src/element_type.rs`，各 match 站点
- **验收**：`cargo clippy --workspace -- -D warnings` 无 unreachable pattern / non-exhaustive

#### M0.5 元素库高阶单元闭环（新增里程碑）

- **动作**：
  - `nedelec/pyramid.rs`：接入 `EDGES` / `TRI_FACES` / `tri_face_dof`，让 PyraND1 通过基本 DOF 一致性测试（partition of unity、tangential trace 匹配）；PyraNDk 高阶留作 M0.5-followup
  - `brezzi_douglas_marini/hex_bdmk.rs`：启用 `gauss_3d`，接入 HexBDMk 3D 装配，跑一个 3D H(div) mass 矩阵 MMS 验证
  - `raviart_thomas/{prism,pyramid}.rs`：用 `n_tri_moments`/`n_quad_moments` 完成 RTk 高阶自由度定义，加 partition of unity 测试
  - `crouzeix_raviart.rs::cr1_tet_grad`：修 `xi` 参数使用逻辑，跑 3D CR1 Poisson MMS 验证收敛率
- **验收**：
  - 上述四处 `cargo clippy` unused_variables / dead_code warning 全消
  - 新增 `#[test]` 覆盖每个修复点，含 MMS 收敛率检查
  - `docs/evaluation/element_library_status.md` 补充每种单元类型的"骨架 / 已完成 / 已验证"三档矩阵

---

### 4.3 P1 示例补齐（M1-M5 里程碑）

#### M1.1 补齐缺失的核心 ex

| 新示例 | 目标 MFEM 对应 |
|---|---|
| `mfem_ex10_hyperelastic_dyn.rs` | ex10 |
| `mfem_ex12_elastic_eigen.rs` | ex12 |
| `mfem_ex14_dg_poisson.rs` | ex14 |
| `mfem_ex19_hyperelastic_dyn_incomp.rs` | ex19 |
| `mfem_ex20_symplectic.rs` | ex20 |
| `mfem_ex21_amr_elasticity.rs` | ex21 |
| `mfem_ex35_multidomain.rs` | ex35 |

**验收**：每个示例
- 输入参数默认可运行 `cargo run --example <name>`
- 输出量与 MFEM 相应示例数量级一致（DOF 数、迭代次数、L² 误差）
- 添加 `#[test] fn ex_smoke_test()` 验证运行不 panic
- README 一对一对齐表更新

#### M1.2 并行 pex 补齐至 ≥ 20 个

- 参考 `mfem_pex1_parallel_poisson.rs` 结构（`ThreadLauncher` 或 `MpiLauncher`）
- 需要新增：pex2 弹性、pex6 AMR、pex7 曲面、pex9 DG advection、pex10 nonlinear heat、pex11 eigen、pex15 dynamic AMR、pex16 nonlinear heat、pex17 DG elasticity、pex18 Euler、pex22 complex Helmholtz、pex23 wave、pex25 PML、pex26 geom_mg、pex27 Robin、pex36 obstacle
- **验收**：`cargo test -p fem-examples --features "" -- --test-threads=1` 至少覆盖 20 个 pex；每个能在 4/8 rank 下线性可扩展至少到中等规模，弱扩展率在 `docs/baselines/` 有记录

#### M1.3 补齐新增模块示例（本轮遗漏）

| 新示例（不再带 `mfem_` 前缀） | 对应模块 |
|---|---|
| `plasticity_j2_bar.rs` | `plasticity` |
| `plasticity_dp_slope.rs` | `plasticity`（Drucker-Prager） |
| `crystal_plasticity_fcc.rs` | `crystal_plasticity` |
| `cahn_hilliard_spinodal.rs` | `cahn_allen` |
| `allen_cahn_evolution.rs` | `cahn_allen` |
| `xfem_crack_sen_tension.rs` | `xfem_crack` |
| `multiphysics_coupled_newton.rs` | `solver/multiphysics` |
| `wg_stokes_cavity.rs` | `wg_stokes` |
| `vem_poisson_polygonal.rs` | `vem_poisson` |
| `hdg_stokes_channel.rs` | `hdg_stokes` |
| `hdg_elasticity_beam.rs` | `hdg_elasticity` |
| `dpg_poisson_2d.rs` | `dpg_2d` |
| `dpg_stokes_2d.rs` | `dpg_stokes` |
| `contact_active_set_3d.rs` | `solver/active_set` |

**验收**：每个示例
- 有一个 MMS 或已知解析解验证收敛率（P1 O(h²)，P2 O(h³)）
- Smoke test 运行不 panic
- 输出结果（VTK 或 CSV）位于 `output/`
- README 添加"额外能力示例"表

#### M1.4 NURBS/IGA 示例补齐

| 新示例 | 内容 |
|---|---|
| `iga_ex_annulus_poisson.rs` | 圆环 NURBS 域上 Poisson |
| `iga_ex_plate_hole_elasticity.rs` | 带孔板 NURBS 弹性 |
| `iga_ex_knot_insertion_amr.rs` | knot insertion 驱动的 h-refinement |
| `iga_ex_bezier_extraction.rs` | Bézier extraction 与 Lagrange 桥接 |
| `iga_ex_shell.rs` | NURBS shell（若 shell 元素可行） |

**验收**：至少 5 个新 IGA 示例；`insert_knot` 在示例中被实际使用；能重现文献解（如 hole in plate 应力集中系数 3.0）

#### M1.5 补齐 miniapp 类演示

| 新示例 | 对齐 MFEM miniapp |
|---|---|
| `fluids_navier_transient.rs` | fluids/navier 时间相关 NS |
| `meshing_tmop_target_matrix.rs` | meshing/mesh-optimizer 完整 TMOP |
| `shifted_sbm_diffusion.rs` | shifted/diffusion |
| `hyperelastic_hooke.rs` | miniapps/hooke |
| `spde_gaussian_field.rs` | miniapps/spde |
| `plor_hex_solve.rs` | solvers/plor-solvers |
| `gslib_field_transfer.rs` | miniapps/gslib（近似 API） |

**验收**：`docs/evaluation/miniapp_alignment.md` 补充每项对齐说明；`cargo test --example` smoke 全部通过

---

### 4.4 P2 生态兼容与性能（M6-M9 里程碑）

#### M6 MFEM 格式互操作

- 新增 `fem-io::mfem::write_mesh`：MFEM `.mesh` v1.0 写出
- 新增 `fem-io::mfem::read_gf` / `write_gf`：MFEM `.gf` GridFunction 读写
- **验收**：`cargo test -p fem-io mfem_roundtrip` — 用 MFEM 生成 `.mesh` + `.gf`，fem-rs 读取，改写，MFEM 再读取，无信息损失

#### M7 GLVis 完整支持

- 完善 `fem-io/glvis.rs` 双向 socket 交互
- **验收**：`cargo run --example mfem_ex1_poisson -- --glvis` 能在 GLVis 4.x 窗口中显示解；`docs/evaluation/glvis_smoke.md` 记录测试步骤截图

#### M8 完整 tensor-product PA GPU

- 完善 `fem-ceed` QFunction 完整实现（H¹ mass/diffusion 3D Hex + Tet）
- 完善 `crates/linalg-gpu/src/pa_apply.rs` 修 wgpu poll timeout warnings
- 添加 GPU assembly kernel（不仅 SpMV）
- **验收**：
  - `cargo bench -p fem-benches --bench pa_gpu` 与 CPU PA 对比 ≥ 5x 加速（对 ≥ 1M DOF）
  - MMS 收敛率一致
  - `docs/baselines/` 添加 GPU-PA 基线

#### M9 HYPRE FFI（可选）

- 通过 `feature = "hypre-ffi"` 提供真实 HYPRE FFI 绑定，独立于当前兼容 API
- **验收**：亿级 DOF 3D Poisson，BoomerAMG，8-16 rank 强扩展效率 ≥ 60%

---

### 4.5 P3 前沿研究（M10-M12，长期）

#### M10 Shifted Boundary Method + Algoim quadrature

- 新 crate 或 `assembly/sbm.rs`
- **验收**：SBM MMS 收敛率 O(h²) 匹配 CutFEM

#### M11 Autodiff 材料模型

- 引入 `enzyme-rust` 或 autodiff crate
- 用 AD 重写 `plasticity::J2` 切线，与手写切线数值对比
- **验收**：AD 版切线与手写切线 Frobenius 差 ≤ 1e-10

#### M12 MHD / Plasma

- 参考 MFEM plasma miniapp 的 six-equation 模型
- **验收**：`plasma_two_fluid.rs` 示例能求解静止解并守恒质量/动量/能量

---

## 5. 验收标准总纲

**每个里程碑必须达成以下 4 项方可 close**：

1. **代码**：新增/修复的模块在 `cargo check --workspace --all-features` 无 error
2. **测试**：新增 `cargo test` 用例覆盖新代码，全部通过；MMS 收敛率符合理论阶数
3. **文档**：在本文件"进度记录"段落追加一条：`- YYYY-MM-DD M<x>.<y> completed: <一句话摘要> (commit=<sha>)`；同时更新 README 对齐表
4. **示例**：如果里程碑涉及示例，`cargo run --example <name>` 需能默认参数下运行成功，输出结果落到 `output/` 目录

**MFEM 对齐口径**（用于回答"我们是否对齐 MFEM"这个问题）：

- **强对齐**：`mfem_ex<N>_*` 前缀的示例必须与 MFEM `examples/exN.cpp` 主题、PDE、方法、BC 一致，否则重命名
- **弱对齐（可选前缀）**：仅具有 MFEM miniapp 主题对应的示例可用 `mfem_` 前缀；其他 fem-rs 独有能力（WG/VEM/晶体塑性等）**禁止**使用 `mfem_` 前缀
- **超越点**：README 中新增 "Beyond MFEM" 段落，列出 fem-rs 独有能力（本文 §2.5）

---

## 6. 可立即执行的行动项清单

按可立即执行的优先级列出：

1. 修 `crates/io/src/vtk.rs:37` 非穷尽 match（P0，30 分钟内）
2. 修 `crates/mesh/src/element_type.rs` unreachable pattern 警告（P0，1 小时内）
3. 完成 HANDOVER_2B_VECTOR_FE_NC.md 中的 Hex/Prism/Pyramid quad face HCurl 约束（P0，1-2 周）
4. 重命名错位的 `mfem_ex10_*`、`mfem_ex14_dc_current`、`mfem_ex19_navier_stokes`、`mfem_ex20_wgpu_poisson`；同时补齐真正的 ex10/14/19/20（P0-P1，1 周）
5. 为每一个"framework 存在但无示例"的模块（DPG/HDG/WG/VEM/塑性/晶体塑性/Cahn-Hilliard/XFEM 裂纹/multiphysics coupled Newton）新增至少一个示例（P1，2-3 周）
6. 添加 5 个新 NURBS/IGA 示例（P1，1-2 周）
7. 补齐 pex 至 20 个（P1，4-6 周）
8. 修 wgpu poll timeout warning（P2，小时级）
9. MFEM `.mesh` writer + `.gf` reader/writer（P2，1 周）
10. GLVis 双向 socket 交互（P2，1-2 周）
11. 元素库高阶单元闭环 M0.5：Nédélec pyramid + BDM Hex 3D + Prism/Pyramid RTk + CR1 tet 梯度（P0-P1，2-3 周）
12. NURBS 双份实现合并：`element/src/nurbs.rs` 归并到 `element/src/iga/mod.rs` 单一 API（P2，2-3 天）

---

## 7. 风险与限制

- **性能对比未做**：`docs/baselines/mfem_parity.md` 只有 fem-rs 自己的性能基线，没有跟 MFEM 同规模问题的直接对比。这不是本次评估的口径，但如果对外声称 "MFEM parity"，性能对比是必要的。建议后续 M8 里程碑同时跑 MFEM 参考问题作 side-by-side 对比。
- **代码规模不等于功能完备**：项目 `assembly` crate 有 80 个模块，但其中相当一部分（DPG/HDG/WG/VEM/xfem_crack 等）只有 framework 或框架代码，缺乏端到端示例和 MMS 验证。README 声明 "core FEM pipeline complete" 主要指基础 P1/P2 + 主流求解器，高级方法的"完成度"需要独立评估。
- **本轮 Phase 3E-3N + Phase 4 commit 太大**：1 次提交 11 个文件 2192 行，属于大颗粒变更。建议后续拆分为可独立回归的小提交。
- **README implementation status 表格过于乐观**：所有 crate 均为 ✅，但本评估显示 tmop/DPG/HDG/WG/VEM/新非线性 都存在示例或验证欠账。建议 README 增加"示例覆盖"列，明确区分"库代码有"和"用户可跑 example"。

---

## 8. 进度记录（append-only）

<!-- 每次完成里程碑或行动项，在此追加一条 -->

- 2026-07-02: 本文档创建，冻结基线口径。当前状态：核心 ex 覆盖率 78% (29/37)，pex 覆盖率 14% (5/36)，电磁 miniapp 100% (4/4)，其他 miniapp 大部分缺失；新增 Phase 3E-3N + 4 模块（塑性、晶体塑性、Cahn-Hilliard、Allen-Cahn、XFEM 裂纹、多物理 Newton、WG、VEM）全部缺示例。
- 2026-07-02 (revision-1): §1.2 fem-element / §1.3 fem-space 段落基于代码逐文件重写。修正三项遗漏：(a) Lagrange 阶数覆盖实际到 Tri P10 / Tet P6 / Seg P6，此前"P1-P3"表述严重低估；(b) Nédélec / RT 已到 NDk/RTk 任意阶泛型；(c) trace space、CR、Bernstein、Serendipity、非协调元 Q1-rot、BlockFESpace、p_refine 存在但此前未列。同步在 §2.4 增列 5 项高阶单元未完成缺口（Nédélec pyramid、BDM Hex 3D、Prism/Pyra RTk、CR1 tet grad、NURBS 双份实现），§2.5 增列 3 项超越点（Tri P10 高阶、NDk/RTk 泛型、Bernstein+Serendipity 双基），§3 P1 增至 10 项、P2 增至 16 项、P3 顺延至 17-22 项。§4 新增里程碑 M0.5（元素库闭环），§6 立即行动项从 10 条扩至 12 条。

---

## 附录 A：文件层数据源清单

- `crates/*/src/lib.rs` 各 `pub mod` 与 `pub use`
- `examples/Cargo.toml`（79 个 `[[example]]` 条目）
- `README.md` §Implementation Status
- `docs/baselines/mfem_parity.md`
- `docs/superpowers/{specs,plans}/*.md`
- `HANDOVER_*.md`（4 份未合并的技术交接文档）
- Git log：`b4e98d9` (Phase 3E-3N + Phase 4) 及其父提交

## 附录 B：如何维护本文件

1. 不要覆盖已有内容，只 append 或 edit
2. 完成一个里程碑，在 §8 追加一行
3. 需要重新评估时，创建姐妹文件 `MFEM_GAP_ANALYSIS_YYYY-MM-DD.md` 并在本文件顶部追加链接
4. README 表格如做兼容对齐更新，同步更新本文件 §2 表格

---
