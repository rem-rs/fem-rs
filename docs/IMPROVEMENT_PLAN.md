# fem-rs vs MFEM 改进计划

_基于 2026-06-29 对 .rs 源码的事实核查（按 D2 不信任旧 MFEM_MAPPING/ALIGNMENT 文档）。引用都是 file:line。_

## 优先级 P0：阻塞用户能力，必须先做

### P0-1. Mesh IO 对齐 MFEM 生态
- 缺 MFEM 原生 `.mesh` v1.0/v1.2 reader → MFEM examples 自带 mesh 文件读不进来
- 缺 GMSH writer（`crates/io/src/gmsh.rs` 文件头声称支持但实际无 `write_msh`）
- 缺 ParaView `.pvd`/`.pvtu` collection 输出
- 缺 CGNS / Exodus / NetCDF / Tecplot / STL / OBJ
- VTK reader 仅 ASCII point-data，不解析 cell/connectivity（`crates/io/src/vtk_reader.rs:1`）

### P0-2. 修高阶单元的"看似有实际不能用"
- `TetRT2` 只有 60 行 placeholder（`crates/element/src/raviart_thomas/tet_rt2.rs:17`）
- `HexRTk::eval_curl` 永远返回 0，`dof_coords` 返回零向量（`hex_rtk.rs:178, 185`）
- `PyraRT0::dof_coords` 返回错误形状的 5 个 barycenter（`pyramid.rs:54`）
- `HexNDk`/`QuadNDk` (k≥2) 只产 edge DOF，face/interior moment 未实现（`hex_ndk.rs:30`、`quad_ndk.rs:28`）
- RT/ND 任意阶缓存硬编码大小 6，超阶 panic via array OOB
- `HCurlSpace` 拒绝 Quad8/Hex20（`crates/space/src/hcurl.rs:105-108`）和混合元（line 99）
- `HDivSpace::interpolate` 返回零（`hdiv.rs:896`），只 `interpolate_vector` 工作
- `HDivSpace` Hex 仅 RT0（`hdiv.rs:130`），无 Hex RTk≥1

### P0-3. MixedAssembler 通用化
- 当前仅 H¹×H¹（`crates/assembly/src/mixed.rs:182`，`ref_elem_vol` 只 dispatch Lagrange，panic on HDiv/L2）
- Darcy（RT-L²）saddle-point 走不通用路径
- 需扩展 HDiv×L²、HCurl×H¹、混合 ref_elem dispatch

### P0-4. 3D `refine_uniform` 直通
- 只有 2D 入口（`crates/mesh/src/amr.rs:1211`），3D 必须绕 NC AMR
- 需 Tet/Hex/Prism uniform refine 直接路径

### P0-5. 工程卫生
- 根目录 log/txt 残留：`solver_err.log`、`solver_err2.log`、`solver_err3.log`、`space_errors.txt`、`test_output*.txt`、`find_results.txt`、`bench_output.txt`
- 误创建目录 `cUsersliluworksfem-rstests/`（路径拼接 bug 痕迹）
- 顶层 `PLAN_622.md`、`TECHNICAL_SPEC.md` 与 `docs/` 下旧文档需标记废弃或重写

## 优先级 P1：扩展求解能力

### P1-1. 统一面装配抽象
- 三套独立 face loop 互不复用：`DgAssembler::assemble_sip`、`assemble_dg_interior_faces<F: DgFaceIntegrator>`、HDG/Euler 手写
- 需 `FaceIntegrator` trait + 通用 driver（DG/HDG/Continuous 统一）

### P1-2. DG 守恒律框架完整化
- Euler3D 仅 stub，face assembly 未完成（`crates/assembly/src/dg_euler_3d.rs`）
- 无通用 `HyperbolicConservationLaw` 抽象（`hyperbolic.rs:17` 仅 1D Euler）
- 缺 HLL/HLLC Riemann flux，仅 LF + 部分 Roe
- 缺 limiter / slope limiter
- 缺专用 Burgers DG

### P1-3. AMS / ADS — Maxwell/H(div) AMG
- 当前 AMG 仅通用 SA-AMG / RS-AMG
- 弹性/Maxwell 大规模求解的关键
- 缺失会让 GPU 求解器对电磁/弹性问题不够用

### P1-4. 非线性求解器工具箱
- 缺 JFNK（Jacobian-free Newton-Krylov）
- 缺 Anderson 加速
- 缺 trust-region（dogleg / 2D subspace）
- 缺 continuation / homotopy
- 缺 AD/符号微分；NonlinearForm 的 Jacobian 必须用户手写（`solver/src/nonlinear.rs:99`）

### P1-5. 弹性 / 大变形完善
- 缺 StressIntegrator 通用接口、几何刚度 API
- 仅紧致 Neo-Hookean（`nonlinear_hyperelasticity.rs:35`）
- 缺 St-Venant-Kirchhoff / Mooney-Rivlin / Ogden
- `ElasticityIntegrator` 无 plane_strain/plane_stress flag，按 dim 自动隐式

## 优先级 P2：性能与异构

### P2-1. GPU 装配 f64 + 3D + 高阶 + Hex/Tet
- 当前 GPU 装配仅 f32（`GpuCooTriplet.val: f32`）
- 仅覆盖 Poisson Tri3、Mass Tri3、Elasticity Tri3 三个 WGSL shader
- 3D / Hex / Tet / 高阶 / Mixed 全无 GPU 路径

### P2-2. CUDA backend 修通
- `backend_cuda.rs` 顶部 `#![cfg]` 写法不合法（MEMORY 已记录）
- 需真正跑通 cuSPARSE/cuBLAS 路径

### P2-3. GPU AMG
- CG/PCG/GMRES 有 GPU 版本，AMG 仍 CPU
- GPU SpMV 已存在但未对接 AMG coarsening / smoothing

### P2-4. 真正分布式 ParMesh
- 当前 `partition_simplex` 每个 rank 先持有全网格再切（`par_simplex.rs:24-26` 自承认）
- 缺 `SharedFace/SharedEdge/SharedVertex` group 抽象（`crates/parallel/src/ghost.rs` 只 node 级）
- 缺增量分布式构建

### P2-5. 分布式求解器补全
- 缺分布式 BiCGStab / IDR(s) / TFQMR / FGMRES / ILU
- 缺分布式直接求解（multifrontal 分布式）
- 缺分布式 EVP / 分布式 ODE
- HYPRE 命名层 `HypreParMatrix`/`hypre_solve_*` 名为分布式实为本地串行包装；真正分布式入口是 `par_solve_*`，两套命名空间未统一

### P2-6. GPU-aware MPI
- 当前完全无

## 优先级 P3：缺失的整块新方向

### P3-1. Phase-field PDE 套件
- Cahn-Hilliard、Allen-Cahn 整套缺失

### P3-2. POD / DEIM / Reduced Order Model
- 完全无

### P3-3. PDE-constrained adjoint optimization
- 完全无（仅 TMOP mesh quality）

### P3-4. Stochastic Galerkin 扩展
- `crates/stochastic` 存在但功能范围未对齐 MFEM `miniapps/uq`
- 缺 PCE 完整路径、Karhunen-Loève 展开成熟实现

### P3-5. 其他单元类型
- BDM / BDFM 完全无
- Crouzeix-Raviart 非协调元完全无
- Pyramid Nedelec ND1 仍未解决（MEMORY 记录三种尝试失败）

## 优先级 P4：基础抽象与 API 表面

### P4-1. Mesh 拓扑层
- `MeshTopology` trait 不暴露 edge/face 邻接、orientation、edge enumeration（`crates/mesh/src/topology.rs`）
- 缺 `n_edges` / edge→element / face orientation

### P4-2. Mesh 几何 API
- 无通用 transform（move/scale/rotate）；只有 `apply_node_displacement` 闭包式（`moving_mesh.rs:54`）
- `make_periodic` 仅 translation，不支持旋转周期（`simplex.rs:277`）
- 3D SubMesh 仅 Tri3 2D（`submesh.rs:70`）
- CurvedMesh 3D 仅 Tet（`curved.rs:57`）；Hex/Prism/Pyramid 曲面无路径

### P4-3. 分区算法
- 真 METIS/ParMETIS binding 无；`crates/rmetis` 是纯 Rust 自实现
- 缺 Hilbert / Z-order / RCB

### P4-4. FE 高阶通用化
- `build_pk` 仅 simplex（`dof_manager.rs:1093`），Quad/Hex P≥3 必须经 `build_variable_order_dof_manager`
- 缺各向异性 tensor-product 阶（Hex p_x≠p_y≠p_z）显式 API
- IGA 单独走自己的代码路径，不与 `FESpace` trait 统一

### P4-5. Transfer / Prolongation
- 缺通用 `GetProlongationMatrix` / `GetTransferMatrix`（mesh→mesh）
- 仅 `transfer_h1_p1_nonmatching_l2_projection`（P1）和 `prolongate_p2_hanging`（2D P2）

### P4-6. Form 高层抽象
- 缺 MFEM 风格 BilinearForm/LinearForm/NonlinearForm 统一类层级与组合（add / MultTranspose）
- BlockNonlinearForm 在 solver crate 而非 assembly（`crates/solver/src/block_operator.rs:158`），与 Form 体系脱钩
- 缺真正的 Hybridization 类；只有 element-level static condensation（`static_cond.rs`）

### P4-7. 边界条件第一类公民
- 缺 Nitsche 统一 BC API
- 缺周期 BC 作为 BC 系统的成员

## 优先级 P5：测试与验证

### P5-1. 收敛测试缺口
- 3D Maxwell MMS — 无（仅 2D ND1/ND2 覆盖）
- 多 rank MMS 收敛 — 无（并行测试只验证 serial vs parallel 一致）
- GPU MMS / GPU 收敛 harness — 无
- Phase-field / Cahn-Hilliard MMS — 无（PDE 都没有）

### P5-2. Complex 路径
- `Complex<f64>` 仅 CPU；GPU 路径无 complex 支持
- `mfem_ex22_complex_helmholtz.rs` 仅 CPU

## 推进建议（按价值/可行性）

1. **P0-5 工程卫生**：半小时清掉根目录日志和误创目录，立即降噪。
2. **P0-2 单元高阶修复**：解开"看似有"的雷区，先让 RT/ND 高阶可用。
3. **P0-1 MFEM `.mesh` reader + GMSH writer + `.pvd/.pvtu`**：打通与 MFEM 生态互操作的最短路径。
4. **P0-3 MixedAssembler 通用化**：兑现 RT/ND 单元的实际价值（Darcy/Stokes/Maxwell mixed）。
5. **P1-1 统一面装配 + P1-2 DG 守恒律框架**：合并三套 face loop 后顺势补 Euler3D / HLL / HLLC / limiter。
6. **P1-3 AMS/ADS**：让 GPU 求解器对真实电磁/弹性问题有竞争力。
7. **P1-4 JFNK + Anderson**：扩充非线性工具箱，成本低收益大。
8. **P2-1/2-2/2-3 GPU 路径加宽 + CUDA 修通 + GPU AMG**：性能层的"下一台阶"。
9. **P2-4/2-5 真正分布式 ParMesh + 分布式求解器补全**：规模上限的"下一台阶"。
10. **P3 系列**：作为长线方向，每条单独立项后再排期。
11. **P4 系列基础抽象**：随 P0/P1 推进时顺手补；不单独立项也无明显阻塞。
12. **P5 测试**：补 3D Maxwell MMS、多 rank MMS、GPU MMS harness 作为质量护栏。

---

每项的具体验收标准（MMS 收敛阶、误差阈值、性能加速比、API 落地清单等）见配套文件 [`IMPROVEMENT_PLAN_ACCEPTANCE.md`](./IMPROVEMENT_PLAN_ACCEPTANCE.md)。
