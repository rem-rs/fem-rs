# fem-rs vs MFEM 改进计划 — 验收标准

_配套 `IMPROVEMENT_PLAN.md`。_

## 通用约定（每条都必须满足）

- (a) 行为正确：MMS 收敛阶达标 + 单元/集成测试通过。
- (b) 工程门禁：`cargo build --workspace` 与 `cargo test --workspace` 全绿；无 `panic!`、`unimplemented!`、`todo!`；无新 warning。
- (c) D1/D2 合规：无外部求解器 FFI；不依赖旧 MFEM_MAPPING/ALIGNMENT 文档；新事实写入 MEMORY.md。
- (d) Demo：至少一个 `examples/` 可编译可运行用例。
- (e) PR 描述附关键 file:line 证据。

## P0 验收

### P0-1. Mesh IO 对齐
- 读取 MFEM v1.0/v1.2 至少 5 个 `.mesh`（含 quadratic），跑通对应 ex1/ex2/ex3。
- `gmsh::write_msh` 文件能被 GMSH GUI 打开；fem-rs 读→写→读 节点/单元/物理组完全一致。
- `.pvd` collection + `.pvtu` 多 rank 输出在 ParaView 5.x 中正确显示场。
- VTK reader 解析 Tri3/Quad4/Tet4/Hex8 connectivity，round-trip 测试通过。
- CGNS / Exodus / NetCDF / Tecplot / STL / OBJ 各 1 个最小 reader + 1 个测试样例。

### P0-2. 单元高阶修复
- `TetRT2`：Darcy MMS 在 Tet 网格 ‖u−u_h‖₀ 收敛阶 ≥ 2.8（期望 3）。
- `HexRTk` (k≥1)：`eval_div`/`eval_curl`/`dof_coords` 与解析/有限差分一致；Hex RT1 Darcy MMS ≥ 1.8 阶。
- `PyraRT0::dof_coords`：5 个面心位置正确，逐面单元测试通过。
- `HexNDk`/`QuadNDk` (k≥2)：补 face/interior moment；Hex ND2 Maxwell MMS ‖curl(E−E_h)‖₀ ≥ 1.8 阶。
- RT/ND 任意阶 k=1..8 跑通，无 OOB panic。
- `HCurlSpace` 接受 Quad8/Hex20 + 混合元混排；MMS 收敛阶不退化。
- `HDivSpace::interpolate` 与 `interpolate_vector` 结果一致；Hex RT1 单元测试通过。

### P0-3. MixedAssembler 通用化
- `ref_elem_vol` dispatch 覆盖 H¹/L²/HDiv/HCurl；不支持的组合返回 `Err` 而非 `panic!`。
- Darcy（RT0-L²、RT1-L²）MMS：压力 ≥ 1 阶，通量 RT0 ≥ 1 阶 / RT1 ≥ 2 阶。
- HCurl×H¹ mixed（如 magnetic vector potential）至少 1 个 MMS 通过。

### P0-4. 3D refine_uniform 直通
- `refine_uniform` 对 Tet4 / Hex8 / Prism6 可直接调用，输出元素数 = 8 × 原元素数。
- Poisson P1 → 2 阶 L²；P2 → 3 阶 L²。
- 与 NC AMR 全标记路径输出节点数 / 单元数一致。

### P0-5. 工程卫生
- 根目录 0 个 `*.log` / `*.txt` 调试残留；`cUsersliluworksfem-rstests/` 删除。
- 旧文档（PLAN_622.md、TECHNICAL_SPEC.md、MFEM_MAPPING.md、ALIGNMENT_TRACKER.md 等）：删除或顶部加 `> DEPRECATED — see docs/IMPROVEMENT_PLAN.md`。
- `.gitignore` 覆盖 `*.log` / `*.txt` 等常见产物。

## P1 验收

### P1-1. 统一面装配
- 新 `FaceIntegrator` trait + 通用 face driver 落地；DG SIP、DG advection、HDG、Euler 全部迁移。
- 原三套 face loop 代码行数减少 ≥ 50%；现有 ≥ 20 个 face 测试全绿。
- 新增至少 1 个跨场景 face integrator 复用 demo。

### P1-2. DG 守恒律框架
- `HyperbolicConservationLaw` trait（state / flux / num_flux / source）落地。
- Euler3D 在 cube / forward-step 跑 ≥ 100 步稳定；与 Euler2D MMS 在常截面上一致。
- HLL/HLLC flux 通过 Sod shock tube 一维基准（L¹ 误差 < 5%）。
- minmod + Barth-Jespersen 限制器至少一种，在含间断算例下抑制振荡（TVB 验证）。
- Burgers DG 1D/2D 各 1 个测试通过。

### P1-3. AMS / ADS
- AMS（Maxwell ND1/ND2）：h 减半时迭代次数增长 < 20%。
- ADS（H(div) Darcy 类）：h-independence 验证。
- 与 SA-AMG/RS-AMG 在标量 Poisson 上的迭代次数对比表写入 `docs/baselines/`。

### P1-4. 非线性工具箱
- JFNK：≥ 1 个 NS/弹性非线性算例达 Newton 二次收敛，log10 残差减少 ≥ 8 阶。
- Anderson 加速：难收敛 Picard 算例迭代数较纯 Picard 减少 ≥ 40%。
- Trust-region dogleg：恶劣初值不发散，胜过 line-search Newton。
- Continuation：参数 0→1 拱形非线性扫描稳定通过。

### P1-5. 弹性 / 大变形
- `StressIntegrator` trait + 几何刚度装配 API；St-VK / Mooney-Rivlin / Ogden 三种本构落地。
- 单轴拉/压、简单剪切解析解对比误差 < 1%。
- `ElasticityIntegrator::with_plane_stress(true)` 显式 flag；2D 板算例验证。

## P2 验收

### P2-1. GPU 装配加宽
- f64 GPU COO 装配（`GpuCooTriplet<f64>`）；CPU/GPU 元素级相对误差 < 1e-12。
- 新增 WGSL shader：Poisson Tet4/Hex8、Mass Tet4/Hex8、Elasticity Tet4/Hex8（≥ 6 个）。
- 高阶 ≥ 1 套 shader（如 Tri P2 Poisson）跑通 MMS。

### P2-2. CUDA backend
- `cargo build --features cuda` 在装 CUDA Toolkit 机器上无错；`backend_cuda.rs` `#![cfg]` 误用修复。
- SpMV / axpy / dot CPU vs CUDA 随机矩阵相对误差 < 1e-10。
- CUDA CG 在 1M DOF 3D Poisson 上比 wgpu 后端快 ≥ 2×（消费级 GPU）。

### P2-3. GPU AMG
- GPU SA-AMG 在 1M DOF Poisson 上迭代次数与 CPU AMG 差 < 10%。
- GPU PCG-AMG 端到端比 GPU PCG-Jacobi 在大规模算例上快 ≥ 3×。

### P2-4. 真正分布式 ParMesh
- `SharedFace/SharedEdge/SharedVertex` group 落地，跨 rank face/edge 邻接一致性校验通过。
- 增量分布式构建：每 rank 内存峰值 ≤ O(N/p)；2 rank 测试比 replicate 模式峰值下降 ≥ 40%。
- 弱可扩展：DOF/rank 固定，至 16 rank 效率 ≥ 60%。

### P2-5. 分布式求解器
- 分布式 BiCGStab / IDR(s) / FGMRES / ILU 各至少 1 个 4-rank MMS 测试。
- 分布式 multifrontal：≤ 1e5 DOF 与 serial LU 相对误差 < 1e-10。
- HYPRE 命名层与 `par_*` 入口合并或显式声明等价；冗余 API 至少删一套或加 deprecation。

### P2-6. GPU-aware MPI
- `mpi + cuda` 双特征启用 GPU 指针直传；与 CPU staging 在 16-rank Poisson 上一致。
- benchmark：GPU↔GPU 通信带宽 ≥ CPU staging 路径 1.5×。

## P3 验收

### P3-1. Phase-field
- Cahn-Hilliard：能量单调下降验证；液滴半径增长率与 Lifshitz-Slyozov 理论 ±10%。
- Allen-Cahn：椭圆收缩率与解析解 ±5%。

### P3-2. POD / DEIM
- POD 投影误差 ≤ 给定截断奇异值上界；ROM 在线求解时间 ≤ FOM 的 5%。
- DEIM 非线性项重构相对误差与基数 m 匹配理论曲线。

### P3-3. PDE-constrained adjoint
- ≥ 1 个稳态算例（Poisson 控制）：finite-difference 校验 adjoint 梯度，每分量相对误差 < 1e-6。
- 优化终止时 KKT 残差 < 1e-8。

### P3-4. Stochastic Galerkin
- PCE 在 1D 椭圆 log-normal 系数上复现解析均值/方差 ±5%。
- Karhunen-Loève 截断协方差 Frobenius 误差 < 1e-3。

### P3-5. 其他单元
- BDM1 / BDM2 在 Tri/Tet 上 MMS 达对应阶。
- Crouzeix-Raviart 在 Stokes 上稳定（LBB 验证）。
- Pyramid ND1：混合 Pyramid+Tet 网格上 Maxwell MMS ≥ 1 阶收敛。

## P4 验收

### P4-1. Mesh 拓扑层
- `MeshTopology` trait 增 `n_edges()` / `edge_iter()` / `face_orientation()` / `edge_to_elements()`。
- 单元测试覆盖 Tri/Quad/Tet/Hex/Prism 全部 orientation 一致性。

### P4-2. Mesh 几何 API
- 通用 `transform(&Fn)` / `translate` / `scale` / `rotate`；2D+3D 单元测试。
- `make_periodic` 支持 rotation；圆环算例验证。
- 3D SubMesh 支持 Tet/Hex/Prism 各至少 1 个测试。
- CurvedMesh Hex/Prism 至少 P2 曲面 1 个测试。

### P4-3. 分区算法
- Hilbert / Z-order / RCB 各 1 个实现；与 `rmetis` 在 edge-cut + imbalance 上对比表。
- 真 METIS binding（feature-gated，仅作分区不作求解，不破坏 D1）：可选。

### P4-4. FE 高阶通用化
- `build_pk` 支持 Quad/Hex P≥3 直接路径，无需绕 `build_variable_order_dof_manager`。
- 各向异性张量阶（Hex p_x≠p_y≠p_z）显式 API + 1 个测试。
- IGA 实现 `FESpace` trait，在通用 assembler 中跑 1 次。

### P4-5. Transfer / Prolongation
- 通用 `GetProlongationMatrix(fine, coarse)` 支持 Lagrange/RT/ND，h-refinement 路径。
- mesh→mesh L² projection 通用化（脱离 H¹-only 限制）；至少 3 种空间组合测试。

### P4-6. Form 高层抽象
- `BilinearForm` / `LinearForm` / `NonlinearForm` 高层类落地，支持 add / MultTranspose / 组合。
- `BlockNonlinearForm` 移至 assembly crate 并与 Form 体系挂接。
- 真正的 `Hybridization` 类：HDG saddle 至少 1 个完整算例跑通（不只是 element-level static_cond）。

### P4-7. BC 第一类公民
- Nitsche BC 统一 API；Poisson + 弹性各 1 个测试。
- Periodic BC 进入 BC 系统枚举；圆环 / 方腔算例验证。

## P5 验收

### P5-1. 收敛测试
- 3D Maxwell MMS（ND1/ND2）测试通过：‖curl(E−E_h)‖₀ 收敛阶 ≥ k。
- 多 rank MMS 收敛：4/8/16 rank 下 P1/P2 收敛阶与 serial 一致。
- GPU MMS harness：CPU/GPU 在同一 MMS 上误差差 < 1e-10。
- Phase-field MMS：Cahn-Hilliard / Allen-Cahn 各 1 个解析或半解析对比。

### P5-2. Complex 路径
- `Complex<f64>` GPU SpMV / CG 与 CPU 结果相对误差 < 1e-10。
- `mfem_ex22_complex_helmholtz` 提供 GPU 版本，与 CPU 一致。
