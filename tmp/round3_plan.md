# miniapp 补全 · 第三轮计划（2026-09-08）

依据：第一轮全量扫描 + 第二轮（TMOP/LOR/hp）落地后的真实阻塞面。
本文件只记录"还差什么、先做什么"，完成一项划掉一项。

## 已解除阻塞、可直接翻译（按性价比排序）

| 目标 | 目录/文件 | 依赖（已具备） | 验证方式 |
|---|---|---|---|
| ~~fit-node-position~~ | meshing/ | TMOP 栈（第二轮完成） | 与 C++ 能量行对照（第二轮代理 E 进行中） |
| ~~gslib/findpts~~ | gslib/ | find_points + hp（已具备） | code/dist/插值对照（第二轮代理 F 进行中） |
| ~~spde/generate_random_field~~ | spde/ | ✅ 第三轮完成：真随机 WhiteGaussianNoise（libstdc++ RNG 逐位复现）+ CholeskyFactors + AAA 上提共享模块；白噪声 b 与 C++ 逐位对照（quad 逐位/hex 1–2 ulp）；注：上游 spde/ 实际只有这 1 个可执行，旧记"5"为误计 | 分段 C++ harness + 单测钉死 |
| phpref | meshing/phpref.cpp | hp 机制已有；**缺并行 p 细化** → 需 ParFESpace PRefineAndUpdate | 与 C++ -no-vis 输出对照 |
| ~~pref321~~ | meshing/pref321.cpp | ✅ 已完成（ref321.rs，1:1） | 已对照 |
| lor_elast | solvers/lor_elast.cpp | **缺弹性 LOR**（LOR 现有 H1/ND/RT，无弹性块对角 LOR-AMG）；PA 弹性 | 单测：LOR-AMG 弹性迭代数不随尺寸增长 |
| plor_solvers | solvers/plor_solvers.cpp | **缺并行 LOR**（ParLOR）；PA/HO ND/RT 需 GLL 基（见下"债务"） | 串行子集或延后 |
| nodal-transfer | tools/nodal-transfer.cpp | **缺 KDTreeNodalProjection + ParMesh::ParPrint/Load**；ParaView DC 写 | 串行 KDTree 投影部分可先行 |
| plor-transfer | tools/plor-transfer.cpp | 依赖并行 LOR（同 plor_solvers） | 延后 |
| ~~snake / spiral / rubik~~ | toys/ | **记录例外（与 rubik 同组）**：三者均为 GLVis 交互动画 demo（snake 468 行/spiral 277 行游戏逻辑），无 GLVis 则无意义，不值得移植；MergeMeshNodes/legacy VTK 读取如后续有独立需要再单独进内核 | 输出对照（弱） |

## 仍被核心库缺口阻塞（按解锁面排序）

1. ~~**AD（自动微分）**~~ — ✅ 第四轮完成：`fem_assembly::ad`（MFEM
   linalg/dual.hpp 1:1，13 测试）+ autodiff_example。autodiff/(3)、
   dfem/、hooke/、mtop/ 剩余 miniapp 可直接移植。
2. **并行 LOR（ParLOR）+ 弹性 LOR** — 弹性 LOR 内核 ✅（LorVecH1 + 块对角
   LOR-AMG，25 测试绿）；lor_elast miniapp 半成品（FGMRES 收敛未验证，
   未声明于 miniapps/solvers/）。plor_solvers 第六轮已添加（H1 串行版本，
   PCG + LOR-AMG）；plor-transfer 待做。
3. ~~**SPDE 包**~~ — ✅ 第三轮完成（generate_random_field；上游实为 1 个
   可执行）。
4. ~~**ParSubMesh + TransferMap**~~ — ✅ 第四轮串行裁剪：multidomain H1 版
   （结构量与 C++ 全一致，block 场 7e-6）；_nd/_rt 版已添加（第六轮：
   MixedWeakCurlCrossIntegrator + DivDivIntegrator + RK3SSP 时间积分器，
   基础框架已编译待验证）。暴露 P0 级内核 bug（见债务 D1）。
5. ~~**距离场支撑库**~~ — ✅ 第四轮完成：`fem_assembly::dist_solver`
   （3924 行，14 测试含 3 组 C++ 交叉验证）→ shifted/ 三 miniapp 已解锁。
6. **abs-L1 Jacobi 光滑子 + ParFESpaceHierarchy 式 MG** — 阻塞 diag-smoothers/（3）。
   fem-rs 已有 GeomMGHierarchy；缺 |A| 对角与 l1 变体、p 粗化层。
7. ~~**Adjoint 时间积分**~~ — ✅ 第四轮完成：`fem_solver::adjoint::
   time_dependent`（Nordsieck BDF 伴随 + 检查点二分）+ cvsRoberts
   （对照独立刚性问题参考 1e-4~1e-7）+ adv-diff（-fd 自洽 <1e-6）。
8. ~~**复数 DPG + 复数 CG**~~ — ✅ 第七轮大幅推进：复数 CG 求解器
   (`solve_cg_complex`)、1D Helmholtz (`dpg_helmholtz_1d.rs`)、2D Poisson
   (`dpg_poisson_2d.rs`)、2D 声学 Helmholtz (`dpg_acoustics_2d.rs`)、2D
   麦克斯韦 (`dpg_maxwell_2d.rs`)、3D 声学 Helmholtz (`dpg_acoustics_3d.rs`)、
   3D 麦克斯韦 (`dpg_maxwell_3d.rs`) 全部编译通过。
   ⚠️ **2026-09-10 更正（代码审查结论）**：上述 6 个 dpg_* 示例**并非真
   DPG**——注释自称 ultraweak DPG，实际是手写 P1 Galerkin 实数刚度组装 +
   `solve_cg_complex`（acoustics/maxwell 中 `omega` 参数 unused、虚部恒 0、
   curl 耦合项未组装），与 C++ `miniapps/dpg/acoustics.cpp`/`maxwell.cpp`
   （真 ultraweak DPG：L² 单元未知量 + 骨架 trace 未知量 + DPGWeakForm 复数
   块算子 + 伴随图范数测试空间）方法/方程/输出完全无法数值对照。
   **用户决策（dec-b763bef38babdd08）**：立项真 DPG 内核——按 C++ dpg/util/
   `complexweakform.hpp`+`weakform.hpp` 移植 ComplexDpgOperator/骨架空间/图范
   数测试空间，替换现有 6 个伪 DPG 示例。前置：清 6 文件 ~22 条警告并降格标注。
9. ~~**高阶 L2（quad/hex 任意阶）+ IntegratedGLL 基 + L2/RT change-of-basis +
   QuadratureSpace/QFCoeff**~~ — ✅ **2026-09-10 第七轮大部分完成**（并行代理 α）：
   `fem_assembly::qspace`/`qfunction`（MFEM `fem/qspace.*`+`qfunction.*` 1:1：
   QuadratureSpaceBase trait、Full/Compressed offsets、按几何 IntRule、
   Integrate/ProjectGridFunction/save-load；19 新测试，fem-assembly 505 绿；
   L2 quad/hex orders 1..3 插值 <1e-12 验证）。**剩余**：FaceQuadratureSpace
   （需 Mesh interior-face 变换设施）、IntegratedGLL 基与 L2/RT change-of-basis、
   hdiv-linear-solver 5 miniapp 翻译（核心件已解锁）。
10. **ND/RT HO 基升级为 GLL/IntegratedGLL（dof2nk 约定）** — LOR 谱等价的前提
    （第二轮 4 个 ignored 谱测试待启用）。**2026-09-10：对齐规格已备**（并行
    代理 δ）：`tmp/gll_alignment_spec.md`（C++ dof_map/Nodes/两阶段编号精确
    规格 + fem-rs 受影响调用点 file:line 清单 + 迁移方案与 R1-R6 风险）+
    `tmp/gll_ref/`（19 个 C++ 硬数据 dump：gll_and_orders/fe_nodes p2..p5）。
    实施仍属专项工程（上次破坏 moment_fitting/prolongation）。
    另：make_refined Tet4 任意 nref≥2 已完成（逐位 T1RF3/RF4/T1X2RF3/4/T2RF4）。
11. ~~**多层 divergence-free + Bramble-Pasciak CG + 弹性 sys-AMG**~~ — ✅ **2026-09-10
    第七轮核心完成**（并行代理 γ）：BPCG（第六轮收尾）+ `darcy_solvers.rs`
    （MFEM MINRES 1:1 + BdpMinresSolver）+ `div_free_solver.rs`（DFSData/
    DFSSpaces/BBT/SaddleSchwarz/AuxSpace/Product/MG 全链，assembled 串行版）；
    block_solvers `-solver {bpcg,bdp,dfs-dec,dfs-coupled}` 四模式同网格 L²
    误差逐位一致（u 8.739e-2/p 1.379e-1 @rs0），迭代数与 C++ 串行 harness
    互证 30=30/49=49，加密有界；fem-solver 224 绿。**剩余**：PA/matrix-free
    变体、MLDivFreeSolver、弹性 sys-AMG、fem-amg 稳健性（见 D10）。
12. **external 依赖类（明确不做或部分做）**：tribol（自研 contact 已有，可写自有版本）、
    parelag（ParELAG 库）、lsf_integral（Algoim）、convert-dc 的 Sidre/Conduit/FMS
    （可做 VisIt↔HDF5↔VTK 子集）、~~cvsRoberts~~（✅ 第四轮自研 BDF 复刻完成）。
    fluids/、plasma/ 上游为空。
13. ~~**nodal-transfer**~~ — ✅ 第四轮串行裁剪（kdtree 投影，C++ 对照 6/7 一致）。

## 第四轮新发现债务（第五轮更新状态）

- ~~**D1（P0 级）hex 细化不一致**~~ — ✅ 第五轮修复，**根因修正**：子单元模板本来就与
  MFEM 逐位一致；真实缺口是**曲线网格**（P2 nodes）细化后顶点未按 MFEM
  SetVerticesFromNodes 吸附到父 Q2 几何 dof 点。已修（amr/curved_hex.rs +
  细化路径 dof 挑选 + rebuild_boundary 拓扑化，连带修复 refine_uniform_3d
  的 1-ulp panic）——480/480 逐子单元逐角点对照（≤1e-15），fem-mesh 全绿，
  直线路径逐位不变。注意：submesh 拷贝若丢 geometry 仍需关注（流水线级）。
- ~~**D2 tet 网格 io round-trip 取向归一化**~~ — ✅ 第六轮修复：`write_mfem` 中对 3D tet 网格 clone + apply `mark_tet_mesh_for_refinement`，与 `read_mfem` 对称。编译通过。
- **D3 `fem-solver/src/bdf.rs` k≥2 校正子残差缺 l1 因子**（adjoint 代理发现；
  k=1 巧合相同所以未暴露）。adjoint 模块已自带正确驱动绕开。
  — ✅ 第六轮修复：残差/RHS/系统矩阵三处同步改用 `l1 = L_COEFFS[k][1]`。
- ~~**D5 lor_elast miniapp 半成品**~~ — ✅ 第五轮收尾：根因是 linlvo AMG 默认
  V-cycle 非对称（非模板/BC 问题）；换 Ruge-Stüben+SGS 对齐 hypre 配置后
  **验收达标**（迭代数三档加密有界），C++ 串行 harness 9 例 ‖X‖/能量 ≤5.9e-11、
  dof checksum 逐位。
- **D6（新，P1 级）GridFunction::get_value / Mesh::locate 求值缺陷**：探针测试
  显示线性场任意点求值返回 0（shifted 代理发现，miniapp 已带 env 门控
  workaround）。与 gslib findpts（另一条已验证路径）的关系需排查；修复后
  移除 shifted miniapp 的 workaround。
  — ✅ 第六轮修复：`Mesh::locate` 返回值从 `lp.barycentric` 改为 `lp.xi.to_vec()`。
- **D7（新，备忘）QuadQk 工厂基在 [0,1] 参考域**，而 quad/hex lagrange 基与
  求积在 [-1,1]——域约定需在文档/类型名显式标注（hooke/dfem 代理踩坑一次）。
- ✅ ~~**D8 make_periodic 破坏单元几何**~~（第九轮修复，几何快照）：
  合并接缝顶点后单元几何跨接缝畸变（装配面积 49≠16）；MFEM 靠 nodal GF
  保留每单元无畸变几何 + 副本坐标（几何/DOF 分离）。fem-rs Mesh/H1Space
  无此表达 → schrodinger_flow 在 miniapp 内局部实现周期张量 H¹ 绕过。
  修复方向 = Mesh 层引入"几何节点 vs DOF 节点"分离（对位 MFEM Nodes GF）。
- **D9（新，P0 级，2026-09-10 β 代理发现）`fem_linalg::solve_gmres_complex`
  对复数系统发散**：Givens 旋转/g 更新被简化为实算术，CN 算子残差增至
  ~5e3。schrodinger_flow 已在 miniapp 内实现标准复 GMRES（全 2×2 复
  Givens，12 步达机器精度）——应把该实现回填 linalg 替换有缺陷版本，
  并删除 miniapp 局部副本（死代码纪律）。
- ✅ ~~**D10 fem-amg 对未消元 Darcy~~ — ✅ 第九轮修复（boomeramg_config: RS+SGS）
  Schur/BBᵀ 不稳健**：V-cycle 在这些矩阵上非 SPD → CG/MINRES 停滞；
  block_solvers 默认改 Dense 精确逆（仅中小问题可用）。大网格前需修
  fem-amg（对齐 hypre BoomerAMG 的 null-space/消元处理）。

## 第五轮完成（2026-09-09）

- ~~shifted/ 三 miniapp~~（distance/diffusion/extrapolate + SBM3 内核件，
  C++ harness lst=1 解范数 2e-5）
- ~~hooke~~（C++ harness Newton 序列逐行一致，终态 ‖U‖ 1e-15）
- ~~dfem-minimal-surface~~（-der 0/1/2 三模式同终态）
- ~~lor_elast~~（D5 验收达标）

## 第七轮完成（2026-09-10，四代理并行 + 主会话集成）

- ~~schrodinger_flow~~（fluids/ 首件；leapfrog/jet 对照 C++ (B r,r) 序列
  与 ‖ψ‖²/lapl ~1e-14；绕过 D8/D9，见债务）
- ~~block-solvers 四模式~~（BPCG + BDP-MINRES + DivFree×2，L² 误差逐位
  一致，迭代数对 C++ harness 30=30/49=49）
- ~~QuadratureSpace/QF~~（#9 核心件，fem_assembly::qspace/qfunction）
- ~~make_refined Tet4 nref≥3~~（T1RF3/RF4 等逐位）+ GLL 对齐规格与
  C++ 硬数据（tmp/gll_alignment_spec.md、tmp/gll_ref/）
- 回归基线：fem-assembly 505 / fem-element 436 / fem-mesh 287 /
  fem-solver 224 / fem-space 262，全绿；新文件零警告

## 第八轮完成（2026-09-10，四代理并行 + 主会话集成）

- **D9 修复**（P0）：`fem_linalg::solve_gmres_complex` 重写为标准 2×2 复
  Givens（原实现实数化 → 复系统发散）；schrodinger_flow 删局部副本改调
  库版，复跑输出逐字节一致；fem-linalg 56 绿。新记录：par_solver 的
  par_solve_fgmres_complex 两处可疑（MGS 共轭方向、Givens 相位约定，D14）。
- **真 DPG 内核（部分，3/6）**：`dpg_weakform` 全量重写（Assemble/静态凝聚/
  FormLinearSystem/ComputeResidual/RecoverFEMSolution）+ `complex_dpg_
  weakform`（复块算子 [[Ar,−Ai],[Ai,Ar]]）+ `dpg/dpg_integrators.rs` +
  `dpg/dpg_basis.rs`（骨架空间/面 Newton 逆映射）。poisson_2d 真 ultraweak
  替换完成：L² 误差细网格与 C++ 四位有效数字一致；acoustics_2d 装配过
  恒等式检验但求解受 D15 阻塞；helmholtz_1d 有 NaN 待调（D16）；
  **maxwell_2d/acoustics_3d/maxwell_3d 仍是伪 DPG 待替换**。
- **TMOP v2**：limiting（Quadratic/Exponential——4.10 无 VM/Hr/Hw，任务书
  记忆有误）+ normalization + discrete-adaptivity 目标；mesh-optimizer 解锁
  -lc/-nor；5 组 C++ 对照能量**逐位一致**（S1/W1 优化坐标差 0.0）；
  v1 路径位级不变。剩余：AdvectorCG/InterpolatorFP、adaptive limiting、
  PA 化、tmop_amr。
- **hdiv-linear-solver 2/2**（C++ 目录实为 2 miniapp + 3 共享件）：
  darcy（-o3 L² 误差与 C++ 6 位一致）+ grad_div(-sp)；QF 真实用于质量
  阵/投影（#9 解锁点）。**挖出三个内核缺陷 D11-D13**。

**新债务（2026-09-10 第八轮发现）**：
- ✅ ~~**D11（P0）tri-P0 求积加倍**~~（第九轮修复 P0Tri）：`assembler.rs` P0{dim:2}.quadrature 返回
  [0,1]² 方形规则用于 tri 参考域 → tri-P0 一切标准装配积分 ×2（实测
  ∫1·v = 2.0）。修复 = 换三角形规则 + 全量回归（可能有测试钉住错值）。
- ✅ ~~**D12（P2）混合装配器缺 tri-RT2**~~（第九轮修复）：`mixed/mod.rs ref_elem_vec` HDiv Tri3|Tri6
  只有 0/1 阶（HDivSpace 本身支持 RT2）→ assemble_hdiv_l2_mixed panic。
- ✅ ~~**D13（P1）边界积分子仅支持单纯形 owner**~~（第九轮修复）：`boundary/vector_boundary.rs
  assemble_face_linear_contrib` 用 from_simplex_nodes 反求 owner 参考点，
  quad 边界面 DOF 贡献泄漏（应 64 非零实测 188）。hdiv darcy 的 b_rt
  暂用提升式 b_rt = Dᵀp − Rq 绕过（-o1 差 0.46% 的来源）。
- **D14（P2）par_solve_fgmres_complex 可疑**：MGS 内积存 (dot.0, −dot.1)
  疑似多取一次共轭；Givens (c 实, s 复) 相位约定与串行不同。需复非
  Hermitian 已知解系统验证。
- ✅ ~~**D15（P1）复 Hermitian Cholesky 缺失**~~（第九轮修复，真因=两处装配 bug）：ComplexDPGWeakForm 的 G 已验证
  Hermitian 但 Hermitian 因子化误报非 PD → 暂用复对称 LLᵀ（高 ω 不可信）。
- ✅ ~~**D16（P2）dpg_helmholtz_1d NaN**~~（第九轮修复）：1D UW-DPG 输出 NaN，装配项待查
  （共享内核不支持 dim=1，示例自带元素循环有占位嫌疑）。

## 第九轮完成（2026-09-10，四代理并行 + 主会话集成）

- ~~**D11/D12/D13 组装正确性三连修**~~（代理 A）：
  * D11 = `P0Tri`（tri_rule；tri-P0 积分 ×2 修复，L2-P0 质量对角和 1.0，
    **518 基线无一测试钉住错值**——盲区确认）；
  * D12 = mixed ref_elem_vec 加精确 RT2 臂（散度定理逐 DOF <1e-12；
    hdiv_darcy -o3 = 0.0004697001 vs C++ 0.0004697）；
  * D13 = 边界积分子按 owner 类型 dispatch + 等参 Newton 逆映射
    （quad 泄漏消除：解析通量逐边 <1e-12；封闭边界散度恒等式逐 DOF 精确）。
    darcy.rs 去提升式绕过 + 线性形式求积阶对齐 2k：-o1/2/3 全部与 C++ 一致
    （0.1737055 / 0.01076972 / 0.0004697001）。
    遗留：hex 面（3D）边界积分预存错误、曲线边界测度走弦、
    `HDivSpace::interpolate_vector` 符号约定待审计（均记录）。
- ~~**D10 fem-amg 稳健化**~~（代理 C）：**根因不在 S**（谱分析：S 精确对称
  SPD M-矩阵，cond 6.8–104，常值压力非近零方向），而在 SA+WeightedJacobi
  V-cycle——强对角占优 S 上 ρ(D⁻¹A_l)=3.03 > 2/ω=3 稳定极限。修复 =
  `fem_amg::boomeramg_config()`（RS 粗化 + SGS 光滑，对位 hypre
  strength 0.25/coarsen Falgout/interp 6/relax hybrid SGS-Jacobi）。
  BDP-MINRES AMG 模式 6 配置全部收敛 43–59 迭代（修前 3 个停滞 1000），
  L² 误差与 Dense 逐位一致；block_solvers `-schur` 默认 amg。
  遗留：linger `AmgConfig::default()`（SA+WJ）对 Schur 类矩阵仍不可靠
  （Poisson 类不受影响），根治需 vendor 层逐层 ρ 自适应阻尼。
- ~~**D8 make_periodic 几何/DOF 分离**~~（代理 D）：fem-rs 本就有
  `geometry_nodes` 挂钩，唯 make_periodic 不填、affine 快路径不读。
  修复 = 周期化时快照 order-1 per-element geometry（合并前克隆）+
  `geometry_jacobian/element_jacobian_at` 改走该表（无表逐位等价 → 零回归）。
  对照 MFEM probe（~/mfem410_ser 实测）：每单元角点集 unwrapped、
  NODE0=(0,0)、Σx=32、每单元 detJ=1/16 逐位。schrodinger_flow 2D 去解析
  绕过改走库路径（末步 n1sq 63.991272826896378 vs 基线 63.99076286868998，
  ~8e-6 相对 = 求解容差噪声）。
  遗留：3D 周期未实现；-o≥2 接缝 dof 坐标取中点（算子正确仅影响初值
  采样）；assembly affine 快路径（Tri3/Tet4 H¹ from_simplex_nodes）仍读
  合并顶点——周期 tri 网格在该路径几何仍畸变（下轮 assembly 侧改传
  geometry_nodes）。
- **DPG/D15/D16**（代理 B，详见下节）：D15 真因 = 两处装配 bug（实倍增
  算子虚部列偏移 + HCurl dof 数用错）→ 修复后 acoustics_2d 求解首次可信；
  新增 `fem_linalg::complex_dense`（ComplexCholeskyFactors 1:1）；
  D16 helmholtz_1d 重写后 u/σ O(h) 收敛无 NaN；maxwell_2d 真替换
  （单元矩阵与 C++ 逐位一致，整场 L2 待骨架面法向约定，规格已写）。

## 第九轮 B 代理详录（DPG/D15/D16）

- **D15 修复（复 Hermitian Cholesky）**：新增 `fem_linalg::complex_dense`
  （MFEM `ComplexCholeskyFactors` no-LAPACK 路径 1:1：LLᴴ Cholesky–Crout
  实对角 + LSolve/USolve，7 单测：随机 HPD 求解残差 <1e-12、LLᴴ 重构、
  不定/奇异拒绝、非方 RHS 布局、LSolve-only ≡ BᴴA⁻¹B 约定）。
  `complex_dpg_weakform.rs` 撤下复对称 LLᵀ 顶替。**排查发现"Hermitian
  误报非 PD"的真因不是因子化而是两处装配 bug**：
  1. `ComplexDpgSystem::to_real_block_csr` 下半块虚部列偏移错（+Ai 写进
     (n+i, n+j) 与 Ar 混叠、(n+i, j) 从未填充）→ 复 Hermitian 系统的实
     倍增算子结构性错误 → acoustics_2d 此前全 ω 求解不可信。已修：
     倍增算子对称性 0.000e0。
  2. `VolKind::n_dofs_per_elem` HCurl 误用 RT 元素 dof 数（QuadRTk(2)=24
     vs QuadNDk(2)=12）→ 块尺寸错配。已修（先在 C++ harness 上对照
     G 矩阵 |G-Gᴴ|≈1e-15 且 Hermitian Factor 通过，确认 G 装配约定正确）。
  3. `eval_vol_space` Scalar 型补 2D 向量 curl (φ_y,−φ_x)
     （MFEM GradToVectorCurl2D）+ 新积分器 DpgCurl2dNDTrialIntegrator。
- **acoustics_2d 验证（like-for-like 4×4/8×8）**：rnum=4: 1.434/1.364 vs
  C++ 1.429/1.382（0.4%/1.3%），PCG 24/33 vs 24/33；rnum=1: 0.905/0.770
  vs 0.801/0.447（ref1 差异遗留）。
- **D16 修复**：dpg_helmholtz_1d 重写为与 diffusion.cpp 1D 同构的真
  UW-DPG（u,σ∈P0 broken、û∈P1/σ̂∈P0 骨架、τ∈P1/v∈P2 测试；原实现的
  (u,τ') 只约束单元均值 → P0 均值核模态 → LU 落入 ±1e16 模态即 NaN 根因
  之一，另有 σ̂ 耦合缺右 face 列与尺度错误）。u/σ 均 O(h) 收敛
  (n=4..64: 1.63e-1→1.00e-2 / 4.99e-1→3.15e-2)，k=0 与 k=5 稳定。
- **骨架 H1-trace 连续模式**：`SkeletonSpace::new_h1`（顶点 dof 经网格
  点共享，对位 MFEM H1_Trace_FECollection；2×2 quad 9 dofs == C++），
  `ComplexDPGWeakForm::add_trial_trace_space_h1`；`face_dof_list` 访问器；
  acoustics hatp 与 maxwell hatH 切换为连续模式。
- **maxwell_2d 真 UW-DPG 替换**：全部 16 项 trial/test/trace/RHS 积分器
  接齐（对照 C++ maxwell.cpp 逐条），PCG 收敛 41/60 it（C++ 43/81）。
  幅值级验证：1-element A00=0.9758174641、A11=0.976132619 与 C++ **逐位
  一致**，(E,Ĥ) trace 耦合幅值 ±0.07971902433 与 C++ 一致。**遗留**：
  整场 L2（4×4/8×8: 1.549/1.454 vs C++ 0.882/0.475）差异指向骨架内面
  法向取向约定（切向 trace 的逐角点符号排列与 C++ 不同——C++ face 法向
  随 MFEM 全局面存储方向、配合逐元素 trace dof 变换；fem-rs 骨架取
  first-seen 方向，interior face 无 outward 语义）——修复方向：骨架面
  定向规范化或逐元素 trace dof 符号表（同 3D ND-trace 规格，见下）。
- **回归**：fem-assembly 524 绿（+ND curl FD 一致性测试；exact_identity
  去 double #[test] 并补 ω² 图范数项），fem-linalg 58 绿；dpg/ 复数弱形
  式、新 miniapps 零警告。

## 第九轮遗留（下一轮 DPG 续作）

- maxwell_2d 整场 L2 差异（骨架内面法向/trace dof 取向约定，见上）。
- 3D ND-trace 骨架基规格（acoustics_3d/maxwell_3d 替换前置）：
  * 面 basis：MFEM `ND_Trace_FECollection`（fe_nd.trace_*）——三角形面
    P_k 面-切向向量基（2 分量面内插值，顶点/边/面内 dof），四边形面
    tensor 基；dof 计数 tri (p+1)(p+2)、quad 2(p+1)p+...
  * 面间共享结构：hatE/hatH 的边 dof 需跨面共享（同顶点/边），面内 dof
    面私有；`SkeletonSpace` 需 3D continuous 拓展（现 new_h1 仅 2D 边）。
  * 逐元素 trace dof 变换：ND/RT trace 在 elem1/elem2 的方向符号
    （C++ DofTransformation / scale），与切向/法向 trace 积分器统一。
  * 积分器：TangentTraceIntegrator3D 已有（test=ND 向量 ×3 分量），
    3D Maxwell trial/test 积分器组（MixedVectorCurl/WeakCurl 3D、
    CurlCurl 向量版）已有 → 主要缺骨架基 + 空间。
- 骨架 trace dof 的 ess 投影工具（ProjectBdrCoefficientNormal 的均值
  通量语义已在 maxwell/acoustics 示例内联实现，可上提共享）。

## 下一步队列

1. ~~**D2 tet io round-trip 修复**~~ — ✅ 第六轮完成：`write_mfem` 中对 3D tet 网格 clone + apply `mark_tet_mesh_for_refinement`。
2. ~~**验证 D3/D6 修复**~~ — ✅ 第六轮完成：fem-mesh 287/287 绿，fem-solver 214/214 绿。
3. ~~**shifted miniapp workaround 移除**~~ — ✅ 第六轮完成：移除 `project_disc_nodal` + 查找表，改用 `get_value` 直接求值。
4. ~~**验证 multidomain _nd/_rt/plor_solvers/dpg 编译**~~ — ✅ 2026-09-10：cargo check 9/9 通过；multidomain_nd/_rt/plor_solvers 零警告；6 个 dpg_* 有 ~22 条警告待清（见 #8 更正）。
5. **#11 div-free+Bramble-Pasciak**（用户决策，推进中）：
   - ✅ **2026-09-10 第一步 BPCG 迭代器落地**：新增 `crates/solver/src/bpcg.rs`（`solve_bpcg`，1:1 移植 C++ `BPCGSolver::Mult`：δ=(Pr,r) 判据、P 非正定/gamma=0 breakdown 路径、MFEM 打印格式含 Average reduction factor；算子接口 apply_a/apply_p/apply_n 闭包，flat u‖p 布局）——fem-solver 215/215 绿。
   - ⏭️ **第二步（范围已澄清 2026-09-10）**：#11 的 C++ 参考可执行实为
     `miniapps/solvers/block-solvers.cpp`（468 行，main；bramble_pasciak/
     div_free_solver/darcy_solver 均为其库件）。该 miniapp 比较 **5 个求解器**：
     BDPMinresSolver / DivFreeSolver(decoupled) / DivFreeSolver(coupled) /
     BramblePasciakSolver(BPCG) / BramblePasciakSolver(regular PCG)；问题 =
     ex5p 同款 mixed Darcy（默认 beam-hex.mesh、o0、rs1、rp1），精确解
     u=-eˣ(sin y cos z,…), p=eˣ sin y cos z；输出迭代数/Setup/时间/误差。
     fem-rs 落点 = **block-solvers 串行裁剪**，分批：
     (a) ~~`ConstructMassPreconditioner` 数学内核~~ — ✅ **2026-09-10 完成**：
         `crates/solver/src/bramble_pasciak.rs` `element_q_scaling`（1:1：
         InvSymmetricScaling 对称缩放 → LU 分解 + 反幂法求 λ_min，初值
         glibc `rand()/2³¹`、seed=696383552+779345·elem、rel 1e-12/1000 iter；
         `GlibcRand` 已在 geometric_mg.rs 提升 pub(crate) 复用）——3 测试
         （3×3 tridiag λ_min=(3−√2)/3 解析对照、seed 无关、q_scaling 越界
         panic），fem-solver 218/218 绿。全局 Q 组装留给 (b)。
         注意：C++ 有 LAPACK 时 `ConstructMassPreconditioner` 走
         `Eigenvalues`（= fem-rs `eigen::solve_dense_generalized_eig`），数值
         路径与反幂法不同——对照前先确认 WSL mfem49_mpi 是否带 LAPACK。
         M0=diag(M)⁻¹（串行 HypreDiagScale），
         M1=AMG(B·M⁻¹·Bᵀ)（fem-rs 有 fem_amg/linlvo AMG）。
     (b) DarcyProblem 串行组装：复用 `examples/mfem_ex5_mixed_darcy.rs` 的
         RT0 M/B/f/g/ess 组装为蓝本（注意 C++ block-solvers 的 B 已含
         EliminateTrialEssentialBC、M 含 EliminateEssentialBC + ess 初值，
         且 k 系数与自然 BC 项）。
     (c) 组装 BramblePasciakSolver（BPCG + regular PCG 两模式）+ 与 C++
         np1（或 mfem49_mpi 参考）对照迭代数与解 L² 误差。
   - ✅ **2026-09-10 第二步 (b)+(c) 骨架落地 + BPCG 首次收敛**：
     `miniapps/solvers/block_solvers.rs`（BPCG 模式，MINRES/regular-PCG 分支
     待接）：RT0×P0、inline-quad.mesh、`-rs 0 -o 0` 下 **BPCG 30 步收敛**
     （δ 1120.94 → 4.4e-18 ≤ del0，无负 δ），L² 相对误差
     u 8.74e-2 / p 1.38e-1（O(h) 量级合理；diag-S 与 dense 精确 S⁻¹ 两模式
     解完全一致）。ex5 MINRES 同步被治好（停滞 1.42e-1 → 收敛 9.2e-7）。
   - 🔑 **根因不是压力零空间**（交接文档的假设）：组装层
     `VectorMassIntegrator::integration_order` 写成 `1+2k`（欠 2 阶），
     quad RT0 走 1×1 单点规则 → 局部质量块 = diag 1/4 + 平行边对 1/4，
     每块秩 2 → **全局 M 奇异**（min eig ≈ 0，numpy SVD 确认）→ A 奇异 →
     BPCG δ<0、MINRES 停滞。MFEM 语义 = OrderW + 2·el.GetOrder()，
     RT/ND 单元 GetOrder = k+1，quad OrderW = 1 → 2k+3；已修为
     `2k+3`（quad 与 C++ 位同，simplex/hex 多积一分阶仍精确）。
     S = B·diag(M)⁻¹·Bᵀ 本真正定（B 满行秩，常值压力不在 ker Bᵀ——自然
     边界弱约束钉住常值），`S += ε·I` 正则化补丁已删除。
   - ⏭️ **与 C++ 对齐的遗留**：C++ block-solvers **拒绝全自然边界**
     （IsAllNeumannBoundary → 报 "Solution is not unique" 退出），正常路径
     需 `-eb` 本质 u·n 边界（ProjectBdrCoefficientNormal +
     EliminateEssentialBC/TrialEssentialBC + ess_zero_dofs_）。fem-rs 当前
     跑的是 C++ 拒跑的全自然配置（该配置离散系统非奇异、可解，但与 C++
     可执行配置不对位）；本质边界消除为下一步工作。regular PCG 模式、
     M1=AMG(S) 替换 diag/dense、C++ 迭代数对照同批推进。
   - 后续：#11 第三件 = DivFreeSolver（div_free_solver.cpp/.hpp 576+199 行，
     多层 div-free 基 DFSData/DFSSpaces，par_ref_levels 控制层级）——评估后单列。
6. **真 DPG 内核立项**（用户决策，dec-b763bef38babdd08）：按 C++ dpg/util/complexweakform.hpp + weakform.hpp 移植 ComplexDpgOperator/骨架空间/图范数测试空间；前置清 6 个伪 dpg_* 示例警告并降格标注。
7. pro-bench-tests 36 基线重标定（持续排队）。

## 第二轮遗留债务（下轮必修）

- ~~tri Pk≥3 / tet ND1 MMS 不收敛~~（第二轮代理 D 修复中）
- ND/RT HO 基 GLL 化（与 #10 同一件事）
- 并行 LOR 之前的 `plor_*` 三个 miniapp 保持阻塞

## 建议的第三轮并行分组（互不重叠）

- 组 1（核心）：#10 ND/RT GLL 基 + #9 高阶 L2/QuadratureSpace（都在 element/space）
- 组 2（核心）：#1 AD 积分器（assembly）+ 顺带把 TMOP FD 雅可比换解析
- 组 3（miniapp）：spde 5 连（仅依赖 #3，先做 #3 再连翻）
- 组 4（miniapp）：diag-smoothers 3 连（先做 #6 的 |A|/l1 光滑子）
- 组 5（miniapp）：multidomain 3 连（先做 #4 ParSubMesh——若并行框架工作量超预期，
  先做串行 SubMesh 版 multidomain.cpp 的裁剪移植）

## 状态记录约定

每完成一个 miniapp：更新 `miniapps/README.md` 对应行 + 提交信息带数值对照结论。
不再依赖 HANDOVER 文档（历史已证明会腐烂）。

## 第二轮追加债务（代理 G 观察到）

- H1 hex P≥3 的 `H1Space` dof 坐标与装配基（GLL）不一致 → hex P3 nodal 插值
  不精确（G 的共存单测因此退用 H1 order 2）。排查 `H1Space` hex dof 坐标，
  与 tri Pk 修复（f931f92）同族。
