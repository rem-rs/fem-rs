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

## 第十轮完成（2026-09-10，五代理并行；A 被中断，主会话接手完成其消费方接线）

- **A（部分交付）DofTransformation 家族 + canonical 面方向**：
  `fem_space::dof_transformation`（doftrans.hpp 1:1：DofTransformation/
  NdDofTransformation/StatelessDofTransformation/FaceGeom/rt_trace_face_sign，
  11 测试）；`SkeletonSpace` 面改按 MFEM canonical 方向存储（2D 边表
  min→max）+ `elem_face_orientation()`；**DPG trace 装配的符号从"首见元素
  ⇒ +1"改为按朝向**（旧规则仅在面按生成元素局部方向存储时等价）。
  中断处置：A 的授权不含消费方 `dpg_weakform.rs`，遗留 1 个失败测试
  （poisson 精确解一致性）；主会话定位根因（法向来自 canonical 而 scale
  仍按首见 ⇒ 符号不一致）并完成接线 + 清未用导入 → DPG 45/45 绿。
  **未达标**：`dpg_maxwell_2d` 整场 L2 仍不收敛（-n 4/8：1.422/1.404
  vs C++ 0.882/0.475；单元矩阵级此前已逐位一致）→ 缺口在骨架/边界，
  见遗留节。
- **B Hybridization 真实现 + constraints 家族**：stub 替换为
  `fem/hybridization.*` 1:1（H1/RT trace Schur、AssembleFaceMatrix、
  GetMatrix/Finalize、消元回代）；`constrained.rs` 由 1/11 扩到
  Elimination×3 / Penalty×3 / Eliminator / EliminationProjection。
  测试 `hdiv_hybrid_matches_direct_face_dg_trace`（RT0/P0 与 P1 trace）、
  `hybrid_vs_direct_vs_saddle_minres` 全绿。
- **C DGMassInverse + L2FaceRestriction + PA 对流**：`dgmassinv.rs`
  （MFEM `DGMassInverse` 1:1；P1/P2/P3 对照 C++ harness ≤1.5e-15，P1 硬编码
  为常驻位级门）；`face_restriction.rs`（L2FaceRestriction Double/SingleValued
  + Conforming；DG 内面 jump 算子经 face-restriction 与逐面直接装配
  **逐位相等**）；`partial.rs` PAConvectionOperator。
  ⁉️ 附带发现既有 bug：`standard::ConvectionIntegrator` 仿射路径丢
  |det J|（unit_square_tri 上 max|K·x − M·1| = 0.75）→ 见遗留 #2。
- **D TMOP 尾巴 + 估计器**：`TmopRemapEvaluator`（AdvectorCG 1:1：移动网格上
  ConvectionIntegrator + Mass PCG、RK4 步进、dt=0.5·h_min/|u|、截断；
  InterpolatorFP 变体）驱动 `UpdateTargetSpecification`；
  `EnableAdaptiveLimiting`；mesh-optimizer 解锁 **-tid 5/6/7/8 + -alc**。
  对照：`-tid 1 -alc 1.0 -nor` **逐位一致**（能量 13 位、网格 RMS 1e-8）；
  `-tid 5` 初始能量逐位（4.6366e-01），remap 噪声后 0.26% 分叉。估计器：
  `lp_error_estimator`、`ls_zz_estimator`、AnisotropicErrorEstimator trait
  + `zz_estimator_aniso` + `amr_refiner::apply_aniso`。
- **E ODE 高阶 + 网格 I/O + 曲率**：`ode/high_order.rs` 九个求解器（Rk2/Rk6/
  Rk8/Ab5/ImplicitMidpoint/Sdirk34/Sdirk33/Esdirk32/Esdirk33，系数 1:1 抄自
  linalg/ode.*，实测阶 + 刚性模型对照 C++ ≤1e-13）；io 新增 **TrueGrid 读
  （对照 C++ dump 逐位）、CUBIT 读、NetCDF-3 自研读写、legacy VTK 写
  （三网格逐字节一致）、Exodus II 写（roundtrip）**；`set_curvature` 补
  Pyramid5（Tet4/Prism6 已有）+ 清 3 条既有警告。
- 回归基线：fem-assembly **555** / fem-solver **244** / fem-mesh **290** /
  fem-space **274** / fem-io **122**，全绿；新文件零警告。

## 第十一轮完成（2026-09-10，四代理 + 两代理续作 + 主会话集成）

- **F 组装正确性四连修（全部含 C++ 对位）**：
  ① **P1**：仿射快路径与等参路径用了两套约定（`{J⁻ᵀ∇φ, |detJ|}` vs MFEM 的
  `{adjJᵀ∇φ, 1/|detJ|}`）→ 按 MFEM 写法写的积分器在单纯形上整体丢 |detJ|
  （`K·x = M·1` 实测 0.75 → 修复后 5.6e-17；tri/tet/quad 均 ≤1e-13）；
  ② hex 边界面被静默截成 3 角点（半面积 + 错法向）→ 按面自身 P1 参考元 +
  面→单元仿射映射；③ 边界求积改为经 owner 等参几何前向映射（曲边弧长
  1.0982466 vs 解析 1.0982301，旧弦线给 1.0000）；④ 单纯形仿射快路径改读
  逐单元 `geometry_nodes`（周期三角网格单元质量阵与未周期化逐位相同）。
- **G：`dpg_maxwell_2d` 整场 L2 收敛（第十轮遗留 #1 结案）**：根因**不是**
  骨架/法向/trace——miniapp 里 ess 边界装配把实/虚分量错位
  （`Ê = (E_y, −E_x)` 是复向量，通量需 `Re/Im(Ê·n)`，旧码把 `Im(E_y)` 当实部
  → 实边界数据恒 0）。对照 C++：rnum=1 n=4 0.8819/0.8782、n=8 0.4753/0.4747、
  n=16 0.2370/0.2369；误差随加密下降（rate −0.94 vs C++ −0.95）。
  算子级证据：元素置换后 `max|A_cpp − S·A_rust·S| = 1.4e-12`（非 ess 自由度）。
- **H：3D trace 骨架基（H1/RT/ND）**：+1590 行，MFEM dof 计数/编号逐位对照
  （hex/tet × 3 类 × p=1..3 共 18 组面 dump 与元素 trace 块 dump **完全一致**，
  面基值 ≤1e-14）；补 `DpgTransposedMixedCurlIntegrator`。**纠正规格**：
  MFEM 无 `ND_TraceElement`，`ND_Trace_FECollection(p,dim) = ND_FECollection(p,dim−1)`。
  3D 示例**未替换**（阻塞见 D24）。
- **I：AMG 粗解 vendor 缺陷（D10 真根因）+ D14 复 FGMRES**：
  `linlvo::direct::SparseLu` 默认 Rcm 重排下 `solve` 返回**置换解**
  （最小复现 `max|A·A⁻¹−I| = 1.0` vs Natural 的 2.2e-16），而 AMG 粗层正用它
  → V-cycle 非对称（`max|B−Bᵀ| = 7.8e-2`）→ CG 停滞。修复 =
  `fem_amg::CorrectedAmgPrecond`（同 cycle，粗解 Natural + 缓存分解）：
  star Schur rs0/1/2 = 3/5/5 迭代（修前 3 个配置停滞）、Darcy Schur
  7/11/16、强对角占优 cond=1.05e3 = 4 迭代；新增 4 测试（含钉住 vendor
  缺陷的 `sparselu_reordering_solve_is_not_an_inverse`）。D14 确认为真错：
  复 FGMRES 的 MGS 内积**双重共轭** → 71→40 迭代、reported/true 残差
  1.19→1.00；残差估计改 `hypot`。
- **K（续作）：NURBS 示例族 API 迁移**——6 个示例恢复编译（详见 D21 的实质
  问题，未擅自修）。
- **L（续作）：fem-parallel P2 分区缺陷（D17 结案）**：`DofManager::new(mesh,2)`
  对 Tri3 走 `build_pk`，边 dof 存在 `edge_pk_map`、返回的 `edge_dof_map`
  是空表，而 partition 按 `order == 2` 选表 → 读到空表 → 所有边 dof 被当
  元素内部 dof（每元素 3 个）而计数按硬编码 1 → 枚举 121 vs 计数 57。
  修法 = 按"哪个表非空"选择 + 内部 dof 数从实际列表导出 + 两条不变量
  debug_assert。fem-parallel **225/225**（前 217过/7败），新增跨分区 P2
  一致性测试；quad Q2 路径无变化。
- **主会话集成：示例编译门禁恢复（7 个坏示例 → 0）**：① 恢复被 `012c2ec`
  以"无外部调用"删除的 `crates/parallel/src/par_dpg_trace.rs`（291 行，
  pex8 实际在用；删除时只注释 import 未改主体）→ pex8 可编译**且可运行**
  （np1: 1361 未知数、PCG 134、收敛）；② `mfem_ex7` 3 处 Mesh 字面量缺
  `vertex_parents`；③ `mfem_pex27`/`pex5` 的 `face_elements` 遮蔽陷阱；
  ④ `mfem_pex4`/`pex5` 误差函数签名漂移（见 D19）。
- 回归基线（全绿）：fem-assembly **570** / fem-solver **244** / fem-mesh **290**
  / fem-space **274** / fem-io **122** / fem-amg **23** / fem-linalg **63** /
  fem-parallel **225**；**`cargo build --release --examples` 0 错误**。

## 第十二轮完成（2026-09-10，四代理并行 + 主会话集成）

- **M：`hdiv_error.rs` 真实现（D19 结案）**：1:1 移植 MFEM
  `GridFunction::ComputeL2Error`（标量）与矢量重载（逆变 Piola 重建 +
  |detJ| 权重 + 逐空间参考元/符号）；**关键正确性修正**：MFEM 的 RT 单元
  `GetOrder()` 存的是 `p+1`（fe_rt.cpp），故默认求积阶是 `2(p+1)+3` 而非
  `2p+3`——用错阶时 quad 上 2×2 规则积 4 次多项式（`u=(x²,0)` 在 RT0-quad
  得 2.6e-16，真值 4.6585e-3）。并行语义新增 `*_filtered` API（owned 谓词）。
  **结果**：pex4 误差从恒 0 → 与 C++ np2 **6~7 位一致**且每次加密减半；
  pex8 L2err 0 → 2.91987e-1（np1==np2 到 1e-5）；14 新测试含 MFEM harness
  对照 <1e-10。
- **N：`face_elements` 遮蔽陷阱消除（D20）+ Hex8 曲率修复（D26）**：
  固有方法改名 `face_adjacent_elems`（更新 4 代码 + 3 测试调用点；顺带修复
  被陷阱长期掩盖的 `autodiff_example.rs` 编译错）；`set_curvature_hex8`
  三个缺陷（边节点搜索用参考坐标比物理坐标 → 12 条边永不命中、trilinear
  回退的顶点 bit 约定与 Hex8 节点序不一致、边节点等距而非 Lobatto）修后与
  MFEM `SetCurvature` 节点差 **1.1e-16**（p=2，1 ulp）/ **逐位**（p=3）；
  单 hex 全部 216 个面 QP `detJ = +0.125`（修前 8 个 QP 精确为 0、最小
  −0.201）；第十轮的曲线 hex 边界测试变通已删除。
- **O：NURBS 二阶导 + ex1/ex3 本质边界**：`basis_funs_and_ders2` 差分模板
  在区段端点被夹断（有效步长 1e-14 却除 ε²）→ 改为解析 Piegl-Tiller A2.3
  （同 MFEM `CalcDnShape`）；6 knot 向量 × 11 采样（含端点）对照 C++
  ≤5.3e-14（修前端点 ~1.6e7）；`nurbs_printfunc` 与 C++ 45 行 ≤1.1e-15。
  ex1/ex3 补全本质边界后**收敛**（ex1: 16641 未知/512 本质 dof/110 迭代、
  ‖u‖_L2 与解析级数差 4e-10；ex3 2D/3D 均收敛、误差 O(h)）。
- **P：3D DPG 五个真缺陷 + acoustics_3d 真替换**：**根因 = hex 参考域错配**
  （fem-element hex 族在 [−1,1]³，而 dpg_basis 用 [0,1]³ 求积/参考坐标 →
  体积映射只覆盖单元 1/8、测度差 8 倍、测试 Gram 近奇异 `G[0,0]=3.5e-5`
  vs C++ 0.22）——修后 3D 声学 L2 由"随加密上升"变为**下降**；另修
  ① 3D 面法向多 0.5 因子（`cross3` = MFEM `CalcOrtho`，散度定理验证
  ∫x·n dS 由 1.5 → 3.0 = 3|Ω|）② `local_face_table` 3D 绕向非外向
  （改 MFEM `FaceVert` 表）③ `SkeletonSpace::new_h1` 3D 分支缺棱/面内部
  dof 且面 dof 序与基函数节点序不符（hex 2×1×1 p=1/2/3 = 12/43/96、tet =
  8/27/64，与 C++ `H1_Trace_FECollection` 逐位一致）④ `face_dof_params`
  三角面枚举非 `eval_face_lagrange` 的逆。`dpg_acoustics_3d` 升级为真
  UW-DPG：dof 数与 C++ 逐位一致、n≤3 的 L2 在 1% 内（n=4 偏 8%，归因于
  trial dof 编号排列差异，已写入文件头）。
- 回归基线（全绿）：fem-assembly **589** / fem-element **437** /
  fem-mesh **292** / fem-space **274** / fem-solver **244** /
  fem-parallel **225** / fem-io **122** / fem-linalg **63** / fem-amg **23**；
  **`cargo build --release --examples --keep-going` 0 错误**。

## 第十三轮完成（2026-09-10，三代理并行 + 主会话接手 D27）

- **D29 修复（quad ND2；k≥2 其余缺口转 D32）**：根因 = `HCurlSpace` 给 Quad4 分配 12 dof（对位 MFEM `ND_QuadrilateralElement(2)` 的 2p(p+1)=12）而装配映射 legacy `QuadND2` 只有 8 边 dof → 4 个内部 dof 载荷恒 0（35.8 垃圾解来源）；双线性泡泡基 `(1−y²)x^m` 亦未从 [−1,1]² 换算到 [0,1]²。修复 = `crates/element/src/nedelec/quad_nd2.rs` 扩为 12 dof 完整元素（协调内部泡泡 `y(1−y)·{1,x}`/`x(1−x)·{1,y}`，切向迹处处零）。**验证**：(1,0) 投影 tri/quad ND1/ND2 全部 ≤1e-15；单元 (K,M) 广义特征值 {0×5, 48, 87.096774, 128.571429} 与 MFEM **逐位一致**；E_exact 投影 ND2 较 ND1 优 ~2×、O(h) 下降；ex3 默认网格输出逐位不变（ND1 路径未受扰动）。element 439 绿。注意：ND 的 GetOrder()=p（不同于 RT 的 p+1）；向量质量/载荷须用 `VectorFEMassIntegrator`/`VectorFEDomainLFIntegrator`（`VectorMassIntegrator` 是 vdim 空间用）。
- **D28 修复**：`HDivSpace::interpolate_vector` 重写为对偶系统引擎：`d_i=u(x_i)·cof(J)n_i`、`W_ij=φ̂_j(x_i)·n_i`、小稠密求解、`g[dofs[i]]=signs[i]·c_i`（对偶矩阵行内容只影响非奇异性不影响精确性，**跨单元一致性**要求 slot↔物理采样点对齐；hex 面 (k+1)² 网格按面取向旋转/镜像 `transform_grid`）。**验证**：tri-RT0 (1,0) 2.0→**7.1e-17**；quad-RT1 1.7→**3.3e-16**；quad-RT2 0.49→1.6e-16；hex-RT0 2.02→3.4e-16；hex-RT1 O(1)→1.7e-15。space 275 绿。C++ 依据：`VectorFiniteElement::Project_RT`（fe_base.cpp:1179）单纯形 RT 全 nodal 法向采样；`RT_QuadrilateralElement` dof_map 负号⇒法向取反；`RT_TetrahedronElement::nk={1,1,1,−1,0,0,0,−1,0,0,0,−1}`。
- **D24 推进（dpg_maxwell_3d 真 UW-DPG 替换；整场 L² 未达标 → D35）**：`add_trial_trace_space_nd` 接入 `dpg_weakform.rs`/`complex_dpg_weakform.rs`（vec_phi=eval_face_nd+协变映射+σ fold+transposed param）；新增 `DpgCurl3dPairingIntegrator`；`dpg_maxwell_3d.rs` 全量重写为真 UW-DPG。**验证**：dof 数与 C++ 逐位一致（n2-o1:156/ref1:984/n2-o2:888/o2-ref1:6192）；单元级 exact-tuple 恒等式机器精度（hex/tet×p1/p2，A x = b 残差 2e-15）；PCG 22/22、48/50 同量级；整场 L2 n2-o1 1.753 vs C++ 1.723、o2-ref1 1.395 vs 0.271。消费方 acoustics_3d(95/1.213/17)/maxwell_2d(33/1.369/20)/poisson_2d 数字不变。两个非显然符号约定已固化入注释：① nd_face_dof_tangents 的 tk 取**参考面 loop 方向**（quad 边 2/3=(−1,0)/(0,−1)，对位 MFEM dof2tk）；② 3D TangentTrace 装配符号相对 elmat 公式**取反**（B_curl·x + B_tan·x = 0 实测）。
- **D27 结案（主会话接手，代理超时前留的诊断探针立功）**：D27 代理留下的探针暴露两层问题：① 探针自身把 `allreduce_sum_f64` 写进 `if rank == 0` 分支 → np2 在 allreduce 永久死锁（0 CPU；np1 单 rank 快速路径无此问题）；② 死锁修复后探针给出决定性证据：算子对称性/配对全部 ~1e-14（矩阵组装正确）、trace(M) 与 np1 一致，但 **L1(b0) np2=152.445 vs np1=152.969**——rhs 丢载荷。**根因** = `assemble_bdr_rhs_par` 按"单元 owner rank"积分边界面，但跨 rank 边界面的 RT1 边 dof 可能归邻居 rank own，贡献落在本地 ghost 段后被 `ParVector::from_local_raw(_, n_u, _)` 截断丢弃。**修复** = rhs 组装后补一次 `u_par.reverse_dof_exchange(&mut bdr_rhs_perm)`（ghost→owner 归还，对位 MFEM GroupComm::Bcast；B 的行归属单元 owner 天然完整、M 经 trace 验证完整，无需改）。**验证**：L1(b0) np2 == np1（1.52969e2）；u 误差 np2 1.95e-1 → **2.92e-5**；rtol 收紧到 1e-10 后 **np1/np2 逐位一致**（-r1: 9.67937e-6，-r2: 1.70021e-6，三档加密单调下降）；rtol=1e-6 下 np2 与 np1 的 ~1.5× 差异为 MINRES 停机噪声（np2 迭代多 ~45%，迭代方向不同），非系统性缺陷。np1 路径逐位不变（1.94103e-5）。**纪律沉淀：并行示例的 rhs 组装后必须 reverse_dof_exchange（凡按单元 owner 组装、dof 归属可能跨 rank 的载荷都适用）**。诊断探针（TEMP 块）已按死代码纪律删除。
- 回归基线（全绿）：element **439** / space **275** / assembly **589**（+6 个 HEAD 预存 ignored：lor_factory×3/hdg/contact/discrete_op debug）/ solver 244 / parallel 225 / mesh 292 / io 122 / linalg 63 / amg 23。

## 第十四轮完成（2026-09-11，四代理并行 + 主会话集成续作）

- **D32 结案（ND2 全几何 nodal 化，tri/quad/tet）**：根因升级表述——MFEM ND dof 真语义 = **点值泛函** `σ(Φ)=Φ(x_i)·t̂_i`（x_i = 边上 Gauss-Legendre 开点，t̂ = 未归一化参考切向；C++ harness 逐位确认），旧矩泛函 `∫Φ·t̂·t^m` 非反射不变 → pairing 破坏。round-13"中心化+单位化配方"失败点 = 单位化只做一侧产生残余 √2；正解是**根本不归一化**（两侧 J·t̂ 都精确等于物理边向量，点值 + 对称 Gauss 点使反向变换恰为符号反对角置换）。修复 = TriND2/TetND2/QuadND2 按 MFEM nodal 基重写（含 QuadND2 张量基 GL open × Lobatto closed、TetND2 面形心点值）；hcurl.rs 边 pairing 改 MFEM 编码（反向边 = 反序 + 符号 −1；k=1 保旧约定 ND1 逐位不变）；interpolate_vector k≥2 全链 nodal；3 点 Gauss 近似常数删除。**验证**：split 网格共享边 trace mismatch ≤1e-13（钉住）；tri ND2 Maxwell MMS rate −0.3 → **1.86**（mms 判据收紧，假绿钉死）；ex3 默认 ND1 输出逐位不变。
- **discrete_op ND2 侧移植（主会话续作 A 代理）**：`gradient_p2_nd2` 重写为点值 + 符号 scatter（删二项式变换）；`curl_2d_nd2_p1/p2` D 行改 canonical 点值（反向边 j_canon 从全局 id 恢复）；`curl_3d_nd2_rt1` 重写为直接形式（参考基 curl 在 RT1 dof 点的通量采样，删 20×20 D⁻¹ 回路）；hcurl.rs 新增 `face_tangent_anchor`。**div_rt1_p1_2d 根因 = 测试场 (x²,y²) 越出 RT1 空间**（RT 实现自洽，通量一致性 2.9e-15），测试换 in-span 场。round-14 期间 18 红（lib 9 + high_order 2 + patch 7）全清。
- **D35 结案（多 hex 反向 quad 面 trace）**：根因 = 反向面（Elem2，ori<0）的单元侧参考点 seed 用"反转 lf + 转置 fparam + Newton 反演"，0.5×1×1 hex210 上 Newton 落在 x_ref=−1+2e-16，而 HexNDk 基在 ±1 处分支差 0.94 → Elem2 trace 块污染（单 hex 全过、hex2 的 elem 2–7 挂的分层解释）。修复 = `dpg_basis::local_face_canonical_order`（MFEM `GetLocalQuadToHexTransformation` 的点矩阵直接插值语义；Newton 只留给曲线单元），四处 trace 路径改精确 seed。**验证**：hex2_p1 转正（F-rows 1.6e-15）；整场 n2-o1 **1.757 vs C++ 1.723（2.0%）**；-sc（complex 弱形式）== 未凝聚（解差 ~1e-12）；**dpg_acoustics_3d n=4 缺口消失**（ref0 0.7757 vs C++ 0.7765、ref1 0.4229 vs 0.4231，<0.1%——round-12 的 8%/60% 挂账结案）。
- **D30 结案**：`dpg_poisson_2d -o 3` 不再 panic（Gram 装配修复）；prob0 -ref2 收敛率 −2.97/−2.99 vs C++ −3.00，同 dofs 误差 6.919e-3 vs 6.932e-3（0.2%）。
- **D33 结案（TetRTk 基）**：两个叠加构造缺陷——①列配对丢失（消元产生列置换 P，旧代码用未置换下标）②右块未转置。修复 `tet_rtk.rs`（`sel.push(cp[c])` + `coeff[i*n+j]=row[j][mt+i]`）后基严格 `D_i(φ̂_j)=δ_ij` 且逐面支撑。tet RT0 插值 5.1e-1 → **7.1e-16**、L² 投影 → **0.0**；tet_rt1/tet_rt2 重写为 MFEM nodal 基（`mfem_nodal_dofs`/TET_NK 公共表）；tri_rt1 块序改空间 TRI_FACES 序。
- **D34 结案（语义统一到 nodal）**：C++ 裁定（harness t6 探针）RT_Triangle/Tetrahedron = nodal 通量对偶基、4.10 源码表构成单位对偶阵。discrete_op 的 RT1/RT2 读法全部改 `dmat[s][k]=sign_s·Φ_k(x_s)·(cof(J)·n̂_s)`，与 interpolate_vector / Piola 重构三方自洽（de Rham 恒等式精确）；`hdiv_interpolate_regression` 4 个 #[ignore] **全部转正**（tri/tet RT0/1/2 插值 ≤2e-15）；ex4 = 2.16192 逐位不变。
- **D25 结案（lor AMG）**：`LorAmgPrecond` + `LorElasticityPrecond` 换 `fem_amg::CorrectedAmgPrecond`（消除粗解置换缺陷的保险；现测试规模下迭代数逐位不变——LOR 层最粗层小，Rcm≈恒等）。plor_solvers 迭代 5/4/5/5/6（rs 0..4）、lor_elast 122/145/131 有界。lor_factory 3 个 ignored 转正失败（走 vendor AmsPrecond/AdsPrecond，不经 LorAmgPrecond；ND hex 残差 8.4e-1）→ 保留 ignore 并把复测数字写入注释（见 D40）。
- **D31 核查完成（精确置换表已备，改动留专项）**：HexQk→MFEM H1 hex 闭式置换 = 边块映射 `[9,10,11,8,3,1,5,7,0,2,6,4]` 且 ei∈{1,3,5,6,7} 块内反序；面块 `[4,2,1,3,0,5]` 且 f=2 行翻、f=3 列翻；顶点/内部恒等（p=2..5 与 C++ dump 全 MATCH，p=2 硬表已钉进 factory.rs 测试）。影响面：① `io/mfem.rs` 高阶 hex `nodes` 读取局部置换 + 全局 face 编号双重失配（读弯曲 hex 网格静默错乱，建议先加拒绝/告警，见 D41）② `HexQ2`/`HexQ3` 手写布局是第三种序（mixed/partial 消费）③ HexQk 内部序改动三者原子迁移、留给 GLL 对齐专项（D7/#10）。
- 回归基线（全绿，`cargo test --lib`）：amg 23 / assembly **585**+6 预存 ignore / element **446** / space 275 / solver 244 / parallel 225 / mesh 292 / io 122 / linalg 63；**`cargo build --release --examples --keep-going` 0 错误**。⚠️ 勘误：round-13 记录的 assembly"589"系多 package 输出归行误差，stash 实测 HEAD（af16729）= 585+6=591 与本轮逐测试一致，**零回归**。
- 抽查（主会话复跑）：dpg_maxwell_3d n2-o1 = 1.757e0 ✓；dpg_acoustics_3d ref0 = 7.757e-1 ✓；ex4 = 2.16192 ✓；ex3 默认 = 3.9163e-1 ✓。

## 第十五轮完成（2026-09-11，四代理并行 + 主会话集成；含 D36 归因更正）

- **D39 结案（real DpgWeakForm 的 -sc）**：两个机械 bug——① `lu_solve` 前代累加 `y[j]` 而非已回代的 `b[j]`（丢 `L[i,j]L[j,k]` 耦合项，n≥3 即错；complex 版用的正是 `br[j]`，是正确基准）→ p=1 时该耦合项恰好为 0，故 round-14 误判"p=1 正常"；② `form_linear_system` 两个分支都缺 MFEM `PartMult` 赋值（`weakform.cpp:561`：`mat_e->AddMult(x,b,-1); mat->PartMult(vdofs,x,b)`），实测这一项**单独**就解释了 p=1 的 46% 偏差（2.836→1.940）。**验证**：-sc == 未凝聚，poisson -o 1/2/3 解相对差 1.1e-14 / 3.2e-13 / 3.3e-12（miniapp 端受求解器容差限制 1e-11 量级）；收敛率不劣化（o3 rate −2.97 vs C++ −2.98）；-sc 的 PCG 迭代反而更少（20/56/113 vs 30/78/154，与 C++ 趋势同）。新增 `tests/dpg_static_cond.rs`（含 complex 基准锁定）。
- **D41 结案（io 弯曲 hex `nodes` 读取）——不是安全网而是**逐位**正确实现**：`build_h1_geometry` 新增 `build_h1_hex_geometry`，复刻 MFEM H1 `nodes` 编号（顶点 | mesh 边块 | mesh 面块 | 单元内部块；边/面按元素遍历 + MFEM `CUBE::Edges`/`FaceVert` 首见规则枚举；边槽从较小顶点 id 端起算 `cor = v[e0]<v[e1] ? 1 : −1`；面槽用首见元素的 `FaceVert` 参数化，其它元素按角点匹配旋转/反射重建 = `QuadDofOrd[orientation]`）。**验证**：4 组夹具（cube P2 8 hex、P3 笛卡尔、P3 fichera 不规则、P3 顶点 id 反序）与 MFEM 4.10 harness 逐槽 **max|Δ| = 0.0（逐位）**；修复前探针在 P3 上 max|Δ| = 1.0e0、cube P2 恰好 0.0（这就是它长期静默的原因）；夹具含反射面（12/48、1/12、9/42），变异测试（关掉边反转分支）只让反序顶点用例失败 ⇒ 夹具具判别力。安全网同样落地：不可映射/混合含 hex 的网格一律 `warning (D41): refusing...` + `geometry = None`（退化为直线顶点，永不静默错乱）。
- **D25/D40 深化（LOR-AMG 与 vendor AMS/ADS 取证）**：D40 判定 **vendor AMS/ADS 无辜**——bisection 显示 AMS 作用在 LOR 矩阵上健康且网格无关（FGMRES(30) rtol 1e-8：n=882/6084 → 33/43 迭代；RT 的 ADS 30/52），而 HO 层经 `LorSolver` 包装即残差 9.06e-1/9.42e-1。进一步诊断：`perm` 是合法带符号双射（882/882，96 负，无重复/未设）；去掉/取逆/单侧 equilibration 均仍失败；**对角传递模型本身无效**（同一光滑场插进两空间后 `|x_HO[|p_i|]/x_LOR[i]|` 散布 1.6×（常值场）… 35×（光滑场））。**根因 = fem-rs 的 LOR 空间未用 MFEM 的 same-functional 约定**（`fem/lor/lor.cpp` 的 `ConstructDofPermutation` + GLL/IntegratedGLL 基对），故包装所需的同余 `A_LOR ≈ S Π A_HO Πᵀ S` 不成立。判定：不可在 `lor_factory.rs`/`solver`/`amg` 内修（根修在 `crates/space/src/lor.rs`）→ **结论式 ignore**（三测试保留 #[ignore] 并写入机制/数字/方案/工作量 1–2 天）。
- **navier 首个 miniapp 交付（最大剩余块开工）**：新增 `crates/solver/src/navier.rs`（1204 行，`navier_solver.{hpp,cpp}` 1:1：BDFk/EXTk 三分支系数含变 dt、`UpdateTimestepHistory`、三次求解与 MFEM rtol/迭代数、`Mv⁻¹`/curl-curl/BDF 代数、`Orthogonalize`/`MeanZero`/`ComputeCFL`/`PrintInfo`/`PrintTimingData`，trait `NavierDiscretization`）+ `miniapps/fluids/navier_kovasznay.rs`（1044 行，1:1 + `[H¹]²×H¹` 离散）。**验证**：与 C++（WSL 串行 harness）**10 步内 err_u/CFL 打印 6 位逐位一致**（step1 err_u 6.24210E-07 == C++、CFL 6.23E-02 ==；err_p 末两位差异）、MVIN/HELM 迭代数相同、PRES ±2（GS 顺序依赖）；`-cr` 双侧 exit 0。fem-solver 250 绿。
- **主会话集成修复（navier 代理挖出的真 bug）**：`VectorConvectionIntegrator`（`standard/vector_convection.rs`）与 `VectorConvectionNLFIntegrator`（`standard/misc_integrators.rs`）用 `qp.weight` 而非 `qp.ref_weight`。**真因不是"该用 phys_weight"**：assembler 的 `grad_phys` 是 **adjJᵀ∇φ（|detJ|-scaled）**而非常物理梯度（`assembler.rs:900-930` 注释明确两族约定），故 convection 族必须配 bare `ip.weight`；实测 `weight` → max|(Ku)_x − Mx| = 8.889e-1、`phys_weight` → 2.963e-1、`ref_weight` → **≤1e-12**（决定性判据：w = u = (x,y) 时 `(Ku)_x = M_scalar·x`；3×2 非单位尺寸网格暴露，2×1 恰好 |detJ|=1 故通过）。新增回归测试钉住；assembly lib 589 绿。
- 回归基线（全绿）：amg 23 / assembly **589**（单跑；+6 预存 ignore）/ element **448** / io **123** / linalg 63 / mesh 292 / parallel 225 / solver **250** / space 275；**`cargo build --release --examples --keep-going` 0 错误**（新注册 `navier_kovasznay`）。
- 抽查（主会话复跑）：navier_kovasznay step1 CFL 6.23E-02 / err_u 6.24210E-07 **与 C++ 逐位一致** ✓；`dpg_poisson_2d -o 1` 未凝聚 30 it vs `-sc` 20 it（新集成测试证明解一致）✓；D41 `curved_hex_nodes_match_mfem_slot_by_slot` 等 3/3 ✓；D36 `d36_hex_nd_nodal` 4/4 ✓。

## 第十六轮完成（2026-09-11，四代理并行 + 主会话集成）

**⚠️ 本轮有一个影响面最广的发现：D-shaped 的 L2 误差度量错误。** fem-rs 的 dpg miniapp 用 `2·order+4` 测 L2，MFEM `GridFunction::ComputeL2Error` 用 `2·fe->GetOrder()+3`（`gridfunc.cpp:3410`）。把 fem-rs 的解代回 MFEM 自己的 `ComputeL2Error` 得到的是 **C++ 的值** ⇒ **解一直是对的，误差度量是错的**。据此更正的既有数字：`dpg_maxwell_2d` 1.369 → **1.381e0（= C++ exactly）**、`dpg_poisson_2d` o1 1.940 → **1.954/1.021/5.149e-1（= C++ 全打印位）**、`dpg_acoustics_2d` **1.222e0/8.008e-1/4.472e-1 = C++ exactly**。5 个 dpg miniapp 的 `2·order+4` → `2·order+3` 已全部更正。

- **D45 结案**：real `DpgWeakForm` 新增 `add_trial_trace_space_h1`（complex 版早已用 `new_h1`），poisson 的 û 切换到 H1-trace 连续空间 → **dof 偏移 [0,4,12,21,33] 与 C++ 逐位一致**、PCG 迭代 **23/66/132 = C++ 精确**。另修三个独立 porting 缺陷：① volume LF 求积阶（fem-rs 默认 6 过度积分非多项式 RHS，MFEM `DomainLFIntegrator` 恰为 `2·fe_order`；修前 σ-RHS 差 2.2e-4 → 修后 1e-16）；② 上述 L2 误差度量；③ face 求积阶（默认固定 4 < MFEM trace 规则 `test_order+p−1`，o3 差 0.18%）。**结果**：poisson o1/o2 与 C++ **所有打印位一致**（raw err_u 0.4700557125278986 vs C++ …8579，13 位），o3 差 0.42%→0.18%；`-sc==未凝聚` 保持 3.6e-14/2.3e-13/5.7e-12。
- **D36 再次更正（重要）**：**G 与 trial 侧被排除**。新增 `tests/dpg_test_norm_regression.rs`（3 测试）钉住装配好的复数法方程块，与 C++ 逐 entry 对照 **max|Δ| = 1.9e-15（o1，900 nnz）/ 6.9e-15（o2，20734 nnz）**，8-hex 上块不变量全同（变异测试：删一个 graph-norm cross 积分器 → 三项全失败，1e-3…2.6e-2，测试有效）。**结论：整个装配矩阵（含 4 个 cross block 与 ND-trace trial 侧）在 o1/o2 都与 C++ ≤7e-15 一致 ⇒ `maxwell_3d -o 2` 的差距在 miniapp 复数路径的下游**（RHS `(J,G)` 与倍增系统偏移/BC 接线）：矩阵相同而解不同（E re/im 范数 4.8373/7.0289 vs C++ 4.8371/4.9530；H 4.9509/7.0517 vs 6.7699/7.0507），而多项式块与 LF 求积规则一致。下轮用既有 `mx3/mwdump.cpp`（打印 `CPPSUM` 逐层块范数）对照 RHS 即可。
- **D38 结案（tri/tet ND k≥3 nodal 化）**：`TriNDk`/`TetNDk` 逐行 1:1 移植 MFEM `ND_TriangleElement(p)`/`ND_TetrahedronElement(p)`（hierarchical Chebyshev raw 基 `T_j(2x−1)` + 方形 Vandermonde 求逆；DOF = `FE::Nodes` + `dof2tk` 点值泛函）；旧的 monomial+moment 机器整段删除。**对照**：dof 点位与 C++ **逐位 0.0**（Tri ND3 / Tet ND3 / Tet ND4）、VShape+curl ≤1.3e-13；共享边/面 trace TriND3 2.2e-15、tet ND2 1.6e-15、tet ND3 4.2e-15；`interpolate_vector` k=3 常量/线性/二次场均 ≤1e-11（实测 ~1e-15）；**k=1/2 路径逐位不回归**（ex3 默认 3.91630900027325e-1 逐位相同）。顺带发现旧 k≥3 元素的 C 矩阵实为 `V_sel⁻¹`（注释写 `V⁻ᵀ`）且 tet 列置换索引写错 ⇒ **旧 k≥3 基本身就是错的**（此前无任何对照测试）。element 454。
- **D37 能力落地（未默认启用，见 D48）**：装配层新增面 dof **2×2 块变换**（`apply_face_block_transform_*`、`assemble_bilinear_nd_canonical`/`assemble_linear_nd_canonical`，+163 行）。关键代数修正：`u_local = S·u_canon ⟹ A ← Sᵀ·A·S、b ← Sᵀ·b`（首版误用 `S A Sᵀ`，被"逐元素质量阵 vs 独立求积"探针抓到 3.75e-2 → 6.9e-17）。**验证**：tet ND2 curl-curl MMS L2 4.570e-1→9.809e-2→4.339e-2（**rate 2.22/2.01**）、空间内场精确解 2.8e-15、同网格 canonical vs element-local **11.1× 改善**（9.81e-2 vs 1.0909e0）、tet ND2 面块 trace 1.519e0 → **1.554e-15**。未默认启用原因：canonical 基下**所有重建消费方**必须做 `u_local = S·u_canon`（`postproc/grid_function.rs`、`postproc/postprocess.rs`、`boundary/vector_boundary.rs`、`mixed/mod.rs`、`hybridization/trace.rs`、`dpg/dpg_basis.rs`、examples 自带 L2 评估器）——与 MFEM `DofTransformation` 同为横切改造。
- **D42 结案（权重约定审计）——又发现 4 个同款 bug**：逐积分器判定（判据：非单位尺寸网格 + 齐次缩放的 `s^(d−2p)` 律 + `MassIntegrator` 参照）。**修复**：`NormalTraceJumpIntegrator`、`NonconservativeDGTraceIntegrator`、`SBM2DirichletIntegrator`、`SBM2NeumannIntegrator`（都是 φφ 型，应用 `phys_weight` 而用了 `weight`）——**实测 9× 偏差（max|K−Mass| = 1.185e0，恰为 |detJ|⁻²）→ 修后 ≤1e-12**。**判为正确无需改**：`VectorDivergenceIntegrator`（对 adj-grad 二次，`weight` 自洽；但**不是 MFEM 同名类**——MFEM 的是矩形 `(Q∇·u,v)`）、`ConvectionIntegrator`/`DiffusionIntegrator`/`ElasticityIntegrator`/`TensorDiffusion`（分类一致）、全部 `VectorQpData` 路径（只有一个 weight 字段，装配时恒为物理测度）、全部 `LinearIntegrator`/`Boundary*`/`bbar`/`shell_mitc4`/`sbm3_*`、PA 算子（`PAConvectionOperator` 等全部正确）。**根源**：`crates/assembly/src/integrator.rs:17` 的 `QpData::weight` 文档说"quadrature weight × |det J|"，与代码（体积路径是 `ip.weight/|det J|`）矛盾——**这个文档 bug 就是整个家族的成因**（见 D51）。新增 5 个回归测试。
- **D43 结案（弯曲 tet nodes 读取，逐位）**：新增 `build_h1_tet_geometry`/`tet_slot_map`（D41 hex 的 tet 类比：MFEM 实体枚举 `TET_EDGES`/`TET_FACES` 首见规则、边槽从较小顶点起算、面块按 `TriDofOrd[orientation]` 重键、内部块按参考序）。**额外发现的缺陷**：MFEM `Mesh(f,1,1)` 除 `MarkForRefinement` 外还跑 `PrepareNodeReorder`/`DoNodeReorder` **重编号 `nodes` 网格函数**以保持几何——故映射器在**旋转前**的网格上求槽位映射（其单元序恰等于文件序）并按物理槽键 `(element, 排序顶点标号下的重心模式)` 重挂文件节点值。**验证**：5 组夹具（escher-p2 42 tets、P3/P4 直线、P3 弯曲、P3 顶点 id 反序）**dof pick / nodes 向量 / 映射全部 max|Δ| = 0.0**；修复前首个节点就错 5.030e-1（与 D43 记录的"11/42 单元、max 1.26"一致）。安全网保留（含 tet 的混合网格拒绝）。
- **D44 结案（`dpg_poisson_2d -tri` panic）**：`tri_rtk.rs` 三个缺陷（D33/TetRTk 修复的镜像）：① **OOB 根因** = 单项式表尺寸写成 `(k+1)(k+2)` 而试验空间需 `[P_k]²` + `x^a y^b·(x,y)` 泡泡块，实为 `3(k+1)(k+2)/2` ⇒ k≥1 全越界；② 转置系数提取（`row[i][mt+j]` 应为 `row[j][mt+i]`）；③ 无行主元 + 静默 `continue`（基损坏而不报错）且 `eval_div` 表尺寸用错。修后 `-tri` 三档**最优收敛率**（o1 −1.04/−1.02、o2 −1.99/−2.00、o3 −2.98/−3.00）；quad 路径类型级不受影响（`TriRTk` 只对 Tri3|Tri6 实例化）。新增 `basis_dual_to_moment_functionals` 等测试。
- **D46 大部结案（navier 内核缺口）**：① `mixed::ref_elem_vol` 按 order 分派（0–3 保原固定表逐位，4..=10 走 order-generic H1 元素；新增 `REF_ELEM_VOL_MAX_ORDER = 10`）；② `ref_elem_face` **修正了 round 15 的建议方向本身**（走 `factory::ref_elem` 会引入新 bug：`SegPk` 等距节点 vs MFEM 边界单元是体积单元的**迹**——闭 Gauss-Lobatto 点 + 拓扑 DOF 序；等距面基下 `Σᵢrhsᵢ` 仍精确但通量被分摊到错 DOF，kovasznay order 6 实测逐 DOF 偏差 0.1，只有逐 DOF 比对才能发现）→ 新增私有 `H1SegPk`（在体积 `QuadQk(p)` 底边取迹，构造上即 `H1_SegmentElement(p)`）、删除错误的 `SegP3` 表项、新增通用 `face_dofs_h1(space)`；③ 新增 `VectorBoundaryNormalLFIntegrator`（对齐 MFEM `BoundaryNormalLFIntegrator(VectorCoefficient&)` = `∫(v·n)φ ds`，**未改动**现有标量同名类型的语义）；④ `Mesh::face_elements` 未构建 `face_to_elem` 时 `debug_assertions` 下打印一次告警；⑥ `G ≠ Dᵀ` 的性质已在 `navier.rs` 模块节与 trait 文档写清（`Gᵀ = −D + B`，`B[i,(k,c)] = ∫_Γ φᵢφ_k n_c`，即 `FText_bdr`/`g_bdr` 携带的通量）。⑤ `NonlinearForm` 未修，但方案明确（见 D52，工作量约 0.5 天）。
- **navier 第二件交付：`navier_mms`**（1173 行，`navier_mms.cpp` 1:1）：与 C++ **step1 err_u 2.75455E-08 / err_p 1.23108E-04 逐位一致**，`MVIN`(8) 与 `PRES` 每步完全相同（66,66,66,67,73,73,73,63,61,61）、`-o 3 -rs 2`（256 元）step1 亦逐位（2.87613E-08）；两边 `-cr` exit 0。`navier_kovasznay` 把 `g_bdr` 切到内核新件后仍保持 round 15 的 6 位一致（step1 err_u 6.24210E-07 / CFL 6.23E-02）。
- 回归基线（全绿）：amg 23 / assembly **605**（单跑；+6 预存 ignore）/ element **454** / io **123** / linalg 63 / mesh 292 / parallel 225 / solver 250 / space 275；**`cargo build --release --examples --keep-going` 0 错误**（新注册 `navier_mms`）。
- 抽查（主会话复跑）：navier_mms step1 err_u 2.75455E-08 / err_p 1.23108E-04 ✓（= C++ 逐位）；`dpg_poisson_2d -o 1` PCG 23（= C++，修前 30）；`dpg_maxwell_2d` **1.381e0**（= C++，修前 1.369 是错度量）；D37 7/7、norm 3/3、D38 4/4+6/6、curved tet/hex nodes 各 3/3 ✓。

## 第十七轮完成（2026-09-11，四代理并行 + 主会话集成）

- **D36 结案（三轮归因的终点）**：根因是 **miniapp 制造解 RHS 写错**——`miniapps/dpg/dpg_maxwell_3d.rs` 的 `Exact::j` 第一分量多了一个负号（`−ωs(ε−2/μ)` 应为 `+ωs(ε−2/μ)`；`J_r(0)=ωεE_i(0)−2ωE_i(0)/μ`）。C++ 探针 `Probe : ComplexDPGWeakForm` 暴露 `y_r/y_i`，在 `x=(0.1,0.2,0.3)` 得 `J_r[0]=−3.69316…` 而 miniapp 给 `+3.693…`，其余 5 分量一致；RHS 分块范数呈"实部/虚部各半对"的指纹（`b_r=B_rᵀf_r+B_iᵀf_i`、`b_i=B_rᵀf_i−B_iᵀf_r` ⇒ 所有 `f_r` 项同错）。**弱形式内核、BC 接线、`to_real_block_csr` 均无需改动**（已验证无辜）。另修测量侧：`dpg_basis::vol_quadrature` 的 Hex8 分支把参数当**点数**（其余几何当**精确次数**）→ 加 `mfem_l2_rule_order()` 换算，使 L2 用 MFEM `ComputeL2Error` 同一规则。**结果**：`-o 2` 由 5.15× 偏差 → **全部对齐打印精度**（9.547e-1 / 2.707e-1 = C++，rate −1.95 对齐，PCG 66/119 vs C++ 66/118），`-o 1` 由 1.757 → **1.723**（= C++）。诊断链与实测表已写入文件头。
- **D47 修复**：`space/src/constraints/dirichlet.rs` 的 `boundary_dofs_hcurl` 原来只收边 + hex 四边形面 dof，tet 三角面（3 节点）被静默跳过；改为按面节点数分派（3 节点面用 `face_dof` + `face_anchor().n_points()`）。回归测试 `d47_boundary_dofs_tet_face`：cube tet ND2 边界 dof **36 → 60**（= 2×18 边 + 2×12 面）。**诚实数字**：ex3 tet ND2 仅 3.8599e0 → 3.8490e0（0.28%）——面 dof 是次要项，主导是 D48。
- **D48 决定性修复（canonical 面块变换接入 ex3，默认启用）**：MFEM 语义核定 `T` 是 local→global（`v_t=T·v`、`A_t=T⁻ᵀAT⁻¹`，`fem/doftrans.hpp` + `gridfunc.cpp:980`），fem-rs 的 `FaceDofBlock`(S) = `T⁻¹` ⇒ round 16 方向正确无需反向。改动：`vector_assembler` 新增 `nd_element_local_dofs`/`nd_element_local_dofs_signed`/`apply_face_blocks_to_local_dofs`，两个 canonical 入口补齐与默认入口一致的按积分器求积阶分派；`examples/mfem_ex3_maxwell_cavity.rs` 统一开关（**默认 canonical=true**，`-no-canonical` 退化），并修 L2 评估器三处（参考元按 order——原 `TetNDk::new(1)` 把 ND2 前 6 dof 与 ND1 基缩并使 `-o 2` 数字无意义；积分阶 `2k+3`；重建 `u_local=T·u_canon`；hex 用三线性 Jacobian——原代码取 0..3 号节点使行列式为 0）；PCG 不收敛改按 MFEM 打印并继续。**结果**：`beam-tet ND2`（166688 dofs）**3.6922e0 → 1.57652991777285e-2**（C++ `0.01576587344824001`，相对差 **3.6e-5**），**同网格 canonical vs 元素局部 234× 改善**；ND1 与 canonical 逐位相同（3.91630923150634e-1 = C++ 8 位）；3D tet MMS 收敛率 L2 `[0.13145, 0.06255]` 比 2.10、curl 比 2.08。
- **D49 结案**：新增 `H1TetPk`（MFEM `H1_TetrahedronElement` 的 GLL 节点 + 层级 Chebyshev 基 Vandermonde 逆，按 order 缓存）——**几何路径专用**，`factory::TetPk` 未动（保所有 L2/DG/tet 解语义不变）。切换 `io/mfem.rs` tet 几何（槽位改用整数标签，消除浮点取整误差路径）与 `assembler::geo_ref_elem`。**弯曲 P3 夹具映射误差 7.53e-2 → 2.442e-15**（顶点反序同）、直 P3 2.44e-15、直 P4 7.11e-15、dof picks 保持逐位 0.0；测试由"仅报告"改为断言（TOL 1e-12）。
- **D50 完成（3D 边界装配可用）**：`boundary_face_geom` 扩到 `dim=3`（`t0×t1` 归一化为法向、`|J_face|=|t0×t1|`、bilinear/affine 面映射；角点经 owner 单元读取以支持周期接缝的 order-1 几何快照）；3D **面基 = 体积单元在面上的迹**：quad 面用 `QuadQk`（[0,1]²，删掉 [−1,1]² 的 `QuadQ1/Q2` 面项）、tri 面用新私有 `H1TetFacePk`（**刻意不用 `H1TriPk`**——fem-rs 的 H¹ tet 是等距而 `H1_TriangleElement` 是 GLL，用 GLL 面基会变成"与空间基在面上限制不同的函数"，正是 D46② 的陷阱）；`face_dofs_h1` 推广到 2/3/4 节点面。**验证**：离散散度定理**逐 DOF** hex ≤2.0e-16（p=1..6，含 2×2×2+1 加密）、tet ≤2.2e-16（p≤4）；全局 `Σ∮(x,y,z)·n = 3|Ω|` 精确打印 3.000000000000 ⇒ 加密后外法向成立；与 MFEM harness 逐 DOF 对照：**hex p=1..6 全匹配**（节点 ≤1.1e-16、值 ≤1.7e-16），tet p=1/2 精确、p≥3 因 H¹ tet 解空间仍等距（属 D49 上一层，fem-rs tet 边界装配自身自洽且散度定理精确）。
- **D51 完成**：`integrator.rs` 重写 `weight`/`phys_weight`/`ref_weight` 三字段文档（`weight` 按装配路径分列两种约定、给出**按被积函数选择**的判据与**反例警示**：mass 型误用 `qp.weight` 差 `1/|detJ|²`，3×2 网格实测 9×，并列出 round 15/16 因此被改的 6 个积分器）；`standard/mod.rs` 加 "Which quadrature weight?" 判据表。
- **navier 第 3/4 件交付**：`navier_shear.rs`（1049 行，双剪切层全周期 1:1）与 `navier_kovasznay_vs.rs`（1258 行，Kovasnay Re=40 + **自适应时间步**：provisional step + CFL 接受/拒绝 + dt 预测 + 历史排队）。**对照**：shear 的 CFL/迭代数/各 L2 范数**全 10 步与 C++ 逐字节一致**（cfl 7.56030E-02、MVIN 4/9、PRES 47/76、HELM 6/6）；vs 的 **CFL/Time/dt 全 5 步逐字节一致**、err_u 到 6 位。`navier.rs` 新增 `dthist()` 访问器 + 纯周期路径 lib 测试。既有 kovasznay/mms 逐位不劣化。solver 251 绿。
- **主会话集成处理**：① **共享 stash 栈被污染**——某代理 `git stash push/pop` 期间栈顶被换成另一分支（`ex4-ads-preconditioner`）的条目，产生 17 项冲突（`examples/compare/compare.sh` UU + 16 个 `*.000000` DU）；已 `checkout HEAD --` 恢复 compare.sh、`git rm -f` 删除 16 个输出产物，**`stash@{0}` 原封保留未动**（⚠️ 见 §备忘）。② `H1TetPk` 补进 `lagrange/mod.rs` 的 re-export。③ 修 `hcurl.rs` 的 `FaceDofBlock` 文档（`A_canon` 公式转置写反：应为 `Sᵀ A_local S`）。④ 修 `vector_assembler::geo_ref_elem_from_mesh` 的 tet 臂（`factory::TetPk` → `H1TetPk`，与 `assembler.rs` 对齐，D49 的跨文件一致性）。
- 回归基线（全绿）：amg 23 / assembly **605**（单跑；+6 预存 ignore）/ element **454** / io **123** / linalg 63 / mesh 292 / parallel 225 / solver **251** / space 275；**`cargo build --release --examples --keep-going` 0 错误**（新注册 `navier_shear`、`navier_kovasznay_vs`）。
- 抽查（主会话复跑）：`dpg_maxwell_3d -n 2 -o 2` = **9.547e-1 / PCG 66**（= C++）✓；`navier_shear` CFL **7.56030E-02**、`|u|_L2 9.31620E-01`、MVIN 4/PRES 47/HELM 6 ✓（= C++）；`cargo test -p fem-assembly --lib` 605 ✓。

## 第十八轮完成（2026-09-11，四代理并行 + 主会话集成；本轮起禁用 git stash）

- **D56 结案（周期网格 DOF 坐标）**：根因 = `DofManager` 全部 builder 的 `dof_coords` 从**折叠后**的 `mesh.node_coords()` 构造——顶点 dof 拿到单一周期像、**边 dof 得到折叠弦中点**（如跨缝边 x=0.45，真实 0.95）。MFEM 语义（`mesh.cpp:6205` 先 `SetCurvature` 物化 per-element Nodes 再合并顶点；`gridfunc.cpp:2473` `ProjectCoefficient` 逐单元求值后写者胜）⇒ DOF 坐标正确来源 = **owner 单元的几何节点**。修复 = `DofManager::new` 按特征签名（`geom_order==1 && geom_n_nodes != n_nodes`，周期 order-1 快照特有）门控进入 `rebuild_dof_coords_periodic`：按单元序经**自身**几何节点仿射映射重算物理位置、按序覆写（槽位数不符的单元跳过保护手写布局）。**验证**：周期 quad Q1–Q4 / tri Q1–Q3 插值 max diff **1.414e0 → <1e-12**；MFEM harness 的 Project 逐单元值表全部一致；**`navier_shear` 删除本地 `project_vel` 绕过、改回 `interpolate_vec` 后 step1 = cfl 7.56030E-02 全列与 C++ 逐字节一致**。`schrodinger_flow` 2D 路径已去绕过（3D 绕过是"局部张量 H¹ 算子"整体实现，非插值问题，见 D61）。
- **D58 落地（canonical 推广为默认装配/重建入口）**：`FESpace` 新增 `element_face_blocks(&self, _elem) -> &[FaceDofBlock]`（默认空 ⇒ 无 blocks 空间零开销逐位不变）；`HCurlSpace` 7 行转发覆写。默认 `assemble_bilinear`/`assemble_linear` 逐元素取 blocks 走块变换；重建消费方逐个改造：`postproc/grid_function.rs`（evaluate_vector/curl、`compute_l2_error_hcurl`）、`postproc/postprocess.rs`、`boundary/vector_boundary.rs`（边界阵 `K←SᵀKS`、载荷 `Sᵀf`）、`mixed/mod.rs`（HCurl/ND 列 `M←M·S`、行 `SᵀM`）；**HDiv-only 路径 no-op**（永无 blocks）；**`discrete_op.rs` 核定无需改动**（round-14 nodal 语义下 ND 列本就是 canonical 泛函 = MFEM TransformPrimal 后的算子）；`hybridization/trace.rs` 2D 无 blocks no-op。**验证**：ex3 走**默认入口** = 1.57652991777285e-2（与 round 17 专用入口**逐位一致**，`-no-canonical` 仍 3.6922e0）；新增 2 个 D58 pin 测试（默认 vs 专用 diff == 0.0 逐位）；d37 套件 9/9。⚠️ hcurl.rs 的 7 行转发是 D58 唯一越界文件改动（当时该文件将被 D55 路使用），主会话追认。
- **D54 结案**：`vol_quadrature` Hex8 分支改委托 `hex_rule(order)`（次数语义，`(order+2)/2` 点）；删除 miniapp 的 `mfem_l2_rule_order()`。**`-o 2 -do 0` ref0 的 0.06% 残余归零（9.482e-1 = C++）**，全部六档 = C++；`dpg_test_norm_regression` 3/3、`dpg_maxwell_3d_identity` 15/15、`dpg_acoustics_3d` 1.212 = C++。
- **D55 关案（假设被 MFEM 源码否定）**：**hex ND 没有 DofTransformation**——`DofTransformationForGeometry` 对 tensor 几何返回 NULL，`TransformPrimal` 只处理三角形面；hex 四边形面跨元素用 `ND_FECollection::QuadDofOrd[ori]` 的**带符号置换**（fem-rs 的 `match_face_dof` 已是该机制），给 hex 加 2×2 面块反而偏离 MFEM ⇒ 保持空块是正确语义（文档已写入 hcurl.rs）。**真根因 1（内核级，主会话集成修复）**：`CurlCurlIntegrator::integration_order` 对所有几何返回 `2k−2`，而 MFEM（bilininteg.cpp:2204）按 `el.Space()` 分派：**Qk（`ND_QuadrilateralElement`/`ND_HexahedronElement`/`ND_WedgeElement`，经 `VectorTensorFiniteElement`）= `2k`，Pk（单纯形）= `2k−2`**——hex curl-curl 欠积分（k=2 时 2 点/向 vs 3 点；单 hex fro² 106.67 vs C++ 142.77 实锤）。修复 = `VectorBilinearIntegrator` 新增 `integration_order_for(space_order, elem_type)`（默认委托 `integration_order`，加法式），`CurlCurlIntegrator` 按 tensor/simplex 覆写，`vector_assembler` 两处 bilinear 派发点传入 `element_type(0)`。**真根因 2**：`solve_report_2d` 重装配未消元矩阵 + 消元右端 = 错系统（与 D57 共因）。ex3 侧已有 `assemble_mat_mfem_rule` 临时规避，现内核修复后两条路一致；`d55_hex_nd2_system_regression.rs` 3 测试钉住。
- **D57 结案（ex3 2D 411×）**：两个 bug——① `l2_err_2d` 的协变 Piola 重建用了 `J⁻¹` 而非 `J⁻ᵀ`（伴随没转置；直角三角形上不可见，剪切三角形出错）；② D55 真根因 2 的同一错系统。**验证**：2-tri 方形 interp L2 1.6262 → **0.5281841004551823**、solve 0.4916776270584529 与 C++ **全位一致**；`beam-tri -o 1` = **0.0801477893043346 vs C++ 0.08014778969562512（4e-9 相对）**；rf0–3 消元不变量+稠密解逐级一致（`d57_ex3_2d_regression.rs`）。
- **D40 完成可修部分（诚实 reframing）**：round 17 结论只对一半——除泛函对应错误外还有两处独立实现缺陷。本轮修复：① `pair_sign()`（用两个空间 canonical `interpolate_vector` 对常场的取值符号度量相对定向 = MFEM `s1·s2·s3·s4`）替换几何 vertex-id 符号推断；② ND 内部 slot 表 **i1/i2 转置**修正（匹配 `HexNDk::eval_basis_vec` 的 z-outer/y-inner 序）；③ RT hex 内部表按 `HexRTk` 实际布局重写；④ `LorSolver` **删除 S 对角平衡**（MFEM 无此物；实验证明有害：k=2 时 S=I 收敛 138 iters、S=diag 失败）。**验证**：常场配对幅度精确（ND k=2 全部 2.0000、RT q=1 全部 1.0000）、负积 0/882、`cos(Πᵀu_HO, u_LOR)=1.000000`（修前 0.16）；perm 结构与 MFEM dump 对齐（RT1 4×4×4: 1728/0 neg 两边同、ND2 4×4×4: 1944/972 两边同）；铅笔质量 exact-inner PCG 修复前 ND hex k=3 停滞 9.1e-1 → **全部收敛**。**诚实结论**：剩余 h-脆弱性 100% 在 **element 层的开放基选择**——fem-rs 的 `HexNDk` 开放模是 GaussLegendre 点值（D36 的选择），而 MFEM 源码自带警告 "LOR is only spectrally equivalent with (Gauss-Lobatto, IntegratedGLL)"；MFEM 自己用默认 (GL, GaussLegendre) 时行为与 fem-rs **定量一致**（ND3: 185→637 vs fem-rs 83→363；IntegratedGLL 下 MFEM 32→33）。三个转正测试保持 `#[ignore]` 并重写文档（阻塞点从"LOR 空间实现缺陷"升级为"element 基偶"，与 **D31/#10 GLL 对齐专项**合流）。plor_solvers 5/4/5/5/6 基线逐位、lor_elast 实际改善（o1 24/31/37）。替代路径（显式稀疏 prolongation，约 1–2 天）已写入文档。
- **主会话集成修复**：`CurlCurlIntegrator` 的 Qk/Pk 积分阶（见 D55；`integration_order_for` 加法式 trait 方法 + 两处派发点传 `element_type(0)`，simplex 路径逐位不变，605/454/275 全绿）。
- 回归基线（全绿）：amg 23 / assembly **605**（+6 预存 ignore）/ element **454** / io 123 / linalg 63 / mesh 292 / parallel 225 / solver **251** / space 275；**`cargo build --release --examples --keep-going` 0 错误**。
- 抽查（主会话复跑）：`navier_shear` step1 cfl **7.56030E-02** 全列 = C++（绕过已删）✓；`dpg_maxwell_3d -o 2 -do 0` = **9.482e-1** = C++ ✓；ex3 2D **8.01477893043346e-2**、3D tet ND2 **1.57652991777406e-2** ✓（均走默认入口）；plor_solvers ✓。

## 第十九轮完成（2026-09-11，四代理并行：三路交付 + GLL 路中途合理停点）

- **D52 结案（NonlinearForm 内核化）**：`NonlinearForm`/`NonlinearFormIntegrator`/`NLQpData` 从 `dist_solver::filter` **迁移**（原文件 re-export 保历史路径）到 `crates/assembly/src/standard/nonlinear_form.rs`，与 `physics::nonlinear::NonlinearForm`（solver 侧 Newton 驱动抽象）按路径区分；框架扩展 `[H¹]^d` 交错布局与 `int_rule_order` 钩子。新 `VectorConvectionNLFIntegrator`（`standard/vector_convection_nlf.rs`）逐式对位 `nonlininteg.cpp:744`（真物理梯度 + `ip.weight·T.Weight()`），**Jacobian 也实现**（初版 gradEF 索引写反被 FD 测试抓出后修正）；旧的错误命名标量版删除（零调用方）。**验证**：单测 4 项（常场 9e-17、线性场 `N·u=M·u`、与本地循环逐位/order 6 2–3 ulp、Jacobian vs FD <1e-8）；`navier_mms` 换内核后除计时行外**逐字节一致**、`navier_kovasznay` 全部 err/CFL 行逐字节一致（step1 6.24210E-07）。
- **navier 第 5 件交付：`navier_tgv` + 3D 周期构造验证**：**重要发现——mesh 层零修改即已可用**：round 9 的"3D 周期未实现"实际已被维度泛型 `Mesh::make_periodic` + `build_pk_hex` + round 18 D56 坐标重建覆盖。3³ torus：27 节点/27 单元/0 边界面、order 4 = 1728/5184 dof（= C++ PrintInfo）；`interpolate_vec` 与逐元素投影 <1e-12（D56 在 hex 上自动生效）。`ComputeCurl3D` 落地（C++ 语义 = 逐点导数投影非 DG 弱形式）。**tgv 对照**：step0 行（u_inf/p_inf/ke）与 C++ **逐字节一致**（9.86914E-01 0 1.24994E-01）、ke 打印位 11 行全同（全精度 rel ≤5.7e-10）、HELM/PRES 迭代**逐位一致**、解析衰减 `ke=⅛e^{−6νt}` 钉住；MVIN 差 0–1（双侧低于 rtol）。`schrodinger_flow` 2D 去绕过路径验证正常收敛。
- **D61 结案（周期 <3 格/向 dof 少计）——附重要实证**：**MFEM 自己在 2 格/向全周期时直接 abort**（"Interior quadrilateral face found connecting elements 0, 1 and 2"）——顶点合并周期化方案在 2 格/向根本无法建网，故"与 C++ dof 数一致"字面上不可能；验收改钉**商复形拓扑真值**（2×2×2 torus：8 顶点+24 边+24 面+8 体）。修法（`dof_manager.rs` ~600 行，零 mesh 改动）：周期网格上在 `UnfoldedPeriodicMesh` 包装器（每元素 pre-merge 几何角点）上跑原 builder，实体 dof 按 **torus 实体商合并**（边签名 = 折叠顶点对 + 展开位移；面 = 角点集 + 相对最低角点像的偏移——平移不变且能区分半周期相位；多 dof 面类第二像按物理位置匹配）。**验证**：Q2 **34 → 64**、Q3 120 → 216（= 真值）；≥3 格与 C++ 交叉核对全部一致（n=3: 216/729、n=4: 512/1728/4096）；**无碰撞（≥3 格）时与折叠构建逐位相等**（pin 测试断言 dofs/实体 map/坐标全同）。
- **D62 结案（周期 + 曲面门控）**：`rebuild_dof_coords_periodic` 的几何求值按 `geom_order()` 选高阶基并对**全部**几何节点求值；曲面周期网格同样走 D61 编号路径（`periodic_geometry_snapshot` 本就克隆 pre-merge 高阶几何）。**验证**：4×4 quad + `set_curvature(2)` + 周期正弦曲面：临时还原旧门实测插值误差 1.280e0，新门 <1e-12；曲面非周期逐位不变。
- **D59 结案（3D 曲线边界面几何）**：`assembler.rs` 新增 `curved_boundary_face_geom`：`geom_order≥2` 的 3D 边界面改用**边界单元自身的曲面映射**（hex 面 `QuadQk(q)`、tet 面 `H1TetFacePk::new(q)`，控制点 = owner 元素的面几何节点）。**验证**：正弦底面 2×2×2 夹具（Q2 可精确表示）解析面积 1.02133645958114，修后装配差 **<1e-12**；角点版探针实测 1.0（差异 2.1e-2 存在性与修复均实证）。
- **D63 推进（GLL 路，中途合理停点——代理超时但树上状态绿且连贯）**：`HexNDk` 新增 `NdOpenBasis::IntegratedGLL` 变体与 `new_integrated_gll(p)`（MFEM `ND_HexahedronElement(p, GaussLobatto, IntegratedGLL)`），`lor_factory` 新增 `d63_igll_pencil_diagnostics` 诊断测试。**诊断结论（重要）**：IntegratedGLL 基偶就位后 RT **quad** 路径收敛而 hex 仍不收敛 ⇒ **配对缺陷不是基缺陷**，剩余嫌疑收窄到 **LOR 空间（`fem_space::lor`）的面/内部 slot pairing 或同余符号**（几何配对审计在边块 dof 上与 `fem_space::lor` 一致）。三个 lor_factory 测试仍 `#[ignore]`（阻塞点 = LOR 空间接线，element 层已就绪）。阶段 A（HexQk 序）未动（按优先级先做 B）。
- 回归基线（全绿，合并树）：amg 23 / assembly **612**（+7 预存 ignore：6 lor/hdg/contact/discrete_op + 1 d63 诊断）/ element **456** / io 123 / linalg 63 / mesh 292 / parallel 225 / solver **252** / space **277**；**`cargo build --release --examples --keep-going` 0 错误**（新注册 `navier_tgv`）。
- 抽查（主会话复跑）：`navier_tgv` step0 = C++ 逐字节 ✓；`navier_shear`/`navier_kovasznay`/`navier_mms` 不劣化 ✓（各代理报告 + D52 逐字节验证）；`schrodinger_flow -jet -no-vis` 2D 去绕过路径收敛（PCG 77 it）✓。

## 第二十轮完成（2026-09-11，四代理：三路交付 + HexQk 路中途连贯停点）

- **D65 关案（重要反转，结论与 round 19 假设相反）**：**LOR hex 的面/内部 slot 配对不是缺陷**——用 MFEM 4.10 的 `ConstructLocalDofPermutation` key 算术与 fem-rs 元素布局双向对照 + 直连 MFEM 的矩阵逐项比对（`tmp/d65_dump.cpp` 导出 MFEM 的 perm/A_HO/A_LOR/锚点），边/面/内部三块表**都一致**（perm 负号数 ND o=2 1³ 27/27、2³ 150/150 与 MFEM 相同；RT0 LOR 矩阵与 MFEM 逐项一致）。**真根因 = LOR 装配的 curl-curl 求积阶**：`curl_curl.rs::integration_order_for` 的 `space_order <= 1 → Some(1)` 捷径对单纯形 ND1 正确（curl 为常数），但 **LOR 空间是有理 1 阶张量元**，`curl` 在每个闭合方向上是二次的 ⇒ 1 点欠积分 ⇒ 结果**恰好 3/4**（中点法则 1/2 vs 精确 2/3）。证据：库装配 1.055556/2.111111/4.222222 vs 同网格直接元素级装配 **1.388889/2.777778/5.555556 = MFEM**（k=2/3 时两者差 8e-15 ⇒ 缺陷仅限 order-1 张量路径）。**修复**：① 代理在 LOR 内加 `LorCurlCurl` 转发核（LOR 形式按构造恒为 order-1 张量，局部合法）；② **主会话采纳一行正解入内核**（`tensor` 分支优先于 `space_order<=1` 捷径，张量元恒用 `2k`）——对常规 ND1 装配是过积分（不改变精确值），回归全绿。**`lor_nd_pcg_iterations_mesh_independent` 转正通过**（网格无关 25→34，标注 `+10`；exact-inner 铅笔 10→11 = MFEM 量级）；RT/quad 仍 ignore（RT 有独立缺陷见 D68）。
- **D23 大幅推进（NURBS FE 集合落地）**：**新增 `crates/element/src/nurbs_fe_collection.rs`**（MFEM `KnotVector` 查询 + `degree_elevate` 1:1 逐行移植 + 8 种 `NurbsElement` 的 dof/order 原式（**含 MFEM 2D 公式的 `py` 笔误——注释说明为何不能"改对"**）+ `Nurbs1DFiniteElement`（Piegl A2.2/A2.3）+ `NurbsScalar2D/3D` span 感知包装 + `NurbsFECollection`/`NurbsHDivFECollection`/`NurbsHCurlFECollection`（VariableOrder/name/几何分派/`SetDim` 的 sFE/qFE/hFE 语义/`DofForGeometry`））与 **`crates/space/src/nurbs_extension.rs`**（NURBS mesh 解析、`edge_to_ukv`、`GenerateOffsets`、`GenerateActiveVertices`、**`GenerateElementDofTable` 1D/2D/3D**（`NURBSPatchMap`/`Or1D/Or2D`/`EC/FC/FCP`/`GenerateFaces`+`GetQuadOrientation`/`activeDof` 压缩））。**对照**：10 个 MFEM mesh 全部**逐位一致**（NKV/orders/GetNV/GetNE/GNBE/GetNTotalDof/GetNDof/每个 KV 的 order-NCP-NE-knots/**完整 el_dof 表**/weights）；稀疏强化对照命中 `nurbs_ex1 -o 2` 的 **4356 dof / 4225 顶点**。测试抓到真 bug：2D span 循环从未推进第二个参数方向（data mesh 该方向恰 1 元故不可见）。顺带修 `nurbs_ex1` 的打印格式偏离（现 `Number of finite element unknowns` → `Average reduction factor` **逐字节一致**）。未闭合：HDiv/HCurl 值求值仍只在单 span patch 正确（修法已明确：`set_ijk` + span 存储，局部下标 = 全局 − span）；`NurbsFESpace`（复刻 `LoadFE`）/加权装配/`UniformRefinement` 是示例切换的前置。
- **navier 第 6 件交付：`navier_bifurcation`（含粒子，完整移植无裁剪）**：新增 `navier_bifurcation.rs`（1353 行）+ `navier_particles.rs`（946 行，1:1）+ 内核增量（`NavierDiscretization::curl_curl_and_vorticity` 默认委托、`NavierSolver::vorticity()`，既有 4 个 miniapp 走默认实现**逐位不变**）。**粒子路径完整**：`crates/mesh/src/findpts`（round 5 移植）语义与 `FindPointsGSLIB` 一致、2D 开箱可用 ⇒ **`crates/mesh/**` 一行未改**；缺失的"定位后插值"在 miniapp 内实现；粒子 ODE/反射 BC/失效搬运/`PrintCSV` 全部移植且 **RNG 逐位复刻**（glibc TYPE_3 + mt19937 + `generate_canonical`）。**对照**：DOF 52866/26433 = C++、step1 CFL **6.03374E-02 逐位**、粒子计数 600 步全一致、CSV 表头逐字节、收敛区 100 步中 97 步 CFL 末位一致、step-50 粒子 id/κ/Order 全同（max rel|ΔX| 1.3e-05）。**过程中查清三个 C++ 侧事实（比表面比对更有价值）**：① **C++ miniapp 的初值实际无效**（`Setup()` 在 `ProjectCoefficient` 之前且内部已 `GetTrueDofs`，投影的抛物线初值永不进求解器——移植按此复刻才逐位）；② MFEM 4.9 串行 `FindPointsGSLIB` 无 `ParticleVector` 重载，用默认 2 参调用时 np≥2 的粒子被按 byNODES 解读而错位（**C++ 侧粒子 CSV 一度是伪结果**）；③ `Apply2DReflectionBC` 里 `beta_k[Order()[i]]` 对 order==3 越界读（C++ UB，移植取 `beta_k[2]` 并标注）。
- **HexQk 序重排（D31/#10 阶段 A，代理超时但树上状态连贯）**：`crates/element/src/lagrange/factory.rs` +406/−290 完成 HexQk 内部序 → MFEM `H1_HexahedronElement` 对齐，**element 469 全绿 + io 弯曲 hex 哨兵 3/3 + 周期测试 14/14 全绿 ⇒ 证实 round 15 的设计意图成立：dof_manager 的 `build_pk_hex` 槽序反推与 io 的 `build_h1_hex_geometry` 都自动跟随**（`space/src/dof_manager.rs` 本轮零改动）。未做：`HexQ2/HexQ3` 手写布局的收敛（mixed/partial 消费）、#10 两个 ignored 谱测试。
- 回归基线（全绿）：amg 23 / assembly **613**（+8 ignore）/ element **469** / io 123 / linalg 63 / mesh 292 / parallel 225 / solver **252** / space **286**；**`cargo build --release --examples --keep-going` 0 错误**（新注册 `navier_bifurcation`）。

## 第二十轮新债务

- **D68（P1）RT1 hex 的 LOR 转移仍有独立缺陷**：RT LOR 矩阵与 MFEM 逐项一致、perm 符号结构与 MFEM 相同（0 neg），但 **RT1 HO 矩阵的排序对角线精确一致而入口和差 ~3%、排序行和最大差 167** ⇒ fem-rs 的 RT1 矩阵与 MFEM 并非仅差 dof 重编号，而是多了一层**逐 dof 定向（符号）模式**；因 `HDivSpace::interpolate_vector` 直接算几何通量泛函（不经元素基对偶），该符号不反映到 perm ⇒ 矩阵与转移的符号约定不一致 ⇒ 铅笔崩坏（RT1 hex 276→793 vs MFEM 18→20）。嫌疑落点：`raviart_thomas/hex_rtk.rs` 内建外法向符号 vs `hdiv.rs` 的 `element_signs`/内部块定向。可复现：`cargo test -p fem-assembly --lib d65_lor_fix_diagnostics -- --ignored --nocapture`。
- **D69（P2）quad ND/RT 的 LOR 测试**仍被 `QuadNDk`/`QuadRTk` 非 MFEM 移植阻塞（quad LOR 路径本身已随 `LorCurlCurl`/正解修好）。
- **D70（P2）NURBS 剩余**：HDiv/HCurl 值求值需 span 存储（修法已明确）；未移植 `patches` 变体/周期 BC/`mesh_elements`/`NCNURBSExtension`/`BoundaryElementDofTable`/`UniformRefinement` 等；`NurbsFESpace` + 加权装配是示例切换前置；`crates/mesh/src/nurbs_mesh.rs` 的 `degree_elevate` 仍是旧的"中点插结"重复实现（应委派 `fem_element::nurbs_fe_collection::degree_elevate`）；`nurbs_ex5`/`ex24` 仍非 C++ 对应示例移植。
- **D71（P3）LOR 的 `LorCurlCurl` 转发核**在 curl_curl 正解入内核后已可删除（需跑 LOR 测试确认无变化）。
- 备忘：① `stash@{0}`（ex4-ads-preconditioner）仍在，待用户决定；② 剩余 navier miniapp 顺序建议：`3dfoc`（最小，需复制 `box-cylinder.mesh` 到 data/）→ `turbchan`（3D 周期 + `AddAccelTerm`）→ `cht`（最难，双网格/OversetFindPointsGSLIB）；③ bifurcation 的 `PCG: No convergence!` 在 rs=3 每步多 2 行（4.9 镜像不打印，`sli.rs` 未在可改范围）。

## 第十九轮新债务

- **D65（P2）LOR hex 面/内部 slot pairing**（D63 诊断的收窄结论）：IntegratedGLL 基偶已就位（element 层），剩余嫌疑 = `fem_space::lor` 的面/内部 slot 配对或同余符号（RT quad 收敛 vs hex 不收敛的分叉证据）；转正三个 lor_factory 测试的最后一步。
- **D66（P3）2D 曲线边界边仍走仿射弦**（D59 的 2D 对应；mesh 侧 `boundary_face_endpoints` 接口）。
- **D67（P3）tgv 的 p_inf 相对差 ≤5.5e-4**：dof 排序不同（C++ torus 序 vs Rust 字典序）下 1e-6 容差停机的迭代噪声；PRES/HELM 迭代数逐位一致说明算子一致。逐位复现需 periodic-cube 排序镜像（不建议）。
- 备忘：① `stash@{0}`（ex4-ads-preconditioner 分支）仍在，待用户决定；② 2 格/向周期网格无 MFEM 对应物（其自身 abort），fem-rs 按商复形真值处理；③ tgv 的 ParaView 落盘与 GLVis socket 未复刻（shear 先例，文档注明）。

## 第十八轮新债务

- **D61（P2）周期网格某方向 <3 格时 dof 少计**：不同环面边/面共享同一顶点对，`EdgeKey`/`QuadFaceKey` 去重碰撞（2×2×2 hex Q2 得 34，Q1=8 正确）；MFEM 按 mesh 实体编号无此碰撞。≥3 格/向的实用网格不受影响（4×4、12×12 与 C++ 精确一致）。属拓扑编号修复，非坐标。
- **D62（P2）周期 + 曲面（geom_order≥2）网格**：D56 的门控不触发，缝 dof 坐标保持折叠值（无测试覆盖）；prism/pyramid P2/P3 手写 builder 布局不对齐 factory 时周期缝坐标跳过修复。
- **D63（P1，与 D31/#10 GLL 专项合流）LOR 谱等价需要 (GaussLobatto, IntegratedGLL) 基偶**：fem-rs 的 `HexNDk` 开放模是 GaussLegendre 点值泛函（D36 的正确选择），不是 LOR 兼容基偶。MFEM 自己用默认基时同样 h-脆弱（定量一致），用 IntegratedGLL 则 14–36 迭代网格无关。三个 lor_factory 测试的转正条件 = element 层提供 IntegratedGLL 变体；替代路径 = 显式稀疏 prolongation（约 1–2 天，方案已写入测试文档）。
- **D64（P3）`plor_solvers` 默认 quad 网格在 `TriPointLocator` panic**（H1 transfer 路径预存限制，与 D40 改动无关）。
- 备忘：D58 的 `element_face_blocks` trait 访问器已让 D55 的"hex 空 blocks"语义自动传播；若未来 hex 出现真块需求需重审 MFEM `QuadDofOrd` 语义。

## 第十七轮新债务

- **D54（P2）`dpg_basis::vol_quadrature` 的 Hex8 分支参数语义不一致**：用 `gauss_legendre_arbitrary(order as usize)` 把参数当**点数**，而 `hex_rule`/`quad_rule_01`/`tri_rule`/`tet_rule` 都当**精确次数**。后果：(a) 3D hex 的 DPG 体积/RHS 装配实际用 6 点/方向（次数 11）而 MFEM 用次数 `2·test_order`（4 点）⇒ RHS 差 ~1e-4（分块范数第 4 位，maxwell_3d `-o 2 -do 0` ref0 残余 0.06% 即此）；(b) 迫使 miniapp 用 `mfem_l2_rule_order()` 换算。**修法**：Hex8 分支改 `(order+2)/2` 点（同 `hex_rule`），**同时删除 miniapp 的 `mfem_l2_rule_order()`**（否则二次换算），并一次跑全 3D DPG 消费方。
- **D55（P2）hex NDk(k≥2) 的 `element_face_blocks` 为空**（只有 sign 匹配 + `quad_face_anchor`），疑似缺 hex 面块/MFEM hex ND DofTransformation ⇒ `beam-hex -o 2` 1.5066e-4 vs C++ 2.5343e-4（1.68× 偏低；ND1 仅差 1.5e-4 相对）。需在 space crate 定案。
- **D56（P1，周期类 miniapp 的头号阻塞）`DofManager::dof_coord`/`VectorH1Space::interpolate_vec` 用折叠后的 `Mesh::coords` 构造 DOF 坐标**，而几何周期网格每单元有自己的几何节点（`geometry_nodes`/`geom_coords_of`）⇒ **seam 单元的 DOF 被放到错误物理位置**，任何非常数系数的节点插值都错。实测后果：`navier_shear` 初始条件若用 `interpolate_vec` 则 cfl 变 1.2e-1（应 7.6e-2）、压力范数大 10³ 倍；代理在 miniapp 内以本地 `project_vel`（复刻 `GridFunction::ProjectCoefficient`，按单元自身几何逐 DOF 求值、后写者胜）绕过。
- **D57（P2）ex3 2D 仍 411× 偏**（`beam-tri -o 1` Rust 32.96 vs C++ 0.0801478）：PCG 历史在 D47 前后逐位相同 ⇒ 与 D47/D48 无关；solve 已收敛（AMS 270 iters 6.9e-13）；2D 齐次 PEC 的 MMS rate 正常 ⇒ 怀疑点收窄到 ex3 自己的 2D 接线或 2D 网格读入路径。（注：C++ 自身在 `star.mesh -o 2` 就打印 `PCG: No convergence!`，该命令不是有效基准。）
- **D58（P1）D48 的 canonical 未成为默认装配入口**：本轮交付按"可整体切换的开关 + ex3 默认走 canonical"路径。全量翻转需 `crates/space/src/fe_space.rs` 给 `FESpace` 加 per-element face-block 访问器（默认空），并同步消费方：`postproc/grid_function.rs:702/737/766/1605/1682`、`postproc/postprocess.rs:314/386`、`boundary/vector_boundary.rs:759/825`、`mixed/mod.rs`×6、`hybridization/trace.rs:399`、`dpg/dpg_basis.rs`、**`discrete_op.rs`（需 `D_t = T·D·T⁻¹`）**。
- **D59（P2）3D 边界面用角点（bilinear/affine）几何**：直线网格精确（已测），高阶曲线面近似——与 MFEM 曲线的边界单元不同。候选债务。
- **D60（P2）`fem_element::lagrange::TetP2::dof_coords()` 自不一致**（20 个坐标对 10 个 DOF，边点在 1/3、2/3）——`assembler.rs` 已绕过。
- 备忘：① **`stash@{0}` 是另一分支（`ex4-ads-preconditioner`）的工作**（"fix(compare): 继续修复 DOF 提取和参数"），本轮只清冲突未动它——**需用户确认后处理**；② `glvis_bidirectional_local_loopback` 在 9-crate 并行跑时偶发失败（单跑通过，端口争用 flaky）；③ `beam-tet -o 3` 与 `star.mesh -o 2` 的 C++ 侧本身不收敛（`PCG: No convergence!`），不可作基准。

## 第十六轮新债务

- **D47（P1）`boundary_dofs_hcurl` 漏 tet 三角面 dof**：`crates/assembly/src/constraints/dirichlet.rs:421` 收集了边 dof 与 hex 四边形面 dof，却漏掉 **tet 三角面 dof（k≥2）** ⇒ tet ND2/ND3 的本质边界条件不完整、解 O(1) 错（这就是 ex3 `-o 2` 现为 3.8599e0 的主因之一）。修法 3 行（用现成 `face_dof` + `order()`）。
- **D48（P1）D37 的 2×2 块变换未接入默认装配/重建入口**：能力与代数已测（见上），但需所有重建消费方同步做 `u_local = S·u_canon`（文件清单：`postproc/grid_function.rs:702`、`postproc/postprocess.rs:314/386`、`boundary/vector_boundary.rs:343/759/825`、`mixed/mod.rs`×5、`hybridization/trace.rs:399`、`dpg/dpg_basis.rs`、examples 自带 L2 评估器）。与 D47 齐备后 ex3 `-o 2` 应显著改善。
- **D49（P2）`H1TetPk` 用等距节点而 MFEM 用 Gauss-Lobatto**：影响**弯曲 p≥3 tet 几何的等参映射**（编号已逐位正确）：实测 2×1×1 正弦网格 curved P3 差 **7.53e-2**（顶点 id 反序同 7.88e-2）。修法 = 在 `crates/element/src/lagrange/factory.rs` 加 GLL 节点的 `H1TetPk`（2D 的 `H1TriPk` 已这样做）。位置在 element crate，本轮该文件由 ND 代理占用。
- **D50（P1）3D 边界装配不可用**：`assembler::boundary_face_geom` 有 `assert_eq!(dim, 2)`；`ref_elem_face` 的 `Tri3/Quad4` 面项对 H1 空间用 `TriPk`/`QuadQk`（后者是 [0,1]² 而 `QuadQ1/Q2` 是 [−1,1]²，且 tri 面在 p≥3 时不是体积 GLL 节点）——与 D46② 同类错误。任何 3D navier miniapp（`navier_3dfoc`、3D TGV）都卡在这里。修法同 D46②：面基取体积单元在面上的迹，并把 `boundary_face_geom` 扩到 dim=3（法向 = `t1×t2` 归一化、`|J_face| = |t1×t2|`）。
- **D51（P2）`QpData::weight` 文档与代码矛盾**（D42 家族的根源）：`crates/assembly/src/integrator.rs:17` 注释称"quadrature weight × |det J|"，实际体积路径是 `ip.weight/|det J|`。**修文档即可预防下一个同类 bug**（本轮代理无权改该文件）。
- **D52（P2）`NonlinearForm`/`VectorConvectionNLFIntegrator` 缺内核化**（约 0.5 天）：`crates/assembly/src/dist_solver/filter.rs` 已有可利用的最小串行 `NonlinearForm`（`add_domain_integrator`/`mult`/`get_gradient`）与 `NonlinearFormIntegrator`+`NLQpData`（携带真物理 `grad_phys` 与 `weight = 规则权×|detJ|`，正是对流族需要的度规）。缺：把 `VectorConvectionNLFIntegrator` 按 MFEM `dshape·adj(J)` + `ip.weight` 约定实现、把三个类型从 `dist_solver::filter` 提升并 re-export（与 `lib.rs:189` 的 `physics::nonlinear::NonlinearForm` 重名需处理）、两个 miniapp 换用 `N.mult` 后保持与 C++ 一致。注：现 `VectorConvectionNLFIntegrator` **不是** MFEM 同名类（fin 实为标量空间算子，在 `[H¹]^d` 上索引越界 panic，且当前无调用方）。
- **D53（P3）无 `H¹ × [H¹]^d` 混合装配路径**：`accumulate_mixed_volume_element` 对非 HDiv/HCurl 的列空间不做分量展开 ⇒ kovasznay/mms 的 `D`/`G` 仍是本地元素循环（round 15 期望的"删 200 行"尚不可达；D46① 已把 order 6 的表打开，剩下的是列空间按分量分块）。另 `BdQpData` 无 `elem_dofs`（边界上无法表达网格函数系数，`FText_bdr` 保留本地面循环）。
- **D36 现状（第三次表述，P1）**：G 与 trial 侧已被排除（装配矩阵与 C++ ≤7e-15）。**剩余唯一嫌疑 = `dpg_maxwell_3d` miniapp 的复数路径 RHS `(J,G)` 与倍增系统偏移/BC 接线**；下轮用 `mx3/mwdump.cpp` 的 `CPPSUM` 对照。
- 备忘：H1-trace 面**内部**节点 fem-rs 等距 vs MFEM Gauss-Lobatto（对矩阵无害＝基变换，但本质 BC 插值点不同 ⇒ poisson o3 残余 0.18% 的主因）；RT-trace 用 nodal Lagrange 而 MFEM 用 INTEGRAL-map（尺度差 2×，无害）；vector-L2 trial 块 fem-rs 按 element-major 而 MFEM `byVDIM`（置换，无害但妨碍逐 entry 对照）；`VectorDivergenceIntegrator`/`VectorConvectionNLFIntegrator` 都不是 MFEM 同名类且当前无调用方。

## 第十五轮新债务

- **D36 更正（P1，重要）**：原归因「HexNDk(2/3) 基与 MFEM 不同 ⇒ dpg_maxwell_3d -o2 差距」**被证伪**。实测：(a) HexNDk 与 MFEM nodal 开模态**张成同一 span**（既有 `nd_hex_span_is_tensor_nedelec` 已钉住）；(b) 完成 nodal 重写前后 `dpg_maxwell_3d` **逐位不变**（o1 1.757 / o2 1.446 / o2-ref1 1.395 三者全同）；(c) o2 差距对测试阶不敏感——`do=0`（test order 2 = HexNDk(2)，span 无疑相同）时差距已是完整 1.57×（C++ 9.482e-1 vs fem-rs 1.489）。⇒ 差距在 **trial 侧**（`L2(1)×3` 体空间或 `ND_Trace(2)` 骨架）或**测试范数 G**（4 个 graph-norm cross block 是当前唯一零覆盖的装配环节——既有恒等式 `A x = BᵀG⁻¹B x = BᵀG⁻¹f` 与 G 无关）。加密不下降（fem-rs 1.446→1.395 rate −0.06 vs C++ 0.9547→0.2707 rate 1.95）⇒ 典型相容性缺陷。**下一步**：给 G 加"范数相关"回归（对照 C++ 的逐元素 whitened Y dump 全模式，harness 在 WSL `~/work/mx3/ydump`）或按 D45 试 H1-trace。
- **D42（P2）convection 族权重约定审计**：本轮已修 `VectorConvectionIntegrator` / `VectorConvectionNLFIntegrator`。**同类可疑处待逐个判定**（都是 `qp.weight`，需按 integrand 是"物理体积分且配 adjJ 型 grad"还是"Diffusion 型"分类）：`misc_integrators.rs` 的 `VectorDivergenceIntegrator`(:16)、`NormalTraceJumpIntegrator`(:181)、`NonconservativeDGTraceIntegrator`(:198)、`MixedWeakGradDotIntegrator`(:223 VectorQpData 路径需单独核对三权重语义)；以及 PA 路径 `partial.rs::PAConvectionOperator`。方法学已备：本轮的 `(Ku)_x == M_scalar·x` 型判据（非单位尺寸网格）。
- **D43（P2）弯曲 tet 网格 `nodes` 读取仍静默错乱**（D41 同类未覆盖）：`data/escher-p2.mesh` → 42 个单元中 11 个拿到别的单元的边 dof，max|Δ| = 1.26（对照 MFEM dump 实测）。需 MFEM `TriDofOrd` 面取向；本轮只加了告警。
- **D44（P2）`dpg_poisson_2d -tri` panic**：`crates/element/src/raviart_thomas/tri_rtk.rs:257` 越界（`-o 1 -tri` 即触发，-sc 与未凝聚同样 panic）；与 D39 无关。
- **D45（P1，D36 的主要候选真因）real DPG 的 û 骨架用面间断空间**：`SkeletonSpace::new`（面间断）vs MFEM `H1_Trace_FECollection`（顶点连续）；complex 版用的是 `new_h1`。**这解释了 real 未凝聚 poisson 与 C++ 的预存 0.7%/0.4%/0.2%/0.04% 差距**（随加密收窄 ⇒ 边界离散效应）。
- **D46（P3）navier 内核缺口（代理清单，均已 miniapp 内绕过）**：① `mixed::ref_elem_vol`（`crates/assembly/src/mixed/mod.rs:1206`）只支持 Quad4/Hex8 到 order 3（固定表）→ `MixedAssembler` 在 order 6 不可用（应委托 `lagrange::factory::ref_elem` + `QuadQk::new(order)`，如 `ref_elem_vol_h1` 那样）；② `ref_elem_face`（`assembler.rs:395`）Line2 只到 order 4 → `assemble_boundary_linear` 在 order 6 panic；③ `BoundaryNormalLFIntegrator` 语义冲突（fem-rs 是标量 `∫gφ ds`，MFEM 是向量系数 `∫(v·n)φ ds`，同名陷阱）；④ `Mesh::face_elements` 未 `build_face_to_elem()` 时静默返回 `(0,None)`；⑤ `NonlinearForm`/`NonlinearFormIntegrator` 缺失（无 `N->Mult`）；⑥ 载荷性质：`G ≠ Dᵀ`（差 `∫_Γ φ_kφ_i n_c ds` 边界项，任何把 G 装成 Dᵀ 的移植都是错的）。
- **D37/D38 状态**：**hex 部分随本轮完成**（`HexNDk` 全阶 nodal，`dof_coords` 的 z-/y- 面第二切向块 open/closed 互换 bug 与 hex 面内 dof 无方向编码的非协调 bug 一并修掉——旋转网格 k=2 trace 从 2.92e-1 → 5.6e-16）。**tri/tet 的 k≥3（D38）与 D37（tet 面 2×2 旋转）本轮未做**，因 `hcurl.rs` 由 hex 一路独占；下轮可开（`hcurl.rs` 的 tri k≥3 / tet k≥3 仍是矩泛函 + 恒等 pairing）。
- 备忘：`crates/element/src/testsupport.rs` 顶部注释称 dump 来自 IntegratedGLL 元素（`mfem_nd2_q0` 现已无引用）；`miniapps/dpg/dpg_maxwell_3d.rs` 头部已按本轮结论更正（见下）。

## 第十四轮新债务

- **D36（P1）HexNDk(2)/(3)（hex 张量 ND）基与 MFEM 不同**（B 发现）：`∫F_0 = (2,1,1)` vs C++ `(0.389,0.278,0.5)`、`∫curl F_0 = −1/6` vs `−1/8`（双方求积阶加密均不动 ⇒ 都精确 ⇒ 基函数确实不同；DPG 算子对测试基变换不变，故 span 相同则解应相同）。位置 `crates/element/src/nedelec/hex_ndk.rs`；**dpg_maxwell_3d -o 2 整场差距（1.395 vs 0.2707）即此**。修法 = 按 D32 同方法 nodal 化 hex 张量基（MFEM `ND_HexahedronElement`：open GL × closed Lobatto），修后复测 o2。
- **D37（P2）tet ND2 面 dof 跨元 2×2 旋转**：MFEM 面切向对相邻元差 T(ori) 矩阵（`ND_DofTransformation::T_data` 六矩阵族），fem-rs 装配层标量符号框架表达不了 2×2 块变换（元素语义已 MFEM 精确；3D ND2 多元 curl-curl 精度受此限制；ex3 3D 默认 ND1 不受影响）。
- **D38（P2）TriNDk/TetNDk/HexNDk（k≥3）仍为矩泛函 + 恒等 pairing**（A）：同族 nodal 化未做（无求解器验收覆盖），建议下轮统一。
- **D39（P2）real DpgWeakForm 的 -sc 与非凝聚解不一致**（B）：round-14 修了 3 个机械 bug（恢复公式双重求解/局部索引散布/gather 错位）后从 panic 变可运行，但仍有布局/符号错位（poisson -o1: 2.836 vs 1.940）；complex 弱形式的 -sc 已完全修好。
- **D40（P2）lor_factory 三 ignored 测试**：走 vendor AmsPrecond/AdsPrecond（粗解不可配置），FAGRDS 假设约束置换 500 步不收敛（ND hex 残差 8.4e-1、RT hex 2.3e-4、quad 5.7e-2）；需 solver 管线换自研 AMS 或修置换路径。
- **D41（P3）io 高阶 hex `nodes` 读取静默错乱**（D31 影响面）：修复前读弯曲 hex 网格得错乱几何且无告警；建议先加显式拒绝/告警，再在 GLL 专项以 D31 置换表 + `make_refined` 编号器实现。
- 备忘：ex3 求解管线 `solve_report_2d` 重组装未消简矩阵 + AMS 梯度每边 1 dof（examples/discrete_op 既有行为）；`ex24 -m beam-tet -p 1 -o 2` 触发 mixed 装配器不支持 HCurl/Tet order-2（`vec_ref_elem` 缺分支，预存）。

## 第十三轮新债务（第 14 轮状态更新）

- **D32** — ✅ 第 14 轮结案（见上）。
- **D33** — ✅ 第 14 轮结案（见上）。
- **D34** — ✅ 第 14 轮结案（统一到 nodal，见上）。
- **D35** — ✅ 第 14 轮结案（maxwell_3d o1 档对齐 C++ 2% 内；o2 档差距转 **D36**）。
- **D30** — ✅ 第 14 轮结案（B 接手完成）。

- **D32（P1）ND k≥2 边 moment 泛函非反射不变**（D29 代理完整诊断，未修）：`∫Φ·t̂·t^m` 在 t→1−t 时常数模差常数偏移（u_0(1−t)=2−u_0(t)），相邻单元参考边不同（长 √2 vs 1）→ 全局切向不连续 → tri ND2 Maxwell 不收敛（rate −0.3；`mms_verification::maxwell_2d_nd2_convergence` 以 `errors[1]<10` 假绿；ex3 -o 2 同因不收敛）。修复配方（实验中 trace mismatch 已达 1.8e-15）：边权重中心化 `{(t−1/2)^m}`（TriND2 Vandermonde r_odd −= ½·r_even、TetND2 边闭包 `*t`→`*(t−0.5)`、QuadND2 边模中心化）、参考切线单位化（TetND2 对角边 /√2、TriND2 e1 行 /√2）、hcurl.rs 边 dof 符号 σ=s·(−1)^m + interpolate_vector k=2 权重同步；卡点 = split 网格斜边两侧参考长度差 √2，需改参考弧长补偿形式。同族：TetND2 的 8 个面 dof 旋转一致性未处理；`HCurlSpace::interpolate_vector` quad NDk≥2 内部 dof 恒 0；hcurl.rs k==2 的 3 点 Gauss 常数是近似写法。
- **D33（P1）`TetRTk(0)` 基非通量对偶**（D28 代理发现）：`tet_rtk.rs` 的 Gauss-Jordan 基构造（`coeff[i*n+j]=row[i][mt+sel[j]]`）得到的基对偶错（D_i(φ̂_j) 非对角：基 0 在面 0 与面 2 同时有常值迹）→ 多四面体网格上任何插值实现都无法精确重构，且 **L² 投影路径同样失败**（(1,0,0) 在 unit_cube_tet(2) 误差 5.1e-1）——属空间/基缺陷。修法 = 对角化选取或转置块；修后 tet RT0/RT1/RT2 插值与 4 个 #[ignore] 回归测试可转正。
- **D34（P2）tri/tet RT1/RT2 dof 语义契约互斥**（D28 代理取舍）：`discrete_op.rs` 的 RT1/RT2 离散散度/curl 算子以 canonical-moment 语义回读 dof，与 nodal 采样插值（MFEM Project_RT）语义互斥 → `interpolate_vector_legacy` 保留旧路径（tri-RT1 (1,0) 0.965 / tri-RT2 (x,y) 0.933 不精确，已在回归测试 #[ignore] 文档化）。解除 = 改 discrete_op 语义（需评估其消费方）。
- **D35（P1）多 hex 反向 quad 面 trace 残差**（D24 代理遗留）：单 hex/tet 恒等式机器精度，2×2×2 hex 反向面 F-rows 残差 ~0.37（`dpg_maxwell_3d_identity::hex2_p1` #[ignore]）；逐项检查（逐元素 Stokes/面插值=Ê_tan/σ·m 跨面/scale/n_can）全过 → 缺口在跨单元装配的某个机械细节，下轮用 C++ Loc2 逐块矩阵对照（harness 在 WSL `~/work/mx3/`）。-sc 静态凝聚恢复解不一致（3.145 vs 1.753）。标量 trace 分支（acoustics n≥4 的 8–60% 缺口）疑似同族反向面问题。

## 第十二轮新债务

- ✅ ~~**D27（P1）pex5 np2 并行解有 ~20% 固定偏差**~~（第十三轮结案，真因不是矩阵也不是求解器）：同网格同维数下
  np1 给 1.94e-5（与 C++ 一致）、np2 给 1.95e-1 且**加密不下降**；
  `nu`/`np`（不依赖解）在 np1/np2 逐位相同 ⇒ owned 过滤/积分/allreduce 正确。
  真因 = **边界 rhs 的跨界 dof 贡献被 ghost 截断**（详见第十三轮 D27 结案记；
  修复 = rhs 组装后 `reverse_dof_exchange`）。
- ✅ ~~**D28（P1）`HDivSpace::interpolate_vector` 与装配基不一致**~~（第十三轮修复，遗留 **D33** tet 基缺陷 + **D34** RT1/RT2 契约互斥）：
  可精确表示的场其插值场严重错误——RT0-quad 上 `(1,0)`/`(x,y)` 正确（~1e-16），
  但 tri-RT0 上 `(1,0)` 误差 **2.0**、quad-RT1 上 O(1)（1.7）；同一场的 L²
  投影（assembler 路径）精确到 1e-15。根因 = 插值 dof 泛函未对齐装配基的
  nodal 采样 + 符号 + 取向约定。影响所有用 `interpolate_vector` 做
  "精确场投影"的示例（pex24 的 `ex_dm`、pex4 初值）。
- ✅ ~~**D29（P1）fem-rs H(curl) 阶 ≥ 2 路径本身是坏的**~~（第十三轮修复 quad ND2；k≥2 边 pairing 转 **D32**）：常量场
  `(1,0)`（任何 ND 空间都能精确表示）的 L² 投影误差在阶 2 为 **35.8**、
  阶 1 仅 4.7e-7，且与求积阶（3…10）无关；对 E_exact 的投影误差阶 2 = 11.3
  （阶 1 = 0.0058）。真因 = HCurl dof 数（Quad4=12）与 legacy QuadND2（8）
  错配 → 内部 4 dof 载荷恒 0 + 泡泡基参考域未换算。
- **D30（P2）`dpg_poisson_2d -o 3` panic** "test-space Gram not SPD on
  element 0"（P 发现，2D quad 高阶图范数 Gram 疑似数值奇异）。
- **D31（P2）`HexQk` 的 dof 序 ≠ MFEM `H1_HexahedronElement` 序**（N 发现）：
  其边块编号用自身 `(face_i,face_j)` 表（MFEM dof 8 在 (0.5,0,0)，HexQk
  dof 8 在 (1,−1,0)）。当前几何按 HexQk 序存储自洽，但依赖"HexQk 序 ==
  MFEM H1 序"的高阶 hex 几何读取（`io/mfem.rs` 尚不支持 Hex27）或 H1 hex
  全局 dof 编号逻辑需另行核查（与 GLL 对齐专项 D7/#10 同族）。
- **D24 第十三轮更新**：`dpg_maxwell_3d` 已替换为真 UW-DPG（`add_trial_trace_space_nd`
  接入弱形式装配，dof 数与 C++ 逐位一致 156/984/888/6192，单元级恒等式机器
  精度），整场 L² 未达 C++（n2-o1 1.753 vs 1.723；o2-ref1 1.395 vs 0.271）
  → 缺口收敛为**多 hex 反向 quad 面 trace 残差**（**D35**）；
  `dpg_acoustics_3d` 的 n≥4/加密档仍有 8%/60% 缺口（疑似同族）。

## 第十一轮新债务

- ~~**D17 fem-parallel P2 分区**~~ — ✅ 第十一轮修复（代理 L）。
- **D18 示例编译门禁缺失（已补）**：`--lib` 测试门不覆盖 examples，
  导致 helmholtz_1d 语法损坏、NURBS 族、pex4/5/8/27、ex7 长期带病入库
  （pex8 自 012c2ec 起编不过）。**纪律更新：收尾验证必须包含
  `cargo build --release --examples --keep-going`**（`--keep-going` 才能一次
  拿到全部失败，否则 cargo 遇错即止、每次只暴露一个）。
- **D19（P1）`hdiv_error.rs` 是占位实现**（全部函数返回 0.0，四个
  `_owned`/`_q` 变体只是转发）→ pex4/pex5/pex8 打印的 L2 误差恒为 0。
  实现路径：用 `qspace/qfunction`（#9 已落地）+ 逐单元 owned 过滤
  （并行语义：只累加本 rank owned 元素，再 allreduce 平方和开方），
  对位 MFEM `ParGridFunction::ComputeL2Error`/`ComputeHdivError`。
- **D20（P1）`face_elements` 遮蔽陷阱**：`Mesh<D>` 固有方法
  `face_elements -> Vec<ElemId>`（simplex.rs:3059）遮蔽
  `MeshTopology::face_elements -> (ElemId, Option<ElemId>)`（:3261），
  两种返回类型在 16 处调用点混用（已两次引发示例编译错）。修法：固有方法
  改名（如 `face_adjacent_elems`）并更新其 7 处调用点。
- **D21 NURBS 示例族的实质缺口**（K 发现，未改）：`nurbs_ex1`/`ex3` 传**空
  essential dof 集**（C++ 把全部边界属性设 essential）→ ex1 PCG 发散 panic、
  ex3 停滞 1.1e-6；`nurbs_ex5`/`nurbs_ex24` **不是** C++ 对应示例的移植
  （实为 H1 NS / mixed-Darcy 草稿，网格/阶/默认值都不同）；ex1 的
  `ref_levels` 常数 50000 vs C++ 5000。
- **D22（P1）fem-element NURBS 二阶导在区段端点错误**：`nurbs.rs:354-357`
  用有限差分 ε=1e-6，但端点把 `xi±ε` 夹到 `xi0+1e-14`/`xi1−1e-14`，
  对称模板被破坏（实测端点 d² ≈ 8e6，应为 ~32）；C++ `PrintFunctions`
  是解析导数。
- **D23 NURBS FE 集合未实现**：`NURBSFECollection`/`NURBS_HCurl`/
  `NURBS_HDiv` + `NURBSExtension`（C++ 的 NURBS 空间），故 NURBS 示例的
  dof 数与误差**不可能**与 C++ 对齐（现用 H1/ND/RT/L2 代替）。
- **D24 DPG 3D 未替换 + 2D 高阶问题**：3D 示例仍伪 DPG，阻塞 =
  `dpg_weakform.rs` 无向量 trace 支持 + `face_geo_at` 的 3D 法向多 0.5 因子
  （MFEM `CalcOrtho` 无因子）；`ComplexDPGWeakForm::assemble` 在 `-o ≥ 2`
  panic "complex test Gram not HPD on element 2"（纯内核高阶问题）；
  骨架 trace 基节点用等距 `k/p` 而 MFEM RT-trace（INTEGRAL+GaussLegendre）
  用 GL 开点（order ≥ 1 需对齐）。
- **D25（P2）`crates/solver/src/lor.rs` 的 `LorAmgPrecond`** 仍直接包
  vendor 的 `linlvo::amg::AmgPrecond` → 带同一粗解置换缺陷；改用
  `fem_amg::CorrectedAmgPrecond`（或在 vendor 层修 `SparseLu` 的重排约定）。
- **D26（P2）3D `Mesh::set_curvature(2)` 对 Hex8 生成退化几何**（F 发现）：
  单 hex 上边界面 QP 处 detJ = 0.0（face 4 的 9 个 QP）与负值（face 3 最小
  −0.236），且 2 阶几何节点错位（`set_curvature_hex8` 的边节点搜索把参考
  坐标与物理坐标比较、trilinear 回退用顶点 bit 约定与 Hex8 节点序不一致）。

## 第十轮遗留（下一轮）

1. **dpg_maxwell_2d 整场 L2 未收敛**（1.42/1.40 vs 0.88/0.48）——单元级
   已逐位一致，故缺口在骨架与边界：候选 = 骨架 trace dof 的 ess 投影
   （`ProjectBdrCoefficientNormal` 等价物的均值通量语义）、RHS 边界项、
   或 ND trace 的逐元素 dof 变换（`NdDofTransformation` 已就位，需接入
   maxwell 装配）。
2. **`standard::ConvectionIntegrator` 仿射路径缺 |det J|**（P1 正确性 bug）：
   `assembler.rs` 仿射分支应对位 MFEM 一律 `Mult(dshape, AdjugateJacobian)`；
   修后全量回归。
3. 3D ND-trace 骨架基（acoustics_3d/maxwell_3d 替换前置；规格见第九轮节）。
4. TMOP `-ae 1`（InterpolatorFP）外推语义与 gslib 差异、tid 4/9–11 仍拒绝；
   C++ 4.10 serial tid 5/6/8 触发 SetSerialDiscreteTargetSize 崩溃（用 4.9 参考）。
5. 其余小尾巴：hex 面边界积分、曲线边界测度、D14 par 复 FGMRES、affine 快
   路径周期几何、fem-amg vendor SA+WJ、BDM 棱柱、GridSfcOrdering3D。

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
