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
   未声明于 miniapps/solvers/）。剩 plor_solvers、plor-transfer。
3. ~~**SPDE 包**~~ — ✅ 第三轮完成（generate_random_field；上游实为 1 个
   可执行）。
4. ~~**ParSubMesh + TransferMap**~~ — ✅ 第四轮串行裁剪：multidomain H1 版
   （结构量与 C++ 全一致，block 场 7e-6）；_nd/_rt 待做。暴露 P0 级内核
   bug（见债务 D1）。
5. ~~**距离场支撑库**~~ — ✅ 第四轮完成：`fem_assembly::dist_solver`
   （3924 行，14 测试含 3 组 C++ 交叉验证）→ shifted/ 三 miniapp 已解锁。
6. **abs-L1 Jacobi 光滑子 + ParFESpaceHierarchy 式 MG** — 阻塞 diag-smoothers/（3）。
   fem-rs 已有 GeomMGHierarchy；缺 |A| 对角与 l1 变体、p 粗化层。
7. ~~**Adjoint 时间积分**~~ — ✅ 第四轮完成：`fem_solver::adjoint::
   time_dependent`（Nordsieck BDF 伴随 + 检查点二分）+ cvsRoberts
   （对照独立刚性问题参考 1e-4~1e-7）+ adv-diff（-fd 自洽 <1e-6）。
8. **复数 DPG + 复数 CG** — 阻塞 dpg/acoustics、dpg/maxwell（2）；dpg 扩散/对流已具备。
9. **高阶 L2（quad/hex 任意阶）+ IntegratedGLL 基 + L2/RT change-of-basis +
   QuadratureSpace/QFCoeff** — 阻塞 hdiv-linear-solver/（5）+ display-basis 的 hex L2≥P2 缺口。
10. **ND/RT HO 基升级为 GLL/IntegratedGLL（dof2nk 约定）** — LOR 谱等价的前提
    （第二轮 4 个 ignored 谱测试待启用）。
11. **多层 divergence-free + Bramble-Pasciak CG + 弹性 sys-AMG** — 阻塞 solvers/ 3 个。
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
- **D2 tet 网格 io round-trip 取向归一化**：细化后 tet 写出/读回做
  `[a,b,c,d]→[b,c,a,d]` 旋转，o2 边 dof id 置换 → 跨 round-trip 的场传输
  错位（nodal-transfer 代理发现，o1 不受影响）。修 crates/io 读端或
  crates/mesh tet finalize。
- **D3 `fem-solver/src/bdf.rs` k≥2 校正子残差缺 l1 因子**（adjoint 代理发现；
  k=1 巧合相同所以未暴露）。adjoint 模块已自带正确驱动绕开。
- ~~**D5 lor_elast miniapp 半成品**~~ — ✅ 第五轮收尾：根因是 linlvo AMG 默认
  V-cycle 非对称（非模板/BC 问题）；换 Ruge-Stüben+SGS 对齐 hypre 配置后
  **验收达标**（迭代数三档加密有界），C++ 串行 harness 9 例 ‖X‖/能量 ≤5.9e-11、
  dof checksum 逐位。
- **D6（新，P1 级）GridFunction::get_value / Mesh::locate 求值缺陷**：探针测试
  显示线性场任意点求值返回 0（shifted 代理发现，miniapp 已带 env 门控
  workaround）。与 gslib findpts（另一条已验证路径）的关系需排查；修复后
  移除 shifted miniapp 的 workaround。
- **D7（新，备忘）QuadQk 工厂基在 [0,1] 参考域**，而 quad/hex lagrange 基与
  求积在 [-1,1]——域约定需在文档/类型名显式标注（hooke/dfem 代理踩坑一次）。

## 第五轮完成（2026-09-09）

- ~~shifted/ 三 miniapp~~（distance/diffusion/extrapolate + SBM3 内核件，
  C++ harness lst=1 解范数 2e-5）
- ~~hooke~~（C++ harness Newton 序列逐行一致，终态 ‖U‖ 1e-15）
- ~~dfem-minimal-surface~~（-der 0/1/2 三模式同终态）
- ~~lor_elast~~（D5 验收达标）

## 下一步队列

1. **D6 locate/get_value 修复**（P1，解锁移除 shifted workaround；顺带排查与
   findpts 的关系）。
2. D2 tet io round-trip、D3 bdf.rs l1 因子（小修）。
4. shifted_diffusion 大规模 tet 工况（AMG 零对角行问题）与 multidomain
   cylinder 复验（D1 修复后）。
4. #2 ParLOR、#8 复数 DPG/CG、#11 div-free+Bramble-Pasciak、#12 convert-dc 子集。
5. pro-bench-tests 36 基线重标定（持续排队）。

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
