# MFEM 简化 collection 逐个盘点（round 72，主会话亲办）

覆盖矩阵 §1.5 的 `?` 行（"常量/线性族（RT0_2D、Const3D、ND1_3D 等简化 collection）——未逐
collection 盘点"）与 §4 未验证队列第 2 条的盘点结果。**方法**：MFEM 4.10
`fem/fe_coll.hpp` 的 class 清单为真值源（`grep -n '^class .*FECollection'`，40 个），
在 fem-rs 侧按**元素 → 空间 → 装配 → pin** 四级查找对应物。**结论口径**：fem-rs 没有
MFEM 那种 collection 类层次（用 `SpaceType` + 元素结构体 + 分派表），所以"对标"= 等价路径
是否存在 + 是否有 pin，而不是类名一一对应。

## 一、有等价物且已被主矩阵覆盖（`?` → 归并到 §1 已有行）

| MFEM collection | fem-rs 等价物 | pin 出处 |
|---|---|---|
| `LinearFECollection`(H1 P1) | `QuadQ1`/`HexQ1`/`H1TriPk(1)`/`H1TetPk(1)`/`H1PrismPk(1)`/pyramid 默认 | §1.1 全行 MACH/BIT（ex1 BIT 等） |
| `QuadraticFECollection`(P2) | `QuadQk(2)`/`HexQk(2)`/`H1TriPk(2)`/`H1TetPk(2)`/… | D157/D158/D113/D368 |
| `CubicFECollection`(P3) | 同族 order 3（`H1_3D_P3` 显式名正常；legacy `Cubic` 名的读侧曾有 D112 坑） | D112/D31、fichera-q3 |
| `RT0_2DFECollection` | `TriRTk::new(0)`/`QuadRTk::new(0)` | §1.3 k=0 行（d468/d555/d560 位级） |
| `RT1_2DFECollection` | `TriRT1`/`QuadRT1` | §1.3（D368/D510、d468 1728/1728） |
| `RT2_2DFECollection` | `TriRT2`/`QuadRTk::new(2)` | §1.3（D575 表修复后） |
| `ND1_3DFECollection` | `HexNDk(1)`/`TetNDk(1)`/`PrismND1`/`PyraND1` | §1.2（D555 逐位、d680 位级门） |
| `Const2D/Const3DFECollection` | `P0Tensor/P0Tri/P0Tet/P0Pyr`（`crates/space/src/ref_elem.rs:77+`，D364 单一真源）+ `L2Space` order 0 | ref_elem/assembler 单测；**无 gf 级 pin**（见二） |
| `CrouzeixRaviartFECollection`(2-D CR) | `crates/element/src/crouzeix_raviart.rs`（`CrTri1/CrTri2/CrouzeixRaviart1/Vec`），DG 侧在用（`dg_base`/`dg_advection`） | 元素级单测；**无空间级 pin** |
| `LinearNonConf3DFECollection` | 2-D 版：`crates/element/src/nonconforming.rs`（`Q1RotRef/QuadQ1Rot/Vec`，Rannacher-Turek 旋转双线性）；3-D hex 版：同文件 `RotTriLinearHex`（round 102/D102 落地，MFEM `RotTriLinearHexFiniteElement` 1:1，face-center dof，[0,1]³ MFEM cube 约定）。collection 其余臂：TRI/SQUARE = P0（等价物在 space 侧 `P0Tri/P0Tensor`），TET = `P1TetNonConfFiniteElement` 仍未移植（D911） | 元素级；hex 版 **MFEM 4.10 对拍位级**（D102：276 点 worst Δ=0，`crates/element/tests/d102_refined_nonconf_mfem_truth.rs`） |
| `RefinedLinearFECollection` | 单纯形三元素 round 102/D102 落地：`crates/element/src/refined_linear.rs`（`RefinedLinear1D`(3 dof)/`RefinedLinear2D`(6)/`RefinedLinear3D`(10)，细分网格分片线性 macro-element，MFEM `fe_fixed_order.cpp` 逐行移植）。SQUARE/CUBE 臂（`RefinedBiLinear2D` 9 dof、`RefinedTriLinear3D` 27 dof）**未移植**（D912/D913） | 元素级 **MFEM 4.10 对拍位级**（D102：含全部分支边界带，worst Δ=0，`d102_refined_nonconf_mfem_truth.rs`）；无空间级消费者（见 D914） |
| `H1Ser_FECollection` | `HexSerendipityPk`/`QuadSerendipityPk` | D743（hex p=1 位级）、D768（2-D 帧，round 72） |
| `H1Pos_FECollection` | `crates/element/src/lagrange/factory.rs:1495` 的 H1Pos 对应物 | 元素级注释 + 单测 |
| `RT_Trace_FECollection` | `crates/space/src/dpg_trace.rs`（ex8 的 trace 空间，逐边 `order+1` dof） | dpg 套件 |
| `GaussLinearDiscont2D` / `GaussQuadraticDiscont2D` 等 | L2 的 GL 开式族（`TriL2GL/TetL2GL/QuadL2GL/HexL2GL`，`L2Basis::GaussLegendre`） | §1.4 行（MACH，D226/D269 等） |
| `P1OnQuadFECollection` | `QuadQ1`（[0,1]² 双线性，同一 node 集） | §1.1 quad 行 |
| `NURBSFECollection`/`NURBS_HDiv`/`NURBS_HCurl` | NURBS patch 装配路径 | §1.5 NURBS 行 BIT |

## 二、真实缺口 / 需登记（本 pass 未盘出等价物或只有元素级）

1. ~~**`RefinedLinearFECollection` — 全库 0 命中**~~ **已落地（round 102/D102）**：单纯形三元素
   `RefinedLinear1D/2D/3D` 在 `crates/element/src/refined_linear.rs`，MFEM 4.10 探针对拍
   **位级一致**（276 点，覆盖全部子域分支内部 + 边界带两侧）。剩余：SQUARE/CUBE 臂
   （D912/D913）与 space 层消费者（D914，MFEM 中该 collection 服务 LOBPCG/LOR 宏元素场景）。
2. **`ND_R2D/RT_R2D` 族（降维嵌入 collection）— 已闭合（round 102 / D102）**：
   `crates/element/src/embedded/`（NdR2dTri/Quad、RtR2dTri/Quad + ND_R1D/RT_R1D
   segment/point 元素；`RT_R2D_SegmentElement` 为上游死代码，有意不移植）+
   `crates/space/src/embedded_r2d.rs`（HCurlR2dSpace/HDivR2dSpace，MFEM
   collection 布局 1:1）。验收：元素探针 1782 项最坏相对差 6.564e-15、空间布局
   10 案例逐位、ex31 4 网格 × 阶 1/2/3 = 12 案例整个 stdout 与 C++ MFEM 4.10
   字节一致（证据 `tmp/d102r2d/REPORT.md`；钉 `d102_embedded_probe.rs` /
   `d102_embedded_space.rs`，先红后绿验牙）。1-D 网格的空间层（ex31 dim==1
   分支）= D901；Trace 变体按映射级落地（无消费者，见 embedded/mod.rs 文档）；
   VectorAssembler embedded 臂缺位 = D900（空间层有意不实现 FESpace 防静默错装）。
3. ~~**`LinearNonConf3DFECollection` 的 3-D 版**（hex 面中点非协调线性）~~ **hex 臂已落地
   （round 102/D102）**：`RotTriLinearHex` 在 `crates/element/src/nonconforming.rs`，MFEM 4.10
   探针对拍**位级一致**（125 点 hex 网格 + 全 collection dof 表）。剩余：TET 臂
   `P1TetNonConfFiniteElement`（4 dof，D911）；TRI/SQUARE 臂 = P0 常量（等价物已在 space 侧）。
   space 层的非协调 hex 空间接线仍无消费者需求（维持"先确认消费者"原则，未为接线而接线）。
4. **`P1OnQuad`/`GaussLinearDiscont` 之类"同族不同基"的 collection**：等价路径在（见上表），
   但**没有以该 collection 语义命名的 pin** ⇒ 归入下面的"pin 深度"问题，不单列缺口。

## 三、这次盘点的真正结论（对 DoD 的影响）

`?` 格的实质**不是"功能缺失"而是"pin 深度"**：元素几乎都在（除 `RefinedLinear`、3-D
`LinearNonConf`、`_R2D` 无逐 collection pin）——其中前两者已于 round 102/D102 补齐元素层
（位级对拍，见上），`_R2D` 仍缺 collection 级 pin——缺的是**以 collection 语义为判据的验收**。
主会话裁决：把矩阵 §1.5 的 `?` 行改写为"**LAT（元素在、无 collection 级 pin）**"，
并把 §4 未验证队列第 2 条的"逐个盘点"标记为**本轮已执行**（剩余 = 上面的 1/2/3 三条子项，
各带 recipe）。这样 §4 的 `?` 集合从"未知"变成"已知但未 pin"，可排单、可收敛。

## 四、本 pass 的边界（如实声明）

- 40 个 MFEM collection 中，本表覆盖了矩阵点名的 9 个 + 盘点中显式命中的 9 个；其余
  （`H1_Trace`/`DG_Interface`/`QuadrilateralFECollection`/`QuadraticPosDiscont*` 等）**未逐条核**，
  归入"同族主路径已覆盖"或"未验证队列"。
- 所有"等价物存在"的判定基于 **grep 命中 + 已存在单测/矩阵行**，未做 collection 级对拍；
  要升 BIT/MACH 需要新的对照（每族一份 MFEM 探针 + pin 测试）。
