# fem-rs ↔ MFEM 4.10 覆盖矩阵（唯一权威完成度清单）

> 建于 round 60（2026-09-22），主会话维护。**使命：把"完整对标 MFEM"从无限目标变成可数、
> 可收敛的清单**——每一格要么有证据指针，要么明确进"未验证队列"。改任何族/路径之前先查此表；
> 关闭任何债之后更新此表。与 `tmp/round3_plan.md`（过程记录）分工：plan 记历史，本表记**现状**。

## 0. 状态图例与完成定义

| 标记 | 含义 |
|---|---|
| **BIT** | 与 MFEM 4.10 逐位/逐字节（测试或落盘 golden 钉死） |
| **MACH** | 机器精度（≤1e-14 级）对拍，有测试 |
| **TOL** | 容差级对拍（口径注明），有测试 |
| **DEV** | 有意分歧（MFEM 之病不镜像或裁剪），出处文档化 |
| **GAP** | 功能缺失（MFEM 有、fem-rs 无/panic） |
| **LAT** | 潜伏债（有实现但约定/表未对齐或未验证，当前无数值路径经过） |
| **?** | 未验证——无已知 pin，需排入队列 |

**完成定义（Definition of Done，提议）**：
1. 元素层矩阵无 `?`、无 `GAP`（LAT 可带条件放行：gate 锁定 + 债号在案）；
2. 路径矩阵五条主线（空间/装配/求解器/并行/io）无 `?`；
3. examples/miniapps 全部为 BIT / MACH / DEV 三态之一（exit(3) 裁剪逐个文档化）；
4. 互操作：mesh 与 gf 文件 C++↔Rust 双向往返干净（mesh 已达，gf = D602 进行中）；
5. 24 个 `#[ignore]` 逐个分类为合法诊断/长验收（round 48 起的既有纪律）。

## 1. 元素层矩阵（几何 × 族）

参考帧注意：GLL vs 等距 vs GL 是**族内分型**（round 32 起四次现形的老坑），表内已按分型拆行。

### 1.1 H1（标量连续）
| 几何 | 分型 | 状态 | 证据 |
|---|---|---|---|
| tri | GLL `H1TriPk` | MACH | D157/D181（全阶对拍） |
| quad | 等距 `QuadQk`/`QuadQ1` | MACH | D368 表、D113 curved-hex 以其 dof_coords 为真源 |
| tet | GLL `H1TetPk` | MACH | D157/D158 |
| hex | `HexQk` | MACH | D113（curved hex -o3/-o4 逐节点 ≤1.1e-15） |
| prism | 实体相位序 `H1PrismPk`（读侧 layer_perm 桥） | MACH | D177/D182/D295 |
| pyramid | Fuentes 默认 + Bergot 显式 | MACH | D299/D304/D324/D348（面取向） |
|Hex20/Hex27/Tet10/Prism15/Pyramid13 的 H1 高阶几何搬运 | — | **GAP（部分）**：D113 剩余——`Mesh::uniform` 丢父几何（Hex27 ~0.5 天/Prism6 ~1 天/Tet4 ~1.5 天/Pyramid5 ~1.5 天） | D113 剩余条目 |

### 1.2 ND（Hcurl）
| 几何 | 状态 | 证据 |
|---|---|---|
| tri | MACH（1 阶逐位） | D555 反转（ND1 与 MFEM 逐位一致） |
| quad | LAT→MACH | D368 表全量；D69 历史债已关（IGLL 对建成） |
| tet | BIT（ND1 逐位 D555；ND≥2 **GF 文件层 BIT**——D602 round 60 裁决：round 59 的"swap 缺陷"是比错列的误诊，fem-rs 与 MFEM 的 .gf 本来 74/74 一致[首遇元素 T=0→T[0]=I 与建面锚点同约定]；本地帧关系不进文件） | D555、D602 三层验收（74/0、跨库重构 4.0e-15、反向 74/74） |
| hex | MACH | D225 dof_map dump、D158、hex ND2 内部 dof 300/300（D525） |
| prism | MACH | **D546-548 关闭**（`ND_WedgeElement` 1:1，基/旋度/Nodes/TK 逐槽 p=1..3，点对偶 σ=δ p≤4） |
| pyramid | MACH | 同上（`ND_FuentesPyramidElement` 1:1） |

### 1.3 RT（Hdiv）——细表见 `tmp/d572/collection_provenance.md`（唯一口径登记处）
| 几何 | k=0 | k≥1 | 证据 |
|---|---|---|---|
| tri | MACH（W=I） | MACH | D529 点值对偶（W 谱 3.3e-16）、D492 联合置换 220/220 |
| quad | MACH | MACH（k≤2；k≥2 表已修=D575） | D368/D510/D575/d459/d468 |
| tet | **BIT**（质量矩阵逐项==MFEM 首次，D560） | MACH | D540（W 1.1e-16）、d555/d560/d392 |
| hex | MACH | MACH | D494 1728/1728、D342 k=3..6、P 路值层 W=(1/4)I 框架差（非病，dof==MFEM Project 逐位） |
| prism | **MACH**（RT0Wdg 收编 D572 + 斜扭 pin D584） | TOL（moment-dual 值非 MFEM 点值——**D444 残留**，k≥1 dof 值口径差在案） | d482 逐位、d572 三件、d584 斜扭 46 项 |
| pyramid | MACH（Fuentes；dof 值=RT0_3D 语义） | MACH（RT1..3 开放） | D534/D535（960 raw 0.0）、D536a、D585 裁决（RT0Pyr W=2I 不采用） |

### 1.4 L2（间断）
| 几何 | 状态 | 证据 |
|---|---|---|
| tri/tet | MACH（GL 开式重心） | D226/D269 |
| quad/hex | MACH | D368 等 |
| prism | MACH | D152 系 |
| pyramid | MACH（Fuentes L2；MFEM 自己拒绝 Bergot L2=D335 结论） | D325/D304 |

### 1.5 其他族
| 族 | 状态 | 证据/缺口 |
|---|---|---|
| BDM | LAT | D587（nodal 语义未定义，访问器 panic） |
| 常量/线性族（RT0_2D、Const3D、ND1_3D 等简化 collection） | **?** | 未逐 collection 盘点——进队列 |
| NURBS | BIT | 补丁装配九档 diff=0、细化控制点逐 dof=0（D531）、A5.5 |
| 高阶 ref_elem 家族 | MACH | **D581 关闭（round 60）**：真缺口是 Hex27（全分派器）+Hex20（除 gll_tensor）；体积 parity Hex27 1.6e-15/Prism18 2.8e-16/Pyramid13 3.5e-18；Pyramid13 实为 15 结点（探针证实）。残：D612（Hex27 slot 序三套并存，p_refine 序 det=−0.242 实锤）、D613（曲线标签 vs 表结点数）、D614（postproc 接线清单） |

## 2. 路径矩阵

| 路径 | 状态 | 关键证据 |
|---|---|---|
| 求积 | **BIT**（1-D GL/Gatteschi-Jacobi 240 组合、tri 全阶含 21-25、**26..33 GM 闭式化后 0 ulp[round 60 D603——修前偶数阶点数差 + 奇数阶 ≥27 权重矩解崩溃（偏差 1e5 级含符号错），均为真缺陷]**、prism qt 40188 点零容差、hex/quad 张量） | D339/D364/D564/D565/D372/D578/D603 |
| 装配（标量/向量/混合） | BIT（并行装配 1-ulp 逃生舱 `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS` 在案） | ex24/ex29/ex31 逐字节、d547 prism/pyr ND 装配、D546 面方向 |
| 鞍点/约束（Dirichlet 消除） | MACH | D409/D411（stokes_darcy 2.9512/3.2973 与 round-48 逐位） |
| 离散算子 | TOL→BIT | curl_2d/curl_3d tet BIT、gradient P2→ND2 hex BIT（d110 双恒等式）；**D120/D121 关闭（round 60）**：curl_3d hex 臂（ND2→RT1 参考元矩阵，MFEM 216 项逐项 12 位一致）、三线性感知逐元投影入口（直/翘 hex parity <1e-9）；**joule hex 电磁半块局部目标 el/W-sum/W-max 全部 ~1e-16 相对偏差**（cylinder-hex -o2），完整 ImplicitSolve 链余项 exit(3) 清单化。残：D618（hex divergence 奇异）、D619（H1 hex slot 序≠MFEM，阻 dof 向量跨栈）、D620（Hex20 曲边 EM 角点三线性） |
| 求解器 | MACH | PCG 位历史（round 32 两套 API 教训在案）、AMG/AMS/ADS、LOR 三腿 mesh-independent、block_solvers dyn（D580 六模式逐位） |
| LOR | MACH | lor_rt_pcg/lor_nd_pcg/lor_rt_quad_pcg；D72 秩亏已修 |
| 并行 | BIT（np1/2/4 键表逐位） | D124/D136/D156/D504；残：D527 连续分块（故意不动）、D122 并行 ND2/RT1 ghost（**GAP**） |
| io mesh | BIT（读侧） | D589（vertices token 流对齐）；INLINE 全类型矩阵 vs MFEM 逐元素全同（D582）；**D117/D118+D624 两段关闭（round 60/61）**：混合高阶 H¹ 编号引擎 → 落存储 attach（ragged CSR 派生、零新字段、uniform 逐位不变；llnl-p3 detJ 1.3e-13、wrong orientation 15/26→0、fichera-mixed-16 翻绿、AMR mindet=粗/4 精确）；**D612/D613 关闭（round 61）**：Hex27 官方 slot 序裁决 + p_refine 修正[连带修体心真 bug] + 曲线标签契约；残：D627/D628/D629/D630/D626/D609/D600/D601 |
| io gridfunction | **BIT（双向）** | **D602 关闭（round 60，裁决=误诊）**：写侧本来就 74/74 一致；真缺口是**原生 gf 读取器缺失**——已补 `read_mfem_gf`；三层验收全过（74/0、跨库重构 4.0e-15=MFEM 自写对照同水平、反向 74/74+120/120）。残：D610（只钉了 straight tet ND2）、D611（face_pair_storage_map 仅 tri-face、O(NE) 扫描） |
| NURBS/IGA | BIT | 九档 diff=0、`-patcha -rint` netlib 位级（D533/D563）、`-pa` 全量（D562，pa_data quirk 镜像 + D593 last-ulp 残差） |
| 线代核 | BIT | QR/NNLS vs LAPACK/netlib 位级（D533/D561）； csr/spmv 既有 |
| tmop | TOL | metric 幅值表 25/40 id；1-D 重心式 = **D579（LAT）** |
| Python 绑定 | 复活（round 59） | D596 修复 + wheel 冒烟；D606 门禁已实测、D607 缺口、D608 零测试 |

## 3. examples / miniapps 矩阵

- **86 examples** 全部移植可编译（每轮 `cargo build --release --examples` 0 错误）。
- **逐例三态台账已建（round 61）**：`tmp/ledger/examples_ledger.md`（85 实跑 + 1 未注册死文件；
  C++ 参考 = 现编 `$HOME/mfem410_ser`，diff 证据 `tmp/ledger/logs/` + `tmp/ledger/ref/`）。
  计数：**BIT 4**（ex1/ex2/ex24/ex31，本轮现编 `$HOME/mfem410_ser` 真对拍逐字节）、
  **RUN 63**（含 ex26 = 机器精度级吻合非逐字节；含 pex15 替代档/pex30 两个 >600 s 长跑）、
  **RUN\* 3**（ex4 终误差 27×、ex5 u_err O(1)、ex20 能量恒 1 —— 数值级失配，已立债）、**CRASH 12**（默认档 panic：6×ex15_dump+ex9+ex15dyn+pex9
  = D633 缺 `star-hilbert/periodic-hexagon` 资产；pex5/pex40 = D635a 分区越界回归）、
  **DEV 2**（pex19/pex32 维度裁剪 exit(3)）、**NOREF 2**。
- **Top 缺口清单**（新债 D633–D635，详examples_ledger §新债）：① D633 `data/` 默认网格资产缺失
  （e1f16f8 误删，真值树可恢复，10 文件默认档受益）；② D634 迭代求解器停机规则/残差口径族
  （ex14 前 309 行逐字节后 C++ 停 Rust 不收敛、ex4 27×、ex5 O(1)、ex6/ex29 过收敛——修一台处收益一片）；
  ③ D635a pex5/pex40 并行分区越界（round 30 绿 → 现挂，回归嫌疑）；④ D635b ex20 symplectic 未演化。
- **miniapps**：`miniapps/README.md` 仍为过程底账（67 处 parity、≈37 处 exit(3) 裁剪标注）；
  round 61 抽样复核 8 个 BIT 档（lor_solvers×3、mesh_bounding_boxes×2、block_solvers×3、
  mg_abs_l1_jacobi、nurbs_patch_ex1 netlib oracle 档、schwarz、**mesh_info 现编 C++ 全新逐字节**）
  **全部仍绿**——`tmp/ledger/miniapps_ledger.md`。逐 miniapp 全量三态化仍进队列（底账可信度高，
  建议按目录分轮抽摊）；joule = 电磁半块推进中（D120/D121 round 60 进行中）。

## 4. 未验证队列（`?` 与台账缺口，round 60 起的排单依据）

1. **examples/miniapps 逐个三态台账**（估计是最大的 "?" 集合——一次性批跑脚本 + 汇总表）；
2. 简化 collection 逐个盘点（LinearFE/QuadraticFE/Const3D/RT0_2D/RT1_2D/RT2_2D/ND1_3D/
   RefinedLinear/LinearNonConf 等——fem-rs 是否都有对应、是否有 pin）；
3. 元素层：ND tri k≥2？L2 高阶 hex/prism/pyr 的 prolongation？H1 pyramid prolongation
   （D536b locator 空缺）；
4. 债务号与矩阵格的双向对账（509 个 D 号 → 每个映射到矩阵格或标记 process-only）。

## 5. 真实功能缺口分诊（用户优先级，round 60 起执行）

| 优先 | 项 | 状态 |
|---|---|---|
| P0 | ~~D602 gf 互操作~~ | **round 60 关闭（裁决=误诊）**：文件层无缺陷；真交付 = 原生 gf 读取器补齐。教训入 §7 |
| P0 | ~~D581 Hex27/Prism15+/Pyramid13+ ref_elem~~ | **round 60 关闭**：真缺口 Hex27/Hex20（Prism15/Pyramid13 臂已在——债文部分失真）；体积 parity 机器精度 |
| P0 | ~~D120/D121 joule hex 电磁半块~~ | **round 60 关闭**：curl_3d hex 臂 216 项 12 位、三线性投影 <1e-9、局部目标 ~1e-16；完整 ImplicitSolve 链余项 exit(3) 清单化（D618/D619/D620 登记） |
| P1 | ~~D603 tri ≥26~~（修过程中挖出 GM 权重矩解崩溃真缺陷，一并闭）+ ~~D591 hex IGLL 接线~~（1800 dof 3.5e-15，框架因子全精度钉死） | **round 60 关闭** |
| P1 | ~~D582 INLINE pyramid~~（+3 真 bug 顺带修） | **round 60 关闭** |
| P1 | ~~D31 hex p=2 原子切换~~ | **round 60 关闭（E 路）**：MFEM 三路互证探针裁决 legacy 槽表全错（p=2 与 p≥3 同构算法）；9 文件修正（dof_manager/factory/hex/gmsh/wgsl/curved_hex）；红证据 "slot 8, left: 16"；**ex26 端到端 vs C++ 数值口径一致**（274625 dof；round 61 台账收窄：机器精度非逐字节） |
| P2 | ~~D117/D118 io 混合高阶几何~~ + ~~D624 落存储~~ | **round 60+61 两段关闭**：编号引擎（60）→ 落存储 attach（61，ragged CSR 零新字段、uniform 逐位不变、llnl-p3 1.3e-13 + wrong orientation 15/26→0 + fichera-mixed-16 翻绿 + AMR mindet=粗/4）；残 D627/D628/D629（写出/细化传播/曲面） |
| P1 | ~~D612/D619/D613 slot 序同族~~ | **round 61 关闭（B 路）**：Hex27 官方序=MFEM FaceVert 序裁决 + p_refine 旧序换掉[**连带修体心坐标真 bug：20 结点求和/8 → 角点均值**]；D619 误诊关闭（D31 已顺带修复，零改动 pin 直接绿）；D613 曲线标签契约+DofManager 高阶编号补齐（红 5/6→绿 6/6） |
| P1 | ~~D614 postproc 接线~~ + ~~D615 hex IGLL 装配腿~~（51984 条目 2.26e-13，此前**静默跌落 GaussLegendre**）+ ~~D617 tet GM fixture~~（红证据：旧矩解 tet 段偏差 1e18..2.5e32） | **round 61 关闭（D 路）** |
| **P1（新，round 61 台账扫出）** | **D634 迭代求解器停机规则/残差口径族**（ex4 提前收敛误差 27×、ex5 MINRES 假收敛 u_err O(1)、ex3/6/8/14/29 同族——修一处收益一片）**← round 62 头号候选**；**D633** data 默认网格资产缺失（e1f16f8 误删，恢复即修 10 例）；**D635a** pex5/pex40 并行分区越界回归（round 30 还绿）；**D635b** ex20 辛积分器未演化（输出 1/0） | **待派** |
| P2 | D113 剩余细化几何搬运（Hex27/Prism6/Tet4/Pyramid5） | 队列 |
| P2 | D122 并行 ND2/RT1 ghost 分区缺陷 | 队列 |
| P3 | D597/D598/D605（prism.rs 可写轮）、D590（GLL 生成器）、D616/D627/D628/D629/D630/D636/D637/D638、D570、D579、D593 | 挂起明确化 |

## 6. 维护规程

- 每轮收尾：更新本表受影响格 + 把新债映射到格（拒绝"只进 plan 不进矩阵"）。
- `?` 格只允许通过"补 pin 或登记 GAP/LAT"消除，不允许静默留白。
- 与 `tmp/d572/collection_provenance.md`（RT 专属细表）保持引用而非复制。

## 7. 登记纪律（round 60 教训固化）

**HYPOTHESIS 规则**：登记"缺陷"时，凡验收判据**尚未实际执行**（如 D559 的三层验收在登记时
未跑过文件层对比），债务条目必须标注 `HYPOTHESIS`，修复简报必须包含"先跑判据，若判据通过
则债务以误诊关闭"分支。round 58→60 四连反转（D584 斜扭、D589 vertices、D602 gf、D581 家族
面部分失真）全部属于"登记时未执行最终判据"。**判据跑过才算缺陷，否则只是假设。**
