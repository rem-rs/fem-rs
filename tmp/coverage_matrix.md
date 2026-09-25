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
|Hex20/Hex27/Tet10/Prism15/Pyramid13 的 H1 高阶几何搬运 | — | **round 72 D113 主体关闭**：新增 `linear_view` 分派层（取角点 + 换线性类型 + **原样携带 `nodes` 几何表**，角点按元素自身阶的参考格定位）⇒ **Hex27/Tet10/Prism18/Tri6/Quad9 与 MFEM 4.10 逐值 ≤2.3e-16 顶点 / ≤1.8e-15 子单元体积**；顺带修 **Gmsh `Prism18` 角点提取真缺陷**（D319 layer-major 置换下"每行前 N 个"会取到边中点 ⇒ 楔形退化）；不可搬运的几何表现在**报 warning**（D113）而非静默丢。**余：D113-2**（Pyramid5+曲率：**MFEM 4.10 自己细化曲面金字塔 SIGSEGV**，真值不可得）、**D113-3**（serendipity 三族 Hex20/Prism15/Pyramid13：读者不附几何表且 MFEM Gmsh reader 拒收码 16-19 ⇒ 无真值） | tmp/d113/README.md、crates/mesh/tests/d113_high_order_uniform_refine.rs |

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
| 常量/线性族（RT0_2D、Const3D、ND1_3D、RefinedLinear、LinearNonConf 等简化 collection） | **LAT（round 72 盘点：元素在、无 collection 级 pin）** | round 72 主会话逐个盘点 `tmp/collections_inventory.md`（MFEM `fe_coll.hpp` 40 类为真值源）：13 族已映射到 §1 已有行（Const2D/3D→`P0Tensor/P0Tri/P0Tet/P0Pyr`+L2 o=0、CR→`crouzeix_raviart.rs`、Serendipity→D743/D768、RT_Trace→`dpg_trace.rs`、Gauss*Discont→L2 GL 开式族、P1OnQuad→`QuadQ1`、NURBS 三件→§1.5 NURBS 行）；**真缺口 = `RefinedLinearFECollection`（全库 0 命中）、3-D `LinearNonConf`（只有 2-D 旋转 Q1）、`ND_R2D/RT_R2D` 族（能力在 dim=3 嵌入臂 + `surface_embed.rs`，无逐 collection pin）** |
| NURBS | BIT | 补丁装配九档 diff=0、细化控制点逐 dof=0（D531）、A5.5 |
| 高阶 ref_elem 家族 | MACH | **D581 关闭（round 60）**：真缺口是 Hex27（全分派器）+Hex20（除 gll_tensor）；体积 parity Hex27 1.6e-15/Prism18 2.8e-16/Pyramid13 3.5e-18；Pyramid13 实为 15 结点（探针证实）。残：D612（Hex27 slot 序三套并存，p_refine 序 det=−0.242 实锤）、D613（曲线标签 vs 表结点数）、D614（postproc 接线清单） |

## 2. 路径矩阵

| 路径 | 状态 | 关键证据 |
|---|---|---|
| 求积 | **BIT**（1-D GL/Gatteschi-Jacobi 240 组合、tri 全阶含 21-25、**26..33 GM 闭式化后 0 ulp[round 60 D603——修前偶数阶点数差 + 奇数阶 ≥27 权重矩解崩溃（偏差 1e5 级含符号错），均为真缺陷]**、prism qt 40188 点零容差、hex/quad 张量） | D339/D364/D564/D565/D372/D578/D603 |
| 装配（标量/向量/混合） | BIT（**D754 round 72：并行线性装配已逐位=串行**——单元求值与 scatter 解耦，并行侧按固定块收集 `(dof,value)`、按 dof 稳定排序、全局元素序折叠；`FEM_ASSEMBLY_PARALLEL_MIN_ELEMS` 只剩"何时并行"语义，**旧"1-ulp 逃生舱"表述作废**；双线性站点判定为已确定性[COO 列表 + `into_csr` 稳定排序]） | ex24/ex29/ex31 逐字节、d547 prism/pyr ND 装配、D546 面方向、d754 五测（线性站修前红）、ex26 hex 5 连跑 sha256 全同 |
| 鞍点/约束（Dirichlet 消除） | MACH | D409/D411（stokes_darcy 2.9512/3.2973 与 round-48 逐位） |
| 离散算子 | TOL→BIT | curl_2d/curl_3d tet BIT、gradient P2→ND2 hex BIT（d110 双恒等式）；**D120/D121 关闭（round 60）**：curl_3d hex 臂（ND2→RT1 参考元矩阵，MFEM 216 项逐项 12 位一致）、三线性感知逐元投影入口（直/翘 hex parity <1e-9）；**joule hex 电磁半块局部目标 el/W-sum/W-max 全部 ~1e-16 相对偏差**（cylinder-hex -o2），完整 ImplicitSolve 链余项 exit(3) 清单化。残：D618（hex divergence 奇异）、D619（H1 hex slot 序≠MFEM，阻 dof 向量跨栈）、D620（Hex20 曲边 EM 角点三线性） |
| 求解器 | MACH | PCG 位历史（round 32 两套 API 教训在案）、AMG/AMS/ADS、LOR 三腿 mesh-independent、block_solvers dyn（D580 六模式逐位） |
| LOR | MACH | lor_rt_pcg/lor_nd_pcg/lor_rt_quad_pcg；D72 秩亏已修 |
| 并行 | BIT（np1/2/4 键表逐位） | D124/D136/D156/D504；**round 72：D122 登记症状 = 过期行**（隔离 worktree `af9c765c` 红证据：cylinder-hex ranks 2/3/4 = 336/816/1172 条 sentinel、64 元 hex = 64 条 + `ghost.rs:178` panic；HEAD 全扫 6 网格 × ranks{1..4} × 9 族 = 216 档全绿 ⇒ 已由 D412[round 48] 关闭）⇒ 现存活缺陷：**D122-1**（H1/ND 边 dof owner = 端点 min ⇒ per-rank TrueVSize 与 MFEM 不等[ND1 596/373 vs MFEM 525/444]；**补丁在案 `tmp/d122/d122_1_patch_dof_partition.diff` ⇒ 八空间逐位=MFEM**，但会平移所有 np≥2 基线 + 需同步三条 pin ⇒ **round 73 头号，与 D122-3 同轮**）、**D122-2**（ghost 层因面闭包 fixpoint 变成传递闭包：np=2 每 rank 持整网，MFEM 应为 1 层 54 元）、**D122-3（数值缺陷）**：NDk k≥2 装配的 `Σ M_ij` 在 np≥2 不守恒（ND2 np2 6.008e-3/np4 1.010e-2、ND3 1.6e-4/4.3e-4；RT0/RT1/RT2/L2/ND1 精确；HYPOTHESIS = D412 的 ND 面 dof 键）；残：D527 连续分块（故意不动） |
| io mesh | BIT（读侧） | D589（vertices token 流对齐）；INLINE 全类型矩阵 vs MFEM 逐元素全同（D582）；**D117/D118+D624 两段关闭（round 60/61）**：混合高阶 H¹ 编号引擎 → 落存储 attach（ragged CSR 派生、零新字段、uniform 逐位不变；llnl-p3 detJ 1.3e-13、wrong orientation 15/26→0、fichera-mixed-16 翻绿、AMR mindet=粗/4 精确）；**D612/D613 关闭（round 61）**：Hex27 官方 slot 序裁决 + p_refine 修正[连带修体心真 bug] + 曲线标签契约；**D651 关闭（round 64）**：直面 tet 均匀细化顶点编号/子单元发射序对齐 MFEM（ex3 3-D 轨迹逐字节）；**D663 关闭（round 65）**：写路径 tet 规范化删除（三问裁决落盘；refined_tet diff 39264→0，常驻回归 d663 双夹具）；**VTK 读入 = D675 关闭（round 65）+ D688 旋钮关闭（round 66）**：`vtk_legacy_reader` 17 cell 类型探针金标 + 三资产 Print 逐字节 + `refine/fix_orientation` 旋钮[简报修正：真实语义 (false,true)] + trimmer 默认档 diff=0 六档全零；残：D627/D628/D629/D630/D626/D609/D600/D601/D689/D686/D687 |
| io gridfunction | **BIT（双向）** | **D602 关闭（round 60，裁决=误诊）**：写侧本来就 74/74 一致；真缺口是**原生 gf 读取器缺失**——已补 `read_mfem_gf`；三层验收全过（74/0、跨库重构 4.0e-15=MFEM 自写对照同水平、反向 74/74+120/120）。残：D610（只钉了 straight tet ND2）、D611（face_pair_storage_map 仅 tri-face、O(NE) 扫描） |
| NURBS/IGA | BIT | 九档 diff=0、`-patcha -rint` netlib 位级（D533/D563）、`-pa` 全量（D562，pa_data quirk 镜像 + D593 last-ulp 残差） |
| 线代核 | BIT | QR/NNLS vs LAPACK/netlib 位级（D533/D561）； csr/spmv 既有 |
| tmop | TOL | metric 幅值表 25/40 id；1-D 重心式 = **D579（LAT）** |
| Python 绑定 | 复活（round 59） | D596 修复 + wheel 冒烟；D606 门禁已实测、D607 缺口、D608 零测试 |

## 3. examples / miniapps 矩阵

- **86 examples** 全部移植可编译（每轮 `cargo build --release --examples` 0 错误）。
- **round 62–64 增量（对照 round 61 初账）**：CRASH 12 → **0**（D633 资产回填 + D635a
  分区越界修复）；RUN\* 3 → **0**（D634 停机规则族 = ex4 646 it=C++、ex5 397 行逐字节、
  ex14/ex6/ex29 同族；D635b ex20 辛积分器步进调用修复）；**ex3 双档逼近 BIT**——3-D
  beam-tet PCG 轨迹 137 行逐字节（round 64 D651）、2-D star.mesh 误差行逐字节级（round 64
  D653），仅剩 3 项打印格式差 = D664；pex40 页脚 np=1 逐字节（D655）。examples 新债
  D672/D673/D674（手写误差评估器残留族，D656 审计 86 例全扫产出入账）。
- **round 66 增量（B/C/D 路）**：**ex1 升 BIT**（D687，第 6 个逐字节档）；**ex24 三口径
  逐字节**（D686；-p1 豁免 5 行 = D700）；**ex22 双根因裁决**（D681 组装无辜[评估器 Q2 角槽
  几何基] + D682 pc 丢 ω/缺 DIAG_ONE——核心交付复数 L2 误差 API，示例侧收口 = D693-695）；
  **pex3 仲裁双重成立**（D683→D697 回撤回归，头号嫌疑 25c4c99）；死 API 删除（D684）；
  警告 **96→5**（余 = D701）。新债 D693-D701。
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
- **miniapps 全量三态化完成（round 63）**：`miniapps/` 100 个 .rs 全部定档——
  `tmp/ledger/miniapps_ledger.md`（round 61 抽样 7 文件 + round 63 全量 93；C++ 参考 =
  现编/现跑 `$HOME/mfem410_ser` 串行 miniapp + `$HOME/work/**` 历史参考二进制；日志
  `tmp/ledger/logs/mini_*.log`、C++ 快照 `tmp/ledger/ref/cpp_*`）。计数：**BIT/BIT\* 21 文件**
  （r61 7 + 本轮 14 真对拍：nurbs ex1/ex3/ex5/ex24/printfunc/curveint/surface、hpref、toroid、
  twist、extruder、field-diff/field-interp/get_values——其中 field-interp SHA256 与 get_values
  数值链为逐字节/精确复现；8 个文件带已立债豁免行）、**RUN 43**（含 RUN-LONG 5：
  multidomain/navier_3dfoc/navier_turbchan/life 无界档/…；pdiffusion/pacoustics/navier_mms/
  dpg_maxwell_3d 等 6 项数值与 C++ 记录**逐位**复现）、**RUN\* 2**、**CRASH 2**、
  **DEV 21**（README「诚实 exit(3)」清单本轮验证 21 处**全部兑现，零回潮**）、**NOREF 6**（spde×5 +
  hdiv_linear_solver 模块）。`miniapps/README.md` 仍为过程底账（67 处 parity、≈37 处 exit(3) 标注）。
- **miniapps 新债（round 63 登记，只登记不修）**：**D657** multidomain_nd/_rt 界面 dof panic
  （CRASH，`BlockToCylinderMap: no block dof`）；**D658** navier_bifurcation PRES PCG 不收敛
  （RUN\*，残差 4.6e+1@200it，与 README 收敛区一致记录冲突）；**D659** trimmer 双重阻塞
  （beam-tet.vtk 资产缺失 = D633 残留 + VTK reader 未实现，rc=1 非声明 exit(3)）。
  另：maxwell.cpp 为 MPI-only（串行树永无现编 oracle）；twist 默认档已随 nodes writer
  落地从 exit(3) 升级为 rc=0（README 记载过期）。joule = 电磁半块推进中（D120/D121/D618-620）。
- **round 65 增量（A/B/C/D+D2 路）**：**ex3 双档升 BIT**（D664：4 处打印格式差全修，轨迹 137/386 行一字不动——examples 第 5 个 BIT 档）；**ex40 终值真值反转**（旧锚 0.0268745 错、核心换算后 0.0269215 = C++ 0.0269214，D674）；ex22 `-o 2` 评估器阶感知修复（-p1 误差 2.0e-1→2.799165e-2；全对齐挡在 D681[D681：-p 0 H1 Q2 路径病]/D682[GMRES 停滞]）；ex18 分派修复默认逐字节不变；**examples 警告清扫 8 文件闭环**（余量 96 条/45 文件为既有积压）。新债 D681-D687。
- **miniapps round 64 增量（B/D 路）**：**CRASH 2 → 0**（D657 multidomain_nd/_rt 崩溃修复：
  实体化配对 + 几何因子定号，dof/ess 计数与 MFEM 全等；数值残差 = D667）；**RUN\* −1**
  （D658 navier_bifurcation 裁定**无回潮**——C++ 串行镜像同档逐位同停滞，系无 hypre 的 MFEM
  固有行为；新增 `-pc amg` 收敛档 24-28 it）；trimmer DEV(阻塞) → RUN(.mesh 对拍，hex diff=0)
  + DEV(.vtk 诚实 exit(3))。**miniapps CRASH 清零**。新债 D667/D668（勘误）/D675（VTK reader
  GAP）/D676（写路径 tet 规范化，与 D663 同族合并追踪）。
- **miniapps round 65 增量（D/D2 路）**：**D675 关闭**——`vtk_legacy_reader` 落地（17 cell 类型
  探针金标 + 三资产 Print 逐字节），**trimmer 默认 .vtk 档撤 exit(3) → 真 RUN**（48 elements/
  36 nodes = C++）；**tet `-a 1` 档 82/83 → 83/83 diff=0**（D685/D676 关闭，D2 修正配方：奇置换
  Swap 实测必要）；默认档残差 91 对保几何循环旋转 = **D688**（reader 缺 refine 旋钮，修法已
  备）；**D665 关闭**（refine_uniform_3d 消费方方向全部稳定/前进/对上 C++，joule rc=3 = 声明
  裁剪）。multidomain RT cyl 残差随 D667 修复 + `-qp 9` 从 ~870× 降至 −16%（1.1962e-6 vs C++
  1.43051e-6；余量 = D677 submesh 曲率 + D678 阶表，RUN* 维持）。

## 4. 未验证队列（`?` 与台账缺口，round 60 起的排单依据）

1. **examples/miniapps 逐个三态台账**（估计是最大的 "?" 集合——一次性批跑脚本 + 汇总表）；
2. 简化 collection 逐个盘点（LinearFE/QuadraticFE/Const3D/RT0_2D/RT1_2D/RT2_2D/ND1_3D/
   RefinedLinear/LinearNonConf 等——fem-rs 是否都有对应、是否有 pin）——**round 72 已执行**
   （`tmp/collections_inventory.md`；§1.5 的 `?` 已改写为 LAT，剩 3 条子项各带 recipe：
   `RefinedLinear`、3-D `LinearNonConf`、`_R2D` 族 pin）；
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
| **P1（round 62 关闭，台账新债清)）** | ~~D634 停机规则族~~（**根因=七例求解器配置漂移，核心 API 两套语义全对**[round 32 对齐守住]：ex4 字面量预开方松 10 量级→646 it=C++ 且 0-588 行逐字节、ex5 改用 1:1 mfem_minres→397 行逐字节、ex6/ex14/ex29 同族、ex3 初值口径、ex8 内层判据；核心唯一真缺陷=mfem_minres 打印门；**回归红线 ex1/ex2/ex24/ex31/d367/d370/mg 全绿**）；~~D635a pex5/pex40 并行越界~~（二分闭环：`0e19c81` Step 0b 三分量硬编码 dot on dim-len 切片；修复后与回归前父构建**逐位一致**）；~~D635b ex20 辛积分器未演化~~（根因=**步进调用被过期 TODO 注释**，SIAVSolver 早已建成；六配置逐字节=C++）；~~D633 data 资产缺失~~（23 件回填 MD5 校验、ex15dyn 513s 全程 rc=0） |
| **P1（round 63 关闭，D634 收口）** | ~~D639 ex4/ex5 误差评估器~~（手写版硬编码三角 RT0 假设 vs quad 网格[三角求积只盖左下半，0.495≈1/2 面积]；换核心后 **ex4 0.0161443 / ex5 0.000143587 与 C++ 逐字节**）；~~D640 ex8 块装配~~（**真核心 bug：sinv.rs 四边形分支 J⁻¹/J⁻ᵀ 写反**——轴对齐单元两者相同故历史验证全盲；16 行修复后 ex8 29 it、DPG 0.0183277 逐字节、square-disc 交叉验证 ✓）；~~D641 EliminateVDofsInRHS~~（`form_linear_system_vdofs` 落地 + **修正 round-62 两处诊断**[PartMult=赋值 ⇒ 消元口径本就等价；IterativeSolver 默认 iterative_mode=true]；ex3 剩余残差铁证 = **tet-ND 边编号置换 vs EnumEdges**[A 对角线前 12 项逐位、第 13 项分叉]→ **D651 round 64 头号候选**）；~~D644 pex5/pex40 参考~~（官方源= ex5p/ex40p；ex5p RUN[Schur AMG 参数族 44vs95 it]、ex40p RUN⁺[≤2e-3]） |
| **P1（round 63 关闭，prism 可写轮五件）** | ~~D597 行表下沉~~（`prism.rs::mfem_nodal_rows` 单一来源，26/26 红线逐位）、~~D605 警告~~、~~D598 coord_twins 夹具~~（2-hex AddHexAsWedges pinched；中和实验证真；**连带更正 d584 docstring：two-wedge 夹具 twin_groups 实为空**）、~~D646 SIAV 双实现~~（symplectic 版零消费方删除，lib 270→264 账目吻合）、~~D647 ex2 升全逐字节~~（**简报前提再修正：MFEM 4.10 ex2.cpp 根本无 Wrote 打印**——删行后 278 行/11783 字节 cmp 全同、stderr 双侧 0） |
| **P1（round 64 关闭）** | ~~D651 tet-ND 边编号置换~~（**根因反转：不在 space[dof_manager/hcurl 无辜]，在 mesh 细化层**——直面 tet 均匀细化顶点编号用 first-touch 而 MFEM `UniformRefinement3D_base` 用字典序 `oedge+e2v`、加镜像子单元发射序[0↔1 转置翻 det(J)]；修 `amr_inner.rs` 两处门后 **ex3 3-D 137 行 (B r,r) 逐字节**；16/16 GetElementDofs 表逐位）；~~D652 sinv 负 det~~（**裁定反转：D640 已顺带消除"垃圾块"[\|det\|]；真缺陷 = 静默修反转单元 vs MFEM Weight() 带符号**——修后带符号语义 + ex8 逐字节不变）；~~D653 ex3 2-D l2_err~~（换核心 compute_hdiv_l2_error，star.mesh `1.34917895677130e-2` = C++ 0.0134918）；~~D657 multidomain_nd/_rt 崩溃~~（实体化配对+几何因子定号；dof/ess 与 MFEM 全等；数值残差 → D667）；~~D658 navier_bifurcation~~（**无回潮裁定**：C++ 串行镜像逐位同停滞；`-pc amg` 档交付）；~~D659 trimmer~~（.vtk 回填+诚实 exit(3)+.mesh 对拍 hex diff=0）；~~D654 ex5p AMG~~（**豁免[结构性]**：hypre 经典 HMIS 族 vs 聚合式不可达，5 组对照实验）；~~D655 Total dofs~~（pex40 页脚改本地真 dof 和，np=1 逐字节）；~~D660 prism 双源~~（验证性关闭，D597 已收敛，26/26 红线）；~~D661 interpolate_vector 重建~~（hoist 4.2×，逐位 pin）；~~D662 漂移审计~~（定向抽样无新漂移）；~~D656 手写误差审计~~（86 例全扫 → D672-674） |
| **P1（round 65 关闭）** | ~~D667 部分~~（**潜伏核心 bug：vector_assembler 两处 det.abs() = D652 同族**，负对角探针红→绿 ≤1e-11；生产网格无折叠单元[诚实披露，几何路径逐点对齐]——2.2× 真驱动 = **D677 submesh 丢曲率 + D678 积分阶表错**[RT1 GetOrder=p+1 勘误、DivDiv=2k=8 点勘误]；`-qp 9` 档 RT cyl 870×→−16%）；~~D672~~（order 感知分派；ex22 -o1 逐字节不变；全对齐挡 D681/D682）；~~D673~~（死函数+文件级 allow 清，13→0 警告）；~~D674~~（ex40/pex40 换核心 + **真值反转**：旧锚 0.0268745 错、新值 0.0269215 = C++）；~~D663~~（写路径规范化删除，refined_tet diff 39264→0）；~~D664~~（**ex3 双档 BIT**，第 4 处格式差连修）；~~D675~~（VTK reader 17 cell 金标 + trimmer 默认 .vtk 真 rc=0）；~~D685/D676~~（tet 83/83 diff=0；奇置换 Swap 修正）；~~D665~~（消费方方向全稳定/前进） |
| **P1（round 66 关闭）** | ~~D677~~（submesh 携带父曲率，MFEM SubMesh 语义；直面惰性——fem-parallel 308/0；**ND block = C++ 6 位**、RT cyl ssq 870×→+21%；残差 D690）；~~D678~~（阶表换 helper，双勘误落地）；~~D681~~（组装证伪嫌疑[4.4e-12]；病灶 = 示例评估器 Q2 角槽几何基；核心交付 `ComplexGridFunction::compute_l2_error`，全管线 **5.643641e-3 = C++ 逐位**）；~~D682~~（系统正确；根因 = 示例 pc 丢 ω + 缺 DIAG_ONE；C++ 全配方 43/116 it = C++）；~~D679 车道内 26 站~~（带符号化 + 红测 pin；余量 D696）；~~D688~~（reader 旋钮；简报修正 = 真实 (false,true)；trimmer 默认档 91 对→diff=0 六档全零）；~~D683~~（**双重成立**：台账过期 且 HEAD 真回归 → D697）；~~D686~~（ex24 三口径逐字节；-p1 = D700）；~~D687~~（**ex1 BIT 收官**）；~~D684~~（死 API 176 行删）；~~警告专项~~（96→5，余 D701） |
| **P1（round 67 关闭）** | ~~D693~~（quad `-p0 -o2` **5.643641e-3 = C++ 逐位**；3-D 同修）；~~D694/D695~~（**裁决：系统侧 ω 早已乘好[complex.rs:271]，缺的只是 pc 直配**——pc 补 ω + DIAG_ONE 对称消元后 `-p1 -o2` **5.622973e-3/6.420253e-3 = C++ 逐位**、hex p1 误差行逐位；迭代 44/119 vs 42/116 = D704）；~~D697~~（**根因反转：crates/parallel 无罪**——`25c4c99` 起示例把 ess 值写死 0.0；pin 测试落地 + **D705 主会话落地验证 ranks1/2 = 87/102 it、2.70053 = C++ 六位**）；~~D690 根因一~~（VectorMass 阶钩子从连接性取 OrderW、曲面 P2 实为 5——geom 感知钩子修复后 block = C++ 6 位；根因二 = **D708** crates/space RT1-hex 曲面槽位）；~~D696 批 1/2~~（vector_assembler 漏网 2 站 + partial 4 + mixed 6 + dpg 六文件，12 pin；批 3/4 余量）；~~D700~~（求积阶翻译错：MFEM RT GetOrder=p+1→默认阶 5，Rust 误用 3——3 误差行逐字节清零；余 2 行 = D712 ulp 噪声[splice 实验证明]）；~~D680~~（家族/几何感知默认阶表 + MFEM 夹具 ≤1e-11；multidomain 轨迹随真实阶 9 移动 = 向 C++ 收敛）；~~D701~~（全清） |
| **P1（round 68+69 关闭）** | ~~D708~~（**槽位假设证伪**：曲面 hex 有序表/局部阵/传递映射与 MFEM 逐位；真凶 = 串行移植数据流[Transfer 读并覆写目的 gf、上游从不用 RK3 结果更新]；gf-shadow 落地后 **rt cyl −6.420799e-6/1.430509e-6 = C++ 全吻合**；d708 七金标 + d709 P-共轭 pin）；~~D702~~（3-D H1 装配无罪 ≤2.4e-15；真凶 = ex22 3-D **虚部 BC 清零**；D718 落地 hex o1/o2 = C++ 括号值）；~~D712~~（**边界裁决**：参考域约定差 [-1,1]³ vs [0,1]³，数学 1e-11 同；收尾 = D721/D722 刻意路径；d712 pin 入库）；~~D713~~（合流重验收完成）；~~D706~~（并行+串行双入口落地，pex3 87/102 it 逐位保持，迁移积压 D739）；~~D696 批 3/4~~（全裁决；**d365 金字塔红灯实探 = MFEM 顶点朝下金字塔 Weight≡+1 ⇒ 帧 |det| 是 parity 归一**，d365 5/5 恢复）；~~D715~~（curved_hex_geometry isoparametric 化，直面 bit-dump diff=0）；~~D724~~（MfemFile+mesh1d segment 分支，三档 Make1D f64 逐位；d602 旧 pin 翻正）；~~D727/D728~~（complex:762 保 abs = MFEM 逐元 fabs 本尊）；~~D719~~（管线证=MFEM，根因在示例评估器 → D738）；~~D720~~（C++ 同档同样不收敛 = parity） |
| **P1（round 69 关闭）** | ~~D721~~（**裁决 = 有条件 go**：爆炸半径清点[求积接现成 gauss_legendre_01、形函数 ~13 文件、mesh 顶点零改动、PA 核 ~6 文件 = 唯一位级风险内核；**hex = 最后一个非 MFEM 域现役族，D721 = D364 分裂画句号**]；试点实证 [0,1]³ RT0 基+规则+MFEM J → 质量阵 **36/36 逐位**；条件 = hex 原子翻转 + d680 升位级门 + D730 裁出 + gauge 同轮）；~~D715~~（curved_hex_geometry isoparametric：两路径 to_bits 相等、直面 bit-dump diff=0）；~~D724~~（MfemFile+mesh1d，三档 Make1D f64 逐位）；~~D706 串行~~（form.rs ElimPolicy+eliminate_ess_tdofs，3/3 位级；迁移 = D739）；~~D727/D728~~；~~D720~~；~~multidomain 升 BIT\*~~（11 格式差对齐后三档各仅 IC 尘行豁免[D735]、**7 位全轨迹对齐**） |
| **P1（round 70 关闭）** | ~~D721 实施~~（**hex [0,1]³ 原子翻转落地**[三轮接力：A 主体 56 文件 → A2 四红灯 → A3 集成残留清剿 51→0]：`hex_rule` 位级 MFEM 表 + **x-fastest 枚举**、`HexQ1` 逐字 `TriLinear3DFiniteElement`、`HexQk` 复用 D364 barycentric、RT/ND hex pull-back 全删、**J op-order 内核 = `geo_ref_elem_from_mesh` 直 hex 返 `HexQ1`**（27/27 位级常驻 pin）、PA 核六文件 [0,1] 化、六处 shim 删除；**验收：ex24 四口径全部逐字节 = C++（`-p1` iter1 1.47776e-22 / ARF 9.13511e-13 归零）⇒ D712 关闭**；d680 RT0/RT1/ND1 位级；multidomain step-250 逐位保持；七 crate 292 targets/4092/0）；~~D738~~（**求积阶不是点数**：`quad_rule_01(n)` = (n+2)/2 点/维，MFEM intorder 2·GetOrder+3 ⇒ ND1 用 5 阶/9 点而 fem-rs 用 6 阶/16 点——一行修复后 2×2 oracle 8 位吻合、quad/tri `-p1 -o1` = C++）；~~D739~~（7 站点迁移：6 站逐字节含 ex31 0.181455；**ex27 `-dbc≠0` 旧管线被证错**——Dirichlet 违反 40%→迁移后 = C++，主会话裁决保留[1:1 前提条款]）；~~D737~~（H1 canonical：官方 PAR-only 实证 → print-ref；7 格式差对齐；**规则修 MFEM 逐积分器默认**[曲 9/9/6=125/125/64pt]；**数据流补上游打印块 GF 刷新** `cyl_gf_state`；alt 档全字节、t005 block 全一致+cyl 第 7 位）；~~D736 收窄~~（rt/nd `-tf 0.5` 50001 step 逐字节全对齐——t≤0.5 通过，t>0.5 HYPOTHESIS）；~~D735 前提被驳~~（MFEM `ProjectBdrCoefficient` **不是 L2 投影**=逐节点求值+触数均值；尘值恒 2^-54 不可复刻且随几何舍入不稳 → 不建 parity API，豁免收窄 = D747）；~~D742/D744/D745~~（A2：RT1 位级[47413 项 dump 零失配]、d386 2/2、LOR rt 43→67 绿）；~~D757~~（A3 清剿 15 站：又修 5 处库内真缺陷——`error_estimate::vertex_shapes` hex 角权、`transfer::hdiv_elem_frame`[d468 hex RT0 48/48 位级]、**`dpg_basis::ref_node_coords(Hex8)` 帧不同步[Stokes 1.8e1→6.7e-16]**、`legacy::LegacyHexQ3`[fichera-q3 6e-15]、`vtk_legacy_reader` type 29[18×2 golden 字节同]） |
| **P1（round 71 关闭）** | ~~D749~~（**数据流落地上游**：对照物逐行审计[GF 触碰点普查确认刷新变体忠实]；端口打印步刷新 gf_state = 模型 ③；三档 vs 刷新 harness 逐字节仅 IC 行；**红线四值重钉：rt cyl −2.137667e-4/1.439350e-6、nd cyl 6.932270e-5/1.594225e-4**；D736：rt t≤0.5 / nd t≤0.2 逐字节）；~~D746~~（**三处并行不确定真缺陷根治**：DofPartition HashMap 迭代序+缺 tie-breaker[主因，解释 -o1 稳/-o2 抖]/ParVector rayon dot/ChannelBackend 到达序 allreduce；5 连跑逐位；pex3 锚复活、round-69 的 …562224e-4 永久作废）；~~D748~~（`paired_vector_reference_element` 公开 API + tet 面块 canonical→element-local 旋转；四族红→绿 = C++；红线逐字节保持）；~~D743~~（HexSerendipityPk → [0,1]³[p=1 位级=MFEM]；**审计更正：nurbs hex 臂本就新帧，漏的是其 quad 兄弟臂**→quad_rule_01；唯一消费方无生产调用者）；~~D729~~（**prism_pk 轴向 shim 真缺陷**：n≥8 落梯形法 ⇒ p≥4 PA 44% 误差→共享 MFEM 表；**geometric_mg SumFact 真缺陷**：[-1,1]² 规则点喂 [0,1]² 公式 ⇒ 5.15e-2→5.4e-16）；~~D760~~（zz 三维采样点=参考质心，3 测）；~~D761~~（Hex20/27 geom_rule+vertex_shapes 补齐；**连带系统性缺口：Tet10 五个 simplex dispatch 漏网**→主会话按 D581 补，d761 6/6）；~~D753~~（ex27 初值 X=R·x + sqrt 容差 ⇒ **双档迭代历史与 C++ 零差异**[29/30 行]） |
| **P1（round 72 关闭账）** | ~~D765~~（**逐槽裁决 + 三通路修复**：`QuadND` = MFEM `ND_QuadrilateralElement(p, GLL, GL)` 逐槽本尊、空间表=MFEM `GetElementDofs`（含 `−1−id`）、legacy `QuadNDk` 24/24 节点失配 ⇒ 装配器/postproc/mixed/dpg/评估器全改 `QuadND` o≥3 + `interpolate_vector` 补 quad k≥3 内部 dof 分支；验收 `-p1 -o3` 9.424581e-3/6.520012e-3 = C++（修前 7.0×）、`-o4` = C++（修前 37×）；红线逐位；space 4/4 + assembly 2/2 修前红）；~~D754~~（**真因反转**：登记归因 `Vector::dot` 被证伪；真缺陷 = 并行**线性**装配 reduce ⇒ 修后**逐位=串行**、ex26 5 连跑 sha256 全同 = 修前串行值；双线性两站经单元测试判定本就确定）；~~D766~~（`Vector::dot/norm` 确定性加固，4 测修前红，无生产调用方）；~~D767~~（**库侧 `mfem_gmres` 逐行移植 + 示例 pc 补 loss 项**：tri `-p2 -o1` 1000 it 打转 → **276 it 收敛**（C++ 266）；tet/tri `-o2` 维持 No convergence = parity；误差行全保；残余 = D780）；~~D771~~（ex27 **49/51 行逐字节** + `-dbc 2.5` 50/52，gold 出处已核；残 = D778/D779）；~~D764~~（旧 print-ref 夹具全打 SUPERSEDED）；~~D763~~（`-vs` 纪律入 README+台账；主会话独立证实打印步长改轨迹）；~~D762~~（**豁免**：漂移由 `beb8442a`（D721 [0,1]³）引入，640 ess 稳定、192 个精确零全在 z 法向面，pin 为结构不变量，三档 stdout 逐字节不变）；~~D768~~（serendipity 2-D → [0,1]²+`quad_rule_01`，红 5/5→绿；审计更正：≠MFEM `H1Ser_*` 类）；~~D769~~（SumFact scratch 字面量 ⇒ p≥5 越界 panic 修复）；~~D770~~（**比登记更宽**：首 QP 标量几何 ⇒ **拉伸仿射棱柱 240% 误差**；改每 QP 解析 Jacobian + 带符号 det） | — |
| **P1（新，round 72 登记）** | **D772**（`mixed::ref_elem_vec` HCurl **tri** 臂缺 o≥2：o=2 拿 `TriNDk(2)` 配 `TriND2` 空间表）；**D773**（`dpg_basis::hcurl_ref_elem(Tri3,p)` p=2 同族潜伏）；**D774**（`pa::prism_pk` 全量跑偶发 flake）；**D775**（`lor_factory.rs:564/1337` 注释"库默认 quad 元 = legacy QuadNDk"已过期）；**D776**（`vector_assembler.rs` 的 `n_i>0 && Quad4` bubble 分支 `[−1,1]²` 语义，当前不可达）；**D777**（**D721 残项：GPU hex WGSL 着色器**——`bary/dary` 节点已在 round 72 重生成（两 pin 修前红），但 `GP/GW` 求积常数仍在 `[-1,1]` 而全树（element `hex_rule`/CPU PA `gauss_legendre_01`）是 `[0,1]` ⇒ 需 generator 改发 `[0,1]` 规则 + q1..q4 重生成 + 求积帧 pin + GPU 数值 pin；分析在 `tmp/d721x/README.md`）；**D778**（ex27 边界平均残差 = `integrate_bc` 半面求积[`[0,1]` 点喂 `[-1,1]` 映射] + 网格构造顺序差；加性账目已闭合，配方在案，pex27 同款）；**D779**（ex27 `-dg` 未对齐：第 0 步 `(B r,r)` 0.142775 vs 0.0220206 + `rtol` 语义）；**D780**（ex22 pc 元素级合成：`‖M b‖` 首值仍 2.20×、收敛档普遍 +1 迭代）；**D781**（`vector_assembler` 矢量线性形并行归约未审计——主会话筛为阴性[ex31 3 连跑+线程扫全同]，配方在案）；**D782**（MFEM 2-D serendipity `H1Ser_*` 语义端口缺失：GLL/`S_p`/`(p²+3p+6)/2` dof）；**D783**（曲面棱柱 PA 侧仍用 6 顶点三线性几何）；**D784**（prism PA 测试均单元素，无多元素共享 dof 的 PA-vs-装配 pin）；**D785**（joule.rs:663 的 `n_bdr` 取自本地网格 ⇒ np≥2 ess 掩码与 C++ 不一致）；**D786**（np > 单元数时空 rank 在 `HCurlSpace::new`[hcurl.rs:852] panic + ThreadLauncher 挂住）；**D787（重要）**：`fem_mesh::transformation::element_jacobian_at` 对**曲面 tet/prism 无等参分支**（走直线映射；实测母单元体积 2.0e-1 vs 真值 1.8298e-1 = **9.3% 偏差**），而 `compute_l2_error`/`hdiv_error`/`flux_recovery`/`dgmassinv`/`complex`/`topology_optimization` 均消费它 ⇒ 曲面单纯形网格上的误差估计器/后处理受影响（配方 `tmp/d113/README.md §4.2`）；**D788**（D113-2：Pyramid5+曲率搬运；**MFEM 4.10 自己细化曲面金字塔必 SIGSEGV**[ASan 栈 `UpdateNodes→RefinementOperator::Mult→Table::GetRow`] ⇒ 真值不可得，HYPOTHESIS 级）；**D789**（D113-3：serendipity 三族 Hex20/Prism15/Pyramid13——读者有意不附几何表 + MFEM 拒绝 gmsh 码 16-19 ⇒ 无真值；配方 = `findpts::incomplete::eval_map` 做父求值器）；D752/D755/D756/D758/D759/D750/D751/D729 余项（pyramid D339）维持。**round 73 头号 = D122-1（边 dof owner）与 D122-3（数值不守恒）同轮** | 待派 |
| P1 | ~~D612/D619/D613 slot 序同族~~ + ~~D614 postproc 接线~~ + ~~D615 hex IGLL 装配腿~~ + ~~D617 tet GM fixture~~ | **round 61 关闭（D 路）** |
| P1 | ~~D632 pyramid 细化家族错配~~（真根因=CurvedMesh 三求值器走 factory；∫|detJ| 0.1434→1/3）+ ~~D636 elem_vol hex 恒 0~~（η 0→√3、ZZ 全 0→全>1e-10）+ ~~D637 flux_recovery 推断~~（显式 D637 拒绝替代静默 0.0） | **round 62 关闭（D 路）** |
| P1 | ~~D638 HCurl hex IGLL 装配腿~~（曾静默跌落 GL，95184 项 1.88e-14、框架因子=1） | **round 62 关闭（C 路）** |
| P2 | D113 剩余细化几何搬运（Hex27/Prism6/Tet4/Pyramid5） | 队列 |
| P2 | D122 并行 ND2/RT1 ghost 分区缺陷 | 队列 |
| P3 | D597/D598/D605（prism.rs 可写轮）、D590、D616/D627/D628/D629/D630/D648（金字塔标记细化越界实录）/D649（hex RT 面旋转）/D650（CurvedMesh JacobianCache 直边化风险）、D570、D579、D593、D644 | 挂起明确化 |

## 6. 维护规程

- 每轮收尾：更新本表受影响格 + 把新债映射到格（拒绝"只进 plan 不进矩阵"）。
- `?` 格只允许通过"补 pin 或登记 GAP/LAT"消除，不允许静默留白。
- 与 `tmp/d572/collection_provenance.md`（RT 专属细表）保持引用而非复制。

## 7. 登记纪律（round 60 教训固化）

**HYPOTHESIS 规则**：登记"缺陷"时，凡验收判据**尚未实际执行**（如 D559 的三层验收在登记时
未跑过文件层对比），债务条目必须标注 `HYPOTHESIS`，修复简报必须包含"先跑判据，若判据通过
则债务以误诊关闭"分支。round 58→60 四连反转（D584 斜扭、D589 vertices、D602 gf、D581 家族
面部分失真）全部属于"登记时未执行最终判据"。**判据跑过才算缺陷，否则只是假设。**
