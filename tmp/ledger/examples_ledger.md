# examples 三态台账（round 61 · C 路 · 2026-09-22/23）

> 86 个 `examples/*.rs` 一次性扫出。**口径**：
> - 实跑环境：Windows release exe（`target/release/examples/`，e1f16f8 后工作树），CWD = `tmp/ledger/rundir/`
>   （带 `data` junction 指向仓库 `data/`，输出文件全部落在 rundir，不入根目录）。
> - pass1 = 无参数默认档；pass2 = 标准档 `-m data/<默认网格> -no-vis`（与 C++ 同参数）。
> - C++ 参考：**本轮现编** `$HOME/mfem410_ser`（MFEM 4.10，g++ -O2 串行），命令与产物在
>   `$HOME/work/r61ref/`，stdout 快照拷贝至 `tmp/ledger/ref/`；diff 全部真跑（`diff` 原文见 logs/ref）。
> - **分类图例**：BIT = 与 C++ stdout 逐字节（注明豁免行）；RUN = 跑通 rc=0、无逐字节对照；
>   **RUN\*** = 跑通但与 C++ 对照存在**数值级失配**（已立债）；CRASH = panic（原文登记）；
>   DEV = 有意分歧/exit(3) 文档化；NOREF = 无 MFEM 对照可能；RUN-LONG = 正常推进但 >600 s 未完。
> - 日志：`tmp/ledger/logs/<name>.log`（同名覆盖时为最后一句真话，正文注明档位）。

## 统计（86 = 45 串行 + 40 并行 + 1 self-test）

**round-84 权威值（C 路盘点 + 抽样重验，HEAD c5a06dd6；下表为本表唯一现行口径）**：

| 分类 | 串行 | 并行 | self-test | 合计 |
|---|---|---|---|---|
| BIT | 11（ex1/ex2/ex3/ex9/ex14/ex24/ex27/ex31 + **r98: ex29/ex20/ex5**） | 0 | 0 | **11**（注：ex17[r89]/ex33 两档[r91]/ex10[r92]/ex6[r93] 的升 BIT 记录在 coverage_matrix §5 各轮行，本表行未逐一回写——历史欠账，r98 起 **BIT 计数以 coverage_matrix 为准**） |
| RUN（含 RUN-LONG 2：pex15 替代档/pex30 单列） | 36 | 38 | 0 | **74** |
| RUN\*（数值失配，已立债） | 0 | 0 | 0 | **0** |
| CRASH | 0 | 0 | 0 | **0** |
| DEV | 0 | 2（pex19/pex32） | 0 | **2** |
| NOREF | 1（ex4_darcy_simple 未注册） | 0 | 1（par_heat self-test） | **2** |
| **合计** | **45** | **40** | **1** | **86** |

**round-61 初账快照（历史，仅存档）**：BIT 4 / RUN 63 / RUN\* 3（ex4/ex5/ex20）/ CRASH 12 / DEV 2 / NOREF 2。
round-62–83 的迁移路径见下文各增量节与「round 84 增量」节的权威三态对照表。

**round-127 增量（MFEM lane：RUN 档抽样升 BIT 第五波，HEAD 1a0d88a4）**：
串行 BIT 15（r126 快照口径）→ **17**（+ex16 +ex21，两行本表体已更新）。
C++ 真值一律现编现跑（`$HOME/mfem410_ser`，g++ -O2，run 目录 `$HOME/work/rr127/`，
两侧同 CWD 形式 `-m data/<mesh>`）；Rust 侧 debug（**未用 release**），CWD =
`tmp/ledger/rundir`。证据：`fem-rs/tmp/rr127mfem/`（双 stdout 快照 + diff + 命令行）。

**round-129 增量（MFEM lane：RUN 档抽样升 BIT 第六波，HEAD fd35de7b）**：
串行 BIT 17 → **18**（+ex23，本表体已更新；ex10 未启动——C 盘 5.5G < 8G 停手纪律，
见 round-129 节）。C++ 真值现编现跑（`$HOME/mfem410_ser`，g++ -std=c++17 -O2，
run 目录 `$HOME/work/rr129/`，两侧同 CWD 形式 `-m data/star.mesh`）；Rust 侧 debug
（**未用 release**），CWD = `tmp/ledger/rundir`。证据：`fem-rs/tmp/rr129mfem/`
（双 stdout 快照 + cmp + 探针链）。

## 串行（45）

| 名字 | MFEM 对应 | 档位 | 分类 | 证据/日志 | 备注 |
|---|---|---|---|---|---|
| mfem_ex0_mesh_intro | ex0 | 默认档 | RUN | logs/mfem_ex0_mesh_intro.log | ARF 0.140201 收敛正常 |
| mfem_ex1_poisson | ex1 | `-m data/star.mesh -no-vis` | **BIT** | logs/… + ref/ex1.out | **本轮现对拍**：除 Rust 缺 10 行 `Options used:` 头外逐字节（111 迭代 + ARF 0.882852 全同）。**round-84:** 现档 BIT（r66 D687 补齐 Options 头后 cmp 全等）；重验 = 仅 mesh 路径行差 ✓ |
| mfem_ex2_elasticity | ex2 | `-m data/beam-tri.mesh -no-vis` | **BIT** | 同上 + ref/ex2.out | **round 63 D647 全流逐字节**：C++ 4.10 ex2 本就不打 `Wrote…`（旧豁免注销），Rust 删该 stderr 行后 stdout 278 行/11783 字节 + stderr 空两侧全同（tmp/d597/ex2run/）；ARF 0.965229/268 迭代全同。**round-84:** 重验 = 仅 mesh 路径行差 ✓ |
| mfem_ex3_maxwell_cavity | ex3 | 默认 beam-tet -o1 | **BIT**（r65 D664 双档） | logs/… + ref/ex3.out | 终值 3.91630923150637e-1 = C++ 0.391631（6 位）；PCG 历史口径不同（Rust 打归一化残差、119 迭代 vs C++ 137/ARF 0.903118）→ D634。**round-84:** 现档 BIT（r64 D651/D653 + r65 D664，PCG 轨迹 137 行逐字节）；重验默认档 vs ref/ex3.out = 仅 mesh 路径回声行（built-in vs 文件） |
| mfem_ex4_darcy | ex4 | `-m data/star.mesh -no-vis` | RUN（r62 D634） | logs/… + ref/ex4.out | **失配**：‖F−F_h‖=0.432497 vs C++ 0.0161443（27×）；Rust 287 迭代即"收敛" vs C++ 646 → D634。**round-84:** 失配已闭（r62 停机规则 + r63 D639 评估器，646 it = C++）；重验：终误差行 0.0161443 逐字同、iter0-588 逐字节，589+ 为 1e-17 噪声级分叉（r62 已记载口径），ARF 第 6 位差 |
| mfem_ex4_darcy_simple | —（无对应） | — | NOREF | — | 未注册死文件、无 exe（round 30 D133 在案）；自述 SIMPLIFIED |
| mfem_ex5_mixed_darcy | ex5 | `-m data/star.mesh -no-vis` | **BIT**（r98，具名豁免 1 行） | logs/… + ref/ex5.out + `tmp/d98runbit/` | **失配**：dim(R/W) 全同（41280/20480），Rust MINRES 423 it ‖r‖/‖b‖=9.24e-7 判收敛但 u_err **1.211582e0** vs C++ 396 it / 1.43587e-4 → D634（块预条件/minres 判据族）。**round-84:** 失配已闭（r63 D639：397 行逐字节、u_err 0.000143587 = C++）；重验：数值行逐字同，豁免 = C++ Options 块 8 行 Rust 不打、Rust 多 `Wrote` 行、wall-clock。**round-98:** 升 **BIT**——414 行 stdout 与 C++ 现编现跑（`$HOME/work/d98runbit/ex5_cpp`，两侧同 `-m data/star.mesh` 路径形式）**逐字节**（396 it 收敛行逐字同），唯一 diff = `MINRES solver took …s.` wall-clock 行（MFEM 自打 RealTime，**具名豁免**）；r84 三类豁免中前两类已不存在 |
| mfem_ex6_flux_recovery | ex6 | 默认 star -o1 -no-vis | RUN | logs/… + ref/ex6.out | 前 4 迭代逐字节（0.441629/0.00864066/2.48721e-06/1.90288e-09）；首个求解 C++@5 停、Rust 拖到 4.6e-40 → D634；AMR 环两端 rc=0（C++ 打 `Reached the maximum number of dofs.`，Rust 打 `Done.`，终态行未逐字节比对）。**round-84:** 重验 rc=0；D634 后首解与 C++ 逐字节 ✓；AMR 标记路径自第 2 环分叉（C++ 76 vs Rust 86 unknowns，r61 已有性质非新漂移；疑与参考命令带 `--no-ls-zz/--max-dofs` 旗标不对齐有关），终态行差维持 |
| mfem_ex7_surface_poisson | ex7 | 默认 | RUN | logs/… | rc=0。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，HEAD c5a06dd6） |
| mfem_ex8_dpg_2x2 | ex8 | `-m data/star.mesh -no-vis` | RUN（数值级 = C++；**r99 D864 后轨迹差 1.3e-5 → 7e-8，仅末位**） | logs/… + ref/ex8.out + `tmp/d98runbit/` | 前 13 迭代逐字节；DPG 范数 0.0181446 vs C++ 0.0183277（~1%）；ARF 0.829413 vs 0.608926 → D634。**round-84:** 数值失配已闭（r63 D640 sinv 四边形分支修复：DPG 0.0183277 与 C++ 逐字同）；重验 ✓，残 = Rust 29 it vs C++ 28 it（停机阈值 1 it 差）、`S^-1` vs `S^{-1}` 打印、Rust 多诊断行。**round-98:** 打印失真**清零**（删多余空行 + `S^{{-1}}`→`S^-1` + 删 Rust 独有 `PCG: iterations=` 摘要行）后 stdout 前 15 行逐字同；**唯一残差 = PCG 轨迹自 iter5 起 1.3e-5 相对分叉**（29 vs 28 it、ARF 0.619477 vs 0.608926）＝核心求解器路径债（D634 族），维持 RUN；真值现编现跑 `$HOME/work/d98runbit/ex8_cpp`。**round-99（D864）：分叉源全部定位**——① `QuadL2GL` 节点用 `gauss_legendre_arbitrary`+映射 ⇒ 比 MFEM `Poly_1D::OpenPoints` **差 1 ulp**（GL 求积点=节点 ⇒ 基函数给 ~1e-17 伪残差而非精确零）；② GL-L2 一维基用直接 Lagrange 乘积 vs MFEM 重心式；③ `SinvBuilder` 手写几何/梯度约定 + 逐列解求逆。三处修后 **F/Bhat/Sinv/S0/Mtest/Ktest 全部逐位 = C++**（`cmp_d99.py`/`cmp_mk.py`），轨迹差降到 iter5 **7.1e-8**（放大率改善 180×），29 行中仅末位翻转；**残余 = B0 混合路径约定（中位 5 ulp、333 条 >100 ulp）+ Shat RAP 求和序（≤5 ulp）+ b（±1e-16）经内层 CG κ≈10³ 放大**——三步配方在 D864 条目。**round-100（D864 第二波）**：① **Shat 逐位达成**——MFEM `RAP(Bhat,Sinv,Bhat)` 匹配的是 `(Rt,A,P)` 重载 = **(Bᵀ·S)·B** 结合序（`CsrMatrix::rap_product` 算的是 Bᵀ·(S·B)），且 MFEM `Mult` 输出行是**首触序**而 Rust 排序（行序传导进下一级累加与 SpMV 求和序）⇒ 新增 `CsrMatrix::multiply_first_touch` + `build_shat` 重接线 = (Bᵀ·S)·B 首触序，**Shat 全表逐位**。② **B0 走 DPG 本地忠实装配器** `dpg::assemble_b0_mfem`（MFEM `DiffusionIntegrator::AssembleElementMatrix2` 标量支逐位镜像：闭式伴随 + `w=ipw/det` + `AddMultABt` 的 k-外层累加；全局 mixed 路径约定切换按㉚仍登记为后续专项）。③ **H1 值路径裁决（1-D 探针四值全中）**：MFEM 的 H1 闭节点基 = **重心式值+导数变体**（`u=l·si·w(i)`），`CalcShape`/`CalcDShape` 同源；曾试 ChangeOfBasis 引擎（Chebyshev+LU）与精确闭式 p=1 特例，均被逐位证据否决后撤销。**现状**：B0 全表 ≤10 ulp（中位 1 ulp）、Shat 逐位、**轨迹分叉点 iter7→iter9**（首叉 8.27615e-08 vs 8.27611e-08 = 5e-13 相对）；**唯一残差定位 = 几何 J 在薄片单元上的 1 ulp**（star 0 号元 det≈8e-4：QP1 的 J[1][1] 差 1 ulp，非求和序/非 FMA/非基函数——入口探针 `$HOME/work/d99/ex8elem_qp.cpp` + `tmp/d98runbit/cmp_qp.py`） |
| mfem_ex9_dg_advection | ex9 | 默认 periodic-hexagon | **BIT**（r77 D812-2，具名豁免） | logs/… + `$HOME/work/d76main/ex9_rerun/` | 默认档 panic（D633 资产缺失）；`-m data/periodic-square.mesh` rc=0；文件头自述 L2 基 ≠ MFEM GLL ⇒ 有意分歧（DEV 性质）双注记。**round-84:** 现档 BIT——r62 资产回填后 rc=0；r77 D812-2 stdout 与 C++ oracle **逐字节**（豁免 = `--mesh` 路径回声 + stderr wall-clock 两处具名）+ r78 D813-3 `ex9-init.gf` 逐字节/final max\|Δ\|=1.0e-08；r82/r83 锚点重验 ✓（旧"L2 基有意分歧"注记作废） |
| mfem_ex10_hyperelastic_dyn | ex10 | 默认 beam-quad | RUN | logs/… + ref/ex10_quad.out | step1..100 的 EE/KE/ΔTE 与 C++ 全部 6 位吻合（0.011958/0.000784/-0.019639）；Newton ‖r‖ 自 iter1 起第 4 位漂移（0.0099624 vs 0.0099476）；打印多诊断行 |
| mfem_ex14_dg_poisson | ex14 | `-m data/star.mesh -no-vis` | **BIT**（r73 D795-1，剥 Options 前缀逐字节） | logs/… + ref/ex14.out | **前 309 行逐字节**；C++@308 收敛（ARF 0.956044），Rust 500 maxiter 不收敛（ARF 0.950077）→ D634。**round-84:** 现档 BIT——r73 D795-1（DG 面规则改等参几何）后迭代史 **311 行逐字节 = C++**（ARF 0.956044 同；豁免 = C++ 12 行 Options 前缀 Rust 不打 = D822-2）；r79/r82/r83 锚点重验 ✓，本轮重验 ✓ |
| mfem_ex15_dynamic_amr | ex15 | 默认 star-hilbert | RUN（r62 D633） | logs/… | 默认档 panic：`data/star-hilbert.mesh` 不存在（D633）；`-m data/star.mesh` 替代档 700 s 内 rc=0（435 s，20 轮 AMR）。**round-84:** 现档 RUN——r62 资产回填后默认档 513 s 全程 rc=0；本轮未重跑（>600 s 预算），维持 r62 定档 |
| mfem_ex15_dump_A_true | —（tools_ex15_ref 配套） | 内置 star-hilbert | RUN（r62 D633） | logs/… | debug dump harness（非用户示例）；panic 于读 `data/star-hilbert.mesh`（D633）。**round-84:** 重验 rc=0（5606 行 PROW dump），D633 修复钉住 |
| mfem_ex15_dump_T002 | 同上 | 同 | RUN（r62 D633） | logs/… | 同上（D633）。**round-84:** 重验 rc=0（2187409 行） |
| mfem_ex15_dump_flow | 同上 | 同 | RUN（r62 D633） | logs/… | 同上（D633）。**round-84:** 重验 rc=0 |
| mfem_ex15_dump_it2_coords | 同上 | 同 | RUN（r62 D633） | logs/… | 同上（D633）。**round-84:** 重验 rc=0 |
| mfem_ex15_dump_p1 | 同上 | 同 | RUN（r62 D633） | logs/… | 同上（D633）。**round-84:** 重验 rc=0，stdout 与 r61-后状态一致（PROW 表） |
| mfem_ex15_dump_p1_it3 | 同上 | 同 | RUN（r62 D633） | logs/… | 同上（D633）。**round-84:** 重验 rc=0 |
| mfem_ex16_nonlinear_heat | ex16 | 默认 star | **BIT**（r127） | logs/… + `tmp/rr127mfem/`（ref/ex16_cpp_default.out + cmp） | rc=0（SDIRK33 时间推进完成）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，chksum 1.072997e6 逐字同）。**round-127:** 升 **BIT**——删 Rust 独有 `=== Comparison Metrics ===` 尾块（C++ stdout 止于 `step 50, t = 0.5`，回归锚点仍留在示例内 `#[cfg(test)]` 钉：dofs 1361/steps 50/norm/chksum 区间）后，`-m data/star.mesh -no-vis` 24 行 stdout 与 C++ 现编现跑（`$HOME/work/rr127/ex16/ex16_rr127`，mfem410_ser）**cmp 逐字节全等**；零代码数值路径改动（时间步进/求解器未动） |
| mfem_ex17_dg_elasticity | ex17 | 默认 beam-tri | RUN（round-84 漂移注记） | logs/… + `tmp/d84c/`（r84 对拍） | rc=0。**round-84:** 数值漂移 = r76 D805-1（DG 弹性罚项改 MFEM 原式）传导：1063→1269 it、‖u_h‖_L2 227.189938→226.758078、checksum −0.15%（旧值系错误罚项产物，非回归）；**新对拍 C++（NOREF-NEW 登记，ref/ex17_cpp_default.out）**：dofs 24576 逐字同、同 PCG+GS+rtol² 配置下 C++ 767 it vs Rust 1269 it → **D822-1**；另 sol.gf 输出表示差（H1 平均位移 vs C++ DG 节点空间）→ 同债 |
| mfem_ex18_euler | ex18 | 默认 periodic-square | **RUN**（round-85 起 C++ 对拍 8 位全对齐 → D822-4） | logs/… + `tmp/d84c/`（二分证据）+ `tmp/d85main/`（D822-4 验收） | rc=0。**round-85（D822-4）**：示例默认 order 对齐 C++（ex18.cpp 默认 3，原 Rust 默认 1）+ `dg_hyperbolic.rs` Quad4 臂放开任意阶（原 `assert_eq!(order,1)`）+ 2-D 面规则改 MFEM 公式（`HyperbolicFormIntegrator` 2p+1 阶 ⇒ p+1 个 GL 点，hyperbolic.cpp:224 + intrules.cpp `SegmentIntegrationRule`；原 `(2p+1).min(4)` 在 p=1 给 3 点/MFEM 2 点）。**默认档（= C++ 默认，order 3）：435 步 = C++，`Solution error: 3.930926246114457e-3` = C++ `0.0039309262` 全部 8 位打印数字逐位一致**；`-o 1` 档：184 步 = C++（原 185），`6.168658610565338e-2` = C++ `0.061686586` 8 位全对齐（原 6.168620814596272e-2 仅 6 位——**旧残差真根因 = 面规则点数差，"fp 排序"归因证伪**）。C++ 真值现编现跑 `$HOME/work/d85main/ex18/`（mfem410_ser 源码），快照 `tmp/d85main/cpp_{default,o1}.out`。历史（round-84 D822-3 回归）：默认档曾 step 27 起 NaN，二分钉 `24e00a8d`，修复后 o1 档终值 6.168620814596272e-2（该值含面规则偏差，已被上值取代） |
| mfem_ex19_hyperelastic_incomp | ex19 | 默认 beam-quad | RUN | logs/… | rc=0（Newton+块 GMRES 收敛）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，Newton 3 it 收敛行逐字同） |
| mfem_ex20_symplectic | ex20 | 默认 -o1 -t 100 步 | **BIT**（r98 双档） | logs/… + ref/ex20.out + `tmp/d98runbit/` | **失配**：能均值/方差 = `1 / 0` vs C++ `1.00204 / 0.0174915`（能量恒 1 ⇒ 积分器未真正演化）→ D635。**round-84:** 失配已闭（r62 D635b 六配置逐字节 = C++）；重验 `-o 1 -t 100 -no-vis`：数值行逐字同，豁免 = C++ Options 回声多 2 行（`--no-visualization/--no-gnuplot`，→ D822-2）。**round-98:** 升 **BIT**——`-no-vis` 档 12 行 stdout 逐字节 = C++（回声块早已补齐，r84 豁免过期）；无参默认档两处 1:1 保真修复后（`visualization` 默认 false→true = C++ ex20.cpp:98；删 Rust 独有 `Wrote ex20_phase…` 行——C++ 走 socketstream 无打印）亦逐字节；vis 副产物文件改为落 rundir |
| mfem_ex21_amr_elasticity | ex21 | 默认 beam-tri | **BIT**（r127） | logs/… + `tmp/rr127mfem/`（ref/ex21_cpp_beamtri.out + ex21_diff_v4.txt=0） | rc=0；历史有机器本地 golden 注记（非 C++ 逐字节）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff）。**round-127:** 升 **BIT**——四处 1:1 保真修复后，`-m data/beam-tri.mesh -no-vis` 132 行 stdout 与 C++ 现编现跑（`$HOME/work/rr127/ex21/ex21_rr127`）**cmp 逐字节全等**（21 个 AMR 周期的 PCG 轨迹+ARF 全同；DOF 序列 36→…→14950 两侧一致）：① Options 块补 3 行（`--no-static-condensation`/`--flux-averaging 0`/`--no-visualization`，PrintOptions 字节对齐）；② PCG 换 legacy 口径 = `rtol 1e-6`（旧 1e-12 实为 1e-24·nom0 判据、每周期打满 2000 it 上限，同 ex14 D795-1 病灶）+ `PrintLevel::FirstAndLast`（MFEM legacy level 3：it0 行带 " ..." + 末行 + ARF）；③ 删 Rust 独有诊断行（`Max err`/`Marked N`/`Wrote …`/`Direct solve`/`SC:`）+ 停机文案对齐 C++（`Reached the maximum number of dofs. Stop.` / `Stopping criterion satisfied. Stop.`）；④ **补上 MFEM 的解延长链路**（旧注记"prolongation 死存储已删"系误删）：C++ `x.Update()` 把上一轮解 prolong 到新网格作 PCG 初值（实证 = 现编探针：PCG 打印的 it0 `(B r,r)` = `(GS(B−A·X), B−A·X)` ≠ `(GS·B, B)`，且 legacy `PCG()` 的 X 不会被清零——initial-guess 模式），P1 延长 = 顶点值复制 + 新中点 `0.5·u_a+0.5·u_b`（MFEM RefinementOperator 行点积口径；中点父边按坐标位模式精确识别，`refine_2d::new_midpoint` 同式）。3-D/Quad 臂维持零初值（无对照档，行为同 r126 前）。C++ 探针（P5/P6）留档 `$HOME/work/rr127/` |
| mfem_ex22_complex_helmholtz | ex22 | 默认 inline-quad | RUN（r67 D693-695 收口） | logs/… | rc=0（复数系统求解完成）。**round-84:** r67 -p0/-p1 双双逐位收官 + r69 D738 评估器求积阶修复 + r71 D748 评估器族配对化（余项 D704/D748/D765/D767）；重验默认档：误差行 1.422826e-1/1.422741e-1 与 r61 后状态逐字同，GMRES 打印已换 MFEM Pass/Iteration 族 |
| mfem_ex23_wave_equation | ex23 | `-m data/star.mesh -no-vis` | **BIT**（r129） | logs/… + `tmp/rr129mfem/`（ref/ex23_cpp_star.out + cmp） | rc=0；历史 golden 本地注记。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，checksum 1.889982e4 / du/dt 2.955466e4 逐字同）。**round-129:** 升 **BIT**——十处 1:1 保真修复后 `-m data/star.mesh -no-vis` 24 行 stdout 与 C++ 现编现跑（`$HOME/work/rr129/ex23/ex23_cpp`，mfem410_ser）**cmp 逐字节全等（0 豁免）**；`ex23-init.gf` 亦**逐字节全等**（新证：投影初值含 ess dof 的 e^-30 级值——C++ 不清零，旧 Rust 违例清零已删）；`ex23-final.gf` 2721/2722 值逐字同，唯一残值 du/dt dof274 末打印位翻转 = 平台 exp() ulp 差（Windows CRT vs glibc，5/1361 初值 1 ulp、exp 参数位级 0 差——探针链闭环，立债）。关键修复：时间循环末步 dt 调整违例（C++ `Step` 永不改 dt）、`solve_pcg_jacobi`→`solve_pcg_dsmoother`（CGSolver+DSmoother DiagScale 逐位移植入口）、**MFEM 4.10 `Vector::Norml2` = dnrm2 风格缩放算法**（vector.cpp:968，非 `sqrt(Σx²)`——plain 版 1/3 dof 差 1-2 ulp，示例内 `mfem_norml2_2d` 逐位镜像）、Options 补 `--ref ` 行/visit 默认 true/GLVis 文案/.gf 双头结构。轨迹锚点入 `#[cfg(test)]`（dofs 1361/steps 50/checksum 区间） |
| mfem_ex24_discrete_ops | ex24 | `-m data/star.mesh -p 0 -o 1 -no-vis` | **BIT**（r70 D712 四口径） | logs/… + ref/ex24_p0o1.out | **本轮现对拍**：数值行逐字节；豁免 = Rust 的 Options 块少 3 行（--no-static-condensation/--no-partial-assembly/--device）。**round-84:** 现档 BIT——r66 D686 三口径 + r70 D712 四口径全部逐字节 = C++（D721 翻转后 -p1 iter1 1.47776e-22 归零）；重验 -p0 -o1 = 仅 mesh 路径行差 ✓ |
| mfem_ex25_pml_maxwell | ex25 | 默认 beam-hex | RUN | logs/… | rc=0（复 PML 系统完成）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff） |
| mfem_ex26_geom_mg | ex26 | 默认 star；hex 档 `-m inline-hex -gr 0 -or 2` | RUN | logs/mfem_ex26_geom_mg.log、logs/mfem_ex26_hex.log + ref/ex26_hex.out | 默认档 rc=0。hex 档本轮**实况对拍**：274625 未知数同、6 迭代轨迹机器精度级吻合（ARF 0.0435272 vs 0.0435273、iter1 1.55112e-6 vs 1.55209e-6），**非逐字节**（Chebyshev smoother 特征值估计 ulp 级分叉）。round 60 存档对 tmp/d31/ex26_{rs,cpp}.log 数值逐字节，但其原命令不可复原 ⇒ 存档对不作为本轮逐字节证据引用。**round-84:** 重验 OK（默认档 stdout 与 r61 记录 **0 diff**；r72 D754 并行装配逐位化后 5 连跑 sha256 全同口径维持） |
| mfem_ex27_robin_bc | ex27 | 默认 `-no-vis` | **BIT\***（round 72 D771） | logs/… + `tmp/d771/`（gold/diff/probe） | **canonical stdout 对齐**：Options 块 12 行 + 去前导空行 + 删非 C++ 的 `Solved in N iterations.` + 平均值行 `", \t"` 与 `%g6`（`fem_solver::fmt_g`）⇒ **default 51 行中 49 行逐字节、`-dbc 2.5` 52 行中 50 行逐字节**（残 2 行/档 = 半面求积 + 网格表示差 **D778**，非格式）；迭代历史 29/30 行零差异保持；**`-dg` 迭代历史逐字节不回退**（其 C++ 差距 = **D779**）。**round-84:** 现档全逐字节——r73 D778（半面求积修为 [0,1] 恒等语义）后 **default 51/51 + `-dbc 2.5` 52/52 逐字节**；r76 D805-3 refined.mesh TOPOLOGY-IDENTICAL 10/10；r82/r83 锚点重验 IDENTICAL ✓ |
| mfem_ex28_sliding_elasticity | ex28 | 默认 | RUN | logs/… | rc=0。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，Pass 15 707 it 轨迹逐字同） |
| mfem_ex29_curved_poisson | ex29 | 默认（-mt 4 -mo 3） | **BIT**（r98 双档） | logs/… + ref/ex29.out + `tmp/d98runbit/` | 迭代 0–7 逐字节；C++@7 停（ARF 0.0969461）Rust@11（ARF 0.0719572）；终误差 ‖u−u_h‖ 0.00138643 / ‖f−f_h‖ 0.00797749 **逐字节同**；Rust 多 2 行头注 → D634。**round-84:** 重验 = 与 C++ **全部数值行逐字节**（含 ARF/停机点，D634/D758b 停机同步达成）；豁免仅 Rust 多 2 行头注（Geometry order/Mesh nodes 诊断行）。**round-98:** 升 **BIT**——`-no-vis` 档 20 行 stdout **逐字节**（r84 记载的「2 行头注豁免」已不存在）；无参默认档两处 1:1 保真修复后（`visualization` 默认 false→true = C++ ex29.cpp:72；bool 对回声补 true 分支 `--visualization`）亦**逐字节**；真值现编现跑 `$HOME/work/d98runbit/ex29_cpp` |
| mfem_ex30_aniso_amr | ex30 | 默认 star | RUN | logs/… | rc=0（三系数预处理完成）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，12572 元/Osc error 0.000619） |
| mfem_ex31_anisotropic_maxwell | ex31 | `-m data/inline-quad.mesh -r 2 -o 1 -no-vis` | **BIT** | logs/… + ref/ex31.out | **本轮现对拍**：除 Options 里 mesh 路径串外**全 stdout 逐字节**（含 0.181455 / ARF 0.829075）；无参默认档会 exit(3)（-vis 未移植，文档化）。**round-84:** 重验 = 仅 mesh 路径行差 ✓；**round-104：1-D 档解锁（D901 关闭）**——`-m data/inline-segment.mesh -r 2 -o 1/2/3 -no-vis` 三档 stdout **逐字节 = C++**（ND_R1D 空间 + 制造源项 f_exact + GS+PCG；RHS 必须用 f_exact 而非 E_exact——C++ f 是 −ΔE−κ²E+ΣE 制造源），证据 `tmp/d104/`；旧 r69 D733 缺口条目作废 |
| mfem_ex31_dump | —（tools/ex31_cpp_helper 配套） | `-m data/inline-quad.mesh -r 2 -o 1` | RUN | logs/… | debug dump harness；rc=0，落盘 rust_*.txt 供 harness 比对（D375 管线） |
| mfem_ex33_fractional_diffusion | ex33 | 默认 star | RUN | logs/… | rc=0；AAA 模块与 spde 共享（那边有逐位锚点），本示例无逐字节记录。**round-84:** 重验 OK——与 r61 记录仅 1 处末位显示差（iter120 `6.40485e-27`→`6.40484e-27`，噪声级），ARF 0.795112 其余逐字同 |
| mfem_ex34_magnetostatics | ex34 | 默认 fichera-mixed | RUN | logs/… | rc=0（SubMesh 电流密度链路完成） |
| mfem_ex36_obstacle | ex36 | 默认 disc_p2 | RUN | logs/… | rc=0（proximal Galerkin Newton 完成）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，L2/H1 误差行逐字同） |
| mfem_ex37_topology_optimization | ex37 | 默认 | RUN | logs/… | rc=0。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，Final step 33 / compliance 0.003875） |
| mfem_ex38_implicit_integration | ex38 | 默认 surface2d | RUN | logs/… | rc=0（moment-fitting 求积完成）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff，Surface 误差 5.4492670572e-5 逐字同） |
| mfem_ex39_compass | ex39 | 默认 compass.msh | RUN | logs/… | rc=0（命名属性集链路完成） |
| mfem_ex40_eikonal | ex40 | 默认 star | RUN | logs/… | rc=0（阻尼拟牛顿完成）。**round-84:** 重验 rc=0；终值 `0.026921483748254076` = r65 D674 真值反转后的 C++ 对齐锚（C++ 0.0269214）✓；Newton 轨迹行 vs r61 记录的差异属 D674 修正预期传导；页脚 Outer 5/Total 6/15520 不变 |
| mfem_ex41_imex | ex41 | 默认 periodic-square -s64 | RUN | logs/… | rc=0（IMEX DIRK3 1000 步完成）。**round-84:** 重验 OK（stdout 与 r61 记录 0 diff）——r82 D817-1 触及 dg_imex bdr K 源、r76 D805-2 加 alpha 字段，默认档均无扰动 |

（注：串行 45 行齐全——含 ex4_darcy_simple（NOREF）与 6 个 ex15_dump（CRASH）；
86 = 45 串行 + 40 并行 + 1 self-test，与盘点清单 `tmp/ledger/ex86_list.txt`
（85 可跑 + 1 未注册死文件）一致。）

## 并行（40）

| 名字 | MFEM 对应 | 档位 | 分类 | 证据/日志 | 备注 |
|---|---|---|---|---|---|
| mfem_pex0_parallel_poisson | ex0p | 默认 --ranks | RUN | logs/… | rc=0 |
| mfem_pex1_parallel_poisson | ex1p | 默认（81920 quads） | RUN | logs/… | rc=0（240 s 内；round 30 曾 150 s 超时） |
| mfem_pex2_parallel_elasticity | ex2p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex3_maxwell_cavity | ex3p | 默认 star（2-D 折叠档） | RUN（数值级 = C++ 6 位） | logs/… | rc=0；历史对照：dofs 10400、‖E−E_h‖=2.70053055689122e-2 vs C++ mpirun 0.0270053（6 位；迭代数 102 vs AMS 17 属求解器栈差异）。**round-84:** r65 漂移（D683）→ r66 仲裁（D697 回归确认）→ r67 D697/D705 翻正：default 87/102 it、2.70053048394657e-2/2.70053055702279e-2 = C++ 六位；r71 L² 锚复活（-o1/-o2 逐位稳定）；r78 四锚点逐位（87/1/95/193 it）；r70 D746 并行确定性根治。维持 RUN（无全 stdout 逐字节主张） |
| mfem_pex4_parallel_hdiv_diffusion | ex4p | 默认 star | RUN | logs/… | rc=0；文件头注明 PCG+Jacobi 替代 AMS（求解器栈差异） |
| mfem_pex5_hdiv_darcy | ex5p | `-m data/star.mesh -no-vis` | RUN（r62 D635a） | logs/… | panic：`crates/parallel/src/launcher/native.rs:154` worker `index out of bounds: len 2 index 2`（round 30 曾 rc=0 ⇒ **回归嫌疑**）→ D635a。**round-84:** 越界已修（r62 D635a 二分闭环）；AMG 配置 = r64 D654 结构性豁免（示例文件有注记）；r70 快查 MINRES 95 it/2.91889e-5 ✓。本轮未重跑，维持 |
| mfem_pex6_parallel_amr | ex6p | 默认 star | RUN | logs/… | rc=0（AMR 环完成） |
| mfem_pex7_parallel_surface | ex7p | 默认 octahedron | RUN | logs/… | rc=0 |
| mfem_pex8_parallel_dpg | ex8p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex9_parallel_dg_advection | ex9p | 默认 periodic-hexagon | RUN（r62 D633） | logs/… | panic 于读 `data/periodic-hexagon.mesh`（D633 资产缺失）。**round-84:** 现档 RUN——r62 资产回填后 rc=0；r76/r78 mass 锚点 3.638348e2、stdout 与改动前逐字节（路径归一化后）。本轮未重跑，维持 |
| mfem_pex10_parallel_hyperelastic | ex10p | 默认 beam-quad | RUN | logs/… | rc=0 |
| mfem_pex11_parallel_eigenvalue | ex11p | 默认 star | RUN | logs/… | rc=0（LOBPCG） |
| mfem_pex12_parallel_elastic_eigen | ex12p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex13_parallel_eigenvalue | ex13p | 默认 beam-tet | RUN | logs/… | rc=0 |
| mfem_pex14_parallel_dg_poisson | ex14p | 默认 star | RUN | logs/… | rc=0 |
| mfem_pex15_parallel_dynamic_amr | ex15p | 默认 star-hilbert | RUN（r62；替代档 RUN-LONG） | logs/… | 默认档 panic（D633）；`-m data/star.mesh` 替代档 **800 s 超时**：日志推进正常（AMR 迭代 2 → 59352 unknowns，带 load rebalance）⇒ 长跑非挂死；对照：串行 ex15 同档 435 s 完成。**round-84:** r62 资产回填 + D635a 修复后默认档 rc=0（CRASH 12→0 批次内）；本轮未重跑（长跑预算），维持 r62 定档 |
| mfem_pex16_parallel_nonlinear_heat | ex16p | 默认 star | RUN | logs/… | rc=0。**round-84:** 重验 OK——默认档（ranks 1）stdout 与 r61 记录 **0 diff**；附带 `--ranks 2` 探针：chksum 1.620186e7→1.618583e7（−9.9e-4 相对，r71-r73 并行分区/确定性改动族的未钉档漂移，该档无既有主张，仅观察记录） |
| mfem_pex17_parallel_dg_elasticity | ex17p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex18_parallel_euler | ex18p | 默认 periodic-square | RUN | logs/… | rc=0 |
| mfem_pex19_parallel_incomp_hyperelastic | ex19p | 默认 beam-tet | DEV | logs/… | exit(3)+文案：本 port 仅 2-D 分支（C++ 默认网格 3-D）——round 31 D137 文档化裁剪 |
| mfem_pex20_parallel_symplectic | ex20p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex21_parallel_amr_elasticity | ex21p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex22_parallel_complex_helmholtz | ex22p | 默认 inline-quad | RUN | logs/… | rc=0 |
| mfem_pex24_parallel_discrete_ops | ex24p | 默认 beam-hex | RUN | logs/… | rc=0 |
| mfem_pex25_pml_maxwell | ex25p | 默认 beam-hex | RUN | logs/… | rc=0 |
| mfem_pex26_parallel_geom_mg | ex26p | 默认 star | RUN | logs/… | rc=0 |
| mfem_pex27_parallel_robin_bc | ex27p | 默认 | RUN | logs/… | rc=0（round 30 的内核 assert panic 已于 D138 修复，本轮绿）。**round-84:** 重验 OK——默认档（ranks 1）stdout 与 r61 记录 **0 diff**（4 个平均值行逐字同）；D778 半面求积同款问题在 r72 已注记（pex27 侧配方在案未动）；附带 `--ranks 2` 探针：PCG 26→38 it（平均值行不变，同上观察记录） |
| mfem_pex28_parallel_sliding_elasticity | ex28p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex29_surface_poisson | ex29p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex30_amr_preprocess | ex30p | `-m data/star.mesh -no-vis` | RUN-LONG | logs/… | 240 s/600 s/**900 s** 三档均超时（rc=124）；日志停在 3341 元素 / Osc error 7.752668e-4，stdout 无时间戳 ⇒ 推进缓慢与停滞不可辨（round 30 同样 150 s 超时）——需专项 |
| mfem_pex31_restricted_hcurl | ex31p | 默认 inline-quad | RUN | logs/… | rc=0（round 30 的 QuadNDk 越界 D137b 已修，本轮绿） |
| mfem_pex32_maxwell_eigenvalue | ex32p | 默认 inline-quad | DEV | logs/… | exit(3)+文案：仅 3-D 路径（C++ 默认网格 2-D，反向维度地雷）——round 31 D137 文档化 |
| mfem_pex33_fractional_laplacian | ex33p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex34_magnetostatics | ex34p | 默认 fichera-mixed | RUN | logs/… | rc=0 |
| mfem_pex35_complex_oscillator | ex35p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex36_obstacle | ex36p | 默认 disc | RUN | logs/… | rc=0 |
| mfem_pex37_topology_optimization | ex37p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex39_named_attributes | ex39p | 默认 compass.msh | RUN | logs/… | rc=0 |
| mfem_pex40_eikonal | ex40p | `-m data/star.mesh -no-vis` | RUN（r62 D635a） | logs/… | panic：`crates/parallel/src/dof_partition.rs:1251 index out of bounds: len 2 index 2` → D635a。**round-84:** 越界已修（r62）；r64 D655 页脚本地真 dof 和（np=1 15520 与 C++ 逐字节）；r65 D674 终值换核心 0.0269215 ✓。本轮未重跑，维持 |
| mfem_pex41_imex | ex41p | 默认 periodic-square | RUN | logs/… | rc=0 |

## self-test（1）

| 名字 | 对应 | 档位 | 分类 | 证据 | 备注 |
|---|---|---|---|---|---|
| par_heat_equation_self_test | —（无 MFEM 对应，文件头自述） | 默认 | NOREF | logs/… | 并行框架自测（分区不变性/dt 精度单测载体），rc=0 |

## 本轮新债（D633–D635 号段）

- **D633（P1）`data/` 默认网格资产缺失**：e1f16f8 "cleanup" 删掉了仍被引用的网格。
  `star-hilbert.mesh`（ex15_dynamic_amr、pex15、6×ex15_dump_*）、`periodic-hexagon.mesh`
  （ex9、pex9）、`ref-cube.mesh`（diag_mg_abs_l1_jacobi 默认）、`triple-pt-1.mesh`/
  `fichera-q2.mesh`（mesh_bounding_boxes 头注示例命令）——受影响默认档一律 panic。
  真值树 `$HOME/mfem410_ser/data/` 都有，恢复即修；验收 = 上述默认档 rc=0。
- **D634（P1）迭代求解器停机规则/口径与 MFEM 不一致（族）**：
  - ex14：前 309 行逐字节后 C++@308 收敛、Rust 500 maxiter 不收敛；
  - ex4：Rust 287 "收敛" vs C++ 646，终误差 27×（0.432497 vs 0.0161443）；
  - ex5：Rust MINRES 423 it 判收敛但 u_err O(1)（C++ 396 it → 1.4e-4）；
  - ex6/ex29：C++ 停得早、Rust 过收敛（ARF 行必然不同）；
  - ex8：DPG 范数 ~1% 差；ex3：残差打印口径（归一化 vs 原始）+ 迭代数不同。
  根因假设（未逐一证实）：`SolverConfig`/`solve_pcg_gssmoother` 等 fem_solver 入口的
  rtol 作用量（(B r,r)² vs sqrt）与 max_iter 默认 ≠ MFEM legacy `PCG()`/MINRES 语义。
  验收 = ex14/ex4/ex29 与 C++ 停在同一迭代、ARF 行逐字节。
- **D635（P1）两项独立失配/回归**：
  - **(a) 并行分区越界回归**：pex5（launcher worker panic）、pex40（`dof_partition.rs:1251`
    index len 2 index 2）——round 30 两者均 rc=0，现默认档必挂；最小复现 = 默认档直接跑。
  - **(b) ex20 symplectic 数值失配**：order-1 oscillator 能量均值/方差 = 1/0（恒等），
    C++ = 1.00204/0.0174915 ⇒ Rust 的 SIAV 未真正演化 q,p（或能量取了初值）。

## 历史 BIT 锚点引用（本轮未重跑、维持原记录）

- ex1：round 32 主会话"逐字节 1 件"（plan L1266）；本轮已复证 ✓
- ex24 `-o 1/2/3`：round 47 D343/D344 三行逐字节（plan L2164/L2205）；本轮 -p0 -o1 复证 ✓
- ex29 默认：round 47 D343"默认档逐字节"——**本轮复证发现 ARF 行不再逐字节**（D634 停机规则，
  终误差行仍逐字节）⇒ 历史"逐字节"结论应收窄为"数值行逐字节"。
- ex31：round 5x stdout 逐字节（plan L5039/L5044）；本轮 -r 2 -o 1 复证 ✓
- ex26：round 60 D31"端到端逐行一致"（覆盖矩阵 §5，存档 tmp/d31/ex26_{rs,cpp}.log 数值逐字节）。
  **本轮复证未闭合**：存档对的原命令不可复原；本轮自建同档（274625 dof）实况对拍为
  **机器精度级吻合而非逐字节**（iter1 起 4 位有效一致、ARF 末位差 1）⇒ ex26 由历史 BIT 降级为
  RUN（数值吻合口径），历史"逐行一致"结论应收窄为"数值逐字节、打印行有 4 行豁免"。

## round 64 增量（主会话收尾注记）

- **ex3**（上行第 35 行的"119 迭代/归一化残差"记录为 round-62 前旧态，已被 round-62 D634
  收口与 round-64 D651 双双超越）：**3-D beam-tet 档 PCG 轨迹 137 行 (B r,r) 与 C++ 4.10
  逐字节全同**（D651 修 mesh 细化顶点编号后）、ARF 0.903118 两侧同、E 0.391631 一致；
  **2-D star.mesh 档误差行 `1.34917895677130e-2` = C++ 0.0134918**（D653 换核心
  hdiv_error）。距 BIT 仅剩 3 项纯打印格式差（缺 `Size of linear system:` 行、多
  `PCG+GSSmoother` 摘要行、E_h 用 `{:.14e}` 而非 cout 6 位）= **D664**（examples 域）。
- **pex5**（ex5p）：维持 RUN；D654 = **豁免（结构性）**——C++ 侧 `HypreBoomerAMG(*S)` 零参数
  覆盖走经典 HMIS+ext-i+aggressive 族（hypre.cpp SetDefaultOptions 钉死），fem-rs
  `crates/parallel/src/par_amg.rs` 为聚合式 AMG 且无粗化/插值族旋钮；示例层 5 组对照实验
  （E0/E1/E2/E4/E6，tmp/d654/）证明同档配置在聚合族上全部更差（95 即最优）。豁免注释已落
  示例文件。
- **pex40**（ex40p）：D655 关闭——页脚 "Total dofs" 从全局口径改 rank 本地真 dof 和
  （= C++ `RTfes.GetTrueVSize()+L2fes.GetTrueVSize()` 语义）；np=1 `15520` 与 C++ 逐字节；
  np=2 语义对齐（7956 vs 7796 = 划分器本地分布差，预期）；Newton 轨迹零漂移。
  顺带清 6 条预存警告（fem-examples 本例零警告）。
- **D656 审计**（86 例全扫，只读）：新债 **D672**（`examples/src/maxwell.rs` 共享
  `l2_error_hcurl_exact` ND1 硬编码 → ex22/pex3 `-o 2` 误差行静默错，实测 -o2 误差大 14×
  且加密反升）、**D673**（ex22 死函数 + 文件级 `#![allow(...)]` 违反死代码零容忍）、
  **D674**（ex40/pex40/ex18 手写 quad-only L2 范数族，部分 HYPOTHESIS）。详表见 round-64
  plan 节。

## round 65 增量（主会话收尾注记）

- **ex3 升 BIT（双档）**：D664 修掉全部 4 处打印格式差（缺 `Size of linear system:` 行、
  多 `PCG+GSSmoother` 摘要行、`{:.14e}` → `fmt_g`、`unknowns` 行前导空行）后，3-D beam-tet
  与 2-D star.mesh stdout 对 C++ 4.10 **diff 仅剩 mesh 路径行**（主会话复跑实证 2 行 diff）；
  PCG 轨迹 137/386 行一字不动。上轮"D664 3 项格式差"记录被第 4 处（空行）补全。
- **ex40 终值真值反转（D674）**：round-64 红线值 `0.02687451685661206` 是**错的**（手写
  QuadQk 范数）——换核心 `compute_l2_error_owned` + 零精确解后 `0.026921483748254076` =
  C++ 4.10 `0.0269214`（前 6 个轨迹行逐行对齐 ~1e-6）。pex40 np=1 页脚 15520 与
  Outer 5/Total 6 不变；ex18 默认档逐字节不变。
- **ex22**：D672 评估器阶感知化（`-o 1` 两档逐字节不变；`-p 1 -o 2` 误差 2.009523e-1 →
  `2.799165e-2`，C++ 5.62297e-3）；**完全对齐挡在 D681（`-p 0` H1 Q2 路径 `-o 2` 误差 90×
  且加密反升——round-64 red 档的真病灶在 H1 Q2 组装/评估）与 D682（`-p 1` GMRES 停滞
  1000 it 残差 1.5e-1 vs C++ BDP+GS 116 it 1e-12）**。ex22 警告 13→0、死函数与文件级
  allow 清除（D673）。
- **pex3**：默认档现值 `1.62385083181174e0`/2085 it 与本台账 round-61 记录
  `2.70053055689122e-2`/102 it **严重漂移**（纯 HEAD 复现两次）→ **D683** 待 MPI 真值
  仲裁（台账过期或回撤回归，二选一）。
- **ex24**：`-p 0` 缺 C++ 混合解 PCG 块（23 it + ARF 0.271788）+ 多 `Wrote` 2 行；数值行
  0.00744893/0.00744895 逐字同 → **D686**（台账旧行"数值行逐字节"失真修正）。
- **ex1**：缺 10 行 `Options used:` 头 → **D687**。
- **警告清扫**：ex0/ex15_dump_p1/ex15_dump_p1_it3/ex15dyn/pex18/ex25/pex26/pex27 八文件
  0 警告（round-65 D675 批次）；余量 96 条/45 文件为既有积压（清单 `tmp/d675/ws_gate2_byfile.txt`）。

## round 66 增量（主会话收尾注记）

- **ex1 升 BIT（D687）**：补 `Options used:` 10 行头 + bool 选项 ENABLE 对规则 + 去前导
  空行后，`-m star.mesh` 与 `-o 2` 两口径 cmp 全等（豁免 = 仅无 `-m` 默认跑的 mesh 路径行）。
- **ex24 三口径逐字节（D686）**：beam-hex -p0 / star -p0 / beam-hex -p2 全 IDENTICAL
  （四处病灶：Options 块、混合解 PCG 换 `solve_pcg_dsmoother` 位级移植、interpolant-norm
  行、多余 Wrote 行）。**口径勘误**：round-65 引用的 23 it/ARF 0.271788 是 star 口径，
  beam-hex = 29 it/0.364936。`-p 1` 豁免 5 行 = **D700**（RT/weak-curl 装配 ~1e-5 相对差，
  HYPOTHESIS，crates 域）。
- **ex22 裁决丰收（D681/D682）**：组装证明正确（单 Q2 quad 4.4e-12）；90× 病灶 = 示例内联
  评估器拿 Q2 角槽位当几何基（核心已交付 `ComplexGridFunction::compute_l2_error`，全管线
  **5.643641e-3 = C++ 逐位**）；GMRES 停滞根因 = 示例 pc 丢 ω + 缺 DIAG_ONE（C++ 全配方
  43/116 it = C++）。示例侧收口 = **D693/D694/D695**（配方与锚点全备）。⚠️ **勘误**：
  round-65 的"ex22 -o1 双 -p 逐字节"对 -p 1 不成立（GMRES 轨迹从未一致）。
- **pex3 仲裁（D683/D697）**：**双重成立**——台账行过期 且 HEAD 真回归。C++ np1/np2 今日
  实跑 0.0270053（17/19 it）= round-30 历史 Rust 值；同 commit fresh log 已是 1.62/2085
  （行文与日志自相矛盾）；HEAD ranks1/2 = 271/2085 it、1.62385083285577e0/1.62385083181174e0
  （rank 无关 = 组装/BC 层）。头号嫌疑 `25c4c99`（D124 HCurl essential）。pex3 行应改 RUN*。
- **警告专项**：96 → **5**（40 文件清零；余 5 条在 pex3[3]/ex40[2] = **D701**）。
- **死 API**：`maxwell.rs::hcurl_error_sq_exact`（176 行）删除（D684）；fem-examples lib 107/107。

## round 67 增量（主会话收尾注记）

- **ex22 -p0/-p1 双双逐位收官（D693/D694/D695）**：`-p 0 -o 2` 误差 **5.643641e-3 = C++
  逐位**；`-p 1 -o 2` **5.622973e-3/6.420253e-3 = C++ 逐位**（pc 补 ω + DIAG_ONE 消元；
  裁决：系统侧 ω 早已在 complex.rs 内乘好，缺的只是 pc 直配）；hex p1 误差行亦逐位。
  迭代数 44/119 vs C++ 42/116（-r1 口径勘误），GMRES 恒多 1-3 it = **D704**（LOW）。
- **ex22 -p0 3-D 残差 = D702**：hex o1 0.2128 vs C++ 0.1459（2-D 逐位、3-D 偏差 → 3-D H1
  装配/BC 投影/几何，crates 域）。
- **pex3 双档翻正（D697/D705 + A 路 -o2 配方）**：default 档 **87/102 it、
  2.70053048394657e-2 / 2.70053055702279e-2** = round-30 历史值恢复、与 C++ 0.0270053
  六位一致（根因 = 25c4c99 起 ess 值被写死 0.0，crates 无罪——D705 主会话落地）；
  `-o 2` 从 10000 it 停 → 404-481 it 收敛（order 门控 AMG 配方；误差行与 22 it 之差 =
  D703 余项[标量 AMG vs HypreAMS 家族]）。pex3 警告 3→0。
- **ex24 -p1 从 5 豁免行 → 2 行（D700）**：三条误差行清零（求积阶翻译错：MFEM RT
  GetOrder=p+1 → 默认阶 5，Rust 误用 3）；余 2 行（iter1/ARF）经 splice 实验证明 =
  1-3 ulp 求和噪声（D712）。**口径新增**：ex24 -p1 beam-hex 全管线 ND 107168/RT 102656 dofs。
- **警告**：ex40 余 2 条清（D701 全关闭）。

## round 69 增量（主会话收尾注记）

- **ex31**：D724 修复后 `-m data/inline-segment.mesh` 读取解锁，但按其声明缺口诚实
  exit(3)（1-D ND_R1D 未移植 = D733）——C++ 金标已留档（`-r 2`：50 unknowns、11 it、
  ARF 0.124887、‖E−E‖ 0.226983）。inline-quad 档 stdout 逐字节保持（0.181455）。
- **ex22**：D719 定位——2-D ND 管线**证明 = MFEM**（单元阵 = D·A_MFEM·D、全局组装逐项同、
  BC 位级同、2×2 稠密解逐系数一致），偏差全在示例评估器 `maxwell.rs::l2_error_hcurl_exact`
  vs MFEM `ComputeL2Error`（+9.5%/−12.9%）→ **D738**（parity oracle `~/work/d719b.cpp` 已备）。
  **D720 关闭**：`-p 2` inline-tet 双侧同样 1000 it 不收敛（C++ 亦打印 No convergence!），
  Im 分量六位逐字同 = parity 达成；可选双侧预条件子升级 = D741。
- **D739**：D706 串行入口消费方迁移积压（ex31_dump:467-468、ex31:427 读投影站点 →
  `eliminate_ess_tdofs`；ex10/26/27/29/39 均质化站点 → `ElimPolicy::DiagOne`）。

## round 70 增量（主会话收尾注记）

- **ex24 四口径全部逐字节 = C++（D712 关闭）**：beam-hex `-p 0/1/2` + star `-p 0` 与
  `tmp/d686/ref_ex24_*.out` 字节相同，含 **`-p 1` iter1 = 1.47776e-22、ARF = 9.13511e-13**
  （翻转前 1.47754e-22 / 9.13443e-13）——D721 翻转让 M/C 条目与 MFEM 位级一致，D712 的
  "参考域约定差" 收尾达成。**D730（solver SpMV/ARF op-order）** 若未来落地可再进一步。
- **D739 迁移（7 站点 → `eliminate_ess_tdofs`/`ElimPolicy`）**：ex31_dump/ex31_aniso 读投影
  站点（DiagKeep）+ ex26/ex29/ex39/ex10 均质化站点（DiagOne）+ ex27。6 站逐字节；
  **ex27 `-dbc≠0` 档旧管线被证为错误实现**（ess 行残留已消元 dof 反应 ⇒ Dirichlet 违反
  40%，平均 3.37 vs C++ 2.5）→ 迁移后 = C++（2.5 / 3.1e-15）；主会话裁决保留（1:1 条款，
  默认档逐字节不变）。残余 = **D753**。
- **D738 关闭（ex22 2-D ND 评估器）**：根因 = 求积**阶**误用（`quad_rule_01(n)` 用 (n+2)/2
  点/维；MFEM intorder = 2·GetOrder+3 ⇒ ND1 = 5 阶/9 点，fem-rs 用 6 阶/16 点）。一行修复后
  2×2 oracle 0.302847805/0.157766105 **8 位吻合**、quad `-p1 -o1` 1.377548e-1/1.407903e-1
  = C++。**D748（新）**：ex22 3-D `-p1 -o2` 27×（l2_error_hcurl_3d ND1 硬编码）、2-D
  `-p2 -o2` 22×（l2_error_hdiv RT0 硬编码）、`-p1 -o3` 7×。
- **D746（新，重要）**：pex3 L² 行**运行间不确定**（~1e-10 相对，5/5 不同值）——任何
  ≥10 位的 pex3 锚点不可复现；round-69 的 `3.05558571562224e-4` 出局，台账口径降档或根治。
- **快查基线（round 70 复核）**：ex26 0.0273569 ✓、ex29 0.00138643 ✓（早停 4 步 → D758b）、
  ex31 inline-quad -r2 = 0.181455 ✓、pex5 MINRES 95 it/2.91889e-5 ✓、pex40 0.0269215 ✓、
  ex34 leg1 与 C++ 逐位（**pex34 ranks2 发散 = D756**）。

## round 71 增量（主会话收尾注记）

- **ex27 数值完全闭合（D753）**：初值改 `X = R·x`（`eliminate_ess_tdofs` 返回值）+ 停机规则对齐
  legacy `PCG()` 包装的 `SetRelTol(sqrt(1e-12)) = 1e-6`（round-32 两套 API 教训）后——
  **default 档 29 行迭代历史与 C++ 零差异**（28 it / ARF 0.607549）、**`-dbc 2.5` 档 30 行零差异**
  （29 it、解平均 2.5 / rel 3e-15）；`-dg` 档逐字节不变。**剩余 = 纯格式**（Options 块 13 行、
  "Solved in N iterations." 多行、平均值 %g6）→ **D771**。
- **ex22 评估器族（D748）**：新增 `fem_assembly::paired_vector_reference_element`（装配器分派的
  公开形式）；四个 3-D/2-D ND/HDiv 评估器改"配对参考元 + order + DOF 数断言"；**tet 面块需
  canonical→element-local 旋转**（候选真发现）。红→绿：3-D hex `-p1 -o2` 27×→1.563985e-2 = C++、
  tet 5.2×→5.189324e-2、2-D `-p2 -o2` 22×→1.595989e-2、3-D hex `-o3` 9.465592e-3。红线
  （-p0 两档、-p1 各档、3-D -p1 -o1）逐字节保持。**D765**：2-D quad NDk k≥3 配对不一致
  （两候选元实测各失败）待裁决；**D767**：HDiv 求解不收敛（tri RT1/tet RT1-2，预处理器侧）。
- **ex27 之外的 D746 连带**：pex3 的 L² 锚复活（`-o1` = 2.70053050699196e-2 逐位稳定、
  `-o2` = 3.05558571614408e-4）——**round-69 的 14 位锚 …562224e-4 是随机编号的一次抽样，
  永久作废**；口径可回到 ~14 位（同命令 5 连跑逐位相同）。

## round 72 增量（C 路：D771 ex27 canonical stdout + D763/D764 纪律）

- **D771 关闭（格式部分）**：`examples/mfem_ex27_robin_bc.rs` 打印面按 MFEM 4.10 实源重塑，
  **数值行一字未动**（迭代历史逐字保持）。五处改动：①新增 `print_options_block`（MFEM
  `OptionsParser::PrintOptions`，optparser.cpp:331：`Options used:` + 每选项一行 `   --<long> <value>`，
  ENABLE 对按当前 bool 选 long_name，数值走 C++ 默认 ostream `%g` = 复用 `fem_solver::fmt_g`）；
  ②`-dg` 时 `--kappa` 打印**替换后**的值（ex27.cpp:137-140 在 PrintOptions 之前）；③去掉
  `println!("\nNumber…")` 前导空行；④删掉 Rust 独有的 `  Solved in N iterations.` 行（**C++ 侧根本
  没有这一行**——简报"Rust 缺 Solved in"的前提是反的）；⑤平均值 4 行改 `", \t"` + `%g6`。
  另按 MFEM 语义把 `-vis/--visualization` 补进参数解析（默认 true，仅喂 Options 块）。
  **验收（cmp，亲自跑）**：default 档 `-no-vis` — 51 行 **49 行逐字节**，`tmp/d771/diff_default.txt`
  只余 2 行；`-dbc 2.5 -no-vis` — 52 行 **50 行逐字节**（`tmp/d771/diff_dbc25.txt`）；`-dg` 迭代
  历史（129 行 = 128 it + ARF）与改造前**逐字节相同**（`tmp/d771/before_dg_iters.txt` vs
  `after_dg_iters.txt`，cmp 全等）。gold = `tmp/d771/cpp_{default,dbc25,dg}.gold`（C++ 现跑，
  binary `$HOME/work/d771/ex27_cpp`；default gold 与 `$HOME/work/d771/ex27_cpp_default.out`
  cmp 全等）+ Rust 侧 `tmp/d771/rust_{default,dbc25,dg}.out`。
- **残 2 行/档 → D778（新债）——两个 ~1e-7 量级的实因，均有实测**：default 的 `Gamma_nbc`
  相对误差（C++ 0.0663354 / Rust 0.0663353）与 `Gamma_nbc0` 平均（-0.00120978 / -0.00120977）；
  dbc25 的 `Gamma_nbc0` 平均（0.000199341 / 0.00019934）与 `Gamma_rbc` 相对误差
  （0.0278363 / 0.0278362）。由打印区间反解，两侧真值至少差 6.3e-9 绝对（~1e-7 相对）。
  **(a) 示例 `integrate_bc` 的"半面"求积（真发现）**：`seg_quad` 取 `SegP1::quadrature` 的
  **[0,1] Gauss 点**（实测 `xi_q=[0.11270, 0.5, 0.88730]`、权重和 1），而 `eip_at` 的八个臂按
  **[-1,1]** 语义做 `0.5*(1±t)`（`deip` 也配 0.5）⇒ **每个面只采到"半条边"**（哪一半取决于面
  结点序相对单元角序的方向）。判决实验（`tmp/d771/ex27_bc_probe.cpp` mode 0/2/3，同网格+同解）：
  全幅/后半幅/前半幅三口径在 `Gamma_nbc` 相差 ~4e-8、`Gamma_rbc` ~2e-8，而 Rust 打印值**落在
  半面区间内**（Rust 网格：mode2=1.0086846150、mode3=1.0086846924、Rust=1.0086846240；
  全幅 mode0=1.0086846538 在区间外）⇒ 半面求积坐实（全表 `tmp/d771/probe_face_modes.txt`）。
  偏差仅 ~1e-8 是因为这些边界积分的被积
  函数沿单面近乎常数——**6 位打印长期掩盖了它**。修法（示例侧；会动 4 个平均值）：
  `eip_at` 改 MFEM `Loc1` 的 [0,1]↦[0,1] 恒等语义（如 `(0,1) => [t, 0.0]`）并去掉
  `deip` 的 0.5，即 `s=t` 而非 `0.5*(1±t)`。
  **(b) 网格构造顺序/表示差（实测）**：C++ = 缝合（folded）→ `SetCurvature(3)` → refine×2 →
  `Transform(trans)`；Rust = refine×2 → `set_curvature(3)` → `transform`（且**未折叠** + DOF 级
  周期）。只依赖网格的探针（同文件 tag3 洞边界）：洞边界曲线差 ~4e-10 相对（`sum_x2`
  25.920000602506406 vs 25.920000591255853、洞周长 nrm 1.25663706399702 vs 1.25663706131693；
  两侧单元体积分布与总面积 1.74867 = 2−2πa² 全同 ⇒ 两网格各自都合法），该差经边界平均值的
  相消放大到 ~1e-7 相对（同网格 mode0 对比：nbc avg 1.0086845458[C++ 网格] vs
  1.0086846538[Rust 网格] = 1.1e-7）。probe 口径自述（`sol.gf` 8 位往返）见
  `tmp/d771/probe_face_modes.txt`（两网格 × 4 口径全表）。
  **两个效应的加性账目闭合（`Gamma_nbc`，default 档）**：C++ 全幅 = 1.0086845457865752
  → [+1.0806e-7 = 网格差（Rust 网格全幅 1.0086846538487606）] → [−2.9872e-8 = 半面求积
  （Rust 实打 1.0086846239769045）] = **net +7.8190e-8 = 观测差**（Rust 打印 1.00868 vs C++
  1.00868，误差行 0.0663353 vs 0.0663354）。同法验 `Gamma_nbc` 误差行：C++ 0.066335350867800846
  → [+4.5903e-8 网格] → [−5.3028e-8 半面] → 0.06633534374250515 = Rust 打印值 ✓（闭合到 1e-17）。
  ⇒ 残 2 行/档**不是格式问题**，而是上述两因叠加；两因都在示例侧（可控）——(a) 改求积点映射
  即恢复 MFEM 口径，(b) 改网格构造顺序/折叠表示才可能达 stdout 全等（都会动这 4 个平均值）。
- **D779（新债）`-dg` 档与 C++ 未对齐（非格式）**：C++ `-dg -no-vis` 104 行（82 it，iter0
  `(B r,r)` = 0.142775）vs Rust 150 行（128 it，iter0 = 0.0220206）——`X0 = 0` 时 iter0 =
  ‖B‖²，**从第 0 步起 RHS/系统就不同**（DG 弱 Dirichlet/Boundary 载荷族或 `DgAssembler` 口径；
  **HYPOTHESIS**：(a) 类求积点映射错（`face_point_and_normal`/`assemble_l2_*` 同样用
  `(1∓xi)/2` 的 [-1,1] 语义配 [0,1] Gauss 点）可能是同族贡献者），且 Rust 用 `rtol 1e-12` 而非
  legacy `PCG()` 的 `sqrt(1e-12)` 语义 → 需专项裁决。**回归口径**：`-dg` 只保证"不回退"
  （D753 立场），与 C++ 的差距是本债内容。
- **D763 `-vs` 纪律（examples 侧交叉引用）**：该纪律的主记录在
  `tmp/ledger/miniapps_ledger.md` §round 72；要点：miniapps/multidomain 的打印步长参与**求解轨迹**，
  任何对比/回归必须显式钉 `-vs`（三档标准命令见该节），未钉 `-vs` 的历史记录一律降档为
  **不可复现**。（ex27 无 `-vs`，不受影响。）
- **D764 旧夹具 SUPERSEDED**：`tmp/dbit/multidomain_{rt,nd}_printref.cpp`（及其逐字节副本
  `tmp/d735/{rt,nd}_ref.cpp`）+ `tmp/dbit/*.txt` 18 件证据快照 + `tmp/d735/evidence.txt` 均已加
  SUPERSEDED 头（旧"never-refresh"模型 (2)，被 round 71 D749 证伪；现行权威 =
  `tmp/d749/multidomain_{rt,nd}_printref_refresh.cpp`、`tmp/d737/multidomain_h1_printref.cpp`；
  重钉四值 rt cyl −2.137667e-4/1.439350e-6、nd cyl 6.932270e-5/1.594225e-4）。历史证据保留未删。

## round 84 增量（C 路：三态台账刷新 + 抽样重验，HEAD c5a06dd6）

> 本节为**现行权威口径**。头部计数表已更新为 round-84 值；表体行仅在状态变化/本轮重验处
> 追加 `round-84:` 注记（历史行未重写）。证据：`tmp/d84c/`（24 档落盘日志 `run/logs/`、
> 对拍摘要、二分记录）。

### r61→r84 状态迁移总账（权威三态对照）

| 行 | r61 档 | r84 权威档 | 迁移轮次/债号 |
|---|---|---|---|
| ex1 | BIT | **BIT** | r66 D687（Options 头补齐后 cmp 全等） |
| ex2 | BIT | **BIT** | r63 D647 全流逐字节 |
| ex3 | RUN | **BIT**（双档） | r64 D651/D653 + r65 D664（4 处打印格式差清零） |
| ex4 | RUN\* | **RUN** | r62 D634（646 it = C++）+ r63 D639（0.0161443 逐字） |
| ex5 | RUN\* | **RUN** | r62/r63 D639（397 行逐字节） |
| ex8 | RUN | **RUN**（数值 = C++） | r63 D640（sinv 四边形分支；DPG 0.0183277 逐字） |
| ex9 | CRASH | **BIT**（具名豁免） | r62 D633 + r77 D812-2（stdout 逐字节）+ r78 D813-3（init.gf 逐字节）；r82/r83 重验 |
| ex14 | RUN | **BIT**（剥 Options 前缀） | r73 D795-1（311 行逐字节、ARF 0.956044）；r79/r82/r83 重验 |
| ex15dyn + 6×ex15_dump | 7×CRASH | **RUN** ×7 | r62 D633（资产回填，ex15dyn 513 s rc=0） |
| ex18 | RUN | **RUN-REG（回归）** | **r84 新发现 → D822-3**（见下） |
| ex20 | RUN\* | **RUN** | r62 D635b（六配置逐字节 = C++） |
| ex22 | RUN | **RUN**（多档逐位） | r67 D693-695 + r69 D738/D720 + r71 D748（余 D704/D748/D765/D767） |
| ex24 | BIT | **BIT**（四口径） | r66 D686 + r70 D712（D721 翻转闭环） |
| ex26 | RUN（r61 自 BIT 收窄） | **RUN** | r72 D754（并行装配逐位化，5 连跑 sha256 同） |
| ex27 | BIT\*（r72） | **BIT**（全逐字节） | r73 D778（51/51 + 52/52）+ r76 D805-3；r82/r83 重验 |
| ex29 | RUN | **RUN**（全数值行 = C++ 逐字节） | r62 D634 + r70 D758b（停机同步达成，本轮复证） |
| ex31 | BIT | **BIT** | r69 D724/D733（inline-segment 档诚实 exit(3)） |
| ex34 | RUN | **RUN**（leg1 逐位） | r70（pex34 ranks2 发散 = D756） |
| ex40 | RUN | **RUN**（终值 = C++） | r65 D674（0.0269215 = C++ 0.0269214） |
| pex3 | RUN | **RUN**（四锚点逐位） | r67 D697/D705 翻正 + r71 D746 + r78 四锚点（87/1/95/193 it） |
| pex5 | CRASH | **RUN** | r62 D635a + r64 D654（AMG 结构性豁免） |
| pex9 | CRASH | **RUN** | r62 D633（r76/r78 mass 锚 3.638348e2） |
| pex15 | CRASH | **RUN**（替代档 RUN-LONG） | r62 D633+D635a |
| pex40 | CRASH | **RUN** | r62 D635a + r64 D655（np1 页脚逐字节） |
| 其余 RUN 行 | RUN | **RUN**（不变） | r62–r83 未触碰；本轮抽样见下 |

### round-84 抽样重验（24 档，全部实跑落盘 `tmp/d84c/run/logs/`）

- **与 r61 记录 0 diff（13 档）**：ex7、ex16、ex19、ex21、ex23、ex25、ex28、ex30、ex36、
  ex37、ex38、ex41（r82 触面无扰动）、pex16/pex27 默认档（ranks 1）。
- **与 r61 记录仅噪声级差（1 档）**：ex33（iter120 末位显示差 1 ulp @1e-27）。
- **BIT/数值档复证（10 档，对 `tmp/ledger/ref/` 金标）**：ex1/ex2/ex3/ex24/ex31（仅 mesh
  路径行差，口径内）、ex14（311 行体逐字节）、ex20（数值行逐字）、ex4（终误差行逐字 +
  iter0-588 逐字节）、ex5（数值行逐字）、ex8（DPG 0.0183277 逐字）、ex29（全数值行逐字节，
  豁免仅 2 行 Rust 头注）、ex40（终值 D674 锚）、ex22（误差行与 r61 后状态逐字同）。
- **ex15_dump×6**：全部 rc=0，D633 修复钉住。
- **SKIP-TRANSIENT：无**（`cargo build --release --examples --keep-going` exit 0，334 exe，
  与他路无编译冲突）。
- **未重跑维持原档**：ex15dyn（513 s）、pex15（长跑）、pex30（RUN-LONG）、pex5/pex9/pex40
  （r62 已钉）、ex10/ex26/ex29/ex31/ex34（r69/r70 已钉）等；miniapps 全量不本轮范围。

### round-84 新债（D822 号段）

- **D822-1（P2）ex17 与 C++ 迭代数/输出表示失配（NOREF-NEW 首对拍）**：
  r76 D805-1 罚项修正传导后 ex17 数值漂移（1063→1269 it、‖u_h‖ −0.19%——旧值为错误罚项
  产物，非回归）。新增 C++ 参考（`ref/ex17_cpp_default.out`，binary `$HOME/work/d84c/ex17_cpp`，
  MFEM 4.10 默认档）：dofs 24576 逐字同；同 PCG+GSSmoother、rtol²=1e-12、maxit 5000 配置下
  **C++ 767 it vs Rust 1269 it**。HYPOTHESIS：Rust 停机用 ‖r‖/‖r0‖<1e-12 而 MFEM `PCG()`
  自由函数按 (B r,r) 比值 <1e-12（= 范数比 1e-6）——D634/D753"两套语义"同族。另：Rust
  `sol.gf` = H1_2D_P1 平均位移场（7.8 KB）vs C++ = L2_T1_2D_P1 DG 节点空间解（293 KB），
  示例内 "matching MFEM ex17 format" 注记失实（示例侧可修，会动输出文件）。
- **D822-2（P3）Options 回声缺口族**：ex14 缺 C++ 全部 12 行 Options 前缀（r73 起的既有
  豁免，本轮升级为债号）；ex20 缺 `--no-visualization/--no-gnuplot` 2 行回声（本轮新发现）。
  D687（ex1）同族收尾项。
- **D822-3（P1，回归）ex18 Euler 默认档 NaN**：HEAD 默认档 step 27（t≈0.07）起发散
  `Solution error: NaN`（rc=0 静默）。**二分钉到 `24e00a8d`（r81 D815-2 dg_hyperbolic
  3-D 面路径重构）**——前一笔 `6e5ef034` 稳定 6.168620814596272e-2（= r61 记录），其后
  `24e00a8d`/`7da0baaf`/`a0ab0f01`/HEAD 全 NaN；该债登记为 3-D 专用但 2-D Euler 被扰动
  （疑似 GenerateFaces 语义配对/face_type_of 分派波及 2-D 臂）。C++ 仲裁（既有 binary
  `$HOME/work/d76main/ex18_cpp`）：默认档 435 步稳定 0.0039309262 ⇒ 真回归，非 MFEM 忠实
  行为。r81 的 8/8 金标全为 3-D oracle，2-D Euler 无回归 pin ⇒ 建议主会话把 ex18 默认档
  加入常驻锚点集。注意：r61 记录的 6.17e-2 与 C++ 0.0039 的差距（185 vs 435 步，dt 计划
  不同）是 RUN 档既有求解器配置差，非本轮内容。

### 其他 round-84 记录

- **NOREF-NEW 登记**：ex17 此前无任何 C++ 参考，本轮现编（`$HOME/mfem410_ser` 串行树，
  g++ -O2）并留档 `ref/ex17_cpp_default.out`；snapshot 二进制 `$HOME/work/d84c/ex17_cpp`。
- **并行 ranks2 观察记录（非债，该档无既有主张）**：pex27 PCG 26→38 it（4 个平均值行逐字
  同）、pex16 chksum −9.9e-4 相对——r71-r73 并行分区/确定性改动族在未钉档上的漂移方向标。
- **矩阵 §4 队列第 1 项**（examples/miniapps 三态台账）的本 examples 半边：本节 + 头部
  round-84 计数表即为权威现状，**可关闭**（miniapps 半边维持 round-63/64/65 定档，待主会话
  复核后落格）。
- **⚠️ 会话事件**：本路工作期间 `fem-rs/data/` 工作树被清空（82 份 tracked `D`，非本路
  所为，命令清单已交主会话）；本路 `tmp/d84c/run/data/` 内有开局完整副本（124 项）可复原。
  本路 example 重跑均在该副本尚在/原 data/ 完好时完成，证据有效性不受影响，但**在 data/
  恢复前任何复跑都会 panic**。

## round 98 增量（主会话：RUN→BIT 第四波，HEAD 42a8b17e 基线）

> 单路主会话，采样原则 = 「距 BIT 最近的豁免收尾」+ 候选实跑现 diff（不照抄台账豁免清单——
> r84 多条豁免注记已过期：ex29「2 行头注」、ex5「Options 块 8 行 + Wrote 行」、
> ex20「C++ 回声多 2 行」均已不存在，系 r62-r83 各轮修复后未回写）。
> C++ 真值一律现编现跑：`$HOME/work/d98runbit/`（mfem410 源码 + mfem410_ser 库，
> `g++ -std=c++17 -O2 -I$HOME/mfem410 -L$HOME/mfem410_ser`）；Rust 侧 `cargo run --release`，
> 对拍副本与 diff 证据 `tmp/d98runbit/`。

- **ex29 升 BIT（双档）**：`-no-vis` 档 20 行 + 无参默认档 20 行均逐字节 = C++（ex29_cpp）。
  代码改动 2 处（示例侧 1:1 保真）：`visualization` 默认 false→true（C++ ex29.cpp:72
  `bool visualization = true`）；bool 对回声补 true 分支（MFEM `PrintOptions` 恰打其一：
  `--visualization`/`--no-visualization`）。修后 `-no-vis` 档不受影响（原已逐字节）。
- **ex20 升 BIT（双档）**：`-no-vis` 档 12 行 + 无参默认档 11 行均逐字节 = C++（ex20_cpp）。
  代码改动 2 处：`visualization` 默认 false→true（C++ ex20.cpp:98）；删 Rust 独有
  `Wrote ex20_phase.mesh, ex20_energy.gf` 行（C++ 走 socketstream 无 stdout）。vis=true 时
  副产物文件落 rundir（C++ 同档仅发 socket，无文件——stdout 不受影响）。
- **ex5 升 BIT（标准档，具名豁免 1 行）**：`-m data/star.mesh -no-vis` 414 行 stdout 与
  C++ 逐字节（396 it 收敛、u_err/p_err 行逐字同）；唯一 diff = `MINRES solver took …s.`
  （MFEM 自打 `chrono.RealTime()`，逐次不同，具名豁免类 = ex9 stderr wall-clock 同族）。
  **零代码改动**——r84 的 Options 块/Wrote 豁免已不存在。⚠️ 网格路径串进 Options 回声，
  两侧必须同形式 `-m data/star.mesh`（WSL 侧 `ln -s $HOME/mfem410/data data`）。
- **ex8 打印失真清零（维持 RUN）**：删 `println!()` 多余空行 + `S^{{-1}}`→`S^-1`
  （C++ ex8.cpp:244 无花括号）+ 删 Rust 独有 `PCG: iterations=` 摘要行后，stdout 前 15 行
  逐字同；**唯一残差 = PCG 轨迹自 iter5 起 1.3e-5 相对分叉**（7.50974e-07 vs 7.50964e-07，
  放大至 29 vs 28 it、ARF 0.619477 vs 0.608926）——非打印、非 ulp 求和噪声量级，
  = B^T·S^{-1}·B / S^{-1}·F 路径上某处真数值差，核心求解器债（D634 族），
  升 BIT 挡在此，建议下波专项（探针入口：iter0-4 逐字节、iter5 首叉）。
- **计数**：BIT 串行 8 → **11**（本波 +ex29/ex20/ex5；ex17/ex33/ex10/ex6 的历史升级见
  coverage_matrix §5，本表未回写）。本轮未动并行档。

## round 99 增量（主会话：D864 ex8 DPG 轨迹分叉专项——三处核心缺陷关闭）

> 单路主会话；真值 = 现编 MFEM 4.10（`$HOME/work/d99/`），逐位对拍机 = `tmp/d98runbit/`
> （`cmp_d99.py`/`cmp_mk.py`/`q_dist.py` + 单元素/扭曲单元/基函数探针）。这是「示例是手段、
> 核心库才是目的」的又一实例：ex8 停在 RUN 的价值是把三处**真核心缺陷**逼了出来。

- **根因链（逐位 dump 定位，非推测）**：把 F/b/B0/Bhat/Sinv/S0/Shat 七件从两侧全精度 dump 后
  逐条比位（`cmp_d99.py`），得三条独立缺陷：
  1. **`QuadL2GL` 节点 1 ulp 偏差**（`crates/element/src/lagrange/factory.rs`）：原用
     `gauss_legendre_arbitrary(n)` 在 [-1,1] 求根再 `0.5*(x+1)`，而 MFEM `Poly_1D::OpenPoints`
     是 128 位 MPFR Newton 直接给 [0,1] 值。因 GL-L2 的求积点与节点重合（order 1 ⇒ 2×2 GL=节点），
     1 ulp 节点差使基函数在求积点给 **~1e-17 伪残差**（MFEM 精确 0）、L2 单元刚度差 1–2 ulp；
     经 DPG 单元块 κ≈10³ 放大成 `S⁻¹` 的 **5e-12 相对误差**。修 = 改用已与 MFEM 逐位钉死的
     `[0,1]` 表 `gauss_legendre_01`（`HexL2GL` 本就如此）。
  2. **GL-L2 一维基用直接 Lagrange 乘积**（同文件）：MFEM `Poly_1D::Basis` 走**重心式**，且
     值/导数两个变体**故意不同式**（`CalcShape`：`u=l·w/(y−x)`；`CalcDShape`：`u=l·(1/(y−x))·w`）。
     逐位移植两变体 + 新 `eval_1d_vals()`（值专用）；postproc L2 误差（`grid_function.rs`）
     切到值专用变体（= MFEM `ComputeL2Error` 的 `CalcShape` 路径）。
  3. **`SinvBuilder` 两处**（`crates/assembly/src/dpg/sinv.rs`）：① 单元矩阵原为手写几何/梯度
     （`J⁻ᵀ` 缩放 + `w=weight·det`）+ k 外层累加 ⇒ 改走**通用体积装配器路径**
     （`Assembler::assemble_bilinear`，其伴随约定已由 S0 逐位证明）；② 求逆原为逐列解单位向量 ⇒
     换 **`mfem_dense_invert`**（MFEM `DenseMatrix::Invert()` 非 LAPACK Gauss–Jordan 分支的逐位移植，
     本构建 `MFEM_USE_LAPACK=NO`）。
- **顺带落地**：新核心积分器 `MixedScalarDiffusionIntegrator`（`crates/assembly/src/mixed/`，
  MFEM `DiffusionIntegrator::AssembleElementMatrix2` 标量支），替换示例与 `dpg_operator` 测试里的
  两份重复自写版（死代码零容忍）。
- **验收（逐位）**：**F、Bhat、Sinv、S0（值）、Mtest、Ktest 全部与 C++ 逐位相同**；
  S0 仅多存零；b 仅 ±1e-16 符号翻转（近零项）；B0 中位 ~5 ulp / 333 条 >100 ulp；
  Shat ≤5 ulp（RAP 求和序）。**ex8 标准档轨迹**：分叉点 iter5 的 1.3e-5 → **7.1e-8**
  （放大率改善 180×），29 行中仅末位数字翻转 ⇒ **维持 RUN**（未逐字节），三步配方入 D864。
- **反回归金标（本轮亲跑）**：workspace 全口径 **5347/0/40**（= r97/r98 锚逐位不变）、doc 28/0/102-gated；
  ex5 0 diff、ex9（hexagon）**逐字节**、ex14 仅 mesh 路径回声、ex24 star-p0-o1 0 diff、
  ex27 default 0 diff、ex29 0 diff；**ex18 反向受益**：误差 3.930926246114457e-3 →
  **3.930926246117042e-3**，对 C++ 17 位真值 `0.003930926246116611` 的相对差
  **5.5e-13 → 1.1e-13（贴近 5×）**；435 步不变。

## round 101 增量（主会话：D864 第三波——几何行序双杀，ex8 fork iter10 仍 RUN）

> 承 round-100 的残差定位（几何 J 1 ulp），本轮把 J 的成因与行序族一并清掉。

- **① 几何派发双轨缺陷（本轮主修复）**：仓库里有**两个**几何元派发器——`assembler.rs::geo_ref_elem`
  （Quad4/g≤1 → 闭式 `BiLinearGeo2D` ✓，S0 路径，故 S0 一直逐位）与
  `vector_assembler.rs::geo_ref_elem_from_mesh`（Quad4 → **重心式工厂 QuadQk** ✗，mixed/b0/向量路径）。
  MFEM 直边 quad 的几何元是 `BiLinear2DFiniteElement` **闭式**（解元是重心式——两者代数同、舍入异），
  `geo_ref_elem_from_mesh` 的 quad 臂漏掉了自家 hex 臂已有的处理（HexQ1 闭式）。实证：star 0 号元
  （det≈8e-4 薄片）QP1 的 `J[1][1]` 差 1 ulp——因子级对拍（PointMat/闭式 dshape/J）后钉死。
  修 = `geo_ref_elem_from_mesh` quad 臂 g≤1 → `BiLinearGeo2D`（hex 臂同款先例）。修后 **B0 全表逐位**。
- **② 行内存储序族（MFEM 链表组装语义）**：MFEM 的 BilinearForm 组装走链表**头插**，
  `Finalize` 保序转 CSR ⇒ **元素私有行存为逆序**（star 实测：Sinv row0 = [3,2,1,0]；S0 共享 dof 行
  = 逆时序 [4257, 1779, …]）。行序对值不可见、对 spmv/乘法累加序致命。三处对齐：
  `assemble_sinv_sparse` 改手工逆序 CSR ⇒ **SinvF 逐位**；`SinvBuilder::{apply,apply_matrix,apply_block}`
  改逆序累加；`assemble_b0_mfem` 改手工逆序 CSR ⇒ **b 逐位**。bhat 在 order-0 trace 下每行单条目
  （序不敏感），高阶 trace 时需同查（记入 D864）。
- **现状**：B0/Shat/SinvF/b **全部逐位 = C++**；ex8 轨迹分叉点 **iter9→iter10**（iter10 = 6.6511e-08 vs
  6.65111e-08）。**唯一残差定位 = S0（H1 共享 dof 行）的行内序**——MFEM 行 = 链表逆时序
  （S0.txt 实测 4961/5281 行非排序），修复需「MFEM 链表语义组装器」（通用装配器级爆炸半径，㉚
  金标账先行），登记为 D864 下一刀。ex8 维持 RUN。
- **门与 pin**：workspace 门 1 靶红 = `d800_straight_mesh_values_are_bitwise_stable` 的 L2-quad4
  自指稳定性 pin（防漂移哨兵，非 MFEM 真值 pin）——本轮几何修复使其 1 ulp 移动
  （1.63299316185545251 → …274），pin 值更新并留 D101 案（其余 10 pin 未受扰）；
  复跑 6/6 绿；其余 446 靶全绿（5346+1 修后 = 5347/0/40 口径）。

## round 127 增量（MFEM lane：RUN→BIT 第五波，HEAD 1a0d88a4）

> 抽样 3 档（ex19/ex16/ex21），收官 2 BIT + 1 立债。C++ 真值现编现跑
> `$HOME/mfem410_ser`（g++ -std=c++17 -O2）；Rust 侧 debug；两侧同参数同 CWD 形式。
> 证据与双 stdout 快照：`tmp/rr127mfem/`；C++ 探针与 run 目录：`$HOME/work/rr127/`。

- **ex16 升 BIT（1 行打印改动，零数值路径改动）**：Rust 独有的
  `=== Comparison Metrics ===` 尾块（L2norm/chksum 等）为 r61 时代的验证脚手架，
  C++ stdout 止于 `step 50, t = 0.5`——删除后 24 行 stdout 与 C++ `cmp` 逐字节全等
  （Options 12 行 + unknowns + 10 个 step 行，`cpp_fmt` 8 位有效数字口径早已对齐）。
  回归锚点无损：dofs 1361/steps 50/norm/chksum 区间断言仍在示例内 `#[cfg(test)]`。
- **ex21 升 BIT（四处 1:1 保真修复，见台账行注记）**：核心是补上被误删的解延长链路。
  旧注记称「prolongation 是死存储已删」——本轮现编探针实证 C++ 语义相反：
  ① `PCG()` 打印的 it0 `(B r, r)` = `(GS(B−A·X), B−A·X)` ≠ `(GS·B, B)`
  （X = 延长后的上一轮解，`P6` 探针 `(GS·B, B)` 恒为 5.078125e-05 而 PCG 打印
  0.00118678）⇒ legacy `PCG()` 跑在 initial-guess 模式，X 不清零；
  ② `B` 恒等于牵引力装配（`P5` 探针：‖B‖ ≡ 0.0070710678118654762，与 ‖b‖ 逐位同）
  ⇒ 同性边界（x[ess]=0）下 `EliminateVDofsInRHS` 无反应项注入——本轮曾实现的
  `B[ess] = −(A·x)` 反应注入被探针证伪后撤销；
  ③ P1 延长按 MFEM `RefinementOperator` 行点积口径（`0.5·u_a+0.5·u_b`），
  新中点父边按坐标位模式识别（`refine_2d::new_midpoint` 同式 `0.5*(a+b)`）。
- **ex19 立债（D905 号段建议：RUN 维持）**：Rust 版 Newton 是自研阻尼线搜索 +
  右预条件 GMRES（restart 30，监控真残差）；C++ 是 MFEM `NewtonSolver`（无线搜索、
  纯全步）+ **左预条件** GMRES + `JacobianPreconditioner`（块消元：mass-PCG/GS、
  stiffness-GMRES/GS、γ=1e-5）。实测分叉：it0 ‖r‖ 一致（2.94392），it1 起路径分叉
  （Rust 2.02052e-1 vs C++ 1.45342e-1，GMRES 16 it vs 15 it）——探针证明非线搜索之差
  （强制 α=1 输出一字不变），为线性求解器/预条件结构差。升 BIT 挡在
  **MFEM GMRESSolver 的逐位移植**（左预条件 + monitor 语义 + D704 族）与示例的
  Newton/预条件 1:1 重写，属核心求解器专项，建议登记后另波专项处理。
  证据：`tmp/rr127mfem/ex19_rs_raw.out` vs `ref/ex19_cpp_beamquad.out`。
- **顺带实证（rr125 线索收口）**：`d817r82_bdr_true_interior_matches_the_mfem_probe`
  （fem-rs io，rr125 曾 FAILED 于曲面 nodes 读回）在 HEAD `1a0d88a4` 已绿
  （`cargo test -p fem-io --test d817r82_bdr_true_interior_ledger` 1 passed）——
  即 D112b 修复件，rr124 报告中的 `test_shell_basic_masonry_contact` 与其无关。

## round 129 增量（MFEM lane：RUN→BIT 第六波，HEAD fd35de7b）

> 抽样计划 3 档（ex23/ex10/ex4 stretch）；**完成 1 档（ex23 升 BIT）后 C 盘余量
> 12G→5.5G 触发 <8G 停手纪律**，ex10 未启动（见下）。C++ 真值现编现跑
> `$HOME/mfem410_ser`（g++ -std=c++17 -O2，run 目录 `$HOME/work/rr129/`，data symlink
> 同 CWD 形式）；Rust 侧 debug exe，CWD = `tmp/ledger/rundir`。证据与探针链：
> `tmp/rr129mfem/`。

- **ex23_wave_equation 升 BIT（十处 1:1 保真修复，全部 examples 域，零核心库改动）**：
  `-m data/star.mesh -no-vis` 24 行 stdout 与 C++ **cmp 逐字节全等（0 豁免）**；
  `ex23-init.gf` **逐字节全等**；`ex23-final.gf` 2721/2722 值逐字同（唯一残值见 D906 候选）。
  修复清单：① Options 回声补 `--ref ` 行（C++ 注册未用的空串选项 ex23.cpp:138，行尾带
  空格）+ bool 对按值打 + double 换 %g8（`cout.precision(8)`）；② 默认值 `visualization`/
  `visit` false→true（ex23.cpp:77-78）；③ 删 Rust 独有 stdout（`ess_bdr count` 行 +
  尾部 stats 8 行块，2D/3D）；④ GLVis 文案对齐；⑤ **时间循环末步 dt 调整违例**
  （Rust `dt_actual = t_final - t`，C++ `GeneralizedAlpha2::Step` 末尾 `t += dt` 永不
  改 dt）→ 恒用 dt 后 t 位级同 C++；⑥ **初值 ess 清零违例**（C++ 投影后保留边界 dof 的
  e^-30 级值喂 `K->FullMult`，Rust 曾清零）→ 删零化（init.gf 因此从 2562/2722 →
  **1361/1361 全对齐**）；⑦ 求解器入口 `solve_pcg_jacobi`（linlvo 通用 CG：预倒数乘法
  1 ulp/次 + 通用停机 + 初值残留）→ **`solve_pcg_dsmoother`**（= MFEM `CGSolver::Mult`
  + DSmoother DiagScale 逐位移植，crates 已有入口，rtol² 判据 + 清零 + 除法 D⁻¹）；
  ⑧ **`Norml2` 逐位镜像**：MFEM 4.10 `Vector::Norml2`（vector.cpp:968）是 LAPACK
  dnrm2 风格**缩放算法**（先除后平方累加 + `scale·sqrt(sum)`），非 `sqrt(Σx²)`——
  plain 版在 ~1/3 dof 上 exp 参数差 1-2 ulp；示例内 `mfem_norml2_2d` 后 **exp 参数
  0/1361 失配**（此发现对其他示例有普适价值：任何用 `Norml2` 的初值/系数都有同款缺口，
  核心 Vector API 落地另波）；⑨ .gf 文件双头结构 + %g8 值（C++ 每次 Save 各写头部）；
  ⑩ 死代码清扫（文件级 allow + 死访问器删除）。
- **D906（候选号，P2）平台 libm 超越函数位差**：Windows CRT `exp` 与 glibc `exp` 在
  ~0.4% 输入上差 1 ulp（本例 5/1361 初值；exp **参数**位级 0 差、纯 libm 输出差）。
  探针因果链：坐标集 1361=1361 → arg 447 差（Norml2）→ 缩放算法后 arg 0 差 → 残
  5 值差（纯 exp）→ step-1 轨迹 2.6e-14 相对分叉 → z274 差 7e-28 绝对（= 值 3.75e-12
  翻转 dof 的 1 ulp 经 K 耦合）→ Tdiag274 位级同。**ex23 final.gf 唯一残值（du/dt
  dof274 末打印位翻转）即此债的下游回声。** 建议：核心库 vendor glibc 位级等价 exp
  （+ Norml2 缩放算法入 Vector API），另波专项；受益面 = 所有 transcendental 系数/初值
  示例（ex16/ex22/ex25/ex40 族）。
- **ex10_hyperelastic_dyn 未启动**：磁盘纪律（C 盘 5.5G < 8G，开工 12G；非本 lane
  消耗——本 lane 增量构建 <0.5G，`target/` 历轮缓存未动）。先验判断维持 brief 所给：
  r84 注记「EE/KE/ΔTE 6 位全吻合、Newton ‖r‖ iter1 起第 4 位漂移」，若漂移根因小
  （判据族）可循 ex21 legacy 口径先例，若 Newton/预条件结构差则如 ex19（rr127）立债。
  下一波候选顺位不变。
- **门自查**：ex23 rc=0 + stdout cmp 逐字节 + init.gf cmp 逐字节；`cargo test -p
  fem-examples --example mfem_ex23_wave_equation` 1 passed（轨迹锚点）；debug 构建
  0 错误；触碰文件零新警告（fem-examples 剩余 1 条 = 预存链接器 LNK4044 /ffast-math，
  vendor/linlvo 20 条 = 预存第三方）；df 开工 12G → 停手时 5.5G（外部 lane 消耗，
  告警已报主会话）。改动物：`examples/mfem_ex23_wave_equation.rs`（工作树）、本台账
  两处、`tmp/coverage_matrix.md`（§3 round-129 bullet）——tmp/ 下文件需主会话
  `add -f` 提交。
