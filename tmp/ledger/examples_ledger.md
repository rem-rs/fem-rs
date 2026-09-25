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

| 分类 | 串行 | 并行 | self-test | 合计 |
|---|---|---|---|---|
| BIT | 4（ex1/ex2/ex24/ex31） | 0 | 0 | **4** |
| RUN（含 RUN-LONG：pex15 替代档/pex30） | 29 | 34 | 0 | **63** |
| RUN\*（数值失配，已立债） | 3（ex4/ex5/ex20） | 0 | 0 | **3** |
| CRASH | 8（6×dump + ex9 + ex15dyn 默认档） | 4（pex5/pex9/pex15 默认档/pex40） | 0 | **12** |
| DEV | 0 | 2（pex19/pex32） | 0 | **2** |
| NOREF | 1（ex4_darcy_simple 未注册） | 0 | 1（par_heat self-test） | **2** |
| **合计** | **45** | **40** | **1** | **86** |

本轮 **RUN→BIT 升级 4 例**（ex1/ex2/ex24/ex31，现编 C++ 真对拍）。

## 串行（45）

| 名字 | MFEM 对应 | 档位 | 分类 | 证据/日志 | 备注 |
|---|---|---|---|---|---|
| mfem_ex0_mesh_intro | ex0 | 默认档 | RUN | logs/mfem_ex0_mesh_intro.log | ARF 0.140201 收敛正常 |
| mfem_ex1_poisson | ex1 | `-m data/star.mesh -no-vis` | **BIT** | logs/… + ref/ex1.out | **本轮现对拍**：除 Rust 缺 10 行 `Options used:` 头外逐字节（111 迭代 + ARF 0.882852 全同） |
| mfem_ex2_elasticity | ex2 | `-m data/beam-tri.mesh -no-vis` | **BIT** | 同上 + ref/ex2.out | **round 63 D647 全流逐字节**：C++ 4.10 ex2 本就不打 `Wrote…`（旧豁免注销），Rust 删该 stderr 行后 stdout 278 行/11783 字节 + stderr 空两侧全同（tmp/d597/ex2run/）；ARF 0.965229/268 迭代全同 |
| mfem_ex3_maxwell_cavity | ex3 | 默认 beam-tet -o1 | RUN | logs/… + ref/ex3.out | 终值 3.91630923150637e-1 = C++ 0.391631（6 位）；PCG 历史口径不同（Rust 打归一化残差、119 迭代 vs C++ 137/ARF 0.903118）→ D634 |
| mfem_ex4_darcy | ex4 | `-m data/star.mesh -no-vis` | **RUN\*** | logs/… + ref/ex4.out | **失配**：‖F−F_h‖=0.432497 vs C++ 0.0161443（27×）；Rust 287 迭代即"收敛" vs C++ 646 → D634 |
| mfem_ex4_darcy_simple | —（无对应） | — | NOREF | — | 未注册死文件、无 exe（round 30 D133 在案）；自述 SIMPLIFIED |
| mfem_ex5_mixed_darcy | ex5 | `-m data/star.mesh -no-vis` | **RUN\*** | logs/… + ref/ex5.out | **失配**：dim(R/W) 全同（41280/20480），Rust MINRES 423 it ‖r‖/‖b‖=9.24e-7 判收敛但 u_err **1.211582e0** vs C++ 396 it / 1.43587e-4 → D634（块预条件/minres 判据族） |
| mfem_ex6_flux_recovery | ex6 | 默认 star -o1 -no-vis | RUN | logs/… + ref/ex6.out | 前 4 迭代逐字节（0.441629/0.00864066/2.48721e-06/1.90288e-09）；首个求解 C++@5 停、Rust 拖到 4.6e-40 → D634；AMR 环两端 rc=0（C++ 打 `Reached the maximum number of dofs.`，Rust 打 `Done.`，终态行未逐字节比对） |
| mfem_ex7_surface_poisson | ex7 | 默认 | RUN | logs/… | rc=0 |
| mfem_ex8_dpg_2x2 | ex8 | `-m data/star.mesh -no-vis` | RUN | logs/… + ref/ex8.out | 前 13 迭代逐字节；DPG 范数 0.0181446 vs C++ 0.0183277（~1%）；ARF 0.829413 vs 0.608926 → D634 |
| mfem_ex9_dg_advection | ex9 | 默认 periodic-hexagon | CRASH | logs/… | 默认档 panic（D633 资产缺失）；`-m data/periodic-square.mesh` rc=0；文件头自述 L2 基 ≠ MFEM GLL ⇒ 有意分歧（DEV 性质）双注记 |
| mfem_ex10_hyperelastic_dyn | ex10 | 默认 beam-quad | RUN | logs/… + ref/ex10_quad.out | step1..100 的 EE/KE/ΔTE 与 C++ 全部 6 位吻合（0.011958/0.000784/-0.019639）；Newton ‖r‖ 自 iter1 起第 4 位漂移（0.0099624 vs 0.0099476）；打印多诊断行 |
| mfem_ex14_dg_poisson | ex14 | `-m data/star.mesh -no-vis` | RUN | logs/… + ref/ex14.out | **前 309 行逐字节**；C++@308 收敛（ARF 0.956044），Rust 500 maxiter 不收敛（ARF 0.950077）→ D634 |
| mfem_ex15_dynamic_amr | ex15 | 默认 star-hilbert | CRASH | logs/… | 默认档 panic：`data/star-hilbert.mesh` 不存在（D633）；`-m data/star.mesh` 替代档 700 s 内 rc=0（435 s，20 轮 AMR） |
| mfem_ex15_dump_A_true | —（tools_ex15_ref 配套） | 内置 star-hilbert | CRASH | logs/… | debug dump harness（非用户示例）；panic 于读 `data/star-hilbert.mesh`（D633） |
| mfem_ex15_dump_T002 | 同上 | 同 | CRASH | logs/… | 同上（D633） |
| mfem_ex15_dump_flow | 同上 | 同 | CRASH | logs/… | 同上（D633） |
| mfem_ex15_dump_it2_coords | 同上 | 同 | CRASH | logs/… | 同上（D633） |
| mfem_ex15_dump_p1 | 同上 | 同 | CRASH | logs/… | 同上（D633） |
| mfem_ex15_dump_p1_it3 | 同上 | 同 | CRASH | logs/… | 同上（D633） |
| mfem_ex16_nonlinear_heat | ex16 | 默认 star | RUN | logs/… | rc=0（SDIRK33 时间推进完成） |
| mfem_ex17_dg_elasticity | ex17 | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_ex18_euler | ex18 | 默认 periodic-square | RUN | logs/… | rc=0（RK4 推进完成） |
| mfem_ex19_hyperelastic_incomp | ex19 | 默认 beam-quad | RUN | logs/… | rc=0（Newton+块 GMRES 收敛） |
| mfem_ex20_symplectic | ex20 | 默认 -o1 -t 100 步 | **RUN\*** | logs/… + ref/ex20.out | **失配**：能均值/方差 = `1 / 0` vs C++ `1.00204 / 0.0174915`（能量恒 1 ⇒ 积分器未真正演化）→ D635 |
| mfem_ex21_amr_elasticity | ex21 | 默认 beam-tri | RUN | logs/… | rc=0；历史有机器本地 golden 注记（非 C++ 逐字节） |
| mfem_ex22_complex_helmholtz | ex22 | 默认 inline-quad | RUN | logs/… | rc=0（复数系统求解完成） |
| mfem_ex23_wave_equation | ex23 | 默认 star | RUN | logs/… | rc=0；历史 golden 本地注记 |
| mfem_ex24_discrete_ops | ex24 | `-m data/star.mesh -p 0 -o 1 -no-vis` | **BIT** | logs/… + ref/ex24_p0o1.out | **本轮现对拍**：数值行逐字节；豁免 = Rust 的 Options 块少 3 行（--no-static-condensation/--no-partial-assembly/--device） |
| mfem_ex25_pml_maxwell | ex25 | 默认 beam-hex | RUN | logs/… | rc=0（复 PML 系统完成） |
| mfem_ex26_geom_mg | ex26 | 默认 star；hex 档 `-m inline-hex -gr 0 -or 2` | RUN | logs/mfem_ex26_geom_mg.log、logs/mfem_ex26_hex.log + ref/ex26_hex.out | 默认档 rc=0。hex 档本轮**实况对拍**：274625 未知数同、6 迭代轨迹机器精度级吻合（ARF 0.0435272 vs 0.0435273、iter1 1.55112e-6 vs 1.55209e-6），**非逐字节**（Chebyshev smoother 特征值估计 ulp 级分叉）。round 60 存档对 tmp/d31/ex26_{rs,cpp}.log 数值逐字节，但其原命令不可复原 ⇒ 存档对不作为本轮逐字节证据引用 |
| mfem_ex27_robin_bc | ex27 | 默认 `-no-vis` | **BIT\***（round 72 D771） | logs/… + `tmp/d771/`（gold/diff/probe） | **canonical stdout 对齐**：Options 块 12 行 + 去前导空行 + 删非 C++ 的 `Solved in N iterations.` + 平均值行 `", \t"` 与 `%g6`（`fem_solver::fmt_g`）⇒ **default 51 行中 49 行逐字节、`-dbc 2.5` 52 行中 50 行逐字节**（残 2 行/档 = 半面求积 + 网格表示差 **D778**，非格式）；迭代历史 29/30 行零差异保持；**`-dg` 迭代历史逐字节不回退**（其 C++ 差距 = **D779**） |
| mfem_ex28_sliding_elasticity | ex28 | 默认 | RUN | logs/… | rc=0 |
| mfem_ex29_curved_poisson | ex29 | 默认（-mt 4 -mo 3） | RUN | logs/… + ref/ex29.out | 迭代 0–7 逐字节；C++@7 停（ARF 0.0969461）Rust@11（ARF 0.0719572）；终误差 ‖u−u_h‖ 0.00138643 / ‖f−f_h‖ 0.00797749 **逐字节同**；Rust 多 2 行头注 → D634 |
| mfem_ex30_aniso_amr | ex30 | 默认 star | RUN | logs/… | rc=0（三系数预处理完成） |
| mfem_ex31_anisotropic_maxwell | ex31 | `-m data/inline-quad.mesh -r 2 -o 1 -no-vis` | **BIT** | logs/… + ref/ex31.out | **本轮现对拍**：除 Options 里 mesh 路径串外**全 stdout 逐字节**（含 0.181455 / ARF 0.829075）；无参默认档会 exit(3)（-vis 未移植，文档化） |
| mfem_ex31_dump | —（tools/ex31_cpp_helper 配套） | `-m data/inline-quad.mesh -r 2 -o 1` | RUN | logs/… | debug dump harness；rc=0，落盘 rust_*.txt 供 harness 比对（D375 管线） |
| mfem_ex33_fractional_diffusion | ex33 | 默认 star | RUN | logs/… | rc=0；AAA 模块与 spde 共享（那边有逐位锚点），本示例无逐字节记录 |
| mfem_ex34_magnetostatics | ex34 | 默认 fichera-mixed | RUN | logs/… | rc=0（SubMesh 电流密度链路完成） |
| mfem_ex36_obstacle | ex36 | 默认 disc_p2 | RUN | logs/… | rc=0（proximal Galerkin Newton 完成） |
| mfem_ex37_topology_optimization | ex37 | 默认 | RUN | logs/… | rc=0 |
| mfem_ex38_implicit_integration | ex38 | 默认 surface2d | RUN | logs/… | rc=0（moment-fitting 求积完成） |
| mfem_ex39_compass | ex39 | 默认 compass.msh | RUN | logs/… | rc=0（命名属性集链路完成） |
| mfem_ex40_eikonal | ex40 | 默认 star | RUN | logs/… | rc=0（阻尼拟牛顿完成） |
| mfem_ex41_imex | ex41 | 默认 periodic-square -s64 | RUN | logs/… | rc=0（IMEX DIRK3 1000 步完成） |

（注：串行 45 行齐全——含 ex4_darcy_simple（NOREF）与 6 个 ex15_dump（CRASH）；
86 = 45 串行 + 40 并行 + 1 self-test，与盘点清单 `tmp/ledger/ex86_list.txt`
（85 可跑 + 1 未注册死文件）一致。）

## 并行（40）

| 名字 | MFEM 对应 | 档位 | 分类 | 证据/日志 | 备注 |
|---|---|---|---|---|---|
| mfem_pex0_parallel_poisson | ex0p | 默认 --ranks | RUN | logs/… | rc=0 |
| mfem_pex1_parallel_poisson | ex1p | 默认（81920 quads） | RUN | logs/… | rc=0（240 s 内；round 30 曾 150 s 超时） |
| mfem_pex2_parallel_elasticity | ex2p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex3_maxwell_cavity | ex3p | 默认 star（2-D 折叠档） | RUN | logs/… | rc=0；历史对照：dofs 10400、‖E−E_h‖=2.70053055689122e-2 vs C++ mpirun 0.0270053（6 位；迭代数 102 vs AMS 17 属求解器栈差异） |
| mfem_pex4_parallel_hdiv_diffusion | ex4p | 默认 star | RUN | logs/… | rc=0；文件头注明 PCG+Jacobi 替代 AMS（求解器栈差异） |
| mfem_pex5_hdiv_darcy | ex5p | `-m data/star.mesh -no-vis` | **CRASH** | logs/… | panic：`crates/parallel/src/launcher/native.rs:154` worker `index out of bounds: len 2 index 2`（round 30 曾 rc=0 ⇒ **回归嫌疑**）→ D635a |
| mfem_pex6_parallel_amr | ex6p | 默认 star | RUN | logs/… | rc=0（AMR 环完成） |
| mfem_pex7_parallel_surface | ex7p | 默认 octahedron | RUN | logs/… | rc=0 |
| mfem_pex8_parallel_dpg | ex8p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex9_parallel_dg_advection | ex9p | 默认 periodic-hexagon | **CRASH** | logs/… | panic 于读 `data/periodic-hexagon.mesh`（D633 资产缺失） |
| mfem_pex10_parallel_hyperelastic | ex10p | 默认 beam-quad | RUN | logs/… | rc=0 |
| mfem_pex11_parallel_eigenvalue | ex11p | 默认 star | RUN | logs/… | rc=0（LOBPCG） |
| mfem_pex12_parallel_elastic_eigen | ex12p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex13_parallel_eigenvalue | ex13p | 默认 beam-tet | RUN | logs/… | rc=0 |
| mfem_pex14_parallel_dg_poisson | ex14p | 默认 star | RUN | logs/… | rc=0 |
| mfem_pex15_parallel_dynamic_amr | ex15p | 默认 star-hilbert | **CRASH**（默认档） | logs/… | 默认档 panic（D633）；`-m data/star.mesh` 替代档 **800 s 超时**：日志推进正常（AMR 迭代 2 → 59352 unknowns，带 load rebalance）⇒ 长跑非挂死；对照：串行 ex15 同档 435 s 完成 |
| mfem_pex16_parallel_nonlinear_heat | ex16p | 默认 star | RUN | logs/… | rc=0 |
| mfem_pex17_parallel_dg_elasticity | ex17p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex18_parallel_euler | ex18p | 默认 periodic-square | RUN | logs/… | rc=0 |
| mfem_pex19_parallel_incomp_hyperelastic | ex19p | 默认 beam-tet | DEV | logs/… | exit(3)+文案：本 port 仅 2-D 分支（C++ 默认网格 3-D）——round 31 D137 文档化裁剪 |
| mfem_pex20_parallel_symplectic | ex20p | 默认 | RUN | logs/… | rc=0 |
| mfem_pex21_parallel_amr_elasticity | ex21p | 默认 beam-tri | RUN | logs/… | rc=0 |
| mfem_pex22_parallel_complex_helmholtz | ex22p | 默认 inline-quad | RUN | logs/… | rc=0 |
| mfem_pex24_parallel_discrete_ops | ex24p | 默认 beam-hex | RUN | logs/… | rc=0 |
| mfem_pex25_pml_maxwell | ex25p | 默认 beam-hex | RUN | logs/… | rc=0 |
| mfem_pex26_parallel_geom_mg | ex26p | 默认 star | RUN | logs/… | rc=0 |
| mfem_pex27_parallel_robin_bc | ex27p | 默认 | RUN | logs/… | rc=0（round 30 的内核 assert panic 已于 D138 修复，本轮绿） |
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
| mfem_pex40_eikonal | ex40p | `-m data/star.mesh -no-vis` | **CRASH** | logs/… | panic：`crates/parallel/src/dof_partition.rs:1251 index out of bounds: len 2 index 2` → D635a |
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
