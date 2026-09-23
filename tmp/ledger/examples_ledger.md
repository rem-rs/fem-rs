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
| mfem_ex27_robin_bc | ex27 | 默认 | RUN | logs/… | rc=0 |
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
