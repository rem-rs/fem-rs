# miniapps 三态台账（round 61 抽样 → round 63 全量三态化 · C 路）

> **底账**：`miniapps/README.md`（1248 行过程台账：67 处 parity 记录、≈37 处 `exit(3)` 裁剪标注）。
> round 61 抽样 8 档全绿（7 文件）；**round 63（本轮）把其余 93 个文件全部三态化**
> （含 6 个纯模块文件 NOREF，87 个实体逐个实跑/对拍）。
>
> **口径**（与 examples_ledger 同款）：
> - 实跑：Windows release exe（`target/release/examples/`，本轮工作树，二进制新鲜已核
>   —— 无任何 crates/examples/miniapps 源文件晚于 exe mtime）；CWD = `tmp/ledger/rundir/`
>   （data junction）。`../../data/…` 形态的默认路径在 rundir 不可解析（rundir/../../data = tmp/data）
>   ——此类按"显式 -m 重跑"定档，路径约定问题单列注记，不计缺陷。
> - C++ 参考：**现编/现跑** `$HOME/mfem410_ser`（MFEM 4.10 串行，MFEM_USE_MPI=NO、
>   MFEM_USE_GSLIB=NO）+ 历史参考二进制 `$HOME/work/**`（nurbs_ex1/ex3、nex24、hpref_cpp、
>   extruder、toroid_cpp 等）；编译产物 `$HOME/work/r63bit/`；stdout 快照 + mesh 产物拷贝至
>   `tmp/ledger/ref/cpp_*`；diff 全部真跑。
> - **图例**：BIT = 与 C++ stdout/产物逐字节（注明豁免行）；BIT\* = 逐字节级但带**已立债**的
>   已知豁免；RUN = 跑通 rc=0、无逐字节对照（或数值级一致）；**RUN\*** = 跑通但与 C++/记录存在
>   数值级失配；RUN-LONG = 正常推进但超预算未完（或与 C++ 同为不可跑/无界档）；
>   CRASH = panic（原文登记）；DEV = 有意分歧/exit(3) 文档化；NOREF = 纯模块文件（无独立 MFEM 对照）。
> - 日志：`tmp/ledger/logs/mini_*.log`；rc 记录 `mini_exit_codes_{dev,bit,run}.txt`。

## 全量统计（100 文件 = 87 本轮 + 7 r61 + 6 模块）

| 分类 | 本轮 | r61 | 合计 |
|---|---|---|---|
| BIT / BIT\* | **14** | 7（8 档） | **21 文件（24 档）** |
| RUN（含 RUN-LONG 5） | **43** | 0 | **43** |
| RUN\*（数值级失配，立债） | **2** | 0 | **2** |
| CRASH（panic，立债） | **2** | 0 | **2** |
| DEV（exit(3)/有意分歧档） | **21** | 0 | **21** |
| NOREF（模块文件） | 6 | 0 | **6** |
| **合计** | **88 行目** | 7 | **100** |

（注：一文件可含多档——如 volta = DEV 默认档 + RUN `-maxit 1` 档、reflector = DEV 默认 + RUN
fichera 档、twist = BIT 两档；表中按文件归主档、备注记副档。21 个 DEV 行目中 6 个同时有
RUN/BIT 副档。）

**round 64 增量（4 文件，B/D 路）**：CRASH 2 → **0**（multidomain_nd/_rt 崩溃修复，转 RUN*）；
navier_bifurcation RUN* → **RUN**（D658 裁定 + `-pc amg` 档）；trimmer DEV(阻塞) →
**RUN(.mesh 对拍) + DEV(.vtk 诚实 exit(3))** 双档。净效应：RUN 43→44、RUN* 2→2（成员换为
multidomain_nd/_rt，数值未逐位 → D667）、CRASH 2→0、DEV 21 不变（trimmer 主档仍 DEV）。

## 本轮 BIT 复跑（14 项，全部真对拍；8 项逐字节、4 项记录档复现、2 项拓扑字节同）

| # | miniapp | 档位 | 对照物 | 结果 | 证据 |
|---|---|---|---|---|---|
| 1 | nurbs/nurbs_ex1 | `-m beam-hex-nurbs` | `$HOME/work/nurbs_ex1`（现跑） | **BIT**：solver 块逐字节（12 迭代 + ARF 0.253583 全同）；豁免 = C++ 多 30 行 Options/Mesh 头（README 在案） | logs/mini_nurbs_ex1_bh.log vs ref/cpp_nurbs_ex1.out |
| 2 | nurbs/nurbs_ex3 | 默认 square-nurbs | `$HOME/work/nurbs_ex3`（现跑） | **BIT**：整 stdout 除 13 行 Options 头外逐字节（165 迭代、ARF 0.918732、L2 8.41665e-06）。⚠️ beam-hex-nurbs（2 patch）档 **C++ 同样拒绝**（`NURBSExtension::GetCurlExtension … single patch`，abort 134）vs fem-rs panic 101 —— 语义一致，仅退出码约定差 | logs/mini_nurbs_ex3_sq.log vs ref/cpp_nurbs_ex3.out |
| 3 | nurbs/nurbs_ex24 | `-r 1 -p 0` | `$HOME/work/nurbs_ex24_ser/nex24`（现跑） | **BIT\***：dof 横幅 + 两条 L² 误差行（0.0224956/0.0039157）逐字节；PCG 迭代块 iter15 起末位 ulp 分叉 = D531 族（插结前后控制网几何），README 在案 | logs/mini_nurbs_ex24_r1p0.log vs ref/cpp_nurbs_ex24.out |
| 4 | nurbs/nurbs_ex5 | 默认 -no-vis | README 记录（现编 C++ 本档 = 同源） | **BIT**（记录复现）：MINRES 462 it、‖r‖_B 4.61012e-09、u-err 8.3193e-08、p-err 1.1665e-07、bnd dof 260/256 全同 | logs/mini_nurbs_ex5.log |
| 5 | nurbs/nurbs_printfunc | 默认 | 现编 4.10 nurbs_printfunc | **BIT**：48 行 diff 为空（第 3 次复核） | logs/mini_printfunc.log vs ref/cpp_printfunc.out |
| 6 | nurbs/nurbs_curveint | `-uw -n 9` | 现编 4.10 nurbs_curveint | **BIT**：豁免 = h_min/h_max/kappa 4 行（D374 在案省略）+ 2 条 fem-rs 注记行；其余逐字节 | logs/mini_curveint_uw9.log vs ref/cpp_curveint_uw9.out |
| 7 | nurbs/nurbs_surface | `-ex 1`（C++）/`-e 1`（Rust） | 现编 4.10 nurbs_surface | **BIT**：3 行特征（4×4 → 2×2 knot o3 → 40×40）逐字节；差异仅 vis 档值 | logs/mini_nurbs_surface.log vs ref/cpp_nurbs_surface_ex1.out |
| 8 | meshing/hpref | `-m inline-quad -pref -n 100` | 现编 4.10 hpref（同旗标） | **BIT**：仅 mesh 路径串 2 行差（历史 hpref_cpp 是无 -pref 的旧版，故现编）；h/p 序列逐字节 | logs/mini_hpref_iq100.log vs ref/cpp_hpref_pref.out |
| 9 | meshing/toroid | `-o 1` | `$HOME/work/toroid_cpp`（现跑） | **BIT**（拓扑字节同）：mesh 头 24 行（elements/boundary 连接表）逐字节；坐标 fem-rs 17 位 vs C++ 8 位（README 记录的 <5e-9 噪声级）；stdout 差 vis 档 + 1 行 trailer | rundir/toroid-wedge-o1-s0.mesh vs ref/toroid-wedge-o1-s0.mesh |
| 10 | meshing/twist | `-o 1 -no-pm`；另默认档 | 现编 4.10 twist（两档） | **BIT**：`-o1` NE=3 NBE=14 NV=16 拓扑逐字节；**默认 `-o 3 -pm` 档本轮 rc=0 写出曲面 nodes 网格且拓扑行同**（README 的 exit(3) 记载已过期——nodes writer 已于 r61/r62 落地，属升级）；坐标同 9 | rundir/twist-hex-o{1,3}*.mesh vs ref/cpp_twist-hex-o{1,3}*.mesh |
| 11 | meshing/extruder | `-m inline-quad -nz 4 -hz 2` | `$HOME/work/extruder`（现跑） | **BIT**：stdout 仅路径串 + vis 档 + Rust 1 行 Wrote-trailer 差；历史 mesh 级对拍（D144 残余偏差 3 条）在案 | logs/mini_extruder_iq.log vs ref/cpp_extruder.out |
| 12 | gslib/field-diff | `-no-vis`（triple-pt 夹具） | round-32 C++ 4.10 记录值 | **BIT**（记录复现）：Max 1.43502 / Avg 0.0949062 / Vol 1.73608 逐位。注：field-diff.cpp 需 GSlib，本机 4.10 串行树无 oracle（fi.err 实证） | logs/mini_field_diff2.log |
| 13 | gslib/field-interp | `-m1 square01 -m2 star` | round-32 记录 SHA `9f39ae2e…` | **BIT**（记录复现）：interpolated.gf SHA256 = `9f39ae2e8cfd18a4…` 逐字节 | logs/mini_field_interp3.log + sha256 |
| 14 | tools/get_values | `-r <现编 C++ ex5 DC>` | **全新真值链**：现编 4.10 ex5 产 DC → 两侧 | **BIT**：`0.790403` / `0.110318` 精确复现 round-32 官方样例值（此前 nan = 输入为 NURBS 场 DC，见注记） | logs/mini_get_values2.log |

## 全量台账（按目录；log = `logs/mini_<tag>.log`）

### electromagnetics/（5）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| maxwell.rs | 记录档 `-m fichera -rs 3 -ts 0.25 -tf 10 -dp …` | **RUN** | rc=0：H(Curl) 12336 / H(Div) 11520 dof = 记录、41 条 Energy 行、能量 8.27e-12J 量级 = 记录复现。C++ maxwell.cpp **MPI-only**（mw.err：`Mpi::Init`/`Hypre` 未声明）→ 串行树无现编 oracle，记录档逐字节主张无法在本机第三方复核。默认网格（NURBS）= exit(3) 声明（`--visit` 默认 ON 与 C++ 一致，已核源码 :136） | 
| joule.rs | 默认 | **DEV** | rc=3 部分交付；banner/Options/skin-depth/unknowns 推进（D120/D121 在案） |
| volta.rs | 默认（-maxit 2）→ DEV；`-maxit 1` → **RUN** | DEV+RUN | 默认 = exit(3)「AMR 需 3-D RT L2ZZ 未移植」声明兑现；-maxit 1：16/32/24/7 dof、Total charge、Solver done = README r31 审计更正复现 |
| tesla.rs | 默认 | **DEV** | rc=3 not_ported 首行退出 ✓ |
| lorentz.rs | 默认 | **DEV** | rc=3 缺口清单 ✓ |

### nurbs/（12）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| nurbs_ex1.rs | BIT | （上表 #1） | |
| nurbs_ex3.rs | BIT | （上表 #2） | |
| nurbs_ex5.rs | BIT | （上表 #4） | |
| nurbs_ex24.rs | BIT\* | （上表 #3） | |
| nurbs_printfunc.rs | BIT | （上表 #5） | |
| nurbs_curveint.rs | BIT | （上表 #6） | |
| nurbs_surface.rs | BIT | （上表 #7） | |
| nurbs_mesh_info.rs / nurbs_patch_ex1.rs | BIT | r61 抽样（mesh_info 现编逐字节；patch_ex1 netlib oracle 档） | |
| nurbs_solenoidal.rs | **DEV** | rc=3 声明 ✓（默认 `../../data` 路径在 rundir 会先 panic 101——路径约定，非缺陷） | |
| nurbs_ex10.rs | **DEV** | rc=3 声明缺口清单（vector NURBS 空间等 5 项）✓ | |
| nurbs_naca_cmesh.rs | **RUN** | rc=0 产网格；VisIt DC / glvis 副本按声明注记跳过 | |

### meshing/（16）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| mesh_bounding_boxes.rs | BIT | r61（10 组 CLI 逐字节） | |
| hpref.rs | BIT | （上表 #8） | |
| phpref.rs | **RUN** | `-n 100`：前 104 行 h/p 表逐字节；solve 块 39 vs 40 it、ARF 0.700042 vs 0.701048、H1 连续性 ~e-17 两侧 —— 求解器栈 ulp 级 | |
| mesh_quality.rs | **RUN** | `-size -aspr -skew`：数值全同（0.0625/0.0625/1/90/90、16 elem 2D）；**打印格式漂移**（skew 标签多 "1"、无 Options 头、多 trailer 行）→ 注记（不立案，见 D659 讨论） | |
| ref321.rs | **RUN** | `-mm -dim 2 -r 100`：unknowns 89=89、H1 连续性 ~0 两侧；C++ -mm 档多打一段 CG 自检块（fem-rs 自检深度不同） | |
| mesh-optimizer.rs | **RUN** | 默认档 rc=0；Final strain energy 7.4854、能量降 53.08%（D369 在案） | |
| fit-node-position.rs | **RUN** | 默认档 Newton 跑满预算未收敛（‖r‖/‖r_0‖=7e-4，线搜索 Neg-detJ 处理正常）；README 记录档为特定 gear（能量相对差 2.3e-10），未复跑该 gear | |
| toroid.rs | BIT | （上表 #9；默认 -o 3 档亦 rc=0 NE=8 NBE=24 NV=24） | |
| twist.rs | BIT | （上表 #10） | |
| extruder.rs | BIT | （上表 #11） | |
| reflector.rs | 默认 NURBS → **DEV** rc=3 ✓；`-m fichera` → **RUN** | fichera 档 NE=14 NBE=40 NV=43 = 记录；**顶点集多重集与 C++ 14/14 全等**（python 实证），角序旋转差 = round 31 有意决定（文件头在案） | |
| shaper.rs | **RUN\*** | `-m inline-quad`：16→64→256→…→65536（refine_uniform 回退）vs C++ NC 16→52→64 —— **D132 复现（仍开）** | |
| polar-nc.rs | **DEV** | rc=3 声明（NC mesh v1.0 writer 缺）✓ | |
| trimmer.rs | **RUN（默认 .vtk 原生）**【round 64 D659 解阻塞 → round 65 D675/D685/D676 关闭】 | `vtk_legacy_reader`（17 cell 类型 C++ 探针金标 + 三资产 Print 逐字节）接线后**默认 `data/beam-tet.vtk` 真 rc=0**（48 elements/36 nodes = C++）；`-m beam-tet.mesh -a 1` tet 档 **83/83 内容行逐字节**（round-64 82/83 → D685 切面 owner 透传+奇置换 Swap；D2 修正 C 配方：MFEM `CheckBdrElementOrientation` 无参 = fix_it=true，奇置换比对必要、循环起点不动）；hex `-a 2` 保持 diff=0；默认档残差 91 对保几何循环旋转 = **D688**（reader 固定 `Mesh(file,1,1)` vs C++ trimmer `(0,0)`——reader 加 refine 旋钮即收官） | |
| mobius-strip.rs / klein-bottle.rs | **RUN** | rc=0，网格写出（D171 记录在案，本轮未重对拍夹具） | |
| mesh-explorer.rs / mesh-quality.rs | explorer **RUN**（`-m beam-tet` rc=0 打印特征 + 写 mesh-explorer.mesh；2-D 输入按实现拒绝）；quality 见上 | |

### tools/（9）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| get_values.rs | BIT | （上表 #14） | NURBS 场 DC → nan（fem-rs 不能求值 NURBS1 场）＝库侧限制注记 |
| compare-dc.rs | **RUN** | `-r0/-r1` 自比对 rc=0、diff=0、|pressure|=60.955；异构 DC（无同名字段）rc=1 合理报错；round-32 逐字节记录在案 | |
| load-dc.rs | **RUN** | rc=1「Connection to localhost failed」= C++ 同款行为（round-32 已对齐） | |
| gridfunction_bounds.rs | **RUN** | `-m triple-pt-1 -s triple-pt-1.gf`：PL bounds 0.11669/3.00575 等全印；33/33 场景 C++ MPI 记录在案（D159/D255-259） | |
| display_basis.rs | **RUN** | `-e 2 -b 3 -o 3` rc=0 | |
| lor_transfer.rs | **RUN** | `-m inline-quad -o 2 -no-vis` rc=0（`-h1/-l2/…` panic! 分支未触碰；README 在案） | |
| nodal_transfer.rs | **RUN** | `-m data/beam-tet` rc=0（37281 unknowns）；默认 `../../data` 路径约定问题（注记） | |
| tmop_check_metric.rs | **DEV** | rc=3 声明 + 40-metric 缺口清单 ✓ | |
| tmop_metric_magnitude.rs | **RUN** | `-mid 7` rc=0 计算正常；无参默认 metric 2 rc=0（与 C++ 默认一致） | |

### toys/（5）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| automata.rs | **RUN** | rc=0，写 automata.mesh/gf | |
| life.rs | **RUN-LONG** | `-sp` 档 120s 未完：Game of Life 无界仿真（C++ 同为交互式无界）——非缺陷 | |
| lissajous.rs | **DEV** | rc=3 声明（2-D 嵌 3-D 面网格缺）✓ | |
| mandel.rs | **DEV** | rc=3；迭代 1（1024）与 C++ 完全相同（round-32 记录复现）；迭代 2 起 GeneralRefinement 缺口分叉（在案） | |
| mondrian.rs | **DEV** | rc=3；迭代 1（16）对齐（在案） | |

### diag-smoothers/（2）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| abs-l1-jacobi.rs | **RUN** | `-m ref-cube`：28 it、ARF 0.415311、L2 0.00143671；C++ 侧 MPI-only（现编失败实证）→ README harness 记录在案 | |
| mg_abs_l1_jacobi.rs | BIT | r61（np1 oracle 逐位） | |

### gslib/（4）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| findpts.rs | **RUN** | `-m rt-2d-q3 -o 8 -mo 4`、`-m inline-hex -o 3 -random 1 -npt 4` 两档 rc=0（max_err ~1e-15 记录在案；C++ 侧需 GSlib 无 oracle） | |
| field-diff.rs | BIT | （上表 #12） | |
| field-interp.rs | BIT | （上表 #13） | |
| schwarz_ex1.rs | BIT | r61（95 迭代逐位） | |

### spde/（6 = 1 实体 + 5 模块）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| generate_random_field.rs | **RUN** | `-m ref-cube -r 2 -rp 1 -no-vis -no-rs` rc=0：SPDE 解 + ParaView 导出（串行 1:1 记录在案） | |
| material_metrics.rs / spde_solver.rs / transformation.rs / util.rs / visualizer.rs | **NOREF** | generate_random_field.rs 的 `mod` 模块（宿主 RUN 覆盖） | |

### multidomain/（3）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| multidomain.rs | **RUN-LONG** | 默认档 300s 推进至 step 8550 / t=0.171（目标 0.25），sum/min/max 正常演化 —— 非挂死 | |
| multidomain_nd.rs | **RUN\***【round 64 D657 崩溃修复】 | 实体化配对（几何因子定号）替换坐标匹配；dof/ess 7708/5664、1168/800 = MFEM **全等**；IC 求和 −4.000000 = C++ 精确；轨迹 block 偏差 0.10→0.26%、cyl 0.5–2.8%（数值未逐位 → **D667 同族**） | |
| multidomain_rt.rs | **RUN\***【round 64 D657 崩溃修复】 | dof/ess 7296/5120、576/640 = MFEM 全等；IC −3.8e-17 vs −4.8e-17；block 轨迹 t=4e-5 偏差 3e-6 → t=0.002 偏差 2%；**cyl 首个 RK3 步后自由接口 Σv² ≈ 2.2× MFEM（发散）→ D667** | |

### shifted/（3）——均需 `-no-vis`（默认档 rc=3 vis 注记，声明兑现）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| shifted_distance.rs | **RUN** | L1 0.00278 / Linf 0.02035 | |
| shifted_diffusion.rs | **RUN** | GMRES(BiCGSTAB) 23 it | |
| shifted_extrapolate.rs | **RUN** | L2 0.8735 | |

### hooke/ dfem/ autodiff/（3）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| hooke.rs | **RUN** | ‖U‖ = 9.5635e-2 检查行 ✓ | |
| dfem_minimal_surface.rs | **RUN** | `-no-vis`：final ‖r‖ = 1.76e-17（三模式同终态记录在案） | |
| autodiff_example.rs | **RUN** | pLaplacian Newton rc=0 | |

### solvers/（4 = 2 本轮 + 2 r61）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| lor_elast.rs | **RUN** | 默认档 36 unknowns、FGMRES 启动正常 rc=0（D5 记录档为特定 gear） | |
| plor_solvers.rs | **RUN** | `star -o 3 -rs 1 -rp 1`：48 it、L2 2.502523e-5 = README 记录（np 无关档） | |
| lor_solvers.rs / block_solvers.rs | BIT | r61 | |

### adjoint/（2）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| adjoint_cvodes_roberts.rs | **RUN** | BDF 统计（392 步/接受 194/拒绝 198）+ checkpoint 重积分 5604 步 | |
| adjoint_advection_diffusion.rs | **RUN** | `-fd 1`：dG/dp2 伴随 1.094e-2 vs FD 1.0944e-2、rel 1.15e-7 PASS（README 5.8e-7/1.2e-7 同量级） | |

### dpg/（10）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| dpg_helmholtz_1d.rs | **RUN** | 16 elem k=0：L2 u 4.013e-2 | |
| dpg_poisson_2d.rs | **RUN** | n=2 o=1：12 dof、PCG 23 | |
| dpg_acoustics_2d.rs / dpg_maxwell_2d.rs | **RUN** | 表头正常（12/1.222e0、33/1.381e0；README 记录档为 rnum=4/-o 2 特定 gear） | |
| dpg_acoustics_3d.rs | **RUN** | 95/1.212e0 | |
| dpg_maxwell_3d.rs | **RUN** | **0 \| 156 \| 6.283 \| 1.723e0 = C++ 逐位记录复现**（round 17 结案值） | |
| pdiffusion.rs | **RUN**（BIT 级） | `0 \| 113 \| 1.021e+00 \| 9.951e-01` **= README 记录逐位** | |
| pacoustics.rs | **RUN**（BIT 级） | `0 \| 113 \| 2.0π \| 8.008e-01 \| 1.374e+00` **= README 记录逐位**（PCG 36 = 记录） | |
| pmaxwell.rs | **RUN** | 默认档两行（113/8.819e-01、417/4.753e-01）自洽；README 逐位档为 -prob 2 等 gear | |
| pconvection_diffusion.rs | **DEV** | rc=3 声明（系数 DPG 积分器缺）✓ | |

### fluids/（10）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| schrodinger_flow.rs | **RUN** | `--leapfrog`：GMRES 82 it、ARF 0.84156（无参 rc=1 用法门 = 设计） | |
| navier_kovasznay.rs | **RUN** | step 表打印（6.57566E-07 err_u 行格式与记录一致） | |
| navier_mms.rs | **RUN**（BIT 级） | `2.75455E-08 1.23108E-04` **= README 记录逐位** | |
| navier_shear.rs / navier_kovasznay_vs.rs / navier_tgv.rs | **RUN** | 表打印正常（3.401e0 SETUP 等） | |
| navier_3dfoc.rs | **RUN-LONG** | 600s：DOF 16956/5652 = 记录、HELM 收敛正常推进（t=6.4e-2 未完） | |
| navier_turbchan.rs | **RUN-LONG** | 默认 o5 = 两侧都 PA-only 不可跑（日志自述，README 在案）；`-o 1`：~140s/步正常收敛（PRES 37/HELM 33/步） | |
| navier_bifurcation.rs | **RUN**【round 64 D658 关闭】 | **回潮裁定 = 不成立**：C++ 4.10 串行镜像同 gear step 1 即 `PRES 200 4.64e+01`（与当前树逐位相同）——无 hypre 的 `OrthoSolver(GSSmoother)` 对 26k dof 纯 Neumann 压力 200 it 打不满是 MFEM 固有行为；新增 `NavierConfig::pressure_amg` + `-pc amg`（HypreBoomerAMG 串行 analogue，`crates/solver/src/navier.rs`）：`-pc amg` 档 PRES 24–28 it（默认 rs=3）/16–17 it（rs=1）全程收敛、零 No convergence；README「97/100 CFL 末位一致」**不可复现**（现树 1/101）→ 勘误 **D668** | |
| navier_cht.rs | **DEV** | rc=3 部分交付（热求解推进行在案）✓ | |

### hdiv_linear_solver/（3 = 2 实体 + 1 模块）
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| darcy.rs | **RUN** | L2 4.697e-4（0.18s） | |
| grad_div.rs | **RUN**（`-sp` 档） | 89 it、L2 2.7638e-4；无旗标默认 = **rc=0 no-op**「No solver enabled」+ `-ams/-lor/-hb` = rc=3 声明（hypre 栈缺，在案）。默认 no-op 行为值得后续改为 usage 提示（注记） | |
| hdiv_linear_solver.rs | **NOREF** | C++ hpp 共享源的模块（darcy/grad_div `#[path]` 引用） | |

### 其余
| 文件 | 档位 | 分类 | 备注 |
|---|---|---|---|
| adjoint_cvodes_roberts.rs 已列；lor_solvers `-fe l` 拒绝档 | DEV | rc=3 声明 ✓（r61 已 BIT 主档） | |

## CRASH / RUN* 与新债（D657–D659，round 63 登记 → round 64 处置）

- **D657（P2，CRASH）——round 64 关闭（B 路）**：根因 = 坐标匹配对 ND/RT 结构性失效
  （canonical 方向 GL 点 / face anchor / RT canonical 顶点帧在两 submesh 编号错位，实测
  两侧各 208 dof 仅 108 命中）+ 块 dof 查询只取块首（ND edge_dof 每边 1 个、RT tri_face_dof
  每面 1 个；FaceKey 未排序与 hdiv 注册键不一致，cyl ess 28 vs 正确 576）。修法 = 实体化配对
  （物理实体分组 + 几何因子 `g(d)` 定号）+ ess 改整块 + IC 对齐 ProjectBdrCoefficient + 积分阶
  逐项对齐 MFEM。两文件崩溃清零，dof/ess 计数与 MFEM 全等。残留数值差 → **D667**。
- **D658（P2，RUN\*）——round 64 关闭（B 路，裁定反转）**：**无回潮**——C++ 4.10 串行镜像
  同 gear 同样 `PRES 200 4.64e+01`（逐位同）；根因 = 串行无 hypre 的 MFEM 固有停滞。
  交付 = `NavierConfig::pressure_amg` + `-pc amg`（BoomerAMG analogue）。README「97/100 CFL
  末位一致」不可复现（1/101）→ 勘误 **D668**。
- **D659（P3，DEV 阻塞）——round 64 关闭（D 路）**：`beam-tet.vtk` 回填
  （MD5 `cfca8a890133d872b1f3f95eb5c064b4`）；默认 .vtk 档改诚实 rc=3 + D675 指向；
  `.mesh` 对拍路线 = GenerateFaces 1:1 重写后 hex 档 trimmer.mesh diff=0、tet 档 82/83
  （残 1 行 → **D676** 写路径）。

## round 64 新债（B/D 路登记）

- **D667（core，P2，已实测）**：multidomain RT cyl 首个 RK3 步后自由接口 Σv² ≈ 2.2× MFEM
  （发散）+ ND 轨迹 0.1–2.8% 偏差同族（传输本身已验证精确拷贝 `cyl_if==blk_if`）。嫌疑：
  RT1-hex 内部基函数 div 在 1-pt DivDiv 规则的采样 / `MixedWeakGradDot` 在非平行四边形
  facet hex 上的求值（crates/element、crates/assembly、crates/space/hdiv）。
- **D668（勘误，已实测）**：miniapps/README.md 与本台账旧记录「navier_bifurcation 收敛区
  100 步中 97 步 CFL 末位一致」在当前树不可复现（101 步仅 step 1 逐位，step 2+ 相对差
  ~1e-4；停更的 200-it 压力解对舍入混沌敏感）——README 批注 erratum 由主会话执行。
- **D675（io GAP，非 HYPOTHESIS）**：fem-io 无 VTK reader（fem-io 仅 MFEM 格式；MFEM
  `Mesh()` 原生读 VTK）。默认输入 `.vtk` 的 miniapp（trimmer 已声明 exit(3)）与未来对照受此限。
- **D676（io 写路径，已实测）**：fem-rs `.mesh` 写出对 Tet4 边界面做 MarkEdge 循环规范化
  （`mark_tet_mesh_for_refinement`），MFEM 写 Finalize 后原循环 → trimmer tet 残差 1 行；
  与 A 路 D663 同族（写路径 tet 朝向/槽位规范化 vs MFEM 存储序），合并追踪。

### 注记（不立案）
- **mesh_quality.rs**：数值与 C++ 全同，仅打印格式漂移（`Min skew 1 (in deg)` 多 "1"、
  缺 `Options used:` 头、多 `Mesh quality check complete` trailer）——README 称 1:1，严格
  逐字节不成立；下次触碰该文件时顺手对齐即可。
- **maxwell.cpp 为 MPI-only**（`Mpi::Init`/`Hypre` 编译失败实证）→ 串行树永远无现编 oracle，
  该 miniapp 的逐字节主张只能依赖 `$HOME/mfem410_mpi` 历史 harness。
- **get_values** 对 NURBS 场 DC 返回 nan（fem-rs 无 NURBS1 场求值）——H1 DC 值链全新复现。
- **hdiv_grad_div** 无旗标默认档 rc=0 静默 no-op（应改 usage）；`-ams/-lor/-hb` rc=3 声明兑现。
- **twist 默认档**已从「exit(3)」升级为 rc=0 真写出曲面 nodes 网格（r61/r62 nodes writer 落地
  红利）——README §meshing 的 exit(3) 记载过期。
- **nurbs_ex3 的 beam-hex-nurbs（2-patch）档**：C++ abort(134) / fem-rs panic(101)，同语义
  拒绝——退出码约定差异与 examples 侧 `MFEM_VERIFY→134` 家族一致，未单独立债。
- 路径约定注记：多处默认 `../../data/…` 仅在 `examples/` CWD 可解析（nodal_transfer、
  solenoidal、reflector 默认、navier_bifurcation 的 data/ 相对链）——与 examples 台账
  r61 注记同族，可用性问题非 parity 问题。
- README「诚实 exit(3)」清单本轮共验证 21 处，**全部兑现**（rc=3 + 声明文案），零回潮。

## round 61 抽样（保留；8 档 7 文件全绿，详见 git 历史）

lor_solvers×3 档、mesh_bounding_boxes×2 档、schwarz_ex1、block_solvers×3 档、
mg_abs_l1_jacobi、nurbs_mesh_info（现编 C++ 全新逐字节）、nurbs_patch_ex1（netlib oracle 档）
——均 BIT。
