# examples + miniapps 三态台账（round-117 lane 1 · 一次性批跑收口 · HEAD `e4a819c5`）

> **本文即 coverage_matrix §4-1 的收口证据**：对 fem-rs 全部 examples（86 文件）+ miniapps（101 文件）
> 的一次性批跑 + 三态定性。历史底账 = `examples_ledger.md`（round 61/84/98/99/101 增量）与
> `miniapps_ledger.md`（round 63/64/72 增量），本文不重写历史行，只给 HEAD 现值。
>
> **三态口径**（本表唯一分类）：
> - **逐字节 BIT**：fresh stdout（及适用时写出产物）对**已提交 oracle/pin/md5** diff=0，
>   豁免仅限具名类（`--mesh` 路径回声行、wall-clock 行）；
> - **数值带 NUM**：无逐字节门，但存在**决定性数值行与 committed oracle/记录逐字相同**
>   （终误差/能量/锚点表行；迭代数与轨迹差异属已立债求解器栈族 D634）；
> - **REPORT-ONLY RO**：rc=0 实跑 + 自锚/历史记录稳定性核对，无本轮 committed oracle 决定性对照。
>   附：DEV = 声明 exit(3) 裁剪兑现；CRASH = panic；RUN-LONG = 推进正常但超本轮预算；UNREG = 未注册。
>
> **实跑环境**：Windows release exe（`target/release/examples/`，HEAD 工作树，`cargo build --release
> --examples` exit 0）；CWD = `tmp/r117/rundir/`（data junction → 仓库 `data/`，另含 r61 rundir 资产副本）；
> 路径约定默认档（`../../data`）从 `tmp/r117exrun/` 复跑。日志 `tmp/r117/logs/`，rc/耗时
> `tmp/r117/{serial,mini,pex}_rc.txt`。新鲜 C++ oracle（本轮现编/现跑，`$HOME/work/r117/`）：
> ex7（tri/quad 双档）、ex17（d84c binary 重跑）。

## 统计总表

| 域 | 逐字节 BIT | 数值带 NUM | REPORT-ONLY RO | DEV | CRASH | RUN-LONG | UNREG | 合计 |
|---|---|---|---|---|---|---|---|---|
| examples 串行 | **14** | 15 | 14 | 0 | 0 | 0 | 1 | **45 文件（44 实跑 + 1 UNREG）** |
| examples 并行 | 0 | 2 | 34 | 1 | 0 | 2 | 0 | **40**（+ranks2 spot 5 档全 rc=0） |
| miniapps | **18 文件** | 31 | 31 | 8 | 0 | 4 | 9（模块/未注册） | **101 文件（92 实体 + 9）** |

（126 档 fresh 日志 + 5 ranks2 spot 在 `tmp/r117/logs/`；rc/耗时 `tmp/r117/{serial,mini,pex}_rc.txt`。
miniapps 102→101 勘误：round-63 台账的 100 文件口径未含 r63 后新增的 `navier_particles.rs`/`sbm_aux.rs`，
本轮按 git 现树 101 文件盘点。）

（计数细节与逐行依据见下文各表；gate pins 本轮全绿：`d106_toys_joule_byteparity` 5/5、
`d106_tesla_d957_ams_parity` 2/2、`d822_ex18_euler_2d_regression` 4/4。）

---

## §1 清单与 oracle 证据源（inventory）

- **examples/*.rs = 86 文件**（45 串行 + 40 并行 + 1 self-test），`examples/Cargo.toml` 注册
  **177 个 example 目标**（86 examples 中 85 注册 + `mfem_ex4_darcy_simple` 未注册死文件[round-30 D133 在案]；
  92 个 miniapp 注册名，含 `mini_*`/`mesh_*`/`gslib_*`/`tools_*`/`toys_*`/`diag_*`/`spde_*`/`hdiv_*`/`miniapp_*`/`shifted_*`/`block_solvers`/`lor_*`/`tmop_*`/`dpg_*`/`p*` 前缀映射）。
- **miniapps/***.rs = 101 文件**：92 注册实体 + **9 个未注册/模块文件**（`fluids/navier_particles.rs`、
  `shifted/sbm_aux.rs`[后两者 **round-63 台账后新增**]、`hdiv_linear_solver/hdiv_linear_solver.rs`[共享模块，
  darcy/grad_div `#[path]` 引用]、`dpg/util/pml.rs`、`spde/{util,spde_solver,material_metrics,transformation,visualizer}.rs`
  [宿主 `generate_random_field` 的 mod 模块]）。
- **committed oracle 清单**（本表比对物）：
  - pins（cargo test，本轮亲跑全绿）：`examples/tests/d106_toys_joule_byteparity.rs`（mandel/mondrian/lissajous/joule，
    md5 stdout+产物 vs MFEM 4.10 oracle）；`crates/solver/tests/d106_tesla_d957_ams_parity.rs`（tesla -bm o2 ×2）；
    `crates/assembly/tests/d822_ex18_euler_2d_regression.rs`（ex18 RK3 终态 ×4）。
  - `tmp/ledger/ref/*.out`（r61/84 C++ 4.10 stdout 快照 ×42）：ex1/ex2/ex3/ex4/ex5[→d98runbit]/ex6/ex8[→d98runbit]/
    ex10_quad/ex14/ex20[→d98runbit]/ex24_p0o1/ex29[→d98runbit]/ex31 + miniapps nurbs/hpref/twist/toroid/extruder/
    curveint/printfunc/field-diff/reflector/mesh-quality 等。
  - `tmp/d98runbit/cpp_*_copy.out`（r98 现编现跑副本）：ex5/ex8/ex20/ex29。
  - `tmp/d89main/cpp_*.out`（r89 现编 ×10）：ex16/ex19/ex21/ex23/ex25/ex36/ex37/ex38(死 oracle: LAPACK 缺)/ex39(死: compass.msh 缺)/ex41。
  - `tmp/d89c/cpp_ex33_fresh.out`、`tmp/d771/cpp_default.gold`、`tmp/ledger/rundir` 资产（square01/triple-pt/DC 夹具）。
  - 本轮新鲜：`tmp/r117/cpp_ex7_tri.out`+`cpp_ex7_default_quad.out`（`$HOME/work/r117/ex7_cpp`，mfem410_ser g++ -O2）、
    `tmp/r117/cpp_ex17.out`（`$HOME/work/d84c/ex17_cpp` 重跑）、`tmp/r117/cpp_ex9{,.out,_final.gf}`（`$HOME/work/d76main/ex9_rerun`）。

## §2 examples 串行（45）

| 例子 | 档位（本轮） | 三态 | 本轮证据（HEAD `e4a819c5`） |
|---|---|---|---|
| mfem_ex0_mesh_intro | `-m data/star.mesh` | **BIT（r135）** | rc=0；r61 后 stdout 无漂移；**r135 补 3 行 Options 回声后**（MFEM 4.10 ParseCheck 会打 PrintOptions）`-m data/star.mesh` 491 字节 vs fresh oracle（`$HOME/work/rr135/ex0`）**cmp 逐字节全等**（0 豁免，sha256 9690d8e2…） |
| mfem_ex1_poisson | `-m data/star.mesh -no-vis` | **BIT** | vs `ref/ex1.out` 2 difflines = mesh 路径行（豁免类） |
| mfem_ex2_elasticity | `-m data/beam-tri.mesh -no-vis` | **BIT** | vs `ref/ex2.out` 2 difflines = mesh 路径行 |
| mfem_ex3_maxwell_cavity | 无参 beam-tet | **BIT** | vs `ref/ex3.out` 2 difflines = mesh 行（built-in vs 文件路径） |
| mfem_ex4_darcy | `-m data/star.mesh -no-vis` | **NUM** | 终误差行 `‖F_h−F‖=0.0161443` 与 C++ **逐字**；589+ 迭代 1e-17 噪声分叉（r62 口径维持） |
| mfem_ex4_darcy_simple | — | **UNREG** | 未注册死文件、无 exe（D133 在案） |
| mfem_ex5_mixed_darcy | `-m data/star.mesh -no-vis` | **BIT** | vs `d98runbit/cpp_ex5_novis_copy.out` 412 行中 1 diff = `MINRES solver took …s.`（MFEM RealTime，具名豁免） |
| mfem_ex6_flux_recovery | 无参 -no-vis | **BIT** | vs `ref/ex6.out` 131 行中 2 difflines = mesh 行（r93 全档 BIT 本轮复证 ✓） |
| mfem_ex7_surface_poisson | 无参 | **BIT（D1273+D1274 已闭，2026-10-07）** | **整段 stdout 1056 字节与 C++ 逐字节一致**（vs `~/work/rr122/ex7run/ex7_cpp_stdout.txt`，258 unknowns/PCG 轨迹 iter0 1.31756…/19 it/ARF 0.482721/L2 **0.00543013** = C++ 真值 5.43012990936432398e-3）。两步合龙：D1273 = 网格管线逐位（tri6 均匀细分按父代 P2 几何插值，init/r1/r2/r2snap 0/774 位差，钉 d1273 2/2）；D1274 = `SurfaceTri6*Integrator` 逐积分器对齐 MFEM 积分规则（`tri_rule_mfem_order` 转正为 pub 供曲面装配消费；钉 `d1274_surface_tri6_int_rules` 2/2；fem-assembly **1379/0** 零回归）。唯一 Rust 侧差异 = 3 行 stderr 单方回声（Mesh 摘要/Total time/Done.，stdout 无涉；D836 族豁免口径）。历史：D1271（L2 度量统一）/D1272（曲面写器）已闭 2026-10-06 |
| mfem_ex8_dpg_2x2 | `-m data/star.mesh -no-vis` | **NUM** | vs `cpp_ex8_novis_copy.out`：前 20 行逐字，分叉 @iter10（6.65111e-08 vs 6.6511e-08）= D864 三波后 recorded 现值；29 vs 28 it |
| mfem_ex9_dg_advection | 无参 -no-vis | **BIT** | vs `$HOME/work/d76main/ex9_rerun` oracle：stdout 2 difflines = mesh 行；`ex9-init.gf` md5 `21f26d72…` = oracle 逐字节；`ex9-final.gf` maxdiff = **1.0e-8**（r78 D813-3 设计档复现） |
| mfem_ex10_hyperelastic_dyn | 无参 beam-quad | **BIT** | vs `ref/ex10_quad.out` 2 difflines = mesh 行；step100 EE/KE/ΔTE = 0.0119584/0.000784203/−0.0196383 = C++（r92 复证 ✓） |
| mfem_ex14_dg_poisson | `-m data/star.mesh -no-vis` | **BIT** | vs `ref/ex14.out` **2 difflines = mesh 行**（r84 的「C++ 12 行 Options 前缀豁免」已不存在——Rust 现亦打 Options 块，D822-2 类收口已落地） |
| mfem_ex15_dynamic_amr | `-m data/star-hilbert.mesh -no-vis` | RO | rc=0 全程 **223 s**（r62 定档 513 s；本轮更快无漂移主张） |
| mfem_ex15_dump_{A_true,T002,flow,it2_coords,p1,p1_it3} | 无参 ×6 | RO ×6 | 全部 rc=0（D633 修复持续钉住；T002 236 s/2.19M 行） |
| mfem_ex16_nonlinear_heat | 无参 | **NUM** | vs `d89main/cpp_ex16.out`：step 线 0 diff、checksum 行 rust=0 命中（Rust 末尾多 16 行 Comparison 块 = 打印差） |
| mfem_ex17_dg_elasticity | 无参 beam-tri | **NUM** | 本轮 C++ oracle 重跑（d84c binary）：**767 it = C++、iter0 303.808 = C++、ARF 0.982095 = C++**（r89 修复维持 ✓）；stdout 不可逐字节（Rust 全轨迹 + 自绘 banner vs C++ SparseMatrix 统计 + 压缩 PCG 打印）。**r135 Lane 8：打印层闭合**——Options 回声 + Assembling 单行 + `print_matrix_info` 库件 + PCG FirstAndLast 全落地，896B vs 884B 仅剩统计块 8 行差 = **D1280 装配语义债**（skip_zeros=1 跳零 + 面等参精确零；数值本体差异，诚实条款不硬凑）；证据 `tmp/rr135lane8/` |
| mfem_ex18_euler | 无参 | **NUM** | `Solution error: 3.930926246117042e-3` = r99 锚（对 C++ 17 位 0.003930926246116611 rel 1.1e-13）；`d822` pin **4/4 绿** |
| mfem_ex19_hyperelastic_incomp | 默认（= C++ beam-tet，r135 起） | **RO（D1256 已闭 r135）** | r117 rc=0 beam-quad 档 Newton 3 it；**r135 默认 mesh 对齐 C++ ex19.cpp:186（beam-quad→beam-tet）**：rerun Newton0 ‖r‖=2.90593 = C++ 逐字、dim(u)=459/dim(p)=36 同；Newton1 起 GMRES 轨迹分叉维持 rr127 求解器结构债 → 维持非 BIT |
| mfem_ex20_symplectic | `-no-vis` | **BIT** | vs `cpp_ex20_novis_copy.out` **0 diff**（r98 双档本轮复证，连豁免都不剩） |
| mfem_ex21_amr_elasticity | 无参 beam-tri | **NUM** | vs `d89main/cpp_ex21.out`：dofs/误差行全同；C++ 打每 AMR 轮全 PCG 史、Rust 压缩（打印差 42 行） |
| mfem_ex22_complex_helmholtz | 无参 inline-quad | **NUM** | Re/Im 误差行 `1.422826e-1 / 1.422741e-1` = r84 锚逐字 |
| mfem_ex23_wave_equation | 无参 star | **NUM** | vs `d89main/cpp_ex23.out`：Rust 多 2 行 checksum（1.889987e4 / 2.955010e4，r61 记录 1.889982e4 = 第 6 位噪声级）；其余数值行全同 |
| mfem_ex24_discrete_ops | `-m data/star.mesh -p 0 -o 1 -no-vis` | **BIT** | vs `ref/ex24_p0o1.out` 2 difflines = mesh 行（四口径 BIT 的本轮复证档） |
| mfem_ex25_pml_maxwell | 无参 | **NUM** | C++ oracle（d89main）为**截断快照**（止于 iter166）；Rust 终值 ‖E‖=3.529612e-2、求解完成；打印族不同（`‖B r‖` vs `(B r,r)`） |
| mfem_ex26_geom_mg | 无参 star | **NUM** | ARF **0.0273569** = r70 快查锚逐字（机器精度级吻合口径，Chebyshev ulp 分叉族） |
| mfem_ex27_robin_bc | `-no-vis` | **BIT** | vs `d771/cpp_default.gold` **0 diff**（r73 全逐字节本轮复证） |
| mfem_ex28_sliding_elasticity | 无参 | RO | rc=0；与 r61 记录 **0 diff** |
| mfem_ex29_curved_poisson | `-no-vis` | **BIT** | vs `cpp_ex29_novis_copy.out` **0 diff**（r98 双档复证） |
| mfem_ex30_aniso_amr | 无参 | RO | rc=0；与 r61 记录仅 1 行 `Total time`（wall-clock 豁免类） |
| mfem_ex31_anisotropic_maxwell | `-m data/inline-quad.mesh -r 2 -o 1 -no-vis` | **BIT** | vs `ref/ex31.out` 2 difflines = mesh 行（1-D 三档/o2-3 解锁 = r104/r106 pins 叙述，本轮未逐一重跑） |
| mfem_ex31_dump | `-m data/inline-quad.mesh -r 2 -o 1` | RO | rc=0（dump harness，落盘 rust_*.txt） |
| mfem_ex33_fractional_diffusion | `-m data/star.mesh -no-vis`（= r135 起的默认档） | **BIT（r91 档 + r135 默认档，D1258 已闭）** | r117 vs `d89c/cpp_ex33_fresh.out` 2 difflines = mesh 行 ✓。**r135 默认值对齐 C++ ex33.cpp:96-101**（order 1/refs 3/alpha 0.5/vis true；旧 Rust-only `-o 2 --alpha 0.33` = 1:1 偏差已除）后，默认档 2599 字节 vs fresh oracle（`$HOME/work/rr135/ex33`）**cmp 逐字节全等**（0 豁免，sha256 9f9ce65f…） |
| mfem_ex34_magnetostatics | 无参 fichera-mixed | RO | rc=0、ARF 0.905223；r70「leg1 与 C++ 逐位」无 committed 快照可复证（证据在历史 tmp） |
| mfem_ex36_obstacle | `-no-vis` | **BIT（r135）** | r117 vs d89main：12 条 error/bounds 行 0 diff、全 stdout 残 23 行（= 4 类打印差）。**r135 修复**（手写 %g 克隆 `cpp_6` 单数位指数 → 库件 `fmt_g`；`Newton_update_size` 行两侧同由 vis 门控，跑 `-no-vis` 即对齐）后 1822 字节 vs fresh oracle **cmp 逐字节全等**（0 豁免，sha256 c0942ea7…；新鲜 C++ 与 d89main 快照逐字节同） |
| mfem_ex37_topology_optimization | 无参 | **NUM** | Final step 33 / compliance 0.003875 = r84 锚逐字；step 表 vs d89main oracle 有打印差（261 行，数值趋势同） |
| mfem_ex38_implicit_integration | 无参 surface2d | **NUM** | Surface 误差 `5.4492670572e-5` = r61 锚逐字（d89main cpp_ex38 为死 oracle：构建缺 LAPACK/ALGOIM，本轮留注） |
| mfem_ex39_compass | `-m data/compass.msh` | **BIT（r135）** | r117 rc=0 与 r61 记录 0 diff。**r135 首次真对拍**（d89main oracle 系 compass abort 死件，D1259 确认）：bool 回声补 `--visualization` 行后 16336 字节 vs fresh oracle（`$HOME/work/rr135/ex39`）**cmp 逐字节全等**（371 行 PCG 轨迹 + ARF 0.963046，0 豁免，sha256 07885bba…） |
| mfem_ex40_eikonal | 无参 star | **NUM** | 终值 `0.026921483748678143`（二连跑确定性 ✓）vs r84 锚 `…83748254076` 差 3.5e-13 相对（13 位以下漂移，C++ 6 位锚 0.0269214 不变） |
| mfem_ex41_imex | `-m data/periodic-square.mesh -no-vis` | **BIT（r135）** | r117 vs `d89main/cpp_ex41.out`：step/time 网格全同、残差仅 Rust 独有 `‖u‖/sum` 诊断列。**r135 删诊断列**（C++ 只打 `time step: ti, time: t`，ex41.cpp:518）后 849 字节 vs fresh oracle（`$HOME/work/rr135/ex41`）**cmp 逐字节全等**（0 豁免，sha256 b962e706…） |

小计（r117 快照原值，**r135 后现状见括注**）：BIT 14（…含 ex10——**r131 计数链对 ex10 双计，
r135 勘误后 r117 真值仍为 14 件 distinct**；**r135 后 = 22 件**：+ex0/ex36/ex39/ex41、
ex33 默认档升格）、
**NUM 15**（…ex36/ex41 已迁出 → r135 后 NUM 13）、
**RO 14**（ex0/ex7[已闭]/…/ex39 已迁出 → r135 后 RO 11）、**DEV 0 / CRASH 0**、
**UNREG 1**（ex4_darcy_simple）。44 实跑 rc=0 率 100%（ex9 首跑 panic 系批跑先于 `../data`
junction 建立，建链后复跑 rc=0 并 BIT）。

> **round-135 五债重审批注**：D1255 **已闭**（rr123 ex7 stdout BIT，2026-10-07）；
> D1256 **已闭**（r135 ex19 默认 mesh → beam-tet，Newton0 = C++ 2.90593 逐字）；
> D1257 **仍开**（无 committed oracle，miniapps/并行域排单）；D1258 **已闭**（r135 ex33
> 默认值对齐，默认档即 BIT 档）；D1259 **已处置**（cpp_ex38/cpp_ex39/cpp_ex25 三件加
> `SUPERSEDED-DEAD-ORACLE` 头 + ex17 替代物指认 `tmp/r117/cpp_ex17.out`；
> **cpp_ex36 从死件名单除名**——r135 实证 d89main 快照 = 新鲜跑逐字节同）。
> 详见 examples_ledger.md round-135 增量节。

## §3 examples 并行（40，Windows native launcher，无参默认档）

| 例子 | 三态 | 本轮证据 |
|---|---|---|
| pex0/pex2/pex4/pex7/pex8/pex10/pex11/pex12/pex13/pex17/pex18/pex20/pex21/pex22/pex24/pex25/pex26/pex28/pex29/pex31/pex33/pex35/pex36/pex37/pex39/pex41 | RO ×26 | 全部 rc=0（日志 `tmp/r117/logs/`）；无 committed oracle，历史定档（r61 RUN）无本轮数值锚变动迹象 |
| mfem_pex1_parallel_poisson | RO | rc=0（158 s；r61 记录 240 s 内，同量级） |
| mfem_pex3_maxwell_cavity | **NUM** | dofs 10400 = r61 记录；**PCG 95 it = r78 四锚点之一**；L2 2.70053057746141e-2（r67 np2 记录 2.70053055702279e-2，8 位级一致；D746 口径=锚随档位） |
| mfem_pex5_hdiv_darcy | RO ⚠ 观察 | rc=0；MINRES **73 it/9.814e-7** vs r70 自锚 95 it/2.91889e-5 —— **自记录漂移**（AMG 结构性豁免 D654 在案，非 C++ parity 破坏）→ 观察 D1257 附注 |
| mfem_pex6_parallel_amr | RO | rc=0（final unknowns 123036 / PCG 739） |
| mfem_pex9_parallel_dg_advection | RO | rc=0（r76/r78 mass 锚 3.638348e2 未在本轮日志复现——档位注记，无回潮迹象） |
| mfem_pex14_parallel_dg_poisson | RO | rc=0（46 s） |
| mfem_pex27_parallel_robin_bc | RO | rc=0 默认档；r84 观察档（ranks2 PCG 26→38 it）本轮 spot 复跑见 `*_r2.log` |
| mfem_pex15_parallel_dynamic_amr | RUN-LONG | rc=124 @700 s（默认档 star-hilbert；r62 定档=长跑非挂死，本轮维持） |
| mfem_pex16_parallel_nonlinear_heat | RO | rc=0（ranks2 spot 见 rc 文件） |
| mfem_pex19_parallel_incomp_hyperelastic | **DEV** | rc=3 文档化裁剪（2-D 分支 only，D137）兑现 ✓ |
| mfem_pex30_amr_preprocess | RUN-LONG | rc=124 @300 s（r61 以来同族，已知长跑） |
| mfem_pex32_maxwell_eigenvalue | **NUM** | **r106 解锁后首次默认档 rc=0**（r63 的 DEV 裁剪已除，D996-999 关闭兑现）：λ 谱打印正常（λ₁ 3.79466047973361…）；谱级 parity 锚 = r106「ex32/pex32 与 C++ 逐模 ≤2e-12、fichera o1/np1 λ₁ Δ=2.0e-14」 |
| mfem_pex34_magnetostatics | RO | rc=0 ranks1；ranks2 spot：D756 发散档未重钉（rc 文件在案） |
| mfem_pex40_eikonal | RO | rc=0（r64 D655 页脚口径维持） |

（ranks2 spot 探针：pex3/pex16/pex27/pex34/pex40 各一档，日志 `*_r2.log`。**勘误**：批脚本首跑
用错 exe 名（5×rc=127），已用正确名重跑 **5/5 rc=0**。锚点：pex3_r2 = 95 it /
2.70053057746141e-2（与默认档逐字同 = 本机默认即 ranks 2）；**pex16_r2 chksum = 1.618583e7
= r84 观察值逐字**（r84 的「漂移方向标」已稳定为 ranks2 锚）；pex27_r2 平均值行/相对误差
0.082736/0.0246717 正常。）

## §4 miniapps（92 实体，按目录归并；档位=本轮标准命令）

### BIT（逐字节，本轮 fresh 复证 ✓）
| 文件 | 档位 | 对照物 | 结果 |
|---|---|---|---|
| nurbs/nurbs_ex1 | `-m data/beam-hex-nurbs.mesh -no-vis` | `ref/cpp_nurbs_ex1.out` | 剥 C++ 30 行头后 **0 diff** |
| nurbs/nurbs_ex3 | `-m data/square-nurbs.mesh -no-vis` | `ref/cpp_nurbs_ex3.out` | 剥 13 行 Options 头后 **0 diff**（165 it/ARF 0.918732/L2 8.41665e-06） |
| nurbs/nurbs_ex24 | `-r 1 -p 0 -no-vis` | `ref/cpp_nurbs_ex24.out` | banner+L2 行 **0 diff**（BIT\*：PCG 尾段 ulp = D531 族维持） |
| nurbs/nurbs_ex5 | 无参 -no-vis | r63 记录（现编同源） | MINRES **462 it**、‖r‖_B 4.61012e-09、u-err 8.31926e-8、p-err 1.16650e-7 = 记录**全同** |
| nurbs/nurbs_printfunc | 无参 | `ref/cpp_printfunc.out` | **0 diff** |
| nurbs/nurbs_curveint | `-uw -n 9` | `ref/cpp_curveint_uw9.out` | 剥 h_min/h_max/kappa 4 行 + 2 注记行后 **0 diff** |
| nurbs/nurbs_surface | `-ex 1` | `ref/cpp_nurbs_surface_ex1.out` | 2 difflines = vis 回声行（`--visualization` vs `--no-visualization`，oracle 以 -no-vis 捕获） |
| meshing/mesh_hpref | `-m data/inline-quad.mesh -pref -n 100 -no-vis` | `ref/cpp_hpref_pref.out` | 2 difflines = mesh 路径行 |
| meshing/mesh_extruder | `-m data/inline-quad.mesh -nz 4 -hz 2.0` | `ref/cpp_extruder.out` | 5 difflines = mesh 行 + vis 行 + Wrote trailer（r63 豁免类全同） |
| meshing/mesh_toroid | `-o 1` | `ref/toroid-wedge-o1-s0.mesh` | mesh 头 24 行拓扑 **0 diff**（坐标 17 vs 8 位 = 记录豁免） |
| meshing/mesh_twist | `-o 1 -no-pm` | `ref/cpp_twist-hex-o1-s2-c.mesh` | 拓扑行 **0 diff**（同上坐标豁免）；默认 -o3 档 rc=0 |
| gslib/gslib_field_interp | `-m1 square01.mesh -m2 data/star.mesh -no-vis` | round-32 SHA 记录 | `interpolated.gf` sha256 **9f39ae2e8cfd18a4… = 记录逐字节** |
| gslib/gslib_field_diff | 无参（triple-pt 夹具） | r63 记录值 | Max **1.43502** / Avg **0.0949062** = 记录逐字 |
| tools/tools_get_values | 现编 C++ ex5 DC 路线 | r63 记录 | r63 BIT 记录（`0.790403`/`0.110318`）维持；本轮夹具 = 旧 NURBS 场 DC → nan（**r63 已登记的库侧限制**非回潮）→ 本轮归 RO（重升 BIT 需现编 C++ ex5 重产 DC，见 §7） |
| toys/toys_mandel + toys_mondrian + toys_lissajous | pin 档 | `d106_toys_joule_byteparity` | **5/5 PASSED at HEAD** |
| electromagnetics/miniapp_joule | pin 档（stdout + visit DC） | 同上 | **PASSED at HEAD** |
| electromagnetics/miniapp_tesla | `-bm o2` 档 ×2 | `d106_tesla_d957_ams_parity` | **2/2 PASSED at HEAD**（D1230 EXACT 8/7 维持） |

### NUM（数值带：决定性数值行 = committed oracle/记录逐字）
| 文件 | 证据 |
|---|---|
| multidomain_nd（`-tf 0.005 -dt 1e-5 -vs 10`） | step250 cyl **sum=6.932270e-5 ssq=1.594225e-4** = D749 红线**逐字** |
| multidomain_rt（同上 + alt `-tf 0.0002 -vs 2`） | step250 cyl **sum=−2.137667e-4 ssq=1.439350e-6** = 红线**逐字**；alt rc=0 |
| multidomain（H1，t005 档） | rc=0；H1 红线为 harness 档（r70 D737），本轮 t005 行自洽 |
| dpg/dpg_maxwell_3d | `0 \| 156 \| 6.283 \| 1.723e0` = round-17 结案值**逐字** |
| dpg/pdiffusion | `0 \| 113 \| 1.021e+00 \| 9.951e-01` = README 记录**逐字** |
| dpg/pacoustics | `0 \| 113 \| 2.0π \| 8.008e-01 \| 1.374e+00` = 记录**逐字**（CG 列 38 vs 记录 36 = 栈差列） |
| dpg/pconvection_diffusion | rc=0（**r105 D963 后 DEV 裁剪已除**）：`113→275 (6.838e-01/6.187e-01)` = r105 记录**逐字** |
| dpg/dpg_helmholtz_1d / dpg_poisson_2d / dpg_acoustics_2d / dpg_maxwell_2d / dpg_acoustics_3d | L2 4.013e-2；12 dof/PCG 23；`12\|1.222e0`；`33\|1.381e0`；`95\|1.212e0` = r63 记录全同 |
| fluids/navier_kovasznay | err 行 `6.57566E-07` = r63 记录逐字 |
| fluids/navier_mms | 默认档 rc=0 收敛（7.75e-9/1.02e-6）；README gear `2.75455E-08` 记录维持（本轮未重跑该 gear） |
| fluids/navier_bifurcation | 默认档 = PRES 200-it 停滞（D658 裁定的无 hypre 固有行为，非回潮）→ 本轮 600 s 预算内未完（RUN-LONG）；**`-pc amg` 档 PRES 27 it/HELM 14 it 正常收敛**（D658 交付维持），400 s 跑 201 步（~2 s/步正常推进；`-tf` 终点远于预算 → RUN-LONG 诚实标注，`mini_navier_bif_amg_full.log`）。**勘误**：r63 台账的 mini_navier_bifurc.log 实为 6 行 mesh-missing panic（D633 时代）——本轮是该文件首次带全资产的全量跑 |
| solvers/plor_solvers | `star -o 3`：PCG **48 it** / L2 **2.502523e-5** = README 记录逐字 |
| solvers/block_solvers | rc=0，u/p 误差行同量级记录 |
| adjoint/adjoint_cvodes_roberts | BDF 392/194/198 + checkpoint 5604 = 记录**全同** |
| adjoint/adjoint_advection_diffusion | dG/dp2 rel **1.15e-7** = 记录逐字，PASS |
| hooke/hooke | ‖U‖ 9.5635044634017119e-2 = 记录逐字 |
| dfem/dfem_minimal_surface | final ‖r‖ 1.7555e-17 = 记录 1.76e-17 |
| diag-smoothers/diag_abs_l1_jacobi | L2 **0.00143671** = r61 记录逐字 |
| tools/tmop_metric_magnitude、tools_display_basis、tools_lor_transfer、tools/tools_nodal_transfer | rc=0 记录档复现 |
| meshing/mesh_quality | 数值 0.0625/0.0625/90 = 记录（打印格式漂移注记维持，不立案） |
| meshing/mesh_phpref / mesh_ref321 / mesh_fit_node_position | 数值记录级（phpref 前表逐字/求解 ulp；fit-node Newton 不收敛 = r61 记录行为） |
| meshing/mesh_optimizer / mesh_trimmer(.mesh) / mesh_bounding_boxes(nurbs 档) / meshing/{mobius,klein,mesh_explorer} | rc=0；trimmer 默认档 **48 elements/36 nodes = C++** |
| spde/spde_generate_random_field、shifted/shifted_extrapolate | rc=0；extrapolate L2 **0.8735** = 记录逐字 |
| tools/compare-dc、tools/load-dc | compare-dc `-r0/-r1` 自比对 \|pressure\|=60.955/diff=0 = r63 记录逐字；load-dc rc=1 localhost = C++ 同款行为 |

### RO / DEV / RUN-LONG（REPORT-ONLY）
- **RO**：gslib/{findpts×2 档, schwarz_ex1(94 it 推进，r61 95 it 记录档)}、toys/automata、toys/life（RUN-LONG 无界档，C++ 同）、
  toys/autodiff_example、solvers/lor_elast、lor_solvers `-fe l`（见 DEV）、
  nurbs/{nurbs_mesh_info,nurbs_patch_ex1}（r61 BIT 记录档，本轮 rc=0 复跑；netlib oracle 无 committed 快照）、
  nurbs/nurbs_naca_cmesh、diag-smoothers/diag_mg_abs_l1_jacobi、spde 档、tools/gridfunction_bounds、
  electromagnetics/miniapp_maxwell `-no-vis`（NURBS 默认 = rc=3 声明；记录档 RUN 维持，MPI-only oracle 无法串行复核）。
- **DEV（exit(3) 声明全部兑现，零回潮）**：miniapp_lorentz、miniapp_volta 默认档、nurbs_ex10、mini_nurbs_solenoidal(-m 档 rc=3)、
  mesh_polar_nc、tmop_check_metric、navier_cht、mesh_reflector 默认 NURBS 档(rc=3)、hdiv_grad_div `-ams/-lor/-hb`、
  miniapp_lor_solvers `-fe l`、pex19、pex32（examples 侧）。
- **RUN-LONG**：navier_3dfoc（240s 超预算，推进正常）、navier_turbchan `-o 1`（240s ≈ 1-2 步，~140 s/步族）、toys/life（无界）。
- **观察（本轮发现，未立案或入 D1257 附注）**：
  - `shifted/shifted_distance`：L1 0.0081437 vs r63 记录 0.00278（Linf 0.01943 vs 0.02035 同量级）；
  - `shifted/shifted_diffusion`：GMRES 29 it vs 记录 23 it —— 两项 = **默认档自记录漂移，无 committed oracle 可仲裁** → **D1257**；
  - `fluids/navier_bifurcation`：**r63 台账的 mini_navier_bifurc.log 实为 6 行 mesh-missing panic**（D633 时代）——
    本轮是该文件首次带全资产的全量跑：默认档 = PRES 200-it 停滞（D658 已裁定固有行为，非回潮，600 s 预算内未完 = RUN-LONG）；
    `-pc amg` 档 PRES 27 it / HELM 14 it 正常收敛（D658 交付维持），solo 探针 ~1.8 s/步正常推进。

## §5 未注册文件（10）
`mfem_ex4_darcy_simple`（死文件）、`miniapps/fluids/navier_particles.rs`、`miniapps/shifted/sbm_aux.rs`（**r63 后新增**）、
`miniapps/hdiv_linear_solver/hdiv_linear_solver.rs` + `miniapps/dpg/util/pml.rs` + `miniapps/spde/{util,spde_solver,material_metrics,transformation,visualizer}.rs`（模块，宿主覆盖）。

## §6 本轮新债（D1255–D1259）

- **D1255（P1，回归）`mfem_ex7_surface_poisson` 数值漂移**：L2 error 9.4026555405e-3（r61/r84 锚）→
  **1.0904089632e-1**（11.6×）；iter0 1.23622→1.25008、ARF 0.454604→0.493151。本轮现编 C++ 4.10 oracle
  （`$HOME/work/r117/ex7_cpp -e 0 -no-vis`，tri 同 258 dof）：C++ L2 **0.00543013**/ARF 0.482721/iter0 1.31756
  ——HEAD 比 r61 **远离** C++（9.4e-3 → 1.09e-1 vs C++ 5.4e-3）。回归窗口 r84(c5a06dd6)..r116(e4a819c5)。
  快照：`tmp/r117/{cpp_ex7_tri,cpp_ex7_default_quad}.out` + fresh log。修复入口候选：ex7 相关面/边界
  求积或 sphere snap 路径在 r85-r116 的触碰（D771 半面求积族、D805-3、D815-3 等）。
- **D1256（P2，1:1 保真）`mfem_ex19_hyperelastic_incomp` 默认 mesh 偏差**：Rust 默认 `data/beam-quad.mesh`
  ≠ C++ ex19.cpp:186 `../data/beam-tet.mesh`。同档（beam-tet -no-vis）Newton0 逐字同、Newton1 起 GMRES
  轨迹分叉（31it/1.45e-12 vs 21it/1.29e-13）。
- **D1257（P3，观察账）shifted/ 自记录漂移 + pex5 迭代漂移**：shifted_distance L1 3×（0.00278→0.008144）、
  shifted_diffusion GMRES 23→29 it、pex5 MINRES 95→73 it/9.814e-7 —— 均为**无 committed oracle 的自锚漂移**，
  不定性为回归；登记待 committed oracle 或档位复原能力。
- **D1258（P3，1:1 保真）`mfem_ex33_fractional_diffusion` 默认档偏差**：Rust 默认 `-o 2 --alpha 0.33`
  ≠ C++ ex33.cpp:99/101 `-o 1 -alpha 0.5`（`-o 1 -a 0.5` 档本轮 BIT ✓，故仅默认档入口偏差）。
- **D1259（P3，证据卫生）死/缺 oracle 快照**：`tmp/d89main/cpp_ex38.out`（LAPACK/ALGOIM abort 输出）、
  `cpp_ex39.out`（compass.msh 缺失 abort）、`cpp_ex25.out`（截断 @iter166）、`ref/ex17_cpp_default.out`
  **缺提交**（r84 声称留档但 ref/ 无此文件；本轮以 d84c binary 重跑补证 `tmp/r117/cpp_ex17.out`）——
  四件 committed oracle 不可用作逐字节对照，使用时必须重新生成。

## §7 remaining（诚实未竟）
- ex31 的 1-D 三档 / `-o 2/3` 15 案例（r104/r106）本轮未逐一重跑（其 pins 叙述在 coverage_matrix §5 行，证据 `tmp/d104/`）。
- 并行 40 例中 26 例仅 rc=0 + 历史定档核对（无 committed oracle 可对照；WSL mpirun 档属于 r105/r106 已钉域）。
- multidomain H1 曲边档（D737「曲边档待复跑」）本轮未重跑；rt/nd 长窗口（t05/20001-step）本轮未重跑（红线四值已逐字）。
- maxwell.rs（MPI-only）记录档仍无本机第三方复核路径（r63 注记维持）。
- `get_values` BIT 路线需现编 C++ ex5 DC 再生成（本轮夹具为旧 NURBS DC → nan = 已登记库侧限制）。
