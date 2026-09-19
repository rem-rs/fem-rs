# miniapps

对应 MFEM `miniapps/` 目录，按子目录组织。

## 目录结构

```
miniapps/
├── tools/                   ← 对应 miniapps/tools/（**round 30 审计 → round 32 全部重测处置**：
│                                30 轮的 (e) 类结论都是枚举 `target/release/examples/*.exe` 得到的
│                                （陈旧二进制污染）⇒ 32 轮用 `cargo run --release --example`
│                                在 MFEM 4.10 下逐件重测（11/11 二进制新鲜性已核），四件做到与
│                                C++ **逐字节相同**，其余改为 `exit(3)` + 缺口清单）
│   ├── compare-dc.rs          ← ✅ **round 32：与 C++ 逐字节相同**（27 行 `diff` 无输出）。
│   │                            修前是三重缺陷：字段遍历用 `HashMap`（C++ `std::map` 字典序，
│   │                            顺序逐次不同）、分隔线 15 个短横（C++ `setw(15+2*len)` 里塞
│   │                            字面量 `" = "` ⇒ `15+2*len-3` 个）、多打一行 `Compare complete.`；
│   │                            **还漏了一条功能性失败**：`read_visit_root` 填的
│   │                            `DcField::values` 是空 vec ⇒ 所有范数打印成 `-0`
│   │                            （C++ 是 `|pressure_0| = 114.455`）。现用
│   │                            `load_visit_collection`（真读 `*.000000` 切片）+ `BTreeMap` +
│   │                            正确的分隔线 + `fmt_g` 6 位有效数字
│   ├── get_values.rs          ← ✅ **round 32：官方 2-D 样例与 C++ 逐字节相同**
│   │                            （`-r Example5 -p "0.5 0.5 0.1 0.1" -fn pressure` →
│   │                            `0.790403`/`0.110318`；`-o <file>` 写文件路径也已核对）。
│   │                            修前：`data_collection_load.rs:101` 硬编码 `mesh3d` ⇒ 2-D 集合
│   │                            直接 `MissingMesh`/exit 1（而 C++ 官方样例 Example5 就是 2-D），
│   │                            且 `-o` 被 `let _ = …` 吞掉。现按 `read_mfem` 分派 `Mesh<2>`/
│   │                            `Mesh<3>`，实现 `-o`（banner 留 stdout、legend+数据进文件）。
│   │                            ⚠️ **仍缺**：3-D 集合的 **ND/RT 分量**与 C++ 不符
│   │                            （hex `velocity` `0.0553179 …` vs C++ `-0.694191 …`；tet 连
│   │                            pressure 也不同）⇒ 这是 fem-rs 3-D H(div)/H(curl) 求值的
│   │                            库侧缺口，非 miniapp 问题
│   ├── load-dc.rs             ← ✅ **round 32：两种档都逐字节相同**（C++ 该程序 stdout 只有
│   │                            `Options used:` banner + `fields: [ pressure, velocity ]`；
│   │                            打开可视化而无 GLVis 服务时打 `Connection to localhost:19916
│   │                            failed.` 并 rc=1 —— 两档均已实测对齐）。修前自创 6 行
│   │                            （`Collection Name:`/`Space Dimension: 3 (assumed)`/`Field
│   │                            details:`/`Load complete.`）C++ 从不打印
│   ├── tmop_check_metric.rs   ← ⚠️ **round 32：改写为 C++ 的 `-mid N` 接口 + 声明式 exit(3)**。
│   │                            id zoo 已按 C++ 源码**实测枚举**（`grep -E "^ *case [0-9]+:"` =
│   │                            40 个 **未注释** case；`211/252/311/352` 在 MFEM 源码里是
│   │                            `// case …` 注释行，**不算**）；未知 id 打印 C++ 原文
│   │                            `Unknown metric_id: <id>` 并 `exit(3)`（C++ `default:` 就是
│   │                            `return 3`，已实测 rc=3）。**仍缺**：C++ 走 mesh/FE 空间的
│   │                            `TMOP_Integrator` + **解析** `AssembleElementVector/Grad` 求
│   │                            EvalP/AssembleH 收敛阶，而 `fem_mesh::tmop` 只有简化元素能量 +
│   │                            有限差分梯度/Hessian ⇒ 21/40 id 可算但无真值，故不跑检查、
│   │                            打印缺口清单后 `exit(3)`（旧的固定 21 项自检是**另一个程序**，
│   │                            且 2-D 恒报 `0/0`）。**未知命令行选项**已按 C++ `OptionsParser`
│   │                            的 `Unrecognized option: <opt>` + **exit 1** 处理（旧版 `_ => {}`
│   │                            会静默丢掉，例如把 `-par` 误写 `-pfa` 会按默认档跑完）
│   ├── tmop_metric_magnitude.rs ← ⚠️ **round 32：id zoo 与出口码对齐 C++，声明式 exit(3)**。
│   │                            C++ 未注释 case 实测 **25 个**；`fem_mesh::tmop` 缺其中
│   │                            `85/98/322`（T-metric）与 `11/36/107`（A-metric）⇒ 这些 id
│   │                            打 C++ 原文 `Unknown metric_id` + rc 3，并在 stderr 说明
│   │                            "C++ 接受它、只是 fem-rs 没实现"；而 C++ 也不认的 id
│   │                            （`999`、注释行里的 `211`）则明确注明"C++ 同样不认"。
│   │                            输出行改用 `fem_solver::fmt_g`（C++ 6 位有效数字）；**未知命令行
│   │                            选项**同样按 `OptionsParser` 打 `Unrecognized option:` + exit 1
│   ├── gridfunction_bounds.rs ← ✅ **round 41-42（D159+D255–D259）：完全体**——
│   │                            `GetElementBounds(…, ref)` 与
│   │                            `EstimateFunctionMinimum/Maximum` 按 `fem/bounds.cpp`
│   │                            全文件逐行移植（PLBound 已晋升
│   │                            `fem_assembly::postproc::plbound`；GL/GLL 节点 =
│   │                            MFEM [0,1] 直接 Newton 的位级端口）；vs C++ MPI 4.10
│   │                            np1 **33/33 场景逐行逐位**（default/-nb/-ref/-bt 1/
│   │                            -l2 -bt 0/1/-visit 含曲面/vdim2/无参）。
│   │                            `exit(3)` 仅余 `-bt 2`（正基，D276）、1-D
│   │                            （D277 `Mesh<1>`）、`-bt≥3`/L2→L2 换基（D278）；
│   │                            `-vis` 为文档化 no-op
│   ├── display_basis.rs       ← (b) 基函数展示 (无 GLVis; H1/ND/RT/L2,
│   │                            vsize 与 C++ 逐位一致, 32/34 组合;
│   │                            hex L2 ≥P2 为 fem-rs 缺口; C++ `-no-vis` 只打 4 行 header)
│   ├── get_values.rs          ← （round 32 处置见上；旧结论"实测失败"已作废）
│   └── lor_transfer.rs        ← (b) LOR 传输 (简化 H1+pointwise 版; C++ 默认走
│                                L2ProjectionGridTransfer 而本文件无参时静默按 pointwise
│                                跑; `-h1/-l2/-t/-w/-ea/-d` 未实现 ⇒ `panic!` 而非
│                                `exit(3)`; HO/R(HO)/LOR 质量与 C++ 逐位一致)
├── electromagnetics/        ← 对应 miniapps/electromagnetics/（**round 31 D139 处置 3 件**）
│   ├── lorentz.rs           ← ⚠️ **round 31 声明式缺口 (exit(3))**：C++ 是 VisIt
│   │                            DataCollection 接口 (`-er/-ef/-ec/-epdc/-epdr/-br/-bf/
│   │                            -bc/-bpdc/-bpdr/-rdf/-rdm/-o/-npt/-m/-q/-xmin/-xmax/
│   │                            -pmin/-pmax/-dt/-nt/-vis/-vt/-vf/-d`)，本文件自造
│   │                            `-ex/-emesh/-efield` CLI 且曾**静默忽略 C++ 选项后 rc=0**
│   │                            （实测 `-er Volta-AMR-Parallel -br Tesla-AMR-Parallel
│   │                            -npt 20 -nt 5` → rc=0、`E=(0,0,1), B=(0,0,0) [constant]`）
│   │                            ⇒ 现任何输入 `exit(3)` + 缺口清单，文件头换成真实 C++
│   │                            选项表 + 偏离说明；缺并行 VisIt DC 读取 / GF 求值 /
│   │                            `-rdf` 重分配（`fem_io::data_collection_load` 存在但未接线）
│   ├── tesla.rs             ← ⚠️ **round 31 声明式缺口 (exit(3))**：stub + 假参数 ——
│   │                            mesh 恒为内置 `unit_cube_tet(2)`、`-m`/`-maxit` 被
│   │                            `let _…` 丢弃、`|A|=|B|=0` 且 `PCG Iterations = 0` 却 rc=0。
│   │                            证据（round 31 实测）：`-m data/beam-tet.mesh -maxit 1
│   │                            -ubbc "0 0 1"` 与无参运行的 dof 计数**完全相同**
│   │                            (H1 27 / H(curl) 98 / H(div) 120 / L2 48)
│   │                            ⇒ 现 `not_ported()` 在 main 第一行、不解析任何参数、
│   │                            `exit(3)` + 缺口清单
│   ├── volta.rs             ← 1:1 到单次解 (`-maxit 1`)：4 空间 + Assemble + Solve +
│   │                            Total charge。**round 31 D139**：`--ranks` 默认
│   │                            **2 → 1**（`>1` 因并行装配 `assert(24≠48)` 原为 panic，
│   │                            现打印根因 + `exit(3)`）；`-vp`/`-nbcs`/AMR(`-maxit>1`)/
│   │                            非 3-D/读文件失败全部 `exit(1)` → **`exit(3)`**。
│   │                            ⚠️ **审计更正**：round 30 记的"hex 全 panic / NURBS 默认
│   │                            网格挂死 >120 s"在当前源码 `--ranks 1 -maxit 1` 下
│   │                            **不复现**（实测 hex / ball-nurbs 均 rc=0 正常跑完）
│   │                            —— 疑似审计跑的是 **8/28 的陈旧 `mfem_miniapp_volta.exe`**
│   │                            （旧注册名；现注册名是 `miniapp_volta`）
│   └── maxwell.rs           ← 全波 Maxwell 偶极子脉冲 (1:1 串行; ND/RT 空间 +
│                                SIAV 辛积分): H(Curl) 12336 / H(Div) 11520 dof
│                                与 C++ 相同, 100 步 40 条 Energy(<t>ns) 行 +
│                                banner + Options dump 与 C++ **逐字节**(主会话
│                                独立复核: 78 行输出仅差 mesh 路径字符串与
│                                Maximum Time Step); dtmax 行因 hypre
│                                Randomize(1234) 种子不可复刻而不同
│                                (0.141749→0.145761ns, 同 SnapTimeStep 档);
│                                -vis/-visit/-cs/-abcs/NURBS/2D 为 exit(3);
│                                SIAV 已于 round 26 下沉到 fem_solver)
│   └── joule.rs             ← 第 5 件 (joule.cpp **部分交付, 退出码 3**,
│                                涡流-热耦合): 1:1 到 dof 横幅 —— banner /
│                                完整 Options dump(含 C++ 的 -p 注册两次) /
│                                两条 skin depth (0.551329, 0.126157) / 四个
│                                空间及阶 / 五行 unknowns (6456/2016/6882/
│                                6456/2443) / true_offset + 6 场 BlockVector +
│                                六个 make_ref 视图 / 四个材料映射 / 三个 BC
│                                掩码 —— 与自编 C++ 4.10 参考**逐行一致**
│                                (主会话独立复核); C++ 在 dof 横幅后立刻进入
│                                hypre 自己的输出 ⇒ 之后无法逐字节
│                                **round 29 推进**：H¹(P2)→ND2 的 3-D hex 离散
│                                梯度已打通（`discrete_op.rs`，两条独立证据：
│                                与 `HCurlSpace::interpolate_vector(∇p)` 逐位一致、
│                                `M1·G == ∫v·∇φ` 弱梯度恒等式 ≤1e-11；也是
│                                joule_solver.cpp:283 注释里的等价路径）。
│                                **仍缺（= 新发现的硬阻塞）**：① `curl_3d`
│                                (ND2→RT1) 只有 tet 版，hex 上 panic（joule 的
│                                `curl→Mult(E,dB)` 正需要它）；② `GetJouleHeating`
│                                的 L² 投影缺"三线性感知"的逐元入口（现有
│                                `evaluate_vector_at_element` 对 Hex8 只做仿射
│                                Jacobian）；③ ≥2 rank 的并行 ND2/RT1 ghost
│                                面/内部 dof 分区缺陷（`ghost.rs:178` panic）；
│                                另有求解器栈（无 AMG/AMS/ADS）与 `.gen`
│                                (netCDF) 读取。端到端目标 = 末两行
│                                `dot(E, J)`（只依赖电磁半块）
├── diag-smoothers/          ← 对应 miniapps/diag-smoothers/
│   ├── abs-l1-jacobi.rs     ← Absolute L(1)-Jacobi 光滑子 (1:1 串行版;
│   │                            mass/diffusion/maxwell 三类系统, SLI/PCG,
│   │                            abs_global + L(p,q) 元素级对角, Kershaw 网格;
│   │                            -a 0/1 下迭代日志/ARF/L2 与 C++ 逐行一致,
│   │                            2D 与 Kershaw 全对比逐位一致;
│   │                            maxwell 3D hex 差 HexNDk 归一化 4×,
│   │                            C++ PARTIAL/NONE 的矩阵免费 AbsMult 为缺口)
│   └── mg_abs_l1_jacobi.rs  ← ✅ **round 48 D376：mg-abs-l1-jacobi 完整 np=1 port**
│                                （此前无对位）。fem-solver 新增 AbsL1 几何多重网格核：
│                                `AbsL1GeometricMultigrid`（ds-common 1:1；VCYCLE/WCYCLE、
│                                `|A|·1` 对角光滑、粗层 SLI/CG + `MG_REL_TOL=√1e-10`/
│                                `MG_MAX_ITER=10`）、`form_fine_linear_system`、
│                                精确 P1 细化延拓（1/0.5/0.25/0.125 二进制常数逐位 =
│                                MFEM RefinementOperator，tri/quad/tet/hex）与
│                                hex ≥2 阶 Newton 嵌套延拓。**验收（ref-cube `-a 0
│                                -rs 3 -gl 1 -ol 1`）与 C++ MPI oracle（mpirun -np 1）
│                                逐位一致**：35937 未知数、20 步 CG 全轨迹
│                                （`154.905 → 3.00892e-19`）、ARF `0.285073`、
│                                L2 `2.66219e-05`；star/beam-quad/beam-tet、SLI、
│                                mass、`-o 2`、`-rp`、`-ol`、monitor CSV 全对齐。
│                                ⚠️ `-s` 同时选粗层求解器（初版硬编码 CG 致 SLI 分叉，
│                                已修）；P1 延拓必须精确二进制常数（barycentric 反求
│                                1-ulp 翻转迭代路径，已修）。残差：star-Kershaw 是
│                                C++ 自身病态（NaN）⇒ **D388**（kershaw_map 不等价）；
│                                `-a≥1` 矩阵自由路径未实现（打印说明走装配）；
│                                **D386**（fem-space 嵌套 3-D 延拓缺 hex，绕过）、
│                                **D387**（ess 行 RHS 约定 DIAG_ONE vs DIAG_KEEP，
│                                ‖b‖ 类诊断不可直接对拍）。
├── nurbs/                   ← 对应 miniapps/nurbs/（**round 30 审计**：15 个可执行中
│                                **4 个真 1:1**（ex1/ex3/ex5/ex24，均有 C++ 对照数字）+
│                                **2 个名不副实待整改**（见下 solenoidal/printfunc）+
│                                **9 个缺失**：ex1p/ex10/ex10p/ex11p/~~curveint~~/mesh_info/
│                                patch_ex1/surface/naca_cmesh；共同地基 = 并行 NURBS、
│                                `Nurbs*Space` 实现 `FESpace`（+vdim）、NURBSPatch 控制网编辑、
│                                v1.1 `spacing` 段、多补丁 `knotvectors` → 逐补丁 `NurbsFile`。
│                                ⚠️ `-pm/-ps/-p` **静默忽略**（实测加不加输出完全相同））
│                                ✅ **round 31 起共同地基已部分落地（D143）**：
│                                `fem_io::nurbs_mesh::{read,write}_nurbs_mesh_doc` —— NURBS 网格
│                                **writer** + **`patches` 变体读写**；11 个 v1.0 夹具 read→write
│                                token 级零差异，8 个与 MFEM 自身 `Mesh::Save(out,16)` **逐字节
│                                相同**（`square-disc-nurbs-patch.mesh` 的 5 个 patch 经 MFEM
│                                归一化输出 145 行逐字节相同）。**仍未解锁 ex10/ex11p/surface**：
│                                还缺并行 NURBS 与 `Nurbs*Space` 实现 `FESpace`
│   ├── nurbs_ex1.rs         ← 1:1（H¹ 标量; NurbsFESpace 真 NURBS 空间,
│   │                            4356 dof / ARF 0.588878, 11 配置 9 网格
│   │                            与 C++ 迭代块逐字节一致; **1D 已支持**:
│   │                            segment-nurbs.mesh 默认 = 4097 dof / 200 迭代
│   │                            块 + 非收敛 trailer 与 C++ **逐字节**（主会话
│   │                            独立复核：迭代块 0 处差异；注：本示例不打印
│   │                            C++ 的 Options used 横幅，2D 路径同）;
│   │                            部分属性 ess_bdr = boundary_dofs_marked）
│   │                            ✅ **round 33（D92）：`-pm/-ps/-p` 已实现**
│   │                            （旧版这三个旗标被 `_ => {}` 静默丢弃）。语义与 C++
│   │                            一致：`-pm` = master 边界属性列表、`-ps` = slave 列表，
│   │                            一起喂给 `NURBSExtension::ConnectBoundaries`；
│   │                            `-p <file>` 从文件读两列表（首 token 是数量）；
│   │                            `-p` 在 MFEM 的线性扫描里被 `--send-port` 遮蔽（照抄）。
│   │                            内核新增 `connect_boundaries`（1D/2D/3D 三版 + `d_to_d`
│   │                            压缩 + 重跑 `GenerateElementDofTable`）、
│   │                            `BdrSegDofMap`/`BdrQuadDofMap`、`NurbsFESpace::with_periodic`。
│   │                            主会话独立复核（重编 C++ 4.10 对拍）：
│   │                            `beam-hex-nurbs -pm 1 -ps 2` 5184 unknowns / ARF 0.200293
│   │                            （18 行整块逐字节，sha256 同）、
│   │                            `pipe-nurbs-2d -o 2 -r 1 -pm 1 -ps 3` 13 / ARF 0.0173099
│   │                            （14 行逐字节）；代理另有 20+ 配置矩阵。同批把选项层对齐
│   │                            MFEM（`-n/--neu`、`ess/neu/per_bdr` 修正环、完整
│   │                            `OptionsParser` 行为、`MFEM_VERIFY` 原文 + `exit(134)`）。
│   │                            **仍缺**：① 奇异系统（`ESS n=0`）的 CG trailer 与 MFEM 不同
│   │                            （缺 `Iteration : 0` 行与 indefinite 诊断）⇒ 1-D 周期档与
│   │                            `pipe-2d -p <both pairs>` 第 3 步后分叉（**空间与消元后的
│   │                            矩阵/RHS 已 17 位精度钉住**，夹具
│   │                            `crates/space/tests/data/nurbs_periodic_mfem.txt`）；②
│   │                            `generate_boundary_elements` 对 `pipe-nurbs.mesh` 给 4 个边界属性
│   │                            而 MFEM 是 1（D169/D170）；③ `square-disc-nurbs-patch.mesh`
│   │                            仍无法解析（D143 残留）
│   ├── nurbs_ex3.rs         ← 1:1（H(curl): NurbsHCurlSpace 分组件 curl
│   │                            扩展 + 合并 elem_dof + Piola 装配 +
│   │                            ProjectCoefficientElementL2 默认投影 =
│   │                            L2 单元局部投影 + LSQ 映回 + ./= Va;
│   │                            默认 -o 1 = 33540 dof / ess 516 /
│   │                            166 迭代块逐字节 / ARF 0.918732 / L2
│   │                            8.41665e-06 全部 = C++; -r 1 = 24 dof /
│   │                            10 迭代块逐字节 / ARF 0.128859 / L2
│   │                            0.0508853 亦 = C++（三档均与 C++ 二进制
│   │                            逐字节；未处理 remaining: sol.gf/refined.mesh））
│   ├── nurbs_ex5.rs         ← 1:1 **完整移植**（NURBS 版 mixed Darcy:
│   │                            H(div)×H¹ + 自然 BC 的边界通量 RHS +
│   │                            BlockDiagonalPreconditioner(M: DSmoother,
│   │                            S: GSSmoother on S = B·diag(M)⁻¹Bᵀ) +
│   │                            MINRES）: 默认档 8580/4225/12805 dof、
│   │                            边界 dof 260/256、**462 迭代**、
│   │                            ||r||_B 4.61014e-09、两个误差范数
│   │                            (8.31927e-08 / 1.1665e-07) 与自编 C++ 4.10
│   │                            参考**全部一致**（残差序列在 C++ 的 6 位
│   │                            量化内一致至第 259 迭代，其后 ~4e-6 = 装配
│   │                            S/GSSmoother 的浮点路径差异）; 唯一偏差:
│   │                            BdpMinresSolver 不暴露 GetFinalNorm ⇒ 摘要
│   │                            行只印迭代数
│   └── nurbs_ex24.rs        ← **1:1 完整移植（round 29）**: `-r 1 -p 0/1/2`
│                                三档的 dof 横幅 + 两条 L² 误差行（0.0224956/
│                                0.0039157、0.488496/0.51453、0.00271413/
│                                0.00260626）与**全新编译**的 C++ 4.10 参考
│                                **逐字节一致**（主会话独立复核，C++ 侧
│                                `$HOME/work/nurbs_ex24_ser/nex24`）；round 29
│                                补齐 NURBS 跨空间 `MixedVectorGradient`/
│                                `MixedVectorCurl`（D98 结案）+ `-p 2` 的标量
│                                单元 L² 投影。**唯一非逐字节处**：PCG 迭代块
│                                （`(B r,r)` 到 ~1e-11 后分叉；真因 = MFEM 对
│                                **插结后**控制网求几何、fem-rs 对**原**控制网在
│                                细化参数区间求几何，同一映射差 ~1 ulp/entry）
│                                仍缺 `refined.mesh`/`sol.gf` + GLVis（步骤
│                                12–13），`-nn` 仍 exit(3)
│   ├── nurbs_solenoidal.rs  ← ⚠️ **round 32：头注释不再自称 "1:1 port"，改为声明式
│   │                            `exit(3)` + 缺口清单**（C++ 4.10 实测参考：
│   │                            NURBS 默认档 dim(R)=8580/dim(W)=4225、MINRES 335 次、
│   │                            `‖u_h−u_ex‖=2.08242e-05`、`‖div‖=1.4113e-13`；`-nn` 档
│   │                            33024/16384、440 次、2.08198e-05、3.55911e-13）。五条缺口：
│   │                            ① 默认 NURBS 档需要 `NURBS_HDivFECollection`/
│   │                            `NURBSExtension`/`NURBSFECollection`，fem-rs 没有实现
│   │                            `FESpace` 的 NURBS 空间（D143）⇒ **exit(3)**；② `-nn`
│   │                            档虽能解，但右端用 `M·I(u_ex)` 而非 C++ 的
│   │                            `VectorFEDomainLFIntegrator`（L² 9.55855e-05 vs 2.08198e-05）；
│   │                            ③ `GridFunction::ComputeDivError` 在 `crates/` 0 命中 ⇒ 该行
│   │                            以显式 `unavailable` 标记而非省掉；④ `GS(M)/GS(S)` 装配了
│   │                            但 `MinresSolver::solve` 不收预条件子 ⇒ 迭代数与 C++ 不同；
│   │                            ⑤ `-vis`/`-d`/VisIt/ParaView 输出缺失。已复刻：CLI、
│   │                            banner（含 `****` 框）、`-df/-p` 分派、`ref_levels` 公式、
│   │                            MINRES 容差与 `exsol.mesh`/`sol_u.gf`/`sol_p.gf` 输出
│   ├── nurbs_curveint.rs    ← ✅ **round 48 D374：nurbs-curveint 1:1 port**（此前缺失）。
│   │                            `fem-mesh` 新建 **`NURBSPatch` 控制点对象层**
│   │                            （`crates/mesh/src/nurbs_patch.rs`：定位 `operator()(i,j,l)`
│   │                            布局 `(i+j·ni)·Dim+l`、`DegreeElevate`、`KnotInsert`、
│   │                            按 C++ `%g` 精确 `Print`），并把原先**孤立无消费方**的
│   │                            `NurbsKnotVector`（nurbs_mesh.rs）迁入补全：
│   │                            Demko–Remez、`GetInterpolant`（MFEM 同款 Gauss–Jordan
│   │                            显式求逆 + kernels::Mult）、`Difference`、`%g` 打印 —— 全部带
│   │                            nurbs.cpp 文件:行号引用。**验收**：`-uw -n 9` stdout 与 C++
│   │                            一致（h/kappa 四行因 NurbsExtension 仍拒绝 patches 格式而
│   │                            如实省略）；`sin-fit.mesh` 头 + 两条 knotvector +
│   │                            **143/153 控制点逐字节相同**，正弦插值控制点
│   │                            （GetDemko/GetInterpolant）**逐位一致**（`%.17g` 探针钉入
│   │                            `d374_curveint_patch.rs`）。10 个差异全在物理零残差行/列
│   │                            （~5e-17，根因 = 委派的 `h_refine_uk` 是逐结点 A5.1 而
│   │                            MFEM 一次 A5.5 精确消去）⇒ 顺带发现 **D380**：
│   │                            `fem_element::nurbs::h_refine_vk`（nurbs.rs:1720）按列重建、
│   │                            按行读回，多结点 v 向插结点返回**错乱数据**（已绕过：
│   │                            转置→h_refine_uk→转置回；上游修复待做）。
│   │                            `-visit`（VisIt DC）未移植 exit 说明。
│   └── nurbs_printfunc.rs   ← ✅ **round 32：48 行与 C++ 逐字节相同**（`diff` 无输出）。
│                                数值本来就对（≤1.1e-15），差异纯粹是格式：C++ `std::cout`
│                                默认 `precision(6)`，本文件此前用 Rust 最短往返表示 ⇒
│                                48 行里 28 行文本不同（`0.005` vs `0.005000000000000001`）；
│                                改用库里已有的 `fem_solver::fmt_g` 即逐字节（D131）
├── meshing/                 ← 对应 miniapps/meshing/（**round 30 审计 → round 31 D126/D132 处置**：
│                                22 个可执行中 5 个 a/b 类 + 10 个缺失；round 30 判为 (e) 类的 7 件已
│                                逐件处置。根因（**D126**）= `write_mfem` 边界面回落硬编码 3 节点/
│                                TRIANGLE，**外加整个 writer 用 1-based 顶点索引**（MFEM 4.10 的
│                                `PrintElementWithoutAttr`/`ReadElementWithoutAttr` 两端都直读
│                                **0-based**，`data/` 全部官方网格含顶点 0）⇒ 修前 MFEM 读本仓
│                                任何 `.mesh` 都 `Invalid mesh topology`/堆崩。已修：面类型按
│                                `face_type_at` 推导 + writer 全面 0-based + reader 0/1-based 判据
│                                加固 + **写前一致性自检**（坏网格返回 `FemError`、不落空文件））
│   ├── shaper.rs            ← ⚠️ **仍开（D132 残留，本路未授权）**：曾标"(1:1) MATERIAL 界面
│   │                            AMR"，实测证伪：quad 分支走 `refine_uniform`（整网格均匀细化、
│   │                            忽略 marked 集合）16→64→…→65536 vs C++ 的 16→52→64 NC 网格；
│   │                            材料属性 `attr(i)` 从未写回
│   ├── extruder.rs          ← ✅ **round 31 修复**：`-m data/inline-quad.mesh` → NE=16 NBE=48
│   │                            NV=50、边界段全为 `1 3 <4 节点>`（与 C++ 相同）、探针
│   │                            `NE=16 NBE=48 NV=50 dim=3 sdim=3 nodes=0` **与 C++ 完全相同**、
│   │                            `mesh-explorer` kappa 4/4 全等。`-trans`/1-D 输入/混合 2-D 输入
│   │                            ⇒ exit(3)+缺口清单（缺曲面 nodes 写出 / `Mesh::Extrude1D`）
│   │                            **仍存偏差（未授权改 `crates/mesh/src/extrusion.rs`，见 D144）**：
│   │                            ① `elem_tags_3d.push(0)` 应 `mesh.elem_tags[e]` ⇒ MFEM 警告
│   │                            `Non-positive attributes`；② 边界面属性 1/2/3 vs C++ 的
│   │                            源属性 1..nba 与底/顶 `nba+elem attr`；③ 顶点编号层优先
│   │                            `j*nv+i` vs C++ 点优先 `i*nvz+j`
│   ├── toroid.rs            ← ✅ **round 31 修复**：① `elem_type` 未随 `-e` 同步（把 6 节点
│   │                            prism 按 Hex8 写出 ⇒ round 30 看到的"6 个 CUBE + 空 boundary"）
│   │                            ② 本地复刻 `FinalizeTopology`/`GenerateBoundaryElements`
│   │                            ③ prism 的 `RemoveInternalBoundaries`（`local_face_verts` 无
│   │                            Prism6 分支 ⇒ 原函数对楔形网格不删任何面）④ face_type/
│   │                            face_types/face_offsets 同步 ⑤ 输出名按 C++
│   │                            `toroid-{wedge,hex}-o*-s*[-r*].mesh` ⇒ `-o 1` → NE=8 NBE=24
│   │                            NV=24，与 C++ `toroid -o 1` **拓扑逐字节相同**、坐标差 <5e-9
│   │                            （C++ 只存 8 位有效数字）。**round 33：`-o > 1` 默认档已解锁**
│   │                            （见下）
│   │                            ✅ **round 33（D151）：默认 `-o 3` → `exit 0`**，写出
│   │                            `toroid-wedge-o3-s0.mesh`。内核新增 prism 的 `nodes` 编号
│   │                            （`prism_h1_slots` 逐行复刻 `fem/fe/fe_h1.cpp:863` 的
│   │                            `H1_WedgeElement`：`t_dof`/`s_dof` 表、底/顶三角形面的内部置换、
│   │                            `SegDofOrd`/`TriDofOrd`/canonical 四边形面、**累积式**面块偏移）。
│   │                            主会话独立复核：`r31_meshread` 两侧
│   │                            `NE=8 NBE=24 NV=24 dim=3 sdim=3 nodes=1`；`r32_probe`
│   │                            `FEC=H1_3D_P3 order=3 vdim=3 ndofs=240 nonpositive=0`；
│   │                            `r31_save`→`r32_cmp` **TOPOLOGY-IDENTICAL**、
│   │                            nodes-dofs=720、max-rel-diff **4.44e-16**（不动点）。
│   │                            ⚠️ **round 33 的诚实代价已被 round 34 清偿**：当时楔形档
│   │                            节点值与 C++ 相差 **7.2e-5**（`PrismPk` 等距 vs MFEM GLL 的
│   │                            族分裂）。
│   │                            ✅ **round 34（D164）：`PrismPk` 格点等距→GLL**，writer 的
│   │                            重插值变成恒等 ⇒ 默认 `-o 3` 楔形档对 **17 位精度** C++ 参考
│   │                            达 **8.12e-15**（`-o 2` = 2.7e-16、`-o 4` = 6.3e-14、hex 无回归
│   │                            2.8e-16；torus 体积变为 C++ 值 0.326483）。**槽序有意保持
│   │                            layer-major**（MFEM 的实体序由场空间侧新元素 `H1PrismPk`
│   │                            承担，D168；dof_manager 的 p=2 prism 布局顺带修复——旧
│   │                            `build_p2_prism` 把三形面 dof 覆写到边槽、16/17 槽别名到顶点 0）。
│   │                            **round 35：toroid 全部门已开**——`-e 0 -dm -o>1` 由 D165
│   │                            解锁（关键发现：`L2_T1` 的 **T1 = BasisType::GaussLobatto**，
│   │                            不连续 wedge 点**也是 GLL 而非等距** ⇒ 纯置换；`-dm -o 3` =
│   │                            `L2_T1_3D_P3`、TOPOLOGY-IDENTICAL、**3.87e-08**）；wedge 的
│   │                            `-e 0 -rs>0 -o>1` 由 D173 解锁（`curved_prism.rs` +
│   │                            `MfemPrismRefineIds`，rs1 = TOPOLOGY-IDENTICAL、**4.65e-08**，
│   │                            rs2 4.83e-08；tri 细化也精确到 5.55e-17）。
│   │                            **round 36（D178）：2-D 三形 `nodes` writer 补齐**——缺口
│   │                            实为**连续 H1 tri**（L2 tri 早在）；点表实证 `H1_TriangleElement`
│   │                            与 `L2_T1` 同为闭 GLL 格（= `H1TriPk`）⇒ 两者皆纯置换；
│   │                            `tri2d_slot_map` + 12 个 17 位精度夹具
│   │                            `crates/io/tests/data/tri2d_*.mesh` 整文件对拍。曲面 tri
│   │                            网格自此可带曲率写出。
│   │                            prism 编号本身用新夹具
│   │                            `crates/io/tests/data/flatprism-p{2,3,4}-m{0,1}.mesh`（MFEM 4.10
│   │                            自己产出）整文件对拍到 **1e-14**
│   ├── reflector.rs         ← ✅ **round 31 修复**：① 重复元素（原来"就地反射 elem 0..ne-1
│   │                            再 append 同样副本" ⇒ 14 元素 = 7 对完全相同）改为"保留原始 +
│   │                            追加反射副本" ② 面内边界面按 C++ 跳过 ③ 反射四边形
│   │                            `rv[0]↔rv[2]` ④ 反射单元按参考立方体奇对称重排（消掉 MFEM
│   │                            `Elements with wrong orientation`）⑤ `minLength` 复刻 C++ 的
│   │                            `GetEdgeVertices(i), i<GetNE()` 怪癖 ⇒ `-m data/fichera.mesh
│   │                            -o '1 0 0' -n '1 0 0'` → NE=14 NBE=40 NV=43，单元/边界面
│   │                            多重集与 C++ **全等**。**NURBS 默认输入**（C++ 默认
│   │                            `data/pipe-nurbs.mesh`，走 `ReflectNURBSMesh`，产物头是
│   │                            `MFEM NURBS mesh v1.0`）⇒ exit(3)+缺口清单，**明确不降级成
│   │                            普通 `MFEM mesh v1.0`**（缺 NURBS 输出 + patch 反射；地基 =
│   │                            `fem_io::nurbs_mesh`，见 D143）
│   ├── twist.rs             ← ⚠️ **round 31 降级 exit(3)**（原 `if per_mesh && false` 静默
│   │                            短路 SetCurvature）：C++ 所有文档档都满足 `order>1 || dg || pm`
│   │                            ⇒ 产物带（L2）`nodes` 段，本仓无 nodes writer。**已 port**：
│   │                            `-o 1 -no-pm` → NE=3 NBE=14 NV=16，与 C++ 同命令 **拓扑逐字节
│   │                            相同**；顺带修了 v2v 顶层置换与 `-e 6`（prism）的 panic
│   ├── polar-nc.rs          ← ⚠️ **round 31 降级 exit(3)**：C++ 产物是 `MFEM NC mesh v1.0`
│   │                            + `vertex_parents`（真 NC）+ `SetCurvature(2)` 曲面 nodes +
│   │                            `-sfc`（`GridSfcOrdering2D`，该 miniapp 的存在理由）；本仓
│   │                            writer 只写 conforming `MFEM mesh v1.0`（旧实现写出的文件
│   │                            MFEM 判 `Invalid mesh topology`）⇒ 缺口清单 5 条；
│   │                            Options dump 与 C++ **逐行一致**
│   ├── mobius-strip.rs      ← ✅ **round 34（D171）：双双解锁，`exit 0`**。主会话亲验
│   │ ── klein-bottle.rs        （对照入库的 C++ 4.10 参考夹具）：
│   │                            mobius = `NE=16 NBE=16 NV=24 dim=2 sdim=3 nodes=1
│   │                            FEC=H1_2D_P3 vdim=3`，TOPOLOGY-IDENTICAL、nodes
│   │                            **4.64e-08**；klein = `NE=128 NBE=0 NV=128 …`（NBE=0 与
│   │                            C++ 一致），TOPOLOGY-IDENTICAL、**4.79e-08**（均为 C++
│   │                            8 位打印噪声级）；`r31_meshread`/`r32_probe` 双侧同。
│   │                            三块新地基：[1] writer 的 `dimension` 行与 nodes FEC 维数
│   │                            改从 `topological_dim()` 取，`nodes_dof_values` 放行（且仅
│   │                            放行）`D=3, dim=2, Quad4` 的忠实曲面情形（连续场 last-
│   │                            writer-wins = MFEM `ProjectCoefficient` 语义）；[2] 新
│   │                            `crates/mesh/src/surface_embed.rs`（`MakeCartesian2D`
│   │                            等价：MFEM 顶点编号 + SFC + 边界段序 1/3/4/2；带几何表
│   │                            重映射的去重顶点 + 曲面版 `RemoveInternalBoundaries`）；
│   │                            [3] 两个 miniapp **主体全移植**。**载荷-bearing 顺序**：
│   │                            `SetCurvature` 必须在端点识别**之前**（节点值保留识别前
│   │                            采样，共享 dof 由 last-writer-wins 收敛）。
│   │                            新测试 `crates/io/tests/mobius_klein_nodes.rs`（13 项）
│   ├── trimmer.rs           ← ⚠️ **仍开**：C++ 默认输入 `data/beam-tet.vtk`（注意是 `.vtk`）
│   │                            **本仓 `data/` 不存在** ⇒ 无法对拍；round 30 记的
│   │                            "边界元素 34 vs 36"需先补该输入才能复现
│   ├── mesh-explorer.rs / mesh-quality.rs
│   ├── ref321.rs            ← 3:1 各向异性细化 (1:1, order 1:
│   │                            unknowns 与 C++ r=1..100 全对齐,
│   │                            H1 连续性 ~0)
│   ├── mesh-optimizer.rs    ← TMOP 网格优化 (1:1, 2D quad/3D hex;
│   │                            icf/cube/jagged 的 min det 与能量
│   │                            与 C++ 逐位一致; 目标 tid 1/2/3,
│   │                            线搜索 = TMOPNewtonSolver)。
│   │                            **round 48 D369：`-qt 3`（ClosedUniform/IntRulesCU）
│   │                            解除封锁** —— `TmopQuadType::ClosedUniform` 按
│   │                            `QuadratureFunctions1D::ClosedUniform`/`CalculateUniformWeights`/
│   │                            `SegmentIntegrationRule`（intrules.cpp:856/964/1029）逐行移植；
│   │                            关键发现：**MFEM 的 ClosedUniform 只改 SEGMENT 规则**，
│   │                            TRI/TET 是与 qt 无关的 Witherden-Vincent 规则。
│   │                            star-q2/beam-hex 双端 2-D/3-D 点数、min det(J)、
│   │                            Newton 能量全部一致；orders 2..=12 的规则点数/权重
│   │                            由 `d369_tmop_closed_uniform_quad.rs` 对探针钉死。
│   │                            顺带修正 qt 相关的 **Prism 点数打印**（先前对
│   │                            -qt 1/3 打错）。⚠️ `fem_element::quadrature::prism_rule`
│   │                            本体仍是 qt 无关欠点规则 ⇒ **D372**。
│   ├── mesh_bounding_boxes.rs ← ✅ **round 48 D371：mesh-bounding-boxes 1:1 port**
│   │                            （此前无对位）。三块核心能力：plbound 全分量
│   │                            bounds（标量路径逐位不变）、
│   │                            `GridFunction::get_element_dof_values`/`get_bounds_vdim`、
│   │                            `Mesh` 的 `GetJacobianDeterminantGF` 对位
│   │                            （`det_order = dim·p−1`、GLL L2 节点、fem-rs hex
│   │                            `[-1,1]³` 参考域对 MFEM `[0,1]³` 的 2^dim 雅可比域因子补偿）。
│   │                            triple-pt-1（2-D 直边 quad）与 fichera-q2（3-D 曲边 hex）
│   │                            共 10 组命令行组合输出与 C++ **逐字节一致**
│   │                            （含 det bounds `0.0699225/1.19096`、nodal `-1.03976…`）。
│   │                            ⚠️ klein-bottle/star-surf 是 `dim 2 / spaceDim 3` 曲面网格，
│   │                            fem-rs 网格 IO 按 D112b 截断 2 分量 ⇒ 驱动检测后
│   │                            **主动 exit 3**（C++ 真值已留档 tmp/d371/）；
│   │                            `-visit` 未移植 exit 3；`-vis` 文档化 no-op。
│   ├── fit-node-position.rs ← TMOP 节点位置拟合到曲面 (1:1, 2D quad;
│   │                            EnableSurfaceFitting: 能量/梯度/Hessian
│   │                            拟合项 + 自适应拟合权 + 拟合误差终止;
│   │                            square01 初始能量与迭代/线搜索决策序列
│   │                            与 C++ 一致, 最终能量相对差 2.3e-10;
│   │                            3D hex 与 tri/tet 裁剪 exit 3)
│   └── hpref.rs             ← 随机 hp 细化 (1:1, 2D quad;
│                                unknowns/h/p/最大阶与 C++ -n
│                                3..1000 全对齐, H1 连续性 ~0)
│   └── phpref.rs            ← 各向异性 p 细化 (串行 -np 1 语义;
│                                unknowns/h/p/最大阶 + order.gf 逐值
│                                与 C++ 串行 harness 全对齐 (-n 100/
│                                1000/200-fo + aniso 变体),
│                                H1 连续性 ~e-17; 迭代行×100 逐字节;
│                                并行 PRefineAndUpdate/-proj/-dim 3
│                                裁剪, 文件头 Port notes 记录)
├── autodiff/                ← 对应 miniapps/autodiff/ (seq_example;
│   └── autodiff_example.rs     pLaplacian 能量 Newton; 依赖
│                                fem_assembly::ad 双数 AD)
├── gslib/                   ← 对应 miniapps/gslib/（**round 30 审计 1/7 → round 31 实质 4/7
│                                → round 32 四个串行件全部与 C++ 4.10 对齐**：findpts 数值
│                                1e-15、schwarz_ex1 逐位、field-diff 三行逐位、field-interp
│                                同一目标网格下 `interpolated.gf` 逐字节）
│   ├── findpts.rs           ← FindPointsGSLIB 找点/插值 (纯 Rust:
│                                BVH + 等参元 Newton, code 0/1/2 与
│                                dist² 语义对齐; glibc rand 逐位复现
│                                随机点; 13 个数值用例 counts 与 C++
│                                全对齐, max_err ~1e-15; -surf/-mpr/
│                                -hr/-ft 2/3 及 NC/mixed/pyramid
│                                网格裁剪 exit 3)
│                                ⚠️ **round 30 修复**：本文件此前**从未在
│                                `examples/Cargo.toml` 注册**（859 行、依赖齐备），
│                                导致 README 下面三条命令全部报
│                                `no example target named gslib_findpts`；已加
│                                注册，三条命令现跑通：max interp error
│                                **1.11e-15 / 6.66e-16 / 1.78e-15**（found 全中、
│                                not-found 0）
│                                **仍缺 6 件**：`field-interp`/`field-diff`/
│                                `schwarz_ex1` 属 MFEM `SEQ_MINIAPPS`（串行可跑）
│                                且只依赖已有 `GslibFindPoints` ⇒ 可直接做；
│                                `pfindpts`/`schwarz_ex1p`/`particles_redist`
│                                需并行 locator 与 `ParticleSet::Redistribute`
│                                （后者全仓 0 命中）⇒ D134
│   ├── schwarz_ex1.rs       ← ✅ **round 31 新增**：重叠网格 Schwarz 迭代解
│   │                            Poisson (1:1; **95 次迭代日志与 C++ 4.10 逐位相同**，
│   │                            其中 90 行逐字符、5 行末位 1 ulp)。移植中必须对齐的三个
│   │                            C++ 细节：① `FormLinearSystem` 会**清零非 essential 项**
│   │                            （Krylov 初值 = 只有 BC）；② 循环内 `Interpolate` 用的是
│   │                            main 里由 interior 列表构造的 `bnd1/bnd2` ⇒ 需对
│   │                            48/32 个 interior 点**重新定位**；③ not-found 点取
│   │                            `default_interp_value = 0`。
│   │                            ⚠️ **分派路径坑（务必先读）**：C++ `schwarz_ex1.cpp:176-186`
│   │                            用 `strcmp(mesh_file_1, "../../data/square-disc.mesh")` 判定
│   │                            —— **只有两个 `-m` 路径字符串逐字等于硬编码默认串时才把
│   │                            `inline-quad.mesh` 按 0.5 缩放到 [0.25,0.75]²**。用
│   │                            **绝对路径**跑 C++ 就不会 rescale ⇒ 子域 2 包住 disc ⇒
│   │                            **重叠退化、1~2 步假收敛到 3.4e-16**（实测 2 vs 95 次迭代）。
│   │                            本 port 以 `DEFAULT_MESH_1/2`（`schwarz_ex1.rs:95-96`）为条件
│   │                            复刻该语义（`:308-313`），故**与 C++ 显式路径档的行为不同**，
│   │                            已写入文件头 doc。
│   │                            附带库层发现：`fem_solver::solve_pcg` 的 `rtol` 作用在
│   │                            **(B r, r) 平方量**上（等价范数意义 1e-6），而 MFEM 判据是
│   │                            `sqrt((B r, r))`；改用 `solve_pcg_precond`(linlvo CG) 后逐位对齐
│   ├── field-diff.rs        ← ✅ **round 32：三行结果与 C++ 4.10 完全相同**（本会话独立复核：
│   │                            重编 C++ 参考后 `Max diff: 1.43502` / `Avg diff: 0.0949062` /
│   │                            `Vol diff: 1.73608` 两侧逐位一致）。round 31 的差异
│   │                            （`Avg` 0.0960922 vs 0.0949062、`Max` 2.58236 vs 1.43502）
│   │                            根因 = findpts 的候选上限：曲面网格上按几何 padding 放大的
│   │                            元素 AABB 大量重叠（单点可达 **26+ 个候选**），真正包含该点
│   │                            的元素排在上限之后 ⇒ 现改为"上限被截断且**有界搜索什么都没
│   │                            找到**时补扫剩余候选"，已定位点的结果不变（故 round 31 已对
│   │                            的 `Vol diff` 保持）
│   └── field-interp.rs      ← ✅ **round 32：同一目标网格下 `interpolated.gf` 与 C++ 4.10
│                                `**逐字节相同**（双侧 SHA256 同为 `9f39ae2e…`，216 行；
│                                本会话用显式 `-m2 data/star.mesh` 在两侧实测）。D146 根因 =
│                                目标求值点取自 `ref_elem(Tri, p)`（DG/L2 族的**等距**
│                                `TriPk`），而 C++ 的 `tar_fes->GetFE(i)->GetNodes()` 是
│                                **H¹ 族**（`H1_FECollection` + `GaussLobatto`）—— p≥3 时两族
│                                不同（1/3,2/3 vs GLL 0.27639,0.72361）⇒ 源场采样点错位。
│                                现所有 H¹ 元素查表走 `h1_ref_elem`（三角形 → `H1TriPk`；
│                                `QuadQk` 本就是 GLL）。**默认档仍不同**：`-m2
│                                data/inline-tri.mesh` 经 `fem-io` 的 INLINE `type=tri` 分支，
│                                其 quad 切分方向与 MFEM `Make2D` 相反（**D154**）⇒ 目标 P3
│                                节点集合本身不同，修 D154 前不可能一致（文件头已写明证据与
│                                替换方案）。`--gfo 1` 且 `-nc != 2` 按 C++ `field-interp.cpp:334`
│                                的越界索引行为显式拒绝（不静默复刻越界）
│                                ⚠️ **仍缺 3 件**：`pfindpts`/`schwarz_ex1p`/
│                                `particles_redist` 需并行 locator 与
│                                `ParticleSet::Redistribute`（`grep Redistribute crates/` = 0）
│                                ⚠️ 夹具 `triple-pt-{1,2}.{mesh,gf}` **不入库**（`data/*.mesh`
│                                被 gitignore）；`field-diff` 带回落：路径不存在时找
│                                `$MFEM_SRC/miniapps/gslib/<basename>`（AGENTS.md 约定变量）
│                                ⇒ `MFEM_SRC=/path/to/mfem cargo run --release --example
│                                gslib_field_diff -- -no-vis` 默认档即可跑
├── spde/                    ← 对应 miniapps/spde/
│   └── generate_random_field.rs ← Matérn 高斯随机场 SPDE (串行 1:1;
│                                WhiteGaussianNoiseDomainLFIntegrator 真
│                                随机版: minstd_rand0 +
│                                libstdc++ normal_distribution 逐位
│                                复现, 元素质量阵 Cholesky·L 乘
│                                (CholeskyFactors 进 fem-linalg);
│                                白噪声 RHS b 在 2×2 quad 网格与 C++
│                                逐位一致, 2×1×1 hex 除 1–2 ulp 求和
│                                顺序差全对齐 (单测钉死); AAA 部分分
│                                式上提共享模块 fem_examples::
│                                rational_approximation (ex33/pex33/
│                                spde 共用); 整数阶+分数阶求解链/
│                                Θ 各向异性张量/η 归一化/octet-truss
│                                +粒子拓扑/URF/scale/offset/level-set
│                                变换/ParaView 导出; 同 seed 可复现,
│                                2D+3D, 场统计物理合理; -cbi
│                                (IntegrateBC) 与非齐次 Dirichlet 裁
│                                剪 exit 3; GLVis 不支持; C++ 为并行
│                                -only miniapp, 全场数值对照不可行
│                                (⚠️ 该对照原在 **mfem49 = MFEM 4.9** 树上做 ⇒ **待用
│                                4.10 重核**；本轮只标记未改结论), 对照走分段
│                                C++ harness)
├── multidomain/             ← 对应 miniapps/multidomain/ (H1 版串行
│   ├── multidomain.rs          裁剪; ParSubMesh→extract_submesh、
│   ├── multidomain_nd.rs       TransferMap→界面 dof 坐标匹配; 结构量
│   └── multidomain_rt.rs       (NE/NV/dofs/ess 数) 与 C++ 全一致,
│                                block 场终态相对差 7e-6; cylinder 案
│                                例促成 D1 修复(曲线 hex 细化顶点吸附,
│                                480/480 逐角点对照), 待流水线复验;
│                                _nd/_rt 版已添加(H(curl)/H(div) 变体)
├── shifted/                 ← 对应 miniapps/shifted/ (SBM3 内核:
│   ├── shifted_distance.rs     sbm3_dirichlet/neumann 积分器 1:1,
│   ├── shifted_diffusion.rs    Nitsche patch test 3D 4e-13;
│   └── shifted_extrapolate.rs  distance/diffusion/extrapolate 串行
│                                驱动, C++ harness lst=1 解范数 2e-5;
│                                -vis/ParaView 裁剪; D6 locate 缺陷
│                                的 workaround 已移除)
├── hooke/                   ← 对应 miniapps/hooke/ (串行 1:1;
│   └── hooke.rs                NeoHookean AD 材料 + matrix-free
│                                弹性算子; C++ harness Newton 序列
│                                逐行一致, 终态 ‖U‖ 1e-15)
├── dfem/                    ← 对应 miniapps/dfem/ (串行 1:1;
│   └── dfem_minimal_surface.rs 极小曲面, -der 0/1/2 三模式
│                                (AD/解析/FD) 同终态; Scherk 边界)
├── solvers/lor_elast.rs     ← 弹性 LOR-AMG (D5 验收达标: 迭代数
│                                三档加密有界 25/32/37/40 等; C++
│                                串行 harness 9 例 ‖X‖/能量 ≤5.9e-11,
│                                dof checksum 逐位; 根因=linlvo AMG
│                                默认 V-cycle 非对称, 换 RS+SGS 对齐
│                                hypre 配置)
├── solvers/lor_solvers.rs   ← ✅ **round 48：D140 关闭 —— 从声明式桩变成真 1:1 driver**。
│                                `-fe h`（默认档 `data/star.mesh`、`-o 3`）的
│                                `Number of DOFs` / `L2 error` **与 C++ 逐字节相同**：
│                                `781 / 0.000395471`（star.mesh，20 个**四边形**元素，
│                                `elements` 段几何码 3）、`625 / 5.56315e-06`（inline-quad）、
│                                `289 / 0.000245071`（inline-quad `-o 2`）。走
│                                `build_lor_amg_h1` → `LorAmgPrecond` → `solve_pcg_lor_amg`。
│                                ⭐ **两个"测出来的"求积细节**（都会动到打印值）：
│                                ① **载荷**规则是 `2·order+1`（**不是** `2·order+2`）——
│                                `f` 是三角函数，RHS 求积改变离散解，实测
│                                `2p+1 → 0.000395471`（= C++）、`2p+2/2p+3 → 0.000395475`；
│                                ② `L2 error` 规则是 MFEM `ComputeL2Error` 默认的
│                                `2·order+3`（`fem/gridfunc.cpp:3410`）。**双线性型**规则
│                                不敏感（6..9 同值，矩阵被精确积分）。
│                                ⚠️ CG **迭代数不算验收**：C++ 无 SuiteSparse 时是
│                                `LORSolver<GSSmoother>`（LOR 矩阵上一次 GS），fem-rs 是
│                                LOR 上的 AMG ⇒ star.mesh 26 次 vs C++ 58 次，**解与
│                                L2 误差相同**。
│                                ✅ **round 48 四批：`-fe n`/`-fe r` 达成逐字节（D368 关闭）**。
│                                二批曾拒绝：D367 落地后 ND 仍不收敛、RT 的 L2 差末位，
│                                根因定位为 HO quad ND/RT 基非 `(GaussLobatto,
│                                IntegratedGLL)` 忠实移植。四批新增 **opt-in** 构造器
│                                `HCurlSpace/HDivSpace::new_gauss_lobatto_integrated_gll`
│                                （默认 `new` 不动，保住全部既有基线）+ 装配器
│                                `vec_ref_elem_with_basis`/`*_quad_igll` 入口
│                                （D347 模式：空间与装配器永不分歧）+
│                                `interpolate_vector` 复现 MFEM `ProjectIntegrated`
│                                子胞积分泛函；`boundary_dofs_hdiv` 在 fem-space 内
│                                正式修复（每条 2-D 边界边暴露全部 `order+1` 个 dof，
│                                MFEM 探针 `ess=96` 对齐），删除驱动侧 workaround。
│                                元素级一致性由 MFEM 探针（`$HOME/work/d368/`）钉入
│                                `crates/space/tests/d368_quad_nd_rt_igll_mfem_parity.rs`（8 项）。
│                                **验收**：`-fe n`/`-fe r` 在 inline-quad `-o 3` 均打印
│                                `1200 / 0.000134744`（**与 C++ 逐字节**；迭代 277/200 vs
│                                C++ 279/268，迭代数不算验收）；`-fe h` 两条基线不动。
│                                ⚠️ 残差 **D377**：`boundary_dofs_hdiv` 的 **3-D** face
│                                分支仍是每面 1 个 dof（tet/hex RT_k 面应为
│                                (k+1)(k+2)/2 / (k+1)² 个），同类缺陷未动（避免扰动
│                                3-D hex RT 基线）。
│                                `-fe l`（DG 面项）与"单形网格上的 `-fe n/r`"（fem-rs 的
│                                ND/RT LOR 只支持张量元）仍拒绝并给出理由。
│                                ⚠️ **另记**：`lor_solvers -m data/inline-tri.mesh -fe h`
│                                在 **C++ 侧自己 abort**（`MFEM_VERIFY(mode == DofToQuad::FULL)`，
│                                `fem/fe/fe_base.cpp:377` —— `SetAssemblyLevel(PARTIAL)` 对
│                                三角形缺张量 `DofToQuad`）⇒ 该组合**没有 C++ oracle**；
│                                fem-rs 全组装可跑（`625` dof，与 C++ abort 前打印的一致），
│                                属"超前于 C++"而非"已对拍"。
├── solvers/plor_solvers.rs  ← LOR 求解器 miniapp（**round 29：并行 H¹ 腿
│                                打通**，1:1 对齐 `plor_solvers.cpp`）: `-m`
│                                `-rs -rp -o -fe -no-vis` + `--ranks/-np N`，
│                                LOR→HO 置换 Π + **真** LOR 矩阵（在
│                                `make_refined_2d` 细网格上重新装 P1，而非
│                                `ΠᵀA_HOΠ` 的置换）、MFEM `FormLinearSystem`
│                                默认 `copy_interior=0` 初值、LOR 预条件的
│                                `EliminateRowColDiag(diag 1)` 消元；
│                                `crates/solver/src/par_lor.rs`（trait 驱动 +
│                                **实测真残差**停机/重启）。数字（`star.mesh
│                                -o 3 -rs 1 -rp 1`）: np=1/2/4 → 48/69/74 迭代、
│                                真残差 6.3e-13、L2 **2.502523e-5 与 np 无关**
│                                且 = C++ 的 2.50252e-05、`GlobalTrueVSize`
│                                3001 = C++；`inline-quad -o 2 -rs 0 -rp 0`
│                                L2 1.930630e-3 vs C++ 1.93092e-3。**诚实非可比**:
│                                迭代数（内层 AMG 不同——fem-parallel 聚合式 vs
│                                hypre BoomerAMG，且我们随 np 增长 48→74）、L2
│                                末位（求积规则）、PA vs 全装配。**阻塞腿**:
│                                ND/RT 需**分布式** AMS/ADS（linger 只有串行版，
│                                估 1–2 周），L²/DG 需并行面积分器（~1 周）⇒
│                                `-fe n|r|l` 显式拒绝并给出原因
├── solvers/block_solvers.rs ← ✅ **round 48 D370：接入 `-solver bp|bp-pcg`**
│                                （MFEM `blocksolvers::BramblePasciakSolver` 的
│                                串行 1:1；此前 4/5 个求解器，`bp` 缺位 exit(2)）。
│                                新核心件：`fem-assembly` 的
│                                `Assembler::assemble_from_element_matrices`
│                                （= `ComputeElementMatrices +
│                                `AssembleElementMatrix(i, Q_i, 1)` 路径，含
│                                `element_signs` 共轭散射）+ `fem-solver` 的
│                                `BPSParameters`/`BramblePasciakSolver`
│                                （`use_bpcg` 两分支）+ `element_q_block`。
│                                **比对中发现并修正 miniapp 装配积分阶**：
│                                M/Q = `2k+2`、B = `2k`（MFEM 积分器默认
│                                `Trans.OrderW()+2·GetOrder()`，RT `GetOrder()=k+1`，
│                                `bilininteg.cpp:2685/1830`）——修前 0 阶 u-误差差 10×。
│                                C++ oracle（`$HOME/work/d370/block_solvers_cpp`，
│                                **MPI 库构建**——block-solvers.cpp 是 MPI miniapp，
│                                串行 lib 链接失败；`mpirun -np 1` 串行协议）：
│                                `-o 0` 的 `bp`/`bp-pcg` L2 `0.0479712` 与 C++ **全部
│                                6 位一致**；`-o 1` `bp-pcg` 迭代数**精确一致（66=66）**、
│                                L2 5–6 位；`-o 2` L2 ~3 位（残差在求解器容差地板）。
│                                剩余迭代数/L2 微差归因 **fem-amg vs hypre BoomerAMG**
│                                （唯一非 1:1 组件，o0/o1 已隔离证明系统/RHS/Q 逐位）。
│                                `DarcySolver` trait 统一**暂缓**（Bdp/BP 表面已一致
│                                ~20 行；并入 DivFreeSolver 中等）。
├── adjoint/                 ← 对应 miniapps/adjoint/
│   ├── adjoint_cvodes_roberts.rs ← Robertson 伴随敏感性 (自研
│   │                              Nordsieck BDF 对位 CVODES 语义,
│   │                              检查点二分; 对照 scipy Radau 参考
│   │                              y(4e7)/G/dGdp 1e-4~1e-7 量化一致)
│   └── adjoint_advection_diffusion.rs ← 串行子集; -fd 1 自洽
│                                  (伴随 vs 有限差分 5.8e-7/1.2e-7)
├── toys/                    ← 对应 miniapps/toys/ (5 件齐全：automata / life /
│                                lissajous / mandel / mondrian；**round 32 重测了后三件**)
│   ├── mandel.rs            ← ⚠️ **round 32 部分修正 + `exit(3)`**：修前是固定 `for iter in
│   │                            0..5` ⇒ 细化 5 次、写出 **59.6 MB / 1,048,576 单元** 的
│   │                            `mandel.mesh`。C++ 在 `-no-vis` 下 `(iter+1)%4==0` 就 break
│   │                            （实测迭代 1024/2254/5884/16006、`mandel.mesh` 926,121 B）。
│   │                            现循环与打印行（含 `"elements. \n"` 的**尾随空格**）已对齐，
│   │                            **迭代 1（1024）与 C++ 完全相同**；但 C++ 用的是
│   │                            `Mesh::GeneralRefinement(refs,-1,nclimit)`（只细化被标记的
│   │                            四边形、非协调），而 `fem_mesh::amr` 只对 `Tri3` 有
│   │                            (NC) 局部细化 ⇒ 四边形退回 `refine_uniform`，迭代 2 起
│   │                            单元数分叉（fem-rs 4096/16384/65536）⇒ 仍写网格但
│   │                            `exit(3)`（`-vis` 不建套接字；C++ 的 `Continue shaping? -->`
│   │                            提示保留，stdin EOF 时 break 以免无限细化）
│   ├── mondrian.rs          ← ⚠️ **round 32 部分修正 + `exit(3)`**：修前固定 `for iter in
│   │                            0..10` 且每轮全单元 ×4 ⇒ 产出 **1.11 GB / 16,777,216 单元**
│   │                            的 `mondrian.mesh`。C++ `-no-vis` 在 `(iter+1)%3==0` break
│   │                            （实测 16/52/145、`mondrian.mesh` 6827 B）。现迭代 1（16）
│   │                            与打印行完全对齐；迭代 2 起因同一 `GeneralRefinement` 缺口
│   │                            分叉（fem-rs 64/256）⇒ `exit(3)`
│   └── lissajous.rs         ← ⚠️ **round 32：改 `exit(3)` + 缺口语明**（旧版静默写两个
│                                **全 0** 的假文件 `lissajous-v.gf`/`lissajous-h.gf` 与自造
│                                `Vertical curve sample at …` 行）。C++ 该程序要建
│                                **2-D 面嵌在 3-D**（`MakeCartesian2D` + `SetCurvature(order,
│                                true, 3, byVDIM)` + `Transform`），实测写出 `lissajous.mesh`
│                                29,968 B 与 `lissajous.gf` 4,829 B（H¹ 场 `u = x[2]`）；
│                                `fem_mesh::Mesh<D>` 的 `sdim == dim`（无嵌入面网格）⇒
│                                网格与场都造不出、写不出，现在不产出任何文件
├── dpg/                     ← 对应 miniapps/dpg/ (真 ultraweak DPG:
│   ├── dpg_poisson_2d.rs       ComplexDPGWeakForm 复块内核 + 骨架
│   ├── dpg_acoustics_2d.rs     空间/Hermitian 复 Cholesky;
│   ├── dpg_maxwell_2d.rs       poisson_2d 真 UW-DPG L² 与 C++ 四位一致;
│   ├── dpg_helmholtz_1d.rs     acoustics_2d rnum=4 4×4/8×8: 1.434/1.364
│   ├── dpg_acoustics_3d.rs     vs C++ 1.429/1.382 (0.4%/1.3%), PCG 24/33
│   └── dpg_maxwell_3d.rs       vs 24/33; maxwell_2d 真 UW-DPG 整场 L²
│                                收敛 (n=4/8/16: 0.882/0.475/0.237 vs C++
│                                0.8819/0.4753/0.2370, rate −0.94 vs
│                                −0.95); maxwell_3d 真 UW-DPG (round 13
│                                替换, round 14 多 hex 反向面 trace 结案:
│                                n2-o1 1.757 vs C++ 1.723 (2.0%), dof 数
│                                逐位 156/984/888/6192, hex2 trace 恒等式
│                                机器精度, -sc==未凝聚); -o2 差距 =
│                                trial 侧 或 测试范数 G (round 15 证伪
│                                "HexNDk 基不同" 说: nodal 重写前后整场
│                                逐位不变且差距对测试阶不敏感, D36 更正);
│                                round 16 更正: 装配矩阵 (含 4 个
│                                graph-norm cross block) 与 C++ ≤7e-15
│                                一致 ⇒ G/trial 均已排除; round 17 结案:
│                                根因是 miniapp 制造解 RHS 的 J_r[0] 多一个
│                                负号 ⇒ -o 2 由 5.15× 偏差 → 全部对齐 C++
│                                (9.547e-1/2.707e-1, rate -1.95, PCG
│                                66/119 vs 66/118), -o 1 → 1.723 = C++;
│                                acoustics_3d 真 UW-DPG (round 14 n=4
│                                缺口消失: 0.7757/0.4229 vs C++
│                                0.7765/0.4231 <0.1%); poisson_2d -o3
│                                结案 (round 14: rate −2.97/−2.99 vs
│                                −3.00, 误差 0.2%); helmholtz_1d 真 1D
│                                UW-DPG (O(h) 收敛 u/σ, k=0/5 稳定)
│   ── **round 33（D141）：并行 DPG 从 0/4 到 1/4**（新 `[[example]]`：
│      `pdiffusion`/`pacoustics`/`pmaxwell`/`pconvection_diffusion`）
│   ├── pdiffusion.rs        ← ✅ **真修**：内核新增 `crates/parallel/src/par_dpg_weakform.rs`
│   │                            （`ParDpgWeakForm`，含分布式迹编号 —— 全局面 id、
│   │                            **面主 = 持该面的最小 rank**、H1 迹角点 = 网格顶点 dof；
│   │                            块延拓 `PᵀAP` + 一次覆盖所有块的 `GhostExchange`；
│   │                            **全局 id** 上的 essential 消元；静态凝聚用**分离的 trial 索引基**
│   │                            + `n_global_trial_dofs()` 让 `-sc` 的 `Dofs` 仍等于 MFEM 的
│   │                            `Σ GlobalTrueDofSize`；带 ghost 填充的解恢复）。
│   │                            主会话独立复核（重编 C++ MPI 参考后逐配置对拍）：
│   │                            `-prob 0 -sref 0` = 113 / 1.021e+00 / 9.951e-01、
│   │                            `-prob 0 -sref 1` = 417 / 5.149e-01 / 5.115e-01、
│   │                            `-prob 1 -sref 0` = 27 / 4.755e-01 / 5.539e-01 ——
│   │                            **`Dofs`/`L2 Error`/`Residual` 三列全部逐位一致**；
│   │                            **PCG 迭代数不复刻**（C++ 自己也不是分区无关；且 fem-rs 对角块用
│   │                            对称 GS on owned part 而非 Hypre 的 `GSSmoother`）。
│   │                            新增 3 项单测（`fem-parallel --lib` 232 → 235），含
│   │                            "改前必失败"证据（回退那一行后 `pdiffusion --ranks 2 -sref 0`
│   │                            打印 `L2 = 1.779e+00`，应为 1.021e+00）
│   ├── pacoustics.rs        ← ✅ **round 35（D172 后半）：转正，`exit 0`**。主会话亲验
│   │                            （默认档 vs C++ MPI 参考日志）：
│   │                            `0 | 113 | 2.0 π | 8.008e-01 | 1.374e+00` —— **Dofs/L2/Residual
│   │                            逐位一致**（仅 PCG 迭代数 36 vs 23：fem-rs 自研复块 sym-GS +
│   │                            rtol 1e-12 vs MFEM Hypre `ComplexPreconditioner` + rtol 1e-6，
│   │                            C++ 自身也分区相关）。代理对拍 7 配置（`-sref 1`/`-sc`/`-sc -sref 1`/
│   │                            `-prob 1` np1+np2）全逐位，含 `-prob 1` 未打印的 p/u 误差拆分
│   │                            （仪器化 C++ 对到 7 位）。内核新增
│   │                            `par_complex_solver.rs`（复块对称 GS + 并行复 Hermitian PCG）
│   │                            + 静态凝聚端到端/解恢复/残差归并；**顺带修掉 round-34 潜伏 bug**
│   │                            （`recover_fem_solution` 从不拷贝 owned 虚部段 ⇒ 复数解恢复后
│   │                            虚部恒 0，L2 1.171 vs 8.008e-01；已被新 np1 测试钉死）。
│   │                            仍 `exit(3)`：`-prob ≥ 2`（PML/scatter/GSLIB 点源）、3-D 网格
│   │                            （round 37 重定性：3-D 并行迹编号已随 pmaxwell 落地，
│   │                            缺的只是本 miniapp 的 3-D 声学块表接线）、`-pref > 0`、`-pmg`
│   ├── pmaxwell.rs          ← ✅ **round 37（D172 3/4）：`-prob 0`（2-D RT/H1-trace +
│   │                            3-D ND-trace）与 `-prob 1`（fichera oven）转正**。
│   │                            内核新增 `par_dpg_numbering.rs` 的 ND-trace 并行编号
│   │                            （每边 p dof、边共享 + MFEM 规范方向符号；面 interior 用
│   │                            交换后全局面键精确前缀，混合 quad/tri 亦精确）与 3-D
│   │                            H1-trace 编号；**顺带修掉零-ghost 死锁**（condensed 3-D 中
│   │                            最低 rank 拥有全部共享迹 dof，`build_ghost_exchange` 在
│   │                            本 rank 零 ghost 时早退 ⇒ 请求方永久等待；`-prob 1 -sc`
│   │                            双 rank 挂死现象）。主会话亲验对拍 C++ MPI 4.10：
│   │                            3-D `-prob 1` np1/np2/`-sc` 三配置 **166 dofs /
│   │                            6.780e-17 / PCG 15/15/7 逐位**；refine 后 1020 /
│   │                            7.092e-01；3-D `-prob 0`（inline-hex，ND-trace）
│   │                            984 / L2 1.313e+00 / 残差 4.706e+00 逐位（仅 PCG 迭代数
│   │                            不复刻：HypreAMS/Jacobi + rtol 1e-6 vs 复块 GS +
│   │                            rtol 1e-12，pacoustics 先例）。
│   │                            仍 `exit(3)`：`-prob 3/4`（scatter.mesh +
│   │                            GSLIB 点源）、`-pmg`、AMR。
│   │                            **round 39（D211 = D172 4/4 全关）**：`-prob 2`
│   │                            转正——`CartesianPML` 移植（`util/pml.rs`）+
│   │                            9 个 Dpg 空间变系数积分器 + 逐 trial 块求积规则
│   │                            （`set_trial_quad_order`，默认路径逐位不变）；
│   │                            2-D 对拍 np1/np2/sc 逐位（113/1.132e+00、
│   │                            417/1.090e+00）。
│   │                            **round 40（D219 关闭）**：3-D `-prob 2` 转正——
│   │                            根因 = 3-D 积分规则漏 MFEM `Trans.OrderW()`
│   │                            （hex = geo·dim−1 = 2；2-D 时 OrderW=1 恰落同
│   │                            Gauss 点数，纯巧合）⇒ `set_test_quad_order`
│   │                            逐 (row,col) 测试块规则（round 39 trial 版的
│   │                            对称仲裁件，默认路径逐位不变）；
│   │                            **984 / 5.891e-01 == C++ 逐位**（`-sref 1`
│   │                            6960/5.406e-01 亦同）。
│   │                            round 38（D195）：serial
│   │                            `ComplexDPGWeakForm::compute_residual` 的 ND 迹
│   │                            双折号已修（MFEM 语义 = 存未折号块 + 取时一次；
│   │                            修后 serial 残差 4.706293799742669 = C++ np1
│   │                            逐字），并行侧绕过退役，三条头条复跑逐位不变
│   └── pconvection_diffusion.rs ← ⚠️ **诚实 exit(3) + 缺口清单**（缺带系数的 DPG 积分器 +
│                                `setup_test_norm_coeffs`）
│      ✅ **round 34（D167）：库层已修**——`from_local_matrix` 的 ghost 列数改从
│      `local.ncols - n_owned` 推导（矩形输入合法）+ `nrows ≥ n_owned` 断言；
│      回归钉两层（`par_csr.rs:750` 矩形用例 + `two_rank_system_matches_serial_full_mesh`，
│      改前回退可复现 `pdiffusion --ranks 2` 打 `L2 = 1.779e+00` vs C++ 1.021e+00）。
│      ⚠️ 代理的诚实声明：fem-rs 的 `--ranks` 走 `ThreadLauncher` **进程内通道**（本仓惯例，
│      如 `mfem_pex8_parallel_dpg`）⇒ 这只对标**分布式算法**（对 C++ 真 MPI 输出），
│      **不是**多地址空间行为（未启用 rsmpi）
├── fluids/schrodinger_flow.rs ← 不可压 Schrödinger 流 (ISF) 串行 1:1:
│                                CN 复 GMRES + 逐 DOF 归一化 + gauge 投影
│                                (OrthoSolver); leapfrog/jet 对照 C++
│                                (B r,r) 序列与 ‖ψ‖²/lapl ~1e-14;
│                                内核缺口绕过: make_periodic 几何畸变
│                                (局部张量 H¹) + solve_gmres_complex
│                                Givens 实数化发散 (局部标准复 GMRES)
├── fluids/navier_kovasznay.rs ← 首个 navier 求解器 miniapp (MFEM
│                                miniapps/fluids/navier/navier_kovasznay.cpp
│                                1:1) + 内核 crates/solver/src/navier.rs
│                                (BDFk/EXTk 含变 dt、三次求解、CFL、
│                                Orthogonalize/MeanZero、PrintInfo 格式);
│                                与 C++ 10 步 err_u/CFL 打印 6 位逐位一致、
│                                MVIN/HELM 迭代数相同、-cr exit 0; 剩余 8 个
│                                navier miniapp 与内核缺口见 round3_plan D46
├── fluids/navier_mms.rs       ← 第 2 个 navier miniapp (navier_mms.cpp
│                                1:1): 与 C++ step1 err_u 2.75455E-08 /
│                                err_p 1.23108E-04 逐位一致, MVIN/PRES
│                                每步相同, -cr exit 0; g_bdr 已切内核
│                                VectorBoundaryNormalLFIntegrator (D46③),
│                                kovasznay 同步切换后仍保持 6 位一致
├── fluids/navier_shear.rs     ← 第 3 个 (navier_shear.cpp 1:1, 双剪切层
│                                全周期): CFL/迭代数/各 L2 范数全 10 步与
│                                C++ 逐字节一致 (cfl 7.56030E-02, MVIN
│                                4/9, PRES 47/76, HELM 6/6); round 18 起
│                                初始条件用库内 interpolate_vec (D56 修复
│                                周期网格 DOF 坐标后本地绕过已删)
├── fluids/navier_kovasznay_vs.rs ← 第 4 个 (自适应时间步: provisional +
│                                CFL 接受/拒绝 + dt 预测 + 历史排队):
│                                CFL/Time/dt 全 5 步逐字节一致, err_u 6 位
├── fluids/navier_tgv.rs       ← 第 5 个 (navier_tgv.cpp 1:1, 3D 周期
│                                Taylor-Green): step0 (u_inf/p_inf/ke) 与
│                                C++ 逐字节一致, ke 打印位 11 行全同
│                                (全精度 rel ≤5.7e-10), HELM/PRES 迭代逐位;
│                                3D 周期构造验证可用 (27 torus 节点,
│                                order 4 = 1728/5184 dof = C++); 解析衰减
│                                ke=⅛·e^(−6νt) 钉住; ComputeCurl3D 入内核
├── fluids/navier_bifurcation.rs ← 第 6 个 (navier_bifurcation.cpp 1:1,
│                                2D 通道分叉 + 粒子追踪): DOF 52866/26433
│                                = C++, step1 CFL 6.03374E-02 逐位, 粒子
│                                计数 600 步全一致, CSV 表头逐字节, 收敛区
│                                100 步中 97 步 CFL 末位一致; 粒子用库内
│                                crates/mesh/findpts (无裁剪, RNG 逐位复刻);
│                                裁剪项: GLVis/ParaView/-traj (文档标注)
├── fluids/navier_3dfoc.rs     ← 第 7 个 (navier_3dfoc.cpp 1:1, 曲线
│                                box-cylinder, 内核零改动): DOF 16956/5652,
│                                vel_ess_tdof 7149 = C++, Time/dt 表逐字节,
│                                MVIN/PRES 迭代数每步相同, 五项矩阵统计
│                                (Σ/Frobenius/迹/A·x) 一致到 1e-15;
│                                两个可复用发现: 曲线网格求积阶必须含几何阶,
│                                MFEM 张量参考单元是单位立方而 fem-rs 在
│                                [-1,1]^d (仅影响 GetElementSize/CFL 类量);
│                                HELM 第 2 步起差 1 = 残差卡阈值舍入;
│                                裁剪: ParaView/PA-LOR-AMG/GLVis/8000 步窗
├── fluids/navier_turbchan.rs  ← 第 8 个 (navier_turbchan.cpp 1:1, 周期
│                                tanh 湍流通道, 内核零改动): MVIN/PRES 6 步
│                                含残差逐位一致, order 5 的 dt/hmin/hmax/
│                                dx+ 与 banner (1470150/490050 dof, vel_ess
│                                36300) 逐位, order 1 的 |u|/|p|/CFL rel
│                                < 1e-6; C++ 侧 UB: navier_turbchan.cpp:155
│                                读未初始化 Array<int> (靠堆恰为 0 侥幸) ⇒
│                                对照运行须显式清零 (否则 HELM 28 vs 30);
│                                order 5 是纯 PA 档 (5.9e9 nnz ≈ 70GB,
│                                两侧都跑不动) ⇒ order 1 是唯一可比档
├── fluids/navier_cht.rs       ← 第 9 件 (navier_cht.cpp **部分交付, 退出码
│                                3**, 重叠网格共轭传热): 已移植双域网格/加密
│                                (默认档 11 elem + 24 solid; -r1 3 -r2 2 =
│                                704/384 elem, VDOF 23042/11521, 热 dof 3185
│                                全部 = C++)、重叠传递算子 (4 阶解析场插值
│                                误差 1.7e-13 / 2.27e-13, 未找到集合 = 流体
│                                挖去的 block 几何)、**热网格 SetCurvature(4)**
│                                与**热求解** K = ∫κ∇T∇v + (u·∇T)v（对流项走新
│                                入库的 MixedDirectionalDerivativeIntegrator,
│                                求积阶 14 = 4+4+(4−1)·2）：对流恒等式
│                                max|K_adv·T − M(u·c)·1| = 6.66e-16、essential
│                                66/3185、热反欧拉 dt=2e-2 PCG(Jacobi) 53 迭代
│                                收敛、‖T₀‖₂ 2.198207E2 → ‖T₁‖₂ 2.064884E2;
│                                缺口: 无 OversetFindPointsGSLIB 对位、流体侧
│                                NavierDiscretization 未移植（对流场用解析替代）、
│                                耦合轨迹不可核（串行 harness 的 FindPoints 漏点）、
│                                C++ 参考不可编（本机所有构型 MFEM_USE_GSLIB=NO）
└── ...                      ← tools/nodal_transfer.rs 已接入 (kd-tree
                                 投影; C++ 对照 6/7 案例一致, 1 例暴露
                                 tet io round-trip 取向归一化内核缺口)
```

## 本轮新增核心库能力 (fem-rs crates)

### round 48（二）— `miniapps/solvers/lor_solvers.rs`：D140 关闭，桩 → 真 1:1 driver

- **背景**：该文件自 round 31 起是**声明式桩**——装好 H¹ 质量阵又**显式丢弃**、只解刚度阵、
  不 import 任何 LOR 符号、任何输入 `exit(3)`。它需要的 LOR 栈此后**全部落地**
  （`fem_space::lor::{LorH1,LorNd,LorRt}`、`fem_assembly::lor_factory`、
  `fem_solver::lor`），所以本轮把它写成真 driver。
- **交付**：`-fe h` 的 `Number of DOFs` / `L2 error` 与 MFEM 4.10 **逐字节相同**：

  | 运行 | C++ | fem-rs |
  |---|---|---|
  | `-m data/star.mesh -fe h` | `781` / `0.000395471` | `781` / `0.000395471` |
  | `-m data/inline-quad.mesh -fe h` | `625` / `5.56315e-06` | `625` / `5.56315e-06` |
  | `-m data/inline-quad.mesh -fe h -o 2` | `289` / `0.000245071` | `289` / `0.000245071` |

  （`star.mesh` 是 **20 个四边形**元素的网格——`elements` 段几何码 3——不是三角形网格；
  之前的记载把它当三角形是错的。）
- **两个"测出来的"求积事实**（都写进代码注释与测试文档）：① **载荷规则 `2·order+1`**
  （`2·order+2` 会给出 `0.000395475`，差在打印的第 6 位）；② `L2 error` 用 MFEM
  `ComputeL2Error` 的默认 `2·order+3`（`fem/gridfunc.cpp:3410`）。双线性型规则不敏感。
- **明确拒绝**（给实测数字，不降级、不伪造）：`-fe n`/`-fe r`（LOR 预条件子必须建在消元后的
  算子上，而 `build_lor_ams_nd_quad`/`build_lor_jacobi_rt_quad` **不收 essential-dof 表**
  ⇒ ND 真残差停在 `1.2070903825e-01`、RT 的 L2 差 C++ 末位）、`-fe l`（DG 面项不是
  `BilinearForm` 积分器）、单形网格上的 `-fe n/r`（fem-rs 的 ND/RT LOR 只支持张量元）。
- **发现的 C++ 自身边界**：`inline-tri.mesh -fe h` 在 C++ 侧 `MFEM_VERIFY` abort
  （PA 对三角形缺张量 `DofToQuad`，`fem/fe/fe_base.cpp:377`）⇒ 该组合无 oracle。

### round 48 — D353 及其同类静默零（3-D 单元上的 L² 度量）

- **D353（P1，已修）**：`fem_assembly::postproc::grid_function::compute_coeff_l2_norm`
  对 **所有 3-D 等参单元恒返回 `0.0`**（`coeff=1, q=6`：Quad4 `1.0` ✓、Tet4
  `sqrt(1/6)` ✓、**Hex8 / Prism6 / Pyramid5 全 `0.0`**）。根因：该文件私有的
  `element_jacobian` 把 Hex8/Prism6/Pyramid5 列进 `needs_iso`，却对每个非 quad
  类型都建 `ref_elem_vol(ElementType::Quad4, 1)` —— **给 3-D 单元配 2-D 基**，
  `J` 第三列恒为 0 ⇒ `det J ≡ 0`。现改为**委派 `fem_mesh::transformation::
  element_jacobian_at`**（mesh crate 的几何 Jacobian 单一真源），几何元素即单元
  自身的类型，并顺带拿到曲面几何表 / 金字塔 `PYR_P1_SLOT_VERTEX` 槽置换 / 曲面
  棱锥的 order-`g` 元素。
- **C++ 对拍**（`tmp/d353_probe.cpp`，MFEM 4.10 `ComputeLpNorm(2.0, coeff, mesh, irs)`，
  `irs[geom] = IntRules.Get(geom, 6)`）：hex8 `1.0` / prism6 `sqrt(1/2)` /
  pyramid5 `sqrt(1/3)`；`coeff = x²` 时 `‖x²‖₂ = (∫x⁴)^{1/2}` 分别
  `sqrt(1/5)`、`sqrt(1/30)`、`sqrt(1/35)`，另加 2×3×4 缩放 hex（`sqrt(24)`）；
  两侧相对差 < 1e-14。测试 `crates/assembly/tests/d340_pyramid_l2_assembly.rs`
  的 `coeff_l2_norm_on_3d_iso_cells` / `coeff_l2_norm_first_n_on_3d_iso_cells` /
  `coeff_l2_norm_matches_the_cpp_compute_lp_norm_oracle`（3 项，原 canary 已转真断言）。
- **同类静默零普查 ⇒ 第二处**：`postproc/flux_recovery.rs::geom_jacobian` 对未特判的
  类型一律走"顶点差"回退，而 **hex 的 `nodes[1..4]` 是两条基边 + 基对角线** ⇒ 三列
  线性相关 ⇒ `det J ≡ 0`；`compute_element_flux` 的 `try_inverse().unwrap_or_default()`
  因此返回**恒零通量**，`compute_flux_energy` 则按 0 加权。此前不可达是因为同一文件
  的 `ref_elem_vol` 直接**拒绝 Hex8**（panic）——即 3-D hex 上根本没有入口。现：
  ① `ref_elem_vol` 增加 `Hex8/Hex20` 臂（`HexQk`，与几何同一元素、同一参考域）；
  ② `geom_jacobian` 增加 Hex 等参臂（曲面读几何表）；
  ③ `fe_order` 推断表补 hex p=2/3（原先 `_ => 1` 在 p≥2 会拿 8 个 dof 的基去配
  27/64 dof 的通量向量）。测试
  `crates/assembly/tests/d353_sibling_silent_zeros.rs`（4）：仿射场 `u = x+2y+3z`
  的恢复通量逐 dof 等于精确梯度；常差通量的能量等于闭式 `κ|v|²|K|`；
  **端到端** `zz_estimator_mfem_nc`（`amr_refiner::ThresholdRefiner` 真正调用的入口）
  在 hex 与 prism 网格上可跑、对仿射场 `total_error < 1e-12`、**且对 P1 不可表示的场
  `sin(x)·y + z²` 每个单元指示子严格为正**（防止"恒零估计子"也能通过）。
  ⇒ 新增能力：**3-D hex / prism 的系数感知 ZZ 通量恢复与误差估计**
  （`ThresholdRefiner` 现在可以在 3-D 上跑；此前 panic）。
  **留白**：Pyramid5 仍无 `ref_elem_vol` 臂（记 D365）。

### round 48（七）— 忽略测试清剿批（D415–D419，全部关闭；`#[ignore]` 仅剩合法诊断/长验收）

用户指令"优先修复忽略的测试"。四路把最后 5 处**因缺陷/缺料被忽略**的测试全部清零；
至此 `#[ignore]` 仅剩合法类别：诊断探针（lor_factory 5、par_lor_h1 2、d269、d224 ×2、
d244、poisson_p3_debug_rates）、基准（ras_benchmark ×3、d260 热路径）、长验收
（d342/d337 32³）、GPU 环境自跳过（见 ①，已非 ignore）。

1. **D415 —— linalg-gpu 11 处 ignore 全部移除，改"适配器条件自跳过"**：本机实测
   wgpu 适配器**存在**但**不支持 SHADER_F64**——原 `ctx().expect("GpuContext")` 在
   无适配器机器上会 panic，这是当年加 ignore 的原因。现全部上下文助手返回 `Option`、
   无适配器/无 f64 时打印可见 `SKIP:` 行后提前返回；`cargo test -p fem-linalg-gpu`
   **28/28 全绿（0 ignored）**，f32 分支真实执行，f64 数值路径待有 f64 适配器的机器
   验证（如实记录）。
2. **D416 —— `curl_3d`"占位"实为历史误标，修正遗留缺陷并补性质检验**：核查发现
   ND1→RT0（拓扑面-边关联）与 ND2→RT1（双基重构，= MFEM `CurlInterpolator`/
   `ProjectCurl3D_RT`，`bilininteg.hpp:4159`、`fe_base.cpp:1385`）**早已完整实现**，
   并行端 `ParDiscreteLinearOperator::curl_3d` 直接复用串行矩阵。本轮修三处遗留：
   调试 `eprintln!("TEMP …")` 残留删除；不支持单元分支误用
   `UnsupportedHCurlOrder{order: elem_type as u8}` 改 `UnsupportedCellType`；
   陈旧 ignore + 测的是**未合成部分乘积**的打印循环改真稠密乘积并加断言。
   **验收**：div∘curl 恒等式达机器精度（ND1→RT0→P0 **4.441e-16**、ND2→RT1→P1
   **1.187e-12**）；制造场收敛率 1.01/1.01（一阶）、渐近 0.96（二阶，O(h) 符合
   预期）；`discrete_op` **47/47 绿（0 ignored）**。
3. **D417 —— HDG 弹性 3-D skeleton（取消 ignore）**：四个真缺陷——**NaN 根因** =
   重建通道硬编码 2-D 行列式（Kuhn 四面体前导 2×2 奇异 ⇒ det=0 → inf·0 = NaN）；
   `face_size` 的 3-D 面-顶点表与 `local_faces` 不一致（每个 ∂K 积分用错面的测度）；
   面 3 求积映射置换了面基-顶点配对；梯度变换用 J⁻¹ 而非 J⁻ᵀ（**两维都有**，斜切
   单元刚度错）+ 内部面 λ 槽按恒等映射绑定（数值通量跨面不单值）。修法：统一为
   维度无关 `build_condensed` + 正确伴随 + λ 槽置换。**验收**：零源问题精确复零解
   （max|u|=max|λ|=0，修前 NaN）；`hdg` **16/16 绿（0 ignored）**。新债
   **D426/D427/D428 均已随修关闭**。
4. **D418/D419 —— contact_mortar 与 schur_s_matrix（取消 ignore）**：前者从标量
   Laplace 占位升级为真 `ElasticityIntegrator` + `VectorH1Space`（关键发现：其全局
   dof 是**分块布局** `dof = comp·n_scalar + node`），补齐消除平移/旋转刚性模态的
   Dirichlet 支承（Jacobi 特征分解证明的铰链机构），`solve_mortar_uzawa` 修正为
   符号物理一致的投影 Uzawa（λ≥0 = 接触压力）；~1400 次 Uzawa 迭代收敛，3/3 绿。
   后者：重新生成 star.mesh 三个 Schur 补 dump（20/80/320 阶）+ 新增结构化回退
   （缺 dump 时 `assemble_schur` 构造同规格矩阵，**永不空转**）；无 dump 5/6/7 次、
   有 dump 5/7/9 次，均收敛 ≤40。

### round 48（六）— 失败/忽略测试修复批（D73a/D73b/D401/D402，全部关闭；唯一红测试清零）

用户指令"先修复失败和忽略的测试"。清单：38 处 `#[ignore]` 逐一分类——**4 处因缺陷被忽略**、
1 处红测试；其余为合法（诊断探针/基准/长验收/手动诊断，如 d342 32³、lor_factory 5 项
打印诊断、ras_benchmark、poisson_p3_debug_rates 等，本轮确认保留）。

1. **D73(a) + D403 —— 唯一红测试 `poisson_nc_amr_convergence` 修复**：根因是
   `ElementIndicators::dorfler_mark` 把 Dörfler 判据写错——按 **η 线性**累加对
   θ·‖η‖₂ 停止，而非标准体判据 **Ση²_marked ≥ θ·Ση²**。对本题近均匀误差分布，
   每轮只标 ~2 个单元（网格 5 轮仅 8→38），AMR 卡在一次性加密水平 7.9e-2。
   修正后（`error_estimate.rs` 一处）测试**原断言直接通过**：末级 L2 **4.16e-2**，
   单调性保持；并用 MFEM 4.10 C++ 对照（ZZ + ThresholdRefiner 0.5，5 轮 4.27e-4）
   证明期望本身保守合理——未放松任何阈值。链路其余环节
   （约束装配残差 ~1e-16、悬挂值恢复精确 C⁰、l2_error 叶元积分）均探针验证健康。
   **新债 D404**：fem-rs 的质心 ZZ 估计子判别力远低于 MFEM 的 L2 投影 ZZ
   （同标定下 4.16e-2 vs 4.27e-4）；MFEM 级路径（`zz_estimator_l2_nc`/`zz_estimator_nodal`）
   已在树上，NC AMR 示例可切换。
2. **D73(b) + D406 —— ams_ads 复数 GMRES-AMS hpc 平台修复（取消 ignore）**：根因是
   驱动层**左预处理收敛判据失真**——预条件残差 ‖M⁻¹r‖ 低估真残差 ~2 个量级，每个
   重启循环提前退出（平台**随 tol 线性移动、与预算无关**——tol 探针 6.8e-7/6.4e-9/
   1.2e-10 证明 (a) 奇异、(c) 循环缺陷均不成立，HANDOVER 旧假设被推翻）。
   修复在 fem-solver 自己的驱动层：`solve_gmres_ams_complex` 改用新实现的
   **右预处理**重启 GMRES（真残差最小化 + 监控 + 每轮重启复核），vendor/linger 未动。
   实测 hpc 16×16：**2000 迭代/6.05e-5 平台 → 25 迭代/6.27e-7 收敛**；default 预设
   同步改善（22→18 迭代）；`ams_ads` 全部 **11 个测试通过（0 ignored）**。
3. **D401 —— stokes_darcy_coupled MMS 取消 ignore**：根因是**两个测试侧缺陷 +
   一个库侧隐患**：(i) 库函数 `apply_dirichlet_keep_diag` 按**对称矩阵**语义从主元行
   取列反力，对 `[A −Bᵀ; B]` 型鞍点系统把**非零**本质值的贡献符号翻转（所有只钉
   零值的既有测试均不受影响 ⇒ 只有此测试暴露）——测试内改真列消元，库侧记
   **D409**；(ii) Stokes 块带 Brinkman 质量项而右端按纯 Stokes 导出（删除）；
   (iii) εI 正则把离散相容性失配 δ=1.5e-3 放大成 1.4e11 的压力常数（改守恒型
   修正，**D411**）。修后四条收敛率恢复理论值：vel **2.95**/p **3.30**/flux **0.95**/
   p **0.96**（修前 0.07/0.31/0.06/0.64）。`poisson.rs:178` 的裸 ignore 确认为合法
   手动诊断（P3 打印、无断言），保留。
   **新债 D410**：RT0 面数据用单点中值而非 ∫_F f·n 矩（O(h²) 偏差，仅影响
   精确数据研究）。
4. **D402 + D412/D413 —— 并行 ND2/RT1 DOF 分区修复（取消 ignore）**：3-D NDk
   (k≥2)/RTk(k≥1) 的**面自由度**被按"首次所见单元"键控与归属（跨 rank 不稳定 ⇒
   ND2 哨兵 gid + GhostExchange panic；RT1 同面 gid 静默别名）；多 dof 边的位置在
   compact 节点模式下取局部序 ⇒ 镜像边 dof 跨 rank 互换。修复 = 拓扑/几何规范键：
   面按（3 个最小全局顶点 + 最小 gid 单元的面块位置）经既有 `exchange_ghost_face_keys`
   轮次交换，边按全局最小端点重定基；2-D 与 ND1/RT0 逐位不变。d110 取消 ignore、
   ranks 2/4 全绿；新增 `d412` 回归测试（**对修复前代码验证过"有牙"**）。
   **新债 D414**：`HDivSpace::dof_coords` 对 RTk 的面/内部 dof 坐标不完整
   （RT0 级锚点 + [0,0,0]），不应作为 dof 身份键使用。

### round 48（五）— 四路并行修复批（D383/D377/D380+D386/D352，全部关闭）

1. **D383（P1）pex31 ∇z 梯度转置约定 —— 关闭**：`mfem_pex31_restricted_hcurl.rs`
   的 `compute_hcurl_error` 误用 J⁻ᵀ 反对角（`dx←jit10`、`dy←jit01`），改为与串行
   ex31/MFEM `GetCurl`（`grad_hat·J⁻¹` 行约定）一致的 `dx = jit00·gξ + jit01·gη`、
   `dy = jit10·gξ + jit11·gη`。**inline-quad（对角 J）逐位不动**：np1-4 的
   `0.0907163` + ‖u‖/sum/checksum 全部保持；**剪切网格恢复正确**：inline-tri
   `0.312913`（1089 unk）、star `0.858735`（1041 unk），与 C++ ex31（-r 2）打印
   精度完全吻合。顺带删除 HEAD 即死代码的 `extract_block`。
2. **D377（P2）`boundary_dofs_hdiv` 3-D 面自由度 —— 关闭**：`HDivSpace` 新增
   `face_dofs(FaceKey)` 整块访问器（tet 面 `(k+1)(k+2)/2`、hex 面 `(k+1)²`），
   3-D 分支镜像 D368 的 2-D 修法。MFEM `GetBoundaryTrueDofs` oracle 在
   beam-tet/inline-hex × k=0..2 上 **8/8 对齐**（含 `GetVSize` 逐项；此前 k≥1 欠约束：
   beam-tet RT1 272 → 816）。**消费者审计：零 pinned baseline 移动**（3-D RT0 与
   2-D 不受影响）。⚠️ 测试需读真实 MFEM 网格 ⇒ `crates/space/Cargo.toml` 增加
   `[dev-dependencies] fem-io`（超出该路许可清单的一处必要偏离，主会话审计后接受）。
   **新债 D392/D393/D394**：tet RT 阶上限 k≤2（MFEM 无上限，k=3 oracle 已备）；
   `build_mixed` 的 tet 面块按 `k+1` 而非 `(k+1)(k+2)/2`；`build_3d_prism` 的 tri 面
   同病。
3. **D380+D386 —— 关闭**：`h_refine_vk` 改为按 `j*nu+i` 精确散写（= MFEM
   `KnotInsert` A5.5 布局），多结点 v 插入与"转置↔`h_refine_uk`↔转置"逐位一致，
   D374 的绕行补丁与死代码删除，`d374_curveint_patch` 字节级验收不变；
   fem-space 嵌套 3-D 分支补 hex 定位器（移植 D376 已验证的 Newton 反演），
   P·1=1、Pᵀ 单亲划分成立，且与 fem-solver Newton 路径及一阶 dyadic 参考
   **max|diff| = 0（逐位）**（order 1 与 2）。
4. **D352（P2，r47 遗留）—— 关闭**：`build_pyramid_pk` 全局编号由"元素首次触及"
   改为 MFEM `Construct` 实体分相序（顶点→全部边→全部面（基四边形在前）→内部），
   与 D177 棱柱同构；单元局部槽表不动。`data/octahedron.mesh` p=1..3 的
   `GetElementDofs` 绝对编号表由 mfem410_ser 探针钉入
   `d352_pyramid_entity_phase_numbering.rs`（含 D348 旋转基 `32 33 30 31`）；
   vsize 6/21/58 不变，round-47 的 `22..25 → 30..33` 已反转。d348/d340/d349/d335
   金字塔套件全绿。**新债 D398**（P3 文档漂移）：d348 测试文档仍描述旧的单趟分配。



1. **D368（quad ND/RT 忠实基）——关闭，`lor_solvers -fe n/r` 达成逐字节**：
   opt-in `new_gauss_lobatto_integrated_gll` 构造器 + `vec_ref_elem_with_basis` 装配入口 +
   `ProjectIntegrated` 泛函 + `boundary_dofs_hdiv` 2-D 正式修复（详见上方 lor_solvers 条）。
   **新债 D377**：`boundary_dofs_hdiv` 的 3-D face 分支同类缺陷（每面 1 个 vs
   (k+1)(k+2)/2 / (k+1)² 个），为不动 3-D hex RT 基线暂缓。
2. **D374（NURBSPatch 对象层 + nurbs_curveint）——关闭**：见上方 nurbs 条。
   关键发现 **D380**：`h_refine_vk` 行列布局缺陷（多结点 v 插结点返回错乱数据）。
3. **D375（串行 ex31）——关闭 D128**：`mfem_ex31_anisotropic_maxwell` 从无条件
   `exit(3)` 桩变为真 1:1 driver（`[H¹(z)|H(curl)(xy)]` 组合空间 = ND_R2D 受限元、
   DIAG_KEEP 消元、GS 预条件 PCG）。**inline-quad 默认工况整段 stdout 与 C++ 逐字节**
   （`0.181455`、74 次 PCG、ARF `0.829075`）；inline-tri `0.312913`、star `0.858735`
   亦逐字节；dump 对比 A/b/消元/x 全部 ~1e-14（修掉 dump 双索引约定缺陷）。
   过程中发现 **D383（真缺陷）**：`mfem_pex31_restricted_hcurl.rs:858` 的 ∇z 物理梯度
   **Jacobian 转置约定反了**（`dx` 应为 `jit00·gξ + jit01·gη`）——对角 J 的
   inline-quad 上不可见（pex31 已发布数字仍有效），斜切三角形上 H(curl) 误差虚大
   ~10×，待修。**D384**（妆饰性）：raw A 多 392 个显式 ≈0 结构项。
4. **D376（mg-abs-l1-jacobi）——关闭**：见上方 diag-smoothers 条。
   新债 **D386**（fem-space 嵌套 3-D 延拓缺 hex）、**D387**（ess 行 RHS 约定差异，
   ‖b‖ 诊断不可直接对拍）、**D388**（`kershaw_map` 与 MFEM `KershawTransformation`
   在非规则网格不等价——C++ 侧自产 NaN）。

### round 48（三）— 四路并行（D367/D369/D370/D371）

1. **D367（LOR essential-dof）**：`build_lor_sgs_nd_quad`/`build_lor_sgs_rt_quad`
   （新）与改造后的 AMS/Jacobi 构造器收 HO essential-dof 表，按 MFEM 串行
   batched 路径 `EliminateBC(ess_dofs, DIAG_KEEP)`（`lor_batched.cpp:726-734`）
   映射到 LOR 编号后消元；新增 `LorSymGs` 适配器（= `LORSolver<GSSmoother>`）。
   顺带发现并修正 `boundary_dofs_hdiv` 每边界边只暴露 1 个 dof 而 `RT_Quad(2)`
   需 `p+1`=3 个的缺陷（MFEM 探针 `ess=96/96` 对齐）。`-fe h` 三组基线不动；
   `-fe n`/`-fe r` 仍拒绝但根因已定位到 **D368/D69**（HO quad ND/RT 单元非忠实
   移植 ⇒ 谱不匹配；IGLL HO 矩阵下同一栈健康）。**`boundary_dofs_hdiv` 的
   per-edge dof 数建议在 fem-space 正式修复**（当前以驱动侧 `boundary_dofs_hdiv_quad_rt`
   变通）——记入 D368 附注。
2. **D369（`-qt 3` ClosedUniform）**：`TmopQuadType::ClosedUniform` +
   `quadrature_functions_1d_closed_uniform`（`intrules.cpp:856/964/1029` 逐行）。
   关键发现：MFEM 的 ClosedUniform **只改 SEGMENT 规则**（TRI/TET 是与 qt 无关的
   Witherden-Vincent 规则）。`mesh-optimizer -qt 3` 解除封锁，双端 2-D/3-D
   点数、min det(J)、Newton 能量全部一致；orders 2..=12 规则钉死
   （`d369_tmop_closed_uniform_quad.rs`）。顺带修正 qt 相关 **Prism 点数打印**。
   ⚠️ `fem_element::quadrature::prism_rule` 本体仍是 qt 无关欠点规则 ⇒ **D372**。
3. **D370（BramblePasciakSolver）**：`fem-assembly` 新增
   `Assembler::assemble_from_element_matrices`（`ComputeElementMatrices +
   AssembleElementMatrix(i, Q_i, 1)` 路径）；`fem-solver` 新增
   `BPSParameters`/`BramblePasciakSolver`（`use_bpcg` 两分支）；`block_solvers`
   接入 `bp`/`bp-pcg`。**比对发现并修正 miniapp 装配积分阶**（M/Q=2k+2、B=2k，
   `bilininteg.cpp:2685/1830`；修前 0 阶 u-误差差 10×）。o0 L2 `0.0479712`
   与 C++ 6 位全同、o1 `bp-pcg` 迭代 66=66；剩余微差归因 fem-amg vs hypre
   BoomerAMG（唯一非 1:1 组件）。
4. **D371（mesh-bounding-boxes）**：1:1 port；plbound 全分量 bounds（标量路径
   逐位不变）+ `get_element_dof_values`/`get_bounds_vdim` +
   `Mesh` 的 `GetJacobianDeterminantGF` 对位（`det_order = dim·p−1`，含 fem-rs
   hex `[-1,1]³` 对 MFEM `[0,1]³` 的 2^dim 域因子补偿）。triple-pt-1 与
   fichera-q2 共 10 组 CLI 输出与 C++ **逐字节一致**。曲面网格（klein-bottle/
   star-surf）因 D112b IO 缺口主动 exit 3（C++ 真值留档 `tmp/d371/`）。


- `fem_assembly::dist_solver` — MFEM `fem/dist_solver.*` 1:1 距离场
  支撑库 (3924 行)：Heat/Normalization/p-Laplace 距离求解器、
  PDEFilter (ScreenedPoisson/PUMPLaplacian)、Extrapolator
  (Aslam/Bochkov)、ShiftedFaceMarker；14 单测含 3 组 C++ 串行交叉
  验证 —— 解锁 shifted/ 三 miniapp
- `fem_solver::adjoint::time_dependent` — TimeDependentAdjointOperator
  对位 (自研 Nordsieck BDF 伴随 + 检查点二分, 8 单测; 过程中发现并
  规避 bdf.rs k≥2 残差缺 l1 因子的既有缺陷)
- 向量 H1 弹性 LOR (space/lor.rs LorVecH1 + solver/lor.rs 块对角
  LOR-AMG 路径, 25 测试绿) — lor_elast miniapp 半成品在
  miniapps/solvers/lor_elast.rs 未声明 (FGMRES 收敛未验证)
- `fem_assembly::ad` — 双数自动微分 (MFEM `linalg/dual.hpp` +
  `miniapps/autodiff/admfem.hpp` 1:1)：Dual 类型 + QFunction/
  QVectorFunc 驱动，13 个单测对 FD/解析导数 <1e-12；示例
  autodiff_example (pLaplacian Newton)
- `fem_mesh::kdtree` — MFEM `fem/kdtree.hpp` KDTreeNodalProjection
  移植 (KdTree + KdTreeNodalProjection, 10 单测)；
  nodal-transfer miniapp 串行部分待接
- `fem_space` hex P≥3 dof 修复 — `build_pk_hex` 与 HexQk GLL 装配基
  对齐（边 slot 枚举/方向、面块序、边 dof GLL 坐标、三线性映射顶点
  序 4 处错误；修复前单元 P3 52/64 slot 错、interpolate 误差 8e-3，
  修复后 <1e-12；5 个新回归测试 hex_p3_gll_repro）
- `fem_linalg::dense::CholeskyFactors` — MFEM `CholeskyFactors` 镜像：
  列主序 Cholesky–Crout 分解 + `LMult`（x←L·x），与 C++ 逐位同序
- `fem_assembly::standard::WhiteGaussianNoiseDomainLFIntegrator`（真随机
  版，替换占位实现）+ `Assembler::assemble_white_gaussian_noise` — MFEM
  白噪声 RHS：libstdc++ `default_random_engine`(=minstd_rand0) +
  `normal_distribution`(polar，含缓存态) 逐位复现，`generate_canonical`
  2 抽取语义、seed≡0(mod 2³¹−1)→1 规则，逐元 Cholesky(质量阵)·L·噪声；
  MFEM `LinearForm::Assemble` 元素循环 1:1
- `fem_assembly::Assembler::mass_element_matrix` — 单元质量阵（列主序，
  供白噪声积分器消费）
- `fem_examples::rational_approximation` — ex33.hpp AAA 有理近似共享模块
  （自 mfem_ex33/mfem_pex33 上提合并，三处共用）
- `fem_assembly::tmop_form` — TMOP 非线性 form：精确导数能量/梯度/Hessian
  （走 `fem_mesh::tmop::metrics` 的 EvalP/AssembleH）、理想形目标
  (UnitSize/EqualSize/GivenSize)、`tmop_newton_solve`（MFEM
  TMOPNewtonSolver 的能量/min-det/残差线搜索 1:1）
- `H1Space::p_refine_update` / `with_element_orders` — MFEM
  `PRefineAndUpdate` / `SetElementOrder+Update`；DofManager variant 式
  变阶边/面 DOF（MFEM `var_edge_dofs` 语义, 最小规则约束）
- `fem_space::lor::{LorNd, LorRt}` — 向量空间 LOR：细化网格上的 ND1/RT0
  + 带符号 dof 置换（MFEM `LORBase::GetDofPermutation`）；
  `build_lor_ams_nd_*` / `build_lor_ads_rt_hex` = `LORSolver<HypreAMS/ADS>`
- `fem_io` 读取 MFEM 弯曲网格 `nodes` 段的高阶 H1 几何（此前被丢弃），
  按 `Mesh::SetVerticesFromNodes` 语义解码顶点
- DofManager hex Qk 全局编号对齐 MFEM `FiniteElementSpace::Construct`
  分块序（顶点→全部边→全部面→全部内部）

上一轮新增（保留）：

- `fem_mesh::amr::general_refinement` — 任意 scale 的 NC quad 细分
  (`Refinement{index,type,scale}`，MFEM `Mesh::GeneralRefinement`)
- `fem_mesh::Mesh::make_cartesian_2d_sfc` — MFEM `MakeCartesian2D`
  默认的 Hilbert SFC 元素序 (`amr::sfc_ordering::grid_sfc_ordering_2d`)
- `fem_mesh::transformation::find_points` — 串行 FindPoints (暴力 + Newton)
- `fem_io` VisIt DC 根文件解析支持 C++ 写出的嵌套 JSON 布局
- `GridFunction::ref_elem_vol` 覆盖 Hex/Prism 全阶

## 命名约定

| C++ 文件 | Rust 文件 |
|---------|----------|
| `tmop-check-metric.cpp` | `tools/tmop_check_metric.rs` |
| `mesh-optimizer.cpp` | `meshing/mesh_optimizer.rs` |
| `lorentz.cpp` | `electromagnetics/lorentz.rs` |
| `ref321.cpp` | `meshing/ref321.rs` |
| `get-values.cpp` | `tools/get_values.rs` |
| `findpts.cpp` | `gslib/findpts.rs` |
| `field-diff.cpp` | `gslib/field-diff.rs` |
| `field-interp.cpp` | `gslib/field-interp.rs` |
| `schwarz_ex1.cpp` | `gslib/schwarz_ex1.rs` |
| `generate_random_field.cpp` | `spde/generate_random_field.rs` |

## 运行

```bash
cargo run --example tmop_check_metric
cargo run --example tmop_metric_magnitude -- -mid 7 -pv 2.0 -par 0.5 -ps 4.0
cargo run --example lorentz
cargo run --example toys_automata
cargo run --example toys_life -- -nx 10 -ny 10 -sp "3 3 0 1 1 1 2 1 1 1"
cargo run --example toys_lissajous -- -no-vis
cargo run --example toys_mondrian -- -i ../../mfem/miniapps/toys/australia.pgm -m data/inline-quad.mesh
cargo run --example toys_mandel -- -m data/inline-quad.mesh
cargo run --example mesh_shaper -- -m data/inline-quad.mesh
cargo run --example mesh_extruder -- -m data/inline-quad.mesh -nz 4 -hz 2.0
cargo run --example mesh_ref321 -- -mm -dim 2 -r 100 -no-vis
cargo run --example tools_display_basis -- -e 2 -b 3 -o 3
cargo run --example tools_get_values -- -r <DC 根路径> -p "x y z ..."
cargo run --example tools_lor_transfer -- -m data/inline-quad.mesh -o 2 -no-vis
cargo run --example spde_generate_random_field -- -m data/ref-cube.mesh -r 2 -rp 1 -no-vis -no-rs
cargo run --example gslib_findpts -- -m data/rt-2d-q3.mesh -o 8 -mo 4 -no-vis
cargo run --example gslib_findpts -- -m data/inline-quad.mesh -o 3 -pr -no-vis
cargo run --example gslib_findpts -- -m data/inline-hex.mesh -o 3 -random 1 -npt 4 -no-vis
```
