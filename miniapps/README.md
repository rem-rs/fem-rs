# miniapps

对应 MFEM `miniapps/` 目录，按子目录组织。

## 目录结构

```
miniapps/
├── tools/                   ← 对应 miniapps/tools/（**round 30 审计**：11 个可执行里只有
│                                display-basis/nodal-transfer 属 (b) 类；其余多为
│                                **(e) 类名不副实**，证据见下与 D129）
│   ├── tmop_check_metric.rs   ← ⚠️ **不是 C++ 同名程序的移植**：C++ 是 `-mid N`
│   │                            单 metric 体检（1000 次随机 T 逐槽 EvalW vs
│   │                            EvalWMatrixForm + dF/ddF 收敛阶），本文件是固定 21 项
│   │                            自检且**静默忽略 `-mid`**（D129）
│   ├── tmop_metric_magnitude.rs ← ⚠️ metric zoo 仅 **22/49** id（缺 85/98/322 等），
│   │                            未知 id `panic!` 而非 C++ 的 `return 3`；输出格式也不同（D129）
│   ├── gridfunction_bounds.rs ← ⚠️ 两列打印**同一数值**（C++ 第二列是真递归收紧界），
│   │                            `-nb/-ref/-bt/-rd/-rt` 全忽略（D129）
│   ├── load-dc.rs             ← ⚠️ C++ 该程序可观察行为只有 GLVis 套接字；本文件
│   │                            自创 6 行输出（含硬编码 `3 (assumed)`）⇒ 已按审计改标注（D129）
│   ├── compare-dc.rs          ← ⚠️ 字段遍历用 `HashMap`（C++ `std::map` 字典序 ⇒
│   │                            顺序逐次不同）、分隔线宽度与多余尾行亦不同（D129）
│   ├── display_basis.rs       ← (b) 基函数展示 (无 GLVis; H1/ND/RT/L2,
│   │                            vsize 与 C++ 逐位一致, 32/34 组合;
│   │                            hex L2 ≥P2 为 fem-rs 缺口; C++ `-no-vis` 只打 4 行 header)
│   ├── get_values.rs          ← ⚠️ **实测失败**：`-r Example23 -p "0 0 0.5"` →
│   │                            `MissingMesh`（`data_collection_load.rs:101` 硬编码
│   │                            `mesh3d`，而 C++ 官方样例 Example5 是 2-D），`-o` 被吞
│   │                            ⇒ 原"与 C++ 输出逐位一致"无可复现命令（D88/D129）
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
│   └── abs-l1-jacobi.rs     ← Absolute L(1)-Jacobi 光滑子 (1:1 串行版;
│                                mass/diffusion/maxwell 三类系统, SLI/PCG,
│                                abs_global + L(p,q) 元素级对角, Kershaw 网格;
│                                -a 0/1 下迭代日志/ARF/L2 与 C++ 逐行一致,
│                                2D 与 Kershaw 全对比逐位一致;
│                                maxwell 3D hex 差 HexNDk 归一化 4×,
│                                C++ PARTIAL/NONE 的矩阵免费 AbsMult 为缺口)
├── nurbs/                   ← 对应 miniapps/nurbs/（**round 30 审计**：15 个可执行中
│                                **4 个真 1:1**（ex1/ex3/ex5/ex24，均有 C++ 对照数字）+
│                                **2 个名不副实待整改**（见下 solenoidal/printfunc）+
│                                **9 个缺失**：ex1p/ex10/ex10p/ex11p/curveint/mesh_info/
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
│   ├── nurbs_solenoidal.rs  ← ⚠️ **名不副实（round 30 审计，(e) 类）**：自称
│   │                            "1:1 port" 但实测走的是 C++ 的 **`-nn`（普通 FE）**
│   │                            档：dim(R)=33024/16384（C++ 默认 NURBS 档 =
│   │                            8580/4225）、缺 `Create NURBS fec and ext` 横幅、
│   │                            缺 `‖div u_h−div u_ex‖` 行（`compute_div_error`
│   │                            在整个 `crates/` 是 0 命中）、L² 9.5586e-05 vs
│   │                            C++ 2.08242e-05、算了 `m_gs/s_gs` 却从不使用
│   │                            ⇒ 重写或 `exit(3)`+缺口清单（D131）
│   └── nurbs_printfunc.rs   ← ⚠️ **格式未达标（round 30 审计，(e) 类）**：数值
│                                早已核过（≤1.1e-15），但 C++ vs Rust
│                                **40/48 行不同**（C++ 6 位有效数字 vs Rust
│                                round-trip）；改用库里已有的
│                                `fem_solver::fmt_g` 即逐字节（D131）
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
│   │                            （C++ 只存 8 位有效数字）。`-o > 1`（含默认 `-o 3`，C++ 产物带
│   │                            `H1_3D_P3` 的 `nodes`）⇒ exit(3)+缺口清单
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
│   ├── mobius-strip.rs      ← ⚠️ **round 31 降级 exit(3)**：C++ 是 **`dimension 2` +
│   │ `klein-bottle.rs`         `Space dimension 3`**（曲面嵌 3-D）+ 非连续高阶 `nodes`
│   │                            （默认 nodes=1；klein 默认 NBE=0）—— **不是 `dimension 3`**；
│   │                            本仓 `Mesh<D>` 把坐标维钉死在拓扑维且无 nodes writer ⇒
│   │                            exit(3)+缺口清单，Options dump 与 C++ **逐行一致**
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
│   │                            线搜索 = TMOPNewtonSolver)
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
├── gslib/                   ← 对应 miniapps/gslib/（**round 30 审计 1/7 → round 31 实质 4/7**）
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
│   ├── field-diff.rs        ← ✅ **round 31 新增**：两网格两场插值对比 (1:1;
│   │                            **`Vol diff` 与 C++ 4.10 逐位相同 1.73608**；
│   │                            `Avg diff` 0.0960922 vs 0.0949062 (+1.2%)；
│   │                            `Max diff` 2.58236 vs 1.43502 —— 差异已定位量化：
│   │                            finder1 [9604 inside, 396 border, 0 not-found] vs
│   │                            finder2 [9599, 396, 5]，**那 5 个近边界点贡献了
│   │                            12.690817 的差和**（点序 4374/4377/4379/4481/4582，
│   │                            其中 4481 的 `v2 = 0` 因 locator 报 code 2）⇒ 不是装配/
│   │                            插值错误，而是 `crates/mesh/src/findpts` 对 0.05% 边际点的
│   │                            border/newton 分类待收口）
│   └── field-interp.rs      ← ✅ **round 31 新增**：源网格场插值到目标网格
│                                (控制台 **4 行与 C++ 逐字相同**；⚠️ 该 miniapp 本身
│                                **没有任何误差行**，全源码只有那 4 行 + 写文件；
│                                `interpolated.gf` 写回**仅前 20/169 DOF 对齐**，
│                                从 DOF 20 起值不同且多重集也不同（max|Δ| = 3.4e-1）
│                                ⇒ H1 P3 三角形 DOF 编号/写回次序 WIP，**已写入文件头 doc
│                                而非静默**）
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
├── solvers/lor_solvers.rs   ← ⚠️ **round 31 D140 降格为声明式桩 (exit(3))**：
│                                **不是** `lor_solvers.cpp` 的 port —— 装好 M 又**显式丢弃**
│                                (`_mass`)、只解 K、不 import 任何 LOR/AMG、`-fe` 非 h 曾
│                                `exit(1)`、旧代码打印 `LOR solvers complete` 且 **rc=0**
│                                ⇒ 现 `not_ported()` 在 main 第一行、`exit(3)` + 缺口清单并
│                                **指向 `solvers/plor_solvers.rs`**（同源 1:1 实现，用那个）。
│                                未删文件之理由：`examples/Cargo.toml` 有注册 + 本 README 引用，
│                                且 Cargo.toml 归主会话独占
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
├── adjoint/                 ← 对应 miniapps/adjoint/
│   ├── adjoint_cvodes_roberts.rs ← Robertson 伴随敏感性 (自研
│   │                              Nordsieck BDF 对位 CVODES 语义,
│   │                              检查点二分; 对照 scipy Radau 参考
│   │                              y(4e7)/G/dGdp 1e-4~1e-7 量化一致)
│   └── adjoint_advection_diffusion.rs ← 串行子集; -fd 1 自洽
│                                  (伴随 vs 有限差分 5.8e-7/1.2e-7)
├── toys/                    ← 对应 miniapps/toys/ (5 个已完成)
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
