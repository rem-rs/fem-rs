# miniapps

对应 MFEM `miniapps/` 目录，按子目录组织。

## 目录结构

```
miniapps/
├── tools/                   ← 对应 miniapps/tools/
│   ├── tmop_check_metric.rs
│   ├── tmop_metric_magnitude.rs
│   ├── gridfunction_bounds.rs
│   ├── load-dc.rs             ← VisIt DC 加载 (1:1)
│   ├── compare-dc.rs          ← DC 比对 (1:1)
│   ├── display_basis.rs       ← 基函数展示 (无 GLVis; H1/ND/RT/L2,
│   │                            vsize 与 C++ 逐位一致, 32/34 组合;
│   │                            hex L2 ≥P2 为 fem-rs 缺口)
│   ├── get_values.rs          ← DC 场采样 (依赖 find_points;
│   │                            与 C++ 输出逐位一致)
│   └── lor_transfer.rs        ← LOR 传输 (简化 H1+pointwise 版;
│                                HO/R(HO)/LOR 质量与 C++ 逐位一致)
├── electromagnetics/        ← 对应 miniapps/electromagnetics/
│   ├── lorentz.rs
│   ├── tesla.rs
│   ├── volta.rs
│   └── maxwell.rs           ← 全波 Maxwell 偶极子脉冲 (1:1 串行; ND/RT 空间 +
│                                SIAV 辛积分): H(Curl) 12336 / H(Div) 11520 dof
│                                与 C++ 相同, 100 步 40 条 Energy(<t>ns) 行 +
│                                banner + Options dump 与 C++ **逐字节**(主会话
│                                独立复核: 78 行输出仅差 mesh 路径字符串与
│                                Maximum Time Step); dtmax 行因 hypre
│                                Randomize(1234) 种子不可复刻而不同
│                                (0.141749→0.145761ns, 同 SnapTimeStep 档);
│                                -vis/-visit/-cs/-abcs/NURBS/2D 为 exit(3);
│                                joule 仅出判定(需要 MakeRef 视图 + 4 块耦合
│                                隐式解 + MFEM ODE 族 + 静态凝聚)
├── diag-smoothers/          ← 对应 miniapps/diag-smoothers/
│   └── abs-l1-jacobi.rs     ← Absolute L(1)-Jacobi 光滑子 (1:1 串行版;
│                                mass/diffusion/maxwell 三类系统, SLI/PCG,
│                                abs_global + L(p,q) 元素级对角, Kershaw 网格;
│                                -a 0/1 下迭代日志/ARF/L2 与 C++ 逐行一致,
│                                2D 与 Kershaw 全对比逐位一致;
│                                maxwell 3D hex 差 HexNDk 归一化 4×,
│                                C++ PARTIAL/NONE 的矩阵免费 AbsMult 为缺口)
├── nurbs/                   ← 对应 miniapps/nurbs/ (4 个 1:1 + 2 个部分移植/exit 3)
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
│   └── nurbs_ex5.rs / nurbs_ex24.rs ← **部分移植（退出码 3）**: 依赖的
│                                `NurbsHDivSpace`（H(div) NURBS 空间）已在
│                                round 26 落地，两个示例重写为「已核对部分
│                                逐字节 + 未移植部分 exit(3) + 文件头双标注」:
│                                ex5 默认档 dim(R)=8580/dim(W)=4225/
│                                dim(R+W)=12805/边界 dof H(div) 260 / H1 256
│                                与 C++ 逐字节；ex24 `-r 1 -p 0/1/2` 的
│                                HCurl144/H127、HCurl144/HDiv108、
│                                HDiv108/L227 六行逐字节。缺: 带符号
│                                `bel_dof`（ex5 自然 BC 的 RHS）、块 MINRES
│                                的 Schur 通路（ex5 迭代块）、NURBS 跨空间
│                                `MixedVectorGradient/CurlIntegrator`（ex24）
├── meshing/                 ← 对应 miniapps/meshing/
│   ├── shaper.rs            ← 材料界面 AMR (1:1)
│   ├── extruder.rs          ← 2D→3D 拉伸 (1:1)
│   ├── twist.rs / klein-bottle.rs / toroid.rs / trimmer.rs / reflector.rs
│   ├── mesh-explorer.rs / mesh-quality.rs
│   ├── polar-nc.rs          ← 极坐标 NC 网格
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
├── gslib/                   ← 对应 miniapps/gslib/
│   └── findpts.rs           ← FindPointsGSLIB 找点/插值 (纯 Rust:
│                                BVH + 等参元 Newton, code 0/1/2 与
│                                dist² 语义对齐; glibc rand 逐位复现
│                                随机点; 13 个数值用例 counts 与 C++
│                                全对齐, max_err ~1e-15; -surf/-mpr/
│                                -hr/-ft 2/3 及 NC/mixed/pyramid
│                                网格裁剪 exit 3)
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
│                                (mfem49 串行), 对照走分段 C++ harness)
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
├── solvers/plor_solvers.rs  ← LOR 求解器 miniapp (H1 空间串行
│                                版本; PCG + LOR-AMG 预条件; 2D
│                                inline-quad.mesh 默认)
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
