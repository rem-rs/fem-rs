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
│   └── volta.rs
├── diag-smoothers/          ← 对应 miniapps/diag-smoothers/
│   └── abs-l1-jacobi.rs     ← Absolute L(1)-Jacobi 光滑子 (1:1 串行版;
│                                mass/diffusion/maxwell 三类系统, SLI/PCG,
│                                abs_global + L(p,q) 元素级对角, Kershaw 网格;
│                                -a 0/1 下迭代日志/ARF/L2 与 C++ 逐行一致,
│                                2D 与 Kershaw 全对比逐位一致;
│                                maxwell 3D hex 差 HexNDk 归一化 4×,
│                                C++ PARTIAL/NONE 的矩阵免费 AbsMult 为缺口)
├── nurbs/                   ← 对应 miniapps/nurbs/ (6 个已完成)
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
├── toys/                    ← 对应 miniapps/toys/ (5 个已完成)
├── fluids/                  ← 对应 miniapps/fluids/ (未开始)
└── ...
```

## 本轮新增核心库能力 (fem-rs crates)

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
