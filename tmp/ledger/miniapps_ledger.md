# miniapps 三态台账（round 61 · C 路 · 抽样复核）

> **底账**：`miniapps/README.md`（1248 行过程台账：67 处 parity 记录、31 处 `exit(3)` 字面标注
> + 若干 `exit 3` 文字标注 ≈ 37）。本轮不重审全部 92 个 miniapp 文件（round 30 已做过 180 项
> 全量判定、后续轮已逐项修复），只做 **BIT 档抽样复核**：重跑 Rust 侧并与"历史 C++ 真值"
> （存储于 `tmp/d371`、`tmp/d533`、README 数字记录）或**现编 C++ 4.10** 对照。

## 抽样复核结果（8 项全部复核，7 项仍绿、1 项带路径坑仍绿）

| # | miniapp | 复核档 | 对照物 | 结果 | 证据 |
|---|---|---|---|---|---|
| 1 | `solvers/lor_solvers.rs`（miniapp_lor_solvers） | `-fe h` 默认 star / `-m inline-quad` / `-o 2` 三档 | README 记录的 C++ 4.10 数字 | **仍绿（3/3 字节同）**：781/0.000395471、625/5.56315e-06、289/0.000245071 | tmp/ledger/logs/sample_lor_*.log |
| 2 | `meshing/mesh_bounding_boxes.rs` | triple-pt-1 `-o 2 --jacobian` + fichera-q2 `-o 2 --jacobian` | tmp/d371/cpp_triple.out、cpp_fichera.out（D371 C++ 真值） | **仍绿（2/2 逐字节）**；唯一差异 = Options 里 mesh 路径串（跑点不同）；det bounds 0.0699225/1.19096、nodal bounds 全同 | logs/sample_bbox_triple.log、sample_bbox_fichera.log |
| 3 | `gslib/schwarz_ex1.rs` | 默认档（square-disc + inline-quad） | README/plan 记录（95 迭代、rtol 1e-8） | **仍绿**：95 次迭代（0..94）、末次残差 8.324015e-09 与记录一致；`$HOME/work/schwarz/cpp_schwarz.txt` 存的是 -m1/-m2 显式档（2 步假收敛档），两档均已文档化 | logs/sample_schwarz.log |
| 4 | `solvers/block_solvers.rs` | `-o 0 -solver bp`、`-o 0 -solver bp-pcg`、`-o 1 -solver bp-pcg` | plan D370 验收记录（C++ MPI oracle，$HOME/work/d370） | **仍绿（3/3）**：o0 L2 4.797124e-2 = C++ 0.0479712（6 位）、o1 bp-pcg 迭代 66=66、u_err 5.561564e-5 | logs/sample_block_*.log |
| 5 | `diag-smoothers/mg_abs_l1_jacobi.rs` | `-a 0 -rs 3 -gl 1 -ol 1`（ref-cube） | plan D376 验收记录（C++ MPI oracle） | **仍绿（逐字节级）**：35937 unknowns、迭代 0 = 154.905、迭代 19 = 3.00892e-19（20 步全轨迹）、ARF 0.285073、L2 2.66219e-05 | logs/sample_mg_abs_l1.log |
| 6 | `nurbs/nurbs_mesh_info.rs`（mini_nurbs_mesh_info） | `-m data/square-nurbs.mesh` | **本轮现编** C++ 4.10 nurbs_mesh_info（$HOME/work/r61ref/nurbs_mesh_info） | **仍绿（全新逐字节）**：stdout 完全一致（36 unknowns 档） | logs/sample_mesh_info.log vs ref/mi_square.out |
| 7 | `nurbs/nurbs_patch_ex1.rs` | `ball -ref 2 -iro 10 -patcha`（九档之一） | tmp/d533/ball_rint_netlib_oracle.log（netlib BLAS C++ oracle） | **仍绿（逐字节）** | logs/sample_patch_ball_rint.log |
| 8 | `nurbs/nurbs_patch_ex1.rs`（默认档冒烟） | `-m data/beam-hex-nurbs.mesh -no-vis` | —（九档 parity 见 #7 与 D531/D533/D553 记录） | rc=0（补丁装配 + 求解完成）；注意 `-patcha -rint`/netlib 规则：**凡 NNLS oracle 必须用 netlib BLAS**（D563 纪律） | logs/sample_patch_ex1.log |

**抽样结论**：README 的 BIT 档记录在本轮工作树上**全部可复现**；无一名不副实回潮。

## 抽样过程中的环境坑（登记，不属 miniapp 缺陷）

- 多个 miniapp 默认网格路径以**仓库根**为基准（`data/…`）或**miniapps 子目录**为基准
  （`../../data/…`）两套并存：从任意 CWD 直接跑都会踩 `NotFound`。本轮统一在
  `tmp/ledger/rundir/`（data junction）+ 显式 `-m` 解决。属**可用性问题**非 parity 问题；
  `block_solvers` 默认 `../data/star.mesh` 同理。
- `data/ref-cube.mesh`、`data/triple-pt-1.mesh`、`data/fichera-q2.mesh` 不在仓库 `data/`
  （本轮从 `$HOME/mfem410_ser` 取用）——与 examples 侧 D633 同族（见 examples_ledger）。

## miniapps 总体三态基线（引用，不重算）

- **总量**：`examples/Cargo.toml` 注册 177 个 example 目标 = 86（examples/，见 examples_ledger）
  + 91（miniapps/）。
- **底账状态**（`miniapps/README.md` + round 30 全量审计 + 后续修复轮）：
  - 真 1:1 / BIT 档：nurbs 4（ex1/ex3/ex5/ex24）+ patch_ex1 + mesh_info、meshing 5
    （mesh-optimizer/hpref/phpref/ref321/fit-node-position）+ mesh_bounding_boxes、
    dpg 6/6、electromagnetics maxwell、solvers（lor_solvers/plor_solvers/lor_elast/
    block_solvers/mg_abs_l1）、diag-smoothers（abs-l1-jacobi）、fluids navier×8 +
    schrodinger_flow、gslib 4/7（findpts/schwarz_ex1/field-diff/field-interp）、
    tools（display-basis/nodal-transfer/lor-transfer）、shifted×3、toys（automata/life）、
    autodiff、hooke、multidomain×3、spde、hdiv×2 等；
  - 诚实裁剪 exit(3)+缺口清单：≈37 处（README 逐条标注；`-vis`/`-visit`/并行分支/
    非默认网格/未移植积分器等）；
  - round 30 的 30 个"(e) 名不副实"在 round 31+ 已全部转正或 exit(3) 文档化
    （D128-D132/D139/D140 关闭记录）。
- **本轮未重审**：上述清单之外的逐项三态复核（建议后续轮按目录抽摊；本轮抽样 8/8 绿
  说明 README 底账可信度高）。
