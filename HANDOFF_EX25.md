# Handoff: MFEM ex25 (PML Maxwell) — 完成

## 状态：✅ 1:1 翻译完成，核心库已补充

### 核心库新增
| 设施 | 位置 | 说明 |
|------|------|------|
| `SesquilinearForm` | `crates/assembly/src/complex.rs` | 复数双线性形式, `add_domain_integrator_pair` |
| `Convention` | `crates/assembly/src/complex.rs` | Hermitian/BlockSymmetric 约定 |
| `RestrictedCoefficient` | `crates/assembly/src/postproc/coefficient.rs` | 标量系数按 element attribute 限制 |
| `VectorRestrictedCoefficient` | 同上 | 矩阵系数按 attribute 限制 |
| `ScalarVectorProductCoefficient` | 同上 | 标量×矩阵系数 |
| `GsPrecond` | `vendor/linger/src/precond/sor.rs` | 纯对称 GS (匹配 MFEM GSSmoother) |
| `project_bdr_coefficient_tangent_2d` | `crates/assembly/src/postproc/grid_function.rs` | 2D HCurl 边界切线投影 |

### 已修复 Bug
1. GsPrecond backward sweep 多除 D[i] — 已修
2. 非 PML 质量矩阵缺 -ω²ε 缩放 — 已修
3. BC 消除仅行归零 → 行列消除 — 已修
4. GSSmoother 用 SSOR（多对角缩放）→ GsPrecond — 已修
5. 系统矩阵缺 BC 消除 — 已修
6. PML 内边界条件未置零 — 已修
7. prob=4 源项在实部→虚部 — 已修
8. GMRES 收敛判据对齐 MFEM（预条件后相对残差）— 已修

### 未推送
网络不通。主仓库 25 提交、`vendor/linger` 子模块 7 提交待推送。

### 收敛差距（待 linlvo 层排查）
- prob=1 square-disc o1 ref0: C++ 251 iters, Rust 394 iters (1.57x)
- 代码逻辑已完全对齐，差距在 linlvo GMRES/GS 内部实现

---

## 下一步: ex26 (Geometric Multigrid)

### 现状
`examples/mfem_ex26_geom_mg.rs` 已存在，199 行。求解 Poisson `−Δu=1` 使用几何多重网格预条件器。需要确认编译运行。

### C++ 参考
`~/works/mfem/examples/ex26.cpp` (WSL)

### 关键比对点
- `DiffusionMultigrid` 多重网格层次构建
- `ChebyshevSmoother` 平滑器
- `PCG` + 多重网格预条件器
- `refined.mesh` + `sol.gf` 输出
- 平均缩减因子 `Average reduction factor`

### 启动提示词
> "推进 MFEM ex26 (geometric multigrid) 的 1:1 翻译比对。确认 `examples/mfem_ex26_geom_mg.rs` 的当前状态，获取 C++ 参考源 `~/works/mfem/examples/ex26.cpp`，编译运行 C++ 版本获取参考输出，然后逐项比对 Rust 版本，补充核心库缺失的 MG 基础设施。"
