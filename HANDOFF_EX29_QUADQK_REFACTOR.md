# 交接: MFEM ex29 (曲面 Poisson) — QuadQk [0,1]² 重构 ✅ 已完成

## 状态
**QuadQk [0,1]² 重构已完成。** L² 误差保持在 0.283（与重构前一致，因为数学上积分是不变量）。
要将 L² 误差降至 0.0014（C++ MFEM 水平），需排查组装层其他问题。

## 已完成修改
| 修改 | 文件 |
|------|------|
| QuadQk `dof_coords()` 映射到 `[0,1]²` | `crates/element/src/lagrange/factory.rs` |
| QuadQk `eval_basis()` 输入映射 `[0,1]→[-1,1]` | 同上 |
| QuadQk `eval_grad_basis()` + 链式法则 ×2 | 同上 |
| QuadQk `eval_hessian()` + 链式法则 ×4 | 同上 |
| QuadQk `quadrature()` 使用 `quad_rule_01` | 同上 |
| 新增 `quad_rule_01` 和 `quad_rule_01_arbitrary` | `crates/element/src/quadrature.rs` |
| `set_curvature` Q1 插值改为 `[0,1]²` 公式 | `crates/mesh/src/simplex.rs` |
| `set_curvature` 边缘变化坐标检测（0/1 边界） | 同上 |
| `dof_manager.rs` 内部 DOF bilinear 映射 `[0,1]²` | `crates/space/src/dof_manager.rs` |
| ex27 `child_refs` 从 `[-1,1]` 改为 `[0,1]` | `examples/mfem_ex27_robin_bc.rs` |
| ex29 `surface_jacobian` QuadQ1 路径加映射 | `examples/mfem_ex29_curved_poisson.rs` |
| 测试：对比 `QuadQk` 与 `QuadQ1/QuadQ2` 时映射坐标 | `crates/element/src/lagrange/factory.rs` |

## 未解决问题（需后续排查）
1. **ex29 L² 误差 0.283 → 0.0014**: 一致的对 `[0,1]²` 约定是必要条件，但不够。建议排查：
   - 表面 Jacobian `det_j` 计算与 MFEM `FiniteElement::Jacobian3D` 的对比
   - `surface_jacobian_quad4` 与 `surface_jacobian` 中 `QuadQk` 路径是否一致
   - 协变度量 G = JᵀJ 和逆变梯度 ∇\_s = J · G⁻¹ · ∇\_ref 的准确性
   - 检查 `quad_rule_01` 的 `n` 点是否足够（`(2*order+1).max(3+3)`=7 应需要 n=4）
2. **p-refinement 延长算子**：`build_prolongation_same_mesh` 中使用 `f_ref.dof_coords()`（GLL 位置）
   与 DofManager 的等距边缘 DOF 位置不一致。这导致无限细格点上的 p-延长误差，建议将边缘 DOF 位置
   也改为 GLL 分布，或在 `build_prolongation_same_mesh` 中用物理坐标→参考坐标反转。

## L² 误差详情（无网格加密，Q3，mesh-type 2）
```
|u - u_h|_2 = 2.826e-01
|f - f_h|_2 = 6.515e-01
```
C++ MFEM 参考: `|u - u_h|_2 = 1.4e-03`
