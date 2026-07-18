# Handoff: MFEM ex29 (Curved-surface Poisson)

## 状态：QuadQk [0,1]² 重构已完成 ✅

**根因已修复：** QuadQk 参考域已从 `[-1,1]²` 改为 `[0,1]²`，与 MFEM `H1_FECollection` 一致。
参见 `HANDOFF_EX29_QUADQK_REFACTOR.md` 获取详细修改清单。

## 剩余问题：L² 误差 0.283 vs C++ 0.0014

数学上积分应不受参数化影响（det(J)·dω 不变性），但 ex29 仍差 ~200×。
需排查组装层以下问题：

1. **表面 Jacobian 精度**：对比 `surface_jacobian` 中的 QuadQk 路径与 MFEM `FiniteElement::Jacobian3D`
2. **协变/逆变映射**：G = JᵀJ 和 ∇_s = J·G⁻¹·∇_ref 的准确性
3. **积分点数**：qo 使用 `(2*order+1).max(3+3)` = 7，需 n=4 点 (2n-1≥7)，`quad_rule_01` 够用
4. **边界条件处理**：Dirichlet BC 消除是否与 MFEM 一致
