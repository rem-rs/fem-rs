# Handoff: MFEM ex29 (Curved-surface Poisson)

## 状态：❌ L² 误差不匹配（~300×）

### 已修复的基础设施
| 修复 | 文件 | 说明 |
|------|------|------|
| API 方法名 | `mfem_ex29_curved_poisson.rs` | `geom_elem_nodes`→`geometry_nodes` 等 |
| `geom_coords_of` 类型适配 | 同上 | `&[f64]` → `[f64;3]` |
| Quad 参考节点对齐 | `assembler.rs`, 示例 | 等距→GLL (`QuadQk`) |

### C++ 参考
- 48 DOFs, PCG+GSSmoother, 7 次迭代
- |u-u_h|₂ = 0.001386, |f-f_h|₂ = 0.007977

### Rust 结果
- 48 DOFs ✅, PCG+GSSmoother ✅
- |u-u_h|₂ = 0.422 (300×), |f-f_h|₂ = 0.582

### 已排除的原因
- ❌ DOF 排序（`build_pk_quad` vs `QuadQ3`/`QuadQk`）：已验证一致
- ❌ 参考节点位置（等距 vs GLL）：已修复，但未改善结果
- ❌ 梯度变换数学公式（`J·G⁻¹·∇_ref` vs MFEM `adj(G)·Jᵀ·∇_ref/det(G)`）：已验证等价

### 未排除的原因
- `DofManager::dof_coord()` 对曲面网格使用线性插值，未使用 Q3 几何节点 → 影响 `||b-A*u_exact||` 测试但不影响 FE 解
- 手动 `surface_jacobian` 实现可能有细节错误（如梯度方向、sigma 投影）
- 建议：重写 ex29 改用 `Assembler::assemble_bilinear` 的 `is_surface=true` 路径 + `TensorDiffusionIntegrator`，传入投影后的 2×2 sigma
