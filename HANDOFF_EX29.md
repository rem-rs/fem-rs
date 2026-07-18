# Handoff: MFEM ex29 (Curved-surface Poisson)

## 最终根因：Quad 参考域不匹配 [0,1]² vs [-1,1]²

**C++ MFEM** 对 Quad 元素使用 **[0,1]²** 参考域。
**Rust QuadQk** 使用 **[-1,1]²** 参考域。

虽然几何节点位置一致（都在 GLL 位置），但基函数梯度的参数化不同：
- 梯度在 [0,1] 与 [-1,1] 之间相差 **2×** 缩放因子
- 导致 Q3 Jacobian 完全不同（不仅幅度，符号也不同）

### 已修复的基础设施
| 修复 | 文件 | 说明 |
|------|------|------|
| GLL 边缘节点 | `mesh/src/simplex.rs` | `set_curvature` 改用 GLL 位置 |
| GLL 参考元素 | `assembly/src/assembler.rs`, 示例 | `ref_elem_vol` 改用 `QuadQk` |
| API 方法名 | 示例 | `geom_elem_nodes`→`geometry_nodes` 等 |

### 需要进一步修复
- `QuadQk` 基函数需适配 [0,1]² 参考域（或加入 `map_to_01` 缩放）
- 影响所有 Quad 元素的相关代码路径
