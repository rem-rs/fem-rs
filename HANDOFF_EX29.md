# Handoff: MFEM ex29 (Curved-surface Poisson)

## 状态：❌ 代码可编译运行，但 L² 误差不匹配

### 已完成
- ✅ 修复 API 方法名（`geom_elem_nodes`→`geometry_nodes`, `geom_node_coords`→`geom_coords_of`）
- ✅ C++ ex29 参考输出：48 DOFs, PCG+GSSmoother 7 次迭代, |u-u_h|₂=0.001386

### 剩余差距
Rust L² 误差 0.422（C++ 0.001386），相差 ~300×。

### 已知问题
- 示例手动实现曲面扩散组装（surface_jacobian + sigma tensor）
- 虽然梯度变换 `J·G⁻¹·∇_ref` 数学上等价于 MFEM 的 `adj(G)·Jᵀ·∇_ref / det(G)³`，但实际数值结果不同
- DOF 排序、积分精度或 sigma 张量应用可能有问题

### 推荐修复
使用 `Assembler::assemble_bilinear` 的 `is_surface=true` 路径，并实现自定义投影后的 2×2 sigma 系数。
