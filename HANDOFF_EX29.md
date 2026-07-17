# Handoff: MFEM ex29 (Curved-surface Poisson)

## 状态：❌ 曲面扩散组装 300×误差，需逐个元素矩阵比对

### 已验证
- ✅ 扁平 Quad4 Laplacian 组装：在其他示例中正常工作
- ✅ 梯度变换数学公式：等价于 MFEM
- ✅ GLL 节点对齐：`ref_elem_vol` 已修复
- ✅ 曲面路径代码逻辑：正确使用 `geometry_nodes` (16 Q3 节点，不是 4 顶点)

### 未解决
- `is_surface=true` 路径的曲面组装在所有场景下都产生 ~0.39 的 L² 误差
- 即使是 σ=I（基本 Laplacian）也失败
- 需要逐个元素矩阵比对（C++ `ComputeElementMatrix` vs Rust 输出）才能定位
- C++ 构建环境需要稳定的 WSL 编译链

### 建议
在可构建 MFEM 的环境中：
1. 修改 C++ ex29 在 `a.Assemble()` 后调用 `a.ComputeElementMatrix(0, elmat)` 输出单元 0 矩阵
2. 在 Rust 中从 Assembler 内部提取相同单元矩阵
3. 逐元素比对找出差异
