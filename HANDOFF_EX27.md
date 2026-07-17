# Handoff: MFEM ex27 (Mixed BCs) — 比对结果

## 状态：❌ 核心库 API 不足，需增强后继续

### 已完成

1. ✅ **C++ ex27 编译运行** — DOFs=302, PCG+GSSmoother 29次迭代
2. ✅ **sol_cpp.gf 参考解已保存**
3. ✅ **添加 2D Quad 曲面网格支持** — 移除 `set_curvature` 的 D==3 断言
4. ✅ **端口 C++ trans/quad_trans** — `hole_transform` 函数

### 剩余的差距

C++ 使用 `make_periodic` 后，底部/顶部边界面上周期识别顶点的坐标是"规范位置"（一侧的），不是用于边界积分时所需的"另一侧"位置。这导致跨越周期缝隙的边界面长度计算错误。

修复这个问题的正确方法比较复杂，需要在核心库层面进行。现有的 CsrMatrix/CooMatrix API 不足以便捷地实现周期 DOF 约束。

### 需要增强的 API

| 需求 | 描述 |
|------|------|
| `CooMatrix::from(&CsrMatrix)` | 从 CSR 转为 COO（方便修改后转回 CSR） |
| `CsrMatrix::to_coo()` | 同上 |
| `CsrMatrix::rows()` / `cols()` | 获取矩阵维度 |
| `CsrMatrix::add_entry(row, col, val)` | 直接修改 CSR 中的某一项 |
| `CsrMatrix::set_diagonal(val)` | 设置对角线 |

### 推荐方案

1. 增强 `CooMatrix` + `CsrMatrix` API 以支持便捷修改
2. 示例中：组装 CooMatrix → 添加周期罚项 → 转为 CSR → 求解
3. 或：在 `make_periodic` 中检测并移除跨越周期缝隙的边界面，然后边界辅助函数正确计算周期边界上的边缘长度
