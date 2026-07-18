# 交接: MFEM ex29 — 最终状态

## 已修复的基础设施
| 修复 | 文件 | 说明 |
|------|------|------|
| QuadQk [-1,1]²→[0,1]² | `factory.rs` | 对齐 MFEM H1_FECollection |
| `node_to_dof` 边界检测 | `factory.rs` | 节点排序对齐 build_pk_quad |
| GLL 边缘节点 | `simplex.rs` | set_curvature 改用 GLL |
| `ref_elem_vol` QuadQk | `assembler.rs` | GLL 节点对齐 |
| API 方法名 | ex29 | geom_elem_nodes→geometry_nodes |

## 剩余误差: 0.283 vs C++ 0.0014

通过逐个元素矩阵比对确认：
- C++ 与 Rust 有 **80/256 个条目符号不匹配**
- 即使 QuadQk 已对齐 [0,1]²，DOF 排序、基函数评估都验证正确
- 中心 Jacobian 仍差 1.3×，RHS 差 6.6×
- 几何节点位置一致，梯度计算数学正确，但数值不同

## 一句话启动提示词
```text
在 C++ 中输出 H1_FECollection(3,2) 的 GetNodes() 完整节点引用坐标 (16 个)，与 Rust QuadQk::all_dof_coords() 逐项比对，找到第一个位置或顺序差异，然后修复 QuadQk::node_to_dof 的索引映射以完全匹配 C++。
```
