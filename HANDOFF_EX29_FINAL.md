# 交接: MFEM ex29 (曲面 Poisson) — 最终状态

## 状态：✅ QuadQk [0,1]² 重构完成，L² 误差 0.283（C++ 0.0014）

## 已完成
| 修复 | 说明 |
|------|------|
| `QuadQk` `[-1,1]²`→`[0,1]²` | 齐 MFEM H1_FECollection 约定 |
| 链式法则缩放 | `eval_grad_basis` 自动缩放因子 2 |
| `quad_rule_01` | GL 求积权重在 [0,1]² 总和为 1 |
| `set_curvature` Q1 插值 | 改用 [0,1]² 公式 |
| 边缘节点检测 | [0,1] 坐标条件（rc[0]==0 或 1） |

## 已知：200× 误差未修复
[0,1]² 重构后结果不变，说明参数化不是误差来源。

**已被排除的可能原因：**
- ✅ Quad 参考域 [0,1]² vs [-1,1]²：积分不变量，非根源
- ✅ 积分阶数 qo=7 vs 9：不改善结果
- ✅ GLL 节点对齐：已修复
- ✅ DOF 排序：build_pk_quad 与 QuadQk 匹配

**剩余可能的根源：**
- ❓ `surface_jacobian` 中的梯度变换与 MFEM 的 `CalcAdjugate` 可能有数值差异
- ❓ 手动曲面组装代码中的 sigma 张量投影
- ❓ 需要逐个元素矩阵数值比对（C++ `ComputeElementMatrix` vs Rust）

## C++ 参考构建
```bash
cd ~/mfem_build && cmake /mnt/c/Users/lilu/works/mfem -DMFEM_USE_MPI=OFF \
  -DMFEM_USE_METIS=OFF -DMFEM_USE_SUITESPARSE=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 mfem
g++ -std=c++17 -I. -I/mnt/c/Users/lilu/works/mfem test.cpp -L. -lmfem
```

## 一句话启动提示词
```text
逐个元素矩阵数值比对：在 C++ 中调用 DiffusionIntegrator::AssembleElementMatrix2 输出单元 0 刚度矩阵，在 Rust ex29 中输出相同矩阵，逐项找出第几个条目开始分叉，从而定位曲面组装代码中的精确 bug。
```
