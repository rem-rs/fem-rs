# fem-rs Crate 改进计划（2026-05-04）

本文把以下四类改进整理为一个可执行的仓库内计划：

- fem-solver：ILU/ILUT 预条件、Block-GMRES、FGMRES 落地扩展
- fem-element：高阶三角积分统一、Tri6 曲线三角几何工具
- fem-mesh：Tri6 网格导入与持久化链路
- fem-linalg：H-matrix 树结构基础设施（可选）

目标不是重复建设仓库里已经存在的能力，而是基于当前状态补齐通用接口、数据通路和验证闭环，使 rem2 与 fem-rs 主线能力共用同一套基础组件。

## 1. 当前状态核查

### 1.1 fem-solver

已存在的基础：

- 标准 GMRES：`solve_gmres`
- FGMRES：`solve_fgmres`、`solve_fgmres_jacobi`
- ILU(0) 预条件：`solve_pcg_ilu0`
- ILDLt、块预条件、Schur complement、MINRES 已存在

当前缺口：

- 缺少 `GMRES + ILU(0)` 的直接入口，非对称系统仍主要靠无预条件 GMRES 或调用方自行拼接
- 没有 ILUT / 可配置填充级别 ILU(k) API
- 没有多右端项 Block-GMRES / batched Krylov 接口
- FGMRES 已有算法入口，但缺少面向“可变预条件器”的标准封装与典型 AMG 内迭代接法

结论：solver 方向不是从零开始，应当视为“补预条件家族 + 扩 Krylov 工作流 + 接入应用”。

### 1.2 fem-element

已存在的基础：

- `tri_rule(order)` 已提供：
  - 1 点规则
  - 3 点规则
  - 7 点 Dunavant 规则（degree 5 exact）
  - 12 点规则（degree 6 exact）
- `TriP2` 已有完整二次三角参考单元基函数与梯度

当前缺口：

- 三角积分规则虽然存在，但还不是一个“可枚举、可命名、可精确选择”的统一高阶规则表 API
- 若 rem2 需要 7/9 阶精度的固定表，当前 `tri_rule(order)` 仍偏“按阶选近似规则”，不利于跨项目一致性
- Tri6 的几何映射/Jacobian 计算目前更偏 mesh 层曲线映射工具，缺少 element 层可复用的几何辅助接口

结论：element 方向的核心不是重新实现 P2 三角形函数，而是统一高阶积分表和几何评估工具边界。

### 1.3 fem-mesh

已存在的基础：

- `ElementType::Tri6` 已定义
- Gmsh type=9 已映射到 `Tri6`
- `CurvedMesh::elevate_to_order2` 已支持 Tri3 -> Tri6 的二次几何提升

当前缺口：

- Gmsh 读取链路最终落到 `MshFile -> SimplexMesh<2>`，Tri6 连接关系能被识别，但没有形成一等公民的“曲线高阶网格对象”导出路径
- 缺少从 Gmsh 二次网格直接生成 `CurvedMesh<2>` 或等价高阶几何对象的标准 API
- 缺少 Tri6 `.msh` v2/v4 回归样例、导入测试、写回测试

结论：mesh 方向不是“能否识别 type=9”，而是“Tri6 数据能否贯穿导入、存储、几何使用、再导出”。

### 1.4 fem-linalg

已存在的基础：

- 稀疏矩阵、块矩阵、向量、张量等常规线性代数设施齐全

当前缺口：

- 未发现 H-matrix / cluster tree / admissibility / ACA 相关基础结构
- 说明该方向基本属于新子系统建设，而不是补现有薄层

结论：linalg 方向应明确标为可选和后置，避免与 solver/mesh/element 的高收益项目争抢优先级。

## 2. 优先级排序

建议按以下顺序推进：

1. fem-solver：ILU/FGMRES 接口补齐
2. fem-element：高阶三角积分 API 统一
3. fem-mesh：Tri6 导入到高阶几何对象
4. fem-solver：Block-GMRES 多右端项
5. fem-element：Tri6 几何评估工具
6. fem-linalg：H-matrix 树结构（探索性）

排序原因：

- solver 改动能最快反哺 rem2 driven/eigenmode 与 fem-rs 自身非对称线性系统
- element 与 mesh 的 Tri6 事项互相依赖，但积分规则统一的边际成本更低、复用更广
- Block-GMRES 有明确价值，但接口设计、内存布局和验收方式比 ILU/FGMRES 更复杂
- H-matrix 是新系统，应在前述高收益项目稳定后再做

## 3. 分阶段实施计划

### Phase 1：solver 快速补强（1 到 2 周）

范围：

- 增加 `solve_gmres_ilu0`
- 增加通用 `solve_gmres_prec` / `solve_fgmres_prec` 风格入口
- 为 AMG / 内迭代预条件器定义稳定包装接口
- 补充非对称基准：Jacobi vs ILU0 vs FGMRES+AMG

交付物：

- 公共 API：统一的 GMRES/FGMRES + preconditioner 入口
- 最少 3 组回归测试：
  - 对流扩散非对称系统
  - H(curl) 或 saddle-point 近似非对称系统
  - AMG 作为可变预条件器的 FGMRES 收敛测试
- 迭代次数/残差基准表

验收标准：

- `GMRES+ILU0` 相比无预条件 GMRES 在至少一个非对称问题上显著降迭代数
- FGMRES 在“每步预条件作用不恒定”的测试中稳定收敛
- 新 API 不破坏现有 `solve_gmres` / `solve_fgmres` 调用面

备注：

- 这一步已经覆盖了你列出的“FGMRES 改进方向”的大部分真实缺口，因为算法本身已存在，真正需要的是接口和验证闭环。

### Phase 2：element 三角积分统一（3 到 5 天）

范围：

- 将 `tri_rule(order)` 重构为“规则表 + 选择器”
- 明确命名规则，例如：`TriangleRule::Dunavant7PtDeg5`、`TriangleRule::Dunavant12PtDeg6`
- 若 rem2 需要更高精度，补入 9 阶精度目标对应的固定表
- 统一测试单项式精确积分阶次

交付物：

- 命名化三角积分规则 API
- 与现有 `ReferenceElement::quadrature(order)` 兼容的适配层
- 文档说明每个规则的点数、精确阶、适用场景

验收标准：

- `TriP1/TriP2/TriP3` 与 Nedelec/RT 三角单元都经由统一规则表取积分
- 对单项式 `x^i y^j` 的精确积分测试覆盖到声明精度
- rem2 可删除自维护三角积分表或只保留兼容层

### Phase 3：mesh Tri6 导入链路打通（1 到 2 周）

范围：

- 在 `fem-io` + `fem-mesh` 间补一个“高阶网格返回类型”路径
- 支持从 `.msh` v2/v4 的 Tri6 直接构造 `CurvedMesh<2>` 或等价结构
- 明确 boundary element 的 `Line3` 处理和边界标签保留策略
- 增加 Tri6 fixture、导入测试、可能的写回测试

交付物：

- 新 API，例如：
  - `MshFile::into_curved_2d()`
  - 或 `CurvedMesh::from_msh(...)`
- 至少 2 个 Gmsh Tri6 fixture：v2、v4
- 标签、节点顺序、Jacobian 正性测试

验收标准：

- Tri6 网格导入后，中点节点位置与连接顺序正确
- 曲线单元的几何 Jacobian 在参考域内为正且有限
- 与线性 `SimplexMesh` 导入路径并存，不破坏旧 API

备注：

- 这里的关键不是 parser 是否接受 type=9，而是导入后能否保留“二次几何语义”。

### Phase 4：Tri6 几何工具下沉到 element（4 到 7 天）

范围：

- 提炼 element 层的几何评估工具，而不是把 Jacobian 逻辑散落在 mesh/assembly
- 基于 `TriP2` 基函数增加几何辅助，例如：
  - 几何插值
  - 参考到物理映射
  - Jacobian / inverse Jacobian / detJ

交付物：

- `fem-element` 中的 Tri6 几何辅助模块
- `fem-mesh::CurvedMesh` 对这些工具的复用接线
- rem2 可直接复用的替代接口说明

验收标准：

- 同一组 Tri6 节点坐标下，element 几何工具与现有 `CurvedMesh` 几何结果一致
- assembly 层不再复制 Tri6 Jacobian 计算逻辑

### Phase 5：Block-GMRES 多右端项（2 到 3 周）

范围：

- 设计多右端输入输出数据结构
- 实现 block Arnoldi、block Hessenberg、小规模 least-squares 求解
- 处理列相关/秩亏 deflation
- 先支持右预条件版本，再评估是否补 flexible 版本

交付物：

- `solve_block_gmres` 公共 API
- 多右端项基准：共享同一矩阵、多个端口激励/载荷
- 与串行逐 RHS 求解的性能和鲁棒性对比

验收标准：

- 多个相近 RHS 时，总 matvec 次数或总 wall time 明显优于逐个 GMRES
- 单 RHS 情况不回退到更差表现
- 内存开销与 restart 参数关系有明确文档

备注：

- 该项更适合作为 solver 的二阶段增强，不建议先于 Tri6 导入链路实施。

### Phase 6：ILUT / ILU(k) 扩展（1 到 2 周）

范围：

- 在 `ILU0` 之上扩展出可配置填充或阈值丢弃策略
- 统一预条件器构造配置，例如：`PrecondKind::Ilu0 | Iluk(k) | Ilut{drop_tol, fill}`

交付物：

- ILU 家族配置对象
- 稳定性/内存占用/构造时间对比基准

验收标准：

- 至少一个难例中 ILUT 比 ILU0 进一步减少迭代数
- 构造成本与收益存在清晰边界，不把 API 做成难以调参的黑盒

备注：

- 如果底层 linger 已经提供现成能力，则此项优先做封装和验证；否则再考虑在 fem-rs 层自行实现。

### Phase 7：H-matrix 树结构预研（可选，2 到 4 周）

范围：

- 只做 cluster tree、bounding boxes、admissibility、block cluster tree
- 暂不承诺完整 H²/ACA 求解链

交付物：

- 独立模块：树构建与块划分
- 至少一个离线 demo：点集生成 block cluster tree
- 为后续 ACA/H-matrix 做数据结构基座

验收标准：

- 树构建复杂度和内存占用可测
- 数据结构不绑定 FEM 特定场景，可复用于 BEM/边界积分

## 4. 依赖关系

推荐依赖图：

1. Phase 1 先做，立即提升现有求解能力
2. Phase 2 与 Phase 3 可并行，但 Phase 3 的几何校验会受益于 Phase 2 的统一积分规则
3. Phase 4 建立在 Phase 3 之上最自然
4. Phase 5 依赖 Phase 1 的预条件器抽象结果
5. Phase 6 可插到 Phase 1 之后任意时点，但建议在 `GMRES+ILU0` 先稳定后再做
6. Phase 7 最后做

## 5. 推荐的仓库拆分方式

### fem-solver

- 第一批 PR：`solve_gmres_ilu0` + 通用 preconditioned GMRES/FGMRES 入口 + 测试
- 第二批 PR：预条件器配置枚举与 AMG 可变预条件包装
- 第三批 PR：Block-GMRES
- 第四批 PR：ILUT / ILU(k)

### fem-element

- 第一批 PR：三角积分规则命名化与测试
- 第二批 PR：Tri6 几何评估辅助

### fem-mesh / fem-io

- 第一批 PR：Tri6 fixture + parser/import API + 回归测试
- 第二批 PR：高阶网格到 assembly/space 的接线样例

### fem-linalg

- 单独 feature branch 做 H-matrix 预研，不和主线求解器/网格改造混在一起

## 6. 风险与规避

### 风险 A：把“已有能力”重复实现一遍

规避：

- solver 直接复用已有 FGMRES / ILU0
- element 直接复用已有 `TriP2` 与 `tri_rule`
- mesh 直接复用已有 `Tri6` 和 `CurvedMesh`

### 风险 B：Tri6 解析了，但几何语义丢失

规避：

- 设计单独高阶网格返回类型
- 强制做 Jacobian 正性和中点节点顺序测试

### 风险 C：Block-GMRES 首版过重

规避：

- 首版只做 dense RHS block + 右预条件 + 固定 restart
- 不在首版加入 flexible block GMRES

### 风险 D：H-matrix 方向吞噬主线精力

规避：

- 明确列为 optional
- 以独立原型和基准为前提，不先承诺用户接口稳定性

## 7. 建议里程碑

### M1：两周内

- `GMRES+ILU0`
- 通用 preconditioned GMRES/FGMRES API
- 三角积分规则命名化

### M2：四周内

- Tri6 `.msh` v2/v4 导入到高阶几何对象
- Tri6 几何 Jacobian 测试

### M3：六周内

- Block-GMRES 首版
- ILUT / ILU(k) 可行性结果

### M4：探索性

- H-matrix 树结构原型和基准

## 8. 最终建议

如果只选三个最值得马上放入 fem-rs 主线的工作项，建议是：

1. fem-solver：补齐 `GMRES/FGMRES + 通用预条件器` 接口
2. fem-mesh + fem-element：打通 Tri6 从 Gmsh 导入到几何使用的完整链路
3. fem-element：把高阶三角积分规则变成统一规则表 API

原因很直接：

- 这三项都已经有现成基础，投入小于从零开发
- 它们既服务 rem2，也服务 fem-rs 主线高阶/曲线/非对称求解问题
- 它们比 H-matrix 更接近当前仓库的主路径和验证能力