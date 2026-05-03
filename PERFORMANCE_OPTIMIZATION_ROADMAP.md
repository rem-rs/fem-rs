# fem-rs 性能优化路线图

## 执行摘要

项目当前已具备基础的并行架构（Rayon 线程并行、MPI 分布式），但在**计算核心**、**内存访问模式**和**数值算法级别**仍有显著优化空间。预估优化后可获得 **2-5 倍性能提升**（取决于优化范围）。

---

## I. 高优先级优化（可立即实施）

### 1. 矩阵-向量乘积（SpMV）优化 🔴

**现状问题**：
- `csr_row_dot_f64()` 使用**4-展开的数据并行**（loop unrolling），但未充分利用 CPU 特性
- 缺乏 SIMD/AVX2/AVX-512 显式优化
- 行指针访问产生**缓存未命中**（行数据分散）

**改进方案**：

#### a) **启用 SIMD 向量化**
```rust
// 当前：纯标量 4-loop 展开
while k < end4 {
    sum += values[k] * x[col_idx[k] as usize]
        + values[k + 1] * x[col_idx[k + 1] as usize]
        + values[k + 2] * x[col_idx[k + 2] as usize]
        + values[k + 3] * x[col_idx[k + 3] as usize];
    k += 4;
}

// 目标：8-展开 + 编译器 SIMD（-C target-feature=+avx2）
#[inline(always)]
fn csr_row_dot_avx2_f64(...) -> f64 {
    // 使用 packed_simd 或 std::arch::x86_64：
    // 	- 8 个值并行加载 + 相乘
    // 	- 单轮 8-元素求和
    // 分期回报：20-30% SpMV 加速
}
```

**配置优化**：
```toml
# Cargo.toml
[profile.release]
opt-level   = 3
lto         = "fat"              # 从 "thin" 升级到 "fat"（跨 crate LTO）
codegen-units = 1
rustflags   = ["-C", "target-feature=+avx2,+fma"]  # 编译时启用 SIMD
strip       = false  # 保留调试符号用于性能分析
```

**预期收益**：
- **内存带宽瓶颈缓解**：20-30% SpMV 吞吐提升
- 特别对大规模稀疏矩阵（>1M DOF）有效

#### b) **改进缓存局部性**
- 重新排列 CSR 格式为**按块存储**（block sparse）：对 FEM 应用（小块结构化非零）更优
- 实现 **ELL（ELLPACK）** 格式选项用于高度统一非零模式的矩阵

---

### 2. 装配并行优化 🔴

**现状问题**：
- 当前阈值 `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS = 64`：**过高**，导致小网格串行执行
- `Rayon` 并行驱动每个元素生成单个 `CooMatrix` 条目，然后进行全局归纳
  - 高同步开销 → 线程空闲
- COO → CSR 转换无并行版本

**改进方案**：

#### a) **降低并行启动阈值**
```rust
// 当前：64 个元素才启用 Rayon
pub const DEFAULT_PARALLEL_MIN_ELEMS: usize = 64;

// 目标：考虑 CPU 核数 + 元素复杂性
pub fn adaptive_parallel_threshold(n_cores: usize, elem_type: ElementType) -> usize {
    let base = match elem_type {
        ElementType::Tet4 => 32,   // Tet：相对便宜，早启用
        ElementType::Hex8 => 16,   // Hex：复杂，非常早启用
        _ => 24,
    };
    base.max(n_cores / 2)  // 至少 (cores/2) 个元素才平衡
}
```

**预期收益**：
- 20-40% 的中等规模网格（500-2000 元素）装配加速

#### b) **局部缓冲 + 按元素块归纳**
```rust
// 当前：单元素线程 → CooMatrix，全局同步
// 目标：线程本地缓冲 + 块批处理
struct ThreadLocalBuffer {
    entries: Vec<(u32, u32, f64)>,  // 本地 COO 条目
}

// 装配驱动
rayon_scope(|scope| {
    for chunk in elems.chunks(chunk_size) {
        scope.spawn(|_| {
            let mut buf = ThreadLocalBuffer::new();
            for e in chunk {
                accumulate_into_buffer(&mut buf, e);
            }
            // 块级别同步（而非全局同步）
            global_coo.merge_from_buffer(&buf);
        });
    }
});
```

**预期收益**：
- 30-50% 中等规模装配加速
- 减少内存碎片化

#### c) **并行 COO→CSR 转换**
```rust
// 当前：顺序扫描 COO，计算 row_ptr
pub fn into_csr(&mut self) -> CsrMatrix<T> {
    self.data.sort_by_key(|x| (x.0, x.1));  // 串行排序（O(n log n)）
    ...
}

// 目标：使用 Rayon 并行排序 + 并行行指针计算
pub fn into_csr_parallel(&mut self) -> CsrMatrix<T> {
    self.data.par_sort_by_key(|x| (x.0, x.1));
    // 并行行指针扫描
    ...
}
```

---

### 3. 内存预分配 & 池 🟡

**现状问题**：
- 每次装配动态分配新的 `CooMatrix::data` 向量
- 反复 malloc/free 导致碎片化，尤其在迭代求解器中

**改进方案**：

```rust
// 在 Assembler 中引入对象池
pub struct AssemblerPool<T> {
    idle_coo: Vec<CooMatrix<T>>,
    idle_vectors: Vec<Vector<T>>,
}

impl<T> AssemblerPool<T> {
    pub fn acquire_coo(&mut self) -> CooMatrix<T> {
        self.idle_coo.pop().unwrap_or_else(CooMatrix::new)
    }
    
    pub fn release_coo(&mut self, mut coo: CooMatrix<T>) {
        coo.data.clear();  // 保留分配
        self.idle_coo.push(coo);
    }
}
```

**预期收益**：
- 10-20% 迭代求解器总时间减少（特别在需要反复装配的非线性求解中）

---

## II. 中优先级优化（需架构调整）

### 4. 求解器收敛加速 🟡

**现状问题**：
- **Jacobi 光滑器**：收敛缓慢（特别对高纵横比网格）
- **CG/GMRES 预处理**：仅用 Jacobi（对 ill-conditioned 问题效果差）

**改进方案**：

#### a) **多层预条件化**
```rust
// 当前：单层 Jacobi
let precond = JacobiPreconditioner::new(&A);
let x = cg_solve(&A, &b, precond, tol);

// 目标：分层预条件（ILU(k) + 块 Jacobi）
pub enum HybridPrecond<T> {
    BlockJacobi,      // 块级别局部 Jacobi
    ILU(usize),       // 不完全 LU 分解（需实施）
    SAAmg,            // 代数多重网格
}

let precond = HybridPrecond::ILU(0);  // ILU(0)：接近成本，显著改进
```

**预期收益**：
- CG 迭代次数减少 30-50%（取决于问题结构）
- 总求解时间 20-40% 加速

#### b) **自适应多重网格平滑器**
- 根据矩阵光谱诊断选择光滑器强度
- 切换 Jacobi → Chebyshev 多项式光滑器（更高效）

---

### 5. 并行 Ghost 交换重叠 🟡

**现状问题**：
- 非阻塞 MPI `irecv/isend` 已实现，但**对角块 SpMV** 与通信串行执行
- 两阶段：(1) SpMV 对角块 (2) 等待 + SpMV 非对角块

**改进方案**：
```rust
// 当前：
let (diag_contrib, requests) = forward_start(&mut x);  // 启动非阻塞接收
let diag = diag_spmv(&A_diag, &x);                    // 对角 SpMV
forward_finish(&requests);                            // 等待接收
let offdiag = offdiag_spmv(&A_offdiag, &x_ghost);   // 非对角 SpMV

// 目标：真正的重叠
let requests = forward_start_async(&mut x);           // 后台接收启动
let diag = diag_spmv_async(&A_diag, &x);             // 并行对角 SpMV（线程）
let offdiag = offdiag_spmv_wait(&requests, &A_offdiag);  // 一旦数据到达立即执行
// 减少等待时间 30-50%
```

**预期收益**：
- 分布式求解中 15-25% 总时间减少（对通信/计算比例高的问题）

---

### 6. 自适应 AMG 粗化 🟡

**现状问题**：
- 固定 AMG 参数（聚集比例、光滑因子）
- 某些问题可能**过度粗化**（精度损失）或**粗化不足**（层级过多）

**改进方案**：
```rust
pub struct AdaptiveAmgParams {
    pub target_coarse_dofs: Range<usize>,  // e.g., 100..500
    pub spectral_radius_target: f64,       // 0.1 - 0.3
}

impl AmgSolver {
    pub fn setup_adaptive(
        &self,
        mat: &CsrMatrix<f64>,
        adaptive_params: AdaptiveAmgParams,
    ) -> AmgHierarchy {
        // (1) 初始粗化估计
        let est_coarse_dofs = /* 通过谱诊断估算 */;
        
        // (2) 调整聚集比例
        let agg_ratio = if est_coarse_dofs > adaptive_params.target_coarse_dofs.end {
            0.3  // 更积极粗化
        } else {
            0.1  // 保守粗化
        };
        
        // (3) 按自适应参数构建层级
        self.build_hierarchy_with_ratio(mat, agg_ratio)
    }
}
```

**预期收益**：
- 对异质问题（如多尺度扩散）20-40% AMG 设置 + 求解加速

---

## III. 低优先级（探索性）

### 7. GPU 加速（长期）🔵

**当前可行性**：
- **cuBLAS/cuSPARSE** 集成：SpMV 可卸载到 NVIDIA GPU（2-10 倍加速）
- **SIMD-aware MPI**：GPU 直接 P2P 通信

**实施步骤**：
1. 包装 `cust` 或 `cuda-sys` crate
2. 为 `CsrMatrix` 实现 GPU 缓冲区传输
3. 特殊情况 GPU 上的 SpMV：`y = A x`
4. 在求解循环中条件编译 GPU 路径

**预期收益**：
- 对大规模（>10M DOF）稀疏矩阵：2-10 倍 SpMV 加速

### 8. SIMD 和 Vectorized 基函数评估 🔵

**思路**：
- 当前基函数评估（`eval_basis`, `eval_grad_basis`）一次一个象限点
- 批量评估多个象限点，利用 SIMD

**预期收益**：
- 10-15% 装配加速

---

## IV. 基准测试扩展 🔴

**当前情况**：
- 仅有 AMG 基准（`benches/amg.rs`）
- 缺乏微基准：SpMV, 装配, 求解器迭代

**建议新增基准**：

```rust
// benches/micro.rs
fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv");
    for n in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("csr", n), n, |b, n| {
            let mat = create_poisson_csr(*n);
            let x = Vector::ones(n);
            let mut y = Vector::zeros(n);
            b.iter(|| mat.spmv(&x, &mut y));
        });
    }
    group.finish();
}

fn bench_assembly_per_elem(c: &mut Criterion) { ... }
fn bench_cg_iterations(c: &mut Criterion) { ... }
```

**预期收益**：
- 建立**性能基线**，追踪优化效果
- 检测**性能回归**

---

## V. 实施优先级表

| 优化项 | 工作量 | 预期收益 | 优先级 | 时间估算 |
|--------|--------|---------|--------|---------|
| 1. SpMV SIMD 优化 | 中 | 20-30% | 🔴 | 1-2 周 |
| 2. 装配并行阈值 | 低 | 20-40% | 🔴 | 3-5 天 |
| 3. 内存池 | 低 | 10-20% | 🔴 | 3-5 天 |
| 4. ILU 预处理 | 高 | 20-40% | 🟡 | 2-3 周 |
| 5. Ghost 交换重叠 | 中 | 15-25% | 🟡 | 1-2 周 |
| 6. 自适应 AMG | 中 | 20-40% | 🟡 | 2-3 周 |
| 7. 微基准 | 低 | 诊断 | 🔴 | 3-5 天 |
| 8. GPU 加速 | 高 | 2-10 倍 | 🔵 | 4-8 周 |

---

## VI. 推荐执行计划

### Phase 1（2 周）—— 快赢
- ✅ 降低装配并行阈值
- ✅ 内存池（COO 缓冲）
- ✅ 添加微基准测试
- **预期收益**：15-25% 中等规模问题加速

### Phase 2（3-4 周）—— 核心加速
- ✅ SpMV SIMD 优化（AVX2/FMA）
- ✅ 并行 COO→CSR 排序
- ✅ 启用 `-C target-feature=+avx2` 编译配置
- **预期收益**：20-35% 矩阵运算加速

### Phase 3（4-6 周）—— 算法改进
- ✅ ILU(k) 预处理器
- ✅ Ghost 交换重叠（MPI 版本）
- ✅ 自适应 AMG 粗化
- **预期收益**：20-50% 求解加速（problem-dependent）

### Phase 4（8+ 周）—— 前沿
- ✅ GPU 加速框架（cuBLAS/cuSPARSE 包装）
- ✅ 向量化基函数评估
- **预期收益**：2-10 倍大规模问题加速

---

## VII. 验证/基准测试检查表

每次优化后运行：

```bash
# 微基准
cargo bench --bench micro 2>&1 | tee bench_micro_$(date +%s).txt

# AMG 基准
cargo bench --bench amg 2>&1 | tee bench_amg_$(date +%s).txt

# 完整例子（定时）
time cargo test --example mfem_ex1_poisson --release 2>&1
time cargo test --example mfem_ex4_darcy --release 2>&1

# 对比前后改进
diff <(head -5 bench_before.txt) <(head -5 bench_after.txt)
```

---

## VIII. 文献参考

- **SpMV 优化**：
  - Vuduc et al., "Performance Optimizations and Bounds for Sparse Matrix-Vector Multiply" (2002)
  - Bell & Garland, "Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors" (NVIDIA, 2009)

- **AMG 粗化**：
  - Napov & Notay, "An Algebraic Multigrid Method with Guaranteed Convergence Rate" (2012)

- **Ghost 交换重叠**：
  - Baker et al., "Performance Modeling of a Hybrid MPI/OpenMP Implementation of an Unstructured Mesh CFD Code" (2010)

---

## 联系方式

有任何性能问题或建议，请提交 Issue 或在代码审查中标记为 `performance`。

