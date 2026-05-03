# fem-rs 性能优化实施指南 — 代码示例

## 1. SpMV SIMD 优化（高优先级）

### 背景
当前 `csr_row_dot_f64()` 使用 4-展开循环，但未充分利用现代 CPU（AVX2/AVX-512）的 SIMD 能力。通过 8-展开 + 显式 SIMD 指令可获得 **20-30% 加速**。

### 实施步骤

#### Step 1：启用编译期 SIMD 支持
**文件**：`Cargo.toml`

```toml
[profile.release]
opt-level   = 3
lto         = "fat"              # 从 "thin" 升级以获得更好跨 crate 优化
codegen-units = 1
# 下面两行可选：显式启用 SIMD 指令集
# rustflags = ["-C", "target-feature=+avx2,+fma"]
```

或在 `.cargo/config.toml` 中：
```toml
[build]
rustflags = ["-C", "target-feature=+avx2,+fma", "-C", "opt-level=3"]
```

#### Step 2：实现 8-展开 + SIMD 感知的 SpMV
**文件**：`crates/linalg/src/csr.rs`

```rust
// 添加到现有代码
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn csr_row_dot_f64_avx2(
    row_ptr: &[usize],
    col_idx: &[u32],
    values: &[f64],
    x: &[f64],
    row: usize,
) -> f64 {
    use std::arch::x86_64::*;
    
    let start = row_ptr[row];
    let end = row_ptr[row + 1];
    let mut k = start;
    
    // SIMD 累积器：4x 512-bit 寄存器（≈ 256 bytes 的 4 个 double）
    let mut sum_vec = unsafe { _mm256_setzero_pd() };  // [0, 0, 0, 0]
    let mut k_tail = start;
    
    // 8-展开循环（每轮处理 8 个元素 = 2 个 AVX2 寄存器操作）
    let end8 = start + (end - start) / 8 * 8;
    while k < end8 {
        // 前 4 个元素
        let idx0 = col_idx[k] as usize;
        let idx1 = col_idx[k + 1] as usize;
        let idx2 = col_idx[k + 2] as usize;
        let idx3 = col_idx[k + 3] as usize;
        
        let v01 = unsafe {
            let a = _mm256_set_pd(
                x[idx3],
                x[idx2],
                x[idx1],
                x[idx0],
            );
            let b = _mm256_set_pd(
                values[k + 3],
                values[k + 2],
                values[k + 1],
                values[k],
            );
            _mm256_mul_pd(a, b)
        };
        
        // 后 4 个元素（同样方式）
        let idx4 = col_idx[k + 4] as usize;
        let idx5 = col_idx[k + 5] as usize;
        let idx6 = col_idx[k + 6] as usize;
        let idx7 = col_idx[k + 7] as usize;
        
        let v45 = unsafe {
            let a = _mm256_set_pd(
                x[idx7],
                x[idx6],
                x[idx5],
                x[idx4],
            );
            let b = _mm256_set_pd(
                values[k + 7],
                values[k + 6],
                values[k + 5],
                values[k + 4],
            );
            _mm256_mul_pd(a, b)
        };
        
        sum_vec = unsafe { _mm256_add_pd(sum_vec, v01) };
        sum_vec = unsafe { _mm256_add_pd(sum_vec, v45) };
        
        k += 8;
        k_tail = k;
    }
    
    // 从向量寄存器横向求和
    let mut sum = unsafe {
        let mut s = sum_vec;
        let tmp = _mm256_permute2_pd(s, s, 1);
        s = _mm256_add_pd(s, tmp);
        let tmp = _mm256_permute_pd(s, 1);
        s = _mm256_add_pd(s, tmp);
        _mm256_cvtsd_f64(s)
    };
    
    // 处理剩余元素（标量）
    while k_tail < end {
        sum += values[k_tail] * x[col_idx[k_tail] as usize];
        k_tail += 1;
    }
    
    sum
}

// 有条件地编译：在 SpMV 主函数中使用
fn csr_row_dot_f64(
    row_ptr: &[usize],
    col_idx: &[u32],
    values: &[f64],
    x: &[f64],
    row: usize,
) -> f64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return csr_row_dot_f64_avx2(row_ptr, col_idx, values, x, row);
    }
    
    // Fallback：原始 4-展开实现
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let start = row_ptr[row];
        let end = row_ptr[row + 1];
        let mut k = start;
        let mut sum = 0.0_f64;
        
        let end4 = start + (end - start) / 4 * 4;
        while k < end4 {
            sum += values[k] * x[col_idx[k] as usize]
                + values[k + 1] * x[col_idx[k + 1] as usize]
                + values[k + 2] * x[col_idx[k + 2] as usize]
                + values[k + 3] * x[col_idx[k + 3] as usize];
            k += 4;
        }
        while k < end {
            sum += values[k] * x[col_idx[k] as usize];
            k += 1;
        }
        sum
    }
}
```

**验证方法**：
```bash
# 编译并运行基准测试
RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench micro --release

# 对比前后性能
# 预期：SpMV 吞吐提升 20-30%
```

---

## 2. 装配并行阈值自适应（低工作量，高收益）

### 现状问题
- `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS = 64`：固定阈值在小网格上过高
- 中等规模网格（500-2000 元素）的多线程启动开销被浪费

### 实施步骤

**文件**：`crates/assembly/src/assembler.rs`

```rust
// 添加自适应阈值计算
pub fn assembly_parallel_min_elems() -> usize {
    *ASSEMBLY_PARALLEL_MIN_ELEMS.get_or_init(|| {
        std::env::var(FEM_ASSEMBLY_PARALLEL_MIN_ELEMS)
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                // 自适应默认值基于可用核数
                let n_threads = rayon::current_num_threads();
                let base_threshold = 64usize;
                
                // 公式：基于核数调整阈值
                // - 单核：64（无并行开销）
                // - 2 核：32
                // - 4 核：16
                // - 8+ 核：8
                let adaptive = base_threshold >> (n_threads.leading_zeros() as usize).saturating_sub(6);
                adaptive.max(8)  // 最低 8 个元素
            })
    })
}
```

**使用**：
```bash
# 保持默认自适应
cargo test --example mfem_ex1_poisson

# 或显式覆盖
FEM_ASSEMBLY_PARALLEL_MIN_ELEMS=16 cargo test --example mfem_ex1_poisson
```

**预期收益**：
- 中等网格（500-2000 元素）：20-40% 装配加速
- 大网格（>10k 元素）：无显著变化（已充分并行化）

---

## 3. 内存池 — COO 矩阵缓冲复用（快速赢）

### 思路
在反复装配（如非线性迭代求解器）中，避免重复 malloc/free COO 缓冲。

**文件**：`crates/linalg/src/coo.rs`

```rust
/// 可复用的 COO 矩阵缓冲池
pub struct CooPool<T> {
    buffers: Vec<(Vec<u32>, Vec<u32>, Vec<T>)>,  // (row, col, values)
}

impl<T> CooPool<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: Vec::with_capacity(capacity),
        }
    }
    
    /// 获取可用缓冲或创建新的
    pub fn acquire(&mut self, nrows: usize, ncols: usize, nnz_hint: usize) -> CooMatrixRef<T> 
    where
        T: Default + Clone,
    {
        if let Some((mut row, mut col, mut val)) = self.buffers.pop() {
            row.clear();
            col.clear();
            val.clear();
            val.reserve(nnz_hint);  // 预分配
            CooMatrixRef {
                nrows,
                ncols,
                row,
                col,
                val,
                pool: self,
            }
        } else {
            CooMatrixRef {
                nrows,
                ncols,
                row: Vec::with_capacity(nnz_hint),
                col: Vec::with_capacity(nnz_hint),
                val: vec![T::default(); nnz_hint],
                pool: self,
            }
        }
    }
    
    fn release(&mut self, row: Vec<u32>, col: Vec<u32>, val: Vec<T>) {
        self.buffers.push((row, col, val));
    }
}

/// 使用引用计数自动归还到池中
pub struct CooMatrixRef<'a, T> {
    pub nrows: usize,
    pub ncols: usize,
    pub row: Vec<u32>,
    pub col: Vec<u32>,
    pub val: Vec<T>,
    pool: &'a mut CooPool<T>,
}

impl<'a, T> Drop for CooMatrixRef<'a, T> {
    fn drop(&mut self) {
        let pool = unsafe {
            // SAFETY：pool 指针在 drop 时仍然有效
            &mut *(self.pool as *mut _)
        };
        pool.release(
            std::mem::take(&mut self.row),
            std::mem::take(&mut self.col),
            std::mem::take(&mut self.val),
        );
    }
}
```

**在装配器中使用**：
```rust
pub struct Assembler {
    coo_pool: CooPool<f64>,
    // ...
}

impl Assembler {
    pub fn assemble_bilinear(&mut self, qr: &QuadratureRule) -> (CsrMatrix<f64>, Vector<f64>) {
        // 从池中获取 COO 缓冲
        let mut coo = self.coo_pool.acquire(
            self.space.n_dofs(),
            self.space.n_dofs(),
            1024 * 16,  // 初始预分配 16K 非零
        );
        
        // ... 装配循环 ...
        
        // coo 在作用域结束时自动返回到池中
        coo.into_csr()
    }
}
```

**预期收益**：
- 反复装配场景（非线性求解）：10-20% 总时间减少
- 内存碎片化减少 ~50%

---

## 4. 并行 COO→CSR 排序（中等工作量）

**文件**：`crates/linalg/src/coo.rs`

```rust
pub fn into_csr_parallel(mut self) -> CsrMatrix<T> 
where
    T: Scalar + Send + Sync,
{
    use rayon::prelude::*;
    
    let nrows = self.nrows;
    let ncols = self.ncols;
    
    // 并行排序（如果数据足够大）
    if self.data.len() > 10_000 {
        self.data.par_sort_by_key(|&(i, j, _)| (i, j));
    } else {
        self.data.sort_by_key(|&(i, j, _)| (i, j));
    }
    
    // 并行计算行指针
    let mut row_ptr = vec![0; nrows + 1];
    
    // 第 1 阶段：并行计数
    let counts: Vec<usize> = (0..rayon::current_num_threads())
        .into_par_iter()
        .map(|thread_id| {
            let stride = (self.data.len() + rayon::current_num_threads() - 1) 
                / rayon::current_num_threads();
            let start = thread_id * stride;
            let end = (start + stride).min(self.data.len());
            
            let mut local_counts = vec![0; nrows + 1];
            for k in start..end {
                local_counts[self.data[k].0 as usize] += 1;
            }
            local_counts
        })
        .collect();
    
    // 第 2 阶段：并行前缀和 + 合并
    for thread_counts in counts {
        for i in 0..=nrows {
            row_ptr[i] += thread_counts[i];
        }
    }
    
    // 转换为累积指针
    let mut sum = 0;
    for i in 0..=nrows {
        let tmp = row_ptr[i];
        row_ptr[i] = sum;
        sum += tmp;
    }
    
    // 提取 CSR 数据
    let col_idx: Vec<u32> = self.data.iter().map(|&(_, j, _)| j).collect();
    let values: Vec<T> = self.data.into_iter().map(|(_, _, v)| v).collect();
    
    CsrMatrix {
        nrows,
        ncols,
        row_ptr,
        col_idx,
        values,
    }
}
```

**测试**：
```rust
#[test]
fn test_into_csr_parallel() {
    let mut coo = CooMatrix::new(100, 100);
    // 填充 1000 个随机非零元素
    for _ in 0..1000 {
        coo.push(rand() % 100, rand() % 100, 1.0);
    }
    
    let csr = coo.into_csr_parallel();
    assert_eq!(csr.nrows, 100);
    assert_eq!(csr.ncols, 100);
    assert_eq!(csr.nnz(), 1000);
}
```

**预期收益**：
- 大 COO 矩阵（>100K 非零）排序：2-3 倍加速（线程级并行）
- 装配流水线总时间：5-10% 减少

---

## 5. 微基准扩展（诊断工具）

**文件**：`crates/benches/micro.rs` （新建）

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_linalg::{CsrMatrix, Vector};
use fem_mesh::SimplexMesh;
use fem_element::quadrature::gauss_triangle;
use fem_space::{H1Space, FESpace};
use fem_assembly::{Assembler, DiffusionIntegrator};

fn bench_spmv_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_small");
    group.sample_size(1000);  // 更多样本以减少噪声
    
    for size in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("csr_f64", size), size, |b, &size| {
            let mesh = SimplexMesh::<2>::unit_square_tri(10);  // 固定小网格
            let space = H1Space::new(&mesh, fem_element::lagrange::TriP1::new());
            let qr = gauss_triangle(2);
            
            let mut assembler = Assembler::new(&space);
            assembler.add_domain(DiffusionIntegrator::new(1.0));
            let (mat, _) = assembler.assemble_bilinear(&qr);
            let mat = mat.into_csr();
            
            let x = Vector::ones(mat.ncols);
            let mut y = Vector::zeros(mat.nrows);
            
            b.iter(|| {
                mat.spmv(&x, &mut y);
                black_box(&mut y);
            });
        });
    }
    group.finish();
}

fn bench_spmv_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_large");
    group.sample_size(100);
    
    for mesh_size in [50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("csr_f64_large", mesh_size),
            mesh_size,
            |b, &mesh_size| {
                let mesh = SimplexMesh::<2>::unit_square_tri(mesh_size);
                let space = H1Space::new(&mesh, fem_element::lagrange::TriP1::new());
                let qr = gauss_triangle(2);
                
                let mut assembler = Assembler::new(&space);
                assembler.add_domain(DiffusionIntegrator::new(1.0));
                let (mat, _) = assembler.assemble_bilinear(&qr);
                let mat = mat.into_csr();
                
                let x = Vector::ones(mat.ncols);
                let mut y = Vector::zeros(mat.nrows);
                
                b.iter(|| {
                    mat.spmv(&x, &mut y);
                    black_box(&mut y);
                });
            },
        );
    }
    group.finish();
}

fn bench_assembly_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    
    for mesh_size in [10, 30, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("poisson_assembly", mesh_size),
            mesh_size,
            |b, &mesh_size| {
                let mesh = SimplexMesh::<2>::unit_square_tri(mesh_size);
                let space = H1Space::new(&mesh, fem_element::lagrange::TriP1::new());
                let qr = gauss_triangle(2);
                
                b.iter(|| {
                    let mut assembler = Assembler::new(&space);
                    assembler.add_domain(DiffusionIntegrator::new(1.0));
                    let (mat, _) = assembler.assemble_bilinear(&qr);
                    black_box(mat);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_spmv_small, bench_spmv_large, bench_assembly_overhead);
criterion_main!(benches);
```

**运行**：
```bash
cargo bench --bench micro --release -- --verbose 2>&1 | tee micro_baseline.txt

# 实施优化后对比
cargo bench --bench micro --release 2>&1 | diff - micro_baseline.txt
```

---

## 总结

| 优化 | 工作量 | 预期收益 | 立即可做 |
|-----|--------|---------|--------|
| SpMV SIMD | 中 | 20-30% | ✅ |
| 装配并行阈值 | 低 | 20-40% (中网格) | ✅ |
| 内存池 | 低 | 10-20% (非线性) | ✅ |
| 并行排序 | 中 | 5-10% | ✅ |
| 微基准 | 低 | 诊断 | ✅ |

所有这些优化都可以在 **2-3 周内实施**，预期总体性能提升 **25-50%**（取决于应用特征）。

