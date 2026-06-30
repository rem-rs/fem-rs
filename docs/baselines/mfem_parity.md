# MFEM 对照基线

## 目标

为 Poisson / Elasticity / Maxwell / Stokes 建立 setup / assemble / solve 三段性能基线，
scale 从 10K DOF 到 1M DOF，作为后续优化的对照。

## 运行方式

```bash
cargo bench -p fem-benches --bench mfem_parity
```

## 基线数据

### Poisson 2D H¹ P1 — Assembly + PCG + Jacobi

| DOF | Assembly+Solve (ms) | Date |
|-----|-------------------|------|
| 25 | 0.11 | 2026-06-30 |
| 81 | 0.43 | 2026-06-30 |
| 289 | 58.1 | 2026-06-30 |
| 625 | 131.0 | 2026-06-30 |

### Elasticity 2D VectorH¹ P1 — Assembly only

| DOF | Assembly (µs) | Date |
|-----|--------------|------|
| 50 | 113 | 2026-06-30 |
| 162 | 262 | 2026-06-30 |
| 578 | 658 | 2026-06-30 |

### Maxwell 2D HCurl ND1 — Assembly only

| DOF | Assembly (µs) | Date |
|-----|--------------|------|
| 56 | 75.8 | 2026-06-30 |
| 208 | 222 | 2026-06-30 |
| 800 | 945 | 2026-06-30 |

### Stokes 2D Mixed VectorH¹ P2 × H¹ P1 — Assembly only

| Velocity DOF | Pressure DOF | Assembly (µs) | Date |
|-------------|--------------|--------------|------|
| 98 | 16 | 312 | 2026-06-30 |
| 338 | 49 | 674 | 2026-06-30 |
| 722 | 100 | 1060 | 2026-06-30 |

## 硬件环境

| 项目 | 值 |
|-----|-----|
| CPU | Intel(R) Core(TM) Ultra 9 285K |
| 核心/线程 | 24C24T |
| RAM | 64 GB |
| OS | Windows 11 |
| Rust | stable (channel) |
