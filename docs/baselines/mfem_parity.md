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

## 分布式可扩展性

### RAS-PCG (ILU0, overlap=1) — 强可扩展性（4225 DOF 固定）

| ranks | iters | time (ms) | 效率 |
|-------|-------|-----------|------|
| 1 | 33 | 92.6 | 100% |
| 2 | 43 | 80.7 | 57% |
| 4 | 40 | 50.5 | 46% |
| 8 | 41 | 43.6 | 27% |

### RAS-PCG — 弱可扩展性（每核 ~1000−1000 DOF）

| ranks | DOF | iters | time (ms) | growth |
|-------|-----|-------|-----------|--------|
| 1 | 1089 | 18 | 14.2 | 1x |
| 2 | 2116 | 31 | 35.5 | 2.5x |
| 4 | 4225 | 40 | 53.3 | 3.8x |
| 8 | 8464 | 60 | 96.2 | 6.8x |

### Schur-GMRES (MultifrontalLu) — 强可扩展性（2401 DOF 固定）

| ranks | iters | time (ms) | 效率 |
|-------|-------|-----------|------|
| 1 | 159 | 228.5 | 100% |
| 2 | 167 | 191.5 | 60% |
| 4 | 117 | 108.4 | 53% |
