# MFEM 对照基线

## 目标

为 Poisson / Elasticity / Maxwell / Stokes 建立 setup / assemble / solve 三段性能基线，
scale 从 10K DOF 到 1M DOF，作为后续优化的对照。

## 运行方式

```bash
cargo bench -p fem-benches --bench mfem_parity
```

## 基线数据

### Poisson 2D H¹ P1 — PCG + Jacobi preconditioner

| DOF     | h       | Assembly (ms) | Solve (ms) | Total (ms) | Date       |
|---------|---------|---------------|------------|------------|------------|
| (待采集) |         |               |            |            |            |

### Elasticity 2D VectorH¹ P1 — PCG + Jacobi

| DOF     | Assembly (ms) | Solve (ms) | Total (ms) | Date       |
|---------|---------------|------------|------------|------------|
| (待采集) |               |            |            |            |

### Maxwell 2D HCurl ND1 — PCG + Jacobi

| DOF     | Assembly (ms) | Solve (ms) | Total (ms) | Date       |
|---------|---------------|------------|------------|------------|
| (待采集) |               |            |            |            |

### Stokes 2D Mixed HDiv×L² — MINRES

| DOF     | Assembly (ms) | Solve (ms) | Total (ms) | Date       |
|---------|---------------|------------|------------|------------|
| (待采集) |               |            |            |            |

## 硬件环境

| 项目       | 值           |
|-----------|--------------|
| CPU       | (待记录)      |
| 核心/线程  |              |
| RAM       |              |
| OS        |              |
| Rust      | (待记录)      |
