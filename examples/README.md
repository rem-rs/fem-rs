# fem-rs Examples

Rust 示例与 C++ MFEM 源码一一对应。

## 目录结构

```
fem-pro/fem-rs/
├── examples/                    ← 对应 mfem/examples/
│   ├── mfem_ex*.rs              ← 串行示例（ex0, ex1, ...）
│   ├── mfem_pex*.rs             ← 并行示例（pex0, pex1, ...）
│   ├── compare/
│   ├── src/
│   ├── README.md
│   └── Cargo.toml
├── miniapps/                    ← 对应 mfem/miniapps/（与 examples/ 并列）
│   ├── tools/                   ← 对应 miniapps/tools/
│   │   ├── tmop_check_metric.rs
│   │   ├── tmop_metric_magnitude.rs
│   │   └── gridfunction_bounds.rs
│   ├── electromagnetics/        ← 对应 miniapps/electromagnetics/
│   │   ├── lorentz.rs
│   │   ├── tesla.rs
│   │   └── volta.rs
│   ├── nurbs/                   ← 对应 miniapps/nurbs/
│   ├── meshing/                 ← 对应 miniapps/meshing/
│   ├── toys/                    ← 对应 miniapps/toys/
│   ├── fluids/                  ← 对应 miniapps/fluids/
│   ├── dpg/                     ← 对应 miniapps/dpg/
│   └── ...
└── crates/
```

## 命名约定

| C++ 文件 | Rust 文件 | 说明 |
|---------|----------|------|
| `ex0.cpp` | `mfem_ex0_mesh_intro.rs` | 串行示例，保留描述性后缀 |
| `ex1_poisson.cpp` | `mfem_ex1_poisson.rs` | 串行示例 |
| `pex0_parallel_poisson.cpp` | `mfem_pex0_parallel_poisson.rs` | 并行示例 |
| `tmop-check-metric.cpp` | `miniapps/tools/tmop_check_metric.rs` | miniapp，`-` → `_` |
| `mesh-optimizer.cpp` | `miniapps/meshing/mesh_optimizer.rs` | miniapp |
| `schrodinger_flow.cpp` | `miniapps/fluids/schrodinger_flow.rs` | miniapp |

## 运行示例

```bash
# 串行示例
cargo run --example mfem_ex1_poisson

# 并行示例（需要 MPI）
cargo run --example mfem_pex1_parallel_poisson

# miniapp
cargo run --example tmop_check_metric
cargo run --example tmop_metric_magnitude -- -mid 7 -pv 2.0 -par 0.5 -ps 4.0
```

## 新增示例

1. 确定 C++ 源码路径（如 `miniapps/meshing/mesh-optimizer.cpp`）
2. 在 `examples/miniapps/meshing/` 下创建 `mesh_optimizer.rs`
3. 在 `examples/Cargo.toml` 中注册：
   ```toml
   [[example]]
   name = "mesh_optimizer"
   path = "miniapps/meshing/mesh_optimizer.rs"
   ```
4. 运行 `cargo run --example mesh_optimizer` 验证

## 与 C++ 的对应关系

- `examples/mfem_ex*.rs` ↔ `mfem/examples/ex*.cpp`（串行）
- `examples/mfem_pex*.rs` ↔ `mfem/examples/ex*.cpp`（并行）
- `examples/miniapps/tools/` ↔ `mfem/miniapps/tools/`
- `examples/miniapps/meshing/` ↔ `mfem/miniapps/meshing/`
- `examples/miniapps/toys/` ↔ `mfem/miniapps/toys/`
- `examples/miniapps/fluids/` ↔ `mfem/miniapps/fluids/`
- `examples/miniapps/dpg/` ↔ `mfem/miniapps/dpg/`
- `examples/miniapps/electromagnetics/` ↔ `mfem/miniapps/electromagnetics/`
- `examples/miniapps/nurbs/` ↔ `mfem/miniapps/nurbs/`
- ...
