# miniapps

对应 MFEM `miniapps/` 目录，按子目录组织。

## 目录结构

```
miniapps/
├── tools/                   ← 对应 miniapps/tools/
│   ├── tmop_check_metric.rs
│   ├── tmop_metric_magnitude.rs
│   └── gridfunction_bounds.rs
├── electromagnetics/        ← 对应 miniapps/electromagnetics/
│   ├── lorentz.rs
│   ├── tesla.rs
│   └── volta.rs
├── nurbs/                   ← 对应 miniapps/nurbs/
├── meshing/                 ← 对应 miniapps/meshing/
├── toys/                    ← 对应 miniapps/toys/
├── fluids/                  ← 对应 miniapps/fluids/
├── dpg/                     ← 对应 miniapps/dpg/
└── ...
```

## 命名约定

| C++ 文件 | Rust 文件 |
|---------|----------|
| `tmop-check-metric.cpp` | `tools/tmop_check_metric.rs` |
| `mesh-optimizer.cpp` | `meshing/mesh_optimizer.rs` |
| `lorentz.cpp` | `electromagnetics/lorentz.rs` |

## 运行

```bash
cargo run --example tmop_check_metric
cargo run --example tmop_metric_magnitude -- -mid 7 -pv 2.0 -par 0.5 -ps 4.0
cargo run --example lorentz
```
