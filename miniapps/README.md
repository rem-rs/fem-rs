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
│   ├── shaper.rs            ← 材料界面 AMR (1:1, 编译+运行通过)
│   └── extruder.rs          ← 2D→3D 拉伸 (1:1, 编译+运行通过)
│   └── (twist/klein-bottle/toroid/trimmer/reflector: 需低层 API)
├── toys/                    ← 对应 miniapps/toys/
│   ├── automata.rs          ← 1D 元胞自动机 (1:1, 编译+运行通过)
│   ├── life.rs              ← Conway 生命游戏 (1:1, 编译通过)
│   ├── lissajous.rs         ← Lissajous 旋转曲面 (核心逻辑 1:1, 文件输出受限)
│   ├── mondrian.rs          ← PGM 图片→AMR 网格 (1:1, 编译+运行通过)
│   ├── mandel.rs            ← Mandelbrot AMR (核心逻辑 1:1, Quad4 退化)
│   └── (snake/rubik/spiral: 需 3D 低层 API 或外部文件)
├── fluids/                  ← 对应 miniapps/fluids/
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
cargo run --example toys_automata
cargo run --example toys_life -- -nx 10 -ny 10 -sp "3 3 0 1 1 1 2 1 1 1"
cargo run --example toys_lissajous -- -no-vis
cargo run --example toys_mondrian -- -i ../../mfem/miniapps/toys/australia.pgm -m data/inline-quad.mesh
cargo run --example toys_mandel -- -m data/inline-quad.mesh
cargo run --example mesh_shaper -- -m data/inline-quad.mesh
cargo run --example mesh_extruder -- -m data/inline-quad.mesh -nz 4 -hz 2.0
```
