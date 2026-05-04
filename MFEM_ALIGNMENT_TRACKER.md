# fem-rs MFEM ?对齐踪?档

> ?026-04-13  
> ? MFEM_MAPPING MAXWELL_GAPS 并为踪口记??对齐 + Maxwell 项

---

## 使说?

- ?档 MFEM ?对齐?主踪?档
- ??来源
  - `MFEM_MAPPING.md`??段?跨模差距
  - `MAXWELL_GAPS.md`Maxwell/H(curl) 项差距路线
- ??
  1. 代码?落
  2. ?档步
  3. ??档不以??可证代码 + ?为?并????档

已· ? ? · ? 计?· 不纳

---

## 1. ?对齐来MFEM_MAPPING

### 1.1 已对齐?主干?

- Mesh / 格?D/3D格Tri3/Tet4并??格混?素设
- ??积?H1NDTri/TetRTTri/Tet张积积?Gauss-Lobatto
- 空DOFH1/L2/HCurl/HDiv/H1TraceDOF ??符号并true dof 路
- 积?线线DG-SIPMixedAssembler离子grad/curl/div
- 线代解CSR/COOKrylovAMGSchurMINRESIDR(s)TFQMR
- 并WASMParCsr/ParVector并并AMGWASM worker E2E 证
- 示?Poisson/Maxwell/Darcy/Stokes/Navier-Stokes/并示已形系

### 1.2 差距跨模

- ✅ HDF5/XDMF 并行 I/O 与 restart 文件链路（Phase 55, 2026-05-02）。
  当前进展：`fem-io` 新增 `hdf5` 模块（串行 HDF5 网格+场读写，GZIP 压缩）与 `xdmf` 模块（XML 元数据生成，支持串行/并行）；`fem-parallel` 新增 `par_hdf5` 模块（PerRank/Gather 两种并行写入模式）与 `checkpoint` 模块（ParVector+ParCsrMatrix 分布式 checkpoint 保存恢复）。
  验收证据：`cargo test -p fem-io --features hdf5` 全通过（18 unit + 7 integration）；`cargo check -p fem-parallel --features hdf5` 零错误。
- 🔲 hypre 绑定（可选 FFI 路线）。
- ✅ **Netgen/Abaqus 网格读取支持**（2026-05-04）。`fem-io` 新增 Prism6、Pyramid5、surfaceelements（Netgen `.vol`）及 C3D5、C3D6、mixed C3D4+C3D6（Abaqus）；XDMF 混合拓扑写入器；72 项单元+集成测试通过。`alignment-smoke` CI 套件 `io-mixed-topology` 覆盖。
- ✅ Quad4/Hex8 非协调 AMR（Phase 67）。`refine_nonconforming_quad`、`NCStateQuad`、`refine_nonconforming_hex`、`unit_cube_hex` 已落地；12 个单元测试通过。
- ✅ 静态凝聚 / 杂化 FEM（Phase 68）。`StaticCondensation`、`GlobalBacksolve`、`condense_global` 已落地（`fem-assembly`）；4 个单元测试通过。
- ✅ AMG WP2 分布式跨 rank 聚合（Phase 69）。`ParAmgHierarchy::build_global()` + `build_coarse_level_global` 已落地；新增 3 个集成测试（串行收敛、双 rank 收敛、四 rank 粗化）并通过。
- ✅ NURBS/IGA 参考元与组装（Phase 70）。`KnotVector`、`BSplineBasis1D`、`NurbsPatch2D/3D`、`NurbsMesh2D/3D`、物理域映射与 2D/3D 全局组装已落地（`fem-element::nurbs` + `fem-assembly::iga`）；`cargo test -p fem-assembly --lib iga` 10 项测试全通过。
- ✅ 几何多重网格 / LOR 预条件器（ex26，2026-05-03）。`mfem_ex26_geom_mg.rs` 完成几何 h-多重网格（4 tests）与 LOR 概念演示（3 tests）两条路线：分别构建 P1（LOR）和 P2 AMG 层次，验证 AMG 比 Jacobi-PCG 迭代次数更少，P2/P1 DOF 比 > 1.4；7/7 测试通过。
- ✅ 分数 Laplacian 稀疏有理逼近（ex33，2026-05-03）。`mfem_ex33_fractional_laplacian.rs` 新增 `SparseRational` 方法：复用 sinc 正交，但每个移位系统 `(K + t_q M) x = b` 用稀疏 Jacobi-PCG 求解（`csr_add_scaled` 构建），不依赖密集 Cholesky；与 dense spectral 基线误差 < 4%；7/7 测试通过。- ✅ 多物理场测试覆盖增强（2026-05-03）。
  - **ex45 ALE**（1→4 tests）：新增网格有效性（无倒置元素）、零振幅精确传递（积分误差 < 1e-12）、多步场范数稳定性验证。
  - **ex49 FSI**（2→6 tests）：新增近刚性壁极限（位移 < 1e-3）、流体积分守恒（< 1e-9）、单速率路径步收敛性、入口幅度单调性。
  - **ex51 EM-thermal-stress**（2→5 tests）：新增近零导电率极限（Joule 功率 < 1e-6）、负温度系数稳定性反馈、σ₀ 单调性。
- ✅ 多物理场测试覆盖二期（2026-05-03）。
  - **ex46 移动网格热方程**（2→4 tests）：新增高扩散率更快衰减单调性、全步有限正值验证。
  - **ex47 模板目录**（1→4 tests）：新增字段/边缘完备性（≥2 字段、≥1 边含"->"）、唯一 Partitioned 模板检查（仅 acoustics_structure）、id/title 唯一非空检查。
  - **ex48 Joule 加热**（3→5 tests）：新增高 kappa 更低温度单调性、正 sigma_beta 增大 Joule 功率反馈。
  - **ex50 声学-结构**（3→5 tests）：新增近刚性结构极小位移限、高驱动幅度增大压力与位移单调性。
  - **ex52 反应-流-热**（4→7 tests）：新增零反应率给出零温升、高热释放增大温度范数、子循环与单速率路径入口单调性一致性。
- ✅ 三维多物理场：ex53 3-D 电热耦合（2026-05-03）。新增 `mfem_ex53_3d_electrothermal.rs`，在 `SimplexMesh::<3>::unit_cube_tet` 上求解稳态电-热耦合：-∇·(σ(T)∇φ)=0（z=0/z=1 Dirichlet 驱动）+ -∇·(κ∇T)=σ|∇φ|²（全边界齐次 Dirichlet），通过带松弛的定点迭代收敛。元素体积直接组装 Joule 热源 RHS（P1 tet：∫_e N_i dx = vol/4）。8/8 测试通过：基础收敛性、(n+1)³ DOF 验证、电功率二次标度（V²）、κ 单调性、σ→0 极限、正 σ_β 自加热反馈、网格细化计数、φ 线性标度。
- ✅ 基础与 AMR 例子测试覆盖三期（2026-05-04）。
  - **ex1 Poisson**（4→8 tests，+4）：新增超粗网格收敛性、P1 h² 收敛阶观测（4.0±10%）、P2 vs P1 速率对比、源项线性缩放验证（2x 源→2x 解）。全部验证超定制造解 sin(πx)sin(πy)。
  - **ex2 弹性**（4→8 tests，+4）：新增超粗网格收敛性、网格细化 DOF 增长、P2 高阶验证、高体力方向位移单调性。VectorH1Space 左墙固定边界 + 重力体力。
  - **ex4 Darcy**（4→8 tests，+4）：新增源项线性缩放、β 参数增大解幅、网格细化单调收敛（n8→n12→n16）、极弱源给出极小解。RT0 H(div) 求解。
  - **ex5 混合 Darcy**（4→8 tests，+4）：新增超粗网格收敛、网格细化 DOF 增长、块残差保持小量、高施加增大解单调性。Uzawa Schur 补求解器。
  - **ex15_dynamic_amr**（3→7 tests，+4）：新增网格细化元素数增长、多轮细化单调递增、误差估计器处处正定有限、Dörfler 标记阈值单调性。ZZ 估计器与限制性标记演示。
  - **ex15_dg_amr**（4→9 tests，+5）：新增 Dörfler 阈值效应验证、超粗网格仍收敛、误差与估计器相关性追踪、DOF 单调增长、元素数单调增长。非协调 AMR 路线验证。
  总计 25 个新测试，涵盖基础例子物理/数值特性与两种 AMR 路线。
- ✅ `ElementTransformation` 统一抽象层（完成）。
  当前进展（2026-04-12）：`assembler`、`vector_assembler`、`mixed`、`vector_boundary` 的仿射 simplex 几何路径已统一切换到 `ElementTransformation`，不再在装配入口重复内联 `J/det(J)/J^{-T}/x(ξ)` 逻辑。
  验收证据（2026-04-12）：`cargo test -p fem-assembly --lib` 通过（118 passed, 0 failed, 1 ignored）。
- ✅ `jsmpi` 依赖来源统一为 crates.io 包（完成）。
  当前进展（2026-04-12）：`fem-parallel` 与 `fem-wasm` 已从 `vendor/jsmpi` path/submodule 依赖切换为 registry 依赖 `jsmpi = "0.1.0"`，并移除仓库 `vendor/jsmpi` 子模块跟踪。
  验收证据（2026-04-12）：`cargo check -p fem-parallel --target wasm32-unknown-unknown` 与 `cargo check -p fem-wasm --target wasm32-unknown-unknown --features wasm-parallel` 通过。

---

## 2. Maxwell 项对齐来MAXWELL_GAPS

### 2.1 ?论

- simplex 格? H(curl) Maxwell 主路已??
- ex3/ex3pMaxwell主线已对齐
- ex31~ex34?/吸/?载荷已形 builder 口
- ex13征已约LOBPCG 工流模??仍?强
- Maxwell solver 已`FirstOrderMaxwellOp` + `FirstOrderMaxwellSolver3D` + ?系

### 2.2 已? Maxwell ?

- H(curl) 件TriND1/2TetND1/2HCurlSpace ND1/ND2边?
- `StaticMaxwellProblem/Builder`marker/tag 语
- 边??PEC?吸齐次?载荷
- ?骨式/CN 步?解封语?
- 3D 骨证`curl_3d` 伴PEC 约语尼项衰

### 2.3 Quad/Hex ?素项注

- 已ND1Quad ND ?
- 已ND1Hex ND ?
- 已ND1HCurlSpace `Quad4`/`Hex8` DOF ??orientation
- 已段Quad/Hex ex3 类端端
  - Quad4 路已?builder 源 PEC smoke
  - Hex8 路已??`ex3_hex8_zero_source_full_pec_smoke` ?

---

## 3. 差距??级

### P0

1. Maxwell 边?条件 API
  026-04-12已 selector 口 `BoundarySelection::{Tags, Marker}``HcurlBoundaryConfig` `StaticMaxwellBuilder` `add_*_on(...)` 边??路?tags/marker API 复该口并保容
2. Maxwell 边?/材?/??
  026-04-12? `boundary_material_frequency_matrix_regression_smoke`isotropic+PECfrequency+impedance(子? marker)+PECanisotropic+absorbing(子? marker)+PEC 路
3. 吸边??边?语
  026-04-12 physical 口 `add_impedance_physical*` / `add_absorbing_physical*` `epsilon/mu` 换 `gamma`并?`gamma` 价?

### P1

1. E-B Maxwell solver ?形走?稳可口
  026-04-12`FirstOrderMaxwellSolver3D` 源模`FirstOrderForceModel3D::{Static, TimeDependent}``set_time_dependent_force(...)`/`clear_force()`并`advance_one`/`advance_with_config` 中?源项
2.  HCurl-HDiv mixed operator 路线
  026-04-12 `HCurlHDivMixedOperators3D` 封`FirstOrderMaxwell3DSkeleton::mixed_operators()` 导口 `H(curl)->H(div)` `H(div)->H(curl)` 路
3. solver ABC/?/源项模??步?
  026-04-12 `new_unit_cube_with_physical_abc_and_impedance_tags(...)``epsilon, mu -> gamma` 缩口并补齐源/边?/混子?

### P2

1. Quad ND2
  026-04-12 `QuadND2` ??并`VectorAssembler` `HCurlSpace``Quad4` order=2路补?空?素?
2. Hex ND2
  026-04-12 `HexND2` ??并`VectorAssembler` `HCurlSpace``Hex8` order=2路补?空?素?
3. Hex8 ex3 类端端示
  026-04-12 `examples/src/maxwell.rs` ?`ex3_hex8_zero_source_full_pec_smoke` ?`cargo test -p fem-space --lib` `cargo test -p fem-examples --lib` ?

### P3

1. ✅ H(curl) partial assembly / matrix-free operator
  2026-04-12 `HcurlMatrixFreeOperator2D` `solve_hcurl_matrix_free(...)` implemented; `apply/solve` confirmed equivalent to assembled matrix.
2. ✅ Maxwell generalized eigenproblem AMG-preconditioned LOBPCG
  `solve_hcurl_eigen_preconditioned_amg(...)` added to `fem-assembly`; uses regularised `K_reg = K_curl + 0.1*M_mass` for AMG; tested on `unit_square_tri(8)`.
3. ✅ Smoke gate tests pass; CI suite `hcurl-maxwell-eigen` added
  `hcurl_eigen_amg_preconditioned_lobpcg_smoke` + `hcurl_eigen_amg_large_scale_smoke` both pass; CI suite added to `alignment-smoke.yml` (now 13 suites).

### ?设

1. HDF5/XDMF 并 I/O段
  026-04-13已rank ???root 端? XDMF sidecar checkpoint ?校
2. restart checkpoint 路段
  026-04-13?中启续中线?`fem-io-hdf5-parallel``hdf5` feature
3. direct backend hooks baseline段
  026-04-13`mumps` `mkl` 容口已可 baseline`linger::{MumpsSolver, MklSolver}` + `fem-solver::{solve_sparse_mumps, solve_sparse_mkl}`??linger ?? multifrontal 载不以?MUMPS/MKL 依为
  证据026-04-13`cargo test --manifest-path vendor/linger/Cargo.toml direct_backend_mkl_solves_system``cargo test --manifest-path vendor/linger/Cargo.toml mkl_solver_solves_single_rhs``cargo test -p fem-solver sparse_mkl_direct` ?
4. native-amg-hardening纯 Rust?额格格式读`linger` AMS/ADS baseline 已可AIR baseline ??已落`CoarsenStrategy::Air` + diagonal-`A_ff` AIR restriction并对称对流?`amg_air_gmres_nonsymmetric_convdiff_1d`续parity hardening ?式/?Abaqus/Netgen 混?section tag 保?

### 主线项2026-04-13

1. direct HDF5 hyperslab ?+??读路baseline 已落缺系 HDF5 ?HDF5/MPI CI ??群端端
2. 跨子项 C2-C4WP2 + WP4-WP6AIR AMS/ADS parity hardeningmkl reed ?落 CI feature matrixGPUjsmpi fallback/CI WP3 mumps/mkl baseline 已
3. ?级 MFEM ?补齐??核?可?核?碍?核???保??核浸没边?保??核

### 补充 2026-05-05

- ✅ **ComplexCoeff / ComplexVectorCoeff** — 已在 `crates/assembly/src/coefficient.rs` 实现（`ComplexCoeff`、`ComplexVectorCoeff`、`ComplexConstCoeff`、`ComplexFromScalars`、`ComplexVectorFromVectors`），`crates/assembly/src/complex.rs` 提供 `ComplexSystem`、`ComplexAssembler`、`NativeComplexAssembler`。
- ✅ **mfem_ex22 时谐 Maxwell** — `examples/mfem_ex22.rs` 8 测试全部通过（Helmholtz 2×2 实块系统）。
- ✅ **AMS parity hardening** — `crates/solver/src/lib.rs` 新增 7 个 AMS/ADS 集成测试（`ams_ads_tests` 模块）：
  - `pcg_ams_hcurl_2d_converges`
  - `gmres_ams_hcurl_2d_converges`
  - `pcg_ams_solution_satisfies_ax_eq_b`
  - `pcg_ams_iteration_count_reasonable`
  - `pcg_ads_hdiv_3d_converges`
  - `gmres_ads_hdiv_3d_converges`
  - `pcg_ads_solution_satisfies_ax_eq_b`
  全部使用真实 FE 装配（HCurlSpace/HDivSpace + DiscreteLinearOperator），76/76 通过。
- ✅ **NamedAttributeSet / NamedAttributeRegistry**（2026-05-05）。`crates/mesh/src/boundary.rs` 中 `NamedAttributeSet`、`NamedAttributeRegistry`、`extract_submesh_by_name` 全部实现并从 `fem-mesh` 导出；`mfem_ex39_named_attributes` 8/8 测试通过。
- ✅ **CI alignment-smoke.yml**（2026-05-05）。`.github/workflows/alignment-smoke.yml` 已创建并扩展至 12 个套件：complex-coeff、complex-ex22、named-attrs、named-attrs-io、electromagnetic-pml、electromagnetic-absorbing、backend-contract、io-mixed-topology、io-mesh-coords-checkpoint、quad-hex-aniso-amr、amg-stress。

### 补充 2026-04-13

-  `.github/workflows/alignment-smoke.yml`对以?提 PR  smoke gate
  1. `ComplexCoeff` / `ComplexVectorCoeff``fem-assembly`
  2. `NamedAttributeSet` / `NamedAttributeRegistry``fem-mesh`
  3. 磁 PML-like 路`mfem_ex3 --pml-like`
  4. 吸边?路`mfem_ex34 --anisotropic`
  5. canonical backend-resource contract`fem-assembly`（`--features reed`）
- 新增 `.github/workflows/backend-feature-matrix.yml`：对 `fem-assembly`（`--features reed`）（workspace 钉 `rem-rs/reed` git 依赖）执行集成测试。

---

## 4. 段?

### 4.1 Maxwell 

- 026-04-12达
- 表PEC/?/吸边?
- isotropic / anisotropic / boundary-loaded 类稳?
- 示不依大?工水

证据026-04-12
- Builder/口??`cargo test -p fem-examples --lib` ?含 `boundary_material_frequency_matrix_regression_smoke``builder_mixed_boundary_matches_low_level_pipeline``builder_anisotropic_diag_matches_anisotropic_matrix_fn_diagonal` 
- 示路稳`cargo test -p fem-examples --example mfem_ex3 --example mfem_ex31 --example mfem_ex32 --example mfem_ex33_tangential_drive_maxwell --example mfem_ex34` ?

### 4.2 Maxwell solver

- 026-04-12达
-  solver 模维 E in H(curl)B in H(div)
-  `ε``μ^{-1}````J` 边?尼项
- /稳?

证据026-04-12
- 3D ?`cargo test -p fem-examples --lib first_order_3d_` ?7/67? `first_order_3d_energy_conserved_sigma0``first_order_3d_sigma_dissipates_energy``first_order_3d_absorbing_boundary_term_dissipates_energy_without_sigma``first_order_3d_impedance_boundary_term_dissipates_energy_without_sigma``first_order_3d_solver_wrapper_time_dependent_force_matches_static_when_constant` 

### 4.3 ?素

- 026-04-12达
- Quad ND1/ND2 可Maxwell
- Hex ND1 可 3D ex3 类
- DOF curl 形边DOForientation ??

证据026-04-12
- ?素ND ?`cargo test -p fem-element --lib nedelec` ?9/19? `nd1_nodal_edge_moments``nd2_quad_basis_and_curl_are_finite``nd1_hex_edge_moments_are_nodal``nd2_hex_basis_and_curl_are_finite`
- 空DOF/边?/orientation`cargo test -p fem-space --lib hcurl` ?3/13? `hcurl_dof_count_quad_nd1``hcurl_dof_count_quad_nd2``hcurl_dof_count_hex_nd1``hcurl_dof_count_hex_nd2``boundary_dofs_hcurl_unit_square``hcurl_signs_consistent_on_shared_edge`
- 端Hex ND1 Maxwell`cargo test -p fem-examples --lib ex3_hex8_zero_source_full_pec_smoke` ?

### 4.4 可

- 026-04-12达
- H(curl) 不式`y = A x` 路线
- 征?大 DOF 模

证据026-04-12
- matrix-free 大模路`cargo test -p fem-examples --lib hcurl_matrix_free_large_dof_apply_smoke` ?`n=40` `HcurlMatrixFreeOperator2D` ?大 DOF `y = A x` `n_dofs > 3000`
- 征模?路`cargo test -p fem-examples --lib hcurl_eigen_amg_preconditioned_lobpcg_smoke` ?AMG 条LOBPCG 大smoke`free DOF > 200`
- ?模?路`cargo test -p fem-examples --lib first_order_3d_large_dof_single_step_smoke` ??3D DOF 步 smoke`n_e > 250`, `n_b > 120`

---

## 5. ?顺序建议

1. ?补Maxwell API ?
2. solver ??mixed operator + 边?/?项
3. Quad/Hex ?素端端Hex ex3
4. ?补PA/matrix-free 可征路线

该顺序维面确保每步以?代+  + ?档

---

## 6. 维约

每次?以

1. ?档?级条
2. `MFEM_MAPPING.md`? remaining items
3. Maxwell 项?证据`MAXWELL_GAPS.md` ?步??该?档已并移?以?档为项来源

---

## 7.  MFEM ?踪Beyond-MFEM
> ?记? MFEM 对齐项??对齐项?工??避被误类为 parity 工?>
> ?已?· ? · ? ?
### 7.1 ?台账已落
| ID | ??| ?| ?| 代码位置 | ?| 责| ?|
|---|---|---|---|---|---|---|---|
| BMF-001 | 象block residual / block Jacobian| ?| P0 | `crates/solver/src/multiphysics.rs` | M1 | TBD | 已? |
| BMF-002 | 线可?换GMRES / 2x2 Schur| ?| P0 | `crates/solver/src/multiphysics.rs` | M1 | TBD | 已? |
| BMF-003 | ??工示稳+ ?split + ?IMEX| ?| P0 | `examples/mfem_ex44_thermoelastic_coupled.rs` | M2 | TBD | 已?示 |
| BMF-004 | 对CSV 导含 dt/steps sweep| ?| P1 | `examples/mfem_ex44_thermoelastic_coupled.rs` | M2 | TBD | 已?示 |
| BMF-005 | 2D Tri3 位barycentric + nearest fallback| ?| P0 | `crates/mesh/src/point_locator.rs` | M1 | TBD | 已? |
| BMF-006 | H1-P1 ?传?source?target| ?| P0 | `crates/assembly/src/transfer.rs` | M1 | TBD | 已? |

### 7.2 路线

| ID |  | ?| ?| ?| 依 | ? |
|---|---|---|---|---|---|---|
| BMF-101 | 传? 3D Tet4locator + transfer| ?| P0 | M3 | BMF-005/BMF-006 | 已? 3D 位 3D 线传?|
| BMF-102 | ?积?驱?L2 ?影 | ?| P0 | M3 | BMF-006 | 已? L2 ?影线精确??|
| BMF-103 | 传?conservative transfer?误差估 | ?| P1 | M4 | BMF-102 | 已提修正?边???误差估并??|
| BMF-104 | 工流??跨格 | ?| P1 | M4 | BMF-101/BMF-102 | ex44 steady/split/IMEX 路已?换格/格并?? |

### 7.3 ?建议
1. M1解象MVP 传??可复?2. M2工示对路稳?3. M3维 L2 ?影?达可线?4. M4传?+ 工流?
### 7.4 令可复
1. `cargo test -p fem-mesh point_locator`
2. `cargo test -p fem-assembly transfer::tests::nonmatching_h1_p1_transfer_is_exact_for_linear_fields`
3. `cargo test -p fem-examples --example mfem_ex44_thermoelastic_coupled`
4. `cargo test -p fem-mesh point_locator::tests::locate_point_in_unit_cube_tet_mesh`
5. `cargo test -p fem-assembly transfer::tests::nonmatching_h1_p1_transfer_is_exact_for_linear_fields_3d`
6. `cargo test -p fem-assembly transfer::tests::nonmatching_h1_p1_l2_projection_is_exact_for_linear_fields`
7. `cargo test -p fem-assembly transfer::tests::nonmatching_h1_p1_l2_projection_l2_error_converges`
8. `cargo test -p fem-assembly transfer::tests::conservative_projection_matches_global_integral`
9. `cargo test -p fem-assembly transfer::tests::boundary_flux_metric_is_consistent_for_exact_linear_transfer`
10. `cargo test -p fem-assembly transfer::tests::conservative_projection_3d_matches_global_integral`
11. `cargo test -p fem-examples --example mfem_ex44_thermoelastic_coupled`

---

## 8. Parity Delivery Execution Pack (2026-04-18)

To move from feature-level alignment to measurable delivery gates, use the
following docs as the execution pack:

1. `docs/mfem-parity-matrix-template.md`
  - Row-based acceptance matrix with explicit thresholds and evidence links.
2. `docs/mfem-6week-plan-estimates.md`
  - Six-week task decomposition with person-day estimates and risk buffers.
3. `docs/mfem-baseline-snapshot-2026-04-18.md`
  - Command-backed baseline snapshot for current parity anchors.

Immediate kickoff checklist:

1. Assign owner and target date for each active parity row (P0/P1 first).
2. Replace all `TODO` evidence links with CI artifact or report links.
3. Use threshold-based closure only; do not close rows by narrative status.
4. Update this tracker and `MFEM_MAPPING.md` when any row status changes.

### 8.1 Execution Progress - Week 2 IO Gate (2026-04-18)

1. Added dedicated IO parity workflow:
  - `.github/workflows/io-parity-hdf5.yml`
  - Includes smoke/full tier split (PR smoke, dispatch/nightly full).
2. Added kickoff execution log:
  - `docs/mfem-w2-io-kickoff-2026-04-18.md`
3. PM-001 status impact:
  - CI lane and artifact path are in place.
  - PM-001 closure evidence is complete (smoke + full URLs backfilled).
4. Local execution fallback (Actions unavailable):
  - Added `scripts/run_io_parity_hdf5.ps1` and generated `docs/mfem-w2-io-local-report-2026-04-18.md`.
  - Switched PM-001 execution path to pure-Rust default backend; native HDF5 dev libs are no longer a hard prerequisite for baseline execution.
  - Validation snapshot on pure-Rust route:
    - `cargo test -p fem-io-hdf5-parallel` passed (6/6).
    - `cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint` passed (4/4).
    - `scripts/run_io_parity_hdf5.ps1 -Mode full -Backend all -Repeat 20` passed.
    - Repeat stability: partitioned 0/20 failures, mpi 0/5 failures.
5. CI recovery backfill template:
  - `docs/mfem-w2-io-ci-backfill-template.md` created as canonical URL/artifact checklist.
  - PM-001 has moved to complete after template URLs were filled from successful CI runs.
6. CI backfill completed:
  - smoke: https://github.com/rem-rs/fem-rs/actions/runs/24606857993
  - full: https://github.com/rem-rs/fem-rs/actions/runs/24606858418


