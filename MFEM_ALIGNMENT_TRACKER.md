# fem-rs 锟?MFEM 鑳藉姏瀵归綈缁熶竴璺熻釜鏂囨。

> 鏇存柊鏃ユ湡锟?026-04-13  
> 鐩殑锛氬皢 MFEM_MAPPING 锟?MAXWELL_GAPS 鍚堝苟涓哄崟涓€璺熻釜鍏ュ彛锛岀粺涓€璁板綍鈥滃叏灞€鑳藉姏瀵归綈 + Maxwell 涓撻」鎺ㄨ繘鈥濓拷?

---

## 浣跨敤璇存槑

- 鏈枃妗ｆ槸 MFEM 鑳藉姏瀵归綈鐨勪富璺熻釜鏂囨。锟?
- 鍘熷缁嗚妭鏉ユ簮锟?
  - `MFEM_MAPPING.md`锛氬叏鍩熻兘鍔涙槧灏勩€侀樁娈甸噷绋嬬銆佽法妯″潡宸窛锟?
  - `MAXWELL_GAPS.md`锛歁axwell/H(curl) 涓撻」宸窛涓庤矾绾垮浘锟?
- 鏇存柊鍘熷垯锟?
  1. 浠ｇ爜涓庢祴璇曞厛钀藉湴锟?
  2. 鏈枃妗ｇ姸鎬佸悓姝ユ洿鏂帮拷?
  3. 鑻ヤ笌鍘熷鏂囨。涓嶄竴鑷达紝浠モ€滄渶鏂板彲楠岃瘉浠ｇ爜 + 娴嬭瘯缁撴灉鈥濅负鍑嗭紝骞跺洖鍐欏師鏂囨。锟?

鐘舵€佸浘渚嬶細锟?宸插畬锟?路 馃敤 閮ㄥ垎瀹屾垚 路 馃敳 璁″垝锟?路 锟?涓嶇撼锟?

---

## 1. 鍏ㄥ眬鑳藉姏瀵归綈鎬昏锛堟潵锟?MFEM_MAPPING锟?

### 1.1 宸插熀鏈榻愮殑涓诲共鑳藉姏

- 锟?Mesh / 鍩虹缃戞牸鑳藉姏锟?D/3D銆侀潪涓€鑷寸綉鏍硷紙Tri3/Tet4锛夈€佸苟琛屽垎鍖恒€佸懆鏈熺綉鏍笺€佹贩鍚堝厓绱犲熀纭€璁炬柦锟?
- 锟?鍙傝€冨厓涓庣Н鍒嗭細H1銆丯D锛圱ri/Tet锛夈€丷T锛圱ri/Tet锛夈€佸紶閲忕Н绉垎銆丟auss-Lobatto锟?
- 锟?绌洪棿锟?DOF锛欻1/L2/HCurl/HDiv/H1Trace銆丏OF 鎷撴墤涓庣鍙峰鐞嗐€佸苟锟?true dof 璺緞锟?
- 锟?瑁呴厤涓庣Н鍒嗗櫒锛氭爣锟?鍚戦噺鍙岀嚎鎬т笌绾挎€ц閰嶃€丏G-SIP銆丮ixedAssembler銆佺鏁ｇ畻瀛愶紙grad/curl/div锛夛拷?
- 锟?绾挎€т唬鏁颁笌姹傝В鍣細CSR/COO銆並rylov銆丄MG銆丼chur銆丮INRES銆両DR(s)銆乀FQMR銆佺█鐤忕洿鎺ユ硶锟?
- 锟?骞惰锟?WASM锛歅arCsr/ParVector銆佸苟琛岃閰嶃€佸苟锟?AMG銆乄ASM 锟?worker 锟?E2E 楠岃瘉锟?
- 锟?绀轰緥瑕嗙洊锛歅oisson/Maxwell/Darcy/Stokes/Navier-Stokes/骞惰绀轰緥宸插舰鎴愪綋绯伙拷?

### 1.2 鍏ㄥ眬鍓╀綑宸窛锛堣法妯″潡锟?

- 馃敤 HDF5/XDMF 骞惰 I/O 锟?restart 鏂囦欢閾捐矾锛坈heckpoint/restart 鍩虹嚎宸茶惤鍦帮紱direct HDF5 hyperslab 鍏ㄥ眬锟?鍒囩墖锟?baseline 宸茶惤鍦帮紝锟?HDF5/MPI 鐜绔埌绔獙鏀讹級锟?
- 馃敤 hypre-equivalent 璺嚎锛堢函 Rust 鑳藉姏杞ㄩ亾锛沗linger` 锟?AMS/ADS baseline 宸插彲鐢紝鍓╀綑锟?AIR 涓庡垎甯冨紡/楂橀樁鑳藉姏琛ラ綈涓轰富锛夛拷?
- 馃敤 Netgen/Abaqus 缃戞牸璇诲彇鏀寔锛圢etgen `.vol` Tet4/Hex8 ASCII uniform/mixed 璇诲彇鍩虹嚎 + Abaqus `.inp` C3D4/C3D8 uniform/mixed 璇诲彇鍩虹嚎宸茶惤鍦帮紱鍐欏嚭涓庢洿锟?section/tag 淇濈湡寰呰ˉ榻愶級锟?
- 馃敤 闈欐€佸嚌鑱氬熀绾跨ず渚嬶紙`mfem_ex8_hybridization`锛屼唬鏁扮害鏉熸秷鍏冭矾寰勶級宸茶惤鍦帮紱娣峰悎/鏉傚寲 FEM 鍐呮牳寰呰ˉ榻愶拷?
- 馃敤 鍒嗘暟锟?Laplacian 鍩虹嚎绀轰緥锛坄mfem_ex33_fractional_laplacian`锛宒ense spectral FE 璺嚎锛夊凡钀藉湴锛涘彲鎵╁睍鐭╅樀鍑芥暟/extension 璺嚎寰呰ˉ榻愶拷?
- 馃敤 闅滅闂鍩虹嚎绀轰緥锛坄mfem_ex36_obstacle`锛宲rimal-dual active-set (PDAS) 鍙樺垎涓嶇瓑寮忓熀绾匡級宸茶惤鍦帮紱semismooth Newton 鍐呮牳寰呰ˉ榻愶拷?
- 馃敤 鎷撴墤浼樺寲鍩虹嚎绀轰緥锛坄mfem_ex37_topology_optimization`锛屾爣锟?SIMP + OC + density filter + Heaviside projection + chain-rule sensitivity锛夊凡钀藉湴锛涘叏寮癸拷?浼撮殢/澶嶆潅绾︽潫璺嚎寰呰ˉ榻愶拷?
- 馃敤 鎴柇绉垎 / 娴告病杈圭晫鍩虹嚎绀轰緥锛坄mfem_ex38_immersed_boundary`锛宑ut-cell subtriangulation + Nitsche-like 锟?Dirichlet锛堝鸡娈佃繎浼硷級锛夊凡钀藉湴锛涘畬锟?cut-FEM/level-set 绋冲仴鍑犱綍涓庨珮闃剁晫闈㈢Н鍒嗚矾绾垮緟琛ラ綈锟?
- 馃敤 鍛藉悕灞炴€ч泦锛坆aseline+锛夛細`fem-mesh` 宸叉彁锟?`NamedAttributeSet` / `NamedAttributeRegistry`锛屾敮锟?mesh named queries 锟?`extract_submesh_by_name(...)`锛宍fem-io` 宸叉彁锟?GMSH `PhysicalNames` -> named registry bridge锛屽苟鏂板 `mfem_ex39_named_attributes` 绀轰緥鎵撻€氱鍒扮璺緞锟?
- 馃敤 鍑犱綍澶氶噸缃戞牸 / LOR锛圥hase 58锛夛細`GeomMGHierarchy` + `GeomMGPrecond` 鍩虹嚎宸插叿澶囷紝骞舵柊锟?`mfem_ex26_geom_mg` 绀轰緥鐢ㄤ簬鎸佺画鍥炲綊锟?
- 锟?`ElementTransformation` 缁熶竴鎶借薄灞傦紙瀹屾垚锛夛拷?
  褰撳墠杩涘睍锟?026-04-12锛夛細`assembler`銆乣vector_assembler`銆乣mixed`銆乣vector_boundary` 鐨勪豢锟?simplex 鍑犱綍璺緞宸茬粺涓€鍒囨崲锟?`ElementTransformation`锛屼笉鍐嶅湪瑁呴厤鍏ュ彛閲嶅鍐呰仈 `J/det(J)/J^{-T}/x(尉)` 閫昏緫锟?
  楠屾敹璇佹嵁锟?026-04-12锛夛細`cargo test -p fem-assembly --lib` 閫氳繃锟?18 passed, 0 failed, 1 ignored锛夛拷?
- 锟?`jsmpi` 渚濊禆鏉ユ簮缁熶竴锟?crates.io 鍖咃紙瀹屾垚锛夛拷?
  褰撳墠杩涘睍锟?026-04-12锛夛細`fem-parallel` 锟?`fem-wasm` 宸蹭粠 `vendor/jsmpi` path/submodule 渚濊禆鍒囨崲锟?registry 渚濊禆 `jsmpi = "0.1.0"`锛屽苟绉婚櫎浠撳簱 `vendor/jsmpi` 瀛愭ā鍧楄窡韪拷?
  楠屾敹璇佹嵁锟?026-04-12锛夛細`cargo check -p fem-parallel --target wasm32-unknown-unknown` 锟?`cargo check -p fem-wasm --target wasm32-unknown-unknown --features wasm-parallel` 閫氳繃锟?

---

## 2. Maxwell 涓撻」瀵归綈鎬昏锛堟潵锟?MAXWELL_GAPS锟?

### 2.1 褰撳墠鑳藉姏缁撹

- 锟?simplex 缃戞牸涓婄殑 H(curl) Maxwell 涓婚摼璺凡鎵撻€氾拷?
- 锟?ex3/ex3p锛堥潤锟?Maxwell锛変富绾垮凡瀵归綈锟?
- 锟?ex31~ex34锛堝悇鍚戝紓锟?闃绘姉/鍚告敹/鍒囧悜杞借嵎锛夊凡褰㈡垚缁熶竴 builder 鍏ュ彛锟?
- 锟?ex13锛堢壒寰佸€硷級宸插叿澶囩害鏉熺█锟?LOBPCG 宸ヤ綔娴侊紙瑙勬ā鍖栬兘鍔涗粛闇€澧炲己锛夛拷?
- 锟?涓€闃跺叏锟?Maxwell solver 宸插畬鎴愶細`FirstOrderMaxwellOp` + `FirstOrderMaxwellSolver3D` + 鍥炲綊娴嬭瘯浣撶郴锟?

### 2.2 宸插畬鎴愮殑 Maxwell 鍏抽敭閲岀▼锟?

- 锟?H(curl) 缁勪欢锛歍riND1/2銆乀etND1/2銆丠CurlSpace ND1/ND2銆佸悜閲忚閰嶄笌杈圭晫瑁呴厤锟?
- 锟?缁熶竴闈欐€佸簲鐢ㄩ摼锛歚StaticMaxwellProblem/Builder`锛宮arker/tag 璇箟缁熶竴锟?
- 锟?杈圭晫鑳藉姏锛歅EC銆侀樆鎶椼€佸惛鏀躲€侀潪榻愭鍒囧悜杞借嵎锟?
- 锟?鏃跺煙涓€闃堕鏋讹細鏄惧紡/CN 姝ヨ繘銆佺姸鎬佸寲姹傝В鍣ㄥ皝瑁呫€佽兘閲忚涔夊洖褰掞拷?
- 锟?3D 楠ㄦ灦楠岃瘉锛歚curl_3d` 缁勮銆佷即闅忎竴鑷存€с€丳EC 绾︽潫璇箟銆侀樆灏奸」鑳介噺琛板噺锟?

### 2.3 Quad/Hex 鍏冪礌鏃忎笓椤癸紙褰撳墠鍏虫敞鐐癸級

- 锟?宸插畬鎴愶紙ND1锛夛細Quad ND 鍏冿拷?
- 锟?宸插畬鎴愶紙ND1锛夛細Hex ND 鍏冿拷?
- 锟?宸插畬鎴愶紙ND1锛夛細HCurlSpace 锟?`Quad4`/`Hex8` 锟?DOF 鎷撴墤锟?orientation锟?
- 锟?宸插畬鎴愶紙闃舵鎬э級锛歈uad/Hex 锟?ex3 绫荤鍒扮锟?
  - 锟?Quad4 璺緞宸茶窇閫氾紙builder 闆舵簮 PEC smoke锛夛拷?
  - 锟?Hex8 璺緞宸叉墦閫氾紙`ex3_hex8_zero_source_full_pec_smoke` 鍥炲綊锛夛拷?

---

## 3. 缁熶竴宸窛娓呭崟锛堟寜浼樺厛绾э級

### P0

1. 锟?Maxwell 缁熶竴杈圭晫鏉′欢 API锛堝畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細宸叉柊澧炵粺涓€ selector 鍏ュ彛 `BoundarySelection::{Tags, Marker}`锛宍HcurlBoundaryConfig` 锟?`StaticMaxwellBuilder` 鍧囨敮锟?`add_*_on(...)` 缁熶竴杈圭晫閫夋嫨璺緞锛屽師锟?tags/marker API 澶嶇敤璇ュ叆鍙ｅ苟淇濇寔鍏煎锟?
2. 锟?Maxwell 杈圭晫/鏉愭枡/棰戝煙鍥炲綊鐭╅樀锛堝畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板鍥炲綊 `boundary_material_frequency_matrix_regression_smoke`锛岃锟?isotropic+PEC銆乫requency+impedance(瀛愰泦 marker)+PEC銆乤nisotropic+absorbing(瀛愰泦 marker)+PEC 缁勫悎璺緞锟?
3. 锟?鍚告敹杈圭晫涓庨樆鎶楄竟鐣岀墿鐞嗗弬鏁拌涔夛紙瀹屾垚锛夛拷?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 physical 鍏ュ彛 `add_impedance_physical*` / `add_absorbing_physical*`锛堢敱 `epsilon/mu` 鑷姩鎹㈢畻 `gamma`锛夛紝骞堕€氳繃涓庢樉锟?`gamma` 绛変环鍥炲綊娴嬭瘯锟?

### P1

1. 锟?涓€锟?E-B Maxwell solver 浠庘€滈洀褰⑩€濊蛋鍚戔€滅ǔ瀹氬彲鎵╁睍鎺ュ彛鈥濓紙瀹屾垚锛夛拷?
  褰撳墠杩涘睍锟?026-04-12锛夛細`FirstOrderMaxwellSolver3D` 鏂板鏃跺彉婧愭ā锟?`FirstOrderForceModel3D::{Static, TimeDependent}`锛屾敮锟?`set_time_dependent_force(...)`/`clear_force()`锛屽苟锟?`advance_one`/`advance_with_config` 涓寜褰撳墠鏃堕棿鍒锋柊婧愰」锟?
2. 锟?瀹屾暣 HCurl-HDiv mixed operator 璺嚎锛堝畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 `HCurlHDivMixedOperators3D` 楂樺眰灏佽锟?`FirstOrderMaxwell3DSkeleton::mixed_operators()` 瀵煎嚭鎺ュ彛锛岀粺涓€ `H(curl)->H(div)` 锟?`H(div)->H(curl)` 搴旂敤璺緞锟?
3. 锟?solver 锟?ABC/闃绘姉/婧愰」妯″瀷鐨勮繘涓€姝ョ墿鐞嗗畬澶囧寲锛堝畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 `new_unit_cube_with_physical_abc_and_impedance_tags(...)`锛坄epsilon, mu -> gamma` 缂╂斁鍏ュ彛锛夛紝骞惰ˉ榻愭椂鍙樻簮/鐗╃悊杈圭晫/娣峰悎绠楀瓙涓€鑷存€у洖褰掞拷?

### P2

1. 锟?Quad ND2锛堝畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 `QuadND2` 鍙傝€冨厓骞舵帴锟?`VectorAssembler` 锟?`HCurlSpace`锛坄Quad4` order=2锛夎矾寰勶紝琛ュ厖绌洪棿涓庡厓绱犲洖褰掞拷?
2. 锟?Hex ND2锛堝畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 `HexND2` 鍙傝€冨厓骞舵帴锟?`VectorAssembler` 锟?`HCurlSpace`锛坄Hex8` order=2锛夎矾寰勶紝琛ュ厖绌洪棿涓庡厓绱犲洖褰掞拷?
3. 锟?Hex8 锟?ex3 绫荤鍒扮绀轰緥涓庨獙鏀堕棴鐜紙瀹屾垚锛夛拷?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 `examples/src/maxwell.rs` 鐨?`ex3_hex8_zero_source_full_pec_smoke` 鍥炲綊锛沗cargo test -p fem-space --lib` 锟?`cargo test -p fem-examples --lib` 鍏ㄩ噺閫氳繃锟?

### P3

1. 锟?H(curl) partial assembly / matrix-free operator锛堥樁娈垫€у畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 `HcurlMatrixFreeOperator2D` 锟?`solve_hcurl_matrix_free(...)`锛屼互 `A x = (1/mu) C^T M_b^{-1} C x + alpha M_e x` 璺嚎閬垮厤缁勮鍏ㄥ眬缁勫悎鐭╅樀锛屽苟琛ラ綈 `apply/solve` 瀵硅閰嶇畻瀛愮殑绛変环鍥炲綊锟?
2. 锟?Maxwell generalized eigenproblem 鐨勫彲鎵╁睍棰勬潯浠惰矾绾匡紙LOBPCG/AMG 缁勫悎锛岄樁娈垫€у畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細锟?`fem-solver` 澧炲姞 `lobpcg_constrained_preconditioned(...)`锛屽苟锟?Maxwell 渚ф帴锟?`solve_hcurl_eigen_preconditioned_amg(...)`锛圓MG 娈嬮噺鍧楅鏉′欢锛夛紝`mfem_ex13_eigenvalue` 榛樿璧拌璺嚎锟?
3. 锟?鏇村ぇ瑙勬ā骞惰/瑙勬ā鍖栭獙璇侊紙闃舵鎬у畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-12锛夛細鏂板 `hcurl_eigen_amg_preconditioned_lobpcg_smoke` 瑙勬ā鍖栧洖褰掞紙`n=10`锛宖ree DOF > 200锛変綔涓哄ぇ瑙勬ā璺緞 smoke gate锟?

### 閫氱敤鍩虹璁炬柦

1. 锟?HDF5/XDMF 骞惰 I/O锛堥樁娈垫€у畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-13锛夛細宸插叿锟?rank 鍒嗙墖鍐欏叆銆乺oot 绔叏灞€鍦虹墿鍖栦笌 XDMF sidecar 杈撳嚭锛屾敮锟?checkpoint 缁撴瀯鏍￠獙锟?
2. 锟?restart checkpoint 閾捐矾锛堥樁娈垫€у畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-13锛夛細鏂板鈥滀腑鏂悗閲嶅惎缁畻涓庢棤涓柇鍩虹嚎涓€鑷粹€濆洖褰掞紙`fem-io-hdf5-parallel`锛宍hdf5` feature锛夛拷?
3. 锟?direct backend hooks baseline锛堥樁娈垫€у畬鎴愶級锟?
  褰撳墠杩涘睍锟?026-04-13锛夛細`mumps` 涓?`mkl` 鍏煎鍏ュ彛鍧囧凡鍏峰鍙敤 baseline锛坄linger::{MumpsSolver, MklSolver}` + `fem-solver::{solve_sparse_mumps, solve_sparse_mkl}`锛夛紱浜岃€呴兘鐢?linger 鍘熺敓 multifrontal 瀹炵幇鎵胯浇锛屼笉浠ュ閮?MUMPS/MKL 渚濊禆涓虹洰鏍囷拷?
  楠屾敹璇佹嵁锟?026-04-13锛夛細`cargo test --manifest-path vendor/linger/Cargo.toml direct_backend_mkl_solves_system`銆乣cargo test --manifest-path vendor/linger/Cargo.toml mkl_solver_solves_single_rhs`銆乣cargo test -p fem-solver sparse_mkl_direct` 閫氳繃锟?
4. hypre-equivalent锛堢函 Rust锛夎兘鍔涙墿灞曚笌棰濆缃戞牸鏍煎紡璇诲彇锛坄linger` 锟?AMS/ADS baseline 宸插彲鐢紱AIR baseline 鑴氭墜鏋跺凡钀藉湴锛歚CoarsenStrategy::Air` + diagonal-`A_ff` AIR restriction锛屽苟鏂板闈炲绉板娴佹墿鏁ｅ洖锟?`amg_air_gmres_nonsymmetric_convdiff_1d`锛涘悗缁仛锟?parity hardening 涓庡垎甯冨紡/楂橀樁鑳藉姏锛汚baqus/Netgen 鎵╁睍锛氭贩鍚堝崟鍏冦€佹洿锟?section 锟?tag 淇濈湡锛夛拷?

### 褰撳墠涓荤嚎鍓╀綑椤癸紙2026-04-13锟?

1. direct HDF5 hyperslab 鍏ㄥ眬鍐欏叆+鍒囩墖璇诲彇璺緞锛坆aseline 宸茶惤鍦帮紱褰撳墠鐜缂哄皯绯荤粺 HDF5 搴擄紝闇€鍦ㄥ叿锟?HDF5/MPI 锟?CI 鎴栭泦缇ょ幆澧冨畬鎴愮鍒扮楠屾敹锛夛拷?
2. 璺ㄥ瓙椤圭洰 C2-C4锛歐P2 + WP4-WP6锛圓IR 锟?AMS/ADS parity hardening銆乵kl 锟?reed 鐨勮惤鍦颁笌 CI feature matrix銆丟PU銆乯smpi fallback/CI 鐭╅樀锛沇P3 mumps/mkl baseline 宸插畬鎴愶級锟?
3. 浣庝紭鍏堢骇 MFEM 鑳藉姏鏃忚ˉ榻愶紙闈欐€佸嚌锟?鏉傚寲鍐呮牳銆佸垎鏁伴樁鍙墿灞曞唴鏍搞€侀殰纰嶉棶棰橀珮闃跺唴鏍搞€佹嫇鎵戜紭鍖栭珮淇濈湡鍐呮牳銆佹蹈娌¤竟鐣岄珮淇濈湡鍐呮牳锛夛拷?

### 鑷姩楠屾敹琛ュ厖锟?026-04-13锟?

- 鏂板 `.github/workflows/alignment-smoke.yml`锛氬浠ヤ笅鑳藉姏鎻愪緵 PR 鑷姩 smoke gate锟?
  1. `ComplexCoeff` / `ComplexVectorCoeff`锛坄fem-assembly`锟?
  2. `NamedAttributeSet` / `NamedAttributeRegistry`锛坄fem-mesh`锟?
  3. 鐢电 PML-like 璺緞锛坄mfem_ex3 --pml-like`锟?
  4. 鍚勫悜寮傛€у惛鏀惰竟鐣岃矾寰勶紙`mfem_ex34 --anisotropic`锟?
  5. canonical backend-resource contract锛坄fem-ceed`锟?
- 鏂板 `.github/workflows/backend-feature-matrix.yml`锛氬 `vendor/reed` backend contract 锟?`baseline/hypre-rs/petsc-rs/mumps/mkl` feature profile 涓嬫墽琛岀煩闃靛寲娴嬭瘯锟?

---

## 4. 闃舵楠屾敹鏍囧噯

### 4.1 Maxwell 搴旂敤锟?

- 褰撳墠鍒ゅ畾锟?026-04-12锛夛細锟?瀹屽叏杈炬爣锟?
- 鑳界洿鎺ヨ〃锟?PEC/闃绘姉/鍚告敹杈圭晫锟?
- isotropic / anisotropic / boundary-loaded 涓夌被闂绋冲畾閫氳繃锟?
- 绀轰緥涓嶄緷璧栧ぇ閲忔墜宸ヨ閰嶈兌姘达拷?

楠屾敹璇佹嵁锟?026-04-12锛夛細
- Builder/缁熶竴鍏ュ彛鑳藉姏涓庡洖褰掞細`cargo test -p fem-examples --lib` 鍏ㄩ噺閫氳繃锛堝惈 `boundary_material_frequency_matrix_regression_smoke`銆乣builder_mixed_boundary_matches_low_level_pipeline`銆乣builder_anisotropic_diag_matches_anisotropic_matrix_fn_diagonal` 绛夛級锟?
- 绀轰緥閾捐矾绋冲畾鎬э細`cargo test -p fem-examples --example mfem_ex3 --example mfem_ex31 --example mfem_ex32 --example mfem_ex33_tangential_drive_maxwell --example mfem_ex34` 鍏ㄩ儴閫氳繃锟?

### 4.2 涓€闃跺叏锟?Maxwell solver

- 褰撳墠鍒ゅ畾锟?026-04-12锛夛細锟?瀹屽叏杈炬爣锟?
- 鐙珛 solver 妯″潡缁存姢 E in H(curl)銆丅 in H(div)锟?
- 鏀寔 `蔚`銆乣渭^{-1}`銆乣蟽`銆乣J` 涓庤竟鐣岄樆灏奸」锟?
- 鍏峰鑳介噺/绋冲畾鎬у洖褰掞拷?

楠屾敹璇佹嵁锟?026-04-12锛夛細
- 涓€闃跺叏锟?3D 鍥炲綊锛歚cargo test -p fem-examples --lib first_order_3d_` 閫氳繃锟?7/67锛夛紝瑕嗙洊 `first_order_3d_energy_conserved_sigma0`銆乣first_order_3d_sigma_dissipates_energy`銆乣first_order_3d_absorbing_boundary_term_dissipates_energy_without_sigma`銆乣first_order_3d_impedance_boundary_term_dissipates_energy_without_sigma`銆乣first_order_3d_solver_wrapper_time_dependent_force_matches_static_when_constant` 绛夛拷?

### 4.3 鍏冪礌鏃忚锟?

- 褰撳墠鍒ゅ畾锟?026-04-12锛夛細锟?瀹屽叏杈炬爣锟?
- Quad ND1/ND2 鍙窇闈欙拷?Maxwell锟?
- Hex ND1 鑷冲皯鍙窇 3D ex3 绫婚棶棰橈拷?
- DOF 鏁般€乧url 褰㈠嚱鏁般€佽竟锟?DOF銆乷rientation 鍧囨湁鍗曞厓娴嬭瘯锟?

楠屾敹璇佹嵁锟?026-04-12锛夛細
- 鍏冪礌锟?ND 瑕嗙洊锛歚cargo test -p fem-element --lib nedelec` 閫氳繃锟?9/19锛夛紝瑕嗙洊 `nd1_nodal_edge_moments`銆乣nd2_quad_basis_and_curl_are_finite`銆乣nd1_hex_edge_moments_are_nodal`銆乣nd2_hex_basis_and_curl_are_finite`锟?
- 绌洪棿锟?DOF/杈圭晫/orientation锛歚cargo test -p fem-space --lib hcurl` 閫氳繃锟?3/13锛夛紝瑕嗙洊 `hcurl_dof_count_quad_nd1`銆乣hcurl_dof_count_quad_nd2`銆乣hcurl_dof_count_hex_nd1`銆乣hcurl_dof_count_hex_nd2`銆乣boundary_dofs_hcurl_unit_square`銆乣hcurl_signs_consistent_on_shared_edge`锟?
- 绔埌锟?Hex ND1 闈欙拷?Maxwell锛歚cargo test -p fem-examples --lib ex3_hex8_zero_source_full_pec_smoke` 閫氳繃锟?

### 4.4 鍙墿灞曟€ц兘锟?

- 褰撳墠鍒ゅ畾锟?026-04-12锛夛細锟?瀹屽叏杈炬爣锟?
- H(curl) 鏀寔涓嶆樉寮忕粍瑁呭叏灞€鐭╅樀锟?`y = A x` 璺嚎锟?
- 鐗瑰緛鍊间笌鏃跺煙闂鑳藉悜鏇村ぇ DOF 瑙勬ā鎺ㄨ繘锟?

楠屾敹璇佹嵁锟?026-04-12锛夛細
- matrix-free 澶ц妯¤矾寰勶細`cargo test -p fem-examples --lib hcurl_matrix_free_large_dof_apply_smoke` 閫氳繃锛岃锟?`n=40` 锟?`HcurlMatrixFreeOperator2D` 鐨勫ぇ DOF `y = A x` 搴旂敤锛坄n_dofs > 3000`锛夛拷?
- 鐗瑰緛鍊艰妯″寲璺緞锛歚cargo test -p fem-examples --lib hcurl_eigen_amg_preconditioned_lobpcg_smoke` 閫氳繃锛岃锟?AMG 棰勬潯锟?LOBPCG 澶ц锟?smoke锛坄free DOF > 200`锛夛拷?
- 鏃跺煙瑙勬ā鍖栬矾寰勶細`cargo test -p fem-examples --lib first_order_3d_large_dof_single_step_smoke` 閫氳繃锛岃鐩栦竴闃跺叏锟?3D 锟?DOF 姝ヨ繘 smoke锛坄n_e > 250`, `n_b > 120`锛夛拷?

---

## 5. 杩戞湡鎵ц椤哄簭锛堝缓璁級

1. 鍏堣ˉ锟?Maxwell 搴旂敤閾撅紙API 涓庡洖褰掔煩闃碉級锟?
2. 鍐嶆帹锟?solver 鏋舵瀯瀹屽鍖栵紙mixed operator + 杈圭晫/鎹熻€楅」锛夛拷?
3. 鍐嶅畬锟?Quad/Hex 鍏冪礌鏃忕鍒扮闂幆锛堜紭锟?Hex ex3锛夛拷?
4. 鏈€鍚庤ˉ鎬ц兘灞傦紙PA/matrix-free 涓庡彲鎵╁睍鐗瑰緛鍊艰矾绾匡級锟?

璇ラ『搴忕敤浜庢帶鍒剁淮鎶ら潰澧為暱锛岀‘淇濇瘡涓€姝ラ兘鑳戒互鈥滀唬锟?+ 娴嬭瘯 + 鏂囨。鐘舵€佲€濋棴鐜獙鏀讹拷?

---

## 6. 缁存姢绾﹀畾

姣忔閲岀▼纰戞帹杩涘悗鑷冲皯鏇存柊浠ヤ笅涓夊锟?

1. 鏈枃妗ｏ細鐘舵€併€佷紭鍏堢骇銆侀獙鏀舵潯鐩拷?
2. `MFEM_MAPPING.md`锛氬叏鍩熸槧灏勪笌 remaining items锟?
3. Maxwell 涓撻」缁嗚妭涓庢祴璇曡瘉鎹細鑻ヤ粨搴撳瓨锟?`MAXWELL_GAPS.md` 鍒欏悓姝ュ洖鍐欙紱鑻ヨ鏂囨。宸插苟锟?绉婚櫎锛屽垯浠ユ湰鏂囨。涓哄敮涓€涓撻」鏉ユ簮锟?

---

## 7. 瓒呰秺 MFEM 鑳藉姏鐐硅窡韪紙Beyond-MFEM锛?
> 鐩殑锛氳褰曗€滈潪 MFEM 瀵归綈椤光€濅笌鈥滃湪鐜版湁瀵归綈椤逛箣涓婃柊澧炵殑宸ョ▼鍖栬兘鍔涒€濓紝閬垮厤鍏惰璇綊绫讳负 parity 宸ヤ綔銆?>
> 鐘舵€佸浘渚嬶細鉁?宸插畬鎴?路 馃敤 杩涜涓?路 馃敳 瑙勫垝涓?
### 7.1 鑳藉姏鍙拌处锛堝凡钀藉湴锛?
| ID | 鑳藉姏鐐?| 鐘舵€?| 浼樺厛绾?| 浠ｇ爜浣嶇疆 | 閲岀▼纰?| 璐熻矗浜?| 楠屾敹鐘舵€?|
|---|---|---|---|---|---|---|---|
| BMF-001 | 缁熶竴澶氱墿鐞嗚€﹀悎闂鎶借薄锛坆lock residual / block Jacobian锛?| 鉁?| P0 | `crates/solver/src/multiphysics.rs` | M1 | TBD | 宸查€氳繃鍗曟祴 |
| BMF-002 | 鍗曚綋鑰﹀悎绾挎€х瓥鐣ュ彲鍒囨崲锛圙MRES / 2x2 Schur锛?| 鉁?| P0 | `crates/solver/src/multiphysics.rs` | M1 | TBD | 宸查€氳繃鍗曟祴 |
| BMF-003 | 鐑?缁撴瀯鑰﹀悎宸ョ▼绀轰緥锛堢ǔ鎬佸崟浣?+ 鐬€?split + 鐬€?IMEX锛?| 鉁?| P0 | `examples/mfem_ex44_thermoelastic_coupled.rs` | M2 | TBD | 宸查€氳繃绀轰緥娴嬭瘯 |
| BMF-004 | 鏂规硶瀵规瘮涓?CSV 瀵煎嚭锛堝惈 dt/steps sweep锛?| 鉁?| P1 | `examples/mfem_ex44_thermoelastic_coupled.rs` | M2 | TBD | 宸查€氳繃绀轰緥娴嬭瘯 |
| BMF-005 | 2D Tri3 鐐瑰畾浣嶅櫒锛坆arycentric + nearest fallback锛?| 鉁?| P0 | `crates/mesh/src/point_locator.rs` | M1 | TBD | 宸查€氳繃鍗曟祴 |
| BMF-006 | H1-P1 闈炲尮閰嶈妭鐐规彃鍊间紶閫掞紙source鈫抰arget锛?| 鉁?| P0 | `crates/assembly/src/transfer.rs` | M1 | TBD | 宸查€氳繃鍗曟祴 |

### 7.2 璺嚎鍥撅紙寰呮帹杩涳級

| ID | 鐩爣 | 鐘舵€?| 浼樺厛绾?| 鐩爣閲岀▼纰?| 渚濊禆 | 楠屾敹鏍囧噯 |
|---|---|---|---|---|---|---|
| BMF-101 | 闈炲尮閰嶄紶閫掓墿灞曞埌 3D Tet4锛坙ocator + transfer锛?| 鉁?| P0 | M3 | BMF-005/BMF-006 | 宸查€氳繃 3D 鐐瑰畾浣嶄笌 3D 绾挎€у満浼犻€掑崟娴?|
| BMF-102 | 浠庤妭鐐规彃鍊兼墿灞曞埌绉垎鐐归┍鍔?L2 鎶曞奖 | 鉁?| P0 | M3 | BMF-006 | 宸查€氳繃 L2 鎶曞奖绾挎€х簿纭€т笌鏀舵暃鐜囧崟娴?|
| BMF-103 | 瀹堟亽浼犻€掞紙conservative transfer锛変笌閫氶噺璇樊璇勪及 | 鉁?| P1 | M4 | BMF-102 | 宸叉彁渚涘畧鎭掍慨姝ｆ姤鍛婁笌杈圭晫鍑€閫氶噺璇樊璇勪及锛屽苟鏈夊崟娴嬮獙鏀?|
| BMF-104 | 涓庡鐗╃悊鑰﹀悎宸ヤ綔娴佹墦閫氾紙璺ㄧ綉鏍艰嚜鍔ㄦ槧灏勶級 | 鉁?| P1 | M4 | BMF-101/BMF-102 | ex44 steady/split/IMEX 璺緞鍧囧凡鏀寔涓€閿垏鎹㈠悓缃戞牸/寮傜綉鏍煎苟閫氳繃鍥炲綊娴嬭瘯 |

### 7.3 閲岀▼纰戝畾涔夛紙寤鸿锛?
1. M1锛氭眰瑙ｅ櫒鎶借薄涓?MVP 浼犻€掕兘鍔涘彲澶嶇幇銆?2. M2锛氬伐绋嬬ず渚嬩笌鏂规硶瀵规瘮閾捐矾绋冲畾銆?3. M3锛氫笁缁翠笌 L2 鎶曞奖鑳藉姏杈惧埌鍙敤鍩虹嚎銆?4. M4锛氬畧鎭掍紶閫?+ 鑷姩鏄犲皠宸ヤ綔娴侀棴鐜€?
### 7.4 楠屾敹鍛戒护锛堝綋鍓嶅彲澶嶇幇锛?
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

