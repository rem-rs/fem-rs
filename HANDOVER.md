# HANDOVER — pex6 深水区交接（2026-08-30）

> 目标：让 `examples/mfem_pex6_parallel_amr.rs`（pex6）在 **np2** 下 unknowns/marked
> AMR 轨迹对齐 C++ MFEM `ex6p`（v4.9，`/home/quan/mfem49_mpi`）。
> C++ 参考二进制：`/home/quan/bin/ex6p_cpp`（含 CXX_MARKED 打印，已编译）。
> 比对脚本：`SKIP_NP4=1 bash examples/compare/pex_compare.sh pex6`。

## 当前状态（务必先跑确认）

```
np1 串行：全轨迹 = C++（已提交 49c7cd1 起保持）
  unknowns 31/101/171/291/386/526/1006/1386/1611/2371/4036 = C++
  marked   20/25/40/25/50/155/115/80/270/530               = C++
  it2 global_max=4.176412896211e-3（= C++）

np2 并行：it0-it3 unknowns 全对齐（31/101/171/291），it2 marked=40 = C++
  it3 marked=1 vs C++/np1 25   ← 遗留深水区（本轮唯一剩余问题）
  it3 global_max≈4.074e-3 vs np1 2.2297e-3（差 1.8 倍）
```

快速验证命令（md 291 只跑到 it3，np2 约 60-90s）：
```bash
./target/release/examples/mfem_pex6_parallel_amr.exe --ranks 2 -m data/star.mesh -md 291 -no-vis
# 期望：unknowns 31/101/171/291 + marked 20/25/40/1（it3 遗留）
```

## 提交历史

| commit | 内容 | 推送 |
|--------|------|------|
| e227741 | PᵀKP/Pᵀf 悬挂约束跨 rank 归并（H1 正确性基础） | 已推 |
| 49c7cd1 | 串行 l2_zz 改 PᵀAP 真解消元 → np1 全轨迹对齐 C++ | 已推 |
| **cf924a3** | par_l2zz 改 PᵀKP 跨 rank + recover 系数修复 + 全局悬挂边 | **未推（网络故障，先 `git push`）** |

## 已确认的关键机制（参考 MFEM 源码）

1. **MFEM RT0 悬挂 flux 约束符号** = `slave 边低端点 == master 边低端点`（全局节点编号序），
   等价于 `GetTransferMatrix`（`fe_base.cpp` `LocalInterpolation_RT`）的几何投影。
2. **MFEM 节点编号**（`ncmesh.cpp:2478` `UpdateVertices`）：SFC 序（`leaf_sfc_index`=Hilbert）
   扫描 leaf 元素，元素内节点 j=0..3 **首次出现**编号新节点。顶层（原始）节点保持 0..N0-1。
3. **MFEM RT0 dof2nk**（`fe_rt.cpp` 构造函数推导）= `[0,1,0,3]`（非 [2,1,2,1]）；
   `nk` 表 = `[(0,-1),(1,0),(0,1),(-1,0)]`。几何法向投影路径需此表（当前未用）。
4. 行替换法（slave 行 x_s−c·x_m=0）**非等价**于 PᵀAP：丢 A_sm/A_ss/b_s 折叠贡献（串行 it3: 40 vs 25）。
5. 串行 PᵀAP（`l2_zz.rs` 344 行起）：A_true=PᵀAP、b_true=Pᵀb、x=Py、slave 链式展开。
   par 用 `apply_hanging_constraints`（`par_csr.rs` 273 行）= PᵀKP/Pᵀf 跨 rank 折叠 + identity 行。
   **recover 系数必须从 c0（±0.5）开始**（不是 1.0）——cf924a3 修的关键 bug。

## it3 遗留根因（本轮精确定位，唯一剩余问题）

np2 it3 marked=1 vs C++ 25：RT0 slave sign 用**本地节点 id 序**（`lo==pa.min(pb)`），
但 np2 的本地 id 序 ≠ MFEM 创建序（partition 重排累积）→ it2 恰好对（40）、it3 错（二级悬挂）。

尝试把 rebuild 的新节点 gid 改为 MFEM 序（按 `(new_elem_gid, 元素内 j)` 全局归并排序），
**失败原因（关键发现）**：
- `new_elem_gid`（`par_amr.rs` 341-413 行）的 child 序 k 是 **每 rank 本地 refined 输出序**
  （`child_k[pg]` 计数器），**跨 rank 不一致**：同一物理 child 在不同 rank 的 new_elem_gid 值不同。
  实证：单 rank 物理边 gkey=(0,11) 的 key=(0,1)，np2 的 key=(0,0)。
- 因此 `(new_elem_gid, j)` 排序**不是全局一致的 MFEM SFC 序** → gid sign 在 np2 错
  （单 rank gid sign it2=40 对因为单 rank partition 是 identity；np2 错 it2=28）。

### 下一步正确方向（二选一，推荐 A）

**A. child 序 k 用几何确定的 Hilbert child index**（跨 rank 一致，推荐）：
- quad 4-split 的 4 个 child 有固定 Hilbert 序（child 0..3 对应确定象限/几何位置）。
- 在 `rebuild_partition_nc` 341-413 行，`child_k[pg]` 计数器改为**按 child 中心坐标/几何确定
  Hilbert index（0..3）**，而非"本地 refined 输出中出现顺序"。
- 这样 `new_elem_gid = prefix[parent_gid] + Hilbert_child` 对同一物理 child 跨 rank 一致，
  `(new_elem_gid, j)` 排序 = 全局 MFEM SFC 序。
- 实现后验证：单 rank 强制 par 的 gid sign it2=40（当前已对）+ **np2 it2 也应 = 40**（当前 28），
  it3 marked 应 → 25。
- 注意：NCStateQuad::refine 输出序声称是 Hilbert 序（341 行注释），但每 rank 的"出现顺序"
  受本地 refine 元素集影响 → 不能直接用出现计数，必须几何确定。

**B. 实现全局 Hilbert SFC**（完整网格 leaf_sfc_index）：改动大，A 失败再考虑。

### 关键验证脚本（调试用）

```bash
# 单 rank 强制 par_l2zz（隔离跨 rank，partition identity → gid=本地 id=MFEM 序）
PEX6_FORCE_PAR_L2ZZ=1 ./target/release/examples/mfem_pex6_parallel_amr.exe --ranks 1 -m data/star.mesh -md 171 -no-vis
# 期望 it2 marked=40、global_max=4.176412896210e-3（= 串行）——当前已满足（PᵀKP 数学正确）

# 对比 it3 global_max（np1 vs np2）
PEX6_TRACE=1 ./target/release/examples/mfem_pex6_parallel_amr.exe --ranks 2 -m data/star.mesh -md 291 -no-vis
# 但注意：PEX6_TRACE 的 PEX6_TH 打印已删（在 pex6 261 行附近），需临时加回

# 加/删调试打印的位置：
#   par_l2zz.rs: slave_deps 检测后（PAR_SLAVES）、recover 后（PAR_RECOVER）
#   l2_zz.rs: 串行 slave_deps（SER_SLAVES）、PᵀAP 解后（SER_SLAVE_FLUX）
#   par_amr.rs: edge_gid 分配（REBUILD_MFEM_ORDER/MFEM_GID_ORDER——注意该方案已回滚）
#   pex6: hang_global 归并后（PEX6_HANG_GLOBAL）、threshold 计算处（PEX6_TH）
```

## 环境与坑

- 工作目录 `C:\Users\lilu\works\fem-pro\fem-rs`（WSL 视角 `/mnt/c/Users/lilu/works/fem-pro/fem-rs`）。
- 环境无 python（用 node 写对比脚本）。`cargo build --release --example mfem_pex6_parallel_amr` 有时慢（后台跑）。
- np2 完整跑 `-md 3000` 会超时（it4+ 慢）；调试用 `-md 171/291/400`。
- **np1/np2 gid 空间不同**：跨 np1/np2 对比必须用**物理坐标**（uc(x,y)），勿用 (gid,li)。
- 临时脚本/产物放 `tmp/` 或 `output/`，勿放项目根目录。
- C++ 端 ex6p.cpp 有调试 patch（CXX_MARKED/CXX_U）——只在重编参考时需要；参考二进制已编译好。
- MFEM 源码参考（WSL）：`/home/quan/mfem49_mpi/fem/fe/fe_base.cpp`（GetTransferMatrix）、
  `fem/fe/fe_rt.cpp`（RT0 dof2nk）、`mesh/ncmesh.cpp`（OrientedPointMatrix:3949、UpdateVertices:2478）、
  `fem/fespace.cpp`（约束构建 944-1270）。

## 测试

```bash
cargo test --release -p fem-assembly -p fem-parallel -p fem-mesh   # 全绿（502+36+60 等）
SKIP_NP4=1 bash examples/compare/pex_compare.sh pex6               # OK (dof=31 三 np 一致)
```

## 提交时注意

- cf924a3 未推送（github 网络故障：`Failed to connect to github.com port 443`）——**先 `git push`**。
- 不提交生成的 mesh/gf/sol 输出文件。
