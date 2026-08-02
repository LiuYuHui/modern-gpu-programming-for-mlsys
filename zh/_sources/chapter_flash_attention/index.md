(chap_flash_attention)=
# Flash Attention 4

:::{admonition} 概览
:class: overview

- Attention 会运行两个 MMA，并在两者之间插入 softmax，因此它不能像 GEMM 那样简单重复一个 MMA。
- 这个 kernel 会把第一部分的硬件 primitive（TMA、`tcgen05`、TMEM、barrier）和第三部分的 GEMM 技术组合起来，并加入 warp role、online-softmax rescaling、causal masking 和 GQA。
:::

Attention 是决定 transformer 能否真正跑起来的 kernel，也是前面构建的所有东西最终必须协同工作的地方。我们为 GEMM 组装的每个部件都会延续到这里：TMA tile movement、`tcgen05` MMA、TMEM、warpgroup register tile，以及显式 barrier。

挑战在于，attention 不是重复一个 MMA。它是两个 MMA，中间夹着真正的工作：online softmax、causal masking，以及把先前 block 和后续 block 放到同一 scale 下的 rescaling。

新的难点就在这个中间阶段。普通 matmul 只需要向 accumulator 里相加；attention 在新的 key 和 value 流入时，必须重新访问并 rescale 已经计算过的结果。softmax 本身也会在两个 Tensor Core MMA 之间运行在 CUDA core 上，因此 exponential 和 row-wise reduction 会直接位于 critical path 上。

这就是为什么 attention 优化中很大一部分其实是 softmax 优化：重写 `exp`，并把 softmax 与 MMA 重叠起来，而不是停下来等待它。

本章目标不是从头重新推导 Flash Attention。我们只保留足够的算法背景，让 kernel 可读；然后把注意力放在真正新的部分：这个算法如何变成 TIRx。

最清晰的切入方式，是跟踪一个 tile 如何流过 kernel。`Q`、`K` 和 `V` 作为输入 tile 进入，从 GMEM 加载到 SMEM。score MMA 把 `Q` 和 `K` 相乘，在 TMEM 中生成 score tile `S`。Softmax 把 `S` 转成 numerator tile `P`，value MMA 再把 `P` 和 `V` 组合起来，更新输出 accumulator `O`。

到目前为止，这看起来像两个 matmul 粘在一起，但其中有一个 GEMM 从未处理过的转折：只要 running softmax maximum 发生变化，到目前为止累积出来的 `O` 就突然处在错误的 scale 上。它必须先被 rescale，下一次 value MMA 才能安全地加到其中。下面几节会先追踪这条路径，然后再展示 TIRx 如何把每个阶段交给一个 warpgroup，并把这些阶段连接起来。

## 算法形状

在把 tile 放进内存之前，我们需要先知道这些 tile 服务的算法。对于一个 query block，Flash Attention 计算：

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

按字面理解，这个公式要求形成完整 score matrix `S = QK^T`，对它做 softmax，再乘以 `V`。这恰好是我们不能使用的方法，因为完整的 `S` 太大了。seq=4096 时，每个 head 大约有 16M 个元素，fp32 下约 64 MB，比 SMEM 或单个 128x512 TMEM region 大几个数量级。片上根本没有地方放它。Flash Attention 的答案是完全不 materialize `S`。相反，它按 block 流式处理 `K/V`，并携带三个 per-row running state 来概括目前看到的一切：

- `row_max`：到目前为止见过的最大 score。
- `row_sum`：softmax 的 running denominator。
- `O`：running output accumulator。

随着新 block 到来，streaming update 会保持这些 state 正确。微妙之处在于，每处理一个 block，running max 都可能变大；一旦变大，所有在旧 max 下计算出的东西就都处在错误 scale 上。因此在加入新贡献之前，我们要先把旧 state 拉回到新的 scale：

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

这里单个 `scale` factor 同时承担两个作用：它 rescale running denominator，也 rescale running output，让早期 block 和后续 block 的贡献最终在同一个 scale 下度量。

上面的伪代码使用自然 `exp` 和显式 `/sqrt(d)`，因为这样最容易读；但 kernel 采用了更便宜的路径。它把 `1/sqrt(d)` 和 `log2(e)` 都折叠进一个常量 `scale_log2 = log2(e)/sqrt(d)`，并对 raw score 使用硬件 `exp2` 计算所有指数，利用恒等式 `exp(x/sqrt(d)) = exp2(x * scale_log2)`。动机很简单：在这个硬件上，`exp2` 比自然 `exp` 更快。

继续之前，有一点值得明确：这里的 `P` *不是*最终归一化后的 attention matrix。它只是当前 K/V block 的 softmax numerator。归一化被刻意推迟，只有在最后一个 block 之后，kernel 才会写出 `O / row_sum`。

对 TIRx 来说，知道算法计算什么只是一半图景。另一半是 kernel 运行时*每个 tile 住在哪里*，因为这决定了 layout 和 barrier 代码。`S`、`P` 和 `O` 都是 tile value，每个都有自己的位置：

- `S` 是 score tile。score MMA 会把它写到 TMEM。
- `P` 是 softmax numerator tile。Softmax 从 TMEM 把 `S` 读入 register，计算 `P = exp((S - m_new) / sqrt(d))`，然后把 `P` 写回 TMEM。
- `O` 是 output accumulator tile。value MMA 从 TMEM 读取 `P`，从 SMEM 读取 `V`，然后累加到 TMEM 中的 `O`。

前面提到的 rescale 也是一个 tile operation，而不是一段 scalar bookkeeping：当 `row_max` 变化时，旧的 `O` 会从 TMEM 读出，在 register 中相乘，再写回 TMEM，然后下一次 value MMA 才会累加到它里面。后续每节都会遵循同样结构：tile placement、hardware path，以及证明下一个 consumer 可以运行的 barrier。

## Tile-Primitive 图

有了 running state 和它们的位置，就可以把算法展开成具体的 tile 移动序列。对一个 K/V block，kernel 会自上而下走过这条 tile 路径：

```text
Q, K, V in GMEM
  -> Q, K, V in SMEM        by TMA load
  -> S in TMEM              by score MMA: QK^T
  -> P in TMEM              by softmax numerator: TMEM -> RF -> TMEM
  -> O in TMEM              by value MMA: P V
  -> O in GMEM              by normalization, SMEM staging, and TMA store
```

与 GEMM 的差异可以归结为一行。GEMM 是重复一条 MMA chain；FA4 有两个 MMA phase，softmax 位于这条 chain 中间。后面几乎所有内容都是这个额外 stage 的结果。

如果把这条短路径展开成显式 producer-consumer edge，就得到完整图：

| Stage | Tile movement 或 compute | TIRx primitive | Hardware path |
|-------|--------------------------|----------------|---------------|
| Load Q/K/V | GMEM tiles -> SMEM tiles | `Tx.copy_async(..., dispatch="tma")` | TMA load |
| Score MMA | SMEM 中的 Q 和 SMEM 中的 K -> TMEM 中的 score tile `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax read | TMEM 中的 `S` -> warpgroup register tile | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax write | register 中的 numerator tile `P` -> fp16 TMEM view | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store，然后 `tcgen05.wait.st()` |
| Value MMA | TMEM 中的 `P` 和 SMEM 中的 V -> TMEM 中的 output accumulator `O` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | 带 TMEM operand 的 `tcgen05.mma` |
| Correction | TMEM 中的 `O` -> register -> TMEM 中的 `O` | TMEM readback、register multiply、TMEM store | `tcgen05.ld` / TMEM store |
| Epilogue | TMEM 中的最终 `O` -> register -> SMEM -> GMEM | TMEM readback、`Tx.copy`、TMA store | `tcgen05.ld` + TMA store |

新增行是 softmax 和 correction。两者都会增加 TMEM -> register -> TMEM traffic，也都会在 score MMA 与 value MMA 之间创建额外 handoff。

**让你的 agent 试试**：让它只追踪上面的短路径。对每个箭头，命名 producer stage、consumer stage、source tile、destination tile 和 hardware path。然后问哪些箭头在 GEMM 章节中不存在。

## Warp 角色和 Scope

确定 data path 之后，自然的下一个问题是每个 stage 实际由谁运行。这里每个 CTA 有 4 个 warpgroup，总共 512 个线程；它们不是按接触哪些数据来拆分，而是按 warpgroup 做*哪类工作*来拆分：

- WG3 驱动硬件 engine：TMA load、MMA 和 TMA store。
- WG0、WG1 和 WG2 执行这些 engine call 之间 register-heavy 的数学工作：softmax、correction 和 epilogue。

具体角色表如下：

| Owner | Role | 工作 |
|-------|------|--------------|
| WG3, warp 1 | TMA load | 把 Q、K、V tile 从 GMEM 加载到 SMEM |
| WG3, warp 0 | MMA | 发起 score MMA 和 value MMA |
| WG3, warp 2 | TMA store | 把最终 O tile 从 SMEM 存回 GMEM |
| WG0 | Q stage 0 的 Softmax | 从 TMEM 读取 S，计算 P，把 P 写回 TMEM |
| WG1 | Q stage 1 的 Softmax | 对第二个 Q pipeline stage 做同样工作 |
| WG2 | Correction 和 epilogue | Rescale TMEM 中的 O，归一化，并 staged 输出 |

很容易把“两个 Q stage”误读成两个 attention head，但它们并不是。它们只是 Q pipeline 中的两个 slot，WG0 拥有一个，WG1 拥有另一个，因此两个 Q tile 可以同时 in flight。这就是 softmax 工作出现两次的原因：一次在 WG0，一次在 WG1。

代码用 symbolic coordinate 选出这些角色：

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

阅读 kernel 时，先找到 role branch。它会告诉你嵌套在里面的每个 tile primitive 由哪支队伍拥有。

- WG3 warp 1 启动 TMA load command。一个 elected lane 发起 copy，TMA engine 移动 tile。
- WG3 warp 0 发起 `tcgen05.mma` 指令。
- WG0 和 WG1 在 full warpgroup scope 下运行 softmax。
- WG2 在 full warpgroup scope 下运行 correction 和 epilogue 工作。

一个不对称性最终塑造了整个 barrier graph：无论 score 还是 value，*每个* MMA 都只由 WG3 warp 0 发起。WG0 和 WG1 完全不发起 MMA。它们只消费 score tile、运行 softmax，并把 `P` 写回 TMEM。

这种分离正是 softmax 周围需要 barrier 的原因。`s_ready` 把 score tile 从 MMA warp 传给 softmax；`p_o_rescale` 则携带 `P`，以及对 value MMA 来说安全的 `O` slot，这个 slot 要么已经被 rescale，要么因为不需要 rescale 而被 release。后面我们会反复回到这两个名字。

## 阅读代码片段

本章中的片段摘自 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)，因此它们不可避免会引用一些在未展示 kernel 部分中定义的名字。那些自解释的名字（`wg_id`、`warp_id`、`BLK_M`/`BLK_N`、`HEAD_DIM`、`kv_stage`、`SMEM_PIPE_DEPTH_*` / `TMEM_PIPE_DEPTH` depth、`should_accumulate`，以及这里为 1 的 `CTA_GROUP`）会在下面第一次需要时介绍。其余名字在这里用一行表格说明，方便你在片段里遇到陌生名字时马上查阅：

| 名称 | 含义 |
|------|---------|
| `q_stage`, `i_q` | Q pipeline stage，0 或 1，也就是哪个 Q tile slot（`SMEM_PIPE_DEPTH_Q = 2`）。在 WG0/WG1 softmax 内部，warpgroup 自己的 `wg_id`（0 或 1）*就是*同一个 stage index，因此 `S_region[q_stage]`、`P_region[wg_id]` 和 `O_region[i_q]` 都选择同一个 Q stage |
| `MMA_N` | TMEM column 中的 score/output tile 宽度（128） |
| `MMA_K` | `P`/`V` 列中的 MMA inner-K step（16）；`K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | value-MMA schedule 的 split point（见*两个 MMA Phase*）；第一个 value MMA 覆盖列 `0:K_SPLIT`（`6 * MMA_K = 96`） |
| `should_rescale` | WG2 per-row flag：下一次 value MMA 前旧的 `O` 是否需要 rescaling（用 `any_sync` 跨 warpgroup reduce） |
| `rescale_threshold` | 小 row-max 变化的 skip threshold；当前 kernel 使用 `8.0`，被跳过的 rescale 会把 `acc_scale` 精确设为 `1.0` |
| `scale_log2` | log2 单位下的 softmax scale，`log2(e)/√d`，因此 `P = exp2((S - m) * scale_log2)` |
| `acc_scale` | softmax 通过 SMEM mailbox 传给 WG2 的 per-row rescale factor |
| `chunk_start`/`chunk_end`, `p_start`/`p_end` | 正在读取 / 写入的 32-wide softmax chunk 的列范围 |

## 两个 MMA Phase

对每个 streamed K/V tile，Flash Attention 运行两个 MMA phase，并由 softmax 连接它们：

```text
Q, K -> score MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

可以把它看成三个 producer 连成的 pipeline。第一个 MMA 产生 attention score `S`，softmax 把 `S` 转成 numerator `P`，第二个 MMA 消费 `P` 来更新 output accumulator `O`。除以 `row_sum` 的归一化会延后到 epilogue，等所有 K/V tile 都贡献完再做。

下面每个 tile op 都会得到与 GEMM step 中相同的 **scope / layout / dispatch** 卡片，并额外增加一行 **Handoff**，用来命名把 tile 传给下一个角色的 barrier。

compute 代码从不直接使用原始 TMEM column number。kernel 会把单个 TMEM allocation 划分成 per-stage view（`S_region`、`P_region`、`O_region`），并按 pipeline stage 索引它们（`S_region[q_stage]`、`O_region[i_q]`、`P_region[i_q, 0:K_SPLIT]`）。这些 view 会在后面的 “TMEM Layout 和复用” 小节中用 `T.TMEMStages` 定义；现在只需要把每个 region 看成同一块物理 TMEM 的命名 slice。

### Score MMA

两个 phase 中的第一个是 score MMA，也就是开启每个 K/V iteration 的 matmul。它计算：

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

并把 `128 x 128` score tile 写入 TMEM：

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

我们可以提出 GEMM 章节对每个 tile op 都问过的四个问题：谁运行它、tile 住在哪里、如何 dispatch，以及如何 handoff：

> **Tile-primitive 读法：Score MMA**
> - Scope：WG3 warp 0 发起它；一个 elected lane arrive `s_ready`。
> - Layout：SMEM 中的 Q、K → TMEM 中的 `S`（`S_region[q_stage]`）。
> - Dispatch：`tcgen05`。
> - Handoff：`s_ready`（→ softmax）。

单个 elected thread 在 `s_ready` 上 arrive，就是完整的 handoff。它宣布这个 score tile 已经完成，softmax warpgroup 现在可以读取它。

### 两个 MMA 之间的 Softmax

两个 MMA 之间是 softmax，这个 stage 会把 score tile `S` 转成 numerator tile `P`。它的读法卡片是：

> **Tile-primitive 读法：Softmax**
> - Scope：WG0（Q stage 0）/ WG1（Q stage 1），full warpgroup。
> - Layout：TMEM 中的 `S` → register → fp16 TMEM 中的 `P`（`P_region[wg_id]`）。
> - Dispatch：用 `tcgen05.ld` 读取，用 TMEM store 写入；中间在 register 中做 row-wise math。
> - Handoff：等待 `s_ready`；arrive `p_o_rescale`（前 96 列）和 `p_ready_2`（最后 32 列）。

这个 stage 完全没有 GEMM 对应物。WG0/WG1 等待 score tile 在 `s_ready` 上到达，然后每次从 TMEM 读取一个 register-sized chunk：

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    S_region[wg_id, chunk_start : chunk_end],
)
```

这是 warpgroup scope 下的 TMEM-to-register tile read。现在 score 已经在 register 中，softmax warpgroup 会按顺序做三件事：

1. 计算 row max 和 row sum，
2. 计算 softmax numerator tile `P`，
3. 把 `P` 作为 fp16 写回 TMEM。

最后一步如下：

```python
Tx.copy_async(
    P_region[wg_id, p_start : p_end],
    p_chunk[:, p_start : p_end],
)
```

既然刚刚已经在 register 中算完 `P`，为什么还要把 `P` 写回 TMEM？因为 value MMA 需要把 `P` 当作 *tile operand*，而 MMA 不能把分散在每个线程中的 scalar register 当成矩阵来读。这个 kernel 中 MMA 可读取的 `P` 形式是 `P_region`，它是 fp16 TMEM alias `tmem_as_f16` 上的一个 view。因此这个 writeback 不是多余的数据移动；它会把 `P` 放进下一个 MMA 唯一能消费的形状中。

### Value MMA

第二个 phase，也就是结束每个 K/V iteration 的阶段，是 value MMA。它计算：

$$O = O + P_{\text{block}}V_{\text{block}}$$

这个 MMA 运行时，`O` 已经被放到当前 K/V block 所需的正确状态：第一个 block 上初始化，后续 block 上完成 rescale。因此 MMA 只需要累加。它与 GEMM 的差异在于 operand 住在哪里：A operand 是 TMEM 中的 `P`，B operand 是 SMEM 中的 `V`，accumulator `O` 也在 TMEM 中：

```python
# First sub-MMA: columns 0:K_SPLIT (the first 96 of P / rows of V).
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
# The second sub-MMA (same form, accum=True, gated on p_ready_2) covers the
# remaining columns K_SPLIT:BLK_N.
```

> **Tile-primitive 读法：Value MMA**
> - Scope：WG3 warp 0。
> - Layout：TMEM 中的 `P` + SMEM 中的 V → TMEM 中的 `O`（`O_region[i_q]`）。
> - Dispatch：带 TMEM operand 的 `tcgen05`。
> - Handoff：等待 `p_o_rescale`、`p_ready_2`、`kv_load.full`；arrive `o_ready`（→ epilogue）。

这个 operand placement 是两个 MMA 在硬件路径上的差异：

- Score MMA 从 SMEM 读取两个 operand：Q 和 K。
- Value MMA 从 TMEM 读取一个 operand：`P`。
- Value MMA 从 SMEM 读取另一个 operand：V。
- 结果累加到 TMEM 中的 `O`。

`accum=should_accumulate` flag 实现了算法中的“初始化还是相加”选择：对 query block 的第一个 K/V tile 它是 false，之后每个 tile 都是 true。

你可能还会注意到，value MMA 不是一次性运行，而是拆成 `96 + 32` 的 schedule：

1. Softmax 用四个 32 列 chunk 写入 `P`。
2. 只要前三个 chunk 准备好，value MMA 就开始处理 `P` 的前 96 列和 `V` 中对应行。
3. 最后 32 列等待 `p_ready_2`。
4. 第二个 MMA 消费最后这个 chunk，并完成整个 tile。

拆分的原因是让 Tensor Core 保持忙碌。如果把 value MMA 作为单条指令运行，整个 phase 都会 stall，直到四个 32 列 `P` chunk 全部完成 exponentiation 并写入。通过立刻对前三个 chunk 发起 MMA，kernel 会把最后一个 chunk 的 `exp` 和 TMEM write 与已经 in flight 的 96-wide MMA 重叠起来，把原本的空闲时间变成有效工作。

## TMEM Layout 和复用

`S`、`P` 和 `O` 都必须共享一个 `128 x 512` TMEM allocation，而它们打包到其中的方式，正是这个 kernel 中 barrier 和 layout 不可分割的原因：

下图直接展示了这种 packing：score slot、numerator slot 和 output slot 全都
共享一个 TMEM allocation，因此 barrier protocol 正是让复用合法的条件。

![TMEM Layout 示意图](../../img/tmem_layout_v3.png)

这张图可以读作一组 tile slot：

- Score slot 保存 `S = QK^T`。
- Numerator slot 保存 softmax exponentiation 之后的 `P` tile。
- Output slot 保存 fp32 `O` accumulator。

这些不是独立 buffer。它们是*同一个* allocation 的不同 region，而且共享并不是风格选择，而是被迫如此。Q-pipeline depth 为 2 时，两个 `S` slot（2 x MMA_N = 256 列）和两个 `O` slot（2 x MMA_N = 256 列）已经占满全部 512 个 fp32 列。没有空间留给 `P`，因此 `P` 只能通过更窄的 fp16 view alias 同一批字节。这样做之所以安全，唯一原因是每个 region 只会在前一个 consumer 完成后才被复用，而这个时机正是 barrier 所保证的。因此在 FA4 中，barrier 不只是调度；它们首先让这个 layout 合法。

这个 aliasing 技巧通过 `T.TMEMPool` 设置。kernel 先取得一个 fp32 view（`tmem`）用于 score 和 output accumulator，然后把 pool base 倒回 0，并在*同一批*物理字节上取得第二个 fp16 view（`tmem_as_f16`）：

```python
tmem_pool = T.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

由于 fp16 元素宽度只有一半，fp16 view 会在同一批字节上暴露两倍数量的可索引列，而 `P` 正是住在这片空间里，fp32 layout 本身没有给它留下空间。有了两个 view 后，kernel 用 `T.TMEMStages` 把 `S`、`P` 和 `O` slot 切成 staged region，让 compute 代码按 pipeline stage 索引，而不是直接使用原始列号：

```python
S_region = T.TMEMStages(tmem,        col_start=0,                       width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
O_region = T.TMEMStages(tmem,        col_start=MMA_N * SMEM_PIPE_DEPTH_Q, width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
P_region = T.TMEMStages(tmem_as_f16, col_start=MMA_N,                   width=BLK_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2)
```

`P_region` stride 中的 `* 2` 是 aliasing 明显泄漏进代码的地方。`S_region` 和 `O_region` 用 fp32 `tmem` 列来度量，而 `P_region` 用 fp16 `tmem_as_f16` 列来度量，后者宽度只有一半，因此 stage-to-stage 移动需要翻倍 stride 才能落到同一批物理字节上。不过一旦 region 定义完成，compute 代码就保持清爽：它写 `S_region[q_stage]`，读 `S_region[wg_id, ...]`，写 `P_region[wg_id, ...]`，并累加到 `O_region[i_q]`，从不触碰原始列索引。

**让你的 agent 试试**：让它解释这个 FA4 kernel 中的 fp32（`tmem`）和 fp16（`tmem_as_f16`）view。哪些物理 TMEM region 保存 `S`、`P` 和 `O`，为什么 `P_region` 的 stride 使用 `MMA_N * 2`？复用问题留到下一节：看完 barrier 表后，再检查每个 region 被复用前必须等哪些 consumer 完成。

## Barrier 如何连接角色

这是 kernel 中最难的部分，因此值得逐步进入。先从沿主 compute path 移动数据的少数 barrier 开始，其余内容可以暂时视作之后再查的 bookkeeping。data-ready handoff 包括：

| Handoff | 含义 |
|---------|---------|
| TMA load -> score/value MMA | Q、K 或 V 已到达 SMEM，可以喂给 MMA |
| score MMA -> softmax | TMEM 中的 `S` 已就绪 |
| softmax/correction -> value MMA | TMEM 中的 `P` 已就绪，并且 `O` 可以安全累加 |
| value MMA -> epilogue | TMEM 中的最终 `O` 已就绪 |
| epilogue -> TMA store | `O_smem` 已准备好存储 |

不在这张表里的东西都是 pipeline bookkeeping：这些 barrier 会释放某个 SMEM、TMEM 或 staging buffer，让另一个角色可以复用它。好处是每个 barrier 的读法都一样，无论它携带数据还是只做 bookkeeping，都可以看成 tile handoff。你只需要问：谁产生数据，谁消费数据，以及二者完成后哪个 buffer 变空。

下一张图把这些 handoff 压缩成两个 MMA phase 的精确 readiness gate：
score MMA 等待什么，以及 value MMA 累加之前必须等待什么。

![Flash Attention 4 MMA 输入 gate](../../img/flash_attention_main_handoff.png)

请把这张图读成一组 correctness gate，而不是 schedule。它回答的是“这个 MMA 发起之前必须满足什么”，并不描述时序。score MMA 等待 SMEM 中的 Q 和 K，然后产生 `S`。value MMA 同时等待三件事：SMEM 中的 V、来自 softmax 的 `P` tile，以及一个已由 WG2 release 或 rescale 的 `O` slot。softmax-to-value gate 被拆开，原因就是前面提到的：一旦 `P` 的前 96 列就位，value MMA 就可以开始，而 `p_ready_2` 会释放最后 32 列。

有一个 handoff 不符合 tile-readiness 模式：softmax-to-correction edge。softmax 不是传递 tile，而是通过一个 one-slot SMEM mailbox 向 WG2 传递一个 scalar（K/V loop 中是 `acc_scale`，epilogue 中是最终 `row_sum`）。由于这个 slot 每次 iteration 都会复用，必须用一对 `full`/`empty` barrier 保护它：

下图放大了这个 mailbox handshake，因此这对 barrier 应该被读成 scalar producer-consumer channel，
而不是 tile-ready gate。

![Flash Attention 4 softmax scale slot handshake](../../img/flash_attention_softmax_correction.png)

请把 `softmax_corr.full` 和 `softmax_corr.empty` 读成一对 producer-consumer barrier：

1. Softmax 在复用 scale/sum slot 前等待 `softmax_corr.empty`。
2. Softmax 把 `acc_scale` 或最终 `row_sum` 写入这个 slot。
3. Softmax 在 `softmax_corr.full` 上 arrive。
4. WG2 等待 `softmax_corr.full`，然后读取这个 slot。
5. WG2 在 `softmax_corr.empty` 上 arrive。
6. softmax warpgroup 可以在下一阶段复用这个 slot。

需要仔细区分 `softmax_corr.empty` 的含义和非含义。它只表示 WG2 已经消费了 scale/sum slot。它完全不说明 `P` 是否就绪，也绝对*不是*允许 value MMA 开始的 gate。那个 gate 是 `p_o_rescale`，它在 `P` 的前 96 列写入完成且 `O` slot 可以安全累加时触发。混淆两者是产生错误结果的经典来源。

有了主路径之后，完整 barrier 列表可以作为参考：

| Barrier | Producer -> consumer | 变得安全的内容 |
|---------|----------------------|-------------------|
| `q_load.full` | TMA load -> score MMA | Q SMEM tile 可以喂给 MMA |
| `q_load.empty` | 这个 Q stage 的所有 score MMA -> TMA load | Q SMEM stage 可以为下一个任务复用 |
| `kv_load.full` | TMA load -> score/value MMA | K 或 V SMEM tile 可以喂给 MMA |
| `kv_load.empty` | score/value MMA -> TMA load | K/V SMEM stage 可以复用 |
| `s_ready` | score MMA -> softmax | S TMEM tile 可以读取 |
| `p_o_rescale` | softmax + WG2 -> value MMA | P 的前 96 列在 TMEM 中，并且 O slot 对 value MMA 安全 |
| `p_ready_2` | softmax -> value MMA | P 的最后四分之一在 TMEM 中 |
| `o_ready` | value MMA -> epilogue | 最终 O accumulator 已就绪 |
| `softmax_corr.full` | softmax -> WG2 | `acc_scale` 或最终 `row_sum` 在 SMEM mailbox 中就绪 |
| `softmax_corr.empty` | WG2 -> softmax | WG2 读完后，同一个 SMEM mailbox slot 可以复用 |
| `corr_epi.full` | epilogue -> TMA store | O_smem 已准备好 store |
| `corr_epi.empty` | TMA store -> epilogue | O_smem stage 可以复用 |

和 GEMM 中一样，可以根据谁产生 signal 来预测 barrier 类型：

- TMA load 使用 `TMABar`，因为 TMA engine 会对自己的完成做 byte-count。
- MMA completion 使用 `TCGen05Bar`，因为 `tcgen05.commit` 会 signal completion group。
- 纯 thread-to-thread handoff 使用 `MBarrier`，参与线程会显式 arrive。

拆分后的 softmax-to-value handoff 值得仔细看。它使用两个 gate：

- `p_o_rescale` 会在 `P` 的前 96 列已经写入且 `O` tile 可以安全累加时允许 value MMA 开始。
- `p_ready_2` 释放 `P` 的最后 32 列，与上一节的 `96 + 32` value-MMA schedule 对应。

第一个 K/V block 是简单情况。WG2 会 pre-arrive `p_o_rescale`，因为还没有旧的 `O` tile 需要 rescale。

后续 block 必须更谨慎。WG2 只有在跳过不必要的 rescale，或已经完成旧 `O` 的 rescale 后，才会 arrive `p_o_rescale`。skip test 是刻意保守的：softmax 会计算 log2-scaled delta `(m_old - m_new) * scale_log2`；如果这个值仍高于 `-rescale_threshold`，说明新的 max 没有移动到值得 rescale 的程度，所以 kernel 保留旧 max，并把 `acc_scale` 精确设为 1.0。只有更大的 max jump 才会走 `exp2` 路径，并要求 WG2 rescale `O`。

随后 WG2 用 `any_sync` 跨 warpgroup reduce `should_rescale`。如果没有任何 row 需要更新，它就保持 `O` 不变。这个 skip 很重要，因为 rescale `O` 是对整个 accumulator 做一次完整的 TMEM -> RF -> TMEM read-modify-write；当 threshold 逻辑已经把 `acc_scale` 保持在 1.0 时，这完全是浪费工作。

注意，所有新的 barrier 都集中在一个地方。`s_ready`、`p_o_rescale`、`p_ready_2` 以及 softmax/correction pair 都是 softmax 周围的 barrier。它们存在的原因只有一个：score MMA 和 value MMA 不再相邻。register math、TMEM rewrite 和 output rescaling 现在位于二者之间，而每一步都需要自己的 handoff。

**让你的 agent 试试**：让它追踪一个 K/V block 如何经过 `s_ready`、`p_o_rescale`、`p_ready_2` 和 `o_ready`。对每个 barrier，问谁在 wait、谁在 arrive、哪个 tile 变得可以安全读取，以及之后哪个 storage 可以复用。

## Pipelining 结构

barrier 告诉我们一个角色消费 tile 前什么必须*就绪*。但它们没有告诉我们实际有哪些东西在*并发*运行，这就是现在要讨论的问题。两者确实不同：一个 correctness gate 可能在 producer 实际运行之前很久满足，也可能在之后很久才满足。

这里没有单一 pipeline depth，因为不同 tile stream 以不同速率移动。因此 kernel 为每类 stream 保留独立 ring：

- Q pipeline depth 2：一个 CTA 处理两个 Q stage。WG0 处理一个 stage，WG1 处理另一个。
- KV pipeline depth 3：K 和 V block 在 inner loop 中流动，同时复用相同的 Q stage。
- TMEM pipeline depth 2：每个 Q stage 拥有自己的 S/P/O TMEM slot，这些 slot 会在对应 barrier 触发后复用。

下图从 correctness gate 切换到 timeline 视角，展示这些独立 ring in flight 之后，哪些角色可以
大致同时活跃。

![Flash Attention 4 pipeline 结构](../../img/flash_attention_pipeline_v2.png)

请把它读成 timeline，而不是 barrier graph。它展示哪些角色在大致相同的时刻活跃；而前面的 barrier-flow 图用于检查精确的 producer-consumer wait。这两张图一起回答了本节开头提出的两个不同问题。

每一行对应代码中的一个 role branch：

- WG3 warp 1 发起 TMA load。
- WG3 warp 0 发起 score MMA 和 value MMA。
- WG0 和 WG1 为两个 Q stage 运行 softmax。
- WG2 release 或 rescale `O`，之后归一化最终输出。
- WG3 warp 2 发起 TMA store。

从左到右阅读这张图，就是在追踪一个代表性的 pipeline wave。load warp 从 `Q0`、`K[n-1]`、`Q1`、`V[n-1]` 开始，然后持续 stream 更低 index 的 K/V block。MMA warp 发起最初的 score MMA，产生 `S0` 和 `S1`，WG0/WG1 再把它们转成 `P0` 和 `P1`。

重要的是，MMA warp *不是*先运行所有 score MMA，再运行所有 value MMA。一旦两个 Q stage 都 primed，它会交错执行两类 MMA：对当前 `V` block 做 value MMA，然后对下一个 `K` block 做 score MMA，依此类推：

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

这种 interleaving 正是图中 score、softmax、correction 和 value 各行相互重叠，而不是整齐顺序执行的原因。

WG2 行标成 `release / rescale`，两半分别对应前面看到的两种情况。在第一个 K/V block 上还没有旧的 `O`，所以 WG2 只参与让 value MMA 继续的 handoff；在后续 block 上，它可能需要在 value MMA 累加之前 rescale 旧的 `O`。Normalization 和 TMA store 只发生一次，在 attention task 的最后一个 K/V block 之后执行。

没有单一 GEMM 风格的 pipeline 能描述 FA4，因为 Q、K/V 和 TMEM slot 都按照独立 schedule 前进。TIRx 把这些 schedule 显式保留下来，表现为独立 tile buffer、`PipelineState` cursor 和 barrier phase，而不是把 kernel 隐藏在一个单体 primitive 后面。代价是 moving parts 更多，收益是复杂性保持可见且可检查。

## Rescaling 和 Writeback

rescale 是必需的，不是可以删掉的优化。Online softmax 会随着每个新 score tile 提高 per-row maximum，而每当它提高时，早期 block 累积出的 `O` 都是按*旧* maximum 缩放的。这会让每个早期项大出 `exp(m_new - m_old)` 倍。如果跳过 correction，这些 block 会被过度加权，最终输出就是错的。修复方式是一个 TMEM → register → TMEM tile operation：

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

这项工作拆分到两个角色。Softmax 计算 per-row scale，并把它放入 SMEM mailbox；WG2 等待 `softmax_corr.full`，从 TMEM 读出当前 `O`，乘以这个 scale，然后把 `O` 写回：

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

值得强调的是，这是覆盖整个 `O` accumulator 的完整 TMEM → register → TMEM tile operation，而不是一点 scalar bookkeeping，它也有和其他 stage 相同的读法卡片：

> **Tile-primitive 读法：Correction（rescale）**
> - Scope：WG2，full warpgroup。
> - Layout：TMEM 中的 `O` → register → TMEM 中的 `O`（`O_region[i_q]`）。
> - Dispatch：用 `tcgen05.ld` 读取，用 TMEM store 写入；中间执行 register multiply。
> - Handoff：等待 `softmax_corr.full`；arrive `p_o_rescale`（→ value MMA）和 `softmax_corr.empty`（→ softmax）。

端到端追踪同步流程：

1. Softmax 把 scale value 写入 SMEM。
2. WG2 等待 `softmax_corr.full`。
3. WG2 rescale TMEM 中的 `O`。
4. WG2 在 `p_o_rescale` 上 arrive。
5. WG3 的 value MMA 现在可以消费 `P`，并累加到 rescaled `O` tile 中。

当 WG2 读完后，`softmax_corr.empty` release SMEM slot，这个 loop 就闭合了；softmax 可以在下一次 iteration 中复用 mailbox。

一旦 K/V loop 结束，WG2 会从 correction 切换到 epilogue。它等待最终的 `row_sum` 和 `o_ready`，从 TMEM 读取最终 `O`，乘以 `1 / row_sum`（最开始被延后的 normalization），cast 为 fp16，然后写入 `O_smem`。随后 WG3 的 TMA store warp 把 `O_smem` 带回 GMEM。

如果你计划扩展这个 kernel，有一个限制值得指出。它只计算 forward output，而 training forward pass 通常还会存储 backward pass 需要的 log-sum-exp（LSE）。加入 LSE 时有个 scaling 细节要注意：这个 kernel 把 `row_max` 保存为*原始*、未缩放 `QK^T` score 的最大值，而 `row_sum` 累加的是 `exp((S - row_max) / sqrt(d))`。因此形成 natural-log LSE 时，必须重新把 `1/\sqrt{d}` factor 应用到 `row_max` 上：

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

这个实现只输出 forward output，不写 LSE。

## Causal Masking

Causal attention 增加了一个约束（query 只能 attend 到位置不晚于自身的 key），kernel 用两种互补方式满足它：一种便宜，一种精确。

便宜的方式是完全跳过工作。很多 K/V block 完全位于对角线之上，对给定 Q block 没有任何贡献，因此 `get_n_block_max(...)` 会计算这个 block 可能需要的最后一个 block，loop 就根本不会加载或计算其余部分。

精确的方式处理跨过对角线的 block，其中有些列有效，有些无效。这些 block 仍然运行 score MMA，但 softmax 会在 exponentiation 之前 mask 掉无效列。对每一行，它会根据该行 query position 和 block offset 推导 column limit，保留不超过这个 limit 的列，并在 register 中把超过它的所有列设为 `-inf`，这样这些列既不会贡献到 row max，也不会贡献到 `exp2` numerator。

实现不会逐元素 branch，而是用 `mask_r2p(...)` 应用这个 limit；它会把 limit 转成覆盖整个 32-wide score chunk 的 bit mask，并一次性 mask 掉 chunk。完全位于对角线下方的 block 会保留所有列，完全不需要 mask。

从 tile-primitive 视角看，causal mode 根本没有改写 data path。它只裁剪 K/V trip count，并在 score MMA 与 `P` writeback 之间，把 masking step 插入到 register-resident softmax 中。

## GQA 支持

Grouped Query Attention 允许多个 query head 共享一个 K/V head。这样可以节省 memory bandwidth，但会带来一个 packing 问题：如何只保留一个 K/V tile，同时仍然让许多 query head 通过它？kernel 的答案是一次性处理一整组 query head，让它们对应同一个 scheduled `kv_head_idx`：

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

技巧在于重新解释 128 个 Q-tile row。对 `GQA_RATIO=4`，它们不再表示 128 个 sequence position；它们表示 32 个 sequence position 乘以 4 个 query head，并打包在一起，让四个 head 乘坐同一个 K/V tile。row decoding 为：

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

Q load 用一个 3D view 表达这种 packing。source 是自然的 `Q[batch, seq, qo_head, dim]` layout，而 destination 是同一个 SMEM tile，score MMA 稍后会把它作为扁平的 `128 x HEAD_DIM` operand 读取。view 负责调和二者，而且不需要任何 copy：

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K 和 V 从不会在内存中 expand，这正是 GQA 的意义：`kv_head_idx` 对应的单个 K/V tile 会被所有打包进 Q 行中的 `GQA_RATIO` 个 query head 复用。输出侧与输入侧镜像对应，epilogue 之后用匹配的 3D view 把 packed row 存回 `O[batch, seq, qo_head, dim]`。

结果是，GQA 完全存在于 Q-load 和 O-store 边界。在 compute path 内部，score MMA 仍然看到一个普通的 `128 x HEAD_DIM` Q tile，tile-primitive graph 的其余部分保持不变。

## Tile Scheduling

scheduler 的工作是把每个 CTA 映射到一个 `(batch, kv_head, m_block)` attention task，而正确策略取决于 masking 是否让这些 task 成本相同：

- Non-causal mode 使用 `FlashAttentionLinearScheduler`。每个 task 做相同数量的工作，因此只需要一个固定 CTA pool 按 `num_ctas` 前进，就能均匀分摊。
- Causal mode 使用 `FlashAttentionLPTScheduler`，因为 causal masking 会让工作量极不均匀：靠近开头的 Q block 大约只 attend 一个 K/V block，而靠近结尾的 Q block 会 attend 所有 block。naive split 会让一些 CTA 比其他 CTA 晚很多完成，因此 longest-processing-time scheduler 会把重 block 前置以均衡完成时间，同时仍然把附近的 batch/head task 放在一起以保持 L2 locality。

尽管二者不同，这两个 scheduler 暴露相同的 loop interface：

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # process one Q block against its K/V block range
    scheduler.next_tile()
```

唯一行为差异在于 `next_tile()` 做什么：在 non-causal mode 中，它把 CTA 推进到另一个 task；在 causal mode 中，它会在当前 task 后结束 loop。不管哪种方式，这都只是 scheduling 决策：它选择 CTA 拥有*哪个* attention tile，而不改变这个 tile 如何计算。loop 内部运行的 local primitive 始终相同：TMA load、score MMA、softmax、value MMA、correction、TMA store。

## 编译和验证

上面的内容都是 excerpt，因此要把它们组合起来并真正运行 kernel，我们需要从 `tirx-kernels` 导入真实实现，编译它，并用 torch reference 检查。完整 kernel 在 `tirx-kernels` 仓库的 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py) 中，本章讲过的所有部件都组装在这个文件里。它与 GEMM verify cell 有两个差异：Flash Attention 的入口更丰富（`get_flash_attention4_kernel`），并且为了内置 profiler 多接受一个 `profiler_buf` 参数。整章只需要运行下面这个 cell：

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA: 32 query heads share 8 KV heads
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)   # ex.mod takes torch tensors directly, like every other chapter
torch.cuda.synchronize()

# torch reference; enable_gqa lets the 32 query heads share the 8 KV heads
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**预期输出**：`... -> PASS`。kernel 会用 fp32 累积 online softmax，但它的结果与高精度 reference 之间仍然有几类近似差异。包括输入和 operand 的 fp16 存储与舍入；基于 `exp2` 的 softmax reformulation（用 `scale_log2 = log2(e)/√d` 重写每个指数）；online-softmax reordering 和 per-row rescaling，它以 running scale 累加 block，而不是一次性求和；最后还有 writeback 时把 `O` cast 为 fp16。这里选择的 `rtol`/`atol` 与源 kernel 自己测试使用的 tolerance 相同，大小足以覆盖这些因素与 torch reference 之间的综合差异，而不是只覆盖 fp16 rounding。因此如果你在这里看到真正的 failure，而不只是边界附近的 near-miss，应把它看作指向 softmax path 的信号：可能漏掉了 `s_ready` / `p_o_rescale` / `p_ready_2` wait，或者 `row_max` / `row_sum` update 没有被 rescale step 正确应用。这些正是本章花大量 barrier 讨论的 handoff。

## 与 GEMM 的差异

下表沿发生变化的维度比较 FA4 和 GEMM：

| 维度 | GEMM | Flash Attention 4 |
|--------|------|-------------------|
| MMA phase | 一个重复的 MMA | score MMA 和 value MMA |
| MMA 之间的工作 | 除 pipeline handoff 外没有 | online softmax、masking 和 O rescaling |
| Running state | 只有 accumulator | row max、row sum、O accumulator |
| 主要中间值 | accumulator TMEM tile | S、P 和 O TMEM tile region |
| Warp role | TMA producer、MMA consumer、writeback | TMA load、MMA、softmax、correction、TMA store |
| Barrier | 主要是 load/compute/writeback handoff | 额外的 score/softmax/value/correction handoff |
| Scheduling unit | output matrix tile | attention task：`(batch, kv_head, m_block)` |

这些差异都可以追溯到本章开头提出的结构变化：第二个 MMA，以及夹在两个 MMA 之间的 softmax。另一方面，底层 TIRx contract 完全没有变化：

- tile primitive 说明哪个 tile 移动或计算，
- surrounding scope 说明哪些线程协作，
- layout 说明 tile 住在哪里，
- barrier 说明下一个角色何时可以消费它。

因此 FA4 比 GEMM 更难，不是因为它依赖不同硬件，而是因为 tile value 更多，tile 之间的 handoff 也更多。

## 练习

1. 与 GEMM 相比，FA4 的两个 MMA phase 之间出现了什么新的 tile handoff？请命名 producer、TMEM tile 和 consumer。
2. 为什么 softmax 要把 numerator tile `P` 写回 TMEM，而不是只把它保留在 register 中供 value MMA 使用？
3. 选择 `p_o_rescale` 或 `p_ready_2`。这个 barrier 精确证明了什么？如果 value MMA 跳过这个 wait，会发生什么错误？

**让你的 agent 试试**：选择一个没有标注的 tile primitive，例如 epilogue `Tx.copy_async`、fp32 -> fp16 的 `Tx.cast`，或第二个 `gemm_pv` sub-MMA。让它给出 scope / layout / dispatch / handoff 卡片，然后根据源代码中的 guard、allocation 和 wait 检查答案。
