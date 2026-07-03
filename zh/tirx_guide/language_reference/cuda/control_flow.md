# Control flow

Control flow 包括 `if`、loop family 和 `while`。它们会映射到直观的 CUDA 结构。

## if

Python `if` / `else` 会变成 CUDA `if` / `else`。可以用 thread/lane 比较来 guard 工作，也可以用 `T.ptx.elect_sync()` 选出一个 issuing thread：

```python
if tx < 128:
    A[tx] = A[tx] * T.float32(2.0)
else:
    A[tx] = A[tx] + T.float32(1.0)

if T.ptx.elect_sync():
    ...                              # one elected lane
```

expression-level 的选择可以用 `T.if_then_else(cond, a, b)`，它会降低成 ternary，不引入 control-flow divergence：

```c++
O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;
```

## Uniform vs. divergent control flow

`if tx < 128` 这样的 per-thread guard 对普通工作没问题，但 **collective** operation 必须由它同步的所有线程 **uniformly** 到达。

例如，`T.cuda.cta_sync()` 会映射到 `__syncthreads()`，要求 thread block 中所有线程到达。它绝不能放在 thread-divergent 或 warpgroup-divergent branch 内；如果放在 `if wg_id == 0:` 里面，其他 warpgroup 永远不会到达，kernel 会死锁。只有一个 warpgroup 需要同步时，应使用 warpgroup scope 的 `T.cuda.warpgroup_sync(id)`，见 {ref}`chap_gemm_advanced` 和 {doc}`threads_sync`。

同样的谨慎也适用于 barrier setup。`mbarrier` 的 `.init()` 会降成 single-thread guard（`if (threadIdx.x < 1)`）。如果把它嵌套到另一个 divergent branch 中，barrier 可能没有初始化，导致 unspecified launch failure。

## loop

loop 有四种形式；普通 Python `range` 会变成 `T.serial`：

- `T.serial(n)`：顺序 loop，ptxas 仍可能展开。
- `T.unroll(n)`：完全展开成 straight-line statement。
- `T.vectorized(n)`：vectorized loop。
- `T.grid(*extents)`：嵌套 loop nest。

loop 内可以使用 `break` / `continue`。

```python
for i, j in T.grid(8, 8):
    B[i, j] = T.max(A[i, j], T.float32(0.0))
```

会降低成普通嵌套 loop：

```c++
for (int i = 0; i < 8; ++i)
  for (int j = 0; j < 8; ++j)
    B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);
```

## while

`while` loop 会一直运行到条件为 false。可变 counter 请使用 mutable scalar（见 {doc}`buffers`）：

```python
i: T.int32 = 0
while i < 64:
    A[i] = A[i] + T.float32(1.0)
    i += 1
```

它会降低成带 early-exit `break` 的 `while (1)`，其中 counter 是一个 one-element register buffer。
