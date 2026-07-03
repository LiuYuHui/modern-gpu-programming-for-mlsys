# CUDA C++/PTX intrinsic

当没有 tile primitive 覆盖你需要的功能时，有两个 escape hatch 可以直接触达硬件：调用 backend intrinsic（来自 `tvm.backend.cuda` 的 `T.cuda.*` / `T.ptx.*` namespace），或者 inline raw CUDA source。

## 调用 backend intrinsic

`T.cuda.*` 和 `T.ptx.*` 会直接暴露 CUDA backend 的 device intrinsic，包括 synchronization、mbarrier、reduction，以及 PTX data-movement / MMA family：

```python
T.cuda.cta_sync()                    # block barrier (__syncthreads)
T.cuda.warp_sync()                   # __syncwarp
T.cuda.warpgroup_sync(8)             # warpgroup barrier
T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))

bar = T.alloc_shared((1,), "uint64")
T.ptx.mbarrier.init(bar.data, 1)
T.ptx.mbarrier.try_wait(bar.data, phase)
```

一个完整可运行的例子是通过 `T.tvm_warp_shuffle_xor` 做 warp all-reduce：

```python
@T.prim_func
def warp_reduce(A_ptr: T.handle):
    A = T.match_buffer(A_ptr, (32,), "float32", align=16)
    T.device_entry()
    cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
    v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
    v[0] = T.float32(31 - lane_id)
    i[0] = 16
    while i[0] >= 1:
        v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
        i[0] = i[0] // 2
    A[lane_id] = v[0]
```

shuffle 会直接降低成 `__shfl_xor_sync`：

```c++
v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);
```

`T.ptx.*` / `T.cuda.*` 下还包含其他 family，例如 `cp_async`（LDGSTS）、`cp_async.bulk.tensor`（TMA）、`ldmatrix` / `stmatrix`、`tcgen05.*`（Blackwell MMA）、`atomic_add`、`fence` 等。完整列表见 backend API reference。

## Synchronization 语义

GEMM 和 Flash Attention kernel 中会反复出现四类同步机制。它们控制 asynchronous engine 和 parallel thread group，误用通常会导致 silent corruption 或 deadlock。

**Mbarrier phase。** Mbarrier 使用单个 internal phase bit 跟踪 arrival。`T.ptx.mbarrier.try_wait(bar, phase)` 会阻塞到 barrier 的 internal phase 与 caller 传入的 `phase` 参数不同。因此复用 barrier 跨 loop iteration 时，caller 必须在每次 wait 后翻转本地 phase tracker（`phase ^= 1`）。如果不这样做，后续 wait 会立即返回，允许 engine 读取半写入的 memory。完整 phase tracking 表见 {ref}`chap_gemm_basics`。

**Election。** `T.ptx.elect_sync()` 会在一个 warp 内选出*单个 active lane*，不是 lane 0，也不是每个 CTA 一个线程。要把 issuer 收窄到恰好一个线程，必须配合 warp-level guard。{ref}`chap_gemm_basics` 中常用 `if warp_id == 0:` 后接 `if T.ptx.elect_sync():` 来发起 `Tx.gemm_async` 和 `tcgen05.commit`。

**Named warpgroup barrier。** `T.cuda.cta_sync()` 映射到 `__syncthreads()`，要求*每个* CTA 线程到达。warpgroups 专门化到不同代码路径后，把 `cta_sync()` 放在 warpgroup branch 中会让 kernel 死锁，因为其他 warpgroup 永远不会到达。硬件提供 16 个 named barrier（ID 0 到 15）；`T.cuda.warpgroup_sync(10)` 只同步一个 warpgroup 的线程。不同 warpgroup 使用不同 ID（例如 `warpgroup_sync(wg_id + 10)`），避免撞到同一个硬件 barrier。见 {ref}`chap_gemm_advanced`。

**Fence。** fence 会把 producer 的写入排在 consumer（通常是 asynchronous engine）读取之前：

| Fence | 排序内容 |
|-------|----------|
| `T.ptx.fence.proxy_async("shared::cta")` | thread-written shared memory 在 async proxy（TMA store / MMA）读取之前 |
| `T.ptx.fence.mbarrier_init()` | mbarrier 初始化在后续 arrival 或 wait 使用该 barrier 之前 |
| `T.ptx.tcgen05.fence.after_thread_sync()` | `tcgen05` writeback edge 上的保守 ordering fence；Step 8/9 会加入它，TMA-to-MMA 路径不需要 |

## Inline raw CUDA

如果没有任何 intrinsic 可用，可以用 `T.cuda.func_call(name, *args, source_code=..., return_type=...)` 从 source string 注入一个 `__device__` function：

```python
SRC = r"""
__device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
"""

@T.prim_func
def k(A_ptr: T.handle, B_ptr: T.handle):
    A = T.match_buffer(A_ptr, (256,), "float32")
    B = T.match_buffer(B_ptr, (256,), "float32")
    T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
    B[tx] = T.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")
```

source 会原样输出，调用也会接到生成代码中：

```c++
__device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
// ...
B_ptr[tx] = my_relu(A_ptr[tx]);
```
