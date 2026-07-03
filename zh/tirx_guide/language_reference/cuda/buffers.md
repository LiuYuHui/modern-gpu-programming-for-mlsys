# Buffer 和内存

kernel 参数 buffer 用 `T.match_buffer` 绑定；body 中的 scratch buffer 用两类声明 API 创建。buffer 可以用 `A[i, j]` 索引，用 `A[m0:m0+BM, 0:BK]` 切片成 `BufferRegion`，也可以用 `A.ptr_to([i, j])` 取得元素 pointer，或用 `A.data` 取得 raw data pointer。

## 声明 buffer

两个基础 API 会创建 buffer：

- `T.alloc_buffer(shape, dtype, scope=..., ...)`：**分配新 storage**，产生 `AllocBuffer` node 并返回 `Buffer`。`T.alloc_shared` / `T.alloc_local` 只是 `scope="shared"` / `scope="local"` 的简写。
- `T.decl_buffer(shape, dtype, data=..., ...)`：在已有 pointer `data` 上**声明 view**，不分配 storage。它用于 alias 或 reinterpret storage，例如 pool 的子区域或 tensor memory 地址。`data=None` 时行为类似 `alloc_buffer`，会分配 storage。

buffer 的 `data` pointer 是 immutable `Var`：`alloc_buffer` 定义它，`decl_buffer` 接收它。若要用 pointer *expression* 支撑 buffer，必须先绑定成 pointer `Var`，见 {doc}`data_types`。

常用 descriptor 参数如下：

| 参数 | 含义 |
|------|------|
| `dtype` | 元素类型，例如 `"float32"`、`"float16"`、`"float4_e2m1fn"` |
| `shape` | 逻辑形状，一组 extent |
| `layout` | 物理映射，见 {ref}`chap_tirx_layout_api`；`"default"` 是 dense row-major |
| `elem_offset` / `allocated_addr` | `elem_offset` 或 `byte_offset` 把 view 放到 `data` 的偏移处；`allocated_addr` 携带预先分配的地址，例如 tensor memory column |
| `align` | data pointer 的字节对齐 |

`scope` 参数选择内存空间：

| Scope | 简写 | 内存 |
|-------|------|------|
| `"global"` | 默认 | device global memory |
| `"shared"` | `T.alloc_shared` | static shared memory（`__shared__`） |
| `"shared.dyn"` | pool | dynamic shared memory |
| `"local"` | `T.alloc_local` | per-thread register |
| `"tmem"` | TMEM pool | Blackwell tensor memory |

```python
A = T.match_buffer(A_ptr, (M, K), "float16", align=16)
As = T.alloc_shared((BM, BK), "float16")
acc = T.alloc_local((4,), "float32")
view = T.decl_buffer((BM, BK), "float16", data=As.data)
```

**ptr-based buffer 只是 pointer 上的 metadata。** 对非 tmem buffer，声明就是 pointer 加 layout；索引会解析成地址：

```text
addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]
```

因此相同的逻辑访问会根据 buffer metadata 生成不同地址算术：

```python
from tvm.tirx.layout import TileLayout, S

B = T.match_buffer(p, (4, 8), "float32")
B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))
B = T.match_buffer(p, (4, 8), "float32", elem_offset=64)
B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)]))
```

生成 CUDA 中 `B[i, j]` 分别类似：

```c++
B_ptr[((i * 8) + j)]        = ...;   // row-major
B_ptr[((j * 4) + i)]        = ...;   // column-major
B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64
B_ptr[((i * 16) + j)]       = ...;   // row stride 16
```

## Shared memory

Shared memory 有两种形式：**static**（compile time 固定大小）和 **dynamic**（launch 时设置大小）。此外还有一个 pool helper 用于管理 dynamic shared memory。

### Static

最简单的是 **static** shared buffer：`T.alloc_shared`，也就是 `scope="shared"`，大小在 compile time 固定。典型模式是先把数据 staged 进去，`cta_sync` 让整个 block 看见写入，然后再读出。

```python
@T.prim_func
def smem_demo(A_ptr: T.handle, B_ptr: T.handle):
    A = T.match_buffer(A_ptr, (128,), "float32")
    B = T.match_buffer(B_ptr, (128,), "float32")
    T.device_entry()
    bx = T.cta_id([1])
    tx = T.thread_id([128])
    sm = T.alloc_shared((128,), "float32")
    sm[tx] = A[tx]
    T.cuda.cta_sync()
    B[tx] = sm[tx] * T.float32(2.0)
```

它会降低成普通 `__shared__` array：

```c++
__shared__ alignas(64) float sm_ptr[128];
sm_ptr[tx] = A_ptr[tx];
__syncthreads();
B_ptr[tx] = sm_ptr[tx] * 2.0f;
```

### Dynamic

**Dynamic** shared memory（`scope="shared.dyn"`）的大小由 launch parameter `sharedMemBytes` 决定，而不是 compile time 固定。一个 kernel 只能有 **一个** dynamic-shared allocation，也就是 arena。因此通常先分配一次 arena，再用 `T.decl_buffer` 搭配 `data=arena.data` 和 `elem_offset` 在其中声明多个 view。

```python
arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")
As = T.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")
Bs = T.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn")
As[tx] = A[tx]
Bs[tx] = B[tx]
T.cuda.cta_sync()
C[tx] = As[tx] + Bs[tx]
```

这两个 view 共享同一个 `extern __shared__` arena：

```c++
extern __shared__ __align__(64) float smem[];
smem[tx]      = A_ptr[tx];
smem[tx + 64] = B_ptr[tx];
__syncthreads();
C_ptr[tx] = smem[tx] + smem[tx + 64];
```

两个独立的 `alloc_buffer(scope="shared.dyn")` 调用是错误的。static shared memory 在 compile time 定大小；dynamic shared memory 是一个 launch-sized arena，并在其中声明 offset view。

TIRx 会自动计算 dynamic-shared size。lowering 会给 device kernel 的 `tirx.kernel_launch_params` 添加 `"tirx.use_dyn_shared_memory"` tag，host launcher 计算总字节数，并把它作为最后一个 launch argument 传入。运行时这个值成为 `cuLaunchKernelEx` 中的 `config.sharedMemBytes`，通常不需要手动设置。

### Pool sugar

`T.SMEMPool` 会自动管理 dynamic shared arena。它用 bump allocator 分配 offset，因此不必手动写 `decl_buffer` view。除了 `alloc` / `commit`，它还支持每个 buffer 的 `align=`，提供 `alloc_mma` helper 来构造 MMA-compatible swizzle layout，并用 `move_base_to` 回退 cursor 以复用空间。

```python
pool = T.SMEMPool()
As = pool.alloc((BM, BK), "float16", align=128)
Bs = pool.alloc((BK, BN), "float16", align=128)
Cs = pool.alloc_mma((BM, BN), "float16")
pool.commit()
# pool.move_base_to(offset) rewinds the cursor to reuse space
```

TMEM pool 会构建在 `SMEMPool` 之上。

## Register

Per-thread scratch 存在 register 中。用 `T.alloc_local(shape, dtype)` 分配，也就是 `scope="local"`。它对每个线程私有，并降低成保存在 register 中的 local array。

```python
r = T.alloc_local((4,), "float32")
for k in T.unroll(4):
    r[k] = A[tx, k]
```

生成 CUDA 可能出现 `alignas(64) float r_ptr[4];`。这里的 `alignas(64)` 是默认 buffer alignment；对 register-resident array 没有性能影响。静态可解析索引的 thread-local array 会被 nvcc/ptxas promoted 到 register，不会真的落到 addressable local memory。这个 over-alignment 是已知粗糙点，后续会用 dtype 的 natural alignment 改进 `local` scope。

## Scalar

scalar 本质上是只有 **一个元素** 的 register array。可以手动分配 size-1 `local` buffer 并用 `[0]` 访问：

```python
phase = T.alloc_local((1,), "int32")
phase[0] = 0
while phase[0] < 4:
    acc = acc + A[tx, phase[0]]
    phase[0] += 1
```

为了避免到处写 `phase[0]`，TIRx 提供 scalar sugar：一个 one-element register buffer，但可以按名字读写。

```python
phase: T.int32 = 0
while phase < 4:
    acc = acc + A[tx, phase]
    phase += 1

s = T.local_scalar("int32")
acc: T.float32 = 0.0
```

两种写法会 parse 成结构相同的 TIRx。`phase: T.int32` 就是 one-element `local` buffer，`phase` / `phase += 1` 就是 `phase[0]` / `phase[0] += 1` 的 sugar。`T.local_scalar` / `T.shared_scalar` / `T.alloc_scalar` 可以显式选择 scope。

注意，TIRx `Var` 是 immutable 的单次绑定（也就是 `T.let` 的结果），不能作为循环或 accumulator 中反复赋值的 mutable scalar。mutable scalar 必须由 one-element buffer 支撑。

## `let`

`T.let` binding 是 **immutable**，表示单个 `LetStmt`，是命名 value，不是 buffer。适合派生常量：

```python
n: T.let = M * K
half: T.let[T.int32] = N // 2
```

它会降低成普通 scalar C variable，不是 array，也没有 `[0]`：

```c++
int half = m * 2;
```

因为值不可变，simplifier 可以传播和 CSE 它。使用点经常会直接看到 `m * 2` 被替换进去，或被共享成 common-subexpression temporary。

immutable binding 的价值在于 arithmetic analyzer 可以把 var 绑定到该值上，关于该值的事实（constant bounds、modular set、range）会传播到所有 use。这会帮助 index simplification、bounds-check elimination 和 alignment/vectorization decision。mutable scalar 是 memory load（`buf[0]`），analyzer 不能假定它保持常量。

## Tensor memory

Blackwell *tensor memory* 不是普通 scratch scope。它必须用 warp-uniform 的 `T.ptx.tcgen05.alloc` / `tcgen05.dealloc` intrinsic 显式 reserve 和 free；每个 tensor 都是其中一个 view，通过 `T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)` 声明。

`allocated_addr` 是必需的 column offset，tensor-core dispatch 会断言它存在。因此 `T.alloc_buffer(scope="tmem")` 不会工作，因为它不会设置 `allocated_addr`。与 shared memory 不同，tensor memory 不能直接寻址，只能通过 `tcgen05` 的 `mma` / `ld` / `st` / `cp` 读写。

手写时，通常由一个 warp 发起 allocation 到 shared slot，再把每个 tensor 声明为某个 column offset 上的 view，最后由一个 warp free：

```python
addr = T.alloc_shared((1,), "uint32")
if warp_id == alloc_warp:
    T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                    allocated_addr=0, layout=tmem_layout)
# ... use acc as a gemm_async / copy_async operand ...
if warp_id == alloc_warp:
    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
    T.ptx.tcgen05.dealloc(addr, n_cols=512, cta_group=cta_group)
```

你需要自己管理 column offset 和 `tmem_layout`。下面的 pool 会自动发出这套 sequence。

### Pool

`T.TMEMPool` 会包装 warp-uniform alloc/dealloc、column bump-allocation 和 datapath layout：

```python
tmem_addr = pool.alloc((1,), "uint32")
tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                       tmem_addr=tmem_addr)
acc = tmem_pool.alloc((CTA_M, 512), "float32")
tmem_pool.commit()
# ... use acc ...
tmem_pool.dealloc()
```

完整例子见第三部分 GEMM kernel。

## Buffer API

`Buffer` 是 pointer 上的 metadata，因此大多数方法都是 compile-time reshape/reinterpret：它们改变 index arithmetic，或交出 pointer，自身不会产生 runtime op。常用方法：

| Method | 含义 |
|--------|------|
| `B.data` | raw data pointer，一个 `Var`，打印为 `B_ptr` |
| `B.ptr_to([i, j])` | 指向元素的 typed pointer，也就是 `address_of` |
| `B.vload([i], dtype="float32x4")` / `B.vstore([i], v)` | vectorized load / store，打印为 `*(float4*)(B_ptr + ...)` |
| `B.view(*shape, layout=...)` | 用新的 shape/layout reinterpret 同一块 storage，不 copy |
| `B.local(*shape, layout=...)` | 取 `local` buffer 中当前线程私有的 register slice |
| `B.permute(*dims)` | 轴置换 view，类似 transposed layout |
| `B.access_ptr(mask, ...)` | masked access pointer，也就是 `tvm_access_ptr` builtin，常用于把 region 传给 intrinsic |

### Pointer：`ptr_to` / `data`

`ptr_to` 用于把元素地址交给 intrinsic 或 inline function；`data` 是 base pointer。

```python
B[tx] = T.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")
```

```c++
B_ptr[tx] = ld(&A_ptr[tx]);
```

### Vectorized access：`vload` / `vstore`

一次移动多个元素，生成 wide transfer：

```python
B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))
```

```c++
*(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);
```

### Reshape / reinterpret：`view` / `permute`

`view` 和 `permute` 都是纯 metadata 操作。data pointer 不变，只改变 index arithmetic。`A.view(64, 4)` 把 256-element buffer 看成 `64x4`；`A.permute(1, 0)` 交换轴：

```python
A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]
At = A.permute(1, 0);   z = At[i, j]
```

### Register：`local`

`local` 会把 thread-axis `local` layout 分解成当前线程的 flat register bundle，tile primitive 中会频繁使用它。

```python
R  = T.alloc_buffer((32, 8), "float32", scope="local",
                    layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
Rl = R.local(8)
```

它对应当前 lane 的私有 register，例如生成 `alignas(64) float Rl_ptr[8];`。
