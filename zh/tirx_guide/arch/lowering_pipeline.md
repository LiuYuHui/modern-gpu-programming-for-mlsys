# TIRx lowering pipeline

`tvm.compile(mod, target, tir_pipeline="tirx")` 会把作者写的 TIRx module 送入 **tirx pipeline**。这个 pipeline 是一组有序的 TIR pass，会把高层构造（tile primitive、带 `TileLayout` 类型的 buffer、execution-scope id）降低成分离的 **host** function 和 **device** function，之后 CUDA backend 再把 device function 渲染成源码。pipeline 定义在 `python/tvm/tirx/compilation_pipeline.py` 的 `tirx_pipeline` 中。

## 位置

`tvm.compile` 先绑定 target，运行 **tirx pipeline**，然后分别对 host 和 device function 应用 finalization pass，最后把每个 device function 交给 CUDA code generator：

```text
authored TIRx  --BindTarget-->  tirx_pipeline  -->  host func  --host finalize-->  C/LLVM
                                      |
                                      +---------->  device func --device finalize--> CUDA
```

## Pass 顺序

`tirx_pipeline` module pass 会按下面顺序应用 pass；其中少数 pass 会受 `PassContext` 配置控制：

| # | Pass | 作用 |
|---|------|------|
| 1 | `LowerTIRx` | 核心 lowering，见下面的 “Inside LowerTIRx” |
| 2 | `UnifyThreadBinding` | 合并等价的 thread-axis binding，让每个 `threadIdx` / `blockIdx` 轴只声明一次 |
| 3 | `StmtSimplify` | statement-level 算术化简 |
| 4 | `LowerTIRxOpaque` | 把剩余 opaque TIRx 构造降成普通 TIR |
| 5 | `FlattenBuffer` | 把多维 `BufferLoad` / `BufferStore` 展平为 1-D |
| 6 | `BF16ComputeLegalize` | 把 `bfloat16` compute 改写成合法形式 |
| 7 | `NarrowDataType(32)` | 在可证明安全时把 index/loop `PrimExpr` dtype 收窄到 32-bit |
| 8 | `VectorizeLoop` | 把 `T.vectorized` loop 转成 vector op；`tir.disable_vectorize` 时跳过 |
| 9 | `UnrollLoop` | 展开标记为 `T.unroll` 的 loop 和小的常量 loop |
| 10 | `StmtSimplify` | vectorize/unroll 暴露常量后再次化简 |
| 11 | `CommonSubexprElim` | 把重复子表达式 hoist 到临时变量；`tir.disable_cse_tir` 时跳过 |
| 12 | `FP8ComputeLegalize` | 把 `float8` compute 改写成合法形式 |
| 13 | `VerifyMemory` | 检查 host-side code 不直接解引用 device memory |
| 14 | `AnnotateEntryFunc` | 把单个 PrimFunc 标记为 module entry point |
| 15 | `SplitHostDevice` | 在 `launch_thread` 边界把 kernel 拆成 host function 和 device function |
| 16 | `MakePackedAPI` | 把 host function 改写成 packed-func ABI |
| 17 | `FP8StorageLegalize` | 合法化 `float8` storage |
| 18 | `BF16StorageLegalize` | 合法化 `bfloat16` storage |

Finalization 会按 function 类型分别运行：

- **host**：`LowerTVMBuiltin`、`LowerIntrin`
- **device**：`LowerWarpMemory`、`StmtSimplify`、`LowerIntrin`

## Inside LowerTIRx

`LowerTIRx` 本身是一个小的 sequence，定义在 `src/tirx/transform/lower_tirx.cc`：

```text
LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])
```

- **`TilePrimitiveDispatch`** 会把每个 `TilePrimitiveCall`（如 `copy`、`gemm`、`reduction`）替换成选定 backend dispatch 生成的实现体。
- **`LowerTIRxCleanup`** 会运行 `LayoutApplier`：它把每个带 `TileLayout` 的 buffer access 解析成具体物理地址算术（`addr = data + elem_offset + layout.apply(coord)`），展平 buffer，并把 execution-scope id（`T.cta_id` / `T.thread_id` 等）通过 `launch_thread` 解析为 `blockIdx` / `threadIdx`。

因此，`LowerTIRx` 之后的 module 已经是普通 TIR：没有 tile primitive，没有 `TileLayout` 间接层，scope id 也已经解析为 thread axis。

## 一个例子

一个一行 scale kernel：

```python
@T.prim_func
def scale(A_ptr: T.handle, B_ptr: T.handle):
    A = T.match_buffer(A_ptr, (256,), "float32")
    B = T.match_buffer(B_ptr, (256,), "float32")
    T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
    B[tx] = A[tx] * T.float32(2.0)
```

经过 `LowerTIRx` 后，scope id 变成真实 thread axis，layout 已应用：

```python
with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
    threadIdx_x = T.launch_thread("threadIdx.x", 256)
    bx: T.let = blockIdx_x
    tx: T.let = threadIdx_x
    B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)
```

经过 `SplitHostDevice` + `MakePackedAPI` 后，一个 function 会变成两个：

```python
@I.ir_module
class Module:
    def main(...):          # host: packed-API launcher
        ...
    def scale_kernel(...):  # device: __global__ body
        ...
```

CUDA backend 最后把 `scale_kernel` 渲染成 `__global__` function，例如 `B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f`。

## 自行复现

可以手动运行 pipeline 的任意前缀来检查某个 stage：

```python
from tvm.tirx import transform as TT

target = tvm.target.Target("cuda")
mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
mod = TT.LowerTIRx()(mod)
print(mod.script())
```

也可以编译完整 module 并读取生成的 CUDA：

```python
exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
print(exe.mod.imports[0].inspect_source())
```
