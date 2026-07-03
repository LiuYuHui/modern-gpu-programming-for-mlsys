# 数据类型和表达式

每个 TIRx expression 同时携带低层 **dtype** 和高层 **type**。

## Expression dtype

`PrimExpr` 的 `.dtype` 是它的 scalar（或 vector）元素类型，例如 `float32`、`float16`、`bfloat16`、`int32`、`uint8`、`bool`，低精度的 `float8_e4m3fn` / `float4_e2m1fn`，`handle`（pointer），以及 `float32x4` 这样的 vector 形式。每种 dtype 都会打印成对应 CUDA 类型。

```python
@T.prim_func
def dtypes(A_ptr: T.handle, O_ptr: T.handle):
    A = T.match_buffer(A_ptr, (256,), "float32")
    O = T.match_buffer(O_ptr, (256,), "float32")
    T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([64])
    f16  = T.alloc_local((1,), "float16")
    bf16 = T.alloc_local((1,), "bfloat16")
    i32  = T.alloc_local((1,), "int32")
    u8   = T.alloc_local((1,), "uint8")
    b1   = T.alloc_local((1,), "bool")
    sm   = T.alloc_shared((64,), "float16")
    v    = T.alloc_local((1,), "float32x4")
    v[0] = A.vload([tx * 4], dtype="float32x4")
    O.vstore([tx * 4], v[0])
```

生成的 CUDA 类型大致对应为：

| dtype | CUDA |
|-------|------|
| `float32` | `float` |
| `float16` | `half` |
| `bfloat16` | `nv_bfloat16` |
| `int32` | `int` |
| `uint8` | `uchar` |
| `bool` | `signed char` |
| `float32x4` | `float4` |
| `handle` | `T*` pointer |

buffer 的 dtype 本身也可以是 **vector type**。例如 `T.alloc_local((1,), "float32x4")` 会直接声明一个 `float4` register，之后用 `v[0]` 索引；`float32x4` 的 `vload` / `vstore` 会把它作为一次 16-byte 访问移动。vector dtype 不只属于 `vload`，任何 buffer 或 scalar 都可以携带它。

## dtype vs type

`dtype` 是低层信息，说明“这些 bit 是什么”。此外，一个值还有高层 **type**：scalar 使用 `PrimType(dtype)`，pointer 使用 `PointerType(PrimType(dtype), scope)`。大多数 expression 是 scalar（`PrimType`）；type system 主要在 **pointer** 上重要。

## Pointer（`handle`）

buffer 的 `data` 是 pointer，也就是 pointer type 的 `Var`，并且它是 **immutable**，pointer 不能被重新赋值。这会影响获得 pointer 的方式：

- `T.alloc_buffer(...)` 会分配 storage，并定义它的 `data` pointer。
- `T.decl_buffer(..., data=ptr)` 会在已有 pointer `Var` `ptr` 上声明 buffer。
- 如果要用 pointer **expression** 支撑 buffer，例如 `T.ptx.map_shared_rank` 产生另一个 cluster CTA 的 shared address，必须先把 expression 绑定为 pointer `Var`。`data` 必须是 `Var`，不能是 expression，可以用 `PointerType` 的 `T.let`：

```python
from tvm.ir.type import PointerType, PrimType

ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = \
    T.reinterpret("handle", T.ptx.map_shared_rank(mbar.ptr_to([0]), 0))
remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")
```
