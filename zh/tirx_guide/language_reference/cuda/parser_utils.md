# Parser utility

少量 helper 会在 **parse time** 生效，也就是 TVMScript 被转换成 TIRx 的阶段。它们用于 inline Python 计算值、抽出可复用片段，以及打包 parser-side state。

## `T.meta_var`：inline Python 值

`T.meta_var(x)` 告诉 parser：`x` 是在 **Python** 中计算出的 compile-time *meta* 值，应直接 inline 到 IR 中，而不是按 script 变量解析。它可以避免一次性 local，也可以驱动 metaprogramming：围绕 meta value 的普通 Python `for` 会在 parser 中展开。

```python
n = T.meta_var(4)              # n is a Python int, inlined
for j in range(n):             # unrolled at parse time
    acc[0] = acc[0] + A[tx, j]
```

## `@T.inline`：inline function

`@T.inline` 定义的 function 会在 parsing 期间被 **inline 到每个 call site**，生成代码中不会出现 function call。它遵循 Python 的 lexical scope（LEGB）和 late binding，因此参数会遮蔽外层变量。

```python
@T.inline
def add_into(acc, x):
    acc[0] = acc[0] + x

add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]
```

## `@T.meta_class`：parser-side state object

`@T.meta_class` 标记一个普通 Python class，其 **instance 是 parser meta value**。它的字段可以保存 buffer 和 scalar，因此可以把相关 allocation 与状态打包成一个对象，并在 kernel body 中使用。

```python
@T.meta_class
class State:
    def __init__(self, smem):
        self.acc = T.alloc_local([1], "float32")
        self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

s = State(smem.data)
s.acc[0] = T.float32(0.0)
# ... s.buf[i] ...
```

这适合把一个 kernel 的 pipeline state（barrier、accumulator、scratch view）组合起来，而不是在 body 中传递许多分散的 local。

## `T.constexpr`

`T.constexpr` 标记 compile-time kernel 参数，由 `@T.jit` 的 `.specialize(...)` 烘焙进去。细节见 {ref}`chap_tirx_primer`。
