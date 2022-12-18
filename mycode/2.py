# import tvm
# import tvm.testing
# from tvm import te
# import numpy as np

# # 全局环境定义

# tgt_host = "c"
# # 如果启用了GPU，则将其更改为相应的GPU，例如：cuda、opencl、rocm
# tgt = "c"
# n = te.var("n")
# A = te.placeholder((n,), name="A")
# B = te.placeholder((n,), name="B")
# C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
# print(type(C))

# s = te.create_schedule(C.op)
# bx, tx = s[C].split(C.op.axis[0], factor=64)
# if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
#     s[C].bind(bx, te.thread_axis("blockIdx.x"))
#     s[C].bind(tx, te.thread_axis("threadIdx.x"))

# fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")


# if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
#     dev_module = fadd.imported_modules[0]
#     print("-----GPU code-----")
#     print(dev_module.get_source())
# else:
#     print(fadd.get_source())
    
# ctx = tvm.context(tgt, 0)
# n = 1024
# a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
# b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
# c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
# fadd(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

# if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
#     dev_module = fadd.imported_modules[0]
#     print("-----GPU code-----")
#     print(dev_module.get_source())
# else:
#     print(fadd.get_source())

import tvm
from tvm import te

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n,n), name='B')
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=32)
yo, yi = s[C].split(s[C].op.axis[1], factor=32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].reorder(xo, yo, yi, xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))

import tvm
from tvm.ir.module import IRModule
from tvm import te

# Step 1: Create two tvm.te.Tensor, A and B
n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n,n), name='B')
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

# Step 2: Create TIR PrimFunc from A and B
func = te.create_prim_func([A, B, C])

# Step 3: Add created PrimFunc to a new TIR IRModule
ir_module_from_te = IRModule({"main": func})

print(ir_module_from_te.script())