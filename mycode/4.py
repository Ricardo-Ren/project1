from te import tvm

A = tvm.placeholder((10, 10))
B = tvm.compute((10, 10), lambda i, j: A[i, j])
s = tvm.create_schedule(B.op)
f = tvm.build(s, [A, B], "hello")
print(f.get_source())