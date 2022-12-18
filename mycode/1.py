import tvm
from tvm.ir.module import IRModule
from tvm import te

# Step 1: Create two tvm.te.Tensor, A and B
A = te.placeholder((8,), dtype="float32", name="A")
B = te.compute((8,), lambda *i: A(*i) + 1.0, name="B")

# Step 2: Create TIR PrimFunc from A and B
func = te.create_prim_func([A, B])

# Step 3: Add created PrimFunc to a new TIR IRModule
ir_module_from_te = IRModule({"main": func})

print(ir_module_from_te)
print(ir_module_from_te.script())
import numpy as np

mod_from_te = tvm.build(ir_module_from_te, target="c")
# mod = tvm.build(ir_module, target="llvm")
# mod = tvm.build(ir_module, target="cuda")

a = tvm.nd.array(np.arange(8).astype("float32"))
print(a)
# [0. 1. 2. 3. 4. 5. 6. 7.]

b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod_from_te(a, b)
print(b)
# [1. 2. 3. 4. 5. 6. 7. 8.]