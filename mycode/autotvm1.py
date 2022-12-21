import logging
import sys

import numpy as np
import tvm
from tvm import te
import tvm.testing

# the module is called `autotvm`
from tvm import autotvm


@autotvm.template("tutorial/matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    # this place directly calculates the MACs 
    
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_k", k, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    ko, ki = cfg["tile_x"].apply(s, C, k)
    s[C].reorder(yo, xo, ko, ki, yi, xi)

    # calculate one what one (xi, yi, computation)'s macs is and map it to the PE 
    # st,  macs < the predefined MACs for one PE
    #print(tvm.lower(s, [A, B, C], simple_mode=True))

    return s, [A, B, C]

N, L, M = 512, 512, 512
task = autotvm.task.create("tutorial/matmul", args=(N, L, M, "float32"), target="llvm")
print(task.config_space)

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

################################################################################
# There are two steps for measuring a config: build and run. By default, we use
# all CPU cores to compile program. We then measure them sequentially. To help
# reduce variance, we take 5 measurements and average them.
measure_option = autotvm.measure_option(builder=autotvm.MYBuilder(), runner=autotvm.MYRunner(key="simulator"))
#measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune_new(
    n_trial=10,
    measure_option=measure_option,
)

################################################################################
# With tuning completed, we can choose the configuration from the log file that
# has the best measured performance and compile the schedule with the
# corresponding parameters. We also do a quick verification that the schedule is
# producing correct answers.  We can call the function :code:`matmul` directly
# under the :any:`autotvm.apply_history_best` context. When we call this
# function, it will query the dispatch context with its argument and get the
# best config with the same argument.

# apply history best from log file
with autotvm.apply_history_best("matmul.log"):
    with tvm.target.Target("llvm"):
        s, arg_bufs = matmul(N, L, M, "float32")
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)