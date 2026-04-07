#!/usr/bin/env python3

import sys,glob,os

num_cores = 1
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cores}"
os.environ["MKL_NUM_THREADS"] = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cores}"


import numpy as np
from cgnaplusparams import cgnaplus
import time


if __name__ == "__main__":

    nbp = 50
    seq = "".join(np.random.choice(list("ACGT"), size=nbp))
    base_fn = 'Test/test'



    cg = cgnaplus(seq,include_stiffness=True)
    x0 = cg.gs_param('X0')

    print(x0)


    # reps = 10
    # t1 = time.time()
    # for i in range(reps):
    #     seq = "".join(np.random.choice(list("ACGT"), size=nbp))
    #     result = cgnaplus2rbp(seq,include_stiffness=True)
    #     conf = rbp_conf(result["gs"])
    # t2 = time.time()
    # print(f"Time taken: {(t2 - t1) / reps:.5f} seconds per sequence ({t2 - t1:.5f} seconds total)")
    # # sys.exit()
    # print(f"Writing visualization to {base_fn}.cxc Open with: chimerax {base_fn}.cxc")
    # visualize_chimerax(base_fn, seq, shape_params=result["gs"],cg=1)
