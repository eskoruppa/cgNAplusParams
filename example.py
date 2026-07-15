#!/usr/bin/env python3

import sys,glob,os

num_cores = 1
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cores}"
os.environ["MKL_NUM_THREADS"] = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cores}"


import numpy as np
from cgnaplusparams import cgnaplus_params
import time


if __name__ == "__main__":

    nbp = 250
    seq = "".join(np.random.choice(list("ACGT"), size=nbp))
    base_fn = 'Test/test'

    cg = cgnaplus_params(seq,include_stiffness=True)

    reps = 10
    t1 = time.time()
    for i in range(reps):
        seq = "".join(np.random.choice(list("ACGT"), size=nbp))
        cgnapp = cgnaplus_params(seq,include_stiffness=True)
        stiff = cgnapp.stiffmat
        gs = cgnapp.gs

    t2 = time.time()
    print(f"Time taken: {(t2 - t1) / reps:.5f} seconds per sequence ({t2 - t1:.5f} seconds total)")
