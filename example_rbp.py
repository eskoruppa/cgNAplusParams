#!/usr/bin/env python3

import sys,glob,os

num_cores = 1
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cores}"
os.environ["MKL_NUM_THREADS"] = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cores}"


import numpy as np
from cgnaplusparams import cgnaplus2rbp, rbp_conf
from cgnaplusparams import visualize_chimerax
from cgnaplusparams import RBP
from cgnaplusparams import RBPConf
import time


if __name__ == "__main__":

    nbp = 200
    seq = "".join(np.random.choice(list("ACGT"), size=nbp))
    base_fn = 'Test/test'

    result = cgnaplus2rbp(seq,include_stiffness=True)
    rbp = RBP(seq, include_stiffness=True)

    if not np.allclose(result["gs"], rbp.gs):
        raise ValueError("gs arrays do not match between cgnaplus2rbp and RBP versions.")
    print(f'gs arrays match between cgnaplus2rbp and RBP versions.')

    dynamic = np.ones(result["gs"].shape) * 0.1

    conf = rbp_conf(result["gs"], dynamic=dynamic)
    rbpconf = RBPConf(rbp, dynamic=dynamic)

    if not np.allclose(conf, rbpconf.poses):
        raise ValueError("RBPConf poses do not match rbp_conf output.")
    print("RBPConf poses match rbp_conf output.")

    rbpconf.to_pdb(base_fn)
    rbpconf.to_chimerax(base_fn)

    reps = 10
    t1 = time.time()
    for i in range(reps):
        seq = "".join(np.random.choice(list("ACGT"), size=nbp))
        rbp = RBP(seq, include_stiffness=True)
        rbpconf = RBPConf(rbp, dynamic=dynamic)
        conf = rbpconf.poses
    t2 = time.time()
    print(f"Time taken: {(t2 - t1) / reps:.5f} seconds per sequence ({t2 - t1:.5f} seconds total)")
    # sys.exit()
    print(f"Writing visualization to {base_fn}.cxc Open with: chimerax {base_fn}.cxc")
    # visualize_chimerax(base_fn, seq, shape_params=result["gs"],cg=1)
