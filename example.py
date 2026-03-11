
import numpy as np
from cgnaplusparams import cgnaplus2rbp, build_conf
from cgnaplusparams import visualize_chimerax
import time


if __name__ == "__main__":

    nbp = 22
    seq = "".join(np.random.choice(list("ACGT"), size=nbp))
    base_fn = 'Test/test'

    result = cgnaplus2rbp(seq,include_stiffness=False)
    conf = build_conf(result["gs"])

    reps = 10000
    t1 = time.time()
    for i in range(reps):
        seq = "".join(np.random.choice(list("ACGT"), size=nbp))
        result = cgnaplus2rbp(seq,include_stiffness=False)
        conf = build_conf(result["gs"])
    t2 = time.time()
    print(f"Time taken: {(t2 - t1) / reps:.5f} seconds per sequence ({t2 - t1:.5f} seconds total)")

    visualize_chimerax(base_fn, seq, shape_params=result["gs"],cg=1)
