from __future__ import annotations

import numpy as np
import so3 
from pycondec import cond_jit

def build_conf(
        rbp_params: np.ndarray,
        orientation: np.ndarray | list | tuple = np.array([0.0, 0.0, 1.0]),
        origin: np.ndarray | list | tuple = np.zeros(3),
        ) -> np.ndarray: 

    if len(rbp_params.shape) != 2:
        raise ValueError(f"rbp_params must be a 2D array, got shape {rbp_params.shape}")
    if rbp_params.shape[1] != 6:
        raise ValueError(f"rbp_params must have 6 columns (3 for rotation, 3 for translation), got {rbp_params.shape[1]}")

    nbps = len(rbp_params)
    nbp = nbps + 1
        
    poses = np.zeros((nbp, 4, 4), dtype=float)
    poses[0] = np.eye(4)
    poses[0][:3, 3] = origin
    if not np.allclose(orientation, np.array([0.0, 0.0, 1.0])):
        R = so3.rotmat_align_vector(np.array([0.0, 0.0, 1.0]), np.asarray(orientation, dtype=float))
        poses[0][:3, :3] = R
    
    gs = so3.se3_euler2rotmat_batch(rbp_params)
    return _build_chain(poses[0], gs)

@cond_jit(nopython=True, cache=True)
def _build_chain(p0: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """JIT-compiled sequential SE(3) prefix product.
    
    Computes poses[i] = p0 @ gs[0] @ gs[1] @ ... @ gs[i-1] for all i.
    The Python loop over small 4×4 matrices is the dominant cost in
    build_conf_vec; compiling it eliminates per-iteration Python overhead
    and yields ~5× speedup.
    """
    n = len(gs)
    poses = np.empty((n + 1, 4, 4))
    poses[0] = p0
    for i in range(n):
        poses[i + 1] = poses[i] @ gs[i]
    return poses
