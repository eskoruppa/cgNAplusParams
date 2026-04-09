from __future__ import annotations

from pathlib import Path
import os
import warnings
import numpy as np
from ..rbp_conf import rbp_conf

# from .. utils.path_methods import create_relative_path

def distance(
    base_fn: str | Path,
    seq: str,
    cg: int,
    *,
    shape_params: np.ndarray | None = None,
    poses: np.ndarray | None = None,
    first_cg: int = 0,
    bead_radius: float | None = None,
    disc_len: float = 0.34,
    include_bps_triads: bool = False
) -> None:

    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")

    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir, exist_ok=True)

    if poses is None:
        poses = rbp_conf(shape_params)

    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgposes = poses[first_cg::cg]
    d = _triads2bild(bildfn, cgposes, alpha=1., scale=1, nm2aa=True, decimals=2)

    return d

def _triads2bild(
    fn: Path | str,
    poses: np.ndarray,
    alpha: float = 1.,
    ucolor: str = 'default',
    vcolor: str = 'default',
    tcolor: str = 'default',
    scale: float = 1,
    nm2aa: bool = True,
    decimals: int = 2
): 

    # extract triads    
    t_hat = np.empty((0,3), float)
    r = np.empty((0,3), float)
    for i,tau in enumerate(poses):
        tau = tau[:3]
        t_hat = np.append(t_hat, np.array([tau[:,2]]), axis=0)
        r = np.append(r, np.array([tau[:,3]]), axis=0)

    # calculate distance based on first and last triad
    d = np.linalg.norm(r[0] - r[-1])

    return d
