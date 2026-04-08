from __future__ import annotations

from pathlib import Path
import os
import warnings
import numpy as np
from ..rbp_conf import rbp_conf

# from .. utils.path_methods import create_relative_path

def curvature(
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
        os.makedirs(outdir)

    if poses is None:
        poses = rbp_conf(shape_params)

    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgposes = poses[first_cg::cg]
    kappa = _triads2bild(bildfn, cgposes, alpha=1., scale=1, nm2aa=True, decimals=2)

    return kappa

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

    nposes = len(poses)
    kappa = 0

    dtheta2 = 0.5*np.acos(np.dot(t_hat[-1],t_hat[nposes//8])) # half angle
    dr = np.linalg.norm(r[-1]-r[nposes//8]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[-1],t_hat[nposes//4])) # half angle
    dr = np.linalg.norm(r[-1]-r[nposes//4]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[-1],t_hat[nposes//2])) # half angle
    dr = np.linalg.norm(r[-1]-r[nposes//2]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[-1],t_hat[7*nposes//8])) # half angle
    dr = np.linalg.norm(r[-1]-r[7*nposes//8]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[0],t_hat[nposes//8])) # half angle
    dr = np.linalg.norm(r[0]-r[nposes//8]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[0],t_hat[nposes//4])) # half angle
    dr = np.linalg.norm(r[0]-r[nposes//4]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[0],t_hat[nposes//2])) # half angle
    dr = np.linalg.norm(r[0]-r[nposes//2]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[0],t_hat[7*nposes//8])) # half angle
    dr = np.linalg.norm(r[0]-r[7*nposes//8]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    dtheta2 = 0.5*np.acos(np.dot(t_hat[0],t_hat[-1])) # half angle
    dr = np.linalg.norm(r[0]-r[-1]) # distance between triad origins
    kappa += 2.0/dr*np.sin(dtheta2)  # curvature

    kappa /= 9

    return kappa
