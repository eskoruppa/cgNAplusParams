from __future__ import annotations

import numpy as np

from ._so3 import so3
from ._pycondec import cond_jit
from .output.write_pdb import gen_pdb
from .output.visualize_rbp import visualize_chimerax
from .rbp_params import RBPParams
from .utils.se3_methods import build_chain


class RBPConf:

    def __init__(
        self,
        rbp: RBPParams,
        dynamic: np.ndarray | None = None,
        orientation: np.ndarray | list | tuple = np.array([0.0, 0.0, 1.0]),
        origin: np.ndarray | list | tuple = np.zeros(3),
    ) -> None:
        self.rbp = rbp
        self.orientation = orientation
        self.origin = origin

        if not rbp.euler_definition:
            raise ValueError("RBPConf currently only supports RBP objects with euler vector definition.")

        self._poses = rbp_conf(
            rbp.gs,
            dynamic=dynamic,
            group_split=rbp.group_split,
            orientation=orientation, 
            origin=origin)

    @property
    def poses(self) -> np.ndarray:
        return self._poses
    
    def to_pdb(self, filename: str) -> None:
        """Write the RBP configuration to a PDB file."""
        gen_pdb(outfn=filename, poses=self.poses, sequence=self.rbp.sequence)

    def to_chimerax(self, base_fn: str) -> None:
        """Write the RBP configuration to a Chimerax file."""
        from .output.visualize_rbp import visualize_chimerax
        visualize_chimerax(base_fn, self.rbp.sequence, cg=1, poses = self.poses)


def rbp_conf(
        rbp_params: np.ndarray,
        dynamic: np.ndarray | None = None,
        group_split: bool = True,
        orientation: np.ndarray | list | tuple = np.array([0.0, 0.0, 1.0]),
        origin: np.ndarray | list | tuple = np.zeros(3),
        ) -> np.ndarray: 

    if len(rbp_params.shape) != 2:
        raise ValueError(f"rbp_params must be a 2D array, got shape {rbp_params.shape}")
    if rbp_params.shape[1] != 6:
        raise ValueError(f"rbp_params must have 6 columns (3 for rotation, 3 for translation), got {rbp_params.shape[1]}")
  
    first_pose = _build_first_pose(orientation=orientation, origin=origin)

    if dynamic is not None:
        if dynamic.shape != rbp_params.shape:
            raise ValueError(f"dynamic shape {dynamic.shape} does not match rbp_params shape {rbp_params.shape}.")
        
        if group_split:
            ss = so3.se3_euler2rotmat_batch(rbp_params)
            ds = so3.se3_euler2rotmat_batch(dynamic)
            gs = ss @ ds            
        else:
            params = rbp_params + dynamic
            gs = so3.se3_euler2rotmat_batch(params)
    else:
        gs = so3.se3_euler2rotmat_batch(rbp_params)

    return build_chain(first_pose, gs)


def _build_first_pose(
        orientation: np.ndarray | list | tuple = np.array([0.0, 0.0, 1.0]),
        origin: np.ndarray | list | tuple = np.zeros(3),
        ) -> np.ndarray:
    """Build the first pose in the chain based on the specified orientation and origin."""
    pose = np.eye(4)
    pose[:3, 3] = origin
    if not np.allclose(orientation, np.array([0.0, 0.0, 1.0])):
        R = so3.rotmat_align_vector(np.array([0.0, 0.0, 1.0]), np.asarray(orientation, dtype=float))
        pose[:3, :3] = R
    return pose

