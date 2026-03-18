from __future__ import annotations

import sys
import numpy as np
from ._so3 import so3
from ._pycondec import cond_jit

from .rbp_conf import _build_first_pose, _build_chain
from .utils.assignment_utils import inter_bp_dof_indices, intra_bp_dof_indices, watson_phosphate_dof_indices, crick_phosphate_dof_indices
from .utils.assignment_utils import INTER_BP_PARAM_NAME, INTRA_BP_PARAM_NAME, B2P_WATSON_PARAM_NAME, B2P_CRICK_PARAM_NAME
from .utils.assignment_utils import dof_index


class cgNAplusConf:

    def __init__(
        self,
        cgnap: dict[str, np.ndarray | bool | str],
        orientation: np.ndarray | list | tuple = np.array([0.0, 0.0, 1.0]),
        origin: np.ndarray | list | tuple = np.zeros(3),
    ) -> None:
        self.cgnap = cgnap
        self.orientation = orientation
        self.origin = origin
        self.conf = cgnaplus_conf(cgnap, orientation=orientation, origin=origin)

        self.poses = self.conf["poses"]
        self.bp_poses = self.conf["bp_poses"]
        self.watson_base_poses = self.conf["watson_base_poses"]
        self.crick_base_poses = self.conf["crick_base_poses"]
        self.watson_phosphate_poses = self.conf["watson_phosphate_poses"]
        self.crick_phosphate_poses = self.conf["crick_phosphate_poses"]
        self._set_named_poses()

    def _set_named_poses(self) -> None:
        """Set attributes like self.inter_bp_poses, self.intra_bp_poses, etc. based on the param_names."""
        self.named_poses = {}
        for i, pose in enumerate(self.bp_poses):
            self.named_poses[f"{INTER_BP_PARAM_NAME}{i}"] = pose
        for i, pose in enumerate(self.watson_base_poses):
            self.named_poses[f"{INTRA_BP_PARAM_NAME}{i}"] = pose
        for i, pose in enumerate(self.crick_base_poses):
            self.named_poses[f"{INTRA_BP_PARAM_NAME}{i}"] = pose
        for i, pose in enumerate(self.watson_phosphate_poses):
            if np.any(pose):  # Check if the pose is not all zeros (i.e., it is contained)
                self.named_poses[f"{B2P_WATSON_PARAM_NAME}{i}"] = pose
        for i, pose in enumerate(self.crick_phosphate_poses):
            if np.any(pose):  # Check if the pose is not all zeros (i.e., it is contained)
                self.named_poses[f"{B2P_CRICK_PARAM_NAME}{i}"] = pose






def cgnaplus_conf(
        cgnap: dict[str, np.ndarray | bool | str],
        orientation: np.ndarray | list | tuple = np.array([0.0, 0.0, 1.0]),
        origin: np.ndarray | list | tuple = np.zeros(3),
        ) -> dict[str, np.ndarray]: 
    
    params = cgnap['gs']
    param_names = cgnap['param_names']
    aligned_strands = cgnap['aligned_strands']

    inter_bp_dof_ids = inter_bp_dof_indices(param_names=param_names)
    intra_bp_dof_ids = intra_bp_dof_indices(param_names=param_names)
    watson_phosphate_dof_ids = watson_phosphate_dof_indices(param_names=param_names)
    crick_phosphate_dof_ids = crick_phosphate_dof_indices(param_names=param_names)

    print(f"number of inter_bp_dof_ids: {len(inter_bp_dof_ids)}")
    print(f"number of intra_bp_dof_ids: {len(intra_bp_dof_ids)}")
    print(f"number of watson_phosphate_dof_ids: {len(watson_phosphate_dof_ids)}")
    print(f"number of crick_phosphate_dof_ids: {len(crick_phosphate_dof_ids)}")

    nbp = len(inter_bp_dof_ids) + 1
    if len(params) != len(param_names):
        raise ValueError(f"Length of params ({len(params)}) does not match length of param_names ({len(param_names)}).")
    if len(inter_bp_dof_ids) != nbp - 1:
        raise ValueError(f"Number of inter-base-pair DOFs ({len(inter_bp_dof_ids)}) does not match expected number ({nbp - 1}).")
    if len(intra_bp_dof_ids) != nbp:
        raise ValueError(f"Number of intra-base-pair DOFs ({len(intra_bp_dof_ids)}) does not match expected number ({nbp}).")
    
    # Convert params to SE(3) group elements (4x4 transformation matrices)
    gs = so3.se3_euler2rotmat_batch(params)

    # generate bp poses
    bp_poses = _build_chain(_build_first_pose(orientation=orientation, origin=origin), gs[inter_bp_dof_ids])

    # generate base poses
    watson_base_poses = np.empty(bp_poses.shape)
    crick_base_poses = np.empty(bp_poses.shape)
    for i in range(len(bp_poses)):
        watson_base_poses[i] = bp_poses[i] @ so3.X2grh(params[intra_bp_dof_ids[i]])
        crick_base_poses[i] = bp_poses[i] @ so3.X2glh_inv(params[intra_bp_dof_ids[i]])

    # generate phosphate poses
    watson_phosphate_poses = np.zeros(bp_poses.shape)
    crick_phosphate_poses = np.zeros(bp_poses.shape)

    watson_phosphate_contained = np.ones(len(bp_poses), dtype=bool)
    crick_phosphate_contained = np.ones(len(bp_poses), dtype=bool)


    for i in range(len(bp_poses)):
        wid = dof_index(f"{B2P_WATSON_PARAM_NAME}{i}", param_names)
        cid = dof_index(f"{B2P_CRICK_PARAM_NAME}{i}", param_names)
        if wid is not None:
            watson_phosphate_poses[i] = watson_base_poses[i] @ so3.X2g(params[wid])
        else:
            watson_phosphate_contained[i] = False
        if cid is not None:
            crick_phosphate_poses[i] = crick_base_poses[i] @ so3.X2g(params[cid])
            crick_phosphate_contained[i] = True
    
    poses = np.concatenate([
        bp_poses, 
        watson_base_poses, 
        crick_base_poses, 
        [watson_phosphate_poses[i] for i in range(len(watson_phosphate_poses)) if watson_phosphate_contained[i]], 
        [crick_phosphate_poses[i] for i in range(len(crick_phosphate_poses)) if crick_phosphate_contained[i]]],axis=0)

    result = {
        "poses" : poses,
        "bp_poses": bp_poses,
        "watson_base_poses": watson_base_poses,
        "crick_base_poses": crick_base_poses,
        "watson_phosphate_poses": watson_phosphate_poses,
        "crick_phosphate_poses": crick_phosphate_poses,
    }

    return result 

