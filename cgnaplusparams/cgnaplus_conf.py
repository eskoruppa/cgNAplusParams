from __future__ import annotations

import numpy as np
from pathlib import Path

from ._so3 import so3
from .utils.assignment_utils import (
    cgnaplus_name_assignment,
    inter_bp_dof_indices,
    intra_bp_dof_indices,
    watson_phosphate_dof_indices,
    crick_phosphate_dof_indices,
    dof_index,
    params_to_names,
    params_to_nbp,
)
from .naming_conventions import (
    BP_NAME,
    LEN_POSE_NAMES,
    WATSON_BASE_NAME,
    CRICK_BASE_NAME,
    WATSON_PHOSPHATE_NAME,
    CRICK_PHOSPHATE_NAME,
    INTRA_BP_JUNC_NAME,
    INTER_BP_JUNC_NAME,
    B2P_WATSON_JUNC_NAME,
    B2P_CRICK_JUNC_NAME,
    BP2W_JUNC_NAME,
    C2BP_JUNC_NAME,
    B2P_WATSON_PARAM_NAME,
    B2P_CRICK_PARAM_NAME,
    INTRA_BP_PARAM_NAME,
    EXCESS_PARAMETER_DEFINITION_ALGEBRA,
    EXCESS_PARAMETER_DEFINITION_GROUP,
)
from .cgnaplus_params import CGNAPlusParams
from .cgnaplus_params import (
    CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME,
)
from .utils.se3_methods import (
    build_chain,
    params_to_chain,
    build_junctions,
    build_cgnaplus_poses,
    bases_to_bp_poses,
    poses_to_juncs
)


CGNAPCONF_DEFAULT_EXCESS_PARAMS_DEF = EXCESS_PARAMETER_DEFINITION_GROUP
CGNAPCONF_REQUIRED_SETTINGS_ROTATION_DEFINITION = 'euler'
# CGNAPCONF_REQUIRED_SETTINGS_GROUP_SPLIT         = 'group'
CGNAPCONF_REQUIRED_SETTINGS_TRANS_NM            = True
CGNAPCONF_REQUIRED_SETTINGS_ALIGNED_STRANDS     = True
CGNAPCONF_REQUIRED_SETTINGS_REMOVE_FACTOR_FIVE  = True



class CGNAPlusConf:


    def __init__(self) -> None:
        
        # input attributes
        self._cgnaplus_params: CGNAPlusParams | None = None
        self._sequence:    str | None = None
        self._n_snapshots: int = 0
        self._nbp:         int = 0
        
        # pose attributes
        self._poses:                    dict[str, np.ndarray] | None = None
        self._bp_poses:                 np.ndarray | None = None
        self._watson_base_poses:        np.ndarray | None = None
        self._crick_base_poses:         np.ndarray | None = None
        self._watson_phosphate_poses:   np.ndarray | None = None  # zeros where absent
        self._crick_phosphate_poses:    np.ndarray | None = None  # zeros where absent
        self._watson_phosphate_mask:    np.ndarray | None = None  # (N,) bool
        self._crick_phosphate_mask:     np.ndarray | None = None  # (N,) bool

        # junction attributes
        self._junction_array: np.ndarray | None = None
        self._junction_names: list[str]  | None = None
        self._junction_mask:  np.ndarray | None = None 

        # parameter attributes 
        self._params_X: np.ndarray      | None = None
        self._params_Xd: np.ndarray     | None = None
        self._params_Xs: np.ndarray     | None = None
        self._ndof:      int             = 0
        self._param_names: list[str]    | None = None
        # self._param_mask:  np.ndarray   | None = None  # (ndof,) bool

        self._excess_zero: bool = True
        self._group_split: bool = True

        self._poses_set = False
        self._params_set = False


    ########################################################################################################################
    # Build from parameters
    ########################################################################################################################

    @classmethod
    def from_params(
        cls,
        cgnaplus_params: CGNAPlusParams,
        excess: np.ndarray | None = None,
    ) -> CGNAPlusConf:

        self = cls()
        self._set_cgnaplus_params(cgnaplus_params)
        self._set_excess_params(excess, cgnaplus_params.group_split)
        self._params_set = True
        return self
    

    @classmethod
    def from_poses(
        cls,
        cgnaplus_params: CGNAPlusParams,
        watson_base_poses: np.ndarray,
        crick_base_poses: np.ndarray,
        watson_phosphate_poses: np.ndarray,
        crick_phosphate_poses: np.ndarray,
        bp_poses: np.ndarray | None = None,
    ) -> CGNAPlusConf:
        self = cls()
        self._set_cgnaplus_params(cgnaplus_params)
        self._group_split = self.cgnaplus_params.group_split
        self._set_poses(watson_base_poses, crick_base_poses, watson_phosphate_poses, crick_phosphate_poses, bp_poses)
        self._poses_set = True
        return self


    ########################################################################################################################
    # Setters
    ########################################################################################################################


    ########################################################################################################################
    # Properties
    ########################################################################################################################

    @property
    def sequence(self) -> str:
        if self._sequence is not None:
            return self._sequence
        elif self._cgnaplus_params is not None:
            return self._cgnaplus_params.sequence
        else:
            raise AttributeError("Sequence not set.  Construct from CGNAPlusParams or supply sequence.")

    @property
    def nbp(self) -> int:
        if self._nbp > 0:
            return self._nbp
        elif self._cgnaplus_params is not None:
            return self._cgnaplus_params.nbp
        else:
            raise AttributeError("Number of base pairs not set.  Construct from CGNAPlusParams or supply sequence.")

    @property
    def param_names(self) -> list[str]:
        if self._param_names is not None:
            return self._param_names
        elif self._cgnaplus_params is not None:
            return self._cgnaplus_params.param_names
        else:
            raise AttributeError("Parameter names not set.  Construct from CGNAPlusParams or supply param_names.")

    @property
    def params_Xs(self) -> np.ndarray:
        if self._cgnaplus_params is not None:
            return self._cgnaplus_params.gs
        else:
            raise AttributeError("Ground state parameters not set.  Construct from CGNAPlusParams or supply raw_params.")
        
    @property
    def gs(self) -> np.ndarray:
        return self.params_Xs
    
    @property
    def groundstate(self) -> np.ndarray:
        return self.gs
    
    @property
    def cgnaplus_params(self) -> CGNAPlusParams:
        if self._cgnaplus_params is not None:
            return self._cgnaplus_params
        else:
            raise AttributeError("CGNAPlusParams not set.  Construct from CGNAPlusParams or supply raw_params.")

    @property
    def excess_params(self) -> np.ndarray | None:
        self._ensure_params_Xd()
        return self._params_Xd
    
    def params_Xd(self) -> np.ndarray:
        return self.excess_params

    @property
    def raw_params(self) -> np.ndarray:
        self._ensure_params_X()
        return self._params_X

    @property
    def poses(self) -> dict[str, np.ndarray]:
        self._ensure_poses()
        return self._poses

    @property
    def bp_poses(self) -> np.ndarray:
        self._ensure_poses()
        return self.poses['bp_poses']

    @property
    def watson_base_poses(self) -> np.ndarray:
        self._ensure_poses()
        return self.poses['watson_base_poses']
    
    @property
    def crick_base_poses(self) -> np.ndarray:
        self._ensure_poses()
        return self.poses['crick_base_poses']

    @property
    def watson_phosphate_poses(self) -> np.ndarray:
        self._ensure_poses()
        return self.poses['watson_phosphate_poses']

    @property
    def crick_phosphate_poses(self) -> np.ndarray:
        self._ensure_poses()
        return self.poses['crick_phosphate_poses']
    
    def get_pose_by_name(self, name: str, snapshot: int = 0) -> np.ndarray:
        self._ensure_poses()
        pose_key, pose_id = self._pose_name_to_type_and_id(name)
        return self._poses[pose_key][snapshot, pose_id]
    
    def get_param_Xs_by_name(self, name: str) -> np.ndarray:
        idx = dof_index(name, self._param_names)
        return self.params_Xs[idx]
    
    def get_param_Xd_by_name(self, name: str, snapshot: int = 0) -> np.ndarray:
        idx = dof_index(name, self._param_names)
        return self.params_Xd[snapshot, idx]

    ########################################################################################################################
    # Internal methods to ensure values are computed and cached
    ########################################################################################################################

    def _ensure_params_Xd(self) -> None:
        if self._params_Xd is not None:
            return
        if not self._poses_set:
            raise AttributeError("Neither poses nore excess params set.")
        _params_Xd = self.poses_to_params(
            self._watson_base_poses,
            self._crick_base_poses,
            self._watson_phosphate_poses,
            self._crick_phosphate_poses,
            self._bp_poses,
            as_exess=True,
            params_Xs=self._cgnaplus_params.gs if self._cgnaplus_params is not None else None,
            group_split=self._group_split
        )
        self._params_Xd = _params_Xd

    def _ensure_params_X(self) -> None:
        if self._params_X is not None:
            return
        # if self._params_Xd is not None:
        #     raise ValueError("BUG HERE, FIX!")
        #     self._params_X = self.excess_to_raw(self._cgnaplus_params, self._params_Xd if not self._excess_zero else None)
        #     return
        if not self._poses_set:
            raise AttributeError("Neither poses nore excess params set.")
        _params = self.poses_to_params(
            self._watson_base_poses,
            self._crick_base_poses,
            self._watson_phosphate_poses,
            self._crick_phosphate_poses,
            self._bp_poses,
            as_exess=False,
            params_Xs=None,
            group_split=self._group_split
        )
        self._params_X = _params

    def _ensure_poses(self) -> None:
        if self._poses is not None:
            return
        if not self._params_set:
            raise AttributeError("Neither parameters nor poses set.")
        _poses = self.params_to_poses(
            params_Xs=self.params_Xs,
            params_Xd=self.excess_params,
            group_split=self._group_split,
            param_names=self._cgnaplus_params.param_names if self._cgnaplus_params is not None else None,
        )
        self._poses = _poses

    ########################################################################################################################
    # Conversion methods
    ########################################################################################################################

    @classmethod
    def excess_to_raw(cls, cgnaplus_params: CGNAPlusParams, excess: np.ndarray) -> np.ndarray:
        if excess is None:
            return cgnaplus_params.gs.copy()
        excess = np.asarray(excess, dtype=float)
        ndof = cgnaplus_params.gs.shape[0]
        if excess.ndim == 2:
            excess = excess[np.newaxis]   # (1, ndof, 6)
        if excess.shape[-1] != 6 or excess.shape[-2] != ndof:
            raise ValueError(
                f"excess shape {excess.shape} incompatible with "
                f"ground state ndof={ndof}."
            )
        n_snap = excess.shape[0]
        raw = np.zeros((n_snap, ndof, 6))
        for i in range(n_snap):
            raw[i] = excess[i] + cgnaplus_params.gs
        return raw
    
    def _excess_to_raw(self) -> np.ndarray:
        if self._cgnaplus_params is None:
            raise AttributeError("CGNAPlusParams not set.  Construct from CGNAPlusParams or supply raw_params.")
        if self._params_Xd is None:
            raise AttributeError("Excess parameters not set.  Supply excess parameters or construct from CGNAPlusParams with excess.")
        return self.excess_to_raw(self._cgnaplus_params, self._params_Xd if not self._excess_zero else None)

    @classmethod
    def params_to_poses(
            cls, 
            params_Xs: np.ndarray, 
            params_Xd: np.ndarray | None = None, 
            group_split: bool = True,
            param_names: list[str] | None = None,
            initial_bp_pose: np.ndarray = np.eye(4),
            ) -> dict[str, np.ndarray]:
        
        if param_names is None:
            param_names = params_to_names(params_Xs)
        else:
            # enforce expected order of parameters for pose construction, rearranging if necessary
            params_Xs, params_Xd, param_names = cls._rearrange_params(param_names=param_names, params_Xs=params_Xs, params_Xd=params_Xd)

        excess_zero = params_Xd is None
        if not excess_zero:
            params_Xd = np.asarray(params_Xd, dtype=float)
            if params_Xd.ndim == 2:
                params_Xd = params_Xd[np.newaxis]
        nsnap = 1 if excess_zero else params_Xd.shape[0]
        nbp = params_Xs.shape[0] // 4 + 1

        inter_bp_dof_ids         = inter_bp_dof_indices(param_names=param_names)
        intra_bp_dof_ids         = intra_bp_dof_indices(param_names=param_names)
        watson_phosphate_dof_ids = watson_phosphate_dof_indices(param_names=param_names)
        crick_phosphate_dof_ids  = crick_phosphate_dof_indices(param_names=param_names)

        bps_params_Xs = params_Xs[inter_bp_dof_ids]
        bps_params_Xd = params_Xd[:, inter_bp_dof_ids, :] if not excess_zero else None
        bp_params_Xs  = params_Xs[intra_bp_dof_ids]
        bp_params_Xd  = params_Xd[:, intra_bp_dof_ids, :] if not excess_zero else None

        # pad watson phosphate: insert dummy at index 0 (no phosphate at 5' end)
        w_params_Xs = np.concatenate([np.zeros((1, 6)), params_Xs[watson_phosphate_dof_ids]], axis=0)
        w_params_Xd = np.concatenate([np.zeros((nsnap, 1, 6)), params_Xd[:, watson_phosphate_dof_ids, :]], axis=1) if not excess_zero else None
        # pad crick phosphate: append dummy at end (no phosphate at 5' end)
        c_params_Xs = np.concatenate([params_Xs[crick_phosphate_dof_ids], np.zeros((1, 6))], axis=0)
        c_params_Xd = np.concatenate([params_Xd[:, crick_phosphate_dof_ids, :], np.zeros((nsnap, 1, 6))], axis=1) if not excess_zero else None

        bp_poses = params_to_chain(initial_bp_pose, bps_params_Xs, bps_params_Xd, group_split=group_split)
        bp_juncs = build_junctions(bp_params_Xs, bp_params_Xd, group_split=group_split)
        w_juncs  = build_junctions(w_params_Xs,  w_params_Xd,  group_split=group_split)
        c_juncs  = build_junctions(c_params_Xs,  c_params_Xd,  group_split=group_split)

        if excess_zero:
            bp_poses = bp_poses[np.newaxis]
            bp_juncs = bp_juncs[np.newaxis]
            w_juncs  = w_juncs[np.newaxis]
            c_juncs  = c_juncs[np.newaxis]

        watson_base_poses, crick_base_poses, watson_phosphate_poses, crick_phosphate_poses = build_cgnaplus_poses(bp_poses, bp_juncs, w_juncs, c_juncs)

        poses = {
            'bp_poses': bp_poses,
            'watson_base_poses': watson_base_poses,
            'crick_base_poses': crick_base_poses,
            'watson_phosphate_poses': watson_phosphate_poses,
            'crick_phosphate_poses': crick_phosphate_poses,
        }
        return poses


    @classmethod
    def poses_to_params(
        cls,
        watson_base_poses: np.ndarray,
        crick_base_poses: np.ndarray,
        watson_phosphate_poses: np.ndarray,
        crick_phosphate_poses: np.ndarray,
        bp_poses: np.ndarray | None = None,
        as_exess: bool = False,
        params_Xs: np.ndarray | None = None,
        group_split: bool = True,
        param_names_Xs: list[str] | None = None,
    ) -> np.ndarray:

        if watson_base_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Expected watson_base_poses to have shape (n_snap, nbp, 4, 4) or (nbp, 4, 4), but got {watson_base_poses.shape}")
        if watson_base_poses.ndim < 3:
            raise ValueError(f"Expected watson_base_poses to have shape (n_snap, nbp, 4, 4) or (nbp, 4, 4), but got {watson_base_poses.shape}")
        if watson_base_poses.ndim > 4:
            raise ValueError(f"Expected watson_base_poses to have shape (n_snap, nbp, 4, 4) or (nbp, 4, 4), but got {watson_base_poses.shape}")
        if crick_base_poses.shape != watson_base_poses.shape:
            raise ValueError(f"Expected crick_base_poses to have the same shape as watson_base_poses, but got {crick_base_poses.shape} and {watson_base_poses.shape}")
        
        if watson_phosphate_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Expected watson_phosphate_poses to have shape (n_snap, nbp, 4, 4) or (nbp, 4, 4), but got {watson_phosphate_poses.shape}")
        if watson_phosphate_poses.ndim < 3:
            raise ValueError(f"Expected watson_phosphate_poses to have shape (n_snap, nbp, 4, 4) or (nbp, 4, 4), but got {watson_phosphate_poses.shape}")
        if watson_phosphate_poses.ndim > 4:
            raise ValueError(f"Expected watson_phosphate_poses to have shape (n_snap, nbp, 4, 4) or (nbp, 4, 4), but got {watson_phosphate_poses.shape}")
        if crick_phosphate_poses.shape != watson_phosphate_poses.shape:
            raise ValueError(f"Expected crick_phosphate_poses to have the same shape as watson_phosphate_poses, but got {crick_phosphate_poses.shape} and {watson_phosphate_poses.shape}")

        if watson_base_poses.ndim != watson_phosphate_poses.ndim:
            raise ValueError(f"Expected watson_base_poses and watson_phosphate_poses to have the same number of dimensions, but got {watson_base_poses.ndim} and {watson_phosphate_poses.ndim}")

        # normalise to 4-D (nsnap, nbp, 4, 4) for consistent indexing below
        if watson_base_poses.ndim == 3:
            watson_base_poses      = watson_base_poses[np.newaxis]
            crick_base_poses       = crick_base_poses[np.newaxis]
            watson_phosphate_poses = watson_phosphate_poses[np.newaxis]
            crick_phosphate_poses  = crick_phosphate_poses[np.newaxis]

        nbases = watson_base_poses.shape[-3]   # nbp axis; shape[-1] == 4 always
        nphos  = watson_phosphate_poses.shape[-3]
        if nphos == nbases:
            # strip the dummy terminal entries inserted by params_to_poses
            watson_phosphate_poses = watson_phosphate_poses[:, 1:,  :, :]
            crick_phosphate_poses  = crick_phosphate_poses[:,  :-1, :, :]
        elif nphos != nbases - 1:
            raise ValueError(f"Expected last two dimensions of phosphate poses to be (nbp, 4, 4) or (nbp-1, 4, 4), but got {watson_phosphate_poses.shape} and {crick_phosphate_poses.shape}")

        if bp_poses is not None:
            if bp_poses.ndim == 3:
                bp_poses = bp_poses[np.newaxis]
            if bp_poses.shape != watson_base_poses.shape:
                raise ValueError(f"Expected bp_poses to have the same shape as watson_base_poses, but got {bp_poses.shape} and {watson_base_poses.shape}")
        else:
            bp_poses = bases_to_bp_poses(watson_base_poses, crick_base_poses)

        # compute intra-bp junctions using the FULL (nbp) base-pose arrays BEFORE trimming
        bp_juncs = poses_to_juncs(crick_base_poses, watson_base_poses)   # (nsnap, nbp,   4, 4)

        # trim base poses to align with the (nbp-1) phosphate pose arrays
        watson_base_poses_w = watson_base_poses[:, 1:,  :, :]   # bp 1..N-1 — match watson phosphate
        crick_base_poses_c  = crick_base_poses[:,  :-1, :, :]   # bp 0..N-2 — match crick  phosphate

        # compute junctions
        bps_juncs = poses_to_juncs(bp_poses)                                     # (nsnap, nbp-1, 4, 4)
        w_juncs   = poses_to_juncs(watson_base_poses_w, watson_phosphate_poses)  # (nsnap, nbp-1, 4, 4)
        c_juncs   = poses_to_juncs(crick_base_poses_c,  crick_phosphate_poses)   # (nsnap, nbp-1, 4, 4)

        nbp  = bp_poses.shape[-3]  
        param_names = cgnaplus_name_assignment("A" * nbp)

        if as_exess:
            if params_Xs is None:
                if not hasattr(cls, '_cgnaplus_params') or cls._cgnaplus_params is None:
                    raise ValueError("Ground state parameters not available.  Supply params_Xs or construct from CGNAPlusParams.")
                params_Xs = cls._cgnaplus_params.gs
                
            # Determine DOF ordering of params_Xs; may differ from the output canonical order.
            _param_names_Xs = param_names_Xs if param_names_Xs is not None else list(param_names)

            if group_split:

                inter_bp_dof_ids_Xs         = inter_bp_dof_indices(param_names=_param_names_Xs)
                intra_bp_dof_ids_Xs         = intra_bp_dof_indices(param_names=_param_names_Xs)
                watson_phosphate_dof_ids_Xs = watson_phosphate_dof_indices(param_names=_param_names_Xs)
                crick_phosphate_dof_ids_Xs  = crick_phosphate_dof_indices(param_names=_param_names_Xs)
                bps_params_Xs = params_Xs[inter_bp_dof_ids_Xs]
                bp_params_Xs  = params_Xs[intra_bp_dof_ids_Xs]
                w_params_Xs   = params_Xs[watson_phosphate_dof_ids_Xs]
                c_params_Xs   = params_Xs[crick_phosphate_dof_ids_Xs]

                bps_gs_inv = so3.se3_inverse_batch(so3.se3_euler2rotmat_batch(bps_params_Xs))
                bps_params_Xd = so3.se3_rotmat2euler_batch(bps_gs_inv @ bps_juncs)

                bp_gs_inv  = so3.se3_inverse_batch(so3.se3_euler2rotmat_batch(bp_params_Xs))
                bp_params_Xd = so3.se3_rotmat2euler_batch(bp_gs_inv @ bp_juncs)

                w_gs_inv   = so3.se3_inverse_batch(so3.se3_euler2rotmat_batch(w_params_Xs))
                w_params_Xd = so3.se3_rotmat2euler_batch(w_gs_inv @ w_juncs)

                c_gs_inv   = so3.se3_inverse_batch(so3.se3_euler2rotmat_batch(c_params_Xs))
                c_params_Xd = so3.se3_rotmat2euler_batch(c_gs_inv @ c_juncs)

                params_Xd = cls._combine_params(
                    bps_params_Xd,
                    bp_params_Xd,
                    w_params_Xd,
                    c_params_Xd,
                    param_names=param_names
                )

            else:
                bps_params = so3.se3_rotmat2euler_batch(bps_juncs)
                bp_params = so3.se3_rotmat2euler_batch(bp_juncs)
                w_params  = so3.se3_rotmat2euler_batch(w_juncs)
                c_params  = so3.se3_rotmat2euler_batch(c_juncs)

                params = cls._combine_params(
                    bps_params,
                    bp_params,
                    w_params,
                    c_params,
                    param_names=param_names
                )

                if params.shape[-params_Xs.ndim:] != params_Xs.shape:
                    raise ValueError(f"Computed params shape {params.shape} does not match params_Xs shape {params_Xs.shape}.")
                params_Xd = params - params_Xs

            return params_Xd

        bps_params = so3.se3_rotmat2euler_batch(bps_juncs)
        bp_params = so3.se3_rotmat2euler_batch(bp_juncs)
        w_params  = so3.se3_rotmat2euler_batch(w_juncs)
        c_params  = so3.se3_rotmat2euler_batch(c_juncs)
        params = cls._combine_params(
            bps_params,
            bp_params,
            w_params,
            c_params,
            param_names=param_names
        )
        return params
    


    ########################################################################################################################
    # setting attributes methods
    ########################################################################################################################

    def _init_cgnaplus_params(self, sequence: str, parameter_set_name: str = CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME, group_split: bool = True, include_stiffness: bool = False) -> None:
        cgnaplus_params = CGNAPlusParams(
            sequence=sequence,
            parameter_set_name=parameter_set_name,
            euler_definition=CGNAPCONF_REQUIRED_SETTINGS_ROTATION_DEFINITION == 'euler',
            group_split=group_split,
            translations_in_nm=CGNAPCONF_REQUIRED_SETTINGS_TRANS_NM,
            aligned_strands=CGNAPCONF_REQUIRED_SETTINGS_ALIGNED_STRANDS,
            include_stiffness=include_stiffness,
            remove_factor_five=CGNAPCONF_REQUIRED_SETTINGS_REMOVE_FACTOR_FIVE,
        )
        self._set_cgnaplus_params(cgnaplus_params)

    def _set_cgnaplus_params(self, cgnaplus_params: CGNAPlusParams) -> None:
        self._cgnaplus_params = cgnaplus_params
        self._param_names = list(cgnaplus_params.param_names)
        self._sequence    = cgnaplus_params.sequence
        self._params_Xs   = cgnaplus_params.gs
        self._nbp         = len(inter_bp_dof_indices(self._param_names)) + 1
        self._ndof        = self._params_Xs.shape[0]
        self._validate_cgnaplus_settings(cgnaplus_params=cgnaplus_params)


    def _set_excess_params(self, excess: np.ndarray | None, group_split: bool = True) -> None:
        excess, excess_zero = self._validate_excess(excess)
        self._group_split = group_split
        self._params_Xd = excess
        self._excess_zero = excess_zero
        self._n_snapshots = 1 if excess_zero else excess.shape[0]

    def _validate_excess(self, excess: np.ndarray | None) -> np.ndarray:
        if self._cgnaplus_params is None:
            raise ValueError("CGNAPlusParams not set.  Construct from CGNAPlusParams or supply raw_params.")

        if excess is None:
            excess = np.zeros(self._params_Xs.shape)
            excess_zero = True
            return excess, excess_zero

        excess_zero = False
        excess = np.asarray(excess, dtype=float)
        if excess.ndim < 2:
            raise ValueError(f"Excess parameters must have at least 2 dimensions (ndof, 6) or (n_snap, ndof, 6), but got {excess.shape}.")
        if excess.ndim > 3:
            raise ValueError(f"Excess parameters must have at most 3 dimensions (ndof, 6) or (n_snap, ndof, 6), but got {excess.shape}.")
        if excess.shape[-1] != 6:
            raise ValueError(f"Last dimension of excess parameters must be 6, but got {excess.shape}.")

        if excess.ndim == 2:
            excess = excess[np.newaxis]   # (1, ndof, 6)
        if excess.shape[-1] != 6 or excess.shape[-2] != self._ndof:
            raise ValueError(
                f"excess shape {excess.shape} incompatible with "
                f"ground state ndof={self._ndof}."
            )
        return excess, excess_zero


    def _set_poses(
            self, 
            watson_base_poses: np.ndarray,
            crick_base_poses: np.ndarray,
            watson_phosphate_poses: np.ndarray,
            crick_phosphate_poses: np.ndarray,
            bp_poses: np.ndarray | None = None,
            ) -> None:
        
        watson_base_poses, crick_base_poses, watson_phosphate_poses, crick_phosphate_poses, bp_poses = self._validate_poses(
            watson_base_poses=watson_base_poses,
            crick_base_poses=crick_base_poses,
            watson_phosphate_poses=watson_phosphate_poses,
            crick_phosphate_poses=crick_phosphate_poses,
            bp_poses=bp_poses
        )
        
        self._bp_poses = bp_poses
        self._watson_base_poses = watson_base_poses
        self._crick_base_poses = crick_base_poses
        self._watson_phosphate_poses = watson_phosphate_poses
        self._crick_phosphate_poses = crick_phosphate_poses
        self._watson_phosphate_mask = np.any(watson_phosphate_poses != 0, axis=(-2, -1)) 
        self._crick_phosphate_mask = np.any(crick_phosphate_poses != 0, axis=(-2, -1)) 
        self._n_snapshots = len(watson_base_poses)
        self._poses = {
            'bp_poses': bp_poses,
            'watson_base_poses': watson_base_poses,
            'crick_base_poses': crick_base_poses,
            'watson_phosphate_poses': watson_phosphate_poses,
            'crick_phosphate_poses': crick_phosphate_poses,
        }


    def _validate_poses(
        self,
        watson_base_poses: np.ndarray,
        crick_base_poses: np.ndarray,
        watson_phosphate_poses: np.ndarray,
        crick_phosphate_poses: np.ndarray,
        bp_poses: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:


        if self._cgnaplus_params is None:
            raise ValueError("CGNAPlusParams not set.  Construct from CGNAPlusParams or supply raw_params.")

        if watson_base_poses.ndim < 3:
            raise ValueError(f"Watson base poses must have at least 3 dimensions (snapshot, nbp, 4, 4), but got {watson_base_poses.shape}.")
        if crick_base_poses.ndim < 3:
            raise ValueError(f"Crick base poses must have at least 3 dimensions (snapshot, nbp, 4, 4), but got {crick_base_poses.shape}.")
        if watson_phosphate_poses.ndim < 3:
            raise ValueError(f"Watson phosphate poses must have at least 3 dimensions (snapshot, nbp, 4, 4), but got {watson_phosphate_poses.shape}.")
        if crick_phosphate_poses.ndim < 3:
            raise ValueError(f"Crick phosphate poses must have at least 3 dimensions (snapshot, nbp, 4, 4), but got {crick_phosphate_poses.shape}.")
        if bp_poses is not None and bp_poses.ndim < 3:
            raise ValueError(f"Base pair poses must have at least 3 dimensions (snapshot, nbp, 4, 4), but got {bp_poses.shape}.")

        if watson_base_poses.ndim > 4:
            raise ValueError(f"Watson base poses must have at most 4 dimensions (snapshot, nbp, 4, 4), but got {watson_base_poses.shape}.")
        if crick_base_poses.ndim > 4:
            raise ValueError(f"Crick base poses must have at most 4 dimensions (snapshot, nbp, 4, 4), but got {crick_base_poses.shape}.")
        if watson_phosphate_poses.ndim > 4:
            raise ValueError(f"Watson phosphate poses must have at most 4 dimensions (snapshot, nbp, 4, 4), but got {watson_phosphate_poses.shape}.")
        if crick_phosphate_poses.ndim > 4:
            raise ValueError(f"Crick phosphate poses must have at most 4 dimensions (snapshot, nbp, 4, 4), but got {crick_phosphate_poses.shape}.")
        if bp_poses is not None and bp_poses.ndim > 4:
            raise ValueError(f"Base pair poses must have at most 4 dimensions (snapshot, nbp, 4, 4), but got {bp_poses.shape}.")
           
        if watson_base_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Watson base poses must have shape (..., 4, 4), but got {watson_base_poses.shape}.")
        if crick_base_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Crick base poses must have shape (..., 4, 4), but got {crick_base_poses.shape}.")
        if watson_phosphate_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Watson phosphate poses must have shape (..., 4, 4), but got {watson_phosphate_poses.shape}.")
        if crick_phosphate_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Crick phosphate poses must have shape (..., 4, 4), but got {crick_phosphate_poses.shape}.")
        if bp_poses is not None and bp_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Base pair poses must have shape (..., 4, 4), but got {bp_poses.shape}.")
        
        if watson_base_poses.shape[-3] != self._nbp:
            raise ValueError(f"Mismatch between shape of provided Watson base poses and cgnaplus_params. Watson base poses must have shape (..., nbp, 4, 4), but got {watson_base_poses.shape} with nbp={watson_base_poses.shape[-3]} while expected nbp={self._nbp}.")
        if crick_base_poses.shape[-3] != self._nbp:
            raise ValueError(f"Mismatch between shape of provided Crick base poses and cgnaplus_params. Crick base poses must have shape (..., nbp, 4, 4), but got {crick_base_poses.shape} with nbp={crick_base_poses.shape[-3]} while expected nbp={self._nbp}.")
        if watson_phosphate_poses.shape[-3] not in [self._nbp, self._nbp - 1]:
            raise ValueError(f"Mismatch between shape of provided Watson phosphate poses and cgnaplus_params. Watson phosphate poses must have shape (..., nbp, 4, 4), but got {watson_phosphate_poses.shape} with nbp={watson_phosphate_poses.shape[-3]} while expected nbp={self._nbp}.")
        if crick_phosphate_poses.shape[-3] not in [self._nbp, self._nbp - 1]:
            raise ValueError(f"Mismatch between shape of provided Crick phosphate poses and cgnaplus_params. Crick phosphate poses must have shape (..., nbp, 4, 4), but got {crick_phosphate_poses.shape} with nbp={crick_phosphate_poses.shape[-3]} while expected nbp={self._nbp}.")
        if bp_poses is not None and bp_poses.shape[-3] != self._nbp:
            raise ValueError(f"Mismatch between shape of provided base pair poses and cgnaplus_params. Base pair poses must have shape (..., nbp, 4, 4), but got {bp_poses.shape} with nbp={bp_poses.shape[-3]} while expected nbp={self._nbp}.")
        

        if watson_base_poses.ndim < 4:
            watson_base_poses = watson_base_poses[np.newaxis]  # add snapshot axis
        if crick_base_poses.ndim < 4:
            crick_base_poses = crick_base_poses[np.newaxis]  # add snapshot axis
        if watson_phosphate_poses.ndim < 4:
            watson_phosphate_poses = watson_phosphate_poses[np.newaxis]  # add snapshot axis
        if crick_phosphate_poses.ndim < 4:
            crick_phosphate_poses = crick_phosphate_poses[np.newaxis]  # add snapshot axis
        if bp_poses is not None and bp_poses.ndim < 4:
            bp_poses = bp_poses[np.newaxis]  # add snapshot axis

        if len(crick_base_poses) != len(watson_base_poses):
            raise ValueError(f"Mismatch in number of snapshots between Watson base poses and Crick base poses. Watson base poses has {len(watson_base_poses)} snapshots while Crick base poses has {len(crick_base_poses)} snapshots.")
        
        if len(watson_phosphate_poses) != len(watson_base_poses):
            raise ValueError(f"Mismatch in number of snapshots between Watson base poses and Watson phosphate poses. Watson base poses has {len(watson_base_poses)} snapshots while Watson phosphate poses has {len(watson_phosphate_poses)} snapshots.")
        
        if len(crick_phosphate_poses) != len(crick_base_poses):
            raise ValueError(f"Mismatch in number of snapshots between Crick base poses and Crick phosphate poses. Crick base poses has {len(crick_base_poses)} snapshots while Crick phosphate poses has {len(crick_phosphate_poses)} snapshots.")

        if bp_poses is not None and len(bp_poses) != len(watson_base_poses):
            raise ValueError(f"Mismatch in number of snapshots between Watson base poses and base pair poses. Watson base poses has {len(watson_base_poses)} snapshots while base pair poses has {len(bp_poses)} snapshots.")

        if bp_poses is None:
            bp_poses = bases_to_bp_poses(watson_base_poses, crick_base_poses)

        if watson_phosphate_poses.shape[-3] == self._nbp - 1:
            # add dummy phosphate pose for missing terminal phosphate
            dummy_phosphate_pose = np.zeros((4, 4))
            watson_phosphate_poses = np.insert(watson_phosphate_poses, 0, dummy_phosphate_pose, axis=-3)
        if crick_phosphate_poses.shape[-3] == self._nbp - 1:
            # add dummy phosphate pose for missing terminal phosphate
            dummy_phosphate_pose = np.zeros((4, 4))
            crick_phosphate_poses = np.insert(crick_phosphate_poses, crick_phosphate_poses.shape[-3], dummy_phosphate_pose, axis=-3)

        return watson_base_poses, crick_base_poses, watson_phosphate_poses, crick_phosphate_poses, bp_poses


    def _wipe_poses(self) -> None:
        self._poses = None
        self._poses_set = False
        self._bp_poses = None
        self._watson_base_poses = None
        self._crick_base_poses = None
        self._watson_phosphate_poses = None
        self._crick_phosphate_poses = None
        self._watson_phosphate_mask = None
        self._crick_phosphate_mask = None

    def _wipe_excess(self) -> None:
        self._params_Xd = None
        self._params_X = None
        self._excess_zero = True


    # def _validate(self) -> bool:
    #     """Validate that the CGNAPlusConf is internally consistent and compatible with its CGNAPlusParams."""
    #     self._validate_cgnaplus_params()
    #     return True

    def _validate_cgnaplus_settings(self, cgnaplus_params: CGNAPlusParams | None = None) -> bool:

        if cgnaplus_params is None:
            cgnaplus_params = self._cgnaplus_params

        if self._cgnaplus_params is None:
            raise ValueError("CGNAPlusParams not set.  Construct from CGNAPlusParams or supply raw_params.")

        
        if self._cgnaplus_params.rotation_definition.lower() != CGNAPCONF_REQUIRED_SETTINGS_ROTATION_DEFINITION.lower():
            raise ValueError(
                f"CGNAPlusParams rotation definition {self._cgnaplus_params.rotation_definition} "
                f"incompatible with CGNAPlusConf requirement of {CGNAPCONF_REQUIRED_SETTINGS_ROTATION_DEFINITION}."
            )

        # if self._cgnaplus_params.splitting_definition.lower() != CGNAPCONF_REQUIRED_SETTINGS_GROUP_SPLIT.lower():
        #     raise ValueError(
        #         f"CGNAPlusParams group_split {self._cgnaplus_params.group_split.lower()} "
        #         f"incompatible with CGNAPlusConf requirement of {CGNAPCONF_REQUIRED_SETTINGS_GROUP_SPLIT.lower()}."
        #     )

        if self._cgnaplus_params.translations_in_nm != CGNAPCONF_REQUIRED_SETTINGS_TRANS_NM:
            raise ValueError(
                f"CGNAPlusParams translations_in_nm {self._cgnaplus_params.translations_in_nm} "
                f"incompatible with CGNAPlusConf requirement of {CGNAPCONF_REQUIRED_SETTINGS_TRANS_NM}."
            )
        if self._cgnaplus_params.aligned_strands != CGNAPCONF_REQUIRED_SETTINGS_ALIGNED_STRANDS:
            raise ValueError(
                f"CGNAPlusParams aligned_strands {self._cgnaplus_params.aligned_strands} "
                f"incompatible with CGNAPlusConf requirement of {CGNAPCONF_REQUIRED_SETTINGS_ALIGNED_STRANDS}."
            )
        
        if self._cgnaplus_params.remove_factor_five != CGNAPCONF_REQUIRED_SETTINGS_REMOVE_FACTOR_FIVE:
            raise ValueError(
                f"CGNAPlusParams remove_factor_five {self._cgnaplus_params.remove_factor_five} "
                f"incompatible with CGNAPlusConf requirement of {CGNAPCONF_REQUIRED_SETTINGS_REMOVE_FACTOR_FIVE}."
            )
        

    ########################################################################################################################
    # Auxiliary methods
    ########################################################################################################################

    @staticmethod
    def _rearrange_params(param_names: list[str], params_Xs: np.ndarray, params_Xd: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        """
        Rearrange params_Xs and params_Xd to the canonical DOF order expected by pose construction,
        based on param_names.  If param_names is already in canonical order, return as is.  Otherwise
        rearrange to canonical order.  Raise an error if param_names is missing any expected DOF name.

        Returns the reordered params_Xs, params_Xd, and the canonical param_names so callers can
        use the returned name list for subsequent DOF index computations on the reordered arrays.
        """

        if len(param_names) != params_Xs.shape[0]:
            raise ValueError(f"Length of param_names {len(param_names)} does not match number of parameters {params_Xs.shape[0]}.")
        
        if params_Xd is not None and len(param_names) != params_Xd.shape[-2]:
            raise ValueError(f"Length of param_names {len(param_names)} does not match number of excess parameters {params_Xd.shape[-2]}.")

        expected_order = params_to_names(params_Xs)

        # if match return as is
        if param_names == expected_order:
            return params_Xs, params_Xd, list(param_names)
        
        # else rearrange according to expected order
        try:
            idx = [param_names.index(name) for name in expected_order]
        except ValueError as e:
            raise ValueError(f"param_names is missing an expected DOF name: {e}") from e

        params_Xs = params_Xs[idx]
        if params_Xd is not None:
            params_Xd = params_Xd[..., idx, :]

        return params_Xs, params_Xd, expected_order
    
    @staticmethod
    def _combine_params(
            bps_params: np.ndarray,
            bp_params: np.ndarray,
            w_params: np.ndarray,
            c_params: np.ndarray,
            param_names: list[str] | None = None,
    ) -> np.ndarray:
        """Reassemble split parameter arrays back into the full cgNA+ DOF array.

        Reverses the splitting performed in ``params_to_poses`` via
        ``inter_bp_dof_indices``, ``intra_bp_dof_indices``,
        ``watson_phosphate_dof_indices``, and ``crick_phosphate_dof_indices``.
        Supports an arbitrary number of leading batch dimensions.

        Parameters
        ----------
        bps_params : np.ndarray, shape (..., N-1, 6)
            Inter-base-pair step parameters.
        bp_params : np.ndarray, shape (..., N, 6)
            Intra-base-pair parameters.
        w_params : np.ndarray, shape (..., N-1, 6)
            Watson-phosphate parameters (W1 … W(N-1)).
        c_params : np.ndarray, shape (..., N-1, 6)
            Crick-phosphate parameters (C0 … C(N-2)).
        param_names : list[str], optional
            DOF name list of length ``4*N - 3``. Inferred from ``N`` if omitted.

        Returns
        -------
        np.ndarray, shape (..., 4*N - 3, 6)
        """
        nbp  = bp_params.shape[-2]          # N (intra-bp has one entry per bp)
        ndof = 4 * nbp - 3

        if param_names is None:
            param_names = cgnaplus_name_assignment("A" * nbp)

        inter_bp_ids = inter_bp_dof_indices(param_names)
        intra_bp_ids = intra_bp_dof_indices(param_names)
        watson_ids   = watson_phosphate_dof_indices(param_names)
        crick_ids    = crick_phosphate_dof_indices(param_names)

        batch_shape = bp_params.shape[:-2]
        params = np.zeros(batch_shape + (ndof, 6))
        params[..., inter_bp_ids, :] = bps_params
        params[..., intra_bp_ids, :] = bp_params
        params[..., watson_ids,   :] = w_params
        params[..., crick_ids,    :] = c_params
        return params
    
    @staticmethod
    def _pose_name_to_type_and_id(name: str) -> tuple[str, int]:
        pose_type = name[:LEN_POSE_NAMES]
        pose_id   = int(name[LEN_POSE_NAMES:])
        if pose_type == BP_NAME:
            type_str = 'bp_poses'
        if pose_type == WATSON_BASE_NAME:
            type_str = 'watson_base_poses'
        elif pose_type == CRICK_BASE_NAME:
            type_str = 'crick_base_poses'
        elif pose_type == WATSON_PHOSPHATE_NAME:
            type_str = 'watson_phosphate_poses'
        elif pose_type == CRICK_PHOSPHATE_NAME: 
            type_str = 'crick_phosphate_poses'
        else:
            raise ValueError(f"Unknown pose type: {pose_type}")
        return type_str, pose_id
    

########################################################################################################################
# Read poses from pdb, cif, or md trajectory
########################################################################################################################

from .input.reader import read_dna

def confs_from_traj(
    filename: Path | str,
    use_aligned_domains: bool = True,
    parameter_set_name: str = CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME,
    include_stiffness: bool = False,
    convention: str = "tsukuba",
    ) -> list[CGNAPlusConf]:
    reader = read_dna(Path(filename))

    # TODO: allow iteration through snaphots

    confs = []

    print(f'Read {len(reader.duplexes)} duplexes from {filename} with use_aligned_domains={use_aligned_domains}.')

    for dup in reader.duplexes:
        if use_aligned_domains:
            print(f'Processing duplex with {len(dup.aligned_domains)} aligned domains.')
            for i,ad in enumerate(dup.aligned_domains):
                poses = ad.fit_frames(convention=convention, crickflip=True)
                sequence = ad.sequence

                print(f'Processing aligned domain {i} with sequence ({len(sequence)}): {sequence}')

                cgnap = CGNAPlusParams(
                    sequence=sequence,
                    parameter_set_name=parameter_set_name,
                    euler_definition=True,
                    group_split=True,
                    translations_in_nm=True,
                    aligned_strands=True,
                    include_stiffness=include_stiffness,
                    remove_factor_five=True,
                )

                conf = CGNAPlusConf.from_poses(
                    cgnap,
                    watson_base_poses=poses['watson_bases'],
                    crick_base_poses=poses['crick_bases'],
                    watson_phosphate_poses=poses['watson_phosphates'],
                    crick_phosphate_poses=poses['crick_phosphates'],
                    )
                
                confs.append(conf)
        else:
            for i,bd in enumerate(dup.bonded_domains):
                poses = bd.fit_frames(convention=convention, crickflip=True)
                sequence = bd.sequence
                cgnap = CGNAPlusParams(
                    sequence=sequence,
                    parameter_set_name=parameter_set_name,
                    euler_definition=True,
                    group_split=True,
                    translations_in_nm=True,
                    aligned_strands=True,
                    include_stiffness=include_stiffness,
                    remove_factor_five=True,
                )

                conf = CGNAPlusConf.from_poses(
                    cgnap,
                    watson_base_poses=poses['watson_bases'],
                    crick_base_poses=poses['crick_bases'],
                    watson_phosphate_poses=poses['watson_phosphates'],
                    crick_phosphate_poses=poses['crick_phosphates'],
                    )
                
                confs.append(conf)
    return confs