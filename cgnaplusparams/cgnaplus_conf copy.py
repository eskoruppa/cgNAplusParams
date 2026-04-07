"""cgNA+ configuration class.

Provides :class:`CGNAPlusConf`, which stores one or more snapshots of a
cgNA+ DNA chain configuration in typed contiguous numpy arrays with a leading
snapshot axis ``(n_snap, ...)``.

Lazy evaluation
---------------
Pose chains, junction frames, and DOF parameter vectors are all computed on
first access and then cached.  When a configuration is initialised from
parameters (ground state ± dynamic deformation), poses are built via the
chain-building algorithm.  When initialised from externally supplied poses
(e.g. fitted from a PDB or MD trajectory), junction frames and DOF parameter
vectors are derived on demand.

Construction
------------
Use the class methods rather than ``__init__`` directly:

* :meth:`CGNAPlusConf.from_params`     – from a :class:`~cgnaplus_params.CGNAPlusParams` object
* :meth:`CGNAPlusConf.from_raw_params` – from a pre-built ``(n_snap, n_dof, 6)`` array
* :meth:`CGNAPlusConf.from_poses`      – from externally supplied SE(3) pose arrays
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._so3 import so3
from .rbp_conf import _build_first_pose
from .utils.se3_methods import build_chain
from .utils.assignment_utils import (
    cgnaplus_name_assignment,
    inter_bp_dof_indices,
    intra_bp_dof_indices,
    dof_index,
)
from .naming_conventions import (
    BP_NAME,
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
)

if TYPE_CHECKING:
    from .cgnaplus_params import CGNAPlusParams


# ---------------------------------------------------------------------------
# Internal SE(3) batch helpers
# ---------------------------------------------------------------------------

def _se3_inv_batch(g: np.ndarray) -> np.ndarray:
    """Batch SE(3) inversion for arrays of shape ``(..., 4, 4)``.

    Uses the SE(3) structure: ``inv([[R, t],[0,1]]) = [[R.T, -R.T t],[0,1]]``.
    """
    inv_g = np.zeros_like(g)
    R  = g[..., :3, :3]
    t  = g[..., :3, 3]
    Rt = R.swapaxes(-1, -2)
    inv_g[..., :3, :3] = Rt
    inv_g[..., :3, 3]  = -np.einsum("...ij,...j->...i", Rt, t)
    inv_g[..., 3, 3]   = 1.0
    return inv_g


def _se3_rel_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Relative SE(3) transform ``inv(A) @ B`` for arrays ``(..., 4, 4)``."""
    return np.matmul(_se3_inv_batch(A), B)


# ---------------------------------------------------------------------------
# CGNAPlusConf
# ---------------------------------------------------------------------------

class CGNAPlusConf:
    """Configuration of a cgNA+ DNA chain — one or more snapshots.

    All pose, junction, and parameter arrays carry a leading snapshot axis so
    that single- and multi-snapshot configurations have consistent shapes.
    Use :attr:`n_snapshots` to query the number of frames, or iterate with
    ``for snap in conf``.

    Attributes
    ----------
    n_snapshots : int
    nbp         : int
    sequence    : str | None
    params      : CGNAPlusParams | None
    """

    # ------------------------------------------------------------------
    # Private __init__ — do not call directly; use classmethods below.
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Source data — set by the classmethod constructors
        self._raw_params:  np.ndarray | None = None   # (n_snap, n_dof, 6)
        self._dynamic:     np.ndarray | None = None   # (n_snap, n_dof, 6)
        self._poses_given: bool = False

        # Typed pose arrays — (n_snap, N, 4, 4), lazy
        self._bp_poses:               np.ndarray | None = None
        self._watson_base_poses:      np.ndarray | None = None
        self._crick_base_poses:       np.ndarray | None = None
        self._watson_phosphate_poses: np.ndarray | None = None  # zeros where absent
        self._crick_phosphate_poses:  np.ndarray | None = None  # zeros where absent
        self._watson_phosphate_mask:  np.ndarray | None = None  # (N,) bool
        self._crick_phosphate_mask:   np.ndarray | None = None  # (N,) bool

        # Junction array — (n_snap, n_junc, 4, 4), lazy; zeros where absent
        self._junction_array: np.ndarray | None = None
        self._junction_names: list[str]  | None = None
        self._junction_mask:  np.ndarray | None = None  # (n_junc,) bool

        # Parameter array — (n_snap, n_dof, 6), lazy; zeros where absent
        self._param_array: np.ndarray | None = None
        self._param_names: list[str]  | None = None
        self._param_mask:  np.ndarray | None = None  # (n_dof,) bool

        # Metadata
        self._params:      "CGNAPlusParams | None" = None
        self._sequence:    str | None = None
        self._n_snapshots: int = 0
        self._nbp:         int = 0
        self._orientation: np.ndarray = np.array([0.0, 0.0, 1.0])
        self._origin:      np.ndarray = np.zeros(3)

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        params: "CGNAPlusParams",
        dynamic: np.ndarray | None = None,
        *,
        orientation: np.ndarray | list = np.array([0.0, 0.0, 1.0]),
        origin:      np.ndarray | list = np.zeros(3),
    ) -> "CGNAPlusConf":
        """Build a configuration from a :class:`~cgnaplus_params.CGNAPlusParams` instance.

        Parameters
        ----------
        params:
            Sequence-dependent cgNA+ parameter object.
        dynamic:
            Optional deformation.  Shape ``(n_dof, 6)`` for a single deformed
            snapshot, or ``(n_snap, n_dof, 6)`` for an ensemble.  Applied on
            top of the ground state via the group product when
            ``params.group_split`` is ``True``, otherwise via vector sum.
        orientation, origin:
            Initial reference frame passed to the chain builder.
        """
        self = cls()
        self._params      = params
        self._param_names = list(params.param_names)
        self._sequence    = params.sequence
        gs    = params.gs   # (n_dof, 6)
        n_dof = gs.shape[0]

        if dynamic is not None:
            dynamic = np.asarray(dynamic, dtype=float)
            if dynamic.ndim == 2:
                dynamic = dynamic[np.newaxis]   # (1, n_dof, 6)
            if dynamic.shape[-1] != 6 or dynamic.shape[-2] != n_dof:
                raise ValueError(
                    f"dynamic shape {dynamic.shape} incompatible with "
                    f"ground state n_dof={n_dof}."
                )
            n_snap = dynamic.shape[0]
            raw = np.broadcast_to(gs[np.newaxis], (n_snap, n_dof, 6)).copy()
            self._dynamic = dynamic
        else:
            n_snap = 1
            raw = gs[np.newaxis].copy()   # (1, n_dof, 6)

        self._raw_params  = raw
        self._n_snapshots = n_snap
        self._nbp         = len(inter_bp_dof_indices(self._param_names)) + 1
        self._orientation = np.asarray(orientation, dtype=float)
        self._origin      = np.asarray(origin,      dtype=float)
        return self

    @classmethod
    def from_raw_params(
        cls,
        raw_params:  np.ndarray,
        param_names: list[str],
        *,
        sequence:    str | None = None,
        dynamic:     np.ndarray | None = None,
        params:      "CGNAPlusParams | None" = None,
        orientation: np.ndarray | list = np.array([0.0, 0.0, 1.0]),
        origin:      np.ndarray | list = np.zeros(3),
    ) -> "CGNAPlusConf":
        """Build a configuration from a raw parameter array.

        Parameters
        ----------
        raw_params:
            Ground-state parameters, shape ``(n_dof, 6)`` or
            ``(n_snap, n_dof, 6)``.
        param_names:
            Ordered DOF name list (same convention as
            :attr:`~cgnaplus_params.CGNAPlusParams.param_names`).
        sequence:
            Nucleotide sequence.  Inferred from *params* when omitted.
        dynamic:
            Optional deformation; same shape rules as :meth:`from_params`.
        params:
            Optional reference :class:`~cgnaplus_params.CGNAPlusParams`.
        orientation, origin:
            Initial reference frame.
        """
        self = cls()
        raw_params = np.asarray(raw_params, dtype=float)
        if raw_params.ndim == 2:
            raw_params = raw_params[np.newaxis]   # (1, n_dof, 6)
        n_snap, n_dof, _ = raw_params.shape

        if dynamic is not None:
            dynamic = np.asarray(dynamic, dtype=float)
            if dynamic.ndim == 2:
                dynamic = dynamic[np.newaxis]
            if dynamic.shape[-2:] != (n_dof, 6):
                raise ValueError(
                    f"dynamic shape {dynamic.shape} incompatible with "
                    f"raw_params n_dof={n_dof}."
                )
            n_snap = max(n_snap, dynamic.shape[0])
            self._dynamic = dynamic

        self._raw_params  = raw_params
        self._n_snapshots = n_snap
        self._param_names = list(param_names)
        self._params      = params
        self._sequence    = sequence or (params.sequence if params else None)
        self._nbp         = len(inter_bp_dof_indices(param_names)) + 1
        self._orientation = np.asarray(orientation, dtype=float)
        self._origin      = np.asarray(origin,      dtype=float)
        return self

    @classmethod
    def from_poses(
        cls,
        bp_poses: np.ndarray,
        *,
        watson_base_poses:      np.ndarray | None = None,
        crick_base_poses:       np.ndarray | None = None,
        watson_phosphate_poses: np.ndarray | None = None,
        crick_phosphate_poses:  np.ndarray | None = None,
        params:   "CGNAPlusParams | None" = None,
        sequence: str | None = None,
        orientation: np.ndarray | list = np.array([0.0, 0.0, 1.0]),
        origin:      np.ndarray | list = np.zeros(3),
    ) -> "CGNAPlusConf":
        """Build a configuration from externally supplied SE(3) pose arrays.

        Parameters
        ----------
        bp_poses:
            Base-pair frames, shape ``(N, 4, 4)`` or ``(n_snap, N, 4, 4)``.
        watson_base_poses, crick_base_poses:
            Watson / Crick base frames; same shape convention.
        watson_phosphate_poses, crick_phosphate_poses:
            Phosphate frames.  Omit or pass zero matrices for absent sites.
            The boolean mask is auto-detected from the first snapshot.
        params:
            Optional reference :class:`~cgnaplus_params.CGNAPlusParams`.
        sequence:
            Nucleotide sequence string.
        orientation, origin:
            Stored for reference; not used in further computation.
        """
        def _norm(arr: np.ndarray, name: str) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 3:
                arr = arr[np.newaxis]   # (1, N, 4, 4)
            if arr.ndim != 4 or arr.shape[-2:] != (4, 4):
                raise ValueError(
                    f"{name}: expected shape (..., N, 4, 4), got {arr.shape}."
                )
            return arr

        self = cls()
        bp_poses = _norm(bp_poses, "bp_poses")
        n_snap, N = bp_poses.shape[:2]

        self._bp_poses = bp_poses

        if watson_base_poses is not None:
            self._watson_base_poses = _norm(watson_base_poses, "watson_base_poses")
        if crick_base_poses is not None:
            self._crick_base_poses = _norm(crick_base_poses, "crick_base_poses")

        if watson_phosphate_poses is not None:
            wp = _norm(watson_phosphate_poses, "watson_phosphate_poses")
            self._watson_phosphate_poses = wp
            self._watson_phosphate_mask  = np.any(wp[0] != 0.0, axis=(-1, -2))
        else:
            self._watson_phosphate_poses = np.zeros((n_snap, N, 4, 4))
            self._watson_phosphate_mask  = np.zeros(N, dtype=bool)

        if crick_phosphate_poses is not None:
            cp = _norm(crick_phosphate_poses, "crick_phosphate_poses")
            self._crick_phosphate_poses = cp
            self._crick_phosphate_mask  = np.any(cp[0] != 0.0, axis=(-1, -2))
        else:
            self._crick_phosphate_poses = np.zeros((n_snap, N, 4, 4))
            self._crick_phosphate_mask  = np.zeros(N, dtype=bool)

        self._poses_given = True
        self._params      = params
        self._sequence    = sequence or (params.sequence if params else None)
        self._n_snapshots = n_snap
        self._nbp         = N

        if params is not None:
            self._param_names = list(params.param_names)
        elif self._sequence is not None:
            self._param_names = cgnaplus_name_assignment(self._sequence)

        self._orientation = np.asarray(orientation, dtype=float)
        self._origin      = np.asarray(origin,      dtype=float)
        return self

    # ------------------------------------------------------------------
    # Lazy-computation guards
    # ------------------------------------------------------------------

    def _ensure_poses(self) -> None:
        if self._bp_poses is not None:
            return
        if self._raw_params is None:
            raise RuntimeError(
                "Cannot compute poses: neither raw parameters nor external "
                "poses have been provided."
            )
        self._compute_poses_from_params()

    def _ensure_junctions(self) -> None:
        self._ensure_poses()
        if self._junction_array is not None:
            return
        self._compute_junctions_from_poses()

    def _ensure_parameters(self) -> None:
        self._ensure_junctions()
        if self._param_array is not None:
            return
        self._compute_parameters_from_junctions()

    # ------------------------------------------------------------------
    # Core computation: parameters → poses
    # ------------------------------------------------------------------

    def _compute_poses_from_params(self) -> None:
        """Run the chain-building algorithm to populate all typed pose arrays.

        Ported from ``build_cgnaplus_conf.cgnaplus_conf()``, extended for the
        snapshot batch dimension.  Uses ``utils.build_chain.build_chain``
        (the old ``rbp_conf._build_chain`` no longer exists).
        """
        if self._param_names is None:
            raise RuntimeError(
                "Cannot compute poses: param_names not set.  Construct via "
                "from_params() or supply param_names to from_raw_params()."
            )

        n_snap      = self._n_snapshots
        raw         = self._raw_params       # (n_snap, n_dof, 6)
        param_names = self._param_names

        inter_bp_ids = inter_bp_dof_indices(param_names)
        intra_bp_ids = intra_bp_dof_indices(param_names)
        N = len(intra_bp_ids)

        # Pre-compute phosphate DOF indices (identical for every snapshot)
        watson_p_exists = np.zeros(N, dtype=bool)
        crick_p_exists  = np.zeros(N, dtype=bool)
        watson_p_idx    = np.full(N, -1, dtype=int)
        crick_p_idx     = np.full(N, -1, dtype=int)
        for i in range(N):
            wid = dof_index(f"{B2P_WATSON_PARAM_NAME}{i}", param_names)
            cid = dof_index(f"{B2P_CRICK_PARAM_NAME}{i}",  param_names)
            if wid is not None:
                watson_p_exists[i] = True
                watson_p_idx[i]    = wid
            if cid is not None:
                crick_p_exists[i] = True
                crick_p_idx[i]    = cid

        # Allocate output arrays
        bp_poses = np.empty((n_snap, N, 4, 4))
        wbase    = np.empty((n_snap, N, 4, 4))
        cbase    = np.empty((n_snap, N, 4, 4))
        wphos    = np.zeros((n_snap, N, 4, 4))
        cphos    = np.zeros((n_snap, N, 4, 4))

        first_pose  = _build_first_pose(orientation=self._orientation, origin=self._origin)
        group_split = self._params.group_split if self._params is not None else True

        for s in range(n_snap):
            params_s = raw[s].copy()   # (n_dof, 6)

            # Apply dynamic deformation
            if self._dynamic is not None:
                dyn_s = self._dynamic[min(s, self._dynamic.shape[0] - 1)]
                if group_split:
                    gs_mat   = so3.se3_euler2rotmat_batch(params_s)
                    ds_mat   = so3.se3_euler2rotmat_batch(dyn_s)
                    combined = gs_mat @ ds_mat
                    params_s = so3.se3_rotmat2euler_batch(combined)
                else:
                    params_s = params_s + dyn_s

            gs = so3.se3_euler2rotmat_batch(params_s)   # (n_dof, 4, 4)

            # Base-pair chain
            bp_chain    = build_chain(first_pose, gs[inter_bp_ids])   # (N+1, 4, 4)
            bp_poses[s] = bp_chain[:N]

            # Watson and Crick base frames from intra-bp DOFs
            for i in range(N):
                Xi          = params_s[intra_bp_ids[i]]
                wbase[s, i] = bp_poses[s, i] @ so3.X2grh(Xi)
                cbase[s, i] = bp_poses[s, i] @ so3.X2glh_inv(Xi)

            # Phosphate frames
            for i in range(N):
                if watson_p_exists[i]:
                    wphos[s, i] = wbase[s, i] @ so3.X2g(params_s[watson_p_idx[i]])
                if crick_p_exists[i]:
                    cphos[s, i] = cbase[s, i] @ so3.X2g(params_s[crick_p_idx[i]])

        self._bp_poses               = bp_poses
        self._watson_base_poses      = wbase
        self._crick_base_poses       = cbase
        self._watson_phosphate_poses = wphos
        self._crick_phosphate_poses  = cphos
        self._watson_phosphate_mask  = watson_p_exists
        self._crick_phosphate_mask   = crick_p_exists

    # ------------------------------------------------------------------
    # Core computation: poses → junctions
    # ------------------------------------------------------------------

    def _compute_junctions_from_poses(self) -> None:
        """Compute SE(3) junction frames from all typed pose arrays.

        The main junction array (first ``n_dof`` entries) mirrors the
        ``param_names`` ordering so that ``junction_names[k].upper() ==
        param_names[k]``.

        Two groups of N extra junctions are appended:

        * ``r{i}`` (:data:`~naming_conventions.BP2W_JUNC_NAME`):
          ``inv(bp[i]) @ watson_base[i]``
        * ``l{i}`` (:data:`~naming_conventions.C2BP_JUNC_NAME`):
          ``inv(crick_base[i]) @ bp[i]``
        """
        N      = self._nbp
        n_snap = self._n_snapshots

        if self._param_names is not None:
            main_junc_names = [n.lower() for n in self._param_names]
        elif self._sequence is not None:
            main_junc_names = [
                n.lower() for n in cgnaplus_name_assignment(self._sequence)
            ]
        else:
            raise RuntimeError(
                "Cannot build junction names: neither param_names nor "
                "sequence is set."
            )

        extra_junc_names = (
            [f"{BP2W_JUNC_NAME}{i}" for i in range(N)]
            + [f"{C2BP_JUNC_NAME}{i}" for i in range(N)]
        )
        all_junc_names = main_junc_names + extra_junc_names
        n_junc = len(all_junc_names)

        junc_array = np.zeros((n_snap, n_junc, 4, 4))
        junc_mask  = np.ones(n_junc, dtype=bool)

        bp    = self._bp_poses
        wb    = self._watson_base_poses
        cb    = self._crick_base_poses
        wp    = self._watson_phosphate_poses
        cp    = self._crick_phosphate_poses
        wmask = self._watson_phosphate_mask
        cmask = self._crick_phosphate_mask

        for k, jname in enumerate(main_junc_names):
            letter = jname[0]
            idx    = int(jname[1:])

            if letter == INTRA_BP_JUNC_NAME:       # 'x' — bp → watson base
                junc_array[:, k] = _se3_rel_batch(bp[:, idx], wb[:, idx])

            elif letter == B2P_CRICK_JUNC_NAME:    # 'c' — crick base → crick phosphate
                if cmask is not None and idx < len(cmask) and cmask[idx]:
                    junc_array[:, k] = _se3_rel_batch(cb[:, idx], cp[:, idx])
                else:
                    junc_mask[k] = False

            elif letter == INTER_BP_JUNC_NAME:     # 'y' — bp[i] → bp[i+1]
                if idx + 1 < N:
                    junc_array[:, k] = _se3_rel_batch(bp[:, idx], bp[:, idx + 1])
                else:
                    junc_mask[k] = False

            elif letter == B2P_WATSON_JUNC_NAME:   # 'w' — watson base → watson phosphate
                if wmask is not None and idx < len(wmask) and wmask[idx]:
                    junc_array[:, k] = _se3_rel_batch(wb[:, idx], wp[:, idx])
                else:
                    junc_mask[k] = False

        # Extra junctions r{i} and l{i}
        if wb is not None and cb is not None:
            n_main = len(main_junc_names)
            for i in range(N):
                junc_array[:, n_main + i]     = _se3_rel_batch(bp[:, i], wb[:, i])
                junc_array[:, n_main + N + i] = _se3_rel_batch(cb[:, i], bp[:, i])

        self._junction_array = junc_array
        self._junction_names = all_junc_names
        self._junction_mask  = junc_mask

    # ------------------------------------------------------------------
    # Core computation: junctions → parameters
    # ------------------------------------------------------------------

    def _compute_parameters_from_junctions(self) -> None:
        """Convert SE(3) junction matrices to 6-DOF parameter vectors.

        * ``X{i}`` — via :func:`so3.grh2X` (right-midpoint inverse).
        * ``Y{i}``, ``W{i}``, ``C{i}`` — via :func:`so3.g2X`.

        ``param_mask`` is inherited from the first ``n_dof`` entries of
        ``junction_mask`` (one-to-one correspondence).
        """
        if self._param_names is None:
            if self._sequence is None:
                raise RuntimeError(
                    "Cannot compute parameters: neither param_names nor "
                    "sequence is set."
                )
            self._param_names = cgnaplus_name_assignment(self._sequence)

        param_names = self._param_names
        n_dof  = len(param_names)
        n_snap = self._n_snapshots

        param_array = np.zeros((n_snap, n_dof, 6))
        param_mask  = self._junction_mask[:n_dof].copy()

        for k, pname in enumerate(param_names):
            if not self._junction_mask[k]:
                continue
            letter = pname[0]
            junc_k = self._junction_array[:, k]   # (n_snap, 4, 4)

            if letter == INTRA_BP_PARAM_NAME:   # 'X' — right-midpoint inverse
                for s in range(n_snap):
                    param_array[s, k] = so3.grh2X(junc_k[s])
            else:                               # 'C', 'Y', 'W' — full SE(3) log
                for s in range(n_snap):
                    param_array[s, k] = so3.g2X(junc_k[s])

        self._param_array = param_array
        self._param_mask  = param_mask

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_pose(self, name: str) -> np.ndarray | None:
        """Return the named pose as ``(n_snap, 4, 4)``, or *None*.

        Parameters
        ----------
        name:
            Pose key built from a prefix in :mod:`~naming_conventions` and a
            0-based index, e.g. ``"bp3"``, ``"bw0"``, ``"bc2"``,
            ``"pw1"``, ``"pc0"``.
        """
        self._ensure_poses()
        for prefix, arr, mask in (
            (BP_NAME,               self._bp_poses,               None),
            (WATSON_BASE_NAME,      self._watson_base_poses,       None),
            (CRICK_BASE_NAME,       self._crick_base_poses,        None),
            (WATSON_PHOSPHATE_NAME, self._watson_phosphate_poses,  self._watson_phosphate_mask),
            (CRICK_PHOSPHATE_NAME,  self._crick_phosphate_poses,   self._crick_phosphate_mask),
        ):
            if name.startswith(prefix):
                try:
                    idx = int(name[len(prefix):])
                except ValueError:
                    return None
                if arr is None or idx < 0 or idx >= arr.shape[1]:
                    return None
                if mask is not None and not mask[idx]:
                    return None
                return arr[:, idx]   # (n_snap, 4, 4)
        return None

    def get_junction(self, name: str) -> np.ndarray | None:
        """Return the named SE(3) junction frame as ``(n_snap, 4, 4)``, or *None*.

        Parameters
        ----------
        name:
            e.g. ``"x0"``, ``"y2"``, ``"w1"``, ``"c0"``, ``"r3"``, ``"l1"``.
        """
        self._ensure_junctions()
        if self._junction_names is None:
            return None
        try:
            k = self._junction_names.index(name)
        except ValueError:
            return None
        if not self._junction_mask[k]:
            return None
        return self._junction_array[:, k]   # (n_snap, 4, 4)

    def get_parameter(self, name: str) -> np.ndarray | None:
        """Return the named DOF parameter vector as ``(n_snap, 6)``, or *None*.

        Parameters
        ----------
        name:
            e.g. ``"X0"``, ``"Y2"``, ``"W1"``, ``"C0"``.
        """
        self._ensure_parameters()
        if self._param_names is None:
            return None
        try:
            k = self._param_names.index(name)
        except ValueError:
            return None
        if not self._param_mask[k]:
            return None
        return self._param_array[:, k]   # (n_snap, 6)

    # ------------------------------------------------------------------
    # Name-list queries
    # ------------------------------------------------------------------

    def pose_names(self) -> list[str]:
        """All valid pose names (respects phosphate masks)."""
        self._ensure_poses()
        N     = self._nbp
        wmask = self._watson_phosphate_mask
        cmask = self._crick_phosphate_mask
        return (
            [f"{BP_NAME}{i}"               for i in range(N)]
            + [f"{WATSON_BASE_NAME}{i}"    for i in range(N)]
            + [f"{CRICK_BASE_NAME}{i}"     for i in range(N)]
            + [f"{WATSON_PHOSPHATE_NAME}{i}" for i in range(N)
               if wmask is not None and wmask[i]]
            + [f"{CRICK_PHOSPHATE_NAME}{i}"  for i in range(N)
               if cmask is not None and cmask[i]]
        )

    def junction_names(self) -> list[str]:
        """All valid junction names (respects junction mask)."""
        self._ensure_junctions()
        return [n for n, m in zip(self._junction_names, self._junction_mask) if m]

    def parameter_names(self) -> list[str]:
        """All valid parameter DOF names (respects param mask)."""
        self._ensure_parameters()
        return [n for n, m in zip(self._param_names, self._param_mask) if m]

    # ------------------------------------------------------------------
    # Convenience array properties
    # ------------------------------------------------------------------

    @property
    def sequence(self) -> str | None:
        """Nucleotide sequence string."""
        return self._sequence

    @property
    def params(self) -> "CGNAPlusParams | None":
        """Reference parameter object, or *None*."""
        return self._params

    @property
    def n_snapshots(self) -> int:
        """Number of structural snapshots."""
        return self._n_snapshots

    @property
    def nbp(self) -> int:
        """Number of base pairs."""
        return self._nbp

    @property
    def dynamic(self) -> np.ndarray | None:
        """Dynamic deformation array ``(n_snap, n_dof, 6)``, or *None*."""
        return self._dynamic

    @property
    def bp_poses(self) -> np.ndarray:
        """Base-pair frames, shape ``(n_snap, N, 4, 4)``."""
        self._ensure_poses()
        return self._bp_poses

    @property
    def watson_base_poses(self) -> np.ndarray:
        """Watson strand base frames, shape ``(n_snap, N, 4, 4)``."""
        self._ensure_poses()
        return self._watson_base_poses

    @property
    def crick_base_poses(self) -> np.ndarray:
        """Crick strand base frames, shape ``(n_snap, N, 4, 4)``."""
        self._ensure_poses()
        return self._crick_base_poses

    @property
    def watson_phosphate_poses(self) -> np.ndarray:
        """Watson phosphate frames ``(n_snap, N, 4, 4)``; zeros where absent."""
        self._ensure_poses()
        return self._watson_phosphate_poses

    @property
    def crick_phosphate_poses(self) -> np.ndarray:
        """Crick phosphate frames ``(n_snap, N, 4, 4)``; zeros where absent."""
        self._ensure_poses()
        return self._crick_phosphate_poses

    @property
    def watson_phosphate_mask(self) -> np.ndarray | None:
        """Boolean mask ``(N,)`` — *True* where Watson phosphate exists."""
        self._ensure_poses()
        return self._watson_phosphate_mask

    @property
    def crick_phosphate_mask(self) -> np.ndarray | None:
        """Boolean mask ``(N,)`` — *True* where Crick phosphate exists."""
        self._ensure_poses()
        return self._crick_phosphate_mask

    @property
    def junction_array(self) -> np.ndarray:
        """Full junction array ``(n_snap, n_junc, 4, 4)``; zeros where absent."""
        self._ensure_junctions()
        return self._junction_array

    @property
    def junction_names_list(self) -> list[str] | None:
        """All junction names including absent entries."""
        self._ensure_junctions()
        return self._junction_names

    @property
    def junction_mask(self) -> np.ndarray | None:
        """Boolean mask ``(n_junc,)`` — *True* where junction is valid."""
        self._ensure_junctions()
        return self._junction_mask

    @property
    def param_array(self) -> np.ndarray:
        """Full parameter array ``(n_snap, n_dof, 6)``; zeros where absent."""
        self._ensure_parameters()
        return self._param_array

    @property
    def param_names_list(self) -> list[str] | None:
        """All parameter DOF names including absent entries."""
        return self._param_names

    @property
    def param_mask(self) -> np.ndarray | None:
        """Boolean mask ``(n_dof,)`` — *True* where parameter is valid."""
        self._ensure_parameters()
        return self._param_mask

    # ------------------------------------------------------------------
    # Snapshot slicing and iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_snapshots

    def __iter__(self):
        for i in range(self._n_snapshots):
            yield self.snapshot(i)

    def snapshot(self, i: int) -> "CGNAPlusConf":
        """Return a single-snapshot view for frame *i* (no recomputation).

        Slices all cached arrays along the snapshot axis; the result has
        ``n_snapshots == 1`` and retains the ``(1, ...)`` leading dimension.
        """
        if i < 0 or i >= self._n_snapshots:
            raise IndexError(
                f"snapshot index {i} out of range for "
                f"n_snapshots={self._n_snapshots}."
            )
        obj = CGNAPlusConf.__new__(CGNAPlusConf)
        obj.__dict__.update(self.__dict__)   # shallow-copy all metadata

        def _sl(arr: np.ndarray | None) -> np.ndarray | None:
            return arr[i : i + 1] if arr is not None else None

        obj._raw_params             = _sl(self._raw_params)
        obj._dynamic                = _sl(self._dynamic)
        obj._bp_poses               = _sl(self._bp_poses)
        obj._watson_base_poses      = _sl(self._watson_base_poses)
        obj._crick_base_poses       = _sl(self._crick_base_poses)
        obj._watson_phosphate_poses = _sl(self._watson_phosphate_poses)
        obj._crick_phosphate_poses  = _sl(self._crick_phosphate_poses)
        obj._junction_array         = _sl(self._junction_array)
        obj._param_array            = _sl(self._param_array)
        obj._n_snapshots            = 1
        return obj

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def to_pdb(self, outfn: str, *, snapshot_idx: int = 0) -> None:
        """Write a PDB file for one snapshot.

        Parameters
        ----------
        outfn:
            Output file path.
        snapshot_idx:
            Index of the snapshot to write (default 0).
        """
        from .output.write_pdb import gen_pdb
        self._ensure_poses()
        if self._sequence is None:
            raise ValueError("sequence must be set to write a PDB.")
        gen_pdb(
            outfn=outfn,
            poses=self._bp_poses[snapshot_idx],
            sequence=self._sequence,
        )

    def to_chimerax(self, base_fn: str, *, snapshot_idx: int = 0) -> None:
        """Write ChimeraX BILD + CXC visualisation files for one snapshot.

        Parameters
        ----------
        base_fn:
            Base file path without file extension.
        snapshot_idx:
            Index of the snapshot to write (default 0).
        """
        from .output.visualize_cgnaplus import visualize_cgnaplus
        self._ensure_poses()
        if self._sequence is None:
            raise ValueError("sequence must be set for ChimeraX visualisation.")
        visualize_cgnaplus(
            base_fn,
            seq=self._sequence,
            poses=self._bp_poses[snapshot_idx],
        )

    def __repr__(self) -> str:
        seq_str = f"'{self._sequence}'" if self._sequence is not None else "None"
        return (
            f"CGNAPlusConf(n_snapshots={self._n_snapshots}, "
            f"nbp={self._nbp}, sequence={seq_str})"
        )
