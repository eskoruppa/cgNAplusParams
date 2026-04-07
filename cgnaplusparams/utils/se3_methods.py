from __future__ import annotations

import numpy as np
from .._so3 import so3
from .._pycondec import cond_jit

_se3_euler2rotmat_sv = so3.se3_euler2rotmat
_g2glh_inv   = so3.g2glh_inv
_g2grh       = so3.g2grh
_g2glh       = so3.g2glh
_se3_inverse = so3.se3_inverse

@cond_jit(nopython=True, cache=True)
def _build_chain(p0: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """JIT-compiled sequential SE(3) prefix product for a single snapshot.

    Parameters
    ----------
    p0 : np.ndarray, shape (4, 4)
        Initial pose.
    gs : np.ndarray, shape (n, 4, 4)
        Sequence of junction SE(3) matrices.

    Returns
    -------
    np.ndarray, shape (n + 1, 4, 4)
        ``poses[i] = p0 @ gs[0] @ ... @ gs[i-1]``.
    """
    n = len(gs)
    poses = np.empty((n + 1, 4, 4))
    poses[0] = p0
    for i in range(n):
        poses[i + 1] = poses[i] @ gs[i]
    return poses


@cond_jit(nopython=True, cache=True)
def _build_chain_batched(p0: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """JIT-compiled loop over a flat batch of snapshots.

    Parameters
    ----------
    p0 : np.ndarray, shape (4, 4)
        Initial pose, shared across all snapshots.
    gs : np.ndarray, shape (n_snap, n, 4, 4)
        Flat batch of junction SE(3) matrices.

    Returns
    -------
    np.ndarray, shape (n_snap, n + 1, 4, 4)
    """
    n_snap = gs.shape[0]
    n = gs.shape[1]
    result = np.empty((n_snap, n + 1, 4, 4))
    for i in range(n_snap):
        result[i] = _build_chain(p0, gs[i])
    return result


def build_chain(p0: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Sequential SE(3) prefix product with support for batched snapshots.

    Batch dimensions in ``gs`` are flattened before the JIT kernel and
    restored in the output, matching the pattern used by ``build_junctions``.

    Parameters
    ----------
    p0 : np.ndarray, shape (4, 4)
        Initial pose, shared across all snapshots.
    gs : np.ndarray, shape (n, 4, 4) or (..., n, 4, 4)
        Junction SE(3) matrices. Leading dimensions beyond the first are
        treated as batch axes.

    Returns
    -------
    np.ndarray
        Shape ``(n + 1, 4, 4)`` for a single snapshot or
        ``(..., n + 1, 4, 4)`` matching the batch dimensions of ``gs``.
    """
    if gs.shape[-2:] != (4, 4):
        raise ValueError(f"Expected gs to have shape (..., n, 4, 4), got {gs.shape}")
    if gs.ndim < 3:
        raise ValueError(f"Expected gs to have at least 3 dimensions (n, 4, 4), got {gs.shape}")
    if gs.ndim > 3:
        batch_shape = gs.shape[:-3]
        n = gs.shape[-3]
        flat = gs.reshape(-1, n, 4, 4)
        result = _build_chain_batched(p0, flat)
        return result.reshape(*batch_shape, n + 1, 4, 4)
    return _build_chain(p0, gs)


def params_to_chain(
        p0: np.ndarray,
        groundstate: np.ndarray,
        excess: np.ndarray | None = None,
        group_split: bool = True
) -> np.ndarray:
    """Build a chain of SE(3) poses from ground-state and optional excess parameters.

    Delegates to ``build_junctions`` (which handles arbitrary batch dimensions
    in ``excess``) and then ``build_chain`` (which handles the resulting batch
    dimensions in the junction matrices).

    Parameters
    ----------
    p0 : np.ndarray, shape (4, 4)
        Initial pose.
    groundstate : np.ndarray, shape (ndof, 6)
        Ground-state Euler parameters.
    excess : np.ndarray, shape (..., ndof, 6), optional
        Excess parameters with optional leading batch dimensions.
    group_split : bool
        If True, combine at the group level. If False, add in the algebra.

    Returns
    -------
    np.ndarray
        Shape ``(ndof + 1, 4, 4)`` for a single snapshot or
        ``(..., ndof + 1, 4, 4)`` matching the batch dimensions of ``excess``.
    """
    junction = build_junctions(groundstate, excess, group_split)
    return build_chain(p0, junction)


@cond_jit(nopython=True, cache=True)
def _build_junctions(
        groundstate: np.ndarray,
        excess: np.ndarray | None = None,
        group_split: bool = True
    ) -> np.ndarray:
    """Build a junction SE(3) matrix for a single snapshot.

    Parameters
    ----------
    groundstate : np.ndarray, shape (ndof, 6)
        Ground-state Euler parameters.
    excess : np.ndarray, shape (ndof, 6), optional
        Excess (displacement) parameters for a single snapshot.
    group_split : bool
        If True, combine groundstate and excess at the group level
        (``R_gs @ R_excess``). If False, add in the algebra first.

    Returns
    -------
    np.ndarray, shape (ndof, 4, 4)
    """
    ndof = groundstate.shape[0]
    result = np.empty((ndof, 4, 4))
    if excess is not None:
        if group_split:
            for k in range(ndof):
                result[k] = _se3_euler2rotmat_sv(groundstate[k]) @ _se3_euler2rotmat_sv(excess[k])
        else:
            for k in range(ndof):
                result[k] = _se3_euler2rotmat_sv(groundstate[k] + excess[k])
    else:
        for k in range(ndof):
            result[k] = _se3_euler2rotmat_sv(groundstate[k])
    return result


@cond_jit(nopython=True, cache=True)
def _build_junctions_batched(
        groundstate: np.ndarray,
        excess: np.ndarray,
        group_split: bool = True
    ) -> np.ndarray:
    """JIT-compiled loop over a flat batch of snapshots.

    Parameters
    ----------
    groundstate : np.ndarray, shape (ndof, 6)
        Ground-state Euler parameters.
    excess : np.ndarray, shape (n_snap, ndof, 6)
        Excess parameters; the first dimension is the flat snapshot index.
    group_split : bool
        Passed through to ``_build_junctions``.

    Returns
    -------
    np.ndarray, shape (n_snap, ndof, 4, 4)
    """
    n_snap = excess.shape[0]
    ndof = groundstate.shape[0]
    result = np.empty((n_snap, ndof, 4, 4))
    for i in range(n_snap):
        result[i] = _build_junctions(groundstate, excess[i], group_split)
    return result


def build_junctions(
        groundstate: np.ndarray,
        excess: np.ndarray | None = None,
        group_split: bool = True
    ) -> np.ndarray:
    """Build junction SE(3) matrices from ground-state and optional excess parameters.

    Supports an arbitrary number of batch dimensions in ``excess``: the batch
    axes are flattened before handing off to the JIT-compiled kernel and
    restored in the output.

    Parameters
    ----------
    groundstate : np.ndarray, shape (ndof, 6)
        Ground-state Euler parameters.
    excess : np.ndarray, shape (..., ndof, 6), optional
        Excess parameters. May have any number of leading batch dimensions.
        If ``None``, the ground-state junction is returned as a single
        snapshot.
    group_split : bool
        If True, combine at the group level (``R_gs @ R_excess``).
        If False, add parameters in the algebra before converting.

    Returns
    -------
    np.ndarray
        Shape ``(ndof, 4, 4)`` when ``excess`` is ``None`` or 2-D, or
        ``(..., ndof, 4, 4)`` matching the batch dimensions of ``excess``.
    """
    if groundstate.shape[-1] != 6:
        raise ValueError(f"Expected groundstate to have shape (..., 6), got {groundstate.shape}")
    if groundstate.ndim != 2:
        raise ValueError(f"Expected groundstate to have exactly 2 dimensions, got {groundstate.shape}")
    if excess is not None:
        if excess.shape[-1] != 6:
            raise ValueError(f"Expected excess to have shape (..., 6), got {excess.shape}")
        if excess.ndim < 2:
            raise ValueError(f"Expected excess to have at least 2 dimensions, got {excess.shape}")
        if excess.shape[-2:] != groundstate.shape:
            raise ValueError(f"Expected excess last two dims {excess.shape[-2:]} to match groundstate shape {groundstate.shape}")
        if excess.ndim > 2:
            batch_shape = excess.shape[:-2]
            flat = excess.reshape(-1, *groundstate.shape)
            result = _build_junctions_batched(groundstate, flat, group_split)
            return result.reshape(*batch_shape, groundstate.shape[0], 4, 4)

    return _build_junctions(groundstate, excess, group_split)


@cond_jit(nopython=True, cache=True)
def build_cgnaplus_poses(
        bp_poses: np.ndarray,
        bp_juncs: np.ndarray,
        w_juncs: np.ndarray,
        c_juncs: np.ndarray,
) -> tuple:
    """JIT-compiled computation of all site SE(3) poses from pre-built junction matrices.

    Parameters
    ----------
    bp_poses : np.ndarray, shape (nsnap, nbp, 4, 4)
        Base-pair frame poses from the inter-bp step chain.
    bp_juncs : np.ndarray, shape (nsnap, nbp, 4, 4)
        Intra-bp junction matrices, one per base pair.
    w_juncs : np.ndarray, shape (nsnap, nbp, 4, 4)
        Watson-phosphate junction matrices padded with a dummy at index 0
        (no Watson phosphate at the 5' end).
    c_juncs : np.ndarray, shape (nsnap, nbp, 4, 4)
        Crick-phosphate junction matrices padded with a dummy at index
        ``nbp - 1`` (no Crick phosphate at the 3' end).

    Returns
    -------
    tuple of four np.ndarray, each shape (nsnap, nbp, 4, 4)
        ``(watson_base_poses, crick_base_poses,
           watson_phosphate_poses, crick_phosphate_poses)``
    """
    nsnap = bp_poses.shape[0]
    nbp   = bp_poses.shape[1]
    watson_base_poses      = np.zeros((nsnap, nbp, 4, 4))
    crick_base_poses       = np.zeros((nsnap, nbp, 4, 4))
    watson_phosphate_poses = np.zeros((nsnap, nbp, 4, 4))
    crick_phosphate_poses  = np.zeros((nsnap, nbp, 4, 4))
    for s in range(nsnap):
        for i in range(nbp):
            bp_glh_inv = _g2glh_inv(bp_juncs[s, i])
            bp_grh     = _g2grh(bp_juncs[s, i])
            wb_pose = bp_poses[s, i] @ bp_grh
            cb_pose = bp_poses[s, i] @ bp_glh_inv
            watson_base_poses[s, i] = wb_pose
            crick_base_poses[s, i]  = cb_pose
            if i > 0:
                watson_phosphate_poses[s, i] = wb_pose @ w_juncs[s, i]
            if i < nbp - 1:
                crick_phosphate_poses[s, i]  = cb_pose @ c_juncs[s, i]
    return watson_base_poses, crick_base_poses, watson_phosphate_poses, crick_phosphate_poses


def bases_to_bp_poses(
        watson_base_poses: np.ndarray,
        crick_base_poses: np.ndarray,
) -> np.ndarray:
    if watson_base_poses.shape != crick_base_poses.shape:
        raise ValueError(f"Expected watson_base_poses and crick_base_poses to have the same shape, got {watson_base_poses.shape} and {crick_base_poses.shape}")
    if watson_base_poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected watson_base_poses and crick_base_poses to have shape (..., 4, 4), got {watson_base_poses.shape} and {crick_base_poses.shape}")
    if watson_base_poses.ndim < 3:
        raise ValueError(f"Expected watson_base_poses and crick_base_poses to have at least 3 dimensions (..., nbp, 4, 4), got {watson_base_poses.shape} and {crick_base_poses.shape}")
    if watson_base_poses.ndim > 4:
        raise ValueError(f"Expected watson_base_poses and crick_base_poses to have at most 4 dimensions (..., nbp, 4, 4), got {watson_base_poses.shape} and {crick_base_poses.shape}")
    if watson_base_poses.ndim == 3:
        watson_base_poses = watson_base_poses[np.newaxis]
        crick_base_poses = crick_base_poses[np.newaxis]
    return _bases_to_bp_poses(watson_base_poses, crick_base_poses)


@cond_jit(nopython=True, cache=True)
def _bases_to_bp_poses(
        watson_base_poses: np.ndarray,
        crick_base_poses: np.ndarray,
) -> np.ndarray:
    """JIT-compiled computation of base-pair poses from Watson and Crick base poses.

    Parameters
    ----------
    watson_base_poses : np.ndarray, shape (nsnap, nbp, 4, 4)
        Watson strand base frame poses.
    crick_base_poses : np.ndarray, shape (nsnap, nbp, 4, 4)
        Crick strand base frame poses.

    Returns
    -------
    np.ndarray, shape (nsnap, nbp, 4, 4)
    """
    nsnap = watson_base_poses.shape[0]
    nbp   = watson_base_poses.shape[1]
    bp_poses = np.zeros((nsnap, nbp, 4, 4))
    for s in range(nsnap):
        for i in range(nbp):
            bp_poses[s, i] = crick_base_poses[s, i] @ _g2glh(_se3_inverse(crick_base_poses[s, i]) @ watson_base_poses[s, i])
    return bp_poses


def poses_to_juncs(
        poses1: np.ndarray,
        poses2: np.ndarray | None = None
) -> np.ndarray:
    
    """JIT-compiled computation of junction matrices from site poses.

    Parameters
    ----------
    poses : np.ndarray, shape (nsnap, nbp, 4, 4)
        Site poses.

    Returns
    -------
    np.ndarray, shape (nsnap, nbp - 1, 4, 4)
        Junction matrices between adjacent sites.
    """
    if poses1.shape[-2:] != (4, 4):
        raise ValueError(f"Expected poses to have shape (..., 4, 4), got {poses1.shape} and {poses2.shape}")
    if poses1.ndim < 3:
        raise ValueError(f"Expected poses to have at least 3 dimensions (..., nbp, 4, 4), got {poses1.shape} and {poses2.shape}")
    if poses1.ndim > 4:
        raise ValueError(f"Expected poses to have at most 4 dimensions (..., nbp, 4, 4), got {poses1.shape} and {poses2.shape}")

    if poses2 is None:
        poses2 = poses1[..., 1: , :, :]
        poses1 = poses1[..., :-1, :, :]
    else:
        if poses1.shape != poses2.shape:
            raise ValueError(f"Expected poses1 and poses2 to have the same shape, got {poses1.shape} and {poses2.shape}")

    if poses1.ndim == 3:
        poses1 = poses1[np.newaxis]
        poses2 = poses2[np.newaxis]

    return _poses_to_juncs(poses1, poses2)

@cond_jit(nopython=True, cache=True)
def _poses_to_juncs(
        poses1: np.ndarray,
        poses2: np.ndarray,
) -> np.ndarray:
    
    """JIT-compiled computation of junction matrices from site poses.

    Parameters
    ----------
    poses : np.ndarray, shape (nsnap, nbp, 4, 4)
        Site poses.

    Returns
    -------
    np.ndarray, shape (nsnap, nbp - 1, 4, 4)
        Junction matrices between adjacent sites.
    """
    nsnap = poses1.shape[0]
    nbps  = poses1.shape[1]
    juncs = np.zeros((nsnap, nbps, 4, 4))
    for s in range(nsnap):
        for i in range(nbps):
            juncs[s, i] = _se3_inverse(poses1[s, i]) @ poses2[s, i]
    return juncs


def juncs_to_params(juncs: np.ndarray) -> np.ndarray:
    """Compute Euler parameters from junction SE(3) matrices.

    Parameters
    ----------
    juncs : np.ndarray, shape (..., n, 4, 4)
        Junction SE(3) matrices with optional leading batch dimensions.

    Returns
    -------
    np.ndarray, shape (..., n, 6)
        Corresponding Euler parameters with the same leading batch dimensions.
    """
    if juncs.shape[-2:] != (4, 4):
        raise ValueError(f"Expected juncs to have shape (..., n, 4, 4), got {juncs.shape}")
    if juncs.ndim < 3:
        raise ValueError(f"Expected juncs to have at least 3 dimensions (..., n, 4, 4), got {juncs.shape}")
    return so3.se3_rotmat2euler_batch(juncs)    
