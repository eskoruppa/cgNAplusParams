from __future__ import annotations
import numpy as np

from .._so3 import so3
from .._pycondec import cond_jit
from .crick_flip import apply_crick_flip
from .assignment_utils import crick_phosphate_dof_indices

_CGNAPLUS_BANDWIDTH = 41  # max |i-j| in assembled stiffness matrix

# ──────────────────────────────────────────────────────────────────────
# JIT-compatible function aliases for _apply_transforms_optimized
# ──────────────────────────────────────────────────────────────────────
_euler2cayley_linearexpansion = so3.euler2cayley_linearexpansion  # (3,) → (3,3)
_cayley2euler_linearexpansion = so3.cayley2euler_linearexpansion  # (3,) → (3,3)
_euler2rotmat = so3.euler2rotmat                                  # (3,) → (3,3)
_cayley2euler = so3.cayley2euler                                  # (3,) → (3,)
_hat_map = so3._hat_map_sv                                        # (3,) → (3,3)
_inverse_right_jacobian = so3.inverse_right_jacobian              # (3,) → (3,3)
_right_jacobian = so3.right_jacobian                              # (3,) → (3,3)


def _apply_transforms(
    gs: np.ndarray,
    stiff,
    nonphosphate_map: list[bool] | bool,
    param_names: list[str],
    remove_factor_five: bool,
    translations_in_nm: bool,
    euler_definition: bool,
    group_split: bool,
    include_stiffness: bool,
    aligned_strands: bool,
) -> tuple[np.ndarray, object]:
    """Apply all coordinate-definition transforms to a raw Cayley/Ångström
    ground-state vector and stiffness matrix.

    Parameters
    ----------
    gs : flat state vector of length 6*N (Cayley, Ångström)
    stiff : stiffness matrix (sparse or dense); ignored when include_stiffness
        is False
    nonphosphate_map : bool list of length N; True where
        translation_as_midstep applies (X- and Y-type DOFs)
    param_names : DOF name list; required for the aligned_strands crick-flip
    remove_factor_five, translations_in_nm, euler_definition, group_split,
        include_stiffness, aligned_strands : transform flags

    Returns
    -------
    gs_vecs : ndarray, shape (N, 6)
    stiff : transformed stiffness (same type as input), or original if
        include_stiffness is False
    """

    if isinstance(nonphosphate_map, bool):
        nonphosphate_map = [nonphosphate_map] * (len(gs) // 6)

    if remove_factor_five:
        factor = 5
        gs = so3.array_conversion(gs, 1. / factor, block_dim=6, dofs=[0, 1, 2])
        if include_stiffness:
            stiff = so3.array_conversion(stiff, factor, block_dim=6, dofs=[0, 1, 2])
    if translations_in_nm:
        factor = 10
        gs = so3.array_conversion(gs, 1. / factor, block_dim=6, dofs=[3, 4, 5])
        if include_stiffness:
            stiff = so3.array_conversion(stiff, factor, block_dim=6, dofs=[3, 4, 5])

    gs = so3.statevec2vecs(gs, vdim=6)

    if euler_definition:
        # cayley2euler_stiffmat requires gs in cayley definition
        if include_stiffness:
            stiff = so3.se3_cayley2euler_stiffmat(gs, stiff, rotation_first=True)
        gs = so3.se3_cayley2euler(gs)

    if group_split:
        if not euler_definition:
            raise ValueError('The group_split option requires euler_definition to be set!')
        if include_stiffness:
            gs, stiff = so3.algebra2group_params(
                gs, stiff, rotation_first=True,
                translation_as_midstep=nonphosphate_map, optimized=True,
            )
        else:
            for i in range(len(gs)):
                if nonphosphate_map[i]:
                    gs[i] = so3.midstep2triad(gs[i], rotation_first=True)

    if aligned_strands:
        gs = so3.vecs2statevec(gs)
        gs, stiff = apply_crick_flip(
            gs,
            stiff if include_stiffness else None,
            param_names,
        )
        gs = so3.statevec2vecs(gs, vdim=6)

    return gs, stiff


# ──────────────────────────────────────────────────────────────────────
# Optimized _apply_transforms
# ──────────────────────────────────────────────────────────────────────
#
# The original implementation builds full dense (6N × 6N) Jacobian matrices
# for the Cayley→Euler and algebra→group stiffness transforms, then performs
# two O((6N)³) dense matrix multiplications.  Both Jacobians are actually
# block-diagonal with 6×6 blocks.
#
# This optimized version:
#   1. Keeps the cheap unit-scaling via array_conversion (~7 ms total).
#   2. Replaces the two expensive stiffness transforms (Cayley→Euler and
#      algebra→group) with a single block-wise congruence on 6×6 blocks.
#
# Complexity drops from O(N³) to O(N · b · 6³) where b is the bandwidth in
# blocks (~15), giving a >95 % wall-clock reduction for typical sequences.
# ──────────────────────────────────────────────────────────────────────


@cond_jit(nopython=True, cache=True)
def _compute_stiff_jacobian_blocks(
    gs_cayley_scaled: np.ndarray,
    nonphosphate_map: np.ndarray,
    do_group_split: bool,
) -> np.ndarray:
    """Compute per-DOF 6×6 Jacobian blocks for the combined stiffness
    congruence: K_final = J^T @ K_scaled @ J.

    The Jacobian combines:
      - T_c2e_inv : Euler→Cayley linearization (se3_euler2cayley_lintrans)
      - HX_inv    : group→algebra linearization (when group_split is True)

    Parameters
    ----------
    gs_cayley_scaled : (N, 6) — groundstate in Cayley coordinates, after unit
        scaling has already been applied.
    nonphosphate_map : (N,) bool — True where midstep translation applies.
    do_group_split : whether to include the algebra→group transform.

    Returns
    -------
    J_all : (N, 6, 6)
    """
    N = gs_cayley_scaled.shape[0]
    J_all = np.empty((N, 6, 6))

    for i in range(N):
        # Convert Cayley rotation → Euler rotation
        cayley_rot = gs_cayley_scaled[i, :3]
        euler_rot = _cayley2euler(cayley_rot)

        # T_c2e_inv block: [[E2C, 0], [0, I_3]]
        E2C = _euler2cayley_linearexpansion(euler_rot)  # 3×3

        if not do_group_split:
            J = np.eye(6)
            for r in range(3):
                for c in range(3):
                    J[r, c] = E2C[r, c]
        else:
            Phi = euler_rot
            zeta_0 = gs_cayley_scaled[i, 3:6]  # algebra translation (unchanged by c2e)

            H_inv = _inverse_right_jacobian(Phi)            # 3×3
            E2C_H_inv = E2C @ H_inv                          # 3×3

            if nonphosphate_map[i]:
                sqrtS = _euler2rotmat(0.5 * Phi)             # R(Ω/2)
                zeta_hat_neg = _hat_map(-zeta_0)             # hat(-ζ)
                H_half = _right_jacobian(0.5 * Phi)          # J_r(Ω/2)

                # HX_inv = [[H_inv, 0],
                #           [Q,     sqrtS]]
                # where Q = -0.5 * hat(-ζ) @ H_half @ H_inv
                Q = -0.5 * (zeta_hat_neg @ (H_half @ H_inv))

                # J = T_c2e_inv @ HX_inv
                #   = [[E2C, 0], [0, I]] @ [[H_inv, 0], [Q, sqrtS]]
                #   = [[E2C @ H_inv, 0], [Q, sqrtS]]
                J = np.zeros((6, 6))
                for r in range(3):
                    for c in range(3):
                        J[r, c] = E2C_H_inv[r, c]
                        J[r + 3, c] = Q[r, c]
                        J[r + 3, c + 3] = sqrtS[r, c]
            else:
                R_phi = _euler2rotmat(Phi)                   # R(Ω)

                # J = [[E2C @ H_inv, 0], [0, R(Ω)]]
                J = np.zeros((6, 6))
                for r in range(3):
                    for c in range(3):
                        J[r, c] = E2C_H_inv[r, c]
                        J[r + 3, c + 3] = R_phi[r, c]

        J_all[i] = J

    return J_all


@cond_jit(nopython=True, cache=True)
def _apply_jacobian_blocks_to_stiff(
    stiff_dense: np.ndarray,
    J_all: np.ndarray,
    bandwidth: int,
) -> np.ndarray:
    """Apply block-diagonal congruence K' = J^T K J to a banded stiffness
    matrix, operating on 6×6 blocks.

    Parameters
    ----------
    stiff_dense : (6N, 6N) dense array (only entries within `bandwidth` of
        the diagonal are assumed nonzero)
    J_all : (N, 6, 6) per-DOF Jacobian blocks
    bandwidth : element-level half-bandwidth of the stiffness

    Returns
    -------
    result : (6N, 6N) dense array
    """
    N = J_all.shape[0]
    dim = 6 * N
    result = np.zeros((dim, dim))

    # Maximum block-index distance that can overlap the band
    max_bd = (bandwidth + 5) // 6  # ceil((bw+5)/6)

    for bi in range(N):
        Ji_T = J_all[bi].T  # 6×6
        ri = 6 * bi
        bj_lo = bi - max_bd
        if bj_lo < 0:
            bj_lo = 0
        bj_hi = bi + max_bd
        if bj_hi >= N:
            bj_hi = N - 1

        for bj in range(bj_lo, bj_hi + 1):
            rj = 6 * bj
            # Extract 6×6 block K[ri:ri+6, rj:rj+6]
            K_block = stiff_dense[ri:ri + 6, rj:rj + 6]

            # Check if block is zero (skip)
            nonzero = False
            for p in range(6):
                for q in range(6):
                    if K_block[p, q] != 0.0:
                        nonzero = True
                        break
                if nonzero:
                    break
            if not nonzero:
                continue

            Jj = J_all[bj]  # 6×6

            # Compute Ji^T @ K_block @ Jj
            # tmp = K_block @ Jj  (6×6 @ 6×6)
            tmp = np.empty((6, 6))
            for r in range(6):
                for c in range(6):
                    s = 0.0
                    for m in range(6):
                        s += K_block[r, m] * Jj[m, c]
                    tmp[r, c] = s

            # result_block = Ji_T @ tmp  (6×6 @ 6×6)
            for r in range(6):
                for c in range(6):
                    s = 0.0
                    for m in range(6):
                        s += Ji_T[r, m] * tmp[m, c]
                    result[ri + r, rj + c] = s

    return result


@cond_jit(nopython=True, cache=True)
def _transform_groundstate_optimized(
    gs_cayley_scaled: np.ndarray,
    nonphosphate_map: np.ndarray,
    do_group_split: bool,
) -> np.ndarray:
    """Transform groundstate: Cayley→Euler + midstep→triad.

    Input is the already-scaled groundstate in Cayley definition.
    """
    N = gs_cayley_scaled.shape[0]
    gs_out = np.empty((N, 6))

    for i in range(N):
        # Cayley→Euler (rotation only, translation unchanged)
        euler_rot = _cayley2euler(gs_cayley_scaled[i, :3])
        for k in range(3):
            gs_out[i, k] = euler_rot[k]
        for k in range(3, 6):
            gs_out[i, k] = gs_cayley_scaled[i, k]

        if do_group_split and nonphosphate_map[i]:
            # midstep→triad: s = R(Ω/2) @ ζ
            sqrtR = _euler2rotmat(0.5 * euler_rot)
            t0 = gs_out[i, 3]
            t1 = gs_out[i, 4]
            t2 = gs_out[i, 5]
            gs_out[i, 3] = sqrtR[0, 0] * t0 + sqrtR[0, 1] * t1 + sqrtR[0, 2] * t2
            gs_out[i, 4] = sqrtR[1, 0] * t0 + sqrtR[1, 1] * t1 + sqrtR[1, 2] * t2
            gs_out[i, 5] = sqrtR[2, 0] * t0 + sqrtR[2, 1] * t1 + sqrtR[2, 2] * t2

    return gs_out


def _apply_transforms_optimized(
    gs: np.ndarray,
    stiff,
    nonphosphate_map: list[bool] | bool,
    param_names: list[str],
    remove_factor_five: bool,
    translations_in_nm: bool,
    euler_definition: bool,
    group_split: bool,
    include_stiffness: bool,
    aligned_strands: bool,
) -> tuple[np.ndarray, object]:
    """Drop-in replacement for ``_apply_transforms`` that is much faster when
    ``include_stiffness=True`` and ``euler_definition=True``.

    Instead of building full (6N × 6N) dense Jacobians and performing dense
    matrix multiplications for the Cayley→Euler and algebra→group transforms,
    this version computes per-DOF 6×6 Jacobian blocks and applies the
    congruence transform block-by-block on the banded stiffness matrix.

    Unit scaling and crick-flip sign changes are folded into the Jacobian
    blocks so that the congruence acts directly on the raw stiffness matrix,
    eliminating the O(dim²) dense crick-flip congruence entirely.

    Parameters and return value are identical to ``_apply_transforms``.
    """

    if isinstance(nonphosphate_map, bool):
        nonphosphate_map = [nonphosphate_map] * (len(gs) // 6)

    if group_split and not euler_definition:
        raise ValueError('The group_split option requires euler_definition to be set!')

    # ── Unit scaling on groundstate (cheap, O(N)) ──
    if remove_factor_five:
        gs = so3.array_conversion(gs, 1.0 / 5, block_dim=6, dofs=[0, 1, 2])
    if translations_in_nm:
        gs = so3.array_conversion(gs, 1.0 / 10, block_dim=6, dofs=[3, 4, 5])

    gs = so3.statevec2vecs(gs, vdim=6)  # (N, 6)

    # ── Fast path: block-wise stiffness congruence ───────────────────
    if include_stiffness and euler_definition:
        nonphosphate_arr = np.asarray(nonphosphate_map, dtype=np.bool_)

        # Compute per-DOF combined Jacobian blocks J = T_c2e_inv @ HX_inv
        J_all = _compute_stiff_jacobian_blocks(
            gs, nonphosphate_arr, do_group_split=group_split,
        )

        # Fold stiffness unit-scaling into Jacobian rows: D @ J
        # This avoids scaling the sparse stiffness matrix separately.
        if remove_factor_five or translations_in_nm:
            d = np.ones(6, dtype=np.float64)
            if remove_factor_five:
                d[:3] = 5.0
            if translations_in_nm:
                d[3:] = 10.0
            J_all = J_all * d[np.newaxis, :, np.newaxis]

        # Fold crick-flip signs into Jacobian columns: J_combined @ S
        # This avoids the O(dim²) dense crick-flip congruence.
        crick_flip_idx = np.empty(0, dtype=int)
        if aligned_strands:
            crick_flip_idx = crick_phosphate_dof_indices(param_names)
            if len(crick_flip_idx) > 0:
                crick_sign = np.array([1., -1., -1., 1., -1., -1.])
                J_all[crick_flip_idx] = (
                    J_all[crick_flip_idx] * crick_sign[np.newaxis, np.newaxis, :]
                )

        # Convert raw sparse stiffness → dense (no unit scaling needed)
        if hasattr(stiff, 'toarray'):
            stiff_dense = stiff.toarray()
        else:
            stiff_dense = np.asarray(stiff, dtype=np.float64)

        # K_final = J_combined^T @ K_raw @ J_combined  (block-by-block)
        stiff = _apply_jacobian_blocks_to_stiff(
            stiff_dense, J_all, _CGNAPLUS_BANDWIDTH,
        )

        # Transform groundstate: Cayley→Euler + midstep→triad
        gs = _transform_groundstate_optimized(
            gs, nonphosphate_arr, do_group_split=group_split,
        )

        # Apply crick-flip to groundstate only (O(N), trivial)
        if len(crick_flip_idx) > 0:
            gs[np.ix_(crick_flip_idx, np.array([1, 2, 4, 5]))] *= -1.0

    else:
        # ── Standard path (no euler stiffness or stiffness not needed) ─
        if include_stiffness:
            if remove_factor_five:
                stiff = so3.array_conversion(stiff, 5, block_dim=6, dofs=[0, 1, 2])
            if translations_in_nm:
                stiff = so3.array_conversion(stiff, 10, block_dim=6, dofs=[3, 4, 5])

        if euler_definition:
            gs = so3.se3_cayley2euler(gs)
            if group_split:
                for i in range(len(gs)):
                    if nonphosphate_map[i]:
                        gs[i] = so3.midstep2triad(gs[i], rotation_first=True)

        if aligned_strands:
            gs = so3.vecs2statevec(gs)
            gs, stiff = apply_crick_flip(
                gs,
                stiff if include_stiffness else None,
                param_names,
            )
            gs = so3.statevec2vecs(gs, vdim=6)

    return gs, stiff