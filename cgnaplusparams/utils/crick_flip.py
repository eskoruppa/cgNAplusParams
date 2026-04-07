#!/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, issparse

from .assignment_utils import cgnaplus_name_assignment, nonphosphate_dof_map
from .assignment_utils import crick_phosphate_dof_indices


def apply_crick_flip(
    gs: np.ndarray,
    stiff: csc_matrix | np.ndarray | None,
    param_names: list[str],
) -> tuple[np.ndarray, csc_matrix | np.ndarray | None]:
    """Flip the orientation of Crick base-to-phosphate junction coordinates.

    The Crick-strand base and phosphate frames are conjugated by
    ``f = diag(1, −1, −1, 1)`` (i.e. ``F = diag(1,−1,−1) ∈ SO(3)``
    applied to both the rotation and translation sub-vectors).  Within each
    C-type 6-DOF block this negates coordinates 1, 2, 4, 5 (coordinates 0
    and 3 are kept).  For the ground state this is a direct negation; for the
    stiffness the block-diagonal congruence ``K' = S K S`` is applied, where
    ``S`` has ``diag(1,−1,−1,1,−1,−1)`` in each Crick-DOF block and the
    identity in all other blocks.

    For a CSC sparse stiffness the congruence is computed via an O(nnz)
    elementwise multiply on the data array – no index copies required.
    For a dense ndarray stiffness it is computed as ``s[:, None] * stiff * s``.

    Parameters
    ----------
    gs:
        Raw Cayley ground-state vector of length N.
    stiff:
        Stiffness matrix as a CSC sparse matrix, a dense ``np.ndarray``, or
        ``None`` when stiffness is not needed.
    param_names:
        List of parameter names.

    Returns
    -------
    gs_flipped, stiff_flipped
    """
    crick_idx = crick_phosphate_dof_indices(param_names)

    flip_within = np.array([1, 2, 4, 5])  # offsets inside each 6-DOF block
    raw_flip = (crick_idx[:, None] * 6 + flip_within).ravel()

    gs_flipped = gs.copy()
    gs_flipped[raw_flip] *= -1.0

    if stiff is None:
        return gs_flipped, None

    N = len(gs)
    s = np.ones(N, dtype=np.float64)
    s[raw_flip] = -1.0

    if issparse(stiff):
        # Congruence S K S on CSC data: multiply each stored value by s[row]*s[col]
        stiff_data = stiff.data * s[stiff.indices] * np.repeat(s, np.diff(stiff.indptr))
        stiff_flipped = csc_matrix(
            (stiff_data, stiff.indices, stiff.indptr),
            shape=stiff.shape,
            copy=False,
        )
    else:
        # Dense congruence S K S via broadcasting
        stiff_flipped = s[:, None] * stiff * s

    return gs_flipped, stiff_flipped
