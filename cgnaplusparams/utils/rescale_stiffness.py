from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
from collections.abc import Sequence
        

def _apply_congruence(stiff: np.ndarray | spmatrix, d: np.ndarray) -> np.ndarray | spmatrix:
    """Return ``D @ stiff @ D`` for the diagonal ``D = diag(d)``, preserving the input type."""
    if sp.sparse.issparse(stiff):
        fmt = getattr(stiff, "format", "csr") or "csr"
        D = sp.sparse.diags(d, format=fmt)
        return D @ stiff @ D
    stiff = np.asarray(stiff)
    return (d[:, None] * stiff) * d[None, :]

def _check_matrix(stiff: np.ndarray | spmatrix, ndims: int) -> int:
    """Validate that ``stiff`` is a square 2D matrix whose size is a multiple of ``ndims``."""
    if not hasattr(stiff, "shape") or len(stiff.shape) != 2:
        raise TypeError("stiff must be a 2D numpy array or a scipy sparse matrix.")
    n, m = stiff.shape
    if n != m:
        raise ValueError(f"stiff must be square, got shape {stiff.shape}.")
    if n == 0:
        raise ValueError("stiff must be non-empty.")
    if n % ndims != 0:
        raise ValueError(f"stiff dimension must be a multiple of {ndims}, got {n}.")
    return n


def rescale_kth(stiff: np.ndarray | spmatrix, k: int, factor: float) -> np.ndarray | spmatrix:
    """
    Rescale the k-th degree of freedom by scaling the corresponding rows and columns.

    This method rescales the stiffness matrix by multiplying all rows and columns
    associated with the k-th degree of freedom (k in 0..5) by ``sqrt(factor)``.
    Equivalently, it applies a similarity transformation

        K -> D K D

    where ``D`` is diagonal with entries equal to ``sqrt(factor)`` for the k-th
    degree of freedom and unity elsewhere.

    As a consequence:
    - diagonal stiffness entries associated with the k-th degree of freedom are
      multiplied by ``factor``,
    - off-diagonal entries coupling the k-th degree of freedom to others are
      multiplied by ``sqrt(factor)``,
    - all other entries are unchanged.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix with dimension a multiple of 6.
    k : int
        Degree-of-freedom index in the range 0..5.
    factor : float
        Positive rescaling factor for the diagonal stiffness entries of the
        selected degree of freedom.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix (same type as input).
    """

    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if not (0 <= k < 6):
        raise ValueError("k must be an integer in the range 0..5.")

    if not isinstance(factor, (int, float, np.number)):
        raise TypeError("factor must be a real number.")
    
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("factor must be finite (not inf/NaN).")
    if factor <= 0:
        raise ValueError(f'factor must be a positive number, encountered factor = {factor}')

    if not hasattr(stiff, "shape") or len(stiff.shape) != 2:
        raise TypeError("stiff must be a 2D numpy array or a scipy sparse matrix.")

    n, m = stiff.shape
    if n != m:
        raise ValueError(f"stiff must be square, got shape {stiff.shape}.")
    if n == 0:
        raise ValueError("stiff must be non-empty.")
    if n % 6 != 0:
        raise ValueError(f"stiff dimension must be a multiple of 6, got {n}.")

    sqfac = np.sqrt(factor)

    d = np.ones(n, dtype=float)
    d[k::6] = sqfac

    if sp.sparse.issparse(stiff):
        fmt = getattr(stiff, "format", "csr") or "csr"
        D = sp.sparse.diags(d, format=fmt)
        return D @ stiff @ D
    else:
        stiff = np.asarray(stiff)
        return (d[:, None] * stiff) * d[None, :]
    

def rescale_stiff(
    stiff: np.ndarray | spmatrix,
    factor: float,
    entries: Sequence[int] | None = None,
) -> np.ndarray | spmatrix:
    """
    Rescale selected degrees of freedom by scaling corresponding rows and columns.

    This method rescales the stiffness matrix by multiplying rows and columns
    associated with selected degrees of freedom by ``sqrt(factor)``. Internally,
    it applies a similarity transformation

        K -> D K D

    where ``D`` is diagonal with entries equal to ``sqrt(factor)`` for the selected
    degrees of freedom and unity elsewhere.

    As a result:
    - diagonal stiffness entries associated with selected degrees of freedom are
      multiplied by ``factor``,
    - off-diagonal entries coupling selected and unselected degrees of freedom
      are multiplied by ``sqrt(factor)``,
    - couplings between two selected degrees of freedom are multiplied by
      ``factor``,
    - all other entries are unchanged.

    If ``entries`` is None or empty, all rows and columns are rescaled uniformly,
    and the stiffness matrix is multiplied by ``factor``.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix with size compatible with 6 degrees of freedom
        per site.
    factor : float
        Positive rescaling factor applied to diagonal stiffness entries of the
        selected degrees of freedom.
    entries : sequence of int or None, optional
        Degree-of-freedom indices to rescale (each in the range 0..5). If None or
        empty, all degrees of freedom are rescaled uniformly.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix. The return type matches the input type.
    """ 

    if not isinstance(factor, (int, float, np.number)):
        raise TypeError("factor must be a real number.")
    
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("factor must be finite (not inf/NaN).")
    if factor <= 0:
        raise ValueError(f'factor must be a positive number, encountered factor = {factor}')

    if not hasattr(stiff, "shape") or len(stiff.shape) != 2:
        raise TypeError("stiff must be a 2D numpy array or a scipy sparse matrix.")
    
    n, m = stiff.shape
    if n != m:
        raise ValueError(f"stiff must be square, got shape {stiff.shape}.")
    if n == 0:
        raise ValueError("stiff must be non-empty.")
    if n % 6 != 0:
        raise ValueError(f"stiff dimension must be a multiple of 6, got {n}.")

    if entries is None or len(entries) == 0:
        return stiff * factor 

    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("entries must be a sequence of ints (e.g., [0, 3, 5]) or None.")

    cleaned: list[int] = []
    for i, k in enumerate(entries):
        if not isinstance(k, int):
            raise TypeError(f"entries[{i}] must be an int, got {type(k)}.")
        if not (0 <= k < 6):
            raise ValueError(f"entries[{i}] must be in range 0..5, got {k}.")
        cleaned.append(k)

    cleaned = sorted(set(cleaned))
    rescaled = stiff.copy() if sp.sparse.issparse(stiff) else np.array(stiff, copy=True)
    for k in cleaned:
        rescaled = rescale_kth(rescaled, k, factor)

    return rescaled


def rescale_stiff_dofs(
    stiff: np.ndarray | spmatrix,
    factors: Sequence[float],
    ndims: int = 6,
) -> np.ndarray | spmatrix:
    """
    Rescale each degree of freedom by its own factor in a single congruence transform.

    This is the per-DOF generalisation of :func:`rescale_stiff`: ``factors`` is a
    length-``ndims`` vector giving an independent positive factor for each DOF.
    Building the diagonal once and applying ``D K D`` a single time is more
    efficient than rescaling DOFs one at a time.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix whose dimension is a multiple of ``ndims``.
    factors : sequence of float
        Length-``ndims`` vector of positive per-DOF rescaling factors. A factor of
        1.0 leaves the corresponding DOF unchanged.
    ndims : int, default=6
        Number of degrees of freedom per site.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix (same type as input).
    """
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 1 or len(factors) != ndims:
        raise ValueError(
            f"factors must be a 1D sequence of length {ndims}, got shape {factors.shape}."
        )
    if not np.all(np.isfinite(factors)):
        raise ValueError("all factors must be finite (not inf/NaN).")
    if np.any(factors <= 0):
        raise ValueError(f"all factors must be positive, got {factors}.")

    n = _check_matrix(stiff, ndims)

    d = np.ones(n, dtype=float)
    for k in range(ndims):
        d[k::ndims] = np.sqrt(factors[k])
    return _apply_congruence(stiff, d)
