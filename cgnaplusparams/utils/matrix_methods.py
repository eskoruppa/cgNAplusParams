from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
from numpy.typing import ArrayLike
        

def matrix_copy(A: np.ndarray | spmatrix) -> np.ndarray | spmatrix:
    """
    Return a deep copy of a dense or sparse matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Input matrix.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        A deep copy of the input matrix.
    """

    if A is None:
        raise TypeError("A must not be None.")

    if not hasattr(A, "shape"):
        raise TypeError("A must be a numpy array or a scipy sparse matrix.")

    if len(A.shape) != 2:
        raise ValueError(f"A must be 2-dimensional, got shape {A.shape}.")

    if sp.sparse.issparse(A):
        return A.copy()
    else:
        return np.array(A, copy=True)
    

def symmetrize_stiffness(
    stiff: np.ndarray | spmatrix | None,
) -> np.ndarray | spmatrix | None:
    """
    Return the symmetric part ``0.5 * (K + K.T)`` of a stiffness matrix.

    Small asymmetries (typically ~1e-11) are introduced by the transform
    congruences and by the Schur-complement marginalisation. Symmetrising
    removes them so that downstream consumers relying on exact symmetry
    (e.g. Cholesky factorisation, Gaussian sampling) behave predictably.

    The sparse/dense type of the input is preserved. cgNA+ stiffness matrices
    are structurally symmetric, so for a sparse input ``K + K.T`` has the same
    sparsity pattern as ``K`` and no additional fill (denseness) is introduced.

    Parameters
    ----------
    stiff : numpy.ndarray, scipy.sparse.spmatrix or None
        Stiffness matrix to symmetrise. ``None`` is passed through unchanged.

    Returns
    -------
    numpy.ndarray, scipy.sparse.spmatrix or None
        Symmetrised matrix of the same type as the input, or ``None``.
    """
    if stiff is None:
        return None
    return (stiff + stiff.T) * 0.5


def is_positive_definite(A: ArrayLike, tol: float = 0.0) -> bool:
    """
    Check whether a matrix is (numerically) symmetric positive definite.

    Parameters
    ----------
    A : array-like
        Input matrix.
    tol : float, optional
        Absolute tolerance used for the symmetry check. Default is 0.0.

    Returns
    -------
    bool
        True if A is symmetric positive definite, False otherwise.
    """
    
    # --- tolerance check ---
    if not isinstance(tol, (int, float, np.number)):
        raise TypeError("tol must be a real number.")
    tol = float(tol)
    if tol < 0:
        raise ValueError("tol must be >= 0.")
    if not np.isfinite(tol):
        raise ValueError("tol must be finite.")

    # --- array conversion ---
    A = np.asarray(A)

    # --- shape check ---
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    # --- finite entries check ---
    if not np.all(np.isfinite(A)):
        return False

    # --- symmetry check ---
    if not np.allclose(A, A.T, atol=tol, rtol=0.0):
        return False

    # --- positive definiteness via Cholesky ---
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

