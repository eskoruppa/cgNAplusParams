"""Rigid-body reference-frame fitting for DNA bases and phosphate groups.

Uses the Kabsch algorithm (via ``scipy.spatial.transform.Rotation.align_vectors``)
to superimpose ideal canonical atom positions onto observed coordinates, yielding
SE(3) rigid-body transformations.

The ideal atom positions come from :mod:`canonical_definitions` (Tsukuba or
Curves+ conventions for bases, a single definition for the phosphate group).

In the ideal configuration the reference frame is the identity — x, y, z axes
map to (1,0,0), (0,1,0), (0,0,1).  After fitting, those axes are rotated
into the **triad** T (a 3×3 SO(3) matrix stored as columns) and the body's
position **r** is the translation.  Both are packed into a 4×4 SE(3) matrix τ::

    τ = | T  r |
        | 0  1 |

where ``T = τ[:3, :3]``, ``r = τ[:3, 3]``, and ``τ[3, :] = [0, 0, 0, 1]``.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional, Tuple

try:
    from .canonical_definitions import TSUKUBA_BASES, CURVESPLUS_BASES, PHOSPHATE
    from .reader_utils import is_dna_residue
except ImportError:
    from canonical_definitions import TSUKUBA_BASES, CURVESPLUS_BASES, PHOSPHATE
    from reader_utils import is_dna_residue

_CONVENTIONS: Dict[str, dict] = {
    "tsukuba": TSUKUBA_BASES,
    "curvesplus": CURVESPLUS_BASES,
}

_RESNAME_MAP: Dict[str, str] = {"DA": "A", "DT": "T", "DG": "G", "DC": "C"}

# Thymine methyl carbon: canonical uses C5M, some PDB files use C7.
_ATOM_ALIASES: Dict[str, List[str]] = {"C5M": ["C5M", "C7"]}

# Ångström → nanometre
_A2NM = 0.1

# Right-multiplication flip that negates the 2nd and 3rd body-frame axes:
#   τ' = τ @ _CRICK_FLIP  ⟹  T' = T @ diag(1,-1,-1),  r' = r
_CRICK_FLIP = np.diag([1., -1., -1., 1.])

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _get_atom_index(residue, atom_name: str) -> Optional[int]:
    """Return the MDtraj atom index for *atom_name* in *residue*, or ``None``."""
    aliases = _ATOM_ALIASES.get(atom_name, [atom_name])
    for name in aliases:
        matches = list(residue.atoms_by_name(name))
        if matches:
            return matches[0].index
    return None


def _collect_matching_atoms(
    residue,
    ideal_atoms: Dict[str, np.ndarray],
    xyz_frame: np.ndarray,
    extra_atoms: Optional[Dict[str, Tuple]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return matched ``(ideal, actual)`` coordinate arrays.

    Parameters
    ----------
    residue :
        MDtraj Residue supplying the main atoms.
    ideal_atoms :
        ``{atom_name: ideal_position_nm}`` for atoms belonging to *residue*.
    xyz_frame : (n_atoms, 3)
        Coordinates for one frame (nm).
    extra_atoms :
        ``{atom_name: (ideal_pos_nm, source_residue)}`` for atoms that live on
        a different residue (e.g. O3' from the preceding nucleotide).
    """
    ideal_list: List[np.ndarray] = []
    actual_list: List[np.ndarray] = []

    for atom_name, ideal_pos in ideal_atoms.items():
        idx = _get_atom_index(residue, atom_name)
        if idx is not None:
            ideal_list.append(ideal_pos)
            actual_list.append(xyz_frame[idx])

    if extra_atoms:
        for atom_name, (ideal_pos, source_res) in extra_atoms.items():
            idx = _get_atom_index(source_res, atom_name)
            if idx is not None:
                ideal_list.append(ideal_pos)
                actual_list.append(xyz_frame[idx])

    if not ideal_list:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.asarray(ideal_list), np.asarray(actual_list)


def _kabsch_se3(ideal: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Kabsch alignment returning a 4×4 SE(3) matrix.

    Both *ideal* and *actual* must be ``(n, 3)`` arrays in the **same** unit
    (nanometres).
    """

    # ideal = ideal*10
    centroid_ideal = ideal.mean(axis=0)
    centroid_actual = actual.mean(axis=0)

    # print(ideal-centroid_ideal)
    # print(actual-centroid_actual)
    # import sys
    # sys.exit()

    rot, _ = Rotation.align_vectors(
        actual - centroid_actual,
        ideal - centroid_ideal,
    )
    T = rot.as_matrix()  # 3×3 SO(3) triad
    r = centroid_actual - T @ centroid_ideal
    tau = np.eye(4)
    tau[:3, :3] = T
    tau[:3, 3] = r
    return tau


# ---------------------------------------------------------------------------
# Single-residue fitting
# ---------------------------------------------------------------------------


def fit_base(
    residue,
    xyz_frame: np.ndarray,
    convention: str = "tsukuba",
) -> Optional[np.ndarray]:
    """Fit a base reference frame for one nucleotide.

    Parameters
    ----------
    residue : mdtraj Residue
    xyz_frame : (n_atoms, 3) array in **nm** — one frame of coordinates.
    convention : ``'tsukuba'`` or ``'curvesplus'``

    Returns
    -------
    tau : (4, 4) SE(3) ndarray, or ``None`` when fewer than 3 atoms match.
    """
    letter = _RESNAME_MAP.get(residue.name)
    if letter is None:
        return None

    canon = _CONVENTIONS[convention].get(letter)
    if canon is None:
        return None

    ideal_atoms = {k: v * _A2NM for k, v in canon.items()}
    ideal, actual = _collect_matching_atoms(residue, ideal_atoms, xyz_frame)
    if len(ideal) < 3:
        return None
    return _kabsch_se3(ideal, actual)


def fit_phosphate(
    residue,
    xyz_frame: np.ndarray,
    prev_residue=None,
) -> Optional[np.ndarray]:
    """Fit a phosphate reference frame for one nucleotide.

    ``O3'`` belongs to *prev_residue* — the 5'-neighbour in the strand.
    If *prev_residue* is ``None`` the ``O3'`` atom is simply omitted from the
    fit.

    Parameters
    ----------
    residue : mdtraj Residue
        Supplies P, O5', OP1, OP2.
    xyz_frame : (n_atoms, 3) array in **nm**.
    prev_residue : mdtraj Residue, optional
        The preceding nucleotide in 5'→3' order (supplies O3').

    Returns
    -------
    tau : (4, 4) SE(3) ndarray, or ``None`` when fewer than 3 atoms match.
    """
    ideal_main: Dict[str, np.ndarray] = {}
    extra: Dict[str, Tuple] = {}

    for atom_name, pos in PHOSPHATE.items():
        pos_nm = pos * _A2NM
        if atom_name == "O3'":
            if prev_residue is not None:
                extra[atom_name] = (pos_nm, prev_residue)
        else:
            ideal_main[atom_name] = pos_nm

    ideal, actual = _collect_matching_atoms(
        residue, ideal_main, xyz_frame, extra_atoms=extra,
    )
    if len(ideal) < 3:
        return None
    tau = _kabsch_se3(ideal, actual)
    # T = tau[:3, :3]
    # T = np.roll(T, shift=1, axis=1)
    # tau[:3, :3] = T
    return tau


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------


def fit_bases(
    residues,
    xyz_frame: np.ndarray,
    convention: str = "tsukuba",
) -> np.ndarray:
    """Fit base frames for a sequence of residues.

    Returns an ``(n, 4, 4)`` array.  Entries are ``NaN`` where fitting failed.
    """
    n = len(residues)
    taus = np.full((n, 4, 4), np.nan)
    for i, res in enumerate(residues):
        tau = fit_base(res, xyz_frame, convention)
        if tau is not None:
            taus[i] = tau
    return taus


def fit_phosphates(
    residues_5to3,
    xyz_frame: np.ndarray,
) -> np.ndarray:
    """Fit phosphate frames for residues ordered **5'→3'**.

    For position *i*, ``O3'`` is taken from position *i − 1*.  The first
    position has no predecessor, so ``O3'`` is omitted from its fit.

    Returns an ``(n, 4, 4)`` array.  Entries are ``NaN`` where fitting failed.
    """
    n = len(residues_5to3)
    taus = np.full((n, 4, 4), np.nan)
    for i, res in enumerate(residues_5to3):
        prev = residues_5to3[i - 1] if i > 0 else None
        tau = fit_phosphate(res, xyz_frame, prev_residue=prev)
        if tau is not None:
            taus[i] = tau
    return taus


# ---------------------------------------------------------------------------
# Duplex-level fitting
# ---------------------------------------------------------------------------


def fit_duplex_frames(
    watson_5to3,
    crick_3to5,
    xyz_frame: np.ndarray,
    convention: str = "tsukuba",
    crickflip: bool = False,
) -> Dict[str, np.ndarray]:
    """Fit base and phosphate frames for a Watson–Crick duplex segment.

    Parameters
    ----------
    watson_5to3 : list of Residue
        Watson-strand residues in 5'→3' order.
    crick_3to5 : list of Residue
        Crick-strand residues in 3'→5' order (antiparallel to Watson).
    xyz_frame : (n_atoms, 3) array in **nm**
        Atom coordinates for a single trajectory frame.
    convention : str
        Base atom convention (``'tsukuba'`` or ``'curvesplus'``).
    crickflip : bool, default False
        When True, apply ``τ' = τ @ f`` with ``f = diag(1,-1,-1,1)`` to all
        Crick base and phosphate frames.  This negates the 2nd and 3rd
        body-frame axes (y and z columns of T) while leaving the position r
        and the 1st axis unchanged.  The result remains in SO(3)×ℝ³.

    Returns
    -------
    dict
        ``watson_bases``       — (n_w, 4, 4)
        ``crick_bases``        — (n_c, 4, 4)
        ``watson_phosphates``  — (n_w, 4, 4)
        ``crick_phosphates``   — (n_c, 4, 4)

        Indexing matches the input residue order.  ``NaN`` where fitting
        failed.
    """
    if convention not in _CONVENTIONS:
        raise ValueError(f"Unsupported convention {convention!r}. Supported: {list(_CONVENTIONS.keys())}")

    watson_bases = fit_bases(watson_5to3, xyz_frame, convention)
    crick_bases = fit_bases(crick_3to5, xyz_frame, convention)

    watson_phosphates = fit_phosphates(watson_5to3, xyz_frame)

    # Crick is stored 3'→5'; reverse to 5'→3' for predecessor logic,
    # then reverse results back so indices match input order.
    crick_5to3 = list(reversed(crick_3to5))
    crick_phos_5to3 = fit_phosphates(crick_5to3, xyz_frame)
    crick_phosphates = crick_phos_5to3[::-1]

    if crickflip:
        # Right-multiply by _CRICK_FLIP: negates columns 1 and 2 of each τ
        # (i.e. the y and z body-frame axes), leaves translation unchanged.
        crick_bases = crick_bases @ _CRICK_FLIP
        crick_phosphates = crick_phosphates @ _CRICK_FLIP

    return {
        "watson_bases": watson_bases,
        "crick_bases": crick_bases,
        "watson_phosphates": watson_phosphates,
        "crick_phosphates": crick_phosphates,
    }
