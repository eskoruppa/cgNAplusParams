"""IO utilities for reading PDB files and MD trajectories.

Functions
---------
fit_pdb:
    Fit a single PDB structure to a cgNA+ configuration.
fit_trajectory:
    Fit an MD trajectory (multiple snapshots) to a cgNA+ configuration.

.. note::
    Both functions are stubs pending full implementation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def fit_pdb(
    pdb_path: "str | Path",
    sequence: str | None = None,
) -> dict:
    """Fit a PDB structure and return named cgNA+ pose arrays.

    Parameters
    ----------
    pdb_path:
        Path to the ``.pdb`` file.
    sequence:
        Expected nucleotide sequence.  When *None*, inferred from the PDB.

    Returns
    -------
    dict with keys ``"bp_poses"`` and optionally ``"watson_base_poses"``,
    ``"crick_base_poses"``, ``"watson_phosphate_poses"``,
    ``"crick_phosphate_poses"``.  Each value has shape ``(N, 4, 4)``.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError(
        "fit_pdb() is not yet implemented.  "
        "Implement this function in cgnaplusparams/io/read_pdb.py."
    )


def fit_trajectory(
    trajectory_path: "str | Path",
    topology_path:   "str | Path",
    sequence: str | None = None,
    frames:   "slice | list | None" = None,
) -> dict:
    """Fit an MD trajectory and return named cgNA+ pose arrays.

    Parameters
    ----------
    trajectory_path:
        Path to the trajectory file (e.g. XTC, DCD, …).
    topology_path:
        Path to the topology / structure file (e.g. PDB, GRO, …).
    sequence:
        Expected nucleotide sequence.
    frames:
        Frame selection: a :class:`slice`, a list of integer indices, or
        *None* for all frames.

    Returns
    -------
    dict with keys ``"bp_poses"`` and optionally the other pose arrays.
    Each value has shape ``(n_frames, N, 4, 4)``.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError(
        "fit_trajectory() is not yet implemented.  "
        "Implement this function in cgnaplusparams/io/read_pdb.py."
    )


