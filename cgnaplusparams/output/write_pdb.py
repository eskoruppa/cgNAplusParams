from __future__ import annotations

import json
import os
import sys
from typing import Any #, Dict, List
import numpy as np

PDP_BPDICTS_FN = "database/bpdicts"

###########################################################################################################################
###########################################################################################################################


def _load_bpdicts(fn: str | None = None) -> dict[str, Any]:
    """
    Parameters:
    fn : str
        bpdict database

    Returns:
    dict - containing residue data
    """
    if fn is None:
        fn = os.path.normpath(os.path.join(os.path.dirname(__file__), PDP_BPDICTS_FN))
    with open(fn, "r") as f:
        bpdicts = json.load(f)
    return bpdicts


###########################################################################################################################
###########################################################################################################################


def _DNA_residue_name(residue_name: str):
    """
    converts basename into pdb residue name
    """
    if residue_name.lower() == "a":
        return "DA"
    if residue_name.lower() == "t":
        return "DT"
    if residue_name.lower() == "g":
        return "DG"
    if residue_name.lower() == "c":
        return "DC"
    return ""


###########################################################################################################################
###########################################################################################################################

def _build_pdb_atomline(
    atomID: int,
    atom_name: str,
    residue_name: str,
    strandID: str,
    residueID: int,
    atom_pos: list[float],
) -> str:
    """
    generates pdb line for atom
    """
    pdbline = (
        "ATOM  "
        + _leftshiftstring(5, str(atomID))
        + " "
        + _leftshiftstring(4, atom_name)
        + " "
        + _leftshiftstring(3, residue_name)
        + " "
        + str(strandID)
        + _leftshiftstring(4, str(residueID))
        + "    "
        + _leftshiftstring(8, "%.3f" % atom_pos[0])
        + _leftshiftstring(8, "%.3f" % atom_pos[1])
        + _leftshiftstring(8, "%.3f" % atom_pos[2])
        + " \n"
    )
    return pdbline

###########################################################################################################################
###########################################################################################################################

def _build_pdb_terline(
    atomID: int, residue_name: str, strandID: str, residueID: int
) -> str:
    """
    generates TER line for pdb strand
    """
    pdbline = (
        "TER   "
        + _leftshiftstring(5, str(atomID))
        + " "
        + _leftshiftstring(4, "")
        + " "
        + _leftshiftstring(3, residue_name)
        + " "
        + str(strandID)
        + _leftshiftstring(4, str(residueID))
        + "    \n"
    )
    return pdbline

###########################################################################################################################
###########################################################################################################################


def _leftshiftstring(total_chars: int, string: str) -> str:
    """
    add lefthand spaces to string to fill number of characters
    """
    chars = len(string)
    shifted_str = ""
    for i in range(total_chars - chars):
        shifted_str += " "
    return shifted_str + string


###########################################################################################################################
###########################################################################################################################


def _random_sequence(N: int) -> list[str]:
    """
    generates random base sequence of length N
    """
    basetypes = ["A", "T", "C", "G"]
    return [basetypes[bt] for bt in np.random.randint(4, size=N)]


###########################################################################################################################
###########################################################################################################################


def _discretization_length(conf: np.ndarray) -> np.ndarray:
    """returns lengths of vectors"""
    ndims = len(np.shape(conf))
    vecs = np.diff(conf, axis=ndims - 2)
    return np.mean(np.linalg.norm(vecs, axis=ndims - 1))


###########################################################################################################################
###########################################################################################################################


def gen_pdb(
    outfn: str,
    poses: np.ndarray,
    sequence: str,
    bpdicts: dict[str, Any] = None,
    center: bool = True,
    ignore_errors: bool = False,
):
    """
    positions needs to be in nm!
    """

    positions = poses[:, :3, 3]
    triads = poses[:, :3, :3]

    if len(positions.shape) > 2:
        raise ValueError(
            f"Wrong dimension provided for positions. Input needs to be a single configuration."
        )
    if len(triads.shape) > 3:
        raise ValueError(
            f"Wrong dimension provided for triads. Input needs to be a single configuration."
        )
    
    if bpdicts is None:
        bpdicts = _load_bpdicts()
    sequence = sequence.upper()

    numbp = len(positions)
    
    disc_len = _discretization_length(positions)
    if np.abs(disc_len - 0.34) / 0.34 > 0.5:
        # wrong discretization length
        if not ignore_errors:
            raise ValueError(
                f"Discretization length needs to be close to 0.34 nm. Provided configuration has discretization length {disc_len} nm!"
            )

    if sequence is None:
        sequence = _random_sequence(numbp)

    if center:
        positions -= np.mean(positions, axis=0)

    # convert to Anstrom
    positions = 10 * np.array(positions)

    with open(outfn, "w") as f:
        atomID = 0
        residueID = 0

        # STRAND A
        strandID = "A"
        residue_name = ""
        for i in range(numbp):
            residueID += 1
            basetype = sequence[i]
            triad = triads[i]
            pos = positions[i]

            bpdict = bpdicts[basetype]
            residue = bpdict["resA"]
            residue_name = residue["resname"]

            for atom in residue["atoms"]:
                atomID += 1
                atom_name = atom["name"]
                atom_pos = atom["pos"]
                # atom_pos = np.dot(atom_pos,triad) + pos
                atom_pos = np.dot(triad, atom_pos) + pos
                pdbline = _build_pdb_atomline(
                    atomID, atom_name, residue_name, strandID, residueID, atom_pos
                )
                f.write(pdbline)

        pdbline = _build_pdb_terline(atomID, residue_name, strandID, residueID)
        f.write(pdbline)

        # STRAND B
        strandID = "B"
        for i in range(numbp - 1, -1, -1):
            residueID += 1
            basetype = sequence[i]
            triad = triads[i]
            pos = positions[i]

            bpdict = bpdicts[basetype]
            residue = bpdict["resB"]
            residue_name = residue["resname"]

            for atom in residue["atoms"]:
                atomID += 1
                atom_name = atom["name"]
                atom_pos = atom["pos"]
                # atom_pos = np.dot( atom_pos,triad) + pos
                atom_pos = np.dot(triad, atom_pos) + pos
                pdbline = _build_pdb_atomline(
                    atomID, atom_name, residue_name, strandID, residueID, atom_pos
                )
                f.write(pdbline)
        pdbline = _build_pdb_terline(atomID, residue_name, strandID, residueID)
        f.write(pdbline)
        f.close()


###########################################################################################################################
###########################################################################################################################
