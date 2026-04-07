"""Standalone utilities for DNA base-pair detection.

Self-contained — no dependencies on the mdna package.
"""

# Watson-Crick complement map (single-letter codes)
BP_MAP = {
    'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
    'U': 'A',
}

# DNA residue names as they appear in PDB / MDtraj topologies
DNA_RESIDUE_NAMES = {'DA', 'DT', 'DG', 'DC', 'DU'}

# Residue name → single-letter code
RESNAME_TO_LETTER = {
    'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C', 'DU': 'U',
    'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'U',
}


def is_dna_residue(residue) -> bool:
    """Check if an MDtraj Residue is a DNA nucleotide."""
    return residue.name in DNA_RESIDUE_NAMES


def one_letter_code(resname: str) -> str:
    """Convert a residue name like 'DA' to single letter 'A'.

    Returns the input unchanged if not recognised.
    """
    return RESNAME_TO_LETTER.get(resname, resname)


def are_complementary(resname_a: str, resname_b: str) -> bool:
    """Check if two residue names are Watson-Crick complements."""
    letter_a = one_letter_code(resname_a)
    letter_b = one_letter_code(resname_b)
    return BP_MAP.get(letter_a) == letter_b


def parse_barnaba_seq_entry(entry: str):
    """Parse a barnaba seq string like ``'DA_-73_0'``.

    Returns:
        (resname, resSeq, chain_index) — e.g. ``('DA', -73, 0)``.
    """
    parts = entry.rsplit('_', 2)
    resname = parts[0]
    resSeq = int(parts[1])
    chain_index = int(parts[2])
    return resname, resSeq, chain_index
