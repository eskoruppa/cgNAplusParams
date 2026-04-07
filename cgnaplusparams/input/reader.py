import os
import io
import sys
import string
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np

try:
    from .reader_utils import (
        DNA_RESIDUE_NAMES,
        are_complementary,
        is_dna_residue,
        one_letter_code,
        parse_barnaba_seq_entry,
    )
except ImportError:
    from reader_utils import (
        DNA_RESIDUE_NAMES,
        are_complementary,
        is_dna_residue,
        one_letter_code,
        parse_barnaba_seq_entry,
    )

_PDB_CHAIN_CHARS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _load_cif_via_gemmi(path: str) -> md.Trajectory:
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(
            "gemmi is required to load CIF / mmCIF files.\n"
            "Install it with:  pip install gemmi"
        ) from exc

    struct = gemmi.read_structure(path)
    struct.remove_waters()
    struct.remove_empty_chains()

    used: set = {chain.name for chain in struct[0] if len(chain.name) == 1}
    char_iter = (c for c in _PDB_CHAIN_CHARS if c not in used)
    for chain in struct[0]:
        if len(chain.name) > 1:
            try:
                chain.name = next(char_iter)
            except StopIteration:
                raise RuntimeError(
                    "Structure has more chains than available single-character "
                    f"PDB chain IDs ({len(_PDB_CHAIN_CHARS)} supported)."
                )

    pdb_str = struct.make_pdb_string()

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        tmp.write(pdb_str)
        tmp_path = tmp.name

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = md.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    return traj


def _load_structure_or_trajectory(path: str | Path, topology: Optional[str]) -> md.Trajectory:
    ext = Path(path).suffix.lower()
    if ext in (".cif", ".mmcif"):
        if topology is not None:
            warnings.warn("The 'topology' argument is ignored for CIF/mmCIF files.", stacklevel=3)
        return _load_cif_via_gemmi(path)
    if topology is not None:
        return md.load(path, top=topology)
    return md.load(path)


# ---------------------------------------------------------------------------
# Domain data containers
# ---------------------------------------------------------------------------


def _format_pair_lines(
    pairs: List[Tuple],
    bonded_mask: Optional[List[bool]] = None,
    w_indices: Optional[List[int]] = None,
    c_indices: Optional[List[int]] = None,
    indent: str = "  ",
) -> List[str]:
    """Render a list of (watson_res, crick_res) pairs as aligned text lines.

    Parameters
    ----------
    pairs :
        List of (watson_residue, crick_residue) tuples.
    bonded_mask :
        When provided, prepend ``(X)`` for bonded pairs and ``(-)`` for
        bubble (non-bonded) positions.  Omit for lists where every pair is
        bonded by definition.
    w_indices :
        0-based position of each Watson residue within its strand's DNA
        residue list.  Printed as a left-hand column when provided.
    c_indices :
        0-based position of each Crick residue within its strand's DNA
        residue list.  Printed as a right-hand column when provided.
    indent :
        Leading whitespace for each line.
    """
    if not pairs:
        return []
    w_strs = [str(w) for w, _ in pairs]
    c_strs = [str(c) for _, c in pairs]
    w_width = max(len(s) for s in w_strs)

    # Width of the index columns (same width on both sides for visual symmetry)
    idx_width = 0
    if w_indices is not None or c_indices is not None:
        all_idxs: List[int] = []
        if w_indices:
            all_idxs.extend(w_indices)
        if c_indices:
            all_idxs.extend(c_indices)
        idx_width = len(str(max(all_idxs))) if all_idxs else 1

    # Crick name needs padding so the trailing index column aligns
    c_width = max(len(s) for s in c_strs) if c_indices is not None else 0

    lines: List[str] = []
    for i, (w_s, c_s) in enumerate(zip(w_strs, c_strs)):
        line = indent
        if bonded_mask is not None:
            line += f"{'(X)' if bonded_mask[i] else '(-)'}  "
        if w_indices is not None:
            line += f"{w_indices[i]:>{idx_width}}  "
        line += f"{w_s:<{w_width}} <-> "
        if c_indices is not None:
            line += f"{c_s:<{c_width}}  {c_indices[i]:>{idx_width}}"
        else:
            line += c_s
        lines.append(line)
    return lines


def _dna_pos_map(chain_obj) -> Dict[int, int]:
    """Map residue.index to 0-based position in *chain_obj*'s DNA residue list."""
    if chain_obj is None:
        return {}
    dna_res = [r for r in chain_obj.residues if is_dna_residue(r)]
    return {r.index: i for i, r in enumerate(dna_res)}


@dataclass
class BondedDomain:
    """A contiguous stretch of Watson-Crick base pairs without gaps in either strand.

    'Contiguous' means adjacent in the chain's DNA residue order:
    Watson positions advance by +1 and Crick positions advance by -1 for each
    successive pair (antiparallel convention).
    """

    pairs: List[Tuple]
    chain_watson: int = 0
    chain_crick: int = 0
    strand_watson: Optional[object] = field(default=None, repr=False, compare=False)
    strand_crick: Optional[object] = field(default=None, repr=False, compare=False)
    traj: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def watson_residues(self) -> list:
        return [p[0] for p in self.pairs]

    def crick_residues(self) -> list:
        return [p[1] for p in self.pairs]

    @property
    def sequence(self) -> str:
        """Watson-strand base sequence as single uppercase letters (5'->3')."""
        return "".join(one_letter_code(r.name) for r in self.watson_residues())

    def __repr__(self) -> str:
        header = f"BondedDomain(n_pairs={self.n_pairs})"
        if not self.pairs:
            return header
        lines = [header]
        if self.strand_watson is not None:
            lines.append(f"  Watson chain {self.chain_watson} | Crick chain {self.chain_crick}")
            wm = _dna_pos_map(self.strand_watson)
            cm = _dna_pos_map(self.strand_crick)
            w_idx = [wm.get(w.index, 0) for w, _ in self.pairs]
            c_idx = [cm.get(c.index, 0) for _, c in self.pairs]
            lines += _format_pair_lines(self.pairs, w_indices=w_idx, c_indices=c_idx)
        else:
            lines += _format_pair_lines(self.pairs)
        return "\n".join(lines)

    def fit_frames(self, frame: int = 0, convention: str = "tsukuba", crickflip: bool = False):
        """Fit base and phosphate SE(3) reference frames for both strands.

        Parameters
        ----------
        frame : int
            Frame index.
        convention : str
            Base atom convention ('tsukuba' or 'curvesplus').
        crickflip : bool
            Apply the crick-flip transformation (negate y and z body-frame
            axes) to all Crick frames.

        Returns
        -------
        dict with 'watson_bases', 'crick_bases', 'watson_phosphates',
        'crick_phosphates' — each an (n, 4, 4) ndarray.
        """
        if self.traj is None:
            raise ValueError(
                "BondedDomain.traj is not set. "
                "Construct via read_dna() or set .traj manually."
            )
        try:
            from .reference_frames import fit_duplex_frames
        except ImportError:
            from reference_frames import fit_duplex_frames
        return fit_duplex_frames(
            self.watson_residues(), self.crick_residues(),
            self.traj.xyz[frame], convention, crickflip=crickflip,
        )


@dataclass
class AlignedDomain:
    """One or more BondedDomains sharing the same sequence-alignment register.

    BondedDomains are merged into one AlignedDomain when the unpaired residues
    (bubble) between them are symmetric: the same number of nucleotides are
    unpaired on both the Watson and Crick strands.  An asymmetric bubble shifts
    the reading frame and starts a new AlignedDomain.

    Attributes
    ----------
    bonded_domains : list of BondedDomain
        Contiguous base-pair runs within this aligned region.
    watson_residues : list of Residue
        All Watson residues spanning this domain (paired and unpaired), 5'->3'.
    crick_residues : list of Residue
        All Crick residues spanning this domain (paired and unpaired), 3'->5'.
    """

    bonded_domains: List[BondedDomain]
    watson_residues: List
    crick_residues: List
    chain_watson: int = 0
    chain_crick: int = 0
    strand_watson: Optional[object] = field(default=None, repr=False, compare=False)
    strand_crick: Optional[object] = field(default=None, repr=False, compare=False)
    traj: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def n_bonded_domains(self) -> int:
        return len(self.bonded_domains)

    @property
    def n_watson(self) -> int:
        return len(self.watson_residues)

    @property
    def n_crick(self) -> int:
        return len(self.crick_residues)

    @property
    def pairs(self) -> List[Tuple]:
        """All aligned (watson, crick) residue pairs in Watson 5'->3' order.

        Includes both bonded (base-paired) and non-bonded (bubble) positions.
        Because only symmetric bubbles are merged into one AlignedDomain,
        ``len(watson_residues) == len(crick_residues)`` is guaranteed and
        every Watson position maps to exactly one Crick position.
        """
        return list(zip(self.watson_residues, self.crick_residues))

    @property
    def bonded_pairs(self) -> List[bool]:
        """Boolean mask over :attr:`pairs` — True where the pair is actually base-paired."""
        paired = {(w.index, c.index) for bd in self.bonded_domains for w, c in bd.pairs}
        return [(w.index, c.index) in paired for w, c in self.pairs]

    def paired_watson_mask(self) -> List[bool]:
        """True at positions where the Watson residue is base-paired."""
        paired = {r.index for bd in self.bonded_domains for r, _ in bd.pairs}
        return [r.index in paired for r in self.watson_residues]

    def paired_crick_mask(self) -> List[bool]:
        """True at positions where the Crick residue is base-paired."""
        paired = {r.index for bd in self.bonded_domains for _, r in bd.pairs}
        return [r.index in paired for r in self.crick_residues]

    @property
    def sequence(self) -> str:
        """Watson-strand base sequence as single uppercase letters (5'->3')."""
        return "".join(one_letter_code(r.name) for r in self.watson_residues)

    def __repr__(self) -> str:
        header = (
            f"AlignedDomain(n_bonded_domains={self.n_bonded_domains}, "
            f"watson={self.n_watson} nt, crick={self.n_crick} nt)"
        )
        lines = [header]
        if self.strand_watson is not None:
            lines.append(f"  Watson chain {self.chain_watson} | Crick chain {self.chain_crick}")
            wm = _dna_pos_map(self.strand_watson)
            cm = _dna_pos_map(self.strand_crick)
            w_idx = [wm.get(w.index, 0) for w, _ in self.pairs]
            c_idx = [cm.get(c.index, 0) for _, c in self.pairs]
            lines += _format_pair_lines(
                self.pairs, bonded_mask=self.bonded_pairs,
                w_indices=w_idx, c_indices=c_idx,
            )
        else:
            lines += _format_pair_lines(self.pairs, bonded_mask=self.bonded_pairs)
        return "\n".join(lines)

    def fit_frames(self, frame: int = 0, convention: str = "tsukuba", crickflip: bool = False):
        """Fit base and phosphate SE(3) reference frames for both strands.

        Parameters
        ----------
        frame : int
            Frame index.
        convention : str
            Base atom convention ('tsukuba' or 'curvesplus').
        crickflip : bool
            Apply the crick-flip transformation (negate y and z body-frame
            axes) to all Crick frames.

        Returns
        -------
        dict with 'watson_bases', 'crick_bases', 'watson_phosphates',
        'crick_phosphates' — each an (n, 4, 4) ndarray.
        """
        if self.traj is None:
            raise ValueError(
                "AlignedDomain.traj is not set. "
                "Construct via read_dna() or set .traj manually."
            )
        try:
            from .reference_frames import fit_duplex_frames
        except ImportError:
            from reference_frames import fit_duplex_frames
        return fit_duplex_frames(
            self.watson_residues, self.crick_residues,
            self.traj.xyz[frame], convention, crickflip=crickflip,
        )


# ---------------------------------------------------------------------------
# Domain computation helpers
# ---------------------------------------------------------------------------


def _split_bonded_domains(
    pairs: List[Tuple],
    watson_all: List,
    crick_all: List,
    chain_watson: int = 0,
    chain_crick: int = 0,
    strand_watson=None,
    strand_crick=None,
    traj=None,
) -> List[BondedDomain]:
    """Split a Watson 5'->3' sorted pair list into contiguous BondedDomains.

    A new domain begins whenever consecutive pairs skip a residue on either
    strand: Watson positions must advance by exactly +1 and Crick positions
    by exactly -1 (antiparallel).

    Parameters
    ----------
    pairs : list of (watson_res, crick_res)
        Base pairs sorted in Watson 5'->3' order.
    watson_all : list of Residue
        All DNA residues of the Watson chain in 5'->3' order.
    crick_all : list of Residue
        All DNA residues of the Crick chain in 5'->3' order.
    """
    if not pairs:
        return []

    w_pos = {r.index: i for i, r in enumerate(watson_all)}
    c_pos = {r.index: i for i, r in enumerate(crick_all)}

    domains: List[BondedDomain] = []
    current = [pairs[0]]

    for prev, curr in zip(pairs[:-1], pairs[1:]):
        dw = w_pos[curr[0].index] - w_pos[prev[0].index]
        dc = c_pos[curr[1].index] - c_pos[prev[1].index]
        # Antiparallel: Watson advances +1, Crick advances -1
        if dw == 1 and dc == -1:
            current.append(curr)
        else:
            domains.append(BondedDomain(
                pairs=list(current),
                chain_watson=chain_watson,
                chain_crick=chain_crick,
                strand_watson=strand_watson,
                strand_crick=strand_crick,
                traj=traj,
            ))
            current = [curr]

    domains.append(BondedDomain(
        pairs=list(current),
        chain_watson=chain_watson,
        chain_crick=chain_crick,
        strand_watson=strand_watson,
        strand_crick=strand_crick,
        traj=traj,
    ))
    return domains


def _split_aligned_domains(
    bonded_domains: List[BondedDomain],
    watson_all: List,
    crick_all: List,
    chain_watson: int = 0,
    chain_crick: int = 0,
    strand_watson=None,
    strand_crick=None,
    traj=None,
) -> List[AlignedDomain]:
    """Group BondedDomains by alignment register into AlignedDomains.

    Two consecutive BondedDomains are merged when the bubble between them is
    symmetric: the same number of unpaired residues on both strands.  An
    asymmetric bubble shifts the register and starts a new AlignedDomain.

    Parameters
    ----------
    bonded_domains : list of BondedDomain
        Sorted in Watson 5'->3' order.
    watson_all : list of Residue
        All DNA residues of the Watson chain in 5'->3' order.
    crick_all : list of Residue
        All DNA residues of the Crick chain in 5'->3' order (used for position
        lookup only; the list is NOT reversed).
    """
    if not bonded_domains:
        return []

    w_pos = {r.index: i for i, r in enumerate(watson_all)}
    c_pos = {r.index: i for i, r in enumerate(crick_all)}

    groups: List[List[BondedDomain]] = [[bonded_domains[0]]]

    for bd_prev, bd_curr in zip(bonded_domains[:-1], bonded_domains[1:]):
        w_last = w_pos[bd_prev.pairs[-1][0].index]
        c_last = c_pos[bd_prev.pairs[-1][1].index]
        w_first = w_pos[bd_curr.pairs[0][0].index]
        c_first = c_pos[bd_curr.pairs[0][1].index]

        # Number of unpaired residues in the bubble on each strand
        w_gap = w_first - w_last - 1
        c_gap = c_last - c_first - 1  # Crick is antiparallel: positions decrease

        if w_gap == c_gap:
            groups[-1].append(bd_curr)
        else:
            groups.append([bd_curr])

    result: List[AlignedDomain] = []
    for group in groups:
        w_start = w_pos[group[0].pairs[0][0].index]
        w_end = w_pos[group[-1].pairs[-1][0].index]
        watson_res = watson_all[w_start : w_end + 1]

        # First pair of first domain has the highest Crick chain position
        c_hi = c_pos[group[0].pairs[0][1].index]
        # Last pair of last domain has the lowest Crick chain position
        c_lo = c_pos[group[-1].pairs[-1][1].index]
        # Reverse to yield 3'->5' order (anti-parallel to Watson 5'->3')
        crick_res = list(reversed(crick_all[c_lo : c_hi + 1]))

        result.append(AlignedDomain(
            bonded_domains=group,
            watson_residues=watson_res,
            crick_residues=crick_res,
            chain_watson=chain_watson,
            chain_crick=chain_crick,
            strand_watson=strand_watson,
            strand_crick=strand_crick,
            traj=traj,
        ))

    return result


@dataclass
class StrandPair:
    """A detected DNA duplex between two chains.

    Attributes
    ----------
    chain_watson : int
        MDtraj chain index of the sense (Watson, 5->3) strand.
    chain_crick : int
        MDtraj chain index of the antisense (Crick, 3->5) strand.
    pairs : list of (Residue, Residue)
        (watson_residue, crick_residue) tuples ordered Watson 5->3 / Crick 3->5.
    method : str
        Detection method: "geometric" (barnaba) or "index_fallback" (MDNA-style).
    """

    chain_watson: int
    chain_crick: int
    pairs: List[Tuple]
    method: str
    bonded_domains: List[BondedDomain] = field(default_factory=list)
    aligned_domains: List[AlignedDomain] = field(default_factory=list)
    strand_watson: Optional[object] = field(default=None, repr=False, compare=False)
    strand_crick: Optional[object] = field(default=None, repr=False, compare=False)
    traj: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def watson_residues(self) -> list:
        return [p[0] for p in self.pairs]

    def crick_residues(self) -> list:
        return [p[1] for p in self.pairs]

    @property
    def sequence(self) -> str:
        """Watson-strand base sequence as single uppercase letters (5'->3')."""
        return "".join(one_letter_code(r.name) for r in self.watson_residues())

    def __str__(self) -> str:
        return (
            f"StrandPair(watson=chain {self.chain_watson}, "
            f"crick=chain {self.chain_crick}, "
            f"n_pairs={self.n_pairs}, method='{self.method}')"
        )

    def __repr__(self) -> str:
        lines = [str(self)]
        if self.strand_watson is not None:
            lines.append(f"  Watson chain {self.chain_watson} | Crick chain {self.chain_crick}")
            wm = _dna_pos_map(self.strand_watson)
            cm = _dna_pos_map(self.strand_crick)
            w_idx = [wm.get(w.index, 0) for w, _ in self.pairs]
            c_idx = [cm.get(c.index, 0) for _, c in self.pairs]
            lines += _format_pair_lines(self.pairs, w_indices=w_idx, c_indices=c_idx)
        else:
            lines += _format_pair_lines(self.pairs)
        return "\n".join(lines)

    def fit_frames(self, frame: int = 0, convention: str = "tsukuba", crickflip: bool = False):
        """Fit base and phosphate SE(3) reference frames for both strands.

        Parameters
        ----------
        frame : int
            Frame index.
        convention : str
            Base atom convention ('tsukuba' or 'curvesplus').
        crickflip : bool
            Apply the crick-flip transformation (negate y and z body-frame
            axes) to all Crick frames.

        Returns
        -------
        dict with 'watson_bases', 'crick_bases', 'watson_phosphates',
        'crick_phosphates' — each an (n, 4, 4) ndarray.
        """
        if self.traj is None:
            raise ValueError(
                "StrandPair.traj is not set. "
                "Construct via read_dna() or set .traj manually."
            )
        try:
            from .reference_frames import fit_duplex_frames
        except ImportError:
            from reference_frames import fit_duplex_frames
        return fit_duplex_frames(
            self.watson_residues(), self.crick_residues(),
            self.traj.xyz[frame], convention, crickflip=crickflip,
        )


class DNAReader:
    """Load a structure or trajectory and detect DNA duplexes via base pairing.

    Parameters
    ----------
    path : str | Path
        Input file path. PDB and CIF/mmCIF for structures; XTC, DCD, TRR, etc.
        for trajectories (require topology).
    topology : str, optional
        Topology file for trajectory formats. Ignored for CIF/mmCIF.
    per_frame : bool, default False
        Re-run detection on every frame when True.

    Attributes
    ----------
    traj : md.Trajectory
    dna_chains : list of int
        MDtraj chain indices of all DNA-containing chains.
    duplexes : list of StrandPair
        Populated after detect() is called.

    Notes
    -----
    Detection strategy:
      1. Geometric (primary) - barnaba.annotate_traj(), inter-chain WC pairs.
      2. Index fallback - positional pairing for equal-length reverse-complement chains.
    """

    def __init__(self, path: str | Path, topology: Optional[str] = None, per_frame: bool = False):
        self.traj = _load_structure_or_trajectory(path, topology)
        self.per_frame = per_frame
        self.dna_chains: List[int] = self._identify_dna_chains()
        self.duplexes: List[StrandPair] = []
        self._frame_duplexes: Dict[int, List[StrandPair]] = {}

    def _identify_dna_chains(self) -> List[int]:
        """Return MDtraj chain indices for all chains with at least one DNA residue."""
        dna_chain_ids = []
        for chain in self.traj.topology.chains:
            for residue in chain.residues:
                if is_dna_residue(residue):
                    dna_chain_ids.append(chain.index)
                    break
        return dna_chain_ids

    def detect(self, frame: int = 0) -> List[StrandPair]:
        """Run base-pair detection and populate duplexes.

        Parameters
        ----------
        frame : int, default 0
            Frame index used for geometric detection.
        """
        duplexes = self._detect_geometric(frame)
        if not duplexes:
            duplexes = self._detect_index_fallback()
        self.duplexes = duplexes

        if self.per_frame:
            for f in range(self.traj.n_frames):
                frame_dups = self._detect_geometric(f)
                if not frame_dups:
                    frame_dups = self._detect_index_fallback()
                self._frame_duplexes[f] = frame_dups

        return self.duplexes

    def _detect_geometric(self, frame_idx: int) -> List[StrandPair]:
        """Detect base pairs using barnaba Leontis-Westhof annotations."""
        try:
            import barnaba
        except ImportError:
            warnings.warn(
                "barnaba is not installed - geometric detection unavailable. "
                "Falling back to index-based detection."
            )
            return []

        single_frame = self.traj[frame_idx]

        _old_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            stackings, pairings, seq = barnaba.annotate_traj(single_frame)
        finally:
            sys.stderr = _old_stderr

        if not pairings or not pairings[0] or not pairings[0][0]:
            return []

        pairs_list = pairings[0][0]
        annotations = pairings[0][1]

        seq_to_residue: Dict[int, md.core.topology.Residue] = {}
        for i, entry in enumerate(seq):
            resname, resSeq, chain_idx = parse_barnaba_seq_entry(entry)
            for residue in self.traj.topology.chain(chain_idx).residues:
                if residue.resSeq == resSeq and residue.name == resname:
                    seq_to_residue[i] = residue
                    break

        chain_pair_groups: Dict[Tuple[int, int], list] = defaultdict(list)

        for pair_idxs, ann in zip(pairs_list, annotations):
            idx1, idx2 = pair_idxs
            if idx1 not in seq_to_residue or idx2 not in seq_to_residue:
                continue
            res1 = seq_to_residue[idx1]
            res2 = seq_to_residue[idx2]
            if res1.chain.index == res2.chain.index:
                continue
            if not is_dna_residue(res1) or not is_dna_residue(res2):
                continue
            if not are_complementary(res1.name, res2.name):
                continue
            c1, c2 = res1.chain.index, res2.chain.index
            if c1 > c2:
                c1, c2 = c2, c1
                res1, res2 = res2, res1
            chain_pair_groups[(c1, c2)].append((res1, res2))

        duplexes: List[StrandPair] = []
        for (c1, c2), raw_pairs in chain_pair_groups.items():
            seen: set = set()
            unique_pairs: List[Tuple] = []
            for res1, res2 in raw_pairs:
                key = (res1.index, res2.index)
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append((res1, res2))

            watson_chain, crick_chain = self._assign_strand_directions(c1, c2, unique_pairs)

            if watson_chain == c1:
                sorted_pairs = sorted(unique_pairs, key=lambda p: p[0].resSeq)
            else:
                sorted_pairs = sorted(
                    [(r2, r1) for r1, r2 in unique_pairs],
                    key=lambda p: p[0].resSeq,
                )

            watson_all = [
                r for r in self.traj.topology.chain(watson_chain).residues
                if is_dna_residue(r)
            ]
            crick_all = [
                r for r in self.traj.topology.chain(crick_chain).residues
                if is_dna_residue(r)
            ]
            strand_w = self.traj.topology.chain(watson_chain)
            strand_c = self.traj.topology.chain(crick_chain)
            bonded_doms = _split_bonded_domains(
                sorted_pairs, watson_all, crick_all,
                chain_watson=watson_chain, chain_crick=crick_chain,
                strand_watson=strand_w, strand_crick=strand_c,
                traj=self.traj,
            )
            aligned_doms = _split_aligned_domains(
                bonded_doms, watson_all, crick_all,
                chain_watson=watson_chain, chain_crick=crick_chain,
                strand_watson=strand_w, strand_crick=strand_c,
                traj=self.traj,
            )

            duplexes.append(StrandPair(
                chain_watson=watson_chain,
                chain_crick=crick_chain,
                pairs=sorted_pairs,
                method="geometric",
                bonded_domains=bonded_doms,
                aligned_domains=aligned_doms,
                strand_watson=strand_w,
                strand_crick=strand_c,
                traj=self.traj,
            ))

        return duplexes

    @staticmethod
    def _assign_strand_directions(c1: int, c2: int, pairs: List[Tuple]) -> Tuple[int, int]:
        """Decide which chain is Watson (5->3) and which is Crick (3->5).

        Uses the antiparallel convention: Watson resSeqs increase while
        Crick resSeqs decrease. Falls back to lower chain index = Watson
        when fewer than 2 pairs are available.
        """
        if len(pairs) < 2:
            return c1, c2
        c1_seqs = np.array([p[0].resSeq for p in pairs])
        c2_seqs = np.array([p[1].resSeq for p in pairs])
        order = np.argsort(c1_seqs)
        c2_sorted = c2_seqs[order]
        diffs = np.diff(c2_sorted)
        if np.sum(diffs < 0) >= np.sum(diffs > 0):
            return c1, c2
        else:
            return c2, c1

    def _detect_index_fallback(self) -> List[StrandPair]:
        """Pair residues positionally for reverse-complement chain pairs (MDNA-style).

        For every ordered pair of DNA chains with equal number of nucleotides
        whose sequences are reverse complements: pair res_A[i] with
        reversed(res_B)[i].
        """
        duplexes: List[StrandPair] = []
        for i, ci in enumerate(self.dna_chains):
            for cj in self.dna_chains[i + 1:]:
                res_i = [r for r in self.traj.topology.chain(ci).residues if is_dna_residue(r)]
                res_j = [r for r in self.traj.topology.chain(cj).residues if is_dna_residue(r)]
                if len(res_i) != len(res_j) or len(res_i) == 0:
                    continue
                res_j_rev = list(reversed(res_j))
                if all(are_complementary(a.name, b.name) for a, b in zip(res_i, res_j_rev)):
                    pairs = list(zip(res_i, res_j_rev))
                    strand_w = self.traj.topology.chain(ci)
                    strand_c = self.traj.topology.chain(cj)
                    bonded_doms = _split_bonded_domains(
                        pairs, res_i, res_j,
                        chain_watson=ci, chain_crick=cj,
                        strand_watson=strand_w, strand_crick=strand_c,
                        traj=self.traj,
                    )
                    aligned_doms = _split_aligned_domains(
                        bonded_doms, res_i, res_j,
                        chain_watson=ci, chain_crick=cj,
                        strand_watson=strand_w, strand_crick=strand_c,
                        traj=self.traj,
                    )
                    duplexes.append(StrandPair(
                        chain_watson=ci,
                        chain_crick=cj,
                        pairs=pairs,
                        method="index_fallback",
                        bonded_domains=bonded_doms,
                        aligned_domains=aligned_doms,
                        strand_watson=strand_w,
                        strand_crick=strand_c,
                        traj=self.traj,
                    ))
        return duplexes

    def get_frame_duplexes(self, frame: int) -> List[StrandPair]:
        """Return duplexes detected for a specific trajectory frame.

        Requires per_frame=True at construction time.
        """
        if frame in self._frame_duplexes:
            return self._frame_duplexes[frame]
        raise ValueError(
            f"No per-frame data for frame {frame}. "
            "Initialise DNAReader with per_frame=True and call detect()."
        )

    def summary(self) -> str:
        """Return a human-readable summary of the detected DNA topology."""
        lines: List[str] = []
        lines.append(
            f"Structure: {self.traj.n_frames} frame(s), "
            f"{self.traj.topology.n_chains} chain(s) total"
        )
        lines.append(f"DNA chains: {self.dna_chains}")
        for chain_idx in self.dna_chains:
            chain = self.traj.topology.chain(chain_idx)
            dna_res = [r for r in chain.residues if is_dna_residue(r)]
            seq = "".join(one_letter_code(r.name) for r in dna_res)
            lines.append(f"  Chain {chain_idx}: {len(dna_res)} nt - 5'-{seq}-3'")
        lines.append(f"\nDuplexes found: {len(self.duplexes)}")
        for dup in self.duplexes:
            watson_seq = "".join(one_letter_code(r.name) for r, _ in dup.pairs)
            crick_seq = "".join(one_letter_code(r.name) for _, r in dup.pairs)
            lines.append(f"  {dup}")
            lines.append(f"    Watson 5'-{watson_seq}-3'")
            lines.append(f"    Crick  3'-{crick_seq}-5'")
            lines.append(
                f"    Bonded domains: {len(dup.bonded_domains)}, "
                f"Aligned domains: {len(dup.aligned_domains)}"
            )
            for j, ad in enumerate(dup.aligned_domains):
                w_mask = ad.paired_watson_mask()
                c_mask = ad.paired_crick_mask()
                w_ann = "".join(
                    one_letter_code(r.name) if p else "."
                    for r, p in zip(ad.watson_residues, w_mask)
                )
                c_ann = "".join(
                    one_letter_code(r.name) if p else "."
                    for r, p in zip(ad.crick_residues, c_mask)
                )
                lines.append(
                    f"    AlignedDomain {j} "
                    f"({ad.n_bonded_domains} bonded domain(s), "
                    f"watson={ad.n_watson} nt, crick={ad.n_crick} nt):"
                )
                lines.append(f"      Watson 5'-{w_ann}-3'")
                lines.append(f"      Crick  3'-{c_ann}-5'")
        return "\n".join(lines)


def read_dna(
    path: str,
    topology: Optional[str] = None,
    per_frame: bool = False,
    frame: int = 0,
) -> DNAReader:
    """Load a structure or trajectory and detect DNA base pairing.

    Structures and trajectories are handled by one entry-point:

    - Structures (PDB, CIF/mmCIF): pass only path.
    - Trajectories (XTC, DCD, TRR, ...): pass path + topology.

    CIF/mmCIF requires gemmi (pip install gemmi).

    Parameters
    ----------
    path : str
        Input file path.
    topology : str, optional
        Topology file for trajectory formats.
    per_frame : bool, default False
        Re-detect pairing on every frame (use get_frame_duplexes to access).
    frame : int, default 0
        Frame used for initial detection.

    Returns
    -------
    DNAReader with duplexes already populated.
    """
    reader = DNAReader(path, topology=topology, per_frame=per_frame)
    reader.detect(frame=frame)
    return reader
