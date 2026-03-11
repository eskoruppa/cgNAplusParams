# cgNAplusParams

Python interface to the [cgDNA+](#ref-cgnaplus) and cgRNA+ coarse-grained nucleic acid model parameter sets. Given a DNA or RNA sequence it returns the sequence-dependent groundstate rigid-body parameters and stiffness matrix, and can build a full SE(3) pose chain for 3-D structure visualisation.

## What it does

- **`constructSeqParms(sequence, parameter_set_name)`** — assembles the cgDNA+/cgRNA+ groundstate vector and stiffness matrix for an arbitrary sequence.
- **`cgnaplus2rbp(sequence, ...)`** — convenience wrapper that returns the base-pair-step rigid-body parameters (`gs`) and optionally the stiffness matrix in physical units.
- **`build_conf(rbp_params, ...)`** — converts a sequence of SE(3) rigid-body parameters into a chain of 4×4 homogeneous transformation matrices (one per base pair).
- **`gen_pdb(...)` / `visualize_chimerax(...)`** — export a coarse-grained structure as PDB or a ChimeraX session.

## Installation

Install the two git-hosted dependencies first, then the package itself:

```bash
pip install git+https://github.com/eskoruppa/SO3.git
pip install git+https://github.com/eskoruppa/pyConDec.git
pip install git+https://github.com/eskoruppa/cgNAplusParams.git
```

## Quick start

```python
import numpy as np
from cgnaplusparams import cgnaplus2rbp, build_conf

sequence = "ATACGCTTGCATGC"

# Groundstate rigid-body parameters and stiffness matrix
result = cgnaplus2rbp(sequence,include_stiffness=True)
gs     = result["gs"]      # (nbp-1, 6) rigid-body parameters
stiff  = result["stiff"]   # (6*(nbp-1), 6*(nbp-1)) stiffness matrix

# 3-D pose chain  →  (nbp, 4, 4) SE(3) matrices
conf = build_conf(gs)
```

### Available parameter sets

| `parameter_set_name` | System |
|---|---|
| `Prmset_cgDNA+_CGF_10mus_int_12mus_ends` *(default)* | DNA, TIP3P/JC/BSC1 |
| `cgDNA+_Curves_BSTJ_10mus_FS` | DNA, Curves+ frames |
| `Prmset_cgRNA+_OL3_CGF_10mus_int_12mus_ends` | RNA |
| `cgHYB+_CGF_OL3_BSC1_10mus_FS_GC_ends` | DNA–RNA hybrid |
| `Di_hmethyl_methylated-hemi_combine` | Methylated DNA |

## References

1. <a name="ref-cgnaplus"></a>**R. Sharma, J. H. Maddocks, and others**, *cgDNA+: A sequence-dependent coarse-grain model of double-stranded DNA*, (2023).

## License

GNU General Public License v2 — see [LICENSE](LICENSE).