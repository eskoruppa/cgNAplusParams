#!/bin/env python3

from __future__ import annotations

import os
import numpy as np
import scipy, scipy.io
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import get_lapack_funcs

from ._so3 import so3
from .utils.assignment_utils import cgnaplus_name_assignment, nonphosphate_dof_map
from .utils.assignment_utils import dof_index
from .utils.crick_flip import apply_crick_flip
from .utils.transforms import _apply_transforms_optimized, _apply_transforms

CGNAPLUSPARAMS_PARAMSPATH = os.path.join(os.path.dirname(__file__), 'Parametersets/')

CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends"


# ──────────────────────────────────────────────────────────────────────
# Caches for the optimized implementation
# ──────────────────────────────────────────────────────────────────────
_cgnaplus_param_cache: dict[str, dict] = {}
_cgnaplus_band_struct_cache: dict[int, dict] = {}
_CGNAPLUS_BANDWIDTH = 41  # max |i-j| in assembled stiffness matrix

# LAPACK banded solver (resolved lazily on first use)
_cgnaplus_gbsv = None


def _get_cgnaplus_gbsv():
    """Lazily resolve and cache the LAPACK dgbsv function."""
    global _cgnaplus_gbsv
    if _cgnaplus_gbsv is None:
        _cgnaplus_gbsv, = get_lapack_funcs(
            ('gbsv',),
            (np.empty((1, 1), dtype=np.float64), np.empty(1, dtype=np.float64)),
        )
    return _cgnaplus_gbsv


def _preprocess_params(ps_name: str) -> dict:
    """Load a parameter set once and convert to fast-lookup plain dicts.

    Stores both the banded-format blocks (for solve) and padded versions
    (for direct LAPACK gbsv which needs an extra kl rows for pivoting).
    """
    if ps_name in _cgnaplus_param_cache:
        return _cgnaplus_param_cache[ps_name]

    ps = scipy.io.loadmat(CGNAPLUSPARAMS_PARAMSPATH + ps_name)
    u = _CGNAPLUS_BANDWIDTH
    bw = 2 * u + 1  # rows in standard band storage
    pw = 3 * u + 1  # rows in padded band storage (for gbsv)

    # Pre-allocate index arrays used in _to_band_block
    _arange42 = np.arange(42)
    _arange36 = np.arange(36)

    def _to_band_block(mat: np.ndarray, m: int, arange_m: np.ndarray) -> np.ndarray:
        """Convert a dense m×m block into banded-format (2u+1, m)."""
        bb = np.zeros((bw, m))
        for p in range(m):
            bb[u + p - arange_m, arange_m] = mat[p, :]
        return bb

    def _to_padded_band_block(band: np.ndarray, m: int) -> np.ndarray:
        """Pad a band block (2u+1, m) to (3u+1, m) for LAPACK gbsv."""
        pb = np.zeros((pw, m))
        pb[u:, :] = band
        return pb

    result: dict = {}
    for cat, m, bkey, skey in [
        ('end5', 36, 'stiff_end5', 'sigma_end5'),
        ('end3', 36, 'stiff_end3', 'sigma_end3'),
        ('int',  42, 'stiff_int',  'sigma_int'),
    ]:
        arange_m = _arange36 if m == 36 else _arange42
        band_dict: dict[str, np.ndarray] = {}
        padded_dict: dict[str, np.ndarray] = {}
        sigma_dict: dict[str, np.ndarray] = {}
        for name in ps[bkey].dtype.names:
            B = np.ascontiguousarray(ps[bkey][name][0][0][:m, :m])
            bb = _to_band_block(B, m, arange_m)
            band_dict[name] = bb
            padded_dict[name] = np.asfortranarray(_to_padded_band_block(bb, m))
            sigma_dict[name] = ps[skey][name][0][0][:m].ravel().copy()
        result[f'stiff_{cat}_band'] = band_dict
        result[f'stiff_{cat}_pad'] = padded_dict
        result[f'sigma_{cat}'] = sigma_dict

    _cgnaplus_param_cache[ps_name] = result
    return result


def _get_band_struct(nbp: int) -> dict:
    """Precompute / cache the CSC extraction structure for a given nbp.

    The extraction operates on the *standard* band portion of the padded
    array, i.e. rows u … 3u of the (3u+1, N) array.
    """
    if nbp in _cgnaplus_band_struct_cache:
        return _cgnaplus_band_struct_cache[nbp]

    u = _CGNAPLUS_BANDWIDTH
    N = 24 * nbp - 18

    # Band rows in the standard (2u+1) representation → 0 .. 2u
    k_arr = np.arange(2 * u + 1)
    j_arr = np.arange(N)
    k_grid, j_grid = np.meshgrid(k_arr, j_arr, indexing='ij')
    i_grid = j_grid - u + k_grid          # actual matrix row
    valid = (i_grid >= 0) & (i_grid < N)

    rows_valid = i_grid[valid]
    cols_valid = j_grid[valid]

    # Sort by (col, row) for native CSC order
    sort_idx = np.lexsort((rows_valid, cols_valid))
    rows_sorted = rows_valid[sort_idx].astype(np.int32)

    # Build indptr from column counts
    _, counts = np.unique(cols_valid[sort_idx], return_counts=True)
    indptr = np.zeros(N + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)

    # Extraction indices into the *padded* array (row offset by u)
    k_padded_sorted = (k_grid[valid] + u)[sort_idx]  # shift k by u for padded
    j_sorted = j_grid[valid][sort_idx]

    # Precompute flat index for F-order (column-major) extraction via np.take
    nrows_padded = 3 * u + 1
    flat_idx_f = j_sorted.astype(np.int64) * nrows_padded + k_padded_sorted.astype(np.int64)

    struct = {
        'rows': rows_sorted,
        'indptr': indptr,
        'k_idx': k_padded_sorted,
        'j_idx': j_sorted,
        'flat_idx_f': flat_idx_f,
        'N': N,
    }
    _cgnaplus_band_struct_cache[nbp] = struct
    return struct


def constructSeqParms(
    sequence: str, ps_name: str
) -> tuple[np.ndarray, csc_matrix]:
    """Drop-in replacement for constructSeqParms – much faster thanks to:

    * Cached parameter loading (avoids repeated scipy.io.loadmat)
    * Precomputed banded-format blocks per dinucleotide
    * Direct LAPACK banded solve (gbsv) instead of sparse LU
    * Precomputed CSC structure for the returned stiffness matrix
    """

    params = _preprocess_params(ps_name)
    s_seq = _seq_edit(sequence)
    nbp = len(s_seq.strip())

    if nbp <= 3:
        raise ValueError(
            f'Sequence length must be greater than or equal to 4. '
            f'Current length is {nbp}.'
        )

    u = _CGNAPLUS_BANDWIDTH
    N = 24 * nbp - 18

    # ── Allocate padded banded matrix (3u+1, N) F-order and sigma ────
    ab = np.zeros((3 * u + 1, N), dtype=np.float64, order='F')
    s = np.zeros(N, dtype=np.float64)

    pad_end5 = params['stiff_end5_pad']
    pad_int  = params['stiff_int_pad']
    pad_end3 = params['stiff_end3_pad']
    sig_end5 = params['sigma_end5']
    sig_int  = params['sigma_int']
    sig_end3 = params['sigma_end3']

    # ── 5ʼ end ───────────────────────────────────────────────────────
    dinuc = s_seq[0:2]
    ab[:, 0:36] += pad_end5[dinuc]
    s[0:36] = sig_end5[dinuc]

    # ── Interior blocks ──────────────────────────────────────────────
    for i in range(2, nbp - 1):
        dinuc = s_seq[i - 1 : i + 1]
        di = 24 * (i - 2) + 18
        ab[:, di : di + 42] += pad_int[dinuc]
        s[di : di + 42] += sig_int[dinuc]

    # ── 3ʼ end ───────────────────────────────────────────────────────
    dinuc = s_seq[nbp - 2 : nbp]
    di = 24 * (nbp - 3) + 18
    ab[:, di : di + 36] += pad_end3[dinuc]
    s[N - 36 : N] += sig_end3[dinuc]

    # ── Build CSC stiffness matrix BEFORE the destructive solve ──────
    bstruct = _get_band_struct(nbp)
    csc_data = np.take(ab.ravel(order='F'), bstruct['flat_idx_f'])
    stiff = csc_matrix(
        (csc_data, bstruct['rows'], bstruct['indptr']),
        shape=(N, N),
        copy=False,
    )

    # ── Solve the banded system via direct LAPACK call ───────────────
    #    ab is already F-order; gbsv overwrites it in-place.
    gbsv = _get_cgnaplus_gbsv()
    _, _, ground_state, info = gbsv(u, u, ab, s, overwrite_ab=True, overwrite_b=True)

    return ground_state, stiff


def constructSeqParms_legacy(sequence: str ,ps_name: str) -> tuple[np.ndarray, csc_matrix]:

    params_path = CGNAPLUSPARAMS_PARAMSPATH
    ps = scipy.io.loadmat(params_path + ps_name)

	#### Following loop take every input sequence and construct shape and stiff matrix ###
    s_seq = _seq_edit(sequence)
    nbp = len(s_seq.strip())
    N = 24*nbp-18

	#### Initialise the sigma vector ###		
    s = np.zeros((N,1))

    #### Error report if sequence provided is less than 2 bp #### 

    if nbp <= 3:
        raise ValueError(f'Sequence length must be greater than or equal to 4. Current length is {nbp}.')

    data,row,col = {},{},{}
    
    ### 5' end #### 
    tmp_ind = np.nonzero(ps['stiff_end5'][s_seq[0:2]][0][0][0:36,0:36])
    row[0],col[0] = tmp_ind[0][:],tmp_ind[1][:]
    data[0] = ps['stiff_end5'][s_seq[0:2]][0][0][row[0],col[0]]
    
    s[0:36] = ps['sigma_end5'][s_seq[0:2]][0][0][0:36]
    #### interior blocks  ###
    for i in range(2,nbp-1):
        tmp_ind = np.nonzero(ps['stiff_int'][s_seq[i-1:i+1]][0][0][0:42, 0:42])
        data[i-1] = ps['stiff_int'][s_seq[i-1:i+1]][0][0][tmp_ind[0][:], tmp_ind[1][:]]
        
        di = 24*(i-2)+18
        row[i-1] = tmp_ind[0][:]+np.ones((1,np.size(tmp_ind[0][:])))*di
        col[i-1] = tmp_ind[1][:]+np.ones((1,np.size(tmp_ind[1][:])))*di
        
        s[di:di+42] = np.add(s[di:di+42],ps['sigma_int'][s_seq[i-1:i+1]][0][0][0:42])
        
    #### 3' end ####
    tmp_ind = np.nonzero(ps['stiff_end3'][s_seq[nbp-2:nbp]][0][0][0:36, 0:36])
    data[nbp-1] = ps['stiff_end3'][s_seq[nbp-2:nbp]][0][0][tmp_ind[0][:], tmp_ind[1][:]]
    
    di = 24*(nbp-3)+18
    row[nbp-1] = tmp_ind[0][:]+np.ones((1,np.size(tmp_ind[0][:])))*di
    col[nbp-1] = tmp_ind[1][:]+np.ones((1,np.size(tmp_ind[1][:])))*di
    s[N-36:N] = s[N-36:N] + ps['sigma_end3'][s_seq[nbp-2:nbp]][0][0][0:36]
    
    tmp = list(row.values())
    row = np.concatenate(tmp,axis=None)
    
    tmp = list(col.values())
    col = np.concatenate(tmp,axis=None)

    tmp = list(data.values())
    data = np.concatenate(tmp,axis=None)
    

    #### Create the sparse Stiffness matrix from data,row_ind,col_ind  ###
    stiff =  csc_matrix((data, (row,col)), shape =(N,N))	

    #### Groudstate calculation ####
    ground_state = spsolve(stiff, s) 

    return ground_state,stiff

def _seq_edit(seq):
	s = seq.upper()
	while s.rfind('_')>0:
		if s[s.rfind('_')-1].isdigit():
			print("Please write the input sequence correctly. Two or more _ can't be put consequently. You can use the brackets. i.e. A_2_2 can be written as [A_2]_2")
			exit()
		if s[s.rfind('_')-1] != ']':
			a = int(_mult(s))
			s = s[:s.rfind('_')-1]+ s[s.rfind('_')-1]*a +  s[s.rfind('_')+1+len(str((a))):]
		if s[s.rfind('_')-1] == ']':
			end,start = _finder(s)
			ka=(2,len(start))
			h=np.zeros(ka)
			for i in range(len(start)):
				h[0][i] = start[i]
				h[1][i] = end[start[i]]	
			ss=  int(max(h[1]))
			ee=  int(h[0][np.argmax(h[1])])
			a = int(_mult(s))
			s =  s[0:ee] + s[ee+1:ss]*a + s[ss+2+len(str((a))):] 
	return s	


def _finder(seq):
	istart = []  
	end = {}
	start = []
	for i, c in enumerate(seq):
		if c == '[':
			istart.append(i)
			start.append(i)
		if c == ']':
			try:
				end[istart.pop()] = i
			except IndexError:
				print('Too many closing parentheses')
	if istart:  # check if stack is empty afterwards
		print('Too many opening parentheses')
	return end, start


def _mult(seq):
	i =seq.rfind('_') 
	if seq[i+1].isdigit():
		a = seq[i+1]
		if seq[i+2].isdigit():
			a = a + seq[i+2]
			if seq[i+3].isdigit():
				a = a + seq[i+3]
				if seq[i+4].isdigit():
					a = a + seq[i+4]
					if seq[i+5].isdigit():
						a = a + seq[i+5]
	return a


########################################################################################################################
# CGNAPlusParams
########################################################################################################################

class CGNAPlusParams:

    def __init__(
            self, 
            sequence: str, 
            parameter_set_name: str = CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME,
            euler_definition: bool = True,
            group_split: bool = True,
            translations_in_nm: bool = True,
            aligned_strands: bool = False,
            include_stiffness: bool = True,
            remove_factor_five: bool = True,
            use_optimized: bool = True,
            ):

        self._parameter_set_name = parameter_set_name
        self._euler_definition = euler_definition
        self._group_split = group_split
        self._translations_in_nm = translations_in_nm
        self._aligned_strands = aligned_strands
        self._include_stiffness = include_stiffness
        self._remove_factor_five = remove_factor_five

        self._sequence = sequence 

        self._stiffmat_initialized = False
        self._gs_initialized = False
        self._use_optimized = use_optimized


    def _init_params(
        self, 
        sequence: str | None,
        include_stiffness: bool | None = None,
    ) -> None:
        
        if include_stiffness is not None:
            self.set_include_stiffness(include_stiffness)
        
        if sequence is not None:
            self.set_sequence(sequence)

        if self._gs_initialized and (not self._include_stiffness or self._stiffmat_initialized):
            return

        gs, stiff = constructSeqParms(self._sequence, self._parameter_set_name)

        param_names = cgnaplus_name_assignment(self._sequence)
        nonphosphate_map = nonphosphate_dof_map(self._sequence, param_names=param_names)

        if self._use_optimized:
            gs, stiff = _apply_transforms_optimized(
                gs, stiff, nonphosphate_map, param_names,
                remove_factor_five=self._remove_factor_five,
                translations_in_nm=self._translations_in_nm,
                euler_definition=self._euler_definition,
                group_split=self._group_split,
                include_stiffness=self._include_stiffness,
                aligned_strands=self._aligned_strands,
            )
        else:
            gs, stiff = _apply_transforms(
                gs, stiff, nonphosphate_map, param_names,
                remove_factor_five=self._remove_factor_five,
                translations_in_nm=self._translations_in_nm,
                euler_definition=self._euler_definition,
                group_split=self._group_split,
                include_stiffness=self._include_stiffness,
                aligned_strands=self._aligned_strands,
            )

        self._gs = gs
        self._param_names = param_names
        self._gs_initialized = True

        if self._include_stiffness:
            self._stiffmat = stiff
            self._stiffmat_initialized = True

        return

    ########################################################################################################################
    # Setters
    ########################################################################################################################

    def set_sequence(self, sequence: str) -> None:
        if sequence != self._sequence:
            self._sequence = sequence
            self._gs_initialized = False
            self._stiffmat_initialized = False

    def set_parameter_set_name(self, parameter_set_name: str) -> None:
        if parameter_set_name != self._parameter_set_name:
            self._parameter_set_name = parameter_set_name
            self._gs_initialized = False
            self._stiffmat_initialized = False

    def set_euler_definition(self, euler_definition: bool) -> None:
        if euler_definition != self._euler_definition:
            self._euler_definition = euler_definition
            self._gs_initialized = False
            self._stiffmat_initialized = False

    def set_group_split(self, group_split: bool) -> None:
        if group_split != self._group_split:
            self._group_split = group_split
            self._gs_initialized = False
            self._stiffmat_initialized = False
    
    def set_translations_in_nm(self, translations_in_nm: bool) -> None:
        if translations_in_nm != self._translations_in_nm:
            self._translations_in_nm = translations_in_nm
            self._gs_initialized = False
            self._stiffmat_initialized = False
    
    def set_aligned_strands(self, aligned_strands: bool) -> None:
        if aligned_strands != self._aligned_strands:
            self._aligned_strands = aligned_strands
            self._gs_initialized = False
            self._stiffmat_initialized = False  

    def set_include_stiffness(self, include_stiffness: bool) -> None:
        self._include_stiffness = include_stiffness

    def set_remove_factor_five(self, remove_factor_five: bool) -> None:
        if remove_factor_five != self._remove_factor_five:
            self._remove_factor_five = remove_factor_five
            self._gs_initialized = False
            self._stiffmat_initialized = False

    ########################################################################################################################
    # Properties
    ########################################################################################################################

    @property
    def sequence(self) -> str:
        return self._sequence
    
    @property
    def parameter_set_name(self) -> str:
        return self._parameter_set_name

    @property
    def euler_definition(self) -> bool:
        return self._euler_definition
    
    @property
    def rotation_definition(self) -> str:
        if self._euler_definition:
            return 'euler'
        else:
            return 'cayley'
    
    @property
    def group_split(self) -> bool:
        return self._group_split
    
    @property
    def splitting_definition(self) -> str:
        if self._group_split:
            return 'group'
        else:
            return 'algebra'
    
    @property
    def translations_in_nm(self) -> bool:
        return self._translations_in_nm
    
    @property
    def translation_units(self) -> str:
        if self._translations_in_nm:
            return 'nm'
        else:
            return 'A'

    @property
    def aligned_strands(self) -> bool:
        return self._aligned_strands
    
    @property
    def remove_factor_five(self) -> bool:
        return self._remove_factor_five
    
    @property
    def gs(self) -> np.ndarray:
        if not self._gs_initialized:
            self._init_params(sequence=None, include_stiffness=None)
        return self._gs
    
    @property
    def groundstate(self) -> np.ndarray:
        return self.gs
    
    @property
    def stiffmat(self) -> np.ndarray:
        if not self._stiffmat_initialized:
            self._init_params(sequence=None, include_stiffness=True)
        return self._stiffmat
    
    @property
    def stiffness(self) -> np.ndarray:
        return self.stiffmat
    
    @property
    def stiffness_matrix(self) -> np.ndarray:
        return self.stiffmat

    @property
    def param_names(self) -> list[str]:
        if not self._gs_initialized:
            self._init_params(sequence=None, include_stiffness=None)
        return self._param_names

    def gs_param(self, name: str) -> np.ndarray:
        idx = dof_index(name, self.param_names)
        if idx is None:
            if name.upper() in self.param_names:
                raise ValueError(f"Parameter name '{name}' not found in param_names. Did you mean '{name.upper()}'?")
            raise ValueError(f"Parameter name '{name}' not found in param_names.")
        return self.groundstate[idx]
    
    def param_by_name(self, name: str) -> np.ndarray:
        return self.gs_param(name)
    
    def param_index(self, name: str) -> int | None:
        return dof_index(name, self.param_names)

    def conf(self, dynamic: np.ndarray | None = None):
        """Return a :class:`~cgnaplus_conf.CGNAplusConf` for the current
        sequence and parameters.

        Parameters
        ----------
        dynamic : optional deformation array added on top of the ground state
            when constructing the frame chain.

        Returns
        -------
        CGNAplusConf
            Lazy object exposing ``bp_poses``, ``watson_base_poses``,
            ``crick_base_poses``, ``watson_phosphate_poses``,
            ``crick_phosphate_poses``, and ``named_poses``.

        Note
        ----
        Requires ``cgnaplus_conf.CGNAplusConf`` to accept a ``CGNAPlusParams``
        instance as its first argument (see planned update to
        ``cgnaplus_conf.py``).
        """
        from .cgnaplus_conf import CGNAPlusConf
        return CGNAPlusConf.from_params(self, dynamic=dynamic)

    def to_rbp(self, rotations_only: bool = False):
        """Return an :class:`~rbp.RBP` instance marginalised to inter-base-pair
        DOFs only.

        If ``self`` is already initialised (``gs`` has been computed), the RBP
        parameters are obtained by directly marginalising the already-transformed
        groundstate and stiffness matrix.  This is mathematically equivalent to
        recomputing from scratch because all coordinate-definition transforms
        (scaling, Cayley→Euler, algebra→group) are block-diagonal congruences
        that commute with the Schur-complement marginalisation.

        If ``self`` has not been initialised yet, a fresh :class:`~rbp.RBP`
        instance is constructed with the same flags and will compute lazily.

        Parameters
        ----------
        rotations_only : if True, further marginalise to rotational DOFs only.

        Returns
        -------
        RBP
        """
        from .rbp_params import RBPParams
        if self._gs_initialized:
            return RBPParams.from_cgnaplus(self, rotations_only=rotations_only)
        return RBPParams(
            self._sequence,
            parameter_set_name=self._parameter_set_name,
            euler_definition=self._euler_definition,
            group_split=self._group_split,
            translations_in_nm=self._translations_in_nm,
            include_stiffness=self._include_stiffness,
            remove_factor_five=self._remove_factor_five,
            rotations_only=rotations_only,
        )
    
    ########################################################################################################################
    # Private Methods
    ########################################################################################################################

    def _set_precomputed(
        self,
        gs: np.ndarray,
        stiff,
        param_names: list[str],
    ) -> None:
        """Directly inject pre-computed (already transformed) cached values,
        bypassing ``_init_params``.

        Parameters
        ----------
        gs : ndarray, shape (N, 6) — groundstate in the coordinate definition
            described by the flags stored on this instance.
        stiff : dense ndarray or None — stiffness matrix in the same definition.
            Pass ``None`` when stiffness is not available.
        param_names : list[str] — DOF name list matching the rows/columns of
            ``gs`` and ``stiff``.
        """
        self._gs = gs
        self._param_names = param_names
        self._gs_initialized = True
        if stiff is not None:
            self._stiffmat = stiff
            self._stiffmat_initialized = True
            self._include_stiffness = True
        else:
            self._stiffmat_initialized = False
            self._include_stiffness = False


########################################################################################################################
# Legacy functions — kept for testing / backward compatibility
########################################################################################################################

def cgnaplusparams(
    sequence: str, 
    parameter_set_name: str = CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME,
    euler_definition: bool = True,
    group_split: bool = True,
    remove_factor_five: bool = True,
    translations_in_nm: bool = True,
    include_stiffness: bool = True,
    aligned_strands: bool = False, 
) -> CGNAPlusParams:
    
    return CGNAPlusParams(
        sequence=sequence,
        parameter_set_name=parameter_set_name,
        euler_definition=euler_definition,
        group_split=group_split,
        remove_factor_five=remove_factor_five,
        translations_in_nm=translations_in_nm,
        include_stiffness=include_stiffness,
        aligned_strands=aligned_strands
    )

def cgnaplus_params(
    sequence: str, 
    parameter_set_name: str = CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME,
    euler_definition: bool = True,
    group_split: bool = True,
    remove_factor_five: bool = True,
    translations_in_nm: bool = True,
    include_stiffness: bool = True,
    aligned_strands: bool = False, 
) -> CGNAPlusParams:
    
    return CGNAPlusParams(
        sequence=sequence,
        parameter_set_name=parameter_set_name,
        euler_definition=euler_definition,
        group_split=group_split,
        remove_factor_five=remove_factor_five,
        translations_in_nm=translations_in_nm,
        include_stiffness=include_stiffness,
        aligned_strands=aligned_strands
    )


def cgnaplusparams_legacy(
    sequence: str, 
    parameter_set_name: str = CGNAPLUSPARAMS_DEFAULT_PARAMETER_SET_NAME,
    euler_definition: bool = True,
    group_split: bool = True,
    remove_factor_five: bool = True,
    translations_in_nm: bool = True,
    include_stiffness: bool = True,
    aligned_strands: bool = False
    ) -> dict[str, np.ndarray | bool | str]:
    
    gs, stiff = constructSeqParms(sequence, parameter_set_name)

    param_names = cgnaplus_name_assignment(sequence)
    nonphosphate_map = nonphosphate_dof_map(sequence, param_names=param_names)

    if aligned_strands:
        gs, stiff = apply_crick_flip(
            gs,
            stiff if include_stiffness else None,
            param_names,
        )

    if remove_factor_five:
        factor = 5
        gs   = so3.array_conversion(gs,1./factor,block_dim=6,dofs=[0,1,2])
        if include_stiffness:
            stiff = so3.array_conversion(stiff,factor,block_dim=6,dofs=[0,1,2])
    if translations_in_nm:
        factor = 10
        gs   = so3.array_conversion(gs,1./factor,block_dim=6,dofs=[3,4,5])
        if include_stiffness:
            stiff = so3.array_conversion(stiff,factor,block_dim=6,dofs=[3,4,5])
    gs = so3.statevec2vecs(gs,vdim=6) 

    if euler_definition:
        # cayley2euler_stiffmat requires gs in cayley definition
        if include_stiffness:
            stiff = so3.se3_cayley2euler_stiffmat(gs,stiff,rotation_first=True)
        gs = so3.se3_cayley2euler(gs)

    if group_split:
        if not euler_definition:
            raise ValueError('The group_split option requires euler_definition to be set!')
        if include_stiffness:
            
            gs,stiff = so3.algebra2group_params(gs, stiff, rotation_first=True, translation_as_midstep=nonphosphate_map, optimized=True) 
        else:
            gs = so3.midstep2triad(gs)

    result = {
        "gs": gs,
        "sequence": sequence,
        "translations_in_nm": translations_in_nm,
        "euler_definition": euler_definition,
        "group_split": group_split,
        "remove_factor_five": remove_factor_five,
        "param_names": param_names,
        "aligned_strands": aligned_strands
    }
    if include_stiffness:
        result["stiffmat"] = stiff

    return result