#!/bin/env python3

from __future__ import annotations

import os
import numpy as np
import scipy, scipy.io
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import get_lapack_funcs

CGDNAPARAMSPATH = os.path.join(os.path.dirname(__file__), 'Parametersets/')

# ──────────────────────────────────────────────────────────────────────
# Caches for the optimized implementation
# ──────────────────────────────────────────────────────────────────────
_param_cache: dict[str, dict] = {}
_band_struct_cache: dict[int, dict] = {}
_BANDWIDTH = 41  # max |i-j| in assembled stiffness matrix

# LAPACK banded solver (resolved lazily on first use)
_gbsv = None


def _get_gbsv():
    """Lazily resolve and cache the LAPACK dgbsv function."""
    global _gbsv
    if _gbsv is None:
        _gbsv, = get_lapack_funcs(
            ('gbsv',),
            (np.empty((1, 1), dtype=np.float64), np.empty(1, dtype=np.float64)),
        )
    return _gbsv


def _preprocess_params(ps_name: str) -> dict:
    """Load a parameter set once and convert to fast-lookup plain dicts.

    Stores both the banded-format blocks (for solve) and padded versions
    (for direct LAPACK gbsv which needs an extra kl rows for pivoting).
    """
    if ps_name in _param_cache:
        return _param_cache[ps_name]

    ps = scipy.io.loadmat(CGDNAPARAMSPATH + ps_name)
    u = _BANDWIDTH
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

    _param_cache[ps_name] = result
    return result


def _get_band_struct(nbp: int) -> dict:
    """Precompute / cache the CSC extraction structure for a given nbp.

    The extraction operates on the *standard* band portion of the padded
    array, i.e. rows u … 3u of the (3u+1, N) array.
    """
    if nbp in _band_struct_cache:
        return _band_struct_cache[nbp]

    u = _BANDWIDTH
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
    _band_struct_cache[nbp] = struct
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

    u = _BANDWIDTH
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
    gbsv = _get_gbsv()
    _, _, ground_state, info = gbsv(u, u, ab, s, overwrite_ab=True, overwrite_b=True)

    return ground_state, stiff

def constructSeqParms_original(sequence: str ,ps_name: str) -> tuple[np.ndarray, csc_matrix]:

    params_path = CGDNAPARAMSPATH
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
