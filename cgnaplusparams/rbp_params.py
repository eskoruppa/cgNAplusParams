#!/bin/env python3

from __future__ import annotations

import numpy as np
import scipy as sp
from ._so3 import so3

from .cgnaplus_params import CGNAPlusParams, _apply_transforms
from .cgnaplus_params import constructSeqParms
from .utils.assignment_utils import cgnaplus_name_assignment
from .utils.assignment_utils import INTER_BP_PARAM_NAME


class RBPParams(CGNAPlusParams):
    """Rigid-base-pair (inter-bp only) model derived from cgNA+.

    ``RBPParams`` inherits the full lazy-init / transform pipeline from
    :class:`~cgnaplus.CGNAPlusParams` but overrides :meth:`_init_params` to
    first marginalise the full cgNA+ parameter set to inter-base-pair
    (``Y``-type) DOFs only via Schur complement, and then applies the
    same coordinate-definition transforms.

    Parameters
    ----------
    sequence : DNA sequence string (upper-case, Watson strand 5'→3')
    parameter_set_name : name of the ``.mat`` parameter set to load
    euler_definition : convert Cayley → Euler rotation parameterisation
    group_split : convert algebra → group parameters (midstep convention)
    translations_in_nm : scale translations from Ångström to nm
    include_stiffness : whether to also compute the stiffness matrix
    remove_factor_five : rescale rotational DOFs by 1/5
    rotations_only : further marginalise to rotational DOFs only after the
        inter-bp marginalisation
    """

    def __init__(
            self,
            sequence: str,
            parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
            euler_definition: bool = True,
            group_split: bool = True,
            translations_in_nm: bool = True,
            include_stiffness: bool = True,
            remove_factor_five: bool = True,
            rotations_only: bool = False,
            ):

        # aligned_strands is not a meaningful concept for RBPParams; fix to False
        super().__init__(
            sequence=sequence,
            parameter_set_name=parameter_set_name,
            euler_definition=euler_definition,
            group_split=group_split,
            translations_in_nm=translations_in_nm,
            aligned_strands=False,
            include_stiffness=include_stiffness,
            remove_factor_five=remove_factor_five,
        )
        self._rotations_only = rotations_only

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_aligned_strands(self, aligned_strands: bool) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "'aligned_strands' is not applicable to the RBP model."
        )

    def set_rotations_only(self, rotations_only: bool) -> None:
        if rotations_only != self._rotations_only:
            self._rotations_only = rotations_only
            self._gs_initialized = False
            self._stiffmat_initialized = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rotations_only(self) -> bool:
        return self._rotations_only


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

        full_names = cgnaplus_name_assignment(self._sequence)
        select_names = [INTER_BP_PARAM_NAME + "*"]

        # --- Schur-complement marginalisation to inter-bp DOFs ---
        if self._include_stiffness:
            stiff = so3.matrix_marginal_assignment(stiff, select_names, full_names, block_dim=6)
            if sp.sparse.issparse(stiff):
                stiff = stiff.toarray()
        gs = so3.vector_marginal_assignment(gs, select_names, full_names, block_dim=6)

        param_names = [n for n in full_names if n.startswith(INTER_BP_PARAM_NAME)]

        gs, stiff = _apply_transforms(
            gs, stiff, True, param_names,
            remove_factor_five=self._remove_factor_five,
            translations_in_nm=self._translations_in_nm,
            euler_definition=self._euler_definition,
            group_split=self._group_split,
            include_stiffness=self._include_stiffness,
            aligned_strands=False,
        )

        if self._rotations_only:
            gs_flat = so3.vecs2statevec(gs)
            gs_flat = so3.vector_rotmarginal(gs_flat)
            gs = so3.statevec2vecs(gs_flat, vdim=3)
            if self._include_stiffness and stiff is not None:
                stiff = so3.matrix_rotmarginal(stiff)

        self._gs = gs
        self._param_names = param_names
        self._gs_initialized = True

        if self._include_stiffness:
            self._stiffmat = stiff
            self._stiffmat_initialized = True

        return

    # ------------------------------------------------------------------
    # conf() factory
    # ------------------------------------------------------------------

    def conf(self, dynamic: np.ndarray | None = None):
        """Return an :class:`~rbp_conf.RBPConf` for the current parameters.

        Note
        ----
        Requires ``rbp_conf.RBPConf`` to accept an ``RBPParams`` instance as its
        first argument (see planned update to ``rbp_conf.py``).
        """
        from .rbp_conf import RBPConf
        return RBPConf(self, dynamic=dynamic)

    # ------------------------------------------------------------------
    # Class-method factory (mirrors CGNAPlusParams.init)
    # ------------------------------------------------------------------

    @classmethod
    def init(  # type: ignore[override]
        cls,
        sequence: str,
        parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
        euler_definition: bool = True,
        group_split: bool = True,
        remove_factor_five: bool = True,
        translations_in_nm: bool = True,
        include_stiffness: bool = True,
        rotations_only: bool = False,
    ) -> RBPParams:

        return cls(
            sequence,
            parameter_set_name=parameter_set_name,
            euler_definition=euler_definition,
            group_split=group_split,
            translations_in_nm=translations_in_nm,
            include_stiffness=include_stiffness,
            remove_factor_five=remove_factor_five,
            rotations_only=rotations_only,
        )

    @classmethod
    def from_cgnaplus(cls, cgnap, rotations_only: bool = False) -> RBPParams:
        """Construct an :class:`RBPParams` by directly marginalising an already-
        transformed :class:`~cgnaplus.CGNAPlusParams` instance.

        All coordinate-definition transforms (unit scaling, Cayley→Euler,
        algebra→group) are block-diagonal congruences that commute with the
        Schur-complement marginalisation to Y-type DOFs. 
        Therefore marginalising the already-converted groundstate and stiffness
        is mathematically identical to recomputing from the raw Cayley
        parameters — but far cheaper because ``constructSeqParms`` and all
        transform steps are skipped.

        Parameters
        ----------
        cgnap : CGNAPlusParams
            Source instance.  Must have ``_gs_initialized = True`` (i.e.
            ``cgnap.gs`` must have been accessed at least once).
        rotations_only : if True, further marginalise to rotational DOFs only
            after the inter-bp marginalisation.

        Returns
        -------
        RBPParams
            Instance whose coordinate-definition flags match ``cgnap``
            (except ``aligned_strands``, which is always ``False`` for RBPParams).

        Raises
        ------
        RuntimeError
            If ``cgnap`` has not been initialised yet.
        """
        if not cgnap._gs_initialized:
            raise RuntimeError(
                "CGNAPlusParams instance has not been initialised yet.  Access "
                "cgnap.gs at least once before calling from_cgnaplus()."
            )

        full_names = cgnap.param_names
        select_names = [INTER_BP_PARAM_NAME + "*"]

        # --- Marginalise groundstate (already in (N, 6) form) ---
        gs_flat = so3.vecs2statevec(cgnap.gs)
        gs_rbp_flat = so3.vector_marginal_assignment(
            gs_flat, select_names, full_names, block_dim=6,
        )
        gs_rbp = so3.statevec2vecs(gs_rbp_flat, vdim=6)

        rbp_param_names = [n for n in full_names if n.startswith(INTER_BP_PARAM_NAME)]

        # --- Marginalise stiffness (Schur complement) if available ---
        stiff_rbp = None
        if cgnap._stiffmat_initialized:
            stiff_rbp = so3.matrix_marginal_assignment(
                cgnap.stiffmat, select_names, full_names, block_dim=6,
            )
            if sp.sparse.issparse(stiff_rbp):
                stiff_rbp = stiff_rbp.toarray()

        # --- Optional further marginalisation to rotations only ---
        if rotations_only:
            gs_rbp_flat2 = so3.vector_rotmarginal(so3.vecs2statevec(gs_rbp))
            gs_rbp = so3.statevec2vecs(gs_rbp_flat2, vdim=3)
            if stiff_rbp is not None:
                stiff_rbp = so3.matrix_rotmarginal(stiff_rbp)

        # --- Build instance and inject pre-computed values ---
        instance = cls(
            cgnap._sequence,
            parameter_set_name=cgnap._parameter_set_name,
            euler_definition=cgnap._euler_definition,
            group_split=cgnap._group_split,
            translations_in_nm=cgnap._translations_in_nm,
            include_stiffness=(stiff_rbp is not None),
            remove_factor_five=cgnap._remove_factor_five,
            rotations_only=rotations_only,
        )
        instance._set_precomputed(gs_rbp, stiff_rbp, rbp_param_names)
        return instance


def rbpparams(
    sequence: str,
    parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
    euler_definition: bool = True,
    group_split: bool = True,
    remove_factor_five: bool = True,
    translations_in_nm: bool = True,
    include_stiffness: bool = True,
    rotations_only: bool = False,
) -> RBPParams:
    """Convenience factory for :class:`RBPParams`.  Mirrors the ``cgnaplus()``
    module-level function."""
    return RBPParams.init(
        sequence=sequence,
        parameter_set_name=parameter_set_name,
        euler_definition=euler_definition,
        group_split=group_split,
        remove_factor_five=remove_factor_five,
        translations_in_nm=translations_in_nm,
        include_stiffness=include_stiffness,
        rotations_only=rotations_only,
    )

def rbp_params(
    sequence: str,
    parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
    euler_definition: bool = True,
    group_split: bool = True,
    remove_factor_five: bool = True,
    translations_in_nm: bool = True,
    include_stiffness: bool = True,
    rotations_only: bool = False,
) -> RBPParams:
    """Convenience factory for :class:`RBPParams`.  Mirrors the ``cgnaplus()``
    module-level function."""
    return RBPParams.init(
        sequence=sequence,
        parameter_set_name=parameter_set_name,
        euler_definition=euler_definition,
        group_split=group_split,
        remove_factor_five=remove_factor_five,
        translations_in_nm=translations_in_nm,
        include_stiffness=include_stiffness,
        rotations_only=rotations_only,
    )


# ======================================================================
# Legacy function — kept for testing / backward compatibility
# ======================================================================

def cgnaplus2rbp(
    sequence: str,
    translations_in_nm: bool = True,
    euler_definition: bool = True,
    group_split: bool = True,
    parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
    remove_factor_five: bool = True,
    rotations_only: bool = False,
    include_stiffness: bool = True,
) -> dict[str, np.ndarray | bool | str]:
    """Legacy dict-returning function.  Prefer the :class:`RBPParams` class."""

    gs, stiff = constructSeqParms(sequence, parameter_set_name)
    names = cgnaplus_name_assignment(sequence)
    select_names = [INTER_BP_PARAM_NAME + "*"]
    if include_stiffness:
        stiff = so3.matrix_marginal_assignment(stiff, select_names, names, block_dim=6)
        if sp.sparse.issparse(stiff):
            stiff = stiff.toarray()
    gs = so3.vector_marginal_assignment(gs, select_names, names, block_dim=6)

    if remove_factor_five:
        factor = 5
        gs    = so3.array_conversion(gs, 1. / factor, block_dim=6, dofs=[0, 1, 2])
        if include_stiffness:
            stiff = so3.array_conversion(stiff, factor, block_dim=6, dofs=[0, 1, 2])

    if translations_in_nm:
        factor = 10
        gs    = so3.array_conversion(gs, 1. / factor, block_dim=6, dofs=[3, 4, 5])
        if include_stiffness:
            stiff = so3.array_conversion(stiff, factor, block_dim=6, dofs=[3, 4, 5])
    gs = so3.statevec2vecs(gs, vdim=6)

    if euler_definition:
        # cayley2euler_stiffmat requires gs in cayley definition
        if include_stiffness:
            stiff = so3.se3_cayley2euler_stiffmat(gs, stiff, rotation_first=True)
        gs = so3.se3_cayley2euler(gs)

    if group_split:
        if not euler_definition:
            raise ValueError('The group_split option requires euler_definition to be set!')
        if include_stiffness:
            gs, stiff = so3.algebra2group_params(
                gs, stiff, rotation_first=True, translation_as_midstep=True, optimized=True,
            )
        else:
            gs = so3.midstep2triad(gs)

    if rotations_only:
        gs = so3.vector_rotmarginal(so3.vecs2statevec(gs))
        if include_stiffness:
            stiff = so3.matrix_rotmarginal(stiff)

    result = {
        "gs": gs,
        "sequence": sequence,
        "translations_in_nm": translations_in_nm,
        "euler_definition": euler_definition,
        "group_split": group_split,
        "remove_factor_five": remove_factor_five,
        "rotations_only": rotations_only,
    }
    if include_stiffness:
        result["stiffmat"] = stiff

    return result
