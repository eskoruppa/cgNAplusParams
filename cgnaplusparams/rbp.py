#!/bin/env python3

from __future__ import annotations

import numpy as np
import so3

from .cgnaplus import constructSeqParms, constructSeqParms_original


def cgnaplus_name_assignment(
    seq: str, 
    dof_names: list[str] = ["W", "x", "C", "y"]
    ) -> list[str]:
    """
    Generates the sequence of contained degrees of freedom for the specified sequence.
    The default names follow the convention introduced on the cgNA+ website
    """
    if len(dof_names) != 4:
        raise ValueError(
            f"Requires 4 names for the degrees of freedom. {len(dof_names)} given."
        )
    N = len(seq)
    if N == 0:  
        return []
    vars = []
    for i in range(1, N + 1):
        vars += [f"{dofn}{i}" for dofn in dof_names]
    return vars[1:-2]


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
    
    gs,stiff = constructSeqParms(sequence,parameter_set_name)
    # gs,stiff = constructSeqParms_original(sequence,parameter_set_name)

    names = cgnaplus_name_assignment(sequence)
    select_names = ["y*"]
    if include_stiffness:
        stiff = so3.matrix_marginal_assignment(stiff,select_names,names,block_dim=6)
        stiff = stiff.toarray()
    gs    = so3.vector_marginal_assignment(gs,select_names,names,block_dim=6)

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
            stiff = so3.algebra2group_stiffmat(gs,stiff,rotation_first=True,translation_as_midstep=True)  
        gs    = so3.midstep2triad(gs)
    
    if rotations_only:
        gs    = so3.vector_rotmarginal(so3.vecs2statevec(gs))
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
