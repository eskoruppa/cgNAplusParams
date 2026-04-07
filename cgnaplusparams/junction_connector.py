from __future__ import annotations
from datetime import time
from os import name

import numpy as np
from sympy import sign
from .utils.assignment_utils import cgnaplus_name_assignment
from .naming_conventions import *


class Pose:
    def __init__(self, type: str, bpid: np.ndarray):
        self._type = type
        self._bpid = bpid

        if not self._valid():
            raise ValueError(f"Invalid pose: type={type}, bpid={bpid}")

    def _valid(self) -> bool:
        if self._type not in [BP_NAME, WATSON_BASE_NAME, CRICK_BASE_NAME, WATSON_PHOSPHATE_NAME, CRICK_PHOSPHATE_NAME]:
            return False
        if self._bpid < 0:
            return False
        return True
    
    @property
    def type(self) -> str:
        return self._type
    
    @property
    def bpid(self) -> np.ndarray:
        return self._bpid
    
    @property
    def name(self) -> str:
        return f"{self._type}{self._bpid}"
    

class Junction:

    param_name_mapper = {
        INTRA_BP_JUNC_NAME : INTRA_BP_PARAM_NAME,
        INTER_BP_JUNC_NAME : INTER_BP_PARAM_NAME,
        B2P_CRICK_JUNC_NAME : B2P_CRICK_PARAM_NAME,
        B2P_WATSON_JUNC_NAME : B2P_WATSON_PARAM_NAME,
        C2BP_JUNC_NAME : INTRA_BP_PARAM_NAME,
        BP2W_JUNC_NAME : INTRA_BP_PARAM_NAME,
    }
    param_style_mapper = {
        INTRA_BP_JUNC_NAME: FULL_JUNCTION_NAME,
        INTER_BP_JUNC_NAME : FULL_JUNCTION_NAME,
        B2P_CRICK_JUNC_NAME : FULL_JUNCTION_NAME,
        B2P_WATSON_JUNC_NAME : FULL_JUNCTION_NAME,
        C2BP_JUNC_NAME : LEFTHAND_JUNCTION_NAME,
        BP2W_JUNC_NAME : RIGHTHAND_JUNCTION_NAME,
    }

    def __init__(self, type: str, bpid: int, sense: int):
        self._type = type
        self._bpid = bpid
        self._sense = sense
        if not self._valid():
            raise ValueError(f"Invalid junction: type={type}, bpid={bpid}, sense={sense}")

    def _valid(self) -> bool:
        if self._type not in [INTER_BP_JUNC_NAME, INTRA_BP_JUNC_NAME, BP2W_JUNC_NAME, C2BP_JUNC_NAME, B2P_WATSON_JUNC_NAME, B2P_CRICK_JUNC_NAME]:
            return False
        if self._bpid < 0:
            return False
        if self._sense not in [-1, 1]:
            return False
        return True
    
    @property
    def type(self) -> str:
        return self._type

    @property
    def bpid(self) -> int:
        return self._bpid
    
    @property
    def sense(self) -> int:
        return self._sense
    
    @property
    def name(self) -> str:
        return f"{self._type}{self._bpid}"
    
    @property
    def param_name(self) -> str:
        return self.name.upper()
    
    @property
    def signed_name(self) -> str:
        sign_str = '+' if self._sense == 1 else '-'
        return f"{self.name}({sign_str})"
    
    @property
    def signed_param_name(self) -> str:
        sign_str = '+' if self._sense == 1 else '-'
        return f"{self.param_name}({sign_str})"
    
    def flip_sense(self) -> Junction:
        return Junction(self._type, self._bpid, -self._sense)
    
    def poses(self):
        if self._type == INTER_BP_JUNC_NAME:
            if self._sense == 1:
                return [Pose(BP_NAME, self._bpid), Pose(BP_NAME, self._bpid + 1)]
            else:
                return [Pose(BP_NAME, self._bpid + 1), Pose(BP_NAME, self._bpid)]
        if self._type == INTRA_BP_JUNC_NAME:
            if self._sense == 1:
                return [Pose(CRICK_BASE_NAME, self._bpid), Pose(WATSON_BASE_NAME, self._bpid)]
            else:
                return [Pose(WATSON_BASE_NAME, self._bpid), Pose(CRICK_BASE_NAME, self._bpid)]

        if self._type == B2P_WATSON_JUNC_NAME:
            if self._sense == 1:
                return [Pose(WATSON_BASE_NAME, self._bpid), Pose(WATSON_PHOSPHATE_NAME, self._bpid)]
            else:
                return [Pose(WATSON_PHOSPHATE_NAME, self._bpid), Pose(WATSON_BASE_NAME, self._bpid)]

        if self._type == B2P_CRICK_JUNC_NAME:
            if self._sense == 1:
                return [Pose(CRICK_BASE_NAME, self._bpid), Pose(CRICK_PHOSPHATE_NAME, self._bpid)]
            else:
                return [Pose(CRICK_PHOSPHATE_NAME, self._bpid), Pose(CRICK_BASE_NAME, self._bpid)]

        if self._type == BP2W_JUNC_NAME:
            if self._sense == 1:
                return [Pose(BP_NAME, self._bpid), Pose(WATSON_BASE_NAME, self._bpid)]
            else:
                return [Pose(WATSON_BASE_NAME, self._bpid), Pose(BP_NAME, self._bpid)]
            
        if self._type == C2BP_JUNC_NAME:
            if self._sense == 1:
                return [Pose(CRICK_BASE_NAME, self._bpid), Pose(BP_NAME, self._bpid)]
            else:
                return [Pose(BP_NAME, self._bpid), Pose(CRICK_BASE_NAME, self._bpid)]
            
        raise ValueError(f"Invalid junction type: {self._type}")

    @property
    def innate(self) -> str:
        return f"{self.param_name_mapper[self._type]}{self._bpid}"

    @property
    def style(self) -> str:
        return self.param_style_mapper[self._type]


def _is_bp(name: str) -> bool:
    return name.startswith(BP_NAME)

def _is_watson_base(name: str) -> bool:
    return name.startswith(WATSON_BASE_NAME)

def _is_crick_base(name: str) -> bool:
    return name.startswith(CRICK_BASE_NAME)

def _is_watson_phosphate(name: str) -> bool:
    return name.startswith(WATSON_PHOSPHATE_NAME)

def _is_crick_phosphate(name: str) -> bool:
    return name.startswith(CRICK_PHOSPHATE_NAME)

def _name2iid(name) -> int | None:
    if _is_crick_phosphate(name):
        return 0
    if _is_crick_base(name):
        return 1
    if _is_watson_base(name):
        return 2
    if _is_watson_phosphate(name):
        return 3
    if _is_bp(name):
        return 1.5
    return None

def _revert_junctions(juncs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [junc.flip_sense() for junc in juncs[::-1]]

def _interal_connect(name1: str, name2: str) -> list[tuple[str, int]] | None:
    bpid1 = int(name1[2:])
    bpid2 = int(name2[2:])
    if bpid1 != bpid2:
        return None
    
    iid1 = _name2iid(name1)
    iid2 = _name2iid(name2)

    if iid1 is None or iid2 is None:
        return None
    if iid1 == iid2:
        return []
    
    reverse = False
    if iid1 > iid2:
        name1, name2 = name2, name1
        iid1, iid2 = iid2, iid1
        reverse = True

    alljuncs = [Junction(B2P_CRICK_JUNC_NAME, bpid1, -1), Junction(INTRA_BP_JUNC_NAME, bpid1, 1), Junction(B2P_WATSON_JUNC_NAME, bpid1, 1)]

    if iid1 == 1.5:
        juncs = [Junction(BP2W_JUNC_NAME, bpid1, 1)]
        juncs += alljuncs[2:iid2]

    elif iid2 == 1.5:
        juncs = alljuncs[:1]
        juncs += [Junction(C2BP_JUNC_NAME, bpid1, 1)]
    else:
        juncs = alljuncs[iid1:iid2]

    if reverse:
        juncs = _revert_junctions(juncs)
    return juncs

def _juncs_to_bp(name: str) -> list[str] | None:
    if _is_bp(name):
        return []
    bpid = int(name[2:])
    if _is_watson_base(name):
        return [Junction(BP2W_JUNC_NAME, bpid, -1)]
    if _is_watson_phosphate(name):
        return [Junction(B2P_WATSON_JUNC_NAME, bpid, -1), Junction(BP2W_JUNC_NAME, bpid, -1)]
    if _is_crick_base(name):
        return [Junction(C2BP_JUNC_NAME, bpid, 1)]
    if _is_crick_phosphate(name):
        return [Junction(B2P_CRICK_JUNC_NAME, bpid, -1), Junction(C2BP_JUNC_NAME, bpid, 1)]
    return None

def _juncs_from_bp(name: str) -> list[str] | None:
    to_bp = _juncs_to_bp(name)
    if to_bp is None or len(to_bp) == 0:
        return to_bp
    from_bp = to_bp[::-1]
    for i in range(len(from_bp)):
        from_bp[i] = from_bp[i].flip_sense()
    return from_bp

def _juncs_from_bp_to_bp(bpid1: int, bpid2: int) -> list[str] | None:
    if bpid1 == bpid2:
        return []
    swap = False
    if bpid1 > bpid2:
        bpid1, bpid2 = bpid2, bpid1
        swap = True

    juncs = []
    for i in range(bpid1, bpid2):
        juncs.append(Junction(INTER_BP_JUNC_NAME, i, 1))
    if swap:
        juncs = juncs[::-1]
        for i in range(len(juncs)):
            juncs[i] = juncs[i].flip_sense()
    return juncs

def check_junctions_consistency(juncs: list[Junction]) -> bool:
    for i in range(len(juncs)-1):
        if juncs[i].poses()[1].name != juncs[i+1].poses()[0].name:
            raise ValueError(f"Mismatch between junction {i} and {i+1}: {juncs[i].poses()[1].name} != {juncs[i+1].poses()[0].name}")
    return True

def vertices2junctions(
        first_name: str,
        second_name: str,
        ) -> list[str] | None:

    bpid1 = int(first_name[2:])
    bpid2 = int(second_name[2:])

    reverse = False
    if bpid1 > bpid2:
        bpid1, bpid2 = bpid2, bpid1
        first_name, second_name = second_name, first_name  
        reverse = True

    if bpid1 == bpid2:
        juncs = _interal_connect(first_name, second_name)
    else:
        juncs =  _juncs_to_bp(first_name)
        juncs += _juncs_from_bp_to_bp(bpid1, bpid2)
        juncs += _juncs_from_bp(second_name)
    if reverse:
        juncs = _revert_junctions(juncs)
    return juncs


def junction_mapper(first_name: str, second_name: str, param_names: list[str]) -> dict[str, Junction]:
    juncs = vertices2junctions(first_name, second_name)
    innates = [junc.innate for junc in juncs]
    indices = [param_names.index(innate) for innate in innates]
    styles  = [junc.style for junc in juncs]
    senses  = [junc.sense for junc in juncs]
    return {
        'indices': indices,
        'styles': styles,
        'senses': senses,
        'junctions': juncs,
    }


if __name__ == "__main__":

    first_name = f"{WATSON_BASE_NAME}0"
    second_name = f"{CRICK_BASE_NAME}0"

    first_name = f"{WATSON_BASE_NAME}10"
    second_name = f"{CRICK_BASE_NAME}10"

    first_name = f"{CRICK_PHOSPHATE_NAME}1"
    second_name = f"{WATSON_PHOSPHATE_NAME}1"

    first_name = f"{CRICK_PHOSPHATE_NAME}1"
    second_name = f"{WATSON_BASE_NAME}1"

    pw = f"{WATSON_PHOSPHATE_NAME}"
    pc = f"{CRICK_PHOSPHATE_NAME}"
    bw = f"{WATSON_BASE_NAME}"
    bc = f"{CRICK_BASE_NAME}"

    first_name = f"{CRICK_PHOSPHATE_NAME}1"
    second_name = f"{BP_NAME}1"

    first_name = f"{BP_NAME}1"
    second_name = f"{CRICK_PHOSPHATE_NAME}1"

    first_name = f"{BP_NAME}1"
    second_name = f"{WATSON_PHOSPHATE_NAME}1"

    first_name = f"{WATSON_PHOSPHATE_NAME}1"
    second_name = f"{BP_NAME}1"

    first_name = f"{CRICK_PHOSPHATE_NAME}1"
    second_name = f"{BP_NAME}1"

    first_name = f"{CRICK_PHOSPHATE_NAME}1"
    second_name = f"{WATSON_PHOSPHATE_NAME}1"

    first_name = f"{WATSON_PHOSPHATE_NAME}1"
    second_name = f"{CRICK_PHOSPHATE_NAME}1"


    juncs = vertices2junctions(first_name, second_name)
    print('#'*80)
    print('# Individual evaluation')
    print('#'*80)

    print(f"len(juncs): {len(juncs)}")

    print(f"{first_name} -> {second_name}:")
    print('----------------------')
    print('Nodes')
    juncstr = " -> ".join([j.signed_name for j in juncs])
    print(juncstr)

    print('----------------------')
    print('Vertices')
    for junc in juncs:
        print(f"{junc.signed_name}: {junc.poses()[0].name} -> {junc.poses()[1].name}")

    check_junctions_consistency(juncs)

    print('All checks passed!\n\n')


    print('#'*80)
    print('# Consistency checks for all junctions between neighboring Vertices')
    print('#'*80)

    n_check = 100000
    max_bpid = 12 

    import time
    t1 = time.time()

    for i in range(n_check):
        pose_types = [BP_NAME, WATSON_BASE_NAME, CRICK_BASE_NAME, WATSON_PHOSPHATE_NAME, CRICK_PHOSPHATE_NAME]
        first_name = pose_types[np.random.randint(5)] + str(np.random.randint(max_bpid))
        second_name = pose_types[np.random.randint(5)] + str(np.random.randint(max_bpid))

        juncs = vertices2junctions(first_name, second_name) 
        check_junctions_consistency(juncs)

    t2 = time.time()
    print(f"Time taken for {n_check} random junction checks: {t2 - t1:.6f} seconds ({(t2 - t1) / n_check:.6f} seconds per check)")
    print(f'Expected time for a nucleosome {(t2 - t1) / n_check*28:.6f} seconds')

    print('All checks passed!\n\n')

    print('#'*80)
    print('# Build parameter index mapping')
    print('#'*80)
    from .cgnaplus_params import cgnaplusparams

    nbp = 7
    seq = ''.join(np.random.choice(list("ACGT"), size=nbp))
    result = cgnaplusparams(seq, include_stiffness=True)

    param_names = result["param_names"]

    first_name = f"{WATSON_PHOSPHATE_NAME}1"
    second_name = f"{CRICK_PHOSPHATE_NAME}5"

    juncs = vertices2junctions(first_name, second_name)

    juncmap = junction_mapper(first_name, second_name, param_names)

    # print(param_names)
    # print(indices)

    print(param_names)
    print(juncmap['indices'])
    print(juncmap['styles'])
    print(juncmap['senses'])