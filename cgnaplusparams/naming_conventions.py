from __future__ import annotations

##################################################
# Parameter names 
INTRA_BP_PARAM_NAME = "B"
INTER_BP_PARAM_NAME = "S"
# INTRA_BP_PARAM_NAME = "X"
# INTER_BP_PARAM_NAME = "Y"
B2P_WATSON_PARAM_NAME = "W"
B2P_CRICK_PARAM_NAME = "C"

PARAM_BASENAMES = [
    B2P_WATSON_PARAM_NAME,
    INTRA_BP_PARAM_NAME,
    B2P_CRICK_PARAM_NAME,
    INTER_BP_PARAM_NAME,
]

##################################################
# Junction names
INTRA_BP_JUNC_NAME = "b"
INTER_BP_JUNC_NAME = "s"
# INTRA_BP_JUNC_NAME = "x"
# INTER_BP_JUNC_NAME = "y"
B2P_WATSON_JUNC_NAME = "w"
B2P_CRICK_JUNC_NAME = "c"

C2BP_JUNC_NAME = "l"
BP2W_JUNC_NAME = "r"

##################################################
# rigid body names
WATSON_BASE_NAME = "bw"
CRICK_BASE_NAME = "bc"
WATSON_PHOSPHATE_NAME = "pw"
CRICK_PHOSPHATE_NAME = "pc"
BP_NAME = "bp"

LEN_POSE_NAMES = 2
LEN_JUNCTION_NAMES = 1
LEN_PARAM_NAMES = 1

##################################################
# junction style names
FULL_JUNCTION_NAME = 'full'
LEFTHAND_JUNCTION_NAME = 'lh'
RIGHTHAND_JUNCTION_NAME = 'rh'

##################################################
# Excess parameter definitions
EXCESS_PARAMETER_DEFINITION_ALGEBRA = 'X'
EXCESS_PARAMETER_DEFINITION_GROUP = 'Y'