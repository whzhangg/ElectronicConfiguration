from .milad_utility import Description
from ..tools import read_generic_to_cpt
from ase.data import chemical_symbols
from .criteria import TM_3D
from .datasets import encode_feature
import numpy as np
import typing

def make_descriptor_from_structure(
    structure: typing.Union[str, tuple], 
    criteria: dict,
    check_neighbor: bool = True
):
    if type(structure) == str:
        c, p, t = read_generic_to_cpt(structure)
    else:
        c, p, t = structure

    sd = Description(cell = c, positions = p, types = t, keepstructure = True)

    uni_index = set(sd.equivalent_atoms)
    unique = [ int(u) for u in uni_index if chemical_symbols[sd.std_types[u]] in TM_3D ]

    features =  sd.get_features( unique, 
                bonded = criteria["bonded_only"],
                firstshell = criteria["firstshell_only"],
                cutoff = criteria["cutoff"],
                smooth = criteria["smooth"],
                check_neighbors= check_neighbor )

    feature_tensor = {}
    for k, feat in features.items():
        feature_tensor[k] = encode_feature(feat)
    # the feature returned here is a dictionary with shape 65 * 117, which need to be transposed to 
    # feed into the network
    return feature_tensor
