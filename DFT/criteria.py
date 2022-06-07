from mldos.parameters import (
    MAIN_GROUP, REDUCED_MAIN_GROUP, 
    TM_3D, TM_4D, TM_5D, NOTCLUDE,
    defined_symbol_tuple_dict
)

# database criteria is ignored

# this criteria should give the most simple
# compounds
simple_criteria = { "database" : "StructureDescription",
             "element_set" : list(REDUCED_MAIN_GROUP | TM_3D),
             "neighbor_set": list(REDUCED_MAIN_GROUP),
             "tot_atom" : 30,
             "poly_set" : [],
             "center_site" : list(TM_3D) ,
             "max_different_type": 1
             }

# this criteria include 65 elements and can have at most 
# 2 different transition metal elements
wider_criteria = { "database" : "StructureDescription",
             "element_set" : list(defined_symbol_tuple_dict.keys()),
             "neighbor_set": list(REDUCED_MAIN_GROUP),
             "tot_atom" : 30,
             "poly_set" : [],
             "center_site" : list(TM_3D),
             "max_different_type": 2}
