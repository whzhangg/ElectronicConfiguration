import numpy as np
from ase.data import chemical_symbols

def get_StructureCollection():
    from pymongo import MongoClient
    Client = MongoClient("mongodb://144.213.165.19:27017/")
    DB = Client["MaterialScience"]
    return DB["StructureDescription"]


def return_doc_from_id(database, id):
    from bson.objectid import ObjectId
    oid = ObjectId(id)
    return database.find({ "_id" : oid })[0]


def get_site_from_doc(document: dict) -> dict:
    types = np.array(document["types"], dtype = int)
    sites = {}

    for site_txt in document["sites"]:
        all_parts = site_txt.split("_")
        wp = all_parts[0]
        unique = int(all_parts[1])
        itype = int(all_parts[2])
        symmetry = all_parts[3]
        eq_indices = all_parts[4]

        # it is possible that site symbol will contain '_', which is resolved here 
        if len(all_parts) == 11:
            poly = all_parts[5]
            csm = all_parts[6]
            neighbor_indices = all_parts[7]
            x = float(all_parts[8])
            y = float(all_parts[9])
            z = float(all_parts[10])
        elif len(all_parts) == 12:
            poly = all_parts[5] + "_" + all_parts[6]
            csm = all_parts[7]
            neighbor_indices = all_parts[8]
            x = float(all_parts[9])
            y = float(all_parts[10])
            z = float(all_parts[11])

        try:
            tmp_eq = []
            tmp = eq_indices.split(".")
            for t in tmp:
                tmp_eq.append(int(t))
            eq_indices = tmp_eq
        except ValueError:
            eq_indices = []

        try:
            csm = float(csm)
        except ValueError:
            csm = 100.0

        try:
            tmp_neighbor = []
            tmp = neighbor_indices.split(".")
            for t in tmp:
                tmp_neighbor.append(int(t))
            neighbor_indices = tmp_neighbor
        except ValueError:
            neighbor_indices = []

        if wp not in sites:
            sites[wp] = {}

        if unique not in sites[wp]:
            sites[wp][unique] = {"chemical_symbol": chemical_symbols[itype],
                                "type" : itype,
                                "symmetry" : symmetry,
                                 "equivalent_indices" : eq_indices,
                                 "poly_symbol" : poly,
                                 "symmetry_measure" : csm,
                                 "neighbors_indices" : neighbor_indices,
                                 "neighbors_symbols" : [chemical_symbols[types[i]] for i in neighbor_indices ],
                                 "wyckoff_position" : [x,y,z]
                                 }
    
    return sites
