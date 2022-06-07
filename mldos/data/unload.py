import os
import typing, pickle
import numpy as np
from mldos.tools import write_cif_file, atom_from_cpt

current_folder = os.path.split(__file__)[0]
_pickle_folder = os.path.join(current_folder, "pickles")
_cif_folder = os.path.join(current_folder, "cif")


def dump_pickle(data, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_pickles(filename: str):
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result


def recover_cifs():
    """
    write all cif files from the pickled data
    """
    if not os.path.exists(_cif_folder):
        os.makedirs(_cif_folder)
    datasets = load_dataset()
    for key, value in datasets.items():
        cell = np.array(value["cell"])
        positions = np.array(value["positions"])
        types = np.array(value["types"])
        atom = atom_from_cpt(cell, positions, types)
        cif_name = os.path.join(_cif_folder, f"{key}.cif")
        write_cif_file(cif_name, atom)
        break


def load_dataset() -> typing.Dict[str, dict]:
    files = sorted([ p for p in os.listdir(_pickle_folder) if ".pkl" in p ])
    datas = {}
    for file in files:
        d = load_pickles(os.path.join(_pickle_folder, file))
        datas.update(d)
    return datas

