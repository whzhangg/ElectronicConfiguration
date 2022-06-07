# make dataset as torch geometry
import typing, re, os, h5py
import numpy as np
from bisect import bisect
from mldos.parameters import symbol_one_hot_dict, TM_3D, estep_dos
from torch_geometric.data import Data
from .unload import load_dataset
from .milad_utility import Description, INVARIANTS
from ..tools import printProgressBar, separate_formula
from .criteria import from_criteria_to_filename

"""
Data(
  pos=np.array([*, 3]),
  types=np.array(),
  onehot={
    "*"=np.array([117]), ...
  },
  y_up   = float,
  y_down = float,
  name   = 'str'
)]
"""

current_folder = os.path.split(__file__)[0]
training_datafolder = os.path.join(current_folder, "hdf")

def normalize_dos(energy, dos_data: dict, efermi: float, erange = 10):
    # we scale the dos so that the energy window is about [ efermi - erange, efermi + erange ]
    
    middle = bisect(energy, efermi)
    up = round(erange / estep_dos)
    down = middle - up
    up = middle + up

    new_energy = np.arange(0.0 - erange, 0.0 + erange ,estep_dos)  # centered at zero
    
    datalength = len(new_energy)

    for dos in dos_data.values():
        for spin, spin_dos in dos.items():
            if up > len(spin_dos):
                tmp = np.pad(spin_dos, (0,up-len(spin_dos)), 'constant' )
            else:
                tmp = spin_dos
            if down >= 0:
                dos[spin] = tmp[down:up]
            else:
                dos[spin] = np.pad(tmp[0:up], (0-down, 0), 'constant')

            if len(dos[spin]) != datalength:
                print(up, down)
                print(len(dos[spin]))
                print(datalength)
                raise

    return new_energy, dos_data


def make_Geometry_data_with_feature(dos_data: dict, criteria: dict) -> typing.List[Data]:
    c = np.array(dos_data["cell"]).reshape((3,3))
    p = np.array(dos_data["positions"]).reshape((-1,3))
    t = dos_data["types"]
    unique = dos_data["unique_tm_index"] 
    # prepare DOS 

    site_dos = {}

    indexs = dos_data["unique_tm_index"] # int
    energy = dos_data["energy"]
    ef = dos_data["efermi"]

    for i in indexs:
        dos = dos_data[str(i)]["3d"]
        tmp = {"up": np.array(dos["ldosup"]), "down": np.array(dos["ldosdw"])}
            #sum_up = np.sum(tmp["up"]) * estep_dos
            #sum_down = np.sum(tmp["down"]) * estep_dos
            #if abs(sum_up - 5.0) > 1 or abs(sum_down - 5.0) > 1:
            #    warnings.warn(prefix + " dos not normalized: {:>4.1f}{:4.1f}".format(sum_up, sum_down))
        site_dos[i] = tmp

    new_energy, site_dos = normalize_dos(energy, site_dos, ef )
    occupation: typing.Dict[int, typing.Dict[str, float]] = {}
    mask = new_energy < 0.0
    for key, value in site_dos.items():
        count = {}
        count["up"] = np.sum(value["up"][mask]) * estep_dos
        count["down"] = np.sum(value["down"][mask]) * estep_dos
        occupation[key] = count


    sd = Description(cell = c, positions = p, types = t, keepstructure = True)

    features = sd.get_features( unique, 
                                bonded = criteria["bonded_only"],
                                firstshell = criteria["firstshell_only"],
                                cutoff = criteria["cutoff"],
                                smooth = criteria["smooth"])

    neighbors = sd.get_neighbors(unique, 
                            bonded = criteria["bonded_only"],
                            firstshell = criteria["firstshell_only"],
                            cutoff = criteria["cutoff"])

    datas: typing.List[Data] = []
    for key, nlist in neighbors.items():
        types = np.array([ sd.std_types[nn.index] for nn in nlist ])
        pos = np.array([ nn.rij for nn in nlist ])
        datas.append(
            Data(
                types = types, pos = pos, 
                onehot = features[key], y_up = occupation[key]["up"], y_down = occupation[key]["down"],
                name = f"{sd.formula}->{key}"
            )
        )

    return datas


def get_all_data_as_geometric_data(criteria: dict, verbose: bool) -> typing.List[Data]:
    datas = load_dataset()
    all_data = []

    tot_num = len(datas.keys())
    step = max( int(tot_num/100), 1)
    if verbose: printProgressBar(0, tot_num)
    count = 0

    for key, value in datas.items():
        all_data += make_Geometry_data_with_feature(value, criteria)
        count += 1
        if verbose and count % step:
            printProgressBar(count+1, tot_num)

    return all_data


def encode_feature(feature_dict):
    """feature dictionary of a particular site, encode into an array"""
    ff = np.zeros((len(symbol_one_hot_dict.keys()), len(INVARIANTS)))
    for ele, mom in feature_dict.items():
        ff[symbol_one_hot_dict[ele]] = mom
    return ff


def write_hdf(criteria: dict, filename: str = ""):
    names = []     # str
    features = []  # numpy arrays
    targets = []  # tuples

    print(f"Generating dataset with the following criteria:")

    for key, value in criteria.items():
        print("{:s} : {:s}".format(key, str(value)))

    all_geometric_data = get_all_data_as_geometric_data(criteria, verbose=True)

    for i, gdata in enumerate(all_geometric_data):
        # check for elements
        compound_name = gdata.name.split("-")[0]
        formula_dict = separate_formula(compound_name)
        contained_ele = set(formula_dict.keys())
        contained_tm = contained_ele & TM_3D
        if criteria["single_tm"] and len(contained_tm) > 1: continue
        if not contained_ele < criteria["elements"]: continue

        feature = gdata.onehot
        if len(feature.keys()) == 0:
            continue
        features.append(encode_feature(feature))
        targets.append((gdata.y_up, gdata.y_down))
        names.append(gdata.name)

    print("Done !")
    print("feature has the shape: " + str(np.array(features).shape))
    print("target  has the shape: " + str(np.array(targets).shape))

    generated_filename = os.path.join(
        training_datafolder, from_criteria_to_filename(criteria, len(names))
    )
    if not filename:
        filename = generated_filename

    print(f"data written to file: {filename}")

    with h5py.File(filename, 'w') as f:
        f.create_dataset("names", data = names)
        f.create_dataset("features", data = np.array(features))
        f.create_dataset("targets", data = np.array(targets))


def open_dataset(filename):
    filename = os.path.join(training_datafolder, filename)
    with h5py.File(filename, 'r') as f:
        features = np.array(f["features"])
        targets = np.array(f["targets"])
        names = list(f["names"])
    return names, features, targets