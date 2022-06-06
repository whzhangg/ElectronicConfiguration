import os
import shutil, typing, pickle
from DFTtools.tools import read_json

json_folder = "/Users/wenhao/work/ml_dos/mldos/json_results/jsons"
pickle_folder = "/Users/wenhao/work/ml_dos/mldos/json_results/pickles"


def dump_pickle(data, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
def load_pickles(filename: str):
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result

def json_to_pickle():
    files = [ p for p in os.listdir(json_folder) if ".json" in p ]
    n = 250

    data = {}
    index = 1
    for i, file in enumerate(files):
        prefix = file.split(".")[0]
        data[prefix] = read_json(os.path.join(json_folder, file))
        if (i + 1) % n == 0:
            filename = os.path.join(pickle_folder, f"result_{index}.pkl")
            dump_pickle(data, filename)
            index += 1
            data = {}
    
    filename = os.path.join(pickle_folder, f"result_{index}.pkl")
    dump_pickle(data, filename)

def load_dataset():
    files = sorted([ p for p in os.listdir(pickle_folder) if ".pkl" in p ])
    datas = {}
    for file in files:
        d = load_pickles(os.path.join(pickle_folder, file))
        datas.update(d)
    print(datas.keys())
    print(len(datas.keys()))

load_dataset()