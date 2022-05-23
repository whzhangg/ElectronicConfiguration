from torch.utils.data import Dataset
import numpy as np
import torch

from mldos.tools import separate_formula
from mldos.data.config import tm_maingroup_metal, tm_maingroup
from mldos.data.data_generation import open_dataset
from ._model_parameter import torch_dtype, torch_device

"""
InputData should be called with the from_file method, which directly load data from a .h5 file
available_dataset = [
    "occ_bonded_rcut=5.0_smoothed_N1205.h5", 
    "occ_firstshell_rcut=5.0_smoothed_N1205.h5"
    ]
"""

def np2torch(feature_np):
    """wrapper to convert numpy to torch"""
    if isinstance(feature_np, np.ndarray):
        return torch.from_numpy(feature_np).to(torch_device, torch_dtype)
    elif isinstance(feature_np, torch.Tensor):
        return feature_np.to(torch_device, torch_dtype)

def partition_data_by_elements(dataset):
    allprefix = [ n.decode('utf-8').split("_")[0] for n in dataset.names ]
    result_dict = {"TM + Main Group": [], "TM + Main Group + Simple Metal": [], "All": []}
        
    for i,prefix in enumerate(allprefix):
            
        parts = set(separate_formula(prefix).keys())

        compound_type = "All"
        if parts < tm_maingroup:
            compound_type = "TM + Main Group"
        elif parts < tm_maingroup_metal:
            compound_type = "TM + Main Group + Simple Metal"

        result_dict.setdefault(compound_type, []).append(i)

    result = {}
    for k,val in result_dict.items():
        indexes = np.array(val)
        names = [ dataset.names[i] for i in indexes ]
        result[k] = InputData(features_np= dataset.features[indexes], 
                                    targets_np= dataset.targets[indexes],
                                    ids = names, datafilename = k)
    
    return result

def partition_data_by_transition_metals(dataset):
    allprefix = [ n.decode('utf-8').split("_")[0] for n in dataset.names ]

    tm_3d = "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
    TM_3D = set(tm_3d)
    result_dict = {tm:[] for tm in tm_3d}
        
    for i,prefix in enumerate(allprefix):
            
        parts = set(separate_formula(prefix).keys())

        tm = parts & TM_3D
        assert len(tm) == 1
        key = tm.pop()


        result_dict.setdefault(key, []).append(i)

    result = {}
    for k,val in result_dict.items():
        indexes = np.array(val)
        names = [ dataset.names[i] for i in indexes ]
        result[k] = InputData(features_np= dataset.features[indexes], 
                                    targets_np= dataset.targets[indexes],
                                    ids = names, datafilename = k)
    
    return result
    #print(len([allprefix[i] for i in result_dict["TM + Main Group"]]))

class InputData(Dataset):

    def __init__(self, features_np, targets_np, ids: list, datafilename = ""):
        """input feature, target_np, 
        ids are optional to keep track of the datas, for example, compound names"""
        assert len(features_np) == len(targets_np)
        if ids is not None:
            assert len(features_np) == len(ids)
            self._names = ids
        else:
            self._names = None

        self._features = np2torch(features_np)
        self._targets = np2torch(targets_np)

        self.nsample = len(features_np)
        self._datafilename = datafilename

    @classmethod
    def from_file(cls, filename):
        """this should be the default method to get a datasets"""
        names, features, targets = open_dataset(filename)  # list, np.array, np.array
        # change the axis of features so that we have (nsample, nfeature, nelements)
        features = features.transpose(0, 2, 1)
        return cls(features_np = features, targets_np = targets, ids = names, datafilename = filename)

    @property
    def features(self):
        return self._features

    @property
    def targets(self):
        return self._targets

    @property
    def names(self):
        return self._names

    @property
    def datafilename(self):
        return self._datafilename

    @property
    def featureshape(self):
        _, *feature_shape = self._features.shape
        return feature_shape

    @property
    def targetshape(self):
        _, *target_shape = self._targets.shape
        return target_shape

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        return self._features[idx], self.targets[idx]

    def split(self, ratio:float = 0.1, indices:tuple = None, seed:int = None):
        # return the ratio set first, then the remaining set, if indices are supplied, use the given indices
        # note that no data is copied
        seed = seed or int(np.random.rand() * 100)

        if not indices:
            nsample_test = int(len(self) * ratio)
            rng = np.random.default_rng(seed = seed)
            shuffled = np.arange(len(self))
            rng.shuffle(shuffled)
            first_indices = shuffled[0:nsample_test]
            second_indices = shuffled[nsample_test:]
        else:
            first_indices, second_indices = indices

        first_names = [ self._names[i] for i in first_indices ]
        second_names = [ self._names[i] for i in second_indices ]

        #first_names = [ n for i,n in enumerate(self._names) if i in first_indices ]
        #second_names = [ n for i,n in enumerate(self._names) if i in second_indices ] 

        first_data = InputData(self._features[first_indices], self._targets[first_indices], ids = first_names, datafilename = self._datafilename)
        second_data  = InputData(self._features[second_indices],  self._targets[second_indices],  ids = second_names, datafilename = self._datafilename )

        return (first_data, second_data)

    def bagging_set(self, seed: int = None):
        seed = seed or int(np.random.rand() * 100)
        rng = np.random.default_rng(seed = seed)
        bagged = rng.integers(low = 0, high = self.nsample, size = self.nsample, dtype = int)
        bagged_ration = len(set(bagged)) / self.nsample
        
        names = [ self._names[i] for i in bagged ]

        return InputData(self._features[bagged], self._targets[bagged], ids = names, datafilename=self._datafilename), bagged_ration

    def nfold(self, n, seed: int = None):
        """return n set of index which can be used to generate the n-fold dataset"""
        seed = seed or int(np.random.rand() * 100)
        nsample = len(self)
        if nsample % n > 0:
            n_per_fold = nsample // n + 1
        else:
            n_per_fold = nsample // n

        
        rng = np.random.default_rng(seed = seed)
        shuffled = np.arange(len(self))
        rng.shuffle(shuffled)
        
        folds = []
        for ni in range(n):
            folds.append(shuffled[ni*n_per_fold: min((ni+1)*n_per_fold, nsample)])
        
        for i in range(n):
            selected = folds[i]
            other = np.array([])
            for j in range(n):
                if j != i:
                    other = np.hstack((other, folds[j]))
                    
            small_names = [ self._names[i] for i in selected ]
            big_names = [ self._names[i] for i in other ]
            #small_names = [ n for i,n in enumerate(self._names) if i in selected ]
            #big_names = [ n for i,n in enumerate(self._names) if i in other ] 

            smallset = InputData(self._features[selected], self._targets[selected], ids = small_names, datafilename = self._datafilename)
            bigset  = InputData(self._features[other],  self._targets[other],  ids = big_names, datafilename = self._datafilename)

            yield (smallset, bigset)
