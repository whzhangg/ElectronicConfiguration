# this class is designed to extract NMF from the density of states.
# But in practice, the class here is not used, because I consider NMF as a process in the network

import numpy as np
from sklearn.decomposition import NMF

class DosNMF:
    # this class is designed to be an interface between scikitlearn's non-negative matrix factorization
    # it will also include methods to scalue the density of states, et. al.
    # and it should be able to keep the shape of the input feature and output feature
    # eg, for input, the density of states is given as an index

    def __init__(self, n_components = 50) -> None:
        self.n_comp = n_components
        self.nmf = None
        self.X = None
        self.nsample = None  # num of sample
        self.nx = None  # length of input feature
        self.features = None
    
    def train(self, entries: dict):
        # entries will be a dictionary with key as prefix, refering to a dictionary of dos
        # the dictionary will be:
        # {"dos"    : {3 : {"up": np.array, "down": np.array}, ... }
        #  "energy" :
        # }

        all_dos = []
        self.reference = {}  # have the same shape as input entries, but remember the index of the density of states
        counter = 0
        for prefix, entry in entries.items():
            ref = {"energy" : entry["energy"],
                   "dos" : {}}
            for index, site_dos in entry["dos"].items():
                ref["dos"][index] = { "up" : counter, "down" : counter + 1 }
                all_dos.append(site_dos["up"])
                all_dos.append(site_dos["down"])
                counter += 2
            self.reference[prefix] = ref
                
        self.X = np.vstack(all_dos)
        self.nsample, self.nx = self.X.shape

        
        self.nmf = NMF(n_components = self.n_comp, max_iter = 500, init = None, solver = 'cd').fit(self.X)
        self.features = self.nmf.transform(self.X)

    def get_dosfeatures(self, prefix):
        assert prefix in self.reference

        site_feature = {}
        site_feature["energy"] = self.reference[prefix]["energy"]
        site_feature["dos"] = {}
        for site_index, lookup in self.reference[prefix]["dos"].items():
            site_feature["dos"][site_index] = {}
            site_feature["dos"][site_index]["up"] = self.features[lookup["up"]]
            site_feature["dos"][site_index]["down"] = self.features[lookup["down"]]
            
        return site_feature

    def plot_components(self):
        datatext = ""
        for i in range(len(self.nmf.components_[0])):
            datatext += "{:>8.2f}".format( -10 + 0.1 * i )
            for j in range(self.n_comp):
                datatext += "{:>14.6e}".format(self.nmf.components_[j,i])
            datatext += "\n"

        with open("nmf_ncomp={:d}.dat".format(self.n_comp), 'w') as f:
            f.write(datatext)

    def _reconstruct_ith(self, index):
        result = np.zeros(self.nx)
        for i in range(self.n_comp):
            result += self.nmf.components_[i] * self.features[index][i]
        return result
    
    def reconstruct(self, prefix):
        assert prefix in self.reference

        site_dos = {}
        site_dos["energy"] = self.reference[prefix]["energy"]
        site_dos["dos"] = {}
        for site_index, lookup in self.reference[prefix]["dos"].items():
            site_dos["dos"][site_index] = {}
            site_dos["dos"][site_index]["up"] = self._reconstruct_ith(lookup["up"])
            site_dos["dos"][site_index]["down"] = self._reconstruct_ith(lookup["down"])
            
        return site_dos
