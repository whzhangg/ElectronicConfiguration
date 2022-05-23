import numpy as np
import os
import h5py
import re
from bisect import bisect
from ase.data import chemical_symbols

from mldos.DFT.job_control import JobBase
from mldos.tools import read_json, printProgressBar, read_generic_to_cpt, separate_formula
from mldos.parameters import symbol_one_hot_dict, TM_3D, estep_dos
from .milad_utility import Description, INVARIANTS
from ._parameters import training_datafolder


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


def _encode_feature(feature_dict):
    """feature dictionary of a particular site, encode into an array"""
    ff = np.zeros((len(symbol_one_hot_dict.keys()), len(INVARIANTS)))
    for ele, mom in feature_dict.items():
        ff[symbol_one_hot_dict[ele]] = mom
    return ff


def open_dataset(filename):
    filename = os.path.join(training_datafolder, filename)
    with h5py.File(filename, 'r') as f:
        features = np.array(f["features"])
        targets = np.array(f["targets"])
        names = list(f["names"])
    return names, features, targets


class Data(JobBase):

    def __init__(self, criteria: dict, exclude: list = []) -> None:
        super().__init__()

        self._dos_result = {}
        self._normalized_dos_length = 0
        self._todo_prefixs = [ p for p in self.done_prefixs if p not in exclude ]

        assert "firstshell_only" in criteria
        assert "bonded_only" in criteria
        assert "cutoff" in criteria
        assert "smooth" in criteria
        assert "target" in criteria
        self.criteria = criteria
        self.target = criteria["target"]

    @classmethod
    def from_filename(cls, filename):
        """this will reverse encode the criteria in the filename, 
        this is for generating consistently the descriptors"""
        
        if "occ" in filename:
            target = "filling"
        elif "dos" in filename:
            target = "dos"
        else:
            raise Exception("filename has no target")
        
        if "bonded" in filename:
            bonded_only = True
            firstshell_only = False
        elif "firstshell" in filename:
            bonded_only = False
            firstshell_only = True
        else:
            bonded_only = False
            firstshell_only = False

        if "smoothed" in filename:
            smooth = True
        else:
            smooth = False

        if "single" in filename:
            single_tm = True
        else:
            single_tm = False

        rcut = re.findall("rcut=\d+.?\d*", filename)[0]
        cutoff = float(re.findall("\d+.?\d*", rcut)[0])

        criteria = {"firstshell_only" : firstshell_only,
                    "bonded_only" : bonded_only,
                    "cutoff" : cutoff,
                    "smooth" : smooth,
                    "target" : target,
                    "single_tm" : single_tm,
                    "elements": set([])}

        #print(criteria)
        return cls(criteria)


    @property
    def todo_prefix(self):
        return self._todo_prefixs


    def _get_dos_result(self, prefix):
        assert prefix in self._todo_prefixs

        if prefix not in self._dos_result:
            file = os.path.join(self.folders[prefix], self.DOSFOLDER, "result.json")
            data = read_json(file)
            self._dos_result[prefix] = data

        return self._dos_result[prefix]


    def get_cpt(self, prefix) -> tuple:
        assert prefix in self._todo_prefixs

        dos_data = self._get_dos_result(prefix)
        c = np.array(dos_data["cell"]).reshape((3,3))
        p = np.array(dos_data["positions"]).reshape((-1,3))
        t = dos_data["types"]
        return (c, p, t)

    def get_local_structure(self, prefix) -> dict:
        # it will return the local structure which is used to generate the descriptor
        # 
        assert prefix in self._todo_prefixs

        dos_data = self._get_dos_result(prefix)
        c = np.array(dos_data["cell"]).reshape((3,3))
        p = np.array(dos_data["positions"]).reshape((-1,3))
        t = dos_data["types"]
        unique = dos_data["unique_tm_index"] 

        sd = Description(cell = c, positions = p, types = t, keepstructure = True)
        return sd.get_neighbors( unique, 
                                bonded = self.criteria["bonded_only"],
                                firstshell = self.criteria["firstshell_only"],
                                cutoff = self.criteria["cutoff"])

    def make_descriptor(self, prefix) -> dict:
        # main method, which generate descriptor
        # return an dict with atomic index as key and feature as value
        assert prefix in self._todo_prefixs

        dos_data = self._get_dos_result(prefix)
        c = np.array(dos_data["cell"]).reshape((3,3))
        p = np.array(dos_data["positions"]).reshape((-1,3))
        t = dos_data["types"]
        unique = dos_data["unique_tm_index"] 

        sd = Description(cell = c, positions = p, types = t, keepstructure = True)
        return sd.get_features( unique, 
                                bonded = self.criteria["bonded_only"],
                                firstshell = self.criteria["firstshell_only"],
                                cutoff = self.criteria["cutoff"],
                                smooth = self.criteria["smooth"] )


    def get_dos(self, prefix) -> dict:
        assert prefix in self._todo_prefixs

        dos_data = self._get_dos_result(prefix)
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
        result = {"energy" : new_energy,
                  "dos"    : site_dos }

        # check of DOS all have the same size
        if self._normalized_dos_length != len(new_energy):
            if self._normalized_dos_length == 0:
                self._normalized_dos_length = len(new_energy)
            else:
                print(prefix)
                print(self._normalized_dos_length)
                print(len(new_energy))
                raise

        return result


    def get_filling(self, prefix) -> dict:
        """ return the electron occupation on TM atoms
        """
        dos = self.get_dos(prefix)
        occupation = {}
        mask = dos["energy"] < 0.0
        for key, value in dos["dos"].items():
            c = {}
            c["up"] = np.sum(value["up"][mask]) * estep_dos
            c["down"] = np.sum(value["down"][mask]) * estep_dos
            occupation[key] = c
        return occupation


    def plot_dos_by_prefix(self, prefix, filename = None):
        """a simple utility to plot density for a compund"""
        import matplotlib.pyplot as plt
        wsize = 6
        hsize = 4

        outputfile = filename or "{:s}.pdf".format(prefix)

        dosdict = self.get_dos(prefix)
        energy = dosdict["energy"]
        dos = dosdict["dos"]

        nsite = len(dos.keys())
        fig = plt.figure()
        if nsite == 1:
            axs = fig.subplots()
            axes = [axs]
            fig_size = (wsize, hsize + 0.5)
        else:
            axes = fig.subplots(nsite, 1, sharex = True)
            fig_size = (wsize, hsize * nsite)
        print("formula {:s} with {:d} sites ".format(prefix, nsite))

        for axs, whichsite in zip(axes, dos.keys()):
            key = prefix + " site: {:d}".format(whichsite + 1)
            axs.plot(energy, dos[whichsite]["up"], "-", color = "black", label = key + " up")
            axs.plot(energy, -1 * dos[whichsite]["down"], "-", color = "red", label = key + "down")
            axs.set_ylabel("DOS")
            axs.legend()
        axs.set_xlabel("Energy (eV)")
        fig.set_size_inches(fig_size[0], fig_size[1])
        fig.savefig(outputfile)


    def make_descriptor_from_structure(self, structfile = None, cell = None, positions = None, types = None, check_neighbor: bool = True):
        if structfile:
            c, p, t = read_generic_to_cpt(structfile)
        else:
            c = cell
            p = positions
            t = types

        sd = Description(cell = c, positions = p, types = t, keepstructure = True)
        
        uni_index = set(sd.equivalent_atoms)
        unique = [ int(u) for u in uni_index if chemical_symbols[sd.std_types[u]] in TM_3D ]

        features =  sd.get_features( unique, 
                                bonded = self.criteria["bonded_only"],
                                firstshell = self.criteria["firstshell_only"],
                                cutoff = self.criteria["cutoff"],
                                smooth = self.criteria["smooth"],
                                check_neighbors= check_neighbor )

        feature_tensor = {}
        for k, feat in features.items():
            feature_tensor[k] = _encode_feature(feat)
        # the feature returned here is a dictionary with shape 65 * 117, which need to be transposed to 
        # feed into the network
        return feature_tensor

    
    def generate_datasets(self, filename : str = None):
        """create dataset by the given criteria, 
           filename is automatically created from the criteria if not given explicitly, in the following form
           occ_bonded_rcut_5.0_smoothed_N1205.h5
           target, neighbor_selection, cutoff, if smoothed or not, N + number of samples
        """
        names = []     # str
        features = []  # numpy arrays
        targets = []  # tuples

        assert self.target in ["filling", "dos"]
        print("Generating dataset for {:s} with the following criteria:".format(self.target))

        for key, value in self.criteria.items():
            print("{:s} : {:s}".format(key, str(value)))

        tot_num = len(self._todo_prefixs)
        step = max( int(tot_num/100), 1)
        printProgressBar(0, tot_num)

        compound_count = 0
        for i, p in enumerate(self._todo_prefixs):

            # this step is added to screem the compounds for the dataset
            formula_dict = separate_formula(p)
            contained_ele = set(formula_dict.keys())
            contained_tm = contained_ele & TM_3D
            if self.criteria["single_tm"] and len(contained_tm) > 1:
                continue
            if not contained_ele < self.criteria["elements"]:
                continue

            feature = self.make_descriptor(p)
            if len(feature.keys()) == 0:
                continue

            compound_count += 1
            if self.target == "filling":
                target = self.get_filling(p)
            elif self.target == "dos":
                target = self.get_dos(p)["dos"]

            for k, feat in feature.items():
                names.append( p + "_" + str(k) )  # formula_siteindex
                features.append(_encode_feature(feat))
                f = target[k]
                if self.target == "filling":
                    targets.append((f["up"], f["down"]))
                elif self.target == "dos":
                    dosup = f["up"] * 5.0 / (np.sum(f["up"]) * estep_dos)
                    dosdw = f["down"] * 5.0 / (np.sum(f["down"]) * estep_dos)
                    targets.append((dosup, dosdw))

            if i % step == 0:
                printProgressBar(i+1, tot_num)
            
        printProgressBar(tot_num, tot_num)
        print("Done !")
        print("total num of compounds :" + str(compound_count))
        print("feature has the shape: " + str(np.array(features).shape))
        print("target  has the shape: " + str(np.array(targets).shape))

        if self.target == "dos":
            fn = "dos_"
        else:
            fn = "occ_"
        if self.criteria["firstshell_only"]:
            fn += "firstshell_"
        elif self.criteria["bonded_only"]:
            fn += "bonded_"
        else:
            fn += "all_"
        fn += "rcut={:.1f}_".format(self.criteria["cutoff"])

        fn += "nele={:d}_".format(len(self.criteria["elements"]))

        if self.criteria["single_tm"]:
            fn += "single_"
        if self.criteria["smooth"]:
            fn += "smoothed_"
        fn += "N{:d}.h5".format(len(names))

        if not filename:
            filename = os.path.join(training_datafolder, fn)

        print("data written to file: " + filename)

        with h5py.File(filename, 'w') as f:
            f.create_dataset("names", data = names)
            f.create_dataset("features", data = np.array(features))
            f.create_dataset("targets", data = np.array(targets))