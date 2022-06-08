# this script provide plotting for a single compounds
import matplotlib.pyplot as plt
import numpy as np
import argparse
from ase.data import chemical_symbols

from mldos.data.criteria import *
from mldos.tools import format_formula, write_cif_file, atom_from_cpt, read_generic_to_cpt
from mldos.data.milad_utility import Description


class SimplePlotter:
    def __init__(self, cell, positions, types, criteria:dict = occ_firstshell_rcut_5_smoothed_all):
        self.description = Description(cell = cell, positions=positions, types=types, keepstructure=True)
        self.criteria = criteria

    @classmethod
    def from_file_criteria(cls, filename, criteria:dict):
        c, p, t = read_generic_to_cpt(filename)
        return cls(c, p, t, criteria)

    @property
    def types(self):
        return self.description.std_types


    def write_cif(self, which: int, filename: str):
        localstructures = self.description.get_neighbors(
            [which], bonded=self.criteria["bonded_only"],
            firstshell = self.criteria["firstshell_only"],
            cutoff = self.criteria["cutoff"]
        )
        localstructure = localstructures[which]
        cellsize = 20.0
        c = np.eye(3) * cellsize
        p = np.vstack([vector for _,vector in localstructure]) / cellsize
        t = [self.description.std_types[i] for i,_ in localstructure]
        write_cif_file(filename, atom_from_cpt(c, p, t))

    def fingerprint(self, which):
        feature = self.description.get_features( [which], 
                                bonded = self.criteria["bonded_only"],
                                firstshell = self.criteria["firstshell_only"],
                                cutoff = self.criteria["cutoff"],
                                smooth = self.criteria["smooth"] )
        return feature[which]


class CompoundPlotter:
    """it will get the compound cif and Fingerprint plot"""
    wsize = 6
    hsize = 4.5
    style = "seaborn-colorblind"

    def __init__(self, prefix: str, criteria:dict = occ_bonded_rcut_5_smoothed_all, filename: str = None):
        self.prefix = prefix
        self.filename = filename or prefix + ".pdf"

        #from mldos.data.data_generation import Data
        data = None

        self.local_structure = data.get_local_structure(self.prefix)
        self.fingerprint = data.make_descriptor(self.prefix)
        

        dosdict = data.get_dos(prefix)
        self.energy = dosdict["energy"]
        self.dos = dosdict["dos"]       # "up" and "down" for each site

        self.c, self.p, self.t = data.get_cpt(prefix) 


    def write_cif(self, site: int = None, filename:str = None):
        if len(self.dos.keys()) > 1 and not site:
            print(self.dos.keys())
            raise Exception("Compounds has multiple site, choose one")

        site = site or list(self.dos.keys())[0]

        filename = filename or self.prefix + "_" + str(site) + ".cif"

        local_structure = self.local_structure[site]

        cellsize = 20.0
        c = np.eye(3) * cellsize
        p = np.vstack([vector for _,vector in local_structure]) / cellsize
        t = [self.t[i] for i,_ in local_structure]

        write_cif_file(filename, atom_from_cpt(c, p, t))


    def plot(self, site: int = None, filename:str = None):
        if len(self.dos.keys()) > 1 and not site:
            print(self.dos.keys())
            raise Exception("Compounds has multiple site, choose one")

        site = site or list(self.dos.keys())[0]

        filename = filename or self.filename
        compoundname = "${:s}$".format(format_formula(self.prefix))
        title = compoundname + " for {:d} atoms ({:s})".format(site+1, chemical_symbols[self.t[site]])

        dos = self.dos[site]
        fp = self.fingerprint[site]

        with plt.style.context(self.style):
            fig = plt.figure()
            axs = fig.subplots(2, 1)
            fig.set_size_inches(self.wsize, self.hsize * 2 - 1)
            
            axs[0].set_title(title)
            axs[0].set_xlim(0, 117)
            axs[0].set_xlabel("Fingerprint index")
            axs[0].set_ylabel("Fingerprint value")
            axs[0].yaxis.set_major_formatter(plt.NullFormatter())

            counter = 0
            for key, val in fp.items():
                x = np.arange(len(val))
                axs[0].plot(x, val + 1.5 * counter, label = key, alpha = 0.75)
                counter += 1
            axs[0].legend(frameon = True, fontsize = 'large', loc = "upper right")

            axs[1].set_xlabel("Energy (eV)")
            axs[1].set_xlim(-7.5, 7.5)
            axs[1].set_ylabel("DoS")
            axs[1].plot(self.energy, dos["up"], label = chemical_symbols[self.t[site]] + " up")
            axs[1].plot(self.energy, -dos["down"], label = chemical_symbols[self.t[site]] + " dw")
            axs[1].legend()
            fig.savefig(filename)


criteria_dictionary = { "first_shell" : occ_firstshell_rcut_5_smoothed_all,
                        "bonded": occ_bonded_rcut_5_smoothed_all,
                        "all" : occ_all_rcut_5_smoothed_all}

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("prefix", type = str, 
                        help = "in case of describe, should be a formula in calculated prefix, in case of predict, it should be file contain a structure")
                        
    parser.add_argument("-mode", type = str, default = "bonded", 
                        help = "first_shell, bonded or all")
    parser.add_argument("-site", type = int, default=None)
    parser.add_argument("-cif", default = False, action='store_true', help = "also write the cif file of the identified structure")
    
    args=parser.parse_args()

    c = criteria_dictionary[args.mode]

    cp = CompoundPlotter(args.prefix, criteria=c)
    cp.plot(args.site)

    if args.cif:
        cp.write_cif(args.site)



    

    
    

