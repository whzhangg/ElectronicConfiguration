
import re
import warnings
import numpy as np
from spglib import get_symmetry_dataset, standardize_cell
from ase.data import chemical_symbols
from collections import Counter

from mldos.tools import read_generic_to_cpt, atom_from_cpt
from ._element_info import pp_info


def make_extended_wychoffs(equivalent, wyckoffs):
    ext_wyckoffs = []  
    count = Counter(equivalent)
    for i,eqat in enumerate(equivalent):
        ext_wyckoffs.append(str(count[eqat]) + wyckoffs[i])
    return ext_wyckoffs


def make_formula(elements:list, reduce = True) -> str:
    #
    # elements -> a list of element types
    # the formula is sorted by the sequence of column
    # the element counts are reduced, with 1 also included in the formula
    #
    counter = Counter(elements)

    to_sort = list(set(elements)) 
    column = []
    for ss in to_sort:
        symbol = chemical_symbols[ss]
        info = pp_info[symbol]
        column.append( info["order"]*1000 + ss) 
        # this ensure that if order is the same, we sort depend on atomic number
    sorting = np.argsort(column)
    sorted_set = []
    for i in sorting:
        sorted_set.append(to_sort[i])
        
    sortedcount = []
    for st in sorted_set:
        sortedcount.append(counter[st])

    if reduce:
        reduced_count = np.array(sortedcount,dtype=int)
        a = np.gcd.reduce(reduced_count)
        reduced_count = reduced_count // int(a)
    else:
        reduced_count = sortedcount

    formula = ""
    for i,t in enumerate(sorted_set):
        formula += chemical_symbols[t] + str(reduced_count[i])
    return formula
      

class StructLite:
    
    def __init__(self, 
                cell = None, positions = None, types = None, 
                occupancies = None, description = "", 
                filename = None, document = None, keepstructure = False,
                tolerance = 1e-3) -> None:
        """
        three way to initialize:
        1. from cell, positions, types:
            in this case, we optionally supply occupancies and description
        2. from filename
            in this case, we only need to supply filename
        3. from a document
            we only need to supply a dict form of document

        keep structure tag prevent standardizing the structures
        """
        
        self.tolerance = tolerance
        self.keepstructure = keepstructure

        self.success = True
        self.error = ""

        if type(cell) != type(None) and type(positions) != type(None) and type(types) != type(None):
            self.description = description
            if type(occupancies) == type(None):
                occupancies == [1.0] * len(positions)
            self._make_structure(cell, positions, types, occupancies)

        elif filename != None:
            self._from_file(filename)

        elif document != None:
            self._from_doc(document)

        else:
            self.success = False
            self.error = "not input structure"


    def _from_doc(self, js:dict) -> None:
        # a wrapper to initialize from document dic:
        self._from_simplestructure_doc(js)


    def _from_simplestructure_doc(self, document):
        formula = document["formula"]
        description = document["description"]
        cell = np.array(document["cell"]).reshape(3,3)
        positions = np.array(document["positions"]).reshape(-1,3)
        types = np.array(document["types"], dtype = int)
        occupancies = np.array(document["occupancies"])
        
        self.description = description
        self.occupancies = occupancies
        self._make_structure(cell, positions, types, occupancies)
        return


    def _from_file(self, filename:str) -> None:
        cell, positions, types = read_generic_to_cpt(filename)
        occupancies = np.array([1.0] * len(positions))
        self.description = None
        self.description = "from file {:>s}".format(filename)
        self._make_structure(cell, positions, types, occupancies)


    def _make_structure(self, cell, positions, types, occupancies):
        # a wrapper to call self._standardize
        self._standardize(cell, positions, types, occupancies, self.tolerance)


    def _standardize(self, cell, positions, types, occupancies, tolerance, load = False):
        # the main method that describe a structure
        #
        struct = (cell,positions,types)

        # not yet to able to treat partial occupancy
        self.occupancies = occupancies 
        
        # idealize will shift position so that wyckoff positions agree with international table

        if not self.keepstructure:
            std_lattice, std_pos, std_types = standardize_cell(struct, symprec = tolerance, no_idealize = False)
        else:
            # if we read from some standardized data, we need to prevent further change which may be introduced by standardize_cell
            std_lattice = cell
            std_pos = positions
            std_types = types

        self.standard = atom_from_cpt(std_lattice, std_pos, std_types) 
        self.std_lattice = std_lattice
        self.std_positions = std_pos
        self.std_types = std_types
        self.std_nat = len(std_types)
        self.valence = self._valence_count(std_types) 
        self.formula = make_formula(std_types)

        
        self.symdataset = get_symmetry_dataset((std_lattice, std_pos, std_types), symprec = tolerance)
        
        self.sgnum = self.symdataset["number"]
        self.sg = self.symdataset["international"]
        self.pointgroup = self.symdataset["pointgroup"]
        self.choice = self.symdataset["choice"]
        self.pearson = self._make_pearson_symbol()

        # information on atoms
        site_symmetry = self.symdataset["site_symmetry_symbols"]
        self.equivalent_atoms = self.symdataset["equivalent_atoms"]
        self.ext_wyckoffs = make_extended_wychoffs(self.equivalent_atoms, self.symdataset["wyckoffs"])
        # eg : with multiplicity ['8c','8c','4a', ...]

        if not load:
            # a simple site, which can be decorate later
            self.sites = {}
            for i, ext_wp in enumerate(self.ext_wyckoffs):
                if ext_wp not in self.sites.keys():
                    self.sites[ext_wp] = {}
                
                if self.equivalent_atoms[i] not in self.sites[ext_wp].keys():
                    neq_wp = {"chemical_symbol":chemical_symbols[self.std_types[i]],
                            "type" : self.std_types[i],
                            "symmetry" : site_symmetry[i],
                            "equivalent_indices" : [i]
                            }
                    self.sites[ext_wp][self.equivalent_atoms[i]] = neq_wp

                else:
                    self.sites[ext_wp][self.equivalent_atoms[i]]["equivalent_indices"].append(i)

            contain_wp_error = False
            if not self.keepstructure:
                from .wyckoff import WyckoffPosition
                for wp in self.sites:
                    for ineq in self.sites[wp]:
                        coords = self.std_positions[self.sites[wp][ineq]["equivalent_indices"]]
                        wyckoff = WyckoffPosition(str(self.sgnum), wp)
                        wyckoff.find_free_parameter(coords)
                        self.sites[wp][ineq]["wyckoff_position"] = wyckoff.xyz
                        self.sites[wp][ineq]["wyckoff_ok"] = wyckoff.success
                        if not wyckoff.success:
                            contain_wp_error = True
                            warnings.warn(str(self.sgnum) + "  " + wp + " site cannot find standard wyckoff position")

            if contain_wp_error:
                self.error += "wyckoff position may not comparable to international table\n"
                    
        abstract_types = []
        for t in self.std_types:
            symbol = chemical_symbols[t]
            info = pp_info[symbol]
            abstract_types.append(info["element_top_of_column"])

        self.absformula = make_formula(abstract_types, reduce = False)
        
        self.names = self._make_names()


    def _make_names(self) -> dict:
        """
        rule: field separated by _, parts in the same field separated by -
        compound name: formula _ spacegroup _ [element - wyckoff] 
        """
        sorted_ele = re.findall(r'[A-z][a-z]*', self.formula)

        wyckoffs = [tmp[-1] for tmp in self.ext_wyckoffs] 
        wyckoff_list = list(set(wyckoffs))  # alphabet [a,b,c] in order
        wyckoff_list.sort()
        sorted_ext_wyckoff = []
        for iw in wyckoff_list:
            for jw in self.ext_wyckoffs:
                if iw in jw:
                    sorted_ext_wyckoff.append(jw)
                    break

        element_wp = {}
        for wp in self.sites:
            for neq in self.sites[wp]:
                tmp = self.sites[wp][neq]
                if tmp["chemical_symbol"] not in element_wp:
                    element_wp[tmp["chemical_symbol"]] = []
                element_wp[tmp["chemical_symbol"]].append(wp)

        all_names = {}
        # compound name
        # FeSb2: Fe1Sb2_58_Fe-2a1_Sb-4g1 : Fe is in a single 2a site
        compound_name = self.formula + "_" + str(self.sgnum) 
        for ele in sorted_ele:
            compound_name += "_" + ele
            wpset = element_wp[ele]
            counter = Counter(wpset)
            
            for iwp in sorted_ext_wyckoff:
                if iwp in counter:
                    compound_name += "-" + iwp + str(counter[iwp])
        all_names["compound"] = compound_name

        # agree with ao flow type: 
        # eg FeSb2
        # aflow : AB2_oP6_58_a_g
        # here  : Fe1Sb2_oP6_58_a_g
        prototype_name = ""   # pearson symbol is removed
        for i, ele in enumerate(sorted_ele):
            if i > 0:
                prototype_name += "_"
            wpset = element_wp[ele]
            counter = Counter(wpset)
            for iwp in sorted_ext_wyckoff:
                if iwp in counter:
                    if counter[iwp] == 1:
                        prototype_name +=  iwp[-1]
                    else:
                        prototype_name +=  str(counter[iwp]) + iwp[-1]

        all_names["prototype"] = prototype_name

        # classification name
        class_name = str(self.sgnum) 

        counter = Counter(self.ext_wyckoffs)

        for iwp in sorted_ext_wyckoff:
            class_name += "_" + iwp + str(counter[iwp])

        class_name += "_" + str(self.absformula)
        class_name += "_" + str(self.valence) 

        all_names["class"] = class_name

        return all_names

    def _make_wyck_list(self, extended_wyckoff_lists):
        # first get the sorted set of wyckoff 
        wlist = list(set(extended_wyckoff_lists))
        wlist.sort(key=lambda word: word[-1])
        wcount = Counter(extended_wyckoff_lists)
        result = ""
        for i,w in enumerate(wlist):
            multiplicity = int(w[:-1])
            assert wcount[w] % multiplicity == 0
            if i > 0:
                result += "_"
            result += "{:d}{:s}".format(int(wcount[w] / multiplicity),  w[-1])
        return result


    def _partial_occupancy(self):
        # partial occupancy is not supported, since I don't have 
        # a way to store recognize the partial occupancy after standardization
        # and other structure types also do not support it well.
        return


    def _valence_count(self, types) -> int:
        count = 0
        for t in types:
            symbol = chemical_symbols[t]
            info = pp_info[symbol]
            count += info["valence"] - info["core"]
        return count


    def _make_pearson_symbol(self) -> str:
        if self.sgnum <= 2:
            p = "a"
        elif self.sgnum <= 15:
            p = "m"
        elif self.sgnum <= 74:
            p = "o"
        elif self.sgnum <= 142:
            p = "t"
        elif self.sgnum <= 194:
            p = "h"
        else:
            p = "c"

        p += self.sg[0]
        p += str(self.std_nat)
        return p


    def __eq__(self, other):
        # two equivalent structure would be equal 
        # formula, space group and wychoff positions 
        return self.names["compound"] == other.names["compound"]


    @property
    def text_description(self):
        text = self.description + "\n"
        text += "chemical formula  : {:s}\n".format(self.formula)
        text += "space group       : {:s} {:>3d}".format(self.sg, self.sgnum)
        if self.choice != "":
            text += "  cell choice {:s}".format(self.choice)
        text += "\n"
        text += "number of atoms   : {:d}\n".format(self.std_nat)
        
        text += "list of wyckoff positions\n"
        text += " ele.     WP   symmetry                 (x, y, z)\n"
        for wp in self.sites:
            for unique in self.sites[wp]:
                tmp = self.sites[wp][unique]
                wkp = tmp["wyckoff_position"]
                text += "  {:>2s}   {:>5s}    {:>7s}    ({:>6.3f},{:>6.3f},{:>6.3f})\n".format(tmp["chemical_symbol"], wp, tmp["symmetry"], 
                                                        wkp[0], wkp[1], wkp[2])
                
        return text
        

def test_230spacegroup():
    import os 
    directory = "../tests/230space_group/"
    files = os.listdir(directory)

    for f in files:
        if '.cif' in f:
            fn = directory + f
            a = StructLite(filename = fn)
            #print(a.text_description)

