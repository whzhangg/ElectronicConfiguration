# this script create the statistics for the DFT data

import os, re
import numpy as np
from ase.data import chemical_symbols, atomic_numbers
from copy import deepcopy
from collections import Counter

from mldos.tools import separate_formula, change_filetype, printProgressBar, write_json, read_json, estimate_average_error
from mldos.parameters import TM_3D, root, Simple_METAL
from mldos.DFT.job_control import JobBase
from mldos.DFT._output_parser import find_gap, SCFout
from mldos.data.milad_utility import Description
from mldos.data.data_generation import Data
from mldos.data.config import tm_maingroup_metal, tm_maingroup, all
from mldos.predict_occ import ensemble_general_predict


def _from_ase_atoms_2_pymatgen(ase_struct):
    from pymatgen.core.structure import Structure
    coord = ase_struct.get_scaled_positions()
    symbols = ase_struct.get_chemical_symbols()
    cell = ase_struct.get_cell()
    return Structure(lattice = cell, species = symbols, coords = coord)


def get_ase_site_env(ase, sites = None, nnn = None) -> list:
    # this method will try to find motif with coordination matching nnn given, but never larger than nnn

    from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
    from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
    from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

    struct = _from_ase_atoms_2_pymatgen(ase)

    result = {}

    if sites is not None:

        if nnn is None:
            nnn = [None] * len(sites)
        elif len(sites) != len(nnn):
            print("number of nn list should have the same length of sites")
            return None
        
        for i, s in enumerate(sites):
            lgf = LocalGeometryFinder()
            lgf.setup_structure(struct)
            #print(s, nnn[i])
            se = lgf.compute_structure_environments(only_indices = [s], max_cn = nnn[i])

            if i == 0:
                # only check at the first time
                symmetrized_struct = lgf.get_structure()

                cell_changed = False
                if not np.isclose(struct.lattice.matrix, symmetrized_struct.lattice.matrix, atol = 1e-2).all():
                    cell_changed = True
                if not np.isclose(np.array(struct.frac_coords), np.array(symmetrized_struct.frac_coords), atol = 1e-2).all():
                    cell_changed = True

                if cell_changed:
                    return None

            strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters() 
            #strategy = SimplestChemenvStrategy()
            lse = LightStructureEnvironments.from_structure_environments(strategy = strategy, structure_environments = se)

            data = lse.coordination_environments[s]
            
            if len(data) == 0:
                # if we cannot find with max_cn, we do it again without restriction
                se = lgf.compute_structure_environments(only_indices = [s], max_cn = None)
                strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters() 
                lse = LightStructureEnvironments.from_structure_environments(strategy = strategy, structure_environments = se)
                data = lse.coordination_environments[s]

            site_info = None
            if len(data) > 0:
                for candidate in data:
                    if len(candidate["permutation"]) == nnn[i]:

                        site_info = {"success" : True,
                                     "symbol" : candidate["ce_symbol"], 
                                     "csm" : candidate["csm"]
                                    }
            
                # data contain no entries
                if site_info is None:
                    
                    site_info = {"success" : True,
                                 "symbol" : data[0]["ce_symbol"], 
                                 "csm" : data[0]["csm"]
                                }

            else:
                site_info = {"success" : False, "symbol" : "", "csm" : 100.0}

            result[s] = site_info

    else:
        # site is None
        lgf = LocalGeometryFinder()
        lgf.setup_structure(struct)
        
        se = lgf.compute_structure_environments()

        symmetrized_struct = lgf.get_structure()

        cell_changed = False
        if not np.isclose(struct.lattice.matrix, symmetrized_struct.lattice.matrix, atol = 1e-2).all():
            cell_changed = True
        if not np.isclose(np.array(struct.frac_coords), np.array(symmetrized_struct.frac_coords), atol = 1e-2).all():
            cell_changed = True

        if cell_changed:
            return None

        strategy = SimplestChemenvStrategy()

        lse = LightStructureEnvironments.from_structure_environments(strategy = strategy, structure_environments = se)

        data = lse.coordination_environments
        for i, site in enumerate(data):

            site_info = {"symbol" : site[0]["ce_symbol"], 
                            "csm" : site[0]["csm"] }

            result[i] = site_info

    #print(result)
    return result
    


def combine_elements2str(eles: set):
    strr = ""
    for i, ele in enumerate(eles):
        if i > 0:
            strr += " "
        strr += str(ele)
    return strr


def format_formula_math(formula: str) -> str:
    parts = separate_formula(formula)
    result = ""
    for ele, count in parts.items():
        result += ele
        if int(count) > 1:
            result += "$_{{{:s}}}$".format(count)
    return result


def _line_from_list(items: list) -> str:
    """given a list, we return a string with & between each items and \\ to the end"""
    result = ""
    for i, item in enumerate(items):
        result += str(item)
        result += " "
        if i < len(items) - 1:
            result += "& "
        else:
            result += "\\\\"
    return result


def print_all_names(formulas: list, outputfile: str, title: str, ncol: int = 5):
    """it simply print all the names in a table"""

    latex  = "\\documentclass{article}\n"
    latex += "\\usepackage{multicol}\n"
    latex += "\\usepackage[margin=0.8in]{geometry}\n"
    latex += "\\begin{document}\n\n"
    latex += "\\title{{{:s}}}\n".format(title)
    latex += "\\date{}\n"
    latex += "\\maketitle\n\n"
       
    latex += "\\begin{{multicols}}{{{:d}}}\n".format(ncol)
    for prefix in formulas:
        latex += "{:s}\n\n".format(format_formula_math(prefix))
    latex += "\\end{multicols}\n"

    latex += "\\end{document}\n"

    with open(outputfile, "w") as f:
        f.write(latex)


toy_criteria = { 
  "firstshell_only" : True,
  "bonded_only" : False, 
  "cutoff" : 5.0, 
  "smooth" : True,
  "target" : "filling"
}


class training_statistics(Data):

    scf_folder = "scf"
    scf_result = "scf.out"

    def __init__(self, criteria: dict) -> None:
        super().__init__(criteria)
        self.scfoutputs = {}
        self.prediction_result = None
        self.compound_info = {}

    
    @classmethod
    def from_json(cls, filename: str):
        result = cls(toy_criteria)
        result.compound_info = read_json(filename)
        return result

    @staticmethod
    def get_shape_distribution(filename):
        ts = training_statistics.from_json(filename)
        shape = [ cinfo["polyhedra"] for cinfo in ts.compound_info.values() ]
        shape2 = []
        for s in shape:
            for p in s.values():
                shape2.append(p)
        return Counter(shape2)

    @staticmethod
    def estmiate_error_by_key(filename, key:str, display = False, given_element = None): 
        """given a key, we print average error estimated"""
        ts = training_statistics.from_json(filename)
        sorted = {}
        key_appearance = []
        for prefix in ts.todo_prefix:
            sites = ts.get_compound_information(prefix, get_prediction=True)["sites"]
            for site in sites:
                whichkey = site[key]
                key_appearance.append(whichkey)
                value = sorted.setdefault(whichkey, {"pred":[], "calc":[]})
                if site["C. up"] > 6.0 or site["C. dw"] > 6.0:  # added to remove the abnormal value in calculated result
                    continue

                if given_element and given_element not in prefix:
                    # only compounds contain given elements are considered
                    continue
                
                value["pred"].append((site["P. up"],site["P. dw"]))
                value["calc"].append((site["C. up"],site["C. dw"]))
        
        freq = Counter(key_appearance)
        for whichkey in freq.keys():
            error = estimate_average_error(sorted[whichkey]["pred"],sorted[whichkey]["calc"])
            sorted[whichkey]["error"] = error
            sorted[whichkey]["freq"] = freq[whichkey]
            if display:
                print("{:>10s}{:>10d}{:>10.3f}".format(whichkey, freq[whichkey], error["mae"]))

        return sorted

    @staticmethod
    def estmiate_error_by_distance(filename, display = False, given_element = None, given_element_set = None):
        """given a key, we print average error estimated"""
        ts = training_statistics.from_json(filename)
        sorted = {}
        freq = {}
        for prefix in ts.todo_prefix:
            compound = ts.get_compound_information(prefix, get_prediction=True)
            sites = compound["sites"]
            dist = round(compound["Dist. (\\r{{A}})"], 1)
            dist = str(dist)
            for site in sites:
                whichkey = dist
                if site["C. up"] > 6.0 or site["C. dw"] > 6.0:
                    continue

                if given_element and given_element not in prefix:
                    # only compounds contain given elements are considered
                    continue
                if given_element_set:
                    if given_element_set not in ["TM + Main Group", "TM + Main Group + Simple Metal", "All"]:
                        raise Exception("given element set should be a given string")
                    
                    if ts.which_subset_by_prefix(prefix) != given_element_set:
                        continue
                    
                freq.setdefault(whichkey, 0)
                freq[whichkey] += 1
                value = sorted.setdefault(whichkey, {"pred":[], "calc":[]})
                value["pred"].append((site["P. up"],site["P. dw"]))
                value["calc"].append((site["C. up"],site["C. dw"]))
        
        for whichkey in freq.keys():
            error = estimate_average_error(sorted[whichkey]["pred"],sorted[whichkey]["calc"])
            sorted[whichkey]["error"] = error
            sorted[whichkey]["freq"] = freq[whichkey]
            if display:
                print("{:>10s}{:>10d}{:>10.3f}".format(whichkey, freq[whichkey], error["mae"]))

        return sorted


    def _get_output(self, prefix):
        if prefix not in self.scfoutputs:
            file = os.path.join(self.folders[prefix], self.scf_folder, self.scf_result)
            self.scfoutputs[prefix] = SCFout(file)

        return self.scfoutputs[prefix]


    def _get_prediction(self, inputprefix):
        if not self.prediction_result:
            print("Making prediction once for all compounds ..")
            self.prediction_result = {}
            structlist = []
            sites = []

            counter = 0
            for i, prefix in enumerate(self._todo_prefixs):
                dos_data = self._get_dos_result(prefix)
                c = np.array(dos_data["cell"]).reshape((3,3))
                p = np.array(dos_data["positions"]).reshape((-1,3))
                t = dos_data["types"]
                unique = dos_data["unique_tm_index"]

                self.prediction_result[prefix] = {"which": [], "site": [], "predict": None}

                for uni in unique:
                    structlist.append((c,p,t))
                    sites.append(uni)
                    self.prediction_result[prefix]["which"].append(counter)
                    self.prediction_result[prefix]["site"].append(uni)
                    counter += 1
        
            result = ensemble_general_predict(structlist, sites)

            for k, v in self.prediction_result.items():
                v["predict"] = {  s : (float(result[w,0]),float(result[w,1])) for s, w in zip(v["site"], v["which"]) }
                # force float for exporting to json
                
        return self.prediction_result[inputprefix]["predict"]


    def get_training_compound_info(self, prefix) -> None:
        """return all useful information for a given compound"""

        assert prefix in self.todo_prefix

        if prefix in self.compound_info.keys():
            return self.compound_info[prefix]

        result = { }

        dos_data = self._get_dos_result(prefix)
        c = np.array(dos_data["cell"]).reshape((3,3))
        p = np.array(dos_data["positions"]).reshape((-1,3))
        t = dos_data["types"]
        unique = dos_data["unique_tm_index"] 

        # get neighbors # as dictionary for each site
        sd = Description(cell = c, positions = p, types = t, keepstructure = True)
        neighbors = sd.get_neighbors( unique, bonded = False, firstshell = True, cutoff = 5.0)
        output = {}
        for iatom, neighbor in neighbors.items():
            # list of tuple with chemical symbol and distance
            output[iatom] = [ (chemical_symbols[sd.std_types[i]], np.linalg.norm(rij)) for i,rij in neighbor]

        result["neighbors"] = output

        # get second vornoi
        neighbors = sd.get_neighbors( unique, bonded = True, firstshell = False, cutoff = 5.0)
        output = {}
        for iatom, neighbor in neighbors.items():
            # list of tuple with chemical symbol and distance
            output[iatom] = [ (chemical_symbols[sd.std_types[i]], np.linalg.norm(rij)) for i,rij in neighbor]

        result["neighbor_voronoi"] = output

        # get second vornoi
        neighbors = sd.get_neighbors( unique, bonded = False, firstshell = False, cutoff = 5.0)
        output = {}
        for iatom, neighbor in neighbors.items():
            # list of tuple with chemical symbol and distance
            output[iatom] = [ (chemical_symbols[sd.std_types[i]], np.linalg.norm(rij)) for i,rij in neighbor]

        result["neighbor_cutoff"] = output

        # get polyhedra shape, if successful, as dictionary for each site
        site_nnn = np.array([ [iatom, len(nn)-1] for iatom, nn in neighbors.items() ], dtype = int)
        try:
            poly = get_ase_site_env(sd.standard, site_nnn[:,0], site_nnn[:,1])
            result["polyhedra"] = { int(k):val["symbol"] for k, val in poly.items() }  # int for exporting to json
        except:
            result["polyhedra"] = {iatom:"" for iatom in neighbors.keys()}

        # get distance, float for structure
        neighbors = sd.get_neighbors( unique, bonded = False,
                                      firstshell = False, cutoff = 10.0)
        distance = 1.0e10 # should be large enough
        for iatom, neighbor in neighbors.items():
            metal_metal_distance = [ np.linalg.norm(rij) for index,rij in neighbor if (chemical_symbols[sd.std_types[index]] in TM_3D and np.linalg.norm(rij) > 0.1) ]
            distance = min([distance, min(metal_metal_distance)])
        result["distance"] = distance # 

        # get structure type, string for structure
        result["prototype"] = sd.names["prototype"]
        result["sgnum"] = sd.sgnum

        # get calculation value and prediction value
        filling = self.get_filling(prefix)
        result["calc"] = { iatom : (f["up"], f["down"]) for iatom, f in filling.items() }

        result["pred"] = self._get_prediction(prefix)

        # get bandgap
        result["gap"] = self.get_bandgap(prefix)

        return result


    def save_all_information(self, filename):
        all = {}
        
        tot_num = len(self.todo_prefix)
        step = max( int(tot_num/100), 1)
        printProgressBar(0, tot_num)

        for i,prefix in enumerate(self.todo_prefix):
            all[prefix] = self.get_training_compound_info(prefix)
            
            if i % step == 0:
                printProgressBar(i+1, tot_num)

        
        printProgressBar(tot_num, tot_num)
        write_json(all, filename)


    def get_bandgap(self, prefix) -> float:
        assert prefix in self._todo_prefixs
        
        scfout = self._get_output(prefix)
        gap = find_gap(scfout.eig, scfout.occupations)
        try:
            gap_value = min([gap[s]["cbm"] for s in gap]) - max([gap[s]["vbm"] for s in gap])
        except:
            gap_value = 0.0
        return max([gap_value, 0.0])


    def which_subset_by_prefix(self, prefix) -> str:
        parts = set(separate_formula(prefix).keys())

        compound_type = "All"
        if parts < tm_maingroup:
            compound_type = "TM + Main Group"
        elif parts < tm_maingroup_metal:
            compound_type = "TM + Main Group + Simple Metal"
            
        return compound_type


    def get_sorted_prefixs(self) -> list:
        """we first sort by TM (atomic number), then by the anions (atomic number)"""

        result_dict = {"TM + Main Group": [], "TM + Main Group + Simple Metal": [], "All": []}
        sorted_result_dict = {}

        for prefix in self._todo_prefixs:
            
            parts = set(separate_formula(prefix).keys())
            if len(parts & TM_3D) > 1:
                continue

            compound_type = self.which_subset_by_prefix(prefix)

            result_dict.setdefault(compound_type, []).append(prefix)

        for key, prefixs in result_dict.items():

            print(key)
            print(len(prefixs))
            
            tosort = []
            for prefix in prefixs:
                sg = self.get_training_compound_info(prefix)["sgnum"]
                ws = self.get_training_compound_info(prefix)["prototype"]
                tosort.append((prefix, sg, ws))


            tosort.sort(key=lambda row: row[2])
            tosort.sort(key=lambda row: row[1])
            
            sorted_result_dict[key] = [ p for p,s,w in tosort ]

        return sorted_result_dict


    def get_compound_information(self, prefix, get_prediction = False) -> dict:
        assert prefix in self._todo_prefixs

        metal_metal = self.get_training_compound_info(prefix)["distance"]
        gap         = self.get_training_compound_info(prefix)["gap"]
        filling     = self.get_training_compound_info(prefix)["calc"]
        prediction  = self.get_training_compound_info(prefix)["pred"]
        first_shell = self.get_training_compound_info(prefix)["neighbors"]
        prototype   = self.get_training_compound_info(prefix)["prototype"]
        shape       = self.get_training_compound_info(prefix)["polyhedra"]
        sg          = self.get_training_compound_info(prefix)["sgnum"]
        prototype = re.sub('_','\_', prototype)

        for k,v in shape.items():
            if v == "O:6":
                shape[k] = "Octa"
            elif v == "T:4":
                shape[k] = "Tetra"
            else:
                shape[k] = "Other"

        sites = []
        for iatom, neighbor in first_shell.items():

            elements = set([symbol for (symbol,rij) in neighbor])
            tm = elements & TM_3D
            anion = elements - tm
            calc = filling[iatom]
            
            info = { "T.M.": combine_elements2str(tm), 
                     "Anion" : combine_elements2str(anion), 
                     "shape" : re.sub('_','', shape[iatom]),
                     "C. up": round(calc[0],2), 
                     "C. dw": round(calc[1],2)
                    }
            
            if get_prediction:
                info["P. up"] = round(prediction[iatom][0],2)
                info["P. dw"] = round(prediction[iatom][1],2)
            
            sites.append(info)
        
        return {"Formula" : format_formula_math(prefix), "SG": sg, "prototype": "${:s}$".format(prototype), "E$_g$ (eV)" : round(gap,2), "Dist. (\\r{{A}})": round(metal_metal,2), "sites": sites}


    def get_compound_information_lines(self, prefix, get_prediction = False) -> dict:

        dictionary = self.get_compound_information(prefix, get_prediction=get_prediction)
        empty = { k:"" for k in dictionary.keys() if k != "sites"}
        not_empty = { k:str(v) for k,v in dictionary.items() if k != "sites"}
        result = []

        for i,site in enumerate(dictionary["sites"]):
            site_str = {k:str(v) for k,v in site.items()}
            if i == 0:
                copy_dict = deepcopy(not_empty)
                not_empty.update(site_str)
                result.append(not_empty)
            else:
                copy_dict = deepcopy(empty)
                copy_dict.update(site_str)
                result.append(copy_dict)
        return result


    def print_all_names(self, ncol = 5, filename = "formula_listing.tex", folder = os.path.join(root, "docs")):
        """it simply print all the names in a table"""
        title = "Compounds List as Training Samples"

        sorted_prefix_dict = self.get_sorted_prefixs()

        latex  = "\\documentclass{article}\n"
        latex += "\\usepackage{multicol}\n"
        latex += "\\usepackage[margin=0.8in]{geometry}\n"
        latex += "\\begin{document}\n\n"
        latex += "\\title{{{:s}}}\n".format(title)
        latex += "\\date{\\today}\n"
        latex += "\\maketitle\n\n"
       
        for key, sorted_prefix in sorted_prefix_dict.items():

            latex += "\\section{{{:>s}}}\n".format(key)
            latex += ""
            latex += "\\begin{{multicols}}{{{:d}}}\n".format(ncol)
            for p in sorted_prefix:
                latex += "{:s}\n\n".format(format_formula_math(p))
            latex += "\\end{multicols}\n"

        latex += "\\end{document}\n"

        with open(filename, "w") as f:
            f.write(latex)

        os.system("pdflatex {:s}".format(filename))
        print("rm {:s}".format(change_filetype(filename, "aux")))
        os.system("rm {:s}".format(change_filetype(filename, "aux")))
        os.system("rm {:s}".format(change_filetype(filename, "log")))
        #os.system("rm {:s}".format(filename))
        os.system("mv {:s} {:s}".format(filename, folder))
        os.system("mv {:s} {:s}".format(change_filetype(filename, "pdf"), folder))


    def print_compound_details(self, filename = "compound_listing.tex", test = False):
        title = "Details of the Training Samples"
        sorted_prefix_dict = self.get_sorted_prefixs()

        latex  = "\\documentclass{article}\n"
        latex += "\\usepackage{longtable}\n"
        latex += "\\usepackage[margin=0.8in]{geometry}\n"
        latex += "\\begin{document}\n\n"
        latex += "\\title{{{:s}}}\n".format(title)
        latex += "\\date{ }\n"
        latex += "\\maketitle\n\n"
        latex += "\n"
        latex += "This list shows all the compounds that we included in the training set. This table is \n"
        latex += "organized into three groups: \n"
        latex += "\\begin{enumerate} \n"
        latex += "    \item \\textbf{TM + Main Group}: contain compounds that have only tranition metal elements and  \n"
        latex += "    anions from the main group (without B, C, N, O, F, as explained in the main article) \n"
        latex += "    \item \\textbf{TM + Main Group + Simple Metal}: In addition to the previous criteria, we allow the  \n"
        latex += "    compounds to contain simple metals with $s$ electrons \n"
        latex += "    \item \\textbf{All}: The compounds in this group can further contain 4$d$ elements from the 5$^{th}$ and  \n"
        latex += "    6$^{th}$ row \n"
        latex += "\\end{enumerate} \n"
        latex += "\n"

        latex += "For each compound, \n"
        latex += "we first report its space group, followed by a structure description (type). The structure type is given as \n"
        latex += "fields separated by underscore \"\_\". Each following field specifies the Wyckoff positions for every element \n"
        latex += "in the chemical formula order. If one elements take multiple Wyckoff positions, all Wyckoff symbols are combined. \n"
        latex += "Numbers indicate number of distinct Wyckoff pisitions. For example, for space group 143, $bc4d$ indicate will correspond to 14 atoms: \n"
        latex += "two at site $b$ and $c$ with multiplicity 1, 12 at 4 different $d$ positions, each with multiplicity 3. \n"
        latex += "\n"
        latex += "Then, we show their calculated band gap, metal metal distance.  \n"
        latex += "Finally, the local tranition metal site information is displayed including the type of transition metal elements \n"
        latex += "at the center and the anions in the first shell.  \n"
        latex += "Finally, the calculated electronic occupation and the predicted occupation are listed in terms of electron counts. \n"
        
        prediction = False
        example_prefix = sorted_prefix_dict[list(sorted_prefix_dict.keys())[0]][0]
        example_dict = self.get_compound_information_lines(example_prefix, get_prediction=prediction)[0]
        keys = list(example_dict.keys())
        arrange = "l"+"r"*(len(keys)-1)

        
        tot_num = sum([ len(val) for val in sorted_prefix_dict.values()])
        step = max( int(tot_num/100), 1)
        print("Generating all compound details")
        printProgressBar(0, tot_num)
        i = 0
        for key, sorted_prefix in sorted_prefix_dict.items():

            latex += "\\section{{{:>s}}}\n".format(key)
            latex += ""
            latex += "\\begin{{longtable}}{{{:s}}}\n".format(arrange)
            latex += _line_from_list(keys) + "\n\\hline \n"
            latex += "\\endhead\n"
            latex += "\\hline \\multicolumn{{{:d}}}{{r}}{{\\textit{{Finish}}}} \\\\\n" .format(len(keys))
            latex += "\\endlastfoot\n"
            latex += "\\hline \\multicolumn{{{:d}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\ \n\\endfoot\n".format(len(keys))
            for p in sorted_prefix:
                lines = self.get_compound_information_lines(p, get_prediction=prediction)
                for line in lines:
                    latex += _line_from_list([it for it in line.values()]) + "\n"

                if i % step == 0:
                    printProgressBar(i+1, tot_num)
                i += 1
                if test and i > 20: 
                    break
            latex += "\\end{longtable}\n"
            latex += "\n"
            latex += "\\newpage\n"

        printProgressBar(tot_num, tot_num)
        latex += "\\end{document}\n"

        with open(filename, "w") as f:
            f.write(latex)



info_file = os.path.join(root, "mldos", "models", "compound_statistic.json")


def search_for_example():
    stat = training_statistics.from_json(info_file)
    for prefix in stat.todo_prefix:
        for site in stat.get_training_compound_info(prefix)["polyhedra"].keys():
            shape = stat.get_training_compound_info(prefix)["polyhedra"][site]

            first = stat.get_training_compound_info(prefix)["neighbors"][site]
            voron = stat.get_training_compound_info(prefix)["neighbor_voronoi"][site]
            cutof = stat.get_training_compound_info(prefix)["neighbor_cutoff"][site]

            voron_not_in_first = [ ]
            for item in voron:
                for item2 in first:
                    if item[0] == item2[0] and item[1] - item2[1] < 0.001 :
                        voron_not_in_first.append(item)
                        break
            
            voron_not_in_first = [ item[0] for item in voron if item not in voron_not_in_first ]
            voron_not_in_first = set(voron_not_in_first)

            cutof_not_in_voron = [ ]
            for item in cutof:
                for item2 in voron:
                    if item[0] == item2[0] and item[1] - item2[1] < 0.001 :
                        cutof_not_in_voron.append(item)
                        break
                    
            cutof_not_in_voron = [ item[0] for item in cutof if item not in cutof_not_in_voron ]
            cutof_not_in_voron = set(cutof_not_in_voron)
            
            
            s_metal = set(Simple_METAL)
            voron_are_s = voron_not_in_first < (s_metal | set(TM_3D))
            cutof_not_s = len(cutof_not_in_voron & s_metal) == 0
            #cutof_not_s = cutof_not_s and len(cutof_not_in_voron & set(TM_3D)) > 0

            contain_s = len(s_metal & set([a[0] for a in cutof])) > 0

            if voron_are_s and cutof_not_s and len(first) < len(voron) and len(voron) < len(cutof) and shape == "O:6":
                print(contain_s)
                print(prefix)
                print(voron_not_in_first)
                print(cutof_not_in_voron)
                print(shape)
                print(stat.get_training_compound_info(prefix)["sgnum"])
                break


if __name__ == "__main__":
    filename = os.path.join("", "compound_listing_training.tex")
    #stat = training_statistics(toy_criteria)
    #stat.save_all_information(info_file)
    #stat = training_statistics.from_json(info_file)
    #stat.print_compound_details(filename)
    search_for_example()
    