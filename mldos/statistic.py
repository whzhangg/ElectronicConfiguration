# this script create the statistics for the DFT data

import os, re
from copy import deepcopy
from collections import Counter

from mldos.tools import separate_formula, change_filetype, printProgressBar, write_json, read_json, estimate_average_error
from mldos.parameters import TM_3D, root
from mldos.data.criteria import tm_maingroup_metal, tm_maingroup


info_filename = "compound_statistic.json"
current_folder = os.path.split(__file__)[0]
info_filepath = os.path.join(current_folder, info_filename)


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


class training_statistics():

    def __init__(self) -> None:
        self.compound_info = {}
        self.todo_prefix = []


    @classmethod
    def from_json(cls, filename: str):
        result = cls()
        result.compound_info = read_json(filename)
        result.todo_prefix = list(result.compound_info.keys())
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


    def get_training_compound_info(self, prefix: str) -> None:
        """return all useful information for a given compound"""
        assert prefix in self.compound_info.keys()
        return self.compound_info[prefix]


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

        for prefix in self.todo_prefix:
            
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
        assert prefix in self.todo_prefix

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


if __name__ == "__main__":
    training_statistics.from_json(info_filepath)    