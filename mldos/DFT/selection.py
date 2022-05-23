import numpy as np
from ase.data import chemical_symbols
from mldos.tools import read_json, write_json, printProgressBar, get_site_from_doc, get_StructureCollection
from .crystal import make_formula


def random_sample(choice, formula, nsample, seed:int = None):
    seed = seed or int(np.random.rand() * 100)

    #assert nsample <= len(choice)
    if nsample >= len(choice):
        return (choice, formula)
    
    rng = np.random.default_rng(seed = seed)
    indices = np.arange(len(choice))
    rng.shuffle(indices)
    return ([choice[i] for i in indices[0: nsample]], [formula[i] for i in indices[0: nsample]])


class BaseFilter:

    # this class provide filtered result for further calculation
    # it should provide an interface between filtered structure and 
    # their database entry
    # it should also save the filtered results

    unique_formula = True

    def __init__(self, 
        criteria:dict, nsample:int, 
        sampled_ids: list = None, all_ids:list = None, all_formulas:list = None
    ):
        self.criteria = criteria
        self.nsample = nsample
        if sampled_ids:
            assert len(sampled_ids) == nsample
            assert len(all_ids) == len(all_formulas)

            self.sampled = sampled_ids
            self.ids = all_ids
            self.formulas = all_formulas
            self.tot_num = len(self.ids)
        else:
            self._filter()  # make ids, formulas
            self.tot_num = len(self.ids)
            self.sampled, self.sampled_formula = random_sample(self.ids, self.formulas, self.nsample)
            print("filtering done, find in total {:>d}".format(self.tot_num))

    @classmethod
    def from_file(cls, filename: str):
        assert ".json" in filename

        data = read_json(filename)

        criteria = data["criteria"]
        sampled_ids = data["sampled"]
        all_ids = data["ids"]
        all_formulas = data["formulas"]

        return cls(criteria, len(sampled_ids), sampled_ids, all_ids, all_formulas)
        
    def write2_json(self, filename: str):
        write_json(self.as_dict, filename)

    @property
    def formula_id(self):
        return { f:id for (f,id) in zip(self.sampled_formula, self.sampled)}

    @property
    def as_dict(self):
        result = {}
        result["criteria"] = self.criteria
        result["tot_num"] = self.tot_num
        result["sample_num"] = self.nsample
        result["sampled"] = self.sampled     # sampled ids
        result["ids"] = self.ids             # all the candidate ids
        result["formulas"] = self.formulas   # all the formulas
        return result
    
    def _filter(self):
        """
        find the entries (id) from the database that 
        match the given criteria
        give value to:
            self.ids
            self.formulas
            self.tot_num
        """
        self.ids = []
        self.formulas = []
        database = self.criteria["database"]
        print("we perform filtering using {:s}".format(database))

        database = get_StructureCollection()
        tot_entries = database.count_documents({"error": ""})
        print("Total number of entries in the database {:>d}".format(tot_entries))
        printProgressBar(0, tot_entries)
        step = max( int(tot_entries/100), 1)
        
        for i, doc in enumerate(database.find({"error": ""})):
        
            types = doc["types"]
            elements_set = set( [ chemical_symbols[t] for t in types] )

            if type(self.criteria["tot_atom"]) is int:
                if len(types) > self.criteria["tot_atom"]:
                    continue
            elif len(self.criteria["tot_atom"]) == 2:
                lb, ub = self.criteria["tot_atom"]
                if len(types) < lb and len(types) > ub:
                    continue
            else:
                raise

            is_subset = elements_set < set(self.criteria["element_set"])
            if not is_subset:
                continue

            contained_center = elements_set & set(self.criteria["center_site"])
            if len(contained_center) > self.criteria["max_different_type"]:
                continue

            sites = get_site_from_doc(doc)
            found = False
            for info in (sites[wp][ineq] for wp in sites for ineq in sites[wp] 
                            if sites[wp][ineq]["chemical_symbol"] in self.criteria["center_site"]):
                neighbor = info["neighbors_symbols"]
                poly = info["poly_symbol"]

                if self.criteria["poly_set"]:
                    if poly not in self.criteria["poly_set"]:
                        continue
                        
                if set(neighbor) < set(self.criteria["neighbor_set"]):
                    found = True
                    break

            if found:
                f = make_formula(types)
                if self.unique_formula and f in self.formulas:
                    pass
                else:
                    self.ids.append(str(doc["_id"]))
                    self.formulas.append(f)

            if i % step == 0:
                printProgressBar(i+1, tot_entries)

        printProgressBar(tot_entries, tot_entries)

