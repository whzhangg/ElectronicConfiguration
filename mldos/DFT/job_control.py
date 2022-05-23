import os
import numpy as np
import warnings

from mldos.parameters import root
from mldos.tools import (
    is_formula, change_filetype, get_StructureCollection, return_doc_from_id,
    atom_from_cpt, write_cif_file, printProgressBar, make_target_directory
)

from ._output_parser import SCFout
from .calculation import SingleCalculation

dos_calculation = os.path.join(root, "dos_calculation")
DONE = os.path.join(root, "dos_calculation/calc/done")
FAIL = os.path.join(root, "dos_calculation/calc/failed")
WORK = os.path.join(root, "dos_calculation/calc/working")

class JobBase:

    # this class take care of prefixes, folders and initializing working folders for each calculation
    # actually, many of the function of this script can be taken care of using a bash script, which is more convenient

    DOSFOLDER = "dos_data"
    SCFFOLDER = "scf"
    
    def __init__(self, 
        donefolder: str = DONE,
        failfolder: str = FAIL,
        workfolder: str = WORK
    ) -> None:
        self.donefolder = donefolder
        self.failfolder = failfolder
        self.workfolder = workfolder

        self.folders = {}        # dictionary from prefix to folder position
        self.done_prefixs = []
        self.working_prefixs = []
        self.failed_prefixs = []  

        self._update_compounds()

    def print_number(self):
        print("number of compounds calculated: {:d}".format(len(self.done_prefixs)))
        print("number of compounds working   : {:d}".format(len(self.working_prefixs)))
        print("number of compounds failed    : {:d}".format(len(self.failed_prefixs)))

    def organize(self, show_only = False):
        """mv entries in the working folder to done or failed
        entry is considered DONE if dosresult file can be found in the entries "result.json"
        entry is considered failed if either scf.1.out, scf.0.out can be found but "result.json" is not present
        """
        possible_scf_out = [ change_filetype(f, "out") for f in SingleCalculation.scf_file ]  # scf_file = [ "scf.0.in", "scf.1.in" ]

        move_done = 0
        move_failed = 0
        for prefix in self.working_prefixs:
            files = os.listdir(self.folders[prefix])
            mv_target = ""
            if SingleCalculation.dos_result in files or "scf" in files:
                # move it to done
                if not show_only:
                    self._organize_folders(prefix)
                mv_target = os.path.join(self.donefolder, prefix)
                move_done += 1
            elif any([ f in files for f in possible_scf_out ]):
                # move it to fail
                mv_target = os.path.join(self.failfolder, prefix)
                move_failed += 1

            if mv_target:
                if os.path.exists(mv_target):
                    warnings.warn("JobBase.organize, Target folder already exist: " + mv_target)
                else:
                    if not show_only:
                        os.system("mv {:s} {:s}".format(self.folders[prefix], mv_target))
            
        print("to move to done: {:d}".format(move_done))
        print("to move to fail: {:d}".format(move_failed))
        self._update_compounds()
        self.print_number()

    def initialize(self, new_ids_dict:dict):
        """accept a dictionary with formula as key and database id as value"""
        StructureCollection = get_StructureCollection()
        print("{:d} Total formula input".format(len(new_ids_dict.keys())))

        all_prefixs = set(self.working_prefixs) | set(self.failed_prefixs) | set(self.done_prefixs)
        not_in = {}
        already_in = []
        for formula, iid in new_ids_dict.items():
            if formula in all_prefixs:
                already_in.append(formula)
            else:
                not_in[formula] = iid

        print("{:d} formula are already in the folder".format(len(already_in)))
        print("we add the remaining in the working folder, with prefix as folder name and a structure file")
        
        for f, iid in not_in.items():
            self.working_prefixs.append(f)
            self.folders[f] = os.path.join(self.WORK, f)

            make_target_directory(self.folders[f])

            doc = return_doc_from_id(StructureCollection, iid)
            c = np.array(doc["cell"]).reshape((3,3))
            p = np.array(doc["positions"]).reshape((-1,3))
            t = np.array(doc["types"])
            struct = atom_from_cpt(c, p, t)
            write_file = os.path.join(self.folders[f], f+".cif")
            write_cif_file(write_file, struct)

    def print_calculation_time(self):
        all_time = []

        tot_num = len(self.done_prefixs)
        step = max( int(tot_num/100), 1)

        print("Processing all scf.out ...")
        printProgressBar(0, tot_num)

        for i, prefix in enumerate(self.done_prefixs):
            folder = self.folders[prefix]
            scfout = os.path.join(folder, self.SCFFOLDER, "scf.out")
            out = SCFout(scfout)
            all_time.append(out.computation_time)
            if i % step == 0:
                printProgressBar(i+1, tot_num)
        printProgressBar(tot_num, tot_num)

        return np.array(all_time)
        
    def _organize_folders(self, prefix):

        folder = self.folders[prefix]
        if "dos_data" in os.listdir(folder):
            return

        dosfolder = os.path.join(folder, "dos_data")
        scffolder = os.path.join(folder, "scf")
        make_target_directory(dosfolder)
        make_target_directory(scffolder)
        current = os.getcwd()
        os.chdir(folder)

        os.system("mv {:s} dos_data".format(prefix+".pdos.*"))
        os.system("mv {:s} dos_data".format(SingleCalculation.dos_result) )
        os.system("mv {:s} scf".format("*.in"))
        os.system("mv {:s} scf".format("*.out"))

        os.chdir(current)

    def _update_compounds(self):
        self.folders = {}        # we clear the entries
        self.done_prefixs = []
        self.working_prefixs = []
        self.failed_prefixs = [] 
        # find all the prefixs
        folders = { self.failfolder: self.failed_prefixs,
                    self.donefolder: self.done_prefixs, 
                    self.workfolder: self.working_prefixs}

        for f, kind in folders.items():
            all_names = os.listdir(f)
            for name in all_names:
                if is_formula(name):
                    kind.append(name)
                    self.folders[name] = os.path.join(f, name)