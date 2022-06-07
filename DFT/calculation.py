import os
import warnings
import argparse
import numpy as np
from copy import deepcopy
from ase.data import chemical_symbols

from mldos.tools import change_filetype, read_generic_to_cpt, write_json
from mldos.parameters import TM_3D, estep_dos
from .crystal import StructLite
from ._qe_parser import FileParser, QeInput
from ._output_parser import parse_pdos_folder, NscfOut, SCFout


class SingleCalculation:
    """ this class produce density of state json result starting from an appropriate structurefile, for example, cif
    """
    EPS = 1e-3
    scf_file = ["scf.0.in", "scf.1.in"]
    dos_result = "result.json"

    def __init__(self, prefix, directory, cell, positions, types):
        self.cell = cell
        self.positions = positions
        self.types = types
        
        self.prefix = prefix
        self.directory = directory
        self.status = 0
        self.fp = FileParser(prefix = prefix, directory = directory)
        return

    @classmethod
    def fromFile(cls, prefix, directory, structfile):
        cell, positions, types = read_generic_to_cpt(structfile)
        #cell, positions, types = read_generic_to_cpt(os.path.join(directory,structfile))  
        # I made a change to use absolute path instead of relative path in terms of structure
        return cls(prefix, directory, cell, positions, types)


    def initialize_scf(self):
        """write two scf file with different starting magnetization"""
        qe = QeInput(prefix = self.prefix, lattice = self.cell, positions = self.positions, types = self.types, nspin = 2)
        scfinput = qe.getinput_pwscf("scf")

        scfinput["&control"]["disk_io"] = 'none'

        input = self._add_starting_magnetization(scfinput, with_3d_moments= False)
        self.fp.write_file(self.scf_file[0], self.fp.make_pwscf(input))

        input = self._add_starting_magnetization(scfinput, with_3d_moments= True)
        self.fp.write_file(self.scf_file[1], self.fp.make_pwscf(input))


    def make_dos_inputs(self):
        result = self._select_ground_state(filenamelist = self.scf_file)
        
        if self.status == -1:
            warnings.warn(self.directory + ": ground state cannot be found")
            return

        qe = QeInput(prefix = self.prefix, lattice = self.cell, positions =  self.positions, 
                     types =  self.types, nspin = 2,  k_grid_multiplier = 1.3)

        basic_input = qe.getinput_pwscf("scf")
        basic_input["&system"]["smearing"] = ""
        basic_input["&system"]["degauss"] = ""
        basic_input["&system"]["occupations"] = "tetrahedra"

        if "scf.1" in result["selected_file"]:
            input = self._add_starting_magnetization(basic_input, with_3d_moments = True)
        else:
            input = self._add_starting_magnetization(basic_input, with_3d_moments = False)
        
        basic_project = qe.getinput_project()
        basic_project["&projwfc"]["DeltaE"] = estep_dos
        basic_project["&projwfc"]["Emin"] = result["efermi"] - 12.0
        basic_project["&projwfc"]["Emax"] = result["efermi"] + 12.0
        basic_project["&projwfc"]["ngauss"] = ""
        basic_project["&projwfc"]["degauss"] = ""
        
        self.fp.write_file("scf.in", self.fp.make_pwscf(input))
        self.fp.write_file("proj.in", self.fp.make_namelist(basic_project))


    def get_dos_result(self, dos_result_file = None):
        if not dos_result_file:
            dos_result_file = os.path.join(self.directory, self.dos_result)

        if os.path.exists(dos_result_file):
            return

        dos_result = {}
        file = os.path.join(self.directory, "scf.out")
        output = NscfOut(file)
        c = output.lattice
        p = output.positions
        t = output.types
        efermi = output.efermi
        struct = StructLite(cell = c, positions = p, types = t, keepstructure = True)
        # keepstructure should prevent standardize the structure

        dos_result["cell"] = c.tolist()
        dos_result["positions"] = p.tolist()
        dos_result["types"] = t
        dos_result["efermi"] = efermi

        uni_index = []
        unique = set(struct.equivalent_atoms)
        for u in unique:
            assert t[u] == struct.std_types[u]
            if chemical_symbols[struct.std_types[u]] in TM_3D:
                uni_index.append(int(u))

        dos_result["unique_tm_index"] = uni_index

        dos_data = parse_pdos_folder(self.directory)

        dos_result["energy"] = dos_data["energy"]
        dos_result["energy_unit"] = dos_data["energy_unit"]
        dos_result["pdos.tot"] = dos_data["pdos.tot"]

        for ui in uni_index:
            dos_result[ui] = dos_data[ui+1]

        write_json(dos_result, dos_result_file)
        return dos_result


    def _add_starting_magnetization(self, input, with_3d_moments = True):
        input2 = deepcopy(input)
        atoms_info = input2["ATOMIC_SPECIES"]["lines"]
        initial_magnetization = {}
        for i,ai in enumerate(atoms_info):
            initial_magnetization[ai.split()[0]] = {"index": i + 1,
                                                        "moment": 0.0 }
            
        if with_3d_moments:
            for e in initial_magnetization:
                if e in TM_3D:
                    initial_magnetization[e]["moment"] = 0.5

        for atomi in initial_magnetization:
            info = initial_magnetization[atomi]
            tag = "starting_magnetization({:d})".format(info["index"])
            input2["&system"][tag] = info["moment"]
        
        return input2


    def _state_similar(self, magnetization, energy, mag_eps = 1e-2):
        """
        compare if magnetization and ground state energy are the same
        """
        energy_eps = len(self.types) * self.EPS
        return (abs(energy[0] - energy[1]) < energy_eps) and (abs(magnetization[0] - magnetization[1]) < mag_eps)


    def _select_ground_state(self, filenamelist = []):
        if not filenamelist:
            filenamelist = self.scf_file

        selected = None
        current_energy = 1e100
        current_magnetization = 0
        current_iteration = 100
        fermi = 0

        for file_in in filenamelist:
            file = os.path.join(self.directory, change_filetype(file_in, "out"))
            result = SCFout(file)
            n_iteration = len(result.iterations)
            if n_iteration > 99:
                continue
            final_energy = result.iterations[-1]["energy"]
            final_moment = result.iterations[-1]["tot_mag"]
            if self._state_similar([final_moment, current_magnetization], [current_energy, final_energy]):
                if n_iteration < current_iteration:
                    current_iteration = n_iteration
                    current_energy = final_energy
                    current_magnetization = final_moment
                    selected = file_in
                    fermi = result.efermi
                    
            elif final_energy < current_energy :
                current_iteration = n_iteration
                current_energy = final_energy
                current_magnetization = final_moment
                selected = file_in
                fermi = result.efermi
        
        result = {}
        result["selected_file"] = selected
        result["energy"] = current_energy
        result["n_iteration"] = current_iteration
        result["total_magnetization"] = current_magnetization
        result["efermi"] = fermi

        if selected:
            self.status = 1
        else:
            self.status = -1

        return result
    

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("prefix", type = str)
    parser.add_argument("file", type = str)
    parser.add_argument("mode", type = str, help= "scf or dos or pp")
    parser.add_argument("-directory", type = str, default=".", help= "directory of calculation")

    args=parser.parse_args()
    calc = SingleCalculation.fromFile(args.prefix, args.directory, args.file)
    if args.mode == "scf":
        calc.initialize_scf()
    elif args.mode == "dos":
        calc.make_dos_inputs()
    elif args.mode == "pp":
        calc.get_dos_result()
    else:
        raise Exception("mode is not supported")
        
    


    
    