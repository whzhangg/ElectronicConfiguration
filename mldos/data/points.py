# this script is intended to put all data as points with density of states associated with it

from mldos.DFT.job_control import JobBase
from torch_geometric.data import Data
import torch
import os
import numpy as np
from copy import deepcopy
from ase.data import chemical_symbols

from mldos.tools import change_filetype, read_generic_to_cpt, write_json
from mldos.parameters import TM_3D, estep_dos
from mldos.DFT.crystal import StructLite
from mldos.DFT._qe_parser import FileParser, QeInput
from mldos.DFT._output_parser import parse_pdos_folder, NscfOut, SCFout

from bisect import bisect

torch_type = torch.float32
dos_range = 10  # -10 - 10

dos_mapper = {
    "s": {
        "pdosup_1" : "s_up",
        "pdosdw_1" : "s_dw"
    },
    "p": {
        "pdosup_1" : "pz_up",
        "pdosdw_1" : "pz_dw",
        "pdosup_2" : "px_up",
        "pdosdw_2" : "px_dw",
        "pdosup_3" : "py_up",
        "pdosdw_3" : "py_dw"
    },
    "d": {
        "pdosup_1" : "dz2_up",
        "pdosdw_1" : "dz2_dw",
        "pdosup_2" : "dzx_up",
        "pdosdw_2" : "dzx_dw",
        "pdosup_3" : "dzy_up",
        "pdosdw_3" : "dzy_dw",
        "pdosup_4" : "dx2-y2_up",
        "pdosdw_4" : "dx2-y2_dw",
        "pdosup_5" : "dxy_up",
        "pdosdw_5" : "dxy_dw"
    }
}

def ranged_dos(input_dos:list, efermi:float, erange:float):
    """we take the list of list of DOS as inputs"""
    energy = input_dos[0]
    new_energy = torch.arange(0.0 - erange, 0.0 + erange ,estep_dos, dtype=torch_type)  # centered at zero

    e0_index = bisect(energy, efermi)
    e0_new = bisect(new_energy.tolist(), 0.0)
    start_old = 0 if e0_new > e0_index else e0_index - e0_new
    start_new = e0_new - e0_index if e0_new > e0_index else 0
    length = min(len(energy) - start_old, len(new_energy)-start_new)

    new = torch.zeros(size = (len(input_dos), len(new_energy)), dtype = torch_type)
    new[0] = new_energy
    torch_old = torch.tensor(input_dos, dtype = torch_type)
    new[1:,start_new:start_new+length] = torch_old[1:,start_old:start_old+length]
    return new


class SingleResult:
    scf_folder = "scf"
    dos_folder = "dos_data"
    def __init__(self, directory):
        self.root = directory
        self.dos_result = {}

        file = os.path.join(self.root, self.scf_folder, "scf.out")
        output = NscfOut(file)
        self.c = output.lattice
        self.p = output.positions
        self.t = output.types
        self.efermi = output.efermi

        struct = StructLite(cell = self.c, positions = self.p, types = self.t, keepstructure = True)
        # keepstructure should prevent standardize the structure


        dos_data = parse_pdos_folder(os.path.join(self.root, self.dos_folder))

        self.dos_energy = dos_data["energy"]
        self.dos_energy_unit = dos_data["energy_unit"]
        self.dos_total = dos_data["pdos.tot"]
        self.pdos = {}

        for i in range(len(self.t)):
            self.pdos[i] = {}
            for key, value in dos_data[i+1].items():
                orbit = key[-1]
                if orbit not in dos_mapper.keys():
                    continue
                m_mapper = dos_mapper[orbit]
                self.pdos[i][key] = {}
                for pdos, data in value.items():
                    if pdos not in m_mapper.keys():
                        continue
                    self.pdos[i][key][m_mapper[pdos]] = data
    @property
    def data(self):
        x = torch.tensor(self.t, dtype = int)
        cell = torch.tensor(self.c, dtype = torch_type)
        pos = torch.tensor(self.p, dtype = torch_type)
        result_list = [self.dos_energy]
        info = {"energy_unit": self.dos_energy_unit}

        for iatom, iatom_data in self.pdos.items():
            for spin in ["up", "dw"]:
                for l, l_data in iatom_data.items():
                    info.setdefault(iatom, {}).setdefault(spin, {}).setdefault(l)
                    start = len(result_list)
                    end = start
                    states = []
                    for m, data in l_data.items():
                        if spin in m:
                            states.append(m[:-3])
                            end += 1
                            result_list.append(data)
                    
                    info.setdefault(iatom, {}).setdefault(spin, {})[l] = {
                        "states": states, 
                        "slice": slice(start, end, 1)
                    }

        dos = ranged_dos(result_list, self.efermi, dos_range)
        dos_old = torch.tensor(result_list, dtype=torch_type)
        print(info)

        return Data(x=x, pos = pos, y = dos, cell = cell, info = info)



if __name__ == "__main__":
    r = SingleResult("/home/wenhao/work/ml_dos/dos_calculation/calc/done/Ag1Zn1P1S4")
    data = r.data
    torch.save(data, "1.pt")