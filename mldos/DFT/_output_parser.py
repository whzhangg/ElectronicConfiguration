# wenhao 2021

import os
import fnmatch
import re
import numpy as np
from ase.data import atomic_numbers
from copy import deepcopy

from scipy.constants import physical_constants
Bohr2A = physical_constants["atomic unit of length"][0] * 1e10


def find_gap(eig: dict, occupancy: dict)->dict:
    # it first identify the nature of the band: 
    # for each spin, it determines whether: metal, semimetal, insulator
    assert eig.keys() == occupancy.keys()

    result = {}
    for spin, energies in eig.items():
        occ = occupancy[spin]
        nbnd = len(energies)
        # find the highest occupied, lowest unoccupied band

        highest_occupied = max( [ ibnd for ibnd in range(nbnd) if (occ[ibnd] > 0.99).any()] )
        lowest_unoccupied = min( [ ibnd for ibnd in range(nbnd) if (occ[ibnd] < 0.01).any()] )

        if highest_occupied >= lowest_unoccupied:
            gap_type = "metal"
            vbm = None
            cbm = None
        else:
            vbm = max(energies[highest_occupied])
            cbm = min(energies[lowest_unoccupied])
            
            if cbm - vbm >= 0.01:
                gap_type = "insulator"
            else:
                gap_type = "semimetal"

        result[spin] = {"type": gap_type, 
                        "vbm" : vbm,
                        "cbm" : cbm}
    
    return result


def qetime2sec(time) -> float:
    "QE time is printed as 6h56m, 12.01s, 10m20.10s"
    parts = re.findall(r"\d+\.?\d*[hms]", time)
    second = 0
    ratio = {"h": 3600, "m": 60, "s" : 1}
    for p in parts:
        number = p[0:-1]
        mark = p[-1]
        second += float(number) * ratio[mark]
    return second


def parse_pdos_folder(folder:str)->dict:
    '''
    result is a dictionary that:
    {
        "energy_unit" : "eV",
        "energy" :
        "pdos.tot" " {
            "dosup(E)" : [...]
            "dosdw(E)" : [...]
            "pdosup(E)" : [...]
            "pdosdw(E)" : [...]
        }
        "atom_index1" : {
            "element" : symbol,
            "s" : { all collomn of the file }
            "p" 
            ...
        }
        "atom_index2" : {
            "element" : symbol,
            "s" : { all collomn of the file }
            "p" 
            ...
        }
    }
    '''

    result = {}

    out_filenames = os.listdir(folder)
    pdos_tot_filenames = fnmatch.filter(out_filenames, '*pdos_tot*')
    for pdos_filename in pdos_tot_filenames:
        with open(os.path.join(folder,pdos_filename), 'r') as pdostot_file:
            # Columns: Energy(eV), Ldos, Pdos
            pdostot_array = np.atleast_2d(np.genfromtxt(pdostot_file))
            nrow, ncol = pdostot_array.shape
            pdostot_file.seek(0)
            firstline = pdostot_file.readline().rstrip().split()
            # E (eV) is splited
            if len(firstline) != ncol + 2:
                raise
            energy = pdostot_array[:,0]

            result["energy_unit"] = "eV"
            result["energy"] =  energy.tolist()

            tot = {}
            for i in range(ncol-1):
                tot[firstline[i+3]] = pdostot_array[:, i+1].tolist()

            result["pdos.tot"] = tot
    
    pdos_atm_filenames = fnmatch.filter(out_filenames, '*pdos_atm*')
    for pdos_filename in pdos_atm_filenames:
        # process the file name
        numbers = [ int(re.sub(r'#', '', dd)) for dd in re.findall(r'#\d+', pdos_filename) ]
        iatom = numbers[0]
        iorbital = numbers[1]
        symbol, orbital = re.findall(r'\([A-Za-z]+\)', pdos_filename)
        symbol = re.sub(r'[\(\)]','', symbol)
        orbital = str(iorbital) + re.sub(r'[\(\)]','', orbital)

        with open(os.path.join(folder,pdos_filename), 'r') as pdostot_file:
            pdostot_array = np.atleast_2d(np.genfromtxt(pdostot_file))
            nrow, ncol = pdostot_array.shape
            pdostot_file.seek(0)
            firstline = pdostot_file.readline().rstrip().split()
            # E (eV) is splited
            if len(firstline) != ncol + 2:
                raise

            tot = {}
            counter = 0
            for i in range(ncol-1):
                title = firstline[i+3][:-3]
                if "ldos" in title:
                    tot[firstline[i+3][:-3]] = pdostot_array[:, i+1].tolist()
                else:
                    tot[firstline[i+3][:-3]+"_"+str(counter//2+1)] = pdostot_array[:, i+1].tolist()
                    counter += 1
            
            if iatom not in result:
                result[iatom] = {}
                result[iatom]["element"] = symbol
            
            result[iatom][orbital] = tot
    
    return result


class PWSCFout_base:

    def __init__(self, filename: str):
        self.output = filename
        self.nele = None
        self.alat = None
        self.nbnd = None

        self.spin_polarized = False

        # crystal structure
        self.positions = None
        self.symbols = None
        self.types = None
        self.lattice = None
        self.reciprocal = None
        self.kpoints = None
        self.kweight = None

        self.low_verbosity = False 
        # low verbosity will not give all information
        self.element_info = {}  # for each element as key, contain valence, mas, pp file
        self.computation_time = 0.0

    def _parse_all(self):
        pass

    def _parse_header(self, fin, rewind = False):
        # it will stop after parsing the kpoints
        alat = 0
        lattice = np.zeros((3,3))
        recip = np.zeros((3,3))
        nbnd = 0
        natom = 0
        ntype = 0
        ecut = 0    # not implemented
        rhocut = 0  # not implemented
        nele = 0
        cart_positions = []
        k_weight = None
        nk = 0
        symbols = []
        k_frac = []

        go_on = True
        pp_file = {}
        atom_info = {}

        while go_on:
            
            aline=fin.readline()
            # read information by checking the flags
            if "lattice parameter (alat)  =" in aline:
                data = aline.split('=')[1]
                data = data.split()
                alat = float(data[0])  # in Bohr

            if "number of Kohn-Sham states" in aline:
                data = aline.split()[-1]
                nbnd = int(data)

            if "number of atoms/cell" in aline:
                data = aline.split()[-1]
                natom = int(data)

            if "number of atomic types" in aline:
                data = aline.split()[-1]
                ntype = int(data)

            if "number of electrons" in aline:
                data = aline.split()[-1]
                nele = float(data)

            if "crystal axes: (cart. coord. in units of alat)" in aline:
                for i in range(3):
                    data = fin.readline().split()[3:6]
                    lattice[i] = np.array(data, dtype = float) 
                lattice *= alat * Bohr2A

            if "reciprocal axes: (cart. coord. in units 2 pi/alat)" in aline:
                for i in range(3):
                    data = fin.readline().split()[3:6]
                    recip[i] = np.array(data, dtype = float)
                recip *= 2 * np.pi / (alat * Bohr2A)

            if "PseudoPot. #" in aline:
                parts = aline.split("for")[1].split()
                ele = parts[0]
                filename = fin.readline().lstrip().rstrip()
                pp_file[ele] = filename

            if "atomic species   valence    mass" in aline:
                for itype in range(ntype):
                    name, valence, mass, *_ = fin.readline().split()
                atom_info[name] = {"valence": float(valence),
                                   "mass": float(mass)}


            if "site n.     atom                  positions (alat units)" in aline:
                for i in range(natom):
                    data = fin.readline()
                    symbols.append(re.findall(r'[A-Z][a-z]*', data)[0])
                    cart_positions.append(np.array(re.findall('-?\d+\.\d+', data), dtype = float) * alat * Bohr2A)
                    # cartesian in unit A

            # reading the k points

            if "Number of k-points >= 100: set verbosity" in aline:
                self.low_verbosity = True
                go_on = False
                # we finish here
            
            if "number of k points= " in aline:
                nk = int( re.findall(r'\d+', aline)[0] )
                
                k_frac = np.zeros((nk,3))
                k_weight = np.zeros(nk)
                aline = fin.readline()
                if "cart. coord." in aline:

                    inv_cell = np.linalg.inv(recip.T)

                    for i in range(nk):
                        aline = fin.readline()
                        parts = aline.split('=')
                        ik = int( re.findall(r'\d+', parts[0])[0] )
                        pos = np.array(re.findall(r'-?\d+\.\d+', parts[1]), dtype = float)
                        k_weight[ik-1] = float(parts[-1])

                        pos *= 2 * np.pi / (alat * Bohr2A)
                        fpos = inv_cell.dot(pos)

                        k_frac[ik-1] = fpos

                    go_on = False

            if go_on == False and rewind:
                fin.seek(0)

        self.alat = alat
        # the part that must present in a nscf.out
        self.lattice = lattice
        self.symbols = symbols 
        self.types = [ atomic_numbers[s] for s in symbols]
        self.nele = nele
        self.nbnd = nbnd

        self.element_info = atom_info
        for k in atom_info:
            atom_info[k]["pseudopotential"] = pp_file[k]

        positions = []
        reverse = np.linalg.inv(self.lattice.T)
        for cp in cart_positions:
            positions.append(reverse.dot(cp))

        self.positions = np.array(positions)
        self.reciprocal = recip

        if len(k_frac) > 0:
            self.kpoints = k_frac
        if len(k_weight) > 0:
            self.kweight = k_weight

    def _parse_kpoint_energy(self, fin, nbnd, dont_read_occupation = False):
        # fin is passed to here, after finding k = xxx line
        # start to read when we find a line with numbers
        # and stop after reading all the occupancy
        aline = fin.readline()
        result = {"energy" : [], "occupation" : []}
        occupation_read = False
        energy_read = False
        while aline:
            if re.search(r'-?\d+\.\d+',aline) != None:

                lenergy = [] # local energy for each k point
                while len(lenergy) < nbnd:
                    data = np.array(re.findall(r'-?\d+\.\d+', aline), dtype = float)
                    for d in data:
                        lenergy.append(d)
                    
                    aline = fin.readline()

                if len(lenergy) > nbnd:
                    print("length of energy: ", len(lenergy))
                    print("nbnd: ", nbnd)
                    raise 

                result["energy"] = np.array(lenergy)
                energy_read = True

            if "occupation numbers" in aline:
                occ = []
                while len(occ) < nbnd:
                    aline = fin.readline()
                    data = np.array(re.findall(r'-?\d+\.\d+', aline), dtype = float)
                    for d in data:
                        occ.append(d)

                if len(occ) > nbnd:
                    raise "length of occupation number > nbnd"
                    
                result["occupation"] = occ
                occupation_read = True

            if energy_read and (dont_read_occupation or occupation_read):
                break
            else:
                aline = fin.readline()

        return result

    def print_coord(self):
        text = ""
        if self.lattice is not None:
            text += "CELL_PARAMETERS (angstrom)\n"
            for i in range(3):
                text += "{:>18.10f}{:>18.10f}{:>18.10f}\n".format(self.lattice[i,0],self.lattice[i,1],self.lattice[i,2])
            text+= "\n"

        text += "ATOMIC_POSITIONS (crystal)\n"
        for i in range(len(self.symbols)):
            text += "{:>2s}{:>18.10f}{:>18.10f}{:>18.10f}\n".format(self.symbols[i], self.positions[i,0], self.positions[i,1],self.positions[i,2])
        
        return text


class SCFout(PWSCFout_base):
    # it is similar to nscf result, but include method to include
    # iteration result

    def __init__(self, filename: str):
        super().__init__(filename)
        self.efermi = None
        self.vbm = None
        self.cbm = None
        self.is_metal = True

        self.eig = None
        self.occupations = None
        self.iterations = None
        self.site_atomic_moment = None
        self.contain_moment = False

        self.final_abs_moment = None
        self.final_tot_moment = None

        self._parse_all()

    def _parse_scf_iteration(self, fin, nat):
        aline = fin.readline()
        iterations = []
        site_magnetic_moment = []
        site_charge = []
        count = 0
        energy = None
        accuracy = None
        tot_magetization = None
        abs_magnetization = None
        contain_magnetic_monent = True
        while "End of self-consistent calculation" not in aline:
            if "iteration #" in aline:
                if count > 0:
                    iterations.append({"energy" : energy,
                                    "accuracy" : accuracy,
                                    "tot_mag" : tot_magetization,
                                    "abs_mag" : abs_magnetization})
                
                count += 1
            
            if "total energy" in aline:
                energy = float(re.findall(r'-?\d+\.\d+', aline)[0])
            if "estimated scf accuracy" in aline:
                accuracy = float(re.findall(r'-?\d+\.\d+E?-?\d+', aline)[0])
            if "total magnetization" in aline:
                tot_magetization = float(re.findall(r'-?\d+\.\d+', aline)[0])
            if "absolute magnetization" in aline:
                abs_magnetization = float(re.findall(r'-?\d+\.\d+', aline)[0])

            # QE 6.4
            if "Magnetic moment per site:" in aline:
                for i in range(nat):
                    parts = fin.readline().split()
                    site_magnetic_moment.append(float(parts[5]))
                    site_charge.append(float(parts[3]))

            # QE 6.8
            if "Magnetic moment per site  (integrated on atomic sphere of radius R)" in aline:
                for i in range(nat):
                    parts = fin.readline().split()
                    site_magnetic_moment.append(float(parts[-1]))
                    site_charge.append(float(parts[-3]))

            aline = fin.readline()

        if " of self-consistent calculation" in aline:
            iterations.append({"energy" : energy,
                            "accuracy" : accuracy,
                            "tot_mag" : tot_magetization,
                            "abs_mag" : abs_magnetization})

        if tot_magetization == None:
            contain_magnetic_monent = False
        return contain_magnetic_monent, iterations, site_magnetic_moment, site_charge

    def _parse_all(self):
        energy = {"spinup" : [],
                  "spindown" : []
                  }

        occupantion = {"spinup" : [],
                       "spindown" : []}

        which = "spinup"  # remember if we are reading spin up or spin down

        with open(self.output, 'r') as f:
            self._parse_header(f, rewind = False)

            aline = f.readline()
            while aline:
                if "Self-consistent Calculation" in aline:
                    # parse until "End of self-consistent calculation"
                    contain_magnetic_monent, iterations, final_site_magnetic_moment, final_site_charge = self._parse_scf_iteration(f, len(self.types))

                if "------ SPIN UP ------------" in aline:
                    which = "spinup"
                    
                if "------ SPIN DOWN ----------" in aline:
                    which = "spindown"

                if re.search('k\s+=\s*-?\d+\.\d+\s*-?\d+\.\d+\s*-?\d+\.\d+\s',aline) != None:
                    tmp_result = self._parse_kpoint_energy(f, self.nbnd)
                    energy[which].append(tmp_result["energy"])
                    occupantion[which].append(tmp_result["occupation"])

                if "the Fermi energy is" in aline:
                    self.efermi = float(re.findall(r'-?\d+\.\d+', aline)[0])

                if "highest occupied, lowest unoccupied level" in aline:
                    self.is_metal = False
                    vbmcbm = np.array(aline.split(":")[1].split(), dtype = float)
                    self.vbm = vbmcbm[0]
                    self.cbm = vbmcbm[1]

                if "total magnetization" in aline:
                    self.final_tot_moment = float(re.findall(r'-?\d+\.\d+', aline)[0])

                if "absolute magnetization" in aline:
                    self.final_abs_moment = float(re.findall(r'-?\d+\.\d+', aline)[0])

                if "PWSCF" in aline and "WALL" in aline:
                    parts = aline.split("CPU")
                    self.computation_time = qetime2sec(parts[0])

                aline = f.readline()

        self.contain_moment = contain_magnetic_monent
        self.iterations = iterations
        self.site_atomic_moment = final_site_magnetic_moment
        self.site_charge = final_site_charge

        self.eig = {}
        if energy["spinup"]:
            self.eig["up"] = np.array(energy["spinup"]).T
        else:
            self.eig["up"] = []

        if energy["spindown"]:
            self.spin_polarized = True
            self.eig["down"] = np.array(energy["spindown"]).T
        else:
            self.eig["down"] = []

        self.occupations = {}
        if occupantion["spinup"]:
            self.occupations["up"] = np.array(occupantion["spinup"]).T
        else:
            self.occupations["up"] = []

        if occupantion["spindown"]:
            self.occupations["down"] = np.array(occupantion["spindown"]).T
        else:
            self.occupations["down"] = []


class NscfOut(PWSCFout_base):

    def __init__(self, filename: str):
        super().__init__(filename)
        self.efermi = None
        self.vbm = None
        self.cbm = None
        self.is_metal = True

        self.eig = None
        self.occupations = None
        self._parse_all()

    def _parse_all(self):
        energy = {"spinup" : [],
                  "spindown" : []
                  }

        occupantion = {"spinup" : [],
                       "spindown" : []}

        which = "spinup"  # remember if we are reading spin up or spin down

        with open(self.output, 'r') as f:
            self._parse_header(f, rewind = False)

            aline = f.readline()
            while aline:
                if "------ SPIN UP ------------" in aline:
                    which = "spinup"
                    
                if "------ SPIN DOWN ----------" in aline:
                    which = "spindown"

                if re.search('k\s+=\s*-?\d+\.\d+\s*-?\d+\.\d+\s*-?\d+\.\d+\s',aline) != None:
                    tmp_result = self._parse_kpoint_energy(f, self.nbnd)
                    energy[which].append(tmp_result["energy"])
                    occupantion[which].append(tmp_result["occupation"])

                if "the Fermi energy is" in aline:
                    self.efermi = float(re.findall(r'-?\d+\.\d+', aline)[0])

                if "highest occupied, lowest unoccupied level" in aline:
                    self.is_metal = False
                    vbmcbm = np.array(aline.split(":")[1].split(), dtype = float)
                    self.vbm = vbmcbm[0]
                    self.cbm = vbmcbm[1]

                if "PWSCF" in aline and "WALL" in aline:
                    parts = aline.split("CPU")
                    self.computation_time = qetime2sec(parts[0])

                aline = f.readline()

        self.eig = {}
        if energy["spinup"]:
            self.eig["up"] = np.array(energy["spinup"]).T
        else:
            self.eig["up"] = []

        if energy["spindown"]:
            self.spin_polarized = True
            self.eig["down"] = np.array(energy["spindown"]).T
        else:
            self.eig["down"] = []

        self.occupations = {}
        if occupantion["spinup"]:
            self.occupations["up"] = np.array(occupantion["spinup"]).T
        else:
            self.occupations["up"] = []

        if occupantion["spindown"]:
            self.occupations["down"] = np.array(occupantion["spindown"]).T
        else:
            self.occupations["down"] = []


class RelaxOut_steps(PWSCFout_base):

    def __init__(self, filename: str):
        super().__init__(filename)

        self.nsteps = 0
        self.final_lattice = None
        self.final_positions = None
        self.steps = []
        self.final_scf_run = None

        self._parse_all()

    def _parse_all(self):

        with open(self.output, 'r') as f:
            self._parse_header(f, rewind = False)

            aline = f.readline()
            while aline:
                if "End of self-consistent calculation" in aline:
                    step = self._parse_step(f)
                    self.steps.append(step)

                if "PWSCF" in aline and "WALL" in aline:
                    parts = aline.split("CPU")
                    self.computation_time = qetime2sec(parts[0])

                aline = f.readline()

        if len(self.steps) > 1:
            if len(self.steps[-1]["lattice"]) == 0 and len(self.steps[-1]["positions"]) == 0 \
                and (len(self.steps[-2]["lattice"]) > 0 or len(self.steps[-2]["positions"]) > 0):
                final_scf = self.steps.pop(-1)

        if len(self.steps) > 0 and len(self.steps[-1]["lattice"]) > 0:
            self.final_lattice = self.steps[-1]["lattice"]
        else:
            self.final_lattice = self.lattice

        if len(self.steps) > 0 and len(self.steps[-1]["positions"]) > 0:
            self.final_positions = self.steps[-1]["positions"]
        else:
            self.final_positions = self.positions

        self.nsteps = len(self.steps)

    def _parse_step(self, f):
        # it will be entered after finding the tag
        # "end of self-consistent calculation"
        empty = {"up" : [],
                 "down" : [] }
        energy = deepcopy(empty)
        occupantion = deepcopy(empty)

        which = "up"  # remember if we are reading spin up or spin down

        result = {}
        go_on = True

        total_energy = 0    # Ry
        force = None        # Ry/bohr
        stress = None

        nat = len(self.symbols)

        lattice = []
        pos = []

        while go_on:
            aline = f.readline()
            if not aline: 
                go_on = False

            if "------ SPIN UP ------------" in aline:
                which = "up"
                
            if "------ SPIN DOWN ----------" in aline:
                self.spin_polarized = True
                which = "down"

            if re.search('k\s+=\s*-?\d+\.\d+\s*-?\d+\.\d+\s*-?\d+\.\d+\s',aline) != None:
                tmp_result = self._parse_kpoint_energy(f, self.nbnd)
                energy[which].append(tmp_result["energy"])
                occupantion[which].append(tmp_result["occupation"])

            if "total energy              =" in aline:
                data = aline.split("=")[1].split()
                total_energy = float(data[0]) # in Ry
            
            if "Forces acting on atoms (cartesian axes, Ry/au):" in aline:
                aline = f.readline()
                force = np.zeros((nat,3))
                for i in range(nat):
                    data = f.readline().split("=")[1]
                    force[i] = np.array(data.split(), dtype = float) # Ry

            if "Computing stress (Cartesian axis) and pressure" in aline:
                aline = f.readline()
                aline = f.readline()
                stress = np.zeros((3,3))
                for i in range(3):
                    datas = f.readline().split()[0:3]
                    stress = np.array(datas, dtype = float)

            
            if "CELL_PARAMETERS" in aline:
                lattice = np.zeros((3,3))
                for i in range(3):
                    lattice[i] = np.array(f.readline().split(), dtype = float)
                    
            if "ATOMIC_POSITIONS" in aline:
                pos = np.zeros((nat, 3))
                for i in range(nat):
                    parts = f.readline().split()
                    pos[i] = np.array(parts[1:], dtype = float)
                
                go_on = False

        result["lattice"] = lattice
        result["positions"] = pos
        result["stress"] = stress # Ry/Bohr**3
        result["forces"] = force
        result["total_energy"] = total_energy

        result["eig"] = {}
        if energy["up"]:
            result["eig"]["up"] = np.array(energy["up"]).T
        else:
            result["eig"]["up"] = []
        if energy["down"]:
            result["eig"]["down"] = np.array(energy["down"]).T
        else:
            result["eig"]["down"] = []

        result["occupations"] = {}
        if occupantion["up"]:
            result["occupations"]["up"] = np.array(occupantion["up"]).T
        else:
            result["occupations"]["up"] = []

        if occupantion["down"]:
            result["occupations"]["down"] = np.array(occupantion["down"]).T
        else:
            result["occupations"]["down"] = []

        return result

    @property
    def coord(self):
        text = ""
        if self.final_lattice is not None:
            l = self.final_lattice
            text += "CELL_PARAMETERS (angstrom)\n"
            for i in range(3):
                text += "{:>18.10f}{:>18.10f}{:>18.10f}\n".format(l[i,0],l[i,1],l[i,2])
            text+= "\n"

        p = self.final_positions
        text += "ATOMIC_POSITIONS (crystal)\n"
        for i in range(len(self.symbols)):
            text += "{:>2s}{:>18.10f}{:>18.10f}{:>18.10f}\n".format(self.symbols[i], p[i,0], p[i,1], p[i,2])
        
        return text

