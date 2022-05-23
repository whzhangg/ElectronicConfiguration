import os
import numpy as np
from ase import Atoms
from seekpath import get_path
from ase.data import atomic_masses,atomic_numbers

from mldos.tools import read_generic_to_cpt, make_target_directory
from ._element_info import pp_info
from ._config import pseudopotential_directory


def find_RCL(a1,a2,a3):
    '''
    parameters: a1,a2,a3 the vector of lattice vector in unit of angstrom 
    return:     b1,b2,b3 the vector of reciprocal vector in 1/A
    '''
    if not isinstance(a1,np.ndarray):
        a1=np.array(a1)
    if not isinstance(a2,np.ndarray):
        a2=np.array(a2)
    if not isinstance(a3,np.ndarray):
        a3=np.array(a3)
    volumn=np.cross(a1,a2).dot(a3)
    b1=(2*np.pi/volumn)*np.cross(a2,a3)
    b2=(2*np.pi/volumn)*np.cross(a3,a1)
    b3=(2*np.pi/volumn)*np.cross(a1,a2)
    return b1,b2,b3

def cell_vector_2_abc(vector):
    """
    Parameter:
        vector: the cell vector in a list form as in the ase.atoms
        (which is just an np.array [[a1,a2,a3],[b1,b2,..],[..]] 
    Return:
        a list
        [a, b, c, alpha, beta, gamme]
    Purpose:
        as the name implies
    """
    unitcell=np.array(vector)
    a=np.sqrt(np.dot(unitcell[0],unitcell[0]))
    b=np.sqrt(np.dot(unitcell[1],unitcell[1]))
    c=np.sqrt(np.dot(unitcell[2],unitcell[2]))
    alpha=(180/np.pi)*np.arccos(np.dot(unitcell[1],unitcell[2])/(b*c))
    beta=(180/np.pi)*np.arccos(np.dot(unitcell[0],unitcell[2])/(a*c))
    gamma=(180/np.pi)*np.arccos(np.dot(unitcell[0],unitcell[1])/(a*b))
    return [a,b,c,alpha,beta,gamma]

def create_MP_01(a1,a2,a3):
    """
    create a grid in the range of [0,1]
    this is necessary for EPW
    """
    mp=[]
    for ii in np.arange(0,1.0,1.0/a1):
        for jj in np.arange(0,1.0,1.0/a2):
            for kk in np.arange(0,1.0,1.0/a3):
                mp.append([ii,jj,kk])
    return mp

def create_MP_unshifted(a1,a2,a3):
    # include gamma 
    mp = create_MP_01(a1,a2,a3)
    mp = np.array(mp)
    sx=(a1//2)/(a1*1.0)
    sy=(a2//2)/(a2*1.0)
    sz=(a3//2)/(a3*1.0)
    all_shift=np.tile(np.array([sx,sy,sz]),(a1*a2*a3,1))
    return mp-all_shift

def create_MP_shifted(a1,a2,a3):
    mp_unshifted = create_MP_unshifted(a1,a2,a3)
    all_shift = np.tile(np.array([0.5/a1,0.5/a2,0.5/a3]),(a1*a2*a3,1))
    return mp_unshifted+all_shift

def get_number_of_kpoints_along_reciprocal_path(k1, k2, recipical_lattice, perpoint=0.015) -> int:
    k1=np.array(k1)
    k2=np.array(k2)
    recipical_lattice=np.array(recipical_lattice)
    k1=k1.dot(recipical_lattice)
    k2=k2.dot(recipical_lattice)
    dif=k2-k1
    # if k1 and k2 are too close, it's possible that it return 0, to prevent that 
    # we return a minimum number 2, which separate from 1
    # but ideally, we should skip this point completely
    return max(int(np.sqrt(dif.dot(dif))/perpoint),2)

def parse_namelist(namelist: dict) -> str:
    text = ""
    for k in namelist:
        if type(namelist[k]) == str and namelist[k] == "":
            text += "  ! " + k + " = \n"
        else:
            text += "  " + k + " = "
            if type(namelist[k]) == str and "tag" in namelist[k]:
                text += namelist[k] + '\n'
            elif type(namelist[k]) == str:
                text += '\"{:s}\"\n'.format(namelist[k])
            #elif type(namelist[k]) == float:
            #    text += "{:.8f}\n".format(namelist[k])
            elif type(namelist[k]) == bool:
                if namelist[k]:
                    text += ".TRUE.\n"
                else:
                    text += ".FALSE.\n"
            else:
                text += str(namelist[k]) + "\n"
    return text

def get_qe_kpoints_data(pointslist, positive=False, withweight=True, shifted=False, withindex = False):
    """
    parameters:
        pointslist: [a1,a2,a3]
        positive: if true, the range of k will only be in [0,1]
        withweight:  if with weight => QE style
                     withweight=Flase => for wannier input 
    return:
        None
    """
    output = []

    a1=pointslist[0]
    a2=pointslist[1]
    a3=pointslist[2]
    if positive==False:
        # two way to generate mp grid if does not require positive
        if shifted:
            mp = create_MP_shifted(a1,a2,a3)
        else:
            mp = create_MP_unshifted(a1,a2,a3)
    else:
        # generate positive mp grid
        mp=create_MP_01(a1,a2,a3)

    weight=1/len(mp)
    
    for i,line in enumerate(mp):
        aline = "{:>15.10f}{:>15.10f}{:>15.10f}".format(line[0],line[1],line[2])
        if withweight:
            aline+="{:>13.8f}".format(weight)
        if withindex:
            aline = "{:>10d}".format(i+1) + aline
        output.append(aline)

    return output
    #if withweight:
    #    for line in mp:
    #        print("{:>15.10f}{:>15.10f}{:>15.10f}{:>13.8f}".format(line[0],line[1],line[2],weight))
    #else:
    #    for line in mp:
    #        print("{:>15.10f}{:>15.10f}{:>15.10f}".format(line[0],line[1],line[2]))


class FileParser:
    """
    this class is designed to handle the result given by the input generator
    """
    import os
    def __init__(self, prefix: str, directory:str = "./"):
        self.prefix = prefix
        self.directory = directory
        make_target_directory(self.directory)

    def make_pwscf(self, inputs:dict) -> str:
        # this use dictionary to parse
        text = ""
        for k in inputs:
            # namelist
            if k[0] == '&':
                text += k + "\n"
                text += parse_namelist(inputs[k])
                text += "/\n"
            else:
                text += k + " " + inputs[k]["option"] + "\n"
                for l in inputs[k]["lines"]:
                    text += l + '\n'

            text += "\n"
        return text
    
    def make_namelist(self, inputs: dict) -> str:
        text = ""
        for k in inputs:
            text += k + "\n"
            text += parse_namelist(inputs[k])
            text += "/\n"
        text += "\n"
        return text

    def write_file(self, filename: str, string: str) -> None:
        with open(os.path.join(self.directory, filename), 'w') as f:
            f.write(string)

class InputStructure:

    KPRA_DENSE = 40000
    KPRA = 5000
    QPRA = 1000
    # k_grid_multiplier can be used to increase kpoint density (applied to all three): KPRA 
    # positions2 and lattice2 allow to modify the structure without modifying the bandlines,
    # however, it is possible that std_lattice and std_position from seekpath do not 
    # agree (transformed)  with lattice2 and should be used with care
    def __init__(self, prefix = None, lattice = None, positions = None, types = None, 
                 filename = None, pp_type = "PBEsol", smear = "mp", nspin: int = 1, 
                 k_grid_multiplier: float = 1.0, use_conventional = False,
                 lattice2 = None, positions2 = None):

        self.prefix = prefix or "material"
        self.pp_dir =pseudopotential_directory[pp_type]
        self.pplib = pp_info
        self.nspin = nspin
        self.use_conventional = use_conventional

        if smear not in ["mp", 'fixed']:
            print("smear is not supported")
            raise
        self.smear = smear

        self.outdir = "./out"
    
        if type(lattice) != type(None) and type(positions) != type(None) and type(types) != type(None):
            kpaths_info = get_path((lattice, positions , types), symprec = 1e-3)
        elif type(filename) != type(None):
            lattice, positions, types = read_generic_to_cpt(filename)
            kpaths_info = get_path((lattice, positions , types), symprec = 1e-3)
        else:
            raise "input structure not specified"

        std_lattice=kpaths_info['primitive_lattice']
        std_position=kpaths_info['primitive_positions']
        std_number=kpaths_info['primitive_types']
        self.struct = Atoms(cell = std_lattice, scaled_positions = std_position)
        self.struct.set_atomic_numbers(std_number)

        conv_lattice=kpaths_info['conv_lattice']
        conv_position=kpaths_info['conv_positions']
        conv_number=kpaths_info['conv_types']
        self.conventional = Atoms(cell = conv_lattice, scaled_positions = conv_position)
        self.conventional.set_atomic_numbers(conv_number)

        self.path_coordinate = kpaths_info['point_coords']
        self.suggested_path = kpaths_info['path']

        if type(lattice2) != type(None):
            self.lattice2 = np.array(lattice2)
        else:
            self.lattice2 = None

        if type(positions2) != type(None):
            self.positions2 = np.array(positions2)
        else:
            self.positions2 = None

        self.elementset = set(self.struct.get_chemical_symbols())
        
        self.k_fine = self._generate_mp(self.KPRA * (k_grid_multiplier)**3)
        self.k_coarse = self._generate_mp(self.QPRA * (k_grid_multiplier)**3)
        self.k_dense = self._generate_mp(self.KPRA_DENSE * (k_grid_multiplier)**3)

        self.cutoff = self._get_cutoff(self.struct.get_chemical_symbols())
        self.conv_thr = 1e-13 # per atom

    def _generate_mp(self, knumber):
        """
        return three integer mp grid for the given kpoint density per reciprocal atoms
        """
        if self.use_conventional:
            cell = self.conventional.get_cell()
            nat = len(self.conventional.get_chemical_symbols())
        else:
            cell = self.struct.get_cell()
            nat = len(self.struct.get_chemical_symbols())

        reciprocal = find_RCL(cell[0], cell[1], cell[2])
        abc = cell_vector_2_abc(reciprocal)
        totkp = knumber / nat
        n = (totkp / (abc[0] * abc[1] * abc[2]))**(1/3)
        nk1 = int(abc[0]*n)
        nk2 = int(abc[1]*n)
        nk3 = int(abc[2]*n)

        return np.array([nk1, nk2, nk3])
        
    def _get_cutoff(self, symbols) -> float:
        cutoff = 0.0
        for elementname in set(symbols):
            try:
                co = self.pplib[elementname]["cutoff_hartree"]
                if co > cutoff:
                    cutoff = co
            except:
                pass

        return cutoff*2

    def bandpath_wannier(self, with_number = False) -> list:
        '''
        * Note, this method generate lines only for primitive cell
        * no \n is included in each line
        return following as lists of lines
            G 0.000 0.000 0.000 X 0.500 0.000 0.500 
            X 0.500 0.000 0.500 U 0.625 0.250 0.625
            ...
        if with_number, add now many points in each path
        '''
        
        cell = self.struct.get_cell()
        reciprocal = find_RCL(cell[0], cell[1], cell[2])
        output = []
        for path_tuple in self.suggested_path:
            s1 = path_tuple[0] # one symbol
            coord1 = self.path_coordinate[s1]
            s2 = path_tuple[1]
            coord2 = self.path_coordinate[s2]
            if s1 == "GAMMA":
                s1 = "G"
            if s2 == "GAMMA":
                s2 = "G"
            aline = "{:>5s}{:>10.6f}{:>10.6f}{:>10.6f}".format(s1,coord1[0],coord1[1],coord1[2])
            aline += "   "
            aline += "{:>5s}{:>10.6f}{:>10.6f}{:>10.6f}".format(s2,coord2[0],coord2[1],coord2[2])
            if with_number:
                tmp_num = get_number_of_kpoints_along_reciprocal_path(coord1, coord2, reciprocal)
                aline += "{:>4d}".format(tmp_num)
            output.append(aline)
        return output
   
    def bandpath_pwscf(self) -> list:
        '''
        * no \n is included in the lines

        return following as lists of lines
        0.000000  0.000000  0.000000   29
        0.500000  0.500000  0.500000   61
        0.812715  0.187285  0.500000    1
        '''
        cell = self.struct.get_cell()
        reciprocal = find_RCL(cell[0], cell[1], cell[2])

        last_path_end=None
        all_path_points=[]
        kpoint_num=[]
        for path_tuple in self.suggested_path:

            if last_path_end==None:
                all_path_points.append(path_tuple[0])
                all_path_points.append(path_tuple[1])
                tmp_num=get_number_of_kpoints_along_reciprocal_path(self.path_coordinate[path_tuple[0]],self.path_coordinate[path_tuple[1]], reciprocal)
                kpoint_num.append(tmp_num)
                last_path_end=path_tuple[1]

            elif path_tuple[0]!=last_path_end:
                kpoint_num.append(1)
                all_path_points.append(path_tuple[0])
                all_path_points.append(path_tuple[1])
                tmp_num=get_number_of_kpoints_along_reciprocal_path(self.path_coordinate[path_tuple[0]],self.path_coordinate[path_tuple[1]], reciprocal)
                kpoint_num.append(tmp_num)
                last_path_end=path_tuple[1]

            else:
                tmp_num=get_number_of_kpoints_along_reciprocal_path(self.path_coordinate[path_tuple[0]],self.path_coordinate[path_tuple[1]], reciprocal)
                kpoint_num.append(tmp_num)
                all_path_points.append(path_tuple[1])
                last_path_end=path_tuple[1]

        kpoint_num.append(1)
        output = []
        all_path_points=np.array(all_path_points)

        for i,_kpoints in enumerate(all_path_points):
            _pos=self.path_coordinate[_kpoints]
            output.append("{:>10.6f}{:>10.6f}{:>10.6f}{:>5d}".format(_pos[0],_pos[1],_pos[2],kpoint_num[i]))
        return output

class QeInput(InputStructure):

    # parse QE inputs based on input structure
    PWJOBS = ["relax", "damp", "scf", "nscf", "nscfdos", "bands"]

    def getinput_pwscf(self, job, nostructure = False) -> dict:

        if job not in self.PWJOBS:
            raise NotImplementedError

        if self.use_conventional and job in ["bands"]:
            raise SystemExit('try to use conventional cell but bands only consider primitive cell')

        elementset = self.elementset
        if self.use_conventional:
            struct = self.conventional
        else:
            struct = self.struct

        inputs = {}
        #inputs["comment"] = "# start_qe_input ===============\n"

        control = {}
        if job == "relax":
            control["calculation"] = "vc-relax"
        if job == "damp":
            control["calculation"] = "relax"
        elif job == "scf":
            control["calculation"] = "scf"
        elif job == "nscf" or job == "nscfdos":
            control["calculation"] = "nscf"
        elif job == "bands":
            control["calculation"] = "bands"

        control["prefix"] = self.prefix
        control["outdir"] = self.outdir
        control["pseudo_dir"] = self.pp_dir
        control["restart_mode"] = "from_scratch"

        if job == "bands" or job == "nscf" or job == "scf" or job == "nscfdos":
            control["verbosity"] = "high"  # print all bands
        else:
            control["verbosity"] = "low"

        if job == "relax" or job == "scf" or job == "damp":
            control["tstress"] = True
            control["tprnfor"] = True
        else:
            control["tstress"] = False
            control["tprnfor"] = False
            
        if job == "relax" or job == "bands" or job == "damp":
            control["disk_io"] = "none"
        elif job == "scf" or job == "nscf" or job == "nscfdos":
            control["disk_io"] = "low"

        if job == "relax"  or job == "damp":
            control["etot_conv_thr"] = 1e-4  # default value
            control["forc_conv_thr"] = 1e-8  # default value = 1e-3

        inputs["&control"] = control  
        # key with & in begining will be reconigized as namelist

        system = {}
        system["ibrav"] = 0
        system["nat"] = len(struct.get_atomic_numbers())
        system["ntyp"] = len(elementset)
        system["ecutwfc"] = self.cutoff
        system["ecutrho"] = 4 * self.cutoff
        
        if self.smear == "mp":
            system["occupations"] = "smearing"
            system["smearing"] = "mp"
            system["degauss"] = 3e-3  # Ry !
        elif self.smear == "fixed":
            system["occupations"] = "fixed"
        
        if job == "nscf" or job == "bands" or job == "nscfdos":
            nbnd = 0
            for ele in struct.get_chemical_symbols():
                ntot = int( self.pplib[ele]["total"] / 2)
                nbnd += ntot
            system["nbnd"] = nbnd

        if job == "nscf" or job == "bands":
            system["nosym"] = True
            system["noinv"] = True  # to prevent nscf generate more points than given

        system["nspin"] = self.nspin
        inputs["&system"] = system

        electrons = {}
        if job == "relax" or job == "damp" or job == "scf":
            # For non-self-consistent calculations, conv_thr is used
            # to set the default value of the threshold (ethr) for iterative diagonalizazion
            # setting it to a low value will give convergence problem "too many band unconverged" for band calculation etc.
            electrons["conv_thr"] = len(struct.get_atomic_numbers()) * self.conv_thr
            electrons["mixing_beta"] = 0.4

        inputs["&electrons"] = electrons
        

        if job == "relax" or job == "damp":
            ions = {}
            if job == "damp":
                ions["ion_dynamics"] = "damp"
            else:
                ions["ion_dynamics"] = "bfgs"
            inputs["&ions"] = ions


        if job == "relax":            
            cell = {}
            cell["press_conv_thr"] = 0.1
            inputs["&cell"] = cell

        # for non namelist inputs, we store all the lines with an options 
        # eg:
        # { "ATOMIS_SPECIES" : {"option"}

        atomic_species = {}
        atomic_species["option"] = ""
        atomic_species["lines"] = []
        for ele in elementset:
            ppname = self.pplib[ele]["file"]
            atomic_species["lines"].append(" {:>2s} {:>8.2f}  ".format(ele,atomic_masses[atomic_numbers[ele]]) + ppname)
        
        inputs["ATOMIC_SPECIES"] = atomic_species

        cell_parameters = {}
        cell_parameters["option"] = "angstrom"
        cell_parameters["lines"] = []
        if type(self.lattice2) != type(None):
            use_cell = self.lattice2
        else:
            use_cell = struct.get_cell()
        for vector in use_cell:
            cell_parameters["lines"].append("{:>13.8f}{:>13.8f}{:>13.8f}".format(vector[0],vector[1],vector[2]))

        atomic_positions = {}
        atomic_positions["option"] = "crystal"
        atomic_positions["lines"] = []
        symbols = struct.get_chemical_symbols()
        
        if type(self.positions2) != type(None):
            pos = self.positions2
        else:
            pos = struct.get_scaled_positions()
        for i, sym in enumerate(symbols):
            atomic_positions["lines"].append("{:>2s}{:>12.8f}{:>12.8f}{:>12.8f}".format(sym, pos[i][0], pos[i][1], pos[i][2]))

        if not nostructure:
            inputs["CELL_PARAMETERS"] = cell_parameters
            inputs["ATOMIC_POSITIONS"] = atomic_positions
        
        k_points = {}
        k_points["lines"] = []
        if job == "relax" or job == "scf" or job == "damp":
            k_points["option"] = "automatic"
            k_points["lines"].append('{:>3d}{:>3d}{:>3d} 0 0 0'.format(self.k_fine[0],self.k_fine[1],self.k_fine[2]))

        elif job == "nscfdos":
            k_points["option"] = "automatic"
            k_points["lines"].append('{:>3d}{:>3d}{:>3d} 0 0 0'.format(self.k_dense[0],self.k_dense[1],self.k_dense[2]))

        elif job == "bands":
            all_bands = self.bandpath_pwscf()
            k_points["option"] = "crystal_b"
            k_points["lines"].append(str(len(all_bands)))
            k_points["lines"] += all_bands

        elif job == "nscf":
            mplines = get_qe_kpoints_data([self.k_coarse[0],self.k_coarse[1],self.k_coarse[2]], positive=True, withweight=True)
            k_points["option"] = "crystal"
            k_points["lines"].append(str(len(mplines)))
            k_points["lines"] += mplines
        
        inputs["K_POINTS"] = k_points
        return inputs

    def getinput_dfpt(self) -> dict:
        """
        this script return a dictionary of inputs 
        for DFPT calculation
        """
        inputs = {}

        inputph = {}
        inputph["prefix"] = self.prefix
        inputph["outdir"] = self.outdir
        inputph["fildyn"] = self.prefix + '.dyn_q'
        inputph["ldisp"] = True
        inputph["fildvscf"] = "dvscf_q"
        inputph["reduce_io"] = True
        inputph["nq1"] = self.k_coarse[0]
        inputph["nq2"] = self.k_coarse[1]
        inputph["nq3"] = self.k_coarse[2]
        inputph["tr2_ph"] = 1.0e-12
        inputph["alpha_mix"] = 0.4

        inputs["&inputph"] = inputph
        return inputs
    
    def getinput_pw2wannier(self) -> dict:

        inputs = {}

        inputpp = {}
        inputpp["outdir"] = self.outdir 
        inputpp["prefix"] = self.prefix 
        inputpp["seedname"] = self.prefix 
        inputpp["spin_component"] = "none"
        inputpp["write_mmn"] = True 
        inputpp["write_amn"] = True
        inputpp["write_unk"] = False
        inputpp["reduce_unk"] = True
        inputpp["wan_mode"] = "standalone"

        inputs["&inputpp"] = inputpp

        return inputs

    def getinput_project(self) -> dict:
        """
        https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html
        """
        projwfc = {}
        projwfc["outdir"] = self.outdir
        projwfc["prefix"] = self.prefix
        projwfc["filpdos"] = self.prefix + ".pdos"
        projwfc["filproj"] = self.prefix + ".proj"
        projwfc["Emin"] = "tag_emin"
        projwfc["Emax"] = "tag_emax"
        projwfc["DeltaE"] = 0.05
        projwfc["ngauss"] = 0
        projwfc["degauss"] = 0.005 # Ry !
        projwfc["kresolveddos"] = False

        inputs = {"&projwfc" : projwfc}
        return inputs

class WannierInput(InputStructure):

    # not finished yet

    def getinput_wannier90(self) -> dict:
        # boltzwann only need .chk and .eig to run
        
        paths = self.bandpath_wannier(with_number = False)

        abc_recip = cell_vector_2_abc(self.reciprocal)
        
        nbnd = 0
        nexclude = 0
        nwann = 0
        for ele in self.struct.get_chemical_symbols():
            info = self.pplib[ele]
            nexclude+= int(info["core"]/2)
            nbnd += int((info["total"] - info["core"])/2)
            nwann += info["nproj"]
        
        exclude = "1-"+str(nexclude)
        # if we didn't find pp, cutoff will just be zero
                
        # boltzwann is prased together
        output  = "!--------\n"
        output += "boltzwann = true\n"
        output += "!boltz_calc_also_dos = true\n"
        output += "!boltz_doc_energy_step = 0.01\n"
        output += "!smr_type = gauss\n"
        output += "!boltz_dos_smr_fixed_en_width = 0.005 \n"
        output += "\n"
        # k mesh is three times of scf kgrid on each side.
        # -> changed to 5 times of scf grid
        k1 = int( self.k_fine[0] // 2 * 10)
        k2 = int( self.k_fine[1] // 2 * 10)
        k3 = int( self.k_fine[2] // 2 * 10)
        output += "kmesh = {:d} {:d} {:d}\n".format(k1,k2,k3)
        # these value should choose according to the VBM and CBM. 
        output += "boltz_mu_min = tag1\n"
        output += "boltz_mu_max = tag2\n"
        output += "boltz_mu_step = 0.025\n" 
        output += "\n"
        output += "boltz_temp_min = 100\n" 
        output += "boltz_temp_max = 1000\n" 
        output += "boltz_temp_step = 50\n" 
        output += "\n"
        output += "boltz_relax_time = 1\n"
        output += "!boltz_bandshift = false\n"
        output += "!boltz_bandshift_firstband = tag4\n"
        output += "!boltz_bandshift_energyshift = -0.06\n"
        output += "\n!"
        output += "!---------------"
        output += '! wannierization\n'
        output+='num_bands = '+ str(nbnd) +'\n'
        output+='num_wann = ' + str(nwann) + '\n'
        output+='exclude_bands : '+ exclude +'\n'
        output+='num_iter = 500 \n'
        output+='dis_froz_max = tag\n'
        output+='\n'
        output+='! restart = plot \n'
        output+='! write_hr = true \n'
        output+='\n'
        output+='bands_plot = true \n'
        output+='bands_plot_format = gnuplot \n'
        output+='bands_num_points = 50 \n'
        output+='begin kpoint_path\n'
        for p in paths:
            output+= p + "\n"
        output+='end kpoint_path\n'
        output+='\n'
        output+='begin atoms_frac\n'
        symbols = self.struct.get_chemical_symbols()
        pos = self.struct.get_scaled_positions()
        for i, sym in enumerate(symbols):
            output+="{:>2s}{:>12.8f}{:>12.8f}{:>12.8f}\n".format(sym, pos[i][0], pos[i][1], pos[i][2])
        output+='\n'
        output+='end atoms_frac\n'
        output+='\n'

        output+='begin projections\n'
        for element in set(symbols):
            output+= "{:>2s} : ".format(element) + self.pplib[element]["wannierproj"] + "\n"
        output+='end projections\n'
        output+='\n'
        
        output+= 'begin unit_cell_cart\n'
        output+= 'ang\n'
        for vector in self.struct.get_cell():
            output+="{:>13.8f}{:>13.8f}{:>13.8f}\n".format(vector[0],vector[1],vector[2])
        output+= 'end unit_cell_cart\n'
        output+= '\n'
        
        output+= 'mp_grid : {:>3d}{:>3d}{:>3d}\n\n'.format(self.k_coarse[0],self.k_coarse[1],self.k_coarse[2])
        mplines = get_qe_kpoints_data([self.k_coarse[0],self.k_coarse[1],self.k_coarse[2]],positive=True,withweight=False)
        output+= 'begin kpoints\n'
        for aline in mplines:
            output+= aline + '\n'
        output+= 'end kpoints\n'
        
        return output

    def getinput_epw(self) -> dict:
        return
