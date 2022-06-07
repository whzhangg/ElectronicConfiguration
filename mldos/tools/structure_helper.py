import re, typing
import numpy as np
from ase.data import chemical_symbols
from ase import io, Atoms
  

def atom_from_cpt(lattice, positions, types) -> Atoms:
    result = Atoms(cell = lattice, scaled_positions = positions)
    if type(types[0]) == str:
        result.set_chemical_symbols(types)
    else:
        result.set_atomic_numbers(types)
    return result


def read_generic_to_cpt(filename):
    struct = None

    if filename[-4:] == '.cif':
        struct=io.read(filename,format='cif')
    elif filename[-3:] == '.in':
        struct=io.read(filename,format='espresso-in')
    elif filename[-6:] == ".coord":
        cell = np.zeros((3,3))
        symbols = []
        pos = []
        with open(filename,'r') as f:
            aline = f.readline()
            while aline:
                if "CELL_PARAMETER" in aline:
                    if "angstrom" in aline:
                        factor = 1.0
                    if "bohr" in aline:
                        factor = 0.529177
                    for i in range(3):
                        cell[i] = np.array(f.readline().split(), dtype = float)
                        cell *= factor
                if "ATOMIC_POSITIONS" in aline:
                    con = True
                    while con:
                        aline = f.readline()
                        data = aline.split()
                        if len(data) == 4:
                            p = np.array(data[1:4], dtype = float)
                            pos.append(p)
                            symbols.append(data[0])
                        else:
                            con = False
                    if con == False:
                        continue
                            
                aline = f.readline()   
        
        struct = Atoms(cell = cell, scaled_positions = pos)
            #self.struct.set_scaled_positions(self.std_position)
        struct.set_chemical_symbols(symbols)
    else:
        print("file should be .cif or .in or .coord , exit! ")

    return struct.get_cell(), struct.get_scaled_positions(), struct.get_atomic_numbers()


def is_formula(formula:str) -> bool:
    # accept a string with only letters and numbers
    symbols = re.findall(r'[A-Z][a-z]*', formula)
    if not symbols:
        return False
    for s in symbols:
        if s not in chemical_symbols:
            return False
    return True


def separate_formula(formula:str) -> dict:
    '''
    separate a formula into a dictionary of elements and their counts
    - any space is removed
    - 1 is inserted if not found
    - fractional counts are accepted
    '''
    formula = re.sub(' ', '', formula)
    parts = re.findall(r'[A-Z][a-z]*[0-9]+[,.]?[0-9]*', re.sub(r'[A-Z][a-z]*(?![\da-z])', r'\g<0>1', formula))

    result = {}
    for p in parts:
        chemical = re.findall(r'[A-Z][a-z]*', p)[0]
        count = re.findall(r'[0-9]+[,.]?[0-9]*', p)[0]
        result[chemical] = count
    return result


def format_formula(formula:str) -> str:
    parts = separate_formula(formula)
    result = ""
    for ele, count in parts.items():
        result += ele
        if int(count) > 1:
            result += "_" + count
    return result


def write_cif_file(filename: str,struct: Atoms) -> None:
    io.write(filename, struct, format='cif')

