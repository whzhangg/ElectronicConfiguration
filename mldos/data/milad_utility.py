from collections import namedtuple
from copy import deepcopy
from operator import itemgetter
import warnings, typing
import numpy as np

from ase.data import chemical_symbols
from milad import invariants
from milad import geometric, zernike
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.structure import Structure

from mldos.parameters import MAIN_GROUP, Simple_METAL
from mldos.crystalstructure import StructLite

allow_neighbors = MAIN_GROUP | Simple_METAL

Neighbor = namedtuple("Neighbor", "index rij")  # Neighbor.index, Neighbor.rij
INVARIANTS = invariants.read(invariants.COMPLEX_INVARIANTS)


def from_ase_atoms_2_pymatgen(ase_struct) -> Structure:
    coord = ase_struct.get_scaled_positions()
    symbols = ase_struct.get_chemical_symbols()
    cell = ase_struct.get_cell()
    return Structure(lattice = cell, species = symbols, coords = coord)


class Description(StructLite):

    def get_all_dictances(self, iatoms, weight_cutoff = 2e-2) -> dict:
        """this method returns all the vector length |r|"""
        structure = from_ase_atoms_2_pymatgen(self.standard)
        neighborNN = VoronoiNN(cutoff = 10.0, compute_adj_neighbors = False)
        temperory = {}

        std_symbols = self.standard.get_chemical_symbols()
        for iatom in iatoms:
            nn_info = neighborNN.get_nn_info(structure, iatom)
            neighbors = []
            for nn in nn_info:
                neighbor = {}
                neighbor["index"] = nn["site_index"]
                neighbor["translation"] = nn["image"]
                neighbor["weight"] = nn["weight"]
                if neighbor["weight"] > weight_cutoff:
                    neighbors.append(neighbor)

            # issue a warning if the nearest neighbor is not in the main group
            sorted_element = [ std_symbols[nb["index"]] for nb in sorted(neighbors, key=itemgetter("weight"), reverse=True) ]
            nearest_set = set(sorted_element[0:4])
            if not nearest_set < MAIN_GROUP:
                text = self.formula + "  4 nearest atoms not in main group: "
                for ele in nearest_set:
                    text += " " + ele
                warnings.warn(text)

            temperory[iatom] = neighbors

        abs_r = {}
        cartesian_pos = self.standard.get_positions()

        for iatom, nns in temperory.items():
            rlist = []
            for nn in nns:
                index = nn["index"]
                t = nn["translation"]
                r = deepcopy(cartesian_pos[index])
                
                for i in range(3):
                    r += self.standard.get_cell()[i] * t[i]
                    
                r -= cartesian_pos[iatom]
                rlist.append(np.linalg.norm(r))
            # print(Counter([ index for index,_ in nnlist]))
            abs_r[iatom] = rlist

        return abs_r


    def _get_coordination_environment(self, iatoms: list, r_cutoff: float = 10, 
                                     weight_cutoff = 2e-2):
        # this method will generate the atomic environment {r_i} from the given 
        # cutoff. The zero of atomic position will be on the center atom

        structure = from_ase_atoms_2_pymatgen(self.standard)
        neighborNN = VoronoiNN(cutoff = r_cutoff, compute_adj_neighbors = False)
        #neighborNN = CrystalNN(x_diff_weight = 0.0, distance_cutoffs = [0.5, cutoff])  
        temperory = {}

        # in VoronoiNN, the weight is calculated from the solid angle
        # which is normalized by weight / max_weight
        # so the maximum weight given is 1
        # we add a small threshold to remove neighbor with too small weight
        std_symbols = self.standard.get_chemical_symbols()
        for iatom in iatoms:
            nn_info = neighborNN.get_nn_info(structure, iatom)
            neighbors = []
            for nn in nn_info:
                neighbor = {}
                neighbor["index"] = nn["site_index"]
                neighbor["translation"] = nn["image"]
                neighbor["weight"] = nn["weight"]
                if neighbor["weight"] > weight_cutoff:
                    neighbors.append(neighbor)

            # issue a warning if the nearest neighbor is not in the main group
            sorted_element = [ std_symbols[nb["index"]] for nb in sorted(neighbors, key=itemgetter("weight"), reverse=True) ]
            nearest_set = set(sorted_element[0:4])
            if not nearest_set < MAIN_GROUP:
                text = self.formula + "  4 nearest atoms not in main group: "
                for ele in nearest_set:
                    text += " " + ele
                warnings.warn(text)

            temperory[iatom] = neighbors

        neighbors = {}
        cartesian_pos = self.standard.get_positions()

        for iatom, nns in temperory.items():
            # first add the center atoms
            nnlist = [ Neighbor(iatom, np.zeros(3)) ]  
            for nn in nns:
                index = nn["index"]
                t = nn["translation"]
                r = deepcopy(cartesian_pos[index])
                
                for i in range(3):
                    r += self.standard.get_cell()[i] * t[i]
                    
                r -= cartesian_pos[iatom]
                nnlist.append(Neighbor(index, r))
            # print(Counter([ index for index,_ in nnlist]))
            neighbors[iatom] = nnlist

        return neighbors


    def _get_sites_in_sphere(self, iatoms: list, r_cutoff: float):
        """return all indexes that is within the given sphere"""
        structure = from_ase_atoms_2_pymatgen(self.standard)
        neighbors = {}
        for iatom in iatoms:
            # see structure.py 2791
            # and local_env.py
            nnn = structure.get_sites_in_sphere(self.standard.get_positions()[iatom], r_cutoff)  
            # all the neighbors within the cutoff sphere
            nnlist = [ Neighbor(n.index, n.coords - self.standard.get_positions()[iatom]) for n in nnn]  
            # I consider n.coords to be coordinate in the periodic image
            neighbors[iatom] = nnlist 

        return neighbors


    def _cutoff(self, rijs, rcut, q):
        """milad paper eq. 38, but with a value q controlling the speed of the decay
            q = 5 is recommended, which is roughly 0.8 at x = 0.8
        """
        weights = np.zeros_like(rijs)
        mask = np.where(rijs < rcut)
        weights[mask] = 0.5 * np.cos( np.pi * (rijs[mask] / rcut)**q ) + 0.5
        return weights


    def _get_first_shell_only(self, neighbors: dict, tolarance: float = 0.2):
        found = {}
        for iatom, neighbor in neighbors.items():
            dist = sorted([ np.linalg.norm(rij) for i,rij in neighbor ])[1] # the closet atom that is not itself
            rmin = dist * 0.98
            rmax = dist * (1 + tolarance)
            found[iatom] = [ n for n in neighbor if np.linalg.norm(n.rij) <= rmax ]

        return found


    def _encode_from_delta(self, vectors: np.array, weights: np.array, max_order : int):
        zernike_moments = zernike.from_deltas(max_order, vectors, weights = weights)
        complex_features = np.array(INVARIANTS(zernike_moments))
        return complex_features
        

    def _print_neighbors(self, neighbors: dict):
        """this function print neighbors for checking"""
        for iatom, neighbor in neighbors.items():
            print("Center atom {:>2s} with index {:>2d}".format(chemical_symbols[self.std_types[iatom]], iatom + 1))
            sorted_neighbors = sorted(neighbor, key = lambda n: np.linalg.norm(n.rij) )
            for i,n in enumerate(sorted_neighbors):
                print("{:>2d}-th, {:>2s} with distance {}".format(i, chemical_symbols[self.std_types[n.index]], np.linalg.norm(n.rij) ))


    def _check_nearest_contain_only_maingroup(self, neighbor):
        nearest_set = set([ chemical_symbols[self.std_types[i]] for i,rij in neighbor if np.linalg.norm(rij) > 0.1])
        return nearest_set < allow_neighbors
        

    def get_neighbors(self, 
        iatoms: list, 
        bonded: bool,
        firstshell: bool,
        cutoff: float
    ) -> typing.Dict[int, typing.List[Neighbor]]:
        """this provide a method to get neighbors only in a dictionary and list of tuples, only the 
        one who's first shell contain only maingroup is returned"""
        if firstshell: 
            bonded = False

        if bonded:
            neighbors = self._get_coordination_environment(iatoms, r_cutoff = cutoff)
        else:
            neighbors = self._get_sites_in_sphere(iatoms, r_cutoff = cutoff)
        
        nearest_neighbors = self._get_first_shell_only(neighbors)

        if firstshell:
            neighbors = nearest_neighbors
        
        return { iatom:neighbor for iatom,neighbor in neighbors.items() if self._check_nearest_contain_only_maingroup(nearest_neighbors[iatom]) }


    def get_features(self, 
        iatoms: list, 
        bonded: bool,
        firstshell: bool,
        cutoff: float,
        smooth: bool,
        method: str = "delta",
        max_order: int = None,
        weight_decay_q : float = 5.0,
        print_neighbors : bool = False,
        check_neighbors : bool = True
    ) -> typing.Dict[int, typing.Dict[str, np.ndarray]]:
        """
        bonded_only will give only the coordinates identified by the VoronoiNN

        """
        max_order = max_order or INVARIANTS.max_order
        if firstshell: 
            bonded = False

        if bonded:
            neighbors = self._get_coordination_environment(iatoms, r_cutoff = cutoff)
        else:
            neighbors = self._get_sites_in_sphere(iatoms, r_cutoff = cutoff)
        
        nearest_neighbors = self._get_first_shell_only(neighbors)

        if firstshell:
            neighbors = nearest_neighbors

        if print_neighbors:
            self._print_neighbors(neighbors)

        descriptor = {}
        for iatom in iatoms:
            if check_neighbors and not self._check_nearest_contain_only_maingroup(nearest_neighbors[iatom]):
                warnings.warn("site {:>3d} of {:s} first_shell contain more than main group".format(iatom + 1, self.formula))
                continue

            rijs = np.array([ np.linalg.norm(rij) for i,rij in neighbors[iatom] ])
            feature = {}
            ele_set = { chemical_symbols[self.std_types[i]] for i,_ in neighbors[iatom] }
            for ele in ele_set:
                vectors = np.array( [ rij for i,rij in neighbors[iatom] if chemical_symbols[self.std_types[i]] == ele ] ) 
                vectors /= cutoff
                if smooth:
                    rijs = np.array([ np.linalg.norm(rij) for i,rij in neighbors[iatom] if chemical_symbols[self.std_types[i]] == ele ])
                    weights = self._cutoff(rijs, cutoff, weight_decay_q)
                else:
                    weights = np.ones(len(vectors))

                feature[ele] = self._encode_from_delta(vectors, weights, max_order)

            one_hot_encode = {}
            for key, item in feature.items():
                one_hot_encode[key] = item

            descriptor[iatom] = one_hot_encode

        return descriptor



def write_cif_localstructure(first_shell = False, bonded = True):
    from ase import Atoms, io
    sd = Description(filename = "../../dos_calculation/calc/done/Ba1Co1S2/scf/scf.in")
    neighbors = sd.get_coordination_environment([3], r_cutoff= 5.0)[3]
    neighbors.append(Neighbor(3, np.zeros(3)))
    symbols = []
    positions = []
    for nn in neighbors:
        symbol = chemical_symbols[sd.std_types[nn.index]]
        pos = nn.rij
        symbols.append(symbol)
        positions.append(pos)
        #print("{:>3s}{:>10.4f}{:>10.4f}{:>10.4f}".format(symbol, *pos))
    A = 20
    cell = np.eye(3) * A
    structure = Atoms(symbols = symbols, positions = positions, cell = cell, pbc = False)
    io.write("Ba1Co1S2_Co.cif", structure, format = 'cif')

    

if __name__ == "__main__":
    print(len(INVARIANTS))