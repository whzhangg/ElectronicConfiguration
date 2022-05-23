import abc
import numpy as np
import torch
from sklearn.decomposition import NMF

from mldos.parameters import estep_dos
from .datasets import Dataset
from ._model_parameter import torch_device, torch_dtype


class Factorize(abc.ABC):

    @abc.abstractmethod
    def transform(self, inputdata):
        """return the transformed data"""
        pass

    @property
    @abc.abstractmethod
    def n_components(self):
        """return the number of components for dos"""
        pass

    @property
    @abc.abstractmethod
    def components(self):
        """return all the components"""
        pass

    @abc.abstractmethod
    def reconstruct(self, coefficients):
        """reconstruct density of states from the coefficients"""
        pass


class NMFactorize(Factorize):
    """NMF components are normalized to 5.0 and the coefficients sum up to 1.0"""
    def __init__(self, n_components: int, random_seed: int) -> None:
        self.seed = np.random.RandomState(random_seed)

        self._ncomp = n_components
        self._components = None
        self.nmf = None
        self.nx = None  # length of input feature
        self.nsample = None
        self.nchannel = None
        self.features = None

    def transform(self, inputdata):
        self.train(inputdata)
        features = torch.from_numpy(self.features).to(torch_device, torch_dtype)
        return torch.reshape(features, (self.nsample, self.nchannel, -1))
    
    def reconstruct(self, coefficients):
        assert len(coefficients) == self._ncomp
        result = np.zeros(self.nx)
        for x, component in zip(coefficients, self._components):
            result += x.item() * component
        return result

    @property
    def n_components(self):
        return self._ncomp

    @property
    def components(self):
        return self._components

    def train(self, all_dos):
        self.nsample, self.nchannel, self.nx = all_dos.shape  # nchannel = 2: spin up and down

        each_dos_sum = torch.sum(all_dos, dim = [2]) * estep_dos
        each_dos_sum = np.expand_dims(each_dos_sum, 2)
        each_dos_sum = np.repeat(each_dos_sum, self.nx, axis = 2)
        assert each_dos_sum.shape == all_dos.shape

        all_dos = all_dos * 5.0 / each_dos_sum
        each_dos_sum = torch.sum(all_dos, axis = 2) * estep_dos
        assert np.isclose(each_dos_sum, 5.0).all()

        X = torch.flatten(all_dos, start_dim = 0, end_dim = 1)

        self.nmf = NMF(n_components = self._ncomp, max_iter = 800, random_state = self.seed, init = "nndsvda", solver = 'cd').fit(X)
        # using the random state ensures the repeatability, see
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF.transform

        self.features = self.nmf.transform(X)
        #print(all_dos.shape)
        #print(self.nmf.n_iter_)
        #print(self.features[0])

        self.normalization = []
        self._components = []
        for component in self.nmf.components_:
            dos_sum = np.sum(component) * estep_dos
            self._components.append(component * 5.0 / dos_sum)
            self.normalization.append(dos_sum / 5.0)

        self.normalization = np.array(self.normalization)
        self._components = np.array(self._components)

        multiply = np.tile(self.normalization, [len(self.features), 1])
        self.features *= multiply

    def plot_components(self):
        datatext = ""
        for i in range(len(self._components[0])):
            datatext += "{:>8.2f}".format( -10 + 0.1 * i )
            for j in range(self._ncomp):
                datatext += "{:>14.6e}".format(self._components[j,i])
            datatext += "\n"

        with open("nmf_ncomp={:d}.dat".format(self._ncomp), 'w') as f:
            f.write(datatext)

    @staticmethod
    def test_nmf(ncomponent = 30, file = ".h5"):
        nmf = NMFactorize(ncomponent)
        data = Dataset.from_file(file)
        target = data.targets
        features = nmf.transform(target)
        print(target.shape)
        print(features.shape)
        print(features[0])
