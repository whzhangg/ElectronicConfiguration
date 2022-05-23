import numpy as np
import torch

from mldos.parameters import estep_dos
from .datasets import InputData
from ._dos_factorize import NMFactorize


class NMFWrapper:
    """this method wrap around a general trainer and provide NMF for both training and prediction"""

    def __init__(self, trainer, ncomponents = 30):
        self.ncomponents = ncomponents
        self.trainer = trainer
        self.nmf = None
        self.nspin = None

    def train_nmf(self, dataset, seed: int = 0):
        """ it trains the nmf and return the transformed dos target that flattened the spin dimension
            to obtain reproduciable factorization, we need to use the same training set"""
        self.nmf = NMFactorize(n_components = self.ncomponents, random_seed = seed)
        targets = dataset.targets
        newtargets = self.nmf.transform(targets) 
        # newtargets have the same size for the first two dimension (nsample, nspin) while the last dimension is reduced

        nsample, self.nspin, ncomp = newtargets.shape
        newtargets = torch.reshape(newtargets, (nsample, self.nspin * ncomp))
        features = dataset.features
        names = dataset.names
        new_set = InputData(features, newtargets, names)
        return new_set

    def train(self, training_set):
        self.trainer.train(training_set)
        return

    def predict(self, x):
        coefficients = self.trainer.predict(x.unsqueeze_(0))
        coefficients.squeeze_(0)
        return coefficients

    def coefficients_to_dos(self, coefficients):
        coefficients = torch.reshape(coefficients, (self.nspin, self.ncomponents))
        doss = []
        for ispin in range(self.nspin):
            dos = self.nmf.reconstruct(coefficients[ispin])
            if np.sum(dos) > 0.1:
                dos *= 5.0 / ( np.sum(dos) * estep_dos )
            doss.append(dos)
        return doss