import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from mldos.models import model_tools
from .nn_train import TrainingBase, TrainingABC
from .datasets import InputData
from copy import deepcopy

class BaggingEnsemble(TrainingABC):
    """this class are required to have the same interface as training Base and 
    directly use training base, but management several of them using the bagging method,
    which is described in Bishop Chapter 14.
    """

    # instead of network, we take a function that generate the network as input
    def __init__(self,
        network_function,
        stop_function,
        nensemble: int = 5,
        optimizer: str = "adamw",
        loss_function: nn.Module = None,
        validation_size: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        batchsize: int = 16,
        input_dropout: float = 0.0,
        verbose: bool = True
    ) -> None:
        self.nensemble = nensemble

        self._eachtraining = []
        for _ in range(self.nensemble):
            stop_function_copy = deepcopy(stop_function)
            loss_function_copy = deepcopy(loss_function)
            train=TrainingBase( network_function(),
                                stop_function_copy,
                                optimizer = optimizer,
                                loss_function = loss_function_copy,
                                validation_size = validation_size,
                                learning_rate = learning_rate,
                                weight_decay = weight_decay,
                                batchsize = batchsize,
                                input_dropout = input_dropout,
                                verbose = verbose)

            self._eachtraining.append(train)

        self.train_scores = []
        self.test_scores = []

        self._datasetname = ""

    def _fit(self, training_set: InputData):
        for each in self._eachtraining:
            bagged_training, ratio = training_set.bagging_set()
            print(ratio)
            each.train(bagged_training)

    def _forward(self, X) -> np.array:
        for i, each in enumerate(self._eachtraining):
            if i >= 1:
                sum_results += each.predict(X).detach().numpy()
            else:
                sum_results = each.predict(X).detach().numpy()
        return sum_results / self.nensemble
    
    def save_model(self, model_prefix: str = ""):
        purpose_name = self._eachtraining[0]._network.name

        if self._datasetname:
            purpose_name += "_" + self._datasetname 
            
        model_prefix = model_prefix or purpose_name
        print(purpose_name)
        # model name will be model_prefix_ensemblei.pt
        for i, each in enumerate(self._eachtraining):
            savename = model_prefix + "_ensemble{:d}.pt".format(i+1)
            model_tools.save_model(each._network, savename)

    def load_model(self, model_prefix: str):
        for i, each in enumerate(self._eachtraining):
            savename = model_prefix + "_ensemble{:d}.pt".format(i+1)
            model_tools.load_model(each._network, savename)
