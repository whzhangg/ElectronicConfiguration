import abc
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from mldos.models import model_tools


class EarlyStop:
    """ callable that give true if training should be stopped
        it either use early stopping or does not use early stopping. 
        For early stopping, algorithm 7.1 is used from DL Goodfellow.

        step defines an average on which we compare the test error
    """
    def __init__(self, 
        min_epochs: int = 30,
        max_epochs: int = 300, 
        early_stop: bool = True, 
        patience: int = 2, 
        step: int = 5,
        verbose: bool = False
    ) -> None:
        self.min_epoch = min_epochs
        self.max_epoch = max_epochs
        self.early_stop = early_stop
        self.patience = patience
        self.step = step
        self.verbose = verbose

        self.tmp_error = []
        self.test_errors = []
        self.i = 0
        self.j = 0

        self.criteria_satisfied = False

    def judge_stop(self):
        # update internal values
        self.i += 1
        if self.i % self.step == 0:
            tmp_error = sum(self.tmp_error) / len(self.tmp_error)

            if (np.abs(np.array(self.tmp_error) - tmp_error) < 1e-10).all():
                return True   # this should prevent training to continue if all error values are the same
            
            if self.early_stop and len(self.test_errors) > 0:
                if tmp_error > self.test_errors[-1]:
                    self.j += 1
                else:
                    self.j = 0

                if self.j > self.patience:
                    return True

            self.test_errors.append(tmp_error)
            self.tmp_error = []

        return False

    def __call__(self, epoch, test_error):
        # update internal values
        self.tmp_error.append(test_error)

        if not self.criteria_satisfied:
            self.criteria_satisfied = self.judge_stop()

        # hard requirement
        if epoch+1 >= self.max_epoch:
            return True
        if epoch+1 <= self.min_epoch:
            return False

        if self.criteria_satisfied:
            return True
        else:
            return False


class NormalizeLoss():
    # callable that return how far the input is from normalized value and 
    # add to the normal loss function given by the input variable
    def __init__(self, lossfunction, normalize_to: float, ratio: float = 1e-5):
        self.loss = lossfunction
        self.ratio = ratio
        self.normalize_to = normalize_to

    def __call__(self, pred, target):
        loss = self.loss(pred,target)
        N, L = pred.shape
        tmp = torch.sum(pred, dim=1)
        tmp = (tmp - self.normalize_to) ** 2
        loss += torch.sum(tmp) / N * self.ratio
        return loss


class TrainingABC(abc.ABC):
    """the abstract base class, we need to implement
    method __init__, _fit and _forward"""

    def predict(self, X):
        """a general method that will yield real y"""
        return self._forward(X)

    def train(self, training_set):
        """this is a wrapper for convenient overloading"""
        self._datasetname = training_set.datafilename # we remember the training set name
        return self._fit(training_set)

    def estimate_error(self, input_dataset):
        dataset = DataLoader(input_dataset)
        mse = 0
        mae = 0
        with torch.no_grad():
            counts = 0
            for (X,y) in dataset:
                predictions = self._forward(X)
                diff = y - predictions
                
                counts += 1
                mae += torch.sum(torch.abs(diff))/2.0  # this is the same as L1Loss, summed up together and divide by counts
                mse += torch.sum(diff**2)/2.0          # this is the same as MSELoss, summed up together and divide by counts, as the following line
                
                #mse += self.loss_function(predictions, y).item()
        mse /= counts
        mae /= counts
        result = {"mae": float(mae), "mse": float(mse)}
        return result

    @abc.abstractmethod
    def _forward(self, X):
        """return output from the network"""
        raise 
    
    @abc.abstractmethod
    def _fit(self, training_set):
        """train the networks contained"""
        raise 
    
    @abc.abstractmethod
    def load_model(self, filename):
        """load model parameters from a file"""
        raise 

    @abc.abstractmethod
    def save_model(self, filename):
        """save model parameters to a file"""
        raise 


class TrainingBase(TrainingABC):
    """ A General Network frame that provide training and predict methods
    """

    def __init__(self, 
        network,
        stop_function,
        optimizer: str = "adamw",
        loss_function: nn.Module = None,
        validation_size: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        batchsize: int = 16,
        input_dropout: float = 0.0,
        verbose: bool = True
    ):
        self._network = network
        self._stop_function = stop_function
        self._loss_function = loss_function or torch.nn.MSELoss()

        self.optimizer = optimizer
        self.validation_size = validation_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batchsize = batchsize
        self.verbose = verbose
        self.input_dropout = input_dropout

        self.train_scores = []
        self.test_scores = []

        self._datasetname = ""

    # this property should be hidden as it is not accessed from outside
    @property
    def network(self):
        return self._network

    @property
    def stop_function(self):
        return self._stop_function

    @property
    def loss_function(self):
        return self._loss_function

    def _fit(self, training_set):

        if self.optimizer == "adamw":
            optimiser = torch.optim.AdamW(self._network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimiser = torch.optim.SGD(self._network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise Exception

        epoch = 0
        loss_result = None
        
        validation, train = training_set.split(self.validation_size)
        validation_loader = DataLoader(validation)
        train_dataloader = DataLoader(train, self.batchsize)

        while True:

            if self.input_dropout > 1e-10:
                small, large = train.split(self.input_dropout)
                train_dataloader = DataLoader(large, self.batchsize)

            for (X,Y) in train_dataloader:
                optimiser.zero_grad()

                predictions = self._forward(X)
                loss_result = self.loss_function(predictions, Y)

                loss_result.backward()
                optimiser.step()

            with torch.no_grad():
                # if more than 1 batch, we evaluate loss for the entire training set
                train_score = 0
                test_score = 0

                counts = 0
                for (X,y) in train_dataloader:
                    predictions = self._forward(X)
                    counts += 1
                    train_score += self.loss_function(predictions, y).item()
                train_score /= counts
                
                counts = 0
                for (X,y) in validation_loader:
                    predictions = self._forward(X)
                    counts += 1
                    test_score += self.loss_function(predictions, y).item()
                test_score /= counts

                self.train_scores.append(train_score)
                self.test_scores.append(test_score)

                if self.verbose:
                    print("Epoch {:>5d} = train_score : {:>16.6e}; test_score : {:>16.6e}".format(epoch+1, train_score, test_score))

                if self.stop_function(epoch, test_score):
                    break
            
            epoch += 1

        return loss_result

    def _forward(self, X):
        return self._network(X)

    def save_model(self, model_filename: str = ""):
        purpose_name = self._network.name + "_" + self._datasetname + ".pt"
        savename = model_filename or purpose_name
        model_tools.save_model(self._network, savename)

    def load_model(self, model_filename: str):
        model_tools.load_model(self._network, model_filename)

