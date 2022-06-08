import numpy as np
from milad import zernike
from mldos.data.milad_utility import INVARIANTS
from mldos.network.nn_train import TrainingBase, EarlyStop
from mldos.network.nn_components import *
from mldos.network.datasets import np2torch
from torch import nn
from torch.utils.data import Dataset

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        # normalization -> linearlayers -> outputlayers
        self.normalization = linearx(3.16)
        self.linearlayer = MultiLinear(117, 1, (50,))

    def forward(self, x):
        out = self.normalization(x)
        out = self.linearlayer(out)
        return out


class InputData(Dataset):

    def __init__(self, features_np, targets_np, ids: list = None, datafilename = ""):
        """input feature, target_np, 
        ids are optional to keep track of the datas, for example, compound names"""
        assert len(features_np) == len(targets_np)
        if ids is not None:
            assert len(features_np) == len(ids)
            self._names = ids
        else:
            self._names = [""] * len(features_np)

        self._features = np2torch(features_np)
        self._targets = np2torch(targets_np)

        self.nsample = len(features_np)
        self._datafilename = datafilename

    @property
    def features(self):
        return self._features

    @property
    def targets(self):
        return self._targets

    @property
    def names(self):
        return self._names

    @property
    def datafilename(self):
        return self._datafilename

    @property
    def featureshape(self):
        _, *feature_shape = self._features.shape
        return feature_shape

    @property
    def targetshape(self):
        _, *target_shape = self._targets.shape
        return target_shape

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        return self._features[idx], self.targets[idx]

    def split(self, ratio:float = 0.1, indices:tuple = None, seed:int = None):
        # return the ratio set first, then the remaining set, if indices are supplied, use the given indices
        # note that no data is copied
        seed = seed or int(np.random.rand() * 100)

        if not indices:
            nsample_test = int(len(self) * ratio)
            rng = np.random.default_rng(seed = seed)
            shuffled = np.arange(len(self))
            rng.shuffle(shuffled)
            first_indices = shuffled[0:nsample_test]
            second_indices = shuffled[nsample_test:]
        else:
            first_indices, second_indices = indices

        first_names = [ self._names[i] for i in first_indices ]
        second_names = [ self._names[i] for i in second_indices ]

        first_data = InputData(self._features[first_indices], self._targets[first_indices], ids = first_names, datafilename = self._datafilename)
        second_data  = InputData(self._features[second_indices],  self._targets[second_indices],  ids = second_names, datafilename = self._datafilename )

        return (first_data, second_data)


class ModelProblem:

    index = {"alpha":0, "beta":1}

    def __init__(self, ntrainingsample):
        self.distortions = np.array([ (45 * np.random.random(), 0.6 + np.random.random()) for i in range(ntrainingsample) ])
        self.features = np.array([ self.distortion_octahedron_feature(*distort) for distort in self.distortions ])
        
        self.trainer = TrainingBase(Network(), 
                                    stop_function = EarlyStop(min_epochs = 200, max_epochs = 400, early_stop=True), 
                                    loss_function = torch.nn.MSELoss(), 
                                    optimizer="adamw", 
                                    learning_rate= 1e-3, batchsize=1,
                                    validation_size=0.2)

    def train_and_test(self, target = "alpha", ntest = 50):

        training_dataset = InputData(self.features, self.distortions[:,self.index[target]])
        self.trainer.train(training_dataset)

        test_distortions = np.array([ (45 * np.random.random(), 0.6 + np.random.random()) for i in range(ntest) ])
        test_features = np.array([ self.distortion_octahedron_feature(*distort) for distort in test_distortions ])
        test_features = np2torch(test_features)
        prediction = self.trainer.predict(test_features).detach().numpy()

        print("{:>20s}{:>20s}".format("Test", "Prediction"))
        result = []
        for i, dist in enumerate(test_distortions):
            print("{:>20.3f}{:>20.3f}".format(dist[self.index[target]], prediction[i][0]))
            result.append((dist[self.index[target]], prediction[i][0]))
        return result
        
    def distortion_octahedron_feature(self, alpha = 0, beta = 1.0):
        assert alpha >= 0 and alpha < 90
        assert beta < 2.0 and beta > 0
        alpha *= np.pi / 180
        vectors = np.array([[ 0, 0, 0.5 * beta],
                            [ 0.5 * np.cos(alpha), 0.5 * np.sin(alpha), 0],
                            [ 0, 0.5, 0],
                            [ 0, 0,- 0.5* beta],
                            [-0.5 * np.cos(alpha), -0.5 * np.sin(alpha), 0],
                            [ 0,-0.5, 0]], dtype = float)
        
        max_order = INVARIANTS.max_order
        zernike_moments = zernike.from_deltas(max_order, vectors)
        complex_features = np.array(INVARIANTS(zernike_moments))

        return complex_features

    def get_continuous_deformation(self, n = 10):
        distortions = np.array([ (0, i) for i in np.arange(0.75, 1.25, 0.5/n) ])
        return np.array([ self.distortion_octahedron_feature(*distort) for distort in distortions ])


if __name__ == "__main__":
    problem = ModelProblem(50)
    problem.train_and_test("beta", 30)
