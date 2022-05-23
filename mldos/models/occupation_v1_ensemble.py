import numpy as np
import torch
from mldos.network import (
    Dataset, ConvNet, SimpleConvolution, EarlyStop, TrainingBase,
    linearx
)

from .model_tools import save_model, load_model

from mldos.tools import write_json
from mldos.parameters import r_seed_testsamples
from mldos.network import encode_feature, torch_device, torch_dtype
from mldos.data import get_features


model_name = "current_model_occupancy.pt"

def ensemble_name(i):
    return "ensemble_model_occupancy_ensemble_{:d}.pt".format(i+1)


def prepare(mode: str):
    assert mode in ["train", "test"]

    alldata = []
    featureshape = tuple(alldata.featureshape)
    conv = SimpleConvolution(featureshape, output_shape=(60,40), weight_sharing = True )

    if mode == "train":
        net = ConvNet(featureshape, 2, convolution = conv, normalization = linearx(3.16), 
                    linearlayer_connection = (500, 200, 100), dropout=(0.0,0.0))
    else:
        net = ConvNet(featureshape, 2, convolution = conv, normalization = linearx(3.16), 
                    linearlayer_connection = (500, 200, 100), dropout=(0.0, 0.0))

    stop = EarlyStop(max_epochs=100, early_stop=True)

    trainer = TrainingBase(net, stop_function = stop, optimizer="adamw", learning_rate= 1e-4, weight_decay = 1e-4, batchsize=32)
    return alldata, trainer


def train_occupancy_ensemble(n_ensemble = 10):
    print("Using random seed {:d}".format(r_seed_testsamples))
    ensemble_results = {}
    for i in range(n_ensemble):
        alldata, trainer = prepare("train")
        _, trainset = alldata.split(0.02, seed = r_seed_testsamples)
        trainer.train(trainset)
        ensemble_results["ensemble_{:>d}".format(i)] = [list(trainer.train_scores), list(trainer.test_scores)]
        
        save_model(trainer.network, filename = ensemble_name(i))
    write_json(ensemble_results, "train_ensembles.json")


def predict_test_ensemble(n_ensemble = 10):
    print("Using random seed {:d}".format(r_seed_testsamples))
    results = {"calculated": [], "each_prediction": [], "average_prediction": []}

    for j in range(n_ensemble):
        alldata, trainer = prepare("train")
        testset, _ = alldata.split(0.02, seed = r_seed_testsamples)
        load_model(trainer.network, ensemble_name(j))
        result_ensemble = np.zeros((len(testset), 2))
        calculated = np.zeros((len(testset), 2))
        for i, (x, y) in enumerate(testset):
            x.unsqueeze_(0)
            pred = trainer.predict(x).squeeze_(0)
            result_ensemble[i] = pred.detach().numpy()
            calculated[i] = y.numpy()
        results["calculated"] = calculated
        results["each_prediction"].append(result_ensemble)
        
        print(j+1)
        for y, pred in zip(calculated, result_ensemble):
                print("Calc. value: {:8.4f}{:8.4f}   / Pred. value: {:8.4f}{:8.4f}".format(*y, *pred))
    
    average = np.zeros(results["calculated"].shape)
    for p in results["each_prediction"]:
        average += p
    results["average_prediction"] = average / len(results["each_prediction"])

    print("average")
    for y, pred in zip(results["calculated"], results["average_prediction"]):
            print("Calc. value: {:8.4f}{:8.4f}   / Pred. value: {:8.4f}{:8.4f}".format(*y, *pred))


def predict_new_ensemble(cell, pos, types, index = None, use_first = False, n_ensemble = 10):
    formula, feature = get_features(cell=cell, positions=pos, types=types, index = index, use_first = use_first)
    
    features = encode_feature( feature )
    X = torch.from_numpy(features.T).to(torch_device, torch_dtype)

    X.unsqueeze_(0)
    results = {"each_prediction": [], "average_prediction": []}

    for j in range(n_ensemble):
        alldata, trainer = prepare("train")
        load_model(trainer.network, ensemble_name(j))
        pred = trainer.predict(X).squeeze_(0)
        results["each_prediction"].append(pred.detach().numpy())
    
    
    average = np.zeros((2))
    for p in results["each_prediction"]:
        average += p
    results["average_prediction"] = average / len(results["each_prediction"])
    return results
    
