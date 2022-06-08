import numpy as np
import torch
import os

from mldos.network.datasets import InputData, np2torch, partition_data_by_elements, partition_data_by_transition_metals
from mldos.network.nn_train import EarlyStop, TrainingBase
from mldos.network.nn_ensemble import BaggingEnsemble
from mldos.tools import write_json, read_json, change_filetype, printProgressBar
from mldos.parameters import r_seed_testsamples
from mldos.network.neuralnetwork import get_network_selfinteraction, get_network_simpleconvolution
from mldos.data.criteria import from_filename_to_criteria
from mldos.data.make_descriptor import make_descriptor_from_structure


datafile = "occ_bonded_rcut=5.0_nele=40_single_smoothed_N1430.h5"
example_model = "self-interaction_occ_bonded_rcut=5.0_nele=65_single_smoothed_N2007.h5_ensemble1.pt"


def prepare(datafile:str, min_ep=100, max_ep=200):
    alldata = InputData.from_file(datafile)
    
    net = get_network_selfinteraction()
    stop = EarlyStop(min_epochs = min_ep, max_epochs = max_ep, early_stop=True)
    trainer = TrainingBase(net, stop_function = stop, loss_function = torch.nn.L1Loss(), optimizer="adamw", 
                            learning_rate= 1e-4, weight_decay = 1e-4, batchsize=32,
                            validation_size=0.005, input_dropout=0.2)
    return alldata, trainer


def train_occupancy():
    alldata, trainer = prepare()
    testset, trainset = alldata.split(0.1, seed = r_seed_testsamples+1)
    trainer.train(trainset)
    score = trainer.estimate_error(testset)
    print(score)
    results = {"results" : [list(trainer.train_scores), list(trainer.test_scores)]}
    #write_json(results, "run_errors_batchsize32.json")
    trainer.save_model()


def predict_new(structfile):
    criteria = from_filename_to_criteria(datafile)
    features = make_descriptor_from_structure(structfile, criteria)
    _, trainer = prepare(datafile)
    trainer.load_model(example_model)

    result = {}
    for key, f in features.items():
        X = np2torch(f.T)
        y = trainer.predict(X.unsqueeze_(0)).squeeze_(0)
        result[key] = y
    print(result)
    return result


def nfold_error_estimation_dataset(dataset, steps = [50, 100, 150, 200]):
    all_errors = {}
    for s in steps:
        alldata, _ = prepare(dataset, min_ep=s, max_ep=s+5)
        all_training_error_single_step = []
        all_testing_error_single_step = []
        for test_set,train_set in alldata.nfold(n = 10):
            _, trainer = prepare(dataset, min_ep=s, max_ep=s+5)
            trainer.train(train_set)
            testscore = trainer.estimate_error(test_set)
            trainscore = trainer.estimate_error(train_set)
            print(testscore)
            all_training_error_single_step.append(trainscore)
            all_testing_error_single_step.append(testscore)
        all_errors[str(s)] = {"train" : all_training_error_single_step, "test": all_testing_error_single_step}
    write_json(all_errors, dataset+".errorlog.json")


def test_single_model():
    folder = os.path.split(__file__)[0]
    file = os.path.join(folder, "tests", "BaCoS2.cif")
    predict_new(file)


