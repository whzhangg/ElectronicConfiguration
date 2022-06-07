import numpy as np
import torch
import os

from mldos.network.datasets import InputData, np2torch, partition_data_by_elements, partition_data_by_transition_metals
from mldos.network.nn_train import EarlyStop, TrainingBase
from mldos.network.nn_ensemble import BaggingEnsemble
from mldos.tools import write_json, read_json, change_filetype, printProgressBar
from mldos.parameters import r_seed_testsamples
from mldos.network.neuralnetwork import get_network_selfinteraction, get_network_simpleconvolution
from mldos.data.data_generation import Data

model_name = "current_model_occupancy.pt"
"""
available_dataset = [
    "occ_bonded_rcut=5.0_smoothed_N1205.h5", 
    "occ_firstshell_rcut=5.0_smoothed_N1205.h5"
    ]
"""

datafile = "occ_bonded_rcut=5.0_nele=40_single_smoothed_N1430.h5"


def prepare(datafile:str, min_ep=100, max_ep=200):
    alldata = InputData.from_file(datafile)
    featureshape = tuple(alldata.featureshape)
    #print("input has the shape " + str(featureshape))
    
    net = get_network_selfinteraction()

    stop = EarlyStop(min_epochs = min_ep, max_epochs = max_ep, early_stop=True)

    trainer = TrainingBase(net, stop_function = stop, loss_function = torch.nn.L1Loss(), optimizer="adamw", 
                            learning_rate= 1e-4, weight_decay = 1e-4, batchsize=32,
                            validation_size=0.005, input_dropout=0.2)
    return alldata, trainer


def train_occupancy():
    #print("Using random seed {:d}".format(r_seed_testsamples))
    alldata, trainer = prepare()
    testset, trainset = alldata.split(0.1, seed = r_seed_testsamples+1)
    #print(testset.names[0:20])
    trainer.train(trainset)
    score = trainer.estimate_error(testset)
    print(score)
    results = {"results" : [list(trainer.train_scores), list(trainer.test_scores)]}
    #write_json(results, "run_errors_batchsize32.json")
    trainer.save_model()
    #save_model(trainer.network, filename = model_name)


def predict_new(structfile, index = None):
    datageneration = Data.from_filename(os.path.join("data/hdf", datafile))
    features = datageneration.make_descriptor_from_structure(structfile = structfile)
    _, trainer = prepare()
    trainer.load_model("self-interaction_occ_bonded_rcut=5.0_nele=40_single_smoothed_N1430.h5.pt")

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


emsemble_datafile = "occ_firstshell_rcut=5.0_nele=65_single_smoothed_N2007.h5"


def prepare_ensemble(datafile:str, min_ep = 100, max_ep = 200):
    alldata = InputData.from_file(datafile)
    featureshape = tuple(alldata.featureshape)
    #print("input has the shape " + str(featureshape))

    stop = EarlyStop(min_epochs = min_ep, max_epochs = max_ep, early_stop=True)

    trainer = BaggingEnsemble(get_network_selfinteraction, stop_function = stop, nensemble=10, 
                            loss_function = torch.nn.L1Loss(), optimizer="adamw", 
                            learning_rate= 1e-4, weight_decay = 1e-4, batchsize=32,
                            validation_size=0.005, input_dropout=0.2)

    return alldata, trainer


def train_ensemble():
    alldata, trainer = prepare_ensemble(emsemble_datafile, min_ep=200, max_ep=205)
    trainer.train(alldata)
    trainscore = trainer.estimate_error(alldata)
    trainer.save_model()
    print(trainscore)


def ensemble_partitioned_predict(mode = "set"):
    alldata, trainer = prepare_ensemble(emsemble_datafile, min_ep=200, max_ep=205)
    trainer.load_model("self-interaction_" + emsemble_datafile)
    if mode == "set":
        results = partition_data_by_elements(alldata)
    elif mode == "tm":
        results = partition_data_by_transition_metals(alldata)
    
    print(results.keys())
    set_value_compare = {}
    for k, data in results.items():
        all_predicted_values = trainer.predict(data.features)
        all_calculated_values = data.targets
        assert all_calculated_values.shape == all_predicted_values.shape
        set_value_compare[k] = {"pred": all_predicted_values, "calc": all_calculated_values}
        #trainscore = trainer.estimate_error(data)
        #print(k)
        #print(trainscore)

    return set_value_compare # this contain the calc. vs pred. value for the three sets


def ensemble_general_predict(structs: list = None, index:list = None):
    """only one index is to be supplied for one structure, however, if you want to extract multiple index, duplicate structure"""
    if not index:
        index = [None] * len(structs)
    if len(index) != len(structs):
        raise

    printprogress = False
    if len(structs) > 100:
        printprogress = True
        tot_num = len(structs)
        step = max( int(tot_num/100), 1)
        printProgressBar(0, tot_num)

    data = Data.from_filename(emsemble_datafile)
    allresult = []
    allinputs = []

    counter = 0
    for s, i in zip(structs, index):
        if type(s) == str:
            features = data.make_descriptor_from_structure(structfile = s, check_neighbor=False)
        elif type(s) == tuple and len(s) == 3:
            features = data.make_descriptor_from_structure(cell=s[0], positions=s[1], types=s[2], check_neighbor=False)
        else:
            raise Exception("Input is nor correct!")

        if i is None:
            i = list(features.keys())[0]
        allinputs.append(features[i].T)

        if printprogress and counter % step == 0:
            printProgressBar(counter+1, tot_num)
        counter += 1

    if printprogress:
        printProgressBar(tot_num, tot_num)

    allinputs = np2torch(np.array(allinputs))

    _, trainer = prepare_ensemble(emsemble_datafile, min_ep=200, max_ep=205)
    trainer.load_model("self-interaction_" + emsemble_datafile)

    return trainer.predict(allinputs)


if __name__ == "__main__":
    #predict_new("/home/wenhao/work/ml_dos/dos_calculation/calc/failed/Cr4As3/scf.1.in")

    #nfold_error_estimation_dataset("occ_firstshell_rcut=5.0_nele=65_single_smoothed_N2007.h5")
    #nfold_error_estimation_dataset("occ_bonded_rcut=5.0_nele=65_single_smoothed_N2007.h5")
    #nfold_error_estimation_dataset("occ_all_rcut=5.0_nele=65_single_smoothed_N2007.h5")

    ensemble_partitioned_predict()

    pass