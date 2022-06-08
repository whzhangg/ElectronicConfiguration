import numpy as np
import torch
from mldos.network.datasets import InputData, np2torch, partition_data_by_elements, partition_data_by_transition_metals
from mldos.network.nn_train import EarlyStop
from mldos.network.nn_ensemble import BaggingEnsemble
from mldos.tools import printProgressBar
from mldos.network.neuralnetwork import get_network_selfinteraction
from mldos.data.criteria import from_filename_to_criteria
from mldos.data import make_descriptor_from_structure


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

    criteria = from_filename_to_criteria(emsemble_datafile)
    allinputs = []

    counter = 0
    for s, i in zip(structs, index):
        features = make_descriptor_from_structure(s, criteria)

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


def test_ensemble_prediction():
    ensemble_partitioned_predict()
